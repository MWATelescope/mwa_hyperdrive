// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to generate sky-model visibilities.

use std::{
    borrow::Cow,
    collections::{hash_map::DefaultHasher, HashMap, HashSet},
    f64::consts::{FRAC_PI_2, LN_2},
    hash::{Hash, Hasher},
};

use hifitime::{Duration, Epoch};
use log::debug;
use marlu::{
    c64,
    pos::xyz::xyzs_to_cross_uvws,
    precession::{get_lmst, precess_time},
    AzEl, Jones, LmnRime, RADec, XyzGeodetic, UVW,
};
use ndarray::{parallel::prelude::*, prelude::*, ArcArray2};
use num_complex::Complex;

use super::{shapelets, ModelError};
use crate::{
    beam::{Beam, BeamError, BeamType},
    constants::*,
    context::Polarisations,
    model::mask_pols,
    srclist::{ComponentList, GaussianParams, PerComponentParams, Source, SourceList},
};

const GAUSSIAN_EXP_CONST: f64 = -(FRAC_PI_2 * FRAC_PI_2) / LN_2;
const SHAPELET_CONST: f64 = SQRT_FRAC_PI_SQ_2_LN_2 / shapelets::SBF_DX;

pub struct SkyModellerCpu<'a> {
    pub(crate) beam: &'a dyn Beam,

    /// The phase centre used for all modelling.
    pub(crate) phase_centre: RADec,
    /// The longitude of the array we're using \[radians\].
    pub(crate) array_longitude: f64,
    /// The latitude of the array we're using \[radians\].
    pub(crate) array_latitude: f64,
    /// The UT1 - UTC offset. If this is 0, effectively UT1 == UTC, which is a
    /// wrong assumption by up to 0.9s. We assume the this value does not change
    /// over the timestamps given to this `SkyModellerCpu`.
    pub(crate) dut1: Duration,
    /// Shift baselines and LSTs back to J2000.
    pub(crate) precess: bool,

    pub(crate) unflagged_fine_chan_freqs: &'a [f64],

    /// The [`XyzGeodetic`] positions of each of the unflagged tiles.
    pub(crate) unflagged_tile_xyzs: &'a [XyzGeodetic],
    pub(crate) unflagged_baseline_to_tile_map: HashMap<usize, (usize, usize)>,

    pub(crate) components: ComponentList,

    tile_index_to_array_index_map: Vec<usize>,
    freq_map: Vec<usize>,
    unique_tiles: Vec<usize>,
    unique_freqs: Vec<f64>,

    pub(crate) pols: Polarisations,
}

impl<'a> SkyModellerCpu<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        beam: &'a dyn Beam,
        source_list: &SourceList,
        pols: Polarisations,
        unflagged_tile_xyzs: &'a [XyzGeodetic],
        unflagged_fine_chan_freqs: &'a [f64],
        flagged_tiles: &'a HashSet<usize>,
        phase_centre: RADec,
        array_longitude_rad: f64,
        array_latitude_rad: f64,
        dut1: Duration,
        apply_precession: bool,
    ) -> SkyModellerCpu<'a> {
        let components = ComponentList::new(
            source_list
                .values()
                .rev()
                .flat_map(|src| src.components.iter()),
            unflagged_fine_chan_freqs,
            phase_centre,
        );
        let maps = crate::math::TileBaselineFlags::new(
            unflagged_tile_xyzs.len() + flagged_tiles.len(),
            flagged_tiles.clone(),
        );

        // Before we get the beam responses, work out the unique tiles and beam-
        // unique frequencies. This means we potentially de-duplicate a bunch
        // of work.
        let total_num_tiles = unflagged_tile_xyzs.len() + flagged_tiles.len();
        // When there are no beam gains or delays, there's a way to cheaply
        // generate the tile map, but I'm lazy right now.
        let gains = beam
            .get_dipole_gains()
            .unwrap_or(ArcArray2::ones((total_num_tiles, 16)));
        let delays = beam
            .get_dipole_delays()
            .unwrap_or(ArcArray2::zeros((total_num_tiles, 16)));
        let mut unique_hashes = vec![];
        let mut unique_tiles = vec![];
        let mut tile_index_to_array_index_map = Vec::with_capacity(total_num_tiles);

        let mut i_array_tile = 0;
        for (i_tile, (gains, delays)) in gains.outer_iter().zip(delays.outer_iter()).enumerate() {
            if flagged_tiles.contains(&i_tile) {
                tile_index_to_array_index_map.push(0);
                continue;
            }

            let (gains, delays) = fix_amps_ndarray(gains, delays);

            let mut unique_tile_hasher = DefaultHasher::new();
            delays.hash(&mut unique_tile_hasher);
            // We can't hash f64 values, but we can hash their bits.
            for gain in gains {
                gain.to_bits().hash(&mut unique_tile_hasher);
            }
            let unique_tile_hash = unique_tile_hasher.finish();
            let index = if let Some((_, index)) = unique_hashes
                .iter()
                .find(|(unique_hash, _)| *unique_hash == unique_tile_hash)
            {
                *index
            } else {
                unique_hashes.push((unique_tile_hash, i_array_tile));
                unique_tiles.push(i_tile);
                i_array_tile += 1;
                i_array_tile - 1
            };
            tile_index_to_array_index_map.push(index);
        }

        let mut unique_beam_freqs = vec![];
        let mut unique_freqs = vec![];
        let mut freq_map = vec![];
        let mut i_array_freq = 0;
        for &freq in unflagged_fine_chan_freqs {
            let beam_freq = beam.find_closest_freq(freq);
            let this_freq_index =
                if let Some((_, index)) = unique_beam_freqs.iter().find(|(f, _)| *f == beam_freq) {
                    *index
                } else {
                    unique_beam_freqs.push((beam_freq, i_array_freq));
                    unique_freqs.push(beam_freq);
                    i_array_freq += 1;
                    i_array_freq - 1
                };
            freq_map.push(this_freq_index);
        }

        SkyModellerCpu {
            beam,
            phase_centre,
            array_longitude: array_longitude_rad,
            array_latitude: array_latitude_rad,
            dut1,
            precess: apply_precession,
            unflagged_fine_chan_freqs,
            unflagged_tile_xyzs,
            unflagged_baseline_to_tile_map: maps.unflagged_cross_baseline_to_tile_map,
            components,
            tile_index_to_array_index_map,
            unique_tiles,
            unique_freqs,
            freq_map,
            pols,
        }
    }

    /// Given source component directions and the MWA latitude, get the beam
    /// responses for all unique tiles and unique beam frequencies.
    /// De-duplicating the work saves a lot of time!
    ///
    /// If the beam is `BeamType::None`, then save even more work up front.
    fn get_beam_responses(
        &self,
        azels: &[AzEl],
        array_latitude_rad: f64,
    ) -> Result<Array3<Jones<f64>>, BeamError> {
        if matches!(self.beam.get_beam_type(), BeamType::None) {
            return Ok(Array3::from_elem(
                (1, self.unflagged_fine_chan_freqs.len(), azels.len()),
                Jones::identity(),
            ));
        }

        let mut beam_responses = Array3::zeros((
            self.unique_tiles.len(),
            self.unique_freqs.len(),
            azels.len(),
        ));
        // The variables are a bit confusing here. `i_tile` is the outer-most
        // index into `beam_responses`, and `i_unique_tile` is the index to feed
        // into the beam calculations.
        for (i_tile, &i_unique_tile) in self.unique_tiles.iter().enumerate() {
            for (slice, freq) in beam_responses
                .slice_mut(s![i_tile, .., ..])
                .as_slice_mut()
                .expect("is contiguous")
                .chunks_exact_mut(azels.len())
                .zip(self.unique_freqs.iter())
            {
                self.beam.calc_jones_array_inner(
                    azels,
                    *freq,
                    Some(i_unique_tile),
                    array_latitude_rad,
                    slice,
                )?;
            }
        }

        Ok(beam_responses)
    }

    /// This function is mostly used for testing. For a single timestep, over
    /// the already-provided baselines and frequencies, generate visibilities
    /// for each specified sky-model point-source component.
    ///
    /// `vis_model_fb`: A mutable view into an `ndarray`. Rather than returning
    /// an array from this function, modelled visibilities are written into
    /// this array. This slice *must* have dimensions `[n1][n2]`, where `n1`
    /// is number of unflagged frequencies and `n2` is the number of unflagged
    /// cross correlation baselines.
    ///
    /// `uvws`: The [`UVW`] coordinates of each baseline \[metres\]. This should
    /// be the same length as `vis_model_fb`'s first axis.
    ///
    /// `lst_rad`: The local sidereal time in \[radians\].
    ///
    /// `array_latitude_rad`: The latitude of the array/telescope/interferometer
    /// in \[radians\].
    pub(super) fn model_points(
        &self,
        mut vis_model_fb: ArrayViewMut2<Jones<f32>>,
        uvws: &[UVW],
        lst_rad: f64,
        array_latitude_rad: f64,
    ) -> Result<(), ModelError> {
        if self.components.points.radecs.is_empty() {
            return Ok(());
        }

        let fds = &self.components.points.flux_densities;
        let lmns = &self.components.points.lmns;
        let azels = &self
            .components
            .points
            .get_azels_mwa_parallel(lst_rad, array_latitude_rad);

        assert_eq!(
            vis_model_fb.len_of(Axis(1)),
            uvws.len(),
            "vis_model_fb.len_of(Axis(1)) != uvws.len()"
        );
        assert_eq!(
            vis_model_fb.len_of(Axis(0)),
            self.unflagged_fine_chan_freqs.len(),
            "vis_model_fb.len_of(Axis(0)) != self.unflagged_fine_chan_freqs.len()"
        );
        assert_eq!(
            fds.len_of(Axis(0)),
            self.unflagged_fine_chan_freqs.len(),
            "fds.len_of(Axis(0)) != self.unflagged_fine_chan_freqs.len()"
        );
        assert_eq!(
            fds.len_of(Axis(1)),
            azels.len(),
            "fds.len_of(Axis(1)) != azels.len()"
        );
        assert_eq!(
            fds.len_of(Axis(1)),
            lmns.len(),
            "fds.len_of(Axis(1)) != lmns.len()"
        );
        assert_eq!(
            uvws.len(),
            self.unflagged_baseline_to_tile_map.len(),
            "uvws.len() != self.unflagged_baseline_to_tile_map.len()"
        );

        let beam_responses = self.get_beam_responses(azels, array_latitude_rad)?;

        // Iterate over the unflagged baseline axis.
        vis_model_fb
            .axis_iter_mut(Axis(1))
            .into_par_iter()
            .zip(uvws.par_iter())
            .enumerate()
            .for_each(|(i_baseline, (mut vis_model_f, &uvw))| {
                let (i_tile1, i_tile2) = self.unflagged_baseline_to_tile_map[&i_baseline];
                // We only need the tile indices for beam responses; use the
                // tile map to access de-duplicated beam responses.
                let i_tile1 = self.tile_index_to_array_index_map[i_tile1];
                let i_tile2 = self.tile_index_to_array_index_map[i_tile2];

                // Unflagged fine-channel axis.
                vis_model_f
                    .iter_mut()
                    .zip(fds.outer_iter())
                    .zip(self.unflagged_fine_chan_freqs)
                    .enumerate()
                    .for_each(|(i_freq, ((vis_model, comp_fds), freq))| {
                        // Access the beam-deduplicated-freq index.
                        let i_freq = self.freq_map[i_freq];
                        let tile1_beam = beam_responses.slice(s![i_tile1, i_freq, ..]);
                        let tile2_beam = beam_responses.slice(s![i_tile2, i_freq, ..]);

                        // Divide UVW by lambda to make UVW dimensionless.
                        let UVW { u, v, w } = uvw * *freq / VEL_C;

                        // Accumulate the double-precision visibilities into
                        // a double-precision Jones matrix before putting that
                        // into the `vis_model_fb`.
                        let mut jones_accum: Jones<f64> = Jones::default();

                        comp_fds
                            .iter()
                            .zip(tile1_beam)
                            .zip(tile2_beam)
                            .zip(lmns.iter())
                            .for_each(|(((comp_fd, beam_1), beam_2), &LmnRime { l, m, n })| {
                                jones_accum += (*beam_1 * *comp_fd * beam_2.h())
                                    * c64::cis(u * l + v * m + w * n);
                            });
                        // Demote to single precision now that all operations are
                        // done.
                        *vis_model += Jones::from(jones_accum);
                    });
            });
        Ok(())
    }

    /// This function is mostly used for testing. For a single timestep, over
    /// the already-provided baselines and frequencies, generate visibilities
    /// for each specified sky-model Gaussian-source component.
    ///
    /// `vis_model_fb`: A mutable view into an `ndarray`. Rather than returning
    /// an array from this function, modelled visibilities are written into
    /// this array. This slice *must* have dimensions `[n1][n2]`, where `n1`
    /// is number of unflagged frequencies and `n2` is the number of unflagged
    /// cross correlation baselines.
    ///
    /// `uvws`: The [`UVW`] coordinates of each baseline \[metres\]. This should
    /// be the same length as `vis_model_fb`'s first axis.
    ///
    /// `lst_rad`: The local sidereal time in \[radians\].
    ///
    /// `array_latitude_rad`: The latitude of the array/telescope/interferometer
    /// in \[radians\].
    pub(super) fn model_gaussians(
        &self,
        mut vis_model_fb: ArrayViewMut2<Jones<f32>>,
        uvws: &[UVW],
        lst_rad: f64,
        array_latitude_rad: f64,
    ) -> Result<(), ModelError> {
        if self.components.gaussians.radecs.is_empty() {
            return Ok(());
        }

        let fds = &self.components.gaussians.flux_densities;
        let lmns = &self.components.gaussians.lmns;
        let azels = &self
            .components
            .gaussians
            .get_azels_mwa_parallel(lst_rad, array_latitude_rad);
        let gaussian_params = &self.components.gaussians.gaussian_params;

        assert_eq!(
            vis_model_fb.len_of(Axis(1)),
            uvws.len(),
            "vis_model_fb.len_of(Axis(1)) != uvws.len()"
        );
        assert_eq!(
            vis_model_fb.len_of(Axis(0)),
            self.unflagged_fine_chan_freqs.len(),
            "vis_model_fb.len_of(Axis(0)) != self.unflagged_fine_chan_freqs.len()"
        );
        assert_eq!(
            fds.len_of(Axis(0)),
            self.unflagged_fine_chan_freqs.len(),
            "fds.len_of(Axis(0)) != self.unflagged_fine_chan_freqs.len()"
        );
        assert_eq!(
            fds.len_of(Axis(1)),
            azels.len(),
            "fds.len_of(Axis(1)) != azels.len()"
        );
        assert_eq!(
            fds.len_of(Axis(1)),
            lmns.len(),
            "fds.len_of(Axis(1)) != lmns.len()"
        );
        assert_eq!(
            uvws.len(),
            self.unflagged_baseline_to_tile_map.len(),
            "uvws.len() != self.unflagged_baseline_to_tile_map.len()"
        );

        let beam_responses = self.get_beam_responses(azels, array_latitude_rad)?;

        // Iterate over the unflagged baseline axis.
        vis_model_fb
            .axis_iter_mut(Axis(1))
            .into_par_iter()
            .zip(uvws.par_iter())
            .enumerate()
            .for_each(|(i_baseline, (mut vis_model_f, &uvw))| {
                let (i_tile1, i_tile2) = self.unflagged_baseline_to_tile_map[&i_baseline];
                // We only need the tile indices for beam responses; use the
                // tile map to access de-duplicated beam responses.
                let i_tile1 = self.tile_index_to_array_index_map[i_tile1];
                let i_tile2 = self.tile_index_to_array_index_map[i_tile2];

                // Unflagged fine-channel axis.
                vis_model_f
                    .iter_mut()
                    .zip(fds.outer_iter())
                    .zip(self.unflagged_fine_chan_freqs)
                    .enumerate()
                    .for_each(|(i_freq, ((vis_model, comp_fds), freq))| {
                        // Access the beam-deduplicated-freq index.
                        let i_freq = self.freq_map[i_freq];
                        let tile1_beam = beam_responses.slice(s![i_tile1, i_freq, ..]);
                        let tile2_beam = beam_responses.slice(s![i_tile2, i_freq, ..]);
                        // Divide UVW by lambda to make UVW dimensionless.
                        let UVW { u, v, w } = uvw * *freq / VEL_C;

                        // Now that we have the UVW coordinates, we can determine
                        // each source component's envelope.
                        let envelopes = gaussian_params.iter().map(|g_params| {
                            let (s_pa, c_pa) = g_params.pa.sin_cos();
                            // Temporary variables for clarity.
                            let k_x = u * s_pa + v * c_pa;
                            let k_y = u * c_pa - v * s_pa;
                            (GAUSSIAN_EXP_CONST
                                * (g_params.maj.powi(2) * k_x.powi(2)
                                    + g_params.min.powi(2) * k_y.powi(2)))
                            .exp()
                        });

                        // Accumulate the double-precision visibilities into
                        // a double-precision Jones matrix before putting that
                        // into the `vis_model_fb`.
                        let mut jones_accum: Jones<f64> = Jones::default();

                        comp_fds
                            .iter()
                            .zip(tile1_beam)
                            .zip(tile2_beam)
                            .zip(lmns.iter())
                            .zip(envelopes)
                            .for_each(
                                |(
                                    (((comp_fd, beam_1), beam_2), &LmnRime { l, m, n }),
                                    envelope,
                                )| {
                                    jones_accum += (*beam_1 * *comp_fd * beam_2.h())
                                        * c64::cis(u * l + v * m + w * n)
                                        * envelope;
                                },
                            );
                        // Demote to single precision now that all operations are
                        // done.
                        *vis_model += Jones::from(jones_accum);
                    });
            });

        Ok(())
    }

    /// This function is mostly used for testing. For a single timestep, over
    /// the already-provided baselines and frequencies, generate visibilities
    /// for each specified sky-model shapelet-source component.
    ///
    /// `vis_model_fb`: A mutable view into an `ndarray`. Rather than returning
    /// an array from this function, modelled visibilities are written into
    /// this array. This slice *must* have dimensions `[n1][n2]`, where `n1`
    /// is number of unflagged frequencies and `n2` is the number of unflagged
    /// cross correlation baselines.
    ///
    /// `uvws`: The [`UVW`] coordinates of each baseline \[metres\]. This should
    /// be the same length as `vis_model_fb`'s first axis.
    ///
    /// `shapelet_uvws` are special UVWs generated as if each shapelet component
    /// was at the phase centre \[metres\]. The first axis is unflagged
    /// baseline, the second shapelet component.
    ///
    /// `lst_rad`: The local sidereal time in \[radians\].
    ///
    /// `array_latitude_rad`: The latitude of the array/telescope/interferometer
    /// in \[radians\].
    pub(super) fn model_shapelets(
        &self,
        mut vis_model_fb: ArrayViewMut2<Jones<f32>>,
        uvws: &[UVW],
        shapelet_uvws: ArrayView2<UVW>,
        lst_rad: f64,
        array_latitude_rad: f64,
    ) -> Result<(), ModelError> {
        if self.components.shapelets.radecs.is_empty() {
            return Ok(());
        }

        let fds = &self.components.shapelets.flux_densities;
        let lmns = &self.components.shapelets.lmns;
        let azels = &self
            .components
            .shapelets
            .get_azels_mwa_parallel(lst_rad, array_latitude_rad);
        let gaussian_params = &self.components.shapelets.gaussian_params;
        let shapelet_coeffs = &self.components.shapelets.shapelet_coeffs;

        assert_eq!(
            vis_model_fb.len_of(Axis(1)),
            uvws.len(),
            "vis_model_fb.len_of(Axis(1)) != uvws.len()"
        );
        assert_eq!(
            vis_model_fb.len_of(Axis(0)),
            self.unflagged_fine_chan_freqs.len(),
            "vis_model_fb.len_of(Axis(0)) != self.unflagged_fine_chan_freqs.len()"
        );
        assert_eq!(
            fds.len_of(Axis(0)),
            self.unflagged_fine_chan_freqs.len(),
            "fds.len_of(Axis(0)) != self.unflagged_fine_chan_freqs.len()"
        );
        assert_eq!(
            fds.len_of(Axis(1)),
            azels.len(),
            "fds.len_of(Axis(1)) != azels.len()"
        );
        assert_eq!(
            fds.len_of(Axis(1)),
            lmns.len(),
            "fds.len_of(Axis(1)) != lmns.len()"
        );
        assert_eq!(
            uvws.len(),
            self.unflagged_baseline_to_tile_map.len(),
            "uvws.len() != self.unflagged_baseline_to_tile_map.len()"
        );
        assert_eq!(
            vis_model_fb.len_of(Axis(1)),
            shapelet_uvws.len_of(Axis(0)),
            "vis_model_fb.len_of(Axis(1)) != shapelet_uvws.len_of(Axis(0))"
        );
        assert_eq!(
            fds.len_of(Axis(1)),
            shapelet_uvws.len_of(Axis(1)),
            "fds.len_of(Axis(1)) != shapelet_uvws.len_of(Axis(1))"
        );

        const I_POWER_TABLE: [c64; 4] = [
            c64::new(1.0, 0.0),
            c64::new(0.0, 1.0),
            c64::new(-1.0, 0.0),
            c64::new(0.0, -1.0),
        ];

        let beam_responses = self.get_beam_responses(azels, array_latitude_rad)?;

        // Iterate over the unflagged baseline axis.
        vis_model_fb
            .axis_iter_mut(Axis(1))
            .into_par_iter()
            .zip(uvws.par_iter())
            .zip(shapelet_uvws.outer_iter())
            .enumerate()
            .for_each(
                |(i_baseline, ((mut vis_model_f, &uvw), shapelet_uvws_per_comp))| {
                    let (i_tile1, i_tile2) = self.unflagged_baseline_to_tile_map[&i_baseline];
                    // We only need the tile indices for beam responses; use the
                    // tile map to access de-duplicated beam responses.
                    let i_tile1 = self.tile_index_to_array_index_map[i_tile1];
                    let i_tile2 = self.tile_index_to_array_index_map[i_tile2];

                    // Preallocate a vector for the envelopes.
                    let mut envelopes = vec![c64::default(); gaussian_params.len()];

                    // Unflagged fine-channel axis.
                    vis_model_f
                        .iter_mut()
                        .zip(fds.outer_iter())
                        .zip(self.unflagged_fine_chan_freqs)
                        .enumerate()
                        .for_each(|(i_freq, ((vis_model, comp_fds), freq))| {
                            // Access the beam-deduplicated-freq index.
                            let i_freq = self.freq_map[i_freq];
                            let tile1_beam = beam_responses.slice(s![i_tile1, i_freq, ..]);
                            let tile2_beam = beam_responses.slice(s![i_tile2, i_freq, ..]);
                            // Divide UVW by lambda to make UVW dimensionless.
                            let one_on_lambda = freq / VEL_C;
                            let UVW { u, v, w } = uvw * one_on_lambda;

                            // Now that we have the UVW coordinates, we can
                            // determine each source component's envelope.
                            envelopes
                                .iter_mut()
                                .zip(gaussian_params.iter())
                                .zip(shapelet_coeffs.iter())
                                .zip(shapelet_uvws_per_comp.iter())
                                .for_each(|(((envelope, g_params), coeffs), shapelet_uvw)| {
                                    let shapelet_u = shapelet_uvw.u * one_on_lambda;
                                    let shapelet_v = shapelet_uvw.v * one_on_lambda;
                                    let GaussianParams { maj, min, pa } = g_params;

                                    let (s_pa, c_pa) = pa.sin_cos();
                                    let x = shapelet_u * s_pa + shapelet_v * c_pa;
                                    let y = shapelet_u * c_pa - shapelet_v * s_pa;
                                    let const_x = maj * SHAPELET_CONST;
                                    let const_y = -min * SHAPELET_CONST;
                                    let x_pos = x * const_x + shapelets::SBF_C;
                                    let y_pos = y * const_y + shapelets::SBF_C;
                                    let x_pos_int = x_pos.floor() as usize;
                                    let y_pos_int = y_pos.floor() as usize;

                                    // Fold the shapelet basis functions (here,
                                    // "coeffs") into a single envelope.
                                    *envelope = coeffs.iter().fold(
                                        Complex::default(),
                                        |envelope_acc, coeff| {
                                            let f_hat = coeff.value;

                                            // Omitting boundary checks speeds
                                            // things up by ~14%.
                                            unsafe {
                                                let x_low = shapelets::SHAPELET_BASIS_VALUES
                                                    .get_unchecked(
                                                        shapelets::SBF_L * usize::from(coeff.n1)
                                                            + x_pos_int,
                                                    );
                                                let x_high = shapelets::SHAPELET_BASIS_VALUES
                                                    .get_unchecked(
                                                        shapelets::SBF_L * usize::from(coeff.n1)
                                                            + x_pos_int
                                                            + 1,
                                                    );
                                                let u_value = x_low
                                                    + (x_high - x_low) * (x_pos - x_pos.floor());

                                                let y_low = shapelets::SHAPELET_BASIS_VALUES
                                                    .get_unchecked(
                                                        shapelets::SBF_L * usize::from(coeff.n2)
                                                            + y_pos_int,
                                                    );
                                                let y_high = shapelets::SHAPELET_BASIS_VALUES
                                                    .get_unchecked(
                                                        shapelets::SBF_L * usize::from(coeff.n2)
                                                            + y_pos_int
                                                            + 1,
                                                    );
                                                let v_value = y_low
                                                    + (y_high - y_low) * (y_pos - y_pos.floor());

                                                envelope_acc
                                                    + I_POWER_TABLE.get_unchecked(usize::from(
                                                        (coeff.n1 + coeff.n2) % 4,
                                                    )) * f_hat
                                                        * u_value
                                                        * v_value
                                            }
                                        },
                                    )
                                });

                            // Accumulate the double-precision visibilities into
                            // a double-precision Jones matrix before putting
                            // that into the `vis_model_fb`.
                            let mut jones_accum: Jones<f64> = Jones::default();

                            comp_fds
                                .iter()
                                .zip(tile1_beam)
                                .zip(tile2_beam)
                                .zip(lmns.iter())
                                .zip(envelopes.iter())
                                .for_each(
                                    |(
                                        (((comp_fd, beam_1), beam_2), &LmnRime { l, m, n }),
                                        envelope,
                                    )| {
                                        jones_accum += (*beam_1 * *comp_fd * beam_2.h())
                                            * c64::cis(u * l + v * m + w * n)
                                            * *envelope;
                                    },
                                );
                            // Demote to single precision now that all operations are
                            // done.
                            *vis_model += Jones::from(jones_accum);
                        });
                },
            );

        Ok(())
    }

    /// For a timestamp, get the LST, tile [`UVW`]s and array latitude. These things
    /// depend on whether we're precessing, so rather than copy+pasting this
    /// code around the place, put it in one spot.
    fn get_lst_uvws_latitude(&self, timestamp: Epoch) -> (f64, Vec<UVW>, f64) {
        let (lst, xyzs, latitude) = if self.precess {
            let precession_info = precess_time(
                self.array_longitude,
                self.array_latitude,
                self.phase_centre,
                timestamp,
                self.dut1,
            );
            // Apply precession to the tile XYZ positions.
            let precessed_tile_xyzs = precession_info.precess_xyz(self.unflagged_tile_xyzs);
            debug!(
                "Modelling GPS timestamp {}, LMST {}°, J2000 LMST {}°",
                timestamp.to_gpst_seconds(),
                precession_info.lmst.to_degrees(),
                precession_info.lmst_j2000.to_degrees()
            );
            (
                precession_info.lmst_j2000,
                Cow::from(precessed_tile_xyzs),
                precession_info.array_latitude_j2000,
            )
        } else {
            let lst = get_lmst(self.array_longitude, timestamp, self.dut1);
            debug!(
                "Modelling GPS timestamp {}, LMST {}°",
                timestamp.to_gpst_seconds(),
                lst.to_degrees()
            );
            (
                lst,
                Cow::from(self.unflagged_tile_xyzs),
                self.array_latitude,
            )
        };

        let uvws = xyzs_to_cross_uvws(&xyzs, self.phase_centre.to_hadec(lst));
        (lst, uvws, latitude)
    }
}

impl<'a> super::SkyModeller<'a> for SkyModellerCpu<'a> {
    fn model_timestep(
        &self,
        timestamp: Epoch,
    ) -> Result<(Array2<Jones<f32>>, Vec<UVW>), ModelError> {
        let (lst, uvws, latitude) = self.get_lst_uvws_latitude(timestamp);
        let shapelet_uvws = self
            .components
            .shapelets
            .get_shapelet_uvws(lst, self.unflagged_tile_xyzs);
        let mut vis_fb = Array2::default((self.unflagged_fine_chan_freqs.len(), uvws.len()));

        self.model_points(vis_fb.view_mut(), &uvws, lst, latitude)?;
        self.model_gaussians(vis_fb.view_mut(), &uvws, lst, latitude)?;
        self.model_shapelets(
            vis_fb.view_mut(),
            &uvws,
            shapelet_uvws.view(),
            lst,
            latitude,
        )?;

        Ok((vis_fb, uvws))
    }

    fn model_timestep_with(
        &self,
        timestamp: Epoch,
        mut vis_fb: ArrayViewMut2<Jones<f32>>,
    ) -> Result<Vec<UVW>, ModelError> {
        let (lst, uvws, latitude) = self.get_lst_uvws_latitude(timestamp);
        let shapelet_uvws = self
            .components
            .shapelets
            .get_shapelet_uvws(lst, self.unflagged_tile_xyzs);

        self.model_points(vis_fb.view_mut(), &uvws, lst, latitude)?;
        self.model_gaussians(vis_fb.view_mut(), &uvws, lst, latitude)?;
        self.model_shapelets(
            vis_fb.view_mut(),
            &uvws,
            shapelet_uvws.view(),
            lst,
            latitude,
        )?;

        // Mask any unavailable polarisations.
        mask_pols(vis_fb, self.pols);

        Ok(uvws)
    }

    fn update_with_a_source(
        &mut self,
        source: &Source,
        phase_centre: RADec,
    ) -> Result<(), ModelError> {
        self.phase_centre = phase_centre;
        self.components = ComponentList::new(
            source.components.iter(),
            self.unflagged_fine_chan_freqs,
            phase_centre,
        );
        Ok(())
    }
}

/// Ensure that any delays of 32 have an amplitude (dipole gain) of 0. The
/// results are bad otherwise! Also ensure that we have 32 dipole gains (amps)
/// here. Also return a Rust array of delays for convenience.
///
/// TODO: This is copy+pasted from `hyperbeam`; make that function public and
/// use it instead.
fn fix_amps_ndarray(amps: ArrayView1<f64>, delays: ArrayView1<u32>) -> ([f64; 32], [u32; 16]) {
    let mut full_amps: [f64; 32] = [1.0; 32];
    full_amps
        .iter_mut()
        .zip(amps.iter().cycle())
        .zip(delays.iter().cycle())
        .for_each(|((out_amp, &in_amp), &delay)| {
            if delay == 32 {
                *out_amp = 0.0;
            } else {
                *out_amp = in_amp;
            }
        });

    // So that we don't have to do .as_slice().unwrap() on our ndarrays outside
    // of this function, return a Rust array of delays here.
    let mut delays_a: [u32; 16] = [0; 16];
    delays_a.iter_mut().zip(delays).for_each(|(da, d)| *da = *d);

    (full_amps, delays_a)
}
