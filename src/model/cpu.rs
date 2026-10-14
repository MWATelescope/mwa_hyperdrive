// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to generate sky-model visibilities.

use std::{
    borrow::Cow,
    collections::{hash_map::DefaultHasher, HashSet},
    f64::consts::{FRAC_PI_2, LN_2},
    hash::{Hash, Hasher},
};

use hifitime::{Duration, Epoch};
use log::debug;
use marlu::{
    c64,
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
    TileBaselineFlags,
};

const GAUSSIAN_EXP_CONST: f64 = -(FRAC_PI_2 * FRAC_PI_2) / LN_2;
const SHAPELET_CONST: f64 = SQRT_FRAC_PI_SQ_2_LN_2 / shapelets::SBF_DX;

pub struct SkyModellerCpu<'a> {
    pub(super) beam: &'a dyn Beam,

    /// The phase centre used for all modelling.
    pub(super) phase_centre: RADec,
    /// The longitude of the array we're using \[radians\].
    pub(super) array_longitude: f64,
    /// The latitude of the array we're using \[radians\].
    pub(super) array_latitude: f64,
    /// The UT1 - UTC offset. If this is 0, effectively UT1 == UTC, which is a
    /// wrong assumption by up to 0.9s. We assume the this value does not change
    /// over the timestamps given to this `SkyModellerCpu`.
    pub(super) dut1: Duration,
    /// Shift baselines and LSTs back to J2000.
    pub(super) precess: bool,

    pub(super) unflagged_fine_chan_freqs: &'a [f64],

    /// The [`XyzGeodetic`] positions of each of the unflagged tiles.
    pub(super) unflagged_tile_xyzs: &'a [XyzGeodetic],
    /// Information on tile/baseline indices and flags.
    pub(super) tile_baseline_flags: TileBaselineFlags,

    pub(super) components: ComponentList,

    tile_index_to_array_index_map: Vec<usize>,
    freq_map: Vec<usize>,
    unique_tiles: Vec<usize>,
    unique_freqs: Vec<f64>,

    pub(super) pols: Polarisations,
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
        let tile_baseline_flags = crate::math::TileBaselineFlags::new(
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

            let (gains, delays) = mwa_hyperbeam::fix_amps_ndarray(gains, delays);

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
            tile_baseline_flags,
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
        let shape = (
            self.unique_tiles.len(),
            self.unique_freqs.len(),
            azels.len(),
        );

        if matches!(self.beam.get_beam_type(), BeamType::None) {
            return Ok(Array3::from_elem(shape, Jones::identity()));
        }

        if azels.is_empty() || self.unique_tiles.is_empty() || self.unique_freqs.is_empty() {
            // Return an empty array to avoid chunk size zero panics
            return Ok(Array3::zeros(shape));
        }

        let mut beam_responses = Array3::zeros(shape);
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
        vis_model_fb: ArrayViewMut2<Jones<f32>>,
        tile_uvws: &[UVW],
        lst_rad: f64,
        array_latitude_rad: f64,
    ) -> Result<(), ModelError> {
        let points = &self.components.points;
        if points.radecs.is_empty() {
            return Ok(());
        }
        let azels = points.get_azels_mwa_parallel(lst_rad, array_latitude_rad);
        let beam_responses = self.get_beam_responses(&azels, array_latitude_rad)?;
        self.model_factorised(
            vis_model_fb,
            tile_uvws,
            &points.lmns,
            points.flux_densities.view(),
            beam_responses.view(),
            None,
        );
        Ok(())
    }

    /// Model Gaussian components; see [`SkyModellerCpu::model_points`].
    pub(super) fn model_gaussians(
        &self,
        vis_model_fb: ArrayViewMut2<Jones<f32>>,
        tile_uvws: &[UVW],
        lst_rad: f64,
        array_latitude_rad: f64,
    ) -> Result<(), ModelError> {
        let gaussians = &self.components.gaussians;
        if gaussians.radecs.is_empty() {
            return Ok(());
        }
        let azels = gaussians.get_azels_mwa_parallel(lst_rad, array_latitude_rad);
        let beam_responses = self.get_beam_responses(&azels, array_latitude_rad)?;
        self.model_factorised(
            vis_model_fb,
            tile_uvws,
            &gaussians.lmns,
            gaussians.flux_densities.view(),
            beam_responses.view(),
            Some(&gaussians.gaussian_params),
        );
        Ok(())
    }

    /// The straightforward (one `cis` per baseline, frequency and component)
    /// point-source model, kept as the reference for testing
    /// [`SkyModellerCpu::model_points`]. `uvws` are baseline UVWs.
    #[cfg(test)]
    pub(super) fn model_points_reference(
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
            self.tile_baseline_flags
                .unflagged_cross_baseline_to_tile_map
                .len(),
            "uvws.len() != self.tile_baseline_flags.unflagged_cross_baseline_to_tile_map.len()"
        );

        let beam_responses = self.get_beam_responses(azels, array_latitude_rad)?;

        // Iterate over the unflagged baseline axis.
        vis_model_fb
            .axis_iter_mut(Axis(1))
            .into_par_iter()
            .zip(uvws.par_iter())
            .enumerate()
            .for_each(|(i_baseline, (mut vis_model_f, &uvw))| {
                let (i_tile1, i_tile2) = self
                    .tile_baseline_flags
                    .unflagged_cross_baseline_to_tile_map[&i_baseline];
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
    /// The reference Gaussian model; see
    /// [`SkyModellerCpu::model_points_reference`].
    #[cfg(test)]
    pub(super) fn model_gaussians_reference(
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
            self.tile_baseline_flags
                .unflagged_cross_baseline_to_tile_map
                .len(),
            "uvws.len() != self.tile_baseline_flags.unflagged_cross_baseline_to_tile_map.len()"
        );

        let beam_responses = self.get_beam_responses(azels, array_latitude_rad)?;

        // Iterate over the unflagged baseline axis.
        vis_model_fb
            .axis_iter_mut(Axis(1))
            .into_par_iter()
            .zip(uvws.par_iter())
            .enumerate()
            .for_each(|(i_baseline, (mut vis_model_f, &uvw))| {
                let (i_tile1, i_tile2) = self
                    .tile_baseline_flags
                    .unflagged_cross_baseline_to_tile_map[&i_baseline];
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
            self.tile_baseline_flags
                .unflagged_cross_baseline_to_tile_map
                .len(),
            "uvws.len() != self.tile_baseline_flags.unflagged_cross_baseline_to_tile_map.len()"
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
                    let (i_tile1, i_tile2) = self
                        .tile_baseline_flags
                        .unflagged_cross_baseline_to_tile_map[&i_baseline];
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
    /// Model point and Gaussian components with the phase term factorised
    /// per tile (the "station phase" trick of DP3's predict): for every
    /// tile, frequency and component a phasor `exp(i (u_t l + v_t m + w_t n))`
    /// is computed once, and a baseline's phasor is the product of its two
    /// tiles' phasors (`cis(a - b) = cis(a) * conj(cis(b))`), so the number of
    /// `sin`/`cos` evaluations drops from baselines x freqs x components to
    /// tiles x freqs x components. The beam-weighted flux density
    /// `B1 F B2^H` is also hoisted out of the baseline loop when it is the
    /// same for every baseline (no beam, or a single unique tile).
    ///
    /// `tile_uvws` are the UVWs of each unflagged tile \[metres\] (not
    /// baselines); their order must match `unflagged_tile_xyzs`.
    /// `gaussian_params`, when given, applies each component's Gaussian
    /// envelope.
    #[allow(clippy::too_many_arguments)]
    fn model_factorised(
        &self,
        mut vis_model_fb: ArrayViewMut2<Jones<f32>>,
        tile_uvws: &[UVW],
        lmns: &[LmnRime],
        fds: ArrayView2<Jones<f64>>,
        beam_responses: ArrayView3<Jones<f64>>,
        gaussian_params: Option<&[GaussianParams]>,
    ) {
        let num_comps = lmns.len();
        if num_comps == 0 {
            return;
        }
        let num_freqs = self.unflagged_fine_chan_freqs.len();
        let num_tiles = tile_uvws.len();
        let pairs = unflagged_tile_pairs(num_tiles);
        assert_eq!(
            pairs.len(),
            vis_model_fb.len_of(Axis(1)),
            "number of unflagged cross baselines != number of unflagged tile pairs"
        );
        assert_eq!(
            pairs.len(),
            self.tile_baseline_flags
                .unflagged_cross_baseline_to_tile_map
                .len()
        );
        assert_eq!(vis_model_fb.len_of(Axis(0)), num_freqs);
        assert_eq!(fds.dim(), (num_freqs, num_comps));
        assert_eq!(beam_responses.len_of(Axis(2)), num_comps);
        if let Some(g) = gaussian_params {
            assert_eq!(g.len(), num_comps);
        }

        // The beam-weighted flux densities, when they don't depend on the
        // baseline: `B1 F B2^H` collapses to `F` without a beam, and with a
        // single unique tile (the usual MWA case of identical dipole gains)
        // every baseline uses the same two beam responses.
        let shared: Option<Array2<Jones<f64>>> =
            if matches!(self.beam.get_beam_type(), BeamType::None) {
                Some(fds.to_owned())
            } else if self.unique_tiles.len() == 1 {
                let mut products = fds.to_owned();
                for (i_freq, mut row) in products.outer_iter_mut().enumerate() {
                    let beam = beam_responses.slice(s![0, self.freq_map[i_freq], ..]);
                    for (j, b) in row.iter_mut().zip(beam.iter()) {
                        *j = *b * *j * b.h();
                    }
                }
                Some(products)
            } else {
                None
            };

        // When the beam differs between tiles, the baseline product
        // `B1 F B2^H` is split as `(B1 L)(B2 L)^H` with `F = L L^H` (Cholesky),
        // so that only one Jones product is left per baseline. That needs
        // every component's flux density to be positive semi-definite (true of
        // physical sources; a negative-flux component falls back to the full
        // per-baseline product).
        let cholesky: Option<Array2<Jones<f64>>> = if shared.is_some() {
            None
        } else {
            let mut l = fds.to_owned();
            let all_psd = l.iter_mut().all(|f| match jones_cholesky(*f) {
                Some(chol) => {
                    *f = chol;
                    true
                }
                None => false,
            });
            all_psd.then_some(l)
        };

        // Components are processed in chunks so that the phasor array
        // (tiles x freqs x components complex doubles) and the half products
        // (unique tiles x freqs x components Jones) stay a sensible size.
        const CHUNK_BYTES: usize = 256 * 1024 * 1024;
        let bytes_per_comp = 16 * num_tiles * num_freqs
            + if cholesky.is_some() {
                64 * self.unique_tiles.len() * num_freqs
            } else {
                0
            };
        let comps_per_chunk = (CHUNK_BYTES / bytes_per_comp).clamp(1, num_comps);

        for chunk_start in (0..num_comps).step_by(comps_per_chunk) {
            let range = chunk_start..(chunk_start + comps_per_chunk).min(num_comps);
            let lmns = &lmns[range.clone()];
            let fds = fds.slice(s![.., range.clone()]);
            let shared = shared.as_ref().map(|p| p.slice(s![.., range.clone()]));
            let beam_responses = beam_responses.slice(s![.., .., range.clone()]);
            let gaussian_params = gaussian_params.map(|g| &g[range.clone()]);
            let phasors = self.tile_phasors(tile_uvws, lmns);
            // Half products `B_t L` per unique tile, frequency and component.
            let half: Option<Array3<Jones<f64>>> = cholesky.as_ref().map(|l| {
                let l = l.slice(s![.., range.clone()]);
                let mut half =
                    Array3::<Jones<f64>>::default((self.unique_tiles.len(), num_freqs, lmns.len()));
                half.outer_iter_mut().into_par_iter().enumerate().for_each(
                    |(i_unique, mut tile_half)| {
                        for (i_freq, mut row) in tile_half.outer_iter_mut().enumerate() {
                            let beam =
                                beam_responses.slice(s![i_unique, self.freq_map[i_freq], ..]);
                            for ((h, b), l) in
                                row.iter_mut().zip(beam.iter()).zip(l.slice(s![i_freq, ..]))
                            {
                                *h = *b * *l;
                            }
                        }
                    },
                );
                half
            });

            vis_model_fb
                .axis_iter_mut(Axis(1))
                .into_par_iter()
                .zip(pairs.par_iter())
                .enumerate()
                .for_each(|(i_baseline, (mut vis_model_f, &(i_tile1, i_tile2)))| {
                    // Beam responses are indexed by unique tile, via the full
                    // tile indices.
                    let (i_tile1_full, i_tile2_full) = self
                        .tile_baseline_flags
                        .unflagged_cross_baseline_to_tile_map[&i_baseline];
                    let i_beam1 = self.tile_index_to_array_index_map[i_tile1_full];
                    let i_beam2 = self.tile_index_to_array_index_map[i_tile2_full];
                    let bl_uvw = tile_uvws[i_tile1] - tile_uvws[i_tile2];

                    for (i_freq, (vis_model, freq)) in vis_model_f
                        .iter_mut()
                        .zip(self.unflagged_fine_chan_freqs)
                        .enumerate()
                    {
                        let ph1 = phasors.slice(s![i_tile1, i_freq, ..]);
                        let ph2 = phasors.slice(s![i_tile2, i_freq, ..]);
                        let ph1 = ph1.as_slice().expect("contiguous");
                        let ph2 = ph2.as_slice().expect("contiguous");
                        // Dimensionless UV for the Gaussian envelopes.
                        let UVW { u, v, .. } = bl_uvw * (*freq / VEL_C);
                        let envelope = |i_comp: usize| -> f64 {
                            match gaussian_params {
                                None => 1.0,
                                Some(g) => gaussian_envelope(&g[i_comp], u, v),
                            }
                        };

                        // Accumulate in double precision, demote at the end.
                        let mut jones_accum: Jones<f64> = Jones::default();
                        match (shared, &half) {
                            (Some(shared), _) => {
                                let shared = shared.slice(s![i_freq, ..]);
                                let shared = shared.as_slice().expect("contiguous");
                                for (i_comp, ((fd, p1), p2)) in
                                    shared.iter().zip(ph1).zip(ph2).enumerate()
                                {
                                    let phasor = p1 * p2.conj();
                                    jones_accum += *fd * (phasor * envelope(i_comp));
                                }
                            }
                            (None, Some(half)) => {
                                let h1 = half.slice(s![i_beam1, i_freq, ..]);
                                let h2 = half.slice(s![i_beam2, i_freq, ..]);
                                let h1 = h1.as_slice().expect("contiguous");
                                let h2 = h2.as_slice().expect("contiguous");
                                for (i_comp, (((h1, h2), p1), p2)) in
                                    h1.iter().zip(h2).zip(ph1).zip(ph2).enumerate()
                                {
                                    let phasor = p1 * p2.conj();
                                    jones_accum += (*h1 * h2.h()) * (phasor * envelope(i_comp));
                                }
                            }
                            (None, None) => {
                                let i_beam_freq = self.freq_map[i_freq];
                                let beam1 = beam_responses.slice(s![i_beam1, i_beam_freq, ..]);
                                let beam2 = beam_responses.slice(s![i_beam2, i_beam_freq, ..]);
                                let comp_fds = fds.slice(s![i_freq, ..]);
                                for (i_comp, ((((fd, b1), b2), p1), p2)) in comp_fds
                                    .iter()
                                    .zip(beam1.iter())
                                    .zip(beam2.iter())
                                    .zip(ph1)
                                    .zip(ph2)
                                    .enumerate()
                                {
                                    let phasor = p1 * p2.conj();
                                    jones_accum +=
                                        (*b1 * *fd * b2.h()) * (phasor * envelope(i_comp));
                                }
                            }
                        }
                        *vis_model += Jones::from(jones_accum);
                    }
                });
        }
    }

    /// The per-tile phasors `exp(i (u l + v m + w n))` for every unflagged
    /// tile, unflagged fine channel and component, as a `[tile][freq][comp]`
    /// array. `tile_uvws` are in metres; they are scaled to wavelengths here.
    fn tile_phasors(&self, tile_uvws: &[UVW], lmns: &[LmnRime]) -> Array3<c64> {
        let mut phasors = Array3::<c64>::zeros((
            tile_uvws.len(),
            self.unflagged_fine_chan_freqs.len(),
            lmns.len(),
        ));
        phasors
            .outer_iter_mut()
            .into_par_iter()
            .zip(tile_uvws.par_iter())
            .for_each(|(mut tile_phasors, tile_uvw)| {
                for (mut freq_phasors, freq) in tile_phasors
                    .outer_iter_mut()
                    .zip(self.unflagged_fine_chan_freqs)
                {
                    let UVW { u, v, w } = *tile_uvw * (*freq / VEL_C);
                    for (phasor, LmnRime { l, m, n }) in freq_phasors.iter_mut().zip(lmns) {
                        *phasor = c64::cis(u * l + v * m + w * n);
                    }
                }
            });
        phasors
    }

    /// The LST, the UVWs of each unflagged tile \[metres\] and the array
    /// latitude at `timestamp`, precessed if requested.
    fn get_lst_tile_uvws_latitude(&self, timestamp: Epoch) -> (f64, Vec<UVW>, f64) {
        let (lst, xyzs, latitude) = self.get_lst_xyzs_latitude(timestamp);
        let hadec = self.phase_centre.to_hadec(lst);
        let (s_ha, c_ha) = hadec.ha.sin_cos();
        let (s_dec, c_dec) = hadec.dec.sin_cos();
        let tile_uvws = xyzs
            .iter()
            .map(|xyz| UVW::from_xyz_inner(*xyz, s_ha, c_ha, s_dec, c_dec))
            .collect();
        (lst, tile_uvws, latitude)
    }

    fn get_lst_xyzs_latitude(&self, timestamp: Epoch) -> (f64, Cow<'_, [XyzGeodetic]>, f64) {
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

        (lst, xyzs, latitude)
    }
}

/// The unflagged tile index pairs `(i, j)`, `i < j`, in the baseline order
/// used throughout (the same order as `xyzs_to_cross_uvws`).
fn unflagged_tile_pairs(num_tiles: usize) -> Vec<(usize, usize)> {
    let mut pairs = Vec::with_capacity(num_tiles * num_tiles.saturating_sub(1) / 2);
    for i in 0..num_tiles {
        for j in i + 1..num_tiles {
            pairs.push((i, j));
        }
    }
    pairs
}

/// Baseline UVWs (`tile1 - tile2`) from tile UVWs, in the same order as
/// `xyzs_to_cross_uvws`.
fn cross_uvws(tile_uvws: &[UVW]) -> Vec<UVW> {
    unflagged_tile_pairs(tile_uvws.len())
        .into_iter()
        .map(|(i, j)| tile_uvws[i] - tile_uvws[j])
        .collect()
}

/// The lower-triangular Cholesky factor `L` of a Hermitian positive
/// semi-definite Jones matrix `F` (`F = L L^H`), or `None` if `F` isn't one.
/// A flux-density Jones matrix `[[I+Q, U+iV], [U-iV, I-Q]]` is one whenever
/// `I >= sqrt(Q^2 + U^2 + V^2)`, i.e. for every physical source.
fn jones_cholesky(f: Jones<f64>) -> Option<Jones<f64>> {
    // Tolerate rounding at the semi-definite boundary.
    let tol = 1e-12 * (f[0].norm() + f[3].norm()).max(f64::MIN_POSITIVE);
    if f[0].im.abs() > tol || f[3].im.abs() > tol || (f[1] - f[2].conj()).norm() > tol {
        return None;
    }
    let f00 = f[0].re;
    let f11 = f[3].re;
    if f00 < -tol || f11 < -tol {
        return None;
    }
    let (a, b) = if f00 <= tol {
        if f[2].norm() > tol {
            return None;
        }
        (0.0, c64::new(0.0, 0.0))
    } else {
        let a = f00.sqrt();
        (a, f[2] / a)
    };
    let c2 = f11 - b.norm_sqr();
    if c2 < -tol {
        return None;
    }
    let c = c2.max(0.0).sqrt();
    Some(Jones::from([
        c64::new(a, 0.0),
        c64::new(0.0, 0.0),
        b,
        c64::new(c, 0.0),
    ]))
}

/// A Gaussian component's envelope at dimensionless `u`, `v`.
fn gaussian_envelope(g_params: &GaussianParams, u: f64, v: f64) -> f64 {
    let (s_pa, c_pa) = g_params.pa.sin_cos();
    let k_x = u * s_pa + v * c_pa;
    let k_y = u * c_pa - v * s_pa;
    (GAUSSIAN_EXP_CONST * (g_params.maj.powi(2) * k_x.powi(2) + g_params.min.powi(2) * k_y.powi(2)))
        .exp()
}

impl<'a> super::SkyModeller<'a> for SkyModellerCpu<'a> {
    // TODO: this impl could be moved into the trait
    fn model_timestep(
        &self,
        timestamp: Epoch,
    ) -> Result<(Array2<Jones<f32>>, Vec<UVW>), ModelError> {
        let num_cross_baselines = self
            .tile_baseline_flags
            .unflagged_cross_baseline_to_tile_map
            .len();
        let mut vis_fb = Array2::zeros((self.unflagged_fine_chan_freqs.len(), num_cross_baselines));
        let uvws = self.model_timestep_with(timestamp, vis_fb.view_mut())?;
        Ok((vis_fb, uvws))
    }

    fn model_timestep_with(
        &self,
        timestamp: Epoch,
        mut vis_fb: ArrayViewMut2<Jones<f32>>,
    ) -> Result<Vec<UVW>, ModelError> {
        let (lst, tile_uvws, latitude) = self.get_lst_tile_uvws_latitude(timestamp);
        let uvws = cross_uvws(&tile_uvws);
        let shapelet_uvws = self
            .components
            .shapelets
            .get_shapelet_uvws(lst, self.unflagged_tile_xyzs);

        self.model_points(vis_fb.view_mut(), &tile_uvws, lst, latitude)?;
        self.model_gaussians(vis_fb.view_mut(), &tile_uvws, lst, latitude)?;
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

    fn model_timestep_autos_with(
        &self,
        timestamp: Epoch,
        mut vis_model_fb: ArrayViewMut2<Jones<f32>>,
    ) -> Result<(), ModelError> {
        assert_eq!(
            vis_model_fb.len_of(Axis(1)),
            self.tile_baseline_flags
                .tile_to_unflagged_auto_index_map
                .len(),
            "vis_fb.len_of(Axis(1)) != self.tile_baseline_flags.tile_to_unflagged_auto_index_map.len()"
        );

        let (lst, latitude) = if self.precess {
            let precession_info = precess_time(
                self.array_longitude,
                self.array_latitude,
                self.phase_centre,
                timestamp,
                self.dut1,
            );
            debug!(
                "Modelling autos for GPS timestamp {}, LMST {}°, J2000 LMST {}°",
                timestamp.to_gpst_seconds(),
                precession_info.lmst.to_degrees(),
                precession_info.lmst_j2000.to_degrees()
            );
            (
                precession_info.lmst_j2000,
                precession_info.array_latitude_j2000,
            )
        } else {
            let lst = get_lmst(self.array_longitude, timestamp, self.dut1);
            debug!(
                "Modelling autos for GPS timestamp {}, LMST {}°",
                timestamp.to_gpst_seconds(),
                lst.to_degrees()
            );
            (lst, self.array_latitude)
        };

        macro_rules! model {
            ($fds:expr, $beam_responses:expr) => {{
                // Iterate over the unflagged baseline axis.
                vis_model_fb
                    .axis_iter_mut(Axis(1))
                    .into_par_iter()
                    .enumerate()
                    .for_each(|(i_unflagged_tile, mut vis_model_f)| {
                        let i_tile = self.tile_index_to_array_index_map[i_unflagged_tile];

                        // Unflagged fine-channel axis.
                        vis_model_f
                            .iter_mut()
                            .zip($fds.outer_iter())
                            .zip(self.unflagged_fine_chan_freqs)
                            .enumerate()
                            .for_each(|(i_freq, ((vis_model, comp_fds), _freq))| {
                                // Access the beam-deduplicated-freq index.
                                let i_freq = self.freq_map[i_freq];
                                let tile_beam = $beam_responses.slice(s![i_tile, i_freq, ..]);

                                // Accumulate the double-precision visibilities
                                // into a double-precision Jones matrix before
                                // putting that into the `vis_model_fb`.
                                let mut jones_accum: Jones<f64> = Jones::default();

                                comp_fds.iter().zip(tile_beam).for_each(|(comp_fd, beam)| {
                                    jones_accum += *beam * *comp_fd * beam.h();
                                });
                                // Demote to single precision now that all
                                // operations are done.
                                *vis_model += Jones::from(jones_accum);
                            });
                    });
            }};
        }

        // Point sources.
        if !self.components.points.radecs.is_empty() {
            let fds = &self.components.points.flux_densities;
            let azels = &self.components.points.get_azels_mwa_parallel(lst, latitude);
            let beam_responses = self.get_beam_responses(azels, latitude)?;
            model!(fds, beam_responses);
        }
        // Gaussians.
        if !self.components.gaussians.flux_densities.is_empty() {
            let fds = &self.components.gaussians.flux_densities;
            let azels = &self
                .components
                .gaussians
                .get_azels_mwa_parallel(lst, latitude);
            let beam_responses = self.get_beam_responses(azels, latitude)?;
            model!(fds, beam_responses);
        }
        // Shapelets sources.
        if !self.components.shapelets.flux_densities.is_empty() {
            let fds = &self.components.shapelets.flux_densities;
            let azels = &self
                .components
                .shapelets
                .get_azels_mwa_parallel(lst, latitude);
            let beam_responses = self.get_beam_responses(azels, latitude)?;
            model!(fds, beam_responses);
        }

        Ok(())
    }
}
