// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to generate sky-model visibilities.

use std::{
    collections::{HashMap, HashSet},
    f64::consts::{FRAC_PI_2, LN_2},
};

use hifitime::{Duration, Epoch};
use marlu::{
    c64,
    pos::xyz::xyzs_to_cross_uvws,
    precession::{get_lmst, precess_time},
    Jones, LmnRime, RADec, XyzGeodetic, UVW,
};
use ndarray::{parallel::prelude::*, prelude::*};
use num_complex::Complex;

use super::ModelError;
use crate::{
    beam::Beam,
    constants::*,
    shapelets,
    srclist::{ComponentList, GaussianParams, PerComponentParams},
};

const GAUSSIAN_EXP_CONST: f64 = -(FRAC_PI_2 * FRAC_PI_2) / LN_2;

pub(crate) struct SkyModellerCpu<'a> {
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

    /// The [XyzGeodetic] positions of each of the unflagged tiles.
    pub(super) unflagged_tile_xyzs: &'a [XyzGeodetic],
    pub(super) flagged_tiles: &'a HashSet<usize>,
    pub(super) unflagged_baseline_to_tile_map: HashMap<usize, (usize, usize)>,

    pub(super) components: ComponentList,
}

impl<'a> SkyModellerCpu<'a> {
    /// This function is mostly used for testing. For a single timestep, over
    /// the already-provided baselines and frequencies, generate visibilities
    /// for each specified sky-model point-source component.
    ///
    /// `vis_model_slice`: A mutable view into an `ndarray`. Rather than
    /// returning an array from this function, modelled visibilities are written
    /// into this array. This slice *must* have dimensions `[n1][n2]`, where
    /// `n1` is number of unflagged frequencies and `n2` is the number of
    /// unflagged cross correlation baselines.
    ///
    /// `uvws`: The [UVW] coordinates of each baseline \[metres\]. This should
    /// be the same length as `vis_model_slice`'s first axis.
    ///
    /// `lst_rad`: The local sidereal time in \[radians\].
    ///
    /// `array_latitude_rad`: The latitude of the array/telescope/interferometer
    /// in \[radians\].
    pub(super) fn model_points(
        &self,
        mut vis_model_slice: ArrayViewMut2<Jones<f32>>,
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
            vis_model_slice.len_of(Axis(1)),
            uvws.len(),
            "vis_model_slice.len_of(Axis(1)) != uvws.len()"
        );
        assert_eq!(
            vis_model_slice.len_of(Axis(0)),
            self.unflagged_fine_chan_freqs.len(),
            "vis_model_slice.len_of(Axis(0)) != self.unflagged_fine_chan_freqs.len()"
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

        // Get beam responses for all unflagged tiles.
        let num_tiles = self.beam.get_num_tiles();
        let mut beam_responses =
            Array3::zeros((num_tiles, self.unflagged_fine_chan_freqs.len(), azels.len()));
        for i_tile in 0..num_tiles {
            if self.flagged_tiles.contains(&i_tile) {
                continue;
            }

            for (i_freq, freq) in self.unflagged_fine_chan_freqs.iter().enumerate() {
                let mut view = beam_responses.slice_mut(s![i_tile, i_freq, ..]);
                let slice = view.as_slice_mut().expect("is contiguous");
                self.beam.calc_jones_array_inner(
                    azels,
                    *freq,
                    Some(i_tile),
                    array_latitude_rad,
                    slice,
                )?;
            }
        }

        // Iterate over the unflagged baseline axis.
        vis_model_slice
            .axis_iter_mut(Axis(1))
            .into_par_iter()
            .zip(uvws.par_iter())
            .enumerate()
            .for_each(|(i_baseline, (mut model_bl_axis, &uvw))| {
                let (i_tile1, i_tile2) = self.unflagged_baseline_to_tile_map[&i_baseline];

                // Unflagged fine-channel axis.
                model_bl_axis
                    .iter_mut()
                    .zip(fds.outer_iter())
                    .zip(self.unflagged_fine_chan_freqs)
                    .zip(beam_responses.slice(s![i_tile1, .., ..]).outer_iter())
                    .zip(beam_responses.slice(s![i_tile2, .., ..]).outer_iter())
                    .for_each(
                        |((((model_vis, comp_fds), freq), tile1_beam), tile2_beam)| {
                            // Divide UVW by lambda to make UVW dimensionless.
                            let UVW { u, v, w } = uvw * *freq / VEL_C;

                            // Accumulate the double-precision visibilities into a
                            // double-precision Jones matrix before putting that into
                            // the `vis_model_slice`.
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
                            *model_vis += Jones::from(jones_accum);
                        },
                    );
            });
        Ok(())
    }

    /// This function is mostly used for testing. For a single timestep, over
    /// the already-provided baselines and frequencies, generate visibilities
    /// for each specified sky-model Gaussian-source component.
    ///
    /// `vis_model_slice`: A mutable view into an `ndarray`. Rather than
    /// returning an array from this function, modelled visibilities are written
    /// into this array. This slice *must* have dimensions `[n1][n2]`, where
    /// `n1` is number of unflagged frequencies and `n2` is the number of
    /// unflagged cross correlation baselines.
    ///
    /// `uvws`: The [UVW] coordinates of each baseline \[metres\]. This should
    /// be the same length as `vis_model_slice`'s first axis.
    ///
    /// `lst_rad`: The local sidereal time in \[radians\].
    ///
    /// `array_latitude_rad`: The latitude of the array/telescope/interferometer
    /// in \[radians\].
    pub(super) fn model_gaussians(
        &self,
        mut vis_model_slice: ArrayViewMut2<Jones<f32>>,
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
            vis_model_slice.len_of(Axis(1)),
            uvws.len(),
            "vis_model_slice.len_of(Axis(1)) != uvws.len()"
        );
        assert_eq!(
            vis_model_slice.len_of(Axis(0)),
            self.unflagged_fine_chan_freqs.len(),
            "vis_model_slice.len_of(Axis(0)) != self.unflagged_fine_chan_freqs.len()"
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

        // Get beam responses for all unflagged tiles.
        let num_tiles = self.beam.get_num_tiles();
        let mut beam_responses =
            Array3::zeros((num_tiles, self.unflagged_fine_chan_freqs.len(), azels.len()));
        for i_tile in 0..num_tiles {
            if self.flagged_tiles.contains(&i_tile) {
                continue;
            }

            for (i_freq, freq) in self.unflagged_fine_chan_freqs.iter().enumerate() {
                let mut view = beam_responses.slice_mut(s![i_tile, i_freq, ..]);
                let slice = view.as_slice_mut().expect("is contiguous");
                self.beam.calc_jones_array_inner(
                    azels,
                    *freq,
                    Some(i_tile),
                    array_latitude_rad,
                    slice,
                )?;
            }
        }

        // Iterate over the unflagged baseline axis.
        vis_model_slice
            .axis_iter_mut(Axis(1))
            .into_par_iter()
            .zip(uvws.par_iter())
            .enumerate()
            .for_each(|(i_baseline, (mut model_bl_axis, &uvw))| {
                let (i_tile1, i_tile2) = self.unflagged_baseline_to_tile_map[&i_baseline];

                // Unflagged fine-channel axis.
                model_bl_axis
                    .iter_mut()
                    .zip(fds.outer_iter())
                    .zip(self.unflagged_fine_chan_freqs)
                    .zip(beam_responses.slice(s![i_tile1, .., ..]).outer_iter())
                    .zip(beam_responses.slice(s![i_tile2, .., ..]).outer_iter())
                    .for_each(
                        |((((model_vis, comp_fds), freq), tile1_beam), tile2_beam)| {
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

                            // Accumulate the double-precision visibilities into a
                            // double-precision Jones matrix before putting that into
                            // the `vis_model_slice`.
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
                            *model_vis += Jones::from(jones_accum);
                        },
                    );
            });
        Ok(())
    }

    /// This function is mostly used for testing. For a single timestep, over
    /// the already-provided baselines and frequencies, generate visibilities
    /// for each specified sky-model shapelet-source component.
    ///
    /// `vis_model_slice`: A mutable view into an `ndarray`. Rather than
    /// returning an array from this function, modelled visibilities are written
    /// into this array. This slice *must* have dimensions `[n1][n2]`, where
    /// `n1` is number of unflagged frequencies and `n2` is the number of
    /// unflagged cross correlation baselines.
    ///
    /// `uvws`: The [UVW] coordinates of each baseline \[metres\]. This should
    /// be the same length as `vis_model_slice`'s first axis.
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
        mut vis_model_slice: ArrayViewMut2<Jones<f32>>,
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
            vis_model_slice.len_of(Axis(1)),
            uvws.len(),
            "vis_model_slice.len_of(Axis(1)) != uvws.len()"
        );
        assert_eq!(
            vis_model_slice.len_of(Axis(0)),
            self.unflagged_fine_chan_freqs.len(),
            "vis_model_slice.len_of(Axis(0)) != self.unflagged_fine_chan_freqs.len()"
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
            vis_model_slice.len_of(Axis(1)),
            shapelet_uvws.len_of(Axis(0)),
            "vis_model_slice.len_of(Axis(1)) != shapelet_uvws.len_of(Axis(0))"
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

        // Get beam responses for all unflagged tiles.
        let num_tiles = self.beam.get_num_tiles();
        let mut beam_responses =
            Array3::zeros((num_tiles, self.unflagged_fine_chan_freqs.len(), azels.len()));
        for i_tile in 0..num_tiles {
            if self.flagged_tiles.contains(&i_tile) {
                continue;
            }

            for (i_freq, freq) in self.unflagged_fine_chan_freqs.iter().enumerate() {
                let mut view = beam_responses.slice_mut(s![i_tile, i_freq, ..]);
                let slice = view.as_slice_mut().expect("is contiguous");
                self.beam.calc_jones_array_inner(
                    azels,
                    *freq,
                    Some(i_tile),
                    array_latitude_rad,
                    slice,
                )?;
            }
        }

        // Iterate over the unflagged baseline axis.
        vis_model_slice
            .axis_iter_mut(Axis(1))
            .into_par_iter()
            .zip(uvws.par_iter())
            .zip(shapelet_uvws.outer_iter())
            .enumerate()
            .for_each(
                |(i_baseline, ((mut model_bl_axis, &uvw), shapelet_uvws_per_comp))| {
                    let (i_tile1, i_tile2) = self.unflagged_baseline_to_tile_map[&i_baseline];

                    // Preallocate a vector for the envelopes.
                    let mut envelopes = vec![c64::default(); gaussian_params.len()];

                    // Unflagged fine-channel axis.
                    model_bl_axis
                        .iter_mut()
                        .zip(fds.outer_iter())
                        .zip(self.unflagged_fine_chan_freqs)
                        .zip(beam_responses.slice(s![i_tile1, .., ..]).outer_iter())
                        .zip(beam_responses.slice(s![i_tile2, .., ..]).outer_iter())
                        .for_each(
                            |((((model_vis, comp_fds), freq), tile1_beam), tile2_beam)| {
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
                                        let const_x =
                                            maj * SQRT_FRAC_PI_SQ_2_LN_2 / shapelets::SBF_DX;
                                        let const_y =
                                            -min * SQRT_FRAC_PI_SQ_2_LN_2 / shapelets::SBF_DX;
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
                                                            shapelets::SBF_L * coeff.n1 + x_pos_int,
                                                        );
                                                    let x_high = shapelets::SHAPELET_BASIS_VALUES
                                                        .get_unchecked(
                                                            shapelets::SBF_L * coeff.n1
                                                                + x_pos_int
                                                                + 1,
                                                        );
                                                    let u_value = x_low
                                                        + (x_high - x_low)
                                                            * (x_pos - x_pos.floor());

                                                    let y_low = shapelets::SHAPELET_BASIS_VALUES
                                                        .get_unchecked(
                                                            shapelets::SBF_L * coeff.n2 + y_pos_int,
                                                        );
                                                    let y_high = shapelets::SHAPELET_BASIS_VALUES
                                                        .get_unchecked(
                                                            shapelets::SBF_L * coeff.n2
                                                                + y_pos_int
                                                                + 1,
                                                        );
                                                    let v_value = y_low
                                                        + (y_high - y_low)
                                                            * (y_pos - y_pos.floor());

                                                    envelope_acc
                                                        + I_POWER_TABLE.get_unchecked(
                                                            (coeff.n1 + coeff.n2) % 4,
                                                        ) * f_hat
                                                            * u_value
                                                            * v_value
                                                }
                                            },
                                        )
                                    });

                                // Accumulate the double-precision visibilities into a
                                // double-precision Jones matrix before putting that
                                // into the `vis_model_slice`.
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
                                *model_vis += Jones::from(jones_accum);
                            },
                        );
                },
            );
        Ok(())
    }
}

impl<'a> super::SkyModeller<'a> for SkyModellerCpu<'a> {
    fn update_source_list(
        &mut self,
        _source_list: &crate::srclist::SourceList,
        _phase_centre: RADec,
    ) -> Result<(), ModelError> {
        todo!()
    }

    fn update_with_a_source(
        &mut self,
        _source: &crate::srclist::IonoSource,
        _phase_centre: RADec,
    ) -> Result<(), ModelError> {
        todo!()
    }

    fn model_timestep(
        &mut self,
        mut vis_model_slice: ArrayViewMut2<Jones<f32>>,
        timestamp: Epoch,
    ) -> Result<Vec<UVW>, ModelError> {
        let (uvws, lst, latitude) = if self.precess {
            let precession_info = precess_time(
                self.array_longitude,
                self.array_latitude,
                self.phase_centre,
                timestamp,
                self.dut1,
            );
            // Apply precession to the tile XYZ positions.
            let precessed_tile_xyzs = precession_info.precess_xyz(self.unflagged_tile_xyzs);
            let uvws = xyzs_to_cross_uvws(
                &precessed_tile_xyzs,
                self.phase_centre.to_hadec(precession_info.lmst_j2000),
            );
            (
                uvws,
                precession_info.lmst_j2000,
                precession_info.array_latitude_j2000,
            )
        } else {
            let lst = get_lmst(self.array_longitude, timestamp, self.dut1);
            let uvws =
                xyzs_to_cross_uvws(self.unflagged_tile_xyzs, self.phase_centre.to_hadec(lst));
            (uvws, lst, self.array_latitude)
        };
        let shapelet_uvws = self
            .components
            .shapelets
            .get_shapelet_uvws(lst, self.unflagged_tile_xyzs);

        self.model_points(vis_model_slice.view_mut(), &uvws, lst, latitude)?;
        self.model_gaussians(vis_model_slice.view_mut(), &uvws, lst, latitude)?;
        self.model_shapelets(
            vis_model_slice.view_mut(),
            &uvws,
            shapelet_uvws.view(),
            lst,
            latitude,
        )?;

        Ok(uvws)
    }
}
