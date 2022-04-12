// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to generate sky-model visibilities.

use super::ModelError;

use std::collections::HashMap;
use std::f64::consts::{FRAC_PI_2, LN_2};

use hifitime::Epoch;
use marlu::{
    pos::xyz::xyzs_to_cross_uvws_parallel,
    precession::{get_lmst, precess_time},
    Complex, Jones, RADec, XyzGeodetic, UVW,
};
use ndarray::{parallel::prelude::*, prelude::*};

use crate::{
    constants::*,
    math::{cexp, exp},
};
use mwa_hyperdrive_beam::Beam;
use mwa_hyperdrive_common::{hifitime, marlu, ndarray, shapelets};
use mwa_hyperdrive_srclist::{ComponentList, GaussianParams, PerComponentParams};

const GAUSSIAN_EXP_CONST: f64 = -(FRAC_PI_2 * FRAC_PI_2) / LN_2;

pub(crate) struct SkyModellerCpu<'a> {
    pub(super) beam: &'a dyn Beam,

    /// The phase centre used for all modelling.
    pub(super) phase_centre: RADec,
    /// The longitude of the array we're using \[radians\].
    pub(super) array_longitude: f64,
    /// The latitude of the array we're using \[radians\].
    pub(super) array_latitude: f64,
    /// Shift baselines and LSTs back to J2000.
    pub(super) precess: bool,

    pub(super) unflagged_fine_chan_freqs: &'a [f64],

    /// The [XyzGeodetic] positions of each of the unflagged tiles.
    pub(super) unflagged_tile_xyzs: &'a [XyzGeodetic],
    pub(super) unflagged_baseline_to_tile_map: HashMap<usize, (usize, usize)>,

    pub(super) components: ComponentList,
}

impl<'a> SkyModellerCpu<'a> {
    /// For a single timestep, over the already-provided baselines and
    /// frequencies, generate visibilities for each specified sky-model
    /// point-source component.
    ///
    /// `vis_model_slice`: A mutable view into an `ndarray`. Rather than
    /// returning an array from this function, modelled visibilities are written
    /// into this array. This slice *must* have dimensions `[n1][n2]`, where
    /// `n1` is number of unflagged cross correlation baselines and `n2` is the
    /// number of unflagged frequencies.
    ///
    /// `uvws`: The [UVW] coordinates of each baseline \[metres\]. This should
    /// be the same length as `vis_model_slice`'s first axis.
    ///
    /// `lst_rad`: The local sidereal time in \[radians\].
    pub(super) fn model_points_inner(
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
            vis_model_slice.len_of(Axis(0)),
            uvws.len(),
            "vis_model_slice.len_of(Axis(0)) != uvws.len()"
        );
        assert_eq!(
            vis_model_slice.len_of(Axis(1)),
            self.unflagged_fine_chan_freqs.len(),
            "vis_model_slice.len_of(Axis(1)) != self.unflagged_fine_chan_freqs.len()"
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

        // Get beam-attenuated flux densities.
        let num_tiles = self.beam.get_num_tiles();
        let mut beam_responses =
            Array3::zeros((num_tiles, self.unflagged_fine_chan_freqs.len(), azels.len()));
        for i_tile in 0..num_tiles {
            for (i_freq, freq) in self.unflagged_fine_chan_freqs.iter().enumerate() {
                let responses = self
                    .beam
                    .calc_jones_array(azels, *freq, i_tile)
                    .expect("Couldn't get beam responses");
                beam_responses
                    .slice_mut(s![i_tile, i_freq, ..])
                    .assign(&Array1::from(responses));
            }
        }

        // Iterate over the unflagged baseline axis.
        vis_model_slice
            .outer_iter_mut()
            .into_par_iter()
            .zip(uvws.par_iter())
            .enumerate()
            .for_each(|(i_baseline, (mut model_bl_axis, uvw_metres))| {
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
                            let uvw = *uvw_metres * *freq / VEL_C;

                            // Accumulate the double-precision visibilities into a
                            // double-precision Jones matrix before putting that into
                            // the `vis_model_slice`.
                            let mut jones_accum: Jones<f64> = Jones::default();

                            comp_fds
                                .iter()
                                .zip(tile1_beam)
                                .zip(tile2_beam)
                                .zip(lmns.iter())
                                .for_each(|(((comp_fd, beam_1), beam_2), lmn)| {
                                    let arg = uvw.u * lmn.l + uvw.v * lmn.m + uvw.w * lmn.n;
                                    let phase = cexp(arg);

                                    let mut fd = *beam_1 * *comp_fd;
                                    fd *= beam_2.h();
                                    // `fd` now contains the beam-attenuated
                                    // instrumental flux density.

                                    jones_accum += fd * phase;
                                });
                            // Demote to single precision now that all operations are
                            // done.
                            *model_vis += Jones::from(jones_accum);
                        },
                    );
            });
        Ok(())
    }

    /// For a single timestep, over the already-provided baselines and
    /// frequencies, generate visibilities for each specified sky-model
    /// Gaussian-source component.
    ///
    /// `vis_model_slice`: A mutable view into an `ndarray`. Rather than
    /// returning an array from this function, modelled visibilities are written
    /// into this array. This slice *must* have dimensions `[n1][n2]`, where
    /// `n1` is number of unflagged cross correlation baselines and `n2` is the
    /// number of unflagged frequencies.
    ///
    /// `uvws`: The [UVW] coordinates of each baseline \[metres\]. This should
    /// be the same length as `vis_model_slice`'s first axis.
    ///
    /// `lst_rad`: The local sidereal time in \[radians\].
    pub(super) fn model_gaussians_inner(
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
            vis_model_slice.len_of(Axis(0)),
            uvws.len(),
            "vis_model_slice.len_of(Axis(0)) != uvws.len()"
        );
        assert_eq!(
            vis_model_slice.len_of(Axis(1)),
            self.unflagged_fine_chan_freqs.len(),
            "vis_model_slice.len_of(Axis(1)) != self.unflagged_fine_chan_freqs.len()"
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

        // Get beam-attenuated flux densities.
        let num_tiles = self.beam.get_num_tiles();
        let mut beam_responses =
            Array3::zeros((num_tiles, self.unflagged_fine_chan_freqs.len(), azels.len()));
        for i_tile in 0..num_tiles {
            for (i_freq, freq) in self.unflagged_fine_chan_freqs.iter().enumerate() {
                let responses = self.beam.calc_jones_array(azels, *freq, i_tile)?;
                beam_responses
                    .slice_mut(s![i_tile, i_freq, ..])
                    .assign(&Array1::from(responses));
            }
        }

        // Iterate over the unflagged baseline axis.
        vis_model_slice
            .outer_iter_mut()
            .into_par_iter()
            .zip(uvws.par_iter())
            .enumerate()
            .for_each(|(i_baseline, (mut model_bl_axis, uvw_metres))| {
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
                            let uvw = *uvw_metres * *freq / VEL_C;

                            // Now that we have the UVW coordinates, we can determine
                            // each source component's envelope.
                            let envelopes = gaussian_params.iter().map(|g_params| {
                                let (s_pa, c_pa) = g_params.pa.sin_cos();
                                // Temporary variables for clarity.
                                let k_x = uvw.u * s_pa + uvw.v * c_pa;
                                let k_y = uvw.u * c_pa - uvw.v * s_pa;
                                exp(GAUSSIAN_EXP_CONST
                                    * (g_params.maj.powi(2) * k_x.powi(2)
                                        + g_params.min.powi(2) * k_y.powi(2)))
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
                                .for_each(|((((comp_fd, beam_1), beam_2), lmn), envelope)| {
                                    let arg = uvw.u * lmn.l + uvw.v * lmn.m + uvw.w * lmn.n;
                                    let phase = cexp(arg) * envelope;

                                    let mut fd = *beam_1 * *comp_fd;
                                    fd *= beam_2.h();
                                    // `fd` now contains the beam-attenuated
                                    // instrumental flux density.

                                    jones_accum += fd * phase;
                                });
                            // Demote to single precision now that all operations are
                            // done.
                            *model_vis += Jones::from(jones_accum);
                        },
                    );
            });
        Ok(())
    }

    /// For a single timestep, over the already-provided baselines and
    /// frequencies, generate visibilities for each specified sky-model
    /// shapelet-source component.
    ///
    /// `vis_model_slice`: A mutable view into an `ndarray`. Rather than
    /// returning an array from this function, modelled visibilities are written
    /// into this array. This slice *must* have dimensions `[n1][n2]`, where
    /// `n1` is number of unflagged cross correlation baselines and `n2` is the
    /// number of unflagged frequencies.
    ///
    /// `uvws`: The [UVW] coordinates of each baseline \[metres\]. This should
    /// be the same length as `vis_model_slice`'s first axis.
    ///
    /// `shapelet_uvws` are special UVWs generated as if each shapelet component was
    /// at the phase centre \[metres\]. The first axis is unflagged baseline, the
    /// second shapelet component.
    ///
    /// `lst_rad`: The local sidereal time in \[radians\].
    pub(super) fn model_shapelets_inner(
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
            vis_model_slice.len_of(Axis(0)),
            uvws.len(),
            "vis_model_slice.len_of(Axis(0)) != uvws.len()"
        );
        assert_eq!(
            vis_model_slice.len_of(Axis(1)),
            self.unflagged_fine_chan_freqs.len(),
            "vis_model_slice.len_of(Axis(1)) != self.unflagged_fine_chan_freqs.len()"
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
            vis_model_slice.len_of(Axis(0)),
            shapelet_uvws.len_of(Axis(0)),
            "vis_model_slice.len_of(Axis(0)) != shapelet_uvws.len_of(Axis(0))"
        );
        assert_eq!(
            fds.len_of(Axis(1)),
            shapelet_uvws.len_of(Axis(1)),
            "fds.len_of(Axis(1)) != shapelet_uvws.len_of(Axis(1))"
        );

        // Get beam-attenuated flux densities.
        let num_tiles = self.beam.get_num_tiles();
        let mut beam_responses =
            Array3::zeros((num_tiles, self.unflagged_fine_chan_freqs.len(), azels.len()));
        for i_tile in 0..num_tiles {
            for (i_freq, freq) in self.unflagged_fine_chan_freqs.iter().enumerate() {
                let responses = self.beam.calc_jones_array(azels, *freq, i_tile)?;
                beam_responses
                    .slice_mut(s![i_tile, i_freq, ..])
                    .assign(&Array1::from(responses));
            }
        }

        // Iterate over the unflagged baseline axis.
        vis_model_slice
            .outer_iter_mut()
            .into_par_iter()
            .zip(uvws.par_iter())
            .zip(shapelet_uvws.outer_iter())
            .enumerate()
            .for_each(
                |(i_baseline, ((mut model_bl_axis, uvw_metres), shapelet_uvws_per_comp))| {
                    let (i_tile1, i_tile2) = self.unflagged_baseline_to_tile_map[&i_baseline];

                    // Preallocate a vector for the envelopes.
                    let mut envelopes: Vec<Complex<f64>> =
                        vec![Complex::default(); gaussian_params.len()];

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
                                let lambda = VEL_C / freq;
                                let uvw = *uvw_metres / lambda;

                                // Now that we have the UVW coordinates, we can
                                // determine each source component's envelope.
                                envelopes
                                    .iter_mut()
                                    .zip(gaussian_params.iter())
                                    .zip(shapelet_coeffs.iter())
                                    .zip(shapelet_uvws_per_comp.iter())
                                    .for_each(|(((envelope, g_params), coeffs), shapelet_uvw)| {
                                        let shapelet_u = shapelet_uvw.u / lambda;
                                        let shapelet_v = shapelet_uvw.v / lambda;
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
                                                        + shapelets::I_POWER_TABLE.get_unchecked(
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
                                    .for_each(|((((comp_fd, beam_1), beam_2), lmn), envelope)| {
                                        let arg = uvw.u * lmn.l + uvw.v * lmn.m + uvw.w * lmn.n;
                                        let phase = cexp(arg) * envelope;

                                        let mut fd = *beam_1 * *comp_fd;
                                        fd *= beam_2.h();
                                        // `fd` now contains the beam-attenuated
                                        // instrumental flux density.

                                        jones_accum += fd * phase;
                                    });
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
    fn model_timestep(
        &self,
        mut vis_model_slice: ArrayViewMut2<Jones<f32>>,
        timestamp: Epoch,
    ) -> Result<Vec<UVW>, ModelError> {
        let (uvws, lst, latitude) = if self.precess {
            let precession_info = precess_time(
                self.phase_centre,
                timestamp,
                self.array_longitude,
                self.array_latitude,
            );
            // Apply precession to the tile XYZ positions.
            let precessed_tile_xyzs =
                precession_info.precess_xyz_parallel(self.unflagged_tile_xyzs);
            let uvws = xyzs_to_cross_uvws_parallel(
                &precessed_tile_xyzs,
                self.phase_centre.to_hadec(precession_info.lmst_j2000),
            );
            (
                uvws,
                precession_info.lmst_j2000,
                precession_info.array_latitude_j2000,
            )
        } else {
            let lst = get_lmst(timestamp, self.array_longitude);
            let uvws = xyzs_to_cross_uvws_parallel(
                self.unflagged_tile_xyzs,
                self.phase_centre.to_hadec(lst),
            );
            (uvws, lst, self.array_latitude)
        };
        let shapelet_uvws = self
            .components
            .shapelets
            .get_shapelet_uvws(lst, self.unflagged_tile_xyzs);

        self.model_points_inner(vis_model_slice.view_mut(), &uvws, lst, latitude)?;
        self.model_gaussians_inner(vis_model_slice.view_mut(), &uvws, lst, latitude)?;
        self.model_shapelets_inner(
            vis_model_slice.view_mut(),
            &uvws,
            shapelet_uvws.view(),
            lst,
            latitude,
        )?;

        Ok(uvws)
    }

    fn model_points(
        &self,
        vis_model_slice: ArrayViewMut2<Jones<f32>>,
        timestamp: Epoch,
    ) -> Result<Vec<UVW>, ModelError> {
        let (uvws, lst, latitude) = if self.precess {
            let precession_info = precess_time(
                self.phase_centre,
                timestamp,
                self.array_longitude,
                self.array_latitude,
            );
            // Apply precession to the tile XYZ positions.
            let precessed_tile_xyzs =
                precession_info.precess_xyz_parallel(self.unflagged_tile_xyzs);
            let uvws = xyzs_to_cross_uvws_parallel(
                &precessed_tile_xyzs,
                self.phase_centre.to_hadec(precession_info.lmst_j2000),
            );
            (
                uvws,
                precession_info.lmst_j2000,
                precession_info.array_latitude_j2000,
            )
        } else {
            let lst = get_lmst(timestamp, self.array_longitude);
            let uvws = xyzs_to_cross_uvws_parallel(
                self.unflagged_tile_xyzs,
                self.phase_centre.to_hadec(lst),
            );
            (uvws, lst, self.array_latitude)
        };
        self.model_points_inner(vis_model_slice, &uvws, lst, latitude)?;
        Ok(uvws)
    }

    fn model_gaussians(
        &self,
        vis_model_slice: ArrayViewMut2<Jones<f32>>,
        timestamp: Epoch,
    ) -> Result<Vec<UVW>, ModelError> {
        let (uvws, lst, latitude) = if self.precess {
            let precession_info = precess_time(
                self.phase_centre,
                timestamp,
                self.array_longitude,
                self.array_latitude,
            );
            // Apply precession to the tile XYZ positions.
            let precessed_tile_xyzs =
                precession_info.precess_xyz_parallel(self.unflagged_tile_xyzs);
            let uvws = xyzs_to_cross_uvws_parallel(
                &precessed_tile_xyzs,
                self.phase_centre.to_hadec(precession_info.lmst_j2000),
            );
            (
                uvws,
                precession_info.lmst_j2000,
                precession_info.array_latitude_j2000,
            )
        } else {
            let lst = get_lmst(timestamp, self.array_longitude);
            let uvws = xyzs_to_cross_uvws_parallel(
                self.unflagged_tile_xyzs,
                self.phase_centre.to_hadec(lst),
            );
            (uvws, lst, self.array_latitude)
        };
        self.model_gaussians_inner(vis_model_slice, &uvws, lst, latitude)?;
        Ok(uvws)
    }

    fn model_shapelets(
        &self,
        vis_model_slice: ArrayViewMut2<Jones<f32>>,
        timestamp: Epoch,
    ) -> Result<Vec<UVW>, ModelError> {
        let (uvws, lst, latitude) = if self.precess {
            let precession_info = precess_time(
                self.phase_centre,
                timestamp,
                self.array_longitude,
                self.array_latitude,
            );
            // Apply precession to the tile XYZ positions.
            let precessed_tile_xyzs =
                precession_info.precess_xyz_parallel(self.unflagged_tile_xyzs);
            let uvws = xyzs_to_cross_uvws_parallel(
                &precessed_tile_xyzs,
                self.phase_centre.to_hadec(precession_info.lmst_j2000),
            );
            (
                uvws,
                precession_info.lmst_j2000,
                precession_info.array_latitude_j2000,
            )
        } else {
            let lst = get_lmst(timestamp, self.array_longitude);
            let uvws = xyzs_to_cross_uvws_parallel(
                self.unflagged_tile_xyzs,
                self.phase_centre.to_hadec(lst),
            );
            (uvws, lst, self.array_latitude)
        };

        let shapelet_uvws = self
            .components
            .shapelets
            .get_shapelet_uvws(lst, self.unflagged_tile_xyzs);
        self.model_shapelets_inner(vis_model_slice, &uvws, shapelet_uvws.view(), lst, latitude)?;

        Ok(uvws)
    }
}
