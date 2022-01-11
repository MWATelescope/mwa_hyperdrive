// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to generate sky-model visibilities. These functions currently work only
//! on a single timestep.

mod error;
pub use error::*;
#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::f64::consts::{FRAC_PI_2, LN_2};

use marlu::{AzEl, Complex, Jones, XyzGeodetic, LMN, UVW};
use ndarray::{parallel::prelude::*, prelude::*};

use crate::{
    constants::*,
    math::{cexp, exp},
    shapelets::*,
};
use mwa_hyperdrive_beam::Beam;
use mwa_hyperdrive_common::{marlu, ndarray};
use mwa_hyperdrive_srclist::{ComponentList, GaussianParams, PerComponentParams, ShapeletCoeff};

const GAUSSIAN_EXP_CONST: f64 = -(FRAC_PI_2 * FRAC_PI_2) / LN_2;

/// Generate simulated visibilities for the given components for a single
/// timestep (a.k.a. generate the sky model).
///
/// `vis_model_slize`: A mutable view into a an `ndarray`. Rather than returning
/// an array from this function, modelled visibilities are written into this
/// array.
///
/// `components`: [ComponentList] has each of the supported sky-model component
/// types bunched together in vectors for maximum computational efficiency. The
/// component [LMN]s should have had 1 subtracted from n and all (l,m,n)
/// multiplied by 2pi (this happens if the [ComponentList] was created with its
/// `new` method). This saves a lot of FLOPs.
///
/// `beam`: An dynamic trait object for [Beam].
///
/// `lst_rad`: The local sidereal time for the timestep \[radians\].
///
/// `tile_xyzs`: The geodetic XYZ positions of the tiles (or antennas or
/// stations or...)
///
/// `unflagged_fine_chan_freqs`: The unflagged fine-channel frequencies to model
/// over \[Hz\]. This should be the same length as `vis_model_slice`'s second
/// axis. Used to scale the UVW coordinates.
pub fn model_timestep(
    mut vis_model_slice: ArrayViewMut2<Jones<f32>>,
    components: &ComponentList,
    beam: &dyn Beam,
    lst_rad: f64,
    tile_xyzs: &[XyzGeodetic],
    uvws: &[UVW],
    unflagged_fine_chan_freqs: &[f64],
    unflagged_baseline_to_tile_map: &HashMap<usize, (usize, usize)>,
) -> Result<(), ModelError> {
    if !components.points.radecs.is_empty() {
        model_points(
            vis_model_slice.view_mut(),
            beam,
            &components.points.get_azels_mwa_parallel(lst_rad),
            &components.points.lmns,
            components.points.flux_densities.view(),
            uvws,
            unflagged_fine_chan_freqs,
            unflagged_baseline_to_tile_map,
        );
    }

    if !components.gaussians.radecs.is_empty() {
        model_gaussians(
            vis_model_slice.view_mut(),
            beam,
            &components.gaussians.get_azels_mwa_parallel(lst_rad),
            &components.gaussians.lmns,
            &components.gaussians.gaussian_params,
            components.gaussians.flux_densities.view(),
            uvws,
            unflagged_fine_chan_freqs,
            unflagged_baseline_to_tile_map,
        );
    }

    if !components.shapelets.radecs.is_empty() {
        let shapelet_uvws = components.shapelets.get_shapelet_uvws(lst_rad, tile_xyzs);
        model_shapelets(
            vis_model_slice.view_mut(),
            beam,
            &components.shapelets.get_azels_mwa_parallel(lst_rad),
            &components.shapelets.lmns,
            &components.shapelets.gaussian_params,
            &components.shapelets.shapelet_coeffs,
            shapelet_uvws.view(),
            components.shapelets.flux_densities.view(),
            uvws,
            unflagged_fine_chan_freqs,
            unflagged_baseline_to_tile_map,
        );
    }

    beam.empty_cache();
    Ok(())
}

/// For a single timestep, over a range of frequencies and baselines, generate
/// visibilities for each specified sky-model point-source component.
///
/// `model_array`: A mutable `ndarray` view of the model of all visibilities.
/// The first axis is unflagged baseline, the second unflagged fine channel.
///
/// `lmns`: The [LMN] coordinates of all sky-model point-source components.
/// These should have had 1 subtracted from n and all (l,m,n) multiplied by 2pi
/// (this happens if the [ComponentList] was created with its `new` method).
/// This saves a lot of FLOPs.
///
/// `beam_corrected_fds`: An `ndarray` view of the instrumental Stokes flux
/// densities of all sky-model source components. The first axis is unflagged
/// fine channel, the second is sky-model component. If beam code is being used,
/// then these flux densities are expected to have the beam response applied to
/// sky-model flux densities.
///
/// `uvws`: The [UVW] coordinates of each baseline \[metres\]. This should be
/// the same length as `model_array`'s first axis.
///
/// `freqs`: The unflagged fine-channel frequencies to model over \[Hz\]. This
/// should be the same length as `model_array`'s second axis. Used to divide the
/// UVW coordinates by wavelength.
pub fn model_points(
    mut model_array: ArrayViewMut2<Jones<f32>>,
    beam: &dyn Beam,
    azels: &[AzEl],
    lmns: &[LMN],
    fds: ArrayView2<Jones<f64>>,
    uvws: &[UVW],
    freqs: &[f64],
    baseline_to_tile_map: &HashMap<usize, (usize, usize)>,
) {
    debug_assert_eq!(
        model_array.len_of(Axis(0)),
        uvws.len(),
        "model_array.len_of(Axis(0)) != uvws.len()"
    );
    debug_assert_eq!(
        model_array.len_of(Axis(1)),
        freqs.len(),
        "model_array.len_of(Axis(1)) != freqs.len()"
    );
    debug_assert_eq!(
        fds.len_of(Axis(0)),
        freqs.len(),
        "fds.len_of(Axis(0)) != freqs.len()"
    );
    debug_assert_eq!(
        fds.len_of(Axis(1)),
        azels.len(),
        "fds.len_of(Axis(1)) != azels.len()"
    );
    debug_assert_eq!(
        fds.len_of(Axis(1)),
        lmns.len(),
        "fds.len_of(Axis(1)) != lmns.len()"
    );
    debug_assert_eq!(
        uvws.len(),
        baseline_to_tile_map.len(),
        "uvws.len() != baseline_to_tile_map.len()"
    );

    // Iterate over the unflagged baseline axis.
    model_array
        .outer_iter_mut()
        .into_par_iter()
        .zip(uvws.par_iter())
        .enumerate()
        .for_each(|(i_baseline, (mut model_bl_axis, uvw_metres))| {
            let (i_tile1, i_tile2) = baseline_to_tile_map[&i_baseline];

            // Unflagged fine-channel axis.
            model_bl_axis
                .iter_mut()
                .zip(fds.outer_iter())
                .zip(freqs)
                .for_each(|((model_vis, comp_fds), freq)| {
                    // Divide UVW by lambda to make UVW dimensionless.
                    let lambda = VEL_C / freq;
                    let uvw = *uvw_metres / lambda;

                    // Accumulate the double-precision visibilities into a
                    // double-precision Jones matrix before putting that into
                    // the `model_array`.
                    let mut jones_accum: Jones<f64> = Jones::default();

                    comp_fds.iter().zip(azels.iter()).zip(lmns.iter()).for_each(
                        |((comp_fd, azel), lmn)| {
                            let arg = uvw.u * lmn.l + uvw.v * lmn.m + uvw.w * lmn.n;
                            let phase = cexp(arg);

                            // Yep, we're unwrapping here. It is *extremely*
                            // unlikely that beam errors can occur, and if its
                            // any consolation, there's no error checking in the
                            // CUDA code...
                            let mut beam_1 = beam.calc_jones(*azel, *freq, i_tile1).unwrap();
                            let beam_2 = beam.calc_jones(*azel, *freq, i_tile2).unwrap();
                            beam_1 *= comp_fd;
                            beam_1 *= beam_2.h();
                            // `beam_1` now contains the beam-attenuated
                            // instrumental flux density.

                            jones_accum += beam_1 * phase;
                        },
                    );
                    // Demote to single precision now that all operations are
                    // done.
                    *model_vis += Jones::from(jones_accum);
                });
        });
}

/// For a single timestep, over a range of frequencies and baselines, generate
/// visibilities for each specified sky-model Gaussian-source component.
///
/// `model_array`: A mutable `ndarray` view of the model of all visibilities.
/// The first axis is unflagged baseline, the second unflagged fine channel.
///
/// `lmns`: The [LMN] coordinates of all sky-model Gaussian-source components.
///
/// `gaussian_params`: [GaussianParams] are the major and minor axes as well as
/// the position angles of the Gaussian-source components.
///
/// `beam_corrected_fds`: An `ndarray` view of the instrumental Stokes flux
/// densities of all sky-model source components. The first axis is unflagged
/// fine channel, the second is sky-model component. If beam code is being used,
/// then these flux densities are expected to have the beam response applied to
/// sky-model flux densities.
///
/// `uvws`: The [UVW] coordinates of each baseline \[metres\]. This should be
/// the same length as `model_array`'s first axis.
///
/// `freqs`: The unflagged fine-channel frequencies to model over \[Hz\]. This
/// should be the same length as `model_array`'s second axis. Used to divide the
/// UVW coordinates by wavelength.
fn model_gaussians(
    mut model_array: ArrayViewMut2<Jones<f32>>,
    beam: &dyn Beam,
    azels: &[AzEl],
    lmns: &[LMN],
    gaussian_params: &[GaussianParams],
    fds: ArrayView2<Jones<f64>>,
    uvws: &[UVW],
    freqs: &[f64],
    baseline_to_tile_map: &HashMap<usize, (usize, usize)>,
) {
    debug_assert_eq!(
        model_array.len_of(Axis(0)),
        uvws.len(),
        "model_array.len_of(Axis(0)) != uvws.len()"
    );
    debug_assert_eq!(
        model_array.len_of(Axis(1)),
        freqs.len(),
        "model_array.len_of(Axis(1)) != freqs.len()"
    );
    debug_assert_eq!(
        fds.len_of(Axis(0)),
        freqs.len(),
        "fds.len_of(Axis(0)) != freqs.len()"
    );
    debug_assert_eq!(
        fds.len_of(Axis(1)),
        azels.len(),
        "fds.len_of(Axis(1)) != azels.len()"
    );
    debug_assert_eq!(
        fds.len_of(Axis(1)),
        lmns.len(),
        "fds.len_of(Axis(1)) != lmns.len()"
    );
    debug_assert_eq!(
        uvws.len(),
        baseline_to_tile_map.len(),
        "uvws.len() != baseline_to_tile_map.len()"
    );

    // Iterate over the unflagged baseline axis.
    model_array
        .outer_iter_mut()
        .into_par_iter()
        .zip(uvws.par_iter())
        .enumerate()
        .for_each(|(i_baseline, (mut model_bl_axis, uvw_metres))| {
            let (i_tile1, i_tile2) = baseline_to_tile_map[&i_baseline];

            // Unflagged fine-channel axis.
            model_bl_axis
                .iter_mut()
                .zip(fds.outer_iter())
                .zip(freqs)
                .for_each(|((model_vis, comp_fds), freq)| {
                    // Divide UVW by lambda to make UVW dimensionless.
                    let lambda = VEL_C / freq;
                    let uvw = *uvw_metres / lambda;

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
                    // the `model_array`.
                    let mut jones_accum: Jones<f64> = Jones::default();

                    comp_fds
                        .iter()
                        .zip(azels.iter())
                        .zip(lmns.iter())
                        .zip(envelopes)
                        .for_each(|(((comp_fd, azel), lmn), envelope)| {
                            let arg = uvw.u * lmn.l + uvw.v * lmn.m + uvw.w * lmn.n;
                            let phase = cexp(arg) * envelope;

                            // Yep, we're unwrapping here. It is *extremely*
                            // unlikely that beam errors can occur, and if its
                            // any consolation, there's no error checking in the
                            // CUDA code...
                            let mut beam_1 = beam.calc_jones(*azel, *freq, i_tile1).unwrap();
                            let beam_2 = beam.calc_jones(*azel, *freq, i_tile2).unwrap();
                            beam_1 *= comp_fd;
                            beam_1 *= beam_2.h();
                            // `beam_1` now contains the beam-attenuated
                            // instrumental flux density.

                            jones_accum += beam_1 * phase;
                        });
                    // Demote to single precision now that all operations are
                    // done.
                    *model_vis += Jones::from(jones_accum);
                });
        });
}

/// For a single timestep, over a range of frequencies and baselines, generate
/// visibilities for each specified sky-model shapelet-source component.
///
/// `model_array`: A mutable `ndarray` view of the model of all visibilities.
/// The first axis is unflagged baseline, the second unflagged fine channel.
///
/// `lmns`: The [LMN] coordinates of all sky-model Gaussian-source components.
///
/// `gaussian_params`: [GaussianParams] are the major and minor axes as well as
/// the position angles of the shapelet-source components.
///
/// `shapelet_coeffs`: A vector of [ShapeletCoeff]s for each shapelet-source
/// component.
///
/// `shapelet_uvws` are special UVWs generated as if each shapelet component was
/// at the phase centre \[metres\]. The first axis is unflagged baseline, the
/// second shapelet component.
///
/// `beam_corrected_fds`: An `ndarray` view of the instrumental Stokes flux
/// densities of all sky-model source components. The first axis is unflagged
/// fine channel, the second is sky-model component. If beam code is being used,
/// then these flux densities are expected to have the beam response applied to
/// sky-model flux densities.
///
/// `uvws`: The [UVW] coordinates of each baseline \[metres\]. This should be
/// the same length as `model_array`'s first axis.
///
/// `freqs`: The unflagged fine-channel frequencies to model over \[Hz\]. This
/// should be the same length as `model_array`'s second axis. Used to divide the
/// UVW coordinates by wavelength.
fn model_shapelets(
    mut model_array: ArrayViewMut2<Jones<f32>>,
    beam: &dyn Beam,
    azels: &[AzEl],
    lmns: &[LMN],
    gaussian_params: &[GaussianParams],
    shapelet_coeffs: &[Vec<ShapeletCoeff>],
    shapelet_uvws: ArrayView2<UVW>,
    fds: ArrayView2<Jones<f64>>,
    uvws: &[UVW],
    freqs: &[f64],
    baseline_to_tile_map: &HashMap<usize, (usize, usize)>,
) {
    debug_assert_eq!(
        model_array.len_of(Axis(0)),
        uvws.len(),
        "model_array.len_of(Axis(0)) != uvws.len()"
    );
    debug_assert_eq!(
        model_array.len_of(Axis(1)),
        freqs.len(),
        "model_array.len_of(Axis(1)) != freqs.len()"
    );
    debug_assert_eq!(
        fds.len_of(Axis(0)),
        freqs.len(),
        "fds.len_of(Axis(0)) != freqs.len()"
    );
    debug_assert_eq!(
        fds.len_of(Axis(1)),
        azels.len(),
        "fds.len_of(Axis(1)) != azels.len()"
    );
    debug_assert_eq!(
        fds.len_of(Axis(1)),
        lmns.len(),
        "fds.len_of(Axis(1)) != lmns.len()"
    );
    debug_assert_eq!(
        uvws.len(),
        baseline_to_tile_map.len(),
        "uvws.len() != baseline_to_tile_map.len()"
    );
    debug_assert_eq!(
        model_array.len_of(Axis(0)),
        shapelet_uvws.len_of(Axis(0)),
        "model_array.len_of(Axis(0)) != shapelet_uvws.len_of(Axis(0))"
    );
    debug_assert_eq!(
        fds.len_of(Axis(1)),
        shapelet_uvws.len_of(Axis(1)),
        "fds.len_of(Axis(1)) != shapelet_uvws.len_of(Axis(1))"
    );

    // Iterate over the unflagged baseline axis.
    model_array
        .outer_iter_mut()
        .into_par_iter()
        .zip(uvws.par_iter())
        .zip(shapelet_uvws.outer_iter())
        .enumerate()
        .for_each(
            |(i_baseline, ((mut model_bl_axis, uvw_metres), shapelet_uvws_per_comp))| {
                let (i_tile1, i_tile2) = baseline_to_tile_map[&i_baseline];

                // Preallocate a vector for the envelopes.
                let mut envelopes: Vec<Complex<f64>> =
                    vec![Complex::default(); gaussian_params.len()];

                // Unflagged fine-channel axis.
                model_bl_axis
                    .iter_mut()
                    .zip(fds.outer_iter())
                    .zip(freqs)
                    .for_each(|((model_vis, comp_fds), freq)| {
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
                                let const_x = maj * SQRT_FRAC_PI_SQ_2_LN_2 / SBF_DX;
                                let const_y = -min * SQRT_FRAC_PI_SQ_2_LN_2 / SBF_DX;
                                let x_pos = x * const_x + SBF_C;
                                let y_pos = y * const_y + SBF_C;
                                let x_pos_int = x_pos.floor() as usize;
                                let y_pos_int = y_pos.floor() as usize;

                                // Fold the shapelet basis functions (here,
                                // "coeffs") into a single envelope.
                                *envelope =
                                    coeffs
                                        .iter()
                                        .fold(Complex::default(), |envelope_acc, coeff| {
                                            let f_hat = coeff.value;

                                            // Omitting boundary checks speeds
                                            // things up by ~14%.
                                            unsafe {
                                                let x_low = SHAPELET_BASIS_VALUES
                                                    .get_unchecked(SBF_L * coeff.n1 + x_pos_int);
                                                let x_high = SHAPELET_BASIS_VALUES.get_unchecked(
                                                    SBF_L * coeff.n1 + x_pos_int + 1,
                                                );
                                                let u_value = x_low
                                                    + (x_high - x_low) * (x_pos - x_pos.floor());

                                                let y_low = SHAPELET_BASIS_VALUES
                                                    .get_unchecked(SBF_L * coeff.n2 + y_pos_int);
                                                let y_high = SHAPELET_BASIS_VALUES.get_unchecked(
                                                    SBF_L * coeff.n2 + y_pos_int + 1,
                                                );
                                                let v_value = y_low
                                                    + (y_high - y_low) * (y_pos - y_pos.floor());

                                                envelope_acc
                                                    + I_POWER_TABLE
                                                        .get_unchecked((coeff.n1 + coeff.n2) % 4)
                                                        * f_hat
                                                        * u_value
                                                        * v_value
                                            }
                                        })
                            });

                        // Accumulate the double-precision visibilities into a
                        // double-precision Jones matrix before putting that
                        // into the `model_array`.
                        let mut jones_accum: Jones<f64> = Jones::default();

                        comp_fds
                            .iter()
                            .zip(azels.iter())
                            .zip(lmns.iter())
                            .zip(envelopes.iter())
                            .for_each(|(((comp_fd, azel), lmn), envelope)| {
                                let arg = uvw.u * lmn.l + uvw.v * lmn.m + uvw.w * lmn.n;
                                let phase = cexp(arg) * envelope;

                                // Yep, we're unwrapping here. It is *extremely*
                                // unlikely that beam errors can occur, and if its
                                // any consolation, there's no error checking in the
                                // CUDA code...
                                let mut beam_1 = beam.calc_jones(*azel, *freq, i_tile1).unwrap();
                                let beam_2 = beam.calc_jones(*azel, *freq, i_tile2).unwrap();
                                beam_1 *= comp_fd;
                                beam_1 *= beam_2.h();
                                // `beam_1` now contains the beam-attenuated
                                // instrumental flux density.

                                jones_accum += beam_1 * phase;
                            });
                        // Demote to single precision now that all operations are
                        // done.
                        *model_vis += Jones::from(jones_accum);
                    });
            },
        );
}
