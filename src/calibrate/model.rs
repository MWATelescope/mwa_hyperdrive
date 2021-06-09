// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to generate sky-model visibilities. These functions currently work only
//! on a single timestep.

use std::f64::consts::{FRAC_PI_2, LN_2, TAU};

use log::trace;
use ndarray::{parallel::prelude::*, prelude::*};
use num::Complex;

use super::CalibrateError;
use crate::beam::Beam;
use crate::{
    constants::*,
    math::{cexp, exp},
    shapelets::*,
};
use mwa_hyperdrive_core::{
    c64, AzEl, ComponentType, EstimateError, Jones, RADec, SourceComponent, SourceList,
    XyzBaseline, LMN, UVW,
};

/// Instrumental flux densities and [LMN] coordinates for a particular type of
/// source component (e.g. point sources). The source components are borrowed
/// from the main source list. The first axis of `instrumental_flux_densities`
/// is unflagged fine channel frequency, the second is the source component. The
/// length of `components`, `instrumental_flux_densities`'s second axis and
/// `lmns` are the same.
struct PerComponentParams<'a> {
    components: Vec<&'a SourceComponent>,
    instrumental_flux_densities: Array2<Jones<f64>>,
    lmns: Vec<LMN>,
}

impl<'a> PerComponentParams<'a> {
    fn get_azels_mwa_parallel(&self, lst_rad: f64) -> Vec<AzEl> {
        self.components
            .par_iter()
            .map(|comp| comp.radec.to_hadec(lst_rad).to_azel(MWA_LAT_RAD))
            .collect()
    }
}

// Anonymous structs to help with type safety.
/// Point-source component parameters. See the documentation for
/// `PerComponentParams` for more info.
struct PointComponentParams<'a>(PerComponentParams<'a>);
/// Gaussian-source component parameters. See the documentation for
/// `PerComponentParams` for more info.
struct GaussianComponentParams<'a>(PerComponentParams<'a>);
/// Shapelet-source component parameters. See the documentation for
/// `PerComponentParams` for more info.
struct ShapeletComponentParams<'a>(PerComponentParams<'a>);

impl<'a> std::ops::Deref for PointComponentParams<'a> {
    type Target = PerComponentParams<'a>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<'a> std::ops::Deref for GaussianComponentParams<'a> {
    type Target = PerComponentParams<'a>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<'a> std::ops::Deref for ShapeletComponentParams<'a> {
    type Target = PerComponentParams<'a>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub(super) struct SplitComponents<'a> {
    point: PointComponentParams<'a>,
    gaussian: GaussianComponentParams<'a>,
    shapelet: ShapeletComponentParams<'a>,
}

/// Given a source list, split the components into each type. These parameters
/// don't change over time, so it's ideal to run this function once.
pub(super) fn split_components<'a>(
    source_list: &'a SourceList,
    unflagged_fine_chan_freqs: &[f64],
    phase_centre: &RADec,
) -> Result<SplitComponents<'a>, EstimateError> {
    let mut point_comps: Vec<&SourceComponent> = vec![];
    let mut gaussian_comps: Vec<&SourceComponent> = vec![];
    let mut shapelet_comps: Vec<&SourceComponent> = vec![];
    for comp in source_list.iter().flat_map(|(_, src)| &src.components) {
        match comp.comp_type {
            ComponentType::Point => point_comps.push(comp),
            ComponentType::Gaussian { .. } => gaussian_comps.push(comp),
            ComponentType::Shapelet { .. } => shapelet_comps.push(comp),
        }
    }

    // Get the LMN coordinates of all source components.
    let get_lmns = |comps: &[&SourceComponent]| -> Vec<LMN> {
        comps
            .par_iter()
            .map(|comp| comp.radec.to_lmn(phase_centre))
            .collect()
    };
    let point_lmns = get_lmns(&point_comps);
    let gaussian_lmns = get_lmns(&gaussian_comps);
    let shapelet_lmns = get_lmns(&shapelet_comps);

    // Get the instrumental flux densities for each component at each frequency.
    // These don't change with time, so we can save a lot of computation by just
    // doing this once.
    trace!("Estimating flux densities for sky-model components at all frequencies");
    let flux_densities = |comps: &[&SourceComponent]| -> Result<Array2<Jones<f64>>, EstimateError> {
        let mut fds = Array2::from_elem(
            (comps.len(), unflagged_fine_chan_freqs.len()),
            Jones::default(),
        );
        let results: Vec<Result<(), _>> = fds
            .outer_iter_mut()
            .into_par_iter()
            .zip(comps.par_iter())
            .map(|(mut comp_axis, comp)| {
                comp_axis
                    .iter_mut()
                    .zip(unflagged_fine_chan_freqs.iter())
                    .try_for_each(|(comp_fd, freq)| {
                        match comp.estimate_at_freq(*freq) {
                            // Estimate was OK; write into the ndarray.
                            Ok(stokes_flux_density) => {
                                let instrumental_flux_density: Jones<f64> =
                                    stokes_flux_density.into();
                                *comp_fd = instrumental_flux_density;
                                Ok(())
                            }
                            // Estimate failed, return the error.
                            Err(e) => Err(e),
                        }
                    })
            })
            .collect();

        // Handle any failures.
        let result = results.into_iter().collect::<Result<Vec<()>, _>>();
        // Flip the array axes; this makes things simpler later.
        result.map(|_| fds.t().to_owned())
    };
    let point_flux_densities = flux_densities(&point_comps)?;
    let gaussian_flux_densities = flux_densities(&gaussian_comps)?;
    let shapelet_flux_densities = flux_densities(&shapelet_comps)?;

    Ok(SplitComponents {
        point: PointComponentParams(PerComponentParams {
            components: point_comps,
            instrumental_flux_densities: point_flux_densities,
            lmns: point_lmns,
        }),
        gaussian: GaussianComponentParams(PerComponentParams {
            components: gaussian_comps,
            instrumental_flux_densities: gaussian_flux_densities,
            lmns: gaussian_lmns,
        }),
        shapelet: ShapeletComponentParams(PerComponentParams {
            components: shapelet_comps,
            instrumental_flux_densities: shapelet_flux_densities,
            lmns: shapelet_lmns,
        }),
    })
}

/// Beam-correct the expected flux densities of the sky-model source components.
/// This function should only be called by `beam_correct_flux_densities`, but is
/// isolated for testing.
///
/// `instrumental_flux_densities`: An ndarray view of the instrumental Stokes
/// flux densities of all sky-model source components (as Jones matrices). The
/// first axis is unflagged fine channel, the second is sky-model component.
///
/// `beam`: A `hyperdrive` [Beam] object.
///
/// `azels`: A collection of [AzEl] structs for each source component. The
/// length and order of this collection should match that of the second axis of
/// `instrumental_flux_densities`.
///
/// `dipole_gains`: The dipole gains to use when calculating beam responses.
/// TODO: Make this different for each tile in a baseline.
///
/// `freqs`: The frequencies to use when calculating the beam responses. The
/// length and order of this collection should match that of the first axis of
/// `instrumental_flux_densities`.
fn beam_correct_flux_densities_inner(
    instrumental_flux_densities: ArrayView2<Jones<f64>>,
    beam: &Box<dyn Beam>,
    azels: &[AzEl],
    dipole_gains: &[f64],
    freqs: &[f64],
) -> Result<Array2<Jones<f64>>, CalibrateError> {
    let mut beam_corrected_fds =
        Array2::from_elem(instrumental_flux_densities.dim(), Jones::default());

    let results = beam_corrected_fds
        .outer_iter_mut()
        .into_par_iter()
        .zip(instrumental_flux_densities.outer_iter())
        .zip(freqs.par_iter())
        .try_for_each(|((mut inst_fds_for_freq, inst_fds), freq)| {
            inst_fds_for_freq
                .iter_mut()
                .zip(inst_fds.iter())
                .zip(azels.iter())
                .try_for_each(|((comp_fd, inst_fd), azel)| {
                    // `jones_1` is the beam response from the first tile in
                    // this baseline.
                    // TODO: Use a Jones matrix cache.
                    let jones_1 = beam.calc_jones(azel, *freq as u32, dipole_gains)?;

                    // `jones_2` is the beam response from the second tile in
                    // this baseline.
                    // TODO: Use a Jones matrix from another tile!
                    let jones_2 = &jones_1;

                    // J . I . J^H
                    *comp_fd = Jones::axb(&jones_1, &Jones::axbh(&inst_fd, jones_2));
                    Ok(())
                })
        });

    // Handle any errors that happened in the closure.
    match results {
        Ok(()) => Ok(beam_corrected_fds),
        Err(e) => Err(e),
    }
}

/// Beam-correct the expected flux densities of the sky-model source components.
///
/// `instrumental_flux_densities`: An ndarray view of the instrumental Stokes
/// flux densities of all sky-model source components (as Jones matrices). The
/// first axis is unflagged fine channel, the second is sky-model component.
fn beam_correct_flux_densities(
    components: &PerComponentParams,
    lst_rad: f64,
    beam: &Box<dyn Beam>,
    dipole_gains: &[f64],
    freqs: &[f64],
) -> Result<Array2<Jones<f64>>, CalibrateError> {
    let c = components;
    let azels = c.get_azels_mwa_parallel(lst_rad);

    debug_assert_eq!(c.instrumental_flux_densities.len_of(Axis(0)), freqs.len());
    debug_assert_eq!(
        c.instrumental_flux_densities.len_of(Axis(1)),
        c.components.len()
    );
    debug_assert_eq!(c.instrumental_flux_densities.len_of(Axis(1)), azels.len());
    debug_assert!(dipole_gains.len() == 16 || dipole_gains.len() == 32);

    beam_correct_flux_densities_inner(
        c.instrumental_flux_densities.view(),
        beam,
        &azels,
        dipole_gains,
        freqs,
    )
}

pub(super) fn model_timestep(
    mut vis_model_slice: ArrayViewMut2<Jones<f32>>,
    weights: ArrayView2<f32>,
    split_components: &SplitComponents,
    beam: &Box<dyn Beam>,
    lst_rad: f64,
    unflagged_baseline_xyzs: &[XyzBaseline],
    uvws: &[UVW],
    unflagged_fine_chan_freqs: &[f64],
) -> Result<(), CalibrateError> {
    let beamed_point_fds = beam_correct_flux_densities(
        &split_components.point,
        lst_rad,
        beam,
        &[1.0; 16],
        unflagged_fine_chan_freqs,
    )?;
    model_points(
        vis_model_slice.view_mut(),
        &split_components.point,
        beamed_point_fds.view(),
        uvws,
        unflagged_fine_chan_freqs,
    );

    let beamed_gaussian_fds = beam_correct_flux_densities(
        &split_components.gaussian,
        lst_rad,
        beam,
        &[1.0; 16],
        unflagged_fine_chan_freqs,
    )?;
    model_gaussians(
        vis_model_slice.view_mut(),
        &split_components.gaussian,
        beamed_gaussian_fds.view(),
        uvws,
        unflagged_fine_chan_freqs,
    );

    let beamed_shapelet_fds = beam_correct_flux_densities(
        &split_components.shapelet,
        lst_rad,
        beam,
        &[1.0; 16],
        unflagged_fine_chan_freqs,
    )?;
    // Shapelets need their own special kind of UVW coordinates.
    let mut shapelet_uvws: Array2<UVW> = Array2::from_elem(
        (
            split_components.shapelet.components.len(),
            vis_model_slice.len_of(Axis(0)),
        ),
        UVW::default(),
    );
    shapelet_uvws
        .outer_iter_mut()
        .into_par_iter()
        .zip(split_components.shapelet.components.par_iter())
        .for_each(|(mut baseline_uvw, comp)| {
            let hadec = comp.radec.to_hadec(lst_rad);
            let shapelet_uvws = UVW::get_baselines_parallel(unflagged_baseline_xyzs, &hadec);
            baseline_uvw.assign(&Array1::from(shapelet_uvws));
        });
    // To ensure that `shapelet_uvws` is being strided efficiently,
    // invert the axes here.
    let shapelet_uvws = shapelet_uvws.t().to_owned();
    model_shapelets(
        vis_model_slice.view_mut(),
        &split_components.shapelet,
        beamed_shapelet_fds.view(),
        shapelet_uvws.view(),
        uvws,
        unflagged_fine_chan_freqs,
    );

    // Scale by weights.
    ndarray::Zip::from(&mut vis_model_slice)
        .and(&weights)
        .par_for_each(|vis, &weight| *vis *= weight);

    Ok(())
}

/// For a single timestep, over a range of frequencies and baselines, generate
/// visibilities for each sky-model source component. Write the model
/// visibilities into the model_array.
///
/// `model_array`: A mutable `ndarray` view of the model of all visibilities.
/// The first axis is unflagged baseline, the second unflagged fine channel.
///
/// `weights`: An `ndarray` view of the weights obtained alongside input data.
/// To make the model visibilities match the input data visibilities, these need
/// to be applied. The shape is the same as `model_array`.
///
/// `component_params`: A single type of sky-model source components (e.g.
/// points, Gaussians, shapelets). This struct can only be created by the
/// `split_components` function.
///
/// `flux_densities`: An `ndarray` view of the instrumental Stokes flux
/// densities of all sky-model source components. The first axis is unflagged
/// fine channel, the second is sky-model component.
///
/// `lmns`: The [LMN] coordinates of all sky-model source components. They
/// should be in the same order as the `components` they correspond to.
///
/// `uvws`: The [UVW] coordinates of each baseline \[metres\]. This should be
/// the same length as `model_array`'s first axis.
///
/// `freqs`: The unflagged fine-channel frequencies to model over \[Hz\]. This
/// should be the same length as `model_array`'s second axis. Used to divide the
/// UVW coordinates by wavelength.
///
/// `envelope_fn`: A function to be used to calculate a component's visibility
/// envelope. This parameter is what makes this function general.
fn model_common<F>(
    mut model_array: ArrayViewMut2<Jones<f32>>,
    component_params: &PerComponentParams,
    beam_corrected_fds: ArrayView2<Jones<f64>>,
    uvws: &[UVW],
    freqs: &[f64],
    envelope_fn: F,
) where
    F: Sync + Fn(&SourceComponent, UVW) -> f64,
{
    // Shortcut.
    let c = component_params;

    debug_assert_eq!(model_array.len_of(Axis(0)), uvws.len());
    debug_assert_eq!(model_array.len_of(Axis(1)), freqs.len());
    debug_assert_eq!(beam_corrected_fds.len_of(Axis(0)), freqs.len());
    debug_assert_eq!(
        beam_corrected_fds.dim(),
        c.instrumental_flux_densities.dim()
    );
    debug_assert_eq!(beam_corrected_fds.len_of(Axis(1)), c.lmns.len());
    debug_assert_eq!(beam_corrected_fds.len_of(Axis(1)), c.components.len());

    // Iterate over the unflagged baseline axis.
    model_array
        .outer_iter_mut()
        .into_par_iter()
        .zip(uvws.par_iter())
        .for_each(|(mut model_bl_axis, uvw)| {
            // Unflagged fine-channel axis.
            model_bl_axis
                .iter_mut()
                .zip(beam_corrected_fds.outer_iter())
                .zip(freqs)
                .for_each(|((model_vis, comp_fds), freq)| {
                    // Divide by lambda to make UVW dimensionless.
                    let uvw = *uvw / (VEL_C / freq);

                    // Now that we have the UVW coordinates, we can determine
                    // each source component's envelope.
                    let envelopes = c.components.iter().map(|&comp| envelope_fn(comp, uvw));

                    comp_fds.iter().zip(c.lmns.iter()).zip(envelopes).for_each(
                        |((comp_fd_c64, lmn), envelope)| {
                            let arg = TAU * (uvw.u * lmn.l + uvw.v * lmn.m + uvw.w * (lmn.n - 1.0));
                            let phase = cexp(arg) * envelope;
                            let vis_c64 = comp_fd_c64.clone() * phase;
                            // Demote to single precision now that all
                            // operations are done.
                            let vis_c32: Jones<f32> = vis_c64.into();
                            *model_vis += vis_c32;
                        },
                    )
                });
        });
}

/// For a single timestep, over a range of frequencies and baselines, generate
/// visibilities for each specified sky-model point-source component. See
/// `model_common` for an explanation of the arguments.
fn model_points(
    model_array: ArrayViewMut2<Jones<f32>>,
    point_comp_params: &PointComponentParams,
    beam_corrected_fds: ArrayView2<Jones<f64>>,
    uvws: &[UVW],
    freqs: &[f64],
) {
    if !point_comp_params.components.is_empty() {
        model_common(
            model_array,
            &point_comp_params,
            beam_corrected_fds,
            uvws,
            freqs,
            // When calculating a point-source visibility, the envelope is always 1.
            |_, _| 1.0,
        );
    }
}

/// For a single timestep, over a range of frequencies and baselines, generate
/// visibilities for each specified sky-model gaussian-source component. See
/// `model_common` for an explanation of the arguments.
fn model_gaussians(
    model_array: ArrayViewMut2<Jones<f32>>,
    gaussian_comp_params: &GaussianComponentParams,
    beam_corrected_fds: ArrayView2<Jones<f64>>,
    uvws: &[UVW],
    freqs: &[f64],
) {
    if !gaussian_comp_params.components.is_empty() {
        model_common(
            model_array,
            &gaussian_comp_params,
            beam_corrected_fds,
            uvws,
            freqs,
            |comp, uvw| {
                match &comp.comp_type {
                    ComponentType::Gaussian { maj, min, pa } => {
                        let (s_pa, c_pa) = pa.sin_cos();
                        // Temporary variables for clarity.
                        let k_x = uvw.u * s_pa + uvw.v * c_pa;
                        let k_y = uvw.u * c_pa - uvw.v * s_pa;
                        exp(-FRAC_PI_2.powi(2) / LN_2
                            * (maj.powi(2) * k_x.powi(2) + min.powi(2) * k_y.powi(2)))
                    }

                    // We only reach here if the function is being misused.
                    _ => unreachable!(),
                }
            },
        );
    }
}

/// For a single timestep, over a range of frequencies and baselines, generate
/// visibilities for each specified sky-model shapelet-source component. See
/// `model_common` for an explanation of the arguments.
///
/// `shapelet_uvws` are special UVWs generated as if each shapelet component was
/// at the phase centre \[metres\]. The first axis is unflagged baseline, the
/// second shapelet component.
fn model_shapelets(
    mut model_array: ArrayViewMut2<Jones<f32>>,
    shapelet_comp_params: &ShapeletComponentParams,
    beam_corrected_fds: ArrayView2<Jones<f64>>,
    shapelet_uvws: ArrayView2<UVW>,
    uvws: &[UVW],
    freqs: &[f64],
) {
    debug_assert_eq!(model_array.len_of(Axis(0)), uvws.len());
    debug_assert_eq!(model_array.len_of(Axis(0)), shapelet_uvws.len_of(Axis(0)));
    debug_assert_eq!(model_array.len_of(Axis(1)), freqs.len());
    debug_assert_eq!(beam_corrected_fds.len_of(Axis(0)), freqs.len());
    debug_assert_eq!(
        beam_corrected_fds.dim(),
        shapelet_comp_params.instrumental_flux_densities.dim()
    );
    debug_assert_eq!(
        beam_corrected_fds.len_of(Axis(1)),
        shapelet_comp_params.lmns.len()
    );
    debug_assert_eq!(
        beam_corrected_fds.len_of(Axis(1)),
        shapelet_comp_params.components.len()
    );

    // Iterate over the unflagged baseline axis.
    model_array
        .outer_iter_mut()
        .into_par_iter()
        .zip(uvws.par_iter())
        .zip(shapelet_uvws.outer_iter().into_par_iter())
        .for_each(|((mut model_bl_axis, uvw), shapelet_uvws_per_comp)| {
            // Unflagged fine-channel axis.
            model_bl_axis
                .iter_mut()
                .zip(beam_corrected_fds.outer_iter())
                .zip(freqs)
                .zip(shapelet_uvws_per_comp.iter())
                .for_each(|(((model_vis, comp_fds), freq), shapelet_uvw)| {
                    // Divide by lambda to make UVW dimensionless.
                    let lambda = VEL_C / freq;
                    let uvw = *uvw / lambda;
                    let shapelet_uvw = *shapelet_uvw / lambda;

                    // Now that we have the UVW coordinates, we can determine
                    // each source component's envelope.
                    let envelopes: Vec<c64> = shapelet_comp_params
                        .components
                        .iter()
                        .map(|&comp| {
                            match &comp.comp_type {
                                ComponentType::Shapelet {
                                    maj,
                                    min,
                                    pa,
                                    coeffs,
                                } => {
                                    let (s_pa, c_pa) = pa.sin_cos();
                                    // The following code borrows from WODEN.
                                    let x = shapelet_uvw.u * s_pa + shapelet_uvw.v * c_pa;
                                    let y = shapelet_uvw.u * c_pa - shapelet_uvw.v * s_pa;
                                    let const_x = maj * SQRT_FRAC_PI_SQ_2_LN_2 / SBF_DX;
                                    let const_y = -min * SQRT_FRAC_PI_SQ_2_LN_2 / SBF_DX;
                                    let x_pos = x * const_x + SBF_C;
                                    let y_pos = y * const_y + SBF_C;
                                    let x_pos_int = x_pos as usize;
                                    let y_pos_int = y_pos as usize;

                                    // Fold the shapelet basis functions (here,
                                    // "coeffs") into a single envelope.
                                    coeffs.iter().fold(Complex::new(0.0, 0.0), |envelope, sbf| {
                                        let f_hat = sbf.coeff;

                                        let x_low =
                                            SHAPELET_BASIS_VALUES[SBF_L * sbf.n1 + x_pos_int];
                                        let x_high =
                                            SHAPELET_BASIS_VALUES[SBF_L * sbf.n1 + x_pos_int + 1];
                                        let u_value =
                                            x_low + (x_high - x_low) * (x_pos - x_pos.floor());

                                        let y_low =
                                            SHAPELET_BASIS_VALUES[SBF_L * sbf.n2 + y_pos_int];
                                        let y_high =
                                            SHAPELET_BASIS_VALUES[SBF_L * sbf.n2 + y_pos_int + 1];
                                        let v_value =
                                            y_low + (y_high - y_low) * (y_pos - y_pos.floor());

                                        envelope
                                            + I_POWER_TABLE[(sbf.n1 + sbf.n2) % 4]
                                                * f_hat
                                                * u_value
                                                * v_value
                                    })
                                }

                                // We only reach here if the function is being
                                // misused.
                                _ => unreachable!(),
                            }
                        })
                        .collect();

                    comp_fds
                        .iter()
                        .zip(shapelet_comp_params.lmns.iter())
                        .zip(envelopes.into_iter())
                        .for_each(|((comp_fd_c64, lmn), envelope)| {
                            let arg = TAU * (uvw.u * lmn.l + uvw.v * lmn.m + uvw.w * (lmn.n - 1.0));
                            let phase = cexp(arg) * envelope;
                            let vis_c64 = comp_fd_c64.clone() * phase;
                            // Demote to single precision now that all
                            // operations are done.
                            let vis_c32: Jones<f32> = vis_c64.into();
                            *model_vis += vis_c32;
                        })
                });
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{calibrate::params::Delays, tests::*};
    use mwa_hyperdrive_core::{FluxDensity, FluxDensityType};
    use mwa_hyperdrive_srclist::SourceListType;

    fn get_small_source_list() -> SourceList {
        let (mut source_list, _) = mwa_hyperdrive_srclist::read::read_source_list_file(
            "test_files/1090008640/srclist_pumav3_EoR0aegean_EoR1pietro+ForA_1090008640_100.yaml",
            Some(SourceListType::Hyperdrive),
        )
        .unwrap();

        // Prune all but two sources from the source list.
        let sources_to_keep = ["J000042-342358", "J000045-272248"];
        let mut sources_to_be_removed = vec![];
        for (name, _) in source_list.iter() {
            if !sources_to_keep.contains(&name.as_str()) {
                sources_to_be_removed.push(name.to_owned());
            }
        }
        for name in sources_to_be_removed {
            source_list.remove(&name);
        }
        source_list
    }

    fn get_big_source_list() -> SourceList {
        let (mut source_list, _) = mwa_hyperdrive_srclist::read::read_source_list_file(
            "test_files/1090008640/srclist_pumav3_EoR0aegean_EoR1pietro+ForA_1090008640_100.yaml",
            Some(SourceListType::Hyperdrive),
        )
        .unwrap();

        // Prune all but four sources from the source list.
        let sources_to_keep = [
            "J000042-342358",
            "J000045-272248",
            "J000816-193957",
            "J001612-312330A",
        ];
        let mut sources_to_be_removed = vec![];
        for (name, _) in source_list.iter() {
            if !sources_to_keep.contains(&name.as_str()) {
                sources_to_be_removed.push(name.to_owned());
            }
        }
        for name in sources_to_be_removed {
            source_list.remove(&name);
        }
        source_list
    }

    fn get_instrumental_flux_densities(srclist: &SourceList, freqs: &[f64]) -> Array2<Jones<f64>> {
        let num_components = srclist.values().fold(0, |a, src| a + src.components.len());

        let inst_flux_densities = {
            let mut fds = Array2::from_elem((num_components, freqs.len()), Jones::default());
            for (mut comp_axis, comp) in fds
                .outer_iter_mut()
                .zip(srclist.iter().map(|(_, src)| &src.components).flatten())
            {
                for (comp_fd, freq) in comp_axis.iter_mut().zip(freqs.iter()) {
                    *comp_fd = comp.estimate_at_freq(*freq).unwrap().into();
                }
            }
            // Flip the array axes; this makes things simpler later.
            fds.t().to_owned()
        };
        assert_eq!(inst_flux_densities.dim(), (freqs.len(), num_components));
        inst_flux_densities
    }

    #[test]
    fn test_beam_correct_flux_densities_no_beam() {
        let freqs = [170e6];
        let lst = 6.261977848;
        let dipole_delays = [0; 16];
        let dipole_gains = [1.0; 16];

        let beam: Box<dyn Beam> = Box::new(crate::beam::NoBeam);
        let srclist = get_small_source_list();
        let inst_flux_densities = get_instrumental_flux_densities(&srclist, &freqs);
        let result = match beam_correct_flux_densities_inner(
            inst_flux_densities.view(),
            &beam,
            &srclist.get_azel_mwa(lst),
            &dipole_gains,
            &freqs,
        ) {
            Ok(fds) => fds,
            Err(e) => panic!("{}", e),
        };
        let num_components = srclist.values().fold(0, |a, src| a + src.components.len());
        assert_eq!(result.dim(), (freqs.len(), num_components));

        // Hand-verified results.
        let expected_comp_fd_1 = Jones::from([
            Complex::new(2.7473072919275476, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(2.7473072919275476, 0.0),
        ]);
        let expected_comp_fd_2 = Jones::from([
            Complex::new(1.7047163998893684, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.7047163998893684, 0.0),
        ]);
        assert_abs_diff_eq!(result[[0, 0]], expected_comp_fd_1, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[0, 1]], expected_comp_fd_2, epsilon = 1e-10);
    }

    #[test]
    #[serial]
    fn test_beam_correct_flux_densities_170_mhz() {
        let freqs = [170e6];
        let lst = 6.261977848;
        let dipole_delays = vec![0; 16];
        let dipole_gains = [1.0; 16];

        let beam: Box<dyn Beam> =
            Box::new(crate::beam::FEEBeam::new_from_env(Delays::Available(dipole_delays)).unwrap());
        let srclist = get_small_source_list();
        let inst_flux_densities = get_instrumental_flux_densities(&srclist, &freqs);

        let result = match beam_correct_flux_densities_inner(
            inst_flux_densities.view(),
            &beam,
            &srclist.get_azel_mwa(lst),
            &dipole_gains,
            &freqs,
        ) {
            Ok(fds) => fds,
            Err(e) => panic!("{}", e),
        };
        let num_components = srclist.values().fold(0, |a, src| a + src.components.len());
        assert_eq!(result.dim(), (freqs.len(), num_components));

        // Hand-verified results.
        let expected_comp_fd_1 = Jones::from([
            Complex::new(2.7473072919275476, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(2.7473072919275476, 0.0),
        ]);
        let expected_jones_1 = Jones::from([
            Complex::new(0.7750324863535399, 0.24282289190335862),
            Complex::new(-0.009009420577898178, -0.002856655664463373),
            Complex::new(0.01021394523909512, 0.0033072019611734838),
            Complex::new(0.7814897063974989, 0.25556799755364396),
        ]);
        assert_abs_diff_eq!(
            result[[0, 0]],
            Jones::axb(
                &expected_jones_1,
                &Jones::axbh(&expected_comp_fd_1, &expected_jones_1)
            ),
            epsilon = 1e-10
        );

        let expected_comp_fd_2 = Jones::from([
            Complex::new(1.7047163998893684, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.7047163998893684, 0.0),
        ]);
        let expected_jones_2 = Jones::from([
            Complex::new(0.9455907247090378, 0.3049292024132071),
            Complex::new(-0.010712295162757346, -0.0033779555969525588),
            Complex::new(0.010367761993275826, 0.003441723575945327),
            Complex::new(0.9450219468106582, 0.30598012238683214),
        ]);
        assert_abs_diff_eq!(
            result[[0, 1]],
            Jones::axb(
                &expected_jones_2,
                &Jones::axbh(&expected_comp_fd_2, &expected_jones_2)
            ),
            epsilon = 1e-10
        );
    }

    #[test]
    #[serial]
    // Same as above, but with a different frequency.
    fn test_beam_correct_flux_densities_180_mhz() {
        let freqs = [180e6];
        let lst = 6.261977848;
        let dipole_delays = vec![0; 16];
        let dipole_gains = [1.0; 16];

        let beam: Box<dyn Beam> =
            Box::new(crate::beam::FEEBeam::new_from_env(Delays::Available(dipole_delays)).unwrap());
        let srclist = get_small_source_list();
        let inst_flux_densities = get_instrumental_flux_densities(&srclist, &freqs);
        let result = match beam_correct_flux_densities_inner(
            inst_flux_densities.view(),
            &beam,
            &srclist.get_azel_mwa(lst),
            &dipole_gains,
            &freqs,
        ) {
            Ok(fds) => fds,
            Err(e) => panic!("{}", e),
        };
        let num_components = srclist.values().fold(0, |a, src| a + src.components.len());
        assert_eq!(result.dim(), (freqs.len(), num_components));

        // Hand-verified results.
        let expected_comp_fd_1 = Jones::from([
            Complex::new(2.60247, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(2.60247, 0.0),
        ]);
        let expected_jones_1 = Jones::from([
            Complex::new(0.7731976406423393, 0.17034253171231564),
            Complex::new(-0.009017301710718753, -0.001961964125441071),
            Complex::new(0.010223521132619665, 0.002456914956330356),
            Complex::new(0.7838681411558177, 0.186582048535625),
        ]);
        assert_abs_diff_eq!(
            result[[0, 0]],
            Jones::axb(
                &expected_jones_1,
                &Jones::axbh(&expected_comp_fd_1, &expected_jones_1)
            ),
            epsilon = 1e-10
        );

        let expected_comp_fd_2 = Jones::from([
            Complex::new(1.61824, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.61824, 0.0),
        ]);
        let expected_jones_2 = Jones::from([
            Complex::new(0.9682339089232415, 0.2198904292735457),
            Complex::new(-0.01090619422142064, -0.0023800302690927533),
            Complex::new(0.010687354909991509, 0.002535994729487373),
            Complex::new(0.9676157155647803, 0.22121720658375732),
        ]);
        assert_abs_diff_eq!(
            result[[0, 1]],
            Jones::axb(
                &expected_jones_2,
                &Jones::axbh(&expected_comp_fd_2, &expected_jones_2)
            ),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_split_components() {
        let freq = 180e6;
        let srclist = get_big_source_list();
        let result = split_components(&srclist, &[freq], &RADec::new_degrees(0.0, -27.0));
        assert!(result.is_ok());
        let split_components = result.unwrap();
        let point = split_components.point;
        let gauss = split_components.gaussian;
        let shapelet = split_components.shapelet;

        let num_point_components = srclist.values().fold(0, |a, src| {
            a + src
                .components
                .iter()
                .filter(|comp| matches!(comp.comp_type, ComponentType::Point))
                .count()
        });
        let num_gauss_components = srclist.values().fold(0, |a, src| {
            a + src
                .components
                .iter()
                .filter(|comp| matches!(comp.comp_type, ComponentType::Gaussian { .. }))
                .count()
        });
        assert_eq!(point.components.len(), num_point_components);
        assert_eq!(point.components.len(), 2);
        assert_eq!(gauss.components.len(), num_gauss_components);
        assert_eq!(gauss.components.len(), 4);
        assert!(shapelet.components.is_empty());

        assert_eq!(point.lmns.len(), num_point_components);
        assert_eq!(gauss.lmns.len(), num_gauss_components);
        assert!(shapelet.lmns.is_empty());
        assert_abs_diff_eq!(point.lmns[0].l, 0.0025326811687516274);
        assert_abs_diff_eq!(point.lmns[0].m, -0.12880688061967666);
        assert_abs_diff_eq!(point.lmns[0].n, 0.9916664625927036);

        assert_eq!(
            point.instrumental_flux_densities.dim(),
            (1, num_point_components)
        );
        assert_eq!(
            gauss.instrumental_flux_densities.dim(),
            (1, num_gauss_components)
        );
        assert_eq!(shapelet.instrumental_flux_densities.dim(), (1, 0));

        // Test one of the component's instrumental flux density.
        let fd = FluxDensityType::List {
            fds: vec![
                FluxDensity {
                    freq: 80e6,
                    i: 2.13017,
                    ..Default::default()
                },
                FluxDensity {
                    freq: 240e6,
                    i: 0.33037,
                    ..Default::default()
                },
            ],
        }
        .estimate_at_freq(freq)
        .unwrap();
        let inst_fd: Jones<f64> = fd.into();
        assert_abs_diff_eq!(gauss.instrumental_flux_densities[[0, 2]], inst_fd);
    }
}
