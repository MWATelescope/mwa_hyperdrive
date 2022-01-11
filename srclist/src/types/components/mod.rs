// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Sky-model component types.

#[cfg(test)]
mod tests;

use marlu::{
    constants::MWA_LAT_RAD, pos::xyz::xyzs_to_cross_uvws_parallel, AzEl, Jones, RADec, XyzGeodetic,
    LMN, UVW,
};
use ndarray::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::SourceList;
use crate::{FluxDensity, FluxDensityType};
use mwa_hyperdrive_beam::{Beam, BeamError};
use mwa_hyperdrive_common::{marlu, ndarray, rayon};

/// Information on a source's component.
#[derive(Clone, Debug, PartialEq)]
pub struct SourceComponent {
    /// Coordinates struct associated with the component.
    pub radec: RADec,
    /// The type of component.
    pub comp_type: ComponentType,
    /// The flux densities associated with this component.
    pub flux_type: FluxDensityType,
}

impl SourceComponent {
    /// Estimate the flux density of this component at a frequency.
    pub fn estimate_at_freq(&self, freq_hz: f64) -> FluxDensity {
        self.flux_type.estimate_at_freq(freq_hz)
    }

    /// Is this component a point source?
    pub fn is_point(&self) -> bool {
        self.comp_type.is_point()
    }

    /// Is this component a gaussian source?
    pub fn is_gaussian(&self) -> bool {
        self.comp_type.is_gaussian()
    }

    /// Is this component a shapelet source?
    pub fn is_shapelet(&self) -> bool {
        self.comp_type.is_shapelet()
    }
}

/// Source component types supported by hyperdrive.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ComponentType {
    #[serde(rename = "point")]
    Point,

    #[serde(rename = "gaussian")]
    Gaussian {
        /// Major axis size \[radians\]
        maj: f64,
        /// Minor axis size \[radians\]
        min: f64,
        /// Position angle \[radians\]
        pa: f64,
    },

    #[serde(rename = "shapelet")]
    Shapelet {
        /// Major axis size \[radians\]
        maj: f64,
        /// Minor axis size \[radians\]
        min: f64,
        /// Position angle \[radians\]
        pa: f64,
        /// Shapelet coefficients
        coeffs: Vec<ShapeletCoeff>,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShapeletCoeff {
    pub n1: usize,
    pub n2: usize,
    pub value: f64,
}

impl ComponentType {
    // The following functions save the caller from using pattern matching to
    // determine the enum variant.

    /// Is this a point source?
    pub fn is_point(&self) -> bool {
        matches!(self, Self::Point)
    }

    /// Is this a gaussian source?
    pub fn is_gaussian(&self) -> bool {
        matches!(self, Self::Gaussian { .. })
    }

    /// Is this a shapelet source?
    pub fn is_shapelet(&self) -> bool {
        matches!(self, Self::Shapelet { .. })
    }
}

/// Major and minor axes as well as a positional angle to describe a Gaussian
/// (or something like a Gaussian, e.g. a shapelet).
#[derive(Clone, Debug, PartialEq)]
pub struct GaussianParams {
    /// Major axis size \[radians\]
    pub maj: f64,
    /// Minor axis size \[radians\]
    pub min: f64,
    /// Position angle \[radians\]
    pub pa: f64,
}

/// [ComponentList] is an alternative to [SourceList] where each of the
/// components and their parameters are arranged into vectors. This improves CPU
/// cache efficiency and allows for easier FFI because elements are contiguous.
///
/// For convenience, the [LMN] coordinates and instrumental flux densities of
/// the components are also provided here.
#[derive(Clone, Debug)]
pub struct ComponentList {
    pub points: PointComponentParams,
    pub gaussians: GaussianComponentParams,
    pub shapelets: ShapeletComponentParams,
}

impl ComponentList {
    /// Given a source list, split the components into each [ComponentType].
    ///
    /// These parameters don't change over time, so it's ideal to run this
    /// function once.
    pub fn new(
        source_list: &SourceList,
        unflagged_fine_chan_freqs: &[f64],
        phase_centre: RADec,
    ) -> ComponentList {
        // Unpack each of the component parameters into vectors.
        let mut point_radecs = vec![];
        let mut point_lmns = vec![];
        let mut point_fds: Vec<FluxDensityType> = vec![];

        let mut gaussian_radecs = vec![];
        let mut gaussian_lmns = vec![];
        let mut gaussian_fds: Vec<FluxDensityType> = vec![];
        let mut gaussian_gaussian_params = vec![];

        let mut shapelet_radecs = vec![];
        let mut shapelet_lmns = vec![];
        let mut shapelet_fds: Vec<FluxDensityType> = vec![];
        let mut shapelet_gaussian_params = vec![];
        let mut shapelet_coeffs: Vec<Vec<ShapeletCoeff>> = vec![];

        for comp in source_list.iter().flat_map(|(_, src)| &src.components) {
            let comp_lmn = comp.radec.to_lmn(phase_centre).prepare_for_rime();
            match &comp.comp_type {
                ComponentType::Point => {
                    point_radecs.push(comp.radec);
                    point_lmns.push(comp_lmn);
                    point_fds.push(comp.flux_type.clone());
                }

                ComponentType::Gaussian { maj, min, pa } => {
                    gaussian_radecs.push(comp.radec);
                    gaussian_lmns.push(comp_lmn);
                    gaussian_fds.push(comp.flux_type.clone());
                    gaussian_gaussian_params.push(GaussianParams {
                        maj: *maj,
                        min: *min,
                        pa: *pa,
                    });
                }

                ComponentType::Shapelet {
                    maj,
                    min,
                    pa,
                    coeffs,
                } => {
                    shapelet_radecs.push(comp.radec);
                    shapelet_lmns.push(comp_lmn);
                    shapelet_fds.push(comp.flux_type.clone());
                    shapelet_gaussian_params.push(GaussianParams {
                        maj: *maj,
                        min: *min,
                        pa: *pa,
                    });
                    shapelet_coeffs.push(coeffs.clone());
                }
            }
        }

        let point_flux_densities =
            get_instrumental_flux_densities(&point_fds, unflagged_fine_chan_freqs);
        let gaussian_flux_densities =
            get_instrumental_flux_densities(&gaussian_fds, unflagged_fine_chan_freqs);
        let shapelet_flux_densities =
            get_instrumental_flux_densities(&shapelet_fds, unflagged_fine_chan_freqs);

        // Attempt to conserve memory. (Does Rust do this anyway?)
        point_radecs.shrink_to_fit();
        point_lmns.shrink_to_fit();
        gaussian_radecs.shrink_to_fit();
        gaussian_lmns.shrink_to_fit();
        gaussian_gaussian_params.shrink_to_fit();
        shapelet_radecs.shrink_to_fit();
        shapelet_lmns.shrink_to_fit();
        shapelet_gaussian_params.shrink_to_fit();
        shapelet_coeffs.shrink_to_fit();

        Self {
            points: PointComponentParams {
                radecs: point_radecs,
                lmns: point_lmns,
                flux_densities: point_flux_densities,
            },
            gaussians: GaussianComponentParams {
                radecs: gaussian_radecs,
                lmns: gaussian_lmns,
                flux_densities: gaussian_flux_densities,
                gaussian_params: gaussian_gaussian_params,
            },
            shapelets: ShapeletComponentParams {
                radecs: shapelet_radecs,
                lmns: shapelet_lmns,
                flux_densities: shapelet_flux_densities,
                gaussian_params: shapelet_gaussian_params,
                shapelet_coeffs,
            },
        }
    }
}

impl ShapeletComponentParams {
    /// Shapelets need their own special kind of UVW coordinates. Each shapelet
    /// component's position is treated as the phase centre.
    ///
    /// The returned array has baseline as the first axis and component as the
    /// second.
    pub fn get_shapelet_uvws(&self, lst_rad: f64, tile_xyzs: &[XyzGeodetic]) -> Array2<UVW> {
        let n = tile_xyzs.len();
        let num_baselines = (n * (n - 1)) / 2;

        let mut shapelet_uvws: Array2<UVW> =
            Array2::from_elem((num_baselines, self.radecs.len()), UVW::default());
        shapelet_uvws
            .axis_iter_mut(Axis(1))
            .into_par_iter()
            .zip(self.radecs.par_iter())
            .for_each(|(mut baseline_uvws, radec)| {
                let hadec = radec.to_hadec(lst_rad);
                let uvws_row = xyzs_to_cross_uvws_parallel(tile_xyzs, hadec);
                baseline_uvws.assign(&Array1::from(uvws_row));
            });
        shapelet_uvws
    }
}

// Get the instrumental flux densities for a bunch of component flux densities.
// The first axis of the returned array is frequency, the second component.
//
// These don't change with time, so we can save a lot of computation by just
// doing this once.
pub fn get_instrumental_flux_densities(
    comp_fds: &[FluxDensityType],
    unflagged_fine_chan_freqs: &[f64],
) -> Array2<Jones<f64>> {
    let mut inst_fds = Array2::from_elem(
        (unflagged_fine_chan_freqs.len(), comp_fds.len()),
        Jones::default(),
    );
    inst_fds
        .axis_iter_mut(Axis(1))
        .into_par_iter()
        .zip(comp_fds.par_iter())
        .for_each(|(mut inst_fd_axis, comp_fd)| {
            inst_fd_axis
                .iter_mut()
                .zip(unflagged_fine_chan_freqs.iter())
                .for_each(|(inst_fd, freq)| {
                    let stokes_flux_density = comp_fd.estimate_at_freq(*freq);
                    let instrumental_flux_density: Jones<f64> =
                        stokes_flux_density.to_inst_stokes();
                    *inst_fd = instrumental_flux_density;
                })
        });
    inst_fds
}

/// Point-source-component parameters.
///
/// The first axis of `flux_densities` is unflagged fine channel
/// frequency, the second is the source component. The length of `radecs`,
/// `lmns`, `flux_densities`'s second axis are the same.
#[derive(Clone, Debug, Default)]
pub struct PointComponentParams {
    pub radecs: Vec<RADec>,
    pub lmns: Vec<LMN>,
    /// Instrumental (i.e. XX, XY, YX, XX).
    pub flux_densities: Array2<Jones<f64>>,
}

/// Gaussian-source-component parameters.
///
/// See the doc comment for [PointComponentParams] for more info.
#[derive(Clone, Debug, Default)]
pub struct GaussianComponentParams {
    pub radecs: Vec<RADec>,
    pub lmns: Vec<LMN>,
    /// Instrumental (i.e. XX, XY, YX, XX).
    pub flux_densities: Array2<Jones<f64>>,
    pub gaussian_params: Vec<GaussianParams>,
}

/// Shapelet-source-component parameters.
///
/// See the doc comment for [PointComponentParams] for more info.
#[derive(Clone, Debug, Default)]
pub struct ShapeletComponentParams {
    pub radecs: Vec<RADec>,
    pub lmns: Vec<LMN>,
    /// Instrumental (i.e. XX, XY, YX, XX).
    pub flux_densities: Array2<Jones<f64>>,
    pub gaussian_params: Vec<GaussianParams>,
    pub shapelet_coeffs: Vec<Vec<ShapeletCoeff>>,
}

/// A trait to abstract common behaviour on the per-component parameters.
pub trait PerComponentParams {
    fn get_azels_mwa_parallel(&self, lst_rad: f64) -> Vec<AzEl>;

    /// Beam-correct the expected flux densities of the sky-model source components.
    fn beam_correct_flux_densities(
        &self,
        lst_rad: f64,
        beam: &dyn Beam,
        dipole_gains: &[f64],
        freqs: &[f64],
    ) -> Result<Array2<Jones<f64>>, BeamError>;
}

fn get_azels_mwa_parallel(radecs: &[RADec], lst_rad: f64) -> Vec<AzEl> {
    radecs
        .par_iter()
        .map(|radec| radec.to_hadec(lst_rad).to_azel(MWA_LAT_RAD))
        .collect()
}

fn beam_correct_flux_densities(
    radecs: &[RADec],
    instrumental_flux_densities: ArrayView2<Jones<f64>>,
    lst_rad: f64,
    beam: &dyn Beam,
    dipole_gains: &[f64],
    freqs: &[f64],
) -> Result<Array2<Jones<f64>>, BeamError> {
    let azels = get_azels_mwa_parallel(radecs, lst_rad);

    debug_assert_eq!(instrumental_flux_densities.len_of(Axis(0)), freqs.len());
    debug_assert_eq!(instrumental_flux_densities.len_of(Axis(1)), radecs.len());
    debug_assert_eq!(instrumental_flux_densities.len_of(Axis(1)), azels.len());
    debug_assert!(dipole_gains.len() == 16 || dipole_gains.len() == 32);

    beam_correct_flux_densities_inner(
        instrumental_flux_densities.view(),
        beam,
        &azels,
        dipole_gains,
        freqs,
    )
}

// Make each of the component types derive the trait.

impl PerComponentParams for PointComponentParams {
    fn get_azels_mwa_parallel(&self, lst_rad: f64) -> Vec<AzEl> {
        get_azels_mwa_parallel(&self.radecs, lst_rad)
    }

    fn beam_correct_flux_densities(
        &self,
        lst_rad: f64,
        beam: &dyn Beam,
        dipole_gains: &[f64],
        freqs: &[f64],
    ) -> Result<Array2<Jones<f64>>, BeamError> {
        beam_correct_flux_densities(
            &self.radecs,
            self.flux_densities.view(),
            lst_rad,
            beam,
            dipole_gains,
            freqs,
        )
    }
}

impl PerComponentParams for GaussianComponentParams {
    fn get_azels_mwa_parallel(&self, lst_rad: f64) -> Vec<AzEl> {
        get_azels_mwa_parallel(&self.radecs, lst_rad)
    }

    fn beam_correct_flux_densities(
        &self,
        lst_rad: f64,
        beam: &dyn Beam,
        dipole_gains: &[f64],
        freqs: &[f64],
    ) -> Result<Array2<Jones<f64>>, BeamError> {
        beam_correct_flux_densities(
            &self.radecs,
            self.flux_densities.view(),
            lst_rad,
            beam,
            dipole_gains,
            freqs,
        )
    }
}

impl PerComponentParams for ShapeletComponentParams {
    fn get_azels_mwa_parallel(&self, lst_rad: f64) -> Vec<AzEl> {
        get_azels_mwa_parallel(&self.radecs, lst_rad)
    }

    fn beam_correct_flux_densities(
        &self,
        lst_rad: f64,
        beam: &dyn Beam,
        dipole_gains: &[f64],
        freqs: &[f64],
    ) -> Result<Array2<Jones<f64>>, BeamError> {
        beam_correct_flux_densities(
            &self.radecs,
            self.flux_densities.view(),
            lst_rad,
            beam,
            dipole_gains,
            freqs,
        )
    }
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
    beam: &dyn Beam,
    azels: &[AzEl],
    dipole_gains: &[f64],
    freqs: &[f64],
) -> Result<Array2<Jones<f64>>, BeamError> {
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
                    // let jones_1 = beam.calc_jones(*azel, *freq, dipole_gains)?;

                    // `jones_2` is the beam response from the second tile in
                    // this baseline.
                    // TODO: Use a Jones matrix from another tile!
                    // let jones_2 = jones_1;

                    // J . I . J^H
                    // *comp_fd = jones_1 * *inst_fd * jones_2.h();
                    Ok(())
                })
        });

    // Handle any errors that happened in the closure.
    match results {
        Ok(()) => Ok(beam_corrected_fds),
        Err(e) => Err(e),
    }
}
