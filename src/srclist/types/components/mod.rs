// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Sky-model component types.

#[cfg(test)]
mod tests;

use std::borrow::Borrow;

use marlu::{pos::xyz::xyzs_to_cross_uvws, AzEl, Jones, LmnRime, RADec, XyzGeodetic, UVW};
use ndarray::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::{FluxDensity, FluxDensityType, SourceList};

/// Information on a source's component.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SourceComponent {
    /// Coordinates struct associated with the component.
    #[serde(flatten)]
    pub radec: RADec,

    /// The type of component.
    pub comp_type: ComponentType,

    /// The flux densities associated with this component.
    pub flux_type: FluxDensityType,
}

impl SourceComponent {
    /// Estimate the flux density of this component at a frequency.
    pub(crate) fn estimate_at_freq(&self, freq_hz: f64) -> FluxDensity {
        self.flux_type.estimate_at_freq(freq_hz)
    }

    /// Is this component a point source?
    pub(crate) fn is_point(&self) -> bool {
        self.comp_type.is_point()
    }

    /// Is this component a gaussian source?
    pub(crate) fn is_gaussian(&self) -> bool {
        self.comp_type.is_gaussian()
    }

    /// Is this component a shapelet source?
    pub(crate) fn is_shapelet(&self) -> bool {
        self.comp_type.is_shapelet()
    }
}

/// Source component types supported by hyperdrive.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ComponentType {
    Point,

    Gaussian {
        /// Major axis size \[radians\]
        #[serde(serialize_with = "radians_to_arcsecs")]
        #[serde(deserialize_with = "arcsecs_to_radians")]
        maj: f64,

        /// Minor axis size \[radians\]
        #[serde(serialize_with = "radians_to_arcsecs")]
        #[serde(deserialize_with = "arcsecs_to_radians")]
        min: f64,

        /// Position angle \[radians\]
        #[serde(serialize_with = "radians_to_degrees")]
        #[serde(deserialize_with = "degrees_to_radians")]
        pa: f64,
    },

    Shapelet {
        /// Major axis size \[radians\]
        #[serde(serialize_with = "radians_to_arcsecs")]
        #[serde(deserialize_with = "arcsecs_to_radians")]
        maj: f64,

        /// Minor axis size \[radians\]
        #[serde(serialize_with = "radians_to_arcsecs")]
        #[serde(deserialize_with = "arcsecs_to_radians")]
        min: f64,

        /// Position angle \[radians\]
        #[serde(serialize_with = "radians_to_degrees")]
        #[serde(deserialize_with = "degrees_to_radians")]
        pa: f64,

        /// Shapelet coefficients
        coeffs: Box<[ShapeletCoeff]>,
    },
}

fn radians_to_arcsecs<S: Serializer>(num: &f64, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_f64(num.to_degrees() * 3600.0)
}

fn radians_to_degrees<S: Serializer>(num: &f64, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_f64(num.to_degrees())
}

fn arcsecs_to_radians<'de, D>(d: D) -> Result<f64, D::Error>
where
    D: Deserializer<'de>,
{
    let num: f64 = Deserialize::deserialize(d)?;
    Ok(num.to_radians() / 3600.0)
}

fn degrees_to_radians<'de, D>(d: D) -> Result<f64, D::Error>
where
    D: Deserializer<'de>,
{
    let num: f64 = Deserialize::deserialize(d)?;
    Ok(num.to_radians())
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShapeletCoeff {
    pub n1: u8,
    pub n2: u8,
    pub value: f64,
}

impl ComponentType {
    // The following functions save the caller from using pattern matching to
    // determine the enum variant.

    /// Is this a point source?
    pub(crate) fn is_point(&self) -> bool {
        matches!(self, Self::Point)
    }

    /// Is this a gaussian source?
    pub(crate) fn is_gaussian(&self) -> bool {
        matches!(self, Self::Gaussian { .. })
    }

    /// Is this a shapelet source?
    pub(crate) fn is_shapelet(&self) -> bool {
        matches!(self, Self::Shapelet { .. })
    }
}

/// Major and minor axes as well as a positional angle to describe a Gaussian
/// (or something like a Gaussian, e.g. a shapelet).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GaussianParams {
    /// Major axis size \[radians\]
    pub(crate) maj: f64,
    /// Minor axis size \[radians\]
    pub(crate) min: f64,
    /// Position angle \[radians\]
    pub(crate) pa: f64,
}

/// [ComponentList] is an alternative to [SourceList] where each of the
/// components and their parameters are arranged into vectors. This improves CPU
/// cache efficiency and allows for easier FFI because elements are contiguous.
///
/// For convenience, the [LMN] coordinates and instrumental flux densities of
/// the components are also provided here.
#[derive(Clone, Debug)]
pub(crate) struct ComponentList {
    pub(crate) points: PointComponentParams,
    pub(crate) gaussians: GaussianComponentParams,
    pub(crate) shapelets: ShapeletComponentParams,
}

impl ComponentList {
    /// Given a source list, split the components into each [ComponentType].
    ///
    /// These parameters don't change over time, so it's ideal to run this
    /// function once.
    pub(crate) fn new(
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

        // Reverse the source list; if the source list has been sorted
        // (brightest sources first), reversing makes the dimmest sources get
        // used first. This is good because floating-point precision errors are
        // smaller when similar values are accumulated. Accumulating into a
        // float starting from the brightest component means that the
        // floating-point precision errors are greater as we work through the
        // source list.
        for comp in source_list
            .iter()
            .rev()
            .flat_map(|(_, src)| src.components.iter())
        {
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
                    shapelet_coeffs.push(coeffs.to_vec());
                }
            }
        }

        let point_flux_densities =
            get_instrumental_flux_densities(&point_fds, unflagged_fine_chan_freqs);
        let gaussian_flux_densities =
            get_instrumental_flux_densities(&gaussian_fds, unflagged_fine_chan_freqs);
        let shapelet_flux_densities =
            get_instrumental_flux_densities(&shapelet_fds, unflagged_fine_chan_freqs);

        // Attempt to conserve memory.
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
    pub(crate) fn get_shapelet_uvws(&self, lst_rad: f64, tile_xyzs: &[XyzGeodetic]) -> Array2<UVW> {
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
                let uvws_row = xyzs_to_cross_uvws(tile_xyzs, hadec);
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
pub fn get_instrumental_flux_densities<T: Borrow<FluxDensityType>>(
    comp_fds: &[T],
    unflagged_fine_chan_freqs: &[f64],
) -> Array2<Jones<f64>> {
    let mut inst_fds = Array2::from_elem(
        (unflagged_fine_chan_freqs.len(), comp_fds.len()),
        Jones::default(),
    );
    inst_fds
        .axis_iter_mut(Axis(1))
        .zip(comp_fds.iter())
        .for_each(|(mut inst_fd_axis, comp_fd)| {
            inst_fd_axis
                .iter_mut()
                .zip(unflagged_fine_chan_freqs.iter())
                .for_each(|(inst_fd, freq)| {
                    let stokes_flux_density = comp_fd.borrow().estimate_at_freq(*freq);
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
pub(crate) struct PointComponentParams {
    pub(crate) radecs: Vec<RADec>,
    pub(crate) lmns: Vec<LmnRime>,
    /// Instrumental (i.e. XX, XY, YX, XX).
    pub(crate) flux_densities: Array2<Jones<f64>>,
}

/// Gaussian-source-component parameters.
///
/// See the doc comment for [PointComponentParams] for more info.
#[derive(Clone, Debug, Default)]
pub(crate) struct GaussianComponentParams {
    pub(crate) radecs: Vec<RADec>,
    pub(crate) lmns: Vec<LmnRime>,
    /// Instrumental (i.e. XX, XY, YX, XX).
    pub(crate) flux_densities: Array2<Jones<f64>>,
    pub(crate) gaussian_params: Vec<GaussianParams>,
}

/// Shapelet-source-component parameters.
///
/// See the doc comment for [PointComponentParams] for more info.
#[derive(Clone, Debug, Default)]
pub(crate) struct ShapeletComponentParams {
    pub(crate) radecs: Vec<RADec>,
    pub(crate) lmns: Vec<LmnRime>,
    /// Instrumental (i.e. XX, XY, YX, XX).
    pub(crate) flux_densities: Array2<Jones<f64>>,
    pub(crate) gaussian_params: Vec<GaussianParams>,
    pub(crate) shapelet_coeffs: Vec<Vec<ShapeletCoeff>>,
}

/// A trait to abstract common behaviour on the per-component parameters.
pub(crate) trait PerComponentParams {
    fn get_azels_mwa_parallel(&self, lst_rad: f64, array_latitude_rad: f64) -> Vec<AzEl>;
}

fn get_azels_mwa_parallel(radecs: &[RADec], lst_rad: f64, array_latitude_rad: f64) -> Vec<AzEl> {
    radecs
        .par_iter()
        .map(|radec| radec.to_hadec(lst_rad).to_azel(array_latitude_rad))
        .collect()
}

// Make each of the component types derive the trait.

impl PerComponentParams for PointComponentParams {
    fn get_azels_mwa_parallel(&self, lst_rad: f64, array_latitude_rad: f64) -> Vec<AzEl> {
        get_azels_mwa_parallel(&self.radecs, lst_rad, array_latitude_rad)
    }
}

impl PerComponentParams for GaussianComponentParams {
    fn get_azels_mwa_parallel(&self, lst_rad: f64, array_latitude_rad: f64) -> Vec<AzEl> {
        get_azels_mwa_parallel(&self.radecs, lst_rad, array_latitude_rad)
    }
}

impl PerComponentParams for ShapeletComponentParams {
    fn get_azels_mwa_parallel(&self, lst_rad: f64, array_latitude_rad: f64) -> Vec<AzEl> {
        get_azels_mwa_parallel(&self.radecs, lst_rad, array_latitude_rad)
    }
}

#[cfg(test)]
impl approx::AbsDiffEq for SourceComponent {
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        f64::EPSILON
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.radec.abs_diff_eq(&other.radec, epsilon)
            && self.comp_type.abs_diff_eq(&other.comp_type, epsilon)
            && self.flux_type.abs_diff_eq(&other.flux_type, epsilon)
    }
}

#[cfg(test)]
impl approx::AbsDiffEq for ComponentType {
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        f64::EPSILON
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        match (self, other) {
            (ComponentType::Point, ComponentType::Point) => true,

            (
                ComponentType::Gaussian { maj, min, pa },
                ComponentType::Gaussian {
                    maj: maj2,
                    min: min2,
                    pa: pa2,
                },
            ) => {
                f64::abs_diff_eq(maj, maj2, epsilon)
                    && f64::abs_diff_eq(min, min2, epsilon)
                    && f64::abs_diff_eq(pa, pa2, epsilon)
            }

            (
                ComponentType::Shapelet {
                    maj,
                    min,
                    pa,
                    coeffs,
                },
                ComponentType::Shapelet {
                    maj: maj2,
                    min: min2,
                    pa: pa2,
                    coeffs: coeffs2,
                },
            ) => {
                maj.abs_diff_eq(maj2, epsilon)
                    && min.abs_diff_eq(min2, epsilon)
                    && pa.abs_diff_eq(pa2, epsilon)
                    && coeffs.eq(coeffs2)
            }

            (
                ComponentType::Point,
                ComponentType::Gaussian { .. } | ComponentType::Shapelet { .. },
            ) => false,
            (
                ComponentType::Gaussian { .. },
                ComponentType::Point | ComponentType::Shapelet { .. },
            ) => false,
            (
                ComponentType::Shapelet { .. },
                ComponentType::Point | ComponentType::Gaussian { .. },
            ) => false,
        }
    }
}

#[cfg(test)]
impl approx::AbsDiffEq for FluxDensityType {
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        f64::EPSILON
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        match (self, other) {
            (FluxDensityType::List(v), FluxDensityType::List(v2)) => {
                approx::abs_diff_eq!(v.as_slice(), v2.as_slice(), epsilon = epsilon)
            }

            (
                FluxDensityType::PowerLaw { si, fd },
                FluxDensityType::PowerLaw { si: si2, fd: fd2 },
            ) => si.abs_diff_eq(si2, epsilon) && fd.abs_diff_eq(fd2, epsilon),

            (
                FluxDensityType::CurvedPowerLaw { si, fd, q },
                FluxDensityType::CurvedPowerLaw {
                    si: si2,
                    fd: fd2,
                    q: q2,
                },
            ) => {
                si.abs_diff_eq(si2, epsilon)
                    && fd.abs_diff_eq(fd2, epsilon)
                    && q.abs_diff_eq(q2, epsilon)
            }

            (
                FluxDensityType::List(_),
                FluxDensityType::PowerLaw { .. } | FluxDensityType::CurvedPowerLaw { .. },
            ) => false,
            (
                FluxDensityType::PowerLaw { .. },
                FluxDensityType::List(_) | FluxDensityType::CurvedPowerLaw { .. },
            ) => false,
            (
                FluxDensityType::CurvedPowerLaw { .. },
                FluxDensityType::List(_) | FluxDensityType::PowerLaw { .. },
            ) => false,
        }
    }
}
