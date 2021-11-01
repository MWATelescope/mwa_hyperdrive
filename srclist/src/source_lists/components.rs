// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Sky-model component types.

use ndarray::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::SourceList;
use crate::FluxDensityType;
use mwa_hyperdrive_beam::{Beam, BeamError};
use mwa_rust_core::{constants::MWA_LAT_RAD, pos::xyz, AzEl, Jones, RADec, XyzGeodetic, LMN, UVW};

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

// TODO: Have in mwa_rust_core
/// Convert [XyzGeodetic] tile coordinates to [UVW] baseline coordinates without
/// having to form [XyzGeodetic] baselines first. This function performs
/// calculations in parallel. Cross-correlation baselines only.
fn xyzs_to_cross_uvws_parallel(
    xyzs: &[mwa_rust_core::XyzGeodetic],
    phase_centre: mwa_rust_core::HADec,
) -> Vec<UVW> {
    let (s_ha, c_ha) = phase_centre.ha.sin_cos();
    let (s_dec, c_dec) = phase_centre.dec.sin_cos();
    // Get a UVW for each tile.
    let tile_uvws: Vec<UVW> = xyzs
        .par_iter()
        .map(|&xyz| UVW::from_xyz_inner(xyz, s_ha, c_ha, s_dec, c_dec))
        .collect();
    // Take the difference of every pair of UVWs.
    let num_tiles = xyzs.len();
    let num_baselines = (num_tiles * (num_tiles - 1)) / 2;
    (0..num_baselines)
        .into_par_iter()
        .map(|i_bl| {
            let (i, j) = mwa_rust_core::math::cross_correlation_baseline_to_tiles(num_tiles, i_bl);
            tile_uvws[i] - tile_uvws[j]
        })
        .collect()
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

#[cfg(test)]
mod tests {
    use std::f64::consts::TAU;
    use std::ops::Deref;

    use approx::assert_abs_diff_eq;
    use serial_test::serial;

    use super::*;
    use crate::{FluxDensity, SourceListType};
    use mwa_hyperdrive_beam::{create_fee_beam_object, Delays, FEEBeam, NoBeam};
    use mwa_rust_core::Complex;

    fn get_small_source_list() -> SourceList {
        let (mut source_list, _) = crate::read::read_source_list_file(
            "test_files/srclist_pumav3_EoR0aegean_EoR1pietro+ForA_1090008640_100.yaml",
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
        let (mut source_list, _) = crate::read::read_source_list_file(
            "test_files/srclist_pumav3_EoR0aegean_EoR1pietro+ForA_1090008640_100.yaml",
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

    // fn get_instrumental_flux_densities_for_srclist(
    //     srclist: &SourceList,
    //     freqs: &[f64],
    // ) -> Array2<Jones<f64>> {
    //     let mut comp_fds: Vec<FluxDensityType> = vec![];
    //     for comp in srclist.iter().flat_map(|(_, src)| &src.components) {
    //         match comp.comp_type {
    //             ComponentType::Point => {
    //                 comp_fds.push(comp.flux_type.clone());
    //             }
    //             ComponentType::Gaussian { .. } => {
    //                 comp_fds.push(comp.flux_type.clone());
    //             }
    //             ComponentType::Shapelet { .. } => {
    //                 comp_fds.push(comp.flux_type.clone());
    //             }
    //         }
    //     }

    //     get_instrumental_flux_densities(&comp_fds, freqs)
    // }

    // #[test]
    // fn test_beam_correct_flux_densities_no_beam() {
    //     let freqs = [170e6];
    //     let lst = 6.261977848;
    //     let dipole_gains = [1.0; 16];

    //     let beam: Box<dyn Beam> = Box::new(NoBeam);
    //     let srclist = get_small_source_list();
    //     let inst_flux_densities = get_instrumental_flux_densities_for_srclist(&srclist, &freqs);
    //     let result = match beam_correct_flux_densities_inner(
    //         inst_flux_densities.view(),
    //         beam.deref(),
    //         &srclist.get_azel_mwa(lst),
    //         &dipole_gains,
    //         &freqs,
    //     ) {
    //         Ok(fds) => fds,
    //         Err(e) => panic!("{}", e),
    //     };
    //     let num_components = srclist.values().fold(0, |a, src| a + src.components.len());
    //     assert_eq!(result.dim(), (freqs.len(), num_components));

    //     // Hand-verified results.
    //     let expected_comp_fd_1 = Jones::from([
    //         Complex::new(2.7473072919275476, 0.0),
    //         Complex::new(0.0, 0.0),
    //         Complex::new(0.0, 0.0),
    //         Complex::new(2.7473072919275476, 0.0),
    //     ]);
    //     let expected_comp_fd_2 = Jones::from([
    //         Complex::new(1.7047163998893684, 0.0),
    //         Complex::new(0.0, 0.0),
    //         Complex::new(0.0, 0.0),
    //         Complex::new(1.7047163998893684, 0.0),
    //     ]);
    //     assert_abs_diff_eq!(result[[0, 0]], expected_comp_fd_1, epsilon = 1e-10);
    //     assert_abs_diff_eq!(result[[0, 1]], expected_comp_fd_2, epsilon = 1e-10);
    // }

    // #[test]
    // #[serial]
    // fn test_beam_correct_flux_densities_170_mhz() {
    //     let freqs = [170e6];
    //     let lst = 6.261977848;
    //     let dipole_delays = vec![0; 16];

    //     let beam: Box<dyn Beam> =
    //         Box::new(FEEBeam::new_from_env(1, Delays::Partial(dipole_delays), None).unwrap());
    //     let srclist = get_small_source_list();
    //     let inst_flux_densities = get_instrumental_flux_densities_for_srclist(&srclist, &freqs);

    //     let result = match beam_correct_flux_densities_inner(
    //         inst_flux_densities.view(),
    //         beam.deref(),
    //         &srclist.get_azel_mwa(lst),
    //         &dipole_gains,
    //         &freqs,
    //     ) {
    //         Ok(fds) => fds,
    //         Err(e) => panic!("{}", e),
    //     };
    //     let num_components = srclist.values().fold(0, |a, src| a + src.components.len());
    //     assert_eq!(result.dim(), (freqs.len(), num_components));

    //     // Hand-verified results.
    //     let expected_comp_fd_1 = Jones::from([
    //         Complex::new(2.7473072919275476, 0.0),
    //         Complex::new(0.0, 0.0),
    //         Complex::new(0.0, 0.0),
    //         Complex::new(2.7473072919275476, 0.0),
    //     ]);
    //     let expected_jones_1 = Jones::from([
    //         Complex::new(0.7750324863535399, 0.24282289190335862),
    //         Complex::new(-0.009009420577898178, -0.002856655664463373),
    //         Complex::new(0.01021394523909512, 0.0033072019611734838),
    //         Complex::new(0.7814897063974989, 0.25556799755364396),
    //     ]);
    //     assert_abs_diff_eq!(
    //         result[[0, 0]],
    //         expected_jones_1 * expected_comp_fd_1 * expected_jones_1.h(),
    //         epsilon = 1e-10
    //     );

    //     let expected_comp_fd_2 = Jones::from([
    //         Complex::new(1.7047163998893684, 0.0),
    //         Complex::new(0.0, 0.0),
    //         Complex::new(0.0, 0.0),
    //         Complex::new(1.7047163998893684, 0.0),
    //     ]);
    //     let expected_jones_2 = Jones::from([
    //         Complex::new(0.9455907247090378, 0.3049292024132071),
    //         Complex::new(-0.010712295162757346, -0.0033779555969525588),
    //         Complex::new(0.010367761993275826, 0.003441723575945327),
    //         Complex::new(0.9450219468106582, 0.30598012238683214),
    //     ]);
    //     assert_abs_diff_eq!(
    //         result[[0, 1]],
    //         expected_jones_2 * expected_comp_fd_2 * expected_jones_2.h(),
    //         epsilon = 1e-10
    //     );
    // }

    // #[test]
    // #[serial]
    // // Same as above, but with a different frequency.
    // fn test_beam_correct_flux_densities_180_mhz() {
    //     let freqs = [180e6];
    //     let lst = 6.261977848;
    //     let dipole_delays = vec![0; 16];
    //     let dipole_gains = [1.0; 16];

    //     let beam: Box<dyn Beam> =
    //         create_fee_beam_object(None, 1, Delays::Partial(dipole_delays), None).unwrap();
    //     let srclist = get_small_source_list();
    //     let inst_flux_densities = get_instrumental_flux_densities_for_srclist(&srclist, &freqs);
    //     let result = match beam_correct_flux_densities_inner(
    //         inst_flux_densities.view(),
    //         beam.deref(),
    //         &srclist.get_azel_mwa(lst),
    //         &dipole_gains,
    //         &freqs,
    //     ) {
    //         Ok(fds) => fds,
    //         Err(e) => panic!("{}", e),
    //     };
    //     let num_components = srclist.values().fold(0, |a, src| a + src.components.len());
    //     assert_eq!(result.dim(), (freqs.len(), num_components));

    //     // Hand-verified results.
    //     let expected_comp_fd_1 = Jones::from([
    //         Complex::new(2.60247, 0.0),
    //         Complex::new(0.0, 0.0),
    //         Complex::new(0.0, 0.0),
    //         Complex::new(2.60247, 0.0),
    //     ]);
    //     let expected_jones_1 = Jones::from([
    //         Complex::new(0.7731976406423393, 0.17034253171231564),
    //         Complex::new(-0.009017301710718753, -0.001961964125441071),
    //         Complex::new(0.010223521132619665, 0.002456914956330356),
    //         Complex::new(0.7838681411558177, 0.186582048535625),
    //     ]);
    //     assert_abs_diff_eq!(
    //         result[[0, 0]],
    //         expected_jones_1 * expected_comp_fd_1 * expected_jones_1.h(),
    //         epsilon = 1e-10
    //     );

    //     let expected_comp_fd_2 = Jones::from([
    //         Complex::new(1.61824, 0.0),
    //         Complex::new(0.0, 0.0),
    //         Complex::new(0.0, 0.0),
    //         Complex::new(1.61824, 0.0),
    //     ]);
    //     let expected_jones_2 = Jones::from([
    //         Complex::new(0.9682339089232415, 0.2198904292735457),
    //         Complex::new(-0.01090619422142064, -0.0023800302690927533),
    //         Complex::new(0.010687354909991509, 0.002535994729487373),
    //         Complex::new(0.9676157155647803, 0.22121720658375732),
    //     ]);
    //     assert_abs_diff_eq!(
    //         result[[0, 1]],
    //         expected_jones_2 * expected_comp_fd_2 * expected_jones_2.h(),
    //         epsilon = 1e-10
    //     );
    // }

    #[test]
    fn test_split_components() {
        let freqs = [180e6];
        let phase_centre = RADec::new_degrees(0.0, -27.0);
        let srclist = get_big_source_list();

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

        let split_components = ComponentList::new(&srclist, &freqs, phase_centre);
        let points = split_components.points;
        let gaussians = split_components.gaussians;
        let shapelets = split_components.shapelets;

        assert_eq!(points.radecs.len(), num_point_components);
        assert_eq!(points.radecs.len(), 2);
        assert_eq!(gaussians.radecs.len(), num_gauss_components);
        assert_eq!(gaussians.radecs.len(), 4);
        assert!(shapelets.radecs.is_empty());

        assert_eq!(points.lmns.len(), num_point_components);
        assert_eq!(gaussians.lmns.len(), num_gauss_components);
        assert!(shapelets.lmns.is_empty());
        assert_abs_diff_eq!(points.lmns[0].l, 0.0025326811687516274 * TAU);
        assert_abs_diff_eq!(points.lmns[0].m, -0.12880688061967666 * TAU);
        assert_abs_diff_eq!(points.lmns[0].n, (0.9916664625927036 - 1.0) * TAU);

        assert_eq!(points.flux_densities.dim(), (1, num_point_components));
        assert_eq!(gaussians.flux_densities.dim(), (1, num_gauss_components));
        assert_eq!(shapelets.flux_densities.dim(), (1, 0));

        // Test one of the component's instrumental flux densities.
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
        .estimate_at_freq(freqs[0]);
        let inst_fd: Jones<f64> = fd.to_inst_stokes();
        assert_abs_diff_eq!(gaussians.flux_densities[[0, 2]], inst_fd);
    }
}
