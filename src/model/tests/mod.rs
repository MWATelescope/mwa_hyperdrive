// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Tests on generating sky-model visibilities.

#[cfg(feature = "cuda")]
mod cuda;

use std::collections::HashSet;
use std::ops::Deref;

use approx::assert_abs_diff_eq;
use mwa_rust_core::{HADec, Jones, RADec, XyzGeodetic};
use ndarray::prelude::*;

use super::*;
use crate::math::TileBaselineMaps;
use mwa_hyperdrive_beam::create_no_beam_object;
use mwa_hyperdrive_srclist::{
    constants::DEFAULT_SPEC_INDEX, ComponentList, ComponentType, FluxDensity, FluxDensityType,
    Source, SourceComponent, SourceList,
};

fn get_simple_point(pos: RADec, flux_density_scale: FluxDensityType) -> SourceComponent {
    SourceComponent {
        radec: pos,
        comp_type: ComponentType::Point,
        flux_type: flux_density_scale,
    }
}

fn get_simple_gaussian(pos: RADec, flux_density_scale: FluxDensityType) -> SourceComponent {
    SourceComponent {
        radec: pos,
        comp_type: ComponentType::Gaussian {
            maj: 0.0,
            min: 0.0,
            pa: 0.0,
        },
        flux_type: flux_density_scale,
    }
}

fn get_simple_shapelet(pos: RADec, flux_density_scale: FluxDensityType) -> SourceComponent {
    SourceComponent {
        radec: pos,
        comp_type: ComponentType::Shapelet {
            maj: 0.0,
            min: 0.0,
            pa: 0.0,
            coeffs: vec![ShapeletCoeff {
                n1: 0,
                n2: 0,
                value: 1.0,
            }],
        },
        flux_type: flux_density_scale,
    }
}

struct ObsParams {
    phase_centre: RADec,
    freqs: Vec<f64>,
    flux_density_scale: FluxDensityType,
    lst: f64,
    xyzs: Vec<XyzGeodetic>,
    uvws: Vec<UVW>,
    beam: Box<dyn Beam>,
    num_unflagged_cross_baselines: usize,
    unflagged_cross_baseline_to_tile_map: HashMap<usize, (usize, usize)>,
    tile_flags: HashSet<usize>,
    array_latitude_rad: f64,
}

impl ObsParams {
    fn list() -> Self {
        let phase_centre = RADec::new_degrees(0.0, -27.0);
        let freqs = vec![150e6, 175e6, 200e6];

        let flux_density_scale = FluxDensityType::List {
            fds: vec![
                FluxDensity {
                    freq: 150e6,
                    i: 1.0,
                    ..Default::default()
                },
                FluxDensity {
                    freq: 175e6,
                    i: 3.0,
                    ..Default::default()
                },
                FluxDensity {
                    freq: 200e6,
                    i: 2.0,
                    ..Default::default()
                },
            ],
        };

        let lst = 0.0;
        let xyzs = vec![
            XyzGeodetic {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            },
            XyzGeodetic {
                x: 0.0,
                y: -1.0,
                z: 0.0,
            },
            XyzGeodetic {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
        ];
        let num_unflagged_cross_baselines = (xyzs.len() * (xyzs.len() - 1)) / 2;
        let uvws = xyzs_to_cross_uvws_parallel(&xyzs, phase_centre.to_hadec(lst));
        let beam = create_no_beam_object(xyzs.len());
        let tile_flags = HashSet::new();
        let maps = TileBaselineMaps::new(xyzs.len(), &tile_flags);
        let array_latitude_rad = MWA_LAT_RAD;

        Self {
            phase_centre,
            freqs,
            flux_density_scale,
            lst,
            xyzs,
            uvws,
            beam,
            num_unflagged_cross_baselines,
            unflagged_cross_baseline_to_tile_map: maps.unflagged_cross_baseline_to_tile_map,
            tile_flags,
            array_latitude_rad,
        }
    }

    fn power_law() -> Self {
        let phase_centre = RADec::new_degrees(0.0, -27.0);
        let freqs = vec![150e6, 175e6, 200e6];

        let flux_density_scale = FluxDensityType::PowerLaw {
            si: DEFAULT_SPEC_INDEX,
            fd: FluxDensity {
                freq: 150e6,
                i: 1.0,
                ..Default::default()
            },
        };

        let lst = 0.0;
        let xyzs = vec![
            XyzGeodetic {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            },
            XyzGeodetic {
                x: 0.0,
                y: -1.0,
                z: 0.0,
            },
            XyzGeodetic {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
        ];
        let num_unflagged_cross_baselines = (xyzs.len() * (xyzs.len() - 1)) / 2;
        let uvws = xyzs_to_cross_uvws_parallel(&xyzs, phase_centre.to_hadec(lst));
        let beam = create_no_beam_object(xyzs.len());
        let tile_flags = HashSet::new();
        let maps = TileBaselineMaps::new(xyzs.len(), &tile_flags);
        let array_latitude_rad = MWA_LAT_RAD;

        Self {
            phase_centre,
            freqs,
            flux_density_scale,
            lst,
            xyzs,
            uvws,
            beam,
            num_unflagged_cross_baselines,
            unflagged_cross_baseline_to_tile_map: maps.unflagged_cross_baseline_to_tile_map,
            tile_flags,
            array_latitude_rad,
        }
    }

    #[cfg(feature = "cuda")]
    fn get_gpu_modeller(&self, srclist: SourceList) -> mwa_hyperdrive_cuda::SkyModellerCuda {
        unsafe {
            mwa_hyperdrive_cuda::SkyModellerCuda::new(
                self.beam.deref(),
                &srclist,
                &self.freqs,
                &self.xyzs,
                &self.tile_flags,
                self.phase_centre,
                self.array_latitude_rad,
                &crate::shapelets::SHAPELET_BASIS_VALUES,
                crate::shapelets::SBF_L,
                crate::shapelets::SBF_N,
                crate::shapelets::SBF_C,
                crate::shapelets::SBF_DX,
            )
            .unwrap()
        }
    }
}

fn assert_list_zenith_visibilities(vis: ArrayView2<Jones<f32>>) {
    // All LMN values are (0, 0, 1). This means that the Fourier transform to
    // make a visibility from LMN and UVW will always just be the input flux
    // density.

    // First column: flux density at first frequency (i.e. 1 Jy XX and YY).
    assert_abs_diff_eq!(
        vis.slice(s![.., 0]),
        Array1::from_elem(vis.dim().0, Jones::identity()),
    );
    // 3 Jy XX and YY.
    assert_abs_diff_eq!(
        vis.slice(s![.., 1]),
        Array1::from_elem(vis.dim().0, Jones::identity() * 3.0),
    );
    // 2 Jy XX and YY.
    assert_abs_diff_eq!(
        vis.slice(s![.., 2]),
        Array1::from_elem(vis.dim().0, Jones::identity() * 2.0),
    );
}

fn assert_list_off_zenith_visibilities(vis: ArrayView2<Jones<f32>>) {
    // This time, all LMN values should be close to, but not the same as, (0, 0,
    // 1). This means that the visibilities should be somewhat close to the
    // input flux densities.

    assert_abs_diff_eq!(
        vis.slice(s![.., 0]),
        array![
            Jones::from([
                Complex::new(0.99522406, 0.09761678),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.99522406, 0.09761678),
            ]),
            Jones::from([
                Complex::new(0.99878436, 0.049292877),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.99878436, 0.049292877),
            ]),
            Jones::from([
                Complex::new(0.9988261, -0.04844065),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.9988261, -0.04844065),
            ]),
        ],
    );

    assert_abs_diff_eq!(
        vis.slice(s![.., 1]),
        array![
            Jones::from([
                Complex::new(2.980504, 0.34146205),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(2.980504, 0.34146205),
            ]),
            Jones::from([
                Complex::new(2.9950366, 0.17249982),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(2.9950366, 0.17249982),
            ]),
            Jones::from([
                Complex::new(2.9952068, -0.1695183),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(2.9952068, -0.1695183),
            ]),
        ],
    );

    assert_abs_diff_eq!(
        vis.slice(s![.., 2]),
        array![
            Jones::from([
                Complex::new(1.9830295, 0.25998875),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(1.9830295, 0.25998875),
            ]),
            Jones::from([
                Complex::new(1.9956784, 0.13140623),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(1.9956784, 0.13140623),
            ]),
            Jones::from([
                Complex::new(1.9958266, -0.12913574),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(1.9958266, -0.12913574),
            ]),
        ],
    );
}

// Put a single point source at zenith.
#[test]
fn point_zenith_cpu() {
    let obs = ObsParams::list();

    let mut srclist = SourceList::new();
    srclist.insert(
        "zenith".to_string(),
        Source {
            components: vec![get_simple_point(
                obs.phase_centre,
                obs.flux_density_scale.clone(),
            )],
        },
    );
    // Get the component parameters via `ComponentList`.
    let comps = ComponentList::new(&srclist, &obs.freqs, obs.phase_centre);

    // Ignore applying the beam.
    let mut visibilities = Array2::from_elem((obs.uvws.len(), obs.freqs.len()), Jones::default());
    model_points(
        visibilities.view_mut(),
        obs.beam.deref(),
        &comps.points.get_azels_mwa_parallel(obs.lst),
        &comps.points.lmns,
        comps.points.flux_densities.view(),
        &obs.uvws,
        &obs.freqs,
        &obs.unflagged_cross_baseline_to_tile_map,
    );
    assert_list_zenith_visibilities(visibilities.view());
}

// Put a single point source just off zenith.
#[test]
fn point_off_zenith_cpu() {
    let obs = ObsParams::list();
    let pos = RADec::new_degrees(1.0, -27.0);

    let mut srclist = SourceList::new();
    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec![get_simple_point(pos, obs.flux_density_scale.clone())],
        },
    );
    let comps = ComponentList::new(&srclist, &obs.freqs, obs.phase_centre);

    let mut visibilities = Array2::from_elem((obs.uvws.len(), obs.freqs.len()), Jones::default());
    model_points(
        visibilities.view_mut(),
        obs.beam.deref(),
        &comps.points.get_azels_mwa_parallel(obs.lst),
        &comps.points.lmns,
        comps.points.flux_densities.view(),
        &obs.uvws,
        &obs.freqs,
        &obs.unflagged_cross_baseline_to_tile_map,
    );
    assert_list_off_zenith_visibilities(visibilities.view());
}

// Put a single Gaussian source at zenith.
#[test]
fn gaussian_zenith_cpu() {
    let obs = ObsParams::list();

    let mut srclist = SourceList::new();
    srclist.insert(
        "zenith".to_string(),
        Source {
            components: vec![get_simple_gaussian(
                obs.phase_centre,
                obs.flux_density_scale.clone(),
            )],
        },
    );
    let comps = ComponentList::new(&srclist, &obs.freqs, obs.phase_centre);

    let mut visibilities = Array2::from_elem((obs.uvws.len(), obs.freqs.len()), Jones::default());
    model_gaussians(
        visibilities.view_mut(),
        obs.beam.deref(),
        &comps.gaussians.get_azels_mwa_parallel(obs.lst),
        &comps.gaussians.lmns,
        &comps.gaussians.gaussian_params,
        comps.gaussians.flux_densities.view(),
        &obs.uvws,
        &obs.freqs,
        &obs.unflagged_cross_baseline_to_tile_map,
    );
    assert_list_zenith_visibilities(visibilities.view());
}

// Put a single Gaussian source just off zenith.
#[test]
fn gaussian_off_zenith_cpu() {
    let obs = ObsParams::list();
    let pos = RADec::new_degrees(1.0, -27.0);

    let mut srclist = SourceList::new();
    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec![get_simple_gaussian(pos, obs.flux_density_scale.clone())],
        },
    );
    let comps = ComponentList::new(&srclist, &obs.freqs, obs.phase_centre);

    let mut visibilities = Array2::from_elem((obs.uvws.len(), obs.freqs.len()), Jones::default());
    model_gaussians(
        visibilities.view_mut(),
        obs.beam.deref(),
        &comps.gaussians.get_azels_mwa_parallel(obs.lst),
        &comps.gaussians.lmns,
        &comps.gaussians.gaussian_params,
        comps.gaussians.flux_densities.view(),
        &obs.uvws,
        &obs.freqs,
        &obs.unflagged_cross_baseline_to_tile_map,
    );
    assert_list_off_zenith_visibilities(visibilities.view());
}

// Put a single shapelet source at zenith.
#[test]
fn shapelet_zenith_cpu() {
    let obs = ObsParams::list();

    let mut srclist = SourceList::new();
    srclist.insert(
        "zenith".to_string(),
        Source {
            components: vec![get_simple_shapelet(
                obs.phase_centre,
                obs.flux_density_scale.clone(),
            )],
        },
    );
    let comps = ComponentList::new(&srclist, &obs.freqs, obs.phase_centre);

    let mut visibilities = Array2::from_elem((obs.uvws.len(), obs.freqs.len()), Jones::default());
    let shapelet_uvws = comps.shapelets.get_shapelet_uvws(obs.lst, &obs.xyzs);
    model_shapelets(
        visibilities.view_mut(),
        obs.beam.deref(),
        &comps.shapelets.get_azels_mwa_parallel(obs.lst),
        &comps.shapelets.lmns,
        &comps.shapelets.gaussian_params,
        &comps.shapelets.shapelet_coeffs,
        shapelet_uvws.view(),
        comps.shapelets.flux_densities.view(),
        &obs.uvws,
        &obs.freqs,
        &obs.unflagged_cross_baseline_to_tile_map,
    );
    assert_list_zenith_visibilities(visibilities.view());
}

// Put a single shapelet source just off zenith.
#[test]
fn shapelet_off_zenith_cpu() {
    let obs = ObsParams::list();
    let pos = RADec::new_degrees(1.0, -27.0);

    let mut srclist = SourceList::new();
    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec![get_simple_shapelet(pos, obs.flux_density_scale.clone())],
        },
    );
    let comps = ComponentList::new(&srclist, &obs.freqs, obs.phase_centre);

    let mut visibilities = Array2::from_elem((obs.uvws.len(), obs.freqs.len()), Jones::default());
    let shapelet_uvws = comps.shapelets.get_shapelet_uvws(obs.lst, &obs.xyzs);
    model_shapelets(
        visibilities.view_mut(),
        obs.beam.deref(),
        &comps.shapelets.get_azels_mwa_parallel(obs.lst),
        &comps.shapelets.lmns,
        &comps.shapelets.gaussian_params,
        &comps.shapelets.shapelet_coeffs,
        shapelet_uvws.view(),
        comps.shapelets.flux_densities.view(),
        &obs.uvws,
        &obs.freqs,
        &obs.unflagged_cross_baseline_to_tile_map,
    );
    assert_list_off_zenith_visibilities(visibilities.view());
}

// TODO: Have in mwa_rust_core
/// Convert [XyzGeodetic] tile coordinates to [UVW] baseline coordinates without
/// having to form [XyzGeodetic] baselines first. This function performs
/// calculations in parallel. Cross-correlation baselines only.
fn xyzs_to_cross_uvws_parallel(xyzs: &[XyzGeodetic], phase_centre: HADec) -> Vec<UVW> {
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
