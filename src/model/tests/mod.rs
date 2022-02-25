// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Tests on generating sky-model visibilities.

#[cfg(feature = "cuda")]
mod cuda;

use std::ops::Deref;

use approx::{assert_abs_diff_eq, assert_abs_diff_ne};
use itertools::Itertools;
use marlu::{
    constants::{MWA_LAT_RAD, MWA_LONG_RAD},
    pos::xyz::xyzs_to_cross_uvws_parallel,
    AzEl, Complex, Jones, RADec, XyzGeodetic,
};
use ndarray::prelude::*;
use serial_test::serial;

use super::*;
use crate::jones_test::TestJones;
#[cfg(feature = "cuda")]
use crate::model::cuda::SkyModellerCuda;
use mwa_hyperdrive_beam::{create_fee_beam_object, create_no_beam_object, Delays};
use mwa_hyperdrive_common::{itertools, marlu, ndarray};
use mwa_hyperdrive_srclist::{
    ComponentType, FluxDensity, FluxDensityType, ShapeletCoeff, Source, SourceComponent, SourceList,
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
    flagged_tiles: Vec<usize>,
    array_longitude_rad: f64,
    array_latitude_rad: f64,
}

impl ObsParams {
    fn list() -> ObsParams {
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
        let uvws = xyzs_to_cross_uvws_parallel(&xyzs, phase_centre.to_hadec(lst));
        let beam = create_no_beam_object(xyzs.len());
        let flagged_tiles = vec![];
        let array_longitude_rad = MWA_LONG_RAD;
        let array_latitude_rad = MWA_LAT_RAD;

        ObsParams {
            phase_centre,
            freqs,
            flux_density_scale,
            lst,
            xyzs,
            uvws,
            beam,
            flagged_tiles,
            array_longitude_rad,
            array_latitude_rad,
        }
    }

    fn power_law() -> ObsParams {
        let phase_centre = RADec::new_degrees(0.0, -27.0);
        let freqs = vec![150e6, 175e6, 200e6];

        let flux_density_scale = FluxDensityType::PowerLaw {
            si: -0.8,
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
        let uvws = xyzs_to_cross_uvws_parallel(&xyzs, phase_centre.to_hadec(lst));
        let beam = create_no_beam_object(xyzs.len());
        let flagged_tiles = vec![];
        let array_longitude_rad = MWA_LONG_RAD;
        let array_latitude_rad = MWA_LAT_RAD;

        ObsParams {
            phase_centre,
            freqs,
            flux_density_scale,
            lst,
            xyzs,
            uvws,
            beam,
            flagged_tiles,
            array_longitude_rad,
            array_latitude_rad,
        }
    }

    fn get_cpu_modeller(&self, srclist: &SourceList) -> SkyModellerCpu {
        new_cpu_sky_modeller_inner(
            self.beam.deref(),
            srclist,
            &self.xyzs,
            &self.freqs,
            &self.flagged_tiles,
            self.phase_centre,
            self.array_longitude_rad,
            self.array_latitude_rad,
            true,
        )
    }

    #[cfg(feature = "cuda")]
    fn get_gpu_modeller(&self, srclist: &SourceList) -> SkyModellerCuda {
        unsafe {
            super::new_cuda_sky_modeller_inner(
                self.beam.deref(),
                srclist,
                &self.xyzs,
                &self.freqs,
                &self.flagged_tiles,
                self.phase_centre,
                self.array_longitude_rad,
                self.array_latitude_rad,
                true,
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
        vis.slice(s![.., 0]).mapv(TestJones::from),
        Array1::from_elem(vis.dim().0, Jones::identity()).mapv(TestJones::from),
    );
    // 3 Jy XX and YY.
    assert_abs_diff_eq!(
        vis.slice(s![.., 1]).mapv(TestJones::from),
        Array1::from_elem(vis.dim().0, Jones::identity() * 3.0).mapv(TestJones::from),
    );
    // 2 Jy XX and YY.
    assert_abs_diff_eq!(
        vis.slice(s![.., 2]).mapv(TestJones::from),
        Array1::from_elem(vis.dim().0, Jones::identity() * 2.0).mapv(TestJones::from),
    );
}

fn assert_list_off_zenith_visibilities(vis: ArrayView2<Jones<f32>>) {
    // This time, all LMN values should be close to, but not the same as, (0, 0,
    // 1). This means that the visibilities should be somewhat close to the
    // input flux densities.

    assert_abs_diff_eq!(
        vis.slice(s![.., 0]).mapv(TestJones::from),
        array![
            TestJones::from([
                Complex::new(0.99522406, 0.09761678),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.99522406, 0.09761678),
            ]),
            TestJones::from([
                Complex::new(0.99878436, 0.049292877),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.99878436, 0.049292877),
            ]),
            TestJones::from([
                Complex::new(0.9988261, -0.04844065),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.9988261, -0.04844065),
            ]),
        ],
    );

    assert_abs_diff_eq!(
        vis.slice(s![.., 1]).mapv(TestJones::from),
        array![
            TestJones::from([
                Complex::new(2.980504, 0.34146205),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(2.980504, 0.34146205),
            ]),
            TestJones::from([
                Complex::new(2.9950366, 0.17249982),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(2.9950366, 0.17249982),
            ]),
            TestJones::from([
                Complex::new(2.9952068, -0.1695183),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(2.9952068, -0.1695183),
            ]),
        ],
    );

    assert_abs_diff_eq!(
        vis.slice(s![.., 2]).mapv(TestJones::from),
        array![
            TestJones::from([
                Complex::new(1.9830295, 0.25998875),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(1.9830295, 0.25998875),
            ]),
            TestJones::from([
                Complex::new(1.9956784, 0.13140623),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(1.9956784, 0.13140623),
            ]),
            TestJones::from([
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
    let modeller = obs.get_cpu_modeller(&srclist);
    let mut visibilities = Array2::zeros((obs.uvws.len(), obs.freqs.len()));
    let result = modeller.model_points_inner(visibilities.view_mut(), &obs.uvws, obs.lst);
    assert!(result.is_ok());
    assert_list_zenith_visibilities(visibilities.view());
}

// Put a single point source just off zenith.
#[test]
fn point_off_zenith_cpu() {
    let obs = ObsParams::list();
    let mut srclist = SourceList::new();
    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec![get_simple_point(
                RADec::new_degrees(1.0, -27.0),
                obs.flux_density_scale.clone(),
            )],
        },
    );
    let modeller = obs.get_cpu_modeller(&srclist);
    let mut visibilities = Array2::zeros((obs.uvws.len(), obs.freqs.len()));
    let result = modeller.model_points_inner(visibilities.view_mut(), &obs.uvws, obs.lst);
    assert!(result.is_ok());
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
    let modeller = obs.get_cpu_modeller(&srclist);
    let mut visibilities = Array2::zeros((obs.uvws.len(), obs.freqs.len()));
    let result = modeller.model_gaussians_inner(visibilities.view_mut(), &obs.uvws, obs.lst);
    assert!(result.is_ok());
    assert_list_zenith_visibilities(visibilities.view());
}

// Put a single Gaussian source just off zenith.
#[test]
fn gaussian_off_zenith_cpu() {
    let obs = ObsParams::list();
    let mut srclist = SourceList::new();
    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec![get_simple_gaussian(
                RADec::new_degrees(1.0, -27.0),
                obs.flux_density_scale.clone(),
            )],
        },
    );
    let modeller = obs.get_cpu_modeller(&srclist);
    let mut visibilities = Array2::zeros((obs.uvws.len(), obs.freqs.len()));
    let result = modeller.model_gaussians_inner(visibilities.view_mut(), &obs.uvws, obs.lst);
    assert!(result.is_ok());
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
    let modeller = obs.get_cpu_modeller(&srclist);
    let shapelet_uvws = modeller
        .components
        .shapelets
        .get_shapelet_uvws(obs.lst, &obs.xyzs);
    let mut visibilities = Array2::zeros((obs.uvws.len(), obs.freqs.len()));
    let result = modeller.model_shapelets_inner(
        visibilities.view_mut(),
        &obs.uvws,
        shapelet_uvws.view(),
        obs.lst,
    );
    assert!(result.is_ok());
    assert_list_zenith_visibilities(visibilities.view());
}

// Put a single shapelet source just off zenith.
#[test]
fn shapelet_off_zenith_cpu() {
    let obs = ObsParams::list();
    let mut srclist = SourceList::new();
    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec![get_simple_shapelet(
                RADec::new_degrees(1.0, -27.0),
                obs.flux_density_scale.clone(),
            )],
        },
    );
    let modeller = obs.get_cpu_modeller(&srclist);
    let shapelet_uvws = modeller
        .components
        .shapelets
        .get_shapelet_uvws(obs.lst, &obs.xyzs);
    let mut visibilities = Array2::zeros((obs.uvws.len(), obs.freqs.len()));
    let result = modeller.model_shapelets_inner(
        visibilities.view_mut(),
        &obs.uvws,
        shapelet_uvws.view(),
        obs.lst,
    );
    assert!(result.is_ok());
    assert_list_off_zenith_visibilities(visibilities.view());
}
