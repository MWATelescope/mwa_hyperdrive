// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Tests on generating sky-model visibilities on a CPU.

use marlu::RADec;
use ndarray::prelude::*;
use serial_test::serial;
use vec1::vec1;

use super::*;
use crate::srclist::{Source, SourceList};

// Put a single point source at zenith.
#[test]
fn point_zenith_cpu() {
    let obs = ObsParams::new(true);
    let modeller = obs.get_cpu_modeller(&POINT_ZENITH_LIST);
    let mut visibilities = Array2::zeros((obs.freqs.len(), obs.uvws.len()));
    let result = modeller.model_points(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_list_zenith_visibilities(visibilities.view());

    let modeller = obs.get_cpu_modeller(&POINT_ZENITH_POWER_LAW);
    visibilities.fill(Jones::default());
    let result = modeller.model_points(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_power_law_zenith_visibilities(visibilities.view());

    let modeller = obs.get_cpu_modeller(&POINT_ZENITH_CURVED_POWER_LAW);
    visibilities.fill(Jones::default());
    let result = modeller.model_points(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_curved_power_law_zenith_visibilities(visibilities.view());
}

// Put a single point source just off zenith.
#[test]
fn point_off_zenith_cpu() {
    let obs = ObsParams::new(true);
    let modeller = obs.get_cpu_modeller(&POINT_OFF_ZENITH_LIST);
    let mut visibilities = Array2::zeros((obs.freqs.len(), obs.uvws.len()));
    let result = modeller.model_points(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_list_off_zenith_visibilities(visibilities.view());

    let modeller = obs.get_cpu_modeller(&POINT_OFF_ZENITH_POWER_LAW);
    visibilities.fill(Jones::default());
    let result = modeller.model_points(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_power_law_off_zenith_visibilities(visibilities.view());

    let modeller = obs.get_cpu_modeller(&POINT_OFF_ZENITH_CURVED_POWER_LAW);
    visibilities.fill(Jones::default());
    let result = modeller.model_points(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_curved_power_law_off_zenith_visibilities(visibilities.view());
}

// Put a single Gaussian source at zenith.
#[test]
fn gaussian_zenith_cpu() {
    let obs = ObsParams::new(true);
    let modeller = obs.get_cpu_modeller(&GAUSSIAN_ZENITH_LIST);
    let mut visibilities = Array2::zeros((obs.freqs.len(), obs.uvws.len()));
    let result = modeller.model_gaussians(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_list_zenith_visibilities(visibilities.view());

    let modeller = obs.get_cpu_modeller(&GAUSSIAN_ZENITH_POWER_LAW);
    visibilities.fill(Jones::default());
    let result = modeller.model_gaussians(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_power_law_zenith_visibilities(visibilities.view());

    let modeller = obs.get_cpu_modeller(&GAUSSIAN_ZENITH_CURVED_POWER_LAW);
    visibilities.fill(Jones::default());
    let result = modeller.model_gaussians(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_curved_power_law_zenith_visibilities(visibilities.view());
}

// Put a single Gaussian source just off zenith.
#[test]
fn gaussian_off_zenith_cpu() {
    let obs = ObsParams::new(true);
    let modeller = obs.get_cpu_modeller(&GAUSSIAN_OFF_ZENITH_LIST);
    let mut visibilities = Array2::zeros((obs.freqs.len(), obs.uvws.len()));
    let result = modeller.model_gaussians(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_list_off_zenith_visibilities(visibilities.view());

    let modeller = obs.get_cpu_modeller(&GAUSSIAN_OFF_ZENITH_POWER_LAW);
    visibilities.fill(Jones::default());
    let result = modeller.model_gaussians(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_power_law_off_zenith_visibilities(visibilities.view());

    let modeller = obs.get_cpu_modeller(&GAUSSIAN_OFF_ZENITH_CURVED_POWER_LAW);
    visibilities.fill(Jones::default());
    let result = modeller.model_gaussians(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_curved_power_law_off_zenith_visibilities(visibilities.view());
}

// Put a single shapelet source at zenith.
#[test]
fn shapelet_zenith_cpu() {
    let obs = ObsParams::new(true);
    let modeller = obs.get_cpu_modeller(&SHAPELET_ZENITH_LIST);
    let shapelet_uvws = modeller
        .components
        .shapelets
        .get_shapelet_uvws(obs.lst, &obs.xyzs);
    let mut visibilities = Array2::zeros((obs.freqs.len(), obs.uvws.len()));
    let result = modeller.model_shapelets(
        visibilities.view_mut(),
        &obs.uvws,
        shapelet_uvws.view(),
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_list_zenith_visibilities(visibilities.view());

    let modeller = obs.get_cpu_modeller(&SHAPELET_ZENITH_POWER_LAW);
    let shapelet_uvws = modeller
        .components
        .shapelets
        .get_shapelet_uvws(obs.lst, &obs.xyzs);
    visibilities.fill(Jones::default());
    let result = modeller.model_shapelets(
        visibilities.view_mut(),
        &obs.uvws,
        shapelet_uvws.view(),
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_power_law_zenith_visibilities(visibilities.view());

    let modeller = obs.get_cpu_modeller(&SHAPELET_ZENITH_CURVED_POWER_LAW);
    let shapelet_uvws = modeller
        .components
        .shapelets
        .get_shapelet_uvws(obs.lst, &obs.xyzs);
    visibilities.fill(Jones::default());
    let result = modeller.model_shapelets(
        visibilities.view_mut(),
        &obs.uvws,
        shapelet_uvws.view(),
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_curved_power_law_zenith_visibilities(visibilities.view());
}

// Put a single shapelet source just off zenith.
#[test]
fn shapelet_off_zenith_cpu() {
    let obs = ObsParams::new(true);
    let modeller = obs.get_cpu_modeller(&SHAPELET_OFF_ZENITH_LIST);
    let shapelet_uvws = modeller
        .components
        .shapelets
        .get_shapelet_uvws(obs.lst, &obs.xyzs);
    let mut visibilities = Array2::zeros((obs.freqs.len(), obs.uvws.len()));
    let result = modeller.model_shapelets(
        visibilities.view_mut(),
        &obs.uvws,
        shapelet_uvws.view(),
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_list_off_zenith_visibilities(visibilities.view());

    let modeller = obs.get_cpu_modeller(&SHAPELET_OFF_ZENITH_POWER_LAW);
    let shapelet_uvws = modeller
        .components
        .shapelets
        .get_shapelet_uvws(obs.lst, &obs.xyzs);
    visibilities.fill(Jones::default());
    let result = modeller.model_shapelets(
        visibilities.view_mut(),
        &obs.uvws,
        shapelet_uvws.view(),
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_power_law_off_zenith_visibilities(visibilities.view());

    let modeller = obs.get_cpu_modeller(&SHAPELET_OFF_ZENITH_CURVED_POWER_LAW);
    let shapelet_uvws = modeller
        .components
        .shapelets
        .get_shapelet_uvws(obs.lst, &obs.xyzs);
    visibilities.fill(Jones::default());
    let result = modeller.model_shapelets(
        visibilities.view_mut(),
        &obs.uvws,
        shapelet_uvws.view(),
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_curved_power_law_off_zenith_visibilities(visibilities.view());
}

// Put a single point source at zenith (FEE beam).
#[test]
#[serial]
fn point_zenith_cpu_fee() {
    let obs = ObsParams::new(false);
    let modeller = obs.get_cpu_modeller(&POINT_ZENITH_LIST);
    let mut visibilities = Array2::zeros((obs.freqs.len(), obs.uvws.len()));
    let result = modeller.model_points(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_list_zenith_visibilities_fee(visibilities.view());

    let modeller = obs.get_cpu_modeller(&POINT_ZENITH_POWER_LAW);
    visibilities.fill(Jones::default());
    let result = modeller.model_points(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_power_law_zenith_visibilities_fee(visibilities.view());

    let modeller = obs.get_cpu_modeller(&POINT_ZENITH_CURVED_POWER_LAW);
    visibilities.fill(Jones::default());
    let result = modeller.model_points(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_curved_power_law_zenith_visibilities_fee(visibilities.view());
}

// Put a single point source just off zenith (FEE beam).
#[test]
#[serial]
fn point_off_zenith_cpu_fee() {
    let obs = ObsParams::new(false);
    let modeller = obs.get_cpu_modeller(&POINT_OFF_ZENITH_LIST);
    let mut visibilities = Array2::zeros((obs.freqs.len(), obs.uvws.len()));
    let result = modeller.model_points(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_list_off_zenith_visibilities_fee(visibilities.view());

    let modeller = obs.get_cpu_modeller(&POINT_OFF_ZENITH_POWER_LAW);
    visibilities.fill(Jones::default());
    let result = modeller.model_points(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_power_law_off_zenith_visibilities_fee(visibilities.view());

    let modeller = obs.get_cpu_modeller(&POINT_OFF_ZENITH_CURVED_POWER_LAW);
    visibilities.fill(Jones::default());
    let result = modeller.model_points(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_curved_power_law_off_zenith_visibilities_fee(visibilities.view());
}

// Put a single Gaussian source at zenith (FEE beam).
#[test]
#[serial]
fn gaussian_zenith_cpu_fee() {
    let obs = ObsParams::new(false);
    let modeller = obs.get_cpu_modeller(&GAUSSIAN_ZENITH_LIST);
    let mut visibilities = Array2::zeros((obs.freqs.len(), obs.uvws.len()));
    let result = modeller.model_gaussians(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_list_zenith_visibilities_fee(visibilities.view());

    let modeller = obs.get_cpu_modeller(&GAUSSIAN_ZENITH_POWER_LAW);
    visibilities.fill(Jones::default());
    let result = modeller.model_gaussians(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_power_law_zenith_visibilities_fee(visibilities.view());

    let modeller = obs.get_cpu_modeller(&GAUSSIAN_ZENITH_CURVED_POWER_LAW);
    visibilities.fill(Jones::default());
    let result = modeller.model_gaussians(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_curved_power_law_zenith_visibilities_fee(visibilities.view());
}

// Put a single Gaussian source just off zenith (FEE beam).
#[test]
#[serial]
fn gaussian_off_zenith_cpu_fee() {
    let obs = ObsParams::new(false);
    let modeller = obs.get_cpu_modeller(&GAUSSIAN_OFF_ZENITH_LIST);
    let mut visibilities = Array2::zeros((obs.freqs.len(), obs.uvws.len()));
    let result = modeller.model_gaussians(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_list_off_zenith_visibilities_fee(visibilities.view());

    let modeller = obs.get_cpu_modeller(&GAUSSIAN_OFF_ZENITH_POWER_LAW);
    visibilities.fill(Jones::default());
    let result = modeller.model_gaussians(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_power_law_off_zenith_visibilities_fee(visibilities.view());

    let modeller = obs.get_cpu_modeller(&GAUSSIAN_OFF_ZENITH_CURVED_POWER_LAW);
    visibilities.fill(Jones::default());
    let result = modeller.model_gaussians(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_curved_power_law_off_zenith_visibilities_fee(visibilities.view());
}

// Put a single shapelet source at zenith (FEE beam).
#[test]
#[serial]
fn shapelet_zenith_cpu_fee() {
    let obs = ObsParams::new(false);
    let modeller = obs.get_cpu_modeller(&SHAPELET_ZENITH_LIST);
    let shapelet_uvws = modeller
        .components
        .shapelets
        .get_shapelet_uvws(obs.lst, &obs.xyzs);
    let mut visibilities = Array2::zeros((obs.freqs.len(), obs.uvws.len()));
    let result = modeller.model_shapelets(
        visibilities.view_mut(),
        &obs.uvws,
        shapelet_uvws.view(),
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_list_zenith_visibilities_fee(visibilities.view());

    let modeller = obs.get_cpu_modeller(&SHAPELET_ZENITH_POWER_LAW);
    let shapelet_uvws = modeller
        .components
        .shapelets
        .get_shapelet_uvws(obs.lst, &obs.xyzs);
    visibilities.fill(Jones::default());
    let result = modeller.model_shapelets(
        visibilities.view_mut(),
        &obs.uvws,
        shapelet_uvws.view(),
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_power_law_zenith_visibilities_fee(visibilities.view());

    let modeller = obs.get_cpu_modeller(&SHAPELET_ZENITH_CURVED_POWER_LAW);
    let shapelet_uvws = modeller
        .components
        .shapelets
        .get_shapelet_uvws(obs.lst, &obs.xyzs);
    visibilities.fill(Jones::default());
    let result = modeller.model_shapelets(
        visibilities.view_mut(),
        &obs.uvws,
        shapelet_uvws.view(),
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_curved_power_law_zenith_visibilities_fee(visibilities.view());
}

// Put a single shapelet source just off zenith (FEE beam).
#[test]
#[serial]
fn shapelet_off_zenith_cpu_fee() {
    let obs = ObsParams::new(false);
    let modeller = obs.get_cpu_modeller(&SHAPELET_OFF_ZENITH_LIST);
    let shapelet_uvws = modeller
        .components
        .shapelets
        .get_shapelet_uvws(obs.lst, &obs.xyzs);
    let mut visibilities = Array2::zeros((obs.freqs.len(), obs.uvws.len()));
    let result = modeller.model_shapelets(
        visibilities.view_mut(),
        &obs.uvws,
        shapelet_uvws.view(),
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_list_off_zenith_visibilities_fee(visibilities.view());

    let modeller = obs.get_cpu_modeller(&SHAPELET_OFF_ZENITH_POWER_LAW);
    let shapelet_uvws = modeller
        .components
        .shapelets
        .get_shapelet_uvws(obs.lst, &obs.xyzs);
    visibilities.fill(Jones::default());
    let result = modeller.model_shapelets(
        visibilities.view_mut(),
        &obs.uvws,
        shapelet_uvws.view(),
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_power_law_off_zenith_visibilities_fee(visibilities.view());

    let modeller = obs.get_cpu_modeller(&SHAPELET_OFF_ZENITH_CURVED_POWER_LAW);
    let shapelet_uvws = modeller
        .components
        .shapelets
        .get_shapelet_uvws(obs.lst, &obs.xyzs);
    visibilities.fill(Jones::default());
    let result = modeller.model_shapelets(
        visibilities.view_mut(),
        &obs.uvws,
        shapelet_uvws.view(),
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_curved_power_law_off_zenith_visibilities_fee(visibilities.view());
}

#[test]
fn shapelet_multiple_components() {
    let obs = ObsParams::new(true);
    let mut srclist = SourceList::new();
    srclist.insert(
        "shapelet".to_string(),
        Source {
            components: vec1![
                get_simple_shapelet(RADec::new_degrees(1.0, -27.0), FluxType::List),
                get_simple_shapelet(RADec::new_degrees(1.1, -27.0), FluxType::List)
            ],
        },
    );
    let modeller = obs.get_cpu_modeller(&srclist);
    let shapelet_uvws = modeller
        .components
        .shapelets
        .get_shapelet_uvws(obs.lst, &obs.xyzs);
    let mut visibilities = Array2::zeros((obs.freqs.len(), obs.uvws.len()));
    let result = modeller.model_shapelets(
        visibilities.view_mut(),
        &obs.uvws,
        shapelet_uvws.view(),
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());

    let expected = array![
        [
            Jones::from([
                Complex::new(1.9894463e0, 2.0495814e-1),
                Complex::new(0e0, 0e0),
                Complex::new(0e0, 0e0),
                Complex::new(1.9894463e0, 2.0495814e-1),
            ]),
            Jones::from([
                Complex::new(1.997311e0, 1.03556715e-1),
                Complex::new(0e0, 0e0),
                Complex::new(0e0, 0e0),
                Complex::new(1.997311e0, 1.03556715e-1),
            ]),
            Jones::from([
                Complex::new(1.9974082e0, -1.0167356e-1),
                Complex::new(0e0, 0e0),
                Complex::new(0e0, 0e0),
                Complex::new(1.9974082e0, -1.0167356e-1),
            ]),
        ],
        [
            Jones::from([
                Complex::new(5.9569197e0, 7.1689516e-1),
                Complex::new(0e0, 0e0),
                Complex::new(0e0, 0e0),
                Complex::new(5.9569197e0, 7.1689516e-1),
            ]),
            Jones::from([
                Complex::new(5.989021e0, 3.6238956e-1),
                Complex::new(0e0, 0e0),
                Complex::new(0e0, 0e0),
                Complex::new(5.989021e0, 3.6238956e-1),
            ]),
            Jones::from([
                Complex::new(5.9894176e0, -3.5580167e-1),
                Complex::new(0e0, 0e0),
                Complex::new(0e0, 0e0),
                Complex::new(5.9894176e0, -3.5580167e-1),
            ]),
        ],
        [
            Jones::from([
                Complex::new(3.9625018e0, 5.4580307e-1),
                Complex::new(0e0, 0e0),
                Complex::new(0e0, 0e0),
                Complex::new(3.9625018e0, 5.4580307e-1),
            ]),
            Jones::from([
                Complex::new(3.9904408e0, 2.760545e-1),
                Complex::new(0e0, 0e0),
                Complex::new(0e0, 0e0),
                Complex::new(3.9904408e0, 2.760545e-1),
            ]),
            Jones::from([
                Complex::new(3.9907863e0, -2.7103797e-1),
                Complex::new(0e0, 0e0),
                Complex::new(0e0, 0e0),
                Complex::new(3.9907863e0, -2.7103797e-1),
            ]),
        ]
    ];
    assert_abs_diff_eq!(expected, visibilities.view());

    let shapelet_uvws = modeller
        .components
        .shapelets
        .get_shapelet_uvws(0.0, &obs.xyzs);
    let expected = array![
        [
            UVW {
                u: 1.9996953903127825,
                v: 0.01584645344024005,
                w: 0.031100415996813357
            },
            UVW {
                u: 1.9996314242432884,
                v: 0.017430912937512553,
                w: 0.034210092851707785
            }
        ],
        [
            UVW {
                u: 1.0173001015936747,
                v: -0.44599812806736405,
                w: -0.8753206115806403
            },
            UVW {
                u: 1.019013154521334,
                v: -0.4451913783247998,
                w: -0.8737372760605702
            }
        ],
        [
            UVW {
                u: -0.9823952887191078,
                v: -0.4618445815076041,
                w: -0.9064210275774537
            },
            UVW {
                u: -0.9806182697219545,
                v: -0.46262229126231236,
                w: -0.9079473689122779
            }
        ]
    ];
    assert_abs_diff_eq!(expected, shapelet_uvws.view());
}
