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
    test_list_zenith_visibilities(visibilities.view());

    let modeller = obs.get_cpu_modeller(&POINT_ZENITH_POWER_LAW);
    visibilities.fill(Jones::default());
    let result = modeller.model_points(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    test_power_law_zenith_visibilities(visibilities.view());

    let modeller = obs.get_cpu_modeller(&POINT_ZENITH_CURVED_POWER_LAW);
    visibilities.fill(Jones::default());
    let result = modeller.model_points(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    test_curved_power_law_zenith_visibilities(visibilities.view());
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
    test_list_off_zenith_visibilities(visibilities.view());

    let modeller = obs.get_cpu_modeller(&POINT_OFF_ZENITH_POWER_LAW);
    visibilities.fill(Jones::default());
    let result = modeller.model_points(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    test_power_law_off_zenith_visibilities(visibilities.view());

    let modeller = obs.get_cpu_modeller(&POINT_OFF_ZENITH_CURVED_POWER_LAW);
    visibilities.fill(Jones::default());
    let result = modeller.model_points(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    test_curved_power_law_off_zenith_visibilities(visibilities.view());
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
    test_list_zenith_visibilities(visibilities.view());

    let modeller = obs.get_cpu_modeller(&GAUSSIAN_ZENITH_POWER_LAW);
    visibilities.fill(Jones::default());
    let result = modeller.model_gaussians(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    test_power_law_zenith_visibilities(visibilities.view());

    let modeller = obs.get_cpu_modeller(&GAUSSIAN_ZENITH_CURVED_POWER_LAW);
    visibilities.fill(Jones::default());
    let result = modeller.model_gaussians(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    test_curved_power_law_zenith_visibilities(visibilities.view());
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
    test_list_off_zenith_visibilities(visibilities.view());

    let modeller = obs.get_cpu_modeller(&GAUSSIAN_OFF_ZENITH_POWER_LAW);
    visibilities.fill(Jones::default());
    let result = modeller.model_gaussians(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    test_power_law_off_zenith_visibilities(visibilities.view());

    let modeller = obs.get_cpu_modeller(&GAUSSIAN_OFF_ZENITH_CURVED_POWER_LAW);
    visibilities.fill(Jones::default());
    let result = modeller.model_gaussians(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    test_curved_power_law_off_zenith_visibilities(visibilities.view());
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
    test_list_zenith_visibilities(visibilities.view());

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
    test_power_law_zenith_visibilities(visibilities.view());

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
    test_curved_power_law_zenith_visibilities(visibilities.view());
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
    test_list_off_zenith_visibilities(visibilities.view());

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
    test_power_law_off_zenith_visibilities(visibilities.view());

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
    test_curved_power_law_off_zenith_visibilities(visibilities.view());
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
    test_list_zenith_visibilities_fee(visibilities.view());

    let modeller = obs.get_cpu_modeller(&POINT_ZENITH_POWER_LAW);
    visibilities.fill(Jones::default());
    let result = modeller.model_points(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    test_power_law_zenith_visibilities_fee(visibilities.view());

    let modeller = obs.get_cpu_modeller(&POINT_ZENITH_CURVED_POWER_LAW);
    visibilities.fill(Jones::default());
    let result = modeller.model_points(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    test_curved_power_law_zenith_visibilities_fee(visibilities.view());
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
    test_list_off_zenith_visibilities_fee(visibilities.view());

    let modeller = obs.get_cpu_modeller(&POINT_OFF_ZENITH_POWER_LAW);
    visibilities.fill(Jones::default());
    let result = modeller.model_points(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    test_power_law_off_zenith_visibilities_fee(visibilities.view());

    let modeller = obs.get_cpu_modeller(&POINT_OFF_ZENITH_CURVED_POWER_LAW);
    visibilities.fill(Jones::default());
    let result = modeller.model_points(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    test_curved_power_law_off_zenith_visibilities_fee(visibilities.view());
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
    test_list_zenith_visibilities_fee(visibilities.view());

    let modeller = obs.get_cpu_modeller(&GAUSSIAN_ZENITH_POWER_LAW);
    visibilities.fill(Jones::default());
    let result = modeller.model_gaussians(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    test_power_law_zenith_visibilities_fee(visibilities.view());

    let modeller = obs.get_cpu_modeller(&GAUSSIAN_ZENITH_CURVED_POWER_LAW);
    visibilities.fill(Jones::default());
    let result = modeller.model_gaussians(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    test_curved_power_law_zenith_visibilities_fee(visibilities.view());
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
    test_list_off_zenith_visibilities_fee(visibilities.view());

    let modeller = obs.get_cpu_modeller(&GAUSSIAN_OFF_ZENITH_POWER_LAW);
    visibilities.fill(Jones::default());
    let result = modeller.model_gaussians(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    test_power_law_off_zenith_visibilities_fee(visibilities.view());

    let modeller = obs.get_cpu_modeller(&GAUSSIAN_OFF_ZENITH_CURVED_POWER_LAW);
    visibilities.fill(Jones::default());
    let result = modeller.model_gaussians(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    test_curved_power_law_off_zenith_visibilities_fee(visibilities.view());
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
    test_list_zenith_visibilities_fee(visibilities.view());

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
    test_power_law_zenith_visibilities_fee(visibilities.view());

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
    test_curved_power_law_zenith_visibilities_fee(visibilities.view());
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
    test_list_off_zenith_visibilities_fee(visibilities.view());

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
    test_power_law_off_zenith_visibilities_fee(visibilities.view());

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
    test_curved_power_law_off_zenith_visibilities_fee(visibilities.view());
}

#[test]
fn gaussian_multiple_components() {
    let obs = ObsParams::new(true);
    let mut srclist = SourceList::new();
    srclist.insert(
        "gaussians".to_string(),
        Source {
            components: vec1![
                get_simple_gaussian(RADec::from_degrees(1.0, -27.0), FluxType::List),
                get_simple_gaussian(RADec::from_degrees(1.1, -27.0), FluxType::List)
            ],
        },
    );
    let modeller = obs.get_cpu_modeller(&srclist);
    let mut visibilities = Array2::zeros((obs.freqs.len(), obs.uvws.len()));
    let result = modeller.model_gaussians(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    test_multiple_gaussian_components(visibilities.view(), 0.0);
}

#[test]
fn shapelet_multiple_components() {
    let obs = ObsParams::new(true);
    let mut srclist = SourceList::new();
    srclist.insert(
        "shapelets".to_string(),
        Source {
            components: vec1![
                get_simple_shapelet(RADec::from_degrees(1.0, -27.0), FluxType::List),
                get_simple_shapelet(RADec::from_degrees(1.1, -27.0), FluxType::List)
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

    let mut shapelet_uvws = modeller
        .components
        .shapelets
        .get_shapelet_uvws(0.0, &obs.xyzs);
    // Set the w terms to 0, because they 0 on the CUDA side, and this way the
    // CPU and CUDA tests can use the same test values.
    shapelet_uvws.iter_mut().for_each(|uvw| uvw.w = 0.0);

    test_multiple_shapelet_components(visibilities.view(), shapelet_uvws.view(), 0.0, 0.0);
}
