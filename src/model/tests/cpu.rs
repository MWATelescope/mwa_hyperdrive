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
    let mut srclist = SourceList::new();
    srclist.insert(
        "zenith".to_string(),
        Source {
            components: vec1![get_simple_point(obs.phase_centre, FluxType::List)],
        },
    );
    let modeller = obs.get_cpu_modeller(&srclist);
    let mut visibilities = Array2::zeros((obs.uvws.len(), obs.freqs.len()));
    let result = modeller.model_points_inner(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_list_zenith_visibilities(visibilities.view(), "Failed on flux-type list");

    srclist.insert(
        "zenith".to_string(),
        Source {
            components: vec1![get_simple_point(obs.phase_centre, FluxType::PowerLaw)],
        },
    );
    let modeller = obs.get_cpu_modeller(&srclist);
    visibilities.fill(Jones::default());
    let result = modeller.model_points_inner(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_power_law_zenith_visibilities(visibilities.view(), "Failed on flux-type power law");

    srclist.insert(
        "zenith".to_string(),
        Source {
            components: vec1![get_simple_point(obs.phase_centre, FluxType::CurvedPowerLaw)],
        },
    );
    let modeller = obs.get_cpu_modeller(&srclist);
    visibilities.fill(Jones::default());
    let result = modeller.model_points_inner(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_curved_power_law_zenith_visibilities(
        visibilities.view(),
        "Failed on flux-type curved power law",
    );
}

// Put a single point source just off zenith.
#[test]
fn point_off_zenith_cpu() {
    let obs = ObsParams::new(true);
    let mut srclist = SourceList::new();
    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec1![get_simple_point(
                RADec::new_degrees(1.0, -27.0),
                FluxType::List
            )],
        },
    );
    let modeller = obs.get_cpu_modeller(&srclist);
    let mut visibilities = Array2::zeros((obs.uvws.len(), obs.freqs.len()));
    let result = modeller.model_points_inner(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_list_off_zenith_visibilities(visibilities.view(), "Failed on flux-type list");

    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec1![get_simple_point(
                RADec::new_degrees(1.0, -27.0),
                FluxType::PowerLaw
            )],
        },
    );
    let modeller = obs.get_cpu_modeller(&srclist);
    visibilities.fill(Jones::default());
    let result = modeller.model_points_inner(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_power_law_off_zenith_visibilities(visibilities.view(), "Failed on flux-type power law");

    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec1![get_simple_point(
                RADec::new_degrees(1.0, -27.0),
                FluxType::CurvedPowerLaw
            )],
        },
    );
    let modeller = obs.get_cpu_modeller(&srclist);
    visibilities.fill(Jones::default());
    let result = modeller.model_points_inner(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_curved_power_law_off_zenith_visibilities(
        visibilities.view(),
        "Failed on flux-type curved power law",
    );
}

// Put a single Gaussian source at zenith.
#[test]
fn gaussian_zenith_cpu() {
    let obs = ObsParams::new(true);
    let mut srclist = SourceList::new();
    srclist.insert(
        "zenith".to_string(),
        Source {
            components: vec1![get_simple_gaussian(obs.phase_centre, FluxType::List)],
        },
    );
    let modeller = obs.get_cpu_modeller(&srclist);
    let mut visibilities = Array2::zeros((obs.uvws.len(), obs.freqs.len()));
    let result = modeller.model_gaussians_inner(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_list_zenith_visibilities(visibilities.view(), "Failed on flux-type list");

    srclist.insert(
        "zenith".to_string(),
        Source {
            components: vec1![get_simple_gaussian(obs.phase_centre, FluxType::PowerLaw)],
        },
    );
    let modeller = obs.get_cpu_modeller(&srclist);
    visibilities.fill(Jones::default());
    let result = modeller.model_gaussians_inner(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_power_law_zenith_visibilities(visibilities.view(), "Failed on flux-type power law");

    srclist.insert(
        "zenith".to_string(),
        Source {
            components: vec1![get_simple_gaussian(
                obs.phase_centre,
                FluxType::CurvedPowerLaw
            )],
        },
    );
    let modeller = obs.get_cpu_modeller(&srclist);
    visibilities.fill(Jones::default());
    let result = modeller.model_gaussians_inner(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_curved_power_law_zenith_visibilities(
        visibilities.view(),
        "Failed on flux-type curved_power law",
    );
}

// Put a single Gaussian source just off zenith.
#[test]
fn gaussian_off_zenith_cpu() {
    let obs = ObsParams::new(true);
    let mut srclist = SourceList::new();
    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec1![get_simple_gaussian(
                RADec::new_degrees(1.0, -27.0),
                FluxType::List
            )],
        },
    );
    let modeller = obs.get_cpu_modeller(&srclist);
    let mut visibilities = Array2::zeros((obs.uvws.len(), obs.freqs.len()));
    let result = modeller.model_gaussians_inner(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_list_off_zenith_visibilities(visibilities.view(), "Failed on flux-type list");

    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec1![get_simple_gaussian(
                RADec::new_degrees(1.0, -27.0),
                FluxType::PowerLaw
            )],
        },
    );
    let modeller = obs.get_cpu_modeller(&srclist);
    visibilities.fill(Jones::default());
    let result = modeller.model_gaussians_inner(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_power_law_off_zenith_visibilities(visibilities.view(), "Failed on flux-type power law");

    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec1![get_simple_gaussian(
                RADec::new_degrees(1.0, -27.0),
                FluxType::CurvedPowerLaw
            )],
        },
    );
    let modeller = obs.get_cpu_modeller(&srclist);
    visibilities.fill(Jones::default());
    let result = modeller.model_gaussians_inner(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_curved_power_law_off_zenith_visibilities(
        visibilities.view(),
        "Failed on flux-type curved power law",
    );
}

// Put a single shapelet source at zenith.
#[test]
fn shapelet_zenith_cpu() {
    let obs = ObsParams::new(true);
    let mut srclist = SourceList::new();
    srclist.insert(
        "zenith".to_string(),
        Source {
            components: vec1![get_simple_shapelet(obs.phase_centre, FluxType::List)],
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
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_list_zenith_visibilities(visibilities.view(), "Failed on flux-type list");

    srclist.insert(
        "zenith".to_string(),
        Source {
            components: vec1![get_simple_shapelet(obs.phase_centre, FluxType::PowerLaw)],
        },
    );
    let modeller = obs.get_cpu_modeller(&srclist);
    let shapelet_uvws = modeller
        .components
        .shapelets
        .get_shapelet_uvws(obs.lst, &obs.xyzs);
    visibilities.fill(Jones::default());
    let result = modeller.model_shapelets_inner(
        visibilities.view_mut(),
        &obs.uvws,
        shapelet_uvws.view(),
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_power_law_zenith_visibilities(visibilities.view(), "Failed on flux-type power law");

    srclist.insert(
        "zenith".to_string(),
        Source {
            components: vec1![get_simple_shapelet(
                obs.phase_centre,
                FluxType::CurvedPowerLaw
            )],
        },
    );
    let modeller = obs.get_cpu_modeller(&srclist);
    let shapelet_uvws = modeller
        .components
        .shapelets
        .get_shapelet_uvws(obs.lst, &obs.xyzs);
    visibilities.fill(Jones::default());
    let result = modeller.model_shapelets_inner(
        visibilities.view_mut(),
        &obs.uvws,
        shapelet_uvws.view(),
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_curved_power_law_zenith_visibilities(
        visibilities.view(),
        "Failed on flux-type curved power law",
    );
}

// Put a single shapelet source just off zenith.
#[test]
fn shapelet_off_zenith_cpu() {
    let obs = ObsParams::new(true);
    let mut srclist = SourceList::new();
    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec1![get_simple_shapelet(
                RADec::new_degrees(1.0, -27.0),
                FluxType::List
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
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_list_off_zenith_visibilities(visibilities.view(), "Failed on flux-type list");

    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec1![get_simple_shapelet(
                RADec::new_degrees(1.0, -27.0),
                FluxType::PowerLaw
            )],
        },
    );
    let modeller = obs.get_cpu_modeller(&srclist);
    let shapelet_uvws = modeller
        .components
        .shapelets
        .get_shapelet_uvws(obs.lst, &obs.xyzs);
    visibilities.fill(Jones::default());
    let result = modeller.model_shapelets_inner(
        visibilities.view_mut(),
        &obs.uvws,
        shapelet_uvws.view(),
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_power_law_off_zenith_visibilities(visibilities.view(), "Failed on flux-type power law");

    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec1![get_simple_shapelet(
                RADec::new_degrees(1.0, -27.0),
                FluxType::CurvedPowerLaw
            )],
        },
    );
    let modeller = obs.get_cpu_modeller(&srclist);
    let shapelet_uvws = modeller
        .components
        .shapelets
        .get_shapelet_uvws(obs.lst, &obs.xyzs);
    visibilities.fill(Jones::default());
    let result = modeller.model_shapelets_inner(
        visibilities.view_mut(),
        &obs.uvws,
        shapelet_uvws.view(),
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_curved_power_law_off_zenith_visibilities(
        visibilities.view(),
        "Failed on flux-type curved power law",
    );
}

// Put a single point source at zenith (FEE beam).
#[test]
#[serial]
fn point_zenith_cpu_fee() {
    let obs = ObsParams::new(false);
    let mut srclist = SourceList::new();
    srclist.insert(
        "zenith".to_string(),
        Source {
            components: vec1![get_simple_point(obs.phase_centre, FluxType::List)],
        },
    );
    let modeller = obs.get_cpu_modeller(&srclist);
    let mut visibilities = Array2::zeros((obs.uvws.len(), obs.freqs.len()));
    let result = modeller.model_points_inner(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_list_zenith_visibilities_fee(visibilities.view(), "Failed on flux-type list");

    srclist.insert(
        "zenith".to_string(),
        Source {
            components: vec1![get_simple_point(obs.phase_centre, FluxType::PowerLaw)],
        },
    );
    let modeller = obs.get_cpu_modeller(&srclist);
    visibilities.fill(Jones::default());
    let result = modeller.model_points_inner(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_power_law_zenith_visibilities_fee(visibilities.view(), "Failed on flux-type power law");

    srclist.insert(
        "zenith".to_string(),
        Source {
            components: vec1![get_simple_point(obs.phase_centre, FluxType::CurvedPowerLaw)],
        },
    );
    let modeller = obs.get_cpu_modeller(&srclist);
    visibilities.fill(Jones::default());
    let result = modeller.model_points_inner(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_curved_power_law_zenith_visibilities_fee(
        visibilities.view(),
        "Failed on flux-type curved power law",
    );
}

// Put a single point source just off zenith (FEE beam).
#[test]
#[serial]
fn point_off_zenith_cpu_fee() {
    let obs = ObsParams::new(false);
    let mut srclist = SourceList::new();
    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec1![get_simple_point(
                RADec::new_degrees(1.0, -27.0),
                FluxType::List
            )],
        },
    );
    let modeller = obs.get_cpu_modeller(&srclist);
    let mut visibilities = Array2::zeros((obs.uvws.len(), obs.freqs.len()));
    let result = modeller.model_points_inner(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_list_off_zenith_visibilities_fee(visibilities.view(), "Failed on flux-type list");

    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec1![get_simple_point(
                RADec::new_degrees(1.0, -27.0),
                FluxType::PowerLaw
            )],
        },
    );
    let modeller = obs.get_cpu_modeller(&srclist);
    visibilities.fill(Jones::default());
    let result = modeller.model_points_inner(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_power_law_off_zenith_visibilities_fee(
        visibilities.view(),
        "Failed on flux-type power law",
    );

    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec1![get_simple_point(
                RADec::new_degrees(1.0, -27.0),
                FluxType::CurvedPowerLaw
            )],
        },
    );
    let modeller = obs.get_cpu_modeller(&srclist);
    visibilities.fill(Jones::default());
    let result = modeller.model_points_inner(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_curved_power_law_off_zenith_visibilities_fee(
        visibilities.view(),
        "Failed on flux-type curved power law",
    );
}

// Put a single Gaussian source at zenith (FEE beam).
#[test]
#[serial]
fn gaussian_zenith_cpu_fee() {
    let obs = ObsParams::new(false);
    let mut srclist = SourceList::new();
    srclist.insert(
        "zenith".to_string(),
        Source {
            components: vec1![get_simple_gaussian(obs.phase_centre, FluxType::List)],
        },
    );
    let modeller = obs.get_cpu_modeller(&srclist);
    let mut visibilities = Array2::zeros((obs.uvws.len(), obs.freqs.len()));
    let result = modeller.model_gaussians_inner(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_list_zenith_visibilities_fee(visibilities.view(), "Failed on flux-type list");

    srclist.insert(
        "zenith".to_string(),
        Source {
            components: vec1![get_simple_gaussian(obs.phase_centre, FluxType::PowerLaw)],
        },
    );
    let modeller = obs.get_cpu_modeller(&srclist);
    visibilities.fill(Jones::default());
    let result = modeller.model_gaussians_inner(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_power_law_zenith_visibilities_fee(visibilities.view(), "Failed on flux-type power law");

    srclist.insert(
        "zenith".to_string(),
        Source {
            components: vec1![get_simple_gaussian(
                obs.phase_centre,
                FluxType::CurvedPowerLaw
            )],
        },
    );
    let modeller = obs.get_cpu_modeller(&srclist);
    visibilities.fill(Jones::default());
    let result = modeller.model_gaussians_inner(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_curved_power_law_zenith_visibilities_fee(
        visibilities.view(),
        "Failed on flux-type curved power law",
    );
}

// Put a single Gaussian source just off zenith (FEE beam).
#[test]
#[serial]
fn gaussian_off_zenith_cpu_fee() {
    let obs = ObsParams::new(false);
    let mut srclist = SourceList::new();
    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec1![get_simple_gaussian(
                RADec::new_degrees(1.0, -27.0),
                FluxType::List
            )],
        },
    );
    let modeller = obs.get_cpu_modeller(&srclist);
    let mut visibilities = Array2::zeros((obs.uvws.len(), obs.freqs.len()));
    let result = modeller.model_gaussians_inner(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_list_off_zenith_visibilities_fee(visibilities.view(), "Failed on flux-type list");

    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec1![get_simple_gaussian(
                RADec::new_degrees(1.0, -27.0),
                FluxType::PowerLaw
            )],
        },
    );
    let modeller = obs.get_cpu_modeller(&srclist);
    visibilities.fill(Jones::default());
    let result = modeller.model_gaussians_inner(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_power_law_off_zenith_visibilities_fee(
        visibilities.view(),
        "Failed on flux-type power law",
    );

    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec1![get_simple_gaussian(
                RADec::new_degrees(1.0, -27.0),
                FluxType::CurvedPowerLaw
            )],
        },
    );
    let modeller = obs.get_cpu_modeller(&srclist);
    visibilities.fill(Jones::default());
    let result = modeller.model_gaussians_inner(
        visibilities.view_mut(),
        &obs.uvws,
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_curved_power_law_off_zenith_visibilities_fee(
        visibilities.view(),
        "Failed on flux-type curved power law",
    );
}

// Put a single shapelet source at zenith (FEE beam).
#[test]
#[serial]
fn shapelet_zenith_cpu_fee() {
    let obs = ObsParams::new(false);
    let mut srclist = SourceList::new();
    srclist.insert(
        "zenith".to_string(),
        Source {
            components: vec1![get_simple_shapelet(obs.phase_centre, FluxType::List)],
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
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_list_zenith_visibilities_fee(visibilities.view(), "Failed on flux-type list");

    srclist.insert(
        "zenith".to_string(),
        Source {
            components: vec1![get_simple_shapelet(obs.phase_centre, FluxType::PowerLaw)],
        },
    );
    let modeller = obs.get_cpu_modeller(&srclist);
    let shapelet_uvws = modeller
        .components
        .shapelets
        .get_shapelet_uvws(obs.lst, &obs.xyzs);
    visibilities.fill(Jones::default());
    let result = modeller.model_shapelets_inner(
        visibilities.view_mut(),
        &obs.uvws,
        shapelet_uvws.view(),
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_power_law_zenith_visibilities_fee(visibilities.view(), "Failed on flux-type power law");

    srclist.insert(
        "zenith".to_string(),
        Source {
            components: vec1![get_simple_shapelet(
                obs.phase_centre,
                FluxType::CurvedPowerLaw
            )],
        },
    );
    let modeller = obs.get_cpu_modeller(&srclist);
    let shapelet_uvws = modeller
        .components
        .shapelets
        .get_shapelet_uvws(obs.lst, &obs.xyzs);
    visibilities.fill(Jones::default());
    let result = modeller.model_shapelets_inner(
        visibilities.view_mut(),
        &obs.uvws,
        shapelet_uvws.view(),
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_curved_power_law_zenith_visibilities_fee(
        visibilities.view(),
        "Failed on flux-type curved power law",
    );
}

// Put a single shapelet source just off zenith (FEE beam).
#[test]
#[serial]
fn shapelet_off_zenith_cpu_fee() {
    let obs = ObsParams::new(false);
    let mut srclist = SourceList::new();
    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec1![get_simple_shapelet(
                RADec::new_degrees(1.0, -27.0),
                FluxType::List
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
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_list_off_zenith_visibilities_fee(visibilities.view(), "Failed on flux-type list");

    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec1![get_simple_shapelet(
                RADec::new_degrees(1.0, -27.0),
                FluxType::PowerLaw
            )],
        },
    );
    let modeller = obs.get_cpu_modeller(&srclist);
    let shapelet_uvws = modeller
        .components
        .shapelets
        .get_shapelet_uvws(obs.lst, &obs.xyzs);
    visibilities.fill(Jones::default());
    let result = modeller.model_shapelets_inner(
        visibilities.view_mut(),
        &obs.uvws,
        shapelet_uvws.view(),
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_power_law_off_zenith_visibilities_fee(
        visibilities.view(),
        "Failed on flux-type power law",
    );

    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec1![get_simple_shapelet(
                RADec::new_degrees(1.0, -27.0),
                FluxType::CurvedPowerLaw
            )],
        },
    );
    let modeller = obs.get_cpu_modeller(&srclist);
    let shapelet_uvws = modeller
        .components
        .shapelets
        .get_shapelet_uvws(obs.lst, &obs.xyzs);
    visibilities.fill(Jones::default());
    let result = modeller.model_shapelets_inner(
        visibilities.view_mut(),
        &obs.uvws,
        shapelet_uvws.view(),
        obs.lst,
        obs.array_latitude_rad,
    );
    assert!(result.is_ok());
    assert_curved_power_law_off_zenith_visibilities_fee(
        visibilities.view(),
        "Failed on flux-type curved power law",
    );
}
