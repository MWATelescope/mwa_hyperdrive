// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Tests on generating sky-model visibilities on a CPU.

use marlu::RADec;
use ndarray::prelude::*;
use vec1::vec1;

use super::*;
use crate::srclist::{Source, SourceList};

macro_rules! test_modelling {
    ($no_beam:expr, $model_fn:expr,
        $list_srclist:expr, $power_law_srclist:expr, $curved_power_law_srclist:expr,
        $list_test_fn:expr, $power_law_test_fn:expr, $curved_power_law_test_fn:expr) => {{
        let obs = ObsParams::new($no_beam);
        let modeller = obs.get_cpu_modeller($list_srclist);
        let mut visibilities = Array2::zeros((obs.freqs.len(), obs.uvws.len()));
        $model_fn(
            &modeller,
            visibilities.view_mut(),
            &obs.uvws,
            obs.lst,
            obs.array_latitude_rad,
        )
        .unwrap();
        let epsilon = if $no_beam { 0.0 } else { 1e-15 };
        $list_test_fn(visibilities.view(), epsilon);

        let modeller = obs.get_cpu_modeller($power_law_srclist);
        visibilities.fill(Jones::default());
        $model_fn(
            &modeller,
            visibilities.view_mut(),
            &obs.uvws,
            obs.lst,
            obs.array_latitude_rad,
        )
        .unwrap();
        $power_law_test_fn(visibilities.view(), epsilon);

        let modeller = obs.get_cpu_modeller($curved_power_law_srclist);
        visibilities.fill(Jones::default());
        $model_fn(
            &modeller,
            visibilities.view_mut(),
            &obs.uvws,
            obs.lst,
            obs.array_latitude_rad,
        )
        .unwrap();
        $curved_power_law_test_fn(visibilities.view(), epsilon);
    }};
}

#[test]
fn point_zenith_cpu() {
    test_modelling!(
        true,
        SkyModellerCpu::model_points,
        &POINT_ZENITH_LIST,
        &POINT_ZENITH_POWER_LAW,
        &POINT_ZENITH_CURVED_POWER_LAW,
        test_list_zenith_visibilities,
        test_power_law_zenith_visibilities,
        test_curved_power_law_zenith_visibilities
    );
}

#[test]
fn point_off_zenith_gpu() {
    test_modelling!(
        true,
        SkyModellerCpu::model_points,
        &POINT_OFF_ZENITH_LIST,
        &POINT_OFF_ZENITH_POWER_LAW,
        &POINT_OFF_ZENITH_CURVED_POWER_LAW,
        test_list_off_zenith_visibilities,
        test_power_law_off_zenith_visibilities,
        test_curved_power_law_off_zenith_visibilities
    );
}

#[test]
fn gaussian_zenith_gpu() {
    test_modelling!(
        true,
        SkyModellerCpu::model_gaussians,
        &GAUSSIAN_ZENITH_LIST,
        &GAUSSIAN_ZENITH_POWER_LAW,
        &GAUSSIAN_ZENITH_CURVED_POWER_LAW,
        test_list_zenith_visibilities,
        test_power_law_zenith_visibilities,
        test_curved_power_law_zenith_visibilities
    );
}

#[test]
fn gaussian_off_zenith_gpu() {
    test_modelling!(
        true,
        SkyModellerCpu::model_gaussians,
        &GAUSSIAN_OFF_ZENITH_LIST,
        &GAUSSIAN_OFF_ZENITH_POWER_LAW,
        &GAUSSIAN_OFF_ZENITH_CURVED_POWER_LAW,
        test_list_off_zenith_visibilities,
        test_power_law_off_zenith_visibilities,
        test_curved_power_law_off_zenith_visibilities
    );
}

#[test]
fn point_zenith_gpu_fee() {
    test_modelling!(
        false,
        SkyModellerCpu::model_points,
        &POINT_ZENITH_LIST,
        &POINT_ZENITH_POWER_LAW,
        &POINT_ZENITH_CURVED_POWER_LAW,
        test_list_zenith_visibilities_fee,
        test_power_law_zenith_visibilities_fee,
        test_curved_power_law_zenith_visibilities_fee
    );
}

#[test]
fn point_off_zenith_gpu_fee() {
    test_modelling!(
        false,
        SkyModellerCpu::model_points,
        &POINT_OFF_ZENITH_LIST,
        &POINT_OFF_ZENITH_POWER_LAW,
        &POINT_OFF_ZENITH_CURVED_POWER_LAW,
        test_list_off_zenith_visibilities_fee,
        test_power_law_off_zenith_visibilities_fee,
        test_curved_power_law_off_zenith_visibilities_fee
    );
}

#[test]
fn gaussian_zenith_gpu_fee() {
    test_modelling!(
        false,
        SkyModellerCpu::model_gaussians,
        &GAUSSIAN_ZENITH_LIST,
        &GAUSSIAN_ZENITH_POWER_LAW,
        &GAUSSIAN_ZENITH_CURVED_POWER_LAW,
        test_list_zenith_visibilities_fee,
        test_power_law_zenith_visibilities_fee,
        test_curved_power_law_zenith_visibilities_fee
    );
}

#[test]
fn gaussian_off_zenith_gpu_fee() {
    test_modelling!(
        false,
        SkyModellerCpu::model_gaussians,
        &GAUSSIAN_OFF_ZENITH_LIST,
        &GAUSSIAN_OFF_ZENITH_POWER_LAW,
        &GAUSSIAN_OFF_ZENITH_CURVED_POWER_LAW,
        test_list_off_zenith_visibilities_fee,
        test_power_law_off_zenith_visibilities_fee,
        test_curved_power_law_off_zenith_visibilities_fee
    );
}

// Ah shapelets, my favourite.
macro_rules! test_modelling_shapelets {
    ($no_beam:expr,
        $list_srclist:expr, $power_law_srclist:expr, $curved_power_law_srclist:expr,
        $list_test_fn:expr, $power_law_test_fn:expr, $curved_power_law_test_fn:expr) => {{
        let obs = ObsParams::new($no_beam);
        let modeller = obs.get_cpu_modeller($list_srclist);
        let shapelet_uvws = modeller
            .components
            .shapelets
            .get_shapelet_uvws(obs.lst, &obs.xyzs);
        let mut visibilities = Array2::zeros((obs.freqs.len(), obs.uvws.len()));
        modeller
            .model_shapelets(
                visibilities.view_mut(),
                &obs.uvws,
                shapelet_uvws.view(),
                obs.lst,
                obs.array_latitude_rad,
            )
            .unwrap();
        let epsilon = if $no_beam { 0.0 } else { 1e-15 };
        $list_test_fn(visibilities.view(), epsilon);

        let modeller = obs.get_cpu_modeller($power_law_srclist);
        visibilities.fill(Jones::default());
        modeller
            .model_shapelets(
                visibilities.view_mut(),
                &obs.uvws,
                shapelet_uvws.view(),
                obs.lst,
                obs.array_latitude_rad,
            )
            .unwrap();
        $power_law_test_fn(visibilities.view(), epsilon);

        let modeller = obs.get_cpu_modeller($curved_power_law_srclist);
        visibilities.fill(Jones::default());
        modeller
            .model_shapelets(
                visibilities.view_mut(),
                &obs.uvws,
                shapelet_uvws.view(),
                obs.lst,
                obs.array_latitude_rad,
            )
            .unwrap();
        $curved_power_law_test_fn(visibilities.view(), epsilon);
    }};
}

#[test]
fn shapelet_zenith_gpu() {
    test_modelling_shapelets!(
        true,
        &SHAPELET_ZENITH_LIST,
        &SHAPELET_ZENITH_POWER_LAW,
        &SHAPELET_ZENITH_CURVED_POWER_LAW,
        test_list_zenith_visibilities,
        test_power_law_zenith_visibilities,
        test_curved_power_law_zenith_visibilities
    );
}

#[test]
fn shapelet_off_zenith_gpu() {
    test_modelling_shapelets!(
        true,
        &SHAPELET_OFF_ZENITH_LIST,
        &SHAPELET_OFF_ZENITH_POWER_LAW,
        &SHAPELET_OFF_ZENITH_CURVED_POWER_LAW,
        test_list_off_zenith_visibilities,
        test_power_law_off_zenith_visibilities,
        test_curved_power_law_off_zenith_visibilities
    );
}

#[test]
fn shapelet_zenith_gpu_fee() {
    test_modelling_shapelets!(
        false,
        &SHAPELET_ZENITH_LIST,
        &SHAPELET_ZENITH_POWER_LAW,
        &SHAPELET_ZENITH_CURVED_POWER_LAW,
        test_list_zenith_visibilities_fee,
        test_power_law_zenith_visibilities_fee,
        test_curved_power_law_zenith_visibilities_fee
    );
}

#[test]
fn shapelet_off_zenith_gpu_fee() {
    test_modelling_shapelets!(
        false,
        &SHAPELET_OFF_ZENITH_LIST,
        &SHAPELET_OFF_ZENITH_POWER_LAW,
        &SHAPELET_OFF_ZENITH_CURVED_POWER_LAW,
        test_list_off_zenith_visibilities_fee,
        test_power_law_off_zenith_visibilities_fee,
        test_curved_power_law_off_zenith_visibilities_fee
    );
}

#[test]
fn non_trivial_gaussian() {
    test_modelling!(
        false,
        SkyModellerCpu::model_gaussians,
        &SourceList::from([(
            "list".to_string(),
            Source {
                components: vec1![get_gaussian2(*OFF_PHASE_CENTRE, FluxType::List)]
            }
        )]),
        &SourceList::from([(
            "power_law".to_string(),
            Source {
                components: vec1![get_gaussian2(*OFF_PHASE_CENTRE, FluxType::PowerLaw)]
            }
        )]),
        &SourceList::from([(
            "curved_power_law".to_string(),
            Source {
                components: vec1![get_gaussian2(*OFF_PHASE_CENTRE, FluxType::CurvedPowerLaw)]
            }
        )]),
        test_non_trivial_gaussian_list,
        test_non_trivial_gaussian_power_law,
        test_non_trivial_gaussian_curved_power_law
    );
}

#[test]
fn gaussian_multiple_components() {
    let obs = ObsParams::new(true);
    let mut srclist = SourceList::new();
    srclist.insert(
        "gaussians".to_string(),
        Source {
            components: vec1![
                get_gaussian(RADec::from_degrees(1.0, -27.0), FluxType::List),
                get_gaussian(RADec::from_degrees(1.1, -27.0), FluxType::List)
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
                get_shapelet(RADec::from_degrees(1.0, -27.0), FluxType::List),
                get_shapelet(RADec::from_degrees(1.1, -27.0), FluxType::List)
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
