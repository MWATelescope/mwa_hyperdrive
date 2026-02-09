// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Tests on generating sky-model visibilities on a CPU.

use marlu::RADec;
use ndarray::prelude::*;

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
                components: vec![get_gaussian2(*OFF_PHASE_CENTRE, FluxType::List)]
                    .into_boxed_slice()
            }
        )]),
        &SourceList::from([(
            "power_law".to_string(),
            Source {
                components: vec![get_gaussian2(*OFF_PHASE_CENTRE, FluxType::PowerLaw)]
                    .into_boxed_slice()
            }
        )]),
        &SourceList::from([(
            "curved_power_law".to_string(),
            Source {
                components: vec![get_gaussian2(*OFF_PHASE_CENTRE, FluxType::CurvedPowerLaw)]
                    .into_boxed_slice()
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
            components: vec![
                get_gaussian(RADec::from_degrees(1.0, -27.0), FluxType::List),
                get_gaussian(RADec::from_degrees(1.1, -27.0), FluxType::List),
            ]
            .into_boxed_slice(),
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
fn precession_off_paths_and_autos() {
    use crate::context::Polarisations;
    use hifitime::Epoch;

    // Build a modeller with precession disabled
    let obs = ObsParams::new(true);
    let modeller = SkyModellerCpu::new(
        &*obs.beam,
        &POINT_ZENITH_LIST,
        Polarisations::default(),
        &obs.xyzs,
        &obs.freqs,
        &obs.flagged_tiles,
        obs.phase_centre,
        obs.array_longitude_rad,
        obs.array_latitude_rad,
        hifitime::Duration::default(),
        false, // apply_precession off
    );

    // model_timestep should succeed and return UVWs
    let (vis, uvws) = modeller
        .model_timestep(Epoch::from_gpst_seconds(1090008640.0))
        .expect("model timestep");
    assert_eq!(vis.dim().0, obs.freqs.len());
    assert_eq!(uvws.len(), obs.uvws.len());
    // With NoBeam and zenith point source, vis should exactly match list FD
    test_list_zenith_visibilities(vis.view(), 0.0);

    // model_timestep_autos_with should also work
    let mut autos = ndarray::Array2::zeros((obs.freqs.len(), obs.xyzs.len()));
    modeller
        .model_timestep_autos_with(Epoch::from_gpst_seconds(1090008640.0), autos.view_mut())
        .expect("model autos");
    test_model_timestep_autos_with_point(autos.view(), 0.0);
}

#[test]
fn update_with_a_source_reconfigures_components() {
    use crate::srclist::{Source, SourceList};
    use marlu::RADec;

    let obs = ObsParams::new(true);
    let mut modeller = obs.get_cpu_modeller(&POINT_ZENITH_LIST);

    // Update with a different single-source list, keeping original phase centre
    let new_phase = RADec::from_degrees(1.0, -27.0);
    let mut sl = SourceList::new();
    sl.insert(
        "p".to_string(),
        Source {
            components: vec![get_point(new_phase, FluxType::List)].into_boxed_slice(),
        },
    );
    let src = sl.values().next().unwrap();
    modeller
        .update_with_a_source(src, obs.phase_centre)
        .expect("update with source");

    // After update, assert visibilities match first-principles expectations for an off-zenith point
    let mut vis = Array2::zeros((obs.freqs.len(), obs.uvws.len()));
    modeller
        .model_points(vis.view_mut(), &obs.uvws, obs.lst, obs.array_latitude_rad)
        .expect("model off-zenith via model_points");
    test_list_off_zenith_visibilities(vis.view(), 0.0);
}

#[test]
fn shapelet_multiple_components() {
    let obs = ObsParams::new(true);
    let mut srclist = SourceList::new();
    srclist.insert(
        "shapelets".to_string(),
        Source {
            components: vec![
                get_shapelet(RADec::from_degrees(1.0, -27.0), FluxType::List),
                get_shapelet(RADec::from_degrees(1.1, -27.0), FluxType::List),
            ]
            .into_boxed_slice(),
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
    // Set the w terms to 0, because they are 0 on the GPU side, and this way
    // the CPU and GPU tests can use the same test values.
    shapelet_uvws.iter_mut().for_each(|uvw| uvw.w = 0.0);

    test_multiple_shapelet_components(visibilities.view(), shapelet_uvws.view(), 0.0, 0.0);
}

#[test]
fn model_timestep_autos_with_point() {
    let obs = ObsParams::new(true);
    let mut srclist = SourceList::new();
    srclist.insert(
        "point".to_string(),
        Source {
            components: vec![get_point(RADec::from_degrees(1.0, -27.0), FluxType::List)]
                .into_boxed_slice(),
        },
    );
    let modeller = obs.get_cpu_modeller(&srclist);
    let mut visibilities = Array2::zeros((obs.freqs.len(), obs.xyzs.len()));
    let timestamp = Epoch::from_gpst_seconds(1090008640.);
    let result = modeller.model_timestep_autos_with(timestamp, visibilities.view_mut());
    assert!(result.is_ok());

    test_model_timestep_autos_with_point(visibilities.view(), 0.0);
}

#[test]
fn model_timestep_autos_with_gaussian() {
    let obs = ObsParams::new(true);
    let mut srclist = SourceList::new();
    srclist.insert(
        "gaussian".to_string(),
        Source {
            components: vec![get_gaussian(
                RADec::from_degrees(1.0, -27.0),
                FluxType::List,
            )]
            .into_boxed_slice(),
        },
    );
    let modeller = obs.get_cpu_modeller(&srclist);
    let mut visibilities = Array2::zeros((obs.freqs.len(), obs.xyzs.len()));
    let timestamp = Epoch::from_gpst_seconds(1090008640.);
    let result = modeller.model_timestep_autos_with(timestamp, visibilities.view_mut());
    assert!(result.is_ok());

    test_model_timestep_autos_with_gaussian(visibilities.view(), 0.0);
}

#[test]
fn model_timestep_autos_with_shapelet() {
    let obs = ObsParams::new(true);
    let mut srclist = SourceList::new();
    srclist.insert(
        "shapelet".to_string(),
        Source {
            components: vec![get_shapelet(
                RADec::from_degrees(1.0, -27.0),
                FluxType::List,
            )]
            .into_boxed_slice(),
        },
    );
    let modeller = obs.get_cpu_modeller(&srclist);
    let mut visibilities = Array2::zeros((obs.freqs.len(), obs.xyzs.len()));
    let timestamp = Epoch::from_gpst_seconds(1090008640.);
    let result = modeller.model_timestep_autos_with(timestamp, visibilities.view_mut());
    assert!(result.is_ok());

    test_model_timestep_autos_with_shapelet(visibilities.view(), 0.0);
}

#[test]
fn model_timestep_autos_sum_of_flux_densities_no_beam() {
    use crate::srclist::{ComponentType, FluxDensity, FluxDensityType, SourceComponent};

    // With NoBeam, each tile's autocorrelation equals the integral of sky flux density,
    // i.e. the sum of component flux densities per frequency, independent of position.
    let obs = ObsParams::new(true);

    // Build a source list with two point sources having known list fluxes per frequency
    let make_point_with_list = |pos: RADec, vals: [f64; 3]| -> SourceComponent {
        SourceComponent {
            radec: pos,
            comp_type: ComponentType::Point,
            flux_type: FluxDensityType::List(vec1::vec1![
                FluxDensity { freq: 150e6, i: vals[0], ..Default::default() },
                FluxDensity { freq: 175e6, i: vals[1], ..Default::default() },
                FluxDensity { freq: 200e6, i: vals[2], ..Default::default() },
            ]),
        }
    };

    let mut srclist = SourceList::new();
    srclist.insert(
        "s1".to_string(),
        Source { components: vec![make_point_with_list(RADec::from_degrees(1.0, -27.0), [1.0, 3.0, 2.0])].into_boxed_slice() },
    );
    srclist.insert(
        "s2".to_string(),
        Source { components: vec![make_point_with_list(RADec::from_degrees(1.1, -27.0), [0.5, 0.25, 1.5])].into_boxed_slice() },
    );

    let modeller = obs.get_cpu_modeller(&srclist);
    let timestamp = Epoch::from_gpst_seconds(1090008640.0);
    let mut autos = Array2::zeros((obs.freqs.len(), obs.xyzs.len()));
    modeller
        .model_timestep_autos_with(timestamp, autos.view_mut())
        .expect("model autos");

    // Expected per-frequency sums replicated across all tiles
    let expected_sums = [1.0 + 0.5, 3.0 + 0.25, 2.0 + 1.5];
    let mut expected = Array2::from_elem((obs.freqs.len(), obs.xyzs.len()), Jones::identity());
    for (fi, sum) in expected_sums.iter().enumerate() {
        let mut row = expected.slice_mut(s![fi, ..]);
        row.fill(Jones::identity() * *sum as f32);
    }

    approx::assert_abs_diff_eq!(autos, expected, epsilon = 0.0);
}

#[test]
fn get_beam_responses_empty_azels() {
    use crate::beam::NoBeam;
    use crate::context::Polarisations;
    use crate::model::cpu::SkyModellerCpu;
    use crate::srclist::SourceList;
    use std::collections::HashSet;

    // Minimal setup for SkyModellerCpu
    let beam = NoBeam { num_tiles: 1 };
    // must have one point source
    let source_list = SourceList::from([(
        "point".to_string(),
        Source {
            components: vec![get_point(RADec::from_degrees(0.0, -27.0), FluxType::List)]
                .into_boxed_slice(),
        },
    )]);

    let pols = Polarisations::default();
    // must have at least one tile
    let unflagged_tile_xyzs = vec![XyzGeodetic {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    }];
    // must have at least one frequency
    let unflagged_fine_chan_freqs = vec![170000000.0];
    let flagged_tiles = HashSet::new();
    let phase_centre = marlu::RADec::from_degrees(0.0, -27.0);
    let array_longitude_rad = 0.0;
    let array_latitude_rad = 0.0;
    let timestamp = Epoch::from_gpst_seconds(1090008640.);
    let dut1 = hifitime::Duration::from_seconds(0.0);
    let apply_precession = false;

    let modeller = SkyModellerCpu::new(
        &beam,
        &source_list,
        pols,
        &unflagged_tile_xyzs,
        &unflagged_fine_chan_freqs,
        &flagged_tiles,
        phase_centre,
        array_longitude_rad,
        array_latitude_rad,
        dut1,
        apply_precession,
    );

    let mut vis_model_fb = Array2::zeros((1, 1));
    let result = modeller.model_timestep_autos_with(timestamp, vis_model_fb.view_mut());

    assert!(result.is_ok());
}
