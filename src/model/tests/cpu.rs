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
            &obs.tile_uvws,
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
            &obs.tile_uvws,
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
            &obs.tile_uvws,
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
        &obs.tile_uvws,
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
        .model_points(
            vis.view_mut(),
            &obs.tile_uvws,
            obs.lst,
            obs.array_latitude_rad,
        )
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
                FluxDensity {
                    freq: 150e6,
                    i: vals[0],
                    ..Default::default()
                },
                FluxDensity {
                    freq: 175e6,
                    i: vals[1],
                    ..Default::default()
                },
                FluxDensity {
                    freq: 200e6,
                    i: vals[2],
                    ..Default::default()
                },
            ]),
        }
    };

    let mut srclist = SourceList::new();
    srclist.insert(
        "s1".to_string(),
        Source {
            components: vec![make_point_with_list(
                RADec::from_degrees(1.0, -27.0),
                [1.0, 3.0, 2.0],
            )]
            .into_boxed_slice(),
        },
    );
    srclist.insert(
        "s2".to_string(),
        Source {
            components: vec![make_point_with_list(
                RADec::from_degrees(1.1, -27.0),
                [0.5, 0.25, 1.5],
            )]
            .into_boxed_slice(),
        },
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

/// The factorised point/Gaussian kernels must agree with the straightforward
/// per-baseline reference, with and without a beam (the no-beam and
/// single-unique-tile cases take the hoisted-product path; a beam with a dead
/// dipole on one tile takes the per-baseline path).
#[test]
fn factorised_model_matches_reference() {
    use crate::beam::Delays;
    use crate::srclist::{ComponentType, FluxDensity, FluxDensityType, SourceComponent};
    use marlu::pos::xyz::xyzs_to_cross_uvws;
    use ndarray::Array2;

    let srclist = {
        let mut srclist = SourceList::new();
        srclist.insert(
            "mixed".to_string(),
            Source {
                components: vec![
                    get_point(RADec::from_degrees(0.0, -27.0), FluxType::List),
                    get_point(RADec::from_degrees(2.0, -25.0), FluxType::PowerLaw),
                    get_point(RADec::from_degrees(355.0, -30.0), FluxType::CurvedPowerLaw),
                    get_gaussian(RADec::from_degrees(1.0, -27.5), FluxType::PowerLaw),
                    get_gaussian2(RADec::from_degrees(358.0, -24.0), FluxType::List),
                    // A polarised (positive semi-definite) component, for the
                    // Cholesky half-product path.
                    SourceComponent {
                        radec: RADec::from_degrees(3.0, -29.0),
                        comp_type: ComponentType::Point,
                        flux_type: FluxDensityType::PowerLaw {
                            si: -0.7,
                            fd: FluxDensity {
                                freq: 150e6,
                                i: 2.0,
                                q: 0.5,
                                u: -0.7,
                                v: 0.3,
                            },
                        },
                    },
                ]
                .into_boxed_slice(),
            },
        );
        srclist
    };
    // A negative-flux component makes the Cholesky path fall back to the full
    // per-baseline products.
    let negative_srclist = {
        let mut srclist = srclist.clone();
        srclist.insert(
            "negative".to_string(),
            Source {
                components: vec![SourceComponent {
                    radec: RADec::from_degrees(357.0, -26.0),
                    comp_type: ComponentType::Point,
                    flux_type: FluxDensityType::PowerLaw {
                        si: -0.7,
                        fd: FluxDensity {
                            freq: 150e6,
                            i: -1.0,
                            q: 0.0,
                            u: 0.0,
                            v: 0.0,
                        },
                    },
                }]
                .into_boxed_slice(),
            },
        );
        srclist
    };

    for (no_beam, dead_dipole, srclist) in [
        (true, false, &srclist),
        (false, false, &srclist),
        (false, true, &srclist),
        (false, true, &negative_srclist),
    ] {
        let mut obs = ObsParams::new(no_beam);
        if dead_dipole {
            // Kill a dipole on the last tile so that the tiles are no longer
            // all identical for the beam.
            let mut gains = Array2::ones((obs.xyzs.len(), 32));
            gains[(obs.xyzs.len() - 1, 3)] = 0.0;
            obs.beam = Box::new(
                crate::beam::FEEBeam::new_from_env(
                    obs.xyzs.len(),
                    Delays::Partial(vec![0; 16]),
                    Some(gains),
                )
                .unwrap(),
            );
        }
        let modeller = obs.get_cpu_modeller(srclist);
        let hadec = obs.phase_centre.to_hadec(obs.lst);
        let tile_uvws: Vec<UVW> = obs
            .xyzs
            .iter()
            .map(|xyz| UVW::from_xyz(*xyz, hadec))
            .collect();
        let bl_uvws = xyzs_to_cross_uvws(&obs.xyzs, hadec);

        let shape = (obs.freqs.len(), bl_uvws.len());
        let mut fast = Array2::<Jones<f32>>::zeros(shape);
        let mut reference = Array2::<Jones<f32>>::zeros(shape);
        modeller
            .model_points(fast.view_mut(), &tile_uvws, obs.lst, obs.array_latitude_rad)
            .unwrap();
        modeller
            .model_gaussians(fast.view_mut(), &tile_uvws, obs.lst, obs.array_latitude_rad)
            .unwrap();
        modeller
            .model_points_reference(
                reference.view_mut(),
                &bl_uvws,
                obs.lst,
                obs.array_latitude_rad,
            )
            .unwrap();
        modeller
            .model_gaussians_reference(
                reference.view_mut(),
                &bl_uvws,
                obs.lst,
                obs.array_latitude_rad,
            )
            .unwrap();
        assert!(
            fast.iter().any(|j| j[0].norm() > 0.0),
            "nothing was modelled"
        );
        assert_abs_diff_eq!(fast, reference, epsilon = 1e-5);
    }
}

/// `jones_cholesky` must factor every positive semi-definite flux density
/// (including ones with a zero XX, e.g. pure -Q) as `L L^H`, and refuse
/// everything else so that the modeller falls back to the full product.
#[test]
fn jones_cholesky_factors_psd_and_rejects_the_rest() {
    use crate::model::cpu::jones_cholesky;
    use marlu::c64;

    let j = |xx: (f64, f64), xy: (f64, f64), yx: (f64, f64), yy: (f64, f64)| {
        Jones::from([
            c64::new(xx.0, xx.1),
            c64::new(xy.0, xy.1),
            c64::new(yx.0, yx.1),
            c64::new(yy.0, yy.1),
        ])
    };

    // Positive semi-definite: I = 2, Q = 0.5, U = 0.3, V = 0.2; unpolarised;
    // zero XX (I = -Q); everything zero.
    for f in [
        j((2.5, 0.0), (0.3, 0.2), (0.3, -0.2), (1.5, 0.0)),
        j((1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 0.0)),
        j((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (2.0, 0.0)),
        j((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
    ] {
        let l = jones_cholesky(f).unwrap_or_else(|| panic!("{f:?} is PSD"));
        assert_eq!(l[1], c64::new(0.0, 0.0), "L is lower-triangular");
        assert_abs_diff_eq!(l * l.h(), f, epsilon = 1e-12);
    }

    for (f, why) in [
        (
            j((1.0, 0.0), (0.3, 0.2), (0.3, 0.2), (1.0, 0.0)),
            "not Hermitian (XY != conj(YX))",
        ),
        (
            j((1.0, 0.5), (0.0, 0.0), (0.0, 0.0), (1.0, 0.0)),
            "not Hermitian (complex XX)",
        ),
        (
            j((-1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 0.0)),
            "negative XX",
        ),
        (
            j((0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0)),
            "zero XX with non-zero XY/YX",
        ),
        (
            j((1.0, 0.0), (2.0, 0.0), (2.0, 0.0), (1.0, 0.0)),
            "negative determinant (I < |polarisation|)",
        ),
    ] {
        assert!(jones_cholesky(f).is_none(), "{why}: {f:?}");
    }
}
