// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use criterion::*;

fn model_benchmarks(c: &mut Criterion) {
    // These "nothing" benchmarks should be commented out when real benchmarking
    // is desired.
    // c.bench_function("nothing", |b| b.iter(marlu::Jones::<f64>::identity));

    use std::collections::HashSet;

    use hifitime::{Duration, Epoch};
    use marlu::{
        constants::{MWA_LAT_RAD, MWA_LONG_RAD},
        Jones, RADec, XyzGeodetic,
    };
    use mwa_hyperdrive::{
        beam::{create_fee_beam_object, Delays},
        model,
        srclist::{
            ComponentType, FluxDensity, FluxDensityType, ShapeletCoeff, Source, SourceComponent,
            SourceList,
        },
    };
    use ndarray::prelude::*;
    use std::ops::Deref;
    use vec1::vec1;

    let num_tiles = 128;
    let num_bls = (num_tiles * (num_tiles - 1)) / 2;
    let phase_centre = RADec::from_degrees(0.0, -27.0);
    let dut1 = Duration::from_total_nanoseconds(0);
    let apply_precession = true;
    let timestamp = Epoch::from_gpst_seconds(1065880128.0);
    let xyzs = vec![XyzGeodetic::default(); num_tiles];
    let flagged_tiles = HashSet::new();
    let beam_file: Option<&str> = None; // Assume the env. variable is set.
    let beam =
        create_fee_beam_object(beam_file, num_tiles, Delays::Partial(vec![0; 16]), None).unwrap();

    let mut points = c.benchmark_group("model FEE points");
    for (num_power_law_points, num_chans) in [(10, 2), (100, 2)] {
        points.bench_function(
            format!("{num_power_law_points} with CPU, {num_tiles} tiles, {num_chans} channels"),
            |b| {
                let freqs = Array1::linspace(150e6, 200e6, num_chans).to_vec();
                let mut vis = Array2::from_elem((num_chans, num_bls), Jones::default());
                let mut source_list = SourceList::default();
                for i in 0..num_power_law_points {
                    source_list.insert(
                        format!("source{i}"),
                        Source {
                            components: vec1![SourceComponent {
                                radec: RADec::from_degrees(0.0, -27.0),
                                comp_type: ComponentType::Point,
                                flux_type: FluxDensityType::PowerLaw {
                                    si: -0.7,
                                    fd: FluxDensity {
                                        freq: 150e6,
                                        i: 1.0,
                                        q: 0.0,
                                        u: 0.0,
                                        v: 0.0,
                                    },
                                },
                            }],
                        },
                    );
                }
                let mut modeller = model::new_sky_modeller(
                    #[cfg(feature = "cuda")]
                    true,
                    beam.deref(),
                    &source_list,
                    &xyzs,
                    &freqs,
                    &flagged_tiles,
                    phase_centre,
                    MWA_LONG_RAD,
                    MWA_LAT_RAD,
                    dut1,
                    apply_precession,
                )
                .unwrap();

                b.iter(|| {
                    modeller.model_timestep(vis.view_mut(), timestamp).unwrap();
                })
            },
        );
    }

    #[cfg(feature = "cuda")]
    for (num_power_law_points, num_chans) in [(100, 2), (1000, 2)] {
        points.bench_function(
            format!("{num_power_law_points} with GPU, 128 tiles, {num_chans} channels"),
            |b| {
                let freqs = Array1::linspace(150e6, 200e6, num_chans).to_vec();
                let mut vis = Array2::from_elem((num_chans, num_bls), Jones::default());
                let mut source_list = SourceList::default();
                for i in 0..num_power_law_points {
                    source_list.insert(
                        format!("source{i}"),
                        Source {
                            components: vec1![SourceComponent {
                                radec: RADec::from_degrees(0.0, -27.0),
                                comp_type: ComponentType::Point,
                                flux_type: FluxDensityType::PowerLaw {
                                    si: -0.7,
                                    fd: FluxDensity {
                                        freq: 150e6,
                                        i: 1.0,
                                        q: 0.0,
                                        u: 0.0,
                                        v: 0.0,
                                    },
                                },
                            }],
                        },
                    );
                }
                let mut modeller = model::new_sky_modeller(
                    false,
                    beam.deref(),
                    &source_list,
                    &xyzs,
                    &freqs,
                    &flagged_tiles,
                    phase_centre,
                    MWA_LONG_RAD,
                    MWA_LAT_RAD,
                    dut1,
                    apply_precession,
                )
                .unwrap();

                b.iter(|| modeller.model_timestep(vis.view_mut(), timestamp).unwrap());
            },
        );
    }

    #[cfg(feature = "cuda")]
    for (num_list_points, num_chans) in [
        (1, 768),
        (32, 768),
        (64, 768),
        (128, 768),
        (256, 768),
        (512, 768),
        (1024, 768),
    ] {
        points.bench_function(
            format!("{num_list_points} with GPU, 128 tiles, {num_chans} channels"),
            |b| {
                let freqs = Array1::linspace(150e6, 200e6, num_chans).to_vec();
                let mut vis = Array2::from_elem((num_chans, num_bls), Jones::default());
                let mut source_list = SourceList::default();
                for i in 0..num_list_points {
                    source_list.insert(
                        format!("source{i}"),
                        Source {
                            components: vec1![SourceComponent {
                                radec: RADec::from_degrees(0.0, -27.0),
                                comp_type: ComponentType::Point,
                                flux_type: FluxDensityType::List(vec1![
                                    FluxDensity {
                                        freq: 150e6,
                                        i: 1.0,
                                        q: 0.0,
                                        u: 0.0,
                                        v: 0.0,
                                    },
                                    FluxDensity {
                                        freq: 175e6,
                                        i: 3.0,
                                        q: 0.0,
                                        u: 0.0,
                                        v: 0.0,
                                    },
                                    FluxDensity {
                                        freq: 200e6,
                                        i: 2.0,
                                        q: 0.0,
                                        u: 0.0,
                                        v: 0.0,
                                    }
                                ]),
                            }],
                        },
                    );
                }
                let mut modeller = model::new_sky_modeller(
                    false,
                    beam.deref(),
                    &source_list,
                    &xyzs,
                    &freqs,
                    &flagged_tiles,
                    phase_centre,
                    MWA_LONG_RAD,
                    MWA_LAT_RAD,
                    dut1,
                    apply_precession,
                )
                .unwrap();

                b.iter(|| modeller.model_timestep(vis.view_mut(), timestamp).unwrap());
            },
        );
    }

    points.finish();

    let mut gaussians = c.benchmark_group("model FEE gaussians");
    gaussians.bench_function("100 with CPU, 128 tiles, 2 channels", |b| {
        let num_gaussians = 100;
        let num_chans = 2;
        let freqs = Array1::linspace(150e6, 200e6, num_chans).to_vec();
        let mut vis = Array2::from_elem((num_chans, num_bls), Jones::default());
        let mut source_list = SourceList::default();
        for i in 0..num_gaussians {
            source_list.insert(
                format!("source{i}"),
                Source {
                    components: vec1![SourceComponent {
                        radec: RADec::from_degrees(0.0, -27.0),
                        comp_type: ComponentType::Gaussian {
                            maj: 1.0,
                            min: 0.5,
                            pa: 0.0,
                        },
                        flux_type: FluxDensityType::PowerLaw {
                            si: -0.7,
                            fd: FluxDensity {
                                freq: 150e6,
                                i: 1.0,
                                q: 0.0,
                                u: 0.0,
                                v: 0.0,
                            },
                        },
                    }],
                },
            );
        }
        let mut modeller = model::new_sky_modeller(
            #[cfg(feature = "cuda")]
            true,
            beam.deref(),
            &source_list,
            &xyzs,
            &freqs,
            &flagged_tiles,
            phase_centre,
            MWA_LONG_RAD,
            MWA_LAT_RAD,
            dut1,
            apply_precession,
        )
        .unwrap();

        b.iter(|| {
            modeller.model_timestep(vis.view_mut(), timestamp).unwrap();
        })
    });

    #[cfg(feature = "cuda")]
    gaussians.bench_function("100 with GPU, 128 tiles, 2 channels", |b| {
        let num_gaussians = 100;
        let num_chans = 2;
        let freqs = Array1::linspace(150e6, 200e6, num_chans).to_vec();
        let mut vis = Array2::from_elem((num_chans, num_bls), Jones::default());
        let mut source_list = SourceList::default();
        for i in 0..num_gaussians {
            source_list.insert(
                format!("source{i}"),
                Source {
                    components: vec1![SourceComponent {
                        radec: RADec::from_degrees(0.0, -27.0),
                        comp_type: ComponentType::Gaussian {
                            maj: 1.0,
                            min: 0.5,
                            pa: 0.0,
                        },
                        flux_type: FluxDensityType::PowerLaw {
                            si: -0.7,
                            fd: FluxDensity {
                                freq: 150e6,
                                i: 1.0,
                                q: 0.0,
                                u: 0.0,
                                v: 0.0,
                            },
                        },
                    }],
                },
            );
        }
        let mut modeller = model::new_sky_modeller(
            #[cfg(feature = "cuda")]
            false,
            beam.deref(),
            &source_list,
            &xyzs,
            &freqs,
            &flagged_tiles,
            phase_centre,
            MWA_LONG_RAD,
            MWA_LAT_RAD,
            dut1,
            apply_precession,
        )
        .unwrap();

        b.iter(|| modeller.model_timestep(vis.view_mut(), timestamp).unwrap());
    });

    #[cfg(feature = "cuda")]
    gaussians.bench_function("1000 with GPU, 128 tiles, 64 channels", |b| {
        let num_gaussians = 1000;
        let num_chans = 64;
        let freqs = Array1::linspace(150e6, 200e6, num_chans).to_vec();
        let mut vis = Array2::from_elem((num_chans, num_bls), Jones::default());
        let mut source_list = SourceList::default();
        for i in 0..num_gaussians {
            source_list.insert(
                format!("source{i}"),
                Source {
                    components: vec1![SourceComponent {
                        radec: RADec::from_degrees(0.0, -27.0),
                        comp_type: ComponentType::Gaussian {
                            maj: 1.0,
                            min: 0.5,
                            pa: 0.0,
                        },
                        flux_type: FluxDensityType::PowerLaw {
                            si: -0.7,
                            fd: FluxDensity {
                                freq: 150e6,
                                i: 1.0,
                                q: 0.0,
                                u: 0.0,
                                v: 0.0,
                            },
                        },
                    }],
                },
            );
        }
        let mut modeller = model::new_sky_modeller(
            #[cfg(feature = "cuda")]
            false,
            beam.deref(),
            &source_list,
            &xyzs,
            &freqs,
            &flagged_tiles,
            phase_centre,
            MWA_LONG_RAD,
            MWA_LAT_RAD,
            dut1,
            apply_precession,
        )
        .unwrap();

        b.iter(|| modeller.model_timestep(vis.view_mut(), timestamp).unwrap());
    });
    gaussians.finish();

    let mut shapelets = c.benchmark_group("model FEE shapelets");
    shapelets.bench_function(
        "100 with CPU (10 coeffs each), 128 tiles, 2 channels",
        |b| {
            let num_shapelets = 100;
            let num_chans = 2;
            let freqs = Array1::linspace(150e6, 200e6, num_chans).to_vec();
            let mut vis = Array2::from_elem((num_chans, num_bls), Jones::default());
            let mut source_list = SourceList::default();
            for i in 0..num_shapelets {
                source_list.insert(
                    format!("source{i}"),
                    Source {
                        components: vec1![SourceComponent {
                            radec: RADec::from_degrees(0.0, -27.0),
                            comp_type: ComponentType::Shapelet {
                                maj: 1.0,
                                min: 0.5,
                                pa: 0.0,
                                coeffs: vec![
                                    ShapeletCoeff {
                                        n1: 0,
                                        n2: 1,
                                        value: 1.0,
                                    };
                                    10
                                ],
                            },
                            flux_type: FluxDensityType::PowerLaw {
                                si: -0.7,
                                fd: FluxDensity {
                                    freq: 150e6,
                                    i: 1.0,
                                    q: 0.0,
                                    u: 0.0,
                                    v: 0.0,
                                },
                            },
                        }],
                    },
                );
            }
            let mut modeller = model::new_sky_modeller(
                #[cfg(feature = "cuda")]
                true,
                beam.deref(),
                &source_list,
                &xyzs,
                &freqs,
                &flagged_tiles,
                phase_centre,
                MWA_LONG_RAD,
                MWA_LAT_RAD,
                dut1,
                apply_precession,
            )
            .unwrap();

            b.iter(|| {
                modeller.model_timestep(vis.view_mut(), timestamp).unwrap();
            })
        },
    );

    #[cfg(feature = "cuda")]
    shapelets.bench_function(
        "100 with GPU (10 coeffs each), 128 tiles, 2 channels",
        |b| {
            let num_shapelets = 100;
            let num_chans = 2;
            let freqs = Array1::linspace(150e6, 200e6, num_chans).to_vec();
            let mut vis = Array2::from_elem((num_chans, num_bls), Jones::default());
            let mut source_list = SourceList::default();
            for i in 0..num_shapelets {
                source_list.insert(
                    format!("source{i}"),
                    Source {
                        components: vec1![SourceComponent {
                            radec: RADec::from_degrees(0.0, -27.0),
                            comp_type: ComponentType::Shapelet {
                                maj: 1.0,
                                min: 0.5,
                                pa: 0.0,
                                coeffs: vec![
                                    ShapeletCoeff {
                                        n1: 0,
                                        n2: 1,
                                        value: 1.0,
                                    };
                                    10
                                ],
                            },
                            flux_type: FluxDensityType::PowerLaw {
                                si: -0.7,
                                fd: FluxDensity {
                                    freq: 150e6,
                                    i: 1.0,
                                    q: 0.0,
                                    u: 0.0,
                                    v: 0.0,
                                },
                            },
                        }],
                    },
                );
            }
            let mut modeller = model::new_sky_modeller(
                #[cfg(feature = "cuda")]
                false,
                beam.deref(),
                &source_list,
                &xyzs,
                &freqs,
                &flagged_tiles,
                phase_centre,
                MWA_LONG_RAD,
                MWA_LAT_RAD,
                dut1,
                apply_precession,
            )
            .unwrap();

            b.iter(|| modeller.model_timestep(vis.view_mut(), timestamp).unwrap());
        },
    );

    #[cfg(feature = "cuda")]
    shapelets.bench_function(
        "1000 with GPU (10 coeffs each), 128 tiles, 64 channels",
        |b| {
            let num_shapelets = 1000;
            let num_chans = 64;
            let freqs = Array1::linspace(150e6, 200e6, num_chans).to_vec();
            let mut vis = Array2::from_elem((num_chans, num_bls), Jones::default());
            let mut source_list = SourceList::default();
            for i in 0..num_shapelets {
                source_list.insert(
                    format!("source{i}"),
                    Source {
                        components: vec1![SourceComponent {
                            radec: RADec::from_degrees(0.0, -27.0),
                            comp_type: ComponentType::Shapelet {
                                maj: 1.0,
                                min: 0.5,
                                pa: 0.0,
                                coeffs: vec![
                                    ShapeletCoeff {
                                        n1: 0,
                                        n2: 1,
                                        value: 1.0,
                                    };
                                    10
                                ],
                            },
                            flux_type: FluxDensityType::PowerLaw {
                                si: -0.7,
                                fd: FluxDensity {
                                    freq: 150e6,
                                    i: 1.0,
                                    q: 0.0,
                                    u: 0.0,
                                    v: 0.0,
                                },
                            },
                        }],
                    },
                );
            }
            let mut modeller = model::new_sky_modeller(
                #[cfg(feature = "cuda")]
                false,
                beam.deref(),
                &source_list,
                &xyzs,
                &freqs,
                &flagged_tiles,
                phase_centre,
                MWA_LONG_RAD,
                MWA_LAT_RAD,
                dut1,
                apply_precession,
            )
            .unwrap();

            b.iter(|| modeller.model_timestep(vis.view_mut(), timestamp).unwrap());
        },
    );
    shapelets.finish();
}

fn calibrate_benchmarks(c: &mut Criterion) {
    // c.bench_function("nothing", |b| b.iter(marlu::Jones::<f64>::identity));

    use hifitime::Epoch;
    use marlu::Jones;
    use ndarray::prelude::*;
    use vec1::{vec1, Vec1};

    use mwa_hyperdrive::{
        averaging::{Chanblock, Timeblock},
        di_calibrate::calibrate_timeblocks,
    };

    let num_timesteps = 10;
    let num_timeblocks = 1;
    let mut timeblocks = Vec::with_capacity(num_timeblocks);
    timeblocks.push(Timeblock {
        index: 0,
        range: 0..num_timesteps,
        timestamps: vec1![
            Epoch::from_gpst_seconds(1090008640.0),
            Epoch::from_gpst_seconds(1090008641.0),
            Epoch::from_gpst_seconds(1090008642.0),
            Epoch::from_gpst_seconds(1090008643.0),
            Epoch::from_gpst_seconds(1090008644.0),
            Epoch::from_gpst_seconds(1090008645.0),
            Epoch::from_gpst_seconds(1090008646.0),
            Epoch::from_gpst_seconds(1090008647.0),
            Epoch::from_gpst_seconds(1090008648.0),
            Epoch::from_gpst_seconds(1090008649.0),
        ],
        median: Epoch::from_gpst_seconds(1090008644.5),
    });
    let timeblocks = Vec1::try_from_vec(timeblocks).unwrap();

    let num_chanblocks = 100;
    let mut chanblocks = Vec::with_capacity(num_chanblocks);
    for i_chanblock in 0..num_chanblocks {
        chanblocks.push(Chanblock {
            chanblock_index: i_chanblock as _,
            unflagged_index: i_chanblock as _,
            _freq: 150e6 + i_chanblock as f64,
        })
    }
    let chanblocks = Vec1::try_from_vec(chanblocks).unwrap();

    let num_tiles = 128;
    let num_baselines = num_tiles * (num_tiles - 1) / 2;

    let vis_shape = (num_timesteps, num_chanblocks, num_baselines);
    let vis_data: Array3<Jones<f32>> = Array3::from_elem(vis_shape, Jones::identity() * 4.0);
    let vis_model: Array3<Jones<f32>> = Array3::from_elem(vis_shape, Jones::identity());

    c.bench_function(
        &format!("calibrate with {num_timesteps} timesteps, {num_baselines} baselines, {num_chanblocks} chanblocks"),
        |b| {
            b.iter(|| {
                calibrate_timeblocks(
                    vis_data.view(),
                    vis_model.view(),
                    &timeblocks,
                    &chanblocks,
                    50,
                    1e-8,
                    1e-4,
                    false,
                    false,
                );
            });
        }
    );
}

fn source_list_benchmarks(c: &mut Criterion) {
    c.bench_function("nothing", |b| b.iter(marlu::Jones::<f64>::identity));

    // use mwa_hyperdrive::srclist::get_instrumental_flux_densities;

    // let num_comps = 1000;
    // let num_freqs = 400;

    // let fds = vec1![
    //     FluxDensity {
    //         freq: 150e6,
    //         i: 1.0,
    //         q: 0.0,
    //         u: 0.0,
    //         v: 0.0,
    //     },
    //     FluxDensity {
    //         freq: 175e6,
    //         i: 3.0,
    //         q: 0.0,
    //         u: 0.0,
    //         v: 0.0,
    //     },
    //     FluxDensity {
    //         freq: 200e6,
    //         i: 2.0,
    //         q: 0.0,
    //         u: 0.0,
    //         v: 0.0,
    //     }
    // ];
    // let comp_fds = vec![FluxDensityType::List { fds }; num_comps];
    // let freqs: Vec<f64> = Array1::linspace(140e6, 210e6, num_comps).to_vec();

    // c.bench_function(
    //     &format!("Estimate flux densities for source list with {num_comps} 'list' components over {num_freqs} frequencies"),
    //     |b| {
    //         b.iter(|| {
    //             get_instrumental_flux_densities(&comp_fds, &freqs);
    //         });
    //     }
    // );
}

criterion_group!(
    name = model;
    config = Criterion::default().sample_size(10);
    targets = model_benchmarks,
);
criterion_group!(
    name = calibrate;
    config = Criterion::default().sample_size(10);
    targets = calibrate_benchmarks,
);
// criterion_group!(
//     name = source_lists;
//     config = Criterion::default().sample_size(100);
//     targets = source_list_benchmarks,
// );
// criterion_main!(model, calibrate, source_lists);
criterion_main!(model, calibrate);
