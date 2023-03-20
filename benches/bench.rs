// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::{
    collections::HashSet,
    path::{Path, PathBuf},
};

use criterion::*;
use hifitime::{Duration, Epoch};
use marlu::{
    constants::{MWA_LAT_RAD, MWA_LONG_RAD},
    Jones, RADec, XyzGeodetic,
};
use ndarray::prelude::*;
use tempfile::Builder;
use vec1::{vec1, Vec1};

use mwa_hyperdrive::{
    calibrate_timeblocks, create_beam_object,
    model::{self, SkyModeller},
    srclist::{
        get_instrumental_flux_densities, ComponentType, FluxDensity, FluxDensityType,
        ShapeletCoeff, Source, SourceComponent, SourceList,
    },
    Chanblock, CrossData, Delays, MsReader, Polarisations, RawDataCorrections, RawDataReader,
    TileBaselineFlags, Timeblock, UvfitsReader,
};

fn model_benchmarks(c: &mut Criterion) {
    let num_tiles = 128;
    let num_bls = (num_tiles * (num_tiles - 1)) / 2;
    let phase_centre = RADec::from_degrees(0.0, -27.0);
    let dut1 = Duration::default();
    let apply_precession = true;
    let timestamp = Epoch::from_gpst_seconds(1065880128.0);
    let xyzs = vec![XyzGeodetic::default(); num_tiles];
    let flagged_tiles = HashSet::new();
    let beam = create_beam_object(Some("fee"), num_tiles, Delays::Partial(vec![0; 16])).unwrap();

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
                            components: vec![SourceComponent {
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
                            }]
                            .into_boxed_slice(),
                        },
                    );
                }
                let modeller = model::SkyModellerCpu::new(
                    &*beam,
                    &source_list,
                    Polarisations::default(),
                    &xyzs,
                    &freqs,
                    &flagged_tiles,
                    phase_centre,
                    MWA_LONG_RAD,
                    MWA_LAT_RAD,
                    dut1,
                    apply_precession,
                );

                b.iter(|| {
                    modeller
                        .model_timestep_with(timestamp, vis.view_mut())
                        .unwrap();
                })
            },
        );
    }

    #[cfg(feature = "cuda")]
    for (num_sources, num_chans) in [
        (1, 768),
        (32, 768),
        (64, 768),
        (128, 768),
        (256, 768),
        (512, 768),
        (1024, 768),
    ] {
        points.bench_function(
            format!("{num_sources} with GPU, 128 tiles, {num_chans} channels"),
            |b| {
                let freqs = Array1::linspace(150e6, 200e6, num_chans).to_vec();
                let mut vis = Array2::from_elem((num_bls, num_chans), Jones::default());
                let mut source_list = SourceList::default();
                for i in 0..num_sources {
                    source_list.insert(
                        format!("source{i}"),
                        Source {
                            components: vec![SourceComponent {
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
                            }]
                            .into_boxed_slice(),
                        },
                    );
                }
                let modeller = model::SkyModellerCuda::new(
                    &*beam,
                    &source_list,
                    Polarisations::default(),
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
                    modeller
                        .model_timestep_with(timestamp, vis.view_mut())
                        .unwrap()
                });
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
                    components: vec![SourceComponent {
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
                    }]
                    .into_boxed_slice(),
                },
            );
        }
        let modeller = model::SkyModellerCpu::new(
            &*beam,
            &source_list,
            Polarisations::default(),
            &xyzs,
            &freqs,
            &flagged_tiles,
            phase_centre,
            MWA_LONG_RAD,
            MWA_LAT_RAD,
            dut1,
            apply_precession,
        );

        b.iter(|| {
            modeller
                .model_timestep_with(timestamp, vis.view_mut())
                .unwrap();
        })
    });

    #[cfg(feature = "cuda")]
    for (num_sources, num_chans) in [
        (1, 768),
        (32, 768),
        (64, 768),
        (128, 768),
        (256, 768),
        (512, 768),
        (1024, 768),
    ] {
        gaussians.bench_function(
            format!("{num_sources} with GPU, 128 tiles, {num_chans} channels"),
            |b| {
                let freqs = Array1::linspace(150e6, 200e6, num_chans).to_vec();
                let mut vis = Array2::from_elem((num_bls, num_chans), Jones::default());
                let mut source_list = SourceList::default();
                for i in 0..num_sources {
                    source_list.insert(
                        format!("source{i}"),
                        Source {
                            components: vec![SourceComponent {
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
                            }]
                            .into_boxed_slice(),
                        },
                    );
                }
                let modeller = model::SkyModellerCuda::new(
                    &*beam,
                    &source_list,
                    Polarisations::default(),
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
                    modeller
                        .model_timestep_with(timestamp, vis.view_mut())
                        .unwrap()
                });
            },
        );
    }
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
                        components: vec![SourceComponent {
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
                                ]
                                .into_boxed_slice(),
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
                        }]
                        .into_boxed_slice(),
                    },
                );
            }
            let modeller = model::SkyModellerCpu::new(
                &*beam,
                &source_list,
                Polarisations::default(),
                &xyzs,
                &freqs,
                &flagged_tiles,
                phase_centre,
                MWA_LONG_RAD,
                MWA_LAT_RAD,
                dut1,
                apply_precession,
            );

            b.iter(|| {
                modeller
                    .model_timestep_with(timestamp, vis.view_mut())
                    .unwrap();
            })
        },
    );

    #[cfg(feature = "cuda")]
    for (num_sources, num_chans) in [
        (1, 768),
        (32, 768),
        (64, 768),
        (128, 768),
        (256, 768),
        (512, 768),
        (1024, 768),
    ] {
        shapelets.bench_function(
            format!("{num_sources} with GPU (10 coeffs each), 128 tiles, {num_chans} channels"),
            |b| {
                let freqs = Array1::linspace(150e6, 200e6, num_chans).to_vec();
                let mut vis = Array2::from_elem((num_bls, num_chans), Jones::default());
                let mut source_list = SourceList::default();
                for i in 0..num_sources {
                    source_list.insert(
                        format!("source{i}"),
                        Source {
                            components: vec![SourceComponent {
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
                                    ]
                                    .into_boxed_slice(),
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
                            }]
                            .into_boxed_slice(),
                        },
                    );
                }
                let modeller = model::SkyModellerCuda::new(
                    &*beam,
                    &source_list,
                    Polarisations::default(),
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
                    modeller
                        .model_timestep_with(timestamp, vis.view_mut())
                        .unwrap()
                });
            },
        );
    }
    shapelets.finish();
}

fn calibrate_benchmarks(c: &mut Criterion) {
    let num_timesteps = 10;
    let num_timeblocks = 1;
    let mut timeblocks = Vec::with_capacity(num_timeblocks);
    timeblocks.push(Timeblock {
        index: 0,
        range: 0..num_timesteps,
        timestamps: vec1![
            (Epoch::from_gpst_seconds(1090008640.0), 0),
            (Epoch::from_gpst_seconds(1090008641.0), 1),
            (Epoch::from_gpst_seconds(1090008642.0), 2),
            (Epoch::from_gpst_seconds(1090008643.0), 3),
            (Epoch::from_gpst_seconds(1090008644.0), 4),
            (Epoch::from_gpst_seconds(1090008645.0), 5),
            (Epoch::from_gpst_seconds(1090008646.0), 6),
            (Epoch::from_gpst_seconds(1090008647.0), 7),
            (Epoch::from_gpst_seconds(1090008648.0), 8),
            (Epoch::from_gpst_seconds(1090008649.0), 9),
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
            freq: 150e6 + i_chanblock as f64,
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
                    Polarisations::default(),
                    false,
                );
            });
        }
    );
}

fn source_list_benchmarks(c: &mut Criterion) {
    let num_comps = 1000;
    let num_freqs = 400;

    let fds = vec1![
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
    ];
    let comp_fds = vec![FluxDensityType::List(fds); num_comps];
    let freqs: Vec<f64> = Array1::linspace(140e6, 210e6, num_comps).to_vec();

    c.bench_function(
        &format!("Estimate flux densities for source list with {num_comps} 'list' components over {num_freqs} frequencies"),
        |b| {
            b.iter(|| {
                get_instrumental_flux_densities(&comp_fds, &freqs);
            });
        }
    );
}

fn io_benchmarks(c: &mut Criterion) {
    let metafits = PathBuf::from("test_files/1090008640/1090008640.metafits");
    // Put the disk data into RAM to help reduce transient IO effects.
    let temp_dir = tempfile::tempdir().unwrap();
    let gpubox = {
        let on_disk =
            PathBuf::from("test_files/1090008640/1090008640_20140721201027_gpubox01_00.fits");
        let in_ram = temp_dir.path().join(on_disk.file_name().unwrap());
        std::fs::copy(&on_disk, &in_ram).unwrap();
        in_ram
    };
    let uvfits = {
        let on_disk = PathBuf::from("test_files/1090008640/1090008640.uvfits");
        let in_ram = Builder::new().suffix(".uvfits").tempfile().unwrap();
        std::fs::copy(on_disk, in_ram.path()).unwrap();
        in_ram
    };
    let ms = {
        let on_disk = PathBuf::from("test_files/1090008640/1090008640.ms");
        let in_ram = Builder::new().suffix(".ms").tempdir().unwrap();
        copy_recursively(on_disk, in_ram.path()).unwrap();
        in_ram
    };

    // Open the readers.
    let gpuboxes = [gpubox];
    let raw = RawDataReader::new(
        &metafits,
        &gpuboxes,
        None,
        RawDataCorrections::do_nothing(),
        None,
    )
    .unwrap();
    let uvfits = UvfitsReader::new(uvfits.path().to_path_buf(), Some(&metafits), None).unwrap();
    let ms = MsReader::new(ms.path().to_path_buf(), None, Some(&metafits), None).unwrap();

    let tile_baseline_flags = TileBaselineFlags::new(128, HashSet::new());

    let mut raw_bench_group = c.benchmark_group("Raw IO");
    raw_bench_group.bench_function(
        "Read crosses for timestep, no channel flags, no corrections",
        |b| {
            // Prepare arrays to be read into.
            let flagged_fine_chans = HashSet::new();
            let mut vis_fb = Array2::default((32 - flagged_fine_chans.len(), 8128));
            let mut weights_fb = Array2::default(vis_fb.raw_dim());

            b.iter(|| {
                let crosses = CrossData {
                    vis_fb: vis_fb.view_mut(),
                    weights_fb: weights_fb.view_mut(),
                    tile_baseline_flags: &tile_baseline_flags,
                };

                raw.read_inner(Some(crosses), None, 0, &flagged_fine_chans)
            });
        },
    );
    raw_bench_group.bench_function(
        "Read crosses for timestep, MWA channel flags, no corrections",
        |b| {
            // Prepare arrays to be read into.
            let flagged_fine_chans = HashSet::from([0, 1, 16, 30, 31]);
            let mut vis_fb = Array2::default((32 - flagged_fine_chans.len(), 8128));
            let mut weights_fb = Array2::default(vis_fb.raw_dim());

            b.iter(|| {
                let crosses = CrossData {
                    vis_fb: vis_fb.view_mut(),
                    weights_fb: weights_fb.view_mut(),
                    tile_baseline_flags: &tile_baseline_flags,
                };

                raw.read_inner(Some(crosses), None, 0, &flagged_fine_chans)
            });
        },
    );
    raw_bench_group.bench_function(
        "Read crosses for timestep, no channel flags, default corrections",
        |b| {
            // Need to re-make the reader with the new corrections.
            let raw = RawDataReader::new(
                &metafits,
                &gpuboxes,
                None,
                RawDataCorrections::default(),
                None,
            )
            .unwrap();

            // Prepare arrays to be read into.
            let flagged_fine_chans = HashSet::new();
            let mut vis_fb = Array2::default((32 - flagged_fine_chans.len(), 8128));
            let mut weights_fb = Array2::default(vis_fb.raw_dim());

            b.iter(|| {
                let crosses = CrossData {
                    vis_fb: vis_fb.view_mut(),
                    weights_fb: weights_fb.view_mut(),
                    tile_baseline_flags: &tile_baseline_flags,
                };

                raw.read_inner(Some(crosses), None, 0, &flagged_fine_chans)
            });
        },
    );
    raw_bench_group.finish();

    let mut uvfits_bench_group = c.benchmark_group("uvfits IO");
    uvfits_bench_group.bench_function("Read crosses for timestep, no channel flags", |b| {
        // Prepare arrays to be read into.
        let flagged_fine_chans = HashSet::new();
        let mut vis_fb = Array2::default((32 - flagged_fine_chans.len(), 8128));
        let mut weights_fb = Array2::default(vis_fb.raw_dim());

        b.iter(|| {
            let crosses = CrossData {
                vis_fb: vis_fb.view_mut(),
                weights_fb: weights_fb.view_mut(),
                tile_baseline_flags: &tile_baseline_flags,
            };

            uvfits.read_inner::<4, 3>(Some(crosses), None, 0, &flagged_fine_chans)
        });
    });
    uvfits_bench_group.bench_function("Read crosses for timestep, MWA channel flags", |b| {
        let flagged_fine_chans = HashSet::from([0, 1, 16, 30, 31]);
        let mut cross_vis = Array2::default((32 - flagged_fine_chans.len(), 8128));
        let mut cross_weights = Array2::default(cross_vis.raw_dim());

        b.iter(|| {
            let crosses = CrossData {
                vis_fb: cross_vis.view_mut(),
                weights_fb: cross_weights.view_mut(),
                tile_baseline_flags: &tile_baseline_flags,
            };

            uvfits.read_inner::<4, 3>(Some(crosses), None, 0, &flagged_fine_chans)
        });
    });
    uvfits_bench_group.finish();

    let mut ms_bench_group = c.benchmark_group("MS IO");
    ms_bench_group.bench_function("Read crosses for timestep, no channel flags", |b| {
        // Prepare arrays to be read into.
        let flagged_fine_chans = HashSet::new();
        let mut cross_vis = Array2::default((32 - flagged_fine_chans.len(), 8128));
        let mut cross_weights = Array2::default(cross_vis.raw_dim());

        b.iter(|| {
            let crosses = CrossData {
                vis_fb: cross_vis.view_mut(),
                weights_fb: cross_weights.view_mut(),
                tile_baseline_flags: &tile_baseline_flags,
            };

            ms.read_inner::<4>(Some(crosses), None, 0, &flagged_fine_chans)
        });
    });
    ms_bench_group.bench_function("Read crosses for timestep, MWA channel flags", |b| {
        let flagged_fine_chans = HashSet::from([0, 1, 16, 30, 31]);
        let mut cross_vis = Array2::default((32 - flagged_fine_chans.len(), 8128));
        let mut cross_weights = Array2::default(cross_vis.raw_dim());

        b.iter(|| {
            let crosses = CrossData {
                vis_fb: cross_vis.view_mut(),
                weights_fb: cross_weights.view_mut(),
                tile_baseline_flags: &tile_baseline_flags,
            };

            ms.read_inner::<4>(Some(crosses), None, 0, &flagged_fine_chans)
        });
    });
    ms_bench_group.finish();
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
criterion_group!(
    name = source_lists;
    config = Criterion::default().sample_size(100);
    targets = source_list_benchmarks,
);
criterion_group!(
    name = io;
    config = Criterion::default().sample_size(10);
    targets = io_benchmarks,
);
criterion_main!(model, calibrate, source_lists, io);

/// Copy files from source to destination recursively.
/// <https://nick.groenen.me/notes/recursively-copy-files-in-rust/>
fn copy_recursively(
    source: impl AsRef<Path>,
    destination: impl AsRef<Path>,
) -> std::io::Result<()> {
    std::fs::create_dir_all(&destination)?;
    for entry in std::fs::read_dir(source)? {
        let entry = entry?;
        let filetype = entry.file_type()?;
        if filetype.is_dir() {
            copy_recursively(entry.path(), destination.as_ref().join(entry.file_name()))?;
        } else {
            std::fs::copy(entry.path(), destination.as_ref().join(entry.file_name()))?;
        }
    }
    Ok(())
}
