// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use approx::assert_abs_diff_eq;
use crossbeam_channel::bounded;
use crossbeam_utils::{atomic::AtomicCell, thread};
use marlu::{Jones, LatLngHeight};
use ndarray::prelude::*;
use scopeguard::defer_on_unwind;
use serial_test::serial;
use tempfile::TempDir;
use vec1::{vec1, Vec1};

use super::*;
use crate::{
    averaging::timesteps_to_timeblocks,
    io::read::{MsReader, UvfitsReader, VisRead},
    math::TileBaselineFlags,
};

fn synthesize_test_data(
    shape: (usize, usize, usize),
) -> (ArcArray<Jones<f32>, Ix3>, ArcArray<f32, Ix3>) {
    let vis_data = ArcArray::<Jones<f32>, Ix3>::from_shape_fn(shape, |(t, c, b)| {
        let t = t as f32;
        let c = c as f32;
        let b = b as f32;
        Jones::from([
            t + b + c,
            t * 2.0 + b + c,
            t * 3.0 + b + c,
            t * 4.0 + b + c,
            t * 5.0 + b + c,
            t * 6.0 + b + c,
            t * 7.0 + b + c,
            t * 8.0 + b + c,
        ])
    });
    let vis_weights =
        ArcArray::<f32, Ix3>::from_shape_fn(shape, |(t, c, b)| (t + c + b + 1) as f32);

    (vis_data, vis_weights)
}

#[test]
#[serial]
fn test_vis_output_no_time_averaging_no_gaps() {
    let vis_time_average_factor = 1;
    let vis_freq_average_factor = 1;

    let num_timesteps = 5;
    let num_channels = 10;
    let ant_pairs = vec![(0, 1), (0, 2), (1, 2)];

    let obsid = 1090000000;
    let start_timestamp = Epoch::from_gpst_seconds(obsid as f64);

    let time_res = Duration::from_seconds(1.);
    let timesteps = vec1![0, 1, 2, 3, 4];
    let timestamps = Vec1::try_from_vec(
        (0..num_timesteps)
            .map(|i| start_timestamp + time_res * i as f64)
            .collect(),
    )
    .unwrap();
    let timeblocks = timesteps_to_timeblocks(&timestamps, vis_time_average_factor, &timesteps);

    let freq_res = 10e3;
    let fine_chan_freqs = Vec1::try_from_vec(
        (0..num_channels)
            .map(|i| 150e6 + freq_res * i as f64)
            .collect(),
    )
    .unwrap();

    let vis_ctx = VisContext {
        num_sel_timesteps: timesteps.len(),
        start_timestamp,
        int_time: time_res,
        num_sel_chans: num_channels,
        start_freq_hz: 128_000_000.,
        freq_resolution_hz: freq_res,
        sel_baselines: ant_pairs.clone(),
        avg_time: 1,
        avg_freq: 1,
        num_vis_pols: 4,
    };
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir");
    let out_vis_paths = vec1![
        (tmp_dir.path().join("vis.uvfits"), VisOutputType::Uvfits),
        (tmp_dir.path().join("vis.ms"), VisOutputType::MeasurementSet)
    ];

    let array_pos = LatLngHeight::mwa();
    let phase_centre = RADec::from_degrees(0., -27.);
    #[rustfmt::skip]
    let tile_xyzs = [
        XyzGeodetic { x: 0., y: 0., z: 0., },
        XyzGeodetic { x: 1., y: 0., z: 0., },
        XyzGeodetic { x: 0., y: 1., z: 0., },
    ];
    let tile_names = ["tile_0_0".into(), "tile_1_0".into(), "tile_0_1".into()];

    let shape = (timesteps.len(), num_channels, ant_pairs.len());
    let (vis_data, vis_weights) = synthesize_test_data(shape);
    let tile_baseline_flags = TileBaselineFlags::new(3, HashSet::new());

    let (tx, rx) = bounded(1);
    let error = AtomicCell::new(false);
    let scoped_threads_result = thread::scope(|scope| {
        // Input visibility-generating thread.
        let data_handle = scope.spawn(|_| {
            for (i_timestep, &timestep) in timesteps.iter().enumerate() {
                let timestamp = timestamps[timestep];
                match tx.send(VisTimestep {
                    cross_data_fb: vis_data.slice(s![i_timestep, .., ..]).to_shared(),
                    cross_weights_fb: vis_weights.slice(s![i_timestep, .., ..]).to_shared(),
                    autos: None,
                    timestamp,
                }) {
                    Ok(()) => (),
                    // If we can't send the message, it's because the channel
                    // has been closed on the other side. That should only
                    // happen because the writer has exited due to error; in
                    // that case, just exit this thread.
                    Err(_) => return Ok(()),
                }
            }

            Ok(())
        });

        // Vis writing thread.
        let write_handle = scope.spawn(|_| {
            defer_on_unwind! { error.store(true); }

            let marlu_mwa_obs_context = None;
            let result = write_vis(
                &out_vis_paths,
                array_pos,
                phase_centre,
                None,
                &tile_xyzs,
                &tile_names,
                Some(obsid),
                &timestamps,
                &timesteps,
                &timeblocks,
                time_res,
                Duration::from_seconds(0.0),
                freq_res,
                &fine_chan_freqs,
                &ant_pairs,
                &HashSet::new(),
                vis_time_average_factor,
                vis_freq_average_factor,
                marlu_mwa_obs_context,
                rx,
                &error,
                None,
            );
            if result.is_err() {
                error.store(true);
            }
            result
        });

        let result: Result<Result<(), VisWriteError>, _> = data_handle.join();
        let result = match result {
            Err(_) | Ok(Err(_)) => result.map(|_| Ok(String::new())),
            Ok(Ok(())) => write_handle.join(),
        };
        result
    });

    match scoped_threads_result {
        Ok(Ok(r)) => r.unwrap(),
        Err(_) | Ok(Err(_)) => panic!("A panic occurred in the async threads"),
    };

    // Read the visibilities in and check everything is fine.
    for (path, vis_type) in out_vis_paths {
        let reader: Box<dyn VisRead> = match vis_type {
            VisOutputType::Uvfits => {
                Box::new(UvfitsReader::new::<&Path, &Path>(&path, None).unwrap())
            }
            VisOutputType::MeasurementSet => {
                Box::new(MsReader::new::<&Path, &Path>(&path, None, None).unwrap())
            }
        };
        let obs_context = reader.get_obs_context();
        assert_eq!(&obs_context.all_timesteps, &timesteps);

        let expected = vec1![
            start_timestamp,
            start_timestamp + time_res,
            start_timestamp + time_res * 2.0,
            start_timestamp + time_res * 3.0,
            start_timestamp + time_res * 4.0,
        ];
        assert_eq!(
            obs_context.timestamps,
            expected,
            "\ngot (GPS): {:?}\nexpected:  {:?}",
            obs_context.timestamps.mapped_ref(|t| t.to_gpst_seconds()),
            expected.mapped_ref(|t| t.to_gpst_seconds())
        );

        assert_eq!(obs_context.time_res, Some(time_res));
        assert_eq!(obs_context.freq_res, Some(freq_res));

        let avg_shape = (
            obs_context.fine_chan_freqs.len(),
            vis_ctx.sel_baselines.len(),
        );
        let mut avg_data = Array2::zeros(avg_shape);
        let mut avg_weights = Array2::zeros(avg_shape);
        let flagged_fine_chans: HashSet<usize> =
            obs_context.flagged_fine_chans.iter().cloned().collect();

        for i_timestep in 0..timesteps.len() {
            reader
                .read_crosses(
                    avg_data.view_mut(),
                    avg_weights.view_mut(),
                    i_timestep,
                    &tile_baseline_flags,
                    &flagged_fine_chans,
                )
                .unwrap();

            assert_abs_diff_eq!(vis_data.slice(s![i_timestep, .., ..]), avg_data);
            assert_abs_diff_eq!(
                vis_weights.slice(s![i_timestep, .., ..]),
                avg_weights.view()
            );
        }
    }
}

#[test]
#[serial]
fn test_vis_output_no_time_averaging_with_gaps() {
    let vis_time_average_factor = 1;
    let vis_freq_average_factor = 1;

    let num_timestamps = 10;
    let num_channels = 10;
    let ant_pairs = vec![(0, 1), (0, 2), (1, 2)];

    let obsid = 1090000000;
    let start_timestamp = Epoch::from_gpst_seconds(obsid as f64);

    let time_res = Duration::from_seconds(1.);
    let timesteps = vec1![1, 3, 9];
    let timestamps = Vec1::try_from_vec(
        (0..num_timestamps)
            .map(|i| start_timestamp + time_res * i as f64)
            .collect(),
    )
    .unwrap();
    let timeblocks = timesteps_to_timeblocks(&timestamps, vis_time_average_factor, &timesteps);

    let freq_res = 10e3;
    let fine_chan_freqs = Vec1::try_from_vec(
        (0..num_channels)
            .map(|i| 150e6 + freq_res * i as f64)
            .collect(),
    )
    .unwrap();

    let vis_ctx = VisContext {
        num_sel_timesteps: timesteps.len(),
        start_timestamp,
        int_time: time_res,
        num_sel_chans: num_channels,
        start_freq_hz: 128_000_000.,
        freq_resolution_hz: freq_res,
        sel_baselines: ant_pairs.clone(),
        avg_time: 1,
        avg_freq: 1,
        num_vis_pols: 4,
    };
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir");
    let out_vis_paths = vec1![
        (tmp_dir.path().join("vis.uvfits"), VisOutputType::Uvfits),
        (tmp_dir.path().join("vis.ms"), VisOutputType::MeasurementSet)
    ];

    let array_pos = LatLngHeight::mwa();
    let phase_centre = RADec::from_degrees(0., -27.);
    #[rustfmt::skip]
    let tile_xyzs = [
        XyzGeodetic { x: 0., y: 0., z: 0., },
        XyzGeodetic { x: 1., y: 0., z: 0., },
        XyzGeodetic { x: 0., y: 1., z: 0., },
    ];
    let tile_names = ["tile_0_0".into(), "tile_1_0".into(), "tile_0_1".into()];

    let shape = (timesteps.len(), num_channels, ant_pairs.len());
    let (vis_data, vis_weights) = synthesize_test_data(shape);
    let tile_baseline_flags = TileBaselineFlags::new(3, HashSet::new());

    let (tx, rx) = bounded(1);
    let error = AtomicCell::new(false);
    let scoped_threads_result = thread::scope(|scope| {
        // Input visibility-generating thread.
        let data_handle = scope.spawn(|_| {
            for (i_timestep, &timestep) in timesteps.iter().enumerate() {
                let timestamp = timestamps[timestep];
                match tx.send(VisTimestep {
                    cross_data_fb: vis_data.slice(s![i_timestep, .., ..]).to_shared(),
                    cross_weights_fb: vis_weights.slice(s![i_timestep, .., ..]).to_shared(),
                    autos: None,
                    timestamp,
                }) {
                    Ok(()) => (),
                    // If we can't send the message, it's because the channel
                    // has been closed on the other side. That should only
                    // happen because the writer has exited due to error; in
                    // that case, just exit this thread.
                    Err(_) => return Ok(()),
                }
            }

            Ok(())
        });

        // Vis writing thread.
        let write_handle = scope.spawn(|_| {
            defer_on_unwind! { error.store(true); }

            let marlu_mwa_obs_context = None;
            let result = write_vis(
                &out_vis_paths,
                array_pos,
                phase_centre,
                None,
                &tile_xyzs,
                &tile_names,
                Some(obsid),
                &timestamps,
                &timesteps,
                &timeblocks,
                time_res,
                Duration::from_seconds(0.0),
                freq_res,
                &fine_chan_freqs,
                &ant_pairs,
                &HashSet::new(),
                vis_time_average_factor,
                vis_freq_average_factor,
                marlu_mwa_obs_context,
                rx,
                &error,
                None,
            );
            if result.is_err() {
                error.store(true);
            }
            result
        });

        let result: Result<Result<(), VisWriteError>, _> = data_handle.join();
        let result = match result {
            Err(_) | Ok(Err(_)) => result.map(|_| Ok(String::new())),
            Ok(Ok(())) => write_handle.join(),
        };
        result
    });

    match scoped_threads_result {
        Ok(Ok(r)) => r.unwrap(),
        Err(_) | Ok(Err(_)) => panic!("A panic occurred in the async threads"),
    };

    // Read the visibilities in and check everything is fine.
    // New timesteps for the gapped data.
    let timesteps = [0, 1, 2];
    for (path, vis_type) in out_vis_paths {
        let reader: Box<dyn VisRead> = match vis_type {
            VisOutputType::Uvfits => {
                Box::new(UvfitsReader::new::<&Path, &Path>(&path, None).unwrap())
            }
            VisOutputType::MeasurementSet => {
                Box::new(MsReader::new::<&Path, &Path>(&path, None, None).unwrap())
            }
        };
        let obs_context = reader.get_obs_context();
        assert_eq!(&obs_context.all_timesteps, &timesteps);

        let expected = vec1![
            Epoch::from_gpst_seconds((obsid + 1) as f64),
            Epoch::from_gpst_seconds((obsid + 3) as f64),
            Epoch::from_gpst_seconds((obsid + 9) as f64),
        ];
        assert_eq!(
            obs_context.timestamps,
            expected,
            "\ngot (GPS): {:?}\nexpected:  {:?}",
            obs_context.timestamps.mapped_ref(|t| t.to_gpst_seconds()),
            expected.mapped_ref(|t| t.to_gpst_seconds())
        );

        // Without the metafits file, the uvfits reader guesses the time
        // resolution from the data. Seeing as the smallest gap is 2s, the time
        // resolution will be reported as 2s, but it should be 1s. This will be
        // fixed in the next version of Marlu.
        assert_ne!(obs_context.time_res, Some(time_res));
        assert_eq!(obs_context.time_res, Some(Duration::from_seconds(2.0)));
        assert_eq!(obs_context.freq_res, Some(freq_res));

        let avg_shape = (
            obs_context.fine_chan_freqs.len(),
            vis_ctx.sel_baselines.len(),
        );
        let mut avg_data = Array2::zeros(avg_shape);
        let mut avg_weights = Array2::zeros(avg_shape);
        let flagged_fine_chans: HashSet<usize> =
            obs_context.flagged_fine_chans.iter().cloned().collect();

        for i_timestep in 0..timesteps.len() {
            reader
                .read_crosses(
                    avg_data.view_mut(),
                    avg_weights.view_mut(),
                    i_timestep,
                    &tile_baseline_flags,
                    &flagged_fine_chans,
                )
                .unwrap();

            assert_abs_diff_eq!(vis_data.slice(s![i_timestep, .., ..]), avg_data);
            assert_abs_diff_eq!(
                vis_weights.slice(s![i_timestep, .., ..]),
                avg_weights.view()
            );
        }
    }
}

#[test]
#[serial]
fn test_vis_output_time_averaging() {
    let vis_time_average_factor = 3;
    let vis_freq_average_factor = 1;

    let num_timestamps = 10;
    let num_channels = 10;
    let ant_pairs = vec![(0, 1), (0, 2), (1, 2)];

    let obsid = 1090000000;
    let start_timestamp = Epoch::from_gpst_seconds(obsid as f64);

    let time_res = Duration::from_seconds(1.);
    // we start at timestep index 1, with averaging 3. Averaged timesteps look like this:
    // [[1, _, 3], [_, _, _], [_, _, 9]]
    let timesteps = vec1![1, 3, 9];
    let timestamps = Vec1::try_from_vec(
        (0..num_timestamps)
            .map(|i| start_timestamp + time_res * i as f64)
            .collect(),
    )
    .unwrap();
    let timeblocks = timesteps_to_timeblocks(&timestamps, vis_time_average_factor, &timesteps);

    let freq_res = 10e3;
    let fine_chan_freqs = Vec1::try_from_vec(
        (0..num_channels)
            .map(|i| 150e6 + freq_res * i as f64)
            .collect(),
    )
    .unwrap();

    let vis_ctx = VisContext {
        num_sel_timesteps: timesteps.len(),
        start_timestamp,
        int_time: time_res,
        num_sel_chans: num_channels,
        start_freq_hz: 128_000_000.,
        freq_resolution_hz: freq_res,
        sel_baselines: ant_pairs.clone(),
        avg_time: 1,
        avg_freq: 1,
        num_vis_pols: 4,
    };
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir");
    let out_vis_paths = vec1![
        (tmp_dir.path().join("vis.uvfits"), VisOutputType::Uvfits),
        (tmp_dir.path().join("vis.ms"), VisOutputType::MeasurementSet)
    ];

    let array_pos = LatLngHeight::mwa();
    let phase_centre = RADec::from_degrees(0., -27.);
    #[rustfmt::skip]
    let tile_xyzs = [
        XyzGeodetic { x: 0., y: 0., z: 0., },
        XyzGeodetic { x: 1., y: 0., z: 0., },
        XyzGeodetic { x: 0., y: 1., z: 0., },
    ];
    let tile_names = ["tile_0_0".into(), "tile_1_0".into(), "tile_0_1".into()];

    let shape = (timesteps.len(), num_channels, ant_pairs.len());
    let (vis_data, mut vis_weights) = synthesize_test_data(shape);
    let tile_baseline_flags = TileBaselineFlags::new(3, HashSet::new());
    // I'm keeping the weights simple because predicting the vis values is
    // hurting my head.
    vis_weights.fill(1.0);

    let (tx, rx) = bounded(1);
    let error = AtomicCell::new(false);
    let scoped_threads_result = thread::scope(|scope| {
        // Input visibility-generating thread.
        let data_handle = scope.spawn(|_| {
            for (i_timestep, &timestep) in timesteps.iter().enumerate() {
                let timestamp = timestamps[timestep];
                match tx.send(VisTimestep {
                    cross_data_fb: vis_data.slice(s![i_timestep, .., ..]).to_shared(),
                    cross_weights_fb: vis_weights.slice(s![i_timestep, .., ..]).to_shared(),
                    autos: None,
                    timestamp,
                }) {
                    Ok(()) => (),
                    // If we can't send the message, it's because the channel
                    // has been closed on the other side. That should only
                    // happen because the writer has exited due to error; in
                    // that case, just exit this thread.
                    Err(_) => return Ok(()),
                }
            }

            Ok(())
        });

        // Vis writing thread.
        let write_handle = scope.spawn(|_| {
            defer_on_unwind! { error.store(true); }

            let marlu_mwa_obs_context = None;
            let result = write_vis(
                &out_vis_paths,
                array_pos,
                phase_centre,
                None,
                &tile_xyzs,
                &tile_names,
                Some(obsid),
                &timestamps,
                &timesteps,
                &timeblocks,
                time_res,
                Duration::from_seconds(0.0),
                freq_res,
                &fine_chan_freqs,
                &ant_pairs,
                &HashSet::new(),
                vis_time_average_factor,
                vis_freq_average_factor,
                marlu_mwa_obs_context,
                rx,
                &error,
                None,
            );
            if result.is_err() {
                error.store(true);
            }
            result
        });

        let result: Result<Result<(), VisWriteError>, _> = data_handle.join();
        let result = match result {
            Err(_) | Ok(Err(_)) => result.map(|_| Ok(String::new())),
            Ok(Ok(())) => write_handle.join(),
        };
        result
    });

    match scoped_threads_result {
        Ok(Ok(r)) => r.unwrap(),
        Err(_) | Ok(Err(_)) => panic!("A panic occurred in the async threads"),
    };

    // Read the visibilities in and check everything is fine.
    // New timesteps for averaged data.
    let timesteps = [0, 1];
    for (path, vis_type) in out_vis_paths {
        let reader: Box<dyn VisRead> = match vis_type {
            VisOutputType::Uvfits => {
                Box::new(UvfitsReader::new::<&Path, &Path>(&path, None).unwrap())
            }
            VisOutputType::MeasurementSet => {
                Box::new(MsReader::new::<&Path, &Path>(&path, None, None).unwrap())
            }
        };
        let obs_context = reader.get_obs_context();
        assert_eq!(&obs_context.all_timesteps, &timesteps);

        let expected = vec1![
            Epoch::from_gpst_seconds((obsid + 2) as f64),
            Epoch::from_gpst_seconds((obsid + 8) as f64),
        ];
        assert_eq!(
            obs_context.timestamps,
            expected,
            "\ngot (GPS): {:?}\nexpected:  {:?}",
            obs_context.timestamps.mapped_ref(|t| t.to_gpst_seconds()),
            expected.mapped_ref(|t| t.to_gpst_seconds())
        );

        // Without the metafits file, the uvfits reader guesses the time
        // resolution from the data. Seeing as the smallest gap is 6s, the time
        // resolution will be reported as 6s, but it should be 3s. This will be
        // fixed in the next version of Marlu.
        assert_ne!(obs_context.time_res, Some(time_res));
        assert_eq!(obs_context.time_res, Some(Duration::from_seconds(6.0)));
        assert_eq!(obs_context.freq_res, Some(freq_res));

        let avg_shape = (
            obs_context.fine_chan_freqs.len(),
            vis_ctx.sel_baselines.len(),
        );
        let mut avg_data = Array2::zeros(avg_shape);
        let mut avg_weights = Array2::zeros(avg_shape);
        let flagged_fine_chans: HashSet<usize> =
            obs_context.flagged_fine_chans.iter().cloned().collect();

        for i_timestep in 0..timesteps.len() {
            reader
                .read_crosses(
                    avg_data.view_mut(),
                    avg_weights.view_mut(),
                    i_timestep,
                    &tile_baseline_flags,
                    &flagged_fine_chans,
                )
                .unwrap();

            match i_timestep {
                0 => {
                    let vis_data = Array2::from_shape_fn(
                        (vis_ctx.num_sel_chans, vis_ctx.sel_baselines.len()),
                        |(b, c)| {
                            // This function similar to that within
                            // `synthesize_test_data`, but modified to match
                            // what we expect is the averaged data for this
                            // timestep.
                            let t = 1.0;
                            let b = (2 * b) as f32;
                            let c = (2 * c) as f32;
                            Jones::from([
                                t + b + c,
                                t * 2.0 + b + c,
                                t * 3.0 + b + c,
                                t * 4.0 + b + c,
                                t * 5.0 + b + c,
                                t * 6.0 + b + c,
                                t * 7.0 + b + c,
                                t * 8.0 + b + c,
                            ])
                        },
                    ) / 2.0;

                    assert_abs_diff_eq!(vis_data, avg_data);
                    assert_abs_diff_eq!(avg_weights, Array2::ones(avg_weights.dim()) * 2.0);
                }
                1 => {
                    let vis_data = Array2::from_shape_fn(
                        (vis_ctx.num_sel_chans, vis_ctx.sel_baselines.len()),
                        |(b, c)| {
                            let t = 2.0;
                            let b = b as f32;
                            let c = c as f32;
                            Jones::from([
                                t + b + c,
                                t * 2.0 + b + c,
                                t * 3.0 + b + c,
                                t * 4.0 + b + c,
                                t * 5.0 + b + c,
                                t * 6.0 + b + c,
                                t * 7.0 + b + c,
                                t * 8.0 + b + c,
                            ])
                        },
                    );

                    assert_abs_diff_eq!(vis_data, avg_data);
                    assert_abs_diff_eq!(avg_weights, Array2::ones(avg_weights.dim()));
                }
                _ => unreachable!(),
            }
        }
    }
}
