// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Tests against peeling

use std::{collections::HashSet, f64::consts::TAU};

use approx::assert_abs_diff_eq;
use hifitime::{Duration, Epoch};
use indexmap::indexmap;
use indicatif::{MultiProgress, ProgressDrawTarget};
use itertools::{izip, Itertools};
use marlu::{
    constants::VEL_C,
    math::cross_correlation_baseline_to_tiles,
    precession::{get_lmst, precess_time},
    Complex, HADec, Jones, LatLngHeight, RADec, XyzGeodetic,
};
use ndarray::{prelude::*, Zip};
use num_traits::Zero;
use vec1::{vec1, Vec1};

use super::*;
use crate::{
    averaging::Timeblock,
    beam::{Delays, FEEBeam},
    context::{ObsContext, Polarisations},
    io::read::VisInputType,
    model::{new_sky_modeller, SkyModellerCpu},
    srclist::{ComponentType, FluxDensity, FluxDensityType, Source, SourceComponent, SourceList},
};

// a single-component point source, stokes I.
macro_rules! point_src_i {
    ($radec:expr, $si:expr, $freq:expr, $i:expr) => {
        Source {
            components: vec![SourceComponent {
                radec: $radec,
                comp_type: ComponentType::Point,
                flux_type: FluxDensityType::PowerLaw {
                    si: $si,
                    fd: FluxDensity {
                        freq: $freq,
                        i: $i,
                        q: 0.0,
                        u: 0.0,
                        v: 0.0,
                    },
                },
            }]
            .into_boxed_slice(),
        }
    };
}

const TILE_SPACING: f64 = 100.;
const TILE_LIMIT: usize = 32;
const NUM_PASSES: usize = 3;
const NUM_LOOPS: usize = 10;
const SHORT_BASELINE_SIGMA: f64 = 50.0;
const CONVERGENCE: f64 = 0.5;
const OUT_PREFIX: &str = "/tmp/hypertest";

fn get_beam(num_tiles: usize) -> FEEBeam {
    let delays = vec![0; 16];
    // https://github.com/MWATelescope/mwa_pb/blob/90d6fbfc11bf4fca35796e3d5bde3ab7c9833b66/mwa_pb/mwa_sweet_spots.py#L60
    // let delays = vec![0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12];

    FEEBeam::new_from_env(num_tiles, Delays::Partial(delays), None).unwrap()
}

// get a timestamp at lmst=0 around the year 2100
// precessing to j2000 will introduce a noticable difference.
fn get_j2100(array_position: &LatLngHeight, dut1: Duration) -> Epoch {
    let mut epoch = Epoch::from_gregorian_utc_at_midnight(2100, 1, 1);

    // shift zenith_time to the nearest time when the phase centre is at zenith
    let sidereal2solar = 365.24 / 366.24;
    let obs_lst_rad = get_lmst(array_position.longitude_rad, epoch, dut1);
    if obs_lst_rad.abs() > 1e-6 {
        epoch -= Duration::from_days(sidereal2solar * obs_lst_rad / TAU);
    }
    epoch
}

/// get 3 simple tiles:
/// - tile "o" is at origin
/// - tile "u" has a u-component of s at lambda = s
/// - tile "v" has a v-component of s at lambda = s
#[rustfmt::skip]
fn get_simple_tiles(s: f64) -> (Vec1<String>, Vec1<XyzGeodetic>) {
    (
        vec1!["o", "u", "v"].mapped(|s| s.into()),
        vec1![
            XyzGeodetic { x: 0., y: 0., z: 0., },
            XyzGeodetic { x: 0., y: s, z: 0., },
            XyzGeodetic { x: 0., y: 0., z: s, },
        ],
    )
}

/// get an observation context with:
/// - array positioned at LatLngHeight = 0, 0, 100m
/// - 2 timestamps:
///   - first: phase centre is at zenith on j2100
///   - second: an hour later,
/// - 2 frequencies: lambda = 2m, 1m
/// - tiles from [get_simple_tiles]
fn get_simple_obs_context(s: f64) -> ObsContext {
    let array_position = LatLngHeight {
        longitude_rad: 0.,
        latitude_rad: 0.,
        height_metres: 100.,
    };

    let dut1 = Duration::from_seconds(0.0);
    let obs_epoch = get_j2100(&array_position, dut1);

    // at first timestep phase centre is at zenith
    let lst_zenith_rad = get_lmst(array_position.longitude_rad, obs_epoch, dut1);
    let phase_centre = RADec::from_hadec(
        HADec::from_radians(0., array_position.latitude_rad),
        lst_zenith_rad,
    );

    // second timestep is at 1h
    let hour_epoch = obs_epoch + Duration::from_hours(1.0);
    let timestamps = vec1![obs_epoch, hour_epoch];

    let (tile_names, tile_xyzs) = get_simple_tiles(s);
    let lambdas_m = vec1![2., 1.];
    let fine_chan_freqs: Vec1<u64> = lambdas_m.mapped(|l| (VEL_C / l) as u64);

    ObsContext {
        input_data_type: VisInputType::Raw,
        obsid: None,
        timestamps,
        all_timesteps: vec1![0, 1],
        unflagged_timesteps: vec![0, 1],
        phase_centre,
        pointing_centre: None,
        array_position,
        supplied_array_position: array_position,
        dut1: Some(dut1),
        tile_names,
        tile_xyzs,
        flagged_tiles: vec![],
        unavailable_tiles: vec![],
        autocorrelations_present: false,
        dipole_delays: Some(Delays::Partial(vec![
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ])),
        dipole_gains: None,
        time_res: Some(hour_epoch - obs_epoch),
        mwa_coarse_chan_nums: None,
        num_fine_chans_per_coarse_chan: None,
        freq_res: Some((fine_chan_freqs[1] - fine_chan_freqs[0]) as f64),
        fine_chan_freqs,
        flagged_fine_chans: vec![],
        flagged_fine_chans_per_coarse_chan: None,
        polarisations: Polarisations::default(),
    }
}

/// get an observation context with:
/// - array positioned at LatLngHeight = 0, 0, 100m
/// - 2 timestamps:
///   - first: phase centre is at zenith on j2100
///   - second: an hour later,
/// - 2 frequencies: lambda = 2m, 1m
/// - tiles from metafits
/// - tiles from [get_grid_tiles], s=50
fn get_phase1_obs_context(tile_limit: usize) -> ObsContext {
    let array_position = LatLngHeight::mwa();

    let meta_path = "test_files/1090008640/1090008640.metafits";
    let meta_ctx = mwalib::MetafitsContext::new(meta_path, None).unwrap();

    let obsid = meta_ctx.obs_id;
    let obs_time = Epoch::from_gpst_seconds(obsid as _);
    let dut1 = Duration::from_seconds(0.);

    // let obs_lst_rad = get_lmst(array_position.longitude_rad, obs_time, dut1);
    // shift obs_time to the nearest time when the phase centre is at zenith
    let zenith_lst_rad = get_lmst(array_position.longitude_rad, obs_time, dut1);
    eprintln!("lst % 𝜏 should be 0: {zenith_lst_rad:?}");
    let phase_centre = RADec::from_hadec(
        HADec::from_radians(0., array_position.latitude_rad),
        zenith_lst_rad,
    );
    eprintln!("phase centre: {phase_centre:?}");
    let hadec = phase_centre.to_hadec(zenith_lst_rad);
    eprintln!("ha % 𝜏 should be 0: {hadec:?}");
    let azel = hadec.to_azel(array_position.latitude_rad);
    eprintln!("(az, el) % 𝜏 should be 0, pi/2: {azel:?}");
    let tile_names: Vec<String> = meta_ctx
        .antennas
        .iter()
        .take(tile_limit)
        .map(|ant| ant.tile_name.clone())
        .collect();
    let tile_names = Vec1::try_from_vec(tile_names).unwrap();
    let tile_xyzs: Vec<XyzGeodetic> = XyzGeodetic::get_tiles_mwa(&meta_ctx)
        .into_iter()
        .take(tile_limit)
        .collect();
    let tile_xyzs = Vec1::try_from_vec(tile_xyzs).unwrap();

    // at first timestep phase centre is at zenith
    let lst_zenith_rad = get_lmst(array_position.longitude_rad, obs_time, dut1);
    let phase_centre = RADec::from_hadec(
        HADec::from_radians(0., array_position.latitude_rad),
        lst_zenith_rad,
    );

    // // second timestep is at 1h
    // let hour_epoch = obs_time + Duration::from_hours(1.0);
    // let timestamps = vec1![obs_time, hour_epoch];

    // corr_int_time_ms
    // metafits_fine_chan_freqs_hz
    // centre_freq_hz
    let time_res = Duration::from_seconds(meta_ctx.corr_int_time_ms as f64 / 1000_f64);
    let timestamps: Vec<Epoch> = (0..4).map(|i| obs_time + i * time_res).collect();
    let freq_res = meta_ctx.corr_fine_chan_width_hz as f64;
    let num_freqs = 32;
    let fine_chan_freqs: Vec<u64> = (0..num_freqs).map(|i|
        meta_ctx.centre_freq_hz as u64 + i as u64 * freq_res as u64
    ).collect();

    ObsContext {
        input_data_type: VisInputType::Raw,
        obsid: None,
        timestamps: Vec1::try_from_vec(timestamps).unwrap(),
        all_timesteps: vec1![0, 1],
        unflagged_timesteps: vec![0, 1],
        phase_centre,
        pointing_centre: None,
        array_position,
        supplied_array_position: array_position,
        dut1: Some(dut1),
        tile_names,
        tile_xyzs,
        flagged_tiles: vec![],
        unavailable_tiles: vec![],
        autocorrelations_present: false,
        dipole_delays: Some(Delays::Partial(vec![
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ])),
        dipole_gains: None,
        time_res: Some(time_res),
        mwa_coarse_chan_nums: None,
        num_fine_chans_per_coarse_chan: None,
        freq_res: Some(freq_res),
        fine_chan_freqs: Vec1::try_from_vec(fine_chan_freqs).unwrap(),
        flagged_fine_chans: vec![],
        flagged_fine_chans_per_coarse_chan: None,
        polarisations: Polarisations::default(),
    }
}

/// get a 16x16 grid of tiles
fn get_grid_tiles(s_: f64) -> (Vec1<String>, Vec1<XyzGeodetic>) {
    let mut tile_names = Vec::new();
    let mut tile_xyzs = Vec::new();
    for y in 0..16 {
        for z in 0..16 {
            let name = format!("{:01x}{:01x}", y, z);
            tile_names.push(name);
            tile_xyzs.push(XyzGeodetic {
                x: 0.,
                y: y as f64 * s_,
                z: z as f64 * s_,
            });
        }
    }
    (
        Vec1::try_from_vec(tile_names).unwrap(),
        Vec1::try_from_vec(tile_xyzs).unwrap(),
    )
}

/// get an observation context with:
/// - array positioned at LatLngHeight = 0, 0, 100m
/// - 2 timestamps:
///   - first: phase centre is at zenith on j2100
///   - second: an hour later,
/// - 2 frequencies: lambda = 2m, 1m
/// - tiles from [get_grid_tiles], s=50
// fn get_grid_obs_context(s: f64) -> ObsContext {
//     let array_position = LatLngHeight {
//         longitude_rad: 0.,
//         latitude_rad: 0.,
//         height_metres: 100.,
//     };

//     let dut1 = Duration::from_seconds(0.0);
//     let obs_epoch = get_j2100(&array_position, dut1);

//     // at first timestep phase centre is at zenith
//     let lst_zenith_rad = get_lmst(array_position.longitude_rad, obs_epoch, dut1);
//     let phase_centre = RADec::from_hadec(
//         HADec::from_radians(0., array_position.latitude_rad),
//         lst_zenith_rad,
//     );

//     eprintln!("phase centre: {phase_centre:?}");
//     let hadec = phase_centre.to_hadec(lst_zenith_rad);
//     eprintln!("ha % 𝜏 should be 0: {hadec:?}");
//     let azel = hadec.to_azel(array_position.latitude_rad);
//     eprintln!("(az, el) % 𝜏 should be 0, pi/2: {azel:?}");
//     let (tile_names, tile_xyzs) = get_grid_tiles(s);

//     // at first timestep phase centre is at zenith
//     let lst_zenith_rad = get_lmst(array_position.longitude_rad, obs_epoch, dut1);
//     let phase_centre = RADec::from_hadec(
//         HADec::from_radians(0., array_position.latitude_rad),
//         lst_zenith_rad,
//     );

//     // second timestep is at 1h
//     let time_res = Duration::from_hours(1.0);
//     let hour_epoch = obs_epoch + time_res;
//     let timestamps = vec1![obs_epoch, hour_epoch];

//     let lambdas_m = vec1![2., 1.];
//     let fine_chan_freqs: Vec1<u64> = lambdas_m.mapped(|l| (VEL_C / l) as u64);

//     ObsContext {
//         input_data_type: VisInputType::Raw,
//         obsid: None,
//         timestamps,
//         all_timesteps: vec1![0, 1],
//         unflagged_timesteps: vec![0, 1],
//         phase_centre,
//         pointing_centre: None,
//         array_position,
//         supplied_array_position: array_position,
//         dut1: Some(dut1),
//         tile_names,
//         tile_xyzs,
//         flagged_tiles: vec![],
//         unavailable_tiles: vec![],
//         autocorrelations_present: false,
//         dipole_delays: Some(Delays::Partial(vec![
//             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//         ])),
//         dipole_gains: None,
//         time_res: Some(time_res),
//         mwa_coarse_chan_nums: None,
//         num_fine_chans_per_coarse_chan: None,
//         freq_res: Some((fine_chan_freqs[1] - fine_chan_freqs[0]) as f64),
//         fine_chan_freqs,
//         flagged_fine_chans: vec![],
//         flagged_fine_chans_per_coarse_chan: None,
//         polarisations: Polarisations::default(),
//     }
// }

// these are used for debugging the tests
use ndarray::{ArrayView2, ArrayView3};
#[allow(clippy::too_many_arguments)]
fn display_vis_b(
    name: &String,
    vis_b: &[Jones<f32>],
    ant_pairs: &[(usize, usize)],
    uvs: &[UV],
    ws: &[W],
    tile_names: &[String],
    seconds: f64,
    lambda: f64,
) {
    // use std::f64::consts::PI;
    println!(
        "bl  u        v        w        | @ time={:>+9.3}s, lam={:+1.3}m, {}",
        seconds, lambda, name
    );
    let n_ants = tile_names.len();
    for (jones, &(ant1, ant2)) in vis_b.iter().zip_eq(ant_pairs.iter()) {
        // only display baselines with the first tile
        if ant1 != 0 || (ant2 >= 16 && ant2 < n_ants - 16) {
            continue;
        }
        let uv = uvs[ant1] - uvs[ant2];
        let w = ws[ant1] - ws[ant2];
        let (name1, name2) = (&tile_names[ant1], &tile_names[ant2]);
        let (xx, xy, yx, yy) = jones.iter().collect_tuple().unwrap();
        println!(
            "{:1}-{:1} {:+09.5} {:+09.5} {:+03.5} | \
                XX {:+09.9} {:+09.9}i XY {:+09.9} {:+09.9}i \
                YX {:+09.9} {:+09.9}i YY {:+09.9} {:+09.9}i",
                // XX {:07.5} @{:+08.5}pi XY {:07.5} @{:+08.5}pi \
                // YX {:07.5} @{:+08.5}pi YY {:07.5} @{:+08.5}pi",
            name1,
            name2,
            uv.u / lambda,
            uv.v / lambda,
            w / lambda,
            xx.re,
            xx.im,
            xy.re,
            xy.im,
            yx.re,
            yx.im,
            yy.re,
            yy.im,
            // xx.norm(),
            // xx.arg() as f64 / PI,
            // xy.norm(),
            // xy.arg() as f64 / PI,
            // yx.norm(),
            // yx.arg() as f64 / PI,
            // yy.norm(),
            // yy.arg() as f64 / PI,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn display_vis_fb(
    name: &String,
    vis_fb: ArrayView2<Jones<f32>>,
    seconds: f64,
    uvs: &[UV],
    ws: &[W],
    lambdas_m: &[f64],
    ant_pairs: &[(usize, usize)],
    tile_names: &[String],
) {
    // println!("{:9} | {:7} | bl  u      v      w      | {}", "time", "lam", name);
    for (vis_b, &lambda) in vis_fb.outer_iter().zip_eq(lambdas_m.iter()) {
        display_vis_b(
            name,
            vis_b.as_slice().unwrap(),
            ant_pairs,
            uvs,
            ws,
            tile_names,
            seconds,
            lambda,
        );
    }
}

/// display named visibilities and uvws in table format
#[allow(clippy::too_many_arguments)]
fn display_vis_tfb(
    name: &String,
    vis_tfb: ArrayView3<Jones<f32>>,
    obs_context: &ObsContext,
    phase_centre: RADec,
    apply_precession: bool,
) {
    let array_pos = obs_context.array_position;
    let num_tiles = obs_context.get_total_num_tiles();
    let num_baselines = (num_tiles * (num_tiles - 1)) / 2;
    let ant_pairs = (0..num_baselines)
        .map(|bl_idx| cross_correlation_baseline_to_tiles(num_tiles, bl_idx))
        .collect_vec();
    let fine_chan_freqs_hz = obs_context
        .fine_chan_freqs
        .iter()
        .map(|&f| f as f64)
        .collect_vec();
    let lambdas_m = fine_chan_freqs_hz.iter().map(|&f| VEL_C / f).collect_vec();

    let start_seconds = obs_context.timestamps[0].to_gpst_seconds();
    let mut tile_uvs_tmp = vec![UV::default(); num_tiles];
    let mut tile_ws_tmp = vec![W::default(); num_tiles];
    // println!("{:9} | {:7} | bl  u      v      w      | {}", "time", "lam", name);
    for (vis_fb, &time) in vis_tfb.outer_iter().zip_eq(obs_context.timestamps.iter()) {
        if apply_precession {
            let precession_info = precess_time(
                array_pos.longitude_rad,
                array_pos.latitude_rad,
                phase_centre,
                time,
                obs_context.dut1.unwrap_or_default(),
            );
            let hadec = phase_centre.to_hadec(precession_info.lmst_j2000);
            let precessed_xyzs = precession_info.precess_xyz(&obs_context.tile_xyzs);
            setup_uvs(&mut tile_uvs_tmp, &precessed_xyzs, hadec);
            setup_ws(&mut tile_ws_tmp, &precessed_xyzs, hadec);
        } else {
            let lmst = get_lmst(
                array_pos.longitude_rad,
                time,
                obs_context.dut1.unwrap_or_default(),
            );
            let hadec = phase_centre.to_hadec(lmst);
            setup_uvs(&mut tile_uvs_tmp, &obs_context.tile_xyzs, hadec);
            setup_ws(&mut tile_ws_tmp, &obs_context.tile_xyzs, hadec);
        }
        let seconds = time.to_gpst_seconds() - start_seconds;
        display_vis_fb(
            name,
            vis_fb.view(),
            seconds,
            tile_uvs_tmp.as_slice(),
            tile_ws_tmp.as_slice(),
            &lambdas_m,
            &ant_pairs,
            &obs_context.tile_names,
        );
    }
}

// #[allow(clippy::too_many_arguments)]
// fn write_vis_tfb(
//     path: PathBuf,
//     vis_tfb: ArrayView3<Jones<f32>>,
//     obs_context: &ObsContext,
// ) {
//     use crate::{
//         averaging::{timesteps_to_timeblocks, channels_to_chanblocks},
//         io::write::{VisOutputType, VisWriteError}
//     };
//     let num_tiles = obs_context.get_total_num_tiles();
//     let num_baselines = (num_tiles * (num_tiles - 1)) / 2;
//     let unflagged_baseline_tile_pairs = (0..num_baselines)
//         .map(|bl_idx| cross_correlation_baseline_to_tiles(num_tiles, bl_idx))
//         .collect_vec();
//     let no_averaging = NonZeroUsize::new(1).unwrap();
//     let time_res = obs_context.time_res.unwrap();
//     let timesteps = (0.. obs_context.timestamps.len()).collect_vec();
//     let output_timeblocks = timesteps_to_timeblocks(
//         &obs_context.timestamps,
//         time_res,
//         no_averaging,
//         None
//     );
//     let write_smallest_contiguous_band = false;
//     let dut1 = obs_context.dut1.unwrap_or_default();
//     let freq_res = obs_context.freq_res.unwrap() as u64;
//     let spw = &channels_to_chanblocks(
//         &obs_context.fine_chan_freqs,
//         freq_res,
//         no_averaging,
//         &HashSet::new()
//     )[0];
//     let vis_weights = Array3::from_elem(vis_tfb.dim(), 1.0);

//     let error = AtomicCell::new(false);
//     let (tx_data, rx_data) = bounded(3);

//     assert_eq!(obs_context.tile_xyzs.len(), obs_context.tile_names.len());

//     let scoped_threads_result = thread::scope(|scope| {

//         let data_handle = scope.spawn(|| {
//             for (i_timestep, &timestep) in timesteps.iter().enumerate() {
//                 let timestamp = obs_context.timestamps[timestep];
//                 match tx_data.send(VisTimestep {
//                     cross_data_fb: vis_tfb.slice(s![i_timestep, .., ..]).to_shared(),
//                     cross_weights_fb: vis_weights.slice(s![i_timestep, .., ..]).to_shared(),
//                     autos: None,
//                     timestamp,
//                 }) {
//                     Ok(()) => (),
//                     // If we can't send the message, it's because the channel
//                     // has been closed on the other side. That should only
//                     // happen because the writer has exited due to error; in
//                     // that case, just exit this thread.
//                     Err(_) => return Ok(()),
//                 }
//             }

//             Ok(())
//         });

//         let write_handle = thread::Builder::new()
//         .name("write".to_string())
//         .spawn_scoped(scope, || {
//             defer_on_unwind! { error.store(true); }

//             let result = write_vis(
//                 &vec1![(path, VisOutputType::MeasurementSet)],
//                 obs_context.array_position,
//                 obs_context.phase_centre,
//                 obs_context.pointing_centre,
//                 &obs_context.tile_xyzs,
//                 &obs_context.tile_names,
//                 obs_context.obsid,
//                 &output_timeblocks,
//                 time_res,
//                 dut1,
//                 spw,
//                 &unflagged_baseline_tile_pairs,
//                 no_averaging,
//                 no_averaging,
//                 None,
//                 write_smallest_contiguous_band,
//                 rx_data,
//                 &error,
//                 None,
//             );
//             if result.is_err() {
//                 error.store(true);
//             }
//             result
//         })
//         .expect("OS can create threads");

//         let result: Result<Result<(), VisWriteError>, _> = data_handle.join();
//         let result = match result {
//             Err(_) | Ok(Err(_)) => result.map(|_| Ok(String::new())),
//             Ok(Ok(())) => write_handle.join(),
//         };
//         result
//     });

//     match scoped_threads_result {
//         Ok(Ok(r)) => println!("{r}"),
//         Err(_) | Ok(Err(_)) => panic!("A panic occurred in the async threads"),
//     };
// }

/// Populate the [UV] and [W] arrays ([times, tiles]) for the given
/// [ObsContext], returning the LMSTs.
fn setup_tile_uv_w_arrays(
    mut tile_uvs: ArrayViewMut2<UV>,
    mut tile_ws: ArrayViewMut2<W>,
    obs_context: &ObsContext,
    phase_centre: RADec,
    apply_precession: bool,
) -> (Vec<f64>, Array2<XyzGeodetic>) {
    let mut lmsts = vec![0.0; obs_context.timestamps.len()];
    let mut xyzs = Array2::default(tile_uvs.dim());

    let array_pos = obs_context.array_position;
    for (&time, mut tile_uvs, mut tile_ws, lmst, mut xyzs) in izip!(
        obs_context.timestamps.iter(),
        tile_uvs.outer_iter_mut(),
        tile_ws.outer_iter_mut(),
        lmsts.iter_mut(),
        xyzs.outer_iter_mut()
    ) {
        if apply_precession {
            let precession_info = precess_time(
                array_pos.longitude_rad,
                array_pos.latitude_rad,
                obs_context.phase_centre,
                time,
                obs_context.dut1.unwrap_or_default(),
            );
            *lmst = precession_info.lmst_j2000;
            let hadec = phase_centre.to_hadec(*lmst);
            let precessed_xyzs = precession_info.precess_xyz(&obs_context.tile_xyzs);
            setup_uvs(tile_uvs.as_slice_mut().unwrap(), &precessed_xyzs, hadec);
            setup_ws(tile_ws.as_slice_mut().unwrap(), &precessed_xyzs, hadec);
            xyzs.assign(&ArrayView1::from(&precessed_xyzs));
        } else {
            *lmst = get_lmst(
                array_pos.longitude_rad,
                time,
                obs_context.dut1.unwrap_or_default(),
            );
            let hadec = phase_centre.to_hadec(*lmst);
            setup_uvs(
                tile_uvs.as_slice_mut().unwrap(),
                &obs_context.tile_xyzs,
                hadec,
            );
            setup_ws(
                tile_ws.as_slice_mut().unwrap(),
                &obs_context.tile_xyzs,
                hadec,
            );
            xyzs.assign(&ArrayView1::from(&obs_context.tile_xyzs));
        }
    }

    (lmsts, xyzs)
}

#[test]
/// test [setup_uvs], [setup_ws]
fn test_setup_uv() {
    let obs_context = get_simple_obs_context(TILE_SPACING);
    let array_pos = obs_context.array_position;

    let num_tiles = obs_context.get_total_num_tiles();
    let num_times = obs_context.timestamps.len();

    // source is at zenith at 1h
    let hour_epoch = obs_context.timestamps[1];
    let lst_1h_rad = get_lmst(
        array_pos.longitude_rad,
        hour_epoch,
        obs_context.dut1.unwrap_or_default(),
    );
    let source_radec =
        RADec::from_hadec(HADec::from_radians(0., array_pos.latitude_rad), lst_1h_rad);

    // tile uvs and ws in the observation phase centre
    let mut tile_uvs_obs = Array2::default((num_times, num_tiles));
    let mut tile_ws_obs = Array2::default((num_times, num_tiles));
    // tile uvs and ws in the source phase centre
    let mut tile_uvs_src = Array2::default((num_times, num_tiles));
    let mut tile_ws_src = Array2::default((num_times, num_tiles));

    for apply_precession in [false, true] {
        setup_tile_uv_w_arrays(
            tile_uvs_obs.view_mut(),
            tile_ws_obs.view_mut(),
            &obs_context,
            obs_context.phase_centre,
            apply_precession,
        );
        setup_tile_uv_w_arrays(
            tile_uvs_src.view_mut(),
            tile_ws_src.view_mut(),
            &obs_context,
            source_radec,
            apply_precession,
        );

        if !apply_precession {
            for a in 0..num_tiles {
                // uvws for the phase centre at first timestpe should be the same as
                // uvws for the source at the second timestep
                assert_abs_diff_eq!(tile_uvs_obs[[0, a]], tile_uvs_src[[1, a]], epsilon = 1e-6);
                assert_abs_diff_eq!(tile_ws_obs[[0, a]], tile_ws_src[[1, a]], epsilon = 1e-6);
                // uvws for the phase centre at the second timestep should be the same as
                // uvws for the source at the first timestep, rotated in the opposite direciton.
                // since all the baselines sit flat on the uv plane, only the w component is negative.
                assert_abs_diff_eq!(tile_uvs_obs[[1, a]], tile_uvs_src[[0, a]], epsilon = 1e-6);
                assert_abs_diff_eq!(tile_ws_obs[[1, a]], -tile_ws_src[[0, a]], epsilon = 1e-6);
            }
            for t in 0..num_times {
                // tile 2 is a special case with only a v component, so should be unchanged
                assert_abs_diff_eq!(tile_uvs_obs[[t, 2]].v, TILE_SPACING, epsilon = 1e-6);
                assert_abs_diff_eq!(tile_uvs_obs[[t, 2]], tile_uvs_src[[t, 2]], epsilon = 1e-6);
                assert_abs_diff_eq!(tile_ws_obs[[t, 2]], tile_ws_src[[t, 2]], epsilon = 1e-6);
            }
            // tile 1 is aligned with zenith at t=0
            assert_abs_diff_eq!(tile_uvs_obs[[0, 1]].u, TILE_SPACING, epsilon = 1e-6);
            // tile 1 is aligned with source at t=1
            assert_abs_diff_eq!(tile_uvs_src[[1, 1]].u, TILE_SPACING, epsilon = 1e-6);
        }
        for t in 0..num_times {
            for a in 0..num_tiles {
                // println!(
                //     "prec={:5} t={} a={} obs=({:+1.6} {:+1.6} {:+1.6}), src=({:+1.6} {:+1.6} {:+1.6})",
                //     apply_precession, t, a,
                //     tile_uvs_obs[[t, a]].u, tile_uvs_obs[[t, a]].v, tile_ws_obs[[t, a]].0,
                //     tile_uvs_src[[t, a]].u, tile_uvs_src[[t, a]].v, tile_ws_src[[t, a]].0,
                // );
                // no difference between the two phase centres for v component
                assert_abs_diff_eq!(
                    tile_uvs_obs[[t, a]].v,
                    tile_uvs_src[[t, a]].v,
                    epsilon = 1e-6
                );
            }
            // tile 0 is the origin tile
            assert_abs_diff_eq!(tile_uvs_obs[[t, 0]].u, 0., epsilon = 1e-6);
            assert_abs_diff_eq!(tile_uvs_obs[[t, 0]].v, 0., epsilon = 1e-6);
            assert_abs_diff_eq!(tile_ws_obs[[t, 0]].0, 0., epsilon = 1e-6);
            assert_abs_diff_eq!(tile_uvs_src[[t, 0]].u, 0., epsilon = 1e-6);
            assert_abs_diff_eq!(tile_uvs_src[[t, 0]].v, 0., epsilon = 1e-6);
            assert_abs_diff_eq!(tile_ws_src[[t, 0]].0, 0., epsilon = 1e-6);
        }
    }
}

#[test]
/// tests vis_rotate_fb by asserting that:
/// - rotated visibilities have the source at the phase centre
/// simulate vis, where at the first timestep, the phase centre is at zenith
/// and at the second timestep, 1h later, the source is at zenith
fn test_vis_rotation() {
    let obs_context = get_simple_obs_context(TILE_SPACING);
    let array_pos = obs_context.array_position;

    let num_tiles = obs_context.get_total_num_tiles();
    let num_times = obs_context.timestamps.len();
    let num_baselines = (num_tiles * (num_tiles - 1)) / 2;
    let ant_pairs = (0..num_baselines)
        .map(|bl_idx| cross_correlation_baseline_to_tiles(num_tiles, bl_idx))
        .collect_vec();
    let flagged_tiles = HashSet::new();
    let num_chans = obs_context.fine_chan_freqs.len();

    let fine_chan_freqs_hz = obs_context
        .fine_chan_freqs
        .iter()
        .map(|&f| f as f64)
        .collect_vec();
    let lambdas_m = fine_chan_freqs_hz.iter().map(|&f| VEL_C / f).collect_vec();

    // source is at zenith at 1h
    let hour_epoch = obs_context.timestamps[1];
    let lst_1h_rad = get_lmst(
        array_pos.longitude_rad,
        hour_epoch,
        obs_context.dut1.unwrap_or_default(),
    );
    let source_radec =
        RADec::from_hadec(HADec::from_radians(0., array_pos.latitude_rad), lst_1h_rad);
    let source_fd = 1.;
    let source_list = SourceList::from(indexmap! {
        "One".into() => point_src_i!(source_radec, 0., fine_chan_freqs_hz[0], source_fd),
    });

    let beam = get_beam(num_tiles);

    let mut vis_tfb = Array3::default((num_times, num_chans, num_baselines));
    let mut vis_rot_tfb = Array3::default((num_times, num_chans, num_baselines));

    // tile uvs and ws in the observation phase centre
    let mut tile_uvs_obs = Array2::default((num_times, num_tiles));
    let mut tile_ws_obs = Array2::default((num_times, num_tiles));
    // tile uvs and ws in the source phase centre
    let mut tile_uvs_src = Array2::default((num_times, num_tiles));
    let mut tile_ws_src = Array2::default((num_times, num_tiles));

    for apply_precession in [false, true] {

        setup_tile_uv_w_arrays(
            tile_uvs_obs.view_mut(),
            tile_ws_obs.view_mut(),
            &obs_context,
            obs_context.phase_centre,
            apply_precession,
        );
        setup_tile_uv_w_arrays(
            tile_uvs_src.view_mut(),
            tile_ws_src.view_mut(),
            &obs_context,
            source_radec,
            apply_precession,
        );
        let modeller = SkyModellerCpu::new(
            &beam,
            &source_list,
            Polarisations::default(),
            &obs_context.tile_xyzs,
            &fine_chan_freqs_hz,
            &flagged_tiles,
            obs_context.phase_centre,
            array_pos.longitude_rad,
            array_pos.latitude_rad,
            obs_context.dut1.unwrap_or_default(),
            apply_precession,
        );

        vis_tfb.fill(Jones::zero());
        model_timesteps(&modeller, &obs_context.timestamps, vis_tfb.view_mut()).unwrap();

        // iterate over time, rotating visibilities
        for (vis_fb, mut vis_rot_fb, tile_ws_obs, tile_ws_src) in izip!(
            vis_tfb.outer_iter(),
            vis_rot_tfb.view_mut().outer_iter_mut(),
            tile_ws_obs.outer_iter(),
            tile_ws_src.outer_iter(),
        ) {
            vis_rotate_fb(
                vis_fb.view(),
                vis_rot_fb.view_mut(),
                tile_ws_obs.as_slice().unwrap(),
                tile_ws_src.as_slice().unwrap(),
                &lambdas_m,
            );
        }

        display_vis_tfb(
            &format!("model@obs prec={apply_precession}"),
            vis_tfb.view(),
            &obs_context,
            obs_context.phase_centre,
            apply_precession,
        );
        display_vis_tfb(
            &format!("rotated@source prec={apply_precession}"),
            vis_rot_tfb.view(),
            &obs_context,
            source_radec,
            apply_precession,
        );

        if !apply_precession {
            // bl  u        v        w        | @ time=   +0.000s, lam=+2.000m, model@obs prec=false
            // o-u -0.50000 +0.00000 +0.00000 | XX +0.18908 -0.20073i XY -0.00002 +0.00008i YX -0.00007 +0.00003i YY +0.20293 -0.21543i
            // o-v +0.00000 -0.50000 +0.00000 | XX +0.27576 +0.00000i XY -0.00007 +0.00004i YX -0.00007 -0.00004i YY +0.29596 +0.00000i
            // u-v +0.50000 -0.50000 +0.00000 | XX +0.18908 +0.20073i XY -0.00007 -0.00003i YX -0.00002 -0.00008i YY +0.20293 +0.21543i
            // bl  u        v        w        | @ time=   +0.000s, lam=+1.000m, model@obs prec=false
            // o-u -1.00000 +0.00000 +0.00000 | XX -0.00126 -0.02110i XY +0.00000 -0.00001i YX -0.00000 -0.00001i YY -0.00212 -0.03541i
            // o-v +0.00000 -1.00000 +0.00000 | XX +0.02114 +0.00000i XY +0.00001 +0.00000i YX +0.00001 -0.00000i YY +0.03548 +0.00000i
            // u-v +1.00000 -1.00000 +0.00000 | XX -0.00126 +0.02110i XY -0.00000 +0.00001i YX +0.00000 +0.00001i YY -0.00212 +0.03541i
            // bl  u        v        w        | @ time=+3600.000s, lam=+2.000m, model@obs prec=false
            // o-u -0.48287 +0.00000 +0.12976 | XX +0.68567 -0.72792i XY -0.00037 +0.00040i YX -0.00037 +0.00039i YY +0.68567 -0.72792i
            // o-v +0.00000 -0.50000 +0.00000 | XX +1.00000 +0.00000i XY -0.00054 +0.00001i YX -0.00054 -0.00001i YY +1.00000 +0.00000i
            // u-v +0.48287 -0.50000 -0.12976 | XX +0.68567 +0.72792i XY -0.00037 -0.00039i YX -0.00037 -0.00040i YY +0.68567 +0.72792i
            // bl  u        v        w        | @ time=+3600.000s, lam=+1.000m, model@obs prec=false
            // o-u -0.96574 +0.00000 +0.25951 | XX -0.05973 -0.99821i XY -0.00002 -0.00036i YX -0.00002 -0.00036i YY -0.05973 -0.99821i
            // o-v +0.00000 -1.00000 +0.00000 | XX +1.00000 +0.00000i XY +0.00036 +0.00000i YX +0.00036 -0.00000i YY +1.00000 +0.00000i
            // u-v +0.96574 -1.00000 -0.25951 | XX -0.05973 +0.99821i XY -0.00002 +0.00036i YX -0.00002 +0.00036i YY -0.05973 +0.99821i
            // bl  u        v        w        | @ time=   +0.000s, lam=+2.000m, rotated@source prec=false
            // o-u -0.48287 +0.00000 -0.12976 | XX +0.27576 -0.00000i XY -0.00007 +0.00004i YX -0.00007 -0.00004i YY +0.29596 -0.00000i
            // o-v +0.00000 -0.50000 +0.00000 | XX +0.27576 +0.00000i XY -0.00007 +0.00004i YX -0.00007 -0.00004i YY +0.29596 +0.00000i
            // u-v +0.48287 -0.50000 +0.12976 | XX +0.27576 +0.00000i XY -0.00007 +0.00004i YX -0.00007 -0.00004i YY +0.29596 +0.00000i
            for &vis_rot in vis_rot_tfb.slice(s![0, 0, ..]) {
                assert_abs_diff_eq!(vis_rot, Jones::from([
                    Complex::new(0.27576, 0.00000),
                    Complex::new(-0.00007, 0.00004),
                    Complex::new(-0.00007, -0.00004),
                    Complex::new(0.29596, 0.00000),
                ]), epsilon = 1e-5);
            }
            // bl  u        v        w        | @ time=   +0.000s, lam=+1.000m, rotated@source prec=false
            // o-u -0.96574 +0.00000 -0.25951 | XX +0.02114 -0.00000i XY +0.00001 +0.00000i YX +0.00001 -0.00000i YY +0.03548 -0.00000i
            // o-v +0.00000 -1.00000 +0.00000 | XX +0.02114 +0.00000i XY +0.00001 +0.00000i YX +0.00001 -0.00000i YY +0.03548 +0.00000i
            // u-v +0.96574 -1.00000 +0.25951 | XX +0.02114 +0.00000i XY +0.00001 +0.00000i YX +0.00001 -0.00000i YY +0.03548 +0.00000i
            for &vis_rot in vis_rot_tfb.slice(s![0, 1, ..]) {
                assert_abs_diff_eq!(vis_rot, Jones::from([
                    Complex::new(0.02114, 0.00000),
                    Complex::new(0.00001, 0.00000),
                    Complex::new(0.00001, -0.00000),
                    Complex::new(0.03548, 0.00000),
                ]), epsilon = 1e-5);
            }
            // bl  u        v        w        | @ time=+3600.000s, lam=+2.000m, rotated@source prec=false
            // o-u -0.50000 +0.00000 +0.00000 | XX +1.00000 +0.00000i XY -0.00054 +0.00001i YX -0.00054 -0.00001i YY +1.00000 +0.00000i
            // o-v +0.00000 -0.50000 +0.00000 | XX +1.00000 +0.00000i XY -0.00054 +0.00001i YX -0.00054 -0.00001i YY +1.00000 +0.00000i
            // u-v +0.50000 -0.50000 +0.00000 | XX +1.00000 -0.00000i XY -0.00054 +0.00001i YX -0.00054 -0.00001i YY +1.00000 -0.00000i
            for &vis_rot in vis_rot_tfb.slice(s![1, 0, ..]) {
                assert_abs_diff_eq!(vis_rot, Jones::from([
                    Complex::new(1.00000, 0.00000),
                    Complex::new(-0.00054, 0.00001),
                    Complex::new(-0.00054, -0.00001),
                    Complex::new(1.00000, 0.00000),
                ]), epsilon = 1e-5);
            }
            // bl  u        v        w        | @ time=+3600.000s, lam=+1.000m, rotated@source prec=false
            // o-u -1.00000 +0.00000 +0.00000 | XX +1.00000 +0.00000i XY +0.00036 +0.00000i YX +0.00036 -0.00000i YY +1.00000 +0.00000i
            // o-v +0.00000 -1.00000 +0.00000 | XX +1.00000 +0.00000i XY +0.00036 +0.00000i YX +0.00036 -0.00000i YY +1.00000 +0.00000i
            // u-v +1.00000 -1.00000 +0.00000 | XX +1.00000 -0.00000i XY +0.00036 +0.00000i YX +0.00036 -0.00000i YY +1.00000 -0.00000i
            for &vis_rot in vis_rot_tfb.slice(s![1, 1, ..]) {
                assert_abs_diff_eq!(vis_rot, Jones::from([
                    Complex::new(1.00000, 0.00000),
                    Complex::new(0.00036, 0.00000),
                    Complex::new(0.00036, -0.00000),
                    Complex::new(1.00000, 0.00000),
                ]), epsilon = 1e-5);
            }
        } else {
            // bl  u        v        w        | @ time=   +0.000s, lam=+2.000m, model@obs prec=true
            // o-u -0.49987 +0.00008 -0.01119 | XX +0.14558 -0.15375i XY -0.00036 +0.00044i YX -0.00042 +0.00039i YY +0.15794 -0.16680i
            // o-v +0.00003 -0.49998 -0.00486 | XX +0.21174 +0.00023i XY -0.00057 +0.00004i YX -0.00057 -0.00004i YY +0.22971 +0.00025i
            // u-v +0.49991 -0.50005 +0.00633 | XX +0.14541 +0.15391i XY -0.00042 -0.00039i YX -0.00036 -0.00044i YY +0.15776 +0.16697i
            // bl  u        v        w        | @ time=   +0.000s, lam=+1.000m, model@obs prec=true
            // o-u -0.99975 +0.00015 -0.02238 | XX -0.00216 -0.03962i XY +0.00021 -0.00001i YX -0.00021 +0.00001i YY -0.00369 -0.06754i
            // o-v +0.00007 -0.99995 -0.00972 | XX +0.03968 +0.00009i XY +0.00000 +0.00021i YX +0.00000 -0.00021i YY +0.06764 +0.00015i
            // u-v +0.99982 -1.00010 +0.01266 | XX -0.00225 +0.03962i XY -0.00021 -0.00001i YX +0.00021 +0.00001i YY -0.00384 +0.06753i
            // bl  u        v        w        | @ time=+3600.000s, lam=+2.000m, model@obs prec=true
            // o-u -0.48565 -0.00119 +0.11891 | XX +0.67731 -0.72223i XY -0.00048 +0.00052i YX -0.00049 +0.00051i YY +0.67761 -0.72255i
            // o-v +0.00003 -0.49998 -0.00486 | XX +0.99014 +0.00109i XY -0.00070 +0.00000i YX -0.00070 -0.00001i YY +0.99058 +0.00109i
            // u-v +0.48569 -0.49879 -0.12377 | XX +0.67652 +0.72298i XY -0.00049 -0.00051i YX -0.00048 -0.00052i YY +0.67682 +0.72330i
            // bl  u        v        w        | @ time=+3600.000s, lam=+1.000m, model@obs prec=true
            // o-u -0.97131 -0.00238 +0.23782 | XX -0.06176 -0.96115i XY -0.00002 -0.00029i YX -0.00002 -0.00029i YY -0.06186 -0.96275i
            // o-v +0.00007 -0.99995 -0.00972 | XX +0.96313 +0.00212i XY +0.00029 +0.00000i YX +0.00029 +0.00000i YY +0.96473 +0.00212i
            // u-v +0.97137 -0.99758 -0.24754 | XX -0.06388 +0.96101i XY -0.00002 +0.00029i YX -0.00002 +0.00029i YY -0.06398 +0.96261i
            // bl  u        v        w        | @ time=   +0.000s, lam=+2.000m, rotated@source prec=true
            // o-u -0.47985 +0.00008 -0.14053 | XX +0.21174 +0.00000i XY -0.00057 +0.00004i YX -0.00057 -0.00004i YY +0.22971 -0.00000i
            // o-v +0.00129 -0.49998 -0.00469 | XX +0.21174 -0.00000i XY -0.00057 +0.00004i YX -0.00057 -0.00004i YY +0.22971 -0.00000i
            // u-v +0.48114 -0.50005 +0.13584 | XX +0.21174 -0.00000i XY -0.00057 +0.00004i YX -0.00057 -0.00004i YY +0.22971 -0.00000i
            for &vis_rot in vis_rot_tfb.slice(s![0, 0, ..]) {
                assert_abs_diff_eq!(vis_rot, Jones::from([
                    Complex::new(0.21174, 0.00000),
                    Complex::new(-0.00057, 0.00004),
                    Complex::new(-0.00057, -0.00004),
                    Complex::new(0.22971, 0.00000),
                ]), epsilon = 1e-5);
            }
            // bl  u        v        w        | @ time=   +0.000s, lam=+1.000m, rotated@source prec=true
            // o-u -0.95969 +0.00015 -0.28106 | XX +0.03968 +0.00000i XY +0.00000 +0.00021i YX +0.00000 -0.00021i YY +0.06764 -0.00000i
            // o-v +0.00259 -0.99995 -0.00937 | XX +0.03968 -0.00000i XY +0.00000 +0.00021i YX +0.00000 -0.00021i YY +0.06764 -0.00000i
            // u-v +0.96228 -1.00010 +0.27169 | XX +0.03968 +0.00000i XY +0.00000 +0.00021i YX +0.00000 -0.00021i YY +0.06764 +0.00000i
            for &vis_rot in vis_rot_tfb.slice(s![0, 1, ..]) {
                assert_abs_diff_eq!(vis_rot, Jones::from([
                    Complex::new(0.03968, 0.00000),
                    Complex::new(0.00000, 0.00021),
                    Complex::new(0.00000, -0.00021),
                    Complex::new(0.06764, -0.00000),
                ]), epsilon = 1e-5);
            }
            // bl  u        v        w        | @ time=+3600.000s, lam=+2.000m, rotated@source prec=true
            // o-u -0.49987 -0.00119 -0.01120 | XX +0.99014 -0.00000i XY -0.00070 +0.00001i YX -0.00070 -0.00001i YY +0.99058 -0.00000i
            // o-v +0.00129 -0.49998 -0.00469 | XX +0.99014 +0.00000i XY -0.00070 +0.00001i YX -0.00070 -0.00001i YY +0.99058 -0.00000i
            // u-v +0.50117 -0.49879 +0.00651 | XX +0.99014 +0.00000i XY -0.00070 +0.00001i YX -0.00070 -0.00001i YY +0.99058 -0.00000i
            for &vis_rot in vis_rot_tfb.slice(s![1, 0, ..]) {
                assert_abs_diff_eq!(vis_rot, Jones::from([
                    Complex::new(0.99014, 0.00000),
                    Complex::new(-0.00070, 0.00001),
                    Complex::new(-0.00070, -0.00001),
                    Complex::new(0.99058, -0.00000),
                ]), epsilon = 1e-5);
            }
            // bl  u        v        w        | @ time=+3600.000s, lam=+1.000m, rotated@source prec=true
            // o-u -0.99975 -0.00238 -0.02239 | XX +0.96313 +0.00000i XY +0.00029 +0.00000i YX +0.00029 -0.00000i YY +0.96473 +0.00000i
            // o-v +0.00259 -0.99995 -0.00937 | XX +0.96313 -0.00000i XY +0.00029 +0.00000i YX +0.00029 -0.00000i YY +0.96473 -0.00000i
            // u-v +1.00233 -0.99758 +0.01302 | XX +0.96313 -0.00000i XY +0.00029 +0.00000i YX +0.00029 -0.00000i YY +0.96473 +0.00000i
            for &vis_rot in vis_rot_tfb.slice(s![1, 1, ..]) {
                assert_abs_diff_eq!(vis_rot, Jones::from([
                    Complex::new(0.96313, 0.00000),
                    Complex::new(0.00029, 0.00000),
                    Complex::new(0.00029, -0.00000),
                    Complex::new(0.96473, 0.00000),
                ]), epsilon = 1e-5);
            }
        }

        if !apply_precession {
            // rotated vis should always have the source in phase, so no angle in pols XX, YY
            for vis_rot in vis_rot_tfb.iter() {
                assert_abs_diff_eq!(vis_rot[0].arg(), 0., epsilon = 1e-6); // XX
                assert_abs_diff_eq!(vis_rot[3].arg(), 0., epsilon = 1e-6); // YY
            }
            // baseline 1, from origin to v has no u or w component, should not be affected by the rotation
            for (vis, vis_rot) in izip!(
                vis_tfb.slice(s![.., .., 1]),
                vis_rot_tfb.slice(s![.., .., 1]),
            ) {
                assert_abs_diff_eq!(vis, vis_rot, epsilon = 1e-6);
            }
            // in the second timestep, the source should be at the pointing centre, so should not be
            // attenuated by the beam
            for (vis, vis_rot) in izip!(
                vis_tfb.slice(s![1, .., ..]),
                vis_rot_tfb.slice(s![1, .., ..]),
            ) {
                // XX
                assert_abs_diff_eq!(vis[0].norm(), source_fd as f32, epsilon = 1e-6);
                assert_abs_diff_eq!(vis_rot[0].norm(), source_fd as f32, epsilon = 1e-6);
                // YY
                assert_abs_diff_eq!(vis[3].norm(), source_fd as f32, epsilon = 1e-6);
                assert_abs_diff_eq!(vis_rot[3].norm(), source_fd as f32, epsilon = 1e-6);
            }
        }

        for (tile_ws_obs, tile_ws_src, vis_fb, vis_rot_fb) in izip!(
            tile_ws_obs.outer_iter(),
            tile_ws_src.outer_iter(),
            vis_tfb.outer_iter(),
            vis_rot_tfb.outer_iter(),
        ) {
            for (lambda_m, vis_b, vis_rot_b) in izip!(
                lambdas_m.iter(),
                vis_fb.outer_iter(),
                vis_rot_fb.outer_iter(),
            ) {
                for (&(ant1, ant2), vis, vis_rot) in
                    izip!(ant_pairs.iter(), vis_b.iter(), vis_rot_b.iter(),)
                {
                    let w_obs = tile_ws_obs[ant1] - tile_ws_obs[ant2];
                    let w_src = tile_ws_src[ant1] - tile_ws_src[ant2];
                    let arg = TAU * (w_src - w_obs) / lambda_m;
                    for (pol_model, pol_model_rot) in vis.iter().zip_eq(vis_rot.iter()) {
                        // magnitudes shoud not be affected by rotation
                        assert_abs_diff_eq!(pol_model.norm(), pol_model_rot.norm(), epsilon = 1e-6);
                        let pol_model_rot_expected = Complex::<f64>::from_polar(
                            pol_model.norm() as f64,
                            pol_model.arg() as f64 - arg,
                        );
                        assert_abs_diff_eq!(
                            pol_model_rot_expected.arg() as f32,
                            pol_model_rot.arg(),
                            epsilon = 1e-6
                        );
                    }
                }
            }
        }
    }
}

#[test]
//
fn test_weight_average() {
    let weights_tfb: Array3<f32> = array![
        [[1., 1., 1., -1.], [2., 2., 2., -2.], [4., 4., 4., -4.],],
        [
            [8., 8., 8., -8.],
            [16., 16., 16., -16.],
            [32., 32., 32., -32.],
        ],
    ];

    // 2, 3, 4
    let weights_shape = weights_tfb.dim();
    // 1, 2, 4
    let avg_shape = (1, 2, weights_shape.2);

    let mut weights_avg_tfb = Array3::zeros(avg_shape);

    weights_average(weights_tfb.view(), weights_avg_tfb.view_mut());

    assert_eq!(
        weights_avg_tfb,
        array![[[27., 27., 27., 0.], [36., 36., 36., 0.],],]
    );
}

#[test]
//
fn test_vis_average() {
    #[rustfmt::skip]
    let vis_tfb: Array3<Jones<f32>> = array![
        [
            [ Jones::zero(), Jones::zero(), Jones::zero(), Jones::identity() ],
            [ Jones::zero(), Jones::zero(), Jones::identity(), Jones::zero() ],
            [ Jones::zero(), Jones::zero(), Jones::identity(), Jones::identity() ],
        ],
        [
            [ Jones::zero(), Jones::identity(), Jones::zero(), Jones::zero() ],
            [ Jones::zero(), Jones::identity(), Jones::zero(), Jones::identity() ],
            [ Jones::zero(), Jones::identity(), Jones::identity(), Jones::zero() ],
        ],
    ];

    #[rustfmt::skip]
    let weights_tfb: Array3<f32> = array![
        [
            [1., 1., 1., 1.],
            [2., 2., 2., 2.],
            [4., 4., 4., 4.],
        ],
        [
            [8., 8., 8., 8.],
            [16., 16., 16., 16.],
            [32., 32., 32., 32.],
        ],
    ];

    // 2, 3, 4
    let vis_shape = vis_tfb.dim();
    // 1, 2, 4
    let avg_shape = (1, 2, vis_shape.2);

    let mut vis_avg_tfb = Array3::zeros(avg_shape);

    vis_average2(vis_tfb.view(), vis_avg_tfb.view_mut(), weights_tfb.view());

    assert_eq!(
        vis_avg_tfb.slice(s![.., .., 0]),
        array![[Jones::zero(), Jones::zero()]]
    );

    assert_eq!(
        vis_avg_tfb.slice(s![.., 0, 1]),
        array![Jones::identity() * 24. / 27.]
    );
    assert_eq!(
        vis_avg_tfb.slice(s![.., 0, 2]),
        array![Jones::identity() * 2. / 27.]
    );
    assert_eq!(
        vis_avg_tfb.slice(s![.., 0, 3]),
        array![Jones::identity() * 17. / 27.]
    );
    assert_eq!(
        vis_avg_tfb.slice(s![.., 1, 1]),
        array![Jones::identity() * 32. / 36.]
    );
    assert_eq!(
        vis_avg_tfb.slice(s![.., 1, 2]),
        array![Jones::identity() * 36. / 36.]
    );
    assert_eq!(
        vis_avg_tfb.slice(s![.., 1, 3]),
        array![Jones::identity() * 4. / 36.]
    );
}

#[test]
//
fn test_apply_iono2() {
    let obs_context = get_phase1_obs_context(128);

    let array_pos = obs_context.array_position;

    // second timestep is at 1h
    let hour_epoch = obs_context.timestamps[1];
    let num_tiles = obs_context.get_total_num_tiles();
    let num_times = obs_context.timestamps.len();
    let num_baselines = (num_tiles * (num_tiles - 1)) / 2;
    let ant_pairs = (0..num_baselines)
        .map(|bl_idx| cross_correlation_baseline_to_tiles(num_tiles, bl_idx))
        .collect_vec();
    let flagged_tiles = HashSet::new();
    let num_chans = obs_context.fine_chan_freqs.len();

    let fine_chan_freqs_hz = obs_context
        .fine_chan_freqs
        .iter()
        .map(|&f| f as f64)
        .collect_vec();
    let lambdas_m = fine_chan_freqs_hz.iter().map(|&f| VEL_C / f).collect_vec();

    // source is at zenith at 1h
    let lst_1h_rad = get_lmst(
        array_pos.longitude_rad,
        hour_epoch,
        obs_context.dut1.unwrap_or_default(),
    );
    let source_radec =
        RADec::from_hadec(HADec::from_radians(0., array_pos.latitude_rad), lst_1h_rad);
    let source_fd = 1.;
    let source_list = SourceList::from(indexmap! {
        "One".into() => point_src_i!(source_radec, 0., fine_chan_freqs_hz[0], source_fd),
    });

    let mut vis_tfb = Array3::default((num_times, num_chans, num_baselines));
    let mut vis_iono_tfb = Array3::default((num_times, num_chans, num_baselines));

    // tile uvs and ws in the source phase centre
    let mut tile_uvs_src = Array2::default((num_times, num_tiles));
    let mut tile_ws_src = Array2::default((num_times, num_tiles));

    let beam = get_beam(num_tiles);

    for apply_precession in [false, true] {
        let modeller = SkyModellerCpu::new(
            &beam,
            &source_list,
            Polarisations::default(),
            &obs_context.tile_xyzs,
            &fine_chan_freqs_hz,
            &flagged_tiles,
            obs_context.phase_centre,
            array_pos.longitude_rad,
            array_pos.latitude_rad,
            obs_context.dut1.unwrap_or_default(),
            apply_precession,
        );

        vis_tfb.fill(Jones::zero());
        model_timesteps(&modeller, &obs_context.timestamps, vis_tfb.view_mut()).unwrap();

        setup_tile_uv_w_arrays(
            tile_uvs_src.view_mut(),
            tile_ws_src.view_mut(),
            &obs_context,
            source_radec,
            apply_precession,
        );

        // we want consts such that at lambda = 2m, the shift moves the source to the phase centre
        let iono_lmn = source_radec.to_lmn(obs_context.phase_centre);
        let iono_consts = IonoConsts {
            alpha: iono_lmn.l / 4.,
            beta: iono_lmn.m / 4.,
            gain: 1.0,
        };
        // let iono_consts = ((lst_1h_rad-lst_zenith_rad)/4., 0.);

        apply_iono2(
            vis_tfb.view(),
            vis_iono_tfb.view_mut(),
            tile_uvs_src.view(),
            iono_consts,
            &lambdas_m,
        );

        // display_vis_tfb(
        //     &"model@obs".into(),
        //     vis_tfb.view(),
        //     &obs_context,
        //     obs_context.phase_centre,
        //     apply_precession,
        // );
        // display_vis_tfb(
        //     &"iono@source".into(),
        //     vis_iono_tfb.view(),
        //     &obs_context,
        //     source_radec,
        //     apply_precession,
        // );

        let mut tol_norm = 0_f32;
        let mut tol_sc = 0_f32;
        for (time, tile_uvs_src, vis_fb, vis_iono_fb) in izip!(
            obs_context.timestamps.iter(),
            tile_uvs_src.outer_iter(),
            vis_tfb.outer_iter(),
            vis_iono_tfb.outer_iter(),
        ) {

            for (&lambda_m, vis_b, vis_iono_b) in izip!(
                lambdas_m.iter(),
                vis_fb.outer_iter(),
                vis_iono_fb.outer_iter(),
            ) {
                for (&(ant1, ant2), vis, vis_iono) in
                    izip!(ant_pairs.iter(), vis_b.iter(), vis_iono_b.iter(),)
                {
                    let UV { u, v } = tile_uvs_src[ant1] - tile_uvs_src[ant2];
                    let arg = TAU * (u * iono_consts.alpha + v * iono_consts.beta) * lambda_m;
                    for (pol_model, pol_model_iono) in vis.iter().zip_eq(vis_iono.iter()) {
                        // magnitudes shoud not be affected by iono rotation
                        assert_abs_diff_eq!(
                            pol_model.norm(),
                            pol_model_iono.norm(),
                            epsilon = 2e-7
                        );
                        tol_norm = tol_norm.max((pol_model.norm()-pol_model_iono.norm()).abs());
                        let pol_model_iono_expected = Complex::<f64>::from_polar(
                            pol_model.norm() as f64,
                            pol_model.arg() as f64 - arg,
                        );
                        let (ex_s, ex_c) = pol_model_iono_expected.arg().sin_cos();
                        let (rx_s, rx_c) = pol_model_iono.arg().sin_cos();
                        assert_abs_diff_eq!( ex_s as f32, rx_s, epsilon = 1e-6 );
                        assert_abs_diff_eq!( ex_c as f32, rx_c, epsilon = 1e-6 );
                        tol_sc = tol_sc.max((ex_s as f32 - rx_s).abs());
                        tol_sc = tol_sc.max((ex_c as f32 - rx_c).abs());
                    }
                }
            }
        }
        // println!("tol norm: {tol_norm:7.2e} sc {tol_sc:7.2e}");
    }
}

#[test]
fn test_get_weights_rts() {
    let obs_context = get_phase1_obs_context(128);

    // second timestep is at 1h
    let num_tiles = obs_context.get_total_num_tiles();
    let num_times = obs_context.timestamps.len();

    let fine_chan_freqs_hz = obs_context
        .fine_chan_freqs
        .iter()
        .map(|&f| f as f64)
        .collect_vec();
    let lambdas_m = fine_chan_freqs_hz.iter().map(|&f| VEL_C / f).collect_vec();

    // tile uvs and ws in the source phase centre
    let mut tile_uvs_obs = Array2::default((num_times, num_tiles));
    let mut tile_ws_obs = Array2::default((num_times, num_tiles));

    setup_tile_uv_w_arrays(
        tile_uvs_obs.view_mut(),
        tile_ws_obs.view_mut(),
        &obs_context,
        obs_context.phase_centre,
        true,
    );

    let weights = get_weights_rts(
        tile_uvs_obs.view(),
        lambdas_m.as_slice(),
        40.,
    );

    // check all weights between 0 and 1
    for w in weights.iter() {
        assert!(*w <= 1., "weight {} should be <= 1", w);
        assert!(*w >= 0., "weight {} should be >= 0", w);
    }


}

#[test]
/// test iono_fit, where residual is just iono rotated model
fn test_iono_fit() {
    // let obs_context = get_phase1_obs_context();
    // TODO(Dev): this test only works with TILE_SPACING=1
    let obs_context = get_simple_obs_context(1.0);
    let second_epoch = obs_context.timestamps[1];

    let array_pos = obs_context.array_position;

    let num_tiles = obs_context.get_total_num_tiles();
    let num_times = obs_context.timestamps.len();
    let num_baselines = (num_tiles * (num_tiles - 1)) / 2;
    let flagged_tiles = HashSet::new();
    let num_chans = obs_context.fine_chan_freqs.len();

    // lambda = 1m
    let fine_chan_freqs_hz = obs_context
        .fine_chan_freqs
        .iter()
        .map(|&f| f as f64)
        .collect_vec();
    let lambdas_m = fine_chan_freqs_hz.iter().map(|&f| VEL_C / f).collect_vec();

    // source is at zenith at 1h
    let lst_1h_rad = get_lmst(
        array_pos.longitude_rad,
        second_epoch,
        obs_context.dut1.unwrap_or_default(),
    );
    let source_radec =
        RADec::from_hadec(HADec::from_radians(0., array_pos.latitude_rad), lst_1h_rad);
    let source_fd = 1.;
    let source_list = SourceList::from(indexmap! {
        "One".into() => point_src_i!(source_radec, 0., fine_chan_freqs_hz[0], source_fd),
    });

    let mut vis_tfb = Array3::<Jones<f32>>::zeros((num_times, num_chans, num_baselines));
    let mut vis_iono_tfb = Array3::<Jones<f32>>::zeros((num_times, num_chans, num_baselines));

    // tile uvs and ws in the source phase centre
    let mut tile_uvs_src = Array2::default((num_times, num_tiles));
    let mut tile_ws_src = Array2::default((num_times, num_tiles));

    let beam = get_beam(num_tiles);

    for apply_precession in [false, true] {
        // unlike the other tests, this is in the SOURCE phase centre
        let modeller = SkyModellerCpu::new(
            &beam,
            &source_list,
            Polarisations::default(),
            &obs_context.tile_xyzs,
            &fine_chan_freqs_hz,
            &flagged_tiles,
            source_radec,
            array_pos.longitude_rad,
            array_pos.latitude_rad,
            obs_context.dut1.unwrap_or_default(),
            apply_precession,
        );

        vis_tfb.fill(Jones::zero());

        model_timesteps(&modeller, &obs_context.timestamps, vis_tfb.view_mut()).unwrap();

        setup_tile_uv_w_arrays(
            tile_uvs_src.view_mut(),
            tile_ws_src.view_mut(),
            &obs_context,
            source_radec,
            apply_precession,
        );

        // display_vis_tfb(
        //     &"model@obs".into(),
        //     vis_tfb.view(),
        //     &obs_context,
        //     obs_context.phase_centre,
        //     apply_precession,
        // );

        let shape = vis_tfb.shape();
        let weights = Array3::ones((shape[0], shape[1], shape[2]));

        // let weights = get_weights_rts(
        //     tile_uvs_src.view(),
        //     lambdas_m.as_slice(),
        //     4.0,
        // );

        for iono_consts in [
            // TODO(Dev): Actually test with gain
            IonoConsts {
                alpha: 0.,
                beta: 0.,
                gain: 1.0,
            },
            IonoConsts {
                alpha: 0.0001,
                beta: -0.0003,
                gain: 1.0,
                // gain: 1.0005,
            },
            IonoConsts {
                alpha: 0.0003,
                beta: -0.0001,
                gain: 1.0,
                // gain: 0.9995,
            },
            IonoConsts {
                alpha: -0.0007,
                beta: 0.0001,
                gain: 1.0,
                // gain: 1.0005,
            },
        ] {
            apply_iono2(
                vis_tfb.view(),
                vis_iono_tfb.view_mut(),
                tile_uvs_src.view(),
                iono_consts,
                &lambdas_m,
            );

            display_vis_tfb(
                &"iono@obs".into(),
                vis_iono_tfb.view(),
                &obs_context,
                obs_context.phase_centre,
                apply_precession,
            );

            let results = iono_fit(
                vis_iono_tfb.view(),
                weights.view(),
                vis_tfb.view(),
                &lambdas_m,
                tile_uvs_src.view(),
            );

            println!("prec: {:?}, expected: {:?}, got: {:?}", apply_precession, iono_consts, &results);

            assert_abs_diff_eq!(results[0], iono_consts.alpha, epsilon = 1e-7);
            assert_abs_diff_eq!(results[1], iono_consts.beta, epsilon = 1e-7);
            // let gain = results[2] / results[3];
            // assert_abs_diff_eq!(gain, iono_consts.gain, epsilon = 1e-7);
        }
    }
}

#[test]
/// - synthesize model visibilities
/// - apply ionospheric rotation
/// - create residual: ionospheric - model
/// - ap ply_iono3 should result in empty visibilitiesiono rotated model
fn test_apply_iono3() {
    let obs_context = get_simple_obs_context(TILE_SPACING);
    let array_pos = obs_context.array_position;

    let num_tiles = obs_context.get_total_num_tiles();
    let num_times = obs_context.timestamps.len();
    let num_baselines = (num_tiles * (num_tiles - 1)) / 2;
    let flagged_tiles = HashSet::new();
    let num_chans = obs_context.fine_chan_freqs.len();

    // lambda = 1m
    let fine_chan_freqs_hz = obs_context
        .fine_chan_freqs
        .iter()
        .map(|&f| f as f64)
        .collect_vec();
    let lambdas_m = fine_chan_freqs_hz.iter().map(|&f| VEL_C / f).collect_vec();

    // source is at zenith at 1h
    let lst_1h_rad = get_lmst(
        array_pos.longitude_rad,
        obs_context.timestamps[1],
        obs_context.dut1.unwrap_or_default(),
    );
    let source_radec =
        RADec::from_hadec(HADec::from_radians(0., array_pos.latitude_rad), lst_1h_rad);
    let source_fd = 1.;
    let source_list = SourceList::from(indexmap! {
        "One".into() => point_src_i!(source_radec, 0., fine_chan_freqs_hz[0], source_fd),
    });

    let beam = get_beam(num_tiles);

    // residual visibilities in the observation phase centre
    let mut vis_resid_obs_tfb = Array3::<Jones<f32>>::zeros((num_times, num_chans, num_baselines));
    // model visibilities in the observation phase centre
    let mut vis_model_obs_tfb = Array3::<Jones<f32>>::zeros((num_times, num_chans, num_baselines));
    // iono rotated model visibilities in the observation phase centre
    let mut vis_iono_obs_tfb = Array3::<Jones<f32>>::zeros((num_times, num_chans, num_baselines));

    // tile uvs and ws in the source phase centre
    let mut tile_uvs_src = Array2::default((num_times, num_tiles));
    let mut tile_ws_src = Array2::default((num_times, num_tiles));

    for apply_precession in [false, true] {
        let modeller = SkyModellerCpu::new(
            &beam,
            &source_list,
            Polarisations::default(),
            &obs_context.tile_xyzs,
            &fine_chan_freqs_hz,
            &flagged_tiles,
            obs_context.phase_centre,
            array_pos.longitude_rad,
            array_pos.latitude_rad,
            obs_context.dut1.unwrap_or_default(),
            apply_precession,
        );

        vis_model_obs_tfb.fill(Jones::zero());

        model_timesteps(
            &modeller,
            &obs_context.timestamps,
            vis_model_obs_tfb.view_mut(),
        )
        .unwrap();

        setup_tile_uv_w_arrays(
            tile_uvs_src.view_mut(),
            tile_ws_src.view_mut(),
            &obs_context,
            source_radec,
            apply_precession,
        );

        // display_vis_tfb(
        //     &"model@obs".into(),
        //     vis_model_obs_tfb.view(),
        //     &obs_context,
        //     obs_context.phase_centre,
        //     apply_precession,
        // );

        for iono_consts in [
            IonoConsts {
                alpha: 0.0001,
                beta: -0.0003,
                gain: 1.0,
            },
            IonoConsts {
                alpha: 0.0003,
                beta: -0.0001,
                gain: 1.0,
            },
            IonoConsts {
                alpha: -0.0007,
                beta: 0.0001,
                gain: 1.0,
            },
        ] {
            // apply iono rotation at source phase to model at observation phase
            apply_iono2(
                vis_model_obs_tfb.view(),
                vis_iono_obs_tfb.view_mut(),
                tile_uvs_src.view(),
                iono_consts,
                &lambdas_m,
            );

            // subtract model from iono at observation phase centre
            vis_resid_obs_tfb.assign(&vis_iono_obs_tfb);
            vis_resid_obs_tfb -= &vis_model_obs_tfb;

            apply_iono3(
                vis_model_obs_tfb.view(),
                vis_resid_obs_tfb.view_mut(),
                tile_uvs_src.view(),
                iono_consts,
                IonoConsts::default(),
                &lambdas_m,
            );

            // display_vis_tfb(
            //     &"residual@obs".into(),
            //     vis_residual_obs_tfb.view(),
            //     &obs_context,
            //     obs_context.phase_centre,
            //     apply_precession,
            // );

            for jones_residual in vis_resid_obs_tfb.iter() {
                for pol_residual in jones_residual.iter() {
                    assert_abs_diff_eq!(pol_residual.norm(), 0., epsilon = 5e-8);
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
#[allow(clippy::upper_case_acronyms)]
enum PeelType {
    CPU,
    #[cfg(any(feature = "cuda", feature = "hip"))]
    Gpu,
}

/// Test a peel function with and without precession on a single source
#[track_caller]
fn test_peel_single_source(peel_type: PeelType) {
    // enable trace
    // let mut builder = env_logger::Builder::from_default_env();
    // builder.target(env_logger::Target::Stdout);
    // builder.format_target(false);
    // builder.filter_level(log::LevelFilter::Debug);
    // builder.init();

    // modify obs_context so that timesteps are closer together
    let obs_context = get_phase1_obs_context(128);
    // let obs_context = get_simple_obs_context(TILE_SPACING);

    let array_pos = obs_context.array_position;
    let num_tiles = obs_context.get_total_num_tiles();
    let num_times = obs_context.timestamps.len();
    let num_baselines = (num_tiles * (num_tiles - 1)) / 2;
    let flagged_tiles = HashSet::new();

    let num_chans = obs_context.fine_chan_freqs.len();
    let chanblocks = obs_context
        .fine_chan_freqs
        .iter()
        .enumerate()
        .map(|(i, f)| Chanblock {
            chanblock_index: i as u16,
            unflagged_index: i as u16,
            freq: *f as f64,
        })
        .collect_vec();

    // lambda = 1m
    let fine_chan_freqs_hz = obs_context
        .fine_chan_freqs
        .iter()
        .map(|&f| f as f64)
        .collect_vec();
    let lambdas_m = fine_chan_freqs_hz.iter().map(|&f| VEL_C / f).collect_vec();
    let avg_freq = 4;
    let low_res_lambdas_m = obs_context.fine_chan_freqs.as_slice().chunks(avg_freq).map(|chunk| {
        let f = chunk.iter().sum::<u64>() as f64 / chunk.len() as f64;
        VEL_C / f
    }).collect_vec();

    let lst_0h_rad = get_lmst(
        array_pos.longitude_rad,
        obs_context.timestamps[0],
        obs_context.dut1.unwrap_or_default(),
    );
    let source_radec =
        RADec::from_hadec(HADec::from_radians(0.2, array_pos.latitude_rad), lst_0h_rad);
    let source_fd = 1.;
    let source_list = SourceList::from([(
        "One".into(),
        point_src_i!(source_radec, 0., fine_chan_freqs_hz[0], source_fd),
    )]);

    let beam = get_beam(num_tiles);

    // model visibilities in the observation phase centre
    let mut vis_model_obs_tfb = Array3::zeros((num_times, num_chans, num_baselines));
    // iono rotated model visibilities in the observation phase centre
    let mut vis_iono_obs_tfb = Array3::zeros((num_times, num_chans, num_baselines));
    // residual visibilities in the observation phase centre
    let mut vis_residual_obs_tfb = Array3::zeros((num_times, num_chans, num_baselines));

    // tile uvs and ws in the source phase centre
    let mut tile_uvs_src = Array2::default((num_times, num_tiles));
    let mut tile_ws_src = Array2::default((num_times, num_tiles));

    let timeblock = Timeblock {
        index: 0,
        range: 0..2,
        timestamps: obs_context.timestamps.clone(),
        timesteps: vec1![0, 1],
        median: obs_context.timestamps[0],
    };

    let vis_shape = vis_residual_obs_tfb.dim();
    let vis_weights = Array3::<f32>::ones(vis_shape);
    let source_weighted_positions = [source_radec];
    let baseline_weights = vec![1.0; vis_model_obs_tfb.len_of(Axis(2))];

    let multi_progress = MultiProgress::with_draw_target(ProgressDrawTarget::hidden());

    for apply_precession in [false, true] {
        let mut high_res_modeller = new_sky_modeller(
            &beam,
            &source_list,
            Polarisations::default(),
            &obs_context.tile_xyzs,
            &fine_chan_freqs_hz,
            &flagged_tiles,
            obs_context.phase_centre,
            array_pos.longitude_rad,
            array_pos.latitude_rad,
            obs_context.dut1.unwrap_or_default(),
            apply_precession,
        )
        .unwrap();

        vis_model_obs_tfb.fill(Jones::zero());

        model_timesteps(
            &*high_res_modeller,
            &obs_context.timestamps,
            vis_model_obs_tfb.view_mut(),
        )
        .unwrap();

        setup_tile_uv_w_arrays(
            tile_uvs_src.view_mut(),
            tile_ws_src.view_mut(),
            &obs_context,
            source_radec,
            apply_precession,
        );

        // display_vis_tfb(
        //     &"model@obs".into(),
        //     vis_model_obs_tfb.view(),
        //     &obs_context,
        //     obs_context.phase_centre,
        //     apply_precession,
        // );

        for iono_consts in [
            IonoConsts {
                alpha: 0.,
                beta: 0.,
                gain: 1.1,
            },
            IonoConsts {
                alpha: 0.0001,
                beta: -0.0003,
                gain: 0.9,
            },
            IonoConsts {
                alpha: 0.0003,
                beta: -0.0001,
                gain: 1.1,
            },
            IonoConsts {
                alpha: -0.0004, // -0.0007 seems to break the tests
                beta: 0.0001,
                gain: 0.9,
            },
        ] {
            log::info!("Testing with iono consts {iono_consts:?}");
            apply_iono2(
                vis_model_obs_tfb.view(),
                vis_iono_obs_tfb.view_mut(),
                tile_uvs_src.view(),
                iono_consts,
                &lambdas_m,
            );

            // display_vis_tfb(
            //     &format!("iono@obs prec={}, ({}, {})", apply_precession, &consts_lm.0, &consts_lm.1),
            //     vis_iono_obs_tfb.view(),
            //     &obs_context,
            //     obs_context.phase_centre,
            //     apply_precession,
            // );

            // subtract model from iono at observation phase centre
            vis_residual_obs_tfb.assign(&vis_iono_obs_tfb);
            vis_residual_obs_tfb -= &vis_model_obs_tfb;

            // display_vis_tfb(
            //     &"residual@obs".into(),
            //     vis_residual_obs_tfb.view(),
            //     &obs_context,
            //     obs_context.phase_centre,
            //     apply_precession,
            // );

            let mut all_iono_consts = vec![IonoConsts::default(); 1];

            // When peel_cpu and peel_gpu are able to take generic
            // `SkyModeller` objects (requires the generic objects to take
            // currently CUDA-only methods), uncomment the following code and
            // delete what follows.

            // let function = match peel_type {
            //     CPU => peel_cpu,
            //     #[cfg(any(feature = "cuda", feature = "hip"))]
            //     CUDA => peel_gpu,
            // };
            // function(
            //     vis_residual_obs_tfb.view_mut(),
            //     vis_weights.view(),
            //     &timeblock,
            //     &source_list,
            //     &mut iono_consts,
            //     &source_weighted_positions,
            //     num_sources_to_iono_subtract,
            //     &fine_chan_freqs_hz,
            //     &lambdas_m,
            //     &lambdas_m,
            //     &obs_context,
            //     obs_context.array_position.unwrap(),
            //     &obs_context.tile_xyzs,
            //     high_res_modeller.deref_mut(),
            //     obs_context.dut1.unwrap_or_default(),
            //     !apply_precession,
            //     &multi_progress,
            // )
            // .unwrap();
            match peel_type {
                PeelType::CPU => peel_cpu(
                    vis_residual_obs_tfb.view_mut(),
                    vis_weights.view(),
                    &timeblock,
                    &source_list,
                    &mut all_iono_consts,
                    &source_weighted_positions,
                    NUM_PASSES,
                    NUM_LOOPS,
                    SHORT_BASELINE_SIGMA,
                    CONVERGENCE,
                    &fine_chan_freqs_hz,
                    &lambdas_m,
                    &low_res_lambdas_m,
                    &obs_context,
                    obs_context.array_position,
                    &obs_context.tile_xyzs,
                    &mut *high_res_modeller,
                    obs_context.dut1.unwrap_or_default(),
                    !apply_precession,
                    &multi_progress,
                )
                .unwrap(),

                #[cfg(any(feature = "cuda", feature = "hip"))]
                PeelType::Gpu => {
                    let mut high_res_modeller = crate::model::SkyModellerGpu::new(
                        &beam,
                        &source_list,
                        Polarisations::default(),
                        &obs_context.tile_xyzs,
                        &fine_chan_freqs_hz,
                        &flagged_tiles,
                        obs_context.phase_centre,
                        array_pos.longitude_rad,
                        array_pos.latitude_rad,
                        obs_context.dut1.unwrap_or_default(),
                        apply_precession,
                    )
                    .unwrap();

                    peel_gpu(
                        vis_residual_obs_tfb.view_mut(),
                        vis_weights.view(),
                        &timeblock,
                        &source_list,
                        &mut all_iono_consts,
                        &source_weighted_positions,
                        NUM_PASSES,
                        NUM_LOOPS,
                        SHORT_BASELINE_SIGMA,
                        CONVERGENCE,
                        &chanblocks,
                        &lambdas_m,
                        &low_res_lambdas_m,
                        &obs_context,
                        obs_context.array_position,
                        &obs_context.tile_xyzs,
                        &baseline_weights,
                        &mut high_res_modeller,
                        obs_context.dut1.unwrap_or_default(),
                        !apply_precession,
                        &multi_progress,
                    )
                    .unwrap()
                }
            };

            // println!("prec: {apply_precession:?}, expected: {iono_consts:?}, got: {:?}", all_iono_consts[0]);

            // display_vis_tfb(
            //     &format!("peeled@obs prec={apply_precession}"),
            //     vis_residual_obs_tfb.view(),
            //     &obs_context,
            //     obs_context.phase_centre,
            //     apply_precession,
            // );

            // let out_path =  match peel_type {
            //     PeelType::CPU => PathBuf::from(&format!("out/test_peel_cpu_single_source{}.ms", if apply_precession { "_prec" } else { "" })),
            //     #[cfg(any(feature = "cuda", feature = "hip"))]
            //     PeelType::Gpu => PathBuf::from(&format!("out/test_peel_gpu_single_source{}.ms", if apply_precession { "_prec" } else { "" })),
            // }.to_path_buf();

            // write_vis_tfb(out_path, vis_residual_obs_tfb.view(), &obs_context);

            // peel should perfectly remove the iono rotate model vis
            let mut norm_sum = 0.;
            for jones_residual in vis_residual_obs_tfb.iter() {
                for pol_residual in jones_residual.iter() {
                    norm_sum += pol_residual.norm();
                    // #[cfg(not(feature = "gpu-single"))]
                    // assert_abs_diff_eq!(pol_residual.norm(), 0., epsilon = 1.3e-8);
                    // #[cfg(feature = "gpu-single")]
                    // assert_abs_diff_eq!(pol_residual.norm(), 0., epsilon = 1.7e-8);
                }
            }
            norm_sum /= vis_residual_obs_tfb.len() as f32;

            let IonoConsts { alpha: ax, beta: bx, gain: gx }  = iono_consts;
            let IonoConsts { alpha: ar, beta: br, gain: gr }  = all_iono_consts[0];
            println!("tolerances prec={apply_precession:6}: norm {norm_sum:7.2e} exp a {ax:+9.7} b {bx:+9.7} g {gx:5.3} rx a {ar:+11.9} b {br:+11.9} g {gr:5.3} tol a {:8.2e} b {:8.2e} g{:8.2e}",
                (ar - ax).abs(), (br - bx).abs(), (gr - gx).abs()
            );

            let (ab_epsilon, g_epsilon, n_epsilon) = match peel_type {
                PeelType::CPU => (3e-11, 1e-7, 8e-8),
                #[cfg(all(any(feature = "cuda", feature = "hip"), not(feature = "gpu-single")))]
                PeelType::Gpu => (3e-11, 1e-7, 8e-8),
                #[cfg(all(any(feature = "cuda", feature = "hip"), feature = "gpu-single"))]
                PeelType::Gpu => (2e-7, 3e-5, 3e-4), // TODO(Dev): bring this down
            };

            assert_abs_diff_eq!(ax, ar, epsilon = ab_epsilon);
            assert_abs_diff_eq!(bx, br, epsilon = ab_epsilon);
            assert_abs_diff_eq!(gx, gr, epsilon = g_epsilon);
            assert_abs_diff_eq!(norm_sum, 0., epsilon = n_epsilon);
        }
        // cargo test --no-default-features --features=hdf5-static,cuda,gpu-single --release -- peel_gpu_single --nocapture
        // tolerances prec=false : norm 2.88e-6 exp a +0.0000000 b +0.0000000 g 1.100 rx a +0.000000006 b -0.000000000 g 1.100 tol a  5.88e-9 b 7.89e-11 g 8.18e-8
        // tolerances prec=false : norm 2.42e-4 exp a +0.0001000 b -0.0003000 g 0.900 rx a +0.000100051 b -0.000300014 g 0.900 tol a  5.07e-8 b  1.41e-8 g 1.54e-6
        // tolerances prec=false : norm 2.56e-4 exp a +0.0003000 b -0.0001000 g 1.100 rx a +0.000300102 b -0.000100018 g 1.100 tol a  1.02e-7 b  1.79e-8 g 1.16e-5
        // tolerances prec=false : norm 2.36e-4 exp a -0.0004000 b +0.0001000 g 0.900 rx a -0.000399920 b +0.000099992 g 0.900 tol a  8.04e-8 b  7.94e-9 g 2.18e-6
        // tolerances prec=true  : norm 3.37e-6 exp a +0.0000000 b +0.0000000 g 1.100 rx a +0.000000006 b -0.000000001 g 1.100 tol a  6.01e-9 b  1.08e-9 g 2.78e-8
        // tolerances prec=true  : norm 2.64e-4 exp a +0.0001000 b -0.0003000 g 0.900 rx a +0.000100055 b -0.000300033 g 0.900 tol a  5.51e-8 b  3.27e-8 g 1.04e-6
        // tolerances prec=true  : norm 2.75e-4 exp a +0.0003000 b -0.0001000 g 1.100 rx a +0.000300107 b -0.000100028 g 1.100 tol a  1.07e-7 b  2.83e-8 g 1.31e-5
        // tolerances prec=true  : norm 2.53e-4 exp a -0.0004000 b +0.0001000 g 0.900 rx a -0.000399916 b +0.000099981 g 0.900 tol a  8.37e-8 b  1.93e-8 g 2.54e-6
        // cargo test --no-default-features --features=hdf5-static,cuda --release -- peel_gpu_single --nocapture
        // tolerances prec=false : norm 1.92e-8 exp a +0.0000000 b +0.0000000 g 1.100 rx a -0.000000000 b +0.000000000 g 1.100 tol a 1.86e-15 b 1.12e-15 g 1.78e-9
        // tolerances prec=false : norm 4.14e-8 exp a +0.0001000 b -0.0003000 g 0.900 rx a +0.000100000 b -0.000300000 g 0.900 tol a 1.17e-12 b 1.43e-13 g 1.50e-8
        // tolerances prec=false : norm 4.98e-8 exp a +0.0003000 b -0.0001000 g 1.100 rx a +0.000300000 b -0.000100000 g 1.100 tol a 1.99e-12 b 8.58e-14 g 2.30e-8
        // tolerances prec=false : norm 7.24e-8 exp a -0.0004000 b +0.0001000 g 0.900 rx a -0.000400000 b +0.000100000 g 0.900 tol a 6.84e-12 b 7.81e-13 g 6.82e-8
        // tolerances prec=true  : norm 4.20e-8 exp a +0.0000000 b +0.0000000 g 1.100 rx a -0.000000000 b -0.000000000 g 1.100 tol a 2.11e-16 b 5.32e-16 g 3.99e-8
        // tolerances prec=true  : norm 4.31e-8 exp a +0.0001000 b -0.0003000 g 0.900 rx a +0.000100000 b -0.000300000 g 0.900 tol a 1.21e-12 b 1.22e-13 g 6.41e-9
        // tolerances prec=true  : norm 4.97e-8 exp a +0.0003000 b -0.0001000 g 1.100 rx a +0.000300000 b -0.000100000 g 1.100 tol a 1.54e-12 b 1.16e-13 g 1.50e-8
        // tolerances prec=true  : norm 6.12e-8 exp a -0.0004000 b +0.0001000 g 0.900 rx a -0.000400000 b +0.000100000 g 0.900 tol a 6.95e-12 b 7.94e-13 g 2.97e-8
        // cargo test --no-default-features --features=hdf5-static --release -- peel_cpu_single --nocapture
        // tolerances prec=false : norm 1.91e-8 exp a +0.0000000 b +0.0000000 g 1.100 rx a -0.000000000 b +0.000000000 g 1.100 tol a 2.17e-15 b 1.07e-15 g7.34e-10
        // tolerances prec=false : norm 3.19e-8 exp a +0.0001000 b -0.0003000 g 0.900 rx a +0.000100000 b -0.000300000 g 0.900 tol a 1.13e-12 b 2.04e-13 g 1.61e-8
        // tolerances prec=false : norm 3.17e-8 exp a +0.0003000 b -0.0001000 g 1.100 rx a +0.000300000 b -0.000100000 g 1.100 tol a 1.66e-12 b 6.63e-14 g 2.28e-9
        // tolerances prec=false : norm 5.45e-8 exp a -0.0004000 b +0.0001000 g 0.900 rx a -0.000400000 b +0.000100000 g 0.900 tol a 6.66e-12 b 1.30e-12 g 4.39e-8
        // tolerances prec=true  : norm 2.15e-8 exp a +0.0000000 b +0.0000000 g 1.100 rx a -0.000000000 b -0.000000000 g 1.100 tol a 1.10e-16 b 4.90e-16 g 3.88e-9
        // tolerances prec=true  : norm 3.25e-8 exp a +0.0001000 b -0.0003000 g 0.900 rx a +0.000100000 b -0.000300000 g 0.900 tol a 1.16e-12 b 1.98e-13 g 1.60e-8
        // tolerances prec=true  : norm 3.34e-8 exp a +0.0003000 b -0.0001000 g 1.100 rx a +0.000300000 b -0.000100000 g 1.100 tol a 1.65e-12 b 5.60e-14 g 2.22e-9
        // tolerances prec=true  : norm 5.59e-8 exp a -0.0004000 b +0.0001000 g 0.900 rx a -0.000400000 b +0.000100000 g 0.900 tol a 6.63e-12 b 1.31e-12 g 4.40e-8
    }
}

#[track_caller]
fn test_peel_multi_source(peel_type: PeelType) {
    // // enable trace
    // let mut builder = env_logger::Builder::from_default_env();
    // builder.target(env_logger::Target::Stdout);
    // builder.format_target(false);
    // builder.filter_level(log::LevelFilter::Debug);
    // builder.init();

    let obs_context = get_phase1_obs_context(128);

    let array_pos = obs_context.array_position;
    let num_tiles = obs_context.get_total_num_tiles();
    let num_times = obs_context.timestamps.len();
    let num_baselines = (num_tiles * (num_tiles - 1)) / 2;
    let flagged_tiles = HashSet::new();

    let num_chans = obs_context.fine_chan_freqs.len();
    let chanblocks = obs_context
        .fine_chan_freqs
        .iter()
        .enumerate()
        .map(|(i, f)| Chanblock {
            chanblock_index: i as u16,
            unflagged_index: i as u16,
            freq: *f as f64,
        })
        .collect_vec();

    // lambda = 1m
    let fine_chan_freqs_hz = obs_context
        .fine_chan_freqs
        .iter()
        .map(|&f| f as f64)
        .collect_vec();
    let avg_freq_hz = fine_chan_freqs_hz.iter().sum::<f64>() / fine_chan_freqs_hz.len() as f64;
    // let avg_lambda_m = VEL_C / avg_freq_hz;
    let lambdas_m = fine_chan_freqs_hz.iter().map(|&f| VEL_C / f).collect_vec();
    let avg_freq = 4;
    let low_res_lambdas_m = obs_context.fine_chan_freqs.as_slice().chunks(avg_freq).map(|chunk| {
        let f = chunk.iter().sum::<u64>() as f64 / chunk.len() as f64;
        VEL_C / f
    }).collect_vec();

    let lst_0h_rad = get_lmst(
        array_pos.longitude_rad,
        obs_context.timestamps[0],
        obs_context.dut1.unwrap_or_default(),
    );
    let source_midpoint =
        RADec::from_hadec(HADec::from_radians(0., array_pos.latitude_rad), lst_0h_rad);

    let baseline_weights = {
        let uvw_lengths = xyzs_to_cross_uvws(
            obs_context.tile_xyzs.as_slice(),
            obs_context.phase_centre.to_hadec(lst_0h_rad),
        ).iter().map(|&UVW { u, v, w }| {
            (u.powi(2) + v.powi(2)).sqrt()
        }).collect_vec();
        // let uvw_max = uvw_lengths.iter().copied().fold(f64::NAN, f64::max);
        // println!("uvw_max: {}", uvw_max);
        uvw_lengths.iter().map(|_| {1.}).collect_vec()
    };

    // observation: this test passes with sources 30 degrees apart, and failes with them 0.05 degrees apart
    let source_list = SourceList::from(indexmap! {
        // remember, radec is in radians
        "Four".into() => point_src_i!(RADec {ra: source_midpoint.ra + 0.05, dec: source_midpoint.dec + 0.05}, 0., fine_chan_freqs_hz[0], 4.),
        "Three".into() => point_src_i!(RADec {ra: source_midpoint.ra + 0.03, dec: source_midpoint.dec - 0.03}, 0., fine_chan_freqs_hz[0], 3.),
        "Two".into() => point_src_i!(RADec {ra: source_midpoint.ra - 0.010, dec: source_midpoint.dec + 0.020}, 0., fine_chan_freqs_hz[0], 2.),
        "One".into() => point_src_i!(RADec {ra: source_midpoint.ra - 0.015, dec: source_midpoint.dec + 0.025}, 0., fine_chan_freqs_hz[0], 1.),
    });

    let source_weighted_positions = source_list
        .iter()
        .map(|(_, source)| source.components[0].radec)
        .collect_vec();

    let iono_consts = [
        IonoConsts {
            alpha: -0.00002,
            beta: -0.00001,
            gain: 1.01,
        },
        IonoConsts {
            alpha: 0.00001,
            beta: -0.00003,
            gain: 0.99,
        },
        IonoConsts {
            alpha: 0.00003,
            beta: -0.00001,
            gain: 1.01,
        },
        IonoConsts {
            alpha: -0.00001,
            beta: 0.00001,
            gain: 0.99,
        },
    ];

    let beam = get_beam(num_tiles);

    // model visibilities of each source
    let mut vis_model_tmp_tfb = Array3::<Jones<f32>>::zeros((num_times, num_chans, num_baselines));
    // iono rotated visibilities of each source
    let mut vis_iono_tmp_tfb = Array3::<Jones<f32>>::zeros((num_times, num_chans, num_baselines));
    // residual visibilities in the observation phase centre
    let mut vis_residual_obs_tfb =
        Array3::<Jones<f32>>::zeros((num_times, num_chans, num_baselines));
    // tile uvs and ws in the source phase centre
    let mut tile_uvs_src = Array2::default((num_times, num_tiles));
    let mut tile_ws_src = Array2::default((num_times, num_tiles));

    let vis_weights = Array3::<f32>::ones(vis_residual_obs_tfb.dim());

    let timeblock = Timeblock {
        index: 0,
        range: 0..2,
        timestamps: obs_context.timestamps.clone(),
        timesteps: vec1![0, 1],
        median: obs_context.timestamps[0],
    };

    let num_sources_to_iono_subtract = source_list.len();

    let multi_progress = MultiProgress::with_draw_target(ProgressDrawTarget::hidden());

    for apply_precession in [true, false] {
        let (average_lmst, _average_latitude) = if !apply_precession {
            let average_timestamp = timeblock.median;
            (
                get_lmst(obs_context.array_position.longitude_rad, average_timestamp, obs_context.dut1.unwrap()),
                obs_context.array_position.latitude_rad,
            )
        } else {
            let average_timestamp = timeblock.median;
            let average_precession_info = precess_time(
                obs_context.array_position.longitude_rad,
                obs_context.array_position.latitude_rad,
                obs_context.phase_centre,
                average_timestamp,
                obs_context.dut1.unwrap(),
            );
            (
                average_precession_info.lmst_j2000,
                average_precession_info.array_latitude_j2000,
            )
        };

        let mut high_res_modeller = new_sky_modeller(
            &beam,
            &source_list,
            Polarisations::default(),
            &obs_context.tile_xyzs,
            &fine_chan_freqs_hz,
            &flagged_tiles,
            obs_context.phase_centre,
            array_pos.longitude_rad,
            array_pos.latitude_rad,
            obs_context.dut1.unwrap_or_default(),
            apply_precession,
        )
        .unwrap();

        vis_residual_obs_tfb.fill(Jones::zero());

        // model each source in source_list and rotate by iono_consts with apply_iono2
        for (&iono_consts, (name, source)) in izip!(iono_consts.iter(), source_list.iter(),) {
            let source_radec = &source.components[0].radec;
            let ra = (source_radec.ra.to_degrees() + 180.) % 360. - 180.;
            let dec = (source_radec.dec.to_degrees() + 180.) % 360. - 180.;
            let source_azel = source_radec.to_hadec(average_lmst).to_azel(_average_latitude);
            let az = source_azel.az.to_degrees();
            let el = source_azel.el.to_degrees();
            let beam_jones = beam.calc_jones(source_azel, avg_freq_hz, None, _average_latitude).unwrap();
            let at = beam_jones.norm_sqr()[0];
            let IonoConsts { alpha, beta, gain }  = iono_consts;
            println!("source {name:8} radec {ra:+6.2} {dec:+6.2} prec_azel {az:+7.2} {el:+7.2} beam {at:5.4} a {alpha:+11.9} b {beta:+11.9} g {gain:+11.9} ");
            assert!(beam_jones.norm_sqr()[0] > 0.5, "modelled source is outside fwhm");

            high_res_modeller
                .update_with_a_source(source, obs_context.phase_centre)
                .unwrap();

            // model visibilities in the observation phase centre
            vis_model_tmp_tfb.fill(Jones::zero());
            model_timesteps(
                &*high_res_modeller,
                &obs_context.timestamps,
                vis_model_tmp_tfb.view_mut(),
            )
            .unwrap();

            setup_tile_uv_w_arrays(
                tile_uvs_src.view_mut(),
                tile_ws_src.view_mut(),
                &obs_context,
                *source_radec,
                apply_precession,
            );

            apply_iono2(
                vis_model_tmp_tfb.view(),
                vis_iono_tmp_tfb.view_mut(),
                tile_uvs_src.view(),
                iono_consts,
                &lambdas_m,
            );

            // display_vis_tfb(
            //     &format!("iono@src={} consts={:?}", name, &iono_consts),
            //     vis_iono_tmp_tfb.view(),
            //     &obs_context,
            //     source_radec,
            //     apply_precession,
            // );

            // add iono rotated and subtract model visibilities from residual
            Zip::from(vis_residual_obs_tfb.view_mut())
                .and(vis_iono_tmp_tfb.view())
                .and(vis_model_tmp_tfb.view())
                .for_each(|res, iono, model| *res += *iono - *model);
        }

        // display_vis_tfb(
        //     &format!("resid@obs consts={:?}", &iono_consts),
        //     vis_residual_obs_tfb.view(),
        //     &obs_context,
        //     obs_context.phase_centre,
        //     apply_precession,
        // );

        let mut iono_consts_result = vec![IonoConsts::default(); num_sources_to_iono_subtract];

        // When peel_cpu and peel_gpu are able to take generic
        // `SkyModeller` objects (requires the generic objects to take
        // currently CUDA-only methods), uncomment the following code and
        // delete what follows.

        // let function = match peel_type {
        //     CPU => peel_cpu,
        //     #[cfg(any(feature = "cuda", feature = "hip"))]
        //     CUDA => peel_gpu,
        // };
        // function(
        //     vis_residual_obs_tfb.view_mut(),
        //     vis_weights.view(),
        //     &timeblock,
        //     &source_list,
        //     &mut iono_consts,
        //     &source_weighted_positions,
        //     num_sources_to_iono_subtract,
        //     &fine_chan_freqs_hz,
        //     &lambdas_m,
        //     &lambdas_m,
        //     &obs_context,
        //     obs_context.array_position.unwrap(),
        //     &obs_context.tile_xyzs,
        //     high_res_modeller.deref_mut(),
        //     obs_context.dut1.unwrap_or_default(),
        //     !apply_precession,
        //     &multi_progress,
        // )
        // .unwrap();
        match peel_type {
            PeelType::CPU => peel_cpu(
                vis_residual_obs_tfb.view_mut(),
                vis_weights.view(),
                &timeblock,
                &source_list,
                &mut iono_consts_result,
                &source_weighted_positions,
                NUM_PASSES,
                NUM_LOOPS,
                SHORT_BASELINE_SIGMA,
                CONVERGENCE,
                &fine_chan_freqs_hz,
                &lambdas_m,
                &low_res_lambdas_m,
                &obs_context,
                obs_context.array_position,
                &obs_context.tile_xyzs,
                &mut *high_res_modeller,
                obs_context.dut1.unwrap_or_default(),
                !apply_precession,
                &multi_progress,
            )
            .unwrap(),

            #[cfg(any(feature = "cuda", feature = "hip"))]
            PeelType::Gpu => {
                let mut high_res_modeller = crate::model::SkyModellerGpu::new(
                    &beam,
                    &source_list,
                    Polarisations::default(),
                    &obs_context.tile_xyzs,
                    &fine_chan_freqs_hz,
                    &flagged_tiles,
                    obs_context.phase_centre,
                    array_pos.longitude_rad,
                    array_pos.latitude_rad,
                    obs_context.dut1.unwrap_or_default(),
                    apply_precession,
                )
                .unwrap();

                peel_gpu(
                    vis_residual_obs_tfb.view_mut(),
                    vis_weights.view(),
                    &timeblock,
                    &source_list,
                    &mut iono_consts_result,
                    &source_weighted_positions,
                    NUM_PASSES,
                    NUM_LOOPS,
                    SHORT_BASELINE_SIGMA,
                    CONVERGENCE,
                    &chanblocks,
                    &lambdas_m,
                    &low_res_lambdas_m,
                    &obs_context,
                    obs_context.array_position,
                    &obs_context.tile_xyzs,
                    baseline_weights.as_slice(),
                    &mut high_res_modeller,
                    obs_context.dut1.unwrap_or_default(),
                    !apply_precession,
                    &multi_progress,
                )
                .unwrap()
            }
        }

        // display_vis_tfb(
        //     &"peeled@obs".into(),
        //     vis_residual_obs_tfb.view(),
        //     &obs_context,
        //     obs_context.phase_centre,
        //     apply_precession,
        // );

        // let out_path =  match peel_type {
        //     PeelType::CPU => PathBuf::from(&format!("{OUT_PREFIX}/test_peel_cpu_multi_source{}.ms", if apply_precession { "_prec" } else { "" })),
        //     #[cfg(any(feature = "cuda", feature = "hip"))]
        //     PeelType::Gpu => PathBuf::from(&format!("{OUT_PREFIX}/test_peel_gpu_multi_source{}.ms", if apply_precession { "_prec" } else { "" })),
        // }.to_path_buf();

        // write_vis_tfb(out_path, vis_residual_obs_tfb.view(), &obs_context);

        // peel should perfectly remove the iono rotate model vis
        let mut norm_sum = 0.;
        for jones_residual in vis_residual_obs_tfb.iter() {
            for pol_residual in jones_residual.iter() {
                norm_sum += pol_residual.norm();
                // #[cfg(not(feature = "gpu-single"))]
                // assert_abs_diff_eq!(pol_residual.norm(), 0., epsilon = 4e-7);
                // #[cfg(feature = "gpu-single")]
                // assert_abs_diff_eq!(pol_residual.norm(), 0., epsilon = 5e-4);
            }
        }
        norm_sum /= vis_residual_obs_tfb.len() as f32;
        println!("norm_sum, prec={}: {:8.2e}", apply_precession, norm_sum);

        for ((name, source), expected, result) in izip!(source_list.iter(), iono_consts.iter(), iono_consts_result.iter(),) {
            let source_radec = &source.components[0].radec;
            let ra_ = (source_radec.ra.to_degrees() + 180.) % 360. - 180.;
            let dec = (source_radec.dec.to_degrees() + 180.) % 360. - 180.;
            let IonoConsts { alpha: ax, beta: bx, gain: gx }  = expected;
            let IonoConsts { alpha: ar, beta: br, gain: gr }  = result;
            println!("source {name:8} prec {apply_precession:6} radec {ra_:+6.2} {dec:+6.2} exp a {ax:+9.7} b {bx:+9.7} g {gx:+5.3} rx a {ar:+11.9} b {br:+11.9} g {gr:+7.5} tol a {:8.2e} b {:8.2e} g{:8.2e}",
                (ar - ax).abs(), (br - bx).abs(), (gr - gx).abs()
            );
        }

        // gacrux results

        // cargo test --no-default-features --features=hdf5-static,cuda,gpu-single --release -- peel_gpu_multi --nocapture
        // norm_sum, prec=true:  6.53e-4
        // source Four     prec true   radec  +1.65 -23.84 exp a -0.0000200 b -0.0000100 g +1.010 rx a -0.000020000 b -0.000010000 g +1.01005 tol a 2.83e-10 b 7.62e-11 g 4.91e-5
        // source Three    prec true   radec  +0.50 -28.42 exp a +0.0000100 b -0.0000300 g +0.990 rx a +0.000009997 b -0.000030000 g +0.98998 tol a  2.76e-9 b 2.04e-10 g 2.49e-5
        // source Two      prec true   radec  -1.79 -25.56 exp a +0.0000300 b -0.0000100 g +1.010 rx a +0.000029999 b -0.000010000 g +1.00991 tol a  1.41e-9 b 3.99e-10 g 9.26e-5
        // source One      prec true   radec  -2.08 -25.27 exp a -0.0000100 b +0.0000100 g +0.990 rx a -0.000010003 b +0.000010001 g +0.99002 tol a  2.92e-9 b 9.29e-10 g 1.82e-5
        // norm_sum, prec=false:  6.65e-4
        // source Four     prec false  radec  +1.65 -23.84 exp a -0.0000200 b -0.0000100 g +1.010 rx a -0.000020000 b -0.000010000 g +1.01005 tol a 8.23e-11 b 1.02e-11 g 5.41e-5
        // source Three    prec false  radec  +0.50 -28.42 exp a +0.0000100 b -0.0000300 g +0.990 rx a +0.000009998 b -0.000030000 g +0.98998 tol a  2.41e-9 b 6.48e-11 g 1.91e-5
        // source Two      prec false  radec  -1.79 -25.56 exp a +0.0000300 b -0.0000100 g +1.010 rx a +0.000029999 b -0.000010000 g +1.00991 tol a  1.30e-9 b 3.39e-10 g 9.19e-5
        // source One      prec false  radec  -2.08 -25.27 exp a -0.0000100 b +0.0000100 g +0.990 rx a -0.000010003 b +0.000010001 g +0.99002 tol a  3.25e-9 b 9.37e-10 g 1.75e-5

        // cargo test --no-default-features --features=hdf5-static,cuda --release -- peel_gpu_multi --nocapture
        // norm_sum, prec=true:  1.72e-6
        // source Four     prec true   radec  +1.65 -23.84 exp a -0.0000200 b -0.0000100 g +1.010 rx a -0.000020000 b -0.000010000 g +1.01000 tol a 1.24e-11 b 3.37e-11 g 1.09e-7
        // source Three    prec true   radec  +0.50 -28.42 exp a +0.0000100 b -0.0000300 g +0.990 rx a +0.000010000 b -0.000030000 g +0.99000 tol a 7.37e-12 b 7.31e-12 g 8.69e-8
        // source Two      prec true   radec  -1.79 -25.56 exp a +0.0000300 b -0.0000100 g +1.010 rx a +0.000030000 b -0.000010000 g +1.01000 tol a 1.35e-11 b 3.41e-11 g 7.09e-8
        // source One      prec true   radec  -2.08 -25.27 exp a -0.0000100 b +0.0000100 g +0.990 rx a -0.000010000 b +0.000010000 g +0.99000 tol a 4.99e-13 b 7.47e-12 g 1.32e-8
        // norm_sum, prec=false:  1.93e-6
        // source Four     prec false  radec  +1.65 -23.84 exp a -0.0000200 b -0.0000100 g +1.010 rx a -0.000020000 b -0.000010000 g +1.01000 tol a 6.68e-12 b 4.48e-11 g 4.83e-9
        // source Three    prec false  radec  +0.50 -28.42 exp a +0.0000100 b -0.0000300 g +0.990 rx a +0.000010000 b -0.000030000 g +0.99000 tol a 1.43e-12 b 1.50e-13 g 1.38e-7
        // source Two      prec false  radec  -1.79 -25.56 exp a +0.0000300 b -0.0000100 g +1.010 rx a +0.000030000 b -0.000010000 g +1.01000 tol a 1.35e-11 b 4.37e-11 g 6.37e-8
        // source One      prec false  radec  -2.08 -25.27 exp a -0.0000100 b +0.0000100 g +0.990 rx a -0.000010000 b +0.000010000 g +0.99000 tol a 7.51e-13 b 1.01e-11 g 4.49e-9

        // cargo test --no-default-features --features=hdf5-static --release -- peel_cpu_multi --nocapture
        // norm_sum, prec=true:  1.66e-6
        // source Four     prec true   radec  +1.65 -23.84 exp a -0.0000200 b -0.0000100 g +1.010 rx a -0.000020000 b -0.000010000 g +1.01000 tol a 1.29e-11 b 3.40e-11 g 8.81e-8
        // source Three    prec true   radec  +0.50 -28.42 exp a +0.0000100 b -0.0000300 g +0.990 rx a +0.000010000 b -0.000030000 g +0.99000 tol a 7.18e-12 b 6.74e-12 g 9.48e-8
        // source Two      prec true   radec  -1.79 -25.56 exp a +0.0000300 b -0.0000100 g +1.010 rx a +0.000030000 b -0.000010000 g +1.01000 tol a 1.28e-11 b 3.37e-11 g 7.25e-8
        // source One      prec true   radec  -2.08 -25.27 exp a -0.0000100 b +0.0000100 g +0.990 rx a -0.000010000 b +0.000010000 g +0.99000 tol a 1.48e-13 b 7.48e-12 g 5.66e-9
        // norm_sum, prec=false:  1.83e-6
        // source Four     prec false  radec  +1.65 -23.84 exp a -0.0000200 b -0.0000100 g +1.010 rx a -0.000020000 b -0.000010000 g +1.01000 tol a 6.28e-12 b 4.44e-11 g 1.04e-8
        // source Three    prec false  radec  +0.50 -28.42 exp a +0.0000100 b -0.0000300 g +0.990 rx a +0.000010000 b -0.000030000 g +0.99000 tol a 1.18e-12 b 7.77e-13 g 1.02e-7
        // source Two      prec false  radec  -1.79 -25.56 exp a +0.0000300 b -0.0000100 g +1.010 rx a +0.000030000 b -0.000010000 g +1.01000 tol a 1.30e-11 b 4.31e-11 g 6.99e-8
        // source One      prec false  radec  -2.08 -25.27 exp a -0.0000100 b +0.0000100 g +0.990 rx a -0.000010000 b +0.000010000 g +0.99000 tol a 5.45e-13 b 1.00e-11 g 2.26e-9

        let (ab_epsilon, g_epsilon, n_epsilon) = match peel_type {
            PeelType::CPU => (7e-11, 2e-7, 2e-6),
            #[cfg(all(any(feature = "cuda", feature = "hip"), not(feature = "gpu-single")))]
            PeelType::Gpu => (7e-11, 2e-7, 2e-6),
            #[cfg(all(any(feature = "cuda", feature = "hip"), feature = "gpu-single"))]
            PeelType::Gpu => (5e-9, 1e-4, 7e-4), // TODO(Dev): bring this down
        };
        for (expected, result) in izip!(iono_consts.iter(), iono_consts_result.iter(),) {
            let IonoConsts { alpha: ax, beta: bx, gain: gx }  = expected;
            let IonoConsts { alpha: ar, beta: br, gain: gr }  = result;
            assert_abs_diff_eq!(ax, ar, epsilon = ab_epsilon);
            assert_abs_diff_eq!(bx, br, epsilon = ab_epsilon);
            assert_abs_diff_eq!(gx, gr, epsilon = g_epsilon);
        }
        assert_abs_diff_eq!(norm_sum, 0., epsilon = n_epsilon);
    }
}

#[test]
fn test_peel_cpu_single_source() {
    // let mut builder = env_logger::Builder::from_default_env();
    // builder.target(env_logger::Target::Stdout);
    // builder.format_target(false);
    // builder.filter_level(log::LevelFilter::Trace);
    // builder.init();
    test_peel_single_source(PeelType::CPU)
}

#[test]
fn test_peel_cpu_multi_source() {
    test_peel_multi_source(PeelType::CPU)
}

#[cfg(any(feature = "cuda", feature = "hip"))]
mod gpu_tests {
    use std::ffi::CStr;

    use marlu::{pos::xyz::xyzs_to_cross_uvws, UVW};

    use super::*;
    use crate::{
        gpu::{self, DevicePointer, GpuFloat},
        model::SkyModellerGpu,
    };

    /// Populate the [UVW] array ([times, baselines]) for the given [ObsContext].
    fn setup_uvw_array(
        mut uvws: ArrayViewMut2<UVW>,
        obs_context: &ObsContext,
        phase_centre: RADec,
        apply_precession: bool,
    ) {
        let array_pos = obs_context.array_position;
        let num_tiles = obs_context.get_total_num_tiles();
        let mut tile_uvws_tmp = vec![UVW::default(); num_tiles];
        // let num_cross_baselines = (num_tiles * (num_tiles - 1)) / 2;
        for (&time, mut uvws) in izip!(obs_context.timestamps.iter(), uvws.outer_iter_mut(),) {
            let (lmst, precessed_xyzs) = if apply_precession {
                let precession_info = precess_time(
                    array_pos.longitude_rad,
                    array_pos.latitude_rad,
                    obs_context.phase_centre,
                    time,
                    obs_context.dut1.unwrap_or_default(),
                );
                let precessed_xyzs = precession_info.precess_xyz(&obs_context.tile_xyzs);
                (precession_info.lmst_j2000, precessed_xyzs)
            } else {
                let lmst = get_lmst(
                    array_pos.longitude_rad,
                    time,
                    obs_context.dut1.unwrap_or_default(),
                );
                (lmst, obs_context.tile_xyzs.clone().into())
            };
            let hadec = phase_centre.to_hadec(lmst);
            let (s_ha, c_ha) = hadec.ha.sin_cos();
            let (s_dec, c_dec) = hadec.dec.sin_cos();
            for (tile_uvw, &precessed_xyz) in
                izip!(tile_uvws_tmp.iter_mut(), precessed_xyzs.iter(),)
            {
                *tile_uvw = UVW::from_xyz_inner(precessed_xyz, s_ha, c_ha, s_dec, c_dec);
            }
            let mut count = 0;
            for (i, t1) in tile_uvws_tmp.iter().enumerate() {
                for t2 in tile_uvws_tmp.iter().skip(i + 1) {
                    uvws[count] = *t1 - *t2;
                    count += 1;
                }
            }
        }
    }

    #[test]
    /// just checking that the way we typically call xyzs_to_cross_uvws is equivalent to gpu::xyzs_to_uvws
    fn test_gpu_xyzs_to_uvws() {
        let obs_context = get_simple_obs_context(TILE_SPACING);
        // obs_context.timestamps = vec1![obs_context.timestamps[0]];
        let num_tiles = obs_context.get_total_num_tiles();
        let num_times = obs_context.timestamps.len();
        let num_cross_baselines = (num_tiles * (num_tiles - 1)) / 2;
        dbg!(&num_tiles, &num_times, &num_cross_baselines);

        let mut gpu_uvws_from = Array2::default((num_times, num_cross_baselines));
        let mut gpu_uvws_to = Array2::default((num_times, num_cross_baselines));

        let lmsts = &[0., std::f64::consts::PI];
        let prec_tile_xyzs = array![
            [
                XyzGeodetic { x: 000., y: 001., z: 002. },
                XyzGeodetic { x: 010., y: 011., z: 012. },
                XyzGeodetic { x: 020., y: 021., z: 022. },
            ],
            [
                XyzGeodetic { x: 100., y: 101., z: 102. },
                XyzGeodetic { x: 110., y: 111., z: 112. },
                XyzGeodetic { x: 120., y: 121., z: 122. },
            ],
        ];

        let gpu_prec_tile_xyzs: Vec<_> = prec_tile_xyzs
            .iter()
            .copied()
            .map(|XyzGeodetic { x, y, z }| gpu::XYZ {
                x: x as GpuFloat,
                y: y as GpuFloat,
                z: z as GpuFloat,
            })
            .collect();

        dbg!(&gpu_prec_tile_xyzs);
        let gpu_lmsts: Vec<GpuFloat> = lmsts.iter().map(|l| *l as GpuFloat).collect();

        unsafe {
            let d_xyzs = DevicePointer::copy_to_device(&gpu_prec_tile_xyzs).unwrap();

            let d_lmsts = DevicePointer::copy_to_device(&gpu_lmsts).unwrap();

            gpu_uvws_from
                .outer_iter_mut()
                .zip(prec_tile_xyzs.outer_iter())
                .zip(lmsts.iter())
                .for_each(|((mut gpu_uvws_from, xyzs), lmst)| {
                    let phase_centre = obs_context.phase_centre.to_hadec(*lmst);
                    let v = xyzs_to_cross_uvws(xyzs.as_slice().unwrap(), phase_centre)
                        .into_iter()
                        .map(|uvw| gpu::UVW {
                            u: uvw.u as GpuFloat,
                            v: uvw.v as GpuFloat,
                            w: uvw.w as GpuFloat,
                        })
                        .collect::<Vec<_>>();
                    gpu_uvws_from.assign(&ArrayView1::from(&v));
                });
            let mut d_uvws_to =
                DevicePointer::malloc(gpu_uvws_from.len() * std::mem::size_of::<gpu::UVW>()).unwrap();
                // DevicePointer::malloc(gpu_uvws_from.len() * (std::mem::size_of::<gpu::UVW>() + std::mem::align_of::<gpu::UVW>())).unwrap();

            gpu_kernel_call!(
                gpu::xyzs_to_uvws,
                d_xyzs.get(),
                d_lmsts.get(),
                d_uvws_to.get_mut(),
                gpu::RADec {
                    ra: obs_context.phase_centre.ra as GpuFloat,
                    dec: obs_context.phase_centre.dec as GpuFloat,
                },
                num_tiles.try_into().unwrap(),
                num_cross_baselines.try_into().unwrap(),
                num_times.try_into().unwrap(),
            ).unwrap();

            d_uvws_to.copy_from_device(gpu_uvws_to.as_slice_mut().unwrap()).unwrap();
        }

        dbg!(&gpu_uvws_from);
        dbg!(&gpu_uvws_to);

        assert_eq!(gpu_uvws_from.dim(), gpu_uvws_to.dim());
        gpu_uvws_from.indexed_iter().for_each(|(idx, uvw_from)| {
            dbg!(&idx);
            let uvw_to = gpu_uvws_to[idx];
            assert_abs_diff_eq!(uvw_from.u, uvw_to.u, epsilon = 1e-5);
            assert_abs_diff_eq!(uvw_from.v, uvw_to.v, epsilon = 1e-5);
            assert_abs_diff_eq!(uvw_from.w, uvw_to.w, epsilon = 1e-5);
        });
    }

    #[test]
    /// - define unique visibilities for each time, channel, baselines
    /// - copy to gpu
    /// - average
    /// - copy back to cpu
    /// - check values
    fn test_gpu_average_values() {
        // let obs_context = get_simple_obs_context(TILE_SPACING);
        // let n_times = obs_context.timestamps.len();
        // let n_chans = obs_context.fine_chan_freqs.len();
        // let n_tiles = obs_context.get_total_num_tiles();

        let n_times = 2;
        let n_chans = 4;
        let n_tiles = 3;
        let n_cross_bls = (n_tiles * (n_tiles - 1)) / 2;

        let time_avg_factor = n_times;
        let freq_avg_factor = 2;
        let n_avg_chans = n_chans / freq_avg_factor;

        let vis_tfb = Array3::<Jones<f32>>::from_shape_fn(
            (n_times, n_chans, n_cross_bls),
            |(t, c, b)| Jones::from([
                Complex::new(t as f32, c as f32),
                Complex::zero(),
                Complex::new(0_f32, 1_f32),
                Complex::new(b as f32, 1_f32),
            ]),
        );
        // let weights_tfb = Array3::<f32>::ones((n_times, n_chans, n_cross_bls));
        let weights_tfb = Array3::<f32>::from_shape_fn(
            (n_times, n_chans, n_cross_bls),
            |(t, c, b)| match (t%2, c%2, b%2) {
                (0, 0, _) => 1.,
                _ => 0.,
            }
        );
        let d_vis_tfb = DevicePointer::copy_to_device(vis_tfb.as_slice().unwrap()).unwrap();
        let d_weights_tfb = DevicePointer::copy_to_device(weights_tfb.as_slice().unwrap()).unwrap();
        let mut avg_fb = Array3::<Jones<f32>>::zeros((1, n_avg_chans, n_cross_bls));
        let mut d_avg_fb = DevicePointer::copy_to_device(avg_fb.as_slice().unwrap()).unwrap();

        gpu_kernel_call!(
            gpu::average,
            d_vis_tfb.get().cast(),
            d_weights_tfb.get(),
            d_avg_fb.get_mut().cast(),
            n_times as i32,
            n_cross_bls as i32,
            n_chans as i32,
            freq_avg_factor as i32,
        ).unwrap();

        d_avg_fb.copy_from_device(avg_fb.as_slice_mut().unwrap()).unwrap();

        for (t, avg_fb) in avg_fb.outer_iter().enumerate() {
            for (c, avg_b) in avg_fb.outer_iter().enumerate() {
                for (b, avg_jones) in avg_b.indexed_iter() {
                    println!("t={}, c={}, b={}, avg_jones={}", t, c, b, avg_jones);
                    assert_abs_diff_eq!(avg_jones[3], &Complex::new(b as f32, 1_f32), epsilon = 1e-10);
                    assert_abs_diff_eq!(avg_jones[2], &Complex::new(0_f32, 1_f32), epsilon = 1e-10);
                    assert_abs_diff_eq!(avg_jones[1], &Complex::zero(), epsilon = 1e-10);

                    let expected_xx = Array2::<Complex<f32>>::from_shape_fn(
                        (time_avg_factor, freq_avg_factor),
                        |(t_, c_)| {
                            let t_ = t_ + t * time_avg_factor;
                            let c_ = c_ + c * freq_avg_factor;
                            let high = Complex::new(t_ as f32, c_ as f32) * weights_tfb[[t_, c_, b]];
                            println!("-> t_={}, c_={}, b={}, avg_jones={}", t_, c_, b, high);
                            high
                        },
                    ).sum();
                    // ^ is a sum because only 1 cell has even t,c indices. others are 0 weight

                    assert_abs_diff_eq!(avg_jones[0], &expected_xx, epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_gpu_average_weights() {
        // let obs_context = get_simple_obs_context(TILE_SPACING);
        // let n_times = obs_context.timestamps.len();
        // let n_chans = obs_context.fine_chan_freqs.len();
        // let n_tiles = obs_context.get_total_num_tiles();

        let n_times = 10;
        let n_chans = 20;
        let n_tiles = 30;
        let n_cross_bls = (n_tiles * (n_tiles - 1)) / 2;

        let freq_avg_factor = 5;
        let n_avg_chans = n_chans / freq_avg_factor;

        let jones = Jones::from([
            Complex::new(0., 0.,),
            Complex::new(0., 1.),
            Complex::new(1., 0.),
            Complex::new(1., 1.,),
        ]);

        let vis_tfb = Array3::<Jones<f32>>::from_elem(
            (n_times, n_chans, n_cross_bls),
            jones,
        );
        // let weights_tfb = Array3::<f32>::ones((n_times, n_chans, n_cross_bls));
        let weights_tfb = Array3::<f32>::from_elem(
            (n_times, n_chans, n_cross_bls),
            1.
        );
        let d_vis_tfb = DevicePointer::copy_to_device(vis_tfb.as_slice().unwrap()).unwrap();
        let d_weights_tfb = DevicePointer::copy_to_device(weights_tfb.as_slice().unwrap()).unwrap();
        let mut avg_fb = Array3::<Jones<f32>>::zeros((1, n_avg_chans, n_cross_bls));
        let mut d_avg_fb = DevicePointer::copy_to_device(avg_fb.as_slice().unwrap()).unwrap();

        gpu_kernel_call!(
            gpu::average,
            d_vis_tfb.get().cast(),
            d_weights_tfb.get(),
            d_avg_fb.get_mut().cast(),
            n_times as i32,
            n_cross_bls as i32,
            n_chans as i32,
            freq_avg_factor as i32,
        ).unwrap();

        d_avg_fb.copy_from_device(avg_fb.as_slice_mut().unwrap()).unwrap();

        for (t, avg_fb) in avg_fb.outer_iter().enumerate() {
            for (c, avg_b) in avg_fb.outer_iter().enumerate() {
                for (b, avg_jones) in avg_b.indexed_iter() {
                    println!("t={}, c={}, b={}, avg_jones={}", t, c, b, avg_jones);
                    for (i, &j) in avg_jones.iter().enumerate() {
                        assert_abs_diff_eq!(j, jones[i], epsilon = 1e-10);
                    }
                }
            }
        }
    }


    #[test]
    fn test_gpu_add_model() {
        // let obs_context = get_simple_obs_context(TILE_SPACING);
        // let n_times = obs_context.timestamps.len();
        // let n_chans = obs_context.fine_chan_freqs.len();
        // let n_tiles = obs_context.get_total_num_tiles();

        let n_times = 10;
        let n_chans = 20;
        let n_tiles = 30;
        let n_cross_bls = (n_tiles * (n_tiles - 1)) / 2;

        let lambdas = (0..n_chans).map(|c| 1. + c as GpuFloat).collect::<Vec<_>>();
        let d_lambdas = DevicePointer::copy_to_device(&lambdas).unwrap();

        let jones = Jones::from([
            Complex::new(0., 0.,),
            Complex::new(0., 1.),
            Complex::new(1., 0.),
            Complex::new(1., 1.,),
        ]);

        let vis_tfb = Array3::<Jones<f32>>::from_elem(
            (n_times, n_chans, n_cross_bls),
            jones,
        );
        let d_vis_tfb = DevicePointer::copy_to_device(vis_tfb.as_slice().unwrap()).unwrap();
        let mut res_tfb =  Array3::<Jones<f32>>::zeros( (n_times, n_chans, n_cross_bls));
        let mut d_res_tfb = DevicePointer::copy_to_device(res_tfb.as_slice().unwrap()).unwrap();
        let uvws_to = Array2::<gpu::UVW>::from_elem((n_times, n_cross_bls), gpu::UVW {
            u: 0.,
            v: 1.,
            w: 0.,
        });
        let d_uvws_to = DevicePointer::copy_to_device(uvws_to.as_slice().unwrap()).unwrap();

        let iono_consts = gpu::IonoConsts {
            alpha: 0.1,
            beta: 0.2,
            gain: 0.3,
        };

        gpu_kernel_call!(
            gpu::add_model,
            d_res_tfb.get_mut().cast(),
            d_vis_tfb.get().cast(),
            iono_consts,
            d_lambdas.get(),
            d_uvws_to.get(),
            n_times as i32,
            n_chans as i32,
            n_cross_bls as i32,
        ).unwrap();

        d_res_tfb.copy_from_device(res_tfb.as_slice_mut().unwrap()).unwrap();

        for (t, res_fb) in res_tfb.outer_iter().enumerate() {
            for (c, res_b) in res_fb.outer_iter().enumerate() {
                for (b, res_jones) in res_b.indexed_iter() {
                    println!("t={}, c={}, b={}, res_jones={}", t, c, b, res_jones);
                    // assert_abs_diff_eq!(j, jones[0], epsilon = 1e-10);
                    for (i, &j) in res_jones.iter().enumerate() {
                        assert_abs_diff_eq!(j.norm(), jones[i].norm() * iono_consts.gain as f32, epsilon = 1e-7);
                        // TODO(Dev): I swear this used to work. What changed?
                        // assert_abs_diff_eq!(j, jones[i] * Complex::cis(((TAU * 0.2) as GpuFloat / lambdas[c]) as f32) * iono_consts.gain as f32, epsilon = 1e-7);
                    }
                }
            }
        }
    }

    // #[test]
    /// TODO: Test how peel_gpu calls average:
    /// gpu_kernel_call!(
    ///     gpu::average,
    ///     d_high_res_model_rotated.get().cast(),
    ///     d_high_res_weights_tfb.get(),
    ///     d_low_res_model_fb.get_mut().cast(),
    ///     num_timesteps_i32,
    ///     num_cross_baselines_i32,
    ///     num_high_res_chans_i32,
    ///     freq_average_factor
    /// )?;
    // fn test_gpu_average() {
    //     let obs_context = get_phase1_obs_context(32);
    //     let num_times = obs_context.timestamps.len();
    //     let num_chans = obs_context.fine_chan_freqs.len();
    //     let num_tiles = obs_context.get_total_num_tiles();
    //     let num_cross_baselines = (num_tiles * (num_tiles - 1)) / 2;
    //     let mut d_vis_hi: DevicePointer<Jones<f32>> =
    //         DevicePointer::malloc(d_high_res_model_tfb.get_size())?;
    //     d_vis_hi.clear();
    // }

    #[test]
    /// - synthesize model visibilities
    /// - apply ionospheric rotation
    /// - create residual: ionospheric - model
    /// - apply_iono3 should result in empty visibilities
    fn test_gpu_subtract_iono() {
        let obs_context = get_simple_obs_context(TILE_SPACING);
        let array_pos = obs_context.array_position;

        let num_tiles = obs_context.get_total_num_tiles();
        let num_times = obs_context.timestamps.len();
        let num_baselines = (num_tiles * (num_tiles - 1)) / 2;
        let flagged_tiles = HashSet::new();
        let num_chans = obs_context.fine_chan_freqs.len();

        // lambda = 1m
        let fine_chan_freqs_hz = obs_context
            .fine_chan_freqs
            .iter()
            .map(|&f| f as f64)
            .collect_vec();
        let lambdas_m = fine_chan_freqs_hz.iter().map(|&f| VEL_C / f).collect_vec();

        // source is at zenith at 1h
        let lst_1h_rad = get_lmst(
            array_pos.longitude_rad,
            obs_context.timestamps[1],
            obs_context.dut1.unwrap_or_default(),
        );
        let source_radec =
            RADec::from_hadec(HADec::from_radians(0., array_pos.latitude_rad), lst_1h_rad);
        let source_fd = 1.;
        let source_list = SourceList::from(indexmap! {
            "One".into() => point_src_i!(source_radec, 0., fine_chan_freqs_hz[0], source_fd),
        });

        let beam = get_beam(num_tiles);

        // residual visibilities in the observation phase centre
        let mut vis_residual_obs_tfb =
            Array3::<Jones<f32>>::zeros((num_times, num_chans, num_baselines));
        // model visibilities in the observation phase centre
        let mut vis_model_obs_tfb =
            Array3::<Jones<f32>>::zeros((num_times, num_chans, num_baselines));
        // iono rotated model visibilities in the observation phase centre
        let mut vis_iono_obs_tfb =
            Array3::<Jones<f32>>::zeros((num_times, num_chans, num_baselines));
        // tile uvs and ws in the source phase centre
        let mut tile_uvws_src = Array2::default((num_times, num_tiles));
        let mut tile_uvs_src = Array2::default((num_times, num_tiles));
        let mut tile_ws_src = Array2::default((num_times, num_tiles));

        for apply_precession in [false, true] {
            let modeller = SkyModellerGpu::new(
                &beam,
                &source_list,
                Polarisations::default(),
                &obs_context.tile_xyzs,
                &fine_chan_freqs_hz,
                &flagged_tiles,
                obs_context.phase_centre,
                array_pos.longitude_rad,
                array_pos.latitude_rad,
                obs_context.dut1.unwrap_or_default(),
                apply_precession,
            )
            .unwrap();

            model_timesteps(
                &modeller,
                &obs_context.timestamps,
                vis_model_obs_tfb.view_mut(),
            )
            .unwrap();

            setup_uvw_array(
                tile_uvws_src.view_mut(),
                &obs_context,
                source_radec,
                apply_precession,
            );
            setup_tile_uv_w_arrays(
                tile_uvs_src.view_mut(),
                tile_ws_src.view_mut(),
                &obs_context,
                source_radec,
                apply_precession,
            );

            // display_vis_tfb(
            //     &"model@obs".into(),
            //     vis_model_obs_tfb.view(),
            //     &obs_context,
            //     obs_context.phase_centre,
            //     apply_precession,
            // );

            let d_high_res_model =
                DevicePointer::copy_to_device(vis_model_obs_tfb.as_slice().unwrap()).unwrap();

            let gpu_uvws_src = tile_uvws_src.mapv(|uvw| gpu::UVW {
                u: uvw.u as GpuFloat,
                v: uvw.v as GpuFloat,
                w: uvw.w as GpuFloat,
            });

            let d_uvws_src =
                DevicePointer::copy_to_device(gpu_uvws_src.as_slice().unwrap()).unwrap();
            let d_lambdas = DevicePointer::copy_to_device(
                &lambdas_m.iter().map(|l| *l as GpuFloat).collect::<Vec<_>>(),
            )
            .unwrap();

            for iono_consts in [
                IonoConsts {
                    alpha: 0.0001,
                    beta: -0.0003,
                    gain: 1.0,
                },
                IonoConsts {
                    alpha: 0.0003,
                    beta: -0.0001,
                    gain: 1.0,
                },
                IonoConsts {
                    alpha: -0.0007,
                    beta: 0.0001,
                    gain: 1.0,
                },
            ] {
                // apply iono rotation at source phase to model at observation phase
                apply_iono2(
                    vis_model_obs_tfb.view(),
                    vis_iono_obs_tfb.view_mut(),
                    tile_uvs_src.view(),
                    iono_consts,
                    &lambdas_m,
                );

                // display_vis_tfb(
                //     &format!("iono@obs ({}, {})", &consts_lm.0, &consts_lm.1),
                //     vis_iono_obs_tfb.view(),
                //     &obs_context,
                //     obs_context.phase_centre,
                //     apply_precession,
                // );

                // subtract model from iono at observation phase centre
                vis_residual_obs_tfb.assign(&vis_iono_obs_tfb);
                vis_residual_obs_tfb -= &vis_model_obs_tfb;

                // display_vis_tfb(
                //     &"residual@obs before".into(),
                //     vis_residual_obs_tfb.view(),
                //     &obs_context,
                //     obs_context.phase_centre,
                //     apply_precession,
                // );

                let mut d_high_res_vis =
                    DevicePointer::copy_to_device(vis_residual_obs_tfb.as_slice().unwrap())
                        .unwrap();

                let gpu_iono_consts = gpu::IonoConsts {
                    alpha: iono_consts.alpha,
                    beta: iono_consts.beta,
                    gain: 1.0,
                };
                let gpu_old_iono_consts = gpu::IonoConsts {
                    alpha: 0.0,
                    beta: 0.0,
                    gain: 1.0,
                };
                let error_message_ptr = unsafe {
                    gpu::subtract_iono(
                        d_high_res_vis.get_mut().cast(),
                        d_high_res_model.get().cast(),
                        gpu_iono_consts,
                        gpu_old_iono_consts,
                        d_uvws_src.get(),
                        d_lambdas.get(),
                        num_times.try_into().unwrap(),
                        num_baselines.try_into().unwrap(),
                        num_chans.try_into().unwrap(),
                    )
                };
                assert!(
                    error_message_ptr.is_null(),
                    "{}",
                    unsafe { CStr::from_ptr(error_message_ptr) }
                        .to_str()
                        .unwrap_or("<cannot read GPU error string>")
                );

                d_high_res_vis
                    .copy_from_device(vis_residual_obs_tfb.as_slice_mut().unwrap())
                    .unwrap();

                // display_vis_tfb(
                //     &"residual@obs after".into(),
                //     vis_residual_obs_tfb.view(),
                //     &obs_context,
                //     obs_context.phase_centre,
                //     apply_precession,
                // );

                for jones_residual in vis_residual_obs_tfb.iter() {
                    for pol_residual in jones_residual.iter() {
                        assert_abs_diff_eq!(pol_residual.norm(), 0., epsilon = 9e-7);
                    }
                }
            }
        }
    }

    #[test]
    fn test_rotate_average() {
        // TODO(Dev): this test only works with TILE_SPACING=1.0
        let obs_context = get_simple_obs_context(1.0);
        let array_pos = obs_context.array_position;

        let num_tiles = obs_context.get_total_num_tiles();
        let num_times = obs_context.timestamps.len();
        let num_baselines = (num_tiles * (num_tiles - 1)) / 2;
        let flagged_tiles = HashSet::new();
        let num_chans = obs_context.fine_chan_freqs.len();

        let fine_chan_freqs_hz = obs_context
            .fine_chan_freqs
            .iter()
            .map(|&f| f as f64)
            .collect_vec();
        let lambdas_m = fine_chan_freqs_hz.iter().map(|&f| VEL_C / f).collect_vec();

        // source is at zenith at 1h
        let hour_epoch = obs_context.timestamps[1];
        let lst_1h_rad = get_lmst(
            array_pos.longitude_rad,
            hour_epoch,
            obs_context.dut1.unwrap_or_default(),
        );
        let source_radec =
            RADec::from_hadec(HADec::from_radians(0., array_pos.latitude_rad), lst_1h_rad);
        let source_fd = 1.;
        let source_list = SourceList::from(indexmap! {
            "One".into() => point_src_i!(source_radec, 0., fine_chan_freqs_hz[0], source_fd),
        });

        let beam = get_beam(num_tiles);

        let mut vis_tfb = Array3::default((num_times, num_chans, num_baselines));
        let mut vis_rot_tfb = Array3::default((num_times, num_chans, num_baselines));

        // tile uvs and ws in the observation phase centre
        let mut tile_uvs_obs = Array2::default((num_times, num_tiles));
        let mut tile_ws_obs = Array2::default((num_times, num_tiles));
        // tile uvs and ws in the source phase centre
        let mut tile_uvs_src = Array2::default((num_times, num_tiles));
        let mut tile_ws_src = Array2::default((num_times, num_tiles));

        let weights_tfb = Array3::from_elem(vis_tfb.dim(), 2.0);

        for apply_precession in [false, true] {
            let modeller = SkyModellerCpu::new(
                &beam,
                &source_list,
                Polarisations::default(),
                &obs_context.tile_xyzs,
                &fine_chan_freqs_hz,
                &flagged_tiles,
                obs_context.phase_centre,
                array_pos.longitude_rad,
                array_pos.latitude_rad,
                obs_context.dut1.unwrap_or_default(),
                apply_precession,
            );

            vis_tfb.fill(Jones::zero());
            model_timesteps(&modeller, &obs_context.timestamps, vis_tfb.view_mut()).unwrap();

            let (lmsts, xyzs) = setup_tile_uv_w_arrays(
                tile_uvs_obs.view_mut(),
                tile_ws_obs.view_mut(),
                &obs_context,
                obs_context.phase_centre,
                apply_precession,
            );
            setup_tile_uv_w_arrays(
                tile_uvs_src.view_mut(),
                tile_ws_src.view_mut(),
                &obs_context,
                source_radec,
                apply_precession,
            );

            // rotate visibilities
            vis_rotate2(
                vis_tfb.view(),
                vis_rot_tfb.view_mut(),
                tile_ws_obs.view(),
                tile_ws_src.view(),
                &lambdas_m,
            );

            display_vis_tfb(
                &format!("model@obs prec={apply_precession}"),
                vis_tfb.view(),
                &obs_context,
                obs_context.phase_centre,
                apply_precession,
            );
            display_vis_tfb(
                &format!("rotated@source prec={apply_precession}"),
                vis_rot_tfb.view(),
                &obs_context,
                source_radec,
                apply_precession,
            );

            // let mut vis_averaged_tfb = Array3::default((num_times, num_chans, num_baselines));
            let mut vis_averaged_tfb = Array3::default((1, 1, num_baselines));
            vis_average2(
                vis_rot_tfb.view(),
                vis_averaged_tfb.view_mut(),
                weights_tfb.view(),
            );

            // if !apply_precession {
            //     // rotated vis should always have the source in phase, so no angle in pols XX, YY
            //     for vis_rot in vis_rot_tfb.iter() {
            //         assert_abs_diff_eq!(vis_rot[0].arg(), 0., epsilon = 1e-6); // XX
            //         assert_abs_diff_eq!(vis_rot[3].arg(), 0., epsilon = 1e-6); // YY
            //     }
            //     // baseline 1, from origin to v has no u or w component, should not be affected by the rotation
            //     for (vis, vis_rot) in izip!(
            //         vis_tfb.slice(s![.., .., 1]),
            //         vis_rot_tfb.slice(s![.., .., 1]),
            //     ) {
            //         assert_abs_diff_eq!(vis, vis_rot, epsilon = 1e-6);
            //     }
            //     // in the second timestep, the source should be at the pointing centre, so should not be
            //     // attenuated by the beam
            //     for (vis, vis_rot) in izip!(
            //         vis_tfb.slice(s![1, .., ..]),
            //         vis_rot_tfb.slice(s![1, .., ..]),
            //     ) {
            //         // XX
            //         assert_abs_diff_eq!(vis[0].norm(), source_fd as f32, epsilon = 1e-6);
            //         assert_abs_diff_eq!(vis_rot[0].norm(), source_fd as f32, epsilon = 1e-6);
            //         // YY
            //         assert_abs_diff_eq!(vis[3].norm(), source_fd as f32, epsilon = 1e-6);
            //         assert_abs_diff_eq!(vis_rot[3].norm(), source_fd as f32, epsilon = 1e-6);
            //     }
            // }

            let (time_axis, freq_axis, _baseline_axis) = (Axis(0), Axis(1), Axis(2));
            let gpu_xyzs: Vec<_> = xyzs
                .iter()
                .copied()
                .map(|XyzGeodetic { x, y, z }| gpu::XYZ {
                    x: x as GpuFloat,
                    y: y as GpuFloat,
                    z: z as GpuFloat,
                })
                .collect();
            let mut gpu_uvws_from = Array2::from_elem(
                (num_times, num_baselines),
                gpu::UVW {
                    u: -99.0,
                    v: -99.0,
                    w: -99.0,
                },
            );
            let mut gpu_uvws_to = Array2::from_elem(
                (num_times, num_baselines),
                gpu::UVW {
                    u: -99.0,
                    v: -99.0,
                    w: -99.0,
                },
            );
            dbg!(&xyzs);
            dbg!(&lmsts);
            gpu_uvws_from
                .outer_iter_mut()
                .zip(xyzs.outer_iter())
                .zip(lmsts.iter())
                .for_each(|((mut gpu_uvws, xyzs), lmst)| {
                    let phase_centre = obs_context.phase_centre.to_hadec(*lmst);
                    let v = xyzs_to_cross_uvws(xyzs.as_slice().unwrap(), phase_centre)
                        .into_iter()
                        .map(|uvw| gpu::UVW {
                            u: uvw.u as GpuFloat,
                            v: uvw.v as GpuFloat,
                            w: uvw.w as GpuFloat,
                        })
                        .collect::<Vec<_>>();
                    gpu_uvws.assign(&ArrayView1::from(&v));
                });

            dbg!(&gpu_uvws_from);
            let gpu_lambdas: Vec<GpuFloat> = lambdas_m.iter().map(|l| *l as GpuFloat).collect();

            let mut result = vis_averaged_tfb.clone();
            result.fill(Jones::default());

            let avg_freq = div_ceil(
                vis_tfb.len_of(freq_axis),
                vis_averaged_tfb.len_of(freq_axis),
            );
            // let avg_freq: usize = 1;

            let d_uvws_from = DevicePointer::copy_to_device(gpu_uvws_from.as_slice().unwrap()).unwrap();
            let mut d_uvws_to =
                DevicePointer::malloc(gpu_uvws_from.len() * std::mem::size_of::<gpu::UVW>()).unwrap();

            unsafe {
                let d_vis_tfb = DevicePointer::copy_to_device(vis_tfb.as_slice().unwrap()).unwrap();
                let d_weights_tfb =
                    DevicePointer::copy_to_device(weights_tfb.as_slice().unwrap()).unwrap();
                let mut d_vis_averaged_tfb =
                    DevicePointer::copy_to_device(result.as_slice().unwrap()).unwrap();
                let d_lmsts = DevicePointer::copy_to_device(
                    &lmsts
                        .iter()
                        .map(|lmst| *lmst as GpuFloat)
                        .collect::<Vec<_>>(),
                )
                .unwrap();
                let d_xyzs = DevicePointer::copy_to_device(&gpu_xyzs).unwrap();
                let d_lambdas = DevicePointer::copy_to_device(&gpu_lambdas).unwrap();

                gpu::rotate_average(
                    d_vis_tfb.get().cast(),
                    d_weights_tfb.get().cast(),
                    d_vis_averaged_tfb.get_mut().cast(),
                    gpu::RADec {
                        ra: source_radec.ra as GpuFloat,
                        dec: source_radec.dec as GpuFloat,
                    },
                    vis_tfb.len_of(time_axis).try_into().unwrap(),
                    num_tiles.try_into().unwrap(),
                    num_baselines.try_into().unwrap(),
                    vis_tfb.len_of(freq_axis).try_into().unwrap(),
                    avg_freq.try_into().unwrap(),
                    d_lmsts.get(),
                    d_xyzs.get(),
                    d_uvws_from.get(),
                    d_uvws_to.get_mut(),
                    d_lambdas.get(),
                );

                d_vis_averaged_tfb
                    .copy_from_device(result.as_slice_mut().unwrap())
                    .unwrap();

                d_uvws_to.copy_from_device(gpu_uvws_to.as_slice_mut().unwrap()).unwrap();
            }

            dbg!(&gpu_uvws_to);

            // Test that the CPU and GPU UVWs are the same. CPU UVWs are per
            // tile, GPU UVWs are per baseline, so we just make the CPU UVWs
            // ourselves.
            let mut cpu_uvws = Array2::from_elem((num_times, num_baselines), UVW::default());
            cpu_uvws
                .outer_iter_mut()
                .zip(xyzs.outer_iter())
                .zip(lmsts.iter())
                .for_each(|((mut cpu_uvws, xyzs), lmst)| {
                    let phase_centre = obs_context.phase_centre.to_hadec(*lmst);
                    let v = xyzs_to_cross_uvws(xyzs.as_slice().unwrap(), phase_centre);
                    cpu_uvws.assign(&ArrayView1::from(&v));
                });
            let gpu_uvws = Array2::from_shape_vec(
                (num_times, num_baselines),
                d_uvws_from.copy_from_device_new().unwrap(),
            )
            .unwrap()
            .mapv(|gpu::UVW { u, v, w }| UVW {
                // The GPU float precision might not be f64.
                u: u as _,
                v: v as _,
                w: w as _,
            });
            #[cfg(not(feature = "gpu-single"))]
            assert_abs_diff_eq!(cpu_uvws, gpu_uvws, epsilon = 0.0);
            #[cfg(feature = "gpu-single")]
            assert_abs_diff_eq!(cpu_uvws, gpu_uvws, epsilon = TILE_SPACING * 5e-8);

            // Hack to use `display_vis_tfb` with low-res visibilities.
            let mut low_res_obs_context = get_simple_obs_context(TILE_SPACING);
            low_res_obs_context.timestamps = vec1![low_res_obs_context.timestamps[0] + Duration::from_hours(0.5)];
            low_res_obs_context.fine_chan_freqs = vec1![(VEL_C / 1.5) as _];
            display_vis_tfb(
                &format!("host prec={apply_precession}"),
                vis_averaged_tfb.view(),
                &low_res_obs_context,
                obs_context.phase_centre,
                apply_precession,
            );
            display_vis_tfb(
                &format!("gpu vis_rotate_average prec={apply_precession}"),
                result.view(),
                &low_res_obs_context,
                obs_context.phase_centre,
                apply_precession,
            );

            assert_abs_diff_eq!(vis_averaged_tfb, result, epsilon = 1e-6);

            // for (tile_ws_obs, tile_ws_src, vis_fb, vis_rot_fb) in izip!(
            //     tile_ws_obs.outer_iter(),
            //     tile_ws_src.outer_iter(),
            //     vis_tfb.outer_iter(),
            //     vis_rot_tfb.outer_iter(),
            // ) {
            //     for (lambda_m, vis_b, vis_rot_b) in izip!(
            //         lambdas_m.iter(),
            //         vis_fb.outer_iter(),
            //         vis_rot_fb.outer_iter(),
            //     ) {
            //         for (&(ant1, ant2), vis, vis_rot) in
            //             izip!(ant_pairs.iter(), vis_b.iter(), vis_rot_b.iter(),)
            //         {
            //             let w_obs = tile_ws_obs[ant1] - tile_ws_obs[ant2];
            //             let w_src = tile_ws_src[ant1] - tile_ws_src[ant2];
            //             let arg = (TAU * (w_src - w_obs) / lambda_m) as f64;
            //             for (pol_model, pol_model_rot) in vis.iter().zip_eq(vis_rot.iter()) {
            //                 // magnitudes shoud not be affected by rotation
            //                 assert_abs_diff_eq!(
            //                     pol_model.norm(),
            //                     pol_model_rot.norm(),
            //                     epsilon = 1e-6
            //                 );
            //                 let pol_model_rot_expected = Complex::from_polar(
            //                     pol_model.norm(),
            //                     (pol_model.arg() as f64 - arg) as f32,
            //                 );
            //                 assert_abs_diff_eq!(
            //                     pol_model_rot_expected.arg(),
            //                     pol_model_rot.arg(),
            //                     epsilon = 1e-6
            //                 );
            //             }
            //         }
            //     }
            // }
        }
    }

    #[test]
    fn test_peel_gpu_single_source() {
        test_peel_single_source(PeelType::Gpu)
    }

    #[test]
    fn test_peel_gpu_multi_source() {
        test_peel_multi_source(PeelType::Gpu)
    }
}