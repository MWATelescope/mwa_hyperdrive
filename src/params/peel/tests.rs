// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Tests against peeling

use std::{collections::HashSet, f64::consts::TAU};

use approx::assert_abs_diff_eq;
use hifitime::{Duration, Epoch};
use indexmap::{indexmap, IndexMap};
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
/// - tile "u" has a u-component of s at lambda = 1m
/// - tile "v" has a v-component of s at lambda = 1m
#[rustfmt::skip]
fn get_simple_tiles(s_: f64) -> (Vec1<String>, Vec1<XyzGeodetic>) {
    (
        vec1!["o", "u", "v"].mapped(|s| s.into()),
        vec1![
            XyzGeodetic { x: 0., y: 0., z: 0., },
            XyzGeodetic { x: 0., y: s_, z: 0., },
            XyzGeodetic { x: 0., y: 0., z: s_, },
        ],
    )
}

/// get an observation context with:
/// - array positioned at LatLngHeight = 0, 0, 100m
/// - 2 timestamps:
///   - first: phase centre is at zenith on j2100
///   - second: an hour later,
/// - 2 frequencies: lambda = 2m, 1m
/// - tiles from [get_simple_tiles], s=1
fn get_simple_obs_context() -> ObsContext {
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

    let (tile_names, tile_xyzs) = get_simple_tiles(1.);
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
/// - tiles from [get_simple_tiles], s=1
fn get_complex_obs_context() -> ObsContext {
    let tile_limit = 32;
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

    // second timestep is at 1h
    let hour_epoch = obs_time + Duration::from_hours(1.0);
    let timestamps = vec1![obs_time, hour_epoch];

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
        time_res: Some(hour_epoch - obs_time),
        mwa_coarse_chan_nums: None,
        num_fine_chans_per_coarse_chan: None,
        freq_res: Some((fine_chan_freqs[1] - fine_chan_freqs[0]) as f64),
        fine_chan_freqs,
        flagged_fine_chans: vec![],
        flagged_fine_chans_per_coarse_chan: None,
        polarisations: Polarisations::default(),
    }
}

// these are used for debugging the tests
use ndarray::{ArrayView2, ArrayView3};
#[allow(clippy::too_many_arguments)]
#[allow(clippy::uninlined_format_args)]
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
    use std::f64::consts::PI;
    println!(
        "bl  u        v        w        | @ time={:>+9.3}s, lam={:+1.3}m, {}",
        seconds, lambda, name
    );
    for (jones, &(ant1, ant2)) in vis_b.iter().zip_eq(ant_pairs.iter()) {
        let uv = uvs[ant1] - uvs[ant2];
        let w = ws[ant1] - ws[ant2];
        let (name1, name2) = (&tile_names[ant1], &tile_names[ant2]);
        let (xx, xy, yx, yy) = jones.iter().collect_tuple().unwrap();
        println!(
            "{:1}-{:1} {:+1.5} {:+1.5} {:+1.5} | \
                XX {:07.5} @{:+08.5}pi XY {:07.5} @{:+08.5}pi \
                YX {:07.5} @{:+08.5}pi YY {:07.5} @{:+08.5}pi",
            name1,
            name2,
            uv.u / lambda,
            uv.v / lambda,
            w / lambda,
            xx.norm(),
            xx.arg() as f64 / PI,
            xy.norm(),
            xy.arg() as f64 / PI,
            yx.norm(),
            yx.arg() as f64 / PI,
            yy.norm(),
            yy.arg() as f64 / PI,
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
    let obs_context = get_simple_obs_context();
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
                assert_abs_diff_eq!(tile_uvs_obs[[t, 2]].v, 1., epsilon = 1e-6);
                assert_abs_diff_eq!(tile_uvs_obs[[t, 2]], tile_uvs_src[[t, 2]], epsilon = 1e-6);
                assert_abs_diff_eq!(tile_ws_obs[[t, 2]], tile_ws_src[[t, 2]], epsilon = 1e-6);
            }
            // tile 1 is aligned with zenith at t=0
            assert_abs_diff_eq!(tile_uvs_obs[[0, 1]].u, 1., epsilon = 1e-6);
            // tile 1 is aligned with source at t=1
            assert_abs_diff_eq!(tile_uvs_src[[1, 1]].u, 1., epsilon = 1e-6);
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
/// and a t the second timestep, 1h later, the source is at zenithiono rotated model
fn test_vis_rotation() {
    let obs_context = get_simple_obs_context();
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
    // let source_iono_consts = IndexMap::new();

    let mut vis_tfb = Array3::default((num_times, num_chans, num_baselines));
    let mut vis_rot_tfb = Array3::default((num_times, num_chans, num_baselines));

    // tile uvs and ws in the observation phase centre
    let mut tile_uvs_obs = Array2::default((num_times, num_tiles));
    let mut tile_ws_obs = Array2::default((num_times, num_tiles));
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

        vis_tfb.fill(Jones::zero());
        model_timesteps(&modeller, &obs_context.timestamps, vis_tfb.view_mut()).unwrap();

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

        // display_vis_tfb(
        //     &"model@obs".into(),
        //     vis_tfb.view(),
        //     &obs_context,
        //     obs_context.phase_centre,
        //     apply_precession,
        // );
        // display_vis_tfb(
        //     &"rotated@source".into(),
        //     vis_rot_tfb.view(),
        //     &obs_context,
        //     source_radec,
        //     apply_precession,
        // );

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
                        let pol_model_rot_expected = Complex::from_polar(
                            pol_model.norm(),
                            (pol_model.arg() as f64 - arg) as f32,
                        );
                        assert_abs_diff_eq!(
                            pol_model_rot_expected.arg(),
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
    let obs_context = get_simple_obs_context();
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
    // let source_iono_consts = IndexMap::new();

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

        for (time, tile_uvs_src, vis_fb, vis_iono_fb) in izip!(
            obs_context.timestamps.iter(),
            tile_uvs_src.outer_iter(),
            vis_tfb.outer_iter(),
            vis_iono_tfb.outer_iter(),
        ) {
            if !apply_precession {
                // baseline 1, from origin to v has no u or w component, should not be affected by the rotation
                for (vis, vis_iono) in izip!(vis_fb.slice(s![.., 1]), vis_iono_fb.slice(s![.., 1]),)
                {
                    assert_abs_diff_eq!(vis, vis_iono, epsilon = 1e-6);
                }
                // in the second timestep, the source should be at the pointing centre, so:
                // - should not be attenuated by the beam
                // - at lambda=2, iono vis should have the source at the phase centre, so no angle in pols XX, YY
                if time == &hour_epoch {
                    for (jones, jones_iono) in izip!(vis_fb.iter(), vis_iono_fb.iter(),) {
                        // XX
                        assert_abs_diff_eq!(jones[0].norm(), source_fd as f32, epsilon = 1e-6);
                        assert_abs_diff_eq!(jones_iono[0].norm(), source_fd as f32, epsilon = 1e-6);
                        // YY
                        assert_abs_diff_eq!(jones[3].norm(), source_fd as f32, epsilon = 1e-6);
                        assert_abs_diff_eq!(jones_iono[3].norm(), source_fd as f32, epsilon = 1e-6);
                    }
                    for vis_iono in vis_iono_fb.slice(s![0, ..]).iter() {
                        assert_abs_diff_eq!(vis_iono[0].arg(), 0., epsilon = 1e-6); // XX
                        assert_abs_diff_eq!(vis_iono[3].arg(), 0., epsilon = 1e-6);
                        // YY
                    }
                }
            }

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
                            epsilon = 1e-6
                        );
                        let pol_model_iono_expected = Complex::from_polar(
                            pol_model.norm(),
                            (pol_model.arg() as f64 - arg) as f32,
                        );
                        assert_abs_diff_eq!(
                            pol_model_iono_expected.arg(),
                            pol_model_iono.arg(),
                            epsilon = 1e-6
                        );
                    }
                }
            }
        }
    }
}

#[test]
/// test iono_fit, where residual is just iono rotated model
fn test_iono_fit() {
    let obs_context = get_simple_obs_context();
    let array_pos = obs_context.array_position;

    // second timestep is at 1h
    let hour_epoch = obs_context.timestamps[1];

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
        hour_epoch,
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
    // let source_iono_consts = IndexMap::new();

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

        for iono_consts in [
            IonoConsts {
                alpha: 0.,
                beta: 0.,
                gain: 1.0,
            },
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
            apply_iono2(
                vis_tfb.view(),
                vis_iono_tfb.view_mut(),
                tile_uvs_src.view(),
                iono_consts,
                &lambdas_m,
            );

            // display_vis_tfb(
            //     &"iono@obs".into(),
            //     vis_iono_tfb.view(),
            //     &obs_context,
            //     obs_context.phase_centre,
            //     apply_precession,
            // );

            let results = iono_fit(
                vis_iono_tfb.view(),
                weights.view(),
                vis_tfb.view(),
                &lambdas_m,
                tile_uvs_src.view(),
            );

            // println!("prec: {:?}, expected: {:?}, got: {:?}", apply_precession, iono_consts, &results);

            assert_abs_diff_eq!(results[0], iono_consts.alpha, epsilon = 1e-8);
            assert_abs_diff_eq!(results[1], iono_consts.beta, epsilon = 1e-8);
        }
    }
}

#[test]
/// - synthesize model visibilities
/// - apply ionospheric rotation
/// - create residual: ionospheric - model
/// - ap ply_iono3 should result in empty visibilitiesiono rotated model
fn test_apply_iono3() {
    let obs_context = get_simple_obs_context();
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
    // let source_iono_consts = IndexMap::new();

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
    // builder.filter_level(log::LevelFilter::Trace);
    // builder.init();

    // modify obs_context so that timesteps are closer together
    let mut obs_context = get_simple_obs_context();
    let hour_epoch = obs_context.timestamps[1];
    let time_res = Duration::from_seconds(1.0);
    let second_epoch = obs_context.timestamps[0] + time_res;
    obs_context.time_res = Some(time_res);
    obs_context.timestamps[1] = second_epoch;

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

    // source is at zenith at 1h (before precession)
    let lst_1h_rad = get_lmst(
        array_pos.longitude_rad,
        hour_epoch,
        obs_context.dut1.unwrap_or_default(),
    );
    let source_radec =
        RADec::from_hadec(HADec::from_radians(0., array_pos.latitude_rad), lst_1h_rad);
    let source_fd = 1.;
    let source_list = SourceList::from([(
        "One".into(),
        point_src_i!(source_radec, 0., fine_chan_freqs_hz[0], source_fd),
    )]);

    let beam = get_beam(num_tiles);
    let source_iono_consts = IndexMap::new();

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
            &source_iono_consts,
        )
        .unwrap();

        let mut low_res_modeller = new_sky_modeller(
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
            &source_iono_consts,
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

        let baseline_weights = vec![1.0; vis_model_obs_tfb.len_of(Axis(2))];

        for iono_consts in [
            IonoConsts {
                alpha: 0.,
                beta: 0.,
                gain: 1.0,
            },
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
            log::info!("Testing with iono consts {iono_consts:?}");
            apply_iono2(
                vis_model_obs_tfb.view(),
                vis_iono_obs_tfb.view_mut(),
                tile_uvs_src.view(),
                iono_consts,
                &lambdas_m,
            );

            // display_vis_tfb(
            //     &format!("iono@obs prec={}, ({}, {})", apply_precession, &iono_consts.0, &iono_consts.1),
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
            //     low_res_modeller.deref_mut(),
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
                    3,
                    1.0,
                    &fine_chan_freqs_hz,
                    &lambdas_m,
                    &lambdas_m,
                    &obs_context,
                    obs_context.array_position,
                    &obs_context.tile_xyzs,
                    &mut *low_res_modeller,
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
                        &source_iono_consts,
                    )
                    .unwrap();

                    peel_gpu(
                        vis_residual_obs_tfb.view_mut(),
                        vis_weights.view(),
                        &timeblock,
                        &source_list,
                        &mut all_iono_consts,
                        &source_weighted_positions,
                        3,
                        &chanblocks,
                        &lambdas_m,
                        &lambdas_m,
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

            println!("prec: {apply_precession:?}, expected: {iono_consts:?}, got: {iono_consts:?}");

            display_vis_tfb(
                &"peeled@obs".into(),
                vis_residual_obs_tfb.view(),
                &obs_context,
                obs_context.phase_centre,
                apply_precession,
            );

            assert_abs_diff_eq!(all_iono_consts[0].alpha, iono_consts.alpha, epsilon = 7e-10);
            assert_abs_diff_eq!(all_iono_consts[0].beta, iono_consts.beta, epsilon = 7e-10);

            // peel should perfectly remove the iono rotate model vis
            for jones_residual in vis_residual_obs_tfb.iter() {
                for pol_residual in jones_residual.iter() {
                    #[cfg(not(feature = "gpu-single"))]
                    assert_abs_diff_eq!(pol_residual.norm(), 0., epsilon = 1.3e-8);
                    #[cfg(feature = "gpu-single")]
                    assert_abs_diff_eq!(pol_residual.norm(), 0., epsilon = 1.7e-8);
                }
            }
        }
    }
}

#[track_caller]
fn test_peel_multi_source(peel_type: PeelType) {
    // // enable trace
    // let mut builder = env_logger::Builder::from_default_env();
    // builder.target(env_logger::Target::Stdout);
    // builder.format_target(false);
    // builder.filter_level(log::LevelFilter::Trace);
    // builder.init();

    // modify obs_context so that timesteps are closer together
    let mut obs_context = get_complex_obs_context();
    let time_res = Duration::from_seconds(1.0);
    let second_epoch = obs_context.timestamps[0] + time_res;
    obs_context.time_res = Some(time_res);
    obs_context.timestamps[1] = second_epoch;

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

    let lst_0h_rad = get_lmst(
        array_pos.longitude_rad,
        obs_context.timestamps[0],
        obs_context.dut1.unwrap_or_default(),
    );
    let source_midpoint =
        RADec::from_hadec(HADec::from_radians(0., array_pos.latitude_rad), lst_0h_rad);

    let source_list = SourceList::from(indexmap! {
        "Four".into() => point_src_i!(RADec {ra: source_midpoint.ra + 0.05, dec: source_midpoint.dec + 0.05}, 0., fine_chan_freqs_hz[0], 4.),
        "Three".into() => point_src_i!(RADec {ra: source_midpoint.ra + 0.03, dec: source_midpoint.dec - 0.03}, 0., fine_chan_freqs_hz[0], 3.),
        // "Two".into() => point_src_i!(RADec {ra: source_midpoint.ra - 0.01, dec: source_midpoint.dec + 0.02}, 0., fine_chan_freqs_hz[0], 2.),
        // "One".into() => point_src_i!(RADec {ra: source_midpoint.ra - 0.02, dec: source_midpoint.dec - 0.01}, 0., fine_chan_freqs_hz[0], 1.),
    });

    let source_weighted_positions = source_list
        .iter()
        .map(|(_, source)| source.components[0].radec)
        .collect_vec();

    let iono_consts = [
        IonoConsts {
            alpha: -0.00002,
            beta: -0.00001,
            gain: 1.0,
        },
        IonoConsts {
            alpha: 0.00001,
            beta: -0.00003,
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
    ];

    let beam = get_beam(num_tiles);
    let source_iono_consts = IndexMap::new();

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
            &source_iono_consts,
        )
        .unwrap();

        let mut low_res_modeller = new_sky_modeller(
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
            &source_iono_consts,
        )
        .unwrap();

        let baseline_weights = vec![1.0; vis_model_tmp_tfb.len_of(Axis(2))];

        vis_residual_obs_tfb.fill(Jones::zero());

        // model each source in source_list and rotate by iono_consts with apply_iono2
        for (&iono_consts, (name, source)) in izip!(iono_consts.iter(), source_list.iter(),) {
            let source_radec = source.components[0].radec;
            println!("source {} radec {:?}", name, &source_radec);

            high_res_modeller
                .update_with_a_source(source, name.as_str(), obs_context.phase_centre)
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
                source_radec,
                apply_precession,
            );

            apply_iono2(
                vis_model_tmp_tfb.view(),
                vis_iono_tmp_tfb.view_mut(),
                tile_uvs_src.view(),
                iono_consts,
                &lambdas_m,
            );

            display_vis_tfb(
                &format!("iono@src={} consts={:?}", name, &iono_consts),
                vis_iono_tmp_tfb.view(),
                &obs_context,
                source_radec,
                apply_precession,
            );

            // add iono rotated and subtract model visibilities from residual
            Zip::from(vis_residual_obs_tfb.view_mut())
                .and(vis_iono_tmp_tfb.view())
                .and(vis_model_tmp_tfb.view())
                .for_each(|res, iono, model| *res += *iono - *model);
        }

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
        //     &mut iono_consts_result,
        //     &source_weighted_positions,
        //     num_sources_to_iono_subtract,
        //     &fine_chan_freqs_hz,
        //     &lambdas_m,
        //     &lambdas_m,
        //     &obs_context,
        //     obs_context.array_position.unwrap(),
        //     &obs_context.tile_xyzs,
        //     low_res_modeller.deref_mut(),
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
                3,
                1.0,
                &fine_chan_freqs_hz,
                &lambdas_m,
                &lambdas_m,
                &obs_context,
                obs_context.array_position,
                &obs_context.tile_xyzs,
                &mut *low_res_modeller,
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
                    &source_iono_consts,
                )
                .unwrap();

                peel_gpu(
                    vis_residual_obs_tfb.view_mut(),
                    vis_weights.view(),
                    &timeblock,
                    &source_list,
                    &mut iono_consts_result,
                    &source_weighted_positions,
                    3,
                    &chanblocks,
                    &lambdas_m,
                    &lambdas_m,
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
        }

        display_vis_tfb(
            &"peeled@obs".into(),
            vis_residual_obs_tfb.view(),
            &obs_context,
            obs_context.phase_centre,
            apply_precession,
        );

        for (expected, result) in izip!(iono_consts.iter(), iono_consts_result.iter()) {
            println!("prec: {apply_precession:?}, expected: {expected:?}, got: {result:?}");
            assert_abs_diff_eq!(expected.alpha, result.alpha, epsilon = 3e-7);
            assert_abs_diff_eq!(expected.beta, result.beta, epsilon = 3e-7);
        }

        // peel should perfectly remove the iono rotate model vis
        for jones_residual in vis_residual_obs_tfb.iter() {
            for pol_residual in jones_residual.iter() {
                #[cfg(not(feature = "gpu-single"))]
                assert_abs_diff_eq!(pol_residual.norm(), 0., epsilon = 4e-7);
                #[cfg(feature = "gpu-single")]
                assert_abs_diff_eq!(pol_residual.norm(), 0., epsilon = 5e-7);
            }
        }
    }
}

#[track_caller]
fn test_peel_single_source_brightness_offset(peel_type: PeelType) {
    // enable trace
    // let mut builder = env_logger::Builder::from_default_env();
    // builder.target(env_logger::Target::Stdout);
    // builder.format_target(false);
    // builder.filter_level(log::LevelFilter::Trace);
    // builder.init();

    // modify obs_context so that timesteps are closer together
    let mut obs_context = get_simple_obs_context();
    let hour_epoch = obs_context.timestamps[1];
    let time_res = Duration::from_seconds(1.0);
    let second_epoch = obs_context.timestamps[0] + time_res;
    obs_context.time_res = Some(time_res);
    obs_context.timestamps[1] = second_epoch;

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

    // source is at zenith at 1h (before precession)
    let lst_1h_rad = get_lmst(
        array_pos.longitude_rad,
        hour_epoch,
        obs_context.dut1.unwrap_or_default(),
    );
    let source_radec =
        RADec::from_hadec(HADec::from_radians(0., array_pos.latitude_rad), lst_1h_rad);
    let source_list = SourceList::from([(
        "One".into(),
        point_src_i!(source_radec, 0., fine_chan_freqs_hz[0], 1.0),
    )]);
    // Introduce a brightness offset.
    let true_source_list = SourceList::from([(
        "One".into(),
        point_src_i!(source_radec, 0., fine_chan_freqs_hz[0], 1.2),
    )]);

    let beam = get_beam(num_tiles);

    // model visibilities in the observation phase centre
    let mut vis_obs_tfb = Array3::zeros((num_times, num_chans, num_baselines));
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

    let multi_progress = MultiProgress::with_draw_target(ProgressDrawTarget::hidden());

    let wip_iono_consts = IndexMap::new();
    for apply_precession in [false, true] {
        let mut high_res_modeller = new_sky_modeller(
            &beam,
            &true_source_list,
            Polarisations::default(),
            &obs_context.tile_xyzs,
            &fine_chan_freqs_hz,
            &flagged_tiles,
            obs_context.phase_centre,
            array_pos.longitude_rad,
            array_pos.latitude_rad,
            obs_context.dut1.unwrap_or_default(),
            apply_precession,
            &wip_iono_consts,
        )
        .unwrap();
        let mut low_res_modeller = new_sky_modeller(
            &beam,
            &true_source_list,
            Polarisations::default(),
            &obs_context.tile_xyzs,
            &fine_chan_freqs_hz,
            &flagged_tiles,
            obs_context.phase_centre,
            array_pos.longitude_rad,
            array_pos.latitude_rad,
            obs_context.dut1.unwrap_or_default(),
            apply_precession,
            &wip_iono_consts,
        )
        .unwrap();

        vis_model_obs_tfb.fill(Jones::zero());

        high_res_modeller
            //.update_with_a_source(&true_source_list[0], source_weighted_positions[0])
            .update_with_a_source(&true_source_list[0], "", obs_context.phase_centre)
            .unwrap();
        model_timesteps(
            &*high_res_modeller,
            &obs_context.timestamps,
            vis_obs_tfb.view_mut(),
        )
        .unwrap();

        high_res_modeller
            //.update_with_a_source(&source_list[0], source_weighted_positions[0])
            .update_with_a_source(&source_list[0], "", obs_context.phase_centre)
            .unwrap();
        model_timesteps(
            &*high_res_modeller,
            &obs_context.timestamps,
            vis_model_obs_tfb.view_mut(),
        )
        .unwrap();
        dbg!(vis_obs_tfb[(0, 0, 0)], vis_model_obs_tfb[(0, 0, 0)]);

        setup_tile_uv_w_arrays(
            tile_uvs_src.view_mut(),
            tile_ws_src.view_mut(),
            &obs_context,
            source_radec,
            apply_precession,
        );

        let baseline_weights = vec![1.0; vis_obs_tfb.len_of(Axis(2))];

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
                gain: 1.0,
            },
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
            log::info!("Testing with iono consts {iono_consts:?}");
            apply_iono2(
                vis_obs_tfb.view(),
                vis_residual_obs_tfb.view_mut(),
                tile_uvs_src.view(),
                iono_consts,
                &lambdas_m,
            );
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
            //     low_res_modeller.deref_mut(),
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
                    3,
                    0.0,
                    &fine_chan_freqs_hz,
                    &lambdas_m,
                    &lambdas_m,
                    &obs_context,
                    obs_context.array_position,
                    &obs_context.tile_xyzs,
                    &mut *low_res_modeller,
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
                        &wip_iono_consts,
                    )
                    .unwrap();

                    peel_gpu(
                        vis_residual_obs_tfb.view_mut(),
                        vis_weights.view(),
                        &timeblock,
                        &source_list,
                        &mut all_iono_consts,
                        &source_weighted_positions,
                        3,
                        &chanblocks,
                        &lambdas_m,
                        &lambdas_m,
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

            println!("prec: {apply_precession:?}, expected: {iono_consts:?}, got: {iono_consts:?}");

            display_vis_tfb(
                &"peeled@obs".into(),
                vis_residual_obs_tfb.view(),
                &obs_context,
                obs_context.phase_centre,
                apply_precession,
            );

            assert_abs_diff_eq!(all_iono_consts[0].alpha, iono_consts.alpha, epsilon = 7e-10);
            assert_abs_diff_eq!(all_iono_consts[0].beta, iono_consts.beta, epsilon = 7e-10);

            // peel should perfectly remove the iono rotate model vis
            for jones_residual in vis_residual_obs_tfb.iter() {
                for pol_residual in jones_residual.iter() {
                    #[cfg(not(feature = "gpu-single"))]
                    assert_abs_diff_eq!(pol_residual.norm(), 0., epsilon = 1.3e-8);
                    #[cfg(feature = "gpu-single")]
                    assert_abs_diff_eq!(pol_residual.norm(), 0., epsilon = 1.7e-8);
                }
            }
        }
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

    use indexmap::IndexMap;
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
    /// - synthesize model visibilities
    /// - apply ionospheric rotation
    /// - create residual: ionospheric - model
    /// - ap ply_iono3 should result in empty visibilitiesiono rotated model
    fn test_gpu_subtract_iono() {
        let obs_context = get_simple_obs_context();
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
        let source_iono_consts = IndexMap::new();

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
                &source_iono_consts,
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

            let mut d_high_res_vis =
                DevicePointer::copy_to_device(vis_residual_obs_tfb.as_slice().unwrap()).unwrap();
            let d_uvws_src =
                DevicePointer::copy_to_device(gpu_uvws_src.as_slice().unwrap()).unwrap();
            let d_lambdas = DevicePointer::copy_to_device(
                &lambdas_m.iter().map(|l| *l as GpuFloat).collect::<Vec<_>>(),
            )
            .unwrap();
            let mut d_iono_consts = DevicePointer::copy_to_device(&[gpu::IonoConsts {
                alpha: 0.0,
                beta: 0.0,
                gain: 1.0,
            }])
            .unwrap();
            let d_old_iono_consts = DevicePointer::copy_to_device(&[gpu::IonoConsts {
                alpha: 0.0,
                beta: 0.0,
                gain: 1.0,
            }])
            .unwrap();

            for (alpha, beta) in [(0.0001, -0.0003), (0.0003, -0.0001), (-0.0007, 0.0001)] {
                let iono_consts = IonoConsts {
                    alpha,
                    beta,
                    ..Default::default()
                };

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
                    //     &format!("iono@obs ({}, {})", &iono_consts.0, &iono_consts.1),
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

                    d_high_res_vis
                        .overwrite(vis_residual_obs_tfb.as_slice().unwrap())
                        .unwrap();
                    d_iono_consts
                        .overwrite(&[gpu::IonoConsts {
                            alpha: iono_consts.alpha,
                            beta: iono_consts.beta,
                            gain: iono_consts.gain,
                        }])
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
            let obs_context = get_simple_obs_context();
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

                let mut vis_averaged_tfb = Array3::default((1, 1, num_baselines));
                vis_average2(
                    vis_rot_tfb.view(),
                    vis_averaged_tfb.view_mut(),
                    weights_tfb.view(),
                );

                // display_vis_tfb(
                //     &"model@obs".into(),
                //     vis_tfb.view(),
                //     &obs_context,
                //     obs_context.phase_centre,
                //     apply_precession,
                // );
                // display_vis_tfb(
                //     &"rotated@source".into(),
                //     vis_rot_tfb.view(),
                //     &obs_context,
                //     source_radec,
                //     apply_precession,
                // );

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
                let mut gpu_uvws = Array2::from_elem(
                    (num_times, num_baselines),
                    gpu::UVW {
                        u: -99.0,
                        v: -99.0,
                        w: -99.0,
                    },
                );
                gpu_uvws
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
                let gpu_lambdas: Vec<GpuFloat> = lambdas_m.iter().map(|l| *l as GpuFloat).collect();

                let mut result = vis_averaged_tfb.clone();
                result.fill(Jones::default());

                let avg_freq = div_ceil(
                    vis_tfb.len_of(freq_axis),
                    vis_averaged_tfb.len_of(freq_axis),
                );

                let d_uvws_from =
                    DevicePointer::copy_to_device(gpu_uvws.as_slice().unwrap()).unwrap();
                let mut d_uvws_to =
                    DevicePointer::malloc(gpu_uvws.len() * std::mem::size_of::<gpu::UVW>())
                        .unwrap();

                unsafe {
                    let d_vis_tfb =
                        DevicePointer::copy_to_device(vis_tfb.as_slice().unwrap()).unwrap();
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
                }

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
                assert_abs_diff_eq!(cpu_uvws, gpu_uvws, epsilon = 5e-8);

                // Hack to use `display_vis_tfb` with low-res visibilities.
                let mut low_res_obs_context = get_simple_obs_context();
                low_res_obs_context.timestamps = vec1![hour_epoch];
                low_res_obs_context.fine_chan_freqs = vec1![VEL_C as _];
                display_vis_tfb(
                    &"host".to_string(),
                    vis_averaged_tfb.view(),
                    &low_res_obs_context,
                    obs_context.phase_centre,
                    apply_precession,
                );
                display_vis_tfb(
                    &"gpu vis_rotate_average".to_string(),
                    result.view(),
                    &low_res_obs_context,
                    obs_context.phase_centre,
                    apply_precession,
                );

                assert_abs_diff_eq!(vis_averaged_tfb, result);

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
    }

    #[test]
    fn test_peel_gpu_single_source() {
        test_peel_single_source(PeelType::Gpu)
    }

    #[test]
    fn test_peel_gpu_multi_source() {
        test_peel_multi_source(PeelType::Gpu)
    }

    #[test]
    fn test_peel_gpu_single_source_brightness_offset() {
        test_peel_single_source_brightness_offset(PeelType::Gpu)
    }
}
