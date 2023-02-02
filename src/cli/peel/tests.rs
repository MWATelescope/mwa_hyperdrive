//! Tests against peeling

use crate::{
    averaging::Timeblock,
    beam::{create_fee_beam_object, Beam, Delays},
    context::ObsContext,
    model::new_sky_modeller,
    srclist::{ComponentType, FluxDensity, FluxDensityType, Source, SourceComponent, SourceList},
};
use approx::assert_abs_diff_eq;
use hifitime::{Duration, Epoch, Unit};
use indexmap::indexmap;
use indicatif::{MultiProgress, ProgressDrawTarget};
use itertools::{izip, Itertools};
use marlu::{
    constants::VEL_C,
    math::cross_correlation_baseline_to_tiles,
    precession::{get_lmst, precess_time},
    Complex, HADec, Jones, LatLngHeight, RADec, XyzGeodetic,
};
use ndarray::{array, s, Array2, Array3, ArrayViewMut2};
use num_traits::Zero;
use std::{
    collections::HashSet,
    f64::consts::TAU,
    ops::{Deref, DerefMut},
    path::PathBuf,
};
use vec1::{vec1, Vec1};

use super::{
    apply_iono2, apply_iono3, iono_fit, model_timesteps, peel_cpu, setup_uvs, setup_ws,
    vis_average2, vis_rotate_fb, weights_average, UV, W,
};

// a single-component point source, stokes I.
macro_rules! point_src_i {
    ($radec:expr, $si:expr, $freq:expr, $i:expr) => {
        Source {
            components: vec1![SourceComponent {
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
            }],
        }
    };
}

fn get_beam(num_tiles: usize) -> Box<dyn Beam> {
    let beam_file: Option<PathBuf> = None;

    #[rustfmt::skip]
    let delays = vec![
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
    ];
    // https://github.com/MWATelescope/mwa_pb/blob/90d6fbfc11bf4fca35796e3d5bde3ab7c9833b66/mwa_pb/mwa_sweet_spots.py#L60
    // let delays = vec![0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12];

    let beam = create_fee_beam_object::<_>(
        beam_file,
        num_tiles,
        Delays::Partial(delays),
        None,
        // Array2::from_elem((obs_context.tile_xyzs.len(), 32), 1.)
    )
    .unwrap();
    beam
}

// get a timestamp at lmst=0 around the year 2100
// precessing to j2000 will introduce a noticable difference.
fn get_j2100(array_position: &LatLngHeight, dut1: Duration) -> Epoch {
    let mut epoch = Epoch::from_gregorian_utc_at_midnight(2100, 1, 1);

    // shift zenith_time to the nearest time when the phase centre is at zenith
    let sidereal2solar = 365.24 / 366.24;
    let obs_lst_rad = get_lmst(array_position.longitude_rad, epoch, dut1);
    if obs_lst_rad.abs() > 1e-6 {
        epoch -= Duration::from_f64(sidereal2solar * obs_lst_rad / TAU, Unit::Day);
    }
    epoch
}

/// get 3 simple tiles:
/// - tile "o" is at origin
/// - tile "u" has a u-component of s at lambda = 1m
/// - tile "v" has a v-component of s at lambda = 1m
fn get_simple_tiles(s_: f64) -> (Vec1<String>, Vec1<XyzGeodetic>) {
    (
        vec1!["o", "u", "v"].mapped(|s| s.into()),
        vec1![
            XyzGeodetic {
                x: 0.,
                y: 0.,
                z: 0.,
            },
            XyzGeodetic {
                x: 0.,
                y: s_,
                z: 0.,
            },
            XyzGeodetic {
                x: 0.,
                y: 0.,
                z: s_,
            },
        ],
    )
}

/// get a simple observation context with:
/// - array positioned at LatLngHeight = 0, 0, 100m
/// - 2 timestamps:
///   - first: phase centre is at zenith on j2100
///   - second: an hour later,
/// - 2 frequencies: lambda = 2m, 1m
/// - tiles from `get_simple_tiles`, s=1
fn get_simple_obs_context() -> ObsContext {
    let array_position = LatLngHeight {
        longitude_rad: 0.,
        latitude_rad: 0.,
        height_metres: 100.,
    };

    let dut1 = Duration::from_f64(0.0, Unit::Second);
    let obs_epoch = get_j2100(&array_position, dut1);

    // at first timestep phase centre is at zenith
    let lst_zenith_rad = get_lmst(array_position.longitude_rad, obs_epoch, dut1);
    let phase_centre = RADec::from_hadec(
        HADec::from_radians(0., array_position.latitude_rad),
        lst_zenith_rad,
    );

    // second timestep is at 1h
    let hour_epoch = obs_epoch + Duration::from_f64(1.0, Unit::Hour);
    let timestamps = vec1![obs_epoch, hour_epoch];

    let (tile_names, tile_xyzs) = get_simple_tiles(1.);
    let lambdas_m = vec1![2., 1.];
    let fine_chan_freqs: Vec1<u64> = lambdas_m.mapped(|l| (VEL_C / l) as u64);

    ObsContext {
        obsid: None,
        timestamps,
        all_timesteps: vec1![0, 1],
        unflagged_timesteps: vec![0, 1],
        phase_centre,
        pointing_centre: None,
        array_position: Some(array_position),
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
        coarse_chan_nums: vec![],
        coarse_chan_freqs: vec![],
        num_fine_chans_per_coarse_chan: 1,
        freq_res: Some((fine_chan_freqs[1] - fine_chan_freqs[0]) as f64),
        fine_chan_freqs,
        flagged_fine_chans: vec![],
        flagged_fine_chans_per_coarse_chan: vec![],
    }
}

// #[allow(clippy::too_many_arguments)]
// fn display_vis_b(
//     name: &String,
//     vis_b: &[Jones<f32>],
//     ant_pairs: &[(usize, usize)],
//     uvs: &[UV],
//     ws: &[W],
//     tile_names: &[String],
//     seconds: f64,
//     lambda: f64
// ) {
//     use std::f64::consts::PI;
//     println!("bl  u      v      w      | @ time={:>+9.3}s, lam={:+1.3}m, {}", seconds, lambda, name);
//     for (jones, &(ant1, ant2)) in vis_b.iter().zip_eq(ant_pairs.iter()) {
//         let uv = uvs[ant1] - uvs[ant2];
//         let w = ws[ant1] - ws[ant2];
//         let (name1, name2) = (&tile_names[ant1], &tile_names[ant2]);
//         let (xx, xy, yx, yy) = jones.iter().collect_tuple().unwrap();
//         println!(
//             "{:1}-{:1} {:+1.3} {:+1.3} {:+1.3} | \
//                 XX {:07.5} @{:+08.5}pi XY {:07.5} @{:+08.5}pi \
//                 YX {:07.5} @{:+08.5}pi YY {:07.5} @{:+08.5}pi",
//             name1, name2, uv.u / lambda, uv.v / lambda, w / lambda,
//             xx.norm(), xx.arg() as f64 / PI,
//             xy.norm(), xy.arg() as f64 / PI,
//             yx.norm(), yx.arg() as f64 / PI,
//             yy.norm(), yy.arg() as f64 / PI,
//         );
//     }
// }

// #[allow(clippy::too_many_arguments)]
// fn display_vis_fb(
//     name: &String,
//     vis_fb: ArrayView2<Jones<f32>>,
//     seconds: f64,
//     uvs: &[UV],
//     ws: &[W],
//     lambdas_m: &[f64],
//     ant_pairs: &[(usize, usize)],
//     tile_names: &[String],
// ) {
//     // println!("{:9} | {:7} | bl  u      v      w      | {}", "time", "lam", name);
//     for (vis_b, &lambda) in vis_fb.outer_iter().zip_eq(lambdas_m.iter()) {
//         display_vis_b(
//             name,
//             vis_b.as_slice().unwrap(),
//             ant_pairs,
//             uvs,
//             ws,
//             tile_names,
//             seconds,
//             lambda
//         );
//     }
// }

// /// display named visibilities and uvws in table format
// #[allow(clippy::too_many_arguments)]
// fn display_vis_tfb(
//     name: &String,
//     vis_tfb: ArrayView3<Jones<f32>>,
//     obs_context: &ObsContext,
//     phase_centre: RADec,
//     apply_precession: bool,
// ) {
//     let array_pos = obs_context.array_position.unwrap();
//     let num_tiles = obs_context.get_num_unflagged_tiles();
//     let num_baselines = (num_tiles * (num_tiles - 1)) / 2;
//     let ant_pairs = (0..num_baselines)
//         .into_iter()
//         .map(|bl_idx| cross_correlation_baseline_to_tiles(num_tiles, bl_idx))
//         .collect_vec();
//     let fine_chan_freqs_hz = obs_context.fine_chan_freqs.iter().map(|&f| f as f64).collect_vec();
//     let lambdas_m = fine_chan_freqs_hz.iter().map(|&f| VEL_C / f).collect_vec();

//     let start_seconds = obs_context.timestamps[0].to_gpst_seconds();
//     let mut tile_uvs_tmp = vec![UV::default(); num_tiles];
//     let mut tile_ws_tmp = vec![W::default(); num_tiles];
//     // println!("{:9} | {:7} | bl  u      v      w      | {}", "time", "lam", name);
//     for (vis_fb, &time) in vis_tfb.outer_iter().zip_eq(obs_context.timestamps.iter()) {
//         if apply_precession {
//             let precession_info = precess_time(
//                 array_pos.longitude_rad,
//                 array_pos.latitude_rad,
//                 phase_centre,
//                 time,
//                 obs_context.dut1.unwrap(),
//             );
//             let hadec = phase_centre.to_hadec(precession_info.lmst_j2000);
//             let precessed_xyzs = precession_info.precess_xyz(&obs_context.tile_xyzs);
//             setup_uvs(&mut tile_uvs_tmp, &precessed_xyzs, hadec);
//             setup_ws(&mut tile_ws_tmp, &precessed_xyzs, hadec);
//         } else {
//             let lmst = get_lmst(array_pos.longitude_rad, time, obs_context.dut1.unwrap());
//             let hadec = phase_centre.to_hadec(lmst);
//             setup_uvs(&mut tile_uvs_tmp, &obs_context.tile_xyzs, hadec);
//             setup_ws(&mut tile_ws_tmp, &obs_context.tile_xyzs, hadec);
//         }
//         let seconds = time.to_gpst_seconds() - start_seconds;
//         display_vis_fb(
//             name,
//             vis_fb.view(),
//             seconds,
//             tile_uvs_tmp.as_slice(),
//             tile_ws_tmp.as_slice(),
//             &lambdas_m,
//             &ant_pairs,
//             &obs_context.tile_names,
//         );
//     }
// }

fn setup_uvw_arrays(
    mut tile_uvs: ArrayViewMut2<UV>,
    mut tile_ws: ArrayViewMut2<W>,
    obs_context: &ObsContext,
    phase_centre: RADec,
    apply_precession: bool,
) {
    let array_pos = obs_context.array_position.unwrap();
    for (&time, mut tile_uvs, mut tile_ws) in izip!(
        obs_context.timestamps.iter(),
        tile_uvs.outer_iter_mut(),
        tile_ws.outer_iter_mut(),
    ) {
        if apply_precession {
            let precession_info = precess_time(
                array_pos.longitude_rad,
                array_pos.latitude_rad,
                phase_centre,
                time,
                obs_context.dut1.unwrap(),
            );
            let hadec = phase_centre.to_hadec(precession_info.lmst_j2000);
            let precessed_xyzs = precession_info.precess_xyz(&obs_context.tile_xyzs);
            setup_uvs(tile_uvs.as_slice_mut().unwrap(), &precessed_xyzs, hadec);
            setup_ws(tile_ws.as_slice_mut().unwrap(), &precessed_xyzs, hadec);
        } else {
            let lmst = get_lmst(array_pos.longitude_rad, time, obs_context.dut1.unwrap());
            let hadec = phase_centre.to_hadec(lmst);
            setup_uvs(tile_uvs.as_slice_mut().unwrap(), &obs_context.tile_xyzs, hadec);
            setup_ws(tile_ws.as_slice_mut().unwrap(), &obs_context.tile_xyzs, hadec);
        }
    }
}

#[test]
/// test `setup_uvs`, `setup_ws`
fn test_setup_uv() {
    let apply_precession = false;

    let obs_context = get_simple_obs_context();
    let array_pos = obs_context.array_position.unwrap();

    let num_tiles = obs_context.get_num_unflagged_tiles();
    let num_times = obs_context.timestamps.len();

    // source is at zenith at 1h
    let hour_epoch = obs_context.timestamps[1];
    let lst_1h_rad = get_lmst(
        array_pos.longitude_rad,
        hour_epoch,
        obs_context.dut1.unwrap(),
    );
    let source_radec =
        RADec::from_hadec(HADec::from_radians(0., array_pos.latitude_rad), lst_1h_rad);

    // tile uvs and ws in the observation phase centre
    let mut tile_uvs_obs = Array2::default((num_times, num_tiles));
    let mut tile_ws_obs = Array2::default((num_times, num_tiles));
    // tile uvs and ws in the source phase centre
    let mut tile_uvs_src = Array2::default((num_times, num_tiles));
    let mut tile_ws_src = Array2::default((num_times, num_tiles));

    setup_uvw_arrays(
        tile_uvs_obs.view_mut(),
        tile_ws_obs.view_mut(),
        &obs_context,
        obs_context.phase_centre,
        apply_precession,
    );
    setup_uvw_arrays(
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
            // tile 0 is the origin tile
            assert_abs_diff_eq!(tile_uvs_obs[[t, 0]].u, 0., epsilon = 1e-6);
            assert_abs_diff_eq!(tile_uvs_obs[[t, 0]].v, 0., epsilon = 1e-6);
            assert_abs_diff_eq!(tile_ws_obs[[t, 0]].0, 0., epsilon = 1e-6);
            // tile 2 is a special case with only a v component, so should be unchanged
            assert_abs_diff_eq!(tile_uvs_obs[[t, 2]].v, 1., epsilon = 1e-6);
            assert_abs_diff_eq!(tile_uvs_obs[[t, 2]], tile_uvs_src[[t, 2]], epsilon = 1e-6);
            assert_abs_diff_eq!(tile_ws_obs[[t, 2]], tile_ws_src[[t, 2]], epsilon = 1e-6);
        }
        // tile 1 is aligned with zenith at t=0
        assert_abs_diff_eq!(tile_uvs_obs[[0, 1]].u, 1., epsilon = 1e-6);
        // tile 1 is aligned with source at t=1
        assert_abs_diff_eq!(tile_uvs_src[[1, 1]].u, 1., epsilon = 1e-6);
    } else {
        panic!("no tests for precession yet");
    }
}

#[test]
/// tests vis_rotate_fb by asserting that:
/// - rotated visibilities have the source at the phase centre
/// simulate vis, where at the first timestep, the phase centre is at zenith
/// and a t the second timestep, 1h later, the source is at zenithiono rotated model
fn test_vis_rotation() {
    let apply_precession = false;
    let obs_context = get_simple_obs_context();
    let array_pos = obs_context.array_position.unwrap();

    let num_tiles = obs_context.get_num_unflagged_tiles();
    let num_times = obs_context.timestamps.len();
    let num_baselines = (num_tiles * (num_tiles - 1)) / 2;
    let ant_pairs = (0..num_baselines)
        .into_iter()
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
        obs_context.dut1.unwrap(),
    );
    let source_radec =
        RADec::from_hadec(HADec::from_radians(0., array_pos.latitude_rad), lst_1h_rad);
    let source_fd = 1.;
    let source_list = SourceList::from(indexmap! {
        "One".into() => point_src_i!(source_radec, 0., fine_chan_freqs_hz[0], source_fd),
    });

    let beam = get_beam(num_tiles);

    let mut modeller = new_sky_modeller(
        #[cfg(feature = "cuda")]
        false,
        beam.deref(),
        &source_list,
        &obs_context.tile_xyzs,
        &fine_chan_freqs_hz,
        &flagged_tiles,
        obs_context.phase_centre,
        array_pos.longitude_rad,
        array_pos.latitude_rad,
        obs_context.dut1.unwrap(),
        apply_precession,
    )
    .unwrap();

    let mut vis_tfb = Array3::default((num_times, num_chans, num_baselines));
    model_timesteps(
        modeller.deref_mut(),
        &obs_context.timestamps,
        vis_tfb.view_mut(),
    )
    .unwrap();

    let mut vis_rot_tfb = Array3::default((num_times, num_chans, num_baselines));

    let mut tile_ws_obs = Array2::default((num_times, num_tiles));
    let mut tile_ws_src = Array2::default((num_times, num_tiles));

    // iterate over time, populating uvws and rotating visibilities
    for (vis_fb, mut vis_rot_fb, &time, mut tile_ws_obs, mut tile_ws_src) in izip!(
        vis_tfb.outer_iter(),
        vis_rot_tfb.view_mut().outer_iter_mut(),
        obs_context.timestamps.iter(),
        tile_ws_obs.outer_iter_mut(),
        tile_ws_src.outer_iter_mut(),
    ) {
        if apply_precession {
            let precession_info = precess_time(
                array_pos.longitude_rad,
                array_pos.latitude_rad,
                obs_context.phase_centre,
                time,
                obs_context.dut1.unwrap(),
            );
            let hadec_obs = obs_context
                .phase_centre
                .to_hadec(precession_info.lmst_j2000);
            let hadec_src = source_radec.to_hadec(precession_info.lmst_j2000);
            let precessed_xyzs = precession_info.precess_xyz(&obs_context.tile_xyzs);
            setup_ws(
                tile_ws_obs.as_slice_mut().unwrap(),
                &precessed_xyzs,
                hadec_obs,
            );
            setup_ws(
                tile_ws_src.as_slice_mut().unwrap(),
                &precessed_xyzs,
                hadec_src,
            );
        } else {
            let lmst = get_lmst(array_pos.longitude_rad, time, obs_context.dut1.unwrap());
            let hadec_obs = obs_context.phase_centre.to_hadec(lmst);
            let hadec_src = source_radec.to_hadec(lmst);
            setup_ws(
                tile_ws_obs.as_slice_mut().unwrap(),
                &obs_context.tile_xyzs,
                hadec_obs,
            );
            setup_ws(
                tile_ws_src.as_slice_mut().unwrap(),
                &obs_context.tile_xyzs,
                hadec_src,
            );
        }

        vis_rotate_fb(
            vis_fb.view(),
            vis_rot_fb.view_mut(),
            tile_ws_obs.as_slice_mut().unwrap(),
            tile_ws_src.as_slice_mut().unwrap(),
            &lambdas_m,
        );
    }

    // display_vis_tfb(
    //     &"model@phase".into(),
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

    // rotated vis should always have the source in phase, so no angle in pols XX, YY
    for vis_rot in vis_rot_tfb.iter() {
        assert_abs_diff_eq!(vis_rot[0].arg(), 0., epsilon = 1e-6); // XX
        assert_abs_diff_eq!(vis_rot[3].arg(), 0., epsilon = 1e-6); // YY
    }

    if !apply_precession {
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
                let arg = (TAU * (w_src - w_obs) / lambda_m) as f64;
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
    #[rustfmt::ignore]
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

    #[rustfmt::ignore]
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
    let apply_precession = false;
    let obs_context = get_simple_obs_context();
    let array_pos = obs_context.array_position.unwrap();

    // second timestep is at 1h
    let hour_epoch = obs_context.timestamps[1];
    let num_tiles = obs_context.get_num_unflagged_tiles();
    let num_times = obs_context.timestamps.len();
    let num_baselines = (num_tiles * (num_tiles - 1)) / 2;
    let ant_pairs = (0..num_baselines)
        .into_iter()
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
        obs_context.dut1.unwrap(),
    );
    let source_radec =
        RADec::from_hadec(HADec::from_radians(0., array_pos.latitude_rad), lst_1h_rad);
    let source_fd = 1.;
    let source_list = SourceList::from(indexmap! {
        "One".into() => point_src_i!(source_radec, 0., fine_chan_freqs_hz[0], source_fd),
    });

    let beam = get_beam(num_tiles);

    let mut modeller = new_sky_modeller(
        #[cfg(feature = "cuda")]
        false,
        beam.deref(),
        &source_list,
        &obs_context.tile_xyzs,
        &fine_chan_freqs_hz,
        &flagged_tiles,
        obs_context.phase_centre,
        array_pos.longitude_rad,
        array_pos.latitude_rad,
        obs_context.dut1.unwrap(),
        apply_precession,
    )
    .unwrap();

    let mut vis_tfb = Array3::default((num_times, num_chans, num_baselines));
    model_timesteps(
        modeller.deref_mut(),
        &obs_context.timestamps,
        vis_tfb.view_mut(),
    )
    .unwrap();

    let mut vis_iono_tfb = Array3::default((num_times, num_chans, num_baselines));

    // tile uvs and ws in the source phase centre
    let mut tile_uvs_src = Array2::default((num_times, num_tiles));
    let mut tile_ws_src = Array2::default((num_times, num_tiles));

    setup_uvw_arrays(
        tile_uvs_src.view_mut(),
        tile_ws_src.view_mut(),
        &obs_context,
        source_radec,
        apply_precession,
    );

    // we want consts such that at lambda = 2m, the shift moves the source to the phase centre
    let iono_lmn = source_radec.to_lmn(obs_context.phase_centre);
    let consts_lm = (iono_lmn.l / 4., iono_lmn.m / 4.);
    // let consts_lm = ((lst_1h_rad-lst_zenith_rad)/4., 0.);

    apply_iono2(
        vis_tfb.view(),
        vis_iono_tfb.view_mut(),
        tile_uvs_src.view(),
        consts_lm,
        &lambdas_m,
    );

    // display_vis_tfb(
    //     &"model@phase".into(),
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
            for (vis, vis_iono) in izip!(vis_fb.slice(s![.., 1]), vis_iono_fb.slice(s![.., 1]),) {
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
                    assert_abs_diff_eq!(vis_iono[3].arg(), 0., epsilon = 1e-6); // YY
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
                let arg = (TAU * (u * consts_lm.0 + v * consts_lm.1) * lambda_m) as f64;
                for (pol_model, pol_model_iono) in vis.iter().zip_eq(vis_iono.iter()) {
                    // magnitudes shoud not be affected by iono rotation
                    assert_abs_diff_eq!(pol_model.norm(), pol_model_iono.norm(), epsilon = 1e-6);
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

#[test]
/// test iono_fit, where residual is just iono rotated model
fn test_iono_fit_easy() {
    let apply_precession = false;
    let obs_context = get_simple_obs_context();
    let array_pos = obs_context.array_position.unwrap();

    // second timestep is at 1h
    let hour_epoch = obs_context.timestamps[1];

    let num_tiles = obs_context.get_num_unflagged_tiles();
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
        obs_context.dut1.unwrap(),
    );
    let source_radec =
        RADec::from_hadec(HADec::from_radians(0., array_pos.latitude_rad), lst_1h_rad);
    let source_fd = 1.;
    let source_list = SourceList::from(indexmap! {
        "One".into() => point_src_i!(source_radec, 0., fine_chan_freqs_hz[0], source_fd),
    });

    let beam = get_beam(num_tiles);

    // unlike the other tests, this is in the SOURCE phase centre
    let mut modeller = new_sky_modeller(
        #[cfg(feature = "cuda")]
        false,
        beam.deref(),
        &source_list,
        &obs_context.tile_xyzs,
        &fine_chan_freqs_hz,
        &flagged_tiles,
        source_radec,
        array_pos.longitude_rad,
        array_pos.latitude_rad,
        obs_context.dut1.unwrap(),
        apply_precession,
    )
    .unwrap();

    let mut vis_tfb = Array3::<Jones<f32>>::zeros((num_times, num_chans, num_baselines));
    model_timesteps(
        modeller.deref_mut(),
        &obs_context.timestamps,
        vis_tfb.view_mut(),
    )
    .unwrap();

    let mut vis_iono_tfb = Array3::<Jones<f32>>::zeros((num_times, num_chans, num_baselines));

    // tile uvs and ws in the source phase centre
    let mut tile_uvs_src = Array2::default((num_times, num_tiles));
    let mut tile_ws_src = Array2::default((num_times, num_tiles));

    setup_uvw_arrays(
        tile_uvs_src.view_mut(),
        tile_ws_src.view_mut(),
        &obs_context,
        source_radec,
        apply_precession,
    );

    // display_vis_tfb(
    //     &"model@phase".into(),
    //     vis_tfb.view(),
    //     &obs_context,
    //     obs_context.phase_centre,
    //     apply_precession,
    // );

    let shape = vis_tfb.shape();
    let weights = Array3::ones((shape[0], shape[1], shape[2]));

    for consts_lm in [(0.0001, -0.0003), (0.0003, -0.0001), (-0.0007, 0.0001)] {
        apply_iono2(
            vis_tfb.view(),
            vis_iono_tfb.view_mut(),
            tile_uvs_src.view(),
            consts_lm,
            &lambdas_m,
        );

        // display_vis_tfb(
        //     &"iono@phase".into(),
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

        assert_abs_diff_eq!(results[0], consts_lm.0, epsilon = 1e-6);
        assert_abs_diff_eq!(results[1], consts_lm.1, epsilon = 1e-6);
    }
}

#[test]
/// - synthesize model visibilities
/// - apply ionospheric rotation
/// - create residual: ionospheric - model
/// - ap ply_iono3 should result in empty visibilitiesiono rotated model
fn test_apply_iono3() {
    let apply_precession = false;
    let obs_context = get_simple_obs_context();
    let array_pos = obs_context.array_position.unwrap();

    let num_tiles = obs_context.get_num_unflagged_tiles();
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
        obs_context.dut1.unwrap(),
    );
    let source_radec =
        RADec::from_hadec(HADec::from_radians(0., array_pos.latitude_rad), lst_1h_rad);
    let source_fd = 1.;
    let source_list = SourceList::from(indexmap! {
        "One".into() => point_src_i!(source_radec, 0., fine_chan_freqs_hz[0], source_fd),
    });

    let beam = get_beam(num_tiles);

    let mut modeller = new_sky_modeller(
        #[cfg(feature = "cuda")]
        false,
        beam.deref(),
        &source_list,
        &obs_context.tile_xyzs,
        &fine_chan_freqs_hz,
        &flagged_tiles,
        obs_context.phase_centre,
        array_pos.longitude_rad,
        array_pos.latitude_rad,
        obs_context.dut1.unwrap(),
        apply_precession,
    )
    .unwrap();

    // residual visibilities in the observation phase centre
    let mut vis_residual_obs_tfb =
        Array3::<Jones<f32>>::zeros((num_times, num_chans, num_baselines));
    // model visibilities in the observation phase centre
    let mut vis_model_obs_tfb = Array3::<Jones<f32>>::zeros((num_times, num_chans, num_baselines));
    // iono rotated model visibilities in the observation phase centre
    let mut vis_iono_obs_tfb = Array3::<Jones<f32>>::zeros((num_times, num_chans, num_baselines));
    model_timesteps(
        modeller.deref_mut(),
        &obs_context.timestamps,
        vis_model_obs_tfb.view_mut(),
    )
    .unwrap();

    // tile uvs and ws in the source phase centre
    let mut tile_uvs_src = Array2::default((num_times, num_tiles));
    let mut tile_ws_src = Array2::default((num_times, num_tiles));

    setup_uvw_arrays(
        tile_uvs_src.view_mut(),
        tile_ws_src.view_mut(),
        &obs_context,
        source_radec,
        apply_precession,
    );

    // display_vis_tfb(
    //     &"model@phase".into(),
    //     vis_model_obs_tfb.view(),
    //     &obs_context,
    //     obs_context.phase_centre,
    //     apply_precession,
    // );

    for consts_lm in [(0.0001, -0.0003), (0.0003, -0.0001), (-0.0007, 0.0001)] {
        // apply iono rotation at source phase to model at observation phase
        apply_iono2(
            vis_model_obs_tfb.view(),
            vis_iono_obs_tfb.view_mut(),
            tile_uvs_src.view(),
            consts_lm,
            &lambdas_m,
        );

        // subtract model from iono at observation phase centre
        vis_residual_obs_tfb.assign(&vis_iono_obs_tfb);
        vis_residual_obs_tfb -= &vis_model_obs_tfb;

        apply_iono3(
            vis_model_obs_tfb.view(),
            vis_residual_obs_tfb.view_mut(),
            tile_uvs_src.view(),
            consts_lm,
            &lambdas_m,
        );

        // display_vis_tfb(
        //     &"residual@phase".into(),
        //     vis_residual_obs_tfb.view(),
        //     &obs_context,
        //     obs_context.phase_centre,
        //     apply_precession,
        // );

        for jones_residual in vis_residual_obs_tfb.iter() {
            for pol_residual in jones_residual.iter() {
                assert_abs_diff_eq!(pol_residual.norm(), 0., epsilon = 1e-6);
            }
        }
    }
}

#[test]
//
fn test_peel_cpu() {
    // enable trace
    // let mut builder = env_logger::Builder::from_default_env();
    // builder.target(env_logger::Target::Stdout);
    // builder.format_target(false);
    // builder.filter_level(log::LevelFilter::Trace);
    // builder.init();

    let apply_precession = false;

    // modify obs_context so that timesteps are closer together
    let mut obs_context = get_simple_obs_context();
    let hour_epoch = obs_context.timestamps[1];
    let time_res = Duration::from_f64(1.0, Unit::Second);
    let second_epoch = obs_context.timestamps[0] + time_res;
    obs_context.time_res = Some(time_res);
    obs_context.timestamps[1] = second_epoch;

    let array_pos = obs_context.array_position.unwrap();
    let num_tiles = obs_context.get_num_unflagged_tiles();
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
        obs_context.dut1.unwrap(),
    );
    let source_radec =
        RADec::from_hadec(HADec::from_radians(0., array_pos.latitude_rad), lst_1h_rad);
    let source_fd = 1.;
    let source_list = SourceList::from(indexmap! {
        "One".into() => point_src_i!(source_radec, 0., fine_chan_freqs_hz[0], source_fd),
    });

    let beam = get_beam(num_tiles);

    let mut high_res_modeller = new_sky_modeller(
        #[cfg(feature = "cuda")]
        false,
        beam.deref(),
        &source_list,
        &obs_context.tile_xyzs,
        &fine_chan_freqs_hz,
        &flagged_tiles,
        obs_context.phase_centre,
        array_pos.longitude_rad,
        array_pos.latitude_rad,
        obs_context.dut1.unwrap(),
        apply_precession,
    )
    .unwrap();

    let mut low_res_modeller = new_sky_modeller(
        #[cfg(feature = "cuda")]
        false,
        beam.deref(),
        &source_list,
        &obs_context.tile_xyzs,
        &fine_chan_freqs_hz,
        &flagged_tiles,
        obs_context.phase_centre,
        array_pos.longitude_rad,
        array_pos.latitude_rad,
        obs_context.dut1.unwrap(),
        apply_precession,
    )
    .unwrap();

    // residual visibilities in the observation phase centre
    let mut vis_residual_obs_tfb =
        Array3::<Jones<f32>>::zeros((num_times, num_chans, num_baselines));
    // model visibilities in the observation phase centre
    let mut vis_model_obs_tfb = Array3::<Jones<f32>>::zeros((num_times, num_chans, num_baselines));
    // iono rotated model visibilities in the observation phase centre
    let mut vis_iono_obs_tfb = Array3::<Jones<f32>>::zeros((num_times, num_chans, num_baselines));
    model_timesteps(
        high_res_modeller.deref_mut(),
        &obs_context.timestamps,
        vis_model_obs_tfb.view_mut(),
    )
    .unwrap();

    // tile uvs and ws in the source phase centre
    let mut tile_uvs_src = Array2::default((num_times, num_tiles));
    let mut tile_ws_src = Array2::default((num_times, num_tiles));

    setup_uvw_arrays(
        tile_uvs_src.view_mut(),
        tile_ws_src.view_mut(),
        &obs_context,
        source_radec,
        apply_precession,
    );

    // display_vis_tfb(
    //     &"model@phase".into(),
    //     vis_model_obs_tfb.view(),
    //     &obs_context,
    //     obs_context.phase_centre,
    //     apply_precession,
    // );

    let timeblock = Timeblock {
        index: 0,
        range: 0..2,
        timestamps: obs_context.timestamps.clone(),
        median: obs_context.timestamps[0],
    };

    let vis_shape = vis_residual_obs_tfb.dim();
    let vis_weights = Array3::<f32>::ones(vis_shape);
    let source_weighted_positions = vec![source_radec; 1];
    let num_srcs_to_iono_subtract = source_list.len();

    let multi_progress = MultiProgress::with_draw_target(ProgressDrawTarget::hidden());

    for consts_lm in [(0.0001, -0.0003), (0.0003, -0.0001), (-0.0007, 0.0001)] {
        apply_iono2(
            vis_model_obs_tfb.view(),
            vis_iono_obs_tfb.view_mut(),
            tile_uvs_src.view(),
            consts_lm,
            &lambdas_m,
        );

        // display_vis_tfb(
        //     &"iono@phase".into(),
        //     vis_iono_obs_tfb.view(),
        //     &obs_context,
        //     obs_context.phase_centre,
        //     apply_precession,
        // );

        // subtract model from iono at source phase centre
        vis_residual_obs_tfb.assign(&vis_iono_obs_tfb);
        vis_residual_obs_tfb -= &vis_model_obs_tfb;

        // display_vis_tfb(
        //     &"residual@phase".into(),
        //     vis_residual_obs_tfb.view(),
        //     &obs_context,
        //     obs_context.phase_centre,
        //     apply_precession,
        // );

        let mut iono_consts = vec![(0., 0.); 1];

        peel_cpu(
            vis_residual_obs_tfb.view_mut(),
            vis_weights.view(),
            &timeblock,
            &source_list,
            &mut iono_consts,
            &source_weighted_positions,
            num_srcs_to_iono_subtract,
            &fine_chan_freqs_hz,
            &lambdas_m,
            &lambdas_m,
            &obs_context,
            obs_context.array_position.unwrap(),
            &obs_context.tile_xyzs,
            low_res_modeller.deref_mut(),
            high_res_modeller.deref_mut(),
            obs_context.dut1.unwrap(),
            !apply_precession,
            &multi_progress,
        )
        .unwrap();

        // println!("expected: {:?}, got: {:?}", consts_lm, iono_consts);

        // display_vis_tfb(
        //     &"peeled@phase".into(),
        //     vis_residual_obs_tfb.view(),
        //     &obs_context,
        //     obs_context.phase_centre,
        //     apply_precession,
        // );

        assert_abs_diff_eq!(iono_consts[0].0, consts_lm.0, epsilon = 1e-6);
        assert_abs_diff_eq!(iono_consts[0].1, consts_lm.1, epsilon = 1e-6);

        // peel should perfectly remove the iono rotate model vis
        for jones_residual in vis_residual_obs_tfb.iter() {
            for pol_residual in jones_residual.iter() {
                assert_abs_diff_eq!(pol_residual.norm(), 0., epsilon = 1e-6);
            }
        }
    }
}
