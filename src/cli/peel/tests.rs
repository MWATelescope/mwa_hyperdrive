//! Tests against peeling

use crate::{
    beam::{create_fee_beam_object, Beam, Delays},
    model::new_sky_modeller,
    srclist::{
        ComponentType, FluxDensity, FluxDensityType, Source,
        SourceComponent, SourceList,
    }, averaging::timesteps_to_timeblocks,
};
use approx::assert_abs_diff_eq;
use hifitime::{Duration, Epoch, Unit};
use indexmap::indexmap;
use itertools::{multiunzip, Itertools, izip};
use marlu::{
    constants::VEL_C,
    math::cross_correlation_baseline_to_tiles,
    precession::{get_lmst, precess_time},
    Complex, HADec, Jones, LatLngHeight,
    RADec, XyzGeodetic,
};
use ndarray::{s, Array2, Array3, ArrayView2, array};
use num_traits::Zero;
use std::{
    f64::consts::{TAU, PI},
    ops::{Deref, DerefMut},
    path::PathBuf,
    collections::HashSet
};
use vec1::vec1;

use super::{model_timesteps, vis_rotate_fb, setup_ws, setup_uvs, UV, W, vis_average2, weights_average, apply_iono2, iono_fit};

// a single-component point source, stokes I.
macro_rules! point_source_i {
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
        // Array2::from_elem((tile_xyzs.len(), 32), 1.)
    )
    .unwrap();
    beam
}

// get a timestamp at lmst=0 around the year 2100
// precessing to j2000 will introduce a noticable difference.
fn get_j2100(array_pos: &LatLngHeight, dut1: Duration) -> Epoch {
    let mut epoch = Epoch::from_gregorian_utc_at_midnight(2100, 1, 1);

    // shift zenith_time to the nearest time when the phase centre is at zenith
    let sidereal2solar = 365.24 / 366.24;
    let obs_lst_rad = get_lmst(array_pos.longitude_rad, epoch, dut1);
    if obs_lst_rad.abs() > 1e-6 {
        epoch -= Duration::from_f64(sidereal2solar * obs_lst_rad / TAU, Unit::Day);
    }
    epoch
}

// tile "o" is at origin
// tile "u" has a u-component of s
// tile "v" has a v-component of s
fn get_simple_tiles(s_: f64) -> (Vec<String>, Vec<XyzGeodetic>) {
    #[rustfmt::skip]
    // todo: what about an unzip_vecs macro?
    let (tile_names, tile_xyzs): (Vec<String>, Vec<XyzGeodetic>) = multiunzip(vec![
        ("o", XyzGeodetic { x: 0., y: 0., z: 0., }),
        // ("u", XyzGeodetic { x: s., y: 0., z: 0., }),
        ("u", XyzGeodetic { x: 0., y: s_, z: 0., }),
        // ("v", XyzGeodetic { x: 0., y: s., z: 0., }),
        ("v", XyzGeodetic { x: 0., y: 0., z: s_, }),
        // ("w", XyzGeodetic { x: 0., y: 0., z: s., }),
        // ("w", XyzGeodetic { x: s_, y: 0., z: 0., }),
        // ("d", XyzGeodetic { x: 0., y: 0., z: 2. * s_, }),
    ].into_iter().map(|(n,x)| (n.into(), x)));
    (tile_names, tile_xyzs)
}

// // display a vis in table format, slice is in baseline dimensions
// #[allow(clippy::too_many_arguments)]
// fn display_vis_b(
//     vis_b: &[Jones<f32>],
//     ant_pairs: &[(usize, usize)],
//     uvs: &[UV],
//     ws: &[W],
//     tile_names: &[String],
//     seconds: f64,
//     lambda: f64
// ) {
//     for (jones, &(ant1, ant2)) in vis_b.iter().zip_eq(ant_pairs.iter()) {
//         let uv = uvs[ant1] - uvs[ant2];
//         let w = ws[ant1] - ws[ant2];
//         let (name1, name2) = (&tile_names[ant1], &tile_names[ant2]);
//         let (xx, xy, yx, yy) = jones.iter().collect_tuple().unwrap();
//         println!(
//             "+{:>8.3} | {:+1.3}m | {:1}-{:1} {:+1.3} {:+1.3} {:+1.3} | \
//                 XX {:07.5} @{:+08.5}pi XY {:07.5} @{:+08.5}pi \
//                 YX {:07.5} @{:+08.5}pi YY {:07.5} @{:+08.5}pi",
//             seconds, lambda, name1, name2, uv.u / lambda, uv.v / lambda, w / lambda,
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
//     println!("{:9} | {:7} | bl  u      v      w      | {}", "time", "lam", name);
//     for (vis_b, &lambda) in vis_fb.outer_iter().zip_eq(lambdas_m.iter()) {
//         display_vis_b(vis_b.as_slice().unwrap(), ant_pairs, uvs, ws, tile_names, seconds, lambda);
//     }
// }

// tests vis_rotate_fb, setup_uvs, setup_ws
#[test]
fn validate_setup_uv() {
    let apply_precession = false;
    let array_pos = LatLngHeight {
        longitude_rad: 0.,
        latitude_rad: 0.,
        height_metres: 100.,
    };

    let dut1: Duration = Duration::from_f64(0.0, Unit::Second);
    let obs_time = get_j2100(&array_pos, dut1);

    // at first timestep phase centre is at zenith
    let lst_zenith_rad = get_lmst(array_pos.longitude_rad, obs_time, dut1);
    let phase_centre = RADec::from_hadec(HADec::from_radians(0., array_pos.latitude_rad), lst_zenith_rad);

    // second timestep is at 1h
    let hour_epoch = obs_time + Duration::from_f64(1.0, Unit::Hour);
    let timestamps = vec![obs_time, hour_epoch];

    let (_, tile_xyzs) = get_simple_tiles(1.);

    let num_tiles = tile_xyzs.len();

    let lst_1h_rad = get_lmst(array_pos.longitude_rad, hour_epoch, dut1);
    let source_radec = RADec::from_hadec(HADec::from_radians(0., array_pos.latitude_rad), lst_1h_rad);

    let mut tile_uvs_phase = Array2::default((timestamps.len(), num_tiles));
    let mut tile_ws_phase = Array2::default((timestamps.len(), num_tiles));
    let mut tile_uvs_source = Array2::default((timestamps.len(), num_tiles));
    let mut tile_ws_source = Array2::default((timestamps.len(), num_tiles));

    // iterate over time
    for (
        &time,
        mut tile_uvs_phase,
        mut tile_ws_phase,
        mut tile_uvs_source,
        mut tile_ws_source,
    ) in izip!(
        timestamps.iter(),
        tile_uvs_phase.outer_iter_mut(),
        tile_ws_phase.outer_iter_mut(),
        tile_uvs_source.outer_iter_mut(),
        tile_ws_source.outer_iter_mut(),
    ) {
        if apply_precession {
            let precession_info = precess_time(
                array_pos.longitude_rad,
                array_pos.latitude_rad,
                phase_centre,
                time,
                dut1,
            );
            let hadec_phase = phase_centre.to_hadec(precession_info.lmst_j2000);
            let hadec_source = source_radec.to_hadec(precession_info.lmst_j2000);
            let precessed_xyzs = precession_info.precess_xyz(&tile_xyzs);
            setup_uvs(tile_uvs_phase.as_slice_mut().unwrap(), &precessed_xyzs, hadec_phase);
            setup_ws(tile_ws_phase.as_slice_mut().unwrap(), &precessed_xyzs, hadec_phase);
            setup_uvs(tile_uvs_source.as_slice_mut().unwrap(), &precessed_xyzs, hadec_source);
            setup_ws(tile_ws_source.as_slice_mut().unwrap(), &precessed_xyzs, hadec_source);
        } else {
            let lmst = get_lmst(array_pos.longitude_rad, time, dut1);
            let hadec_phase = phase_centre.to_hadec(lmst);
            let hadec_source = source_radec.to_hadec(lmst);
            setup_uvs(tile_uvs_phase.as_slice_mut().unwrap(), &tile_xyzs, hadec_phase);
            setup_ws(tile_ws_phase.as_slice_mut().unwrap(), &tile_xyzs, hadec_phase);
            setup_uvs(tile_uvs_source.as_slice_mut().unwrap(), &tile_xyzs, hadec_source);
            setup_ws(tile_ws_source.as_slice_mut().unwrap(), &tile_xyzs, hadec_source);
        }
    }

    if !apply_precession {
        for i in 0..num_tiles {
            // uvws for the phase centre at first timestpe should be the same as
            // uvws for the source at the second timestep
            assert_abs_diff_eq!(tile_uvs_phase[[0, i]], tile_uvs_source[[1, i]], epsilon = 1e-6);
            assert_abs_diff_eq!(tile_ws_phase[[0, i]], tile_ws_source[[1, i]], epsilon = 1e-6);
            // uvws for the phase centre at the second timestep should be the same as
            // uvws for the source at the first timestep, rotated in the opposite direciton.
            // since all the baselines sit flat on the uv plane, only the w component is negative.
            assert_abs_diff_eq!(tile_uvs_phase[[1, i]], tile_uvs_source[[0, i]], epsilon = 1e-6);
            assert_abs_diff_eq!(tile_ws_phase[[1, i]], -tile_ws_source[[0, i]], epsilon = 1e-6);
            // tile 2 is a special case, with only a v component, so should be unchanged
            if i == 2 {
                assert_abs_diff_eq!(tile_uvs_phase[[0, i]], tile_uvs_source[[0, i]], epsilon = 1e-6);
                assert_abs_diff_eq!(tile_ws_phase[[0, i]], tile_ws_source[[0, i]], epsilon = 1e-6);
                assert_abs_diff_eq!(tile_uvs_phase[[1, i]], tile_uvs_source[[1, i]], epsilon = 1e-6);
                assert_abs_diff_eq!(tile_ws_phase[[1, i]], tile_ws_source[[1, i]], epsilon = 1e-6);
            }
        }
    } else {
        panic!("no tests for precession yet");
    }
}

#[test]
// tests vis_rotate_fb, setup_uvs, setup_ws
// simulate vis, where at the first timestep, the phase centre is at zenith
// and at the second timestep, 1h later, the source is at zenith
fn validate_vis_rotation() {
    let apply_precession = false;
    let array_pos = LatLngHeight {
        longitude_rad: 0.,
        latitude_rad: 0.,
        height_metres: 100.,
    };

    let dut1: Duration = Duration::from_f64(0.0, Unit::Second);
    let obs_time = get_j2100(&array_pos, dut1);

    // at first timestep phase centre is at zenith
    let lst_zenith_rad = get_lmst(array_pos.longitude_rad, obs_time, dut1);
    let phase_centre = RADec::from_hadec(HADec::from_radians(0., array_pos.latitude_rad), lst_zenith_rad);

    // second timestep is at 1h
    let hour_epoch = obs_time + Duration::from_f64(1.0, Unit::Hour);
    let timestamps = vec![obs_time, hour_epoch];

    let (tile_names, tile_xyzs) = get_simple_tiles(1.);

    let num_tiles = tile_xyzs.len();
    let num_baselines = (num_tiles * (num_tiles - 1)) / 2;
    let ant_pairs = (0..num_baselines)
        .into_iter()
        .map(|bl_idx| cross_correlation_baseline_to_tiles(num_tiles, bl_idx))
        .collect_vec();
    let flagged_tiles = HashSet::new();

    // lambda = 1m
    let lambdas_m = vec![2., 1.];
    let freqs_hz = lambdas_m.iter().map(|&l| VEL_C / l).collect_vec();

    // source is at zenith at 1h
    let lst_1h_rad = get_lmst(array_pos.longitude_rad, hour_epoch, dut1);
    let source_radec = RADec::from_hadec(HADec::from_radians(0., array_pos.latitude_rad), lst_1h_rad);
    let source_fd = 1.;
    let source_list = SourceList::from(indexmap! {
        "One".into() => point_source_i!(source_radec, 0., freqs_hz[0], source_fd),
    });

    let beam = get_beam(num_tiles);

    let mut modeller = new_sky_modeller(
        #[cfg(feature = "cuda")]
        false,
        beam.deref(),
        &source_list,
        &tile_xyzs,
        &freqs_hz,
        &flagged_tiles,
        phase_centre,
        array_pos.longitude_rad,
        array_pos.latitude_rad,
        dut1,
        apply_precession,
    )
    .unwrap();

    let mut vis_tfb = Array3::from_elem((timestamps.len(), freqs_hz.len(), ant_pairs.len()), Jones::<f32>::zero());
    model_timesteps(modeller.deref_mut(), &timestamps, vis_tfb.view_mut()).unwrap();

    let mut vis_rot_tfb = Array3::from_elem((timestamps.len(), freqs_hz.len(), ant_pairs.len()), Jones::<f32>::zero());

    let mut tile_uvs_phase = Array2::default((timestamps.len(), num_tiles));
    let mut tile_ws_phase = Array2::default((timestamps.len(), num_tiles));
    let mut tile_uvs_source = Array2::default((timestamps.len(), num_tiles));
    let mut tile_ws_source = Array2::default((timestamps.len(), num_tiles));

    // iterate over time
    let start_seconds = obs_time.to_gpst_seconds();
    for (
        vis_fb,
        mut vis_rot_fb,
        &time,
        mut tile_uvs_phase,
        mut tile_ws_phase,
        mut tile_uvs_source,
        mut tile_ws_source,
    ) in izip!(
        vis_tfb.outer_iter(),
        vis_rot_tfb.view_mut().outer_iter_mut(),
        timestamps.iter(),
        tile_uvs_phase.outer_iter_mut(),
        tile_ws_phase.outer_iter_mut(),
        tile_uvs_source.outer_iter_mut(),
        tile_ws_source.outer_iter_mut(),
    ) {
        if apply_precession {
            let precession_info = precess_time(
                array_pos.longitude_rad,
                array_pos.latitude_rad,
                phase_centre,
                time,
                dut1,
            );
            let hadec_phase = phase_centre.to_hadec(precession_info.lmst_j2000);
            let hadec_source = source_radec.to_hadec(precession_info.lmst_j2000);
            let precessed_xyzs = precession_info.precess_xyz(&tile_xyzs);
            setup_uvs(tile_uvs_phase.as_slice_mut().unwrap(), &precessed_xyzs, hadec_phase);
            setup_ws(tile_ws_phase.as_slice_mut().unwrap(), &precessed_xyzs, hadec_phase);
            setup_uvs(tile_uvs_source.as_slice_mut().unwrap(), &precessed_xyzs, hadec_source);
            setup_ws(tile_ws_source.as_slice_mut().unwrap(), &precessed_xyzs, hadec_source);
        } else {
            let lmst = get_lmst(array_pos.longitude_rad, time, dut1);
            let hadec_phase = phase_centre.to_hadec(lmst);
            let hadec_source = source_radec.to_hadec(lmst);
            setup_uvs(tile_uvs_phase.as_slice_mut().unwrap(), &tile_xyzs, hadec_phase);
            setup_ws(tile_ws_phase.as_slice_mut().unwrap(), &tile_xyzs, hadec_phase);
            setup_uvs(tile_uvs_source.as_slice_mut().unwrap(), &tile_xyzs, hadec_source);
            setup_ws(tile_ws_source.as_slice_mut().unwrap(), &tile_xyzs, hadec_source);
        }

        let seconds = time.to_gpst_seconds() - start_seconds;
        // display_vis_fb(
        //     &"model".into(),
        //     vis_fb.view(),
        //     seconds,
        //     tile_uvs_phase.as_slice().unwrap(),
        //     tile_ws_phase.as_slice().unwrap(),
        //     &lambdas_m,
        //     &ant_pairs,
        //     &tile_names,
        // );

        vis_rotate_fb(
            vis_fb.view(),
            vis_rot_fb.view_mut(),
            tile_ws_phase.as_slice_mut().unwrap(),
            tile_ws_source.as_slice_mut().unwrap(),
            &lambdas_m
        );

        // display_vis_fb(
        //     &"rotated".into(),
        //     vis_rot_fb.view(),
        //     seconds,
        //     tile_uvs_source.as_slice().unwrap(),
        //     tile_ws_source.as_slice().unwrap(),
        //     &lambdas_m,
        //     &ant_pairs,
        //     &tile_names,
        // );
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
        // rotated vis should always have the source at the phase centre, so no angle in pols XX, YY
        for vis_rot in vis_rot_tfb.iter() {
            assert_abs_diff_eq!(vis_rot[0].arg(), 0., epsilon = 1e-6); // XX
            assert_abs_diff_eq!(vis_rot[3].arg(), 0., epsilon = 1e-6); // YY
        }
    }

    for (
        tile_ws_phase,
        tile_ws_source,
        vis_fb,
        vis_rot_fb
    ) in izip!(
        tile_ws_phase.outer_iter(),
        tile_ws_source.outer_iter(),
        vis_tfb.outer_iter(),
        vis_rot_tfb.outer_iter(),
    ) {
        for (
            lambda_m,
            vis_b,
            vis_rot_b,
        ) in izip!(
            lambdas_m.iter(),
            vis_fb.outer_iter(),
            vis_rot_fb.outer_iter(),
        ) {
            for (
                &(ant1, ant2),
                vis,
                vis_rot,
            ) in izip!(
                ant_pairs.iter(),
                vis_b.iter(),
                vis_rot_b.iter(),
             ) {
                let w_phase = tile_ws_phase[ant1] - tile_ws_phase[ant2];
                let w_source = tile_ws_source[ant1] - tile_ws_source[ant2];
                let arg = (TAU * (w_source - w_phase) / lambda_m) as f64;
                for (pol_model, pol_model_rot) in vis.iter().zip_eq(vis_rot.iter()) {
                    // magnitudes shoud not be affected by rotation
                    assert_abs_diff_eq!(pol_model.norm(), pol_model_rot.norm(), epsilon = 1e-6);
                    let pol_model_rot_expected = Complex::from_polar(pol_model.norm(), (pol_model.arg() as f64 - arg) as f32);
                    assert_abs_diff_eq!(pol_model_rot_expected.arg(), pol_model_rot.arg(), epsilon = 1e-6);
                }
            }
        }
    }
}

#[test]
fn validate_weight_average() {
    let weights_tfb: Array3<f32> = array![
        [
            [1., 1., 1., -1.],
            [2., 2., 2., -2.],
            [4., 4., 4., -4.],
        ],
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

    weights_average(
        weights_tfb.view(),
        weights_avg_tfb.view_mut(),
    );

    assert_eq!(weights_avg_tfb, array![
        [
            [27., 27., 27., 0.],
            [36., 36., 36., 0.],
        ],
    ]);
}

#[test]
fn validate_vis_average() {
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

    vis_average2(
        vis_tfb.view(),
        vis_avg_tfb.view_mut(),
        weights_tfb.view(),
    );

    assert_eq!(vis_avg_tfb.slice(s![.., .., 0]), array![[Jones::zero(), Jones::zero()]]);

    assert_eq!(vis_avg_tfb.slice(s![.., 0, 1]), array![Jones::identity() * 24./27.]);
    assert_eq!(vis_avg_tfb.slice(s![.., 0, 2]), array![Jones::identity() * 2./27.]);
    assert_eq!(vis_avg_tfb.slice(s![.., 0, 3]), array![Jones::identity() * 17./27.]);
    assert_eq!(vis_avg_tfb.slice(s![.., 1, 1]), array![Jones::identity() * 32./36.]);
    assert_eq!(vis_avg_tfb.slice(s![.., 1, 2]), array![Jones::identity() * 36./36.]);
    assert_eq!(vis_avg_tfb.slice(s![.., 1, 3]), array![Jones::identity() * 4./36.]);
}
