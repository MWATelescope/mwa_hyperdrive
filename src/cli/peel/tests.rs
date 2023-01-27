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

