// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use hifitime::{Duration, Epoch};
use marlu::{LatLngHeight, RADec, XyzGeodetic};
use vec1::vec1;

use crate::{beam::Delays, context::ObsContext};

fn get_minimal_obs_context() -> ObsContext {
    ObsContext {
        obsid: None,
        timestamps: vec1![Epoch::from_gpst_seconds(1090008640.0)],
        all_timesteps: vec1![0],
        unflagged_timesteps: vec![0],
        phase_centre: RADec::default(),
        pointing_centre: Some(RADec::default()),
        array_position: LatLngHeight::mwa(),
        _supplied_array_position: None,
        dut1: None,
        tile_names: vec1!["Tile00".into()],
        tile_xyzs: vec1![XyzGeodetic::default()],
        flagged_tiles: vec![],
        unavailable_tiles: vec![],
        autocorrelations_present: false,
        dipole_delays: Some(Delays::Partial(vec![0; 16])),
        dipole_gains: None,
        time_res: None,
        mwa_coarse_chan_nums: None,
        num_fine_chans_per_coarse_chan: None,
        freq_res: None,
        fine_chan_freqs: vec1![128_000_000],
        flagged_fine_chans: vec![],
        flagged_fine_chans_per_coarse_chan: None,
    }
}

#[test]
fn test_guess_time_res() {
    let mut obs_ctx = get_minimal_obs_context();

    // test fallback to 1s
    obs_ctx.time_res = None;
    obs_ctx.timestamps = vec1![Epoch::from_gpst_seconds(1090000000.0)];

    assert_eq!(obs_ctx.guess_time_res(), Duration::from_seconds(1.));

    // test use direct time_res over min_delta
    obs_ctx.time_res = Some(Duration::from_seconds(2.));
    obs_ctx.timestamps = vec1![
        Epoch::from_gpst_seconds(1090000000.0),
        Epoch::from_gpst_seconds(1090000010.0),
        Epoch::from_gpst_seconds(1090000013.0),
    ];

    assert_eq!(obs_ctx.guess_time_res(), Duration::from_seconds(2.));
}

#[test]
fn test_guess_freq_res() {
    let mut obs_ctx = get_minimal_obs_context();

    // test fallback to 1s
    obs_ctx.freq_res = None;
    obs_ctx.fine_chan_freqs = vec1![128_000_000];

    assert_eq!(obs_ctx.guess_freq_res(), 10_000.);

    // test use direct freq_res over min_delta
    obs_ctx.freq_res = Some(30_000.);
    obs_ctx.fine_chan_freqs = vec1![128_000_000, 128_100_000, 128_120_000];

    assert_eq!(obs_ctx.guess_freq_res(), 30_000.);
}

#[test]
fn test_veto_freqs() {
    let mut obs_ctx = get_minimal_obs_context();
    obs_ctx.fine_chan_freqs = vec1![182335000, 182415000];

    assert_eq!(&obs_ctx.get_veto_freqs(), &[181760000.0, 183040000.0]);
}
