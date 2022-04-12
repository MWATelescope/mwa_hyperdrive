// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use hifitime::{Duration, Epoch, Unit};
use marlu::{RADec, XyzGeodetic};
use vec1::vec1;

use crate::context::ObsContext;
use mwa_hyperdrive_beam::Delays;
use mwa_hyperdrive_common::{hifitime, marlu, vec1};

fn get_minimal_obs_context() -> ObsContext {
    ObsContext {
        obsid: None,
        timestamps: vec1![Epoch::from_gpst_seconds(1090008640.0)],
        all_timesteps: vec1![0],
        unflagged_timesteps: vec![0],
        phase_centre: RADec::default(),
        pointing_centre: Some(RADec::default()),
        tile_names: vec1!["Tile00".into()],
        tile_xyzs: vec1![XyzGeodetic::default()],
        flagged_tiles: vec![],
        autocorrelations_present: false,
        dipole_delays: Some(Delays::Partial(vec![0; 16])),
        dipole_gains: None,
        time_res: None,
        array_position: None,
        coarse_chan_nums: vec![],
        coarse_chan_freqs: vec![],
        num_fine_chans_per_coarse_chan: 1,
        freq_res: None,
        fine_chan_freqs: vec1![128_000_000],
        flagged_fine_chans: vec![],
        flagged_fine_chans_per_coarse_chan: vec![],
    }
}

#[test]
pub fn test_guess_time_res() {
    let mut obs_ctx = get_minimal_obs_context();

    // test fallback to 1s
    obs_ctx.time_res = None;
    obs_ctx.timestamps = vec1![Epoch::from_gpst_seconds(1090000000.0)];

    assert_eq!(
        obs_ctx.guess_time_res(),
        Duration::from_f64(1., Unit::Second)
    );

    // test use direct time_res over min_delta
    obs_ctx.time_res = Some(Duration::from_f64(2., Unit::Second));
    obs_ctx.timestamps = vec1![
        Epoch::from_gpst_seconds(1090000000.0),
        Epoch::from_gpst_seconds(1090000010.0),
        Epoch::from_gpst_seconds(1090000013.0),
    ];

    assert_eq!(
        obs_ctx.guess_time_res(),
        Duration::from_f64(2., Unit::Second)
    );
}

#[test]
pub fn test_guess_freq_res() {
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
