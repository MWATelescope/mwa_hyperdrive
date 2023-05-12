// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use hifitime::Epoch;
use marlu::{LatLngHeight, RADec, XyzGeodetic};
use vec1::vec1;

use super::Polarisations;
use crate::{beam::Delays, context::ObsContext, io::read::VisInputType};

fn get_minimal_obs_context() -> ObsContext {
    ObsContext {
        input_data_type: VisInputType::Raw,
        obsid: None,
        timestamps: vec1![Epoch::from_gpst_seconds(1090008640.0)],
        all_timesteps: vec1![0],
        unflagged_timesteps: vec![0],
        phase_centre: RADec::default(),
        pointing_centre: Some(RADec::default()),
        array_position: LatLngHeight::mwa(),
        supplied_array_position: LatLngHeight::mwa(),
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
        polarisations: Polarisations::default(),
    }
}

#[test]
fn test_veto_freqs() {
    let mut obs_ctx = get_minimal_obs_context();
    obs_ctx.fine_chan_freqs = vec1![182335000, 182415000];

    assert_eq!(&obs_ctx.get_veto_freqs(), &[181760000.0, 183040000.0]);
}
