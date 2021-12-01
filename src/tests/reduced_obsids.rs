// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! This module provides functions for tests against observations that occupy a
//! small amount of data. These tests are useful as it allows hyperdrive to
//! perform units tests without requiring MWA data for a full observation.

use super::*;
use crate::calibrate::args::CalibrateUserArgs;

/// Get the calibration arguments associated with the obsid 1090008640. This
/// observational data is inside the hyperdrive git repo, but has been reduced;
/// there is only 1 coarse channel and 1 timestep.
pub(crate) fn get_1090008640() -> CalibrateUserArgs {
    // Ensure that the required files are there.
    let data = vec![
        "test_files/1090008640/1090008640.metafits".to_string(),
        "test_files/1090008640/1090008640_20140721201027_gpubox01_00.fits".to_string(),
    ];
    for file in &data {
        let pb = PathBuf::from(file);
        assert!(
            pb.exists(),
            "Could not find {}, which is required for this test",
            pb.display()
        );
    }

    let srclist = PathBuf::from(
        "test_files/1090008640/srclist_pumav3_EoR0aegean_EoR1pietro+ForA_1090008640_100.yaml",
    );
    assert!(
        srclist.exists(),
        "Could not find {}, which is required for this test",
        srclist.display()
    );

    CalibrateUserArgs {
        data: Some(data),
        source_list: Some(path_to_string(&srclist)),
        no_beam: true,
        ..Default::default()
    }
}
