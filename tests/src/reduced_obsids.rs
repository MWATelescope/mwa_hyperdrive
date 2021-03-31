// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
This module provides functions for tests against observations that occupy a
small amount of data. These tests are useful as it allows hyperdrive to perform
units tests without requiring MWA data for a full observation.
 */

use super::*;

use mwa_hyperdrive::calibrate::args::CalibrateUserArgs;

/// Get the calibration arguments associated with the obsid 1090008640. This
/// observational data is inside the hyperdrive git repo, but has been reduced;
/// there are only 3 coarse bands and 3 timesteps.
pub fn get_1090008640() -> CalibrateUserArgs {
    // Ensure that the required files are there.
    let metafits = PathBuf::from("tests/1090008640/1090008640.metafits");
    assert!(
        metafits.exists(),
        "Could not find {}, which is required for this test",
        metafits.display()
    );

    let mut gpuboxes = vec![];
    // Ensure that the required files are there.
    for &f in &[
        "1090008640_20140721201027_gpubox01_00.fits",
        "1090008640_20140721201027_gpubox02_00.fits",
        "1090008640_20140721201027_gpubox03_00.fits",
    ] {
        let pathbuf = PathBuf::from(format!("tests/1090008640/{}", f));
        assert!(
            pathbuf.exists(),
            "Could not find {}, which is required for this test",
            f
        );
        gpuboxes.push(path_to_string(&pathbuf));
    }

    let srclist = PathBuf::from(
        "tests/1090008640/srclist_pumav3_EoR0aegean_EoR1pietro+ForA_1090008640_100.yaml",
    );
    assert!(
        srclist.exists(),
        "Could not find {}, which is required for this test",
        srclist.display()
    );

    CalibrateUserArgs {
        metafits: Some(metafits.display().to_string()),
        gpuboxes: Some(gpuboxes),
        source_list: Some(srclist.display().to_string()),
        ..Default::default()
    }
}

/// Get the calibration arguments associated with the obsid 1090008640. This
/// observational data is inside the hyperdrive git repo, but has been reduced;
/// there is only 1 coarse band and 3 timesteps.
pub fn get_1090008640_smallest() -> CalibrateUserArgs {
    // We can just use the other function and remove all but the first gpubox
    // file.
    let mut args = get_1090008640();
    let gpuboxes: Option<Vec<String>> = args.gpuboxes;
    let first = gpuboxes.unwrap()[0].clone();
    args.gpuboxes = Some(vec![first]);
    args
}
