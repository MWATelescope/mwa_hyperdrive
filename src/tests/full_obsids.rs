// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! This module provides functions for tests against full observations.
//!
//! The required gpubox and/or mwaf files need to be in hyperdrive's
//! "tests/{obsid}" directory. These could be symlinked from somewhere else for
//! convenience. Because this data should not be expected to always be there (it
//! occupies a large volume), any tests using these functions should have a
//!
//! #[ignore]
//!
//! annotation on the test function. "Ignored" tests are run with something
//! like:
//!
//! `cargo test -- --ignored`

use super::*;

/// Get the calibration arguments associated with the 1065880128 observation
/// (including gpubox files and mwaf files).
pub(crate) fn get_1065880128() -> CalibrateUserArgs {
    let metafits = PathBuf::from("test_files/1065880128/1065880128.metafits");
    assert!(
        metafits.exists(),
        "Could not find {}, which is required for this test",
        metafits.display()
    );

    let mut gpuboxes = vec![];
    let mut mwafs = vec![];
    // Ensure that the required files are there.
    for &f in &[
        "1065880128_20131015134830_gpubox01_00.fits",
        "1065880128_20131015134830_gpubox02_00.fits",
        "1065880128_20131015134830_gpubox03_00.fits",
        "1065880128_20131015134830_gpubox04_00.fits",
        "1065880128_20131015134830_gpubox05_00.fits",
        "1065880128_20131015134830_gpubox06_00.fits",
        "1065880128_20131015134830_gpubox07_00.fits",
        "1065880128_20131015134830_gpubox08_00.fits",
        "1065880128_20131015134830_gpubox09_00.fits",
        "1065880128_20131015134830_gpubox10_00.fits",
        "1065880128_20131015134830_gpubox11_00.fits",
        "1065880128_20131015134830_gpubox12_00.fits",
        "1065880128_20131015134830_gpubox13_00.fits",
        "1065880128_20131015134830_gpubox14_00.fits",
        "1065880128_20131015134830_gpubox15_00.fits",
        "1065880128_20131015134830_gpubox16_00.fits",
        "1065880128_20131015134830_gpubox17_00.fits",
        "1065880128_20131015134830_gpubox18_00.fits",
        "1065880128_20131015134830_gpubox19_00.fits",
        "1065880128_20131015134830_gpubox20_00.fits",
        "1065880128_20131015134830_gpubox21_00.fits",
        "1065880128_20131015134830_gpubox22_00.fits",
        "1065880128_20131015134830_gpubox23_00.fits",
        "1065880128_20131015134830_gpubox24_00.fits",
        "1065880128_20131015134930_gpubox01_01.fits",
        "1065880128_20131015134930_gpubox02_01.fits",
        "1065880128_20131015134930_gpubox03_01.fits",
        "1065880128_20131015134930_gpubox04_01.fits",
        "1065880128_20131015134930_gpubox05_01.fits",
        "1065880128_20131015134930_gpubox06_01.fits",
        "1065880128_20131015134930_gpubox07_01.fits",
        "1065880128_20131015134930_gpubox08_01.fits",
        "1065880128_20131015134930_gpubox09_01.fits",
        "1065880128_20131015134930_gpubox10_01.fits",
        "1065880128_20131015134930_gpubox11_01.fits",
        "1065880128_20131015134930_gpubox12_01.fits",
        "1065880128_20131015134930_gpubox13_01.fits",
        "1065880128_20131015134930_gpubox14_01.fits",
        "1065880128_20131015134930_gpubox15_01.fits",
        "1065880128_20131015134930_gpubox16_01.fits",
        "1065880128_20131015134930_gpubox17_01.fits",
        "1065880128_20131015134930_gpubox18_01.fits",
        "1065880128_20131015134930_gpubox19_01.fits",
        "1065880128_20131015134930_gpubox20_01.fits",
        "1065880128_20131015134930_gpubox21_01.fits",
        "1065880128_20131015134930_gpubox22_01.fits",
        "1065880128_20131015134930_gpubox23_01.fits",
        "1065880128_20131015134930_gpubox24_01.fits",
        "1065880128_01.mwaf",
        "1065880128_02.mwaf",
        "1065880128_03.mwaf",
        "1065880128_04.mwaf",
        "1065880128_05.mwaf",
        "1065880128_06.mwaf",
        "1065880128_07.mwaf",
        "1065880128_08.mwaf",
        "1065880128_09.mwaf",
        "1065880128_10.mwaf",
        "1065880128_11.mwaf",
        "1065880128_12.mwaf",
        "1065880128_13.mwaf",
        "1065880128_14.mwaf",
        "1065880128_15.mwaf",
        "1065880128_16.mwaf",
        "1065880128_17.mwaf",
        "1065880128_18.mwaf",
        "1065880128_19.mwaf",
        "1065880128_20.mwaf",
        "1065880128_21.mwaf",
        "1065880128_22.mwaf",
        "1065880128_23.mwaf",
        "1065880128_24.mwaf",
    ] {
        let pathbuf = PathBuf::from(format!("test_files/1065880128/{}", f));
        assert!(
            pathbuf.exists(),
            "Could not find {}, which is required for this test",
            f
        );
        if f.contains("gpubox") {
            gpuboxes.push(path_to_string(&pathbuf));
        } else if f.contains("mwaf") {
            mwafs.push(path_to_string(&pathbuf));
        }
    }

    let srclist = PathBuf::from(
        "test_files/1065880128/srclist_pumav3_EoR0aegean_EoR1pietro+ForA_1065880128_100.yaml",
    );
    assert!(
        srclist.exists(),
        "Could not find {}, which is required for this test",
        srclist.display()
    );

    let mut data = vec![path_to_string(&metafits)];
    data.append(&mut gpuboxes);
    data.append(&mut mwafs);
    CalibrateUserArgs {
        data: Some(data),
        source_list: Some(path_to_string(&srclist)),
        ..Default::default()
    }
}
