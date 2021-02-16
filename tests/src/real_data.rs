// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
This module provides functions for tests against real data.

Any required gpubox files need to be in hyperdrive's "tests" directory. These
could be symlinked from somewhere else for convenience. Because this data should
not be expected to always be there (it occupies a large volume), any tests using
these functions should have a

#[ignore]

annotation on the test function. "Ignored" tests are with something like:

`cargo test -- --ignored`
 */

use super::*;

pub struct MwaData {
    /// The MWA observation GPS time.
    pub obsid: u32,

    /// The metafits file associated with the observation.
    pub metafits: String,

    /// Raw MWA gpubox files.
    pub gpuboxes: Vec<String>,

    /// cotter mwaf files. Can be empty.
    pub mwafs: Vec<String>,

    /// Sky-model source list.
    pub source_list: Option<String>,
}

/// Get the calibration arguments associated with the 1065880128 observation.
pub fn get_1065880128() -> MwaData {
    let mut gpuboxes = vec![];
    let mut mwafs = vec![];
    let mut srclist = String::new();
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
        "srclist_pumav3_EoR0aegean_EoR1pietro+ForA_1065880128_100.yaml",
    ] {
        let pathbuf = PathBuf::from(format!("tests/{}", f));
        assert!(
            pathbuf.exists(),
            "Could not find {}, which is required for this test",
            f
        );
        if f.contains("gpubox") {
            gpuboxes.push(path_to_string(&pathbuf));
        } else if f.contains("mwaf") {
            mwafs.push(path_to_string(&pathbuf));
        } else if f.contains("srclist") {
            srclist = path_to_string(&pathbuf);
        }
    }

    // We already have the real metafits file in hyperdrive's git repo.
    let metafits = "tests/1065880128.metafits";
    let metafits_pb = PathBuf::from(&metafits);
    assert!(metafits_pb.exists());

    MwaData {
        obsid: 1065880128,
        metafits: metafits.to_string(),
        gpuboxes,
        mwafs,
        source_list: Some(srclist),
    }
}
