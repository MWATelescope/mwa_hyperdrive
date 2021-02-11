// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
 * This module tests hyperdrive calibration with real data.
 *
 * Any required gpubox files need to be in hyperdrive's "tests" directory. These
 * could be symlinked from somewhere else for convenience. Run these "ignored"
 * tests with something like:
 *
 * cargo test -- --ignored
 */

use super::*;

#[cfg(test)]
mod tests {
    use super::*;

    fn get_1065880128() -> (CalibrateUserArgs, TempDir) {
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
            let pathbuf = PathBuf::from(format!("tests/{}", f));
            assert!(
                pathbuf.exists(),
                "Could not find {}, which is required for this test",
                f
            );
            if f.contains("gpubox") {
                gpuboxes.push(path_to_string(&pathbuf));
            }
            if f.contains("mwaf") {
                mwafs.push(path_to_string(&pathbuf));
            }
        }

        // We already have the real metafits file in hyperdrive's git repo.
        let metafits = "tests/1065880128.metafits";
        let metafits_pb = PathBuf::from(&metafits);
        assert!(metafits_pb.exists());

        // Make a toml parameter file for testing.
        let tmp_dir = TempDir::new().expect("couldn't make a temp dir");
        let args = CalibrateUserArgs {
            metafits: metafits_pb.to_str().map(|s| s.to_string()),
            gpuboxes: Some(gpuboxes),
            mwafs: Some(mwafs),
            source_list: Some(
                "tests/pumav3_EoR0aegean_EoR1pietro+ForA_1065880128_2000.yaml".to_string(),
            ),
            source_list_type: None,
            num_sources: Some(1000),
            veto_threshold: Some(0.01),
            time_res: None,
            freq_res: None,
        };

        (args, tmp_dir)
    }

    #[test]
    #[ignore]
    /// Obsid 1065880128 is an ionospherically-active MWA EoR observation. 40
    /// kHz, 0.5s. EoR-0 field. EoR high band (roughly 170-200 MHz).
    ///
    /// Relies on the 48 associated gpubox files being available in the "tests"
    /// directory.
    fn calibrate_1065880128() {
        let (args, tmp_dir) = get_1065880128();
        let (toml_pb, mut toml) = make_file_in_dir(&"calibrate.toml", tmp_dir.path());
        serialise_args_toml(&args, &mut toml);

        // Now we can test hyperdrive.
        //
        // Because the calibration doesn't do anything yet, it will panic unless
        // we turn "dry-run" on. When doing a dry run, hyperdrive will print out
        // high-level information.
        let cmd = hyperdrive()
            .arg("calibrate")
            .arg(&format!("{}", toml_pb.display()))
            .ok();
        assert!(cmd.is_err(), "{:?}", cmd.unwrap());
        let (_, stderr) = get_cmd_output(cmd);
        assert!(&stderr.contains("not yet implemented"), "{}", stderr);

        let cmd = hyperdrive()
            .arg("calibrate")
            .arg(&format!("{}", toml_pb.display()))
            .arg("--dry-run")
            .ok();
        assert!(cmd.is_ok(), "{}", cmd.unwrap_err());
        let (stdout, stderr) = get_cmd_output(cmd);
        assert!(stderr.is_empty(), "{}", stderr);
        // From mwalib context output
        assert!(&stdout.contains("Actual UNIX start time:   1381844910"));
        // From the cotter flags
        assert!(&stdout.contains("occupancy: {1: [1.0, 1.0, 0.06320619, 0.061518785"));
    }
}
