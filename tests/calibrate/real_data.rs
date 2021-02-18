// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
This module tests hyperdrive calibration with real data.

Any required gpubox files need to be in hyperdrive's "tests" directory. These
could be symlinked from somewhere else for convenience. Run these "ignored"
tests with something like:

cargo test -- --ignored
 */

#[cfg(test)]
mod tests {
    use mwa_hyperdrive_tests::gpuboxes::*;
    use mwa_hyperdrive_tests::*;

    fn args_1065880128() -> (CalibrateUserArgs, TempDir) {
        let data = get_1065880128();
        let tmp_dir = TempDir::new().expect("couldn't make a temp dir");

        let args = CalibrateUserArgs {
            metafits: Some(data.metafits),
            gpuboxes: Some(data.gpuboxes),
            mwafs: Some(data.mwafs),
            source_list: data.source_list,
            source_list_type: None,
            num_sources: Some(50),
            source_dist_cutoff: None,
            veto_threshold: Some(0.01),
            time_res: None,
            freq_res: None,
            tile_flags: None,
            ignore_metafits_flags: None,
            fine_chan_flags: None,
        };
        (args, tmp_dir)
    }

    #[test]
    #[serial]
    #[ignore]
    /// Obsid 1065880128 is an ionospherically-active MWA EoR observation. 40
    /// kHz, 0.5s. EoR-0 field. EoR high band (roughly 170-200 MHz).
    ///
    /// Relies on the 48 associated gpubox files being available in the "tests"
    /// directory.
    fn calibrate_1065880128() {
        let (args, tmp_dir) = args_1065880128();
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
