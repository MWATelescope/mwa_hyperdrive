// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
This module tests the "calibrate" command-line interface in hyperdrive *without*
real data. It runs the program in various ways to hopefully ensure that it
continues to work as expected.

These tests won't do much computation, because e.g. mwalib complains that the
gpubox files aren't real. But, we can still determine if our CLI-reading code
behaves as expected.
 */

use super::*;

#[cfg(test)]
mod tests {
    use super::*;
    use mwa_hyperdrive::calibrate::args::CalibrateUserArgs;

    /// This function creates a temporary working directory to test files. It
    /// copies the existing metafits, mwaf and source list files to the temp
    /// directory. The results of this function can be manipulated by each unit
    /// test to test various configurations of files.
    fn get_args() -> (CalibrateUserArgs, TempDir) {
        // Make some dummy gpubox files.
        let tmp_dir = TempDir::new().expect("couldn't make tmp dir");

        let mut gpuboxes = vec![];
        for &f in &[
            "1065880128_20131015134830_gpubox01_00.fits",
            "1065880128_20131015134830_gpubox02_00.fits",
        ] {
            let (pathbuf, _) = make_file_in_dir(f, tmp_dir.path());
            gpuboxes.push(path_to_string(&pathbuf));
        }

        // Get the real mwaf files. These are gzipped.
        let mut mwafs = vec![];
        for &f in &["1065880128_01.mwaf", "1065880128_02.mwaf"] {
            let (mwaf_pb, mut mwaf_file) = make_file_in_dir(f, tmp_dir.path());
            let real_filename = format!("tests/{}.gz", f);
            deflate_gz(&real_filename, &mut mwaf_file);
            mwafs.push(path_to_string(&mwaf_pb));
        }

        // The metafits file can be real.
        let (metafits_pb, mut metafits_file) =
            make_file_in_dir(&"1065880128.metafits", tmp_dir.path());
        {
            let mut real_meta = File::open("tests/1065880128.metafits").unwrap();
            std::io::copy(&mut real_meta, &mut metafits_file).unwrap();
        }

        // Copy the source list.
        let (source_list_pb, mut source_list_file) = make_file_in_dir(
            &"pumav3_EoR0aegean_EoR1pietro+ForA_1065880128_2000.yaml",
            tmp_dir.path(),
        );
        {
            let mut real_source_list =
                File::open("tests/pumav3_EoR0aegean_EoR1pietro+ForA_1065880128_2000.yaml").unwrap();
            std::io::copy(&mut real_source_list, &mut source_list_file).unwrap();
        }

        let args = CalibrateUserArgs {
            metafits: metafits_pb.to_str().map(|s| s.to_string()),
            gpuboxes: Some(gpuboxes),
            mwafs: Some(mwafs),
            source_list: Some(source_list_pb.display().to_string()),
            source_list_type: None,
            num_sources: Some(1000),
            veto_threshold: Some(0.01),
            time_res: None,
            freq_res: None,
        };

        (args, tmp_dir)
    }

    #[test]
    /// Make a toml parameter file with all of the absolute paths to files.
    fn toml_file_absolute_paths() {
        let (args, tmp_dir) = get_args();

        let (toml_pb, mut toml) = make_file_in_dir(&"calibrate.toml", tmp_dir.path());
        serialise_args_toml(&args, &mut toml);
        let cmd = hyperdrive()
            .arg("calibrate")
            .arg(&format!("{}", toml_pb.display()))
            .ok();
        // The command fails because mwalib can't read the fake gpubox files.
        assert!(cmd.is_err());
        let (_, stderr) = get_cmd_output(cmd);
        assert!(&stderr.contains("mwalib error:"));
        assert!(&stderr.contains("tried to move past end of file"));
    }

    #[test]
    /// Make a json parameter file with all of the absolute paths to files.
    fn json_file_absolute_paths() {
        let (args, tmp_dir) = get_args();

        let (json_pb, mut json) = make_file_in_dir(&"calibrate.json", tmp_dir.path());
        serialise_args_json(&args, &mut json);
        let cmd = hyperdrive()
            .current_dir(tmp_dir.path())
            .arg("calibrate")
            .arg(&format!("{}", json_pb.display()))
            .ok();
        // The command fails because mwalib can't read the fake gpubox files.
        assert!(cmd.is_err());
        let (_, stderr) = get_cmd_output(cmd);
        assert!(&stderr.contains("mwalib error:"));
        assert!(&stderr.contains("tried to move past end of file"));
    }

    #[test]
    /// Make a toml parameter file with globs of absolute paths to files.
    fn toml_file_absolute_globs() {
        let (mut args, tmp_dir) = get_args();
        let tmp_dir_str = path_to_string(tmp_dir.path());

        args.metafits = Some(format!("{}/*.metafits", tmp_dir_str));
        args.gpuboxes = Some(vec![format!("{}/*gpubox*", tmp_dir_str)]);
        args.mwafs = Some(vec![format!("{}/*.mwaf", tmp_dir_str)]);

        let (toml_pb, mut toml) = make_file_in_dir(&"calibrate.toml", tmp_dir.path());
        serialise_args_toml(&args, &mut toml);
        let cmd = hyperdrive()
            .arg("calibrate")
            .arg(&format!("{}", toml_pb.display()))
            .ok();
        // The command fails because mwalib can't read the fake gpubox files.
        assert!(cmd.is_err());
        let (_, stderr) = get_cmd_output(cmd);
        assert!(&stderr.contains("mwalib error:"));
        assert!(&stderr.contains("tried to move past end of file"));
    }

    #[test]
    /// Make a json parameter file with globs of absolute paths to files.
    fn json_file_absolute_globs() {
        let (mut args, tmp_dir) = get_args();
        let tmp_dir_str = path_to_string(tmp_dir.path());

        args.metafits = Some(format!("{}/*.metafits", tmp_dir_str));
        args.gpuboxes = Some(vec![format!("{}/*gpubox*", tmp_dir_str)]);
        args.mwafs = Some(vec![format!("{}/*.mwaf", tmp_dir_str)]);

        let (json_pb, mut json) = make_file_in_dir(&"calibrate.json", tmp_dir.path());
        serialise_args_json(&args, &mut json);
        let cmd = hyperdrive()
            .arg("calibrate")
            .arg(&format!("{}", json_pb.display()))
            .ok();
        // The command fails because mwalib can't read the fake gpubox files.
        assert!(cmd.is_err());
        let (_, stderr) = get_cmd_output(cmd);
        assert!(&stderr.contains("mwalib error:"));
        assert!(&stderr.contains("tried to move past end of file"));
    }

    #[test]
    /// Make a toml parameter file with globs of relative paths to files.
    fn toml_file_relative_globs() {
        let (mut args, tmp_dir) = get_args();

        args.metafits = Some(format!("*.metafits"));
        args.gpuboxes = Some(vec![format!("*gpubox*")]);
        args.mwafs = Some(vec![format!("*.mwaf")]);
        args.source_list = Some("pumav3_*.yaml".to_string());

        let (toml_pb, mut toml) = make_file_in_dir(&"calibrate.toml", tmp_dir.path());
        serialise_args_toml(&args, &mut toml);
        let cmd = hyperdrive()
            .current_dir(tmp_dir.path())
            .arg("calibrate")
            .arg(&format!("{}", toml_pb.display()))
            .ok();
        // The command fails because mwalib can't read the fake gpubox files.
        assert!(cmd.is_err());
        let (_, stderr) = get_cmd_output(cmd);
        assert!(&stderr.contains("mwalib error:"));
        assert!(&stderr.contains("tried to move past end of file"));
    }

    #[test]
    /// Make a json parameter file with globs of relative paths to files.
    fn json_file_relative_globs() {
        let (mut args, tmp_dir) = get_args();

        args.metafits = Some(format!("*.metafits"));
        args.gpuboxes = Some(vec![format!("*gpubox*")]);
        args.mwafs = Some(vec![format!("*.mwaf")]);

        let (json_pb, mut json) = make_file_in_dir(&"calibrate.json", tmp_dir.path());
        serialise_args_json(&args, &mut json);
        let cmd = hyperdrive()
            .current_dir(tmp_dir.path())
            .arg("calibrate")
            .arg(&format!("{}", json_pb.display()))
            .ok();
        // The command fails because mwalib can't read the fake gpubox files.
        assert!(cmd.is_err());
        let (_, stderr) = get_cmd_output(cmd);
        assert!(&stderr.contains("mwalib error:"));
        assert!(&stderr.contains("tried to move past end of file"));
    }

    // The above tests should establish that reading from toml or json files
    // works fine. We can do more tests with only toml.

    #[test]
    /// Make a toml parameter file with too many file globs.
    fn param_file_multiple_globs() {
        let (mut args, tmp_dir) = get_args();

        args.metafits = Some(format!("*.metafits"));
        // Even though the first glob is valid, hyperdrive will fail (via
        // mwalib), because it only handles globs when there is a single element
        // in the vector; hyperdrive will treat these strings as real file
        // paths, and mwalib blows up when it can't access them.
        args.gpuboxes = Some(vec![format!("*gpubox*"), format!("*asdf*")]);
        args.mwafs = Some(vec![format!("*.mwaf")]);

        let (toml_pb, mut toml) = make_file_in_dir(&"calibrate.toml", tmp_dir.path());
        serialise_args_toml(&args, &mut toml);

        let cmd = hyperdrive()
            .current_dir(tmp_dir.path())
            .arg("calibrate")
            .arg(&format!("{}", toml_pb.display()))
            .ok();
        assert!(cmd.is_err());
        let (_, stderr) = get_cmd_output(cmd);
        assert!(
            &stderr.contains("mwalib error: Could not identify the gpubox filename structure for")
        );
    }

    #[test]
    /// Make a toml parameter file without the metafits.
    fn param_file_missing_metafits() {
        let (mut args, tmp_dir) = get_args();

        args.metafits = None;
        args.gpuboxes = Some(vec![format!("*gpubox*")]);
        args.mwafs = Some(vec![format!("*.mwaf")]);

        let (toml_pb, mut toml) = make_file_in_dir(&"calibrate.toml", tmp_dir.path());
        serialise_args_toml(&args, &mut toml);

        let cmd = hyperdrive()
            .current_dir(tmp_dir.path())
            .arg("calibrate")
            .arg(&format!("{}", toml_pb.display()))
            .ok();
        assert!(cmd.is_err());
        let (_, stderr) = get_cmd_output(cmd);
        assert!(&stderr.contains("Error: No metafits file supplied"));

        // Try again, but this time specifying the metafits file from the CLI.
        let cmd = hyperdrive()
            .current_dir(tmp_dir.path())
            .arg("calibrate")
            .arg(&format!("{}", toml_pb.display()))
            .arg("--metafits=*.metafits")
            .ok();
        // The command fails because mwalib can't read the fake gpubox files.
        assert!(cmd.is_err());
        let (_, stderr) = get_cmd_output(cmd);
        assert!(&stderr.contains("mwalib error:"));
        assert!(&stderr.contains("tried to move past end of file"));
    }

    #[test]
    /// Make a toml parameter file without the gpubox files.
    fn param_file_missing_gpuboxes() {
        let (mut args, tmp_dir) = get_args();

        args.metafits = Some(format!("*.metafits"));
        args.gpuboxes = None;
        args.mwafs = Some(vec![format!("*.mwaf")]);

        let (toml_pb, mut toml) = make_file_in_dir(&"calibrate.toml", tmp_dir.path());
        serialise_args_toml(&args, &mut toml);

        let cmd = hyperdrive()
            .current_dir(tmp_dir.path())
            .arg("calibrate")
            .arg(&format!("{}", toml_pb.display()))
            .ok();
        assert!(cmd.is_err());
        let (_, stderr) = get_cmd_output(cmd);
        assert!(&stderr.contains("Error: No gpubox files supplied"));

        // Try again, but this time specifying the gpubox files from the CLI.
        let cmd = hyperdrive()
            .current_dir(tmp_dir.path())
            .arg("calibrate")
            .arg(&format!("{}", toml_pb.display()))
            .arg("--gpuboxes=*gpubox*")
            .ok();
        // The command fails because mwalib can't read the fake gpubox files.
        assert!(cmd.is_err());
        let (_, stderr) = get_cmd_output(cmd);
        assert!(&stderr.contains("mwalib error:"));
        assert!(&stderr.contains("tried to move past end of file"));
    }

    #[test]
    /// Make a toml parameter file without the mwaf files.
    fn param_file_missing_mwafs() {
        let (mut args, tmp_dir) = get_args();

        args.metafits = Some(format!("*.metafits"));
        args.gpuboxes = Some(vec![format!("*gpubox*")]);
        args.mwafs = None;

        let (toml_pb, mut toml) = make_file_in_dir(&"calibrate.toml", tmp_dir.path());
        serialise_args_toml(&args, &mut toml);

        // Because the mwaf files are optional, the mwalib error will occur as
        // with all other tests. hyperdrive requires a mwalib context to analyse
        // the cotter flags, so we can't test any further.
        let cmd = hyperdrive()
            .current_dir(tmp_dir.path())
            .arg("calibrate")
            .arg(&format!("{}", toml_pb.display()))
            .ok();
        assert!(cmd.is_err());
        let (_, stderr) = get_cmd_output(cmd);
        assert!(&stderr.contains("mwalib error:"));
        assert!(&stderr.contains("tried to move past end of file"));
    }
}
