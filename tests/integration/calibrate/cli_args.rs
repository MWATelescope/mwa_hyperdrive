// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Tests against the command-line interface for calibration.

mod tests {
    use crate::*;

    #[test]
    fn test_calibrate_help_is_correct() {
        // First with --help
        let cmd = hyperdrive().arg("--help").ok();
        assert!(cmd.is_ok());
        let (stdout, stderr) = get_cmd_output(cmd);
        assert!(stderr.is_empty());
        assert!(stdout.contains("calibrate"));
        assert!(stdout.contains("Perform direction-independent calibration on the input MWA data"));

        // Second with -h
        let cmd = hyperdrive().arg("-h").ok();
        assert!(cmd.is_ok());
        let (stdout, stderr) = get_cmd_output(cmd);
        assert!(stderr.is_empty());
        assert!(stdout.contains("calibrate"));
        assert!(stdout.contains("Perform direction-independent calibration on the input MWA data"));
    }
}
