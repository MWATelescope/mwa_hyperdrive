// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Tests against the command-line interface for DI calibration.

use itertools::Itertools;

use crate::{get_cmd_output, hyperdrive};

#[test]
fn test_hyperdrive_help_is_correct() {
    let mut stdouts = vec![];

    // First with --help
    let cmd = hyperdrive().arg("--help").ok();
    assert!(cmd.is_ok());
    let (stdout, stderr) = get_cmd_output(cmd);
    assert!(stderr.is_empty());
    stdouts.push(stdout);

    // Then with -h
    let cmd = hyperdrive().arg("-h").ok();
    assert!(cmd.is_ok());
    let (stdout, stderr) = get_cmd_output(cmd);
    assert!(stderr.is_empty());
    stdouts.push(stdout);

    for stdout in stdouts {
        assert!(stdout.contains("di-calibrate"));
        assert!(stdout.contains("Perform direction-independent calibration on the input MWA data"));
    }
}

#[test]
fn test_di_calibrate_help_is_correct() {
    let mut stdouts = vec![];

    // First with --help
    let cmd = hyperdrive().args(["di-calibrate", "--help"]).ok();
    assert!(cmd.is_ok());
    let (stdout, stderr) = get_cmd_output(cmd);
    assert!(stderr.is_empty());
    stdouts.push(stdout);

    // Then with -h
    let cmd = hyperdrive().args(["di-calibrate", "-h"]).ok();
    assert!(cmd.is_ok());
    let (stdout, stderr) = get_cmd_output(cmd);
    assert!(stderr.is_empty());
    stdouts.push(stdout);

    for stdout in stdouts {
        // Output visibility formats are correctly specified.
        let mut iter = stdout.split("\n\n").filter(|s| s.contains("--outputs "));
        let outputs_line = iter.next();
        assert!(
            outputs_line.is_some(),
            "No lines containing '--outputs ' were found in di-calibrate's help text"
        );
        assert!(
            iter.next().is_none(),
            "More than one '--outputs ' was found; this should not be possible"
        );
        let outputs_line = outputs_line.unwrap().split_ascii_whitespace().join(" ");
        assert!(
            outputs_line.contains("Supported formats: fits, bin"),
            "--outputs did not list expected solution outputs. The line is: {outputs_line}"
        );

        let mut iter = stdout
            .split("\n\n")
            .filter(|s| s.contains("--model-filenames "));
        let model_line = iter.next();
        assert!(
            model_line.is_some(),
            "No lines containing '--model-filenames ' were found in di-calibrate's help text"
        );
        assert!(
            iter.next().is_none(),
            "More than one '--model-filenames ' was found; this should not be possible"
        );
        let model_line = model_line.unwrap().split_ascii_whitespace().join(" ");
        assert!(
            model_line.contains("Supported formats: uvfits, ms"),
            "--model-filenames did not list expected vis outputs. The line is: {model_line}"
        );
    }
}
