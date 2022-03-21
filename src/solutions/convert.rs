// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to convert calibration solutions.

use std::path::PathBuf;

use clap::Parser;
use log::info;

use super::CalibrationSolutions;
use crate::HyperdriveError;
use mwa_hyperdrive_common::{clap, log};

#[derive(Parser, Debug, Default)]
pub struct SolutionsConvertArgs {
    /// The path to the input file. If this is a directory instead, then we
    /// attempt to read RTS calibration files in the directory.
    #[clap(name = "INPUT_SOLUTIONS_FILE", parse(from_os_str))]
    input: PathBuf,

    /// The path to the output file. If this is a directory instead, then we
    /// attempt to write RTS calibration files to the directory.
    #[clap(name = "OUTPUT_SOLUTIONS_FILE", parse(from_os_str))]
    output: PathBuf,

    /// The metafits file associated with the solutions. This may be required.
    #[clap(short, long, parse(from_str))]
    metafits: Option<PathBuf>,
}

impl SolutionsConvertArgs {
    pub fn run(self) -> Result<(), HyperdriveError> {
        let sols =
            CalibrationSolutions::read_solutions_from_ext(&self.input, self.metafits.as_ref())?;
        sols.write_solutions_from_ext(&self.output, self.metafits.as_deref())?;

        info!(
            "Converted {} to {}",
            self.input.display(),
            self.output.display()
        );

        Ok(())
    }
}
