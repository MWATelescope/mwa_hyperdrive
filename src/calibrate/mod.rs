// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to handle calibration.

pub mod args;
pub mod di;
mod error;
pub(crate) mod params;

use args::CalibrateUserArgs;
pub use error::CalibrateError;

use std::path::Path;

use log::{debug, info, trace};

use crate::solutions::{self, CalSolutionType, CalibrationSolutions};
use mwa_hyperdrive_common::log;

pub fn di_calibrate(
    cli_args: Box<CalibrateUserArgs>,
    args_file: Option<&Path>,
    dry_run: bool,
) -> Result<Option<CalibrationSolutions>, CalibrateError> {
    let args = if let Some(f) = args_file {
        trace!("Merging command-line arguments with the argument file");
        Box::new(cli_args.merge(&f)?)
    } else {
        cli_args
    };
    debug!("{:#?}", &args);
    trace!("Converting arguments into calibration parameters");
    let parameters = args.into_params()?;

    if dry_run {
        info!("Dry run -- exiting now.");
        return Ok(None);
    }

    let sols = di::di_calibrate(&parameters)?;

    // Write out the solutions.
    if parameters.output_solutions_filenames.len() == 1 {
        let (sol_type, file) = &parameters.output_solutions_filenames[0];
        match sol_type {
            CalSolutionType::Fits => solutions::hyperdrive::write(&sols, file)?,
            CalSolutionType::Bin => solutions::ao::write(&sols, file)?,
        }
        info!("Calibration solutions written to {}", file.display());
    } else {
        for (i, (sol_type, file)) in parameters
            .output_solutions_filenames
            .into_iter()
            .enumerate()
        {
            match sol_type {
                CalSolutionType::Fits => solutions::hyperdrive::write(&sols, &file)?,
                CalSolutionType::Bin => solutions::ao::write(&sols, &file)?,
            }
            if i == 0 {
                info!("Calibration solutions written to:");
            }
            info!("  {}", file.display());
        }
    }

    Ok(Some(sols))
}
