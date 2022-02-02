// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to handle calibration.

pub mod args;
mod di;
mod error;
pub(crate) mod params;
pub mod solutions;

use args::CalibrateUserArgs;
pub use error::CalibrateError;
use solutions::CalibrationSolutions;

use std::ops::Range;
use std::path::Path;

use hifitime::Epoch;
use log::{debug, info, trace};

use mwa_hyperdrive_common::{hifitime, log};

pub fn di_calibrate<P: AsRef<Path>>(
    cli_args: Box<CalibrateUserArgs>,
    args_file: Option<P>,
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
    Ok(Some(sols))
}

/// A collection of timesteps to average together *during* calibration.
#[derive(Debug)]
struct Timeblock {
    /// The timeblock index. e.g. If all observation timesteps are being used in
    /// a single calibration timeblock, then its index is 0.
    index: usize,

    /// The range of indices into an *unflagged* array of visibilities. e.g. If
    /// timeblock 0 represents timestep 10 and timeblock 1 represents timesteps
    /// 15 and 16 (and these are the only timesteps used for calibration), then
    /// timeblock 0's range is 0..1, whereas timeblock 1's range is 1..3.
    ///
    /// We can use a range because the timesteps belonging to a timeblock are
    /// always contiguous.
    range: Range<usize>,

    /// The timestamp representing the start of the timeblock.
    start: Epoch,

    /// The timestamp representing the end of the timeblock.
    end: Epoch,

    /// The timestamp best representing the entire timeblock.
    average: Epoch,
}

/// A collection of fine-frequency channels to average together *before*
/// calibration.
#[derive(Debug)]
struct Chanblock {
    /// The chanblock index, regardless of flagging. e.g. If all channels are
    /// unflagged, then the first chanblock used in calibration has an index of
    /// 0. But, if the first two channels are flagged, then the first chanblock
    /// used in calibration has an index of 2.
    chanblock_index: usize,

    /// The index into an *unflagged* array of visibilities. Regardless of the
    /// first unflagged chanblock's index, its unflagged index is 0.
    unflagged_index: usize,

    /// The frequency that this chanblock best represents \[Hz\].
    freq: f64,
}
