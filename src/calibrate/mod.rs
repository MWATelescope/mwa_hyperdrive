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
    fn inner(
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
        Ok(Some(sols))
    }
    inner(cli_args, args_file.as_ref().map(|f| f.as_ref()), dry_run)
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

    /// The timestamp (centroid) representing the start of the timeblock.
    start: Epoch,

    /// The timestamp (centroid) representing the end of the timeblock.
    end: Epoch,

    /// The timestamp best representing the entire timeblock.
    ///
    /// e.g. If a timeblock comprised GPS times 10 and 11, the average would be
    /// GPS time 10.5.
    average: Epoch,
}

/// A collection of fine-frequency channels to average together *before*
/// calibration.
#[derive(Debug, Clone)]
pub(crate) struct Chanblock {
    /// The chanblock index, regardless of flagging. e.g. If the first two
    /// calibration chanblocks are flagged, then the first unflagged chanblock
    /// has a chanblock_index of 2 but an unflagged_index of 0.
    pub(crate) chanblock_index: u16,

    /// The index into an *unflagged* array of visibilities. Regardless of the
    /// first unflagged chanblock's index, its unflagged index is 0.
    pub(crate) unflagged_index: u16,

    /// The centroid frequency for this chanblock \[Hz\].
    pub(crate) freq: f64,
}

/// A spectral windows, a.k.a. a contiguous-band of fine-frequency channels
/// (possibly made up of multiple contiguous coarse channels). Multiple `Fence`s
/// allow a "picket fence" observation to be represented. Calibration is run on
/// each independent `Fence`.
#[derive(Debug)]
pub(crate) struct Fence {
    /// The unflagged calibration [Chanblock]s in this [Fence].
    pub(crate) chanblocks: Vec<Chanblock>,

    /// The indices of the flagged chanblocks.
    ///
    /// The type is `u16` to keep the memory usage down; these probably need to
    /// be promoted to `usize` when being used.
    pub(crate) flagged_chanblock_indices: Vec<u16>,

    /// The first chanblock's centroid frequency (may be flagged) \[Hz\].
    pub(crate) first_freq: f64,

    /// The frequency gap between consecutive chanblocks \[Hz\]. If this isn't
    /// defined, it's because there's only one chanblock.
    pub(crate) freq_res: Option<f64>,
}

impl Fence {
    fn get_total_num_chanblocks(&self) -> usize {
        self.chanblocks.len() + self.flagged_chanblock_indices.len()
    }

    /// Get the centre frequency of this [Fence], considering all chanblocks
    /// (flagged and unflagged) \[Hz\].
    fn get_centre_freq(&self) -> f64 {
        if let Some(freq_res) = self.freq_res {
            let n = self.get_total_num_chanblocks();
            self.first_freq + (n / 2) as f64 * freq_res
        } else {
            self.first_freq
        }
    }

    fn get_freqs(&self) -> Vec<f64> {
        if let Some(freq_res) = self.freq_res {
            (0..self.get_total_num_chanblocks())
                .into_iter()
                .map(|i_chanblock| self.first_freq + i_chanblock as f64 * freq_res)
                .collect()
        } else {
            vec![self.first_freq]
        }
    }
}
