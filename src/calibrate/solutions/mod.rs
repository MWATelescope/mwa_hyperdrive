// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to read and write calibration solutions.
//!
//! See for more info:
//! https://github.com/MWATelescope/mwa_hyperdrive/wiki/Calibration-solutions

mod ao;
mod error;
mod hyperdrive;
#[cfg(feature = "plotting")]
mod plotting;
mod rts;

pub(crate) use error::*;
pub(crate) use rts::RtsReadSolsError;

use std::collections::HashSet;
use std::path::Path;
use std::str::FromStr;

use hifitime::Epoch;
use marlu::{time::epoch_as_gps_seconds, Jones};
use ndarray::prelude::*;
use strum_macros::{Display, EnumIter, EnumString};

use mwa_hyperdrive_common::{hifitime, marlu, ndarray};

#[derive(Debug, Display, EnumIter, EnumString)]
pub(crate) enum CalSolutionType {
    /// hyperdrive's preferred format.
    #[strum(serialize = "fits")]
    Fits,

    /// The "André Offringa" format used by mwa-reduce.
    #[strum(serialize = "bin")]
    Bin,
}

pub struct CalibrationSolutions {
    pub di_jones: Array3<Jones<f64>>,
    pub num_timeblocks: usize,
    pub total_num_tiles: usize,
    pub total_num_fine_freq_chans: usize,

    /// The start timestamps of each timeblock used to produce these calibration
    /// solutions.
    pub start_timestamps: Vec<Epoch>,
    pub obsid: Option<u32>,

    /// The number of seconds per timeblock, or, the time resolution of the
    /// calibration solutions. Only really useful if there are multiple
    /// timeblocks.
    pub time_res: Option<f64>,
}

impl CalibrationSolutions {
    /// Read in calibration solutions from a file. The format of the file is
    /// determined by the file's extension (e.g. ".fits"). Mostly useful for
    /// testing.
    pub fn read_solutions_from_ext<T: AsRef<Path>>(file: T) -> Result<Self, ReadSolutionsError> {
        match file.as_ref().extension().and_then(|s| s.to_str()) {
            Some("fits") => hyperdrive::read(file),
            Some("bin") => ao::read(file),
            s => {
                let ext = s.unwrap_or("<no extension>").to_string();
                Err(ReadSolutionsError::UnsupportedExt { ext })
            }
        }
    }

    pub(super) fn write_solutions_from_ext<T: AsRef<Path>>(
        &self,
        file: T,
        tile_flags: &HashSet<usize>,
        unflagged_fine_chans: &HashSet<usize>,
    ) -> Result<(), WriteSolutionsError> {
        let ext = file.as_ref().extension().and_then(|e| e.to_str());
        match ext.and_then(|s| CalSolutionType::from_str(s).ok()) {
            Some(CalSolutionType::Fits) => {
                self.write_hyperdrive_fits(file, tile_flags, unflagged_fine_chans)
            }
            Some(CalSolutionType::Bin) => {
                self.write_andre_binary(file, tile_flags, unflagged_fine_chans)
            }
            None => Err(WriteSolutionsError::UnsupportedExt {
                ext: ext.unwrap_or("<no extension>").to_string(),
            }),
        }
    }

    pub(super) fn write_hyperdrive_fits<T: AsRef<Path>>(
        &self,
        file: T,
        tile_flags: &HashSet<usize>,
        unflagged_fine_chans: &HashSet<usize>,
    ) -> Result<(), WriteSolutionsError> {
        hyperdrive::write(self, file, tile_flags, unflagged_fine_chans)
    }

    pub(super) fn write_andre_binary<T: AsRef<Path>>(
        &self,
        file: T,
        tile_flags: &HashSet<usize>,
        unflagged_fine_chans: &HashSet<usize>,
    ) -> Result<(), WriteSolutionsError> {
        ao::write(self, file, tile_flags, unflagged_fine_chans)
    }

    #[cfg(feature = "plotting")]
    pub fn plot<T: AsRef<Path>, S: AsRef<str>>(
        &self,
        filename_base: T,
        plot_title: &str,
        ref_tile: Option<usize>,
        tile_names: Option<&[S]>,
    ) -> Result<Vec<String>, ()> {
        plotting::plot_sols(self, filename_base, plot_title, ref_tile, tile_names)
    }
}
