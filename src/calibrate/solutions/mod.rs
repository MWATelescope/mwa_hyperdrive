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
#[cfg(test)]
mod tests;

pub(crate) use error::*;

use std::path::Path;
use std::str::FromStr;

use hifitime::Epoch;
use marlu::Jones;
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
    /// The direction-independent calibration solutions. This has dimensions of
    /// (num_timeblocks, total_num_tiles, total_num_chanblocks). Note that this
    /// potentially includes flagged data; other struct members help to
    /// determine what is flagged. These Jones matrices, when applied to data
    /// Jones matrices, should approximate the model Jones matrices used in
    /// calibration.
    pub di_jones: Array3<Jones<f64>>,

    /// Which tiles are flagged? Zero indexed.
    pub flagged_tiles: Vec<usize>,

    /// Which chanblocks are flagged? Zero indexed.
    pub flagged_chanblocks: Vec<u16>,

    /// The start timestamps (centroids) of each timeblock used to produce these
    /// calibration solutions. This is allowed to be empty; in this case, no
    /// timestamp information is provided. It may also have a different length
    /// to the first dimension of `di_jones` due to inadequate information.
    pub start_timestamps: Vec<Epoch>,

    /// The end timestamps (centroids) of each timeblock used to produce these
    /// calibration solutions. This is allowed to be empty; in this case, no
    /// timestamp information is provided. It may also have a different length
    /// to the first dimension of `di_jones` due to inadequate information.
    pub end_timestamps: Vec<Epoch>,

    /// The average timestamps of each timeblock used to produce these
    /// calibration solutions. This is allowed to be empty; in this case, no
    /// timestamp information is provided. It may also have a different length
    /// to the first dimension of `di_jones` due to inadequate information.
    pub average_timestamps: Vec<Epoch>,

    /// The MWA observation ID. Allowed to be optional as not all formats
    /// provide it.
    pub obsid: Option<u32>,
}

impl CalibrationSolutions {
    /// Read in calibration solutions from a file. The format of the file is
    /// determined by the file's extension (e.g. ".fits"). If the file is
    /// actually a directory, we attempt to read RTS DI calibration solution
    /// files from the directory.
    pub fn read_solutions_from_ext<T: AsRef<Path>, T2: AsRef<Path>>(
        file: T,
        metafits: Option<T2>,
    ) -> Result<Self, ReadSolutionsError> {
        let f = file.as_ref();
        if f.is_dir() {
            let metafits = metafits.ok_or(ReadSolutionsError::RtsMetafitsRequired)?;
            rts::read(f, metafits).map_err(ReadSolutionsError::from)
        } else {
            match f.extension().and_then(|s| s.to_str()) {
                Some("fits") => hyperdrive::read(file),
                Some("bin") => ao::read(file),
                s => {
                    let ext = s.unwrap_or("<no extension>").to_string();
                    Err(ReadSolutionsError::UnsupportedExt { ext })
                }
            }
        }
    }

    pub fn write_solutions_from_ext<T: AsRef<Path>>(
        &self,
        file: T,
    ) -> Result<(), WriteSolutionsError> {
        let f = file.as_ref();
        if f.is_dir() {
            //rts::write
            todo!()
        } else {
            let ext = file.as_ref().extension().and_then(|e| e.to_str());
            match ext.and_then(|s| CalSolutionType::from_str(s).ok()) {
                Some(CalSolutionType::Fits) => hyperdrive::write(self, file),
                Some(CalSolutionType::Bin) => ao::write(self, file),
                None => Err(WriteSolutionsError::UnsupportedExt {
                    ext: ext.unwrap_or("<no extension>").to_string(),
                }),
            }
        }
    }

    #[cfg(feature = "plotting")]
    #[allow(clippy::too_many_arguments)]
    pub fn plot<T: AsRef<Path>, S: AsRef<str>>(
        &self,
        filename_base: T,
        plot_title: &str,
        ref_tile: Option<usize>,
        no_ref_tile: bool,
        tile_names: Option<&[S]>,
        ignore_cross_pols: bool,
        min_amp: Option<f64>,
        max_amp: Option<f64>,
    ) -> Result<Vec<String>, ()> {
        plotting::plot_sols(
            self,
            filename_base,
            plot_title,
            ref_tile,
            no_ref_tile,
            tile_names,
            ignore_cross_pols,
            min_amp,
            max_amp,
        )
    }
}
