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
use log::debug;
use marlu::Jones;
use ndarray::prelude::*;
use strum_macros::{Display, EnumIter, EnumString};

use mwa_hyperdrive_common::{hifitime, log, marlu, ndarray};

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

    /// The MWA observation ID. Allowed to be optional as not all formats
    /// provide it.
    pub obsid: Option<u32>,

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
}

impl CalibrationSolutions {
    /// Read in calibration solutions from a file. The format of the file is
    /// determined by the file's extension (e.g. ".fits"). If the file is
    /// actually a directory, we attempt to read RTS DI calibration solution
    /// files from the directory.
    pub fn read_solutions_from_ext<P: AsRef<Path>, P2: AsRef<Path>>(
        file: P,
        metafits: Option<P2>,
    ) -> Result<CalibrationSolutions, ReadSolutionsError> {
        fn inner(
            file: &Path,
            metafits: Option<&Path>,
        ) -> Result<CalibrationSolutions, ReadSolutionsError> {
            if file.is_dir() {
                debug!(
                    "Got a directory '{}', looking for RTS solutions...",
                    file.display()
                );
                let metafits = metafits.ok_or(ReadSolutionsError::RtsMetafitsRequired)?;
                rts::read(file, metafits).map_err(ReadSolutionsError::from)
            } else {
                match file.extension().and_then(|s| s.to_str()) {
                    Some("fits") => hyperdrive::read(file),
                    Some("bin") => ao::read(file),
                    s => {
                        let ext = s.unwrap_or("<no extension>").to_string();
                        Err(ReadSolutionsError::UnsupportedExt { ext })
                    }
                }
            }
        }
        inner(file.as_ref(), metafits.as_ref().map(|f| f.as_ref()))
    }

    pub fn write_solutions_from_ext<P: AsRef<Path>, P2: AsRef<Path>, P3: AsRef<Path>>(
        &self,
        file: P,
        metafits: Option<P2>,
        fee_beam_file: Option<P3>,
    ) -> Result<(), WriteSolutionsError> {
        fn inner(
            sols: &CalibrationSolutions,
            file: &Path,
            metafits: Option<&Path>,
            fee_beam_file: Option<&Path>,
        ) -> Result<(), WriteSolutionsError> {
            if file.is_dir() {
                debug!("{file:?} is a directiory - looking for RTS solution files");
                let metafits = metafits.ok_or(WriteSolutionsError::RtsMetafitsRequired)?;
                // rts::write(sols, file, metafits, fee_beam_file, None)?;
                rts::write(sols, file, metafits, None)?;
            } else {
                let ext = file.extension().and_then(|e| e.to_str());
                match ext.and_then(|s| CalSolutionType::from_str(s).ok()) {
                    Some(CalSolutionType::Fits) => hyperdrive::write(sols, file),
                    Some(CalSolutionType::Bin) => ao::write(sols, file),
                    None => Err(WriteSolutionsError::UnsupportedExt {
                        ext: ext.unwrap_or("<no extension>").to_string(),
                    }),
                }?;
            }

            Ok(())
        }
        inner(
            self,
            file.as_ref(),
            metafits.as_ref().map(|f| f.as_ref()),
            fee_beam_file.as_ref().map(|f| f.as_ref()),
        )
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
    ) -> Result<Vec<String>, Box<dyn std::error::Error>> {
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
