// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to read and write RTS calibration solutions.
//!
//! See for more info:
//! https://github.com/MWATelescope/mwa_hyperdrive/wiki/Calibration-solutions

// The RTS calls DI_JonesMatrices files "alignment files". The first two lines
// of these files correspond to "alignment flux density" (or "flux scale") and
// "post-alignment matrix". The reference line is structured the same as all
// those that follow it; 4 complex number pairs. All lines after the first two
// are "pre-alignment matrices". When reading in this data, the post-alignment
// matrix is inverted and applied to each pre-alignment matrix (as in, A*B where
// A is pre- and B- is inverse post).

mod read_files;

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use crate::glob::get_all_matches_from_glob;
use byteorder::{ByteOrder, LittleEndian, ReadBytesExt, WriteBytesExt};
use hifitime::{Duration, Epoch};
use marlu::{
    time::{epoch_as_gps_seconds, gps_to_epoch},
    Jones,
};
use ndarray::prelude::*;
use thiserror::Error;

use super::CalibrationSolutions;
use mwa_hyperdrive_common::{hifitime, marlu, ndarray, thiserror, Complex};

pub(super) fn read<T: AsRef<Path>>(dir: T) -> Result<CalibrationSolutions, RtsReadSolsError> {
    // Search the current directory for DI_JonesMatrices_node???.dat and
    // BandpassCalibration_node???.dat files.
    let di_jm = get_all_matches_from_glob(&format!(
        "{}/DI_JonesMatrices_node???.dat",
        dir.as_ref().display()
    ))?;
    let bp_cal = get_all_matches_from_glob(&format!(
        "{}/BandpassCalibration_node???.dat",
        dir.as_ref().display()
    ))?;

    // There should be an equal number of files for each type. The number of
    // files corresponds to the number of coarse channels.
    let num_coarse_chans = di_jm.len();
    if bp_cal.len() != num_coarse_chans {
        return Err(RtsReadSolsError::UnequalFileCount {
            dir: dir.as_ref().display().to_string(),
            di_jm_count: di_jm.len(),
            bp_cal_count: bp_cal.len(),
        });
    }

    // Unpack the DI JM files.

    Ok(CalibrationSolutions {
        di_jones: todo!(),
        num_timeblocks: todo!(),
        total_num_tiles: todo!(),
        total_num_fine_freq_chans: todo!(),
        start_timestamps: todo!(),
        obsid: todo!(),
        time_res: todo!(),
    })
}

#[derive(Error, Debug)]
pub enum RtsReadSolsError {
    #[error("In directory {dir}, found {di_jm_count} DI_JonesMatrices_node???.dat files, but {bp_cal_count} BandpassCalibration_node???.dat files.\nThere must be an equal number of these files.")]
    UnequalFileCount {
        dir: String,
        di_jm_count: usize,
        bp_cal_count: usize,
    },

    #[error("{0}")]
    Glob(#[from] crate::glob::GlobError),
}
