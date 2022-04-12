// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::path::PathBuf;

use thiserror::Error;

use crate::help_texts::VIS_OUTPUT_EXTENSIONS;
use mwa_hyperdrive_common::{marlu, thiserror};

#[derive(Error, Debug)]
pub enum VisWriteError {
    #[error(
        "An invalid output format was specified ({0}). Supported:\n{}",
        *VIS_OUTPUT_EXTENSIONS,
    )]
    InvalidOutputFormat(String),

    #[error("Irregular timestamps; first timestamp (GPS) is {first}, but timestamp {bad} is not a multiple of the time resolution ({time_res}s) from the first")]
    IrregularTimestamps { first: f64, bad: f64, time_res: f64 },

    #[error(transparent)]
    FileWrite(#[from] FileWriteError),

    #[error(transparent)]
    UvfitsWrite(#[from] marlu::UvfitsWriteError),

    #[error(transparent)]
    MsWrite(#[from] marlu::io::MeasurementSetWriteError),

    #[error(transparent)]
    MarluIO(#[from] marlu::io::error::IOError),

    #[error(transparent)]
    IO(#[from] std::io::Error),
}

#[derive(Error, Debug)]
pub enum FileWriteError {
    #[error("Cannot write to the specified file '{file}'. Do you have write permissions set?")]
    FileNotWritable { file: String },

    #[error(
        "Couldn't create directory '{0}' for output files. Do you have write permissions set?"
    )]
    NewDirectory(PathBuf),

    #[error(transparent)]
    IO(#[from] std::io::Error),
}
