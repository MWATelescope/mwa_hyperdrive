// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Errors associated with interacting with CASA measurement sets.
 */

use std::path::PathBuf;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum NewMSError {
    #[error("Supplied file path {0} does not exist or is not readable!")]
    BadFile(PathBuf),

    #[error("The main table of the measurement set contains no rows!")]
    Empty,

    #[error("Couldn't work out the good start and end times of the measurement set; are all visibilities flagged?")]
    AllFlagged,

    #[error("{0}")]
    GeneralMS(#[from] MSError),

    // // TODO: Kill failure
    // #[error("{0}")]
    // Casacore(#[from] rubbl_casatables::CasacoreError),
    #[error("{0}")]
    Glob(#[from] crate::glob::GlobError),
}

#[derive(Error, Debug)]
pub enum MSError {
    #[error("Specified table name {0} does not exist")]
    TableDoesntExist(String),

    #[error("When reading in measurment set, ERFA function eraGd2gc failed to convert geodetic coordinates to geocentric. Is something wrong with your ANTENNA/POSITION column?")]
    Geodetic2Geocentric,

    #[error("Error when trying to interface with measurement set: {0}")]
    RubblError(String),
}
