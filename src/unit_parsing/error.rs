// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use super::*;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum UnitParseError {
    #[error(
        "Successfully parsed a time unit, but could not parse the numerical component of '{0}'"
    )]
    GotTimeUnitButCantParse(String),

    #[error(
        "Successfully parsed a frequency unit, but could not parse the numerical component of '{0}'"
    )]
    GotFreqUnitButCantParse(String),

    #[error("Could not parse '{0}' as a time")]
    Unknown(String),
}
