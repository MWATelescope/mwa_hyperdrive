// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use super::*;

use thiserror::Error;

use mwa_hyperdrive_common::thiserror;

#[derive(Debug, Error)]
pub enum UnitParseError {
    #[error(
        "Successfully parsed a time unit ('{unit}'), but could not parse the numerical component of '{0}'"
    )]
    GotTimeUnitButCantParse { input: String, unit: &'static str },

    #[error(
        "Successfully parsed a frequency unit, but could not parse the numerical component of '{0}'"
    )]
    GotFreqUnitButCantParse(String),

    #[error("Could not parse '{input}' as a number or quantity of {unit_type}")]
    Unknown {
        input: String,
        unit_type: &'static str,
    },
}
