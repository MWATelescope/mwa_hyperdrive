// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use thiserror::Error;

use mwa_hyperdrive_common::thiserror;

#[derive(Debug, Error)]
pub(crate) enum UnitParseError {
    #[error(
        "Successfully parsed a {unit_type} unit ('{unit}'), but could not parse the numerical component of '{input}'"
    )]
    GotUnitButCantParse {
        input: String,
        unit_type: &'static str,
        unit: &'static str,
    },

    #[error("Parsed '{input}' as a number, but this quantity requires a {unit_type} unit")]
    UnitRequired {
        input: String,
        unit_type: &'static str,
    },

    #[error("Could not parse '{input}' as a number or quantity of {unit_type}")]
    Unknown {
        input: String,
        unit_type: &'static str,
    },
}
