// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use thiserror::Error;

use mwa_hyperdrive_common::thiserror;

#[derive(Error, Debug)]
pub(crate) enum AverageFactorError {
    #[error("The user input was 0; this is not permitted")]
    Zero,

    #[error("The user input has no units and isn't an integer; this is not permitted")]
    NotInteger,

    #[error("The user input isn't an integer multiple of the resolution: {out} vs {inp}")]
    NotIntegerMultiple { out: f64, inp: f64 },

    #[error(transparent)]
    Parse(#[from] crate::unit_parsing::UnitParseError),
}
