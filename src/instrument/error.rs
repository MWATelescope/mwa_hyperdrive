// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Errors associated with instrumental calibration.
*/

use thiserror::Error;

#[derive(Error, Debug)]
pub enum InstrumentError {
    /// There was a mismatch in expected number of dipoles.
    #[error("{source_file}:{source_line}\nExpected {expected} dipoles, got {got}")]
    DipoleCountMismatch {
        expected: u32,
        got: u32,
        source_file: String,
        source_line: u32,
    },
}
