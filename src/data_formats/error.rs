// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Errors from building a `InputData` trait instance.
 */

use thiserror::Error;

#[derive(Error, Debug)]
pub enum ReadInputDataError {
    // TODO: Tidy.
    #[error("{0}")]
    MS(#[from] super::ms::error::MSError),
}
