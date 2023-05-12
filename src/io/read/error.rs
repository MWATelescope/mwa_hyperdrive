// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#[derive(thiserror::Error, Debug)]
pub enum VisReadError {
    #[error("Output {array_type} array did not have expected {expected_len} elements on axis {axis_num}")]
    BadArraySize {
        array_type: &'static str,
        expected_len: usize,
        axis_num: usize,
    },

    #[error(transparent)]
    Raw(#[from] super::raw::RawReadError),

    #[error(transparent)]
    MS(#[from] super::ms::MsReadError),

    #[error(transparent)]
    Uvfits(#[from] super::uvfits::UvfitsReadError),
}
