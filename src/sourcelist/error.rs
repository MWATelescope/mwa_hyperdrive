// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::error::Error;
use std::fmt;

#[derive(Debug)]
pub enum ErrorKind {
    /// Used for impractical situations, like trying to estimate the flux
    /// density of a source without any components.
    Insane(String),
    /// Parse error from the nom crate.
    ParseError(String),
}

impl Error for ErrorKind {
    fn description(&self) -> &str {
        match *self {
            ErrorKind::Insane(ref err) => err,
            ErrorKind::ParseError(ref err) => err,
        }
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ErrorKind::Insane(ref err) => err.fmt(f),
            ErrorKind::ParseError(ref err) => err.fmt(f),
        }
    }
}
