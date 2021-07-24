// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Core code to describe coordinate transformations, Jones matrices, abstract
//! beam code...

pub mod constants;
pub mod coord;
pub mod jones;
pub mod sexagesimal;

// Re-exports.
pub use coord::*;
pub use jones::Jones;
pub use sexagesimal::*;

pub use num::complex::{Complex32 as c32, Complex64 as c64};

#[cfg(feature = "mwalib")]
pub use mwalib;

#[cfg(feature = "beam")]
pub mod beam;
#[cfg(feature = "beam")]
pub use beam::Beam;

#[cfg(feature = "erfa")]
pub use erfa_sys;
