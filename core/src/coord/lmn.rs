// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Handle interferometric direction-cosine coordinates.
//!
//! This coordinate system is discussed at length in Interferometry and
//! Synthesis in Radio Astronomy, Third Edition, Section 4: Geometrical
//! Relationships, Polarimetry, and the Measurement Equation.

/// (l,m,n) direction-cosine coordinates. All units are in radians.
///
/// This coordinate system is discussed at length in Interferometry and
/// Synthesis in Radio Astronomy, Third Edition, Section 4: Geometrical
/// Relationships, Polarimetry, and the Measurement Equation.
#[derive(Debug)]
#[allow(clippy::upper_case_acronyms)]
pub struct LMN {
    /// l-coordinate \[radians\]
    pub l: f64,
    /// m-coordinate \[radians\]
    pub m: f64,
    /// n-coordinate \[radians\]
    pub n: f64,
}
