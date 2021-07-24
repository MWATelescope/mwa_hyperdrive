// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Interferometric direction-cosine coordinates.
//!
//! This coordinate system is discussed at length in Interferometry and
//! Synthesis in Radio Astronomy, Third Edition, Section 3: Analysis of the
//! Interferometer Response.

/// (l,m,n) direction-cosine coordinates. There are no units (i.e.
/// dimensionless).
///
/// This coordinate system is discussed at length in Interferometry and
/// Synthesis in Radio Astronomy, Third Edition, Section 3: Analysis of the
/// Interferometer Response.
#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
pub struct LMN {
    /// l coordinate \[dimensionless\]
    pub l: f64,
    /// m coordinate \[dimensionless\]
    pub m: f64,
    /// n coordinate \[dimensionless\]
    pub n: f64,
}
