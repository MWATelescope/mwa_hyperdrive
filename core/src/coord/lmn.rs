// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Handle interferometric direction-cosine coordinates.
 */

/// The (l,m,n) direction-cosine coordinates of a point. All units are in
/// radians.
///
/// This coordinate system is discussed at length in Interferometry and
/// Synthesis in Radio Astronomy, Third Edition, Section 4: Geometrical
/// Relationships, Polarimetry, and the Measurement Equation.
#[derive(Debug)]
pub struct LMN {
    /// l-coordinate [radians]
    pub l: f64,
    /// m-coordinate [radians]
    pub m: f64,
    /// n-coordinate [radians]
    pub n: f64,
}

impl LMN {
    /// Convert a vector of `LMN` structs to L, M, and N vectors. Useful for
    /// FFI.
    pub fn decompose(mut lmn: Vec<Self>) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let length = lmn.len();
        let mut l = Vec::with_capacity(length);
        let mut m = Vec::with_capacity(length);
        let mut n = Vec::with_capacity(length);
        for elem in lmn.drain(..) {
            l.push(elem.l as f32);
            m.push(elem.m as f32);
            n.push(elem.n as f32);
        }
        // Ensure that the capacity of the vectors matches their length.
        l.shrink_to_fit();
        m.shrink_to_fit();
        n.shrink_to_fit();
        (l, m, n)
    }
}
