// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to cache Jones matrices.
//!
//! Because it's very likely that the Jones matrices derived from different
//! tiles are exactly the same (especially if we ignore the positions of the
//! tiles), the code here exists to cache Jones matrices given the parameters
//! that would be given to beam code.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use marlu::AzEl;

use mwa_hyperdrive_common::marlu;

/// A special hash used to determine what's in our Jones cache.
#[derive(Hash, Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct JonesHash(u64);

impl JonesHash {
    /// Create a new [JonesHash].
    ///
    /// It hashes the input parameters for a unique hash. If these parameters
    /// are re-used, the same hash will be generated, and we can use the cache
    /// that these `JonesHash`es guard.
    pub(crate) fn new(azel: AzEl, freq_hz: f64, delays: &[u32], amps: &[f64]) -> JonesHash {
        let mut hasher = DefaultHasher::new();
        // We can't hash f64 values, so use their bits.
        azel.az.to_bits().hash(&mut hasher);
        azel.el.to_bits().hash(&mut hasher);
        freq_hz.to_bits().hash(&mut hasher);
        delays.hash(&mut hasher);
        for a in amps {
            a.to_bits().hash(&mut hasher);
        }
        Self(hasher.finish())
    }
}
