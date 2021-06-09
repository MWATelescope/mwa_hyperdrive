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

use dashmap::DashMap;
use mwa_hyperbeam::fee::*;
use num::Complex;

use super::Jones;

/// A special hash used to determine what's in our Jones cache.
#[derive(Hash, Debug, Clone, Eq, PartialEq)]
struct JonesHash(u64);

impl JonesHash {
    /// Create a new `JonesHash`.
    ///
    /// It hashes the input parameters for a unique hash. If these parameters
    /// are re-used, the same hash will be generated, and we can use the cache
    /// that these `JonesHash`es guard.
    fn new(
        az_rad: f64,
        za_rad: f64,
        freq_hz: u32,
        delays: &[u32],
        amps: &[f64],
        norm_to_zenith: bool,
    ) -> Self {
        let mut hasher = DefaultHasher::new();
        // We can't hash f64 values, so convert them to ints. Multiply by a big
        // number to get away from integer rounding.
        let to_int = |x: f64| (x * 1e8) as u32;
        to_int(az_rad).hash(&mut hasher);
        to_int(za_rad).hash(&mut hasher);
        freq_hz.hash(&mut hasher);
        delays.hash(&mut hasher);
        for &a in amps {
            to_int(a).hash(&mut hasher);
        }
        norm_to_zenith.hash(&mut hasher);
        Self(hasher.finish())
    }
}

/// A cache for Jones matrices.
///
/// Given parameters that are needed for the beam code, `JonesCache` hashes them
/// to see if they've been used before. If so, we can avoid re-calculating the
/// same Jones matrix.
#[derive(Default)]
pub struct JonesCache(DashMap<JonesHash, Jones<f64>>);

impl JonesCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Give beam parameters and retrieve a Jones matrix. If the matrix wasn't
    /// already in the cache, it is populated.
    #[allow(clippy::too_many_arguments)]
    pub fn get_jones(
        &self,
        beam: &FEEBeam,
        az_rad: f64,
        za_rad: f64,
        freq_hz: u32,
        delays: &[u32],
        amps: &[f64],
        norm_to_zenith: bool,
    ) -> Result<Jones<f64>, FEEBeamError> {
        // The FEE beam is defined only on coarse-band frequencies. For this
        // reason, rather than making a unique has for every single different
        // frequency, round specified frequency (`freq_hz`) to the nearest
        // beam frequency and use that for the hash.
        let beam_freq = beam.find_closest_freq(freq_hz);

        // Are the input settings already cached? Hash them to check.
        let hash = JonesHash::new(az_rad, za_rad, beam_freq, delays, amps, norm_to_zenith);

        // If the cache for this hash exists, we can return a copy of the Jones
        // matrix.
        if self.0.contains_key(&hash) {
            // TODO: Can we avoid clone here?
            return Ok(self.0.get(&hash).unwrap().clone());
        }

        // If we hit this part of the code, the relevant Jones matrix was not in
        // the cache.
        let jones = beam.calc_jones(az_rad, za_rad, beam_freq, delays, amps, norm_to_zenith)?;
        let jones = Jones::from([
            Complex::new(jones[0].re, jones[0].im),
            Complex::new(jones[1].re, jones[1].im),
            Complex::new(jones[2].re, jones[2].im),
            Complex::new(jones[3].re, jones[3].im),
        ]);
        self.0.insert(hash.clone(), jones);
        Ok(self.0.get(&hash).unwrap().clone())
    }

    /// Get the size of the cache.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Is the cache empty?
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Clear the cache.
    pub fn clear(&self) {
        self.0.clear()
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::*;

    use super::*;
    // Serial tests needed when working with beam code, because the HDF5 C
    // library is not reentrant.
    use serial_test::serial;

    #[test]
    fn hash_same() {
        let az = FRAC_PI_4;
        let za = FRAC_PI_4 - 0.1;
        let freq_hz = 167000000;
        let delays = [0; 16];
        let amps = [1.0; 16];
        let norm_to_zenith = true;

        let hash1 = JonesHash::new(az, za, freq_hz, &delays, &amps, norm_to_zenith);
        let hash2 = JonesHash::new(az, za, freq_hz, &delays, &amps, norm_to_zenith);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn hash_different() {
        let az = FRAC_PI_4;
        let za = FRAC_PI_4 - 0.1;
        let freq_hz = 167000000;
        let delays = [0; 16];
        let amps = [1.0; 16];
        let norm_to_zenith = true;

        let hash1 = JonesHash::new(az, za, freq_hz, &delays, &amps, norm_to_zenith);
        let hash2 = JonesHash::new(az, za, freq_hz + 1, &delays, &amps, norm_to_zenith);
        assert_ne!(hash1, hash2);
    }

    #[test]
    #[serial]
    fn cache_same() {
        let beam_res = FEEBeam::new_from_env();
        assert!(beam_res.is_ok());
        let beam = beam_res.unwrap();
        let az = FRAC_PI_4;
        let za = FRAC_PI_4 - 0.1;
        let freq_hz = 167000000;
        let delays = [0; 16];
        let amps = [1.0; 16];
        let norm_to_zenith = true;

        let cache = JonesCache::new();
        let jones1 = cache.get_jones(&beam, az, za, freq_hz, &delays, &amps, norm_to_zenith);
        assert!(jones1.is_ok(), "{:?}", jones1.unwrap_err());
        let jones2 = cache.get_jones(&beam, az, za, freq_hz, &delays, &amps, norm_to_zenith);
        assert!(jones2.is_ok(), "{:?}", jones2.unwrap_err());
        assert_eq!(jones1.unwrap(), jones2.unwrap());

        assert_eq!(cache.0.len(), 1);
    }

    #[test]
    #[serial]
    // The same as above, but the frequency is slightly different. Despite the
    // difference, the result is the same, because the FEE beam's frequencies
    // are much coarser.
    fn cache_same_2() {
        let beam_res = FEEBeam::new_from_env();
        assert!(beam_res.is_ok());
        let beam = beam_res.unwrap();
        let az = FRAC_PI_4;
        let za = FRAC_PI_4 - 0.1;
        let freq_hz = 167000000;
        let delays = [0; 16];
        let amps = [1.0; 16];
        let norm_to_zenith = true;

        let cache = JonesCache::new();
        let jones1 = cache.get_jones(&beam, az, za, freq_hz, &delays, &amps, norm_to_zenith);
        assert!(jones1.is_ok(), "{:?}", jones1.unwrap_err());
        let jones2 = cache.get_jones(&beam, az, za, freq_hz + 1, &delays, &amps, norm_to_zenith);
        assert!(jones2.is_ok(), "{:?}", jones2.unwrap_err());
        assert_eq!(jones1.unwrap(), jones2.unwrap());

        assert_eq!(cache.0.len(), 1);
    }

    #[test]
    #[serial]
    fn cache_different() {
        let beam_res = FEEBeam::new_from_env();
        assert!(beam_res.is_ok());
        let beam = beam_res.unwrap();
        let az = FRAC_PI_4;
        let za = FRAC_PI_4 - 0.1;
        let freq_hz = 167000000;
        let delays = [0; 16];
        let amps = [1.0; 16];
        let norm_to_zenith = true;

        let cache = JonesCache::new();
        let jones1 = cache.get_jones(&beam, az, za, freq_hz, &delays, &amps, norm_to_zenith);
        assert!(jones1.is_ok(), "{:?}", jones1.unwrap_err());
        let jones2 = cache.get_jones(
            &beam,
            az,
            za,
            freq_hz + 1280000,
            &delays,
            &amps,
            norm_to_zenith,
        );
        assert!(jones2.is_ok(), "{:?}", jones2.unwrap_err());
        assert_ne!(jones1.unwrap(), jones2.unwrap());

        assert_eq!(cache.0.len(), 2);
    }
}
