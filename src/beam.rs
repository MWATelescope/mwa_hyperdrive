// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to abstract beam calculations.
//!
//! `Beam` is a trait detailing how to perform various beam-related tasks. By
//! making this trait, we can neatly abstract over multiple beam codes,
//! including a simple `NoBeam` type (which just returns identity matrices).
//!
//! Note that (where applicable) `norm_to_zenith` is always true; the
//! implication being that a sky-model source's brightness is always assumed to
//! be correct when at zenith.

use ndarray::prelude::*;
use rayon::prelude::*;
use thiserror::Error;

use mwa_hyperdrive_core::{mwa_hyperbeam, AzEl, Jones};

use crate::calibrate::params::Delays;

/// A trait abstracting beam code functions.
pub(crate) trait Beam: Sync + Send {
    /// Calculate the Jones matrices for an [AzEl] direction. The pointing
    /// information is not needed because it was provided when `self` was
    /// created.
    ///
    /// `amps` *must* have 16 elements (each corresponds to an MWA dipole in a
    /// tile, in the M&C order; see
    /// https://wiki.mwatelescope.org/pages/viewpage.action?pageId=48005139).
    fn calc_jones(&self, azel: &AzEl, freq_hz: u32, amps: &[f64]) -> Result<Jones<f64>, BeamError>;

    /// Calculate the Jones matrices for many [AzEl] directions. The pointing
    /// information is not needed because it was provided when `self` was
    /// created.
    ///
    /// `amps` *must* have 16 elements (each corresponds to an MWA dipole in a
    /// tile, in the M&C order; see
    /// https://wiki.mwatelescope.org/pages/viewpage.action?pageId=48005139).
    fn calc_jones_array(
        &self,
        azels: &[AzEl],
        freq_hz: u32,
        amps: &[f64],
    ) -> Result<Array1<Jones<f64>>, BeamError>;

    /// Given a frequency in Hz, find the closest frequency that the beam code is
    /// defined for. This is important because, for example, the FEE beam code
    /// can only give beam responses at specific frequencies. On the other hand,
    /// the analytic beam can be used at any frequency.
    fn find_closest_freq(&self, desired_freq_hz: f64) -> f64;

    /// If the beam struct supports it, empty its associated coefficient cache.
    fn empty_cache(&self);
}

#[derive(Error, Debug)]
pub enum BeamError {
    #[error("Tried to create a beam object, but MWA dipole delay information isn't available!")]
    NoDelays,

    #[error("hyperbeam error: {0}")]
    Hyperbeam(#[from] mwa_hyperbeam::fee::FEEBeamError),

    #[error("hyperbeam init error: {0}")]
    HyperbeamInit(#[from] mwa_hyperbeam::fee::InitFEEBeamError),
}

/// A hyperdrive-specific wrapper of the `FEEBeam` struct in hyperbeam.
pub(crate) struct FEEBeam {
    beam: mwa_hyperbeam::fee::FEEBeam,
    delays: Vec<u32>,
}

impl FEEBeam {
    pub(crate) fn new<T: AsRef<std::path::Path>>(
        file: T,
        delays: Delays,
    ) -> Result<Self, BeamError> {
        // Wrap the `FEEBeam` out of hyperbeam with our own `FEEBeam`.
        let beam = mwa_hyperbeam::fee::FEEBeam::new(file)?;
        let delays = match delays {
            Delays::Available(d) => d,
            _ => return Err(BeamError::NoDelays),
        };

        Ok(FEEBeam { beam, delays })
    }

    pub(crate) fn new_from_env(delays: Delays) -> Result<Self, BeamError> {
        let beam = mwa_hyperbeam::fee::FEEBeam::new_from_env()?;
        let delays = match delays {
            Delays::Available(d) => d,
            _ => return Err(BeamError::NoDelays),
        };

        Ok(FEEBeam { beam, delays })
    }
}

impl Beam for FEEBeam {
    fn calc_jones(&self, azel: &AzEl, freq_hz: u32, amps: &[f64]) -> Result<Jones<f64>, BeamError> {
        let j = self
            .beam
            .calc_jones(azel.az, azel.za(), freq_hz, &self.delays, amps, true)?;
        Ok(Jones::from(j))
    }

    fn calc_jones_array(
        &self,
        azels: &[AzEl],
        freq_hz: u32,
        amps: &[f64],
    ) -> Result<Array1<Jones<f64>>, BeamError> {
        // Letting hyperbeam calculate Jones matrices in parallel is likely more
        // efficient than running `calc_jones` in parallel here. For that
        // reason, unpack azimuth and zenith angles before calling hyperbeam.
        let mut az = Vec::with_capacity(azels.len());
        let mut za = Vec::with_capacity(azels.len());
        azels
            .par_iter()
            .map(|azel| (azel.az, azel.za()))
            .unzip_into_vecs(&mut az, &mut za);
        let j = self
            .beam
            .calc_jones_array(&az, &za, freq_hz, &self.delays, amps, true)?;

        // hyperbeam returns a 1D ndarray of [Complex64; 4]. The version of
        // ndarray used by hyperbeam may be different than that of hyperdrive,
        // as well as the version of num or num-complex for Complex64. Turn each
        // sub-array into our special `Jones` wrapper. A benchmark analysis
        // suggests that the compiler doesn't actually do any copying; it knows
        // that this is a no-op.
        let j = j.mapv(|j| Jones::from(j));
        let j = Array1::from(j.to_vec());

        Ok(j)
    }

    fn find_closest_freq(&self, desired_freq_hz: f64) -> f64 {
        self.beam.find_closest_freq(desired_freq_hz as _) as _
    }

    fn empty_cache(&self) {
        self.beam.empty_cache()
    }
}

// `NoBeam` is a beam implementation that only returns identity Jones matrices
// for all beam calculations.
pub(crate) struct NoBeam;

impl Beam for NoBeam {
    fn calc_jones(
        &self,
        _azel: &AzEl,
        _freq_hz: u32,
        _amps: &[f64],
    ) -> Result<Jones<f64>, BeamError> {
        Ok(Jones::identity())
    }

    fn calc_jones_array(
        &self,
        azels: &[AzEl],
        _freq_hz: u32,
        _amps: &[f64],
    ) -> Result<Array1<Jones<f64>>, BeamError> {
        Ok(Array1::from_elem(azels.len(), Jones::identity()))
    }

    fn find_closest_freq(&self, desired_freq_hz: f64) -> f64 {
        desired_freq_hz
    }

    fn empty_cache(&self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use mwa_hyperbeam::fee::FEEBeam;
    use serial_test::serial;

    #[test]
    #[serial]
    fn fee_beam_values_are_sensible() {
        let delays = vec![0; 16];
        let freq = 150e6 as u32;
        let azels = [
            AzEl { az: 0.0, el: 0.0 },
            AzEl { az: 1.0, el: 0.1 },
            AzEl { az: -1.0, el: 0.2 },
        ];
        let (azs, zas): (Vec<_>, Vec<_>) = azels.iter().map(|azel| (azel.az, azel.za())).unzip();

        // Get the beam values right out of hyperbeam.
        let hyperbeam = FEEBeam::new_from_env().unwrap();
        let hyperbeam_values = hyperbeam
            .calc_jones_array(&azs, &zas, freq, &delays, &[1.0; 16], true)
            .unwrap();
        // Put the hyperbeam results into hyperdrive `Jones` objects.
        let hyperbeam_values = hyperbeam_values.mapv(|v| Jones::from(v));
        let hyperbeam_values = Array1::from(hyperbeam_values.to_vec());

        // Compare these with the hyperdrive `Beam` trait.
        let delays = Delays::Available(delays);
        let hyperdrive = super::FEEBeam::new_from_env(delays).unwrap();
        let hyperdrive_values: Array1<Jones<f64>> = hyperdrive
            .calc_jones_array(&azels, freq, &[1.0; 16])
            .unwrap();

        for (beam, drive) in hyperbeam_values
            .into_iter()
            .zip(hyperdrive_values.into_iter())
        {
            assert_abs_diff_eq!(beam[0], drive[0]);
            assert_abs_diff_eq!(beam[1], drive[1]);
            assert_abs_diff_eq!(beam[2], drive[2]);
            assert_abs_diff_eq!(beam[3], drive[3]);
        }
    }

    #[test]
    #[serial]
    fn no_beam_means_no_beam() {
        let azels = [
            AzEl { az: 0.0, el: 0.0 },
            AzEl { az: 1.0, el: 0.1 },
            AzEl { az: -1.0, el: 0.2 },
        ];
        let beam = NoBeam;
        let values: Array1<Jones<f64>> = beam
            .calc_jones_array(&azels, 150e6 as _, &[1.0; 16])
            .unwrap();

        for value in values.into_iter() {
            assert_abs_diff_eq!(value[0].re, 1.0);
            assert_abs_diff_eq!(value[0].im, 0.0);
            assert_abs_diff_eq!(value[1].re, 0.0);
            assert_abs_diff_eq!(value[1].im, 0.0);
            assert_abs_diff_eq!(value[2].re, 0.0);
            assert_abs_diff_eq!(value[2].im, 0.0);
            assert_abs_diff_eq!(value[3].re, 1.0);
            assert_abs_diff_eq!(value[3].im, 0.0);
        }
    }
}
