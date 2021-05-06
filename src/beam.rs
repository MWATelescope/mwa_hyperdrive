// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Code to abstract beam calculations.

`Beam` is a trait detailing how to perform various beam-related tasks. By
making this trait, we can neatly abstract over multiple beam codes, including
a simple `NoBeam` type (which just returns identity matrices).
 */

use ndarray::prelude::*;
use thiserror::Error;

use mwa_hyperdrive_core::{mwa_hyperbeam, Jones};

/// A trait abstracting beam code functions.
pub(crate) trait Beam: Sync + Send {
    /// Calculate the Jones matrices for a direction given a pointing.
    ///
    /// `delays` and `amps` *must* have 16 elements (each corresponds to an MWA
    /// dipole in a tile, in the M&C order; see
    /// https://wiki.mwatelescope.org/pages/viewpage.action?pageId=48005139).
    fn calc_jones(
        &self,
        az_rad: f64,
        za_rad: f64,
        freq_hz: u32,
        delays: &[u32],
        amps: &[f64],
        norm_to_zenith: bool,
    ) -> Result<Jones<f64>, BeamError>;

    /// Calculate the Jones matrices for many directions given a pointing.
    ///
    /// `delays` and `amps` *must* have 16 elements (each corresponds to an MWA
    /// dipole in a tile, in the M&C order; see
    /// https://wiki.mwatelescope.org/pages/viewpage.action?pageId=48005139).
    fn calc_jones_array(
        &self,
        az_rad: &[f64],
        za_rad: &[f64],
        freq_hz: u32,
        delays: &[u32],
        amps: &[f64],
        norm_to_zenith: bool,
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
    #[error("hyperbeam error: {0}")]
    Hyperbeam(#[from] mwa_hyperbeam::fee::FEEBeamError),

    #[error("hyperbeam init error: {0}")]
    HyperbeamInit(#[from] mwa_hyperbeam::fee::InitFEEBeamError),
}

/// A hyperdrive-specific wrapper of the `FEEBeam` struct in hyperbeam.
pub(crate) struct FEEBeam(mwa_hyperbeam::fee::FEEBeam);

impl FEEBeam {
    pub(crate) fn new<T: AsRef<std::path::Path>>(
        file: T,
    ) -> Result<Self, mwa_hyperbeam::fee::InitFEEBeamError> {
        // Wrap the `FEEBeam` out of hyperbeam with our own `FEEBeam`.
        mwa_hyperbeam::fee::FEEBeam::new(file).map(FEEBeam)
    }

    pub(crate) fn new_from_env() -> Result<Self, mwa_hyperbeam::fee::InitFEEBeamError> {
        mwa_hyperbeam::fee::FEEBeam::new_from_env().map(FEEBeam)
    }
}

impl Beam for FEEBeam {
    fn calc_jones(
        &self,
        az_rad: f64,
        za_rad: f64,
        freq_hz: u32,
        delays: &[u32],
        amps: &[f64],
        norm_to_zenith: bool,
    ) -> Result<Jones<f64>, BeamError> {
        let j = self
            .0
            .calc_jones(az_rad, za_rad, freq_hz, delays, amps, norm_to_zenith)?;
        // hyperbeam returns a 1D ndarray of [Complex64; 4]. The version of
        // ndarray used by hyperbeam may be different than that of hyperdrive,
        // as well as the version of num or num-complex for Complex64. Turn each
        // sub-array into our special `Jones` wrapper. Using transmute is ugly,
        // but it neatly converts potentially mis-matched crate versions without
        // copying anything, as the types are exactly the same.
        let j = unsafe { std::mem::transmute::<[_; 4], Jones<f64>>(j) };
        Ok(j)
    }

    fn calc_jones_array(
        &self,
        az_rad: &[f64],
        za_rad: &[f64],
        freq_hz: u32,
        delays: &[u32],
        amps: &[f64],
        norm_to_zenith: bool,
    ) -> Result<Array1<Jones<f64>>, BeamError> {
        let j = self
            .0
            .calc_jones_array(az_rad, za_rad, freq_hz, delays, amps, norm_to_zenith)?;

        // hyperbeam returns a 1D ndarray of [Complex64; 4]. The version of
        // ndarray used by hyperbeam may be different than that of hyperdrive,
        // as well as the version of num or num-complex for Complex64. Turn each
        // sub-array into our special `Jones` wrapper. Using transmute is ugly,
        // but it neatly converts potentially mis-matched crate versions without
        // copying anything, as the types are exactly the same.
        debug_assert_eq!(j.shape()[1], 4);
        let j = unsafe { std::mem::transmute::<_, Array1<Jones<f64>>>(j) };
        Ok(j)
    }

    fn find_closest_freq(&self, desired_freq_hz: f64) -> f64 {
        self.0.find_closest_freq(desired_freq_hz as _) as _
    }

    fn empty_cache(&self) {
        self.0.empty_cache()
    }
}

// `NoBeam` is a beam implementation that only returns identity Jones matrices
// for all beam calculations.
pub(crate) struct NoBeam;

impl Beam for NoBeam {
    fn calc_jones(
        &self,
        _az_rad: f64,
        _za_rad: f64,
        _freq_hz: u32,
        _delays: &[u32],
        _amps: &[f64],
        _norm_to_zenith: bool,
    ) -> Result<Jones<f64>, BeamError> {
        Ok(Jones::identity())
    }

    fn calc_jones_array(
        &self,
        az_rad: &[f64],
        _za_rad: &[f64],
        _freq_hz: u32,
        _delays: &[u32],
        _amps: &[f64],
        _norm_to_zenith: bool,
    ) -> Result<Array1<Jones<f64>>, BeamError> {
        Ok(Array1::from_elem(az_rad.len(), Jones::identity()))
    }

    fn find_closest_freq(&self, desired_freq_hz: f64) -> f64 {
        desired_freq_hz
    }

    fn empty_cache(&self) {}
}
