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

mod error;
mod fee;
pub use error::*;
pub use fee::*;

use std::path::Path;

use log::{info, trace};
use ndarray::prelude::*;

use crate::{AzEl, Jones};

/// A trait abstracting beam code functions.
pub trait Beam: Sync + Send {
    /// Calculate the Jones matrices for an [AzEl] direction. The pointing
    /// information is not needed because it was provided when `self` was
    /// created.
    ///
    /// `amps` *must* have 16 elements (each corresponds to an MWA dipole in a
    /// tile, in the M&C order; see
    /// <https://wiki.mwatelescope.org/pages/viewpage.action?pageId=48005139>.
    fn calc_jones(&self, azel: AzEl, freq_hz: u32, amps: &[f64]) -> Result<Jones<f64>, BeamError>;

    /// Calculate the Jones matrices for many [AzEl] directions. The pointing
    /// information is not needed because it was provided when `self` was
    /// created.
    ///
    /// `amps` *must* have 16 or 32 elements (each corresponds to an MWA dipole
    /// in a tile, in the M&C order; see
    /// <https://wiki.mwatelescope.org/pages/viewpage.action?pageId=48005139>.
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

/// An enum to track whether MWA dipole delays are provided and/or necessary.
pub enum Delays {
    /// Delays are available.
    Available(Vec<u32>),

    /// Delays have not been provided, but are necessary for calibration.
    None,

    /// Delays are not necessary, probably because no beam code is being used.
    NotNecessary,
}

/// A beam implementation that returns only identity Jones matrices for all beam
/// calculations.
pub struct NoBeam;

impl Beam for NoBeam {
    fn calc_jones(
        &self,
        _azel: AzEl,
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

pub fn create_fee_beam_object<T: AsRef<Path>>(
    delays: Delays,
    beam_file: Option<T>,
) -> Result<Box<dyn Beam>, BeamError> {
    trace!("Setting up FEE beam");
    let beam = if let Some(bf) = beam_file {
        // Set up the FEE beam struct from the specified beam file.
        Box::new(FEEBeam::new(&bf, delays)?)
    } else {
        // Set up the FEE beam struct from the MWA_BEAM_FILE environment
        // variable.
        Box::new(FEEBeam::new_from_env(delays)?)
    };
    info!("Using FEE beam with delays {:?}", beam.get_delays());
    Ok(beam)
}

pub fn create_beam_object<T: AsRef<Path>>(
    no_beam: bool,
    delays: Delays,
    beam_file: Option<T>,
) -> Result<Box<dyn Beam>, BeamError> {
    if no_beam {
        info!("Not using a beam");
        Ok(Box::new(NoBeam))
    } else {
        create_fee_beam_object(delays, beam_file)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
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
