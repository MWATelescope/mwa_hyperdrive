// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code for FEE beam calculations.

use mwa_hyperbeam;
use ndarray::prelude::*;
use rayon::prelude::*;

use super::{Beam, BeamError, Delays};
use crate::{AzEl, Jones};

/// A wrapper of the `FEEBeam` struct in hyperbeam that implements the [Beam]
/// trait.
pub struct FEEBeam {
    beam: mwa_hyperbeam::fee::FEEBeam,
    delays: Vec<u32>,
}

impl FEEBeam {
    fn new_inner(beam: mwa_hyperbeam::fee::FEEBeam, delays: Delays) -> Result<Self, BeamError> {
        let delays = match delays {
            Delays::Available(d) => d,
            Delays::None | Delays::NotNecessary => return Err(BeamError::NoDelays),
        };

        // Wrap the `FEEBeam` out of hyperbeam with our own `FEEBeam`.
        Ok(FEEBeam { beam, delays })
    }

    pub fn new<T: AsRef<std::path::Path>>(file: T, delays: Delays) -> Result<Self, BeamError> {
        let beam = mwa_hyperbeam::fee::FEEBeam::new(file)?;
        Self::new_inner(beam, delays)
    }

    pub fn new_from_env(delays: Delays) -> Result<Self, BeamError> {
        let beam = mwa_hyperbeam::fee::FEEBeam::new_from_env()?;
        Self::new_inner(beam, delays)
    }

    pub fn get_delays(&self) -> &[u32] {
        &self.delays
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
        let j = j.mapv(Jones::from);
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
}