// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code for FEE beam calculations.

use std::path::Path;

use dashmap::DashMap;
use log::trace;
use mwa_rust_core::{AzEl, Jones};
use ndarray::prelude::*;

use super::{cache::JonesHash, Beam, BeamError, BeamType, Delays};

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        use mwa_hyperbeam::cuda::*;
        use super::{BeamCUDA, CudaFloat};
    }
}

/// A wrapper of the `FEEBeam` struct in hyperbeam that implements the [Beam]
/// trait.
pub struct FEEBeam {
    hyperbeam_object: mwa_hyperbeam::fee::FEEBeam,
    delays: Array2<u32>,
    gains: Array2<f64>,
    cache: DashMap<JonesHash, Jones<f64>>,

    /// If all the tile delays are the same, then provide a single tile's delays
    /// here for convenience.
    delays_flattened: Option<Array2<u32>>,
}

impl FEEBeam {
    fn new_inner(
        hyperbeam_object: mwa_hyperbeam::fee::FEEBeam,
        num_tiles: usize,
        delays: Delays,
        gains: Option<Array2<f64>>,
    ) -> Result<Self, BeamError> {
        let delays = match delays {
            Delays::Full(d) => d,
            Delays::Partial(d) => partial_to_full(d, num_tiles),
            Delays::None | Delays::NotNecessary => return Err(BeamError::NoDelays),
        };

        // If no gains were provided, assume all are alive.
        let gains = match gains {
            Some(g) => g,
            None => Array2::from_elem(delays.dim(), 1.0),
        };

        // Complain if the dimensions of delays and gains don't match.
        // TODO: Tidy.
        if delays.dim().0 != gains.dim().0 {
            return Err(BeamError::DelayGainsDimensionMismatch {
                delays: delays.dim().0,
                gains: gains.dim().0,
            });
        }

        // Are all the delays the same? If so, keep a record of that for
        // convenience.
        let delays_flattened = {
            // Innocent until proven guilty.
            let mut all_equal = true;
            for delay_row in delays.outer_iter() {
                if delay_row != delays.slice(s![0, ..]) {
                    all_equal = false;
                    break;
                }
            }
            if all_equal {
                let slice = delays.slice(s![0, ..]);
                let mut new = Array2::zeros((1, slice.dim()));
                new.slice_mut(s![0, ..]).assign(&slice);
                log::debug!("Using delays: {:?}", slice.as_slice().unwrap());
                Some(new)
            } else {
                None
            }
        };

        // Wrap the `FEEBeam` out of hyperbeam with our own `FEEBeam`.
        Ok(FEEBeam {
            hyperbeam_object,
            delays,
            gains,
            cache: DashMap::new(),
            delays_flattened,
        })
    }

    pub fn new<T: AsRef<std::path::Path>>(
        file: T,
        num_tiles: usize,
        delays: Delays,
        gains: Option<Array2<f64>>,
    ) -> Result<FEEBeam, BeamError> {
        let hyperbeam_object = mwa_hyperbeam::fee::FEEBeam::new(file)?;
        Self::new_inner(hyperbeam_object, num_tiles, delays, gains)
    }

    pub fn new_from_env(
        num_tiles: usize,
        delays: Delays,
        gains: Option<Array2<f64>>,
    ) -> Result<FEEBeam, BeamError> {
        let hyperbeam_object = mwa_hyperbeam::fee::FEEBeam::new_from_env()?;
        Self::new_inner(hyperbeam_object, num_tiles, delays, gains)
    }

    fn calc_jones_inner(
        &self,
        azel: AzEl,
        freq_hz: f64,
        delays: &[u32],
        amps: &[f64],
    ) -> Result<Jones<f64>, mwa_hyperbeam::fee::FEEBeamError> {
        self.hyperbeam_object
            .calc_jones(azel.az, azel.za(), freq_hz as _, delays, amps, true)
    }

    /// Get the dipole delays that are used for this [FEEBeam].
    pub fn get_delays(&self) -> ArrayView2<u32> {
        if let Some(d) = &self.delays_flattened {
            d.view()
        } else {
            self.delays.view()
        }
    }
}

impl Beam for FEEBeam {
    fn calc_jones(
        &self,
        azel: AzEl,
        freq_hz: f64,
        tile_index: usize,
    ) -> Result<Jones<f64>, BeamError> {
        // Determine whether the input settings correspond to a Jones matrix in
        // the cache. If so, use the cache. Otherwise, calculate the Jones
        // matrix and populate the cache.

        // The FEE beam is defined only at specific frequencies. For this
        // reason, rather than making a unique hash for every single different
        // frequency, round specified frequency (`freq_hz`) to the nearest beam
        // frequency and use that for the hash.
        let beam_freq = self.find_closest_freq(freq_hz);

        // Are the input settings already cached? Hash them to check.
        // TODO: Check tile_index isn't too big.
        let delays = self.delays.slice(s![tile_index, ..]);
        let amps = self.gains.slice(s![tile_index, ..]);
        let hash = JonesHash::new(
            azel,
            beam_freq,
            // unwrap is safe because these array slices are contiguous.
            delays.as_slice().unwrap(),
            amps.as_slice().unwrap(),
        );

        // If the cache for this hash exists, we can return a copy of the Jones
        // matrix.
        if self.cache.contains_key(&hash) {
            // TODO: Can we avoid clone here?
            return Ok(*self.cache.get(&hash).unwrap());
        }

        // If we hit this part of the code, the relevant Jones matrix was not in
        // the cache.
        let jones = self.calc_jones_inner(
            azel,
            beam_freq,
            delays.as_slice().unwrap(),
            amps.as_slice().unwrap(),
        )?;
        self.cache.insert(hash, jones);
        Ok(*self.cache.get(&hash).unwrap())
    }

    fn find_closest_freq(&self, desired_freq_hz: f64) -> f64 {
        self.hyperbeam_object
            .find_closest_freq(desired_freq_hz as _) as _
    }

    fn get_beam_type(&self) -> BeamType {
        BeamType::FEE
    }

    fn len(&self) -> usize {
        self.cache.len()
    }

    fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    fn empty_cache(&self) {
        self.cache.clear();
    }

    fn empty_coeff_cache(&self) {
        self.hyperbeam_object.empty_cache();
    }

    #[cfg(feature = "cuda")]
    unsafe fn prepare_cuda_beam(&self, freqs_hz: &[u32]) -> Result<Box<dyn BeamCUDA>, BeamError> {
        let cuda_beam = self.hyperbeam_object.cuda_prepare(
            freqs_hz,
            self.delays.view(),
            self.gains.view(),
            true,
        )?;
        Ok(Box::new(FEEBeamCUDA {
            hyperbeam_object: cuda_beam,
        }))
    }
}

#[cfg(feature = "cuda")]
pub struct FEEBeamCUDA {
    hyperbeam_object: mwa_hyperbeam::fee::FEEBeamCUDA,
}

#[cfg(feature = "cuda")]
impl BeamCUDA for FEEBeamCUDA {
    unsafe fn calc_jones(
        &self,
        azels: &[AzEl],
    ) -> Result<DevicePointer<Jones<CudaFloat>>, BeamError> {
        let (azs, zas): (Vec<CudaFloat>, Vec<CudaFloat>) = azels
            .iter()
            .map(|&azel| (azel.az as CudaFloat, azel.za() as CudaFloat))
            .unzip();
        self.hyperbeam_object
            .calc_jones_device(&azs, &zas, true)
            .map_err(BeamError::from)
    }

    fn get_beam_type(&self) -> BeamType {
        BeamType::FEE
    }

    fn get_beam_jones_map(&self) -> *const u64 {
        self.hyperbeam_object.get_beam_jones_map()
    }

    fn get_num_unique_freqs(&self) -> i32 {
        self.hyperbeam_object.get_num_unique_freqs()
    }
}

/// Create an FEE beam object. The dipole delays and amps (a.k.a. gains) must
/// have the same number of rows; these correspond to individual tiles.
///
/// `amps` *must* have 16 or 32 elements per row (each corresponds to an MWA
/// dipole in a tile, in the M&C order; see
/// <https://wiki.mwatelescope.org/pages/viewpage.action?pageId=48005139>).
pub fn create_fee_beam_object<T: AsRef<Path>>(
    beam_file: Option<T>,
    num_tiles: usize,
    delays: Delays,
    gains: Option<Array2<f64>>,
) -> Result<Box<dyn Beam>, BeamError> {
    trace!("Setting up FEE beam");
    let beam = if let Some(bf) = beam_file {
        // Set up the FEE beam struct from the specified beam file.
        Box::new(FEEBeam::new(&bf, num_tiles, delays, gains)?)
    } else {
        // Set up the FEE beam struct from the MWA_BEAM_FILE environment
        // variable.
        Box::new(FEEBeam::new_from_env(num_tiles, delays, gains)?)
    };
    Ok(beam)
}

/// Assume that the dipole delays for all tiles is the same as the delays for
/// one tile.
fn partial_to_full(delays: Vec<u32>, num_tiles: usize) -> Array2<u32> {
    let mut out = Array2::zeros((num_tiles, 16));
    let d = Array1::from(delays);
    out.outer_iter_mut().for_each(|mut tile_delays| {
        tile_delays.assign(&d);
    });
    out
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
        let amps = [1.0; 16];
        let freq = 150e6;
        let azels = [
            AzEl { az: 0.0, el: 0.0 },
            AzEl { az: 1.0, el: 0.1 },
            AzEl { az: -1.0, el: 0.2 },
        ];
        let (azs, zas): (Vec<_>, Vec<_>) = azels.iter().map(|azel| (azel.az, azel.za())).unzip();

        // Get the beam values right out of hyperbeam.
        let hyperbeam = FEEBeam::new_from_env().unwrap();
        let hyperbeam_values = hyperbeam
            .calc_jones_array(&azs, &zas, freq as u32, &delays, &amps, true)
            .unwrap();
        // Put the hyperbeam results into hyperdrive `Jones` objects.
        let hyperbeam_values = Array1::from(hyperbeam_values.to_vec());

        // Compare these with the hyperdrive `Beam` trait.
        let gains = array![amps];
        let hyperdrive =
            super::FEEBeam::new_from_env(1, Delays::Partial(delays), Some(gains)).unwrap();
        let hyperdrive_values: Vec<Jones<f64>> = azels
            .iter()
            .map(|&azel| hyperdrive.calc_jones(azel, freq, 0).unwrap())
            .collect();

        assert_abs_diff_eq!(Array1::from(hyperdrive_values), hyperbeam_values);
    }
}
