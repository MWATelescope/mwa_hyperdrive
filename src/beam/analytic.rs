// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code for analytic beam calculations.

use log::debug;
use marlu::{AzEl, Jones};
use mwa_hyperbeam::analytic::AnalyticType;
use ndarray::prelude::*;

use super::{partial_to_full, validate_delays, Beam, BeamError, BeamType, Delays};

#[cfg(any(feature = "cuda", feature = "hip"))]
use super::{BeamGpu, DevicePointer, GpuFloat};

/// A wrapper of the `AnalyticBeam` struct in hyperbeam that implements the
/// [`Beam`] trait.
pub(crate) struct AnalyticBeam {
    hyperbeam_object: mwa_hyperbeam::analytic::AnalyticBeam,
    analytic_type: AnalyticType,
    delays: Array2<u32>,
    gains: Array2<f64>,
    ideal_delays: [u32; 16],
}

impl AnalyticBeam {
    pub(crate) fn new_mwa_pb(
        num_tiles: usize,
        delays: Delays,
        gains: Option<Array2<f64>>,
    ) -> Result<AnalyticBeam, BeamError> {
        Self::new_inner(AnalyticType::MwaPb, num_tiles, delays, gains)
    }

    pub(crate) fn new_rts(
        num_tiles: usize,
        delays: Delays,
        gains: Option<Array2<f64>>,
    ) -> Result<AnalyticBeam, BeamError> {
        Self::new_inner(AnalyticType::Rts, num_tiles, delays, gains)
    }

    fn new_inner(
        at: AnalyticType,
        num_tiles: usize,
        delays: Delays,
        gains: Option<Array2<f64>>,
    ) -> Result<AnalyticBeam, BeamError> {
        // Check that the delays are sensible.
        validate_delays(&delays, num_tiles)?;

        let ideal_delays = delays.get_ideal_delays();
        debug!("Ideal dipole delays: {:?}", ideal_delays);

        let delays = match delays {
            Delays::Full(d) => d,
            Delays::Partial(d) => partial_to_full(d, num_tiles),
        };

        // If no gains were provided, assume all are alive.
        let gains = match gains {
            Some(g) => {
                debug!("Using supplied dipole gains");
                g
            }
            None => {
                debug!("No dipole gains supplied; setting all to 1");
                Array2::ones((delays.len_of(Axis(0)), 32))
            }
        };

        // Complain if the dimensions of delays and gains don't match.
        if delays.dim().0 != gains.dim().0 {
            return Err(BeamError::DelayGainsDimensionMismatch {
                delays: delays.dim().0,
                gains: gains.dim().0,
            });
        }

        // Wrap the `AnalyticBeam` out of hyperbeam with our own `AnalyticBeam`.
        let hyperbeam_object = mwa_hyperbeam::analytic::AnalyticBeam::new_custom(
            at,
            at.get_default_dipole_height(),
            4,
        );
        Ok(AnalyticBeam {
            hyperbeam_object,
            analytic_type: at,
            delays,
            gains,
            ideal_delays,
        })
    }

    fn calc_jones_inner(
        &self,
        azel: AzEl,
        freq_hz: f64,
        delays: &[u32],
        amps: &[f64],
        latitude_rad: f64,
    ) -> Result<Jones<f64>, mwa_hyperbeam::analytic::AnalyticBeamError> {
        self.hyperbeam_object.calc_jones_pair(
            azel.az,
            azel.za(),
            freq_hz as _,
            delays,
            amps,
            latitude_rad,
            true,
        )
    }

    fn _calc_jones_array(
        &self,
        azels: &[AzEl],
        freq_hz: f64,
        delays: &[u32],
        amps: &[f64],
        latitude_rad: f64,
    ) -> Result<Vec<Jones<f64>>, mwa_hyperbeam::analytic::AnalyticBeamError> {
        self.hyperbeam_object.calc_jones_array(
            azels,
            freq_hz as _,
            delays,
            amps,
            latitude_rad,
            true,
        )
    }

    fn calc_jones_array_inner(
        &self,
        azels: &[AzEl],
        freq_hz: f64,
        delays: &[u32],
        amps: &[f64],
        latitude_rad: f64,
        results: &mut [Jones<f64>],
    ) -> Result<(), mwa_hyperbeam::analytic::AnalyticBeamError> {
        self.hyperbeam_object.calc_jones_array_inner(
            azels,
            freq_hz as _,
            delays,
            amps,
            latitude_rad,
            true,
            results,
        )
    }
}

impl Beam for AnalyticBeam {
    fn get_beam_type(&self) -> BeamType {
        match self.analytic_type {
            AnalyticType::MwaPb => BeamType::AnalyticMwaPb,
            AnalyticType::Rts => BeamType::AnalyticRts,
        }
    }

    fn get_num_tiles(&self) -> usize {
        self.delays.len_of(Axis(0))
    }

    fn get_ideal_dipole_delays(&self) -> Option<[u32; 16]> {
        Some(self.ideal_delays)
    }

    fn get_dipole_delays(&self) -> Option<ArcArray<u32, Dim<[usize; 2]>>> {
        Some(self.delays.to_shared())
    }

    fn get_dipole_gains(&self) -> Option<ArcArray<f64, Dim<[usize; 2]>>> {
        Some(self.gains.to_shared())
    }

    fn get_beam_file(&self) -> Option<&std::path::Path> {
        None
    }

    fn calc_jones(
        &self,
        azel: marlu::AzEl,
        freq_hz: f64,
        tile_index: Option<usize>,
        latitude_rad: f64,
    ) -> Result<Jones<f64>, BeamError> {
        if let Some(tile_index) = tile_index {
            if tile_index > self.delays.len_of(Axis(0)) {
                return Err(BeamError::BadTileIndex {
                    got: tile_index,
                    max: self.delays.len_of(Axis(0)),
                });
            }
            let delays = self.delays.slice(s![tile_index, ..]);
            let amps = self.gains.slice(s![tile_index, ..]);
            let j = self.calc_jones_inner(
                azel,
                freq_hz,
                delays.as_slice().unwrap(),
                amps.as_slice().unwrap(),
                latitude_rad,
            )?;
            Ok(j)
        } else {
            let delays = &self.ideal_delays;
            let amps = [1.0; 32];
            let j = self.calc_jones_inner(azel, freq_hz, delays, &amps, latitude_rad)?;
            Ok(j)
        }
    }

    fn calc_jones_array(
        &self,
        azels: &[AzEl],
        freq_hz: f64,
        tile_index: Option<usize>,
        latitude_rad: f64,
    ) -> Result<Vec<marlu::Jones<f64>>, BeamError> {
        let mut jones = vec![Jones::default(); azels.len()];
        Beam::calc_jones_array_inner(self, azels, freq_hz, tile_index, latitude_rad, &mut jones)?;
        Ok(jones)
    }

    fn calc_jones_array_inner(
        &self,
        azels: &[marlu::AzEl],
        freq_hz: f64,
        tile_index: Option<usize>,
        latitude_rad: f64,
        results: &mut [marlu::Jones<f64>],
    ) -> Result<(), BeamError> {
        if let Some(tile_index) = tile_index {
            if tile_index > self.delays.len_of(Axis(0)) {
                return Err(BeamError::BadTileIndex {
                    got: tile_index,
                    max: self.delays.len_of(Axis(0)),
                });
            }
            let delays = self.delays.slice(s![tile_index, ..]);
            let amps = self.gains.slice(s![tile_index, ..]);
            self.calc_jones_array_inner(
                azels,
                freq_hz,
                delays.as_slice().unwrap(),
                amps.as_slice().unwrap(),
                latitude_rad,
                results,
            )?;
        } else {
            let delays = &self.ideal_delays;
            let amps = [1.0; 32];
            self.calc_jones_array_inner(azels, freq_hz, delays, &amps, latitude_rad, results)?;
        }
        Ok(())
    }

    fn find_closest_freq(&self, desired_freq_hz: f64) -> f64 {
        desired_freq_hz
    }

    fn empty_coeff_cache(&self) {}

    #[cfg(any(feature = "cuda", feature = "hip"))]
    fn prepare_gpu_beam(&self, freqs_hz: &[u32]) -> Result<Box<dyn BeamGpu>, BeamError> {
        let gpu_beam = unsafe {
            self.hyperbeam_object
                .gpu_prepare(self.delays.view(), self.gains.view())?
        };
        let freq_map = (0..freqs_hz.len()).map(|i| i as i32).collect::<Vec<_>>();
        let d_freq_map = DevicePointer::copy_to_device(&freq_map)?;
        Ok(Box::new(AnalyticBeamGpu {
            hyperbeam_object: gpu_beam,
            d_freqs_hz: DevicePointer::copy_to_device(freqs_hz)?,
            d_freq_map,
        }))
    }
}

#[cfg(any(feature = "cuda", feature = "hip"))]
struct AnalyticBeamGpu {
    hyperbeam_object: mwa_hyperbeam::analytic::AnalyticBeamGpu,
    d_freqs_hz: DevicePointer<u32>,
    d_freq_map: DevicePointer<i32>,
}

#[cfg(any(feature = "cuda", feature = "hip"))]
impl BeamGpu for AnalyticBeamGpu {
    unsafe fn calc_jones_pair(
        &self,
        az_rad: &[GpuFloat],
        za_rad: &[GpuFloat],
        latitude_rad: f64,
        d_jones: *mut std::ffi::c_void,
    ) -> Result<(), BeamError> {
        let d_az_rad = DevicePointer::copy_to_device(az_rad)?;
        let d_za_rad = DevicePointer::copy_to_device(za_rad)?;
        self.hyperbeam_object.calc_jones_device_pair_inner(
            d_az_rad.get(),
            d_za_rad.get(),
            az_rad.len().try_into().expect("not bigger than i32::MAX"),
            self.d_freqs_hz.get(),
            self.d_freqs_hz
                .get_num_elements()
                .try_into()
                .expect("not bigger than i32::MAX"),
            latitude_rad as GpuFloat,
            true,
            d_jones,
        )?;
        Ok(())
    }

    fn get_beam_type(&self) -> BeamType {
        BeamType::FEE
    }

    fn get_tile_map(&self) -> *const i32 {
        self.hyperbeam_object.get_device_tile_map()
    }

    fn get_freq_map(&self) -> *const i32 {
        self.d_freq_map.get()
    }

    fn get_num_unique_tiles(&self) -> i32 {
        self.hyperbeam_object.get_num_unique_tiles()
    }

    fn get_num_unique_freqs(&self) -> i32 {
        self.d_freqs_hz
            .get_num_elements()
            .try_into()
            .expect("not bigger than i32::MAX")
    }
}
