// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code for FEE beam calculations.

use std::path::{Path, PathBuf};

use log::debug;
use marlu::{AzEl, Jones};
use ndarray::prelude::*;

use super::{partial_to_full, validate_delays, Beam, BeamError, BeamType, Delays};

#[cfg(any(feature = "cuda", feature = "hip"))]
use super::{BeamGpu, DevicePointer, GpuFloat, GpuJones};

#[cfg(feature = "cuda")]
use cuda_runtime_sys::{
    cudaMemcpy as gpuMemcpy, cudaMemcpyKind::cudaMemcpyDeviceToHost as gpuMemcpyDeviceToHost,
    cudaMemcpyKind::cudaMemcpyHostToDevice as gpuMemcpyHostToDevice,
};
#[cfg(feature = "hip")]
use hip_sys::hiprt::{
    hipMemcpy as gpuMemcpy, hipMemcpyKind::hipMemcpyDeviceToHost as gpuMemcpyDeviceToHost,
    hipMemcpyKind::hipMemcpyHostToDevice as gpuMemcpyHostToDevice,
};

/// A wrapper of the `FEEBeam` struct in hyperbeam that implements the [`Beam`]
/// trait.
pub(crate) struct FEEBeam {
    hyperbeam_object: mwa_hyperbeam::fee::FEEBeam,
    delays: Array2<u32>,
    gains: Array2<f64>,
    ideal_delays: Box<[u32; 16]>,
    file: PathBuf,

    /// If there's CRAM tile info, this is the tile index and dipole gains.
    cram_tile: Option<(usize, Box<[f64; 64]>)>,
}

impl FEEBeam {
    fn new_inner(
        hyperbeam_object: mwa_hyperbeam::fee::FEEBeam,
        num_tiles: usize,
        delays: Delays,
        gains: Option<Array2<f64>>,
        file: Option<&Path>,
        cram_tile: Option<(usize, Box<[f64; 64]>)>,
    ) -> Result<FEEBeam, BeamError> {
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

        // Wrap the `FEEBeam` out of hyperbeam with our own `FEEBeam`.
        Ok(FEEBeam {
            hyperbeam_object,
            delays,
            gains,
            ideal_delays: Box::new(ideal_delays),
            file: match file {
                Some(p) => p.to_path_buf(),
                None => PathBuf::from(std::env::var("MWA_BEAM_FILE").unwrap()),
            },
            cram_tile,
        })
    }

    pub(crate) fn new(
        file: &Path,
        num_tiles: usize,
        delays: Delays,
        gains: Option<Array2<f64>>,
        cram_tile: Option<(usize, Box<[f64; 64]>)>,
    ) -> Result<FEEBeam, BeamError> {
        let hyperbeam_object = mwa_hyperbeam::fee::FEEBeam::new(file)?;
        Self::new_inner(
            hyperbeam_object,
            num_tiles,
            delays,
            gains,
            Some(file),
            cram_tile,
        )
    }

    pub(crate) fn new_from_env(
        num_tiles: usize,
        delays: Delays,
        gains: Option<Array2<f64>>,
        cram_tile: Option<(usize, Box<[f64; 64]>)>,
    ) -> Result<FEEBeam, BeamError> {
        let hyperbeam_object = mwa_hyperbeam::fee::FEEBeam::new_from_env()?;
        Self::new_inner(hyperbeam_object, num_tiles, delays, gains, None, cram_tile)
    }

    fn calc_jones_inner(
        &self,
        azel: AzEl,
        freq_hz: f64,
        delays: &[u32],
        amps: &[f64],
        latitude_rad: f64,
    ) -> Result<Jones<f64>, mwa_hyperbeam::fee::FEEBeamError> {
        self.hyperbeam_object.calc_jones_pair(
            azel.az,
            azel.za(),
            freq_hz as _,
            delays,
            amps,
            true,
            Some(latitude_rad),
            false,
        )
    }

    fn _calc_jones_array(
        &self,
        azels: &[AzEl],
        freq_hz: f64,
        delays: &[u32],
        amps: &[f64],
        latitude_rad: f64,
    ) -> Result<Vec<Jones<f64>>, mwa_hyperbeam::fee::FEEBeamError> {
        self.hyperbeam_object.calc_jones_array(
            azels,
            freq_hz as _,
            delays,
            amps,
            true,
            Some(latitude_rad),
            false,
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
    ) -> Result<(), mwa_hyperbeam::fee::FEEBeamError> {
        self.hyperbeam_object.calc_jones_array_inner(
            azels,
            freq_hz as _,
            delays,
            amps,
            true,
            Some(latitude_rad),
            false,
            results,
        )
    }
}

impl Beam for FEEBeam {
    fn get_beam_type(&self) -> BeamType {
        BeamType::FEE
    }

    fn get_num_tiles(&self) -> usize {
        self.delays.len_of(Axis(0))
    }

    fn get_ideal_dipole_delays(&self) -> Option<&[u32; 16]> {
        Some(&self.ideal_delays)
    }

    fn get_dipole_delays(&self) -> Option<ArcArray<u32, Dim<[usize; 2]>>> {
        Some(self.delays.to_shared())
    }

    fn get_dipole_gains(&self) -> Option<ArcArray<f64, Dim<[usize; 2]>>> {
        Some(self.gains.to_shared())
    }

    fn get_beam_file(&self) -> Option<&Path> {
        Some(&self.file)
    }

    fn calc_jones(
        &self,
        azel: AzEl,
        freq_hz: f64,
        tile_index: Option<usize>,
        latitude_rad: f64,
    ) -> Result<Jones<f64>, BeamError> {
        // The FEE beam is defined only at specific frequencies. For this
        // reason, rather than making a unique hash for every single different
        // frequency, round specified frequency (`freq_hz`) to the nearest beam
        // frequency and use that for the hash.
        let beam_freq = self.find_closest_freq(freq_hz);

        let jones = if let Some(tile_index) = tile_index {
            if tile_index > self.delays.len_of(Axis(0)) {
                return Err(BeamError::BadTileIndex {
                    got: tile_index,
                    max: self.delays.len_of(Axis(0)),
                });
            }
            let delays = self.delays.slice(s![tile_index, ..]);
            let amps = self.gains.slice(s![tile_index, ..]);
            self.calc_jones_inner(
                azel,
                beam_freq,
                delays.as_slice().unwrap(),
                amps.as_slice().unwrap(),
                latitude_rad,
            )?
        } else {
            let delays = &self.ideal_delays;
            let amps = [1.0; 32];
            self.calc_jones_inner(azel, beam_freq, delays.as_slice(), &amps, latitude_rad)?
        };
        Ok(jones)
    }

    fn calc_jones_array(
        &self,
        azels: &[AzEl],
        freq_hz: f64,
        tile_index: Option<usize>,
        latitude_rad: f64,
    ) -> Result<Vec<Jones<f64>>, BeamError> {
        let mut jones = vec![Jones::default(); azels.len()];
        Beam::calc_jones_array_inner(self, azels, freq_hz, tile_index, latitude_rad, &mut jones)?;
        Ok(jones)
    }

    fn calc_jones_array_inner(
        &self,
        azels: &[AzEl],
        freq_hz: f64,
        tile_index: Option<usize>,
        latitude_rad: f64,
        results: &mut [Jones<f64>],
    ) -> Result<(), BeamError> {
        // The FEE beam is defined only at specific frequencies. For this
        // reason, rather than making a unique hash for every single different
        // frequency, round specified frequency (`freq_hz`) to the nearest beam
        // frequency and use that for the hash.
        let beam_freq = self.find_closest_freq(freq_hz);

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
                beam_freq,
                delays.as_slice().unwrap(),
                amps.as_slice().unwrap(),
                latitude_rad,
                results,
            )?;
        } else {
            let delays = &self.ideal_delays;
            let amps = [1.0; 32];
            self.calc_jones_array_inner(
                azels,
                beam_freq,
                delays.as_slice(),
                &amps,
                latitude_rad,
                results,
            )?;
        }
        Ok(())
    }

    fn find_closest_freq(&self, desired_freq_hz: f64) -> f64 {
        self.hyperbeam_object
            .find_closest_freq(desired_freq_hz as _) as _
    }

    fn empty_coeff_cache(&self) {
        self.hyperbeam_object.empty_cache();
    }

    fn get_cram_tile(&self) -> Option<(usize, &[f64; 64])> {
        self.cram_tile.as_ref().map(|(i, g)| (*i, &**g))
    }

    #[cfg(any(feature = "cuda", feature = "hip"))]
    fn prepare_gpu_beam(&self, freqs_hz: &[u32]) -> Result<Box<dyn BeamGpu>, BeamError> {
        let gpu_beam = unsafe {
            self.hyperbeam_object.gpu_prepare(
                freqs_hz,
                self.delays.view(),
                self.gains.view(),
                true,
            )?
        };

        // hyperbeam only knows about normal tiles, not the CRAM tile. The
        // current approach of dealing with the CRAM tile is to consider it
        // as the last tile. This means that the tile map out of hyperbeam is
        // wrong. Copy and edit hyperbeam's tile map and use our own.
        let total_num_tiles = gpu_beam.get_total_num_tiles();
        let (gpu_cram_beam, tile_map) =
            if let Some((i_cram_tile, cram_amps)) = self.cram_tile.as_ref() {
                let d_hyperbeam_tile_map = gpu_beam.get_device_tile_map();
                let mut hyperbeam_tile_map = vec![0; total_num_tiles];
                unsafe {
                    gpuMemcpy(
                        hyperbeam_tile_map.as_mut_ptr().cast(),
                        d_hyperbeam_tile_map.cast(),
                        total_num_tiles * std::mem::size_of::<i32>(),
                        gpuMemcpyDeviceToHost,
                    );
                }

                // Adjust the map for the CRAM tile.
                hyperbeam_tile_map[*i_cram_tile] =
                    hyperbeam_tile_map.iter().copied().max().expect("not empty") + 1;
                let tile_map = DevicePointer::copy_to_device(&hyperbeam_tile_map)?;

                // Set up a new beam object, RTS style because mwa_pb doesn't
                // look as good.
                use mwa_hyperbeam::analytic::{AnalyticBeam, AnalyticType};
                let at = AnalyticType::Rts;
                let cram_beam = AnalyticBeam::new_custom(at, at.get_default_dipole_height(), 8);
                let gpu_cram_beam = unsafe {
                    cram_beam.gpu_prepare(
                        Array2::zeros((1, 64)).view(),
                        ArrayView2::from_shape((1, 64), cram_amps.as_slice()).expect("valid"),
                    )?
                };

                (Some(gpu_cram_beam), tile_map)
            } else {
                // No adjustment needed, but we need to own the tile map, so copy
                // it.
                let our_map = vec![0; gpu_beam.get_total_num_tiles()];
                let mut d_our_map = DevicePointer::copy_to_device(&our_map)?;
                unsafe {
                    gpuMemcpy(
                        d_our_map.get_mut().cast(),
                        our_map.as_ptr().cast(),
                        total_num_tiles * std::mem::size_of::<i32>(),
                        gpuMemcpyHostToDevice,
                    );
                }

                (None, d_our_map)
            };

        // We only need the FEE-unique frequencies.
        let freqs_hz = {
            let mut unique = std::collections::HashSet::with_capacity(freqs_hz.len());
            unique.extend(
                freqs_hz
                    .iter()
                    .copied()
                    .map(|f| self.hyperbeam_object.find_closest_freq(f)),
            );
            let mut v: Vec<_> = unique.into_iter().collect();
            v.sort_unstable();
            DevicePointer::copy_to_device(&v)?
        };

        Ok(Box::new(FEEBeamGpu {
            hyperbeam_object: gpu_beam,
            freqs_hz,
            tile_map,
            cram_beam: gpu_cram_beam,
        }))
    }
}

#[cfg(any(feature = "cuda", feature = "hip"))]
struct FEEBeamGpu {
    hyperbeam_object: mwa_hyperbeam::fee::FEEBeamGpu,

    /// FEE-unique frequencies \[Hz\].
    freqs_hz: DevicePointer<u32>,

    tile_map: DevicePointer<i32>,

    /// If this is set, then there is a CRAM tile present, and this object will
    /// be used to supply beam responses.
    cram_beam: Option<mwa_hyperbeam::analytic::AnalyticBeamGpu>,
}

#[cfg(any(feature = "cuda", feature = "hip"))]
impl BeamGpu for FEEBeamGpu {
    unsafe fn calc_jones_pair(
        &self,
        d_az_rad: &DevicePointer<GpuFloat>,
        d_za_rad: &DevicePointer<GpuFloat>,
        latitude_rad: f64,
        d_jones: &mut DevicePointer<GpuJones>,
    ) -> Result<(), BeamError> {
        let d_array_latitude_rad = DevicePointer::copy_to_device(&[latitude_rad as GpuFloat])?;
        let num_directions = d_az_rad
            .get_num_elements()
            .try_into()
            .expect("not bigger than i32::MAX");
        self.hyperbeam_object.calc_jones_device_pair_inner(
            d_az_rad.get(),
            d_za_rad.get(),
            num_directions,
            d_array_latitude_rad.get(),
            false,
            d_jones.get_mut().cast(),
        )?;

        // Overwrite beam responses for the tile corresponding to the CRAM tile,
        // if we have that info.
        if let Some(cram_beam) = self.cram_beam.as_ref() {
            if !matches!(self.get_beam_type(), BeamType::None) {
                cram_beam.calc_jones_device_pair_inner(
                    d_az_rad.get(),
                    d_za_rad.get(),
                    num_directions,
                    self.freqs_hz.get(),
                    self.freqs_hz.get_num_elements().try_into().expect("valid"),
                    latitude_rad as GpuFloat,
                    true,
                    // The CRAM tile always comes last.
                    d_jones
                        .get_mut()
                        .add(
                            self.hyperbeam_object.get_num_unique_tiles() as usize
                                * self.hyperbeam_object.get_num_unique_freqs() as usize
                                * d_az_rad.get_num_elements(),
                        )
                        .cast(),
                )?;
            }
        }

        Ok(())
    }

    fn get_beam_type(&self) -> BeamType {
        BeamType::FEE
    }

    fn get_tile_map(&self) -> *const i32 {
        self.tile_map.get()
    }

    fn get_freq_map(&self) -> *const i32 {
        self.hyperbeam_object.get_device_freq_map()
    }

    fn get_num_unique_tiles(&self) -> i32 {
        self.hyperbeam_object.get_num_unique_tiles() + if self.cram_beam.is_some() { 1 } else { 0 }
    }

    fn get_num_unique_freqs(&self) -> i32 {
        self.hyperbeam_object.get_num_unique_freqs()
    }
}
