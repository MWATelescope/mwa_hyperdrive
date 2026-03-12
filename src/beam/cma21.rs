// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Provisional 21CMA beam implementations.
//!
//! `Cma21GaussianBeam` is intentionally conservative. It only models the
//! published NCP-centred main lobe and ignores poorly constrained sidelobe and
//! polarisation structure. The current approximation is:
//!
//! - pointing centre fixed at the North Celestial Pole;
//! - circular Gaussian power beam;
//! - sigma = 3.33 deg * (nu / 100 MHz)^-1.14.
//!
//! This follows the public 21CMA NCP primary-beam fit reported by Zheng et al.
//! (2024), which models the 24h-averaged beam as a Gaussian profile with
//! `theta_b = 3.33 deg * (nu / 100 MHz)^-1.14`.

#[cfg(any(feature = "cuda", feature = "hip"))]
use std::f64::consts::FRAC_PI_2;

use marlu::{AzEl, Jones};
use ndarray::prelude::*;
use num_complex::Complex;

use super::{Beam, BeamError, BeamType};
#[cfg(any(feature = "cuda", feature = "hip"))]
use super::{BeamGpu, DevicePointer, GpuFloat};

const CMA21_REF_FREQ_HZ: f64 = 100e6;
const CMA21_MAIN_LOBE_SIGMA_DEG_AT_100MHZ: f64 = 3.33;
const CMA21_MAIN_LOBE_SIGMA_FREQ_INDEX: f64 = -1.14;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Cma21GaussianBeam {
    pub(crate) num_tiles: usize,
}

impl Cma21GaussianBeam {
    fn main_lobe_sigma_rad(freq_hz: f64) -> f64 {
        let scale = (freq_hz / CMA21_REF_FREQ_HZ).powf(CMA21_MAIN_LOBE_SIGMA_FREQ_INDEX);
        CMA21_MAIN_LOBE_SIGMA_DEG_AT_100MHZ.to_radians() * scale
    }

    fn north_celestial_pole_azel(latitude_rad: f64) -> AzEl {
        AzEl::from_radians(0.0, latitude_rad)
    }

    fn angular_separation_rad(azel: AzEl, latitude_rad: f64) -> f64 {
        let ncp = Self::north_celestial_pole_azel(latitude_rad);
        let cos_sep = (azel.el.sin() * ncp.el.sin()
            + azel.el.cos() * ncp.el.cos() * (azel.az - ncp.az).cos())
        .clamp(-1.0, 1.0);
        cos_sep.acos()
    }

    fn calc_amplitude(azel: AzEl, freq_hz: f64, latitude_rad: f64) -> f64 {
        if !freq_hz.is_finite() || freq_hz <= 0.0 || azel.el <= 0.0 {
            return 0.0;
        }

        let sigma = Self::main_lobe_sigma_rad(freq_hz);
        let theta = Self::angular_separation_rad(azel, latitude_rad);

        // Zheng et al. (2024) fit the 24h-averaged NCP primary beam with a
        // Gaussian power profile F = exp(-theta^2 / (2 sigma^2)). The Jones
        // amplitude is the square root of that power response.
        (-(theta * theta) / (4.0 * sigma * sigma)).exp()
    }

    fn calc_scalar_jones(azel: AzEl, freq_hz: f64, latitude_rad: f64) -> Jones<f64> {
        let amp = Self::calc_amplitude(azel, freq_hz, latitude_rad);
        Jones::from([
            Complex::new(amp, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(amp, 0.0),
        ])
    }
}

impl Beam for Cma21GaussianBeam {
    fn get_beam_type(&self) -> BeamType {
        BeamType::Cma21Gaussian
    }

    fn get_num_tiles(&self) -> usize {
        self.num_tiles
    }

    fn get_dipole_delays(&self) -> Option<ArcArray<u32, Dim<[usize; 2]>>> {
        None
    }

    fn get_ideal_dipole_delays(&self) -> Option<[u32; 16]> {
        None
    }

    fn get_dipole_gains(&self) -> Option<ArcArray<f64, Dim<[usize; 2]>>> {
        None
    }

    fn get_beam_file(&self) -> Option<&std::path::Path> {
        None
    }

    fn calc_jones(
        &self,
        azel: AzEl,
        freq_hz: f64,
        _tile_index: Option<usize>,
        latitude_rad: f64,
    ) -> Result<Jones<f64>, BeamError> {
        Ok(Self::calc_scalar_jones(azel, freq_hz, latitude_rad))
    }

    fn calc_jones_array(
        &self,
        azels: &[AzEl],
        freq_hz: f64,
        tile_index: Option<usize>,
        latitude_rad: f64,
    ) -> Result<Vec<Jones<f64>>, BeamError> {
        let mut results = vec![Jones::default(); azels.len()];
        self.calc_jones_array_inner(azels, freq_hz, tile_index, latitude_rad, &mut results)?;
        Ok(results)
    }

    fn calc_jones_array_inner(
        &self,
        azels: &[AzEl],
        freq_hz: f64,
        _tile_index: Option<usize>,
        latitude_rad: f64,
        results: &mut [Jones<f64>],
    ) -> Result<(), BeamError> {
        azels
            .iter()
            .copied()
            .zip(results.iter_mut())
            .for_each(|(azel, result)| {
                *result = Self::calc_scalar_jones(azel, freq_hz, latitude_rad);
            });
        Ok(())
    }

    fn find_closest_freq(&self, desired_freq_hz: f64) -> f64 {
        desired_freq_hz
    }

    fn empty_coeff_cache(&self) {}

    #[cfg(any(feature = "cuda", feature = "hip"))]
    fn prepare_gpu_beam(&self, freqs_hz: &[u32]) -> Result<Box<dyn BeamGpu>, BeamError> {
        let tile_map = DevicePointer::copy_to_device(&vec![0; self.num_tiles])?;
        let freq_map = DevicePointer::copy_to_device(
            &(0..freqs_hz.len()).map(|i| i as i32).collect::<Vec<_>>(),
        )?;
        Ok(Box::new(Cma21GaussianBeamGpu {
            cpu_object: *self,
            freqs_hz: freqs_hz.to_vec(),
            tile_map,
            freq_map,
        }))
    }
}

#[cfg(any(feature = "cuda", feature = "hip"))]
pub(crate) struct Cma21GaussianBeamGpu {
    cpu_object: Cma21GaussianBeam,
    freqs_hz: Vec<u32>,
    tile_map: DevicePointer<i32>,
    freq_map: DevicePointer<i32>,
}

#[cfg(any(feature = "cuda", feature = "hip"))]
impl BeamGpu for Cma21GaussianBeamGpu {
    unsafe fn calc_jones_pair(
        &self,
        az_rad: &[GpuFloat],
        za_rad: &[GpuFloat],
        latitude_rad: f64,
        d_jones: *mut std::ffi::c_void,
    ) -> Result<(), BeamError> {
        #[cfg(all(any(feature = "cuda", feature = "hip"), not(feature = "gpu-single")))]
        let azels = az_rad
            .iter()
            .zip(za_rad.iter())
            .map(|(&az, &za)| AzEl::from_radians(az, FRAC_PI_2 - za))
            .collect::<Vec<_>>();
        #[cfg(feature = "gpu-single")]
        let azels = az_rad
            .iter()
            .zip(za_rad.iter())
            .map(|(&az, &za)| AzEl::from_radians(az as f64, FRAC_PI_2 - za as f64))
            .collect::<Vec<_>>();

        let mut beam_cube = Array2::zeros((self.freqs_hz.len(), az_rad.len()));
        #[cfg(feature = "gpu-single")]
        let mut tmp = vec![Jones::default(); az_rad.len()];
        for (mut out_row, &freq_hz) in beam_cube.outer_iter_mut().zip(self.freqs_hz.iter()) {
            let out_row = out_row
                .as_slice_mut()
                .expect("beam output row should be contiguous");
            cfg_if::cfg_if! {
                if #[cfg(feature = "gpu-single")] {
                    self.cpu_object.calc_jones_array_inner(
                        &azels,
                        f64::from(freq_hz),
                        None,
                        latitude_rad,
                        &mut tmp,
                    )?;
                    out_row
                        .iter_mut()
                        .zip(tmp.iter())
                        .for_each(|(dst, src)| *dst = Jones::<f32>::from(*src));
                } else {
                    self.cpu_object.calc_jones_array_inner(
                        &azels,
                        f64::from(freq_hz),
                        None,
                        latitude_rad,
                        out_row,
                    )?;
                }
            }
        }

        #[cfg(feature = "cuda")]
        use cuda_runtime_sys::{
            cudaMemcpy as gpuMemcpy,
            cudaMemcpyKind::cudaMemcpyHostToDevice as gpuMemcpyHostToDevice,
        };
        #[cfg(feature = "hip")]
        use hip_sys::hiprt::{
            hipMemcpy as gpuMemcpy, hipMemcpyKind::hipMemcpyHostToDevice as gpuMemcpyHostToDevice,
        };

        gpuMemcpy(
            d_jones,
            beam_cube.as_ptr().cast(),
            beam_cube.len() * std::mem::size_of::<Jones<GpuFloat>>(),
            gpuMemcpyHostToDevice,
        );
        crate::gpu::check_for_errors(crate::gpu::GpuCall::CopyToDevice)?;
        Ok(())
    }

    fn get_beam_type(&self) -> BeamType {
        BeamType::Cma21Gaussian
    }

    fn get_tile_map(&self) -> *const i32 {
        self.tile_map.get()
    }

    fn get_freq_map(&self) -> *const i32 {
        self.freq_map.get()
    }

    fn get_num_unique_tiles(&self) -> i32 {
        1
    }

    fn get_num_unique_freqs(&self) -> i32 {
        self.freqs_hz.len() as i32
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn main_lobe_is_unity_at_ncp() {
        let latitude = 42.9242_f64.to_radians();
        let azel = AzEl::from_radians(0.0, latitude);
        let amp = Cma21GaussianBeam::calc_amplitude(azel, 125e6, latitude);
        assert_abs_diff_eq!(amp, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn off_axis_response_decreases() {
        let latitude = 42.9242_f64.to_radians();
        let beam = Cma21GaussianBeam { num_tiles: 1 };
        let on_axis = beam
            .calc_jones(AzEl::from_radians(0.0, latitude), 125e6, None, latitude)
            .unwrap();
        let off_axis = beam
            .calc_jones(
                AzEl::from_radians(0.0, latitude - 5f64.to_radians()),
                125e6,
                None,
                latitude,
            )
            .unwrap();
        assert!(off_axis[0].norm() < on_axis[0].norm());
        assert!(off_axis[3].norm() < on_axis[3].norm());
    }

    #[test]
    fn horizon_is_suppressed() {
        let latitude = 42.9242_f64.to_radians();
        let beam = Cma21GaussianBeam { num_tiles: 1 };
        let jones = beam
            .calc_jones(AzEl::from_radians(0.0, 0.0), 125e6, None, latitude)
            .unwrap();
        assert!(jones[0].norm() < 1e-10);
        assert!(jones[3].norm() < 1e-10);
    }
}
