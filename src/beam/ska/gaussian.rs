// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::f64::consts::FRAC_PI_2;

use marlu::{AzEl, Jones, RADec, LMN};
use ndarray::prelude::*;
use num_complex::Complex;
use rayon::prelude::*;

use super::{NUM_STATIONS, PHASE_CENTRE, REF_FREQ_HZ, SKA_LATITUDE_RAD};
use crate::beam::{Beam, BeamError, BeamType};
#[cfg(any(feature = "cuda", feature = "hip"))]
use crate::beam::{BeamGpu, DevicePointer, GpuFloat};

const FWHM_RAD: f64 = 0.07452555906;
const FWHM_FACTOR: f64 = 2.35482004503;

#[derive(Clone, Copy)]
pub(crate) struct SkaGaussianBeam;

/// Analytic Beam implementation.
impl SkaGaussianBeam {
    /// Explicitly a 2D gaussian function
    #[allow(clippy::too_many_arguments)]
    fn gaussian_2d(
        x: f64,
        y: f64,
        xo: f64,
        yo: f64,
        sigma: f64,
        // sigma_x: f64,
        // sigma_y: f64,
        // cos_theta: f64,
        // sin_theta: f64,
        // sin_2theta: f64,
    ) -> f64 {
        // // these are related to position angle, which I'm setting to zero
        // let cos_theta = 1.0;
        // let sin_theta = 0.0;
        // let sin_2theta = 0.0;

        // let sigma_x_2 = sigma_x * sigma_x;
        // let sigma_y_2 = sigma_y * sigma_y;
        // let sin_theta_2 = sin_theta * sin_theta;
        // let cos_theta_2 = cos_theta * cos_theta;

        // let a = cos_theta_2 / (2. * sigma_x_2) + sin_theta_2 / (2. * sigma_y_2);
        // let b = -sin_2theta / (4. * sigma_x_2) + sin_2theta / (4. * sigma_y_2);
        // let c = sin_theta_2 / (2. * sigma_x_2) + cos_theta_2 / (2. * sigma_y_2);

        // (-(a * (x - xo) * (x - xo) + 2. * b * (x - xo) * (y - yo) + c * (y - yo) * (y - yo))).exp()

        let sigma_2 = sigma * sigma;
        let a = 1.0 / (2. * sigma_2);
        let _b = 0.0;
        let c = 1.0 / (2. * sigma_2);

        let x_diff = x - xo;
        let y_diff = y - yo;

        (-(a * x_diff * x_diff + c * y_diff * y_diff)).exp()
    }

    fn calc_jones_inner(
        azel: AzEl,
        lst_rad: f64,
        zenith_radec: RADec,
        cent_l: f64,
        cent_m: f64,
        sigma: f64,
    ) -> Jones<f64> {
        let beam_radec = azel.to_hadec(SKA_LATITUDE_RAD).to_radec(lst_rad);
        let LMN {
            l: beam_l,
            m: beam_m,
            ..
        } = beam_radec.to_lmn(zenith_radec);

        let beam_real = SkaGaussianBeam::gaussian_2d(beam_l, beam_m, cent_l, cent_m, sigma);

        Jones::from([
            Complex::new(beam_real, 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(beam_real, 0.),
        ])
    }
}

impl Beam for SkaGaussianBeam {
    fn get_beam_type(&self) -> BeamType {
        BeamType::SkaGaussian
    }

    fn get_num_tiles(&self) -> usize {
        NUM_STATIONS
    }

    fn get_dipole_gains(&self) -> Option<ArcArray<f64, Dim<[usize; 2]>>> {
        None
    }

    /// Derived with help from Dev Null and Jack Line.
    fn calc_jones(
        &self,
        azel: AzEl,
        freq_hz: f64,
        _tile_index: Option<usize>,
        latitude_rad: f64,
    ) -> Result<Jones<f64>, BeamError> {
        let lst_rad = latitude_rad;
        let zenith_radec = RADec::from_radians(lst_rad, SKA_LATITUDE_RAD);
        let LMN {
            l: cent_l,
            m: cent_m,
            ..
        } = PHASE_CENTRE.to_lmn(zenith_radec);

        // scale fwhm to be in l,m coords
        let fwhm_lm = FWHM_RAD.sin();
        let std = (fwhm_lm / FWHM_FACTOR) * (REF_FREQ_HZ / freq_hz);

        Ok(SkaGaussianBeam::calc_jones_inner(
            azel,
            lst_rad,
            zenith_radec,
            cent_l,
            cent_m,
            std,
        ))
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
        let lst_rad = latitude_rad;
        let zenith_radec = RADec::from_radians(lst_rad, SKA_LATITUDE_RAD);
        let LMN {
            l: cent_l,
            m: cent_m,
            ..
        } = PHASE_CENTRE.to_lmn(zenith_radec);

        // scale fwhm to be in l,m coords
        let fwhm_lm = FWHM_RAD.sin();
        let std = (fwhm_lm / FWHM_FACTOR) * (REF_FREQ_HZ / freq_hz);

        azels
            .par_iter()
            .zip(results.par_iter_mut())
            .for_each(|(&azel, result)| {
                *result = SkaGaussianBeam::calc_jones_inner(
                    azel,
                    lst_rad,
                    zenith_radec,
                    cent_l,
                    cent_m,
                    std,
                );
            });
        Ok(())
    }

    #[cfg(any(feature = "cuda", feature = "hip"))]
    fn prepare_gpu_beam(&self, freqs_hz: &[u32]) -> Result<Box<dyn BeamGpu>, BeamError> {
        // All "tiles" have the same response.
        let tile_map = DevicePointer::copy_to_device(&vec![0; NUM_STATIONS])?;
        // Each frequency is distinct.
        let freq_map = DevicePointer::copy_to_device(
            &(0..freqs_hz.len())
                .map(|usize| usize as i32)
                .collect::<Vec<_>>(),
        )?;
        let obj = SkaGaussianBeamGpu {
            cpu_object: *self,
            freqs_hz: freqs_hz.to_vec(),
            tile_map,
            freq_map,
        };
        Ok(Box::new(obj))
    }

    fn get_dipole_delays(&self) -> Option<ArcArray<u32, Dim<[usize; 2]>>> {
        None
    }

    fn get_ideal_dipole_delays(&self) -> Option<[u32; 16]> {
        None
    }

    fn get_beam_file(&self) -> Option<&std::path::Path> {
        None
    }

    fn find_closest_freq(&self, desired_freq_hz: f64) -> f64 {
        desired_freq_hz
    }

    fn empty_coeff_cache(&self) {}
}

#[cfg(any(feature = "cuda", feature = "hip"))]
pub(crate) struct SkaGaussianBeamGpu {
    cpu_object: SkaGaussianBeam,
    freqs_hz: Vec<u32>,
    tile_map: DevicePointer<i32>,
    freq_map: DevicePointer<i32>,
}

#[cfg(any(feature = "cuda", feature = "hip"))]
impl BeamGpu for SkaGaussianBeamGpu {
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

        let mut a = Array2::zeros((self.freqs_hz.len(), az_rad.len()));
        #[cfg(feature = "gpu-single")]
        let mut v = vec![Jones::default(); az_rad.len()];
        for (mut a, &freq) in a.outer_iter_mut().zip(self.freqs_hz.iter()) {
            let a = a
                .as_slice_mut()
                .expect("cannot fail as memory is contiguous");
            let freq = f64::from(freq);

            cfg_if::cfg_if! {
                if #[cfg(feature = "gpu-single")] {
                    self.cpu_object
                        .calc_jones_array_inner(&azels, freq, None, latitude_rad, &mut v)?;
                    a.iter_mut()
                        .zip(v.iter())
                        .for_each(|(a, v)| *a = Jones::<f32>::from(*v));
                } else {
                    self.cpu_object
                        .calc_jones_array_inner(&azels, freq, None, latitude_rad, a)?;
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
            a.as_ptr().cast(),
            a.len() * std::mem::size_of::<Jones<GpuFloat>>(),
            gpuMemcpyHostToDevice,
        );
        crate::gpu::check_for_errors(crate::gpu::GpuCall::CopyToDevice)?;

        Ok(())
    }

    fn get_beam_type(&self) -> BeamType {
        BeamType::SkaGaussian
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
    use super::*;
    use approx::assert_abs_diff_eq;
    use marlu::AzEl;

    #[test]
    fn test_gaussian_2d() {
        let std = 0.031618803234858744;
        let cent_l = 0.4252937845833011;
        let cent_m = -0.10576131883022044;
        let beam_l = 0.48339108;
        let beam_m = -0.22339675;
        let beam_real = SkaGaussianBeam::gaussian_2d(beam_l, beam_m, cent_l, cent_m, std);
        assert_abs_diff_eq!(beam_real, 0.00018248210368566883, epsilon = 1e-6);
    }

    #[test]
    fn test_gaussian_calc_jones_inner() {
        let freq_hz = 106000000.;
        let lst_rad = 5.769848203643869;
        let beam = SkaGaussianBeam;

        let azel = AzEl::from_radians(2.00370398, 1.00922628);
        let jones = beam.calc_jones(azel, freq_hz, None, lst_rad).unwrap();
        let expected = 0.00018248210368566883;
        assert_abs_diff_eq!(jones[0], Complex::new(expected, 0.0), epsilon = 1e-6);
        assert_abs_diff_eq!(jones[1], Complex::new(0.0, 0.0), epsilon = 1e-6);
        assert_abs_diff_eq!(jones[2], Complex::new(0.0, 0.0), epsilon = 1e-6);
        assert_abs_diff_eq!(jones[3], Complex::new(expected, 0.0), epsilon = 1e-6);
    }
}
