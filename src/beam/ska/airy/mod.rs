// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::f64::consts::{FRAC_PI_2, PI};

use marlu::{AzEl, Jones, RADec, LMN};
use ndarray::prelude::*;
use rayon::prelude::*;

use super::{NUM_STATIONS, PHASE_CENTRE, REF_FREQ_HZ, SKA_LATITUDE_RAD};
use crate::beam::{Beam, BeamError, BeamType};
#[cfg(any(feature = "cuda", feature = "hip"))]
use crate::beam::{BeamGpu, DevicePointer, GpuFloat};

include!("bindings.rs");

/// `scipy.special.jn_zeros(1, 1)[0] / np.pi`
const J_ZERO_THINGY: f64 = 1.2196698912665045;

lazy_static::lazy_static! {
    static ref AIRY_CONST: f64 = PI * J_ZERO_THINGY / (5.15_f64.to_radians() * REF_FREQ_HZ);
}

#[derive(Clone, Copy)]
pub(crate) struct SkaAiryBeam;

impl SkaAiryBeam {
    fn calc_jones_inner(
        azel: AzEl,
        freq_hz: f64,
        lst_rad: f64,
        zenith_radec: RADec,
        cent_l: f64,
        cent_m: f64,
    ) -> Jones<f64> {
        let hadec = azel.to_hadec(SKA_LATITUDE_RAD);
        let beam_radec = hadec.to_radec(lst_rad);
        let LMN {
            l: beam_l,
            m: beam_m,
            ..
        } = beam_radec.to_lmn(zenith_radec);

        let dist = ((beam_l - cent_l).powi(2) + (beam_m - cent_m).powi(2)).sqrt();

        // More explicit.
        // let radius = 5.15_f64.to_radians() * REF_FREQ_HZ / freq_hz;
        // let rt = dist / (radius / J_ZERO_THINGY) * PI;
        let rt = dist * freq_hz * *AIRY_CONST;

        let z = (2.0 * unsafe { j1(rt) } / rt).abs();

        Jones::from([z, 0.0, 0.0, 0.0, 0.0, 0.0, z, 0.0])
    }
}

impl Beam for SkaAiryBeam {
    fn get_beam_type(&self) -> BeamType {
        BeamType::SkaAiry
    }

    fn get_num_tiles(&self) -> usize {
        NUM_STATIONS
    }

    fn get_dipole_gains(&self) -> Option<ArcArray<f64, Dim<[usize; 2]>>> {
        None
    }

    /// Derived with help from Jack Line, Dev Null and
    /// <https://docs.astropy.org/en/stable/_modules/astropy/modeling/functional_models.html#AiryDisk2D.evaluate>
    fn calc_jones(
        &self,
        azel: AzEl,
        freq_hz: f64,
        _tile_index: Option<usize>,
        lst_rad: f64,
    ) -> Result<Jones<f64>, BeamError> {
        let zenith_radec = RADec::from_radians(lst_rad, SKA_LATITUDE_RAD);
        let LMN {
            l: cent_l,
            m: cent_m,
            ..
        } = PHASE_CENTRE.to_lmn(zenith_radec);

        Ok(SkaAiryBeam::calc_jones_inner(
            azel,
            freq_hz,
            lst_rad,
            zenith_radec,
            cent_l,
            cent_m,
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
        lst_rad: f64,
        results: &mut [Jones<f64>],
    ) -> Result<(), BeamError> {
        let zenith_radec = RADec::from_radians(lst_rad, SKA_LATITUDE_RAD);
        let LMN {
            l: cent_l,
            m: cent_m,
            ..
        } = PHASE_CENTRE.to_lmn(zenith_radec);

        azels
            .par_iter()
            .zip(results.par_iter_mut())
            .for_each(|(&azel, result)| {
                *result = SkaAiryBeam::calc_jones_inner(
                    azel,
                    freq_hz,
                    lst_rad,
                    zenith_radec,
                    cent_l,
                    cent_m,
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
        let obj = SkaAiryBeamGpu {
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
pub(crate) struct SkaAiryBeamGpu {
    cpu_object: SkaAiryBeam,
    freqs_hz: Vec<u32>,
    tile_map: DevicePointer<i32>,
    freq_map: DevicePointer<i32>,
}

#[cfg(any(feature = "cuda", feature = "hip"))]
impl BeamGpu for SkaAiryBeamGpu {
    unsafe fn calc_jones_pair(
        &self,
        az_rad: &[GpuFloat],
        za_rad: &[GpuFloat],
        latitude_rad: f64,
        d_jones: *mut std::ffi::c_void,
    ) -> Result<(), BeamError> {
        let lst_rad = latitude_rad;

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

        let mut a: Array2<Jones<GpuFloat>> = Array2::zeros((self.freqs_hz.len(), az_rad.len()));
        #[cfg(feature = "gpu-single")]
        let mut v = vec![Jones::default(); az_rad.len()];
        for (mut a, &freq) in a.outer_iter_mut().zip(self.freqs_hz.iter()) {
            let freq = f64::from(freq);

            cfg_if::cfg_if! {
                if #[cfg(feature = "gpu-single")] {
                    self.cpu_object
                        .calc_jones_array_inner(&azels, freq, None, lst_rad, &mut v)?;
                    a.iter_mut()
                        .zip(v.iter())
                        .for_each(|(a, v)| *a = Jones::<f32>::from(*v));
                } else {
                    let a = a
                        .as_slice_mut()
                        .expect("cannot fail as memory is contiguous");
                    self.cpu_object
                        .calc_jones_array_inner(&azels, freq, None, lst_rad, a)?;
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
        BeamType::SkaAiry
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
    fn test_airy_calc_jones_inner() {
        let freq_hz = 106000000.;
        let lst_rad = 5.769848203643869;
        let beam = SkaAiryBeam;

        let azel = AzEl::from_radians(2.00370398, 1.00922628);
        let jones = beam.calc_jones(azel, freq_hz, None, lst_rad).unwrap();
        let expected = 0.0143450805168023_f64.sqrt();
        assert_abs_diff_eq!(jones[0].re, expected, epsilon = 1e-6);
        assert_abs_diff_eq!(jones[0].im, 0.0);
        assert_abs_diff_eq!(jones[1].re, 0.0);
        assert_abs_diff_eq!(jones[1].im, 0.0);
        assert_abs_diff_eq!(jones[2].re, 0.0);
        assert_abs_diff_eq!(jones[2].im, 0.0);
        assert_abs_diff_eq!(jones[3].re, expected, epsilon = 1e-6);
        assert_abs_diff_eq!(jones[3].im, 0.0);

        let azel = AzEl::from_radians(0.1, 0.1);
        let jones = beam.calc_jones(azel, freq_hz, None, lst_rad).unwrap();
        let expected = 1.20727184e-5_f64.sqrt();
        assert_abs_diff_eq!(jones[0].re, expected, epsilon = 1e-6);
        assert_abs_diff_eq!(jones[0].im, 0.0);
        assert_abs_diff_eq!(jones[1].re, 0.0);
        assert_abs_diff_eq!(jones[1].im, 0.0);
        assert_abs_diff_eq!(jones[2].re, 0.0);
        assert_abs_diff_eq!(jones[2].im, 0.0);
        assert_abs_diff_eq!(jones[3].re, expected, epsilon = 1e-6);
        assert_abs_diff_eq!(jones[3].im, 0.0);
    }
}
