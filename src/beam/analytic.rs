//! Code for Analytic beam calculations.

use super::{Beam, BeamError, BeamType};
// use rayon::iter::{IntoParallelRefIterator, IndexedParallelIterator, IntoParallelRefMutIterator, P};
#[cfg(feature = "cuda")]
use super::BeamCUDA;
use marlu::{AzEl, Jones, RADec};
use ndarray::prelude::*;
use num_complex::Complex64;
use rayon::prelude::*;

pub struct AnalyticBeam {
    pub num_stations: usize,
    pub fwhm_rad: f64,
    pub ref_freq_hz: f64,
    pub phase_centre: RADec,
    // TODO: check if code actually uses radians
    pub latitude_rad: f64,
}

/// Analytic Beam implementation.
impl AnalyticBeam {

    /// Explicitly a 2D gaussian function
    #[allow(clippy::too_many_arguments)]
    fn gaussian_2d(
        x: f64,
        y: f64,
        xo: f64,
        yo: f64,
        sigma_x: f64,
        sigma_y: f64,
        cos_theta: f64,
        sin_theta: f64,
        sin_2theta: f64,
    ) -> f64 {
        let a = (cos_theta * cos_theta) / (2. * sigma_x * sigma_x)
            + (sin_theta * sin_theta) / (2. * sigma_y * sigma_y);
        let b = -sin_2theta / (4. * sigma_x * sigma_x) + sin_2theta / (4. * sigma_y * sigma_y);
        let c = (sin_theta * sin_theta) / (2. * sigma_x * sigma_x)
            + (cos_theta * cos_theta) / (2. * sigma_y * sigma_y);

        (-(a * (x - xo) * (x - xo) + 2. * b * (x - xo) * (y - yo) + c * (y - yo) * (y - yo))).exp()
    }

    // #[allow(clippy::too_many_arguments)]
    // fn gaussian_2d_array(
    //     xs: &[f64],
    //     ys: &[f64],
    //     xo: f64,
    //     yo: f64,
    //     sigma_x: f64,
    //     sigma_y: f64,
    //     cos_theta: f64,
    //     sin_theta: f64,
    //     sin_2theta: f64,
    // ) -> Vec<f64> {
    //     let a = (cos_theta * cos_theta) / (2. * sigma_x * sigma_x)
    //         + (sin_theta * sin_theta) / (2. * sigma_y * sigma_y);
    //     let b = -sin_2theta / (4. * sigma_x * sigma_x) + sin_2theta / (4. * sigma_y * sigma_y);
    //     let c = (sin_theta * sin_theta) / (2. * sigma_x * sigma_x)
    //         + (cos_theta * cos_theta) / (2. * sigma_y * sigma_y);

    //     // ( -( a*(x-xo)*(x-xo) + 2.*b*(x-xo)*(y-yo) + c*(y-yo)*(y-yo) )).exp()
    //     todo!()
    // }

    fn calc_jones_inner(
        &self,
        azel: AzEl,
        freq_hz: f64,
        lst_rad: f64,
    ) -> Result<Jones<f64>, BeamError> {
        let beam_radec = azel.to_hadec(self.latitude_rad).to_radec(lst_rad);
        let zenith_radec = RADec::from_radians(lst_rad, self.latitude_rad);
        let beam_lmn = beam_radec.to_lmn(zenith_radec);
        let cent_lmn = self.phase_centre.to_lmn(zenith_radec);

        // these are related to position angle, which I'm setting to zero
        let cos_theta = 1.0;
        let sin_theta = 0.0;
        let sin_2theta = 0.0;

        // scale fwhm to be in l,m coords
        let fwhm_lm = self.fwhm_rad.sin();

        let fwhm_factor = 2.35482004503;
        let std = (fwhm_lm / fwhm_factor) * (self.ref_freq_hz / freq_hz);

        let beam_real = AnalyticBeam::gaussian_2d(
            beam_lmn.l, beam_lmn.m, cent_lmn.l, cent_lmn.m, //cent_l, cent_m,
            std, std, cos_theta, sin_theta, sin_2theta,
        );

        // TODO: what's the conversion between stokes I and XX again?
        Ok(Jones::from([
            Complex64::new(beam_real, 0.),
            Complex64::new(0., 0.),
            Complex64::new(0., 0.),
            Complex64::new(beam_real, 0.),
        ]))
    }

    // fn calc_jones_array(
    //     &self,
    //     azels: &[AzEl],
    //     freq_hz: f64,
    //     lst: f64,
    // ) -> Result<Vec<Jones<f64>>, BeamError> {

    // }
}

impl Beam for AnalyticBeam {
    fn get_beam_type(&self) -> BeamType {
        BeamType::Analytic
    }

    fn get_num_tiles(&self) -> usize {
        self.num_stations
    }

    fn get_dipole_gains(&self) -> Option<ArcArray<f64, Dim<[usize; 2]>>> {
        None
    }

    /// Warning: latitude_rad is used for LST.
    fn calc_jones(
        &self,
        azel: AzEl,
        freq_hz: f64,
        _tile_index: Option<usize>,
        latitude_rad: f64,
    ) -> Result<Jones<f64>, BeamError> {
        self.calc_jones_inner(azel, freq_hz, latitude_rad)
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
            .par_iter()
            .zip(results.par_iter_mut())
            .for_each(|(&azel, result)| {
                // TODO: WWCHJD?
                // NOTE: latitude_rad is actually LST.
                *result = self.calc_jones_inner(azel, freq_hz, latitude_rad).unwrap();
            });
        Ok(())
    }

    fn uses_lst(&self) -> bool {
        true
    }

    #[cfg(feature = "cuda")]
    fn prepare_cuda_beam(&self, _freqs_hz: &[u32]) -> Result<Box<dyn BeamCUDA>, BeamError> {
        unimplemented!()
    }
}

/// Create an analytic beam object
pub fn create_analytic_beam_object(
    num_stations: usize,
    ref_freq_hz: f64,
    fwhm_rad: f64,
    latitude_rad: f64,
    phase_centre: RADec,
) -> Box<dyn Beam> {
    Box::new(AnalyticBeam {
        num_stations,
        ref_freq_hz,
        fwhm_rad,
        latitude_rad,
        phase_centre,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use marlu::{AzEl, RADec};
    #[test]
    fn test_gaussian_2d() {
        let cos_theta = 1.0;
        let sin_theta = 0.0;
        let sin_2theta = 0.0;
        let std = 0.031618803234858744;
        let cent_l = 0.4252937845833011;
        let cent_m = -0.10576131883022044;
        let beam_l = 0.48339108;
        let beam_m = -0.22339675;
        let beam_real = AnalyticBeam::gaussian_2d(
            beam_l, beam_m, cent_l, cent_m, std, std, cos_theta, sin_theta, sin_2theta,
        );
        assert_abs_diff_eq!(beam_real, 0.00018248210368566883, epsilon = 1e-6);
    }

    #[test]
    fn test_analytic_calc_jones_inner() {
        let freq_hz = 106000000.;
        let beam = AnalyticBeam {
            num_stations: 1,
            ref_freq_hz: freq_hz,
            fwhm_rad: 4.27_f64.to_radians(),
            latitude_rad: -26.82472208_f64.to_radians(),
            phase_centre: RADec::from_radians(0.0_f64.to_radians(), -30.0_f64.to_radians()),
        };
        let azel = AzEl::from_radians(2.00370398, 1.00922628);
        let lst_rad = 5.769848203643869;
        let jones = beam.calc_jones_inner(azel, freq_hz, lst_rad).unwrap();
        let beam_real = 0.00018248210368566883;
        assert_abs_diff_eq!(jones[0], Complex64::new(beam_real, 0.0), epsilon = 1e-6);
        assert_abs_diff_eq!(jones[1], Complex64::new(0.0, 0.0), epsilon = 1e-6);
        assert_abs_diff_eq!(jones[2], Complex64::new(0.0, 0.0), epsilon = 1e-6);
        assert_abs_diff_eq!(jones[3], Complex64::new(beam_real, 0.0), epsilon = 1e-6);
    }
}
