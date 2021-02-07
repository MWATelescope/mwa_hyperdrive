// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Flux density structures.
 */

use log::{debug, trace};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::constants::*;

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
/// At a frequency, four flux densities for each Stokes parameter.
pub struct FluxDensity {
    /// The frequency that these flux densities apply to [Hz]
    pub freq: f64,
    /// The flux density of Stokes I [Jy]
    pub i: f64,
    /// The flux density of Stokes Q [Jy]
    pub q: f64,
    /// The flux density of Stokes U [Jy]
    pub u: f64,
    /// The flux density of Stokes V [Jy]
    pub v: f64,
}

impl std::ops::Mul<f64> for FluxDensity {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self {
        FluxDensity {
            freq: self.freq,
            i: self.i * rhs,
            q: self.q * rhs,
            u: self.u * rhs,
            v: self.v * rhs,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum FluxDensityType {
    /// A list of flux densities specified at multiple frequencies.
    /// Interpolation/extrapolation is needed to get flux densities at
    /// non-specified frequencies.
    List { fds: Vec<FluxDensity> },

    /// $S_\nu = a \nu^{\alpha}$
    PowerLaw {
        /// Spectral index (alpha)
        si: f64,
        /// Flux density (a)
        fd: FluxDensity,
    },

    /// Similar to a power law. See Callingham et al. 2017, section 4.1.
    ///
    /// S_\nu = a \nu^{\alpha} e^{q(\ln{v})^2}
    CurvedPowerLaw {
        /// Spectral index (alpha)
        si: f64,
        /// Flux density (a)
        fd: FluxDensity,
        /// Spectral curvature (q)
        q: f64,
    },
}

/// Given two Stokes I flux densities and their frequencies, calculate the
/// spectral index that fits them. The first argument is expected to be the flux
/// density with the smaller frequency.
fn calc_spec_index(fd1: &FluxDensity, fd2: &FluxDensity) -> f64 {
    (fd2.i / fd1.i).ln() / (fd2.freq / fd1.freq).ln()
}

/// Given a spectral index, determine the flux-density ratio of two frequencies.
fn calc_flux_ratio(desired_freq_hz: f64, cat_freq_hz: f64, spec_index: f64) -> f64 {
    (desired_freq_hz / cat_freq_hz).powf(spec_index)
}

impl FluxDensityType {
    /// Given flux density information, estimate the flux density at a
    /// particular frequency.
    ///
    /// If enum variant is FluxDensityType::List, then the entries must be
    /// sorted by frequency (which should be the case if the source list was
    /// read by hyperdrive).
    ///
    /// The estimated flux density is based off of the Stokes I component, so
    /// any other Stokes parameters may be poorly estimated.
    pub fn estimate_at_freq(&self, freq_hz: f64) -> Result<FluxDensity, EstimateError> {
        match self {
            FluxDensityType::List { fds } => {
                if fds.is_empty() {
                    return Err(EstimateError::NoFluxDensities);
                }

                let mut old_freq = -1.0;

                // `smaller_flux_density` is a bad name given to the component's flux
                // density corresponding to a frequency smaller than but nearest to the
                // specified frequency.
                let (spec_index, smaller_flux_density) = {
                    // If there's only one source component, then we must assume the
                    // spectral index for extrapolation.
                    if fds.len() == 1 {
                        debug!("Only one flux density in a component's list; extrapolating with spectral index {}", DEFAULT_SPEC_INDEX);
                        (DEFAULT_SPEC_INDEX, &fds[0])
                    }
                    // Otherwise, find the frequencies that bound the given frequency. As we
                    // assume that the input source components are sorted by frequency, we
                    // can assume that the comp. with the smallest frequency is at index 0,
                    // and similarly for the last component.
                    else {
                        let mut smaller_comp_index: usize = 0;
                        let mut larger_comp_index: usize = fds.len() - 1;
                        for (i, fd) in fds.iter().enumerate() {
                            let f = fd.freq;
                            // Bail if this frequency is smaller than the old
                            // frequency; we require the list of flux densities
                            // to be sorted by frequency.
                            if f < old_freq {
                                return Err(EstimateError::FluxDensityListNotSorted(fds.clone()));
                            }
                            old_freq = f;

                            // Iterate until we hit a catalogue frequency bigger than the
                            // desired frequency.

                            // If this freq and the specified freq are the same...
                            if (f - freq_hz).abs() < 1e-3 {
                                // ... then just return the flux density information from
                                // this frequency.
                                return Ok(*fd);
                            }
                            // If this freq is smaller than the specified freq...
                            else if f < freq_hz {
                                // Check if we've iterated to the last catalogue component -
                                // if so, then we must extrapolate (i.e. the specified
                                // freq. is bigger than all catalogue frequencies).
                                if i == fds.len() - 1 {
                                    smaller_comp_index = fds.len() - 2;
                                }
                            }
                            // We only arrive here if f > freq.
                            else {
                                // Because the array is sorted, we now know the closest two
                                // frequencies. The only exception is if this is the first
                                // catalogue frequency (i.e. the desired freq is smaller
                                // than all catalogue freqs -> extrapolate).
                                if i == 0 {
                                    larger_comp_index = 1;
                                } else {
                                    smaller_comp_index = i - 1;
                                }
                                break;
                            }
                        }

                        let mut spec_index =
                            calc_spec_index(&fds[smaller_comp_index], &fds[larger_comp_index]);

                        // Stop stupid spectral indices.
                        if spec_index < SPEC_INDEX_CAP {
                            trace!(
                                "Component had a spectral index {}; capping at {}",
                                spec_index,
                                SPEC_INDEX_CAP
                            );
                            spec_index = SPEC_INDEX_CAP;
                        }

                        (
                            spec_index,
                            // If our last component's frequency is smaller than the specified
                            // freq., then we should use that for flux densities.
                            if fds[larger_comp_index].freq < freq_hz {
                                &fds[larger_comp_index]
                            } else {
                                &fds[smaller_comp_index]
                            },
                        )
                    }
                };

                // Now scale the flux densities given the calculated spectral index.
                let flux_ratio = calc_flux_ratio(freq_hz, smaller_flux_density.freq, spec_index);

                Ok(FluxDensity {
                    freq: freq_hz,
                    ..*smaller_flux_density
                } * flux_ratio)
            }

            FluxDensityType::PowerLaw { si, fd } => {
                let ratio = calc_flux_ratio(freq_hz, fd.freq, *si);
                let mut new_fd = *fd * ratio;
                new_fd.freq = freq_hz;
                Ok(new_fd)
            }

            FluxDensityType::CurvedPowerLaw { si, fd, q } => {
                let mut power_law_component = *fd * calc_flux_ratio(freq_hz, fd.freq, *si);
                power_law_component.freq = freq_hz;
                let curved_component = (q * (freq_hz / fd.freq).ln().powi(2)).exp();
                Ok(power_law_component * curved_component)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::{FRAC_PI_4, PI};

    use approx::*;

    use super::*;
    use crate::coord::RADec;
    use crate::source::*;

    #[test]
    fn calc_freq_ratio_1() {
        let desired_freq = 160.0;
        let cat_freq = 150.0;
        let spec_index = -0.6;
        let ratio = calc_flux_ratio(desired_freq, cat_freq, spec_index);
        let expected = 0.9620170425907598;
        assert_abs_diff_eq!(ratio, expected, epsilon = 1e-10);
    }

    #[test]
    fn calc_freq_ratio_2() {
        let desired_freq = 140.0;
        let cat_freq = 150.0;
        let spec_index = -0.6;
        let ratio = calc_flux_ratio(desired_freq, cat_freq, spec_index);
        let expected = 1.0422644718599143;
        assert_abs_diff_eq!(ratio, expected, epsilon = 1e-10);
    }

    fn get_list_source_1() -> Source {
        Source {
            components: vec![SourceComponent {
                radec: RADec::new(PI, FRAC_PI_4),
                comp_type: ComponentType::Point,
                flux_type: FluxDensityType::List {
                    fds: vec![
                        FluxDensity {
                            freq: 100.0,
                            i: 10.0,
                            q: 7.0,
                            u: 6.0,
                            v: 1.0,
                        },
                        FluxDensity {
                            freq: 150.0,
                            i: 8.0,
                            q: 5.0,
                            u: 4.0,
                            v: 0.25,
                        },
                        FluxDensity {
                            freq: 200.0,
                            i: 6.0,
                            q: 4.0,
                            u: 3.0,
                            v: 0.1,
                        },
                    ],
                },
            }],
        }
    }

    // TODO: Write more tests using this source.
    // This is an extreme example; particularly useful for verifying a SI cap.
    fn get_list_source_2() -> Source {
        Source {
            components: vec![SourceComponent {
                radec: RADec::new(PI - 0.82, -FRAC_PI_4),
                comp_type: ComponentType::Point,
                flux_type: FluxDensityType::List {
                    fds: vec![
                        FluxDensity {
                            freq: 100.0,
                            i: 10.0,
                            q: 7.0,
                            u: 6.0,
                            v: 1.0,
                        },
                        FluxDensity {
                            freq: 200.0,
                            i: 1.0,
                            q: 2.0,
                            u: 3.0,
                            v: 4.1,
                        },
                    ],
                },
            }],
        }
    }

    #[test]
    fn list_calc_spec_index_source_1() {
        let source = get_list_source_1();
        let fds = match &source.components[0].flux_type {
            FluxDensityType::List { fds } => fds,
            _ => unreachable!(),
        };

        let si = calc_spec_index(&fds[0], &fds[1]);
        let expected = -0.5503397132132084;
        assert_abs_diff_eq!(si, expected, epsilon = 1e-10);

        let si = calc_spec_index(&fds[1], &fds[2]);
        let expected = -1.0000000000000002;
        assert_abs_diff_eq!(si, expected, epsilon = 1e-10);

        // This is an invalid usage of the `calc_spec_index`, but the invalidity
        // is not checked.
        let si = calc_spec_index(&fds[2], &fds[0]);
        let expected = -0.7369655941662062;
        assert_abs_diff_eq!(si, expected, epsilon = 1e-10);
    }

    #[test]
    fn list_calc_spec_index_source_2() {
        let source = get_list_source_2();
        let fds = match &source.components[0].flux_type {
            FluxDensityType::List { fds } => fds,
            _ => unreachable!(),
        };

        let si = calc_spec_index(&fds[0], &fds[1]);
        let expected = -3.321928094887362;
        assert_abs_diff_eq!(si, expected, epsilon = 1e-10);
    }

    #[test]
    fn list_estimate_flux_density_at_freq_no_comps() {
        let mut source = get_list_source_1();
        // Delete all flux densities.
        match &mut source.components[0].flux_type {
            FluxDensityType::List { fds } => fds.drain(..),
            _ => unreachable!(),
        };

        let fd = &source.components[0].flux_type.estimate_at_freq(90.0);
        assert!(fd.is_err());
    }

    #[test]
    fn list_estimate_flux_density_at_freq_extrapolation_single_comp1() {
        let mut source = get_list_source_1();
        match &mut source.components[0].flux_type {
            FluxDensityType::List { fds } => fds.drain(1..),
            _ => unreachable!(),
        };

        let fd = source.components[0].flux_type.estimate_at_freq(90.0);
        assert!(fd.is_ok());
        let fd = fd.unwrap();
        assert_abs_diff_eq!(fd.i, 10.879426248455298, epsilon = 1e-10);
        assert_abs_diff_eq!(fd.q, 7.615598373918708, epsilon = 1e-10);
        assert_abs_diff_eq!(fd.u, 6.527655749073178, epsilon = 1e-10);
        assert_abs_diff_eq!(fd.v, 1.0879426248455297, epsilon = 1e-10);

        let fd = source.components[0].flux_type.estimate_at_freq(110.0);
        assert!(fd.is_ok());
        let fd = fd.unwrap();
        assert_abs_diff_eq!(fd.i, 9.265862513558696, epsilon = 1e-10);
        assert_abs_diff_eq!(fd.q, 6.486103759491087, epsilon = 1e-10);
        assert_abs_diff_eq!(fd.u, 5.559517508135217, epsilon = 1e-10);
        assert_abs_diff_eq!(fd.v, 0.9265862513558696, epsilon = 1e-10);
    }

    #[test]
    fn list_estimate_flux_density_at_freq_extrapolation_source_1() {
        let source = get_list_source_1();
        let fd = source.components[0].flux_type.estimate_at_freq(90.0);
        assert!(fd.is_ok());
        let fd = fd.unwrap();
        assert_abs_diff_eq!(fd.i, 10.596981209124532, epsilon = 1e-10);
        assert_abs_diff_eq!(fd.q, 7.417886846387172, epsilon = 1e-10);
        assert_abs_diff_eq!(fd.u, 6.358188725474719, epsilon = 1e-10);
        assert_abs_diff_eq!(fd.v, 1.0596981209124532, epsilon = 1e-10);

        let fd = source.components[0].flux_type.estimate_at_freq(210.0);
        assert!(fd.is_ok());
        let fd = fd.unwrap();
        assert_abs_diff_eq!(fd.i, 5.7142857142857135, epsilon = 1e-10);
        assert_abs_diff_eq!(fd.q, 3.8095238095238093, epsilon = 1e-10);
        assert_abs_diff_eq!(fd.u, 2.8571428571428568, epsilon = 1e-10);
        assert_abs_diff_eq!(fd.v, 0.09523809523809523, epsilon = 1e-10);
    }

    #[test]
    fn list_estimate_flux_density_at_freq_source_2() {
        // Verify that the spectral index is capped if the calculated value
        // is below SPEC_INDEX_CAP. Use a freq. of 210 MHz.
        let source = get_list_source_2();
        let fds = match &source.components[0].flux_type {
            FluxDensityType::List { fds } => fds,
            _ => unreachable!(),
        };

        let desired_freq = 210.0;
        let fd = source.components[0]
            .flux_type
            .estimate_at_freq(desired_freq);
        assert!(fd.is_ok());
        let fd = fd.unwrap();
        let freq_ratio_capped = calc_flux_ratio(desired_freq, fds[1].freq, SPEC_INDEX_CAP);
        let expected = fds[1].clone() * freq_ratio_capped;
        assert_abs_diff_eq!(fd.i, expected.i, epsilon = 1e-10);
        assert_abs_diff_eq!(fd.q, expected.q, epsilon = 1e-10);
        assert_abs_diff_eq!(fd.u, expected.u, epsilon = 1e-10);
        assert_abs_diff_eq!(fd.v, expected.v, epsilon = 1e-10);

        // Calculate the estimated flux density manually.
        // si should be -3.321928094887362 (this is also tested above).
        let si = calc_spec_index(&fds[0], &fds[1]);
        let expected_si = -3.321928094887362;
        assert_abs_diff_eq!(si, expected_si, epsilon = 1e-10);
        let freq_ratio = calc_flux_ratio(desired_freq, fds[1].freq, si);
        let manual = fds[1].clone() * freq_ratio;

        // Compare our manually-calculated values with the output of the
        // estimation function. Our manually-calculated flux densities should be
        // smaller than those of the function, as our SI has not been capped.
        assert!(
            fd.i > manual.i,
            "Stokes I FD ({}) was smaller than expected!",
            fd.i,
        );
        assert!(
            fd.q > manual.q,
            "Stokes Q FD ({}) was smaller than expected!",
            fd.q,
        );
        assert!(
            fd.u > manual.u,
            "Stokes U FD ({}) was smaller than expected!",
            fd.u,
        );
        assert!(
            fd.v > manual.v,
            "Stokes V FD ({}) was smaller than expected!",
            fd.v,
        );
    }

    #[test]
    fn unsorted_list_error() {
        let mut source = get_list_source_2();
        match &mut source.components[0].flux_type {
            FluxDensityType::List { fds } => {
                fds.reverse();
            }
            _ => unreachable!(),
        };

        let result = source.components[0].flux_type.estimate_at_freq(210.0);
        assert!(result.is_err());
        assert_eq!(
            std::mem::discriminant(&result.unwrap_err()),
            std::mem::discriminant(&EstimateError::FluxDensityListNotSorted(vec![]))
        );
    }

    #[test]
    fn test_power_law() {
        let mut source = get_list_source_1();
        source.components[0].flux_type = FluxDensityType::PowerLaw {
            si: -0.6,
            fd: FluxDensity {
                freq: 150.0,
                i: 10.0,
                q: 7.0,
                u: 6.0,
                v: 1.0,
            },
        };
        let result = source.components[0].flux_type.estimate_at_freq(200.0);
        assert!(result.is_ok());
        let new_fd = result.unwrap();
        assert_abs_diff_eq!(new_fd.i, 8.414663590846496, epsilon = 1e-10);
        assert_abs_diff_eq!(new_fd.q, 5.890264513592547, epsilon = 1e-10);
        assert_abs_diff_eq!(new_fd.u, 5.048798154507898, epsilon = 1e-10);
        assert_abs_diff_eq!(new_fd.v, 0.8414663590846496, epsilon = 1e-10);

        let result = source.components[0].flux_type.estimate_at_freq(100.0);
        assert!(result.is_ok());
        let new_fd = result.unwrap();
        assert_abs_diff_eq!(new_fd.i, 12.754245006257909, epsilon = 1e-10);
        assert_abs_diff_eq!(new_fd.q, 8.927971504380537, epsilon = 1e-10);
        assert_abs_diff_eq!(new_fd.u, 7.652547003754746, epsilon = 1e-10);
        assert_abs_diff_eq!(new_fd.v, 1.275424500625791, epsilon = 1e-10);
    }

    #[test]
    fn test_curved_power_law() {
        let mut source = get_list_source_1();
        source.components[0].flux_type = FluxDensityType::CurvedPowerLaw {
            si: -0.6,
            fd: FluxDensity {
                freq: 150.0,
                i: 10.0,
                q: 7.0,
                u: 6.0,
                v: 1.0,
            },
            q: 0.5,
        };
        let result = source.components[0].flux_type.estimate_at_freq(200.0);
        assert!(result.is_ok());
        let new_fd = result.unwrap();
        assert_abs_diff_eq!(new_fd.i, 8.770171284543176, epsilon = 1e-10);
        assert_abs_diff_eq!(new_fd.q, 6.139119899180223, epsilon = 1e-10);
        assert_abs_diff_eq!(new_fd.u, 5.262102770725906, epsilon = 1e-10);
        assert_abs_diff_eq!(new_fd.v, 0.8770171284543176, epsilon = 1e-10);

        let result = source.components[0].flux_type.estimate_at_freq(100.0);
        assert!(result.is_ok());
        let new_fd = result.unwrap();
        assert_abs_diff_eq!(new_fd.i, 13.846951980528223, epsilon = 1e-10);
        assert_abs_diff_eq!(new_fd.q, 9.692866386369756, epsilon = 1e-10);
        assert_abs_diff_eq!(new_fd.u, 8.308171188316935, epsilon = 1e-10);
        assert_abs_diff_eq!(new_fd.v, 1.3846951980528224, epsilon = 1e-10);

        // Just ensure that the results coming out of a curved power law don't
        // match those of a normal power law.
        source.components[0].flux_type = FluxDensityType::PowerLaw {
            si: -0.6,
            fd: FluxDensity {
                freq: 150.0,
                i: 10.0,
                q: 7.0,
                u: 6.0,
                v: 1.0,
            },
        };
        let result = source.components[0].flux_type.estimate_at_freq(200.0);
        assert!(result.is_ok());
        let norm_power_law = result.unwrap();
        assert_abs_diff_ne!(new_fd.i, norm_power_law.i);
        assert_abs_diff_ne!(new_fd.q, norm_power_law.q);
        assert_abs_diff_ne!(new_fd.u, norm_power_law.u);
        assert_abs_diff_ne!(new_fd.v, norm_power_law.v);
    }
}

#[derive(Error, Debug, PartialEq)]
pub enum EstimateError {
    #[error("Tried to estimate a flux density for a component, but it had no flux densities")]
    NoFluxDensities,

    #[error("The list of flux densities used for estimation were not sorted:\n{0:#?}")]
    FluxDensityListNotSorted(Vec<FluxDensity>),
}
