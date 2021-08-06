// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Structures to describe sky-model sources and their components.

use rayon::prelude::*;

use super::ComponentType;
use crate::{FluxDensity, FluxDensityType};
use mwa_hyperdrive_core::{RADec, LMN};

#[derive(Clone, Debug, PartialEq)]
/// A collection of components.
pub struct Source {
    /// The components associated with the source.
    pub components: Vec<SourceComponent>,
}

impl Source {
    /// Calculate the (l,m,n) coordinates of each component's (RA,Dec).
    pub fn get_lmn(&self, phase_centre: RADec) -> Vec<LMN> {
        self.components
            .iter()
            .map(|comp| comp.radec.to_lmn(phase_centre))
            .collect()
    }

    /// Calculate the (l,m,n) coordinates of each component's (RA,Dec). The
    /// calculation is done in parallel.
    pub fn get_lmn_parallel(&self, phase_centre: RADec) -> Vec<LMN> {
        self.components
            .par_iter()
            .map(|comp| comp.radec.to_lmn(phase_centre))
            .collect()
    }

    /// Estimate the flux densities for each of a source's components given a
    /// frequency.
    pub fn get_flux_estimates(&self, freq_hz: f64) -> Vec<FluxDensity> {
        self.components
            .iter()
            .map(|comp| comp.flux_type.estimate_at_freq(freq_hz))
            .collect()
    }

    /// Estimate the flux densities for each of a source's components given a
    /// frequency. The calculation is done in parallel.
    pub fn get_flux_estimates_parallel(&self, freq_hz: f64) -> Vec<FluxDensity> {
        self.components
            .par_iter()
            .map(|comp| comp.flux_type.estimate_at_freq(freq_hz))
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq)]
/// Information on a source's component.
pub struct SourceComponent {
    /// Coordinates struct associated with the component.
    pub radec: RADec,
    /// The type of component.
    pub comp_type: ComponentType,
    /// The flux densities associated with this component.
    pub flux_type: FluxDensityType,
}

impl SourceComponent {
    /// Estimate the flux density of this component at a frequency.
    pub fn estimate_at_freq(&self, freq_hz: f64) -> FluxDensity {
        self.flux_type.estimate_at_freq(freq_hz)
    }

    /// Is this component a point source?
    pub fn is_point(&self) -> bool {
        self.comp_type.is_point()
    }

    /// Is this component a gaussian source?
    pub fn is_gaussian(&self) -> bool {
        self.comp_type.is_shapelet()
    }

    /// Is this component a shapelet source?
    pub fn is_shapelet(&self) -> bool {
        self.comp_type.is_shapelet()
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::{FRAC_PI_4, PI};

    use approx::*;

    use super::*;
    use crate::{calc_flux_ratio, constants::SPEC_INDEX_CAP};
    use mwa_hyperdrive_core::RADec;

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

        let si = fds[0].calc_spec_index(&fds[1]);
        assert_abs_diff_eq!(si, -0.5503397132132084, epsilon = 1e-10);

        let si = fds[1].calc_spec_index(&fds[2]);
        assert_abs_diff_eq!(si, -1.0000000000000002, epsilon = 1e-10);

        let si = fds[0].calc_spec_index(&fds[2]);
        assert_abs_diff_eq!(si, -0.7369655941662062, epsilon = 1e-10);

        // Despite using the same two flux densities in the opposite way, the SI
        // comes out the same.
        let si = fds[2].calc_spec_index(&fds[0]);
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

        let si = fds[0].calc_spec_index(&fds[1]);
        assert_abs_diff_eq!(si, -3.321928094887362, epsilon = 1e-10);
    }

    #[test]
    #[should_panic]
    fn list_estimate_flux_density_at_freq_no_comps() {
        let mut source = get_list_source_1();
        // Delete all flux densities.
        match &mut source.components[0].flux_type {
            FluxDensityType::List { fds } => fds.drain(..),
            _ => unreachable!(),
        };
        let _fd = source.components[0].flux_type.estimate_at_freq(90.0);
    }

    #[test]
    fn list_estimate_flux_density_at_freq_extrapolation_single_comp1() {
        let mut source = get_list_source_1();
        match &mut source.components[0].flux_type {
            FluxDensityType::List { fds } => fds.drain(1..),
            _ => unreachable!(),
        };

        let fd = source.components[0].flux_type.estimate_at_freq(90.0);
        assert_abs_diff_eq!(fd.i, 10.879426248455298, epsilon = 1e-10);
        assert_abs_diff_eq!(fd.q, 7.615598373918708, epsilon = 1e-10);
        assert_abs_diff_eq!(fd.u, 6.527655749073178, epsilon = 1e-10);
        assert_abs_diff_eq!(fd.v, 1.0879426248455297, epsilon = 1e-10);

        let fd = source.components[0].flux_type.estimate_at_freq(110.0);
        assert_abs_diff_eq!(fd.i, 9.265862513558696, epsilon = 1e-10);
        assert_abs_diff_eq!(fd.q, 6.486103759491087, epsilon = 1e-10);
        assert_abs_diff_eq!(fd.u, 5.559517508135217, epsilon = 1e-10);
        assert_abs_diff_eq!(fd.v, 0.9265862513558696, epsilon = 1e-10);
    }

    #[test]
    fn list_estimate_flux_density_at_freq_extrapolation_source_1() {
        let source = get_list_source_1();
        let fd = source.components[0].flux_type.estimate_at_freq(90.0);
        assert_abs_diff_eq!(fd.i, 10.596981209124532, epsilon = 1e-10);
        assert_abs_diff_eq!(fd.q, 7.417886846387172, epsilon = 1e-10);
        assert_abs_diff_eq!(fd.u, 6.358188725474719, epsilon = 1e-10);
        assert_abs_diff_eq!(fd.v, 1.0596981209124532, epsilon = 1e-10);

        let fd = source.components[0].flux_type.estimate_at_freq(210.0);
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
        let freq_ratio_capped = calc_flux_ratio(desired_freq, fds[1].freq, SPEC_INDEX_CAP);
        let expected = fds[1].clone() * freq_ratio_capped;
        assert_abs_diff_eq!(fd.i, expected.i, epsilon = 1e-10);
        assert_abs_diff_eq!(fd.q, expected.q, epsilon = 1e-10);
        assert_abs_diff_eq!(fd.u, expected.u, epsilon = 1e-10);
        assert_abs_diff_eq!(fd.v, expected.v, epsilon = 1e-10);

        // Calculate the estimated flux density manually.
        // si should be -3.321928094887362 (this is also tested above).
        let si = fds[0].calc_spec_index(&fds[1]);
        assert_abs_diff_eq!(si, -3.321928094887362, epsilon = 1e-10);
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
    #[should_panic]
    fn unsorted_list_error() {
        let mut source = get_list_source_2();
        match &mut source.components[0].flux_type {
            FluxDensityType::List { fds } => {
                fds.reverse();
            }
            _ => unreachable!(),
        };
        let _fd = source.components[0].flux_type.estimate_at_freq(210.0);
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
        let new_fd = source.components[0].flux_type.estimate_at_freq(200.0);
        assert_abs_diff_eq!(new_fd.i, 8.414663590846496, epsilon = 1e-10);
        assert_abs_diff_eq!(new_fd.q, 5.890264513592547, epsilon = 1e-10);
        assert_abs_diff_eq!(new_fd.u, 5.048798154507898, epsilon = 1e-10);
        assert_abs_diff_eq!(new_fd.v, 0.8414663590846496, epsilon = 1e-10);

        let new_fd = source.components[0].flux_type.estimate_at_freq(100.0);
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
        let new_fd = source.components[0].flux_type.estimate_at_freq(200.0);
        assert_abs_diff_eq!(new_fd.i, 8.770171284543176, epsilon = 1e-10);
        assert_abs_diff_eq!(new_fd.q, 6.139119899180223, epsilon = 1e-10);
        assert_abs_diff_eq!(new_fd.u, 5.262102770725906, epsilon = 1e-10);
        assert_abs_diff_eq!(new_fd.v, 0.8770171284543176, epsilon = 1e-10);

        let new_fd = source.components[0].flux_type.estimate_at_freq(100.0);
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
        let norm_power_law = source.components[0].flux_type.estimate_at_freq(200.0);
        assert_abs_diff_ne!(new_fd.i, norm_power_law.i);
        assert_abs_diff_ne!(new_fd.q, norm_power_law.q);
        assert_abs_diff_ne!(new_fd.u, norm_power_law.u);
        assert_abs_diff_ne!(new_fd.v, norm_power_law.v);
    }
}
