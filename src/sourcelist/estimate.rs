// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use super::*;
use crate::constants::*;

/// Given two Stokes I flux densities and their frequencies, calculate the
/// spectral index that fits them. The first argument is expected to be the flux
/// density with the smaller frequency.
fn calc_spec_index(fd1: &FluxDensity, fd2: &FluxDensity) -> f64 {
    (fd2.i / fd1.i).ln() / (fd2.freq / fd1.freq).ln()
}

/// Given a spectral index, determine the flux-density ratio of two frequencies.
pub fn calc_flux_ratio(desired_freq: f64, cat_freq: f64, spec_index: f64) -> f64 {
    (desired_freq / cat_freq).powf(spec_index)
}

/// Estimate the flux density of a source at a particular frequency. The input
/// source components must be sorted by frequency.
///
/// The interpolated/extrapolated flux density is based off of the Stokes I of
/// the input source, so any other Stokes parameters may be poorly estimated.
pub fn estimate_flux_density_at_freq(
    comp: &SourceComponent,
    freq: f64,
) -> Result<FluxDensity, ErrorKind> {
    // fds == flux densities.
    let fds = &comp.flux_densities;
    if fds.is_empty() {
        return Err(ErrorKind::Insane(
            "estimate_flux_density_at_freq: Received no flux densities!".to_string(),
        ));
    }

    // `smaller_flux_density` is a bad name given to the component's flux
    // density corresponding to a frequency smaller than but nearest to the
    // specified frequency.
    let (spec_index, smaller_flux_density) = {
        // If there's only one source component, then we must assume the
        // spectral index for extrapolation.
        if fds.len() == 1 {
            // CHJ: Should we warn here? If so, it needs to be done better than
            // just printing to stderr.
            // eprintln!(
            //     "estimate_flux_density_at_freq: WARNING: Component has only one flux density; extrapolating with spectral index {}",
            //     *DEFAULT_SPEC_INDEX
            // );
            (*DEFAULT_SPEC_INDEX, &fds[0])
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

                // Iterate until we hit a catalogue frequency bigger than the
                // desired frequency.

                // If this freq and the specified freq are the same...
                if (f - freq).abs() < 1e-3 {
                    // ... then just return the flux density information from
                    // this frequency.
                    return Ok(*fd);
                }
                // If this freq is smaller than the specified freq...
                else if f < freq {
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

            let mut spec_index = calc_spec_index(&fds[smaller_comp_index], &fds[larger_comp_index]);

            // Stop stupid spectral indices.
            if spec_index < *SPEC_INDEX_CAP {
                eprintln!(
                    "estimate_flux_density_at_freq: WARNING: Component had a spectral index {}; capping at {}",
                    spec_index, *SPEC_INDEX_CAP
                );
                spec_index = *SPEC_INDEX_CAP;
            }

            (
                spec_index,
                // If our last component's frequency is smaller than the specified
                // freq., then we should use that for flux densities.
                if fds[larger_comp_index].freq < freq {
                    &fds[larger_comp_index]
                } else {
                    &fds[smaller_comp_index]
                },
            )
        }
    };

    // Now scale the flux densities given the calculated spectral index.
    let flux_ratio = calc_flux_ratio(freq, smaller_flux_density.freq, spec_index);

    Ok(FluxDensity {
        freq,
        ..*smaller_flux_density
    } * flux_ratio)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coord::types::RADec;
    use float_cmp::approx_eq;

    fn get_test_source_1() -> Source {
        Source {
            name: "test_source_1".to_string(),
            components: vec![SourceComponent {
                radec: RADec::new(std::f64::consts::PI, std::f64::consts::FRAC_PI_4),
                ctype: ComponentType::Point,
                flux_densities: vec![
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
            }],
        }
    }

    // TODO: Write more tests using this source.
    // This is an extreme example; particularly useful for verifying a SI cap.
    fn get_test_source_2() -> Source {
        Source {
            name: "test_source_2".to_string(),
            components: vec![SourceComponent {
                radec: RADec::new(std::f64::consts::PI - 0.82, -std::f64::consts::FRAC_PI_4),
                ctype: ComponentType::Point,
                flux_densities: vec![
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
            }],
        }
    }

    #[test]
    fn test_calc_spec_index_source_1() {
        let source = get_test_source_1();
        let si = calc_spec_index(
            &source.components[0].flux_densities[0],
            &source.components[0].flux_densities[1],
        );
        let expected = -0.5503397132132084;
        assert!(
            approx_eq!(f64, si, expected, epsilon = 1e-10),
            "Calculated SI ({}) did not match expected SI ({})!",
            si,
            expected
        );

        let si = calc_spec_index(
            &source.components[0].flux_densities[1],
            &source.components[0].flux_densities[2],
        );
        let expected = -1.0000000000000002;
        assert!(
            approx_eq!(f64, si, expected, epsilon = 1e-10),
            "Calculated SI ({}) did not match expected SI ({})!",
            si,
            expected
        );

        // This is an invalid usage of the `calc_spec_index`, but the invalidity
        // is not checked.
        let si = calc_spec_index(
            &source.components[0].flux_densities[2],
            &source.components[0].flux_densities[0],
        );
        let expected = -0.7369655941662062;
        assert!(
            approx_eq!(f64, si, expected, epsilon = 1e-10),
            "Calculated SI ({}) did not match expected SI ({})!",
            si,
            expected
        );
    }

    #[test]
    fn test_calc_spec_index_source_2() {
        let source = get_test_source_2();

        let si = calc_spec_index(
            &source.components[0].flux_densities[0],
            &source.components[0].flux_densities[1],
        );
        let expected = -3.321928094887362;
        assert!(
            approx_eq!(f64, si, expected, epsilon = 1e-10),
            "Calculated SI ({}) did not match expected SI ({})!",
            si,
            expected
        );
    }

    #[test]
    fn test_calc_freq_ratio_1() {
        let desired_freq = 160.;
        let cat_freq = 150.;
        let spec_index = -0.6;
        let ratio = calc_flux_ratio(desired_freq, cat_freq, spec_index);
        let expected = 0.9620170425907598;
        assert!(
            approx_eq!(f64, ratio, expected, epsilon = 1e-10),
            "Calculated freq. ratio ({}) did not match expected value ({})!",
            ratio,
            expected
        );
    }

    #[test]
    fn test_calc_freq_ratio_2() {
        let desired_freq = 140.;
        let cat_freq = 150.;
        let spec_index = -0.6;
        let ratio = calc_flux_ratio(desired_freq, cat_freq, spec_index);
        let expected = 1.0422644718599143;
        assert!(
            approx_eq!(f64, ratio, expected, epsilon = 1e-10),
            "Calculated freq. ratio ({}) did not match expected value ({})!",
            ratio,
            expected
        );
    }

    #[test]
    fn test_estimate_flux_density_at_freq_no_comps() {
        let mut source = get_test_source_1();
        // Delete all flux densities.
        source.components[0].flux_densities.remove(2);
        source.components[0].flux_densities.remove(1);
        source.components[0].flux_densities.remove(0);

        let fd = estimate_flux_density_at_freq(&source.components[0], 90.);
        assert!(fd.is_err());
    }

    #[test]
    fn test_estimate_flux_density_at_freq_extrapolation_single_comp1() {
        let mut source = get_test_source_1();
        // Delete all but the first flux density.
        source.components[0].flux_densities.remove(2);
        source.components[0].flux_densities.remove(1);

        let fd = estimate_flux_density_at_freq(&source.components[0], 90.);
        assert!(fd.is_ok());
        let fd = fd.unwrap();
        assert!(approx_eq!(f64, fd.i, 10.879426248455298, epsilon = 1e-10));
        assert!(approx_eq!(f64, fd.q, 7.615598373918708, epsilon = 1e-10));
        assert!(approx_eq!(f64, fd.u, 6.527655749073178, epsilon = 1e-10));
        assert!(approx_eq!(f64, fd.v, 1.0879426248455297, epsilon = 1e-10));

        let fd = estimate_flux_density_at_freq(&source.components[0], 110.);
        assert!(fd.is_ok());
        let fd = fd.unwrap();
        assert!(approx_eq!(f64, fd.i, 9.265862513558696, epsilon = 1e-10));
        assert!(approx_eq!(f64, fd.q, 6.486103759491087, epsilon = 1e-10));
        assert!(approx_eq!(f64, fd.u, 5.559517508135217, epsilon = 1e-10));
        assert!(approx_eq!(f64, fd.v, 0.9265862513558696, epsilon = 1e-10));
    }

    #[test]
    fn test_estimate_flux_density_at_freq_extrapolation_source_1() {
        let source = get_test_source_1();
        let fd = estimate_flux_density_at_freq(&source.components[0], 90.);
        assert!(fd.is_ok());
        let fd = fd.unwrap();
        assert!(approx_eq!(f64, fd.i, 10.596981209124532, epsilon = 1e-10));
        assert!(approx_eq!(f64, fd.q, 7.417886846387172, epsilon = 1e-10));
        assert!(approx_eq!(f64, fd.u, 6.358188725474719, epsilon = 1e-10));
        assert!(approx_eq!(f64, fd.v, 1.0596981209124532, epsilon = 1e-10));

        let fd = estimate_flux_density_at_freq(&source.components[0], 210.);
        assert!(fd.is_ok());
        let fd = fd.unwrap();
        assert!(approx_eq!(f64, fd.i, 5.7142857142857135, epsilon = 1e-10));
        assert!(approx_eq!(f64, fd.q, 3.8095238095238093, epsilon = 1e-10));
        assert!(approx_eq!(f64, fd.u, 2.8571428571428568, epsilon = 1e-10));
        assert!(approx_eq!(f64, fd.v, 0.09523809523809523, epsilon = 1e-10));
    }

    #[test]
    fn test_estimate_flux_density_at_freq_source_2() {
        // Verify that the spectral index is capped if the calculated value
        // is below SPEC_INDEX_CAP. Use a freq. of 210 MHz.
        let source = get_test_source_2();
        let desired_freq = 210.;
        let fd = estimate_flux_density_at_freq(&source.components[0], desired_freq);
        assert!(fd.is_ok());
        let fd = fd.unwrap();
        let freq_ratio_capped = calc_flux_ratio(
            desired_freq,
            source.components[0].flux_densities[1].freq,
            *SPEC_INDEX_CAP,
        );
        let expected = source.components[0].flux_densities[1].clone() * freq_ratio_capped;
        assert!(
            approx_eq!(f64, fd.i, expected.i, epsilon = 1e-10),
            "Stokes I FD ({}) did not match expected value ({})!",
            fd.i,
            expected.i,
        );
        assert!(
            approx_eq!(f64, fd.q, expected.q, epsilon = 1e-10),
            "Stokes Q FD ({}) did not match expected value ({})!",
            fd.i,
            expected.i,
        );
        assert!(
            approx_eq!(f64, fd.u, expected.u, epsilon = 1e-10),
            "Stokes U FD ({}) did not match expected value ({})!",
            fd.i,
            expected.i,
        );
        assert!(
            approx_eq!(f64, fd.v, expected.v, epsilon = 1e-10),
            "Stokes V FD ({}) did not match expected value ({})!",
            fd.i,
            expected.i,
        );

        // Calculate the estimated flux density manually.
        // si should be -3.321928094887362 (this is also tested above).
        let si = calc_spec_index(
            &source.components[0].flux_densities[0],
            &source.components[0].flux_densities[1],
        );
        let expected_si = -3.321928094887362;
        assert!(
            approx_eq!(f64, si, expected_si, epsilon = 1e-10),
            "Calculated SI ({}) did not match expected SI ({})!",
            si,
            expected_si,
        );
        let freq_ratio = calc_flux_ratio(
            desired_freq,
            source.components[0].flux_densities[1].freq,
            si,
        );
        let manual = source.components[0].flux_densities[1].clone() * freq_ratio;

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
}
