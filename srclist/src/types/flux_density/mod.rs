// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Flux density structures.

#[cfg(test)]
mod tests;

use marlu::Jones;
use serde::{Deserialize, Serialize};
use vec1::Vec1;

use crate::constants::*;
use mwa_hyperdrive_common::{marlu, vec1, Complex};

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
/// At a frequency, four flux densities for each Stokes parameter.
// When serialising/deserialising, ignore Stokes Q U V if they are zero.
pub struct FluxDensity {
    /// The frequency that these flux densities apply to \[Hz\]
    pub freq: f64,

    /// The flux density of Stokes I \[Jy\]
    pub i: f64,

    /// The flux density of Stokes Q \[Jy\]
    #[serde(default)]
    #[serde(skip_serializing_if = "is_zero")]
    pub q: f64,

    /// The flux density of Stokes U \[Jy\]
    #[serde(default)]
    #[serde(skip_serializing_if = "is_zero")]
    pub u: f64,

    /// The flux density of Stokes V \[Jy\]
    #[serde(default)]
    #[serde(skip_serializing_if = "is_zero")]
    pub v: f64,
}

impl FluxDensity {
    /// Given two flux densities, calculate the spectral index that fits them.
    /// Uses only Stokes I.
    pub fn calc_spec_index(&self, fd2: &Self) -> f64 {
        (fd2.i.abs() / self.i.abs()).ln() / (fd2.freq / self.freq).ln()
    }

    /// Convert a `FluxDensity` into a [Jones] matrix representing instrumental
    /// Stokes (i.e. XX, XY, YX, YY).
    pub fn to_inst_stokes(&self) -> Jones<f64> {
        Jones::from([
            Complex::new(self.i + self.q, 0.0),
            Complex::new(self.u, self.v),
            Complex::new(self.u, -self.v),
            Complex::new(self.i - self.q, 0.0),
        ])
    }
}

/// This is only used for serialisation
// https://stackoverflow.com/questions/53900612/how-do-i-avoid-generating-json-when-serializing-a-value-that-is-null-or-a-defaul
#[allow(clippy::trivially_copy_pass_by_ref)]
fn is_zero(num: &f64) -> bool {
    num.abs() < f64::EPSILON
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

    /// A list of flux densities specified at multiple frequencies.
    /// Interpolation/extrapolation is needed to get flux densities at
    /// non-specified frequencies.
    List { fds: Vec1<FluxDensity> },
}

impl FluxDensityType {
    /// Given flux density information, estimate the flux density at a
    /// particular frequency. For power laws / curved power laws, the "ratio" of
    /// the reference frequency and the specified frequencies is used to scale
    /// the reference flux density.
    ///
    /// If enum variant is FluxDensityType::List, then the entries must be
    /// sorted by frequency (which should be the case if the source list was
    /// read by hyperdrive). The estimated flux density is based off of the
    /// Stokes I component, so any other Stokes parameters may be poorly
    /// estimated.
    pub fn estimate_at_freq(&self, freq_hz: f64) -> FluxDensity {
        match self {
            FluxDensityType::PowerLaw { si, fd } => {
                let ratio = calc_flux_ratio(freq_hz, fd.freq, *si);
                let mut new_fd = fd.clone() * ratio;
                new_fd.freq = freq_hz;
                new_fd
            }

            FluxDensityType::CurvedPowerLaw { si, fd, q } => {
                let mut power_law_component = fd.clone() * calc_flux_ratio(freq_hz, fd.freq, *si);
                power_law_component.freq = freq_hz;
                let curved_component = (q * (freq_hz / fd.freq).ln().powi(2)).exp();
                power_law_component * curved_component
            }

            FluxDensityType::List { fds } => {
                let mut old_freq = -1.0;

                // `smaller_flux_density` is a bad name given to the component's flux
                // density corresponding to a frequency smaller than but nearest to the
                // specified frequency.
                let (spec_index, smaller_flux_density) = {
                    // If there's only one source component, then we must assume the
                    // spectral index for extrapolation.
                    if fds.len() == 1 {
                        // trace!("Only one flux density in a component's list; extrapolating with spectral index {}", DEFAULT_SPEC_INDEX);
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
                                panic!("The list of flux densities used for estimation were not sorted");
                            }
                            old_freq = f;

                            // Iterate until we hit a catalogue frequency bigger than the
                            // desired frequency.

                            // If this freq and the specified freq are the same...
                            if (f - freq_hz).abs() < 1e-3 {
                                // ... then just return the flux density information from
                                // this frequency.
                                return fd.clone();
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
                            fds[smaller_comp_index].calc_spec_index(&fds[larger_comp_index]);

                        // Stop stupid spectral indices.
                        if spec_index < SPEC_INDEX_CAP {
                            // trace!(
                            //     "Component had a spectral index {}; capping at {}",
                            //     spec_index,
                            //     SPEC_INDEX_CAP
                            // );
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

                // Now scale the flux densities given the calculated
                // spectral index.
                let flux_ratio = calc_flux_ratio(freq_hz, smaller_flux_density.freq, spec_index);
                let fd = FluxDensity {
                    freq: freq_hz,
                    ..*smaller_flux_density
                } * flux_ratio;

                // Handle NaNs by setting the flux densities to 0.
                if fd.i.is_nan() {
                    FluxDensity {
                        freq: fd.freq,
                        ..Default::default()
                    }
                } else {
                    fd
                }
            }
        }
    }
}

/// Given a spectral index, determine the flux-density ratio of two frequencies.
pub fn calc_flux_ratio(desired_freq_hz: f64, cat_freq_hz: f64, spec_index: f64) -> f64 {
    (desired_freq_hz / cat_freq_hz).powf(spec_index)
}

#[cfg(test)]
impl approx::AbsDiffEq for FluxDensity {
    type Epsilon = f64;

    fn default_epsilon() -> f64 {
        f64::EPSILON
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: f64) -> bool {
        f64::abs_diff_eq(&self.freq, &other.freq, epsilon)
            && f64::abs_diff_eq(&self.i, &other.i, epsilon)
            && f64::abs_diff_eq(&self.q, &other.q, epsilon)
            && f64::abs_diff_eq(&self.u, &other.u, epsilon)
            && f64::abs_diff_eq(&self.v, &other.v, epsilon)
    }
}
