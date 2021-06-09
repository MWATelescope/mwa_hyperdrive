// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Specialised code for calibrating sky-model sources.
//!
//! The `SourceList` type is deliberately not sorted by flux density, or anything
//! else, so the `RankedSource` type here helps calibration by providing a
//! beam-attenuated flux density, as well as other things.

use mwa_hyperdrive_core::*;

/// A source's name as well as its apparent flux density.
#[derive(Debug)]
pub(crate) struct RankedSource {
    /// The name of the source. This can be used as a key for a `SourceList`.
    pub(crate) name: String,

    /// The smallest apparent flux density across all frequencies of the
    /// observation [Jy].
    pub(crate) apparent_fd: f64,

    /// The flux densities of this source for each of the observation's coarse
    /// channels.
    pub(crate) coarse_chan_fds: Vec<FluxDensity>,

    /// The flux-density-weighted average position of all of the source's
    /// components. Only Stokes I flux densities are used in this calculation.
    pub(crate) weighted_pos: Option<RADec>,
}

impl RankedSource {
    /// Create a new `RankedSource` struct. The `src` and `freq_hz` variables
    /// are used to generated the flux-density-weighted position here.
    pub(crate) fn new(
        name: String,
        apparent_fd_jy: f64,
        coarse_chan_fds: Vec<FluxDensity>,
        src: &Source,
        freq_hz: f64,
    ) -> Result<Self, EstimateError> {
        let mut weights = vec![];
        for c in &src.components {
            let stokes_i = c.flux_type.estimate_at_freq(freq_hz)?.i;
            weights.push(stokes_i);
        }
        let weighted_pos = RADec::weighted_average(
            &src.components
                .iter()
                .map(|c| &c.radec)
                .collect::<Vec<&RADec>>(),
            &weights,
        );
        Ok(Self {
            name,
            apparent_fd: apparent_fd_jy,
            coarse_chan_fds,
            weighted_pos,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_weighted_pos() {
        // Only the src variable matters here; the rest of the variables aren't
        // used when constructing `RankedSource`.

        // Simple case: both components have a Stokes I FD of 1.0.
        let mut src = Source {
            components: vec![
                SourceComponent {
                    // From astropy docs.
                    radec: RADec::new_degrees(10.0, 9.0),
                    comp_type: ComponentType::Point,
                    flux_type: FluxDensityType::List {
                        fds: vec![FluxDensity {
                            freq: 150e6,
                            i: 1.0,
                            q: 0.0,
                            u: 0.0,
                            v: 0.0,
                        }],
                    },
                },
                SourceComponent {
                    radec: RADec::new_degrees(11.0, 10.0),
                    comp_type: ComponentType::Point,
                    flux_type: FluxDensityType::List {
                        fds: vec![FluxDensity {
                            freq: 150e6,
                            i: 1.0,
                            q: 0.0,
                            u: 0.0,
                            v: 0.0,
                        }],
                    },
                },
            ],
        };
        let coarse_chan_fds = vec![FluxDensity {
            freq: 150e6,
            i: 1.0,
            q: 0.0,
            u: 0.0,
            v: 0.0,
        }];
        let ranked_source = match RankedSource::new(
            "asdf".to_string(),
            10.0,
            coarse_chan_fds.clone(),
            &src,
            150e6,
        ) {
            Ok(rs) => rs,
            Err(e) => panic!("{}", e),
        };
        let weighted_pos = ranked_source.weighted_pos.unwrap();
        assert_abs_diff_eq!(weighted_pos.ra, 10.5_f64.to_radians(), epsilon = 1e-10);
        assert_abs_diff_eq!(weighted_pos.dec, 9.5_f64.to_radians(), epsilon = 1e-10);

        // Complex case: both components have different Stokes I FDs. Modify the
        // FD of the first component.
        match &mut src.components[0].flux_type {
            FluxDensityType::List { fds } => fds[0].i = 3.0,
            _ => unreachable!(),
        }
        let ranked_source =
            match RankedSource::new("asdf".to_string(), 10.0, coarse_chan_fds, &src, 150e6) {
                Ok(rs) => rs,
                Err(e) => panic!("{}", e),
            };
        let weighted_pos = ranked_source.weighted_pos.unwrap();
        assert_abs_diff_eq!(weighted_pos.ra, 10.25_f64.to_radians(), epsilon = 1e-10);
        assert_abs_diff_eq!(weighted_pos.dec, 9.25_f64.to_radians(), epsilon = 1e-10);
    }
}
