// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to remove sources from a source list.
//!
//! A sources can be removed either because its position in the beam attenuates
//! its brightness too severely (veto), or we request a certain number of
//! sources (say N) and there are more than N sources in the source list.

use log::{debug, trace, log_enabled, Level::Trace};
use rayon::{iter::Either, prelude::*};
use thiserror::Error;
use marlu::{RADec, Jones};

use crate::{FluxDensity, SourceList, constants::*};
use mwa_hyperdrive_beam::{BeamError, Beam};
use mwa_hyperdrive_common::{log, rayon, thiserror, marlu};

#[derive(Debug)]
struct RankedSource<'a> {
    name: &'a String,
    apparent_fd: f64,
}

/// This function mutates the input source list, removing any sources that have
/// beam-attenuated flux densities less than the threshold, and/or remove
/// sources that aren't in the top N sources specified by `num_sources`. The
/// name of the apparently brightest source is returned.
///
/// This is important for calibration, because it is expensive to generate a sky
/// model, and using only dim sources would result in poor calibration.
///
/// If the input `source_list` has more sources than `num_sources`, then
/// `source_cutoff_dist` is used to keep the number of calculations down (filter
/// any sources that are more than that many degrees away). Otherwise, only
/// sources that need to be vetoed due to their positions in the beam are
/// vetoed.
///
/// `coarse_chan_freqs_hz`: The centre frequencies of each of the coarse
/// channels of this observation \[Hz\].
///
/// Assume an ideal array (all dipoles with unity gain). Also assume that the
/// observation does not elapse enough time to shift sources into beam nulls
/// compared to the obs start.
// TODO: Ensure the documentation is accurate.
#[allow(clippy::too_many_arguments)]
pub fn veto_sources(
    source_list: &mut SourceList,
    phase_centre: RADec,
    lst_rad: f64,
    array_latitude_rad: f64,
    coarse_chan_freqs_hz: &[f64],
    beam: &dyn Beam,
    num_sources: Option<usize>,
    source_dist_cutoff_deg: f64,
    veto_threshold: f64,
) -> Result<Option<String>, VetoError> {
    let dist_cutoff = source_dist_cutoff_deg.to_radians();

    // TODO: This step is relatively expensive!
    let (vetoed_sources, mut not_vetoed_sources): (Vec<Result<&String, VetoError>>, Vec<RankedSource>) = source_list
        .par_iter()
        .partition_map(|(source_name, source)| {
            // For this source, work out its smallest flux density at any of the
            // coarse channel frequencies. This is how we determine which
            // sources are "best".
            let mut smallest_fd = std::f64::INFINITY;

            // Filter trivial sources: are any of this source's components too
            // low in elevation? Or too far from the phase centre?
            let mut azels = vec![];
            for comp in &source.components {
                let azel = comp.radec.to_hadec(lst_rad).to_azel(array_latitude_rad);
                if azel.el.to_degrees() < ELEVATION_LIMIT {
                    if log_enabled!(Trace) {
                        trace!("A component's elevation ({}°, source {}) was below the limit ({}°)", azel.el.to_degrees(), source_name, ELEVATION_LIMIT);
                    }
                    return Either::Left(Ok(source_name));
                }
                let separation = comp.radec.separation(phase_centre);
                if separation > dist_cutoff {
                    if log_enabled!(Trace) {
                        trace!("A component (source {}) was too far from the phase centre (separation {}°)", source_name, separation);
                    }
                    return Either::Left(Ok(source_name));
                }
                azels.push(azel);
            }

            // Iterate over each frequency. Is the total flux density
            // acceptable for each frequency?
            for &cc_freq in coarse_chan_freqs_hz {
                // `fd` is the sum of the source's component XX+YY flux
                // densities at this coarse-channel frequency.
                let mut fd = 0.0;
                
                for (comp, azel) in source.components.iter().zip(azels.iter()) {
                    // Get the beam response at this source position and
                    // frequency.
                    let j = match beam.calc_jones(
                            *azel,
                            cc_freq,
                            // Have to assume that tile 0 is sensible.
                        0) {
                            Ok(j) => j,
                            Err(e) => {
                                trace!("Beam error for source {}", source_name);
                                return Either::Left(Err(e.into()))
                            },
                        };

                    let comp_fd = comp.estimate_at_freq(cc_freq);
                    fd += get_beam_attenuated_flux_density(&comp_fd, j);
                }
                
                if fd < veto_threshold {
                    if log_enabled!(Trace) {
                        trace!(
                            "Source {}'s XX+YY brightness ({} Jy) is less than the veto threshold ({} Jy)",
                            source_name,
                            fd,
                            veto_threshold
                        );
                    }
                    return Either::Left(Ok(source_name));
                }
                smallest_fd = fd.min(smallest_fd);
            }

            // If we got this far, the source should not be vetoed.
            Either::Right(RankedSource {
                name: source_name,
                apparent_fd: smallest_fd,
            })
        });

    // Handle potential errors while vetoing (such as the beam code failing).
    let mut vetoed_sources = vetoed_sources.into_iter().collect::<Result<Vec<_>, _>>()?;

    // Reverse-sort the sources by brightness.
    not_vetoed_sources.par_sort_unstable_by(|a, b| {
        b.apparent_fd
            .partial_cmp(&a.apparent_fd)
            .unwrap_or_else(|| panic!("Couldn't compare {} to {}", a.apparent_fd, b.apparent_fd))
    });

    // Reduce the number of sources if we have to.
    if let Some(n) = num_sources {
        if not_vetoed_sources.len() > n {
            // Add the not-top-N sources into `vetoed_sources`.
            let mut dimmer_sources = not_vetoed_sources.drain(n..).map(|ranked_source| ranked_source.name).collect();
            vetoed_sources.append(&mut dimmer_sources);
        }
    }

    let first_name = not_vetoed_sources.first().map(|rs| rs.name.clone());
    drop(not_vetoed_sources);

    debug!(
        "{} sources were vetoed from the source list",
        vetoed_sources.len()
    );
    trace!(
        "The following {} sources were vetoed from the source list: {:?}",
        vetoed_sources.len(),
        vetoed_sources
    );

    // TODO: Please, someone, help me!
    let asdf: Vec<String> = vetoed_sources.iter().map(|s| s.to_owned().to_owned()).collect();
    drop(vetoed_sources);
    for name in asdf {
        source_list.remove(&name);
    }

    // If there are fewer sources than requested after vetoing, we need to bail
    // out.
    if let Some(n) = num_sources {
        if n > source_list.len() {
            return Err(VetoError::TooFewSources {
                requested: n,
                available: source_list.len(),
            });
        }
    }

    Ok(first_name)
}

/// Convert a Stokes flux densities into instrumental flux densities, and
/// multiply by a beam-response Jones matrix. Return the sum of the response XX
/// and YY flux densities as the "beam attenuated flux density".
// This function is isolated for testing.
fn get_beam_attenuated_flux_density(fd: &FluxDensity, j: Jones<f64>) -> f64 {
    // Get the instrumental flux densities as a Jones matrix.
    let i = fd.to_inst_stokes();
    // Calculate: J . I . J^H
    // where J is the beam-response Jones matrix and I are the instrumental flux
    // densities.
    let jijh = j * Jones::axbh(i, j);
    // Use the trace of `jijh` as the total source flux density.
    // Using the determinant instead of the trace might be more
    // realistic; uncomment the line below to do that.
    jijh[0].norm() + jijh[3].norm()
    // (jijh[0].norm() * jijh[3].norm()) - (jijh[1].norm() * jijh[2].norm())
}

#[derive(Error, Debug)]
pub enum VetoError {
    #[error("Tried to use {requested} sources, but only {available} sources were available after vetoing")]
    TooFewSources { requested: usize, available: usize },

    #[error("{0}")]
    Beam(#[from] BeamError),
}

#[cfg(test)]
mod tests {
    use std::ops::Deref;

    use mwa_hyperdrive_beam::{Delays, create_fee_beam_object, create_no_beam_object};
    use marlu::{AzEl, constants::MWA_LAT_RAD};
    use approx::assert_abs_diff_eq;
    use serial_test::*;
    use vec1::vec1;

    use super::*;
    use crate::{ComponentType, FluxDensityType, Source, SourceComponent, read::read_source_list_file};
    use mwa_hyperdrive_common::vec1;

    #[test]
    fn test_beam_attenuated_flux_density_no_beam() {
        let beam = create_no_beam_object(1);
        let jones_pointing_centre = beam
            .calc_jones(AzEl::new_degrees(0.0, 90.0), 180e6, 0)
            .unwrap();
        let jones_null = beam
            .calc_jones(AzEl::new_degrees(10.0, 10.0), 180e6, 0)
            .unwrap();
        let fd = FluxDensity {
            freq: 180e6,
            i: 1.0,
            q: 0.0,
            u: 0.0,
            v: 0.0,
        };
        let bafd_pc = get_beam_attenuated_flux_density(&fd, jones_pointing_centre);
        assert_abs_diff_eq!(bafd_pc, 2.0);

        let bafd_null = get_beam_attenuated_flux_density(&fd, jones_null);
        assert_abs_diff_eq!(bafd_null, 2.0);
    }

    #[test]
    #[serial]
    fn test_beam_attenuated_flux_density_fee_beam() {
        let beam_file: Option<&str> = None;
        let beam = create_fee_beam_object(beam_file, 1, Delays::Partial(vec![0; 16]), None).unwrap();
        let jones_pointing_centre = beam
            .calc_jones(AzEl::new_degrees(0.0, 89.0), 180e6, 0)
            .unwrap();
        let jones_null = beam
            .calc_jones(AzEl::new_degrees(10.0, 10.0), 180e6, 0)
            .unwrap();
        let fd = FluxDensity {
            freq: 180e6,
            i: 1.0,
            q: 0.0,
            u: 0.0,
            v: 0.0,
        };
        let bafd_pc = get_beam_attenuated_flux_density(&fd, jones_pointing_centre);
        assert_abs_diff_eq!(bafd_pc, 1.9857884953095866);

        let bafd_null = get_beam_attenuated_flux_density(&fd, jones_null);
        assert_abs_diff_eq!(bafd_null, 0.002789795062384414);
    }

    #[test]
    #[serial]
    fn veto() {
        let beam_file: Option<&str> = None;
        let beam = create_fee_beam_object(beam_file, 1, Delays::Partial(vec![0; 16]), None).unwrap();
        let (mut source_list, _) = read_source_list_file("../test_files/1090008640/srclist_pumav3_EoR0aegean_EoR1pietro+ForA_1090008640_peel100.txt", None).unwrap();

        // For testing's sake, keep only the following bright sources.
        let sources = &[
            "J002549-260211",
            "J004616-420739",
            "J233426-412520",
            "J235701-344532",
        ];
        let keys: Vec<String> = source_list.keys().cloned().collect();
        for source_name in keys {
            if !sources.contains(&source_name.as_str()) {
                source_list.remove(source_name.as_str());
            }
        }

        // Add some sources that are in beam nulls. Despite being very bright,
        // they should be vetoed.
        source_list.insert(
            "bad_source1".to_string(),
            Source {
                components: vec1![SourceComponent {
                    radec: RADec::new_degrees(330.0, -80.0),
                    comp_type: ComponentType::Point,
                    flux_type: FluxDensityType::PowerLaw {
                        si: -0.8,
                        fd: FluxDensity {
                            freq: 180e6,
                            i: 10.0,
                            q: 0.0,
                            u: 0.0,
                            v: 0.0,
                        },
                    },
                }],
            },
        );
        source_list.insert(
            "bad_source2".to_string(),
            Source {
                components: vec1![SourceComponent {
                    radec: RADec::new_degrees(30.0, -80.0),
                    comp_type: ComponentType::Point,
                    flux_type: FluxDensityType::PowerLaw {
                        si: -0.8,
                        fd: FluxDensity {
                            freq: 180e6,
                            i: 10.0,
                            q: 0.0,
                            u: 0.0,
                            v: 0.0,
                        },
                    },
                }],
            },
        );
        source_list.insert(
            "bad_source3".to_string(),
            Source {
                components: vec1![SourceComponent {
                    radec: RADec::new_degrees(285.0, 40.0),
                    comp_type: ComponentType::Point,
                    flux_type: FluxDensityType::PowerLaw {
                        si: -0.8,
                        fd: FluxDensity {
                            freq: 180e6,
                            i: 10.0,
                            q: 0.0,
                            u: 0.0,
                            v: 0.0,
                        },
                    },
                }],
            },
        );

        let phase_centre = RADec::new_degrees(0.0, -27.0);
        let result = veto_sources(
            &mut source_list,
            phase_centre,
            0.0,
            MWA_LAT_RAD,
            &[167.68e6, 197.12e6],
            beam.deref(),
            None,
            180.0,
            0.1,
        );
        assert!(result.is_ok());
        result.unwrap();

        // Only the first four sources are kept.
        assert_eq!(
            source_list.len(),
            4,
            "Expected only five sources to not get vetoed: {:#?}",
            source_list.keys()
        );
    }

    #[test]
    fn top_n_sources() {
        let beam = create_no_beam_object(1);
        let (mut source_list, _) = read_source_list_file("../test_files/1090008640/srclist_pumav3_EoR0aegean_EoR1pietro+ForA_1090008640_peel100.txt", None).unwrap();

        // For testing's sake, keep only the following sources.
        let sources = &[
            "J000042-342358",
            "J000045-272248",
            "J000105-165921",
            "J000143-305731",
            "J000217-253912",
            "J000245-302825",
        ];
        let keys: Vec<String> = source_list.keys().cloned().collect();
        for source_name in keys {
            if !sources.contains(&source_name.as_str()) {
                source_list.remove(source_name.as_str());
            }
        }

        let phase_centre = RADec::new_degrees(0.0, -27.0);
        let result = veto_sources(
            &mut source_list,
            phase_centre,
            0.0,
            MWA_LAT_RAD,
            &[167.68e6, 197.12e6],
            beam.deref(),
            Some(3),
            180.0,
            0.1,
        );
        assert!(result.is_ok(), "{:?}", result.unwrap_err());
        result.unwrap();
        for &name in &["J000042-342358", "J000105-165921", "J000143-305731"] {
            assert!(
                &source_list.contains_key(name),
                "Expected to find {} in the source list after vetoing",
                name
            );
        }
    }
}
