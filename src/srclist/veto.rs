// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to remove sources from a source list.
//!
//! A sources can be removed either because its position in the beam attenuates
//! its brightness too severely (veto), or we request a certain number of
//! sources (say N) and there are more than N sources in the source list.

use std::collections::BTreeMap;

use log::{debug, log_enabled, trace, Level::Trace};
use marlu::{Jones, RADec};
use rayon::{iter::Either, prelude::*};

use crate::{
    beam::Beam,
    constants::*,
    srclist::{FluxDensity, ReadSourceListError, SourceList},
};

/// This function mutates the input source list, removing any sources that have
/// components below the elevation limit, components that are too far from the
/// phase centre, components with beam-attenuated flux densities less than the
/// threshold, and/or remove sources that aren't in the top N sources specified
/// by `num_sources`. The source list is also sorted by reverse brightness; i.e.
/// the brightest source is first. ([`SourceList`] is actually an
/// `indexmap::IndexMap`, which is like an order-preserved `HashMap`.)
///
/// This is important for calibration, because it is expensive to generate a sky
/// model, and using only dim sources would result in poor calibration.
///
/// Sources are vetoed if any of their components are further away from the
/// phase centre than `source_dist_cutoff_deg` or their beam attenuated flux
/// densities are less than `veto_threshold`.
///
/// If there are fewer sources than that of `num_sources`, an error is returned;
/// it's up to the caller to handle this if they want to.
///
/// `freqs_hz`: The frequencies to use for beam calculations \[Hz\]. These are
/// traditionally multiples of 1.28 MHz.
#[allow(clippy::too_many_arguments)]
pub(crate) fn veto_sources(
    source_list: &mut SourceList,
    phase_centre: RADec,
    lst_rad: f64,
    array_latitude_rad: f64,
    freqs_hz: &[f64],
    beam: &dyn Beam,
    num_sources: Option<usize>,
    source_dist_cutoff_deg: f64,
    veto_threshold: f64,
) -> Result<(), ReadSourceListError> {
    let dist_cutoff = source_dist_cutoff_deg.to_radians();

    // TODO: This step is relatively expensive!
    let (vetoed_sources, not_vetoed_sources): (Vec<Result<String, ReadSourceListError>>, BTreeMap<String, f64>) = source_list
        .par_iter()
        .partition_map(|(source_name, source)| {
            let source_name = source_name.to_owned();

            // For this source, work out its smallest flux density at any of the
            // coarse channel frequencies. This is how we determine which
            // sources are "best".
            let mut smallest_fd = f64::INFINITY;

            // Filter trivial sources: are any of this source's components too
            // low in elevation? Or too far from the phase centre?
            let mut azels = vec![];
            for comp in source.components.iter() {
                let azel = comp.radec.to_hadec(lst_rad).to_azel(array_latitude_rad);
                if azel.el.to_degrees() < ELEVATION_LIMIT {
                    if log_enabled!(Trace) {
                        trace!("A component's elevation ({}°, source {source_name}) was below the limit ({ELEVATION_LIMIT}°)", azel.el.to_degrees());
                    }
                    return Either::Left(Ok(source_name));
                }
                let separation = comp.radec.separation(phase_centre);
                if separation > dist_cutoff {
                    if log_enabled!(Trace) {
                        trace!("A component (source {source_name}) was too far from the phase centre (separation {}°)", separation.to_degrees());
                    }
                    return Either::Left(Ok(source_name));
                }
                azels.push(azel);
            }

            // Iterate over each frequency. Is the total flux density
            // acceptable for each frequency?
            for &cc_freq in freqs_hz {
                // `fd` is the sum of the source's component XX+YY flux
                // densities at this coarse-channel frequency.
                let mut fd = 0.0;

                for (comp, azel) in source.components.iter().zip(azels.iter()) {
                    // Get the beam response at this source position and
                    // frequency.
                    let j = match beam.calc_jones(
                            *azel,
                            cc_freq,
                        None,
                        array_latitude_rad) {
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
            Either::Right((source_name, smallest_fd))
        });

    // Handle potential errors while vetoing (such as the beam code failing).
    let mut vetoed_sources = vetoed_sources.into_iter().collect::<Result<Vec<_>, _>>()?;

    // Remove vetoed sources from the source list.
    for name in vetoed_sources.iter() {
        source_list.remove_entry(name);
    }

    // Now that only not-vetoed sources are left in the source list, sort the
    // sources *descendingly* with respect to brightness. i.e. The apparently
    // brightest source is first.
    source_list.par_sort_unstable_by(|a_key, _, b_key, _| {
        let a_brightness = not_vetoed_sources[a_key];
        let b_brightness = not_vetoed_sources[b_key];
        b_brightness
            .partial_cmp(&a_brightness)
            // No NaNs should be here.
            .unwrap()
    });
    drop(not_vetoed_sources);

    // Reduce the number of sources if we have to.
    if let Some(n) = num_sources {
        if source_list.len() > n {
            // Add the not-top-N sources into `vetoed_sources`.
            source_list
                .drain(n..)
                .for_each(|(name, _)| vetoed_sources.push(name));
        }
    }

    debug!(
        "{} sources were vetoed from the source list",
        vetoed_sources.len()
    );
    if log_enabled!(Trace) {
        trace!(
            "The following {} sources were vetoed from the source list:",
            vetoed_sources.len()
        );
        for vetoed_sources in vetoed_sources.chunks(5) {
            trace!("  {vetoed_sources:?}");
        }
    }

    // If there are fewer sources than requested after vetoing, we need to bail
    // out.
    if let Some(n) = num_sources {
        if n > source_list.len() {
            return Err(ReadSourceListError::VetoTooFewSources {
                requested: n,
                available: source_list.len(),
            });
        }
    }

    Ok(())
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

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use marlu::{constants::MWA_LAT_RAD, AzEl};
    use serial_test::*;

    use super::*;
    use crate::{
        beam::{Delays, FEEBeam, NoBeam},
        srclist::{
            read::read_source_list_file, ComponentType, FluxDensityType, Source, SourceComponent,
        },
    };

    #[test]
    fn test_beam_attenuated_flux_density_no_beam() {
        let beam = NoBeam { num_tiles: 1 };
        let jones_pointing_centre = beam
            .calc_jones(AzEl::from_degrees(0.0, 90.0), 180e6, None, MWA_LAT_RAD)
            .unwrap();
        let jones_null = beam
            .calc_jones(AzEl::from_degrees(10.0, 10.0), 180e6, None, MWA_LAT_RAD)
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
        let beam = FEEBeam::new_from_env(1, Delays::Partial(vec![0; 16]), None).unwrap();
        let jones_pointing_centre = beam
            .calc_jones(AzEl::from_degrees(0.0, 89.0), 180e6, None, MWA_LAT_RAD)
            .unwrap();
        let jones_null = beam
            .calc_jones(AzEl::from_degrees(10.0, 10.0), 180e6, None, MWA_LAT_RAD)
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
        let beam = FEEBeam::new_from_env(1, Delays::Partial(vec![0; 16]), None).unwrap();
        let (mut source_list, _) = read_source_list_file("test_files/1090008640/srclist_pumav3_EoR0aegean_EoR1pietro+ForA_1090008640_peel100.txt", None).unwrap();

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
                components: vec![SourceComponent {
                    radec: RADec::from_degrees(330.0, -80.0),
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
                }]
                .into_boxed_slice(),
            },
        );
        source_list.insert(
            "bad_source2".to_string(),
            Source {
                components: vec![SourceComponent {
                    radec: RADec::from_degrees(30.0, -80.0),
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
                }]
                .into_boxed_slice(),
            },
        );
        source_list.insert(
            "bad_source3".to_string(),
            Source {
                components: vec![SourceComponent {
                    radec: RADec::from_degrees(285.0, 40.0),
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
                }]
                .into_boxed_slice(),
            },
        );

        let phase_centre = RADec::from_degrees(0.0, -27.0);
        let result = veto_sources(
            &mut source_list,
            phase_centre,
            0.0,
            MWA_LAT_RAD,
            &[167.68e6, 197.12e6],
            &beam,
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
        let beam = NoBeam { num_tiles: 1 };
        let (mut source_list, _) = read_source_list_file("test_files/1090008640/srclist_pumav3_EoR0aegean_EoR1pietro+ForA_1090008640_peel100.txt", None).unwrap();

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

        let phase_centre = RADec::from_degrees(0.0, -27.0);
        let result = veto_sources(
            &mut source_list,
            phase_centre,
            0.0,
            MWA_LAT_RAD,
            &[167.68e6, 197.12e6],
            &beam,
            Some(3),
            180.0,
            0.1,
        );
        assert!(result.is_ok(), "{:?}", result.unwrap_err());
        result.unwrap();
        for &name in &["J000042-342358", "J000105-165921", "J000143-305731"] {
            assert!(
                &source_list.contains_key(name),
                "Expected to find {name} in the source list after vetoing"
            );
        }
    }

    #[test]
    fn sorted_by_reverse_brightness() {
        let beam = NoBeam { num_tiles: 1 };
        let (mut source_list, _) = read_source_list_file("test_files/1090008640/srclist_pumav3_EoR0aegean_EoR1pietro+ForA_1090008640_peel100.txt", None).unwrap();

        let phase_centre = RADec::from_degrees(0.0, -27.0);
        let result = veto_sources(
            &mut source_list,
            phase_centre,
            0.0,
            MWA_LAT_RAD,
            &[167.68e6, 197.12e6],
            &beam,
            None,
            180.0,
            0.1,
        );
        assert!(result.is_ok(), "{:?}", result.unwrap_err());
        result.unwrap();

        assert_eq!(source_list.len(), 100);
        assert_eq!(source_list.get_index(0).unwrap().0, "J004616-420739");
        assert_eq!(source_list.get_index(99).unwrap().0, "J000217-253912");
    }
}
