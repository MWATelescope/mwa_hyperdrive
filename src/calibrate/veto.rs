// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Code to remove sources from a source list.

Sources can be either because their position in the beam attenuates them
sufficiently (veto), or we request a certain number of sources.
 */

// use mwalib::{coarse_channel::CoarseChannel, MetafitsContext};
use mwalib::{CoarseChannel, MetafitsContext};
use rayon::{iter::Either, prelude::*};
use thiserror::Error;

use super::params::RankedSource;
use crate::*;
use mwa_hyperdrive_core::*;

/// This function mutates the input source list, removing any sources that have
/// beam-attenuated flux densities less than the threshold, and/or remove
/// sources that aren't in the top n sources specified by `num_sources`.
///
/// This is important for calibration, because a source might be too dim to be
/// practically used, or its flux density might be too attenuated at its
/// position in the beam (at any observed frequency).
///
/// If the input `source_list` has more sources than `num_sources`, then
/// `source_cutoff_dist` is used to keep the number of calculations down (filter
/// any sources that are more than that many degrees away). Otherwise, only
/// sources that need to be vetoed due to their positions in the beam are
/// vetoed.
///
/// Assume an ideal array (all dipoles with unity gain). Also assume that the
/// observation does not elapse enough time to shift sources into beam nulls
/// compared to the obs start.
pub(crate) fn veto_sources(
    source_list: &mut SourceList,
    metafits: &MetafitsContext,
    coarse_chans: &[CoarseChannel],
    beam: &mwa_hyperbeam::fee::FEEBeam,
    num_sources: Option<usize>,
    source_dist_cutoff: f64,
    veto_threshold: f64,
) -> Result<Vec<RankedSource>, VetoError> {
    // Do we need to consider the source distances? Determine this from the
    // final number of sources requested.
    let dist_cutoff: Option<f64> = {
        match num_sources {
            Some(n) => {
                if source_list.len() < n {
                    {
                        debug!("The supplied source list has enough sources to satisfy the requested number of sources; no distance cutoff necessary");
                        None
                    }
                } else {
                    {
                        debug!("The supplied source list has more sources than the requested number of sources; using a distance cutoff of {} degrees", source_dist_cutoff);
                        Some(source_dist_cutoff.to_radians())
                    }
                }
            }
            None => {
                debug!("The number of sources to use was not specified; using a distance cutoff of {} degrees to filter sources", source_dist_cutoff);
                Some(source_dist_cutoff.to_radians())
            }
        }
    };
    let pc_radec = RADec::new_degrees(
        metafits.ra_tile_pointing_degrees,
        metafits.dec_tile_pointing_degrees,
    );

    // We want to store the flux densities for each source at the "mean
    // frequency". Store the middle coarse channel number to achieve this later.
    let middle_coarse_chan = coarse_chans[coarse_chans.len() / 2].corr_chan_number;

    let (vetoed_sources, mut not_vetoed_sources): (Vec<_>, Vec<_>) = source_list
        .par_iter()
        .partition_map(|(source_name, source)| {
            // Are any of this source's components too low in elevation? Or too
            // far from the pointing centre?
            for comp in &source.components {
                let azel = comp.radec.to_hadec(metafits.lst_rad).to_azel_mwa();
                if azel.el < ELEVATION_LIMIT {
                    trace!("A component of source {}'s elevation ({} radians) was below the limit ({} radians)",
                           source_name,
                           azel.el,
                           ELEVATION_LIMIT);
                    return Either::Left(source_name.clone());
                } else if let Some(d) = dist_cutoff {
                    if comp.radec.separation(&pc_radec) > d {
                    trace!("A component of source {}'s elevation ({} radians) was too far from the pointing centre ({} radians)",
                           source_name,
                           azel.el,
                           d);
                    return Either::Left(source_name.clone());
                    };
                }
            }

            // For this source, work out its smallest flux density at any of the
            // observing frequencies. This is how we determine which sources are
            // "best".
            let mut smallest_fd = std::f64::INFINITY;

            // `cc_fds` are the flux densities for each coarse channel.
            let mut cc_fds = vec![];

            // Iterate over each frequency. Is the total flux density acceptable
            // for each frequency?
            for coarse_chan in coarse_chans {
                // `fd` is the sum of the source's component flux densities.
                let mut fd = 0.0;

                // Get the beam response at this source position and frequency.
                let j = Jones::from(beam.calc_jones(
                        metafits.az_rad,
                        metafits.za_rad,
                        coarse_chan.chan_centre_hz,
                        // Use the ideal delays.
                        &metafits.delays,
                        &[1.0; 16],
                    true)
                    // unwrap is ugly, but an error would indicate a serious
                    // problem with hyperbeam. The alternative is lots of
                    // "and_then" control flow operators.
                              .unwrap());
                // TODO: Remove unwrap.

                for source_fd in source
                    .get_flux_estimates(coarse_chan.chan_centre_hz as _)
                    .unwrap()
                {
                    fd += get_beam_attenuated_flux_density(&source_fd, &j);
                    cc_fds.push(source_fd);
                }

                if fd < veto_threshold {
                    trace!(
                        "Source {}'s brightness ({}) is less than the veto threshold ({})",
                        source_name,
                        fd,
                        veto_threshold
                    );
                    return Either::Left(source_name.clone());
                }
                smallest_fd = fd.min(smallest_fd);
            }

            // If we got this far, the source should not be vetoed.
            Either::Right(RankedSource::new( source_name.to_owned(),
                 smallest_fd,
                 cc_fds,
                 source,
                 // Use the centre observation frequency.
                 metafits.centre_freq_hz as f64,
            ).unwrap())
        });

    trace!(
        "The following {} sources were vetoed from the source list: {:?}",
        vetoed_sources.len(),
        vetoed_sources
    );
    for name in &vetoed_sources {
        source_list.remove(name);
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

    // Sort the source list by flux density from brightest to dimmest.
    //
    // par_sort_unstable_by is a parallel sort. Don't let the "unstable"
    // scare you; this just means it's the fastest kind of sort.
    //
    // The argument to par_sort_unstable_by compares two elements in
    // not_vetoed_sources. We want to sort by the smallest flux densities of
    // each source; that's the 2nd element of each tuple.
    //
    // The reverse comparison (b against a) is deliberate; we want the
    // sources reverse-sorted by beam-attenuated flux density.
    not_vetoed_sources.par_sort_unstable_by(|a, b| {
        b.apparent_fd
            .partial_cmp(&a.apparent_fd)
            .unwrap_or_else(|| panic!("Couldn't compare {} to {}", a.apparent_fd, b.apparent_fd))
    });

    // If we were requested to use n number of sources, remove all sources after n.
    if let Some(n) = num_sources {
        for source_to_be_removed in not_vetoed_sources.iter().skip(n) {
            source_list.remove(&source_to_be_removed.name);
        }
    }

    Ok(not_vetoed_sources)
}

// This function is isolated for testing.
#[inline(always)]
fn get_beam_attenuated_flux_density(fd: &FluxDensity, j: &Jones) -> f64 {
    // Form an ideal flux-density Jones matrix for each of the
    // source components.
    let i = Jones::from([
        c64::new(fd.i + fd.q, 0.0),
        c64::new(fd.u, fd.v),
        c64::new(fd.u, -fd.v),
        c64::new(fd.i - fd.q, 0.0),
    ]);
    // Calculate: J . I . J^H
    // where J is the beam-response Jones matrix and I is the
    // source's ideal Jones matrix.
    let ji = j.clone() * i;
    let jijh = ji.mul_hermitian(&j);
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
    Hyperbeam(#[from] mwa_hyperbeam::fee::FEEBeamError),
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::*;
    // Need to use serial tests because HDF5 is not necessarily reentrant.
    use serial_test::serial;

    use mwa_hyperdrive_srclist::SourceListType;
    use mwalib::CorrelatorVersion;

    fn get_params() -> (
        mwalib::MetafitsContext,
        mwa_hyperbeam::fee::FEEBeam,
        SourceList,
    ) {
        (
            MetafitsContext::new(&"tests/1065880128/1065880128.metafits").unwrap(),
            mwa_hyperbeam::fee::FEEBeam::new_from_env().unwrap(),
            mwa_hyperdrive_srclist::read::read_source_list_file(
                &PathBuf::from(
                    "tests/1065880128/srclist_pumav3_EoR0aegean_EoR1pietro+ForA_1065880128_100.yaml",
                ),
                &SourceListType::Hyperdrive,
            )
            .unwrap(),
        )
    }

    #[test]
    #[serial]
    fn test_beam_attenuated_flux_density() {
        let (metafits, beam, _) = get_params();
        let jones_pointing_centre = Jones::from(
            beam.calc_jones(
                metafits.az_rad,
                metafits.za_rad,
                180e6 as _,
                &metafits.delays,
                &[1.0; 16],
                true,
            )
            .unwrap(),
        );
        let radec_null = RADec::new_degrees(
            metafits.ra_tile_pointing_degrees + 80.0,
            metafits.dec_tile_pointing_degrees + 80.0,
        );
        let azel_null = radec_null.to_hadec(metafits.lst_rad).to_azel_mwa();
        let jones_null = Jones::from(
            beam.calc_jones(
                azel_null.az,
                azel_null.za(),
                180e6 as _,
                &metafits.delays,
                &[1.0; 16],
                true,
            )
            .unwrap(),
        );
        let fd = FluxDensity {
            freq: 180e6,
            i: 1.0,
            q: 0.0,
            u: 0.0,
            v: 0.0,
        };
        let bafd_pc = get_beam_attenuated_flux_density(&fd, &jones_pointing_centre);
        assert_abs_diff_eq!(bafd_pc, 1.9867421166394585, epsilon = 1e-10);

        let bafd_null = get_beam_attenuated_flux_density(&fd, &jones_null);
        assert_abs_diff_eq!(bafd_null, 0.000000017940424114049793, epsilon = 1e-10);
    }

    #[test]
    #[serial]
    fn veto() {
        let (metafits, beam, mut source_list) = get_params();
        // For testing's sake, keep only the following sources. The first five
        // are bright and won't get vetoed and the last three are dim.
        let sources = &[
            "J002549-260211",
            "J004616-420739",
            "J010817-160418",
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
        // they will be vetoed.
        source_list.insert(
            "bad_source1".to_string(),
            Source {
                components: vec![SourceComponent {
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
                components: vec![SourceComponent {
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
                components: vec![SourceComponent {
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

        let result = veto_sources(
            &mut source_list,
            &metafits,
            &metafits
                .get_expected_coarse_channels(CorrelatorVersion::Legacy)
                .unwrap(),
            &beam,
            None,
            180.0,
            20.0,
        );
        assert!(result.is_ok());
        result.unwrap();
        // Only the first five are kept.
        assert_eq!(
            source_list.len(),
            5,
            "Expected only five sources to not get vetoed: {:#?}",
            source_list.keys()
        );
    }

    #[test]
    #[serial]
    fn top_n_sources() {
        let (metafits, beam, mut source_list) = get_params();
        // For testing's sake, keep only the following sources.
        let sources = &[
            "J002549-260211",
            "J004209-441404",
            "J004616-420739",
            "J010817-160418",
            "J012027-152011",
            "J013411-362913",
            "J233426-412520",
            "J235701-344532",
        ];
        let keys: Vec<String> = source_list.keys().cloned().collect();
        for source_name in keys {
            if !sources.contains(&source_name.as_str()) {
                source_list.remove(source_name.as_str());
            }
        }

        let result = veto_sources(
            &mut source_list,
            &metafits,
            &metafits
                .get_expected_coarse_channels(CorrelatorVersion::Legacy)
                .unwrap(),
            &beam,
            Some(5),
            180.0,
            0.1,
        );
        assert!(result.is_ok());
        result.unwrap();
        for &name in &[
            "J002549-260211",
            "J004616-420739",
            "J010817-160418",
            "J233426-412520",
            "J235701-344532",
        ] {
            assert!(
                &source_list.contains_key(name),
                "Expected to find {} in the source list after vetoing",
                name
            );
        }
    }
}
