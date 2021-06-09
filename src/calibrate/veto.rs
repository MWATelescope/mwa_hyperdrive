// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to remove sources from a source list.
//!
//! A sources can be removed either because its position in the beam attenuates
//! its brightness too severely (veto), or we request a certain number of
//! sources (say N) and there are more than N sources in the source list.

use log::{debug, trace, warn};
use rayon::{iter::Either, prelude::*};
use thiserror::Error;

use super::params::RankedSource;
use crate::{beam::Beam, *};
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
/// `coarse_chan_freqs`: The centre frequencies of each of the coarse channels
/// of this observation [Hz].
///
/// Assume an ideal array (all dipoles with unity gain). Also assume that the
/// observation does not elapse enough time to shift sources into beam nulls
/// compared to the obs start.
pub(crate) fn veto_sources(
    source_list: &mut SourceList,
    phase_centre: &RADec,
    lst_rad: f64,
    array_latitude_rad: f64,
    coarse_chan_freqs: &[f64],
    beam: &Box<dyn Beam>,
    num_sources: Option<usize>,
    source_dist_cutoff: f64,
    veto_threshold: f64,
) -> Result<Vec<RankedSource>, VetoError> {
    let phase_azel = phase_centre.to_hadec(lst_rad).to_azel(array_latitude_rad);

    // We want to store the flux densities for each source at the "mean
    // frequency". Store the middle coarse channel number to achieve this later.
    let average_freq = coarse_chan_freqs.iter().sum::<f64>() / coarse_chan_freqs.len() as f64;

    // Do we need to consider the source distances? Determine this from the
    // final number of sources requested.
    let dist_cutoff: Option<f64> = {
        match num_sources {
            Some(n) => {
                if source_list.len() <= n {
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
                warn!("The number of sources to use was not specified; using a distance cutoff of {} degrees to filter sources", source_dist_cutoff);
                Some(source_dist_cutoff.to_radians())
            }
        }
    };

    let (vetoed_sources, not_vetoed_sources): (Vec<Result<String, VetoError>>, Vec<Result<RankedSource, EstimateError>>) = source_list
        .par_iter()
        .partition_map(|(source_name, source)| {
            // Are any of this source's components too low in elevation? Or too
            // far from the pointing centre?
            for comp in &source.components {
                let azel = comp.radec.to_hadec(lst_rad).to_azel(array_latitude_rad);
                if azel.el < ELEVATION_LIMIT {
                    trace!("A component's elevation ({} radians, source {}) was below the limit ({} radians)",
                           azel.el,
                           source_name,
                           ELEVATION_LIMIT);
                    return Either::Left(Ok(source_name.clone()));
                } else if let Some(d) = dist_cutoff {
                    if comp.radec.separation(&phase_centre) > d {
                    trace!("A component (source {}) was too far from the pointing centre (separation {} radians)",
                           source_name,
                           d);
                    return Either::Left(Ok(source_name.clone()));
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
            for &cc_freq in coarse_chan_freqs {
                // `fd` is the sum of the source's component flux densities.
                let mut fd = 0.0;

                // Get the beam response at this source position and frequency.
                let j = match beam.calc_jones(
                        &phase_azel,
                        cc_freq as _,
                        &[1.0; 16]) {
                        Ok(j) => j,
                        Err(e) => return Either::Left(Err(e.into())),
                    };

                let source_fds = match source.get_flux_estimates(cc_freq) {
                    Ok(fds) => fds,
                    Err(e) => return Either::Left(Err(e.into())),
                };

                for source_fd in source_fds {
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
                    return Either::Left(Ok(source_name.clone()));
                }
                smallest_fd = fd.min(smallest_fd);
            }

            // If we got this far, the source should not be vetoed.
            let ranked_source  = RankedSource::new(source_name.to_owned(),
            smallest_fd,
            cc_fds,
            source,
            average_freq
            );
            Either::Right(ranked_source)
        });

    // Handle potential errors while vetoing (such as the beam code failing).
    let vetoed_sources = vetoed_sources.into_iter().collect::<Result<Vec<_>, _>>()?;
    let mut not_vetoed_sources = not_vetoed_sources
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;

    debug!(
        "{} sources were vetoed from the source list",
        vetoed_sources.len()
    );
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

/// Convert a Stokes flux densities into instrumental flux densities, and
/// multiply by a beam-response Jones matrix. Return the sum of the response XX
/// and YY flux densities as the "beam attenuated flux density".
// This function is isolated for testing.
#[inline(always)]
fn get_beam_attenuated_flux_density(fd: &FluxDensity, j: &Jones<f64>) -> f64 {
    // Get the instrumental flux densities as a Jones matrix.
    let i = Jones::from(fd);
    // Calculate: J . I . J^H
    // where J is the beam-response Jones matrix and I are the instrumental flux
    // densities.
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
    Beam(#[from] crate::beam::BeamError),

    #[error("{0}")]
    Estimate(#[from] mwa_hyperdrive_core::flux_density::EstimateError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::precession::get_unprecessed_lmst;
    use crate::tests::*;
    use mwa_hyperdrive_core::mwalib::MWA_LONGITUDE_RADIANS;
    use reduced_obsids::get_1090008640_smallest;

    #[test]
    #[serial]
    fn test_beam_attenuated_flux_density() {
        let mut args = get_1090008640_smallest();
        args.no_beam = false;
        let params = args.into_params().unwrap();
        let obs_context = params.input_data.get_obs_context();
        let phase = &obs_context.phase_centre;
        let lmst = get_unprecessed_lmst(&obs_context.timesteps[0], MWA_LONGITUDE_RADIANS);
        let phase_azel = phase.to_hadec(lmst).to_azel_mwa();

        let jones_pointing_centre = params
            .beam
            .calc_jones(&phase_azel, 180e6 as _, &[1.0; 16])
            .unwrap();
        let radec_null =
            RADec::new_degrees(phase.ra.to_degrees() + 80.0, phase.dec.to_degrees() + 80.0);
        assert!(radec_null.dec.to_degrees() > -90.0);
        assert!(radec_null.dec.to_degrees() < 90.0);
        let azel_null = radec_null.to_hadec(lmst).to_azel_mwa();
        let jones_null = params
            .beam
            .calc_jones(&azel_null, 180e6 as _, &[1.0; 16])
            .unwrap();
        let fd = FluxDensity {
            freq: 180e6,
            i: 1.0,
            q: 0.0,
            u: 0.0,
            v: 0.0,
        };
        let bafd_pc = get_beam_attenuated_flux_density(&fd, &jones_pointing_centre);
        assert_abs_diff_eq!(bafd_pc, 1.9822303442965272, epsilon = 1e-10);

        let bafd_null = get_beam_attenuated_flux_density(&fd, &jones_null);
        assert_abs_diff_eq!(bafd_null, 0.0000000005317170873153842, epsilon = 1e-10);
    }

    #[test]
    #[serial]
    fn veto() {
        // let (metafits, beam, mut source_list) = get_params();
        let mut args = get_1090008640_smallest();
        args.no_beam = false;
        let mut params = args.into_params().unwrap();
        let obs_context = params.input_data.get_obs_context();
        let lmst = get_unprecessed_lmst(&obs_context.timesteps[0], MWA_LONGITUDE_RADIANS);
        // For testing's sake, keep only the following bright sources.
        let sources = &[
            "J002549-260211",
            "J004616-420739",
            "J233426-412520",
            "J235701-344532",
        ];
        let keys: Vec<String> = params.source_list.keys().cloned().collect();
        for source_name in keys {
            if !sources.contains(&source_name.as_str()) {
                params.source_list.remove(source_name.as_str());
            }
        }

        // Add some sources that are in beam nulls. Despite being very bright,
        // they should be vetoed.
        params.source_list.insert(
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
        params.source_list.insert(
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
        params.source_list.insert(
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
            &mut params.source_list,
            &obs_context.phase_centre,
            lmst,
            MWA_LAT_RAD,
            &params.input_data.get_freq_context().coarse_chan_freqs,
            &params.beam,
            None,
            180.0,
            20.0,
        );
        assert!(result.is_ok());
        result.unwrap();
        // Only the first four are kept.
        assert_eq!(
            params.source_list.len(),
            4,
            "Expected only five sources to not get vetoed: {:#?}",
            params.source_list.keys()
        );
    }

    #[test]
    #[serial]
    fn top_n_sources() {
        let mut args = get_1090008640_smallest();
        args.no_beam = false;
        let mut params = args.into_params().unwrap();
        let obs_context = params.input_data.get_obs_context();
        let lmst = get_unprecessed_lmst(&obs_context.timesteps[0], MWA_LONGITUDE_RADIANS);

        // For testing's sake, keep only the following sources.
        let sources = &[
            "J000042-342358",
            "J000045-272248",
            "J000105-165921",
            "J000143-305731",
            "J000217-253912",
            "J000245-302825",
        ];
        let keys: Vec<String> = params.source_list.keys().cloned().collect();
        for source_name in keys {
            if !sources.contains(&source_name.as_str()) {
                params.source_list.remove(source_name.as_str());
            }
        }

        let result = veto_sources(
            &mut params.source_list,
            &obs_context.phase_centre,
            lmst,
            MWA_LAT_RAD,
            &params.input_data.get_freq_context().coarse_chan_freqs,
            &params.beam,
            Some(3),
            180.0,
            0.1,
        );
        assert!(result.is_ok(), "{:?}", result.unwrap_err());
        result.unwrap();
        for &name in &["J000042-342358", "J000105-165921", "J000143-305731"] {
            assert!(
                &params.source_list.contains_key(name),
                "Expected to find {} in the source list after vetoing",
                name
            );
        }
    }
}
