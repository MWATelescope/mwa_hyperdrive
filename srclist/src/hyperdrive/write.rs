// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to write out hyperdrive source lists.
//!
//! To make the source list files a little easier to read and write,
//! [SourceList] isn't directly serialisable or deserialisable. Use temporary
//! types here to do the serde magic and write out a [SourceList].

use log::warn;

use super::*;
use mwa_hyperdrive_common::{log, serde_json};

fn source_list_to_tmp_sl(sl: &SourceList, num_sources: Option<usize>) -> TmpSourceList {
    let mut tmp_sl: BTreeMap<String, Vec<TmpComponent>> = BTreeMap::new();

    let mut num_written_sources = 0;
    // Note that, if sorted, each source in the source list is dimmer than the
    // last!
    for (name, source) in sl.iter() {
        // If `num_sources` is supplied, then check that we're not writing out
        // too many sources.
        if let Some(num_sources) = num_sources {
            if num_written_sources == num_sources {
                break;
            }
        }

        let mut tmp_comps = Vec::with_capacity(source.components.len());
        for comp in &source.components {
            let comp_type = match &comp.comp_type {
                ComponentType::Point => ComponentType::Point,
                ComponentType::Gaussian { maj, min, pa } => ComponentType::Gaussian {
                    maj: maj.to_degrees() * 3600.0,
                    min: min.to_degrees() * 3600.0,
                    pa: pa.to_degrees(),
                },
                ComponentType::Shapelet {
                    maj,
                    min,
                    pa,
                    coeffs,
                } => ComponentType::Shapelet {
                    maj: maj.to_degrees() * 3600.0,
                    min: min.to_degrees() * 3600.0,
                    pa: pa.to_degrees(),
                    coeffs: coeffs.clone(),
                },
            };

            tmp_comps.push(TmpComponent {
                ra: comp.radec.ra.to_degrees(),
                dec: comp.radec.dec.to_degrees(),
                comp_type,
                flux_type: match &comp.flux_type {
                    FluxDensityType::List { fds } => TmpFluxDensityType::List(fds.to_vec()),
                    FluxDensityType::PowerLaw { si, fd } => TmpFluxDensityType::PowerLaw {
                        si: *si,
                        fd: fd.clone(),
                    },
                    FluxDensityType::CurvedPowerLaw { si, fd, q } => {
                        TmpFluxDensityType::CurvedPowerLaw {
                            si: *si,
                            fd: fd.clone(),
                            q: *q,
                        }
                    }
                },
            })
        }

        tmp_sl.insert(name.clone(), tmp_comps);
        num_written_sources += 1;
    }

    if let Some(num_sources) = num_sources {
        if num_sources > num_written_sources {
            warn!("Couldn't write the requested number of sources ({num_sources}): wrote {num_written_sources}")
        }
    }

    tmp_sl
}

/// Write a `SourceList` to a yaml file.
pub fn source_list_to_yaml<T: std::io::Write>(
    buf: &mut T,
    sl: &SourceList,
    num_sources: Option<usize>,
) -> Result<(), WriteSourceListError> {
    let tmp_sl = source_list_to_tmp_sl(sl, num_sources);
    serde_yaml::to_writer(buf, &tmp_sl)?;
    Ok(())
}

/// Write a `SourceList` to a json file.
pub fn source_list_to_json<T: std::io::Write>(
    buf: &mut T,
    sl: &SourceList,
    num_sources: Option<usize>,
) -> Result<(), WriteSourceListError> {
    let tmp_sl = source_list_to_tmp_sl(sl, num_sources);
    serde_json::to_writer_pretty(buf, &tmp_sl)?;
    Ok(())
}
