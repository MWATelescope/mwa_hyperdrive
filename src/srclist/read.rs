// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Common code for reading sky-model source list files.

use std::{fs::File, path::Path};

use log::{debug, trace, warn};

use super::{error::ReadSourceListError, FluxDensityType, SourceList, SourceListType};
use crate::{
    constants::DEFAULT_SPEC_INDEX,
    srclist::{ao, dp3, fits, hyperdrive, rts, woden},
};

/// Given the path to a sky-model source list file (and optionally its type,
/// e.g. "RTS style"), return a [SourceList] object. The [SourceListType] is
/// also returned in case that's interesting to the caller.
pub(crate) fn read_source_list_file<P: AsRef<Path>>(
    path: P,
    sl_type: Option<SourceListType>,
) -> Result<(SourceList, SourceListType), ReadSourceListError> {
    fn inner(
        path: &Path,
        sl_type: Option<SourceListType>,
    ) -> Result<(SourceList, SourceListType), ReadSourceListError> {
        debug!("Attempting to read source list");

        // If the file extension corresponds to YAML or JSON, we know what to
        // target.
        let ext = path
            .extension()
            .and_then(|os_str| os_str.to_str())
            .map(|s| s.to_lowercase());
        match ext.as_deref() {
            Some("yaml" | "yml") => {
                debug!("Read as hyperdrive yaml");
                let mut f = std::io::BufReader::new(File::open(path)?);
                return hyperdrive::source_list_from_yaml(&mut f)
                    .map(|r| (r, SourceListType::Hyperdrive));
            }
            Some("json") => {
                debug!("Read as hyperdrive json");
                let mut f = std::io::BufReader::new(File::open(path)?);
                return hyperdrive::source_list_from_json(&mut f)
                    .map(|r| (r, SourceListType::Hyperdrive));
            }
            _ => (),
        }

        // We're guessing what the format is here.
        match sl_type {
            Some(SourceListType::Hyperdrive) => {
                // Can be either yaml or json.
                let mut f = std::io::BufReader::new(File::open(path)?);
                let json_err = match hyperdrive::source_list_from_json(&mut f) {
                    Ok(sl) => return Ok((sl, SourceListType::Hyperdrive)),
                    Err(e @ ReadSourceListError::Json(_)) => e.to_string(),
                    Err(e) => return Err(e),
                };
                let mut f = std::io::BufReader::new(File::open(path)?);
                let yaml_err = match hyperdrive::source_list_from_yaml(&mut f) {
                    Ok(sl) => return Ok((sl, SourceListType::Hyperdrive)),
                    // If there was an error in parsing, then pass the parse
                    // error along and try JSON.
                    Err(e @ ReadSourceListError::Yaml(_)) => e.to_string(),
                    Err(e) => return Err(e),
                };
                Err(ReadSourceListError::FailedToDeserialise { yaml_err, json_err })
            }

            Some(SourceListType::Fits) => match fits::parse_source_list(path) {
                Ok(sl) => Ok((sl, SourceListType::Fits)),
                Err(e) => panic!("{:?}", e),
            },

            Some(SourceListType::Rts) => {
                let mut f = std::io::BufReader::new(File::open(path)?);
                match rts::parse_source_list(&mut f) {
                    Ok(sl) => Ok((sl, SourceListType::Rts)),
                    Err(e) => Err(e),
                }
            }

            Some(SourceListType::AO) => {
                let mut f = std::io::BufReader::new(File::open(path)?);
                match ao::parse_source_list(&mut f) {
                    Ok(sl) => Ok((sl, SourceListType::AO)),
                    Err(e) => Err(e),
                }
            }

            Some(SourceListType::Dp3) => {
                let mut f = std::io::BufReader::new(File::open(path)?);
                dp3::parse_source_list(&mut f).map(|sl| (sl, SourceListType::Dp3))
            }

            Some(SourceListType::Woden) => {
                let mut f = std::io::BufReader::new(File::open(path)?);
                match woden::parse_source_list(&mut f) {
                    Ok(sl) => Ok((sl, SourceListType::Woden)),
                    Err(e) => Err(e),
                }
            }

            None => {
                // DP3/BBS sky models (usually with a .skymodel or .txt
                // extension) are identified by their format line; if there is
                // one, then any error reading the file is the interesting one.
                if dp3::looks_like_dp3(path) {
                    debug!("Read as a DP3/BBS sky model");
                    let mut f = std::io::BufReader::new(File::open(path)?);
                    return dp3::parse_source_list(&mut f).map(|sl| (sl, SourceListType::Dp3));
                }
                trace!("Source list does not look like a DP3/BBS sky model (no format line)");

                // Try all kinds.
                match fits::parse_source_list(path) {
                    Ok(sl) => return Ok((sl, SourceListType::Fits)),
                    Err(_) => trace!("Failed to read source list as a fits file"),
                }

                let mut f = std::io::BufReader::new(File::open(path)?);
                match rts::parse_source_list(&mut f) {
                    Ok(sl) => return Ok((sl, SourceListType::Rts)),
                    Err(_) => {
                        trace!("Failed to read source list as rts-style");
                        // Even a failed attempt to read the file alters the buffer. Open it
                        // again.
                        f = std::io::BufReader::new(File::open(path)?);
                    }
                }
                match ao::parse_source_list(&mut f) {
                    Ok(sl) => return Ok((sl, SourceListType::AO)),
                    Err(_) => {
                        trace!("Failed to read source list as ao-style");
                        f = std::io::BufReader::new(File::open(path)?);
                    }
                }
                match woden::parse_source_list(&mut f) {
                    Ok(sl) => return Ok((sl, SourceListType::Woden)),
                    Err(_) => {
                        trace!("Failed to read source list as woden-style");
                        f = std::io::BufReader::new(File::open(path)?);
                    }
                }
                match hyperdrive::source_list_from_json(&mut f) {
                    Ok(sl) => return Ok((sl, SourceListType::Hyperdrive)),
                    Err(_) => {
                        trace!("Failed to read source list as hyperdrive-style json");
                        f = std::io::BufReader::new(File::open(path)?);
                    }
                }
                match hyperdrive::source_list_from_yaml(&mut f) {
                    Ok(sl) => return Ok((sl, SourceListType::Hyperdrive)),
                    Err(_) => {
                        trace!("Failed to read source list as hyperdrive-style yaml");
                    }
                }
                Err(ReadSourceListError::FailedToReadAsAnyType)
            }
        }
    }
    let (source_list, source_list_type) = inner(path.as_ref(), sl_type)?;
    let n = count_single_entry_flux_lists(&source_list);
    if n > 0 {
        warn!(
            "{n} source-list component{} {} a flux-density list with a single entry; \
             such components are extrapolated with a spectral index of {DEFAULT_SPEC_INDEX} \
             (write a power law with a spectral index of 0 for a flat spectrum)",
            if n == 1 { "" } else { "s" },
            if n == 1 { "has" } else { "have" }
        );
    }
    Ok((source_list, source_list_type))
}

/// The number of components whose flux density is a list with one entry.
/// Those can't be interpolated, so hyperdrive extrapolates them with
/// [`DEFAULT_SPEC_INDEX`], which is easy to miss.
fn count_single_entry_flux_lists(source_list: &SourceList) -> usize {
    source_list
        .values()
        .flat_map(|source| source.components.iter())
        .filter(|comp| matches!(&comp.flux_type, FluxDensityType::List(fds) if fds.len() == 1))
        .count()
}

#[cfg(test)]
mod tests {
    use marlu::RADec;
    use vec1::vec1;

    use super::*;
    use crate::srclist::{ComponentType, FluxDensity, Source, SourceComponent};

    fn comp(flux_type: FluxDensityType) -> SourceComponent {
        SourceComponent {
            radec: RADec::from_degrees(0.0, -27.0),
            comp_type: ComponentType::Point,
            flux_type,
        }
    }

    fn fd(freq: f64) -> FluxDensity {
        FluxDensity {
            freq,
            i: 1.0,
            q: 0.0,
            u: 0.0,
            v: 0.0,
        }
    }

    #[test]
    fn single_entry_flux_lists_are_counted() {
        let mut sl = SourceList::new();
        sl.insert(
            "one".to_string(),
            Source {
                components: vec![
                    comp(FluxDensityType::List(vec1![fd(150e6)])),
                    comp(FluxDensityType::List(vec1![fd(150e6), fd(180e6)])),
                    comp(FluxDensityType::PowerLaw {
                        si: 0.0,
                        fd: fd(150e6),
                    }),
                ]
                .into_boxed_slice(),
            },
        );
        sl.insert(
            "two".to_string(),
            Source {
                components: vec![comp(FluxDensityType::List(vec1![fd(150e6)]))].into_boxed_slice(),
            },
        );
        assert_eq!(count_single_entry_flux_lists(&sl), 2);
        assert_eq!(count_single_entry_flux_lists(&SourceList::new()), 0);
    }
}
