// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Writing "André Offringa"-style text source lists.

use log::{debug, warn};
use marlu::sexagesimal::*;

use crate::srclist::{error::WriteSourceListError, ComponentType, FluxDensityType, SourceList};

pub(crate) fn write_source_list<T: std::io::Write>(
    buf: &mut T,
    sl: &SourceList,
    num_sources: Option<usize>,
) -> Result<(), WriteSourceListError> {
    // The AO format can't handle curved-power-law flux types or shapelet
    // components.
    let mut warned_shapelets = false;
    let mut warned_curved_power_laws = false;

    writeln!(buf, "skymodel fileformat 1.1")?;

    let mut num_written_sources = 0;
    // Note that, if sorted, each source in the source list is dimmer than the
    // last!
    for (name, source) in sl.iter() {
        let (any_curved_power_laws, any_shapelets) =
            source.components.iter().fold((false, false), |acc, comp| {
                (
                    acc.0 || matches!(comp.flux_type, FluxDensityType::CurvedPowerLaw { .. }),
                    acc.1 || matches!(comp.comp_type, ComponentType::Shapelet { .. }),
                )
            });
        if any_curved_power_laws {
            if !warned_curved_power_laws {
                warn!("AO source lists don't support curved-power-law flux densities.");
                warn!("Any sources containing them won't be written.");
                warned_curved_power_laws = true;
            }
            debug!("Ignoring source {name} as it contains a curved power law");
            continue;
        }
        if any_shapelets {
            if !warned_shapelets {
                warn!("AO source lists don't support shapelet components.");
                warn!("Any sources containing them won't be written.");
                warned_shapelets = true;
            }
            debug!("Ignoring source {name} as it contains a shapelet component");
            continue;
        }

        // If `num_sources` is supplied, then check that we're not writing out
        // too many sources.
        if let Some(num_sources) = num_sources {
            if num_written_sources == num_sources {
                break;
            }
        }

        writeln!(buf, "source {{")?;
        writeln!(buf, "  name \"{name}\"")?;
        for c in &source.components {
            writeln!(buf, "  component {{")?;

            let (comp_type, shape) = match c.comp_type {
                ComponentType::Point => ("point", None),
                ComponentType::Gaussian { maj, min, pa } => ("gaussian", Some((maj, min, pa))),
                ComponentType::Shapelet { .. } => {
                    return Err(WriteSourceListError::UnsupportedComponentType {
                        source_list_type: "AO",
                        comp_type: "shapelet",
                    })
                }
            };
            writeln!(buf, "    type {comp_type}")?;

            writeln!(
                buf,
                "    position {} {}",
                degrees_to_sexagesimal_hms(c.radec.ra.to_degrees()),
                degrees_to_sexagesimal_dms(c.radec.dec.to_degrees()),
            )?;

            if comp_type == "gaussian" {
                let (maj, min, pa) = shape.unwrap();
                writeln!(
                    buf,
                    "    shape {} {} {}",
                    maj.to_degrees() * 3600.0,
                    min.to_degrees() * 3600.0,
                    pa.to_degrees(),
                )?;
            }

            match &c.flux_type {
                FluxDensityType::PowerLaw { fd, si } => {
                    writeln!(buf, "    sed {{")?;
                    writeln!(buf, "      frequency {} MHz", fd.freq / 1e6)?;
                    writeln!(
                        buf,
                        "      fluxdensity Jy {} {} {} {}",
                        fd.i, fd.q, fd.u, fd.v
                    )?;
                    writeln!(buf, "      spectral-index {{ {si} 0.00 }}")?;
                    writeln!(buf, "    }}")?;
                }

                FluxDensityType::List { fds } => {
                    for fd in fds {
                        writeln!(buf, "    measurement {{")?;
                        writeln!(buf, "      frequency {} MHz", fd.freq / 1e6)?;
                        writeln!(
                            buf,
                            "      fluxdensity Jy {} {} {} {}",
                            fd.i, fd.q, fd.u, fd.v
                        )?;
                        writeln!(buf, "    }}")?;
                    }
                }

                FluxDensityType::CurvedPowerLaw { .. } => {
                    return Err(WriteSourceListError::UnsupportedFluxDensityType {
                        source_list_type: "AO",
                        fd_type: "curved power law",
                    })
                }
            }
            writeln!(buf, "  }}")?;
        }

        writeln!(buf, "}}")?;
        num_written_sources += 1;
    }
    buf.flush()?;

    if let Some(num_sources) = num_sources {
        if num_sources > num_written_sources {
            warn!("Couldn't write the requested number of sources ({num_sources}): wrote {num_written_sources}")
        }
    }

    Ok(())
}
