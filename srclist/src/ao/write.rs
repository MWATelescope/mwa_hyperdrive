// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Writing "André Offringa"-style text source lists.

use marlu::sexagesimal::*;

use super::*;

pub fn write_source_list<T: std::io::Write>(
    buf: &mut T,
    sl: &SourceList,
) -> Result<(), WriteSourceListError> {
    writeln!(buf, "skymodel fileformat 1.1")?;

    for (name, source) in sl.iter() {
        writeln!(buf, "source {{")?;
        writeln!(buf, "  name \"{}\"", name)?;
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
            writeln!(buf, "    type {}", comp_type)?;

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
                    writeln!(buf, "      spectral-index {{ {} 0.00 }}", si)?;
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
    }
    buf.flush()?;

    Ok(())
}
