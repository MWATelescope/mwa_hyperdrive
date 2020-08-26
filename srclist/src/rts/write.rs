// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Writing RTS-style text source lists.
 */

use super::*;

fn write_comp_type<T: std::io::Write>(
    buf: &mut T,
    comp_type: &ComponentType,
) -> Result<(), WriteSourceListError> {
    match comp_type {
        ComponentType::Point => (),

        ComponentType::Gaussian { maj, min, pa } => writeln!(
            buf,
            "GAUSSIAN {} {} {}",
            pa.to_degrees(),
            maj.to_degrees() * 60.0,
            min.to_degrees() * 60.0
        )?,

        ComponentType::Shapelet {
            maj,
            min,
            pa,
            coeffs,
        } => {
            writeln!(
                buf,
                "SHAPELET2 {} {} {}",
                pa.to_degrees(),
                maj.to_degrees() * 60.0,
                min.to_degrees() * 60.0
            )?;
            for c in coeffs {
                writeln!(buf, "COEFF {} {} {}", c.n1, c.n2, c.coeff)?;
            }
        }
    }

    Ok(())
}

fn write_flux_type<T: std::io::Write>(
    buf: &mut T,
    flux_type: &FluxDensityType,
) -> Result<(), WriteSourceListError> {
    match &flux_type {
        FluxDensityType::List { fds } => {
            for fd in fds {
                writeln!(
                    buf,
                    "FREQ {:+e} {} {} {} {}",
                    fd.freq, fd.i, fd.q, fd.u, fd.v
                )?;
            }
        }

        // If there are only two "list type" flux densities, the RTS uses them
        // as a power law.
        FluxDensityType::PowerLaw { .. } => {
            let fd_150 = flux_type.estimate_at_freq(150e6)?;
            let fd_200 = flux_type.estimate_at_freq(200e6)?;
            for fd in &[fd_150, fd_200] {
                writeln!(
                    buf,
                    "FREQ {:+e} {} {} {} {}",
                    fd.freq, fd.i, fd.q, fd.u, fd.v
                )?;
            }
        }

        FluxDensityType::CurvedPowerLaw { .. } => {
            return Err(WriteSourceListError::UnsupportedFluxDensityType {
                source_list_type: "RTS".to_string(),
                fd_type: "curved power law".to_string(),
            })
        }
    }

    Ok(())
}

pub fn write_source_list<T: std::io::Write>(
    buf: &mut T,
    sl: &SourceList,
) -> Result<(), WriteSourceListError> {
    for (name, source) in sl {
        // Write out the first component as the RTS base source.
        let first_comp = match source.components.first() {
            Some(c) => c,
            None => return Err(WriteSourceListError::NoComponents(name.clone())),
        };
        writeln!(
            buf,
            "SOURCE {} {} {}",
            name,
            first_comp.radec.ra.to_degrees() / 15.0,
            first_comp.radec.dec.to_degrees()
        )?;

        write_comp_type(buf, &first_comp.comp_type)?;
        write_flux_type(buf, &first_comp.flux_type)?;

        // Write out any other components.
        for comp in source.components.iter().skip(1) {
            writeln!(
                buf,
                "COMPONENT {} {}",
                comp.radec.ra.to_degrees() / 15.0,
                comp.radec.dec.to_degrees()
            )?;

            write_comp_type(buf, &comp.comp_type)?;
            write_flux_type(buf, &comp.flux_type)?;

            writeln!(buf, "ENDCOMPONENT")?;
        }

        writeln!(buf, "ENDSOURCE")?;
    }

    Ok(())
}
