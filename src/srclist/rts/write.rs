// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Writing RTS-style text source lists.

use log::debug;

use crate::{
    cli::Warn,
    srclist::{ComponentType, FluxDensityType, SourceList, WriteSourceListError},
};

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
            for c in coeffs.iter() {
                writeln!(buf, "COEFF {} {} {}", c.n1, c.n2, c.value)?;
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
        FluxDensityType::List(fds) => {
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
            let fd_150 = flux_type.estimate_at_freq(150e6);
            let fd_200 = flux_type.estimate_at_freq(200e6);
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
                source_list_type: "RTS",
                fd_type: "curved power law",
            })
        }
    }

    Ok(())
}

pub(crate) fn write_source_list<T: std::io::Write>(
    buf: &mut T,
    sl: &SourceList,
    num_sources: Option<usize>,
) -> Result<(), WriteSourceListError> {
    // The RTS format can't handle curved-power-law flux types.
    let mut warned_curved_power_laws = false;

    let mut num_written_sources = 0;
    for (name, source) in sl.iter() {
        if source
            .components
            .iter()
            .any(|comp| matches!(comp.flux_type, FluxDensityType::CurvedPowerLaw { .. }))
        {
            if !warned_curved_power_laws {
                [
                    "RTS source lists don't support curved-power-law flux densities.".into(),
                    "Any sources containing them won't be written.".into(),
                ]
                .warn();
                warned_curved_power_laws = true;
            }
            debug!("Ignoring source {name} as it contains a curved power law");
            continue;
        }

        // If `num_sources` is supplied, then check that we're not writing out
        // too many sources.
        if let Some(num_sources) = num_sources {
            if num_written_sources == num_sources {
                break;
            }
        }

        // Write out the first component as the RTS base source.
        let first_comp = source
            .components
            .first()
            .expect("source's components aren't empty");
        writeln!(
            buf,
            "SOURCE {} {} {}",
            name.replace(' ', "_"),
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
        num_written_sources += 1;
    }
    buf.flush()?;

    if let Some(num_sources) = num_sources {
        if num_sources > num_written_sources {
            format!("Couldn't write the requested number of sources ({num_sources}): wrote {num_written_sources}").warn()
        }
    }

    Ok(())
}

/// Write an RTS-style source list, but with the sources ordered according to
/// `source_name_order`.
pub(crate) fn write_source_list_with_order<T: std::io::Write>(
    buf: &mut T,
    sl: &SourceList,
    source_name_order: Vec<String>,
) -> Result<(), WriteSourceListError> {
    // The RTS format can't handle curved-power-law flux types.
    let mut warned_curved_power_laws = false;

    for name in source_name_order {
        let source = &sl[&name];

        if source
            .components
            .iter()
            .any(|comp| matches!(comp.flux_type, FluxDensityType::CurvedPowerLaw { .. }))
        {
            if !warned_curved_power_laws {
                [
                    "RTS source lists don't support curved-power-law flux densities.".into(),
                    "Any sources containing them won't be written.".into(),
                ]
                .warn();
                warned_curved_power_laws = true;
            }
            debug!("Ignoring source {name} as it contains a curved power law");
            continue;
        }

        // Write out the first component as the RTS base source.
        let first_comp = source
            .components
            .first()
            .expect("source's components aren't empty");
        writeln!(
            buf,
            "SOURCE {} {} {}",
            name.replace(' ', "_"),
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
    buf.flush()?;

    Ok(())
}
