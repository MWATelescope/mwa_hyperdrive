// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Writing WODEN-style text source lists.
//!
//! WODEN only allows for a single flux density per component. For this reason,
//! only the first flux density in a list of flux densities will be written
//! here.

use log::debug;

use crate::{
    cli::Warn,
    srclist::{error::WriteSourceListError, ComponentType, FluxDensityType, SourceList},
};

fn write_comp_type<T: std::io::Write>(
    buf: &mut T,
    comp_type: &ComponentType,
) -> Result<(), WriteSourceListError> {
    match comp_type {
        ComponentType::Point => (),

        ComponentType::Gaussian { maj, min, pa } => writeln!(
            buf,
            "GPARAMS {} {} {}",
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
                "SPARAMS {} {} {}",
                pa.to_degrees(),
                maj.to_degrees() * 60.0,
                min.to_degrees() * 60.0
            )?;
            for c in coeffs.iter() {
                writeln!(buf, "SCOEFF {} {} {}", c.n1, c.n2, c.value)?;
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
            // Only use the first. WODEN can't use multiple.
            let fd = &fds[0];
            writeln!(
                buf,
                "FREQ {:+e} {} {} {} {}",
                fd.freq, fd.i, fd.q, fd.u, fd.v
            )?;
        }

        FluxDensityType::PowerLaw { fd, si } => {
            writeln!(
                buf,
                "LINEAR {:+e} {} {} {} {} {}",
                fd.freq, fd.i, fd.q, fd.u, fd.v, si
            )?;
        }

        FluxDensityType::CurvedPowerLaw { .. } => {
            return Err(WriteSourceListError::UnsupportedFluxDensityType {
                source_list_type: "WODEN",
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
    // The WODEN format can't handle curved-power-law flux types.
    let mut warned_curved_power_laws = false;

    let mut num_written_sources = 0;
    // Note that, if sorted, each source in the source list is dimmer than the
    // last!
    for (name, source) in sl.iter() {
        if source
            .components
            .iter()
            .any(|comp| matches!(comp.flux_type, FluxDensityType::CurvedPowerLaw { .. }))
        {
            if !warned_curved_power_laws {
                [
                    "WODEN source lists don't support curved-power-law flux densities.".into(),
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

        // Get the counts of each type of component.
        let mut num_points = 0;
        let mut num_gaussians = 0;
        let mut num_shapelets = 0;
        let mut num_shapelet_coeffs = 0;
        for comp in source.components.iter() {
            match &comp.comp_type {
                ComponentType::Point => num_points += 1,
                ComponentType::Gaussian { .. } => num_gaussians += 1,
                ComponentType::Shapelet { coeffs, .. } => {
                    num_shapelets += 1;
                    num_shapelet_coeffs += coeffs.len() as u32;
                }
            }
        }

        writeln!(
            buf,
            "SOURCE {} P {} G {} S {} {}",
            name.replace(' ', "_"),
            num_points,
            num_gaussians,
            num_shapelets,
            num_shapelet_coeffs
        )?;

        // Write out the components.
        for comp in source.components.iter() {
            let comp_type_str = match comp.comp_type {
                ComponentType::Point => "POINT",
                ComponentType::Gaussian { .. } => "GAUSSIAN",
                ComponentType::Shapelet { .. } => "SHAPELET",
            };

            writeln!(
                buf,
                "COMPONENT {} {} {}",
                comp_type_str,
                comp.radec.ra.to_degrees() / 15.0,
                comp.radec.dec.to_degrees()
            )?;

            write_flux_type(buf, &comp.flux_type)?;
            write_comp_type(buf, &comp.comp_type)?;

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
