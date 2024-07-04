// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Parsing of RTS source lists.
//!
//! See for more info:
//! <https://mwatelescope.github.io/mwa_hyperdrive/defs/source_list_rts.html>

use marlu::RADec;
use vec1::vec1;

use crate::{
    cli::Warn,
    srclist::{
        error::{ReadSourceListCommonError, ReadSourceListError, ReadSourceListRtsError},
        ComponentType, FluxDensity, FluxDensityType, ShapeletCoeff, Source, SourceComponent,
        SourceList,
    },
};

/// Parse a buffer containing an RTS-style source list into a `SourceList`.
pub(crate) fn parse_source_list<T: std::io::BufRead>(
    buf: &mut T,
) -> Result<SourceList, ReadSourceListError> {
    let mut line = String::new();
    let mut line_num: u32 = 0;
    let mut in_source = false;
    let mut in_component = false;
    let mut in_shapelet = false;
    let mut in_shapelet2 = false;
    let mut found_shapelets = vec![];
    let mut component_type_set = false;
    let mut source_name = String::new();
    let mut components: Vec<SourceComponent> = vec![];
    let mut shapelet_coeffs = vec![];
    let mut source_list = SourceList::new();

    let parse_float = |string: &str, line_num: u32| -> Result<f64, ReadSourceListCommonError> {
        string
            .parse()
            .map_err(|_| ReadSourceListCommonError::ParseFloatError {
                line_num,
                string: string.to_string(),
            })
    };

    let float_to_int = |float: f64, line_num: u32| -> Result<usize, ReadSourceListCommonError> {
        if float < 0.0 || float > u8::MAX as f64 {
            Err(ReadSourceListCommonError::FloatToIntError { line_num, float })
        } else {
            Ok(float as _)
        }
    };

    while buf.read_line(&mut line)? > 0 {
        line_num += 1;

        // Handle lines that aren't intended to parsed (comments and blank
        // lines).
        if line.starts_with('#') | line.starts_with('\n') {
            line.clear();
            continue;
        }
        // We ignore any lines starting with whitespace, but emit a warning.
        else if line.starts_with(' ') | line.starts_with('\t') {
            format!(
                "Source list line {} starts with whitespace; ignoring it",
                line_num
            )
            .warn();
            line.clear();
            continue;
        }

        let mut items = line.split_ascii_whitespace();
        match items.next() {
            Some("SOURCE") => {
                if in_source {
                    return Err(ReadSourceListCommonError::NestedSources(line_num).into());
                } else {
                    in_source = true;

                    // SOURCE lines must have at least 4 elements (including SOURCE).
                    match items.next() {
                        Some(name) => source_name.push_str(name),
                        None => {
                            return Err(
                                ReadSourceListCommonError::IncompleteSourceLine(line_num).into()
                            )
                        }
                    };
                    let hour_angle = match items.next() {
                        Some(ha) => parse_float(ha, line_num)?,
                        None => {
                            return Err(
                                ReadSourceListCommonError::IncompleteSourceLine(line_num).into()
                            )
                        }
                    };
                    let declination = match items.next() {
                        Some(dec) => parse_float(dec, line_num)?,
                        None => {
                            return Err(
                                ReadSourceListCommonError::IncompleteSourceLine(line_num).into()
                            )
                        }
                    };
                    if items.next().is_some() {
                        format!(
                            "Source list line {}: Ignoring trailing contents after declination",
                            line_num
                        )
                        .warn();
                    }

                    // Validation and conversion.
                    if !(0.0..=24.0).contains(&hour_angle) {
                        return Err(ReadSourceListError::InvalidHa(hour_angle));
                    }
                    if !(-90.0..=90.0).contains(&declination) {
                        return Err(ReadSourceListError::InvalidDec(declination));
                    }

                    components.push(SourceComponent {
                        radec: RADec::from_degrees(hour_angle * 15.0, declination),
                        // Assume the base source is a point source. If we find
                        // component type information, we can overwrite this.
                        comp_type: ComponentType::Point,
                        // RTS source lists do not handle curved power laws. We
                        // use this here as a placeholder stating that it must
                        // be changed by the time we have finished reading in
                        // the component.
                        flux_type: FluxDensityType::CurvedPowerLaw {
                            si: 0.0,
                            fd: FluxDensity::default(),
                            q: 0.0,
                        },
                    });
                }
            }

            // Flux density line.
            Some("FREQ") => {
                if !in_source {
                    return Err(ReadSourceListCommonError::OutsideSource {
                        line_num,
                        keyword: "FREQ",
                    }
                    .into());
                }

                // FREQ lines must have at least 6 elements (including FREQ).
                let freq_hz = match items.next() {
                    Some(f) => parse_float(f, line_num)?,
                    None => {
                        return Err(ReadSourceListCommonError::IncompleteFluxLine(line_num).into())
                    }
                };
                let stokes_i = match items.next() {
                    Some(f) => parse_float(f, line_num)?,
                    None => {
                        return Err(ReadSourceListCommonError::IncompleteFluxLine(line_num).into())
                    }
                };
                let stokes_q = match items.next() {
                    Some(f) => parse_float(f, line_num)?,
                    None => {
                        return Err(ReadSourceListCommonError::IncompleteFluxLine(line_num).into())
                    }
                };
                let stokes_u = match items.next() {
                    Some(f) => parse_float(f, line_num)?,
                    None => {
                        return Err(ReadSourceListCommonError::IncompleteFluxLine(line_num).into())
                    }
                };
                let stokes_v = match items.next() {
                    Some(f) => parse_float(f, line_num)?,
                    None => {
                        return Err(ReadSourceListCommonError::IncompleteFluxLine(line_num).into())
                    }
                };
                if items.next().is_some() {
                    format!(
                        "Source list line {}: Ignoring trailing contents after Stokes V",
                        line_num
                    )
                    .warn();
                }

                if stokes_i.is_nan() || stokes_q.is_nan() || stokes_u.is_nan() || stokes_v.is_nan()
                {
                    return Err(ReadSourceListError::NaNsInComponent { source_name });
                }

                let fd = FluxDensity {
                    freq: freq_hz,
                    i: stokes_i,
                    q: stokes_q,
                    u: stokes_u,
                    v: stokes_v,
                };

                match components.iter_mut().last().map(|c| &mut c.flux_type) {
                    Some(FluxDensityType::List(fds)) => fds.push(fd),
                    Some(c @ FluxDensityType::CurvedPowerLaw { .. }) => {
                        *c = FluxDensityType::List(vec1![fd])
                    }
                    _ => unreachable!(),
                }
            }

            // Gaussian component type.
            Some("GAUSSIAN") => {
                if !in_source {
                    return Err(ReadSourceListCommonError::OutsideSource {
                        line_num,
                        keyword: "GAUSSIAN",
                    }
                    .into());
                }

                // GAUSSIAN lines must have at least 4 elements (including
                // GAUSSIAN).
                let mut position_angle = match items.next() {
                    Some(f) => parse_float(f, line_num)?,
                    None => {
                        return Err(ReadSourceListRtsError::IncompleteGaussianLine(line_num).into())
                    }
                };
                let maj_arcmin = match items.next() {
                    Some(f) => parse_float(f, line_num)?,
                    None => {
                        return Err(ReadSourceListRtsError::IncompleteGaussianLine(line_num).into())
                    }
                };
                let min_arcmin = match items.next() {
                    Some(f) => parse_float(f, line_num)?,
                    None => {
                        return Err(ReadSourceListRtsError::IncompleteGaussianLine(line_num).into())
                    }
                };
                if items.next().is_some() {
                    format!(
                        "Source list line {}: Ignoring trailing contents after minor axis",
                        line_num
                    )
                    .warn();
                }

                // Ensure the position angle is positive.
                if position_angle < 0.0 {
                    position_angle += 360.0;
                }

                let comp_type = ComponentType::Gaussian {
                    maj: maj_arcmin.to_radians() / 60.0,
                    min: min_arcmin.to_radians() / 60.0,
                    pa: position_angle.to_radians(),
                };

                // Have we already set the component type?
                if component_type_set {
                    return Err(ReadSourceListCommonError::MultipleComponentTypes(line_num).into());
                }
                components.iter_mut().last().unwrap().comp_type = comp_type;
                component_type_set = true;
            }

            // Shapelet component type.
            Some("SHAPELET2") => {
                if !in_source {
                    return Err(ReadSourceListCommonError::OutsideSource {
                        line_num,
                        keyword: "SHAPELET2",
                    }
                    .into());
                }

                // SHAPELET2 lines must have at least 4 elements (including
                // SHAPELET2).
                let mut position_angle = match items.next() {
                    Some(f) => parse_float(f, line_num)?,
                    None => {
                        return Err(ReadSourceListRtsError::IncompleteShapelet2Line(line_num).into())
                    }
                };
                let maj_arcmin = match items.next() {
                    Some(f) => parse_float(f, line_num)?,
                    None => {
                        return Err(ReadSourceListRtsError::IncompleteShapelet2Line(line_num).into())
                    }
                };
                let min_arcmin = match items.next() {
                    Some(f) => parse_float(f, line_num)?,
                    None => {
                        return Err(ReadSourceListRtsError::IncompleteShapelet2Line(line_num).into())
                    }
                };
                if items.next().is_some() {
                    format!(
                        "Source list line {}: Ignoring trailing contents after minor axis",
                        line_num
                    )
                    .warn();
                }

                // Ensure the position angle is positive.
                if position_angle < 0.0 {
                    position_angle += 360.0;
                }

                let comp_type = ComponentType::Shapelet {
                    maj: maj_arcmin.to_radians() / 60.0,
                    min: min_arcmin.to_radians() / 60.0,
                    pa: position_angle.to_radians(),
                    coeffs: vec![].into_boxed_slice(),
                };

                // Have we already set the component type?
                if component_type_set {
                    return Err(ReadSourceListCommonError::MultipleComponentTypes(line_num).into());
                }
                components.iter_mut().last().unwrap().comp_type = comp_type;
                component_type_set = true;
                in_shapelet2 = true;
            }

            // "Jenny's shapelet type" - not to be used any more.
            Some("SHAPELET") => {
                if !in_source {
                    return Err(ReadSourceListCommonError::OutsideSource {
                        line_num,
                        keyword: "SHAPELET",
                    }
                    .into());
                }

                // Make the component type a "dummy shapelet component",
                // identifiable by its position angle of -999.
                let comp_type = ComponentType::Shapelet {
                    maj: 0.0,
                    min: 0.0,
                    pa: -999.0,
                    coeffs: vec![].into_boxed_slice(),
                };
                components.iter_mut().last().unwrap().comp_type = comp_type;

                format!("Source list line {}: Ignoring SHAPELET component", line_num).warn();
                component_type_set = true;
                in_shapelet = true;
            }

            // Shapelet coefficient.
            Some("COEFF") => {
                if !in_source {
                    return Err(ReadSourceListCommonError::OutsideSource {
                        line_num,
                        keyword: "COEFF",
                    }
                    .into());
                }

                // Did we parse a SHAPELET or SHAPELET2 line earlier?
                if !in_shapelet && !in_shapelet2 {
                    return Err(ReadSourceListRtsError::MissingShapeletLine(line_num).into());
                }

                // COEFF lines must have at least 4 elements (including COEFF).
                let n1 = match items.next() {
                    Some(f) => parse_float(f, line_num)?,
                    None => {
                        return Err(ReadSourceListRtsError::IncompleteCoeffLine(line_num).into())
                    }
                };
                let n2 = match items.next() {
                    Some(f) => parse_float(f, line_num)?,
                    None => {
                        return Err(ReadSourceListRtsError::IncompleteCoeffLine(line_num).into())
                    }
                };
                let value = match items.next() {
                    Some(f) => parse_float(f, line_num)?,
                    None => {
                        return Err(ReadSourceListRtsError::IncompleteCoeffLine(line_num).into())
                    }
                };
                if items.next().is_some() {
                    format!(
                        "Source list line {}: Ignoring trailing contents after minor axis",
                        line_num
                    )
                    .warn();
                }

                // Because we ignore SHAPELET components, only add this COEFF
                // line if we're dealing with a SHAPELET2.
                if in_shapelet2 {
                    let n1 = float_to_int(n1, line_num)
                        .map_err(|_| ReadSourceListCommonError::ShapeletBasisNotInt(n1))?;
                    let n2 = float_to_int(n2, line_num)
                        .map_err(|_| ReadSourceListCommonError::ShapeletBasisNotInt(n2))?;
                    let shapelet_coeff = ShapeletCoeff {
                        n1: n1
                            .try_into()
                            .expect("shapelet coeff is not bigger than u8::MAX"),
                        n2: n2
                            .try_into()
                            .expect("shapelet coeff is not bigger than u8::MAX"),
                        value,
                    };
                    if !matches!(
                        components.iter_mut().last().unwrap().comp_type,
                        ComponentType::Shapelet { .. }
                    ) {
                        return Err(ReadSourceListRtsError::MissingShapeletLine(line_num).into());
                    }
                    shapelet_coeffs.push(shapelet_coeff);
                }
            }

            // New component.
            Some("COMPONENT") => {
                if !in_source {
                    return Err(ReadSourceListCommonError::OutsideSource {
                        line_num,
                        keyword: "COMPONENT",
                    }
                    .into());
                }

                // Before we start working with a new component, check that the
                // last component struct added actually has flux densities (this
                // struct corresponds to the "base source"). RTS source lists
                // can only have the "list" type.
                match &components.last().unwrap().flux_type {
                    FluxDensityType::List(fds) => {
                        if fds.is_empty() {
                            return Err(ReadSourceListCommonError::NoFluxDensities(line_num).into());
                        }
                    }
                    _ => unreachable!(),
                }
                // If we were reading a SHAPELET2 component, check that shapelet
                // coefficients were read and populate the boxed slice.
                if in_shapelet2 {
                    if shapelet_coeffs.is_empty() {
                        return Err(ReadSourceListRtsError::NoShapeletCoeffs(line_num).into());
                    }
                    match &mut components.last_mut().expect("not empty").comp_type {
                        ComponentType::Shapelet { coeffs, .. } => {
                            *coeffs = shapelet_coeffs.clone().into_boxed_slice();
                            shapelet_coeffs.clear();
                        }
                        _ => unreachable!("should be inside a SHAPELET2"),
                    }
                }

                if in_component {
                    return Err(ReadSourceListCommonError::NestedComponents(line_num).into());
                }
                in_component = true;

                // COMPONENT lines must have at least 3 elements (including COMPONENT).
                let hour_angle = match items.next() {
                    Some(ha) => parse_float(ha, line_num)?,
                    None => {
                        return Err(ReadSourceListCommonError::IncompleteSourceLine(line_num).into())
                    }
                };
                let declination = match items.next() {
                    Some(dec) => parse_float(dec, line_num)?,
                    None => {
                        return Err(ReadSourceListCommonError::IncompleteSourceLine(line_num).into())
                    }
                };
                if items.next().is_some() {
                    format!(
                        "Source list line {}: Ignoring trailing contents after declination",
                        line_num
                    )
                    .warn();
                }

                // Validation and conversion.
                if !(0.0..=24.0).contains(&hour_angle) {
                    return Err(ReadSourceListError::InvalidHa(hour_angle));
                }
                if !(-90.0..=90.0).contains(&declination) {
                    return Err(ReadSourceListError::InvalidDec(declination));
                }
                let radec = RADec::from_degrees(hour_angle * 15.0, declination);

                components.push(SourceComponent {
                    radec,
                    // Assume the base source is a point source. If we find
                    // component type information, we can overwrite this.
                    comp_type: ComponentType::Point,
                    // RTS source lists do not handle curved power laws. We use
                    // this here as a placeholder stating that it must be
                    // changed by the time we have finished reading in the
                    // component.
                    flux_type: FluxDensityType::CurvedPowerLaw {
                        si: 0.0,
                        fd: FluxDensity::default(),
                        q: 0.0,
                    },
                });

                in_shapelet = false;
                in_shapelet2 = false;
                component_type_set = false;
            }

            Some("ENDCOMPONENT") => {
                if !in_component {
                    return Err(ReadSourceListCommonError::EarlyEndComponent(line_num).into());
                }

                // Check that the last component struct added actually has flux
                // densities. RTS source lists can only have the "list" type.
                let comp = components.iter_mut().last().unwrap();
                match &mut comp.flux_type {
                    FluxDensityType::List(fds) => {
                        // Sort the existing flux densities by frequency.
                        fds.sort_unstable_by(|a, b| {
                            a.freq.partial_cmp(&b.freq).unwrap_or_else(|| {
                                panic!("Couldn't compare {} to {}", a.freq, b.freq)
                            })
                        });
                    }
                    FluxDensityType::CurvedPowerLaw { .. } => {
                        return Err(ReadSourceListRtsError::MissingFluxes {
                            line_num,
                            comp_type: match comp.comp_type {
                                ComponentType::Point => "Point",
                                ComponentType::Gaussian { .. } => "Gaussian",
                                ComponentType::Shapelet { .. } => "Shapelet",
                            },
                            ra: comp.radec.ra,
                            dec: comp.radec.dec,
                        }
                        .into());
                    }
                    _ => unreachable!(),
                }

                // If we were reading a SHAPELET2 component, check that shapelet
                // coefficients were read and populate the boxed slice.
                if in_shapelet2 {
                    if shapelet_coeffs.is_empty() {
                        return Err(ReadSourceListRtsError::NoShapeletCoeffs(line_num).into());
                    }
                    match &mut components.last_mut().expect("not empty").comp_type {
                        ComponentType::Shapelet { coeffs, .. } => {
                            *coeffs = shapelet_coeffs.clone().into_boxed_slice();
                            shapelet_coeffs.clear();
                        }
                        _ => unreachable!("should be inside a SHAPELET2"),
                    }
                }

                in_component = false;
                in_shapelet = false;
                in_shapelet2 = false;
                component_type_set = false;
            }

            Some("ENDSOURCE") => {
                if !in_source {
                    return Err(ReadSourceListCommonError::EarlyEndSource(line_num).into());
                } else if in_component {
                    return Err(ReadSourceListCommonError::MissingEndComponent(line_num).into());
                }

                // Ensure that no curved power law placeholders exist.
                for comp in &components {
                    if matches!(comp.flux_type, FluxDensityType::CurvedPowerLaw { .. }) {
                        return Err(ReadSourceListRtsError::MissingFluxes {
                            line_num,
                            comp_type: match comp.comp_type {
                                ComponentType::Point => "Point",
                                ComponentType::Gaussian { .. } => "Gaussian",
                                ComponentType::Shapelet { .. } => "Shapelet",
                            },
                            ra: comp.radec.ra,
                            dec: comp.radec.dec,
                        }
                        .into());
                    }
                }

                let mut source = Source {
                    components: components.clone().into_boxed_slice(),
                };
                components.clear();

                // Find any SHAPELET components (not SHAPELET2 components). If
                // we find one, we ignore it, and we don't need to return an
                // error if this source has no components. Also ensure that the
                // sum of each Stokes' flux densities is positive.
                let mut sum_i = 0.0;
                let mut sum_q = 0.0;
                let mut sum_u = 0.0;
                let mut sum_v = 0.0;
                for (i, c) in source.components.iter().enumerate() {
                    if let ComponentType::Shapelet { pa, .. } = c.comp_type {
                        if (pa + 999.0).abs() < 1e-3 {
                            found_shapelets.push(i);
                        }
                    }

                    match &c.flux_type {
                        FluxDensityType::List(fds) => {
                            for fd in fds {
                                sum_i += fd.i;
                                sum_q += fd.q;
                                sum_u += fd.u;
                                sum_v += fd.v;
                            }
                        }
                        _ => unreachable!(),
                    }
                }
                if sum_i < 0.0 {
                    return Err(ReadSourceListError::InvalidFluxDensitySum {
                        sum: sum_i,
                        stokes_comp: "I",
                        source_name,
                    });
                } else if sum_q < 0.0 {
                    return Err(ReadSourceListError::InvalidFluxDensitySum {
                        sum: sum_q,
                        stokes_comp: "Q",
                        source_name,
                    });
                } else if sum_u < 0.0 {
                    return Err(ReadSourceListError::InvalidFluxDensitySum {
                        sum: sum_u,
                        stokes_comp: "U",
                        source_name,
                    });
                } else if sum_v < 0.0 {
                    return Err(ReadSourceListError::InvalidFluxDensitySum {
                        sum: sum_v,
                        stokes_comp: "V",
                        source_name,
                    });
                }

                // Delete any found shapelets.
                if source.components.len() == found_shapelets.len() {
                    return Err(ReadSourceListCommonError::NoNonShapeletComponents(line_num).into());
                }
                if !found_shapelets.is_empty() {
                    let mut comps = source.components.to_vec();
                    for &i in found_shapelets.iter().rev() {
                        comps.remove(i);
                    }
                    source.components = comps.into_boxed_slice();
                }

                if source.components.is_empty() && found_shapelets.is_empty() {
                    return Err(ReadSourceListCommonError::NoComponents(line_num).into());
                }
                found_shapelets.clear();

                // Check that the last component struct added actually has flux
                // densities. RTS source lists can only have the "list" type.
                if !source.components.is_empty() {
                    match &mut source.components.iter_mut().last().unwrap().flux_type {
                        FluxDensityType::List(fds) => {
                            if fds.is_empty() {
                                return Err(
                                    ReadSourceListCommonError::NoFluxDensities(line_num).into()
                                );
                            } else {
                                // Sort the existing flux densities by frequency.
                                fds.sort_unstable_by(|a, b| {
                                    a.freq.partial_cmp(&b.freq).unwrap_or_else(|| {
                                        panic!("Couldn't compare {} to {}", a.freq, b.freq)
                                    })
                                });
                            }
                        }
                        _ => unreachable!(),
                    }
                }

                // If we were reading a SHAPELET2 component, check that shapelet
                // coefficients were read and populate the boxed slice.
                if in_shapelet2 {
                    if shapelet_coeffs.is_empty() {
                        return Err(ReadSourceListRtsError::NoShapeletCoeffs(line_num).into());
                    }
                    match &mut source.components.last_mut().expect("not empty").comp_type {
                        ComponentType::Shapelet { coeffs, .. } => {
                            *coeffs = shapelet_coeffs.clone().into_boxed_slice();
                            shapelet_coeffs.clear();
                        }
                        _ => unreachable!("should be inside a SHAPELET2"),
                    }
                }

                // If we found SHAPELET components, but now there are no
                // components left, don't add this source to the source list.
                if !source.components.is_empty() {
                    source_list.insert(source_name.clone(), source);
                }

                in_source = false;
                in_shapelet = false;
                in_shapelet2 = false;
                component_type_set = false;
                source_name.clear();
            }

            Some(k) => {
                return Err(ReadSourceListCommonError::UnrecognisedKeyword {
                    line_num,
                    keyword: k.to_string(),
                }
                .into())
            }

            // Empty line, continue.
            None => (),
        }

        line.clear(); // clear to reuse the buffer line.
    }

    // If we're still "in a source", but we've finished reading lines, then an
    // ENDSOURCE must be missing.
    if in_source {
        return Err(ReadSourceListCommonError::MissingEndSource(line_num).into());
    }

    // Complain if no sources were read.
    if source_list.is_empty() {
        return Err(ReadSourceListCommonError::NoSources(line_num).into());
    }

    Ok(source_list)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use approx::*;
    // indoc allows us to write test source lists that look like they would in a
    // file.
    use indoc::indoc;
    use marlu::constants::DH2R;

    use super::*;

    #[test]
    fn parse_source_1() {
        let mut sl = Cursor::new(indoc! {"
        SOURCE VLA_ForA 3.40182 -37.5551
        FREQ 180e+6 1.0 0 0 0
        ENDSOURCE
        SOURCE VLA_ForB 3.40166 -37.0551
        FREQ 180e+6 2.0 0 0 0
        ENDSOURCE
        "});

        let result = parse_source_list(&mut sl);
        assert!(result.is_ok(), "{result:?}");
        let sl = result.unwrap();
        assert_eq!(sl.len(), 2);

        assert!(sl.contains_key("VLA_ForA"));
        let s = sl.get("VLA_ForA").unwrap();
        assert_eq!(s.components.len(), 1);
        let comp = &s.components[0];
        assert_abs_diff_eq!(comp.radec.ra, 3.40182 * DH2R);
        assert_abs_diff_eq!(comp.radec.dec, -37.5551_f64.to_radians());
        assert!(matches!(comp.flux_type, FluxDensityType::List { .. }));
        match &comp.flux_type {
            FluxDensityType::List(fds) => {
                assert_abs_diff_eq!(fds[0].freq, 180e6);
                assert_abs_diff_eq!(fds[0].i, 1.0);
                assert_abs_diff_eq!(fds[0].q, 0.0);
            }

            FluxDensityType::PowerLaw { .. } | FluxDensityType::CurvedPowerLaw { .. } => {
                unreachable!()
            }
        };

        assert!(sl.contains_key("VLA_ForB"));
        let s = sl.get("VLA_ForB").unwrap();
        assert_eq!(s.components.len(), 1);
        let comp = &s.components[0];
        assert_abs_diff_eq!(comp.radec.ra, 3.40166 * DH2R);
        assert_abs_diff_eq!(comp.radec.dec, -37.0551_f64.to_radians());
        assert!(matches!(comp.flux_type, FluxDensityType::List { .. }));
        match &comp.flux_type {
            FluxDensityType::List(fds) => {
                assert_abs_diff_eq!(fds[0].freq, 180e6);
                assert_abs_diff_eq!(fds[0].i, 2.0);
                assert_abs_diff_eq!(fds[0].q, 0.0);
            }
            _ => unreachable!(),
        };
    }

    #[test]
    fn parse_source_1_no_trailing_newline() {
        let mut sl = Cursor::new(indoc! {"
        SOURCE VLA_ForA 3.40182 -37.5551
        FREQ 180e+6 1.0 0 0 0
        ENDSOURCE
        SOURCE VLA_ForB 3.40166 -37.0551
        FREQ 180e+6 2.0 0 0 0
        ENDSOURCE"});

        let result = parse_source_list(&mut sl);
        assert!(result.is_ok(), "{result:?}");
        let sl = result.unwrap();
        assert_eq!(sl.len(), 2);

        assert!(sl.contains_key("VLA_ForA"));
        assert!(sl.contains_key("VLA_ForB"));
    }

    #[test]
    fn parse_source_2() {
        let mut sl = Cursor::new(indoc! {"
        SOURCE VLA_ForA 3.40182 -37.5551
        FREQ 180e+6 1.0 0 0 0
        FREQ 190e+6 0.123 0.5 0 0
        COMPONENT 3.40200 -37.6
        GAUSSIAN 90 1.0 0.5
        FREQ 180e+6 0.5 0 0 0
        FREQ 170e+6 1.0 0 0.2 0
        ENDCOMPONENT
        COMPONENT 3.40200 -37.6
        SHAPELET2 70 1.5 0.75
        COEFF 0.0e+00   0.0e+00   5.0239939e-02
        COEFF 9.0e+00   0.0e+00  -8.7418484e-03
        FREQ 180e+6 0.5 0 0 0
        FREQ 170e+6 1.0 0 0.2 0
        ENDCOMPONENT
        ENDSOURCE
        SOURCE VLA_ForB 3.40166 -37.0551
        FREQ 180e+6 2.0 0 0 0
        ENDSOURCE
        "});

        let result = parse_source_list(&mut sl);
        assert!(result.is_ok(), "{result:?}");
        let sl = result.unwrap();
        assert_eq!(sl.len(), 2);

        assert!(sl.contains_key("VLA_ForA"));
        let s = sl.get("VLA_ForA").unwrap();
        assert_eq!(s.components.len(), 3);
        let comp = &s.components[0];
        assert_abs_diff_eq!(comp.radec.ra, 3.40182 * DH2R);
        assert_abs_diff_eq!(comp.radec.dec, -37.5551_f64.to_radians());
        assert!(matches!(comp.flux_type, FluxDensityType::List { .. }));
        match &comp.flux_type {
            FluxDensityType::List(fds) => {
                assert_abs_diff_eq!(fds[0].freq, 180e6);
                assert_abs_diff_eq!(fds[0].i, 1.0);
                assert_abs_diff_eq!(fds[0].q, 0.0);
                assert_abs_diff_eq!(fds[1].i, 0.123);
                assert_abs_diff_eq!(fds[1].q, 0.5);
            }
            _ => unreachable!(),
        };

        let comp = &s.components[1];
        assert_abs_diff_eq!(comp.radec.ra, 3.402 * DH2R);
        assert_abs_diff_eq!(comp.radec.dec, -37.6_f64.to_radians());
        assert!(matches!(comp.flux_type, FluxDensityType::List { .. }));
        match &comp.flux_type {
            FluxDensityType::List(fds) => {
                // Note that 180 MHz isn't the first FREQ specified; the list
                // has been sorted.
                assert_abs_diff_eq!(fds[0].freq, 170e6);
                assert_abs_diff_eq!(fds[0].i, 1.0);
                assert_abs_diff_eq!(fds[0].q, 0.0);
                assert_abs_diff_eq!(fds[0].u, 0.2);
                assert_abs_diff_eq!(fds[1].freq, 180e6);
                assert_abs_diff_eq!(fds[1].i, 0.5);
                assert_abs_diff_eq!(fds[1].u, 0.0);
            }
            _ => unreachable!(),
        };

        assert!(sl.contains_key("VLA_ForB"));
        let s = sl.get("VLA_ForB").unwrap();
        assert_eq!(s.components.len(), 1);
        let comp = &s.components[0];
        assert_abs_diff_eq!(comp.radec.ra, 3.40166 * DH2R);
        assert_abs_diff_eq!(comp.radec.dec, -37.0551_f64.to_radians());
        assert!(matches!(comp.flux_type, FluxDensityType::List { .. }));
        match &comp.flux_type {
            FluxDensityType::List(fds) => {
                assert_abs_diff_eq!(fds[0].i, 2.0);
                assert_abs_diff_eq!(fds[0].q, 0.0);
            }
            _ => unreachable!(),
        };
    }

    #[test]
    fn parse_source_2_comps_freqs_swapped() {
        let mut sl = Cursor::new(indoc! {"
        SOURCE VLA_ForA 3.40182 -37.5551
        FREQ 180e+6 1.0 0 0 0
        FREQ 190e+6 0.123 0.5 0 0
        COMPONENT 3.40200 -37.6
        FREQ 180e+6 0.5 0 0 0
        FREQ 170e+6 1.0 0 0.2 0
        GAUSSIAN 90 1.0 0.5
        ENDCOMPONENT
        COMPONENT 3.40200 -37.6
        FREQ 180e+6 0.5 0 0 0
        FREQ 170e+6 1.0 0 0.2 0
        SHAPELET2 70 1.5 0.75
        COEFF 0.0e+00   0.0e+00   5.0239939e-02
        COEFF 9.0e+00   0.0e+00  -8.7418484e-03
        ENDCOMPONENT
        ENDSOURCE
        SOURCE VLA_ForB 3.40166 -37.0551
        FREQ 180e+6 2.0 0 0 0
        ENDSOURCE
        "});

        let result = parse_source_list(&mut sl);
        assert!(result.is_ok(), "{result:?}");
        let sl = result.unwrap();
        assert_eq!(sl.len(), 2);
    }

    #[test]
    fn parse_source_3() {
        let mut sl = Cursor::new(indoc! {"
        SOURCE VLA_ForA 3.40182 -37.5551
        FREQ 180e+6 1.0 0 0 0
        # FREQ 190e+6 0.123 0.5 0 0
        ENDSOURCE
        SOURCE VLA_ForB 3.40166 -37.0551 # Fornax B >>> Fornax A
        GAUSSIAN 90 1.0 0.5
        # FREQ 180e+6 0.5 0 0 0
        FREQ 170e+6 1.0 0 0.2 0
        ENDSOURCE
        "});

        let result = parse_source_list(&mut sl);
        assert!(result.is_ok(), "{result:?}");
        let sl = result.unwrap();
        assert_eq!(sl.len(), 2);
    }

    #[test]
    fn parse_source_4() {
        let mut sl = Cursor::new(indoc! {"
        SOURCE VLA_ForA 3.40182 -37.5551
        FREQ 180e+6 1.0 0 0 0
        GAUSSIAN 90 1.0 0.5
        COMPONENT 3.40166 -37.0551
        GAUSSIAN 90 1.0 0.5
        FREQ 170e+6 1.0 0 0.2 0
        ENDCOMPONENT
        ENDSOURCE
        "});

        let result = parse_source_list(&mut sl);
        assert!(result.is_ok(), "{result:?}");
        let sl = result.unwrap();
        assert_eq!(sl.len(), 1);
        let comps = &sl.get("VLA_ForA").unwrap().components;
        assert_eq!(comps.len(), 2);
    }

    #[test]
    fn parse_flawed_source_commented() {
        let mut sl = Cursor::new(indoc! {"
        # SOURCE VLA_ForA 3.40182 -37.5551
        FREQ 190e+6 0.123 0.5 0 0
        ENDSOURCE
        "});
        let result = parse_source_list(&mut sl);
        assert!(&result.is_err(), "{:?}", &result);
    }

    #[test]
    fn parse_flawed_freq_commented() {
        let mut sl = Cursor::new(indoc! {"
        SOURCE VLA_ForA 3.40182 -37.5551
        # FREQ 190e+6 0.123 0.5 0 0
        ENDSOURCE
        "});
        let result = parse_source_list(&mut sl);
        assert!(&result.is_err(), "{:?}", &result);
    }

    #[test]
    fn parse_flawed_endsource_commented() {
        let mut sl = Cursor::new(indoc! {"
        SOURCE VLA_ForA 3.40182 -37.5551
        FREQ 190e+6 0.123 0.5 0 0
        # ENDSOURCE
        "});
        let result = parse_source_list(&mut sl);
        assert!(&result.is_err(), "{:?}", &result);
    }

    #[test]
    fn parse_flawed_endsource_indented() {
        let mut sl = Cursor::new(indoc! {"
        SOURCE VLA_ForA 3.40182 -37.5551
        FREQ 190e+6 0.123 0.5 0 0
         ENDSOURCE
        "});
        let result = parse_source_list(&mut sl);
        assert!(&result.is_err(), "{:?}", &result);
    }

    #[test]
    fn parse_empty_component() {
        let mut sl = Cursor::new(indoc! {"
        SOURCE VLA_ForA 3.40182 -37.5551
        FREQ 190e+6 0.123 0.5 0 0
        COMPONENT 3.40200 -37.6
        ENDCOMPONENT
        ENDSOURCE
        "});
        let result = parse_source_list(&mut sl);
        assert!(&result.is_err(), "{:?}", &result);
    }

    #[test]
    fn parse_flawed_shapelet() {
        // Because we ignore SHAPELET components, this source list has no sources.
        let mut sl = Cursor::new(indoc! {"
        SOURCE VLA_ForA 3.40182 -37.5551
        FREQ 170e6 235.71 0 0 0
        SHAPELET -77.99222 8.027761 4.820640
        COEFF 0.0e+00   0.0e+00   5.0239939e-02
        COEFF 0.0e+00   2.0e+00   2.0790306e-02
        ENDSOURCE
        "});
        let result = parse_source_list(&mut sl);
        assert!(&result.is_err(), "{:?}", &result);
    }

    #[test]
    fn parse_flawed_shapelet_comp() {
        // Because we ignore SHAPELET components, this source list has one
        // component.
        let mut sl = Cursor::new(indoc! {"
        SOURCE VLA_ForA 3.40182 -37.5551
        FREQ 170e6 235.71 0 0 0
        COMPONENT 3.40200 -37.6
        FREQ 170e6 200 0 0 0
        SHAPELET -77.99222 8.027761 4.820640
        COEFF 0.0e+00   0.0e+00   5.0239939e-02
        COEFF 0.0e+00   2.0e+00   2.0790306e-02
        ENDCOMPONENT
        ENDSOURCE
        "});
        let result = parse_source_list(&mut sl);
        assert!(&result.is_ok(), "{:?}", &result);
        let sl = result.unwrap();
        assert_eq!(sl.len(), 1);
        let comps = &sl.get("VLA_ForA").unwrap().components;
        assert_eq!(comps.len(), 1);
    }

    #[test]
    fn parse_flawed_no_shapelet2_coeffs() {
        // No shapelet COEFFs.
        let mut sl = Cursor::new(indoc! {"
        SOURCE VLA_ForA 3.40182 -37.5551
        FREQ 170e6 235.71 0 0 0
        SHAPELET2 -77.99222 8.027761 4.820640
        # COEFF 0.0e+00   0.0e+00   5.0239939e-02
        # COEFF 0.0e+00   2.0e+00   2.0790306e-02
        ENDSOURCE
        "});
        let result = parse_source_list(&mut sl);
        assert!(&result.is_err(), "{:?}", &result);
    }

    #[test]
    fn parse_flawed_no_shapelet2_coeffs_comp() {
        let mut sl = Cursor::new(indoc! {"
        SOURCE VLA_ForA 3.40182 -37.5551
        FREQ 170e6 235.71 0 0 0
        COMPONENT 3.40200 -37.6
        FREQ 170e6 200 0 0 0
        SHAPELET2 -77.99222 8.027761 4.820640
        # COEFF 0.0e+00   0.0e+00   5.0239939e-02
        # COEFF 0.0e+00   2.0e+00   2.0790306e-02
        ENDCOMPONENT
        ENDSOURCE
        "});
        let result = parse_source_list(&mut sl);
        assert!(&result.is_err(), "{:?}", &result);
    }

    #[test]
    fn parse_flawed_endcomponent_commented() {
        let mut sl = Cursor::new(indoc! {"
        SOURCE VLA_ForA 3.40182 -37.5551
        FREQ 190e+6 0.123 0.5 0 0
        COMPONENT 3.40200 -37.6
        FREQ 180e+6 0.5 0 0 0
        # ENDCOMPONENT
        ENDSOURCE
        "});
        let result = parse_source_list(&mut sl);
        assert!(&result.is_err(), "{:?}", &result);
    }

    #[test]
    fn parse_flawed_endcomponent_indented() {
        let mut sl = Cursor::new(indoc! {"
        SOURCE VLA_ForA 3.40182 -37.5551
        FREQ 190e+6 0.123 0.5 0 0
        COMPONENT 3.40200 -37.6
        FREQ 180e+6 0.5 0 0 0
        \tENDCOMPONENT
        ENDSOURCE
        "});
        let result = parse_source_list(&mut sl);
        assert!(&result.is_err(), "{:?}", &result);
    }

    #[test]
    fn invalid_flux_sum1() {
        let mut sl = Cursor::new(indoc! {"
        SOURCE VLA_ForA 3.40182 -37.5551
        FREQ 180e+6 0 1.0 0 0
        ENDSOURCE
        SOURCE VLA_ForB 3.40166 -37.0551
        FREQ 180e+6 0 -2.0 0 0
        ENDSOURCE
        "});

        let result = parse_source_list(&mut sl);
        assert!(result.is_err(), "{result:?}");
        let err_message = format!("{}", result.unwrap_err());
        assert_eq!(
            err_message,
            "Source VLA_ForB: The sum of all Stokes Q flux densities was negative (-2)"
        );
    }

    #[test]
    fn invalid_flux_sum2() {
        let mut sl = Cursor::new(indoc! {"
        SOURCE VLA_ForA 3.40182 -37.5551
        FREQ 180e+6 0 1.0 0 0
        COMPONENT 3.40166 -37.0551
        FREQ 180e+6 0 -2.0 0 0
        ENDCOMPONENT
        ENDSOURCE
        "});

        let result = parse_source_list(&mut sl);
        assert!(result.is_err(), "{result:?}");
        let err_message = format!("{}", result.unwrap_err());
        assert_eq!(
            err_message,
            "Source VLA_ForA: The sum of all Stokes Q flux densities was negative (-1)"
        );
    }

    #[test]
    fn shapelet_gets_ignored() {
        // We ignore "SHAPELET", using only "SHAPELET2".
        let mut sl = Cursor::new(indoc! {"
        SOURCE VLA_ForA 3.40182 -37.5551
        FREQ 185.0e+6 209.81459 0 0 0
        SHAPELET 68.70984356 3.75 4.0
        COEFF 0.0 0.0 0.099731291104
        COMPONENT 3.40200 -37.6
        GAUSSIAN 90 1.0 0.5
        FREQ 180e+6 0.5 0 0 0
        FREQ 170e+6 1.0 0 0.2 0
        ENDCOMPONENT
        COMPONENT 3.40200 -37.6
        FREQ 185.0e+6 209.81459 0 0 0
        SHAPELET 68.70984356 3.75 4.0
        COEFF 0.0 0.0 0.099731291104
        ENDCOMPONENT
        ENDSOURCE
        SOURCE VLA_ForB 3.40182 -37.5551
        FREQ 185.0e+6 209.81459 0 0 0
        SHAPELET2 68.70984356 3.75 4.0
        COEFF 0.0 0.0 0.099731291104
        COMPONENT 3.40200 -37.6
        GAUSSIAN 90 1.0 0.5
        FREQ 180e+6 0.5 0 0 0
        FREQ 170e+6 1.0 0 0.2 0
        ENDCOMPONENT
        ENDSOURCE
        "});
        let result = parse_source_list(&mut sl);
        assert!(result.is_ok(), "{result:?}");
        let sl = result.unwrap();
        assert_eq!(sl["VLA_ForA"].components.len(), 1);
        assert_eq!(sl["VLA_ForB"].components.len(), 2);
    }

    #[test]
    fn invalid_ha() {
        let mut sl = Cursor::new(indoc! {"
        SOURCE J235959-180236 24.999841465546492 -18.047459423190038
        FREQ 170e6 235.71 0 0 0
        ENDSOURCE
        "});
        let result = parse_source_list(&mut sl);
        assert!(&result.is_err(), "{:?}", &result);

        let mut sl = Cursor::new(indoc! {"
        SOURCE J235959-180236 23.999841465546492 -18.047459423190038
        FREQ 170e6 235.71 0 0 0
        COMPONENT 24.005069897856296291 -20.092982953337557
        GAUSSIAN 90 1.0 0.5
        FREQ 180e+6 0.5 0 0 0
        FREQ 170e+6 1.0 0 0.2 0
        ENDCOMPONENT
        ENDSOURCE
        "});
        let result = parse_source_list(&mut sl);
        assert!(&result.is_err(), "{:?}", &result);
    }

    #[test]
    fn units() {
        // The first component is taken from one of Jack's source lists; dunno
        // why the major (3.75) is smaller than the minor (4.0)! I suppose it
        // doesn't matter if the PA is right...
        let mut sl = Cursor::new(indoc! {"
        SOURCE VLA_ForA 3.40182 -37.5551
        FREQ 185.0e+6 209.81459 0 0 0
        SHAPELET2 68.70984356 3.75 4.0
        COEFF 0.0 0.0 0.099731291104
        COMPONENT 3.40200 -37.6
        GAUSSIAN 90 1.0 0.5
        FREQ 180e+6 0.5 0 0 0
        FREQ 170e+6 1.0 0 0.2 0
        ENDCOMPONENT
        ENDSOURCE
        "});
        let result = parse_source_list(&mut sl);
        assert!(result.is_ok(), "{result:?}");
        let sl = result.unwrap();
        let source = &sl["VLA_ForA"];

        match &source.components[0].comp_type {
            ComponentType::Shapelet {
                maj,
                min,
                pa,
                coeffs,
            } => {
                assert_abs_diff_eq!(*maj, 3.75_f64.to_radians() / 60.0);
                assert_abs_diff_eq!(*min, 4.0_f64.to_radians() / 60.0);
                assert_abs_diff_eq!(*pa, 68.70984356_f64.to_radians());
                assert_eq!(coeffs.len(), 1);
                assert_eq!(coeffs[0].n1, 0);
                assert_eq!(coeffs[0].n2, 0);
                assert_abs_diff_eq!(coeffs[0].value, 0.099731291104);
            }

            _ => unreachable!(),
        }

        match &source.components[1].comp_type {
            ComponentType::Gaussian { maj, min, pa } => {
                assert_abs_diff_eq!(*maj, 1.0_f64.to_radians() / 60.0);
                assert_abs_diff_eq!(*min, 0.5_f64.to_radians() / 60.0);
                assert_abs_diff_eq!(*pa, 90_f64.to_radians());
            }

            _ => unreachable!(),
        }
    }
}
