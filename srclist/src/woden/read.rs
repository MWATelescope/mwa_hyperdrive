// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Parsing of WODEN source lists.

WODEN source lists are similar to RTS source lists, but a little more modern.
There is no "base source", and point-source-component types are explicit. SOURCE
lines include a count of how many of each component type is associated with the
source.

Coordinates are hour angle and declination, which have units of decimal hours
(i.e. 0 - 24) and degrees, respectively.

Gaussian and shapelet sizes are specified in arcminutes, whereas position angles
are in degrees. All frequencies are in Hz.

All flux densities are specified in the "list" style, and all have units of Jy.

Keywords like SOURCE, COMPONENT, POINT etc. must be at the start of a line (i.e.
no preceeding space). COMPONENT must be followed by one of POINT, GAUSSIAN or
SHAPELET. Each component will have a corresponding GPARAMS or SPARAMS (nothing
needed for a point source).
 */

use std::convert::TryInto;

use log::warn;

use super::*;

/// Parse a buffer containing a WODEN-style source list into a `SourceList`.
pub fn parse_source_list<T: std::io::BufRead>(
    buf: &mut T,
) -> Result<SourceList, ReadSourceListError> {
    let mut line = String::new();
    let mut line_num: u32 = 0;
    let mut in_source = false;
    let mut in_component = false;
    let mut num_points = 0;
    let mut num_gaussians = 0;
    let mut num_shapelets = 0;
    let mut num_shapelet_coeffs = 0;
    let mut source_name = String::new();
    let mut components: Vec<SourceComponent> = vec![];
    let mut source_list: SourceList = BTreeMap::new();

    let parse_float = |string: &str, line_num: u32| -> Result<f64, ReadSourceListCommonError> {
        string
            .parse()
            .map_err(|_| ReadSourceListCommonError::ParseFloatError {
                line_num,
                string: string.to_string(),
            })
    };

    let float_to_int = |float: f64, line_num: u32| -> Result<u32, ReadSourceListCommonError> {
        if float < 0.0 || float > std::u32::MAX as _ {
            Err(ReadSourceListCommonError::FloatToIntError { line_num, float })
        } else {
            Ok(float as u32)
        }
    };

    while buf.read_line(&mut line).expect("IO error") > 0 {
        line_num += 1;

        // Handle lines that aren't intended to parsed (comments and blank
        // lines).
        if line.starts_with('#') | line.starts_with('\n') {
            line.clear();
            continue;
        }
        // We ignore any lines starting with whitespace, but emit a warning.
        else if line.starts_with(' ') | line.starts_with('\t') {
            warn!(
                "Source list line {} starts with whitespace; ignoring it",
                line_num
            );
            line.clear();
            continue;
        }

        let mut items = line.split_whitespace();
        match items.next() {
            Some("SOURCE") => {
                if in_source {
                    return Err(ReadSourceListCommonError::NestedSources(line_num).into());
                } else {
                    // SOURCE lines must have at least 9 elements (including SOURCE).
                    match items.next() {
                        Some(name) => source_name.push_str(name),
                        None => {
                            return Err(
                                ReadSourceListCommonError::IncompleteSourceLine(line_num).into()
                            )
                        }
                    };
                    match items.next() {
                        Some("P") => (),
                        Some(k) => {
                            return Err(ReadSourceListCommonError::UnrecognisedKeyword {
                                line_num,
                                keyword: k.to_string(),
                            }
                            .into())
                        }
                        None => {
                            return Err(
                                ReadSourceListCommonError::IncompleteSourceLine(line_num).into()
                            )
                        }
                    };
                    num_points = match items.next() {
                        Some(v) => {
                            parse_float(v, line_num).and_then(|f| float_to_int(f, line_num))?
                        }
                        None => {
                            return Err(
                                ReadSourceListCommonError::IncompleteSourceLine(line_num).into()
                            )
                        }
                    };
                    match items.next() {
                        Some("G") => (),
                        Some(k) => {
                            return Err(ReadSourceListCommonError::UnrecognisedKeyword {
                                line_num,
                                keyword: k.to_string(),
                            }
                            .into())
                        }
                        None => {
                            return Err(
                                ReadSourceListCommonError::IncompleteSourceLine(line_num).into()
                            )
                        }
                    };
                    num_gaussians = match items.next() {
                        Some(v) => {
                            parse_float(v, line_num).and_then(|f| float_to_int(f, line_num))?
                        }
                        None => {
                            return Err(
                                ReadSourceListCommonError::IncompleteSourceLine(line_num).into()
                            )
                        }
                    };
                    match items.next() {
                        Some("S") => (),
                        Some(k) => {
                            return Err(ReadSourceListCommonError::UnrecognisedKeyword {
                                line_num,
                                keyword: k.to_string(),
                            }
                            .into())
                        }
                        None => {
                            return Err(
                                ReadSourceListCommonError::IncompleteSourceLine(line_num).into()
                            )
                        }
                    };
                    num_shapelets = match items.next() {
                        Some(v) => {
                            parse_float(v, line_num).and_then(|f| float_to_int(f, line_num))?
                        }
                        None => {
                            return Err(
                                ReadSourceListCommonError::IncompleteSourceLine(line_num).into()
                            )
                        }
                    };
                    num_shapelet_coeffs = match items.next() {
                        Some(v) => {
                            parse_float(v, line_num).and_then(|f| float_to_int(f, line_num))?
                        }
                        None => {
                            return Err(
                                ReadSourceListCommonError::IncompleteSourceLine(line_num).into()
                            )
                        }
                    };
                    if items.next().is_some() {
                        warn!(
                            "Source list line {}: Ignoring trailing contents after declination",
                            line_num
                        );
                    }

                    in_source = true;
                }
            }

            // New component.
            Some("COMPONENT") => {
                if !in_source {
                    return Err(ReadSourceListCommonError::OutsideSource {
                        line_num,
                        keyword: "COMPONENT".to_string(),
                    }
                    .into());
                }
                if in_component {
                    return Err(ReadSourceListCommonError::NestedComponents(line_num).into());
                }

                // COMPONENT lines must have at least 4 elements (including COMPONENT).
                let comp_type = match items.next() {
                    Some("POINT") => ComponentType::Point,
                    Some("GAUSSIAN") => ComponentType::Gaussian {
                        maj: 0.0,
                        min: 0.0,
                        pa: 0.0,
                    },
                    Some("SHAPELET") => ComponentType::Shapelet {
                        maj: 0.0,
                        min: 0.0,
                        pa: 0.0,
                        coeffs: vec![],
                    },
                    Some(k) => {
                        return Err(ReadSourceListCommonError::UnrecognisedKeyword {
                            line_num,
                            keyword: k.to_string(),
                        }
                        .into())
                    }
                    None => {
                        return Err(ReadSourceListCommonError::IncompleteSourceLine(line_num).into())
                    }
                };
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
                    warn!(
                        "Source list line {}: Ignoring trailing contents after declination",
                        line_num
                    );
                }

                // Validation and conversion.
                if !(0.0..=24.0).contains(&hour_angle) {
                    return Err(ReadSourceListError::InvalidHa(hour_angle));
                }
                if !(-90.0..=90.0).contains(&declination) {
                    return Err(ReadSourceListError::InvalidDec(declination));
                }
                let radec = RADec::new(hour_angle * DH2R, declination.to_radians());

                components.push(SourceComponent {
                    radec,
                    // Assume the base source is a point source. If we find
                    // component type information, we can overwrite this.
                    comp_type,
                    flux_type: FluxDensityType::List { fds: vec![] },
                });

                in_component = true;
            }

            // Flux density line.
            Some("FREQ") => {
                if !in_source {
                    return Err(ReadSourceListCommonError::OutsideSource {
                        line_num,
                        keyword: "FREQ".to_string(),
                    }
                    .into());
                } else if !in_component {
                    return Err(ReadSourceListCommonError::OutsideComponent {
                        line_num,
                        keyword: "FREQ".to_string(),
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
                    warn!(
                        "Source list line {}: Ignoring trailing contents after Stokes V",
                        line_num
                    );
                }

                let fd = FluxDensity {
                    freq: freq_hz,
                    i: stokes_i,
                    q: stokes_q,
                    u: stokes_u,
                    v: stokes_v,
                };

                match components.iter_mut().last().map(|c| &mut c.flux_type) {
                    Some(FluxDensityType::List { fds }) => fds.push(fd),
                    _ => unreachable!(),
                }
            }

            // Gaussian parameters.
            Some("GPARAMS") => {
                if !in_source {
                    return Err(ReadSourceListCommonError::OutsideSource {
                        line_num,
                        keyword: "GPARAMS".to_string(),
                    }
                    .into());
                } else if !in_component {
                    return Err(ReadSourceListCommonError::OutsideComponent {
                        line_num,
                        keyword: "GPARAMS".to_string(),
                    }
                    .into());
                }

                // GPARAMS lines must have at least 4 elements (including
                // GPARAMS).
                let mut position_angle = match items.next() {
                    Some(f) => parse_float(f, line_num)?,
                    None => {
                        return Err(ReadSourceListWodenError::IncompleteGParamsLine(line_num).into())
                    }
                };
                let maj_arcmin = match items.next() {
                    Some(f) => parse_float(f, line_num)?,
                    None => {
                        return Err(ReadSourceListWodenError::IncompleteGParamsLine(line_num).into())
                    }
                };
                let min_arcmin = match items.next() {
                    Some(f) => parse_float(f, line_num)?,
                    None => {
                        return Err(ReadSourceListWodenError::IncompleteGParamsLine(line_num).into())
                    }
                };
                if items.next().is_some() {
                    warn!(
                        "Source list line {}: Ignoring trailing contents after minor axis",
                        line_num
                    );
                }

                // Ensure the position angle is positive.
                if position_angle < 0.0 {
                    position_angle += 360.0;
                }

                let comp_type = ComponentType::Gaussian {
                    maj: (maj_arcmin / 60.0).to_radians(),
                    min: (min_arcmin / 60.0).to_radians(),
                    pa: position_angle.to_radians(),
                };

                // Does this component type match what was on the COMPONENT
                // line?
                match components.last().unwrap().comp_type {
                    ComponentType::Gaussian { .. } => (),
                    _ => {
                        return Err(
                            ReadSourceListCommonError::MultipleComponentTypes(line_num).into()
                        )
                    }
                }
                components.iter_mut().last().unwrap().comp_type = comp_type;
            }

            // Shapelet parameters.
            Some("SPARAMS") => {
                if !in_source {
                    return Err(ReadSourceListCommonError::OutsideSource {
                        line_num,
                        keyword: "SPARAMS".to_string(),
                    }
                    .into());
                } else if !in_component {
                    return Err(ReadSourceListCommonError::OutsideComponent {
                        line_num,
                        keyword: "SPARAMS".to_string(),
                    }
                    .into());
                }

                // SPARAMS lines must have at least 4 elements (including
                // SPARAMS).
                let mut position_angle = match items.next() {
                    Some(f) => parse_float(f, line_num)?,
                    None => {
                        return Err(ReadSourceListWodenError::IncompleteSParamsLine(line_num).into())
                    }
                };
                let maj_arcmin = match items.next() {
                    Some(f) => parse_float(f, line_num)?,
                    None => {
                        return Err(ReadSourceListWodenError::IncompleteSParamsLine(line_num).into())
                    }
                };
                let min_arcmin = match items.next() {
                    Some(f) => parse_float(f, line_num)?,
                    None => {
                        return Err(ReadSourceListWodenError::IncompleteSParamsLine(line_num).into())
                    }
                };
                if items.next().is_some() {
                    warn!(
                        "Source list line {}: Ignoring trailing contents after minor axis",
                        line_num
                    );
                }

                // Ensure the position angle is positive.
                if position_angle < 0.0 {
                    position_angle += 360.0;
                }

                let comp_type = ComponentType::Shapelet {
                    maj: (maj_arcmin / 60.0).to_radians(),
                    min: (min_arcmin / 60.0).to_radians(),
                    pa: position_angle.to_radians(),
                    coeffs: vec![],
                };

                // Does this component type match what was on the COMPONENT
                // line?
                match components.last().unwrap().comp_type {
                    ComponentType::Shapelet { .. } => (),
                    _ => {
                        return Err(
                            ReadSourceListCommonError::MultipleComponentTypes(line_num).into()
                        )
                    }
                }
                components.iter_mut().last().unwrap().comp_type = comp_type;
            }

            // Shapelet coefficient.
            Some("SCOEFF") => {
                if !in_source {
                    return Err(ReadSourceListCommonError::OutsideSource {
                        line_num,
                        keyword: "SCOEFF".to_string(),
                    }
                    .into());
                } else if !in_component {
                    return Err(ReadSourceListCommonError::OutsideComponent {
                        line_num,
                        keyword: "SCOEFF".to_string(),
                    }
                    .into());
                }

                // SCOEFF lines must have at least 4 elements (including
                // SCOEFF).
                let n1 = match items.next() {
                    Some(f) => parse_float(f, line_num)?,
                    None => {
                        return Err(ReadSourceListWodenError::IncompleteSCoeffLine(line_num).into())
                    }
                };
                let n2 = match items.next() {
                    Some(f) => parse_float(f, line_num)?,
                    None => {
                        return Err(ReadSourceListWodenError::IncompleteSCoeffLine(line_num).into())
                    }
                };
                let coeff = match items.next() {
                    Some(f) => parse_float(f, line_num)?,
                    None => {
                        return Err(ReadSourceListWodenError::IncompleteSCoeffLine(line_num).into())
                    }
                };
                if items.next().is_some() {
                    warn!(
                        "Source list line {}: Ignoring trailing contents after minor axis",
                        line_num
                    );
                }

                let shapelet_coeff = ShapeletCoeff {
                    n1: float_to_int(n1, line_num)?.try_into().unwrap(),
                    n2: float_to_int(n2, line_num)?.try_into().unwrap(),
                    coeff,
                };

                match &mut components.iter_mut().last().unwrap().comp_type {
                    ComponentType::Shapelet { coeffs, .. } => coeffs.push(shapelet_coeff),
                    // If the last component isn't a shapelet, we've got problems.
                    _ => return Err(ReadSourceListWodenError::MissingSParamsLine(line_num).into()),
                }
            }

            Some("ENDCOMPONENT") => {
                if !in_component {
                    return Err(ReadSourceListCommonError::EarlyEndComponent(line_num).into());
                }

                // Check that the last component struct added actually has flux
                // densities. WODEN source lists can only have the "list" type.
                match &mut components.iter_mut().last().unwrap().flux_type {
                    FluxDensityType::List { fds } => {
                        if fds.is_empty() {
                            return Err(ReadSourceListCommonError::NoFluxDensities(line_num).into());
                        } else {
                            // Sort the existing flux densities by frequency.
                            fds.sort_unstable_by(|&a, &b| {
                                a.freq.partial_cmp(&b.freq).unwrap_or_else(|| {
                                    panic!("Couldn't compare {} to {}", a.freq, b.freq)
                                })
                            });
                        }
                    }
                    _ => unreachable!(),
                }

                // If we were reading a shapelet component, check that shapelet
                // coefficients were read.
                if let ComponentType::Shapelet { coeffs, .. } =
                    &components.last().unwrap().comp_type
                {
                    if coeffs.is_empty() {
                        return Err(ReadSourceListWodenError::NoShapeletSCoeffs(line_num).into());
                    }
                }

                in_component = false;
            }

            Some("ENDSOURCE") => {
                if !in_source {
                    return Err(ReadSourceListCommonError::EarlyEndSource(line_num).into());
                } else if in_component {
                    return Err(ReadSourceListCommonError::MissingEndComponent(line_num).into());
                }
                let mut source = Source { components: vec![] };
                source.components.append(&mut components);

                if source.components.is_empty() {
                    return Err(ReadSourceListCommonError::NoComponents(line_num).into());
                }

                // Check that the specified counts of each component type were
                // read in. Also ensure that the sum of each Stokes' flux
                // densities is positive.
                let mut read_num_points = 0;
                let mut read_num_gaussians = 0;
                let mut read_num_shapelets = 0;
                let mut read_num_shapelet_coeffs = 0;
                let mut sum_i = 0.0;
                let mut sum_q = 0.0;
                let mut sum_u = 0.0;
                let mut sum_v = 0.0;
                for comp in &source.components {
                    match &comp.comp_type {
                        ComponentType::Point => read_num_points += 1,
                        ComponentType::Gaussian { .. } => read_num_gaussians += 1,
                        ComponentType::Shapelet { coeffs, .. } => {
                            read_num_shapelets += 1;
                            read_num_shapelet_coeffs += coeffs.len() as u32;
                        }
                    }

                    match &comp.flux_type {
                        FluxDensityType::List { fds } => {
                            for fd in fds {
                                sum_i += fd.i;
                                sum_q += fd.q;
                                sum_u += fd.u;
                                sum_v += fd.v;
                            }
                        }

                        FluxDensityType::PowerLaw { fd, .. } => {
                            sum_i += fd.i;
                            sum_q += fd.q;
                            sum_u += fd.u;
                            sum_v += fd.v;
                        }

                        FluxDensityType::CurvedPowerLaw { fd, .. } => {
                            sum_i += fd.i;
                            sum_q += fd.q;
                            sum_u += fd.u;
                            sum_v += fd.v;
                        }
                    }
                }
                if read_num_points != num_points {
                    return Err(ReadSourceListWodenError::CompCountMismatch {
                        line_num,
                        expected: num_points,
                        got: read_num_points,
                        comp_type: "POINT".to_string(),
                    }
                    .into());
                }
                if read_num_gaussians != num_gaussians {
                    return Err(ReadSourceListWodenError::CompCountMismatch {
                        line_num,
                        expected: num_gaussians,
                        got: read_num_gaussians,
                        comp_type: "GAUSSIAN".to_string(),
                    }
                    .into());
                }
                if read_num_shapelets != num_shapelets {
                    return Err(ReadSourceListWodenError::CompCountMismatch {
                        line_num,
                        expected: num_shapelets,
                        got: read_num_shapelets,
                        comp_type: "SHAPELET".to_string(),
                    }
                    .into());
                }
                if read_num_shapelet_coeffs != num_shapelet_coeffs {
                    return Err(ReadSourceListWodenError::ShapeletCoeffCountMismatch {
                        line_num,
                        expected: num_shapelet_coeffs,
                        got: read_num_shapelet_coeffs,
                    }
                    .into());
                }
                if sum_i < 0.0 {
                    return Err(ReadSourceListError::InvalidFluxDensitySum {
                        sum: sum_i,
                        stokes_comp: "I".to_string(),
                        source_name,
                    });
                } else if sum_q < 0.0 {
                    return Err(ReadSourceListError::InvalidFluxDensitySum {
                        sum: sum_q,
                        stokes_comp: "Q".to_string(),
                        source_name,
                    });
                } else if sum_u < 0.0 {
                    return Err(ReadSourceListError::InvalidFluxDensitySum {
                        sum: sum_u,
                        stokes_comp: "U".to_string(),
                        source_name,
                    });
                } else if sum_v < 0.0 {
                    return Err(ReadSourceListError::InvalidFluxDensitySum {
                        sum: sum_v,
                        stokes_comp: "V".to_string(),
                        source_name,
                    });
                }

                source_list.insert(source_name.clone(), source);

                in_source = false;
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

    use super::*;

    #[test]
    fn parse_point_source() {
        let mut sl = Cursor::new(indoc! {"
        SOURCE point_source P 1 G 0 S 0 0
        COMPONENT POINT 3.378 -37.2
        FREQ 1.8e+08 10.0 0 0 0
        ENDCOMPONENT
        ENDSOURCE
        "});

        let result = parse_source_list(&mut sl);
        assert!(result.is_ok(), "{:?}", result);
        let sl = result.unwrap();
        assert_eq!(sl.len(), 1);

        assert!(sl.contains_key("point_source"));
        let s = sl.get("point_source").unwrap();
        assert_eq!(s.components.len(), 1);
        let comp = &s.components[0];
        assert_abs_diff_eq!(comp.radec.ra, 3.378 * DH2R);
        assert_abs_diff_eq!(comp.radec.dec, -37.2_f64.to_radians());
        assert!(match comp.flux_type {
            FluxDensityType::List { .. } => true,
            _ => false,
        });
        let fds = match &comp.flux_type {
            FluxDensityType::List { fds } => fds,
            _ => unreachable!(),
        };
        assert_abs_diff_eq!(fds[0].freq, 180e6);
        assert_abs_diff_eq!(fds[0].i, 10.0);
        assert_abs_diff_eq!(fds[0].q, 0.0);

        assert!(match &comp.comp_type {
            ComponentType::Point => true,
            _ => false,
        });
    }

    #[test]
    fn parse_point_source2() {
        let mut sl = Cursor::new(indoc! {"
        SOURCE point_source P 1 G 0 S 0 0
        COMPONENT POINT 3.378 -37.2
        FREQ 1.8e+08 10.0 0 0 0
        FREQ 1.7e+08 8.0 0 0 0.2
        ENDCOMPONENT
        ENDSOURCE
        "});

        let result = parse_source_list(&mut sl);
        assert!(result.is_ok(), "{:?}", result);
        let sl = result.unwrap();
        assert_eq!(sl.len(), 1);

        assert!(sl.contains_key("point_source"));
        let s = sl.get("point_source").unwrap();
        assert_eq!(s.components.len(), 1);
        let comp = &s.components[0];
        assert_abs_diff_eq!(comp.radec.ra, 3.378 * DH2R);
        assert_abs_diff_eq!(comp.radec.dec, -37.2_f64.to_radians());
        assert!(match comp.flux_type {
            FluxDensityType::List { .. } => true,
            _ => false,
        });
        let fds = match &comp.flux_type {
            FluxDensityType::List { fds } => fds,
            _ => unreachable!(),
        };

        // Note that 180 MHz isn't the first FREQ specified; the list has been
        // sorted.
        assert_abs_diff_eq!(fds[0].freq, 170e6);
        assert_abs_diff_eq!(fds[0].i, 8.0);
        assert_abs_diff_eq!(fds[0].v, 0.2);
        assert_abs_diff_eq!(fds[1].freq, 180e6);
        assert_abs_diff_eq!(fds[1].i, 10.0);
        assert_abs_diff_eq!(fds[1].q, 0.0);

        assert!(match &comp.comp_type {
            ComponentType::Point => true,
            _ => false,
        });
    }

    #[test]
    fn parse_gaussian_source() {
        let mut sl = Cursor::new(indoc! {"
        SOURCE gaussian_source P 0 G 1 S 0 0
        COMPONENT GAUSSIAN 3.378 -37.2
        FREQ 1.8e+08 10.0 0 0 0
        GPARAMS 45.0000000000 6.0 3.0
        ENDCOMPONENT
        ENDSOURCE
        "});

        let result = parse_source_list(&mut sl);
        assert!(result.is_ok(), "{:?}", result);
        let sl = result.unwrap();
        assert_eq!(sl.len(), 1);

        assert!(sl.contains_key("gaussian_source"));
        let s = sl.get("gaussian_source").unwrap();
        assert_eq!(s.components.len(), 1);
        let comp = &s.components[0];
        assert_abs_diff_eq!(comp.radec.ra, 3.378 * DH2R);
        assert_abs_diff_eq!(comp.radec.dec, -37.2_f64.to_radians());
        assert!(match comp.flux_type {
            FluxDensityType::List { .. } => true,
            _ => false,
        });
        let fds = match &comp.flux_type {
            FluxDensityType::List { fds } => fds,
            _ => unreachable!(),
        };
        assert_abs_diff_eq!(fds[0].freq, 180e6);
        assert_abs_diff_eq!(fds[0].i, 10.0);
        assert_abs_diff_eq!(fds[0].q, 0.0);

        assert!(match &comp.comp_type {
            ComponentType::Gaussian { .. } => true,
            _ => false,
        });
        match &comp.comp_type {
            ComponentType::Gaussian { maj, min, pa } => {
                assert_abs_diff_eq!(*maj, (6.0 / 60.0_f64).to_radians());
                assert_abs_diff_eq!(*min, (3.0 / 60.0_f64).to_radians());
                assert_abs_diff_eq!(*pa, 45.0_f64.to_radians());
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn parse_shapelet_source() {
        let mut sl = Cursor::new(indoc! {"
        SOURCE shapelet_source P 0 G 0 S 1 1
        COMPONENT SHAPELET 3.378 -37.2
        FREQ 1.8e+08 10.0 0 0 0
        SPARAMS 45.0000000000 6.0 3.0
        SCOEFF 0 0 1.0
        ENDCOMPONENT
        ENDSOURCE
        "});

        let result = parse_source_list(&mut sl);
        assert!(result.is_ok(), "{:?}", result);
        let sl = result.unwrap();
        assert_eq!(sl.len(), 1);

        assert!(sl.contains_key("shapelet_source"));
        let s = sl.get("shapelet_source").unwrap();
        assert_eq!(s.components.len(), 1);
        let comp = &s.components[0];
        assert_abs_diff_eq!(comp.radec.ra, 3.378 * DH2R);
        assert_abs_diff_eq!(comp.radec.dec, -37.2_f64.to_radians());
        assert!(match comp.flux_type {
            FluxDensityType::List { .. } => true,
            _ => false,
        });
        let fds = match &comp.flux_type {
            FluxDensityType::List { fds } => fds,
            _ => unreachable!(),
        };
        assert_abs_diff_eq!(fds[0].freq, 180e6);
        assert_abs_diff_eq!(fds[0].i, 10.0);
        assert_abs_diff_eq!(fds[0].q, 0.0);

        assert!(match &comp.comp_type {
            ComponentType::Shapelet { .. } => true,
            _ => false,
        });
        match &comp.comp_type {
            ComponentType::Shapelet {
                maj,
                min,
                pa,
                coeffs,
            } => {
                assert_abs_diff_eq!(*maj, (6.0 / 60.0_f64).to_radians());
                assert_abs_diff_eq!(*min, (3.0 / 60.0_f64).to_radians());
                assert_abs_diff_eq!(*pa, 45.0_f64.to_radians());

                assert_eq!(coeffs[0].n1, 0);
                assert_eq!(coeffs[0].n2, 0);
                assert_eq!(coeffs[0].coeff, 1.0);
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn invalid_flux_sum() {
        let mut sl = Cursor::new(indoc! {"
        SOURCE bad_source P 2 G 0 S 0 0
        COMPONENT POINT 3.378 -37.2
        FREQ 1.8e+08 10.0 0 0 0
        ENDCOMPONENT
        COMPONENT POINT 3.378 -37.2
        FREQ 1.8e+08 -12.0 0 0 0
        ENDCOMPONENT
        ENDSOURCE
        "});

        let result = parse_source_list(&mut sl);
        assert!(result.is_err(), "{:?}", result);
        let err_message = format!("{}", result.unwrap_err());
        assert_eq!(
            err_message,
            "Source bad_source: The sum of all Stokes I flux densities was negative (-2)"
        );
    }
}
