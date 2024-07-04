// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Parsing of WODEN source lists.
//!
//! WODEN source lists are similar to RTS source lists, but a little more modern.
//! There is no "base source", and point-source-component types are explicit. SOURCE
//! lines include a count of how many of each component type is associated with the
//! source.
//!
//! Coordinates are hour angle and declination, which have units of decimal hours
//! (i.e. 0 - 24) and degrees, respectively.
//!
//! Gaussian and shapelet sizes are specified in arcminutes, whereas position angles
//! are in degrees. All frequencies are in Hz.
//!
//! All flux densities are specified as power laws, and all have units of Jy. If
//! a flux density is listed as FREQ, then it uses a default spectral index. The
//! alternative is LINEAR, which lists the SI. There is only one flux density per component.
//!
//! Keywords like SOURCE, COMPONENT, POINT etc. must be at the start of a line (i.e.
//! no preceeding space). COMPONENT must be followed by one of POINT, GAUSSIAN or
//! SHAPELET. Each component will have a corresponding GPARAMS or SPARAMS (nothing
//! needed for a point source).

use std::convert::TryInto;

use marlu::{constants::DH2R, RADec};

use crate::{
    cli::Warn,
    constants::DEFAULT_SPEC_INDEX,
    srclist::{
        error::{ReadSourceListCommonError, ReadSourceListError, ReadSourceListWodenError},
        ComponentType, FluxDensity, FluxDensityType, ShapeletCoeff, Source, SourceComponent,
        SourceList,
    },
};

/// Parse a buffer containing a WODEN-style source list into a `SourceList`.
pub(crate) fn parse_source_list<T: std::io::BufRead>(
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

    let float_to_int = |float: f64, line_num: u32| -> Result<u32, ReadSourceListCommonError> {
        if float < 0.0 || float > u32::MAX as f64 {
            Err(ReadSourceListCommonError::FloatToIntError { line_num, float })
        } else {
            Ok(float as u32)
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
                        format!(
                            "Source list line {}: Ignoring trailing contents after declination",
                            line_num
                        )
                        .warn();
                    }

                    in_source = true;
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
                        coeffs: vec![].into_boxed_slice(),
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
                let radec = RADec::from_radians(hour_angle * DH2R, declination.to_radians());

                components.push(SourceComponent {
                    radec,
                    // Assume the base source is a point source. If we find
                    // component type information, we can overwrite this.
                    comp_type,
                    flux_type: FluxDensityType::PowerLaw {
                        fd: FluxDensity::default(),
                        si: 0.0,
                    },
                });

                in_component = true;
            }

            // Flux density line.
            Some("FREQ") => {
                if !in_source {
                    return Err(ReadSourceListCommonError::OutsideSource {
                        line_num,
                        keyword: "FREQ",
                    }
                    .into());
                } else if !in_component {
                    return Err(ReadSourceListCommonError::OutsideComponent {
                        line_num,
                        keyword: "FREQ",
                    }
                    .into());
                }

                // FREQ lines must have at least 6 elements (including FREQ).
                let freq = match items.next() {
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

                match components.iter_mut().last().map(|c| &mut c.flux_type) {
                    Some(FluxDensityType::PowerLaw { fd, si }) => {
                        // If the frequency is set (i.e. not 0), the ignore
                        // additional flux density lines for this component.
                        if fd.freq > f64::EPSILON {
                            format!("Ignoring FREQ line {}", line_num).warn();
                        } else {
                            *fd = FluxDensity {
                                freq,
                                i: stokes_i,
                                q: stokes_q,
                                u: stokes_u,
                                v: stokes_v,
                            };
                            *si = DEFAULT_SPEC_INDEX;
                        }
                    }
                    _ => unreachable!(),
                }
            }

            // Flux density line for a power law.
            Some("LINEAR") => {
                if !in_source {
                    return Err(ReadSourceListCommonError::OutsideSource {
                        line_num,
                        keyword: "LINEAR",
                    }
                    .into());
                } else if !in_component {
                    return Err(ReadSourceListCommonError::OutsideComponent {
                        line_num,
                        keyword: "LINEAR",
                    }
                    .into());
                }

                // LINEAR lines must have at least 7 elements (including LINEAR).
                let freq = match items.next() {
                    Some(f) => parse_float(f, line_num)?,
                    None => {
                        return Err(ReadSourceListCommonError::IncompleteLinearLine(line_num).into())
                    }
                };
                let stokes_i = match items.next() {
                    Some(f) => parse_float(f, line_num)?,
                    None => {
                        return Err(ReadSourceListCommonError::IncompleteLinearLine(line_num).into())
                    }
                };
                let stokes_q = match items.next() {
                    Some(f) => parse_float(f, line_num)?,
                    None => {
                        return Err(ReadSourceListCommonError::IncompleteLinearLine(line_num).into())
                    }
                };
                let stokes_u = match items.next() {
                    Some(f) => parse_float(f, line_num)?,
                    None => {
                        return Err(ReadSourceListCommonError::IncompleteLinearLine(line_num).into())
                    }
                };
                let stokes_v = match items.next() {
                    Some(f) => parse_float(f, line_num)?,
                    None => {
                        return Err(ReadSourceListCommonError::IncompleteLinearLine(line_num).into())
                    }
                };
                let spectral_index = match items.next() {
                    Some(f) => parse_float(f, line_num)?,
                    None => {
                        return Err(ReadSourceListCommonError::IncompleteLinearLine(line_num).into())
                    }
                };
                if items.next().is_some() {
                    format!(
                        "Source list line {}: Ignoring trailing contents after spectral index",
                        line_num
                    )
                    .warn();
                }

                if stokes_i.is_nan()
                    || stokes_q.is_nan()
                    || stokes_u.is_nan()
                    || stokes_v.is_nan()
                    || spectral_index.is_nan()
                {
                    return Err(ReadSourceListError::NaNsInComponent { source_name });
                }

                match components.iter_mut().last().map(|c| &mut c.flux_type) {
                    Some(FluxDensityType::PowerLaw { fd, si }) => {
                        // If the frequency is set (i.e. not 0), the ignore
                        // additional flux density lines for this component.
                        if fd.freq > f64::EPSILON {
                            format!("Ignoring LINEAR line {}", line_num).warn();
                        } else {
                            *fd = FluxDensity {
                                freq,
                                i: stokes_i,
                                q: stokes_q,
                                u: stokes_u,
                                v: stokes_v,
                            };
                            *si = spectral_index;
                        }
                    }
                    _ => unreachable!(),
                }
            }

            // Gaussian parameters.
            Some("GPARAMS") => {
                if !in_source {
                    return Err(ReadSourceListCommonError::OutsideSource {
                        line_num,
                        keyword: "GPARAMS",
                    }
                    .into());
                } else if !in_component {
                    return Err(ReadSourceListCommonError::OutsideComponent {
                        line_num,
                        keyword: "GPARAMS",
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
                        keyword: "SPARAMS",
                    }
                    .into());
                } else if !in_component {
                    return Err(ReadSourceListCommonError::OutsideComponent {
                        line_num,
                        keyword: "SPARAMS",
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
                    maj: (maj_arcmin / 60.0).to_radians(),
                    min: (min_arcmin / 60.0).to_radians(),
                    pa: position_angle.to_radians(),
                    coeffs: vec![].into_boxed_slice(),
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
                        keyword: "SCOEFF",
                    }
                    .into());
                } else if !in_component {
                    return Err(ReadSourceListCommonError::OutsideComponent {
                        line_num,
                        keyword: "SCOEFF",
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
                let value = match items.next() {
                    Some(f) => parse_float(f, line_num)?,
                    None => {
                        return Err(ReadSourceListWodenError::IncompleteSCoeffLine(line_num).into())
                    }
                };
                if items.next().is_some() {
                    format!(
                        "Source list line {}: Ignoring trailing contents after minor axis",
                        line_num
                    )
                    .warn();
                }

                let shapelet_coeff = ShapeletCoeff {
                    n1: float_to_int(n1, line_num)
                        .map_err(|_| ReadSourceListCommonError::ShapeletBasisNotInt(n1))?
                        .try_into()
                        .expect("shapelet coeff is not bigger than u8::MAX"),
                    n2: float_to_int(n2, line_num)
                        .map_err(|_| ReadSourceListCommonError::ShapeletBasisNotInt(n2))?
                        .try_into()
                        .expect("shapelet coeff is not bigger than u8::MAX"),
                    value,
                };
                shapelet_coeffs.push(shapelet_coeff);

                // If the last component isn't a shapelet, we've got problems.
                if !matches!(
                    components.iter_mut().last().unwrap().comp_type,
                    ComponentType::Shapelet { .. }
                ) {
                    return Err(ReadSourceListWodenError::MissingSParamsLine(line_num).into());
                }
            }

            Some("ENDCOMPONENT") => {
                if !in_component {
                    return Err(ReadSourceListCommonError::EarlyEndComponent(line_num).into());
                }

                // Check that the last component struct added actually has flux
                // densities.
                match &mut components.iter_mut().last().unwrap().flux_type {
                    FluxDensityType::PowerLaw { fd, si } => {
                        if fd.i.abs() < f64::EPSILON
                            && fd.q.abs() < f64::EPSILON
                            && fd.u.abs() < f64::EPSILON
                            && fd.v.abs() < f64::EPSILON
                            && si.abs() < f64::EPSILON
                        {
                            return Err(ReadSourceListCommonError::NoFluxDensities(line_num).into());
                        }
                    }
                    _ => unreachable!(),
                }

                // If we were reading a shapelet component, check that shapelet
                // coefficients were read and populate the boxed slice.
                if let ComponentType::Shapelet { coeffs, .. } =
                    &mut components.last_mut().expect("not empty").comp_type
                {
                    if shapelet_coeffs.is_empty() {
                        return Err(ReadSourceListWodenError::NoShapeletSCoeffs(line_num).into());
                    }
                    *coeffs = shapelet_coeffs.clone().into_boxed_slice();
                    shapelet_coeffs.clear();
                }

                in_component = false;
            }

            Some("ENDSOURCE") => {
                if !in_source {
                    return Err(ReadSourceListCommonError::EarlyEndSource(line_num).into());
                } else if in_component {
                    return Err(ReadSourceListCommonError::MissingEndComponent(line_num).into());
                }
                if components.is_empty() {
                    return Err(ReadSourceListCommonError::NoComponents(line_num).into());
                }

                let source = Source {
                    components: components.clone().into_boxed_slice(),
                };
                components.clear();

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
                for comp in source.components.iter() {
                    match &comp.comp_type {
                        ComponentType::Point => read_num_points += 1,
                        ComponentType::Gaussian { .. } => read_num_gaussians += 1,
                        ComponentType::Shapelet { coeffs, .. } => {
                            read_num_shapelets += 1;
                            read_num_shapelet_coeffs += coeffs.len() as u32;
                        }
                    }

                    match &comp.flux_type {
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

                        FluxDensityType::List { .. } => unreachable!(),
                    }
                }
                if read_num_points != num_points {
                    return Err(ReadSourceListWodenError::CompCountMismatch {
                        line_num,
                        expected: num_points,
                        got: read_num_points,
                        comp_type: "POINT",
                    }
                    .into());
                }
                if read_num_gaussians != num_gaussians {
                    return Err(ReadSourceListWodenError::CompCountMismatch {
                        line_num,
                        expected: num_gaussians,
                        got: read_num_gaussians,
                        comp_type: "GAUSSIAN",
                    }
                    .into());
                }
                if read_num_shapelets != num_shapelets {
                    return Err(ReadSourceListWodenError::CompCountMismatch {
                        line_num,
                        expected: num_shapelets,
                        got: read_num_shapelets,
                        comp_type: "SHAPELET",
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
        assert!(result.is_ok(), "{result:?}");
        let sl = result.unwrap();
        assert_eq!(sl.len(), 1);

        assert!(sl.contains_key("point_source"));
        let s = sl.get("point_source").unwrap();
        assert_eq!(s.components.len(), 1);
        let comp = &s.components[0];
        assert_abs_diff_eq!(comp.radec.ra, 3.378 * DH2R);
        assert_abs_diff_eq!(comp.radec.dec, -37.2_f64.to_radians());
        assert!(matches!(comp.flux_type, FluxDensityType::PowerLaw { .. }));
        let (fd, si) = match &comp.flux_type {
            FluxDensityType::PowerLaw { fd, si } => (fd, si),
            _ => unreachable!(),
        };
        assert_abs_diff_eq!(fd.freq, 180e6);
        assert_abs_diff_eq!(fd.i, 10.0);
        assert_abs_diff_eq!(fd.q, 0.0);
        assert_abs_diff_eq!(*si, -0.8);

        assert!(matches!(&comp.comp_type, ComponentType::Point));
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
        assert!(result.is_ok(), "{result:?}");
        let sl = result.unwrap();
        assert_eq!(sl.len(), 1);

        assert!(sl.contains_key("point_source"));
        let s = sl.get("point_source").unwrap();
        assert_eq!(s.components.len(), 1);
        let comp = &s.components[0];
        assert_abs_diff_eq!(comp.radec.ra, 3.378 * DH2R);
        assert_abs_diff_eq!(comp.radec.dec, -37.2_f64.to_radians());
        assert!(matches!(comp.flux_type, FluxDensityType::PowerLaw { .. }));
        let (fd, si) = match &comp.flux_type {
            FluxDensityType::PowerLaw { fd, si } => (fd, si),
            _ => unreachable!(),
        };

        // Note that the 170 MHz line isn't included; only one flux density can
        // be associated with a component.
        assert_abs_diff_eq!(fd.freq, 180e6);
        assert_abs_diff_eq!(fd.i, 10.0);
        assert_abs_diff_eq!(fd.q, 0.0);
        assert_abs_diff_eq!(*si, -0.8);

        assert!(matches!(&comp.comp_type, ComponentType::Point));
    }

    #[test]
    fn parse_point_source3() {
        let mut sl = Cursor::new(indoc! {"
        SOURCE point_source P 1 G 0 S 0 0
        COMPONENT POINT 3.378 -37.2
        LINEAR 1.7e+08 8.0 0 0 0.2 -0.5
        ENDCOMPONENT
        ENDSOURCE
        "});

        let result = parse_source_list(&mut sl);
        assert!(result.is_ok(), "{result:?}");
        let sl = result.unwrap();
        assert_eq!(sl.len(), 1);

        assert!(sl.contains_key("point_source"));
        let s = sl.get("point_source").unwrap();
        assert_eq!(s.components.len(), 1);
        let comp = &s.components[0];
        assert_abs_diff_eq!(comp.radec.ra, 3.378 * DH2R);
        assert_abs_diff_eq!(comp.radec.dec, -37.2_f64.to_radians());
        assert!(matches!(comp.flux_type, FluxDensityType::PowerLaw { .. }));
        let (fd, si) = match &comp.flux_type {
            FluxDensityType::PowerLaw { fd, si } => (fd, si),
            _ => unreachable!(),
        };

        assert_abs_diff_eq!(fd.freq, 170e6);
        assert_abs_diff_eq!(fd.i, 8.0);
        assert_abs_diff_eq!(fd.v, 0.2);
        assert_abs_diff_eq!(*si, -0.5);

        assert!(matches!(&comp.comp_type, ComponentType::Point));
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
        assert!(result.is_ok(), "{result:?}");
        let sl = result.unwrap();
        assert_eq!(sl.len(), 1);

        assert!(sl.contains_key("gaussian_source"));
        let s = sl.get("gaussian_source").unwrap();
        assert_eq!(s.components.len(), 1);
        let comp = &s.components[0];
        assert_abs_diff_eq!(comp.radec.ra, 3.378 * DH2R);
        assert_abs_diff_eq!(comp.radec.dec, -37.2_f64.to_radians());
        assert!(matches!(comp.flux_type, FluxDensityType::PowerLaw { .. }));
        let (fd, si) = match &comp.flux_type {
            FluxDensityType::PowerLaw { fd, si } => (fd, si),
            _ => unreachable!(),
        };
        assert_abs_diff_eq!(fd.freq, 180e6);
        assert_abs_diff_eq!(fd.i, 10.0);
        assert_abs_diff_eq!(fd.q, 0.0);
        assert_abs_diff_eq!(*si, -0.8);

        assert!(matches!(&comp.comp_type, ComponentType::Gaussian { .. }));
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
        assert!(result.is_ok(), "{result:?}");
        let sl = result.unwrap();
        assert_eq!(sl.len(), 1);

        assert!(sl.contains_key("shapelet_source"));
        let s = sl.get("shapelet_source").unwrap();
        assert_eq!(s.components.len(), 1);
        let comp = &s.components[0];
        assert_abs_diff_eq!(comp.radec.ra, 3.378 * DH2R);
        assert_abs_diff_eq!(comp.radec.dec, -37.2_f64.to_radians());
        assert!(matches!(comp.flux_type, FluxDensityType::PowerLaw { .. }));
        let (fd, si) = match &comp.flux_type {
            FluxDensityType::PowerLaw { fd, si } => (fd, si),
            _ => unreachable!(),
        };
        assert_abs_diff_eq!(fd.freq, 180e6);
        assert_abs_diff_eq!(fd.i, 10.0);
        assert_abs_diff_eq!(fd.q, 0.0);
        assert_abs_diff_eq!(*si, -0.8);

        assert!(matches!(&comp.comp_type, ComponentType::Shapelet { .. }));
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
                assert_abs_diff_eq!(coeffs[0].value, 1.0);
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
        assert!(result.is_err(), "{result:?}");
        let err_message = format!("{}", result.unwrap_err());
        assert_eq!(
            err_message,
            "Source bad_source: The sum of all Stokes I flux densities was negative (-2)"
        );
    }
}
