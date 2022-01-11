// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Parsing of "André Offringa"-style source lists.
//!
//! The code here is probably incomplete, but it should work for the majority of
//! source lists.

use marlu::{sexagesimal::*, RADec};

use super::*;

/// Parse a buffer containing an AO-style source list into a [SourceList].
pub fn parse_source_list<T: std::io::BufRead>(
    buf: &mut T,
) -> Result<SourceList, ReadSourceListError> {
    let mut line = String::new();
    let mut line_num: u32 = 0;
    let mut in_source = false;
    let mut in_component = false;
    let mut in_sed = false;
    let mut in_measurement = false;
    let mut measurement_got_freq = false;
    let mut measurement_got_fd = false;
    let mut measurement_made_fd = false;
    let mut source_name = String::new();
    let mut components: Vec<SourceComponent> = vec![];
    let mut source_list = SourceList::new();

    let parse_float = |string: &str, line_num: u32| -> Result<f64, ReadSourceListCommonError> {
        string
            .parse()
            .map_err(|_| ReadSourceListCommonError::ParseFloatError {
                line_num,
                string: string.to_string(),
            })
    };

    while buf.read_line(&mut line)? > 0 {
        line_num += 1;

        // Handle lines that aren't intended to parsed (comments and blank
        // lines).
        if line.starts_with('#') | line.starts_with('\n') {
            line.clear();
            continue;
        }

        let mut items = line.split_whitespace();
        match items.next() {
            // Ignore the "fileformat" line.
            Some("skymodel") => (),

            Some("source") => {
                if in_source {
                    return Err(ReadSourceListCommonError::NestedSources(line_num).into());
                } else {
                    in_source = true;
                }
            }

            Some("name") => {
                if !in_source {
                    return Err(ReadSourceListCommonError::OutsideSource {
                        line_num,
                        keyword: "name",
                    }
                    .into());
                } else {
                    // Sigh.
                    let name_with_quotes: String = items.collect::<Vec<_>>().join(" ");
                    match name_with_quotes.split('\"').nth(1) {
                        None => return Err(ReadSourceListAOError::NameNotQuoted(line_num).into()),
                        Some(name) => source_name.push_str(name),
                    }
                }
            }

            Some("component") => {
                if !in_source {
                    return Err(ReadSourceListCommonError::OutsideSource {
                        line_num,
                        keyword: "name",
                    }
                    .into());
                }
                if in_component {
                    return Err(ReadSourceListCommonError::NestedComponents(line_num).into());
                }

                // We have to assume everything about the component, and
                // overwrite it later.
                components.push(SourceComponent {
                    radec: RADec::new(0.0, 0.0),
                    comp_type: ComponentType::Point,
                    flux_type: FluxDensityType::PowerLaw {
                        si: 0.0,
                        fd: FluxDensity::default(),
                    },
                });

                in_component = true;
            }

            // Component type.
            Some("type") => {
                if !in_source {
                    return Err(ReadSourceListCommonError::OutsideSource {
                        line_num,
                        keyword: "type",
                    }
                    .into());
                }
                if !in_component {
                    return Err(ReadSourceListCommonError::OutsideComponent {
                        line_num,
                        keyword: "type",
                    }
                    .into());
                }

                match items.next() {
                    Some("point") => {
                        components.iter_mut().last().unwrap().comp_type = ComponentType::Point
                    }
                    Some("gaussian") => {
                        components.iter_mut().last().unwrap().comp_type = ComponentType::Gaussian {
                            maj: 0.0,
                            min: 0.0,
                            pa: 0.0,
                        }
                    }
                    Some(t) => {
                        return Err(ReadSourceListAOError::UnrecognisedComponentType {
                            line_num,
                            comp_type: t.to_string(),
                        }
                        .into())
                    }
                    None => {
                        return Err(ReadSourceListAOError::MissingComponentType(line_num).into())
                    }
                }
            }

            Some("position") => {
                if !in_source {
                    return Err(ReadSourceListCommonError::OutsideSource {
                        line_num,
                        keyword: "position",
                    }
                    .into());
                }
                if !in_component {
                    return Err(ReadSourceListCommonError::OutsideComponent {
                        line_num,
                        keyword: "position",
                    }
                    .into());
                }

                let right_ascension = match items.next() {
                    Some(ra) => {
                        let mut deg = sexagesimal_hms_string_to_degrees(ra)?;
                        if deg < 0.0 {
                            deg += 360.0;
                        }
                        deg
                    }
                    None => {
                        return Err(ReadSourceListAOError::IncompletePositionLine(line_num).into())
                    }
                };

                let declination = match items.next() {
                    Some(dec) => sexagesimal_dms_string_to_degrees(dec)?,
                    None => {
                        return Err(ReadSourceListAOError::IncompletePositionLine(line_num).into())
                    }
                };

                // Validation.
                if right_ascension > 360.0 {
                    return Err(ReadSourceListError::InvalidHa(right_ascension));
                }
                if !(-90.0..=90.0).contains(&declination) {
                    return Err(ReadSourceListError::InvalidDec(declination));
                }

                components.iter_mut().last().unwrap().radec =
                    RADec::new_degrees(right_ascension, declination);
            }

            Some("shape") => {
                if !in_source {
                    return Err(ReadSourceListCommonError::OutsideSource {
                        line_num,
                        keyword: "shape",
                    }
                    .into());
                }
                if !in_component {
                    return Err(ReadSourceListCommonError::OutsideComponent {
                        line_num,
                        keyword: "shape",
                    }
                    .into());
                }

                let maj_arcsec = match items.next() {
                    Some(maj) => parse_float(maj, line_num)?,
                    None => return Err(ReadSourceListAOError::IncompleteShapeLine(line_num).into()),
                };
                let min_arcsec = match items.next() {
                    Some(min) => parse_float(min, line_num)?,
                    None => return Err(ReadSourceListAOError::IncompleteShapeLine(line_num).into()),
                };
                let mut position_angle_deg = match items.next() {
                    Some(maj) => parse_float(maj, line_num)?,
                    None => return Err(ReadSourceListAOError::IncompleteShapeLine(line_num).into()),
                };

                // Ensure the position angle is positive.
                if position_angle_deg < 0.0 {
                    position_angle_deg += 360.0;
                }

                components.iter_mut().last().unwrap().comp_type = ComponentType::Gaussian {
                    maj: maj_arcsec.to_radians() / 3600.0,
                    min: min_arcsec.to_radians() / 3600.0,
                    pa: position_angle_deg.to_radians(),
                }
            }

            // Power-law flux density type.
            Some("sed") => {
                in_sed = true;

                components.iter_mut().last().unwrap().flux_type = FluxDensityType::PowerLaw {
                    si: 0.0,
                    fd: FluxDensity::default(),
                };
            }

            // A flux-density at a single frequency.
            Some("measurement") => {
                in_measurement = true;

                if !measurement_made_fd {
                    components.iter_mut().last().unwrap().flux_type =
                        FluxDensityType::List { fds: vec![] };
                    measurement_made_fd = true;
                }
            }

            Some("frequency") => {
                if !in_source {
                    return Err(ReadSourceListCommonError::OutsideSource {
                        line_num,
                        keyword: "frequency",
                    }
                    .into());
                }
                if !in_component {
                    return Err(ReadSourceListCommonError::OutsideComponent {
                        line_num,
                        keyword: "frequency",
                    }
                    .into());
                }

                let freq = match items.next() {
                    Some(f) => parse_float(f, line_num)?,
                    None => {
                        return Err(ReadSourceListAOError::IncompleteFrequencyLine(line_num).into())
                    }
                };

                let freq_unit = match items.next() {
                    // It appears that mwa-reduce doesn't even read the units,
                    // it just assumes that the frequency is in MHz!
                    Some("MHz") => 1e6,
                    Some(u) => {
                        return Err(ReadSourceListAOError::UnhandledFrequencyUnits {
                            line_num,
                            units: u.to_string(),
                        }
                        .into())
                    }
                    None => {
                        return Err(ReadSourceListAOError::IncompleteFrequencyLine(line_num).into())
                    }
                };

                if in_sed {
                    match &mut components.iter_mut().last().unwrap().flux_type {
                        FluxDensityType::PowerLaw { fd, .. } => fd.freq = freq * freq_unit,
                        _ => unreachable!(),
                    }
                } else if in_measurement {
                    match &mut components.iter_mut().last().unwrap().flux_type {
                        FluxDensityType::List { fds } => {
                            // If both bools are false, then we need to make a
                            // new `FluxDensity` struct.
                            if !measurement_got_freq && !measurement_got_fd {
                                fds.push(FluxDensity {
                                    freq: freq * freq_unit,
                                    i: 0.0,
                                    q: 0.0,
                                    u: 0.0,
                                    v: 0.0,
                                });
                                measurement_got_freq = true;
                            } else {
                                // We should edit the last `FluxDensity` struct.
                                fds.iter_mut().last().unwrap().freq = freq * freq_unit;
                            }
                        }
                        _ => unreachable!(),
                    }
                } else {
                    unreachable!()
                }
            }

            Some("fluxdensity") => {
                if !in_source {
                    return Err(ReadSourceListCommonError::OutsideSource {
                        line_num,
                        keyword: "fluxdensity",
                    }
                    .into());
                }
                if !in_component {
                    return Err(ReadSourceListCommonError::OutsideComponent {
                        line_num,
                        keyword: "fluxdensity",
                    }
                    .into());
                }

                let fd_unit = match items.next() {
                    Some("Jy") => 1.0,
                    Some(u) => {
                        return Err(ReadSourceListAOError::UnhandledFluxDensityUnits {
                            line_num,
                            units: u.to_string(),
                        }
                        .into())
                    }
                    None => {
                        return Err(
                            ReadSourceListAOError::IncompleteFluxDensityLine(line_num).into()
                        )
                    }
                };

                let i = match items.next() {
                    Some(f) => parse_float(f, line_num)?,
                    None => {
                        return Err(
                            ReadSourceListAOError::IncompleteFluxDensityLine(line_num).into()
                        )
                    }
                };
                let q = match items.next() {
                    Some(f) => parse_float(f, line_num)?,
                    None => {
                        return Err(
                            ReadSourceListAOError::IncompleteFluxDensityLine(line_num).into()
                        )
                    }
                };
                let u = match items.next() {
                    Some(f) => parse_float(f, line_num)?,
                    None => {
                        return Err(
                            ReadSourceListAOError::IncompleteFluxDensityLine(line_num).into()
                        )
                    }
                };
                let v = match items.next() {
                    Some(f) => parse_float(f, line_num)?,
                    None => {
                        return Err(
                            ReadSourceListAOError::IncompleteFluxDensityLine(line_num).into()
                        )
                    }
                };

                if in_sed {
                    match &mut components.iter_mut().last().unwrap().flux_type {
                        FluxDensityType::PowerLaw { fd, .. } => {
                            fd.i = i * fd_unit;
                            fd.q = q * fd_unit;
                            fd.u = u * fd_unit;
                            fd.v = v * fd_unit;
                        }
                        _ => unreachable!(),
                    }
                } else if in_measurement {
                    match &mut components.iter_mut().last().unwrap().flux_type {
                        FluxDensityType::List { fds } => {
                            // If both bools are false, then we need to make a
                            // new `FluxDensity` struct.
                            if !measurement_got_freq && !measurement_got_fd {
                                fds.push(FluxDensity {
                                    freq: 0.0,
                                    i: i * fd_unit,
                                    q: q * fd_unit,
                                    u: u * fd_unit,
                                    v: v * fd_unit,
                                });
                                measurement_got_fd = true;
                            } else {
                                // We should edit the last `FluxDensity` struct.
                                let fd = fds.iter_mut().last().unwrap();
                                fd.i = i * fd_unit;
                                fd.q = q * fd_unit;
                                fd.u = u * fd_unit;
                                fd.v = v * fd_unit;
                            }
                        }
                        _ => unreachable!(),
                    }
                } else {
                    unreachable!()
                }
            }

            Some("spectral-index") => {
                if !in_source {
                    return Err(ReadSourceListCommonError::OutsideSource {
                        line_num,
                        keyword: "spectral-index",
                    }
                    .into());
                }
                if !in_component {
                    return Err(ReadSourceListCommonError::OutsideComponent {
                        line_num,
                        keyword: "spectral-index",
                    }
                    .into());
                }

                match items.next() {
                    Some("{") => (),
                    _ => return Err(ReadSourceListAOError::MissingOpeningCurly(line_num).into()),
                }

                let spec_index = match items.next() {
                    Some(f) => parse_float(f, line_num)?,
                    None => {
                        return Err(
                            ReadSourceListAOError::IncompleteFluxDensityLine(line_num).into()
                        )
                    }
                };
                // I don't know what this is.
                let _: f64 = match items.next() {
                    Some(f) => parse_float(f, line_num)?,
                    None => {
                        return Err(
                            ReadSourceListAOError::IncompleteFluxDensityLine(line_num).into()
                        )
                    }
                };

                match items.next() {
                    Some("}") => (),
                    _ => return Err(ReadSourceListAOError::MissingClosingCurly(line_num).into()),
                }

                match &mut components.iter_mut().last().unwrap().flux_type {
                    FluxDensityType::PowerLaw { si, .. } => *si = spec_index,
                    _ => unreachable!(),
                }
            }

            Some("}") => {
                if in_sed {
                    in_sed = false;
                } else if in_measurement {
                    in_measurement = false;
                    measurement_got_freq = false;
                    measurement_got_fd = false;
                } else if in_component {
                    in_component = false;
                    measurement_made_fd = false;

                    // Check that the last component struct added actually has flux
                    // densities.
                    match &components.iter().last().unwrap().flux_type {
                        FluxDensityType::PowerLaw { fd, .. } => {
                            if (fd.i + fd.q + fd.u + fd.v) < 1e-6 {
                                return Err(
                                    ReadSourceListCommonError::NoFluxDensities(line_num).into()
                                );
                            }
                        }
                        FluxDensityType::List { fds } => {
                            let fd = &fds.last().unwrap();
                            if (fd.i + fd.q + fd.u + fd.v) < 1e-6 {
                                return Err(
                                    ReadSourceListCommonError::NoFluxDensities(line_num).into()
                                );
                            }
                        }
                        _ => unreachable!(),
                    }
                } else if in_source {
                    let mut source = Source { components: vec![] };
                    source.components.append(&mut components);

                    // Ensure that the sum of each Stokes' flux densities is
                    // positive.
                    let mut sum_i = 0.0;
                    let mut sum_q = 0.0;
                    let mut sum_u = 0.0;
                    let mut sum_v = 0.0;
                    for c in &source.components {
                        match &c.flux_type {
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

                            FluxDensityType::List { fds } => {
                                for fd in fds {
                                    sum_i += fd.i;
                                    sum_q += fd.q;
                                    sum_u += fd.u;
                                    sum_v += fd.v;
                                }
                            }
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

                    source_list.insert(source_name.clone(), source);

                    in_source = false;
                    source_name.clear();
                }
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

    // If we're still "in a source", but we've finished reading lines, then a
    // final closing curly bracket must be missing.
    if in_source {
        return Err(ReadSourceListCommonError::MissingEndSource(line_num).into());
    }

    if in_component {
        return Err(ReadSourceListCommonError::MissingEndComponent(line_num).into());
    }

    if in_sed {
        return Err(ReadSourceListAOError::MissingEndSed(line_num).into());
    }

    // Complain if no sources were read.
    if source_list.is_empty() {
        return Err(ReadSourceListCommonError::NoSources(line_num).into());
    }

    Ok(source_list)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;
    use std::io::Cursor;

    #[test]
    fn parse_source_1() {
        let mut sl = Cursor::new(
            r#"
skymodel fileformat 1.1
source {
  name "GLEAM J113423-172750"
  component {
    type point
    position 11h34m23.7854s -17d27m50.6772s
    sed {
      frequency 200 MHz
      fluxdensity Jy 8.9289444 0 0 0
      spectral-index { -0.81 0.00 }
    }
  }
}
"#,
        );

        let result = parse_source_list(&mut sl);
        assert!(result.is_ok(), "{:?}", result);
        let sl = result.unwrap();
        assert_eq!(sl.len(), 1);

        assert!(sl.contains_key("GLEAM J113423-172750"));
        let s = sl.get("GLEAM J113423-172750").unwrap();
        assert_eq!(s.components.len(), 1);
        let comp = &s.components[0];
        assert_abs_diff_eq!(comp.radec.ra, 3.0298759753097615, epsilon = 1e-10);
        assert_abs_diff_eq!(comp.radec.dec, -0.30480564447181374, epsilon = 1e-10);
        assert!(match comp.flux_type {
            FluxDensityType::PowerLaw { .. } => true,
            _ => false,
        });
        let (fd, si) = match &comp.flux_type {
            FluxDensityType::PowerLaw { fd, si } => (fd, si),
            _ => unreachable!(),
        };
        assert_abs_diff_eq!(fd.freq, 200e6, epsilon = 1e-10);
        assert_abs_diff_eq!(fd.i, 8.9289444, epsilon = 1e-10);
        assert_abs_diff_eq!(fd.q, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(*si, -0.81, epsilon = 1e-10);

        assert!(match comp.comp_type {
            ComponentType::Point => true,
            _ => false,
        });
    }

    #[test]
    fn parse_source_2() {
        let mut sl = Cursor::new(
            r#"source {
  name "GLEAM J114901-120412"
  component {
    type gaussian
    position 11h49m01.062s -12d04m12.1809s
    shape 155.7 148.7 87.9
    sed {
      frequency 200 MHz
      fluxdensity Jy 4.3064720 0 0 0
      spectral-index { -0.88 0.00 }
    }
  }
}"#,
        );

        let result = parse_source_list(&mut sl);
        assert!(result.is_ok(), "{:?}", result);
        let sl = result.unwrap();
        assert_eq!(sl.len(), 1);

        assert!(sl.contains_key("GLEAM J114901-120412"));
        let s = sl.get("GLEAM J114901-120412").unwrap();
        assert_eq!(s.components.len(), 1);
        let comp = &s.components[0];
        assert_abs_diff_eq!(comp.radec.ra, 3.09367332997935, epsilon = 1e-10);
        assert_abs_diff_eq!(comp.radec.dec, -0.2106621177436647, epsilon = 1e-10);
        assert!(match comp.flux_type {
            FluxDensityType::PowerLaw { .. } => true,
            _ => false,
        });
        let (fd, si) = match &comp.flux_type {
            FluxDensityType::PowerLaw { fd, si } => (fd, si),
            _ => unreachable!(),
        };
        assert_abs_diff_eq!(fd.freq, 200e6, epsilon = 1e-10);
        assert_abs_diff_eq!(fd.i, 4.306472, epsilon = 1e-10);
        assert_abs_diff_eq!(fd.q, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(*si, -0.88, epsilon = 1e-10);

        assert!(match comp.comp_type {
            ComponentType::Gaussian { .. } => true,
            _ => false,
        });
        let (maj, min, pa) = match &comp.comp_type {
            ComponentType::Gaussian { maj, min, pa } => (maj, min, pa),
            _ => unreachable!(),
        };
        assert_abs_diff_eq!(*maj, 0.0007548549014875475, epsilon = 1e-10);
        assert_abs_diff_eq!(*min, 0.0007209179438098799, epsilon = 1e-10);
        assert_abs_diff_eq!(*pa, 1.5341444125030157, epsilon = 1e-10);
    }

    #[test]
    fn parse_source_3() {
        let mut sl = Cursor::new(
            r#"source {
  name "3C444"
  component {
    type point
    position -01h45m34.4s -17d01m30s
    measurement {
      frequency 167.355 MHz
      fluxdensity Jy 37.0557 0 0 37.0557
    }
    measurement {
      frequency 197.435 MHz
      fluxdensity Jy 32.5745 0 0 32.5745
    }
  }
}"#,
        );

        let result = parse_source_list(&mut sl);
        assert!(result.is_ok(), "{:?}", result);
        let sl = result.unwrap();
        assert_eq!(sl.len(), 1);

        assert!(sl.contains_key("3C444"));
        let s = sl.get("3C444").unwrap();
        assert_eq!(s.components.len(), 1);
        let comp = &s.components[0];
        assert_abs_diff_eq!(comp.radec.ra, 5.82253473993655, epsilon = 1e-10);
        assert_abs_diff_eq!(comp.radec.dec, -0.2971423051520346, epsilon = 1e-10);
        assert!(matches!(comp.flux_type, FluxDensityType::List { .. }));
        match &comp.flux_type {
            FluxDensityType::List { fds } => {
                assert_abs_diff_eq!(fds[0].freq, 167.355e6, epsilon = 1e-10);
                assert_abs_diff_eq!(fds[0].i, 37.0557, epsilon = 1e-10);
                assert_abs_diff_eq!(fds[0].q, 0.0, epsilon = 1e-10);
                assert_abs_diff_eq!(fds[0].v, 37.0557, epsilon = 1e-10);
                assert_abs_diff_eq!(fds[1].freq, 197.435e6, epsilon = 1e-10);
                assert_abs_diff_eq!(fds[1].i, 32.5745, epsilon = 1e-10);
                assert_abs_diff_eq!(fds[1].q, 0.0, epsilon = 1e-10);
                assert_abs_diff_eq!(fds[1].v, 32.5745, epsilon = 1e-10);
            }
            _ => unreachable!(),
        };

        assert!(matches!(comp.comp_type, ComponentType::Point { .. }));
    }

    #[test]
    fn parse_source_4() {
        let mut sl = Cursor::new(
            r#"source {
  name "3C444"
  component {
    type point
    position -01h45m34.4s -17d01m30s
    measurement {
      frequency 167.355 MHz
      fluxdensity Jy 37.0557 0 0 37.0557
    }
    measurement {
      frequency 197.435 MHz
      fluxdensity Jy 32.5745 0 0 32.5745
    }
  }
  component {
    type gaussian
    position 11h49m01.062s -12d04m12.1809s
    shape 155.7 148.7 87.9
    sed {
      frequency 200 MHz
      fluxdensity Jy 4.3064720 0 0 0
      spectral-index { -0.88 0.00 }
    }
  }
}"#,
        );

        let result = parse_source_list(&mut sl);
        assert!(result.is_ok(), "{:?}", result);
        let sl = result.unwrap();
        assert_eq!(sl.len(), 1);

        assert!(sl.contains_key("3C444"));
        let s = sl.get("3C444").unwrap();
        assert_eq!(s.components.len(), 2);
        let comp = &s.components[0];
        assert_abs_diff_eq!(comp.radec.ra, 5.82253473993655, epsilon = 1e-10);
        assert_abs_diff_eq!(comp.radec.dec, -0.2971423051520346, epsilon = 1e-10);
        assert!(matches!(comp.flux_type, FluxDensityType::List { .. }));
        match &comp.flux_type {
            FluxDensityType::List { fds } => {
                assert_abs_diff_eq!(fds[0].freq, 167.355e6, epsilon = 1e-10);
                assert_abs_diff_eq!(fds[0].i, 37.0557, epsilon = 1e-10);
                assert_abs_diff_eq!(fds[0].q, 0.0, epsilon = 1e-10);
                assert_abs_diff_eq!(fds[0].v, 37.0557, epsilon = 1e-10);
                assert_abs_diff_eq!(fds[1].i, 32.5745, epsilon = 1e-10);
                assert_abs_diff_eq!(fds[1].q, 0.0, epsilon = 1e-10);
                assert_abs_diff_eq!(fds[1].v, 32.5745, epsilon = 1e-10);
            }
            _ => unreachable!(),
        };

        assert!(matches!(comp.comp_type, ComponentType::Point { .. }));

        let comp = &s.components[1];
        assert_abs_diff_eq!(comp.radec.ra, 3.09367332997935, epsilon = 1e-10);
        assert_abs_diff_eq!(comp.radec.dec, -0.2106621177436647, epsilon = 1e-10);
        assert!(matches!(comp.flux_type, FluxDensityType::PowerLaw { .. }));
        match &comp.flux_type {
            FluxDensityType::PowerLaw { si, fd } => {
                assert_abs_diff_eq!(*si, -0.88, epsilon = 1e-10);
                assert_abs_diff_eq!(fd.freq, 200e6, epsilon = 1e-10);
                assert_abs_diff_eq!(fd.i, 4.306472, epsilon = 1e-10);
                assert_abs_diff_eq!(fd.q, 0.0, epsilon = 1e-10);
            }
            _ => unreachable!(),
        };

        assert!(matches!(comp.comp_type, ComponentType::Gaussian { .. }));
        match &comp.comp_type {
            ComponentType::Gaussian { maj, min, pa } => {
                assert_abs_diff_eq!(*maj, 0.0007548549014875475, epsilon = 1e-10);
                assert_abs_diff_eq!(*min, 0.0007209179438098799, epsilon = 1e-10);
                assert_abs_diff_eq!(*pa, 1.5341444125030157, epsilon = 1e-10);
            }
            _ => unreachable!(),
        };
    }

    #[test]
    fn parse_source_5() {
        let mut sl = Cursor::new(
            r#"source {
  name "3C444"
  component {
    type point
    position -01h45m34.4s -17d01m30s
    measurement {
      frequency 167.355 MHz
      fluxdensity Jy 37.0557 0 0 37.0557
    }
    measurement {
      frequency 197.435 MHz
      fluxdensity Jy 32.5745 0 0 32.5745
    }
  }
  component {
    type gaussian
    position 11h49m01.062s -12d04m12.1809s
    shape 155.7 148.7 87.9
    measurement {
      frequency 157.355 MHz
      fluxdensity Jy 27.0557 0 0 37.0557
    }
    measurement {
      frequency 187.435 MHz
      fluxdensity Jy 22.5745 0 0 32.5745
    }
  }
}"#,
        );

        let result = parse_source_list(&mut sl);
        assert!(result.is_ok(), "{:?}", result);
        let sl = result.unwrap();
        assert_eq!(sl.len(), 1);

        assert!(sl.contains_key("3C444"));
        let s = sl.get("3C444").unwrap();
        assert_eq!(s.components.len(), 2);
        let comp = &s.components[0];
        assert_abs_diff_eq!(comp.radec.ra, 5.82253473993655, epsilon = 1e-10);
        assert_abs_diff_eq!(comp.radec.dec, -0.2971423051520346, epsilon = 1e-10);
        assert!(matches!(comp.flux_type, FluxDensityType::List { .. }));
        match &comp.flux_type {
            FluxDensityType::List { fds } => {
                assert_abs_diff_eq!(fds[0].freq, 167.355e6, epsilon = 1e-10);
                assert_abs_diff_eq!(fds[0].i, 37.0557, epsilon = 1e-10);
                assert_abs_diff_eq!(fds[0].q, 0.0, epsilon = 1e-10);
                assert_abs_diff_eq!(fds[0].v, 37.0557, epsilon = 1e-10);
                assert_abs_diff_eq!(fds[1].freq, 197.435e6, epsilon = 1e-10);
                assert_abs_diff_eq!(fds[1].i, 32.5745, epsilon = 1e-10);
                assert_abs_diff_eq!(fds[1].q, 0.0, epsilon = 1e-10);
                assert_abs_diff_eq!(fds[1].v, 32.5745, epsilon = 1e-10);
            }
            _ => unreachable!(),
        };

        assert!(matches!(comp.comp_type, ComponentType::Point { .. }));

        let comp = &s.components[1];
        assert_abs_diff_eq!(comp.radec.ra, 3.09367332997935, epsilon = 1e-10);
        assert_abs_diff_eq!(comp.radec.dec, -0.2106621177436647, epsilon = 1e-10);
        assert!(matches!(comp.flux_type, FluxDensityType::List { .. }));
        match &comp.flux_type {
            FluxDensityType::List { fds } => {
                assert_abs_diff_eq!(fds[0].freq, 157.355e6, epsilon = 1e-10);
                assert_abs_diff_eq!(fds[0].i, 27.0557, epsilon = 1e-10);
                assert_abs_diff_eq!(fds[0].q, 0.0, epsilon = 1e-10);
                assert_abs_diff_eq!(fds[0].v, 37.0557, epsilon = 1e-10);
                assert_abs_diff_eq!(fds[1].freq, 187.435e6, epsilon = 1e-10);
                assert_abs_diff_eq!(fds[1].i, 22.5745, epsilon = 1e-10);
                assert_abs_diff_eq!(fds[1].q, 0.0, epsilon = 1e-10);
                assert_abs_diff_eq!(fds[1].v, 32.5745, epsilon = 1e-10);
            }
            _ => unreachable!(),
        };
    }
}
