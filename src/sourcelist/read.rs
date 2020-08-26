// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Source-list parsing.
 */

use nom::{
    bytes::complete::tag,
    character::complete::{alpha1, multispace0, multispace1, newline},
    eof, many_till,
    multi::{many0, many1},
    named,
    number::complete::double,
    take_until, take_until1, IResult,
};

use super::error::ErrorKind;
use super::types::flux_density::*;
use super::types::source::*;
use crate::*;

/// Parse any line that starts with a hash symbol (#), or has whitespace before
/// a hash symbol. The result is discarded.
fn comment_parser(s: &str) -> IResult<&str, ()> {
    // Ignore leading whitespace.
    let (s, _) = multispace0(s)?;
    // If we hit a hash symbol...
    let (s, _) = tag("#")(s)?;
    // ... consume the input until we hit a newline.
    named!(remainder<&str, &str>, take_until!("\n"));
    let (s, _) = remainder(s)?;
    let (s, _) = newline(s)?;
    Ok((s, ()))
}

/// Parse a single `FluxDensity` struct, e.g. FREQ 180e+6 0.0000105513 0 0 0
#[allow(clippy::many_single_char_names)]
fn flux_density_parser(s: &str) -> IResult<&str, FluxDensity> {
    // Parse any comments.
    let (s, _) = many0(comment_parser)(s)?;

    let (s, _) = tag("FREQ")(s)?;
    let (s, _) = multispace1(s)?;
    let (s, freq) = double(s)?;
    let (s, _) = multispace1(s)?;
    // The Stokes I flux density.
    let (s, i) = double(s)?;
    let (s, _) = multispace1(s)?;
    let (s, q) = double(s)?;
    let (s, _) = multispace1(s)?;
    let (s, u) = double(s)?;
    let (s, _) = multispace1(s)?;
    let (s, v) = double(s)?;
    let (s, _) = newline(s)?;
    Ok((s, FluxDensity { freq, i, q, u, v }))
}

/// Parse a single COMPONENT.
fn component_parser(s: &str) -> IResult<&str, SourceComponent> {
    // Parse any comments.
    let (s, _) = many0(comment_parser)(s)?;

    let (s, _) = tag("COMPONENT")(s)?;
    let (s, _) = multispace1(s)?;
    // Only alphabetical characters allowed here; nothing like "SHAPELET2"
    // allowed.
    let (s, component_type) = alpha1(s)?;
    let _component_type = match component_type {
        "POINT" => ComponentType::Point,
        _ => unimplemented!(),
    };
    let (s, _) = multispace1(s)?;
    // `ra` is specified as an hour angle.
    let (s, ra) = double(s)?;
    let (s, _) = multispace1(s)?;
    let (s, dec) = double(s)?;
    let (s, _) = multispace1(s)?;

    // Parse the flux density lines. At least one must exist.
    let (s, flux_densities) = many1(flux_density_parser)(s)?;

    let (s, _) = tag("ENDCOMPONENT")(s)?;
    let (s, _) = newline(s)?;

    Ok((
        s,
        SourceComponent {
            radec: RADec::new(ra * *DH2R, dec.to_radians()),
            flux_densities,
            ctype: ComponentType::Point,
        },
    ))
}

/// Parse a single SOURCE.
fn source_parser(s: &str) -> IResult<&str, Source> {
    // Parse any comments.
    let (s, _) = many0(comment_parser)(s)?;

    let (s, _) = tag("SOURCE")(s)?;
    let (s, _) = multispace1(s)?;
    // Parse the source name.
    named!(sname<&str, &str>, take_until1!(" "));
    let (s, name) = sname(s)?;
    // TODO: Handle component type counts?
    // Ignore component type counts.
    named!(counts<&str, &str>, take_until1!("\n"));
    let (s, _) = counts(s)?;
    let (s, _) = newline(s)?;

    // Parse any comments.
    let (s, _) = many0(comment_parser)(s)?;

    // Parse the components. At least one must exist.
    let (s, components) = many1(component_parser)(s)?;

    let (s, _) = tag("ENDSOURCE")(s)?;
    // Trailing newlines are optional.
    // TODO: Also handle any comments after a ENDSOURCE.
    let (s, _) = many0(newline)(s)?;

    Ok((
        s,
        Source {
            name: name.to_string(),
            components,
        },
    ))
}

/// Parse multiple SOURCEs.
fn parse_sources(s: &str) -> IResult<&str, Vec<Source>> {
    named!(srcs<&str, (Vec<Source>, &str)>,
           many_till!(source_parser, eof!()));
    let (s, (sources, _)) = srcs(s)?;
    Ok((s, sources))
}

/// Parse a source list into a vector of `Source`s.
pub fn parse_source_list(contents: &str) -> Result<Vec<Source>, ErrorKind> {
    // So that the caller doesn't have to know about the nom crate, unwrap the
    // nom result here.
    let (_, s) = match parse_sources(contents) {
        Ok(r) => r,
        Err(e) => return Err(ErrorKind::ParseError(e.to_string())),
    };
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;

    #[test]
    fn flux_density_parser_test_1() {
        let s = "FREQ 180e+6 0.0000105513 0 0 0\n";
        let (_, fd) = flux_density_parser(s).expect("Parser failed");
        assert_abs_diff_eq!(fd.freq, 180e6, epsilon = 5e-7);
        assert_abs_diff_eq!(fd.i, 0.0000105513, epsilon = 5e-7);
        assert_abs_diff_eq!(fd.q, 0.0, epsilon = 5e-7);
        assert_abs_diff_eq!(fd.u, 0.0, epsilon = 5e-7);
        assert_abs_diff_eq!(fd.v, 0.0, epsilon = 5e-7);
    }

    #[test]
    fn flux_density_parser_test_2() {
        let s = "FREQ 165e+6 0.0002097382 0 0 0\n";
        let (_, fd) = flux_density_parser(s).expect("Parser failed");
        assert_abs_diff_eq!(fd.freq, 165e6, epsilon = 5e-7);
        assert_abs_diff_eq!(fd.i, 0.0002097382, epsilon = 5e-7);
        assert_abs_diff_eq!(fd.q, 0.0, epsilon = 5e-7);
        assert_abs_diff_eq!(fd.u, 0.0, epsilon = 5e-7);
        assert_abs_diff_eq!(fd.v, 0.0, epsilon = 5e-7);
    }

    #[test]
    fn flux_density_parser_empty1() {
        let s = "";
        assert!(flux_density_parser(s).is_err());
    }

    #[test]
    fn flux_density_parser_empty2() {
        let s = "# FREQ 165e+6 0.0002097382 0 0 0\n";
        assert!(flux_density_parser(s).is_err());
    }

    #[test]
    fn component_parser_test_1() {
        let s = r#"COMPONENT POINT 3.40182 -37.5551
FREQ 180e+6 0.0000105513 0 0 0
FREQ 165e+6 0.0002097382 0 0 0
ENDCOMPONENT
"#;
        let (_, c) = component_parser(s).expect("Parser failed");
        assert_abs_diff_eq!(c.radec.ra, 3.40182_f64 * *DH2R, epsilon = 5e-7);
        assert_abs_diff_eq!(c.radec.dec, (-37.5551_f64).to_radians(), epsilon = 5e-7);
        assert_eq!(c.flux_densities.len(), 2);
    }

    #[test]
    fn component_parser_empty1() {
        let s = r#"COMPONENT POINT 1 2
ENDCOMPONENT"#;
        assert!(component_parser(s).is_err());
    }

    #[test]
    fn component_parser_empty2() {
        let s = r#"COMPONENT POINT 1 2
# FREQ 180e+6 0.0000105513 0 0 0
ENDCOMPONENT"#;
        assert!(component_parser(s).is_err());
    }

    #[test]
    fn source_parser_test_1() {
        let s = r#"SOURCE VLA_ForA P 129561 G 0 S 0 0
COMPONENT POINT 3.40182 -37.5551
FREQ 180e+6 0.0000105513 0 0 0
ENDCOMPONENT
COMPONENT POINT 3.40166 -37.5551
FREQ 180e+6 0.0002097382 0 0 0
ENDCOMPONENT
ENDSOURCE
"#;
        let (_, src) = source_parser(s).expect("Parser failed");
        assert_eq!(src.name, "VLA_ForA");
        assert_eq!(src.components.len(), 2);
    }

    #[test]
    fn source_parser_with_comments_test_1() {
        let s = r#"#SOURCE VLA_ForA P 129561 G 0 S 0 0
SOURCE testsrc P 129561 G 0 S 0 0
 # COMPONENT POINT 3.40182 -37.5551
COMPONENT POINT 1.40182 -37.5551
FREQ 180e+6 0.0000105513 0 0 0
ENDCOMPONENT
COMPONENT POINT 3.40166 -37.5551
# FREQ 180e+6 0.0002097382 0 0 0
FREQ 180e+6 0.0003097382 0 0 0
ENDCOMPONENT
ENDSOURCE
"#;
        let (_, src) = source_parser(s).expect("Parser failed");
        assert_eq!(src.name, "testsrc");
        assert_eq!(src.components.len(), 2);
        assert_abs_diff_eq!(
            src.components[0].radec.ra,
            0.3669956178046037,
            epsilon = 1e-10
        );
        assert_abs_diff_eq!(
            src.components[1].flux_densities[0].i,
            0.0003097382,
            epsilon = 1e-10
        );
    }

    #[test]
    fn source_parser_empty1() {
        let s = r#"SOURCE VLA_ForA P 129561 G 0 S 0 0
ENDSOURCE"#;
        assert!(source_parser(s).is_err());
    }

    #[test]
    fn source_parser_empty2() {
        let s = r#"SOURCE VLA_ForA P 129561 G 0 S 0 0
# COMPONENT POINT 3.40166 -37.5551
# FREQ 180e+6 0.0002097382 0 0 0
ENDSOURCE"#;
        assert!(source_parser(s).is_err());
    }

    #[test]
    fn parse_sources_test() {
        let s = r#"SOURCE VLA_ForA P 129561 G 0 S 0 0
COMPONENT POINT 3.40182 -37.5551
FREQ 180e+6 0.0000105513 0 0 0
ENDCOMPONENT
ENDSOURCE
SOURCE VLA_ForB P 5 G 1 S 1 2
COMPONENT POINT 3.40166 -37.5551
FREQ 180e+6 0.0002097382 0 0 0
ENDCOMPONENT
ENDSOURCE
"#;
        let (_, srcs) = parse_sources(s).expect("Parser failed");
        assert_eq!(srcs.len(), 2);
        assert_eq!(srcs[0].name, "VLA_ForA");
        assert_eq!(srcs[0].components.len(), 1);
        assert_eq!(srcs[1].name, "VLA_ForB");
        assert_eq!(srcs[1].components.len(), 1);
    }
}
