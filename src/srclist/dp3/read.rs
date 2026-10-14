// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Parsing of DP3/BBS-style ("makesourcedb") text sky models.
//!
//! This mirrors DP3's reader (`sky_model/ReadSkyModel.cc`) and the evaluation
//! of spectral terms in DP3's `base/PointSource.cc`:
//!
//! - The format line (`Format = Name, Type, ...`, or the older
//!   `# (Name, Type, ...) = format`) names the fields; names are
//!   case-insensitive, may come in any order, and may carry a default
//!   (`ReferenceFrequency='150e6'`) or a fixed value (`I=fixed'1.0'`).
//! - A line with an empty Name and a Patch defines a patch. Each patch becomes
//!   one hyperdrive [`Source`] named after the patch, containing all components
//!   that name it in their Patch column. Components without a patch become a
//!   source of their own named by the component name.
//! - Only POINT and GAUSSIAN components are supported.
//!
//! Spectral terms (DP3 only applies them to Stokes I) with LogarithmicSI=true
//! are `S = I (ν/ν0)^(c0 + c1 log10(ν/ν0) + c2 log10²(ν/ν0) + ...)`. One term
//! is a hyperdrive power law; two terms are a hyperdrive curved power law with
//! `q = c1 / ln(10)`; no terms is a flat spectrum (a power law with spectral
//! index 0, which is what DP3 predicts). Anything else (three or more
//! non-zero log terms, or LogarithmicSI=false polynomials
//! `S = I + c0 x + c1 x² + ...` with `x = ν/ν0 - 1`) is evaluated at
//! [`LIST_FREQS_HZ`] into a flux-density list.

use std::{
    f64::consts::{FRAC_PI_2, LN_10},
    io::BufRead,
    path::Path,
};

use indexmap::IndexMap;
use log::debug;
use marlu::RADec;
use vec1::Vec1;

use crate::{
    cli::Warn,
    srclist::{
        error::{ReadSourceListDp3Error, ReadSourceListError},
        ComponentType, FluxDensity, FluxDensityType, Source, SourceComponent, SourceList,
    },
};

/// The number of frequencies that spectral models that cannot be represented
/// exactly by hyperdrive are evaluated at.
const NUM_LIST_FREQS: usize = 16;
/// The lowest frequency that spectral models that cannot be represented exactly
/// by hyperdrive are evaluated at \[Hz\].
const LIST_FREQ_MIN_HZ: f64 = 70e6;
/// The highest frequency that spectral models that cannot be represented
/// exactly by hyperdrive are evaluated at \[Hz\].
const LIST_FREQ_MAX_HZ: f64 = 300e6;

lazy_static::lazy_static! {
    /// The frequencies (log-spaced between [`LIST_FREQ_MIN_HZ`] and
    /// [`LIST_FREQ_MAX_HZ`]) at which spectral models that cannot be
    /// represented exactly by hyperdrive are evaluated \[Hz\].
    static ref LIST_FREQS_HZ: [f64; NUM_LIST_FREQS] = {
        let mut freqs = [0.0; NUM_LIST_FREQS];
        let ratio = (LIST_FREQ_MAX_HZ / LIST_FREQ_MIN_HZ).ln();
        for (i, f) in freqs.iter_mut().enumerate() {
            *f = LIST_FREQ_MIN_HZ * (ratio * i as f64 / (NUM_LIST_FREQS - 1) as f64).exp();
        }
        freqs
    };
}

/// The reference frequency used for components without spectral terms and
/// without a reference frequency (their flux density is flat, so this is
/// arbitrary) \[Hz\].
const FLAT_SPECTRUM_FREQ_HZ: f64 = 150e6;

/// The fields that DP3 knows about. The order matters: the first
/// [`NUM_STANDARD_FIELDS`] are used (in this order) when the format string is
/// empty, as DP3 does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Field {
    Name,
    Type,
    RefType,
    Ra,
    Dec,
    I,
    Q,
    U,
    V,
    SpectralIndex,
    LogarithmicSI,
    ReferenceFrequency,
    MajorAxis,
    MinorAxis,
    Orientation,
    OrientationIsAbsolute,
    RotationMeasure,
    PolarizedFraction,
    PolarizationAngle,
    ReferenceWavelength,
    IShapelet,
    QShapelet,
    UShapelet,
    VShapelet,
    Category,
    Patch,
    Rah,
    Rad,
    Ram,
    Ras,
    Dech,
    Decd,
    Decm,
    Decs,
}

const NUM_FIELDS: usize = 34;
const NUM_STANDARD_FIELDS: usize = 24;

const FIELDS: [(Field, &str); NUM_FIELDS] = [
    (Field::Name, "Name"),
    (Field::Type, "Type"),
    (Field::RefType, "RefType"),
    (Field::Ra, "Ra"),
    (Field::Dec, "Dec"),
    (Field::I, "I"),
    (Field::Q, "Q"),
    (Field::U, "U"),
    (Field::V, "V"),
    (Field::SpectralIndex, "SpectralIndex"),
    (Field::LogarithmicSI, "LogarithmicSI"),
    (Field::ReferenceFrequency, "ReferenceFrequency"),
    (Field::MajorAxis, "MajorAxis"),
    (Field::MinorAxis, "MinorAxis"),
    (Field::Orientation, "Orientation"),
    (Field::OrientationIsAbsolute, "OrientationIsAbsolute"),
    (Field::RotationMeasure, "RotationMeasure"),
    (Field::PolarizedFraction, "PolarizedFraction"),
    (Field::PolarizationAngle, "PolarizationAngle"),
    (Field::ReferenceWavelength, "ReferenceWavelength"),
    (Field::IShapelet, "IShapelet"),
    (Field::QShapelet, "QShapelet"),
    (Field::UShapelet, "UShapelet"),
    (Field::VShapelet, "VShapelet"),
    (Field::Category, "Category"),
    (Field::Patch, "Patch"),
    (Field::Rah, "rah"),
    (Field::Rad, "rad"),
    (Field::Ram, "ram"),
    (Field::Ras, "ras"),
    (Field::Dech, "dech"),
    (Field::Decd, "decd"),
    (Field::Decm, "decm"),
    (Field::Decs, "decs"),
];

impl Field {
    fn from_lowercase(name: &str) -> Option<Field> {
        FIELDS
            .iter()
            .find(|(_, n)| n.eq_ignore_ascii_case(name))
            .map(|(f, _)| *f)
    }

    fn name(self) -> &'static str {
        FIELDS[self as usize].1
    }
}

/// A single field of the format line.
#[derive(Debug)]
struct FieldSpec {
    /// The separator following this field's value.
    sep: u8,
    /// The value to use if the field is empty on a line.
    default: Option<String>,
    /// The value to always use for this field (nothing is read from lines).
    fixed: Option<String>,
}

/// The parsed format line.
#[derive(Debug)]
struct Format {
    specs: Vec<FieldSpec>,
    /// For each [`Field`], the index into `specs` (if the field is present).
    index: [Option<usize>; NUM_FIELDS],
}

fn ltrim(b: &[u8], mut st: usize, end: usize) -> usize {
    while st < end && (b[st] == b' ' || b[st] == b'\t') {
        st += 1;
    }
    st
}

fn rtrim(b: &[u8], st: usize, mut end: usize) -> usize {
    while end > st && (b[end - 1] == b' ' || b[end - 1] == b'\t') {
        end -= 1;
    }
    end
}

fn is_alnum_or_underscore(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

/// Parse the format string (the part after "format =" or inside the
/// parentheses of "# (...) = format"). This is a port of DP3's `getFormat`.
fn parse_format(format: &str) -> Result<Format, String> {
    let b = format.as_bytes();
    let mut end = b.len();
    let mut st = ltrim(b, 0, end);
    end = rtrim(b, st, end);

    let mut specs = vec![];
    let mut index = [None; NUM_FIELDS];

    // An empty format means all standard fields, separated by commas (or by
    // the single given character).
    if st + 1 >= end {
        let sep = if st + 1 == end { b[st] } else { b',' };
        for (i, idx) in index.iter_mut().take(NUM_STANDARD_FIELDS).enumerate() {
            *idx = Some(i);
            specs.push(FieldSpec {
                sep,
                default: None,
                fixed: None,
            });
        }
        return Ok(Format { specs, index });
    }

    let handle_name = |name: &str, specs: &[FieldSpec], index: &mut [Option<usize>]| {
        let lname = name.to_ascii_lowercase();
        if lname == "spectralindexdegree" {
            return Err(
                "Use SpectralIndex=[v1,v2,...] instead of SpectralIndexDegree and SpectralIndex:i"
                    .to_string(),
            );
        }
        if !lname.is_empty() && lname != "dummy" {
            if let Some(f) = Field::from_lowercase(&lname) {
                index[f as usize] = Some(specs.len());
            } else {
                debug!("DP3 sky model: ignoring unknown field '{name}'");
            }
        }
        Ok(())
    };

    let mut i = st;
    while i < end {
        let c = b[i];
        if is_alnum_or_underscore(c) || (c == b':' && i > st) {
            i += 1;
            continue;
        }

        // End of a name.
        let name = &format[st..i];
        handle_name(name, &specs, &mut index)?;

        // See if a default or fixed value is given.
        let mut default = None;
        let mut fixed = None;
        i = ltrim(b, i, end);
        if i < end && b[i] == b'=' {
            i = ltrim(b, i + 1, end);
            let mut is_default = true;
            if i + 5 < end
                && format
                    .get(i..i + 5)
                    .is_some_and(|s| s.eq_ignore_ascii_case("fixed"))
            {
                is_default = false;
                i += 5;
            }
            if i >= end {
                return Err(format!("No value given after {name}="));
            }
            let quote = b[i];
            if quote != b'"' && quote != b'\'' {
                return Err(format!("Value after {name}= must start with a quote"));
            }
            let pos = match b[i + 1..end].iter().position(|&c| c == quote) {
                Some(p) => i + 1 + p,
                None => return Err(format!("No closing value quote given after {name}=")),
            };
            let value = format[i + 1..pos].to_string();
            if is_default {
                default = Some(value);
            } else {
                fixed = Some(value);
            }
            i = ltrim(b, pos + 1, end);
        }

        // Now look for a separator. (DP3 reads a NUL past the end of the
        // string, which means "the rest of the line".)
        let mut sep = b' ';
        let c = if i < end { b[i] } else { 0 };
        if !is_alnum_or_underscore(c) {
            if c == b'"' || c == b'\'' {
                return Err("A quote is found as separator; probably a quote around a value in the format string is missing".to_string());
            }
            if !c.is_ascii() {
                return Err(format!(
                    "Non-ASCII separator after field {name} is not supported"
                ));
            }
            sep = c;
            i = ltrim(b, i + 1, end);
        }
        specs.push(FieldSpec {
            sep,
            default,
            fixed,
        });
        st = i;
    }
    if st < end {
        // The last item was just a name.
        let name = &format[st..end];
        handle_name(name, &specs, &mut index)?;
        specs.push(FieldSpec {
            sep: b' ',
            default: None,
            fixed: None,
        });
    }

    // Make sure that RA and Dec are given correctly.
    for (main, h_or_d, d, m, s, what) in [
        (
            Field::Ra,
            Field::Rah,
            Field::Rad,
            Field::Ram,
            Field::Ras,
            "Ra",
        ),
        (
            Field::Dec,
            Field::Dech,
            Field::Decd,
            Field::Decm,
            Field::Decs,
            "Dec",
        ),
    ] {
        let given = |f: Field| index[f as usize].is_some();
        if given(h_or_d) && given(d) {
            return Err(format!(
                "{} and {} cannot both be used",
                h_or_d.name(),
                d.name()
            ));
        }
        let any_part = given(h_or_d) || given(d) || given(m) || given(s);
        if given(main) && any_part {
            return Err(format!("{what} cannot be combined with its h/d/m/s parts"));
        }
        if !given(main) && !any_part {
            return Err(format!("No {what} info given"));
        }
    }

    Ok(Format { specs, index })
}

/// Get the next value by looking for the separator. The separator is ignored
/// in parts enclosed in quotes or square brackets. Square brackets can be
/// nested (they indicate arrays). This is a port of DP3's `nextValue`.
fn next_value(b: &[u8], sep: u8, mut st: usize, end: usize) -> Result<usize, String> {
    let mut nbracket = 0;
    let mut posbracket = 0;
    while st < end {
        let c = b[st];
        if c == b'\'' || c == b'"' {
            match b[st + 1..end].iter().position(|&x| x == c) {
                Some(p) => st += 1 + p,
                None => return Err(format!("Unbalanced quoted string at position {st}")),
            }
        } else if c == b'[' {
            nbracket += 1;
            posbracket = st;
        } else if c == b']' {
            if nbracket <= 0 {
                return Err(format!("Unbalanced square brackets at position {st}"));
            }
            nbracket -= 1;
        } else if nbracket == 0 && c == sep {
            return Ok(st);
        }
        st += 1;
    }
    if nbracket != 0 {
        return Err(format!(
            "Unbalanced square brackets at position {posbracket}"
        ));
    }
    Ok(end)
}

/// Remove a matching pair of quotes surrounding a value.
fn unquote(s: &str) -> &str {
    let b = s.as_bytes();
    if b.len() > 1 {
        let last = b.len() - 1;
        if (b[0] == b'"' && b[last] == b'"') || (b[0] == b'\'' && b[last] == b'\'') {
            return &s[1..last];
        }
    }
    s
}

/// Return the line without a trailing `#` comment (a `#` inside quotes is
/// kept).
fn strip_comment(line: &str) -> &str {
    let b = line.as_bytes();
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            q @ (b'\'' | b'"') => match b[i + 1..].iter().position(|&x| x == q) {
                Some(p) => i += 1 + p,
                None => return line,
            },
            b'#' => return &line[..i],
            _ => (),
        }
        i += 1;
    }
    line
}

/// If this line is a "format = ..." line, return the format string.
fn format_from_format_line(line: &str) -> Option<&str> {
    let t = line.trim_start_matches([' ', '\t']);
    // `get` (rather than slicing) so that a non-ASCII character within the
    // first six bytes yields `None` instead of panicking on a non-boundary.
    if t.get(..6).is_some_and(|h| h.eq_ignore_ascii_case("format")) {
        let rest = t[6..].trim_start_matches([' ', '\t']);
        rest.strip_prefix('=')
    } else {
        None
    }
}

/// If this line is a "# (...) = format" comment, return the format string.
fn format_from_comment_line(line: &str) -> Option<&str> {
    let t = line.trim_matches([' ', '\t']);
    let t = t.strip_prefix('#')?.trim_start_matches([' ', '\t']);
    let t = t.strip_prefix('(')?;
    let t = t.trim_end_matches([' ', '\t']);
    // `split_at_checked` (rather than slicing) so that a non-ASCII character
    // within the last six bytes yields `None` instead of panicking.
    let (t, tail) = t.split_at_checked(t.len().checked_sub(6)?)?;
    if !tail.eq_ignore_ascii_case("format") {
        return None;
    }
    let t = t.trim_end_matches([' ', '\t']);
    let t = t.strip_suffix('=')?.trim_end_matches([' ', '\t']);
    t.strip_suffix(')')
}

/// Quickly determine if the file at the given path looks like a DP3/BBS sky
/// model, i.e. if it has a format line before its first data line.
pub(crate) fn looks_like_dp3(path: &Path) -> bool {
    let f = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return false,
    };
    let mut buf = std::io::BufReader::new(f);
    let mut line = String::new();
    // Don't read too much of a file that is perhaps binary.
    for _ in 0..1000 {
        line.clear();
        match buf.read_line(&mut line) {
            Ok(0) | Err(_) => return false,
            Ok(_) => (),
        }
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        if t.starts_with('#') {
            if format_from_comment_line(t).is_some() {
                return true;
            }
            continue;
        }
        return format_from_format_line(t).is_some();
    }
    false
}

/// Parse a real number; empty strings give the default.
fn parse_real(
    s: &str,
    default: f64,
    field: Field,
    line_num: u32,
) -> Result<f64, ReadSourceListDp3Error> {
    let s = s.trim();
    if s.is_empty() {
        return Ok(default);
    }
    s.parse().map_err(|_| ReadSourceListDp3Error::ParseFloat {
        line_num,
        field: field.name(),
        value: s.to_string(),
    })
}

/// Parse a boolean like DP3 does (only the first character matters); empty
/// strings give the default.
fn parse_bool(
    s: &str,
    default: bool,
    field: Field,
    line_num: u32,
) -> Result<bool, ReadSourceListDp3Error> {
    match s.trim().as_bytes().first() {
        None => Ok(default),
        Some(b't' | b'T' | b'1' | b'y' | b'Y') => Ok(true),
        Some(b'f' | b'F' | b'0' | b'n' | b'N') => Ok(false),
        Some(_) => Err(ReadSourceListDp3Error::ParseBool {
            line_num,
            field: field.name(),
            value: s.to_string(),
        }),
    }
}

/// Parse the leading unsigned integer of `b[*pos..]` (0 if there are no
/// digits), like casacore's `MUString::getuInt`.
fn read_uint(b: &[u8], pos: &mut usize) -> f64 {
    let mut v = 0.0;
    while *pos < b.len() && b[*pos].is_ascii_digit() {
        v = v * 10.0 + f64::from(b[*pos] - b'0');
        *pos += 1;
    }
    v
}

/// Length of the longest prefix of `b` that is a (possibly signed) decimal
/// floating-point number.
fn float_prefix_len(b: &[u8]) -> usize {
    let mut i = 0;
    if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
        i += 1;
    }
    let digits_start = i;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
    }
    let mut n_digits = i - digits_start;
    if i < b.len() && b[i] == b'.' {
        i += 1;
        let frac_start = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        n_digits += i - frac_start;
    }
    if n_digits == 0 {
        return 0;
    }
    if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
        let mut j = i + 1;
        if j < b.len() && (b[j] == b'+' || b[j] == b'-') {
            j += 1;
        }
        let exp_start = j;
        while j < b.len() && b[j].is_ascii_digit() {
            j += 1;
        }
        if j > exp_start {
            i = j;
        }
    }
    i
}

/// Parse the (possibly empty) unsigned decimal number of `b[*pos..]` (0 if
/// there is none), like casacore's `MUString::getDouble`.
fn read_udouble(s: &str, pos: &mut usize) -> f64 {
    let b = s.as_bytes();
    let len = float_prefix_len(&b[*pos..]);
    let v = if len == 0 {
        0.0
    } else {
        s[*pos..*pos + len].parse().unwrap_or(0.0)
    };
    *pos += len;
    v
}

/// Read an angle the way casacore's `MVAngle::read` does (as used by DP3's
/// `string2pos`), returning degrees. Accepted forms (separator characters are
/// case insensitive, `+` is an optional sign):
///
/// - `+d.m.s` (e.g. `-37.08.37.076`): degrees, arcminutes, arcseconds;
/// - `+dDmMsS` (e.g. `-37d08m37.076s`, also `10d`): degrees;
/// - `+h:m:s` or `+h:m` (e.g. `03:21:29.744`): hours;
/// - `+hHmMsS` (e.g. `03h21m29.744s`): hours;
/// - `+number[unit]` where unit is one of `deg`, `rad`, `arcmin`, `arcsec`,
///   `mas`, `'`, `"`, `h`, `min`, `s`; a number without a unit is in
///   **radians** (as in casacore).
///
/// Unlike casacore, trailing characters that are not part of these forms are
/// an error rather than silently ignored.
fn read_angle_deg(s: &str) -> Option<f64> {
    let s = s.trim();
    let b = s.as_bytes();
    let mut pos = 0;
    let sign = match b.first() {
        Some(b'-') => {
            pos += 1;
            -1.0
        }
        Some(b'+') => {
            pos += 1;
            1.0
        }
        _ => 1.0,
    };
    let peek = |pos: usize| b.get(pos).copied();

    let mut r = read_uint(b, &mut pos);
    // 0: not sexagesimal, 1: d, 2: h, 3: colon, 4: d.m.s
    let mut tp = 0;
    match peek(pos) {
        Some(b'.') => {
            let save = pos;
            pos += 1;
            let r1 = read_uint(b, &mut pos);
            if peek(pos) == Some(b'.') {
                pos += 1;
                r += r1 / 60.0 + read_udouble(s, &mut pos) / 3600.0;
                tp = 4;
            } else {
                pos = save;
            }
        }
        Some(b'd' | b'D') => {
            pos += 1;
            tp = 1;
        }
        Some(b'h' | b'H') => {
            pos += 1;
            tp = 2;
        }
        Some(b':') => {
            pos += 1;
            tp = 3;
        }
        _ => (),
    }
    if (1..=3).contains(&tp) {
        if matches!(peek(pos), Some(b'm' | b'M' | b':')) {
            tp = 0;
        } else {
            let r1 = read_uint(b, &mut pos);
            let tc_matches = matches!(
                (tp, peek(pos)),
                (3, Some(b':')) | (1 | 2, Some(b'm' | b'M'))
            );
            if tc_matches {
                pos += 1;
                r += r1 / 60.0 + read_udouble(s, &mut pos) / 3600.0;
                if tp != 3 && matches!(peek(pos), Some(b's' | b'S')) {
                    pos += 1;
                }
            } else if tp == 3
                && (peek(pos) == Some(b'Z')
                    || !matches!(peek(pos), Some(c) if c == b'.' || c == b'/' || c.is_ascii_alphabetic()))
            {
                r += r1 / 60.0;
            } else if tp == 1 && r1 == 0.0 && !matches!(peek(pos), Some(b'.' | b'/')) {
                // Something like "10d", meaning 10 degrees.
            } else {
                tp = 0;
            }
        }
    }
    // If there are trailing characters, try reading as a number with a unit
    // instead (e.g. "12deg" gets here).
    if tp != 0 && (pos == b.len() || (pos == b.len() - 1 && b[pos] == b'Z' && tp == 3)) {
        r *= sign;
        return Some(match tp {
            1 | 4 => r,
            _ => r * 15.0,
        });
    }

    // Number with a unit.
    let len = float_prefix_len(b);
    if len == 0 {
        return None;
    }
    let v: f64 = s[..len].parse().ok()?;
    let unit = s[len..].trim();
    let deg = match unit {
        "" | "rad" => v.to_degrees(),
        "deg" => v,
        "arcmin" | "'" => v / 60.0,
        "arcsec" | "\"" | "''" => v / 3600.0,
        "mas" => v / 3.6e6,
        "h" => v * 15.0,
        "min" => v / 4.0,
        "s" => v / 240.0,
        _ => return None,
    };
    Some(deg)
}

/// Warnings that are only emitted once per file (with counts).
#[derive(Default)]
struct DeferredWarnings {
    many_log_terms: usize,
    linear_si: usize,
    pol_with_spectral_terms: usize,
    orientation_is_absolute: usize,
    empty_patches: usize,
}

/// The format DP3 uses when a file has no format line (`ReadFormat` in
/// `sky_model/ReadSkyModel.cc`).
const DEFAULT_FORMAT: &str = "Name,Type,Ra,Dec,I,Q,U,V,MajorAxis,MinorAxis,Orientation";

/// Parse a buffer containing a DP3/BBS-style ("makesourcedb") text sky model
/// into a [`SourceList`].
///
/// Like DP3, if no format line is found before the first data line, the
/// default format ([`DEFAULT_FORMAT`]) is used and a warning is emitted.
/// (Auto-detection of the file type, [`looks_like_dp3`], still requires a
/// format line, so this fallback only applies when the type is given
/// explicitly.)
pub(crate) fn parse_source_list<T: BufRead>(
    buf: &mut T,
) -> Result<SourceList, ReadSourceListError> {
    let mut line = String::new();
    let mut line_num: u32 = 0;
    let mut format: Option<Format> = None;
    let mut comment_format: Option<(String, u32)> = None;
    // Patch/source name -> components. An IndexMap keeps the file's order.
    let mut sources: IndexMap<String, Vec<SourceComponent>> = IndexMap::new();
    let mut warnings = DeferredWarnings::default();

    loop {
        line.clear();
        if buf.read_line(&mut line)? == 0 {
            break;
        }
        line_num += 1;
        let l = line.trim_end_matches(['\n', '\r']);

        // Comments and blank lines.
        let first = l.trim_start_matches([' ', '\t']);
        if first.is_empty() {
            continue;
        }
        if first.starts_with('#') {
            if format.is_none() && comment_format.is_none() {
                if let Some(f) = format_from_comment_line(first) {
                    comment_format = Some((f.to_string(), line_num));
                }
            }
            continue;
        }

        // A "format = ..." line. Like DP3, only the first one counts (it must
        // be the first data line) and later ones are ignored.
        if let Some(f) = format_from_format_line(first) {
            if format.is_none() {
                format = Some(
                    parse_format(f)
                        .map_err(|msg| ReadSourceListDp3Error::InvalidFormat { line_num, msg })?,
                );
            }
            continue;
        }

        let format = match &mut format {
            Some(f) => f,
            None => match comment_format.take() {
                Some((f, f_line_num)) => {
                    format = Some(parse_format(&f).map_err(|msg| {
                        ReadSourceListDp3Error::InvalidFormat {
                            line_num: f_line_num,
                            msg,
                        }
                    })?);
                    format.as_mut().unwrap()
                }
                None => {
                    format!("DP3 sky model: No format line found before the first data line; using DP3's default format '{DEFAULT_FORMAT}'").warn();
                    format =
                        Some(parse_format(DEFAULT_FORMAT).expect("the default format is valid"));
                    format.as_mut().unwrap()
                }
            },
        };

        process_line(
            strip_comment(l),
            line_num,
            format,
            &mut sources,
            &mut warnings,
        )?;
    }

    let mut source_list = SourceList::new();
    for (name, comps) in sources {
        if comps.is_empty() {
            warnings.empty_patches += 1;
            debug!("DP3 sky model: patch '{name}' has no components; ignoring it");
            continue;
        }
        source_list.insert(
            name,
            Source {
                components: comps.into_boxed_slice(),
            },
        );
    }

    let DeferredWarnings {
        many_log_terms,
        linear_si,
        pol_with_spectral_terms,
        orientation_is_absolute,
        empty_patches,
    } = warnings;
    if many_log_terms > 0 {
        format!("DP3 sky model: {many_log_terms} components have 3 or more logarithmic spectral terms; these have been evaluated at {NUM_LIST_FREQS} frequencies between {} and {} MHz into flux-density lists", LIST_FREQ_MIN_HZ / 1e6, LIST_FREQ_MAX_HZ / 1e6).warn();
    }
    if linear_si > 0 {
        format!("DP3 sky model: {linear_si} components have LogarithmicSI=false spectral terms; these have been evaluated at {NUM_LIST_FREQS} frequencies between {} and {} MHz into flux-density lists", LIST_FREQ_MIN_HZ / 1e6, LIST_FREQ_MAX_HZ / 1e6).warn();
    }
    if pol_with_spectral_terms > 0 {
        format!("DP3 sky model: {pol_with_spectral_terms} components have non-zero Q, U or V and spectral terms; DP3 keeps Q, U and V constant with frequency, but hyperdrive scales them like I").warn();
    }
    if orientation_is_absolute > 0 {
        format!("DP3 sky model: {orientation_is_absolute} Gaussian components have OrientationIsAbsolute=true; hyperdrive evaluates Gaussian orientations in the phase-centre frame (DP3's default, OrientationIsAbsolute=false)").warn();
    }
    if empty_patches > 0 {
        format!("DP3 sky model: ignored {empty_patches} patches without components").warn();
    }

    if source_list.is_empty() {
        return Err(ReadSourceListDp3Error::NoComponents.into());
    }

    Ok(source_list)
}

/// Handle a single data (patch or component) line.
fn process_line(
    line: &str,
    line_num: u32,
    format: &Format,
    sources: &mut IndexMap<String, Vec<SourceComponent>>,
    warnings: &mut DeferredWarnings,
) -> Result<(), ReadSourceListDp3Error> {
    let malformed = |msg: String| ReadSourceListDp3Error::Malformed { line_num, msg };

    // Split the line into values, using the format's separators, defaults and
    // fixed values.
    let b = line.as_bytes();
    let end = b.len();
    let mut st = ltrim(b, 0, end);
    let mut values: Vec<&str> = Vec::with_capacity(format.specs.len());
    for spec in &format.specs {
        let mut value: &str = "";
        if let Some(fixed) = &spec.fixed {
            value = fixed;
        } else if st < end {
            let pos = next_value(b, spec.sep, st, end).map_err(malformed)?;
            value = &line[st..rtrim(b, st, pos)];
            st = ltrim(b, pos + 1, end);
        }
        if value.is_empty() {
            if let Some(default) = &spec.default {
                value = default;
            }
        }
        values.push(value);
    }
    let get = |f: Field| -> &str {
        match format.index[f as usize] {
            Some(i) => unquote(values[i]),
            None => "",
        }
    };

    let name = get(Field::Name);
    let patch = get(Field::Patch);

    let ref_type = get(Field::RefType);
    if !ref_type.is_empty() && !ref_type.eq_ignore_ascii_case("J2000") {
        return Err(ReadSourceListDp3Error::UnsupportedRefType {
            line_num,
            ref_type: ref_type.to_string(),
        });
    }

    // A patch definition.
    if name.is_empty() {
        if patch.is_empty() {
            return Err(ReadSourceListDp3Error::NoName { line_num });
        }
        // The patch position isn't used by hyperdrive (and DP3 replaces it
        // with the flux-weighted position of its components).
        sources.entry(patch.to_string()).or_default();
        return Ok(());
    }

    // A component.
    let comp_type_str = get(Field::Type);
    let is_gaussian = match comp_type_str.to_ascii_lowercase().as_str() {
        "" | "point" => false,
        "gaussian" => true,
        "disk" | "shapelet" => {
            return Err(ReadSourceListDp3Error::UnsupportedComponentType {
                line_num,
                comp_type: comp_type_str.to_string(),
            })
        }
        _ => {
            return Err(ReadSourceListDp3Error::InvalidComponentType {
                line_num,
                comp_type: comp_type_str.to_string(),
            })
        }
    };

    for f in [
        Field::RotationMeasure,
        Field::PolarizedFraction,
        Field::PolarizationAngle,
        Field::ReferenceWavelength,
    ] {
        if !get(f).is_empty() {
            return Err(ReadSourceListDp3Error::UnsupportedRotationMeasure { line_num });
        }
    }

    // Position.
    let read_pos = |main: Field,
                    h: Option<Field>,
                    d: Field,
                    m: Field,
                    s: Field,
                    is_dec: bool|
     -> Result<f64, ReadSourceListDp3Error> {
        if format.index[main as usize].is_some() {
            let value = get(main).trim();
            if value.is_empty() {
                return Err(ReadSourceListDp3Error::MissingPosition {
                    line_num,
                    name: name.to_string(),
                    field: main.name(),
                });
            }
            if is_dec && value.contains(':') {
                return Err(ReadSourceListDp3Error::ColonInDec {
                    line_num,
                    value: value.to_string(),
                });
            }
            return read_angle_deg(value).ok_or_else(|| ReadSourceListDp3Error::InvalidAngle {
                line_num,
                field: main.name(),
                value: value.to_string(),
            });
        }
        // Separate h/d, m and s fields.
        let mut found = false;
        let mut deg = 0.0;
        let mut is_hours = false;
        if let Some(h) = h.filter(|h| format.index[*h as usize].is_some()) {
            let v = get(h);
            if !v.is_empty() {
                deg = parse_real(v, 0.0, h, line_num)?;
                found = true;
            }
            is_hours = true;
        } else {
            let v = get(d);
            if !v.is_empty() {
                deg = parse_real(v, 0.0, d, line_num)?;
                found = true;
            }
        }
        let mut ms = 0.0;
        let v = get(m);
        if !v.is_empty() {
            ms = parse_real(v, 0.0, m, line_num)?;
            found = true;
        }
        let v = get(s);
        if !v.is_empty() {
            ms += parse_real(v, 0.0, s, line_num)? / 60.0;
            found = true;
        }
        if !found {
            return Err(ReadSourceListDp3Error::MissingPosition {
                line_num,
                name: name.to_string(),
                field: main.name(),
            });
        }
        if deg < 0.0 {
            deg -= ms / 60.0;
        } else {
            deg += ms / 60.0;
        }
        if is_hours {
            deg *= 15.0;
        }
        Ok(deg)
    };
    let ra_deg = read_pos(
        Field::Ra,
        Some(Field::Rah),
        Field::Rad,
        Field::Ram,
        Field::Ras,
        false,
    )?;
    let dec_deg = read_pos(
        Field::Dec,
        Some(Field::Dech),
        Field::Decd,
        Field::Decm,
        Field::Decs,
        true,
    )?;
    // DP3 accepts RAs within (-6.3, 6.3) radians.
    let ra_ok = ra_deg.abs() < 6.3_f64.to_degrees();
    let dec_ok = dec_deg.to_radians().abs() <= FRAC_PI_2 + 1e-12;
    if !ra_ok || !dec_ok {
        return Err(ReadSourceListDp3Error::PositionOutOfRange {
            line_num,
            ra: ra_deg,
            dec: dec_deg,
        });
    }
    let mut ra_deg = ra_deg.rem_euclid(360.0);
    if ra_deg >= 360.0 {
        ra_deg = 0.0;
    }
    let dec_deg = dec_deg.clamp(-90.0, 90.0);

    // Flux densities.
    let i = parse_real(get(Field::I), 1.0, Field::I, line_num)?;
    let q = parse_real(get(Field::Q), 0.0, Field::Q, line_num)?;
    let u = parse_real(get(Field::U), 0.0, Field::U, line_num)?;
    let v = parse_real(get(Field::V), 0.0, Field::V, line_num)?;
    let ref_freq = parse_real(
        get(Field::ReferenceFrequency),
        0.0,
        Field::ReferenceFrequency,
        line_num,
    )?;
    let log_si = parse_bool(
        get(Field::LogarithmicSI),
        true,
        Field::LogarithmicSI,
        line_num,
    )?;

    // Spectral terms: a single value, or a bracketed list.
    let mut terms: Vec<f64> = vec![];
    let si_str = get(Field::SpectralIndex);
    if !si_str.is_empty() {
        let sb = si_str.as_bytes();
        let last = sb.len() - 1;
        if sb.len() < 2 || sb[0] != b'[' || sb[last] != b']' {
            terms.push(parse_real(si_str, 0.0, Field::SpectralIndex, line_num)?);
        } else {
            let mut st = ltrim(sb, 1, last);
            let end = rtrim(sb, st, last);
            while st < end {
                let pos = next_value(sb, b',', st, end).map_err(malformed)?;
                terms.push(parse_real(
                    &si_str[st..rtrim(sb, st, pos)],
                    0.0,
                    Field::SpectralIndex,
                    line_num,
                )?);
                st = ltrim(sb, pos + 1, end);
            }
        }
    }
    if !terms.is_empty() && ref_freq <= 0.0 {
        return Err(ReadSourceListDp3Error::MissingReferenceFrequency { line_num });
    }
    if !terms.is_empty() && (q != 0.0 || u != 0.0 || v != 0.0) {
        warnings.pol_with_spectral_terms += 1;
    }

    let fd = FluxDensity {
        freq: if ref_freq > 0.0 {
            ref_freq
        } else {
            FLAT_SPECTRUM_FREQ_HZ
        },
        i,
        q,
        u,
        v,
    };
    // Trailing zero terms don't change anything, and dropping them may let us
    // represent the spectrum exactly.
    while terms.last() == Some(&0.0) {
        terms.pop();
    }
    let flux_type = match (log_si, terms.as_slice()) {
        (_, []) => FluxDensityType::PowerLaw { si: 0.0, fd },
        (true, [si]) => FluxDensityType::PowerLaw { si: *si, fd },
        (true, [si, c1]) => FluxDensityType::CurvedPowerLaw {
            si: *si,
            fd,
            q: c1 / LN_10,
        },
        (true, _) => {
            warnings.many_log_terms += 1;
            eval_list(&fd, |freq| {
                let base = (freq / fd.freq).log10();
                let exponent = terms.iter().rev().fold(0.0, |acc, c| acc * base + c);
                fd.i * 10_f64.powf(base * exponent)
            })
        }
        (false, _) => {
            warnings.linear_si += 1;
            eval_list(&fd, |freq| {
                let x = freq / fd.freq - 1.0;
                let val = terms.iter().rev().fold(0.0, |acc, c| acc * x + c);
                fd.i + val * x
            })
        }
    };

    let comp_type = if is_gaussian {
        let maj = parse_real(get(Field::MajorAxis), 1.0, Field::MajorAxis, line_num)?;
        let min = parse_real(get(Field::MinorAxis), 1.0, Field::MinorAxis, line_num)?;
        let pa = parse_real(get(Field::Orientation), 1.0, Field::Orientation, line_num)?;
        if parse_bool(
            get(Field::OrientationIsAbsolute),
            false,
            Field::OrientationIsAbsolute,
            line_num,
        )? {
            warnings.orientation_is_absolute += 1;
        }
        ComponentType::Gaussian {
            maj: (maj / 3600.0).to_radians(),
            min: (min / 3600.0).to_radians(),
            pa: pa.to_radians(),
        }
    } else {
        ComponentType::Point
    };

    let comp = SourceComponent {
        radec: RADec::from_degrees(ra_deg, dec_deg),
        comp_type,
        flux_type,
    };

    if patch.is_empty() {
        sources.entry(name.to_string()).or_default().push(comp);
    } else {
        match sources.get_mut(patch) {
            Some(comps) => comps.push(comp),
            None => {
                return Err(ReadSourceListDp3Error::UndefinedPatch {
                    line_num,
                    patch: patch.to_string(),
                })
            }
        }
    }

    Ok(())
}

/// Evaluate a Stokes I spectral model at [`LIST_FREQS_HZ`]; Q, U and V are
/// kept constant (as DP3 does).
fn eval_list<F: Fn(f64) -> f64>(fd: &FluxDensity, stokes_i: F) -> FluxDensityType {
    let fds: Vec<FluxDensity> = LIST_FREQS_HZ
        .iter()
        .map(|&freq| FluxDensity {
            freq,
            i: stokes_i(freq),
            q: fd.q,
            u: fd.u,
            v: fd.v,
        })
        .collect();
    // There are always NUM_LIST_FREQS (> 0) entries.
    FluxDensityType::List(Vec1::try_from_vec(fds).unwrap())
}

#[cfg(test)]
mod tests {
    use std::io::{Cursor, Write};

    use approx::assert_abs_diff_eq;

    use super::*;
    use crate::srclist::{read::read_source_list_file, SourceListType};

    fn parse(s: &str) -> Result<SourceList, ReadSourceListError> {
        parse_source_list(&mut Cursor::new(s))
    }

    fn dp3_err(s: &str) -> ReadSourceListDp3Error {
        match parse(s) {
            Err(ReadSourceListError::Dp3(e)) => e,
            other => panic!("expected a DP3 error, got {other:?}"),
        }
    }

    /// DP3's evaluation of Stokes I (base/PointSource.cc).
    fn dp3_stokes_i(i: f64, ref_freq: f64, terms: &[f64], log_si: bool, freq: f64) -> f64 {
        if log_si {
            let base = (freq / ref_freq).log10();
            let exponent = terms.iter().rev().fold(0.0, |acc, c| acc * base + c);
            i * 10_f64.powf(base * exponent)
        } else {
            let x = freq / ref_freq - 1.0;
            let val = terms.iter().rev().fold(0.0, |acc, c| acc * x + c);
            i + val * x
        }
    }

    #[test]
    fn angles() {
        let cases = [
            // Sexagesimal hours.
            ("03:21:29.744", (3.0 + 21.0 / 60.0 + 29.744 / 3600.0) * 15.0),
            ("-01:30:00", -22.5),
            ("12:30", 187.5),
            (
                "03h21m29.744s",
                (3.0 + 21.0 / 60.0 + 29.744 / 3600.0) * 15.0,
            ),
            ("03H21M29.744", (3.0 + 21.0 / 60.0 + 29.744 / 3600.0) * 15.0),
            // Sexagesimal degrees.
            ("-37.08.37.076", -(37.0 + 8.0 / 60.0 + 37.076 / 3600.0)),
            ("+12.00.00", 12.0),
            ("-00.30.00", -0.5),
            ("5..2.59", 5.0 + 2.59 / 3600.0),
            ("-37d08m37.076s", -(37.0 + 8.0 / 60.0 + 37.076 / 3600.0)),
            ("37D08M37.076S", 37.0 + 8.0 / 60.0 + 37.076 / 3600.0),
            ("10d", 10.0),
            // Numbers with units; without a unit, radians (as casacore).
            ("50.3739deg", 50.3739),
            ("-37.1436 deg", -37.1436),
            ("12deg", 12.0),
            ("1.5rad", 1.5_f64.to_degrees()),
            ("0.5", 0.5_f64.to_degrees()),
            ("30arcmin", 0.5),
            ("1800arcsec", 0.5),
            ("2h", 30.0),
        ];
        for (s, expected) in cases {
            let deg = read_angle_deg(s).unwrap_or_else(|| panic!("failed to read '{s}'"));
            assert_abs_diff_eq!(deg, expected, epsilon = 1e-12);
        }
        for s in ["", "abc", "12.5x", "03:21:29.7xyz", "12.5.6.7.8", "3:21.5"] {
            assert!(read_angle_deg(s).is_none(), "'{s}' should not be readable");
        }
    }

    #[test]
    fn format_lines() {
        assert_eq!(
            format_from_format_line("  FORMAT = Name, Type"),
            Some(" Name, Type")
        );
        assert_eq!(format_from_format_line("format=Name"), Some("Name"));
        assert_eq!(format_from_format_line("formation, POINT"), None);
        assert_eq!(
            format_from_comment_line("# (Name, Type, Ra, Dec, I) = format"),
            Some("Name, Type, Ra, Dec, I")
        );
        assert_eq!(
            format_from_comment_line("#(Name,Type)=FORMAT"),
            Some("Name,Type")
        );
        assert_eq!(format_from_comment_line("# a comment"), None);

        // Multi-byte characters near the "format" keyword must not panic
        // (these functions run on arbitrary files during auto-detection).
        assert_eq!(format_from_format_line("abcdeéfgh"), None);
        assert_eq!(format_from_format_line("éformat"), None);
        assert_eq!(format_from_comment_line("# (Name) = fé"), None);
        assert_eq!(format_from_comment_line("# (Name) = abcdeéf"), None);
        assert_eq!(format_from_comment_line("# (Name) = é"), None);
        assert!(parse_format("Name, Type=éabcde, Ra, Dec").is_err());

        let f = parse_format(
            "Name, Type, Ra, Dec, I, ReferenceFrequency='150e6', SpectralIndex='[]', LogarithmicSI='true'",
        )
        .unwrap();
        assert_eq!(f.specs.len(), 8);
        assert_eq!(f.index[Field::Ra as usize], Some(2));
        assert_eq!(f.index[Field::Patch as usize], None);
        assert_eq!(f.specs[5].default.as_deref(), Some("150e6"));
        assert_eq!(f.specs[6].default.as_deref(), Some("[]"));
        assert_eq!(f.specs[7].default.as_deref(), Some("true"));
        assert_eq!(f.specs[0].sep, b',');

        // Field names are case insensitive; unknown fields are ignored.
        let f = parse_format("NAME, type, rA, DEC, Something, i").unwrap();
        assert_eq!(f.index[Field::Name as usize], Some(0));
        assert_eq!(f.index[Field::Type as usize], Some(1));
        assert_eq!(f.index[Field::I as usize], Some(5));

        // Fixed values.
        let f = parse_format("Name, Type=fixed'GAUSSIAN', Ra, Dec").unwrap();
        assert_eq!(f.specs[1].fixed.as_deref(), Some("GAUSSIAN"));

        // Position is required.
        assert!(parse_format("Name, Type, Ra, I").is_err());
        assert!(parse_format("Name, Type, Ra, Dec, rah").is_err());
    }

    #[test]
    fn wsclean_style() {
        // As written by WSClean's -save-source-list.
        let sl = parse(indoc::indoc! {"
            Format = Name, Type, Ra, Dec, I, SpectralIndex, LogarithmicSI, ReferenceFrequency='150000000.0', MajorAxis, MinorAxis, Orientation
            s0c0,POINT,08:28:05.152,39.35.08.511,0.3,[-0.7],true,,,,
            s0c1,GAUSSIAN,08:28:05.152,-39.35.08.511,0.4,[-0.8,0.1],true,125584411.621094,60.0,30.0,45.0
            s0c2,POINT,-01:30:00.0,+00.00.00,0.5,[0.2,-0.1],false,125584411.621094,,,
        "})
        .unwrap();
        assert_eq!(sl.len(), 3);
        assert_eq!(sl.keys().collect::<Vec<_>>(), ["s0c0", "s0c1", "s0c2"]);

        let c = &sl["s0c0"].components[0];
        assert!(c.is_point());
        assert_abs_diff_eq!(
            c.radec.ra.to_degrees(),
            (8.0 + 28.0 / 60.0 + 5.152 / 3600.0) * 15.0,
            epsilon = 1e-10
        );
        assert_abs_diff_eq!(
            c.radec.dec.to_degrees(),
            39.0 + 35.0 / 60.0 + 8.511 / 3600.0,
            epsilon = 1e-10
        );
        match &c.flux_type {
            FluxDensityType::PowerLaw { si, fd } => {
                assert_abs_diff_eq!(*si, -0.7);
                // The default reference frequency is used.
                assert_abs_diff_eq!(fd.freq, 150e6);
                assert_abs_diff_eq!(fd.i, 0.3);
            }
            ft => panic!("unexpected {ft:?}"),
        }

        let c = &sl["s0c1"].components[0];
        assert_abs_diff_eq!(
            c.radec.dec.to_degrees(),
            -(39.0 + 35.0 / 60.0 + 8.511 / 3600.0),
            epsilon = 1e-10
        );
        match c.comp_type {
            ComponentType::Gaussian { maj, min, pa } => {
                assert_abs_diff_eq!(maj.to_degrees() * 3600.0, 60.0, epsilon = 1e-10);
                assert_abs_diff_eq!(min.to_degrees() * 3600.0, 30.0, epsilon = 1e-10);
                assert_abs_diff_eq!(pa.to_degrees(), 45.0, epsilon = 1e-10);
            }
            ref ct => panic!("unexpected {ct:?}"),
        }
        match &c.flux_type {
            FluxDensityType::CurvedPowerLaw { si, fd, q } => {
                assert_abs_diff_eq!(*si, -0.8);
                assert_abs_diff_eq!(*q, 0.1 / LN_10);
                assert_abs_diff_eq!(fd.freq, 125584411.621094);
            }
            ft => panic!("unexpected {ft:?}"),
        }
        // The curved power law agrees with DP3's formula.
        for freq in [70e6, 125e6, 200e6, 300e6] {
            assert_abs_diff_eq!(
                c.estimate_at_freq(freq).i,
                dp3_stokes_i(0.4, 125584411.621094, &[-0.8, 0.1], true, freq),
                epsilon = 1e-12
            );
        }

        // Negative RA in hours is wrapped.
        let c = &sl["s0c2"].components[0];
        assert_abs_diff_eq!(c.radec.ra.to_degrees(), 337.5, epsilon = 1e-10);
        assert_abs_diff_eq!(c.radec.dec, 0.0);
        // LogarithmicSI=false: an ordinary polynomial, evaluated into a list.
        match &c.flux_type {
            FluxDensityType::List(fds) => {
                assert_eq!(fds.len(), NUM_LIST_FREQS);
                assert_abs_diff_eq!(fds.first().freq, LIST_FREQ_MIN_HZ, epsilon = 1e-3);
                assert_abs_diff_eq!(fds.last().freq, LIST_FREQ_MAX_HZ, epsilon = 1e-3);
                for fd in fds {
                    assert_abs_diff_eq!(
                        fd.i,
                        dp3_stokes_i(0.5, 125584411.621094, &[0.2, -0.1], false, fd.freq),
                        epsilon = 1e-12
                    );
                }
            }
            ft => panic!("unexpected {ft:?}"),
        }
    }

    #[test]
    fn patches_comments_and_stokes() {
        let sl = parse(indoc::indoc! {"
            # A comment before the format line.

            FORMAT = Name, Type, Patch, Ra, Dec, I, Q, U, V, ReferenceFrequency, SpectralIndex, LogarithmicSI, MajorAxis, MinorAxis, Orientation

            # Patch definitions have an empty name and type.
             , , FornaxA, 03:21:29.7, -37.08.37.1
            FornaxA_0, GAUSSIAN, FornaxA, 03h21m29.744s, -37d08m37.076s, 198.6, 1.0, -2.0, 0.5, 1.8879e+08, [-0.8], true, 822.5, 411.2, 10 # a trailing comment
            FornaxA_1, POINT, FornaxA, 50.3739deg, -37.1436deg, 97.5, 0, 0, 0, 1.8879e+08, [-0.8], true, , ,
            lonely, POINT, , 00:00:01, +00.00.01, 2.0, , , , 1.5e8, [-0.5, 0, 0], true, , ,
             , , EmptyPatch, 00:00:00, +00.00.00

             , , 'Quoted # patch', 01:00:00, +01.00.00
            q0, POINT, 'Quoted # patch', 01:00:00, +01.00.00, 3.0, , , , , , , , ,
        "})
        .unwrap();
        // The empty patch is dropped.
        assert_eq!(
            sl.keys().collect::<Vec<_>>(),
            ["FornaxA", "lonely", "Quoted # patch"]
        );

        let fnx = &sl["FornaxA"];
        assert_eq!(fnx.components.len(), 2);
        let c = &fnx.components[0];
        assert!(c.is_gaussian());
        assert_abs_diff_eq!(
            c.radec.ra.to_degrees(),
            (3.0 + 21.0 / 60.0 + 29.744 / 3600.0) * 15.0,
            epsilon = 1e-10
        );
        assert_abs_diff_eq!(
            c.radec.dec.to_degrees(),
            -(37.0 + 8.0 / 60.0 + 37.076 / 3600.0),
            epsilon = 1e-10
        );
        match &c.flux_type {
            FluxDensityType::PowerLaw { si, fd } => {
                assert_abs_diff_eq!(*si, -0.8);
                assert_eq!(
                    *fd,
                    FluxDensity {
                        freq: 1.8879e8,
                        i: 198.6,
                        q: 1.0,
                        u: -2.0,
                        v: 0.5
                    }
                );
            }
            ft => panic!("unexpected {ft:?}"),
        }
        match c.comp_type {
            ComponentType::Gaussian { pa, .. } => {
                assert_abs_diff_eq!(pa.to_degrees(), 10.0, epsilon = 1e-10)
            }
            ref ct => panic!("unexpected {ct:?}"),
        }
        let c = &fnx.components[1];
        assert!(c.is_point());
        assert_abs_diff_eq!(c.radec.ra.to_degrees(), 50.3739, epsilon = 1e-10);
        assert_abs_diff_eq!(c.radec.dec.to_degrees(), -37.1436, epsilon = 1e-10);

        // Trailing zero spectral terms are dropped, so this is a power law.
        let c = &sl["lonely"].components[0];
        assert_abs_diff_eq!(c.radec.ra.to_degrees(), 15.0 / 3600.0, epsilon = 1e-12);
        assert_abs_diff_eq!(c.radec.dec.to_degrees(), 1.0 / 3600.0, epsilon = 1e-12);
        assert!(matches!(
            c.flux_type,
            FluxDensityType::PowerLaw { si, fd: FluxDensity { i, q: 0.0, u: 0.0, v: 0.0, .. } }
                if si == -0.5 && i == 2.0
        ));

        // No spectral terms: a flat spectrum, as in DP3.
        let c = &sl["Quoted # patch"].components[0];
        match &c.flux_type {
            FluxDensityType::PowerLaw { si, fd } => {
                assert_abs_diff_eq!(*si, 0.0);
                assert_abs_diff_eq!(fd.i, 3.0);
            }
            ft => panic!("unexpected {ft:?}"),
        }
        assert_abs_diff_eq!(c.estimate_at_freq(80e6).i, 3.0);
    }

    #[test]
    fn old_style_format_and_many_log_terms() {
        let terms = [-0.7, 0.1, -0.05];
        let sl = parse(indoc::indoc! {"
            # (Name, Type, Ra, Dec, I, ReferenceFrequency, SpectralIndex) = format
            # Another comment
            src, POINT, 12:00:00, -45.00.00, 10.0, 1.4e8, [-0.7, 0.1, -0.05]
        "})
        .unwrap();
        let c = &sl["src"].components[0];
        assert_abs_diff_eq!(c.radec.ra.to_degrees(), 180.0, epsilon = 1e-10);
        assert_abs_diff_eq!(c.radec.dec.to_degrees(), -45.0, epsilon = 1e-10);
        match &c.flux_type {
            FluxDensityType::List(fds) => {
                assert_eq!(fds.len(), NUM_LIST_FREQS);
                for fd in fds {
                    assert_abs_diff_eq!(
                        fd.i,
                        dp3_stokes_i(10.0, 1.4e8, &terms, true, fd.freq),
                        epsilon = 1e-12
                    );
                }
            }
            ft => panic!("unexpected {ft:?}"),
        }
    }

    #[test]
    fn separate_position_fields() {
        let sl = parse(indoc::indoc! {"
            format = Name Type rah ram ras decd decm decs I
            src POINT 1 30 0 -10 30 0 1.0
        "})
        .unwrap();
        let c = &sl["src"].components[0];
        assert_abs_diff_eq!(c.radec.ra.to_degrees(), 22.5, epsilon = 1e-10);
        assert_abs_diff_eq!(c.radec.dec.to_degrees(), -10.5, epsilon = 1e-10);
    }

    #[test]
    fn errors() {
        // No data lines at all.
        assert_eq!(
            dp3_err("# only comments\n\n"),
            ReadSourceListDp3Error::NoComponents
        );
        assert_eq!(dp3_err(""), ReadSourceListDp3Error::NoComponents);

        let header = "Format = Name, Type, Patch, Ra, Dec, I, ReferenceFrequency, SpectralIndex\n";

        // Unsupported component types.
        for t in ["SHAPELET", "disk"] {
            let e = dp3_err(&format!("{header}src, {t}, , 00:00:00, +00.00.00, 1.0\n"));
            assert_eq!(
                e,
                ReadSourceListDp3Error::UnsupportedComponentType {
                    line_num: 2,
                    comp_type: t.to_string()
                }
            );
            assert!(e.to_string().contains(t));
        }
        assert!(matches!(
            dp3_err(&format!("{header}src, BLOB, , 00:00:00, +00.00.00, 1.0\n")),
            ReadSourceListDp3Error::InvalidComponentType { line_num: 2, .. }
        ));

        // A patch must be defined before use.
        assert!(matches!(
            dp3_err(&format!(
                "{header}src, POINT, P1, 00:00:00, +00.00.00, 1.0\n"
            )),
            ReadSourceListDp3Error::UndefinedPatch { line_num: 2, .. }
        ));

        // Spectral terms need a reference frequency.
        assert_eq!(
            dp3_err(&format!(
                "{header}src, POINT, , 00:00:00, +00.00.00, 1.0, , [-0.8]\n"
            )),
            ReadSourceListDp3Error::MissingReferenceFrequency { line_num: 2 }
        );

        // Colons are not allowed in declinations.
        assert!(matches!(
            dp3_err(&format!("{header}src, POINT, , 00:00:00, -10:00:00, 1.0\n")),
            ReadSourceListDp3Error::ColonInDec { line_num: 2, .. }
        ));

        // Bad numbers and positions.
        assert!(matches!(
            dp3_err(&format!("{header}src, POINT, , 00:00:00, +00.00.00, one\n")),
            ReadSourceListDp3Error::ParseFloat { line_num: 2, .. }
        ));
        assert!(matches!(
            dp3_err(&format!("{header}src, POINT, , nowhere, +00.00.00, 1.0\n")),
            ReadSourceListDp3Error::InvalidAngle { line_num: 2, .. }
        ));
        assert!(matches!(
            dp3_err(&format!("{header}src, POINT, , , +00.00.00, 1.0\n")),
            ReadSourceListDp3Error::MissingPosition { line_num: 2, .. }
        ));
        assert!(matches!(
            dp3_err(&format!("{header}src, POINT, , 00:00:00, +95.00.00, 1.0\n")),
            ReadSourceListDp3Error::PositionOutOfRange { line_num: 2, .. }
        ));

        // Unbalanced brackets.
        assert!(matches!(
            dp3_err(&format!(
                "{header}src, POINT, , 00:00:00, +00.00.00, 1.0, 1e8, [-0.8\n"
            )),
            ReadSourceListDp3Error::Malformed { line_num: 2, .. }
        ));

        // Rotation measures are not supported.
        assert_eq!(
            dp3_err("Format = Name, Type, Ra, Dec, I, Q, U, RotationMeasure\nsrc, POINT, 00:00:00, +00.00.00, 1.0, 0.1, 0.1, 5\n"),
            ReadSourceListDp3Error::UnsupportedRotationMeasure { line_num: 2 }
        );

        // Nothing in the file.
        assert_eq!(dp3_err(header), ReadSourceListDp3Error::NoComponents);
    }

    /// DP3's `read_format` test (sky_model/test/unit/tReadSkyModel.cc): a
    /// file without a format line is read with the default format.
    #[test]
    fn default_format() {
        let f = parse_format(DEFAULT_FORMAT).unwrap();
        assert_eq!(f.specs.len(), 11);
        for (i, field) in [
            Field::Name,
            Field::Type,
            Field::Ra,
            Field::Dec,
            Field::I,
            Field::Q,
            Field::U,
            Field::V,
            Field::MajorAxis,
            Field::MinorAxis,
            Field::Orientation,
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(f.index[field as usize], Some(i));
        }

        let sl = parse("center, POINT, 16:38:28.205000, +63.44.34.314000, 1, , , , , \n").unwrap();
        assert_eq!(sl.len(), 1);
        let (name, src) = sl.first().unwrap();
        assert_eq!(name, "center");
        assert_eq!(src.components.len(), 1);
        let comp = &src.components[0];
        assert_abs_diff_eq!(
            comp.radec.ra,
            (16.0 + 38.0 / 60.0 + 28.205 / 3600.0) * 15.0_f64.to_radians(),
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            comp.radec.dec,
            (63.0 + 44.0 / 60.0 + 34.314 / 3600.0_f64).to_radians(),
            epsilon = 1e-12
        );
        assert!(matches!(comp.comp_type, ComponentType::Point));
        match &comp.flux_type {
            FluxDensityType::PowerLaw { si, fd } => {
                assert_abs_diff_eq!(*si, 0.0);
                assert_abs_diff_eq!(fd.i, 1.0);
                assert_abs_diff_eq!(fd.q, 0.0);
                assert_abs_diff_eq!(fd.u, 0.0);
                assert_abs_diff_eq!(fd.v, 0.0);
            }
            other => panic!("expected a power law, got {other:?}"),
        }

        // The same with a Gaussian, exercising the trailing default fields,
        // and with a leading comment (which is not a format comment).
        let sl = parse(
            "# no format here\ng, GAUSSIAN, 16:38:28.205, +63.44.34.314, 2, 0.1, , , 60, 30, 45\n",
        )
        .unwrap();
        let comp = &sl["g"].components[0];
        match &comp.comp_type {
            ComponentType::Gaussian { maj, min, pa } => {
                assert_abs_diff_eq!(*maj, (60.0 / 3600.0_f64).to_radians(), epsilon = 1e-12);
                assert_abs_diff_eq!(*min, (30.0 / 3600.0_f64).to_radians(), epsilon = 1e-12);
                assert_abs_diff_eq!(*pa, 45.0_f64.to_radians(), epsilon = 1e-12);
            }
            other => panic!("expected a Gaussian, got {other:?}"),
        }

        // The fallback only applies to the first data line; a later format
        // line is ignored, as in DP3.
        let sl = parse(
            "a, POINT, 00:00:00, +00.00.00, 1\nFormat = Name, Ra, Dec, I\nb, POINT, 00:00:00, +00.00.00, 1\n",
        )
        .unwrap();
        assert_eq!(sl.len(), 2);

        // From a file, the fallback applies when the type is given explicitly,
        // but such a file is not auto-detected as a DP3 sky model.
        let mut f = tempfile::Builder::new()
            .suffix(".skymodel")
            .tempfile()
            .unwrap();
        writeln!(
            f,
            "center, POINT, 16:38:28.205000, +63.44.34.314000, 1, , , , , "
        )
        .unwrap();
        f.flush().unwrap();
        let (sl, sl_type) = read_source_list_file(f.path(), Some(SourceListType::Dp3)).unwrap();
        assert_eq!(sl_type, SourceListType::Dp3);
        assert_eq!(sl.len(), 1);
        assert!(!looks_like_dp3(f.path()));
        if let Ok((_, sl_type)) = read_source_list_file(f.path(), None) {
            assert_ne!(sl_type, SourceListType::Dp3);
        }
    }

    #[test]
    fn source_list_type_names() {
        use std::str::FromStr;
        for s in ["dp3", "skymodel", "bbs"] {
            assert_eq!(SourceListType::from_str(s).unwrap(), SourceListType::Dp3);
        }
        assert_eq!(SourceListType::Dp3.to_string(), "dp3");
    }

    #[test]
    fn auto_detect() {
        let mut f = tempfile::Builder::new().suffix(".txt").tempfile().unwrap();
        writeln!(
            f,
            "# A wsclean-style model\nFormat = Name, Type, Ra, Dec, I\nsrc, POINT, 00:00:00, +00.00.00, 1.0"
        )
        .unwrap();
        f.flush().unwrap();
        assert!(looks_like_dp3(f.path()));
        let (sl, sl_type) = read_source_list_file(f.path(), None).unwrap();
        assert_eq!(sl_type, SourceListType::Dp3);
        assert_eq!(sl.len(), 1);

        // An RTS-style source list is not mistaken for a DP3 one.
        let mut f = tempfile::Builder::new().suffix(".txt").tempfile().unwrap();
        writeln!(f, "SOURCE src 0.0 0.0\nFREQ 180e+6 1.0 0 0 0\nENDSOURCE").unwrap();
        f.flush().unwrap();
        assert!(!looks_like_dp3(f.path()));
        let (_, sl_type) = read_source_list_file(f.path(), None).unwrap();
        assert_eq!(sl_type, SourceListType::Rts);

        // A UTF-8 file whose first line has a multi-byte character within
        // the first six bytes must not panic auto-detection.
        let mut f = tempfile::Builder::new().suffix(".txt").tempfile().unwrap();
        writeln!(f, "abcdeéfgh\n# (Name) = fé").unwrap();
        f.flush().unwrap();
        assert!(!looks_like_dp3(f.path()));
    }
}
