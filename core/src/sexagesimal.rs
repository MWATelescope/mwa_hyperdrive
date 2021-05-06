// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Code for handling conversion to and from sexagesimal.
 */

use thiserror::Error;

/// Convert a sexagesimal-formatted string delimited by colons to a float
/// \[degrees\]. The input is assumed to be in "degrees minutes seconds".
///
/// # Examples
///
/// ```
/// # use mwa_hyperdrive_core::sexagesimal::*;
/// # use approx::*;
/// # fn main() -> Result<(), SexagesimalError> {
/// let f = sexagesimal_colon_str_to_degrees("-22:58:52.56")?;
/// assert_abs_diff_eq!(f, -22.981267, epsilon = 1e-6);
/// # Ok(())
/// # }
/// ```
pub fn sexagesimal_colon_str_to_degrees(s: &str) -> Result<f64, SexagesimalError> {
    let mut split = Vec::with_capacity(3);
    for elem in s.split(':') {
        split.push(elem.parse()?);
    }
    if split.len() != 3 {
        return Err(SexagesimalError::WrongFieldCount(s.to_string()));
    }
    let h = split[0];
    let m = split[1];
    let s = split[2];
    Ok(sexagesimal_dms_to_degrees(h, m, s))
}

pub fn sexagesimal_dms_to_degrees(d: f64, m: f64, s: f64) -> f64 {
    let (negative, d_abs) = if d < 0.0 { (true, d.abs()) } else { (false, d) };
    let num = d_abs + m / 60.0 + s / 3600.0;
    if negative {
        -num
    } else {
        num
    }
}

/// Convert a sexagesimal-formatted string in "degrees minutes seconds" to a
/// float \[degrees\].
///
/// # Examples
///
/// ```
/// # use mwa_hyperdrive_core::sexagesimal::*;
/// # use approx::*;
/// # fn main() -> Result<(), SexagesimalError> {
/// let f = sexagesimal_dms_string_to_degrees("-11d49m01.062s")?;
/// assert_abs_diff_eq!(f, -11.81696167, epsilon = 1e-6);
/// # Ok(())
/// # }
/// ```
pub fn sexagesimal_dms_string_to_degrees(dms: &str) -> Result<f64, SexagesimalError> {
    let mut split = dms.split('d');
    let d = match split.next() {
        None => return Err(SexagesimalError::MissingD(dms.to_string())),
        Some(d) => d.parse()?,
    };

    let mut split = match split.next() {
        None => return Err(SexagesimalError::MissingM(dms.to_string())),
        Some(s) => s.split('m'),
    };
    let m = match split.next() {
        None => return Err(SexagesimalError::MissingM(dms.to_string())),
        Some(m) => m.parse()?,
    };

    let mut split = match split.next() {
        None => return Err(SexagesimalError::MissingS(dms.to_string())),
        Some(s) => s.split('s'),
    };
    let s = match split.next() {
        None => return Err(SexagesimalError::MissingS(dms.to_string())),
        Some(s) => s.parse()?,
    };

    Ok(sexagesimal_dms_to_degrees(d, m, s))
}

/// Convert a sexagesimal-formatted string in "hours minutes seconds" to a
/// float \[degrees\].
///
/// # Examples
///
/// ```
/// # use mwa_hyperdrive_core::sexagesimal::*;
/// # use approx::*;
/// # fn main() -> Result<(), SexagesimalError> {
/// let s = "-11h49m01.062s";
/// let f = sexagesimal_hms_string_to_degrees(s)?;
/// assert_abs_diff_eq!(f, -177.254425, epsilon = 1e-6);
/// # Ok(())
/// # }
/// ```
pub fn sexagesimal_hms_string_to_degrees(hms: &str) -> Result<f64, SexagesimalError> {
    let mut split = hms.split('h');
    let h = match split.next() {
        None => return Err(SexagesimalError::MissingH(hms.to_string())),
        Some(h) => h.parse()?,
    };

    let mut split = match split.next() {
        None => return Err(SexagesimalError::MissingM(hms.to_string())),
        Some(s) => s.split('m'),
    };
    let m = match split.next() {
        None => return Err(SexagesimalError::MissingM(hms.to_string())),
        Some(m) => m.parse()?,
    };

    let mut split = match split.next() {
        None => return Err(SexagesimalError::MissingS(hms.to_string())),
        Some(s) => s.split('s'),
    };
    let s = match split.next() {
        None => return Err(SexagesimalError::MissingS(hms.to_string())),
        Some(s) => s.parse()?,
    };

    Ok(sexagesimal_hms_to_float(h, m, s))
}

pub fn sexagesimal_hms_to_float(h: f64, m: f64, s: f64) -> f64 {
    sexagesimal_dms_to_degrees(15.0 * h, 15.0 * m, 15.0 * s)
}

/// Convert a number in degrees to a sexagesimal-formatted string in "degrees
/// minutes seconds".
///
/// # Examples
///
/// ```
/// # use mwa_hyperdrive_core::sexagesimal::*;
/// let dms = degrees_to_sexagesimal_dms(-165.0169619);
/// assert_eq!(dms, "-165d01m01.0628s");
/// ```
pub fn degrees_to_sexagesimal_dms(f: f64) -> String {
    let negative = f < 0.0;
    let f_abs = f.abs();
    let degrees = f_abs.floor();
    let minutes = (f_abs - degrees) * 60.0;
    let seconds = (minutes - minutes.floor()) * 60.0;

    format!(
        "{sign}{deg}d{min:02}m{sec:02}.{frac:04}s",
        sign = if negative { "-" } else { "" },
        deg = degrees as u8,
        min = minutes.floor() as u8,
        sec = seconds.floor() as u8,
        // The 4 in 1e4 gives that many decimal places.
        frac = ((seconds - seconds.floor()) * 1e4) as u32,
    )
}

/// Convert a number in degrees to a sexagesimal-formatted string in "hours
/// minutes seconds".
///
/// # Examples
///
/// ```
/// # use mwa_hyperdrive_core::sexagesimal::*;
/// let hms = degrees_to_sexagesimal_hms(-177.254425);
/// assert_eq!(hms, "-11h49m01.0619s");
/// ```
pub fn degrees_to_sexagesimal_hms(f: f64) -> String {
    let negative = f < 0.0;
    let f_abs = f.abs();
    let hours = (f_abs / 15.0).floor();
    let minutes = ((f_abs / 15.0 - hours) * 60.0).floor();
    let seconds = (((f_abs / 15.0 - hours) * 60.0) - minutes) * 60.0;

    format!(
        "{sign}{hrs}h{min:02}m{sec:02}.{frac:04}s",
        sign = if negative { "-" } else { "" },
        hrs = hours as u8,
        min = minutes.floor() as u8,
        sec = seconds.floor() as u8,
        // The 4 in 1e4 gives that many decimal places.
        frac = ((seconds - seconds.floor()) * 1e4) as u32,
    )
}

#[derive(Error, Debug)]
pub enum SexagesimalError {
    /// Three numbers (fields) are expected; this error is used when the number
    /// of fields is not three.
    #[error("Did not get three sexagesimal fields: {0}")]
    WrongFieldCount(String),

    #[error("Did not find 'h' when attempting to read sexagesigmal string: {0}")]
    MissingH(String),

    #[error("Did not find 'd' when attempting to read sexagesigmal string: {0}")]
    MissingD(String),

    #[error("Did not find 'm' when attempting to read sexagesigmal string: {0}")]
    MissingM(String),

    #[error("Did not find 's' when attempting to read sexagesigmal string: {0}")]
    MissingS(String),

    #[error("{0}")]
    ParseFloat(#[from] std::num::ParseFloatError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;

    #[test]
    fn test_sex2float_1() {
        let result = sexagesimal_colon_str_to_degrees("-22:58:52.56");
        assert!(result.is_ok());
        assert_abs_diff_eq!(result.unwrap(), -22.981266666666667, epsilon = 1e-10);
    }

    #[test]
    fn test_sex2float_2() {
        let result = sexagesimal_colon_str_to_degrees("12:30:45");
        assert!(result.is_ok());
        assert_abs_diff_eq!(result.unwrap(), 12.5125, epsilon = 1e-10);
    }

    #[test]
    fn test_sex_hms_1() {
        let result = sexagesimal_hms_string_to_degrees("11h34m23.7854s");
        assert!(result.is_ok(), "{}", result.unwrap_err());
        assert_abs_diff_eq!(result.unwrap(), 173.59910583333334);
    }
}
