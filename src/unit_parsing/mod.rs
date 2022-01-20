// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to parse strings into plain numbers or some quantity with a unit.

mod error;
#[cfg(test)]
mod tests;

pub(crate) use error::*;

use itertools::Itertools;
use strum::IntoEnumIterator;
use strum_macros::{Display, EnumIter, EnumString, IntoStaticStr};

use mwa_hyperdrive_common::{itertools, lazy_static};

lazy_static::lazy_static! {
    pub static ref WAVELENGTH_FORMATS: String = InputWavelengthFormat::iter().join(", ");
}

#[derive(Debug, Clone, Copy, PartialEq, EnumIter, EnumString, IntoStaticStr)]
pub(crate) enum TimeFormat {
    /// Seconds
    S,

    /// Milliseconds
    Ms,

    NoUnit,
}

/// Parse a string that may have a unit of time attached to it.
pub(crate) fn parse_time(s: &str) -> Result<(f64, TimeFormat), UnitParseError> {
    // Try to parse a naked number.
    let maybe_number: Option<f64> = s.trim().parse().ok();
    if let Some(number) = maybe_number {
        return Ok((number, TimeFormat::NoUnit));
    };

    // That didn't work; let's search over our supported formats.
    for time_format in TimeFormat::iter().filter(|&tf| tf != TimeFormat::NoUnit) {
        let time_format_str: &'static str = time_format.into();
        let suffix = s
            .trim()
            .trim_start_matches(|c| char::is_numeric(c) || c == '.')
            .trim();
        if suffix.to_uppercase() == time_format_str.to_uppercase() {
            let prefix = s.trim().trim_end_matches(char::is_alphabetic).trim();
            let number: f64 = match prefix.parse() {
                Ok(n) => n,
                Err(_) => {
                    return Err(UnitParseError::GotUnitButCantParse {
                        input: s.to_string(),
                        unit_type: "time",
                        unit: time_format_str,
                    })
                }
            };
            return Ok((number, time_format));
        }
    }

    // If we made it this far, we don't know how to parse the string.
    Err(UnitParseError::Unknown {
        input: s.to_string(),
        unit_type: "time",
    })
}

#[derive(Debug, Clone, Copy, PartialEq, EnumIter, EnumString, IntoStaticStr)]
#[allow(non_camel_case_types)]
pub(crate) enum FreqFormat {
    /// Hertz
    Hz,

    /// kiloHertz
    kHz,

    NoUnit,
}

/// Parse a string that may have a unit of frequency attached to it.
pub(crate) fn parse_freq(s: &str) -> Result<(f64, FreqFormat), UnitParseError> {
    // Try to parse a naked number.
    let maybe_number: Option<f64> = s.trim().parse().ok();
    if let Some(number) = maybe_number {
        return Ok((number, FreqFormat::NoUnit));
    };

    // That didn't work; let's search over our supported formats.
    for freq_format in FreqFormat::iter().filter(|&tf| tf != FreqFormat::NoUnit) {
        let freq_format_str: &'static str = freq_format.into();
        let suffix = s
            .trim()
            .trim_start_matches(|c| char::is_numeric(c) || c == '.')
            .trim();
        if suffix.to_uppercase() == freq_format_str.to_uppercase() {
            let prefix = s.trim().trim_end_matches(char::is_alphabetic).trim();
            let number: f64 = match prefix.parse() {
                Ok(n) => n,
                Err(_) => {
                    return Err(UnitParseError::GotUnitButCantParse {
                        input: s.to_string(),
                        unit_type: "freq",
                        unit: freq_format_str,
                    })
                }
            };
            return Ok((number, freq_format));
        }
    }

    // If we made it this far, we don't know how to parse the string.
    Err(UnitParseError::Unknown {
        input: s.to_string(),
        unit_type: "frequency",
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(non_camel_case_types)]
pub(crate) enum WavelengthUnit {
    /// Lambda (λ)
    L,

    /// Metres
    M,
}

#[derive(Debug, Display, Clone, Copy, EnumIter, EnumString, IntoStaticStr)]
#[allow(non_camel_case_types)]
enum InputWavelengthFormat {
    /// Lambda (λ)
    #[strum(serialize = "λ")]
    λ,

    /// kiloLambda
    #[strum(serialize = "kλ")]
    kλ,

    /// Lambda (λ)
    #[strum(serialize = "l")]
    l,

    /// kiloLambda
    #[strum(serialize = "kl")]
    kl,

    /// Lambda (λ)
    #[strum(serialize = "lambda")]
    lambda,

    /// kiloLambda
    #[strum(serialize = "klambda")]
    klambda,

    /// Metres
    #[strum(serialize = "m")]
    m,

    /// kiloMetres
    #[strum(serialize = "km")]
    km,
}

impl InputWavelengthFormat {
    fn into_internal(self, quantity: f64) -> (f64, WavelengthUnit) {
        match self {
            InputWavelengthFormat::l | InputWavelengthFormat::lambda | InputWavelengthFormat::λ => {
                (quantity, WavelengthUnit::L)
            }

            InputWavelengthFormat::kl
            | InputWavelengthFormat::klambda
            | InputWavelengthFormat::kλ => (quantity * 1e3, WavelengthUnit::L),

            InputWavelengthFormat::m => (quantity, WavelengthUnit::M),

            InputWavelengthFormat::km => (quantity * 1e3, WavelengthUnit::M),
        }
    }
}

/// Parse a wavelength string. Must have a unit attached to it.
pub(crate) fn parse_wavelength(s: &str) -> Result<(f64, WavelengthUnit), UnitParseError> {
    // Try to parse a naked number. As we require a unit, error out if we the
    // parse succeeds.
    let maybe_number: Option<f64> = s.trim().parse().ok();
    if maybe_number.is_some() {
        return Err(UnitParseError::UnitRequired {
            input: s.to_string(),
            unit_type: "wavelength",
        });
    };

    // Search over our supported formats.
    for wavelength_format in InputWavelengthFormat::iter() {
        let wavelength_format_str: &'static str = wavelength_format.into();
        let suffix = s
            .trim()
            .trim_start_matches(|c| char::is_numeric(c) || c == '.')
            .trim();
        if suffix.to_uppercase() == wavelength_format_str.to_uppercase() {
            let prefix = s.trim().trim_end_matches(char::is_alphabetic).trim();
            let quantity: f64 = match prefix.parse() {
                Ok(n) => n,
                Err(_) => {
                    return Err(UnitParseError::GotUnitButCantParse {
                        input: s.to_string(),
                        unit_type: "wavelength",
                        unit: wavelength_format_str,
                    })
                }
            };

            // Convert the input format to an internal format.
            return Ok(wavelength_format.into_internal(quantity));
        }
    }

    // If we made it this far, we don't know how to parse the string.
    Err(UnitParseError::Unknown {
        input: s.to_string(),
        unit_type: "wavelength",
    })
}
