// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use super::*;

use approx::assert_abs_diff_eq;

#[test]
fn test_parse_time_str_without_units() {
    let result = parse_time("1");
    assert!(result.is_ok(), "{:?}", result.unwrap_err());
    let pair = result.unwrap();
    assert_abs_diff_eq!(pair.0, 1.0);
    assert_eq!(pair.1, None);

    let result = parse_time("1.0");
    assert!(result.is_ok(), "{:?}", result.unwrap_err());
    let pair = result.unwrap();
    assert_abs_diff_eq!(pair.0, 1.0);
    assert_eq!(pair.1, None);

    let result = parse_time(" 1.0 ");
    assert!(result.is_ok(), "{:?}", result.unwrap_err());
    let pair = result.unwrap();
    assert_abs_diff_eq!(pair.0, 1.0);
    assert_eq!(pair.1, None);
}

#[test]
fn test_parse_time_str_with_units() {
    // Iterate over all possible units.
    for time_format in TimeFormat::iter() {
        let time_format_str: &'static str = time_format.into();
        for time_format_str in [
            time_format_str.to_lowercase(),
            time_format_str.to_uppercase(),
        ] {
            let result = parse_time(&format!("1{}", time_format_str));
            assert!(result.is_ok(), "{:?}", result.unwrap_err());
            let pair = result.unwrap();
            assert_abs_diff_eq!(pair.0, 1.0);
            assert_eq!(pair.1, Some(time_format));

            let result = parse_time(&format!("1.0{}", time_format_str));
            assert!(result.is_ok(), "{:?}", result.unwrap_err());
            let pair = result.unwrap();
            assert_abs_diff_eq!(pair.0, 1.0);
            assert_eq!(pair.1, Some(time_format));

            let result = parse_time(&format!(" 1.0{} ", time_format_str));
            assert!(result.is_ok(), "{:?}", result.unwrap_err());
            let pair = result.unwrap();
            assert_abs_diff_eq!(pair.0, 1.0);
            assert_eq!(pair.1, Some(time_format));

            let result = parse_time(&format!(" 1.0 {} ", time_format_str));
            assert!(result.is_ok(), "{:?}", result.unwrap_err());
            let pair = result.unwrap();
            assert_abs_diff_eq!(pair.0, 1.0);
            assert_eq!(pair.1, Some(time_format));
        }
    }
}

#[test]
fn test_parse_freq_str_without_units() {
    let result = parse_freq("20");
    assert!(result.is_ok(), "{:?}", result.unwrap_err());
    let pair = result.unwrap();
    assert_abs_diff_eq!(pair.0, 20.0);
    assert_eq!(pair.1, None);

    let result = parse_freq("40.0");
    assert!(result.is_ok(), "{:?}", result.unwrap_err());
    let pair = result.unwrap();
    assert_abs_diff_eq!(pair.0, 40.0);
    assert_eq!(pair.1, None);

    let result = parse_freq(" 40.0 ");
    assert!(result.is_ok(), "{:?}", result.unwrap_err());
    let pair = result.unwrap();
    assert_abs_diff_eq!(pair.0, 40.0);
    assert_eq!(pair.1, None);
}

#[test]
fn test_parse_freq_str_with_units() {
    // Iterate over all possible units.
    for freq_format in FreqFormat::iter() {
        let freq_format_str: &'static str = freq_format.into();
        for freq_format_str in [
            freq_format_str.to_lowercase(),
            freq_format_str.to_uppercase(),
        ] {
            let result = parse_freq(&format!("20{}", freq_format_str));
            assert!(result.is_ok(), "{:?}", result.unwrap_err());
            let pair = result.unwrap();
            assert_abs_diff_eq!(pair.0, 20.0);
            assert_eq!(pair.1, Some(freq_format));

            let result = parse_freq(&format!("10.0{}", freq_format_str));
            assert!(result.is_ok(), "{:?}", result.unwrap_err());
            let pair = result.unwrap();
            assert_abs_diff_eq!(pair.0, 10.0);
            assert_eq!(pair.1, Some(freq_format));

            let result = parse_freq(&format!(" 40.0{} ", freq_format_str));
            assert!(result.is_ok(), "{:?}", result.unwrap_err());
            let pair = result.unwrap();
            assert_abs_diff_eq!(pair.0, 40.0);
            assert_eq!(pair.1, Some(freq_format));

            let result = parse_freq(&format!(" 40.0 {} ", freq_format_str));
            assert!(result.is_ok(), "{:?}", result.unwrap_err());
            let pair = result.unwrap();
            assert_abs_diff_eq!(pair.0, 40.0);
            assert_eq!(pair.1, Some(freq_format));
        }
    }
}

#[test]
fn test_parse_wavelength_str() {
    // Iterate over all possible unit inputs.
    for wavelength_format in InputWavelengthFormat::iter() {
        let wavelength_format_str: &'static str = wavelength_format.into();
        for wavelength_format_str in [
            wavelength_format_str.to_lowercase(),
            wavelength_format_str.to_uppercase(),
        ] {
            // Iterate over a bunch of inputs.
            for (mut expected, input) in [
                (20.0, format!("20{}", wavelength_format_str)),
                (10.0, format!("10.0{}", wavelength_format_str)),
                (40.0, format!(" 40.0{} ", wavelength_format_str)),
                (40.0, format!(" 40.0 {} ", wavelength_format_str)),
            ] {
                let result = parse_wavelength(&input);
                assert!(result.is_ok(), "{:?}", result.unwrap_err());
                let pair = result.unwrap();

                if wavelength_format_str.starts_with('k') || wavelength_format_str.starts_with('K')
                {
                    expected *= 1e3;
                }
                assert_abs_diff_eq!(pair.0, expected);
                if wavelength_format_str.contains('l')
                    || wavelength_format_str.contains('L')
                    || wavelength_format_str.contains('λ')
                {
                    assert_eq!(pair.1, WavelengthUnit::L);
                } else if wavelength_format_str.contains('m') || wavelength_format_str.contains('M')
                {
                    assert_eq!(pair.1, WavelengthUnit::M);
                }
            }
        }
    }
}
