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
    assert_eq!(pair.1, TimeFormat::NoUnit);

    let result = parse_time("1.0");
    assert!(result.is_ok(), "{:?}", result.unwrap_err());
    let pair = result.unwrap();
    assert_abs_diff_eq!(pair.0, 1.0);
    assert_eq!(pair.1, TimeFormat::NoUnit);

    let result = parse_time(" 1.0 ");
    assert!(result.is_ok(), "{:?}", result.unwrap_err());
    let pair = result.unwrap();
    assert_abs_diff_eq!(pair.0, 1.0);
    assert_eq!(pair.1, TimeFormat::NoUnit);
}

#[test]
fn test_parse_time_str_with_units() {
    // Iterate over all possible units.
    for time_format in TimeFormat::iter().filter(|&tf| tf != TimeFormat::NoUnit) {
        let time_format_str: &'static str = time_format.into();
        for time_format_str in [
            time_format_str.to_lowercase(),
            time_format_str.to_uppercase(),
        ] {
            let result = parse_time(&format!("1{}", time_format_str));
            assert!(result.is_ok(), "{:?}", result.unwrap_err());
            let pair = result.unwrap();
            assert_abs_diff_eq!(pair.0, 1.0);
            assert_eq!(pair.1, time_format);

            let result = parse_time(&format!("1.0{}", time_format_str));
            assert!(result.is_ok(), "{:?}", result.unwrap_err());
            let pair = result.unwrap();
            assert_abs_diff_eq!(pair.0, 1.0);
            assert_eq!(pair.1, time_format);

            let result = parse_time(&format!(" 1.0{} ", time_format_str));
            assert!(result.is_ok(), "{:?}", result.unwrap_err());
            let pair = result.unwrap();
            assert_abs_diff_eq!(pair.0, 1.0);
            assert_eq!(pair.1, time_format);

            let result = parse_time(&format!(" 1.0 {} ", time_format_str));
            assert!(result.is_ok(), "{:?}", result.unwrap_err());
            let pair = result.unwrap();
            assert_abs_diff_eq!(pair.0, 1.0);
            assert_eq!(pair.1, time_format);
        }
    }
}

#[test]
fn test_parse_freq_str_without_units() {
    let result = parse_freq("20");
    assert!(result.is_ok(), "{:?}", result.unwrap_err());
    let pair = result.unwrap();
    assert_abs_diff_eq!(pair.0, 20.0);
    assert_eq!(pair.1, FreqFormat::NoUnit);

    let result = parse_freq("40.0");
    assert!(result.is_ok(), "{:?}", result.unwrap_err());
    let pair = result.unwrap();
    assert_abs_diff_eq!(pair.0, 40.0);
    assert_eq!(pair.1, FreqFormat::NoUnit);

    let result = parse_freq(" 40.0 ");
    assert!(result.is_ok(), "{:?}", result.unwrap_err());
    let pair = result.unwrap();
    assert_abs_diff_eq!(pair.0, 40.0);
    assert_eq!(pair.1, FreqFormat::NoUnit);
}

#[test]
fn test_parse_freq_str_with_units() {
    // Iterate over all possible units.
    for freq_format in FreqFormat::iter().filter(|&tf| tf != FreqFormat::NoUnit) {
        let freq_format_str: &'static str = freq_format.into();
        for freq_format_str in [
            freq_format_str.to_lowercase(),
            freq_format_str.to_uppercase(),
        ] {
            let result = parse_freq(&format!("20{}", freq_format_str));
            assert!(result.is_ok(), "{:?}", result.unwrap_err());
            let pair = result.unwrap();
            assert_abs_diff_eq!(pair.0, 20.0);
            assert_eq!(pair.1, freq_format);

            let result = parse_freq(&format!("10.0{}", freq_format_str));
            assert!(result.is_ok(), "{:?}", result.unwrap_err());
            let pair = result.unwrap();
            assert_abs_diff_eq!(pair.0, 10.0);
            assert_eq!(pair.1, freq_format);

            let result = parse_freq(&format!(" 40.0{} ", freq_format_str));
            assert!(result.is_ok(), "{:?}", result.unwrap_err());
            let pair = result.unwrap();
            assert_abs_diff_eq!(pair.0, 40.0);
            assert_eq!(pair.1, freq_format);

            let result = parse_freq(&format!(" 40.0 {} ", freq_format_str));
            assert!(result.is_ok(), "{:?}", result.unwrap_err());
            let pair = result.unwrap();
            assert_abs_diff_eq!(pair.0, 40.0);
            assert_eq!(pair.1, freq_format);
        }
    }
}
