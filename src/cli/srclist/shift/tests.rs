// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Tests for shifting source lists.

use std::{fs::File, io::BufReader, path::Path};

use approx::assert_abs_diff_eq;
use tempfile::NamedTempFile;

use super::shift;
use crate::srclist::{
    hyperdrive::source_list_from_json, read::read_source_list_file, SourceList, SrclistError,
};

/// An RTS-style sky-model. `shift` keeps the order of the sources in an RTS
/// source list.
const RTS_SRCLIST_PATH: &str =
    "test_files/1090008640/srclist_pumav3_EoR0aegean_EoR1pietro+ForA_1090008640_peel100.txt";

/// A hyperdrive-style sky-model.
const HYP_SRCLIST_PATH: &str =
    "test_files/1090008640/srclist_pumav3_EoR0aegean_EoR1pietro+ForA_1090008640_100.yaml";

/// The metafits used to collapse source lists.
const METAFITS_PATH: &str = "test_files/1090008640/1090008640.metafits";

/// A source's shift [degrees].
#[derive(Clone, Copy)]
struct Shift {
    ra_deg: f64,
    dec_deg: f64,
}

/// Write a shifts .json file for `shift`.
fn write_shifts(shifts: &[(&str, Shift)]) -> NamedTempFile {
    let map: serde_json::Map<String, serde_json::Value> = shifts
        .iter()
        .map(|(name, s)| {
            (
                name.to_string(),
                serde_json::json!({ "ra": s.ra_deg, "dec": s.dec_deg }),
            )
        })
        .collect();
    let temp = tempfile::Builder::new().suffix(".json").tempfile().unwrap();
    serde_json::to_writer(File::create(temp.path()).unwrap(), &map).unwrap();
    temp
}

/// Read a hyperdrive .json source list.
fn read_json(path: &Path) -> SourceList {
    let mut f = BufReader::new(File::open(path).unwrap());
    source_list_from_json(&mut f).unwrap()
}

/// Get the names of the first `n` sources, in file order.
fn first_names(sl: &SourceList, n: usize) -> Vec<String> {
    sl.keys().take(n).cloned().collect()
}

/// Assert that `got` has the (RA, Dec) of `expected` [radians], component by
/// component and in order.
fn assert_radecs_eq(got: &[(f64, f64)], expected: &[(f64, f64)]) {
    assert_eq!(got.len(), expected.len());
    for (g, e) in got.iter().zip(expected) {
        assert_abs_diff_eq!(g.0, e.0, epsilon = 1e-12);
        assert_abs_diff_eq!(g.1, e.1, epsilon = 1e-12);
    }
}

/// Get the (RA, Dec) of each of a source's components [radians].
fn source_radecs(sl: &SourceList, name: &str) -> Vec<(f64, f64)> {
    sl[name]
        .components
        .iter()
        .map(|c| (c.radec.ra, c.radec.dec))
        .collect()
}

/// Get the (RA, Dec) of each of a source's components [radians], after a shift.
fn shifted_radecs(sl: &SourceList, name: &str, shift: Shift) -> Vec<(f64, f64)> {
    source_radecs(sl, name)
        .into_iter()
        .map(|(ra, dec)| {
            (
                ra + shift.ra_deg.to_radians(),
                dec + shift.dec_deg.to_radians(),
            )
        })
        .collect()
}

/// Only the sources in the shifts file are written, and each is shifted by
/// its own amount. (The first sources are near RA 0°, so the RA shifts are
/// positive; `shift` doesn't wrap RA into [0°, 360°).)
#[test]
fn test_shift_writes_only_shifted_sources() {
    let (sl, _) = read_source_list_file(HYP_SRCLIST_PATH, None).unwrap();
    let names = first_names(&sl, 2);
    let shifts = [
        Shift {
            ra_deg: 0.5,
            dec_deg: -0.25,
        },
        Shift {
            ra_deg: 1.0,
            dec_deg: 0.75,
        },
    ];
    let shifts_file = write_shifts(&[(&names[0], shifts[0]), (&names[1], shifts[1])]);

    let output = tempfile::Builder::new().suffix(".json").tempfile().unwrap();
    shift(
        Path::new(HYP_SRCLIST_PATH),
        shifts_file.path(),
        Some(output.path()),
        None,
        None,
        false,
        false,
        None,
    )
    .unwrap();

    let shifted = read_json(output.path());
    assert_eq!(shifted.len(), 2);
    for (name, s) in names.iter().zip(shifts) {
        assert_radecs_eq(
            &source_radecs(&shifted, name),
            &shifted_radecs(&sl, name, s),
        );
    }
}

/// With `include_unshifted_sources`, the sources not in the shifts file are
/// written too, unchanged.
#[test]
fn test_shift_include_unshifted_sources() {
    let (sl, _) = read_source_list_file(HYP_SRCLIST_PATH, None).unwrap();
    let name = first_names(&sl, 1).remove(0);
    let s = Shift {
        ra_deg: 0.1,
        dec_deg: 0.2,
    };
    let shifts_file = write_shifts(&[(&name, s)]);

    let output = tempfile::Builder::new().suffix(".json").tempfile().unwrap();
    shift(
        Path::new(HYP_SRCLIST_PATH),
        shifts_file.path(),
        Some(output.path()),
        None,
        None,
        false,
        true,
        None,
    )
    .unwrap();

    let shifted = read_json(output.path());
    assert_eq!(shifted.len(), sl.len());
    for other in sl.keys() {
        let expected = if *other == name {
            shifted_radecs(&sl, other, s)
        } else {
            source_radecs(&sl, other)
        };
        assert_radecs_eq(&source_radecs(&shifted, other), &expected);
    }
}

/// Collapsing a non-RTS source list puts the apparently brightest source's
/// components first, then those of the other sources in file order, each
/// with its own shift. (Before indexmap v2's `shift_remove_entry` was used,
/// the last source's components came straight after the base source's.)
#[test]
fn test_shift_collapse_keeps_file_order() {
    let (sl, _) = read_source_list_file(HYP_SRCLIST_PATH, None).unwrap();
    // Shift every source by a different amount, so a component that ends up
    // in the wrong place, or with another source's shift, is caught. Shift RA
    // towards 180°, because `shift` doesn't wrap RA into [0°, 360°).
    let shifts: Vec<(String, Shift)> = sl
        .iter()
        .enumerate()
        .map(|(i, (name, src))| {
            let ra_sign = if src.components[0].radec.ra.to_degrees() < 180.0 {
                1.0
            } else {
                -1.0
            };
            (
                name.clone(),
                Shift {
                    ra_deg: ra_sign * 0.01 * (i + 1) as f64,
                    dec_deg: -0.02 * (i + 1) as f64,
                },
            )
        })
        .collect();
    let shifts_file = write_shifts(
        &shifts
            .iter()
            .map(|(n, s)| (n.as_str(), *s))
            .collect::<Vec<_>>(),
    );

    let output = tempfile::Builder::new().suffix(".json").tempfile().unwrap();
    shift(
        Path::new(HYP_SRCLIST_PATH),
        shifts_file.path(),
        Some(output.path()),
        None,
        None,
        true,
        false,
        Some(Path::new(METAFITS_PATH)),
    )
    .unwrap();

    let collapsed = read_json(output.path());
    assert_eq!(collapsed.len(), 1);
    let (base_name, _) = collapsed.get_index(0).unwrap();
    let base_index = sl.get_index_of(base_name).unwrap();
    // If the base were the last source, swapping and shifting would give the
    // same order, and this test would not check anything.
    assert_ne!(base_index, sl.len() - 1);

    let expected: Vec<(f64, f64)> = std::iter::once(&shifts[base_index])
        .chain(shifts.iter().filter(|(n, _)| n != base_name))
        .flat_map(|(n, s)| shifted_radecs(&sl, n, *s))
        .collect();
    assert_radecs_eq(&source_radecs(&collapsed, base_name), &expected);
}

/// Collapsing an RTS source list uses the first shifted source in the file as
/// the base, and adds the other shifted sources in file order. (This used to
/// panic, because the names of the collapsed sources were still written.)
#[test]
fn test_shift_collapse_rts_keeps_file_order() {
    let (sl, _) = read_source_list_file(RTS_SRCLIST_PATH, None).unwrap();
    // Pick a few sources, out of file order, and leave the rest unshifted (and
    // so dropped).
    let picked: Vec<String> = [7, 2, 4]
        .iter()
        .map(|&i| sl.get_index(i).unwrap().0.clone())
        .collect();
    let s = Shift {
        ra_deg: 0.3,
        dec_deg: -0.1,
    };
    let shifts_file = write_shifts(&picked.iter().map(|n| (n.as_str(), s)).collect::<Vec<_>>());

    let output = tempfile::Builder::new().suffix(".txt").tempfile().unwrap();
    shift(
        Path::new(RTS_SRCLIST_PATH),
        shifts_file.path(),
        Some(output.path()),
        None,
        None,
        true,
        false,
        Some(Path::new(METAFITS_PATH)),
    )
    .unwrap();

    let (collapsed, _) = read_source_list_file(output.path(), None).unwrap();
    assert_eq!(collapsed.len(), 1);
    let file_order: Vec<&String> = [2, 4, 7]
        .iter()
        .map(|&i| sl.get_index(i).unwrap().0)
        .collect();
    assert_eq!(collapsed.get_index(0).unwrap().0, file_order[0]);
    let expected: Vec<(f64, f64)> = file_order
        .iter()
        .flat_map(|n| shifted_radecs(&sl, n, s))
        .collect();
    // The RTS format writes fewer digits than a .json, so allow for that.
    let got = source_radecs(&collapsed, file_order[0]);
    assert_eq!(got.len(), expected.len());
    for (g, e) in got.iter().zip(&expected) {
        assert_abs_diff_eq!(g.0, e.0, epsilon = 1e-6);
        assert_abs_diff_eq!(g.1, e.1, epsilon = 1e-6);
    }
}

/// Collapsing needs a metafits.
#[test]
fn test_shift_collapse_needs_metafits() {
    let (sl, _) = read_source_list_file(HYP_SRCLIST_PATH, None).unwrap();
    let name = first_names(&sl, 1).remove(0);
    let shifts_file = write_shifts(&[(
        &name,
        Shift {
            ra_deg: 0.0,
            dec_deg: 0.0,
        },
    )]);
    let output = tempfile::Builder::new().suffix(".json").tempfile().unwrap();
    let result = shift(
        Path::new(HYP_SRCLIST_PATH),
        shifts_file.path(),
        Some(output.path()),
        None,
        None,
        true,
        false,
        None,
    );
    assert!(matches!(result, Err(SrclistError::MissingMetafits)));
}
