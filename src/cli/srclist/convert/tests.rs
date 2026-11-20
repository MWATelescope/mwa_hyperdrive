// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Tests for converting source lists.

use std::{fs::File, io::BufReader, path::Path};

use approx::assert_abs_diff_eq;

use super::convert;
use crate::srclist::{hyperdrive::source_list_from_json, read::read_source_list_file, SourceList};

/// The sky-model used by these tests.
const SRCLIST_PATH: &str =
    "test_files/1090008640/srclist_pumav3_EoR0aegean_EoR1pietro+ForA_1090008640_peel100.txt";

/// The metafits used by these tests.
const METAFITS_PATH: &str = "test_files/1090008640/1090008640.metafits";

/// Get the sorted (RA, Dec) [radians] of all components in a source list.
fn get_sorted_component_radecs(sl: &SourceList) -> Vec<(f64, f64)> {
    let mut radecs: Vec<(f64, f64)> = sl
        .values()
        .flat_map(|src| src.components.iter())
        .map(|comp| (comp.radec.ra, comp.radec.dec))
        .collect();
    radecs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    radecs
}

/// Collapsing a source list into a single source must keep every component
/// exactly once, whatever order the other sources are visited in.
#[test]
fn test_convert_collapse_keeps_all_components() {
    let (sl, _) = read_source_list_file(SRCLIST_PATH, None).unwrap();
    assert!(sl.len() > 1);
    let expected = get_sorted_component_radecs(&sl);

    let temp = tempfile::Builder::new().suffix(".json").tempfile().unwrap();
    convert::<&Path, &str>(
        Path::new(SRCLIST_PATH),
        temp.path(),
        None,
        None,
        true,
        Some(Path::new(METAFITS_PATH)),
        false,
        false,
        false,
    )
    .unwrap();

    let mut f = BufReader::new(File::open(temp.path()).unwrap());
    let collapsed = source_list_from_json(&mut f).unwrap();
    assert_eq!(collapsed.len(), 1);
    let result = get_sorted_component_radecs(&collapsed);
    assert_eq!(result.len(), expected.len());
    for (got, expected) in result.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(got.0, expected.0, epsilon = 1e-9);
        assert_abs_diff_eq!(got.1, expected.1, epsilon = 1e-9);
    }
}

/// Collapsing a source list puts the apparently brightest source's components
/// first, then those of the other sources in file order. (Before indexmap v2's
/// `shift_remove_entry` was used, the last source's components came straight
/// after the base source's.)
#[test]
fn test_convert_collapse_keeps_file_order() {
    let (sl, _) = read_source_list_file(SRCLIST_PATH, None).unwrap();

    let temp = tempfile::Builder::new().suffix(".json").tempfile().unwrap();
    convert::<&Path, &str>(
        Path::new(SRCLIST_PATH),
        temp.path(),
        None,
        None,
        true,
        Some(Path::new(METAFITS_PATH)),
        false,
        false,
        false,
    )
    .unwrap();

    let mut f = BufReader::new(File::open(temp.path()).unwrap());
    let collapsed = source_list_from_json(&mut f).unwrap();
    assert_eq!(collapsed.len(), 1);
    let (base_name, base_src) = collapsed.get_index(0).unwrap();
    let base_index = sl.get_index_of(base_name).unwrap();
    // If the base were the last source, swapping and shifting would give the
    // same order, and this test would not check anything.
    assert_ne!(base_index, sl.len() - 1);

    let expected: Vec<(f64, f64)> = std::iter::once(&sl[base_index])
        .chain(
            sl.iter()
                .filter(|(n, _)| *n != base_name)
                .map(|(_, src)| src),
        )
        .flat_map(|src| src.components.iter())
        .map(|comp| (comp.radec.ra, comp.radec.dec))
        .collect();
    assert_eq!(base_src.components.len(), expected.len());
    for (got, expected) in base_src.components.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(got.radec.ra, expected.0, epsilon = 1e-9);
        assert_abs_diff_eq!(got.radec.dec, expected.1, epsilon = 1e-9);
    }
}
