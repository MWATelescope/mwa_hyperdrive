// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::{fs::File, io::BufReader, path::PathBuf};

use approx::assert_abs_diff_eq;

use super::SrclistByBeamArgs;
use crate::{
    cli::common::BeamArgs,
    srclist::{hyperdrive::source_list_from_json, read::read_source_list_file},
};

#[test]
fn test_srclist_by_beam() {
    let sl_path = PathBuf::from(
        "test_files/1090008640/srclist_pumav3_EoR0aegean_EoR1pietro+ForA_1090008640_peel100.txt",
    );
    let (sl, _) = read_source_list_file(&sl_path, None).unwrap();
    let n = 5;

    let temp = tempfile::Builder::new().suffix(".json").tempfile().unwrap();
    SrclistByBeamArgs {
        input_source_list: sl_path,
        output_source_list: Some(temp.path().to_path_buf()),
        input_type: None,
        output_type: None,
        metafits: Some(PathBuf::from("test_files/1090008640/1090008640.metafits")),
        array_position: None,
        lst_rad: None,
        phase_centre: None,
        freqs_hz: None,
        number: n,
        source_dist_cutoff: None,
        veto_threshold: None,
        elevation_limit: None,
        filter_points: false,
        filter_gaussians: false,
        filter_shapelets: false,
        collapse_into_single_source: false,
        rts_base_source: None,
        beam_args: BeamArgs {
            beam_type: Some("fee".to_string()),
            no_beam: false,
            delays: None,
            unity_dipole_gains: false,
            beam_file: None,
        },
    }
    .run()
    .unwrap();

    let f = File::open(temp.path()).unwrap();
    let mut f = BufReader::new(f);
    let new_sl = source_list_from_json(&mut f).unwrap();
    assert_eq!(new_sl.len(), n);
    for i in 0..n {
        assert_abs_diff_eq!(sl[i], new_sl[i]);
    }
}

/// The sky-model used by the by-beam tests.
const SRCLIST_PATH: &str =
    "test_files/1090008640/srclist_pumav3_EoR0aegean_EoR1pietro+ForA_1090008640_peel100.txt";

/// The metafits used by the by-beam tests.
const METAFITS_PATH: &str = "test_files/1090008640/1090008640.metafits";

/// Make by-beam arguments for the 1090008640 test files.
fn get_args(
    output: &std::path::Path,
    number: usize,
    collapse_into_single_source: bool,
    rts_base_source: Option<String>,
) -> SrclistByBeamArgs {
    SrclistByBeamArgs {
        input_source_list: PathBuf::from(SRCLIST_PATH),
        output_source_list: Some(output.to_path_buf()),
        input_type: None,
        output_type: None,
        metafits: Some(PathBuf::from(METAFITS_PATH)),
        array_position: None,
        lst_rad: None,
        phase_centre: None,
        freqs_hz: None,
        number,
        source_dist_cutoff: None,
        veto_threshold: None,
        elevation_limit: None,
        filter_points: false,
        filter_gaussians: false,
        filter_shapelets: false,
        collapse_into_single_source,
        rts_base_source,
        beam_args: BeamArgs {
            beam_type: Some("fee".to_string()),
            no_beam: false,
            delays: None,
            unity_dipole_gains: false,
            beam_file: None,
        },
    }
}

/// When the source list is collapsed into a single source, the base source's
/// components come first, then the components of the next `number` sources in
/// their original (brightest first) order. Removing the base source from the
/// list must therefore keep the order of the other sources. (`number` limits
/// the other sources, so the collapsed source has the base plus `number`
/// others.)
#[test]
fn test_srclist_by_beam_collapse_keeps_source_order() {
    let n = 5;
    let base_index = 1;

    // Get the (vetoed and ordered) sources without collapsing. Ask for the
    // base source plus `n` others.
    let temp_sl = tempfile::Builder::new().suffix(".json").tempfile().unwrap();
    get_args(temp_sl.path(), n + 1, false, None).run().unwrap();
    let mut f = BufReader::new(File::open(temp_sl.path()).unwrap());
    let sl = source_list_from_json(&mut f).unwrap();
    assert_eq!(sl.len(), n + 1);
    let base_name = sl.get_index(base_index).unwrap().0.clone();

    // Collapse, using a source that is not the first as the base.
    let temp_collapsed = tempfile::Builder::new().suffix(".json").tempfile().unwrap();
    get_args(temp_collapsed.path(), n, true, Some(base_name.clone()))
        .run()
        .unwrap();
    let mut f = BufReader::new(File::open(temp_collapsed.path()).unwrap());
    let collapsed = source_list_from_json(&mut f).unwrap();
    assert_eq!(collapsed.len(), 1);
    let (collapsed_name, collapsed_src) = collapsed.get_index(0).unwrap();
    assert_eq!(collapsed_name, &base_name);

    // The base source's components, then the others' in their original order.
    let expected_comps: Vec<_> = sl
        .get_index(base_index)
        .unwrap()
        .1
        .components
        .iter()
        .chain(
            sl.iter()
                .enumerate()
                .filter(|(i, _)| *i != base_index)
                .flat_map(|(_, (_, src))| src.components.iter()),
        )
        .collect();
    assert_eq!(collapsed_src.components.len(), expected_comps.len());
    for (got, expected) in collapsed_src.components.iter().zip(expected_comps) {
        assert_abs_diff_eq!(got.radec.ra, expected.radec.ra, epsilon = 1e-9);
        assert_abs_diff_eq!(got.radec.dec, expected.radec.dec, epsilon = 1e-9);
    }
}
