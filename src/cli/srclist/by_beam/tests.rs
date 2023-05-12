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
        filter_points: false,
        filter_gaussians: false,
        filter_shapelets: false,
        collapse_into_single_source: false,
        rts_base_source: None,
        beam_args: BeamArgs {
            beam_file: None,
            unity_dipole_gains: false,
            delays: None,
            no_beam: false,
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
