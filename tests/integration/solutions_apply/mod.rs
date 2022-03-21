// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Integration tests for solutions-apply.

mod cli_args;

use std::path::Path;

use marlu::Jones;
use ndarray::prelude::*;

use mwa_hyperdrive::solutions::CalibrationSolutions;
use mwa_hyperdrive_common::{marlu, ndarray};

pub(crate) fn get_1090008640_identity_solutions_file(file: &Path) {
    let sols = CalibrationSolutions {
        di_jones: Array3::from_elem((1, 128, 32), Jones::identity()),
        ..Default::default()
    };
    sols.write_solutions_from_ext::<&Path, &Path>(file, None)
        .unwrap();
}
