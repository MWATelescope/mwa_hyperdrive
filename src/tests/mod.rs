// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code for easing testing. This module should only be used during testing.

pub(crate) mod calibrate;
pub(crate) mod full_obsids;
pub(crate) mod reduced_obsids;

pub(crate) use std::fs::File;
pub(crate) use std::io::Write;
pub(crate) use std::path::{Path, PathBuf};

pub(crate) use approx::assert_abs_diff_eq;
pub(crate) use tempfile::{NamedTempFile, TempPath};
// Need to use serial tests because HDF5 is not necessarily reentrant.
pub(crate) use serial_test::serial;

pub(crate) use crate::calibrate::args::CalibrateUserArgs;

pub(crate) fn make_file_in_dir<T: AsRef<Path>, U: AsRef<Path>>(
    filename: T,
    dir: U,
) -> (PathBuf, File) {
    let path = dir.as_ref().join(filename);
    let f = File::create(&path).expect("couldn't make file");
    (path, f)
}

pub(crate) fn path_to_string(p: &Path) -> String {
    p.to_str().map(|s| s.to_string()).unwrap()
}

pub(crate) fn deflate_gz_into_file<T: AsRef<Path>>(gz_file: T, out_file: &mut File) {
    let mut gz = flate2::read::GzDecoder::new(File::open(gz_file).unwrap());
    std::io::copy(&mut gz, out_file).unwrap();
}

pub(crate) fn deflate_gz_into_tempfile<T: AsRef<Path>>(file: T) -> TempPath {
    let (mut temp_file, temp_path) = NamedTempFile::new().unwrap().into_parts();
    deflate_gz_into_file(file, &mut temp_file);
    temp_path
}

pub(crate) fn serialise_args_toml(args: &CalibrateUserArgs, file: &mut File) {
    let ser = toml::to_string_pretty(&args).expect("couldn't serialise CalibrateUserArgs as toml");
    write!(file, "{}", ser).unwrap();
}

pub(crate) fn serialise_args_json(args: &CalibrateUserArgs, file: &mut File) {
    let ser =
        serde_json::to_string_pretty(&args).expect("couldn't serialise CalibrateUserArgs as json");
    write!(file, "{}", ser).unwrap();
}
