// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Code for easing testing. This module should only be used during testing.
 */

pub mod gpuboxes;
pub mod no_gpuboxes;

pub use std::fs::File;
pub use std::io::Write;
pub use std::path::{Path, PathBuf};
pub use std::process::Output;
pub use std::str::from_utf8;

pub use assert_cmd::{output::OutputError, Command};
pub use serial_test::serial;
pub use tempfile::TempDir;

pub use mwa_hyperdrive::calibrate::args::CalibrateUserArgs;

pub struct MwaData {
    /// The MWA observation GPS time.
    pub obsid: u32,

    /// The metafits file associated with the observation.
    pub metafits: String,

    /// Raw MWA gpubox files.
    pub gpuboxes: Vec<String>,

    /// cotter mwaf files. Can be empty.
    pub mwafs: Vec<String>,

    /// Sky-model source list.
    pub source_list: Option<String>,
}

pub fn make_file_in_dir<T: AsRef<Path> + ?Sized>(filename: &T, dir: &Path) -> (PathBuf, File) {
    let mut path = dir.to_path_buf();
    path.push(filename);
    let f = File::create(&path).expect("couldn't make file");
    (path, f)
}

pub fn path_to_string(p: &Path) -> String {
    p.to_str().map(|s| s.to_string()).unwrap()
}

pub fn deflate_gz<T: AsRef<Path>>(gz_file: &T, out_file: &mut File) {
    let mut gz = flate2::read::GzDecoder::new(File::open(gz_file).unwrap());
    std::io::copy(&mut gz, out_file).unwrap();
}

pub fn serialise_args_toml(args: &CalibrateUserArgs, file: &mut File) {
    let ser = toml::to_string_pretty(&args).expect("couldn't serialise CalibrateUserArgs as toml");
    write!(file, "{}", ser).unwrap();
}

pub fn serialise_args_json(args: &CalibrateUserArgs, file: &mut File) {
    let ser =
        serde_json::to_string_pretty(&args).expect("couldn't serialise CalibrateUserArgs as json");
    write!(file, "{}", ser).unwrap();
}

pub fn get_cmd_output(result: Result<Output, OutputError>) -> (String, String) {
    let output = match result {
        Ok(o) => o,
        Err(o) => o.as_output().unwrap().clone(),
    };
    (
        from_utf8(&output.stdout).unwrap().to_string(),
        from_utf8(&output.stderr).unwrap().to_string(),
    )
}

pub fn hyperdrive() -> Command {
    Command::cargo_bin("hyperdrive").unwrap()
}
