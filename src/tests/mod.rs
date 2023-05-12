// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Helpful functions for tests.

use std::{
    fs::File,
    path::{Path, PathBuf},
};

use tempfile::{NamedTempFile, TempPath};

pub(crate) fn deflate_gz_into_file<T: AsRef<Path>>(gz_file: T, out_file: &mut File) {
    let mut gz = flate2::read::GzDecoder::new(File::open(gz_file).unwrap());
    std::io::copy(&mut gz, out_file).unwrap();
}

pub(crate) fn deflate_gz_into_tempfile<T: AsRef<Path>>(file: T) -> TempPath {
    let (mut temp_file, temp_path) = NamedTempFile::new().unwrap().into_parts();
    deflate_gz_into_file(file, &mut temp_file);
    temp_path
}

const DATA_DIR_1090008640: &str = "test_files/1090008640";

pub(crate) struct DataAsStrings {
    pub(crate) metafits: String,
    pub(crate) vis: Vec<String>,
    pub(crate) mwafs: Vec<String>,
    pub(crate) srclist: String,
}

pub(crate) struct DataAsPathBufs {
    pub(crate) metafits: PathBuf,
    pub(crate) vis: Vec<PathBuf>,
    pub(crate) mwafs: Vec<PathBuf>,
    pub(crate) srclist: PathBuf,
}

pub(crate) fn get_reduced_1090008640_raw() -> DataAsStrings {
    DataAsStrings {
        metafits: format!("{DATA_DIR_1090008640}/1090008640.metafits"),
        vis: vec![format!(
            "{DATA_DIR_1090008640}/1090008640_20140721201027_gpubox01_00.fits"
        )],
        mwafs: vec![format!("{DATA_DIR_1090008640}/1090008640_01.mwaf")],
        srclist: format!(
            "{DATA_DIR_1090008640}/srclist_pumav3_EoR0aegean_EoR1pietro+ForA_1090008640_100.yaml"
        ),
    }
}

pub(crate) fn get_reduced_1090008640_ms() -> DataAsStrings {
    let mut data = get_reduced_1090008640_raw();
    data.vis[0] = format!("{DATA_DIR_1090008640}/1090008640.ms");
    data
}

pub(crate) fn get_reduced_1090008640_uvfits() -> DataAsStrings {
    let mut data = get_reduced_1090008640_raw();
    data.vis[0] = format!("{DATA_DIR_1090008640}/1090008640.uvfits");
    data
}

pub(crate) fn get_reduced_1090008640_raw_pbs() -> DataAsPathBufs {
    let DataAsStrings {
        metafits,
        vis,
        mwafs,
        srclist,
    } = get_reduced_1090008640_raw();
    let pbs = DataAsPathBufs {
        metafits: PathBuf::from(metafits).canonicalize().unwrap(),
        vis: vis
            .into_iter()
            .map(|s| PathBuf::from(s).canonicalize().unwrap())
            .collect(),
        mwafs: mwafs
            .into_iter()
            .map(|s| PathBuf::from(s).canonicalize().unwrap())
            .collect(),
        srclist: PathBuf::from(srclist).canonicalize().unwrap(),
    };

    // Ensure that the required files are there.
    for file in [&pbs.metafits]
        .into_iter()
        .chain(pbs.vis.iter())
        .chain(pbs.mwafs.iter())
        .chain([&pbs.srclist].into_iter())
    {
        assert!(
            file.exists(),
            "Could not find '{}', which is required for this test",
            file.display()
        );
    }

    pbs
}

pub(crate) fn get_reduced_1090008640_ms_pbs() -> DataAsPathBufs {
    let mut data = get_reduced_1090008640_raw_pbs();
    data.vis[0] = PathBuf::from(DATA_DIR_1090008640)
        .canonicalize()
        .unwrap()
        .join("1090008640.ms");
    data
}

pub(crate) fn get_reduced_1090008640_uvfits_pbs() -> DataAsPathBufs {
    let mut data = get_reduced_1090008640_raw_pbs();
    data.vis[0] = PathBuf::from(DATA_DIR_1090008640)
        .canonicalize()
        .unwrap()
        .join("1090008640.uvfits");
    data
}
