// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to parse filenames.
//!
//! [InputDataTypes] is the struct to be used here. It is constructed from a
//! slice of string filenames, and it enforces things like allowing only one
//! metafits file to be present.

use std::fs::OpenOptions;
use std::path::{Path, PathBuf};

use regex::{Regex, RegexBuilder};
use thiserror::Error;
use vec1::Vec1;

use crate::io::{get_all_matches_from_glob, read::VisReadError, GlobError};

lazy_static::lazy_static! {
    // gpubox files should not be renamed in any way! This includes the case of
    // the letters in the filename. mwalib should complain if this is not the
    // case.
    static ref RE_GPUBOX: Regex =
        RegexBuilder::new(r".*gpubox.*\.fits$")
            .case_insensitive(false).build().unwrap();

    static ref RE_MWAX: Regex =
        RegexBuilder::new(r"\d{10}_\d{8}(.)?\d{6}_ch\d{3}_\d{3}\.fits$")
            .case_insensitive(false).build().unwrap();
}

pub(super) const SUPPORTED_INPUT_FILE_COMBINATIONS: &str =
    "gpubox + metafits (+ mwaf)\nms (+ metafits)\nuvfits (+ metafits)";

pub(super) const SUPPORTED_CALIBRATED_INPUT_FILE_COMBINATIONS: &str =
    "ms (+ metafits)\nuvfits (+ metafits)";

#[derive(Debug)]
/// Supported input data types for calibration.
pub(crate) struct InputDataTypes {
    pub(crate) metafits: Option<Vec1<PathBuf>>,
    pub(crate) gpuboxes: Option<Vec1<PathBuf>>,
    pub(crate) mwafs: Option<Vec1<PathBuf>>,
    pub(crate) ms: Option<Vec1<PathBuf>>,
    pub(crate) uvfits: Option<Vec1<PathBuf>>,
}

// The same as `InputDataTypes`, but all types are allowed to be multiples. This
// makes coding easier.
#[derive(Debug, Default)]
struct InputDataTypesTemp {
    metafits: Vec<PathBuf>,
    gpuboxes: Vec<PathBuf>,
    mwafs: Vec<PathBuf>,
    ms: Vec<PathBuf>,
    uvfits: Vec<PathBuf>,
}

impl InputDataTypes {
    /// From an input collection of filename or glob strings, disentangle the
    /// file types and populate [InputDataTypes].
    pub(super) fn new(files: &[String]) -> Result<InputDataTypes, VisReadError> {
        let mut temp = InputDataTypesTemp::default();

        for file in files.iter().map(|f| f.as_str()) {
            file_checker(&mut temp, file)?;
        }

        Ok(Self {
            metafits: if temp.metafits.is_empty() {
                None
            } else {
                Some(Vec1::try_from_vec(temp.metafits).unwrap())
            },
            gpuboxes: if temp.gpuboxes.is_empty() {
                None
            } else {
                Some(Vec1::try_from_vec(temp.gpuboxes).unwrap())
            },
            mwafs: if temp.mwafs.is_empty() {
                None
            } else {
                Some(Vec1::try_from_vec(temp.mwafs).unwrap())
            },
            ms: if temp.ms.is_empty() {
                None
            } else {
                Some(Vec1::try_from_vec(temp.ms).unwrap())
            },
            uvfits: if temp.uvfits.is_empty() {
                None
            } else {
                Some(Vec1::try_from_vec(temp.uvfits).unwrap())
            },
        })
    }
}

fn exists_and_is_readable(file: &Path) -> Result<(), InputFileError> {
    if !file.exists() {
        return Err(InputFileError::DoesNotExist(file.display().to_string()));
    }
    match OpenOptions::new()
        .read(true)
        .open(file)
        .map_err(|io_error| io_error.kind())
    {
        Ok(_) => (),
        Err(std::io::ErrorKind::PermissionDenied) => {
            return Err(InputFileError::CouldNotRead(file.display().to_string()))
        }
        Err(e) => return Err(InputFileError::IO(file.display().to_string(), e.into())),
    }

    Ok(())
}

// Given a file (as a string), check it exists and is readable, then determine
// what type it is, and add it to the provided file types struct. If the file
// string doesn't exist, then check if it's a glob string, and act recursively
// on the glob results.
fn file_checker(file_types: &mut InputDataTypesTemp, file: &str) -> Result<(), InputFileError> {
    let file_pb = PathBuf::from(file);
    // Is this a file, and is it readable?
    match exists_and_is_readable(&file_pb) {
        Ok(_) => (),

        // If this string isn't a file, maybe it's a glob.
        Err(InputFileError::DoesNotExist(f)) => {
            match get_all_matches_from_glob(file) {
                Ok(glob_results) => {
                    // If there were no glob matches, then just return the
                    // original error (the file does not exist).
                    if glob_results.is_empty() {
                        return Err(InputFileError::DoesNotExist(f));
                    }

                    // Iterate over all glob results, adding them to the file
                    // types.
                    for pb in glob_results {
                        file_checker(file_types, pb.display().to_string().as_str())?;
                    }
                    return Ok(());
                }

                // Propagate all other errors.
                Err(e) => return Err(InputFileError::from(e)),
            }
        }

        // Propagate all other errors.
        Err(e) => return Err(e),
    };
    if file.contains("_metafits_ppds.fits") {
        return Err(InputFileError::PpdMetafitsUnsupported(file.to_string()));
    }
    match (
        file.ends_with(".metafits") || file.ends_with("_metafits.fits"),
        RE_GPUBOX.is_match(file),
        RE_MWAX.is_match(file),
        file.ends_with(".mwaf"),
        file.ends_with(".ms"),
        file.ends_with(".uvfits"),
    ) {
        (true, false, false, false, false, false) => file_types.metafits.push(file_pb),
        (false, true, false, false, false, false) => file_types.gpuboxes.push(file_pb),
        (false, false, true, false, false, false) => file_types.gpuboxes.push(file_pb),
        (false, false, false, true, false, false) => file_types.mwafs.push(file_pb),
        (false, false, false, false, true, false) => file_types.ms.push(file_pb),
        (false, false, false, false, false, true) => file_types.uvfits.push(file_pb),
        _ => return Err(InputFileError::NotRecognised(file.to_string())),
    }

    Ok(())
}

#[derive(Debug, Error)]
pub enum InputFileError {
    #[error("Specified file does not exist: {0}")]
    DoesNotExist(String),

    #[error("Could not read specified file: {0}")]
    CouldNotRead(String),

    #[error("The specified file '{0}' is a \"PPDs metafits\" and is not supported. Please use a newer metafits file.")]
    PpdMetafitsUnsupported(String),

    #[error("The specified file '{0}' was not a recognised file type.")]
    NotRecognised(String),

    #[error(transparent)]
    Glob(#[from] GlobError),

    #[error("IO error when attempting to read file '{0}': {1}")]
    IO(String, std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::{Builder, TempDir, TempPath};

    fn make_new_dir() -> TempDir {
        TempDir::new().unwrap()
    }

    fn make_file(dir: &Path, suffix: &str) -> TempPath {
        Builder::new()
            .prefix("asdf")
            .suffix(suffix)
            .rand_bytes(1)
            .tempfile_in(dir)
            .unwrap()
            .into_temp_path()
    }

    fn make_metafits(dir: &Path) -> TempPath {
        make_file(dir, ".metafits")
    }

    fn make_metafits2(dir: &Path) -> TempPath {
        make_file(dir, "_metafits.fits")
    }

    fn make_ms(dir: &Path) -> TempPath {
        make_file(dir, ".ms")
    }

    fn make_uvfits(dir: &Path) -> TempPath {
        make_file(dir, ".uvfits")
    }

    fn make_legacy_gpubox(dir: &Path) -> TempPath {
        Builder::new()
            .prefix("1065880128_01234567890123_gpubox01_00")
            .suffix(".fits")
            .rand_bytes(1)
            .tempfile_in(dir)
            .unwrap()
            .into_temp_path()
    }

    // The regex we use to match MWAX gpuboxes is too strict to have random
    // characters. Make a file normally in the TempDir. It will get deleted when
    // the TempDir is dropped.
    fn make_mwax_gpubox(dir: &Path) -> PathBuf {
        let file_path = dir.join("1247842824_20190722150006_ch113_000.fits");
        std::fs::File::create(&file_path).unwrap();
        file_path
    }

    #[test]
    fn test_non_existent_file() {
        let result = exists_and_is_readable(&PathBuf::from("/does/not/exist.metafits"));
        assert!(result.is_err());
        match result {
            Err(InputFileError::DoesNotExist(_)) => (),
            Err(e) => panic!("Unexpected error kind! {e:?}"),
            Ok(_) => unreachable!(),
        }
    }

    #[test]
    #[cfg(unix)]
    fn test_unreadable_file() {
        // This test only works on "unix" because windows can't reliably alter
        // write permissions.
        use std::os::unix::fs::PermissionsExt;

        // Make a temporary file and remove write permissions from it.
        let tmp_file = tempfile::NamedTempFile::new().expect("Couldn't make a temp file");
        let mut perms = tmp_file
            .as_file()
            .metadata()
            .expect("Couldn't get file metadata")
            .permissions();
        perms.set_mode(0o000); // No read/write for anyone.
        tmp_file
            .as_file()
            .set_permissions(perms)
            .expect("Couldn't set permissions");

        let result = exists_and_is_readable(tmp_file.path());
        assert!(result.is_err());
        match result {
            Err(InputFileError::CouldNotRead(_)) => (),
            Err(e) => panic!("Unexpected error kind! {e:?}"),
            Ok(_) => unreachable!(),
        }

        // Set read/write for the owner so the file can be deleted.
        let mut perms = tmp_file
            .as_file()
            .metadata()
            .expect("Couldn't get file metadata")
            .permissions();
        perms.set_mode(0o600);
        tmp_file
            .as_file()
            .set_permissions(perms)
            .expect("Couldn't set permissions");
    }

    #[test]
    fn test_unrecognised_file() {
        let mut input = InputDataTypesTemp::default();
        // Is /tmp always present?
        let result = file_checker(&mut input, "/tmp");
        assert!(result.is_err());
        match result {
            Err(InputFileError::NotRecognised(_)) => (),
            Err(e) => panic!("Unexpected error kind! {e:?}"),
            Ok(_) => unreachable!(),
        }
    }

    #[test]
    fn test_non_existent_file2() {
        let mut input = InputDataTypesTemp::default();
        let result = file_checker(&mut input, "/does/not/exist.metafits");
        assert!(result.is_err());
        match result {
            Err(InputFileError::DoesNotExist(_)) => (),
            Err(e) => panic!("Unexpected error kind! {e:?}"),
            Ok(_) => unreachable!(),
        }
    }

    #[test]
    fn test_found_metafits() {
        let mut input = InputDataTypesTemp::default();
        let dir = make_new_dir();
        let metafits = make_metafits(dir.path());
        let result = file_checker(&mut input, metafits.to_str().unwrap());
        assert!(result.is_ok());
        assert_eq!(input.metafits.len(), 1);

        let mut input = InputDataTypesTemp::default();
        let dir = make_new_dir();
        let metafits = make_metafits2(dir.path());
        let result = file_checker(&mut input, metafits.to_str().unwrap());
        assert!(result.is_ok());
        assert_eq!(input.metafits.len(), 1);
    }

    #[test]
    fn test_found_ms() {
        let mut input = InputDataTypesTemp::default();
        let dir = make_new_dir();
        let ms = make_ms(dir.path());
        let result = file_checker(&mut input, ms.to_str().unwrap());
        assert!(result.is_ok());
        assert_eq!(input.ms.len(), 1);
    }

    #[test]
    fn test_found_uvfits() {
        let mut input = InputDataTypesTemp::default();
        let dir = make_new_dir();
        let uvfits = make_uvfits(dir.path());
        let result = file_checker(&mut input, uvfits.to_str().unwrap());
        assert!(result.is_ok());
        assert_eq!(input.uvfits.len(), 1);
    }

    #[test]
    fn test_found_legacy_gpubox() {
        let mut input = InputDataTypesTemp::default();
        let dir = make_new_dir();
        let gpubox = make_legacy_gpubox(dir.path());
        let result = file_checker(&mut input, gpubox.to_str().unwrap());
        assert!(result.is_ok());
        assert_eq!(input.gpuboxes.len(), 1);
    }

    #[test]
    fn test_found_mwax_gpubox() {
        let mut input = InputDataTypesTemp::default();
        let dir = make_new_dir();
        let gpubox = make_mwax_gpubox(dir.path());
        let result = file_checker(&mut input, gpubox.to_str().unwrap());
        assert!(result.is_ok(), "{}", result.unwrap_err());
        assert_eq!(input.gpuboxes.len(), 1);
    }
}
