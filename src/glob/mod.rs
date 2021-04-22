// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Functions to glob files.
 */

use std::path::PathBuf;

use glob::glob;
use thiserror::Error;

/// Given a glob pattern, get all of the matches from the filesystem.
///
/// # Examples
///
/// ```
/// # use std::path::PathBuf;
/// # use mwa_hyperdrive::glob::*;
/// # fn main() -> Result<(), GlobError> {
/// let glob = "Cargo*";
/// let matches: Vec<PathBuf> = get_all_matches_from_glob(&glob)?;
/// assert!(&matches.contains(&PathBuf::from("Cargo.toml")));
/// # Ok(())
/// # }
/// ```
pub fn get_all_matches_from_glob(g: &str) -> Result<Vec<PathBuf>, GlobError> {
    let mut entries = vec![];
    for entry in glob(g)? {
        match entry {
            Ok(e) => entries.push(e),
            Err(e) => return Err(GlobError::GlobError(e)),
        }
    }
    Ok(entries)
}

/// The same as `get_all_matches_from_glob`, but only a single result is
/// expected to be returned from the glob match. If there are no results, or
/// more than one, an error is returned.
///
/// # Examples
///
/// ```
/// # use std::path::PathBuf;
/// # use mwa_hyperdrive::glob::*;
/// # fn main() -> Result<(), GlobError> {
/// let glob = "Cargo.t*l";
/// assert_eq!(get_single_match_from_glob(&glob)?, PathBuf::from("Cargo.toml"));
///
/// let glob = "Cargo.t??l";
/// assert_eq!(get_single_match_from_glob(&glob)?, PathBuf::from("Cargo.toml"));
///
/// let glob = "../Cargo*";
/// // Matches "Cargo.lock" and "Cargo.toml".
/// assert!(get_single_match_from_glob(&glob).is_err());
/// # Ok(())
/// # }
/// ```
pub fn get_single_match_from_glob(g: &str) -> Result<PathBuf, GlobError> {
    let entries = get_all_matches_from_glob(g)?;
    match entries.as_slice() {
        [] => Err(GlobError::NoMatches {
            glob: g.to_string(),
        }),
        [e] => Ok(e.clone()),
        _ => Err(GlobError::MoreThanOneMatch {
            glob: g.to_string(),
        }),
    }
}

#[derive(Error, Debug)]
/// Error type associated with glob helper functions.
pub enum GlobError {
    #[error("No glob matches were found for {glob}")]
    NoMatches { glob: String },

    #[error("More than one glob matches were found for {glob}; we require only one match")]
    MoreThanOneMatch { glob: String },

    #[error("{0}")]
    GlobError(#[from] glob::GlobError),

    #[error("{0}")]
    PatternError(#[from] glob::PatternError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glob_cargo() {
        let result = get_all_matches_from_glob("./Cargo*");
        assert!(result.is_ok());
        let entries = result.unwrap();
        assert!(&entries.contains(&PathBuf::from("Cargo.lock")));
        assert!(&entries.contains(&PathBuf::from("Cargo.toml")));
    }

    #[test]
    fn glob_cargo_toml() {
        let result = get_single_match_from_glob("./Cargo*");
        assert!(result.is_err());

        let result = get_single_match_from_glob("core/Cargo*");
        assert!(result.is_ok());
        let entry = result.unwrap();
        assert_eq!(entry, PathBuf::from("core/Cargo.toml"));
    }
}
