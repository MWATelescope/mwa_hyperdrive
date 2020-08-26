// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Handling for calibration arguments.
 */

use anyhow::bail;
use serde::{Deserialize, Serialize};
use structopt::StructOpt;

use super::CalibrateParams;
use crate::glob::*;
use crate::*;

/// Arguments that are exposed to users. These are digested by hyperdrive and
/// used to populate `CalibrateArgs`, which is used throughout hyperdrive.
#[derive(StructOpt, Debug, Default, Serialize, Deserialize)]
pub struct CalibrateUserArgs {
    /// Path to the metafits file.
    #[structopt(short, long)]
    pub metafits: Option<String>,

    /// Paths to gpubox files.
    #[structopt(short, long)]
    pub gpuboxes: Option<Vec<String>>,

    /// Paths to mwaf files.
    #[structopt(long)]
    pub mwafs: Option<Vec<String>>,
}

/// Both command-line and parameter-file arguments overlap in terms of what is
/// available; this function consolidates everything that was specified into a
/// single struct. Where applicable, it will prefer CLI parameters over those in
/// the file.
///
/// Also verify that files exist and parameters are sensible, etc.
pub fn merge_cli_and_file_args(
    cli_args: CalibrateUserArgs,
    param_file: Option<PathBuf>,
) -> Result<CalibrateParams, anyhow::Error> {
    // If available, read in the parameter file.
    let file_params: CalibrateUserArgs = if let Some(pf) = &param_file {
        debug!(
            "Found a parameter file {}; attempting to parse...",
            pf.display()
        );

        let mut contents = String::new();
        match pf.extension().and_then(|e| e.to_str()) {
            Some("toml") => {
                debug!("Parsing toml file...");
                let mut fh = File::open(&pf)?;
                fh.read_to_string(&mut contents)?;
                match toml::from_str(&contents) {
                    Ok(p) => p,
                    Err(e) => bail!(
                        "Couldn't decode toml structure from {}:\n{}",
                        pf.display(),
                        e
                    ),
                }
            }
            Some("json") => {
                debug!("Parsing json file...");
                let mut fh = File::open(&pf)?;
                fh.read_to_string(&mut contents)?;
                match serde_json::from_str(&contents) {
                    Ok(p) => p,
                    Err(e) => bail!(
                        "Couldn't decode json structure from {}:\n{}",
                        pf.display(),
                        e
                    ),
                }
            }
            _ => bail!(
                "Parameter file {} doesn't have a recognised file extension!
Valid extensions are .toml and .json",
                pf.display()
            ),
        }
    } else {
        std::default::Default::default()
    };

    let metafits = match (cli_args.metafits, file_params.metafits) {
        // Use the metafits from the CLI, if available.
        (Some(m), _) => m,
        // There is a metafits in the parameter file; use it.
        (None, Some(m)) => m,
        // No metafits in the parameter file, we need to bail.
        (None, None) => bail!("No metafits file supplied"),
    };
    // If the specified metafits file can't be found, treat it as a glob and
    // expand it to find a match.
    let mut metafits_pb = PathBuf::from(&metafits);
    if !metafits_pb.exists() {
        metafits_pb = get_single_match_from_glob(&metafits)?;
    }

    let gpuboxes: Vec<PathBuf> = {
        let gpuboxes: Vec<String> = match (cli_args.gpuboxes, file_params.gpuboxes) {
            (Some(m), _) => m,
            (None, Some(m)) => m,
            (None, None) => bail!("No gpubox files supplied"),
        };
        if gpuboxes.is_empty() {
            bail!("No gpubox files supplied");
        }
        // If a single gpubox file was specified, and it isn't a real file, treat it
        // as a glob and expand it to find matches.
        match gpuboxes.len() {
            1 => {
                let pb = PathBuf::from(&gpuboxes[0]);
                if pb.exists() {
                    vec![pb]
                } else {
                    let entries = get_all_matches_from_glob(&gpuboxes[0])?;
                    if entries.is_empty() {
                        bail!("The lone gpubox entry is neither a file nor a glob pattern that matched any files");
                    } else {
                        entries
                    }
                }
            }
            _ => gpuboxes.into_iter().map(PathBuf::from).collect(),
        }
    };

    let mwafs: Option<Vec<PathBuf>> = {
        let mwafs = match (cli_args.mwafs, file_params.mwafs) {
            (Some(m), _) => Some(m),
            (None, Some(m)) => Some(m),
            // mwaf files are optional.
            (None, None) => None,
        };
        // If a single mwaf file was specified, and it isn't a real file, treat it
        // as a glob and expand it to find matches.
        match mwafs {
            None => None,
            Some(ms) => match ms.len() {
                1 => {
                    let pb = PathBuf::from(&ms[0]);
                    if pb.exists() {
                        Some(vec![pb])
                    } else {
                        let entries = get_all_matches_from_glob(&ms[0])?;
                        if entries.is_empty() {
                            bail!("The lone mwaf entry is neither a file nor a glob pattern that matched any files");
                        } else {
                            Some(entries)
                        }
                    }
                }
                _ => Some(ms.into_iter().map(PathBuf::from).collect()),
            },
        }
    };

    Ok(CalibrateParams {
        metafits: metafits_pb,
        gpuboxes,
        mwafs,
    })
}
