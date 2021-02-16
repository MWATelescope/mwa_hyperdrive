// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Handling of calibration arguments.

Strategy: Users give arguments to hyperdrive (handled by calibrate::args).
hyperdrive turns arguments into parameters (handled by calibrate::params). Using
this terminology, the code to handle arguments and parameters (and associated
errors) can be neatly split.
 */

use std::fs::File;
use std::io::Read;

use serde::{Deserialize, Serialize};
use structopt::StructOpt;
use thiserror::Error;

use crate::calibrate::params::{CalibrateParams, InvalidArgsError};
use crate::*;

lazy_static::lazy_static! {
    static ref SOURCE_DIST_CUTOFF_HELP: String =
        format!("The sky-model source cutoff distance (degrees). This is only used if the input sky-model source list has more sources than specified by num_sources. Default: {}", CUTOFF_DISTANCE);

    static ref VETO_THRESHOLD_HELP: String =
        format!("The smallest possible beam-attenuated flux density any sky-model source is allowed to have. Default: {}", DEFAULT_VETO_THRESHOLD);

    static ref SOURCE_LIST_TYPE_HELP: String =
        format!(r#"The type of sky-model source list. Valid types are: {}

If not specified, the program will assume .txt files are RTS type source lists"#, *mwa_hyperdrive_srclist::SOURCE_LIST_FILE_TYPES_COMMA_SEPARATED);
}

/// Arguments that are exposed to users. All arguments should be optional.
///
/// These are digested by hyperdrive and used to eventually populate
/// `CalibrateParams`, which is used throughout hyperdrive.
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

    /// Path to the sky-model source list file.
    #[structopt(short, long)]
    pub source_list: Option<String>,

    #[structopt(long, help = SOURCE_LIST_TYPE_HELP.as_str())]
    pub source_list_type: Option<String>,

    /// The number of sources to use in the source list. The default is to use
    /// them all.
    ///
    /// Example: If 1000 sources are specified here, then the top 1000 sources
    /// are used (based on their flux densities after the beam attenuation)
    /// within the specified source distance cutoff.
    #[structopt(long)]
    pub num_sources: Option<usize>,

    #[structopt(long, help = SOURCE_DIST_CUTOFF_HELP.as_str())]
    pub source_dist_cutoff: Option<f64>,

    #[structopt(long, help = VETO_THRESHOLD_HELP.as_str())]
    pub veto_threshold: Option<f64>,

    /// The calibration time resolution. This must be a multiple of the
    /// observation's native time resolution. If not supplied, then the
    /// observation's native time resolution is used.
    #[structopt(short, long)]
    pub time_res: Option<f64>,

    /// The calibration fine-channel frequency resolution. This must be a
    /// multiple of the observation's native frequency resolution. If not
    /// supplied, then the observation's native frequency resolution is used.
    #[structopt(short, long)]
    pub freq_res: Option<f64>,

    /// The fine channels to be flagged in each coarse band. e.g. 0 1 16 30 31
    ///
    /// If this is not specified, it defaults to flagging the centre channel, as
    /// well as 80 kHz (or as close to this as possible) at the edges.
    #[structopt(long)]
    pub fine_chan_flags: Option<Vec<usize>>,
}

impl CalibrateUserArgs {
    /// Both command-line and file arguments overlap in terms of what is
    /// available; this function consolidates everything that was specified into
    /// a single struct. Where applicable, it will prefer CLI parameters over
    /// those in the file.
    ///
    /// The argument to this function is the `Path` to the arguments file.
    pub(crate) fn merge(self, arg_file: Option<PathBuf>) -> Result<Self, CalibrateArgsError> {
        // Make it abundantly clear that "self" should be considered the
        // command-line arguments.
        let cli_args = self;

        // If available, read in the parameter file.
        let file_args: Self = if let Some(pf) = &arg_file {
            debug!(
                "Found a argument file {}; attempting to parse...",
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
                        Err(e) => {
                            return Err(CalibrateArgsError::TomlDecode {
                                file: pf.display().to_string(),
                                err: e.to_string(),
                            })
                        }
                    }
                }

                Some("json") => {
                    debug!("Parsing json file...");
                    let mut fh = File::open(&pf)?;
                    fh.read_to_string(&mut contents)?;
                    match serde_json::from_str(&contents) {
                        Ok(p) => p,
                        Err(e) => {
                            return Err(CalibrateArgsError::JsonDecode {
                                file: pf.display().to_string(),
                                err: e.to_string(),
                            })
                        }
                    }
                }

                _ => {
                    return Err(CalibrateArgsError::UnrecognisedArgFileExt(
                        pf.display().to_string(),
                    ))
                }
            }
        } else {
            Self::default()
        };

        // Merge all the arguments, preferring the CLI args when available.
        Ok(Self {
            metafits: cli_args.metafits.or(file_args.metafits),
            gpuboxes: cli_args.gpuboxes.or(file_args.gpuboxes),
            mwafs: cli_args.mwafs.or(file_args.mwafs),
            source_list: cli_args.source_list.or(file_args.source_list),
            source_list_type: cli_args.source_list_type.or(file_args.source_list_type),
            num_sources: cli_args.num_sources.or(file_args.num_sources),
            source_dist_cutoff: cli_args.source_dist_cutoff.or(file_args.source_dist_cutoff),
            veto_threshold: cli_args.veto_threshold.or(file_args.veto_threshold),
            time_res: cli_args.time_res.or(file_args.time_res),
            freq_res: cli_args.freq_res.or(file_args.freq_res),
            fine_chan_flags: cli_args.fine_chan_flags.or(file_args.fine_chan_flags),
        })
    }

    pub fn to_params(self) -> Result<CalibrateParams, InvalidArgsError> {
        CalibrateParams::new_from_args(self)
    }
}

/// Errors associated with merging `CalibrateUserArgs` structs.
#[derive(Error, Debug)]
pub enum CalibrateArgsError {
    #[error("Argument file {0} doesn't have a recognised file extension! Valid extensions are .toml and .json")]
    UnrecognisedArgFileExt(String),

    #[error("Couldn't decode toml structure from {file}:\n{err}")]
    TomlDecode { file: String, err: String },

    #[error("Couldn't decode json structure from {file}:\n{err}")]
    JsonDecode { file: String, err: String },

    #[error("{0}")]
    IO(#[from] std::io::Error),
}
