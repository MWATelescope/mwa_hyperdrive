// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! The main hyperdrive binary.

mod common;
use common::*;

use std::path::PathBuf;

use log::{debug, info, trace};
use structopt::{clap::AppSettings, StructOpt};

use mwa_hyperdrive::{
    calibrate::{args::*, calibrate},
    simulate_vis::{simulate_vis, SimulateVisArgs},
    HyperdriveError,
};
use mwa_hyperdrive_srclist::{
    CONVERT_INPUT_TYPE_HELP, CONVERT_OUTPUT_TYPE_HELP, SOURCE_DIST_CUTOFF_HELP, VETO_THRESHOLD_HELP,
};

#[derive(StructOpt)]
#[structopt(name = "hyperdrive", author, about,
            version = HYPERDRIVE_VERSION.as_str(),
            global_settings = &[AppSettings::ColoredHelp,
                                AppSettings::ArgRequiredElseHelp,
                                AppSettings::DeriveDisplayOrder])]
enum Args {
    /// Perform direction-independent calibration on the input MWA data.
    ///
    /// See for more info:
    /// https://github.com/MWATelescope/mwa_hyperdrive/wiki/Calibration-usage
    #[structopt(alias = "calibrate")]
    DiCalibrate {
        // Share the arguments that could be passed in via a parameter file.
        #[structopt(flatten)]
        cli_args: CalibrateUserArgs,

        /// All of the arguments to calibrate may be specified in a toml or json
        /// file. Any CLI arguments override parameters set in the file.
        #[structopt(name = "ARGUMENTS_FILE", parse(from_os_str))]
        args_file: Option<PathBuf>,

        /// The verbosity of the program. Increase by specifying multiple times
        /// (e.g. -vv). The default is to print only high-level information.
        #[structopt(short, long, parse(from_occurrences))]
        verbosity: u8,

        /// Don't actually do calibration; just verify that arguments were
        /// correctly ingested and print out high-level information.
        #[structopt(long)]
        dry_run: bool,
    },

    /// Simulate visibilities of a sky-model source list.
    SimulateVis {
        #[structopt(flatten)]
        args: SimulateVisArgs,

        /// The verbosity of the program. The default is to print high-level
        /// information.
        #[structopt(short, long, parse(from_occurrences))]
        verbosity: u8,

        /// Don't actually do any work; just verify that the input arguments
        /// were correctly ingested and print out high-level information.
        #[structopt(long)]
        dry_run: bool,

        /// Use the CPU for visibility generation. This is deliberately made
        /// non-default because using a GPU is much faster.
        #[cfg(feature = "cuda")]
        #[structopt(short, long)]
        cpu: bool,
    },

    /// Reduce a sky-model source list to the top N brightest sources, given
    /// pointing information.
    ///
    /// See for more info:
    /// https://github.com/MWATelescope/mwa_hyperdrive/wiki/Source-lists
    SrclistByBeam {
        /// Path to the source list to be converted.
        #[structopt(name = "INPUT_SOURCE_LIST", parse(from_os_str))]
        input_source_list: PathBuf,

        /// Path to the output source list. If not specified, then then "_N" is
        /// appended to the filename.
        #[structopt(name = "OUTPUT_SOURCE_LIST", parse(from_os_str))]
        output_source_list: Option<PathBuf>,

        #[structopt(short = "i", long, parse(from_str), help = CONVERT_INPUT_TYPE_HELP.as_str())]
        input_type: Option<String>,

        #[structopt(short = "o", long, parse(from_str), help = CONVERT_OUTPUT_TYPE_HELP.as_str())]
        output_type: Option<String>,

        /// Reduce the input source list to the brightest N sources and write
        /// them to the output source list. If the input source list has less
        /// than N sources, then all sources are used.
        #[structopt(short = "n", long)]
        number: usize,

        /// Path to the metafits file. If this isn't supplied, then the dipole
        /// delays must be supplied.
        #[structopt(short = "m", long, parse(from_str))]
        metafits: PathBuf,

        #[structopt(long, help = SOURCE_DIST_CUTOFF_HELP.as_str())]
        source_dist_cutoff: Option<f64>,

        #[structopt(long, help = VETO_THRESHOLD_HELP.as_str())]
        veto_threshold: Option<f64>,

        /// Path to the MWA FEE beam file. If this is not specified, then the
        /// MWA_BEAM_FILE environment variable should contain the path.
        #[structopt(short, long)]
        beam_file: Option<PathBuf>,

        /// Attempt to convert flux density lists to power laws. See the online
        /// help for more information.
        #[structopt(short, long)]
        convert_lists: bool,

        /// Collapse all of the sky-model components into a single source; the
        /// apparently brightest source is used as the base source. This is
        /// suitable for an "RTS patch source list".
        #[structopt(long)]
        collapse_into_single_source: bool,

        /// The verbosity of the program. The default is to print high-level
        /// information.
        #[structopt(short, long, parse(from_occurrences))]
        verbosity: u8,
    },

    /// Convert a sky-model source list from one format to another.
    ///
    /// See for more info:
    /// https://github.com/MWATelescope/mwa_hyperdrive/wiki/Source-lists
    SrclistConvert {
        #[structopt(short = "i", long, parse(from_str), help = CONVERT_INPUT_TYPE_HELP.as_str())]
        input_type: Option<String>,

        /// Path to the source list to be converted.
        #[structopt(name = "INPUT_SOURCE_LIST", parse(from_os_str))]
        input_source_list: PathBuf,

        #[structopt(short = "o", long, parse(from_str), help = CONVERT_OUTPUT_TYPE_HELP.as_str())]
        output_type: Option<String>,

        /// Path to the output source list.
        #[structopt(name = "OUTPUT_SOURCE_LIST", parse(from_os_str))]
        output_source_list: PathBuf,

        /// Attempt to convert flux density lists to power laws. See the online
        /// help for more information.
        #[structopt(short, long)]
        convert_lists: bool,

        /// The verbosity of the program. The default is to print high-level
        /// information.
        #[structopt(short, long, parse(from_occurrences))]
        verbosity: u8,
    },

    /// Verify that sky-model source lists can be read by hyperdrive.
    ///
    /// See for more info:
    /// https://github.com/MWATelescope/mwa_hyperdrive/wiki/Source-lists
    SrclistVerify {
        /// Path to the source list(s) to be verified.
        #[structopt(name = "SOURCE_LISTS", parse(from_os_str))]
        source_lists: Vec<PathBuf>,

        /// The verbosity of the program. The default is to print high-level
        /// information.
        #[structopt(short, long, parse(from_occurrences))]
        verbosity: u8,
    },
}

fn main() {
    // Stolen from BurntSushi. We don't return Result from main because it
    // prints the debug representation of the error. The code below prints the
    // "display" or human readable representation of the error.
    if let Err(e) = try_main() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn try_main() -> Result<(), HyperdriveError> {
    // Get the command-line arguments.
    let args = Args::from_args();

    // Set up logging.
    let verbosity = match args {
        Args::DiCalibrate { verbosity, .. } => verbosity,
        Args::SimulateVis { verbosity, .. } => verbosity,
        Args::SrclistByBeam { verbosity, .. } => verbosity,
        Args::SrclistConvert { verbosity, .. } => verbosity,
        Args::SrclistVerify { verbosity, .. } => verbosity,
    };
    setup_logging(verbosity).expect("Failed to initialise logging.");

    // Print the version of hyperdrive and its build-time information.
    info!(
        "hyperdrive {}",
        match args {
            Args::DiCalibrate { .. } => "di-calibrate",
            Args::SimulateVis { .. } => "simulate-vis",
            Args::SrclistByBeam { .. } => "srclist-by-beam",
            Args::SrclistConvert { .. } => "srclist-convert",
            Args::SrclistVerify { .. } => "srclist-verify",
        }
    );
    info!("Version {}", env!("CARGO_PKG_VERSION"));
    match GIT_HEAD_REF {
        Some(hr) => {
            let dirty = GIT_DIRTY.unwrap_or(false);
            info!(
                "Compiled on git commit hash: {}{}",
                GIT_COMMIT_HASH.unwrap(),
                if dirty { " (dirty)" } else { "" }
            );
            info!("            git head ref: {}", hr);
        }
        None => info!("Compiled on git commit hash: <no git info>"),
    }
    info!("            {}", BUILT_TIME_UTC);
    info!("         with compiler {}", RUSTC_VERSION);
    info!("");

    match Args::from_args() {
        Args::DiCalibrate {
            cli_args,
            args_file,
            verbosity: _,
            dry_run,
        } => {
            let args = if let Some(f) = args_file {
                trace!("Merging command-line arguments with the argument file");
                cli_args.merge(&f)?
            } else {
                cli_args
            };
            debug!("{:#?}", &args);
            trace!("Converting arguments into calibration parameters");
            let parameters = args.into_params()?;

            if dry_run {
                info!("Dry run -- exiting now.");
                return Ok(());
            }

            calibrate(parameters)?;

            info!("hyperdrive calibrate complete.");
        }

        Args::SimulateVis {
            args,
            verbosity: _,
            dry_run,
            #[cfg(feature = "cuda")]
            cpu,
        } => {
            #[cfg(not(feature = "cuda"))]
            simulate_vis(args, dry_run)?;

            #[cfg(feature = "cuda")]
            simulate_vis(args, cpu, dry_run)?;

            info!("hyperdrive simulate-vis complete.");
        }

        Args::SrclistByBeam {
            input_source_list,
            output_source_list,
            input_type,
            output_type,
            number,
            metafits,
            source_dist_cutoff,
            veto_threshold,
            beam_file,
            convert_lists,
            collapse_into_single_source,
            verbosity: _,
        } => mwa_hyperdrive_srclist::by_beam(
            &input_source_list,
            output_source_list.as_ref(),
            input_type,
            output_type,
            number,
            &metafits,
            source_dist_cutoff,
            veto_threshold,
            beam_file.as_ref(),
            convert_lists,
            collapse_into_single_source,
        )?,

        Args::SrclistConvert {
            input_source_list,
            output_source_list,
            input_type,
            output_type,
            convert_lists,
            verbosity: _,
        } => mwa_hyperdrive_srclist::convert(
            &input_source_list,
            &output_source_list,
            input_type,
            output_type,
            convert_lists,
        )?,

        Args::SrclistVerify {
            source_lists,
            verbosity: _,
        } => {
            mwa_hyperdrive_srclist::verify(&source_lists)?;
        }
    }

    Ok(())
}
