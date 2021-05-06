// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

mod common;
use common::*;

use std::path::PathBuf;

use log::{debug, info};
use structopt::{clap::AppSettings, StructOpt};

use mwa_hyperdrive::{calibrate::calibrate, simulate_vis::simulate_vis, *};

#[derive(StructOpt)]
#[structopt(name = "hyperdrive", author, about,
            version = HYPERDRIVE_VERSION.as_str(),
            global_settings = &[AppSettings::ColoredHelp,
                                AppSettings::ArgRequiredElseHelp,
                                AppSettings::DeriveDisplayOrder])]
enum Args {
    /// Perform direction-independent calibration on the input MWA data.
    Calibrate {
        // Share the arguments that could be passed in via a parameter file.
        #[structopt(flatten)]
        cli_args: mwa_hyperdrive::calibrate::args::CalibrateUserArgs,

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
        #[structopt(short = "n", long)]
        dry_run: bool,
    },

    /// Simulate visibilities of a source list like WODEN. Defaults are "CHIPS
    /// settings".
    SimulateVis {
        // Share the arguments that could be passed in via a parameter file.
        #[structopt(flatten)]
        cli_args: mwa_hyperdrive::simulate_vis::SimulateVisArgs,

        /// All of the arguments to simulate-vis may be specified in a toml or
        /// json file. Any CLI arguments override parameters set in the file.
        #[structopt(name = "PARAMETER_FILE", parse(from_os_str))]
        param_file: Option<PathBuf>,

        /// Use the CPU for visibility generation. This is deliberately made
        /// non-default, because using a GPU is much faster.
        #[structopt(short, long)]
        cpu: bool,

        /// Also write visibility data to text files.
        #[structopt(long = "text")]
        write_to_text: bool,

        /// The verbosity of the program. The default is to print high-level
        /// information.
        #[structopt(short, long, parse(from_occurrences))]
        verbosity: u8,

        /// Don't actually do any work; just verify that the input arguments
        /// were correctly ingested and print out high-level information.
        #[structopt(short = "n", long)]
        dry_run: bool,
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
        Args::Calibrate { verbosity, .. } => verbosity,
        Args::SimulateVis { verbosity, .. } => verbosity,
    };
    setup_logging(verbosity).expect("Failed to initialise logging.");

    // Print the version of hyperdrive and its build-time information.
    info!(
        "hyperdrive {}",
        match args {
            Args::Calibrate { .. } => "calibrate",
            Args::SimulateVis { .. } => "simulate-vis",
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
        None => info!("<no git info>"),
    }
    info!("            {}", BUILT_TIME_UTC);
    info!("         with compiler {}", RUSTC_VERSION);
    info!("");

    match Args::from_args() {
        Args::Calibrate {
            cli_args,
            args_file,
            verbosity: _,
            dry_run,
        } => {
            let args = if let Some(f) = args_file {
                debug!("Merging command-line arguments with the argument file");
                cli_args.merge(&f)?
            } else {
                cli_args
            };
            debug!("{:#?}", &args);
            debug!("Converting arguments into calibration parameters");
            let parameters = args.into_params()?;

            if dry_run {
                info!("Dry run -- exiting now.");
                return Ok(());
            }

            calibrate(parameters)?;

            info!("hyperdrive calibrate complete.");
        }

        Args::SimulateVis {
            cli_args,
            param_file,
            cpu,
            write_to_text,
            verbosity: _,
            dry_run,
        } => {
            // Handle requesting CUDA if CUDA isn't available.
            #[cfg(not(feature = "cuda"))]
            if !cpu {
                return Err(HyperdriveError::NoGpuCompiled);
            }

            simulate_vis(cli_args, param_file, cpu, write_to_text, dry_run)?
        }
    }

    Ok(())
}
