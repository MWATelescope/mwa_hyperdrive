// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

mod common;
mod simulate_vis;

use common::*;
use simulate_vis::*;

use std::path::PathBuf;

use anyhow::bail;
use log::{debug, info};
use structopt::{clap::AppSettings, StructOpt};

use mwa_hyperdrive::*;

#[derive(StructOpt, Debug)]
#[structopt(author, name = "hyperdrive", version = HYPERDRIVE_VERSION.as_str(), about, global_settings = &[AppSettings::ColoredHelp, AppSettings::ArgRequiredElseHelp])]
enum Args {
    /// Calibrate the input data. WIP.
    Calibrate {
        // Share the arguments that could be passed in via a parameter file.
        #[structopt(flatten)]
        cli_args: mwa_hyperdrive::calibrate::args::CalibrateUserArgs,

        /// All of the arguments to calibrate may be specified in a toml or json
        /// file. Any CLI arguments override parameters set in the file.
        #[structopt(name = "ARGUMENTS_FILE", parse(from_os_str))]
        args_file: Option<PathBuf>,

        /// The verbosity of the program. The default is to print high-level
        /// information.
        #[structopt(short, long, parse(from_occurrences))]
        verbosity: u8,

        /// Don't actually do calibration; just verify that data was correctly
        /// ingested and print out high-level information.
        #[structopt(short = "n", long)]
        dry_run: bool,
    },

    /// Simulate visibilities of a source list like WODEN. Defaults are "CHIPS
    /// settings".
    SimulateVis {
        // Share the arguments that could be passed in via a parameter file.
        #[structopt(flatten)]
        cli_args: SimulateVisArgs,

        /// All of the arguments to simulate-vis may be specified in a toml or
        /// json file. Any CLI arguments override parameters set in the file.
        #[structopt(name = "PARAMETER_FILE", parse(from_os_str))]
        param_file: Option<PathBuf>,

        /// Use the CPU for visibility generation.
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

fn main() -> Result<(), anyhow::Error> {
    // Set up logging.
    let args = Args::from_args();
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
            use mwa_hyperdrive::calibrate::*;

            debug!("Merging command-line arguments with the argument file");
            let args = cli_args.merge(args_file)?;
            debug!("{:#?}", &args);
            debug!("Converting arguments into calibration parameters");
            let parameters = args.into_params()?;

            if dry_run {
                info!("Dry run -- exiting now.");
                return Ok(());
            }

            di_cal(&parameters)?;

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
            #[cfg(not(cuda))]
            if !cpu {
                bail!("Requested GPU processing, but the CUDA feature was not enabled when hyperdrive was compiled.")
            }
            simulate_vis(cli_args, param_file, cpu, write_to_text, dry_run)?
        }
    }

    Ok(())
}
