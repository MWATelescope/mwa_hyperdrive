// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! The main hyperdrive binary.

use std::path::PathBuf;

use log::{debug, info, trace};
use structopt::{clap::AppSettings, StructOpt};

use mwa_hyperdrive::{
    calibrate::{args::CalibrateUserArgs, di_calibrate},
    simulate_vis::{simulate_vis, SimulateVisArgs},
    HyperdriveError,
};
use mwa_hyperdrive_common::{display_build_info, setup_logging};
use mwa_hyperdrive_srclist::utilities::*;

#[derive(StructOpt)]
#[structopt(name = "hyperdrive", about,
            author = env!("CARGO_PKG_HOMEPAGE"),
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
        #[cfg(any(feature = "cuda-double", feature = "cuda-single"))]
        #[structopt(short, long)]
        cpu: bool,
    },

    SrclistByBeam {
        #[structopt(flatten)]
        args: ByBeamArgs,
    },

    SrclistConvert {
        #[structopt(flatten)]
        args: ConvertArgs,
    },

    SrclistShift {
        #[structopt(flatten)]
        args: ShiftArgs,
    },

    SrclistVerify {
        #[structopt(flatten)]
        args: VerifyArgs,
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
    let verbosity = match &args {
        Args::DiCalibrate { verbosity, .. } => verbosity,
        Args::SimulateVis { verbosity, .. } => verbosity,
        Args::SrclistByBeam { args, .. } => &args.verbosity,
        Args::SrclistConvert { args, .. } => &args.verbosity,
        Args::SrclistShift { args, .. } => &args.verbosity,
        Args::SrclistVerify { args, .. } => &args.verbosity,
    };
    setup_logging(*verbosity).expect("Failed to initialise logging.");

    // Print the version of hyperdrive and its build-time information.
    info!(
        "hyperdrive {} {}",
        match args {
            Args::DiCalibrate { .. } => "di-calibrate",
            Args::SimulateVis { .. } => "simulate-vis",
            Args::SrclistByBeam { .. } => "srclist-by-beam",
            Args::SrclistConvert { .. } => "srclist-convert",
            Args::SrclistShift { .. } => "srclist-shift",
            Args::SrclistVerify { .. } => "srclist-verify",
        },
        env!("CARGO_PKG_VERSION")
    );
    display_build_info();

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

            di_calibrate(&parameters)?;

            info!("hyperdrive di-calibrate complete.");
        }

        Args::SimulateVis {
            args,
            verbosity: _,
            dry_run,
            #[cfg(any(feature = "cuda-double", feature = "cuda-single"))]
            cpu,
        } => {
            #[cfg(not(any(feature = "cuda-double", feature = "cuda-single")))]
            simulate_vis(args, dry_run)?;

            #[cfg(any(feature = "cuda-double", feature = "cuda-single"))]
            simulate_vis(args, cpu, dry_run)?;

            info!("hyperdrive simulate-vis complete.");
        }

        // Source list utilities.
        Args::SrclistByBeam { args } => args.run()?,
        Args::SrclistConvert { args } => args.run()?,
        Args::SrclistShift { args } => args.run()?,
        Args::SrclistVerify { args } => args.run()?,
    }

    Ok(())
}
