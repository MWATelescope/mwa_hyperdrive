// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! The main hyperdrive binary.

use std::path::PathBuf;

use clap::{AppSettings, Parser};
use log::info;

use mwa_hyperdrive::{
    calibrate::{args::CalibrateUserArgs, di_calibrate},
    solutions::{apply::SolutionsApplyArgs, convert::SolutionsConvertArgs, SolutionsPlotArgs},
    vis_utils::{simulate::VisSimulateArgs, subtract::VisSubtractArgs},
    HyperdriveError,
};
use mwa_hyperdrive_common::{clap, display_build_info, log, setup_logging};
use mwa_hyperdrive_srclist::utilities::*;

#[derive(Parser)]
#[clap(
    version,
    author,
    about = r#"Calibration software for the Murchison Widefield Array (MWA) radio telescope
Documentation: https://mwatelescope.github.io/mwa_hyperdrive
Source:        https://github.com/MWATelescope/mwa_hyperdrive"#
)]
#[clap(global_setting(AppSettings::DeriveDisplayOrder))]
#[clap(disable_help_subcommand = true)]
#[clap(infer_subcommands = true)]
#[clap(propagate_version = true)]
enum Args {
    #[clap(alias = "calibrate")]
    #[clap(
        about = r#"Perform direction-independent calibration on the input MWA data.
https://mwatelescope.github.io/mwa_hyperdrive/user/di_cal/intro.html"#
    )]
    #[clap(arg_required_else_help = true)]
    #[clap(infer_long_args = true)]
    DiCalibrate {
        // Share the arguments that could be passed in via a parameter file.
        #[clap(flatten)]
        cli_args: Box<CalibrateUserArgs>,

        /// All of the arguments to di-calibrate may be specified in a toml or
        /// json file. Any CLI arguments override parameters set in the file.
        #[clap(name = "ARGUMENTS_FILE", parse(from_os_str))]
        args_file: Option<PathBuf>,

        /// The verbosity of the program. Increase by specifying multiple times
        /// (e.g. -vv). The default is to print only high-level information.
        #[clap(short, long, parse(from_occurrences))]
        verbosity: u8,

        /// Don't actually do calibration; just verify that arguments were
        /// correctly ingested and print out high-level information.
        #[clap(long)]
        dry_run: bool,
    },

    #[clap(alias = "simulate-vis")]
    #[clap(about = r#"Simulate visibilities of a sky-model source list.
https://mwatelescope.github.io/mwa_hyperdrive/user/vis_simulate/intro.html"#)]
    #[clap(arg_required_else_help = true)]
    #[clap(infer_long_args = true)]
    VisSimulate {
        #[clap(flatten)]
        args: VisSimulateArgs,

        /// The verbosity of the program. The default is to print high-level
        /// information.
        #[clap(short, long, parse(from_occurrences))]
        verbosity: u8,

        /// Don't actually do any work; just verify that the input arguments
        /// were correctly ingested and print out high-level information.
        #[clap(long)]
        dry_run: bool,
    },

    #[clap(alias = "subtract-vis")]
    #[clap(about = "Subtract sky-model sources from supplied visibilities.
https://mwatelescope.github.io/mwa_hyperdrive/user/vis_subtract/intro.html")]
    #[clap(arg_required_else_help = true)]
    #[clap(infer_long_args = true)]
    VisSubtract {
        #[clap(flatten)]
        args: VisSubtractArgs,

        /// The verbosity of the program. The default is to print high-level
        /// information.
        #[clap(short, long, parse(from_occurrences))]
        verbosity: u8,

        /// Don't actually do any work; just verify that the input arguments
        /// were correctly ingested and print out high-level information.
        #[clap(long)]
        dry_run: bool,
    },

    #[clap(alias = "apply-solutions")]
    #[clap(about = r#"Apply calibration solutions to input data.
https://mwatelescope.github.io/mwa_hyperdrive/user/solutions_apply/intro.html"#)]
    #[clap(arg_required_else_help = true)]
    #[clap(infer_long_args = true)]
    SolutionsApply {
        #[clap(flatten)]
        args: SolutionsApplyArgs,

        /// The verbosity of the program. The default is to print high-level
        /// information.
        #[clap(short, long, parse(from_occurrences))]
        verbosity: u8,

        /// Don't actually do any work; just verify that the input arguments
        /// were correctly ingested and print out high-level information.
        #[clap(long)]
        dry_run: bool,
    },

    #[clap(alias = "convert-solutions")]
    #[clap(about = "Convert between calibration solution file formats.")]
    #[clap(arg_required_else_help = true)]
    #[clap(infer_long_args = true)]
    SolutionsConvert {
        #[clap(flatten)]
        args: SolutionsConvertArgs,

        /// The verbosity of the program. Increase by specifying multiple times
        /// (e.g. -vv). The default is to print only high-level information.
        #[clap(short, long, parse(from_occurrences))]
        verbosity: u8,
    },

    #[clap(alias = "plot-solutions")]
    #[clap(
        about = "Plot calibration solutions. Only available if compiled with the \"plotting\" feature."
    )]
    #[clap(arg_required_else_help = true)]
    #[clap(infer_long_args = true)]
    SolutionsPlot {
        #[clap(flatten)]
        args: SolutionsPlotArgs,

        /// The verbosity of the program. Increase by specifying multiple times
        /// (e.g. -vv). The default is to print only high-level information.
        #[clap(short, long, parse(from_occurrences))]
        verbosity: u8,
    },

    #[clap(arg_required_else_help = true)]
    #[clap(infer_long_args = true)]
    SrclistByBeam {
        #[clap(flatten)]
        args: ByBeamArgs,
    },

    #[clap(arg_required_else_help = true)]
    #[clap(infer_long_args = true)]
    SrclistConvert {
        #[clap(flatten)]
        args: ConvertArgs,
    },

    #[clap(arg_required_else_help = true)]
    #[clap(infer_long_args = true)]
    SrclistShift {
        #[clap(flatten)]
        args: ShiftArgs,
    },

    #[clap(arg_required_else_help = true)]
    #[clap(infer_long_args = true)]
    SrclistVerify {
        #[clap(flatten)]
        args: VerifyArgs,
    },

    #[clap(arg_required_else_help = true)]
    #[clap(infer_long_args = true)]
    DipoleGains {
        #[clap(flatten)]
        args: mwa_hyperdrive::utilities::dipole_gains::DipoleGains,
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
    let args = Args::parse();

    // Set up logging.
    let (verbosity, sub_command) = match &args {
        Args::DiCalibrate { verbosity, .. } => (verbosity, "di-calibrate"),
        Args::VisSimulate { verbosity, .. } => (verbosity, "vis-simulate"),
        Args::VisSubtract { verbosity, .. } => (verbosity, "vis-subtract"),
        Args::SolutionsApply { verbosity, .. } => (verbosity, "solutions-apply"),
        Args::SolutionsConvert { verbosity, .. } => (verbosity, "solutions-convert"),
        Args::SolutionsPlot { verbosity, .. } => (verbosity, "solutions-plot"),
        Args::SrclistByBeam { args, .. } => (&args.verbosity, "srclist-by-beam"),
        Args::SrclistConvert { args, .. } => (&args.verbosity, "srclist-convert"),
        Args::SrclistShift { args, .. } => (&args.verbosity, "srclist-shift"),
        Args::SrclistVerify { args, .. } => (&args.verbosity, "srclist-verify"),
        Args::DipoleGains { .. } => (&0, "dipole-gains"),
    };
    setup_logging(*verbosity).expect("Failed to initialise logging.");

    // Print the version of hyperdrive and its build-time information.
    info!("hyperdrive {} {}", sub_command, env!("CARGO_PKG_VERSION"));
    display_build_info();

    match Args::parse() {
        Args::DiCalibrate {
            cli_args,
            args_file,
            verbosity: _,
            dry_run,
        } => {
            di_calibrate(cli_args, args_file.as_deref(), dry_run)?;
        }

        Args::VisSimulate {
            args,
            verbosity: _,
            dry_run,
        } => {
            args.run(dry_run)?;
        }

        Args::VisSubtract {
            args,
            verbosity: _,
            dry_run,
        } => {
            args.run(dry_run)?;
        }

        Args::SolutionsApply {
            args,
            verbosity: _,
            dry_run,
        } => {
            args.run(dry_run)?;
        }

        Args::SolutionsConvert { args, verbosity: _ } => {
            args.run()?;
        }

        Args::SolutionsPlot { args, verbosity: _ } => {
            args.run()?;
        }

        // Source list utilities.
        Args::SrclistByBeam { args } => args.run()?,
        Args::SrclistConvert { args } => args.run()?,
        Args::SrclistShift { args } => args.run()?,
        Args::SrclistVerify { args } => args.run()?,

        // Misc. utilities.
        Args::DipoleGains { args } => args.run().unwrap(),
    }

    info!("hyperdrive {} complete.", sub_command);
    Ok(())
}
