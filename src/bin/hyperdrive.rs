// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! The main hyperdrive binary.

use clap::{AppSettings, Parser};
use log::info;

use mwa_hyperdrive::HyperdriveError;

// Add build-time information from the "built" crate.
include!(concat!(env!("OUT_DIR"), "/built.rs"));

fn main() {
    // Stolen from BurntSushi. We don't return Result from main because it
    // prints the debug representation of the error. The code below prints the
    // "display" or human readable representation of the error.
    if let Err(e) = cli() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

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
        cli_args: Box<mwa_hyperdrive::DiCalArgs>,

        /// The verbosity of the program. Increase by specifying multiple times
        /// (e.g. -vv). The default is to print only high-level information.
        #[clap(short, long, parse(from_occurrences))]
        verbosity: u8,

        /// Don't actually do calibration; just verify that arguments were
        /// correctly ingested and print out high-level information.
        #[clap(long)]
        dry_run: bool,
    },

    #[clap(about = r#"Peeling!"#)]
    #[clap(arg_required_else_help = true)]
    #[clap(infer_long_args = true)]
    Peel {
        // Share the arguments that could be passed in via a parameter file.
        #[clap(flatten)]
        cli_args: Box<mwa_hyperdrive::PeelArgs>,

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
        args: mwa_hyperdrive::VisSimulateArgs,

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
        args: mwa_hyperdrive::VisSubtractArgs,

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
        args: mwa_hyperdrive::SolutionsApplyArgs,

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
        args: mwa_hyperdrive::SolutionsConvertArgs,

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
        args: mwa_hyperdrive::SolutionsPlotArgs,

        /// The verbosity of the program. Increase by specifying multiple times
        /// (e.g. -vv). The default is to print only high-level information.
        #[clap(short, long, parse(from_occurrences))]
        verbosity: u8,
    },

    #[clap(arg_required_else_help = true)]
    #[clap(infer_long_args = true)]
    SrclistByBeam {
        #[clap(flatten)]
        args: mwa_hyperdrive::SrclistByBeamArgs,

        /// The verbosity of the program. The default is to print high-level
        /// information.
        #[clap(short, long, parse(from_occurrences))]
        verbosity: u8,
    },

    #[clap(arg_required_else_help = true)]
    #[clap(infer_long_args = true)]
    SrclistConvert {
        #[clap(flatten)]
        args: mwa_hyperdrive::SrclistConvertArgs,

        /// The verbosity of the program. The default is to print high-level
        /// information.
        #[clap(short, long, parse(from_occurrences))]
        verbosity: u8,
    },

    #[clap(arg_required_else_help = true)]
    #[clap(infer_long_args = true)]
    SrclistShift {
        #[clap(flatten)]
        args: mwa_hyperdrive::SrclistShiftArgs,

        /// The verbosity of the program. The default is to print high-level
        /// information.
        #[clap(short, long, parse(from_occurrences))]
        verbosity: u8,
    },

    #[clap(arg_required_else_help = true)]
    #[clap(infer_long_args = true)]
    SrclistVerify {
        #[clap(flatten)]
        args: mwa_hyperdrive::SrclistVerifyArgs,

        /// The verbosity of the program. The default is to print high-level
        /// information.
        #[clap(short, long, parse(from_occurrences))]
        verbosity: u8,
    },

    #[clap(arg_required_else_help = true)]
    #[clap(infer_long_args = true)]
    DipoleGains {
        #[clap(flatten)]
        args: mwa_hyperdrive::DipoleGainsArgs,

        /// The verbosity of the program. The default is to print high-level
        /// information.
        #[clap(short, long, parse(from_occurrences))]
        verbosity: u8,
    },
}

/// Run `hyperdrive`.
fn cli() -> Result<(), HyperdriveError> {
    // Get the command-line arguments.
    let args = Args::parse();

    // Set up logging.
    let (verbosity, sub_command) = match &args {
        Args::DiCalibrate { verbosity, .. } => (verbosity, "di-calibrate"),
        Args::Peel { verbosity, .. } => (verbosity, "peel"),
        Args::VisSimulate { verbosity, .. } => (verbosity, "vis-simulate"),
        Args::VisSubtract { verbosity, .. } => (verbosity, "vis-subtract"),
        Args::SolutionsApply { verbosity, .. } => (verbosity, "solutions-apply"),
        Args::SolutionsConvert { verbosity, .. } => (verbosity, "solutions-convert"),
        Args::SolutionsPlot { verbosity, .. } => (verbosity, "solutions-plot"),
        Args::SrclistByBeam { verbosity, .. } => (verbosity, "srclist-by-beam"),
        Args::SrclistConvert { verbosity, .. } => (verbosity, "srclist-convert"),
        Args::SrclistShift { verbosity, .. } => (verbosity, "srclist-shift"),
        Args::SrclistVerify { verbosity, .. } => (verbosity, "srclist-verify"),
        Args::DipoleGains { verbosity, .. } => (verbosity, "dipole-gains"),
    };
    setup_logging(*verbosity).expect("Failed to initialise logging.");

    // Print the version of hyperdrive and its build-time information.
    info!("hyperdrive {} {}", sub_command, env!("CARGO_PKG_VERSION"));
    display_build_info();

    match Args::parse() {
        Args::DiCalibrate {
            cli_args,
            verbosity: _,
            dry_run,
        } => {
            cli_args.run(dry_run)?;
        }

        Args::Peel {
            cli_args,
            verbosity: _,
            dry_run,
        } => {
            cli_args.run(dry_run)?;
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
        Args::SrclistByBeam { args, .. } => args.run()?,
        Args::SrclistConvert { args, .. } => args.run()?,
        Args::SrclistShift { args, .. } => args.run()?,
        Args::SrclistVerify { args, .. } => args.run()?,

        // Misc. utilities.
        Args::DipoleGains { args, .. } => args.run()?,
    }

    info!("hyperdrive {} complete.", sub_command);
    Ok(())
}

/// Activate a logger. All log messages are put onto `stdout`. `env_logger`
/// automatically only uses colours and fancy symbols if we're on a tty (e.g. a
/// terminal); piped output will be formatted sensibly. Source code lines are
/// displayed in log messages when verbosity >= 3.
fn setup_logging(verbosity: u8) -> Result<(), log::SetLoggerError> {
    let mut builder = env_logger::Builder::from_default_env();
    builder.target(env_logger::Target::Stdout);
    builder.format_target(false);
    match verbosity {
        0 => builder.filter_level(log::LevelFilter::Info),
        1 => builder.filter_level(log::LevelFilter::Debug),
        2 => builder.filter_level(log::LevelFilter::Trace),
        _ => {
            builder.filter_level(log::LevelFilter::Trace);
            builder.format(|buf, record| {
                use std::io::Write;

                // TODO: Add colours.
                let timestamp = buf.timestamp();
                let level = record.level();
                let target = record.target();
                let line = record.line().unwrap_or(0);
                let message = record.args();

                writeln!(buf, "[{timestamp} {level} {target}:{line}] {message}")
            })
        }
    };
    builder.init();

    Ok(())
}

/// Write many info-level log lines of how this executable was compiled.
fn display_build_info() {
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
}
