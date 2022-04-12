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
use mwa_hyperdrive_common::{clap, display_build_info, log, mwalib, setup_logging};
use mwa_hyperdrive_srclist::utilities::*;

#[derive(Parser)]
#[clap(name = "hyperdrive", version, author = env!("CARGO_PKG_HOMEPAGE"), about)]
#[clap(global_setting(AppSettings::DeriveDisplayOrder))]
#[clap(disable_help_subcommand = true)]
#[clap(infer_subcommands = true)]
#[clap(propagate_version = true)]
enum Args {
    /// Perform direction-independent calibration on the input MWA data. See for
    /// more info:
    /// https://github.com/MWATelescope/mwa_hyperdrive/wiki/Calibration-usage
    #[clap(alias = "calibrate")]
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

    /// Simulate visibilities of a sky-model source list.
    #[clap(alias = "simulate-vis")]
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

    /// Subtract sky-model sources from supplied visibilities.
    #[clap(alias = "subtract-vis")]
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

    /// Apply calibration solutions to input data.
    #[clap(alias = "apply-solutions")]
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

    /// Convert between calibration solution file formats.
    #[clap(alias = "convert-solutions")]
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

    /// Plot calibration solutions.
    #[clap(alias = "plot-solutions")]
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

    /// Print information on the dipole gains listed by a metafits file.
    #[clap(arg_required_else_help = true)]
    #[clap(infer_long_args = true)]
    DipoleGains {
        #[clap(name = "METAFITS_FILE", parse(from_os_str))]
        metafits: PathBuf,
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
    let verbosity = match &args {
        Args::DiCalibrate { verbosity, .. } => verbosity,
        Args::VisSimulate { verbosity, .. } => verbosity,
        Args::VisSubtract { verbosity, .. } => verbosity,
        Args::SolutionsApply { verbosity, .. } => verbosity,
        Args::SolutionsConvert { verbosity, .. } => verbosity,
        Args::SolutionsPlot { verbosity, .. } => verbosity,
        Args::SrclistByBeam { args, .. } => &args.verbosity,
        Args::SrclistConvert { args, .. } => &args.verbosity,
        Args::SrclistShift { args, .. } => &args.verbosity,
        Args::SrclistVerify { args, .. } => &args.verbosity,
        Args::DipoleGains { .. } => &0,
    };
    setup_logging(*verbosity).expect("Failed to initialise logging.");

    // Print the version of hyperdrive and its build-time information.
    info!(
        "hyperdrive {} {}",
        match args {
            Args::DiCalibrate { .. } => "di-calibrate",
            Args::VisSimulate { .. } => "vis-simulate",
            Args::VisSubtract { .. } => "vis-subtract",
            Args::SolutionsApply { .. } => "solutions-apply",
            Args::SolutionsConvert { .. } => "solutions-convert",
            Args::SolutionsPlot { .. } => "solutions-plot",
            Args::SrclistByBeam { .. } => "srclist-by-beam",
            Args::SrclistConvert { .. } => "srclist-convert",
            Args::SrclistShift { .. } => "srclist-shift",
            Args::SrclistVerify { .. } => "srclist-verify",
            Args::DipoleGains { .. } => "dipole-gains",
        },
        env!("CARGO_PKG_VERSION")
    );
    display_build_info();

    match Args::parse() {
        Args::DiCalibrate {
            cli_args,
            args_file,
            verbosity: _,
            dry_run,
        } => {
            di_calibrate(cli_args, args_file.as_deref(), dry_run)?;

            info!("hyperdrive di-calibrate complete.");
        }

        Args::VisSimulate {
            args,
            verbosity: _,
            dry_run,
        } => {
            args.run(dry_run)?;

            info!("hyperdrive vis-simulate complete.");
        }

        Args::VisSubtract {
            args,
            verbosity: _,
            dry_run,
        } => {
            args.run(dry_run)?;

            info!("hyperdrive vis-subtract complete.");
        }

        Args::SolutionsApply {
            args,
            verbosity: _,
            dry_run,
        } => {
            args.run(dry_run)?;

            info!("hyperdrive solutions-apply complete.");
        }

        Args::SolutionsConvert { args, verbosity: _ } => {
            args.run()?;

            info!("hyperdrive solutions-convert complete.");
        }

        Args::SolutionsPlot { args, verbosity: _ } => {
            args.run()?;

            info!("hyperdrive solutions-plot complete.");
        }

        Args::DipoleGains { metafits } => {
            let meta = mwalib::MetafitsContext::new(&metafits, None).unwrap();
            let gains = mwa_hyperdrive::metafits::get_dipole_gains(&meta);
            let mut all_unity = vec![];
            let mut non_unity = vec![];
            for (i, tile_gains) in gains.outer_iter().enumerate() {
                if tile_gains
                    .iter()
                    .all(|&g| g.is_finite() && (g - 1.0).abs() < f64::EPSILON)
                {
                    all_unity.push(i);
                } else {
                    non_unity.push((i, tile_gains));
                }
            }

            if all_unity.len() == meta.num_ants {
                info!("All dipoles on all tiles have a gain of 1.0!");
            } else {
                info!(
                    "Tiles with all dipoles alive ({}): {:?}",
                    all_unity.len(),
                    all_unity
                );
                info!("Other tiles:");
                let mut bad_x = Vec::with_capacity(16);
                let mut bad_y = Vec::with_capacity(16);
                let mut bad_string = String::new();
                for (tile_num, tile_gains) in non_unity {
                    let tile_gains = tile_gains.as_slice().unwrap();
                    tile_gains[..16].iter().enumerate().for_each(|(i, &g)| {
                        if (g - 1.0).abs() > f64::EPSILON {
                            bad_x.push(i);
                        }
                    });
                    tile_gains[16..].iter().enumerate().for_each(|(i, &g)| {
                        if (g - 1.0).abs() > f64::EPSILON {
                            bad_y.push(i);
                        }
                    });
                    bad_string.push_str(&format!("    Tile {:>3}: ", tile_num));
                    if !bad_x.is_empty() {
                        bad_string.push_str(&format!("X {:?}", &bad_x));
                    }
                    if !bad_x.is_empty() && !bad_y.is_empty() {
                        bad_string.push_str(", ");
                    }
                    if !bad_y.is_empty() {
                        bad_string.push_str(&format!("Y {:?}", &bad_y));
                    }
                    info!("{}", bad_string);
                    bad_x.clear();
                    bad_y.clear();
                    bad_string.clear();
                }
            }
        }

        // Source list utilities.
        Args::SrclistByBeam { args } => args.run()?,
        Args::SrclistConvert { args } => args.run()?,
        Args::SrclistShift { args } => args.run()?,
        Args::SrclistVerify { args } => args.run()?,
    }

    Ok(())
}
