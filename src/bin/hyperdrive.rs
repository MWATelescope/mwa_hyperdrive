// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! The main hyperdrive binary.

use std::path::PathBuf;

use clap::{AppSettings, Parser};
use log::{debug, info, trace, warn};

use mwa_hyperdrive::{
    calibrate::{args::CalibrateUserArgs, di_calibrate, solutions::CalibrationSolutions},
    simulate_vis::{simulate_vis, SimulateVisArgs},
    HyperdriveError,
};
use mwa_hyperdrive_common::{clap, display_build_info, log, mwalib, setup_logging};
use mwa_hyperdrive_srclist::utilities::*;

#[derive(Parser)]
#[clap(name = "hyperdrive", version, author = env!("CARGO_PKG_HOMEPAGE"), about)]
#[clap(global_setting(AppSettings::ArgRequiredElseHelp))]
#[clap(global_setting(AppSettings::DeriveDisplayOrder))]
#[clap(global_setting(AppSettings::DisableHelpSubcommand))]
#[clap(global_setting(AppSettings::InferLongArgs))]
#[clap(global_setting(AppSettings::InferSubcommands))]
#[clap(global_setting(AppSettings::PropagateVersion))]
enum Args {
    /// Perform direction-independent calibration on the input MWA data.
    ///
    /// See for more info:
    /// https://github.com/MWATelescope/mwa_hyperdrive/wiki/Calibration-usage
    #[clap(alias = "calibrate")]
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
    SimulateVis {
        #[clap(flatten)]
        args: SimulateVisArgs,

        /// The verbosity of the program. The default is to print high-level
        /// information.
        #[clap(short, long, parse(from_occurrences))]
        verbosity: u8,

        /// Don't actually do any work; just verify that the input arguments
        /// were correctly ingested and print out high-level information.
        #[clap(long)]
        dry_run: bool,

        /// Use the CPU for visibility generation. This is deliberately made
        /// non-default because using a GPU is much faster.
        #[cfg(feature = "cuda")]
        #[clap(long)]
        cpu: bool,
    },

    /// Plot calibration solutions.
    #[cfg(feature = "plotting")]
    SolutionsPlot {
        #[clap(name = "SOLUTIONS_FILES", parse(from_os_str))]
        files: Vec<PathBuf>,

        /// The metafits file associated with the solutions. This provides
        /// additional information on the plots, like the antenna names.
        #[clap(short, long, parse(from_str))]
        metafits: Option<PathBuf>,

        #[clap(short, long)]
        ref_ant: Option<usize>,

        /// The verbosity of the program. Increase by specifying multiple times
        /// (e.g. -vv). The default is to print only high-level information.
        #[clap(short, long, parse(from_occurrences))]
        verbosity: u8,
    },

    SrclistByBeam {
        #[clap(flatten)]
        args: ByBeamArgs,
    },

    SrclistConvert {
        #[clap(flatten)]
        args: ConvertArgs,
    },

    SrclistShift {
        #[clap(flatten)]
        args: ShiftArgs,
    },

    SrclistVerify {
        #[clap(flatten)]
        args: VerifyArgs,
    },

    /// Print information on the dipole gains listed by a metafits file.
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
        Args::SimulateVis { verbosity, .. } => verbosity,
        #[cfg(feature = "plotting")]
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
            Args::SimulateVis { .. } => "simulate-vis",
            #[cfg(feature = "plotting")]
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
            let args = if let Some(f) = args_file {
                trace!("Merging command-line arguments with the argument file");
                Box::new(cli_args.merge(&f)?)
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
            #[cfg(feature = "cuda")]
            cpu,
        } => {
            #[cfg(not(feature = "cuda"))]
            simulate_vis(args, dry_run)?;

            #[cfg(feature = "cuda")]
            simulate_vis(args, cpu, dry_run)?;

            info!("hyperdrive simulate-vis complete.");
        }

        #[cfg(feature = "plotting")]
        Args::SolutionsPlot {
            files,
            ref_ant,
            metafits,
            verbosity: _,
        } => {
            if metafits.is_none() {
                warn!("No metafits supplied; the obsid and tile names won't be on the plots");
            }

            let mwalib_context = metafits.map(|m| mwalib::MetafitsContext::new(&m, None).unwrap());
            let tile_names: Option<Vec<&str>> = mwalib_context.as_ref().map(|m| {
                m.rf_inputs
                    .iter()
                    .filter(|rf| rf.pol == mwalib::Pol::X)
                    .map(|rf| rf.tile_name.as_str())
                    .collect()
            });

            for solutions_file in files {
                let sols = CalibrationSolutions::read_solutions_from_ext(&solutions_file).unwrap();
                let base = solutions_file
                    .file_stem()
                    .and_then(|os_str| os_str.to_str())
                    .expect("Calibration solutions filename contains invalid UTF-8");
                let plot_title = format!(
                    "obsid {}",
                    mwalib_context
                        .as_ref()
                        .map(|m| m.obs_id)
                        .or(sols.obsid)
                        .map(|o| o.to_string())
                        .unwrap_or_else(|| "<unknown>".to_string())
                );
                let plot_files = sols
                    .plot(base, &plot_title, ref_ant, tile_names.as_deref())
                    .unwrap();
                info!("Wrote {:?}", plot_files);
            }
        }

        // Source list utilities.
        Args::SrclistByBeam { args } => args.run()?,
        Args::SrclistConvert { args } => args.run()?,
        Args::SrclistShift { args } => args.run()?,
        Args::SrclistVerify { args } => args.run()?,

        Args::DipoleGains { metafits } => {
            let meta = mwalib::MetafitsContext::new(&metafits, None).unwrap();
            let gains = mwa_hyperdrive::data_formats::metafits::get_dipole_gains(&meta);
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
    }

    Ok(())
}
