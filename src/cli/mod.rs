// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Command-line interface code. More specific options for `hyperdrive`
//! subcommands are contained in modules.
//!
//! All booleans must have `#[serde(default)]` annotated, and anything that
//! isn't a boolean must be optional. This allows all arguments to be optional
//! *and* usable in an arguments file.
//!
//! Only 3 things should be public in this module: `Hyperdrive`,
//! `Hyperdrive::run`, and `HyperdriveError`.

#[macro_use]
mod common;
mod di_calibrate;
mod dipole_gains;
mod error;
mod solutions;
mod srclist;
mod vis_convert;
mod vis_simulate;
mod vis_subtract;

pub(crate) use common::Warn;
pub use error::HyperdriveError;

use std::path::PathBuf;

use clap::{AppSettings, Args, Parser, Subcommand};
use log::info;

use crate::PROGRESS_BARS;

// Add build-time information from the "built" crate.
include!(concat!(env!("OUT_DIR"), "/built.rs"));

#[derive(Debug, Parser)]
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
#[clap(infer_long_args = true)]
pub struct Hyperdrive {
    #[clap(flatten)]
    global_opts: GlobalArgs,

    #[clap(subcommand)]
    command: Command,
}

#[derive(Debug, Args)]
struct GlobalArgs {
    /// Don't draw progress bars.
    #[clap(long)]
    #[clap(global = true)]
    no_progress_bars: bool,

    /// The verbosity of the program. Increase by specifying multiple times
    /// (e.g. -vv). The default is to print only high-level information.
    #[clap(short, long, parse(from_occurrences))]
    #[clap(global = true)]
    verbosity: u8,

    /// Only verify that arguments were correctly ingested and print out
    /// high-level information.
    #[clap(long)]
    #[clap(global = true)]
    dry_run: bool,

    /// Save the input arguments into a new TOML file that can be used to
    /// reproduce this run.
    #[clap(long)]
    #[clap(global = true)]
    save_toml: Option<PathBuf>,
}

#[derive(Debug, Subcommand)]
#[clap(arg_required_else_help = true)]
enum Command {
    #[clap(alias = "calibrate")]
    #[clap(
        about = r#"Perform direction-independent calibration on the input MWA data.
https://mwatelescope.github.io/mwa_hyperdrive/user/di_cal/intro.html"#
    )]
    DiCalibrate(di_calibrate::DiCalArgs),

    #[clap(alias = "convert-vis")]
    #[clap(about = r#"Convert visibilities from one type to another.
https://mwatelescope.github.io/mwa_hyperdrive/user/vis_convert/intro.html"#)]
    VisConvert(vis_convert::VisConvertArgs),

    #[clap(alias = "simulate-vis")]
    #[clap(about = r#"Simulate visibilities of a sky-model source list.
https://mwatelescope.github.io/mwa_hyperdrive/user/vis_simulate/intro.html"#)]
    VisSimulate(vis_simulate::VisSimulateArgs),

    #[clap(alias = "subtract-vis")]
    #[clap(about = "Subtract sky-model sources from supplied visibilities.
https://mwatelescope.github.io/mwa_hyperdrive/user/vis_subtract/intro.html")]
    VisSubtract(vis_subtract::VisSubtractArgs),

    #[clap(alias = "apply-solutions")]
    #[clap(about = r#"Apply calibration solutions to input data.
https://mwatelescope.github.io/mwa_hyperdrive/user/solutions_apply/intro.html"#)]
    SolutionsApply(solutions::SolutionsApplyArgs),

    #[clap(alias = "plot-solutions")]
    #[clap(
        about = r#"Plot calibration solutions. Only available if compiled with the "plotting" feature.
https://mwatelescope.github.io/mwa_hyperdrive/user/plotting.html"#
    )]
    SolutionsPlot(solutions::SolutionsPlotArgs),

    #[clap(alias = "convert-solutions")]
    #[clap(about = "Convert between calibration solution file formats.")]
    SolutionsConvert(solutions::SolutionsConvertArgs),

    SrclistByBeam(srclist::SrclistByBeamArgs),

    SrclistConvert(srclist::SrclistConvertArgs),

    SrclistVerify(srclist::SrclistVerifyArgs),

    SrclistShift(srclist::SrclistShiftArgs),

    DipoleGains(dipole_gains::DipoleGainsArgs),
}

impl Hyperdrive {
    pub fn run(self) -> Result<(), HyperdriveError> {
        // Set up logging.
        let GlobalArgs {
            verbosity,
            dry_run,
            no_progress_bars,
            save_toml,
        } = self.global_opts;
        setup_logging(verbosity).expect("Failed to initialise logging.");
        // Enable progress bars if the user didn't say "no progress bars".
        if !no_progress_bars {
            PROGRESS_BARS.store(true);
        }

        // Print the version of hyperdrive and its build-time information.
        let sub_command = match &self.command {
            Command::DiCalibrate(_) => "di-calibrate",
            Command::VisConvert(_) => "vis-convert",
            Command::VisSimulate(_) => "vis-simulate",
            Command::VisSubtract(_) => "vis-subtract",
            Command::SolutionsApply(_) => "solutions-apply",
            Command::SolutionsConvert(_) => "solutions-convert",
            Command::SolutionsPlot(_) => "solutions-plot",
            Command::SrclistByBeam(_) => "srclist-by-beam",
            Command::SrclistConvert(_) => "srclist-convert",
            Command::SrclistShift(_) => "srclist-shift",
            Command::SrclistVerify(_) => "srclist-verify",
            Command::DipoleGains(_) => "dipole-gains",
        };
        info!("hyperdrive {} {}", sub_command, env!("CARGO_PKG_VERSION"));
        display_build_info();

        macro_rules! merge_save_run {
            ($args:expr) => {{
                let args = $args.merge()?;
                if let Some(toml) = save_toml {
                    use std::{
                        fs::File,
                        io::{BufWriter, Write},
                    };

                    let mut f = BufWriter::new(File::create(toml)?);
                    let toml_str = toml::to_string(&args).expect("toml serialisation error");
                    f.write_all(toml_str.as_bytes())?;
                }
                args.run(dry_run)?;
            }};
        }

        match self.command {
            Command::DiCalibrate(args) => {
                merge_save_run!(args)
            }

            Command::VisConvert(args) => {
                merge_save_run!(args)
            }

            Command::VisSimulate(args) => {
                merge_save_run!(args)
            }

            Command::VisSubtract(args) => {
                merge_save_run!(args)
            }

            Command::SolutionsApply(args) => {
                merge_save_run!(args)
            }

            Command::SolutionsConvert(args) => {
                args.run()?;
            }

            Command::SolutionsPlot(args) => {
                args.run()?;
            }

            // Source list utilities.
            Command::SrclistByBeam(args) => args.run()?,
            Command::SrclistConvert(args) => args.run()?,
            Command::SrclistShift(args) => args.run()?,
            Command::SrclistVerify(args) => args.run()?,

            // Misc. utilities.
            Command::DipoleGains(args) => args.run()?,
        }

        info!("hyperdrive {} complete.", sub_command);
        Ok(())
    }
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
    let dirty = match GIT_DIRTY {
        Some(true) => " (dirty)",
        _ => "",
    };
    match GIT_COMMIT_HASH_SHORT {
        Some(hash) => {
            info!("Compiled on git commit hash: {hash}{dirty}");
        }
        None => info!("Compiled on git commit hash: <no git info>"),
    }
    if let Some(hr) = GIT_HEAD_REF {
        info!("            git head ref: {}", hr);
    }
    info!("            {}", BUILT_TIME_UTC);
    info!("         with compiler {}", RUSTC_VERSION);
    info!("");
}
