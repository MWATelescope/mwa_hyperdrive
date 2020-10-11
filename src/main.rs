// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use structopt::StructOpt;

use mwa_hyperdrive::*;

// Modules only for use from main.
mod main_funcs;
use main_funcs::calibrate::calibrate;
use main_funcs::simulate_vis::*;
use main_funcs::verify_srclist::*;
use main_funcs::HYPERDRIVE_VERSION;

fn setup_logging(level: u8) -> Result<(), fern::InitError> {
    fern::Dispatch::new()
        .format(|out, message, record| {
            out.finish(format_args!(
                "{}[{}][{}] {}",
                chrono::Local::now().format("[%Y-%m-%d %H:%M:%S]"),
                record.target(),
                record.level(),
                message
            ))
        })
        .level(match level {
            0 => log::LevelFilter::Info,
            1 => log::LevelFilter::Debug,
            _ => log::LevelFilter::Trace,
        })
        .chain(std::io::stdout())
        .apply()?;
    Ok(())
}

#[derive(StructOpt, Debug)]
#[structopt(author, name = "hyperdrive", version = HYPERDRIVE_VERSION.as_str(), about)]
enum Opt {
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
    },

    /// Verify the arguments that can be passed to "simulate-vis".
    VerifySimulateVis {
        // Share the arguments that could be passed in via a parameter file.
        #[structopt(flatten)]
        cli_args: SimulateVisArgs,

        /// All of the arguments to simulate-vis may be specified in a toml or
        /// json file. Any CLI arguments override parameters set in the file.
        #[structopt(name = "PARAMETER_FILE", parse(from_os_str))]
        param_file: Option<PathBuf>,
    },

    /// Verify that a source list can be read by the hyperdrive.
    VerifySrclist {
        /// Path to the source list(s) to be verified.
        #[structopt(name = "SOURCE_LISTS", parse(from_os_str))]
        source_lists: Vec<PathBuf>,
    },

    /// Calibrate the input data. WIP.
    Calibrate {
        // Share the arguments that could be passed in via a parameter file.
        #[structopt(flatten)]
        cli_args: mwa_hyperdrive::calibrate::args::CalibrateUserArgs,

        /// All of the arguments to calibrate may be specified in a toml or json
        /// file. Any CLI arguments override parameters set in the file.
        #[structopt(name = "PARAMETER_FILE", parse(from_os_str))]
        param_file: Option<PathBuf>,

        /// The verbosity of the program. The default is to print high-level
        /// information.
        #[structopt(short, long, parse(from_occurrences))]
        verbosity: u8,

        /// Don't actually do calibration; just verify that data was correctly
        /// ingested and print out high-level information.
        #[structopt(short = "n", long)]
        dry_run: bool,
    },
}

fn main() -> Result<(), anyhow::Error> {
    match Opt::from_args() {
        Opt::SimulateVis {
            cli_args,
            param_file,
            cpu,
            write_to_text,
            verbosity,
        } => {
            setup_logging(verbosity).expect("Failed to initialize logging.");
            simulate_vis(cli_args, param_file, cpu, write_to_text)?
        }

        Opt::VerifySimulateVis {
            cli_args,
            param_file,
        } => verify_simulate_vis_args(cli_args, param_file)?,

        Opt::VerifySrclist { source_lists } => verify_srclist(source_lists)?,

        Opt::Calibrate {
            cli_args,
            param_file,
            verbosity,
            dry_run,
        } => {
            setup_logging(verbosity).expect("Failed to initialize logging.");
            calibrate(cli_args, param_file, dry_run)?;
        }
    }

    Ok(())
}
