// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use structopt::StructOpt;

use mwa_hyperdrive::*;

mod calibrate;
mod common;
mod simulate_vis;
use calibrate::calibrate;
use common::{setup_logging, HYPERDRIVE_VERSION};
use simulate_vis::*;

#[derive(StructOpt, Debug)]
// #[structopt(author, name = "hyperdrive", version = HYPERDRIVE_VERSION.as_str(), about, global_settings = &[AppSettings::ColoredHelp, AppSettings::ArgRequiredElseHelp])]
#[structopt(author, name = "hyperdrive", version = HYPERDRIVE_VERSION.as_str(), about)]
enum Args {
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
    match Args::from_args() {
        Args::Calibrate {
            cli_args,
            param_file,
            verbosity,
            dry_run,
        } => {
            setup_logging(verbosity).expect("Failed to initialize logging.");
            calibrate(cli_args, param_file, dry_run)?;
        }

        Args::SimulateVis {
            cli_args,
            param_file,
            cpu,
            write_to_text,
            verbosity,
            dry_run,
        } => {
            setup_logging(verbosity).expect("Failed to initialize logging.");
            simulate_vis(cli_args, param_file, cpu, write_to_text, dry_run)?
        }
    }

    Ok(())
}
