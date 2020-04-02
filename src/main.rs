// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::path::PathBuf;

use structopt::StructOpt;

// Modules only for use from main.
mod main_funcs;
use main_funcs::*;

#[derive(StructOpt, Debug)]
#[structopt(author, about = HYPERDRIVE_VERSION.as_str())]
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
        #[structopt(long)]
        text: bool,

        /// Display more message when running.
        #[structopt(long)]
        debug: bool,
    },

    /// Verify that a source list can be read by the hyperdrive.
    VerifySrclist {
        /// Path to the source list(s) to be verified.
        #[structopt(name = "SOURCE_LISTS", parse(from_os_str))]
        source_lists: Vec<PathBuf>,
    },
}

fn main() -> Result<(), anyhow::Error> {
    match Opt::from_args() {
        Opt::SimulateVis {
            cli_args,
            param_file,
            cpu,
            text,
            debug,
        } => simulate_vis(cli_args, param_file, cpu, text, debug)?,

        Opt::VerifySrclist { source_lists } => verify_srclist(source_lists)?,
    }

    Ok(())
}
