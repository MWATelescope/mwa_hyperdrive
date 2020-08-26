// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use structopt::StructOpt;

use mwa_hyperdrive::*;

// Modules only for use from main.
mod main_funcs;
use main_funcs::*;

fn setup_logging(debug: bool) -> Result<(), fern::InitError> {
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
        .level(if debug {
            log::LevelFilter::Debug
        } else {
            log::LevelFilter::Info
        })
        .chain(std::io::stdout())
        .apply()?;
    Ok(())
}

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
        #[structopt(long = "text")]
        write_to_text: bool,

        /// Display more messages when running.
        #[structopt(long)]
        debug: bool,
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

    /// Testing RTS functionality
    TestRts {
        /// Path to the metafits file.
        #[structopt(short, long, parse(from_os_str))]
        metafits: PathBuf,
    },
}

fn main() -> Result<(), anyhow::Error> {
    match Opt::from_args() {
        Opt::SimulateVis {
            cli_args,
            param_file,
            cpu,
            write_to_text,
            debug,
        } => {
            setup_logging(debug).expect("Failed to initialize logging.");
            simulate_vis(cli_args, param_file, cpu, write_to_text)?
        }

        Opt::VerifySimulateVis {
            cli_args,
            param_file,
        } => verify_simulate_vis_args(cli_args, param_file)?,

        Opt::VerifySrclist { source_lists } => verify_srclist(source_lists)?,

        Opt::TestRts { metafits } => {
            setup_logging(true).expect("Failed to initialize logging.");

            let context = mwalibContext::new(&metafits, &[])?;
            let beam = PrimaryBeam::default(instrument::BeamType::Mwa32T, 0, Pol::X, &context);
            let scaling = instrument::BeamScaling::None;
            let azel = AzEl::new(0.61086524, 1.4835299);
            let j = instrument::tile_response(&beam, &azel, 180e6, &scaling, &[0.0; 16])?;
            dbg!(j);
        }
    }

    Ok(())
}
