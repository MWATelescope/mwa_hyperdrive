// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Utilities for source list files.

use structopt::{clap::AppSettings, StructOpt};

use mwa_hyperdrive_common::setup_logging;
use mwa_hyperdrive_srclist::{utilities::*, *};

#[derive(StructOpt, Debug)]
#[structopt(name = "hyperdrive srclist", about,
            author = env!("CARGO_PKG_HOMEPAGE"),
            global_settings = &[AppSettings::ColoredHelp,
                                AppSettings::ArgRequiredElseHelp,
                                AppSettings::DeriveDisplayOrder])]
enum Args {
    ByBeam {
        #[structopt(flatten)]
        args: ByBeamArgs,
    },

    Convert {
        #[structopt(flatten)]
        args: ConvertArgs,
    },

    Shift {
        #[structopt(flatten)]
        args: ShiftArgs,
    },

    Verify {
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

fn try_main() -> Result<(), SrclistError> {
    match Args::from_args() {
        Args::ByBeam { args } => {
            setup_logging(args.verbosity).expect("Failed to initialise logging.");
            args.run()?
        }

        Args::Convert { args } => {
            setup_logging(args.verbosity).expect("Failed to initialise logging.");
            args.run()?
        }

        Args::Shift { args } => {
            setup_logging(args.verbosity).expect("Failed to initialise logging.");
            args.run()?
        }

        Args::Verify { args } => {
            setup_logging(args.verbosity).expect("Failed to initialise logging.");
            args.run()?
        }
    }

    Ok(())
}
