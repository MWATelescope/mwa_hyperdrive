// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! The main hyperdrive binary.

use clap::Parser;

fn main() {
    // Run hyperdrive, only performing extra steps if it returns an error.
    //
    // Stolen from BurntSushi. We don't return Result from main because it
    // prints the debug representation of the error. The code below prints the
    // "display" or human readable representation of the error.
    if let Err(e) = mwa_hyperdrive::Hyperdrive::parse().run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
