// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::path::PathBuf;
use std::time::Duration;

use console::Term;
use crossbeam_channel::bounded;
use mwalib::{CorrelatorContext, MwalibError};

use mwa_hyperdrive_common::{is_a_tty, mwalib};

/// Wait this many seconds before printing a message that we're still waiting to
/// read gpubox files.
const READ_GPUBOX_WAIT_SECS: u64 = 2;

/// Return a [CorrelatorContext] given the necessary files.
///
/// It can take a while to create the correlator context because all gpubox
/// files need to be iterated over (big IO operation). To make the UX a bit
/// nicer, this function prints a message that we're waiting if the operation
/// takes a while.
pub(super) fn get_mwalib_correlator_context(
    metafits: PathBuf,
    gpuboxes: Vec<PathBuf>,
) -> Result<CorrelatorContext, MwalibError> {
    let (tx_context, rx_context) = bounded(1);
    std::thread::spawn(move || tx_context.send(CorrelatorContext::new(&metafits, &gpuboxes)));
    let mwalib_context = {
        // Only print messages if we're in an interactive terminal.
        let term = is_a_tty().then(Term::stderr);

        let mut total_wait_time = Duration::from_secs(0);
        let inc_wait_time = Duration::from_millis(500);
        let mut printed_wait_line = false;
        // Loop forever until the context is ready.
        loop {
            // If the channel is full, then the context is ready.
            if rx_context.is_full() {
                // Clear the waiting line.
                if let Some(term) = term.as_ref() {
                    if printed_wait_line {
                        term.move_cursor_up(1).expect("Couldn't move cursor up");
                        term.clear_line().expect("Couldn't clear line");
                    }
                }
                break;
            }
            // Otherwise we must wait longer.
            std::thread::sleep(inc_wait_time);
            total_wait_time += inc_wait_time;
            if let Some(term) = term.as_ref() {
                if total_wait_time.as_secs() >= READ_GPUBOX_WAIT_SECS {
                    if printed_wait_line {
                        term.move_cursor_up(1).expect("Couldn't move cursor up");
                        term.clear_line().expect("Couldn't clear line");
                    }
                    term.write_line(&format!(
                        "Still waiting to inspect all gpubox metadata: {:.2}s",
                        total_wait_time.as_secs_f64()
                    ))
                    .expect("Couldn't write line");
                    printed_wait_line = true;
                }
            }
        }
        // Receive the context result. We can safely unwrap because we only
        // break the loop when the channel is populated.
        rx_context.recv().unwrap()
    };
    mwalib_context
}
