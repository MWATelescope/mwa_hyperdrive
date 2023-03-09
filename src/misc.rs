// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Miscellaneous things.

use std::thread;

use console::Term;
use crossbeam_channel::bounded;
use is_terminal::IsTerminal;

pub(crate) fn is_a_tty() -> bool {
    std::io::stdout().is_terminal() || std::io::stderr().is_terminal()
}

/// Perform this expensive operation as a normal Rust function, but if it takes
/// more than a certain amount of time, display a message to the user that
/// you're still waiting for this operation to complete.
pub(crate) fn expensive_op<F, R>(func: F, wait_message: &str) -> R
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    use std::time::Duration;

    const INITIAL_WAIT_TIME: Duration = Duration::from_secs(1);
    const INC_WAIT_TIME: Duration = Duration::from_millis(250);

    let (tx, rx) = bounded(1);

    thread::scope(|s| {
        s.spawn(|| {
            let r = func();
            tx.send(r).expect("receiver is not disconnected");
        });

        // Only print messages if we're in an interactive terminal.
        let term = is_a_tty().then(Term::stderr);

        let mut total_wait_time = Duration::from_secs(0);
        let mut printed_wait_line = false;
        // Loop forever until the return value is ready.
        loop {
            match rx.try_recv() {
                // If the channel received a value, then we need to clean up
                // before returning it to the caller.
                Ok(r) => {
                    // Clear the waiting line.
                    if let Some(term) = term.as_ref() {
                        if printed_wait_line {
                            term.move_cursor_up(1).expect("Couldn't move cursor up");
                            term.clear_line().expect("Couldn't clear line");
                        }
                    }

                    return r;
                }
                // Otherwise we must wait longer.
                Err(_) => {
                    thread::sleep(INC_WAIT_TIME);
                    total_wait_time += INC_WAIT_TIME;
                    if let Some(term) = term.as_ref() {
                        if total_wait_time >= INITIAL_WAIT_TIME {
                            if printed_wait_line {
                                term.move_cursor_up(1).expect("Couldn't move cursor up");
                                term.clear_line().expect("Couldn't clear line");
                            }
                            term.write_line(&format!(
                                "{wait_message}: {:.2}s",
                                total_wait_time.as_secs_f64()
                            ))
                            .expect("Couldn't write line");
                            printed_wait_line = true;
                        }
                    }
                }
            }
        }
    })
}
