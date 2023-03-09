// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Miscellaneous things.

use std::thread;

use console::Term;
use crossbeam_channel::bounded;
use hifitime::{Duration, Epoch, Unit};
use is_terminal::IsTerminal;

/// Some timestamps may be read in ever so slightly off from their true values
/// because of float errors. This function checks if a supplied [Epoch], when
/// represented as GPS seconds, is really close to a neat value in the
/// hundredths. If so, the value is rounded and returned.
///
/// e.g. The GPS time 1090008639.999405 should be 1090008634.0. Other examples
/// of usage are in the tests alongside this function.
pub(crate) fn round_hundredths_of_a_second(e: Epoch) -> Epoch {
    let e_gps = e.to_gpst_seconds() * 100.0;
    if (e_gps.round() - e_gps).abs() < 0.1 {
        Epoch::from_gpst_seconds(e_gps.round() / 100.0)
    } else {
        e
    }
}

/// Quantize a duration to the nearest multiple of q nanoseconds
pub(crate) fn quantize_duration(d: Duration, q_nanos: f64) -> Duration {
    let d_nanos = d.to_unit(Unit::Nanosecond);
    let d_nanos = (d_nanos / q_nanos).round() * q_nanos;
    Duration::from_f64(d_nanos, Unit::Nanosecond)
}

fn is_a_tty() -> bool {
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

    const INITIAL_WAIT_TIME: Duration = Duration::from_secs(2);
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

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn test_round_seconds() {
        let e = Epoch::from_gpst_seconds(1090008639.999405);
        assert_abs_diff_eq!(
            round_hundredths_of_a_second(e).to_gpst_seconds(),
            1090008640.0
        );

        let e = Epoch::from_gpst_seconds(1090008640.251);
        assert_abs_diff_eq!(
            round_hundredths_of_a_second(e).to_gpst_seconds(),
            1090008640.25
        );

        let e = Epoch::from_gpst_seconds(1090008640.24999);
        assert_abs_diff_eq!(
            round_hundredths_of_a_second(e).to_gpst_seconds(),
            1090008640.25
        );

        // No rounding.
        let e = Epoch::from_gpst_seconds(1090008640.26);
        assert_abs_diff_eq!(
            round_hundredths_of_a_second(e).to_gpst_seconds(),
            1090008640.26
        );
    }

    #[test]
    fn test_round_duration() {
        let half_day = Duration::from_f64(0.5, Unit::Day);
        let millis = Duration::from_f64(1., Unit::Millisecond);
        let quanta = 10_000_000.;
        assert_eq!(half_day, quantize_duration(half_day + millis, quanta));
        assert_eq!(half_day, quantize_duration(half_day + 4 * millis, quanta));
        assert_eq!(half_day, quantize_duration(half_day - 4 * millis, quanta));
        assert_eq!(
            half_day + 10 * millis,
            quantize_duration(half_day + 11 * millis, quanta)
        );
    }
}
