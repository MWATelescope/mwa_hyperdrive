// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to be shared between hyperdrive-related binaries.

use fern::colors::{Color, ColoredLevelConfig};
use log::info;

// Add build-time information from the "built" crate.
include!(concat!(env!("OUT_DIR"), "/built.rs"));

/// Activate a logger. Use colours only if we're on a tty (e.g. a terminal) and
/// display source lines in log messages with verbosity >= 3.
// This is pretty dirty code. Can it be cleaned up?
pub fn setup_logging(verbosity: u8) -> Result<(), log::SetLoggerError> {
    let is_a_tty = atty::is(atty::Stream::Stdout) || atty::is(atty::Stream::Stderr);

    let (high_level_messages, low_level_messages) = if is_a_tty {
        let colours = ColoredLevelConfig::new()
            .warn(Color::Red)
            .info(Color::Green)
            .debug(Color::Blue)
            .trace(Color::Yellow);
        let colours2 = colours;

        (
            fern::Dispatch::new().format(move |out, message, record| {
                out.finish(format_args!(
                    "{} {:<5} {}",
                    chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
                    colours.color(record.level()),
                    message
                ))
            }),
            fern::Dispatch::new().format(move |out, message, record| {
                out.finish(format_args!(
                    "{} {} line {:<3} {:<5} {}",
                    chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
                    record.target(),
                    record.line().unwrap_or(0),
                    colours2.color(record.level()),
                    message
                ))
            }),
        )
    } else {
        (
            fern::Dispatch::new().format(move |out, message, record| {
                out.finish(format_args!(
                    "{} {:<5} {}",
                    chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
                    record.level(),
                    message
                ))
            }),
            fern::Dispatch::new().format(move |out, message, record| {
                out.finish(format_args!(
                    "{} {} line {:<3} {:<5} {}",
                    chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
                    record.target(),
                    record.line().unwrap_or(0),
                    record.level(),
                    message
                ))
            }),
        )
    };

    let logger = match verbosity {
        0 => high_level_messages.level(log::LevelFilter::Info),
        1 => high_level_messages.level(log::LevelFilter::Debug),
        2 => high_level_messages.level(log::LevelFilter::Trace),
        _ => low_level_messages.level(log::LevelFilter::Trace),
    };
    logger.chain(std::io::stdout()).apply()
}

/// Write many info-level log lines of how this executable was compiled.
pub fn display_build_info() {
    match GIT_HEAD_REF {
        Some(hr) => {
            let dirty = GIT_DIRTY.unwrap_or(false);
            info!(
                "Compiled on git commit hash: {}{}",
                GIT_COMMIT_HASH.unwrap(),
                if dirty { " (dirty)" } else { "" }
            );
            info!("            git head ref: {}", hr);
        }
        None => info!("Compiled on git commit hash: <no git info>"),
    }
    info!("            {}", BUILT_TIME_UTC);
    info!("         with compiler {}", RUSTC_VERSION);
    info!("");
}
