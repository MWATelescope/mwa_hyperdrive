// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

pub(crate) fn setup_logging(level: u8) -> Result<(), fern::InitError> {
    fern::Dispatch::new()
        .format(|out, message, record| {
            out.finish(format_args!(
                "[{}][{} line {}][{}] {}",
                chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
                record.target(),
                record.line().unwrap_or(0),
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

// Add build-time information from the "built" crate.
include!(concat!(env!("OUT_DIR"), "/built.rs"));

// Not sure how to format this string nicely without the "lazy_static" crate.
use lazy_static::lazy_static;
lazy_static! {
    /// A formatted string detailing which git commit of hyperdrive was used,
    /// what compiler version was used, and when the executable was built.
    pub static ref HYPERDRIVE_VERSION: String =
        format!(r#"{ver}
Compiled on git commit hash: {hash}{dirty}
                head ref:    {head_ref}
         at: {time}
         with compiler: {compiler}"#,
                ver = env!("CARGO_PKG_VERSION"),
                hash = GIT_COMMIT_HASH.unwrap_or("<no git info>"),
                dirty = match GIT_DIRTY {
                    Some(true) => " (dirty)",
                    _ => "",
                },
                head_ref = GIT_HEAD_REF.unwrap_or("<no git info>"),
                time = BUILT_TIME_UTC,
                compiler = RUSTC_VERSION,
        );
    // These lines prevent warnings about unused built variables.
    static ref _RUSTDOC_VERSION: &'static str = RUSTDOC_VERSION;
    static ref _GIT_VERSION: Option<&'static str> = GIT_VERSION;
}
