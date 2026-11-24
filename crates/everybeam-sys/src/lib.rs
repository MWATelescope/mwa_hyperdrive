// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Raw bindings to [EveryBeam](https://git.astron.nl/RD/EveryBeam), via a
//! small C shim (see `shim/shim.h`).
//!
//! By default, an existing installation of EveryBeam (>= 0.9) and casacore is
//! used; see the crate's `build.rs` for how these are found. With the
//! `vendored` feature, casacore and EveryBeam are downloaded, built and linked
//! statically, and EveryBeam's element-response coefficients are embedded (see
//! [`data`]).
//!
//! Note that EveryBeam is licensed under the GPL-3.0; binaries that link
//! against it are subject to its terms.

#![allow(non_camel_case_types)]

use std::ffi::{c_char, c_double, c_int};

/// Whether EveryBeam was built by this crate (the `vendored` feature).
pub const VENDORED: bool = cfg!(feature = "vendored");

/// An opaque handle to a loaded EveryBeam telescope.
#[repr(C)]
pub struct eb_telescope {
    _private: [u8; 0],
}

/// Options used when loading a telescope. Any null strings use EveryBeam's
/// defaults.
#[repr(C)]
pub struct eb_options {
    pub element_response_model: *const c_char,
    pub beam_mode: *const c_char,
    pub beam_normalisation_mode: *const c_char,
    pub coeff_path: *const c_char,
    pub data_column_name: *const c_char,
    pub use_channel_frequency: c_int,
    pub frequency_interpolation: c_int,
}

extern "C" {
    /// Load the telescope described by the measurement set at `ms_path`. On
    /// failure, null is returned and an error message is written into `err`.
    pub fn eb_load(
        ms_path: *const c_char,
        options: *const eb_options,
        err: *mut c_char,
        err_len: usize,
    ) -> *mut eb_telescope;

    /// Free a telescope created by [`eb_load`].
    pub fn eb_free(telescope: *mut eb_telescope);

    /// The number of stations in the telescope.
    pub fn eb_num_stations(telescope: *const eb_telescope) -> usize;

    /// Compute beam responses for J2000 (RA, Dec) directions at a single time
    /// (MJD seconds, UTC); see `shim.h` for details.
    #[allow(clippy::too_many_arguments)]
    pub fn eb_point_responses(
        telescope: *const eb_telescope,
        time_mjd_s: c_double,
        field_id: usize,
        ras: *const c_double,
        decs: *const c_double,
        num_directions: usize,
        freqs_hz: *const c_double,
        num_freqs: usize,
        stations: *const usize,
        num_stations: usize,
        out: *mut c_double,
        station_stride: usize,
        freq_stride: usize,
        err: *mut c_char,
        err_len: usize,
    ) -> c_int;

    /// Set the directory that a vendored EveryBeam uses for its data files
    /// (element-response coefficients). This has no effect for a system
    /// EveryBeam, which uses its own data directory. The string is copied.
    pub fn eb_set_data_dir(dir: *const c_char);
}

/// EveryBeam's data files (element-response coefficients), embedded at build
/// time. Only available with the `vendored` feature.
#[cfg(feature = "vendored")]
pub mod data {
    include!(concat!(env!("OUT_DIR"), "/everybeam_data.rs"));
}
