// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// A small C interface to EveryBeam (https://git.astron.nl/RD/EveryBeam), so
// that it can be used from Rust.

#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// An opaque handle to a loaded EveryBeam telescope.
typedef struct eb_telescope eb_telescope;

// Options used when loading a telescope. Any NULL strings use EveryBeam's
// defaults.
typedef struct eb_options {
    // e.g. "default", "hamaker", "lobes", "oskar_dipole",
    // "oskar_spherical_wave", "skala40_wave", "skalow_feko", ...
    const char *element_response_model;
    // e.g. "full", "array_factor", "element"
    const char *beam_mode;
    // e.g. "none", "full", "amplitude", "preapplied"
    const char *beam_normalisation_mode;
    // Path to element response coefficients (telescope dependent).
    const char *coeff_path;
    // The data column to use for LOFAR "preapplied" beam information.
    const char *data_column_name;
    // Boolean; use channel frequencies rather than the subband frequency.
    int use_channel_frequency;
    // Boolean; interpolate the (MWA) beam over frequency.
    int frequency_interpolation;
} eb_options;

// Load the telescope described by the measurement set at `ms_path`. On failure,
// NULL is returned and an error message is written into `err`.
eb_telescope *eb_load(const char *ms_path, const eb_options *options, char *err, size_t err_len);

// Set the directory that a vendored EveryBeam uses for its data files
// (element-response coefficients). The string is copied. This has no effect
// for a system EveryBeam.
void eb_set_data_dir(const char *dir);

// Free a telescope created by `eb_load`.
void eb_free(eb_telescope *telescope);

// The number of stations in the telescope.
size_t eb_num_stations(const eb_telescope *telescope);

// Directions prepared for beam-response calculations at a single time.
typedef struct eb_directions eb_directions;

// Prepare J2000 (RA, Dec) directions [radians] for beam-response calculations
// at a single time (MJD seconds, UTC). For phased-array telescopes (e.g.
// SKA-Low, LOFAR, OSKAR), every direction is converted to ITRF here, once, so
// that no coordinate conversions are needed when calculating responses. The
// coordinates are copied.
//
// On failure, NULL is returned and an error message is written into `err`.
eb_directions *eb_directions_new(const eb_telescope *telescope, double time_mjd_s,
                                 const double *ras, const double *decs, size_t num_directions,
                                 char *err, size_t err_len);

// Free directions created by `eb_directions_new`.
void eb_directions_free(eb_directions *directions);

// Compute beam responses for `num_directions` of the prepared directions,
// starting from index `first_direction`, for every combination of direction,
// frequency and station. Each station's response in a direction is calculated
// for all frequencies in one EveryBeam call.
//
// Each response is a 2x2 complex Jones matrix written as 8 doubles (re, im for
// each of j00, j01, j10, j11). The response for station index `s` (into
// `stations`), frequency index `f` and direction index `d` (relative to
// `first_direction`) is written at
// `out + 8 * (s * station_stride + f * freq_stride + d)`.
//
// This function may be called from multiple threads simultaneously with the
// same telescope and directions, as long as the output locations don't overlap.
//
// Returns 0 on success; otherwise an error message is written into `err`.
int eb_station_responses(const eb_telescope *telescope, const eb_directions *directions,
                         size_t first_direction, size_t num_directions, size_t field_id,
                         const double *freqs_hz, size_t num_freqs, const size_t *stations,
                         size_t num_stations, double *out, size_t station_stride,
                         size_t freq_stride, char *err, size_t err_len);

#ifdef __cplusplus
}
#endif
