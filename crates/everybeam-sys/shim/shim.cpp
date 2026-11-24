// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// A small C interface to EveryBeam, so that it can be used from Rust. All C++
// exceptions are caught here and converted into error messages.

#include "shim.h"

#include <complex>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <string>

#include <EveryBeam/beammode.h>
#include <EveryBeam/beamnormalisationmode.h>
#include <EveryBeam/elementresponse.h>
#include <EveryBeam/load.h>
#include <EveryBeam/options.h>
#include <EveryBeam/pointresponse/pointresponse.h>
#include <EveryBeam/telescope/telescope.h>

struct eb_telescope {
    std::unique_ptr<everybeam::telescope::Telescope> telescope;
    everybeam::BeamMode beam_mode;
};

namespace {

void write_error(const char *msg, char *err, size_t err_len) {
    if (err == nullptr || err_len == 0)
        return;
    std::strncpy(err, msg, err_len - 1);
    err[err_len - 1] = '\0';
}

bool is_set(const char *s) { return s != nullptr && s[0] != '\0'; }

// From HDF5's H5Epublic.h and H5public.h (stable since HDF5 1.10, where
// hid_t is 64 bits). Declared here so that HDF5's headers aren't needed.
// hbool_t's size varies between HDF5 builds, so a zeroed 64-bit buffer is
// passed for it.
extern "C" int H5Eset_auto2(int64_t estack_id, void *func, void *client_data);
extern "C" int H5is_library_threadsafe(void *is_ts);
constexpr int64_t H5E_DEFAULT = 0;

bool hdf5_is_threadsafe() {
    static const bool threadsafe = [] {
        uint64_t is_ts = 0;
        H5is_library_threadsafe(&is_ts);
        return is_ts != 0;
    }();
    return threadsafe;
}

// EveryBeam's OSKAR coefficient reader probes for HDF5 datasets that may not
// exist, and calls H5::Exception::dontPrint() to silence HDF5's error
// messages. With a thread-safe HDF5, that only applies to the calling thread,
// so silence them for each thread that calculates responses. Without a
// thread-safe HDF5, the setting is global, and HDF5 must not be called
// concurrently (EveryBeam serialises its own HDF5 reads), so it's only done
// when loading a telescope (which is serialised by the caller).
void silence_hdf5_errors_in_this_thread() {
    thread_local bool silenced = false;
    if (!silenced && hdf5_is_threadsafe()) {
        H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
        silenced = true;
    }
}

// Creating a point-response object sets up casacore measures conversions,
// which aren't reliably thread-safe; serialise that (but not the responses).
std::mutex point_response_mutex;

std::mutex data_dir_mutex;
std::string data_dir;

} // namespace

// When EveryBeam is vendored, its GetDataDirectory() is patched to call this
// first; a null return falls back to EveryBeam's usual logic.
extern "C" const char *hyperdrive_everybeam_data_dir() {
    std::lock_guard<std::mutex> lock(data_dir_mutex);
    return data_dir.empty() ? nullptr : data_dir.c_str();
}

extern "C" {

eb_telescope *eb_load(const char *ms_path, const eb_options *options, char *err, size_t err_len) {
    try {
        everybeam::Options eb_options;
        everybeam::BeamMode beam_mode = everybeam::BeamMode::kFull;
        if (options != nullptr) {
            if (is_set(options->element_response_model))
                eb_options.element_response_model =
                    everybeam::ElementResponseModelFromString(options->element_response_model);
            else
                eb_options.element_response_model = everybeam::ElementResponseModel::kDefault;
            if (is_set(options->beam_mode))
                beam_mode = everybeam::ParseBeamMode(options->beam_mode);
            if (is_set(options->beam_normalisation_mode))
                eb_options.beam_normalisation_mode =
                    everybeam::ParseBeamNormalisationMode(options->beam_normalisation_mode);
            if (is_set(options->coeff_path))
                eb_options.coeff_path = options->coeff_path;
            if (is_set(options->data_column_name))
                eb_options.data_column_name = options->data_column_name;
            eb_options.use_channel_frequency = options->use_channel_frequency != 0;
            eb_options.frequency_interpolation = options->frequency_interpolation != 0;
        }
        eb_options.beam_mode = beam_mode;

        if (!hdf5_is_threadsafe()) {
            H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
        }
        auto telescope = everybeam::Load(std::string(ms_path), eb_options);
        if (!telescope) {
            write_error("EveryBeam did not return a telescope for this measurement set", err,
                        err_len);
            return nullptr;
        }
        return new eb_telescope{std::move(telescope), beam_mode};
    } catch (const std::exception &e) {
        write_error(e.what(), err, err_len);
    } catch (...) {
        write_error("Unknown C++ exception when loading EveryBeam telescope", err, err_len);
    }
    return nullptr;
}

void eb_set_data_dir(const char *dir) {
    std::lock_guard<std::mutex> lock(data_dir_mutex);
    data_dir = dir == nullptr ? "" : dir;
}

void eb_free(eb_telescope *telescope) { delete telescope; }

size_t eb_num_stations(const eb_telescope *telescope) {
    return telescope->telescope->GetNrStations();
}

int eb_point_responses(const eb_telescope *telescope, double time_mjd_s, size_t field_id,
                       const double *ras, const double *decs, size_t num_directions,
                       const double *freqs_hz, size_t num_freqs, const size_t *stations,
                       size_t num_stations, double *out, size_t station_stride,
                       size_t freq_stride, char *err, size_t err_len) {
    try {
        silence_hdf5_errors_in_this_thread();
        // Each call gets its own point-response object; they are not
        // thread-safe, but the (const) telescope is.
        std::unique_ptr<everybeam::pointresponse::PointResponse> point_response;
        {
            std::lock_guard<std::mutex> lock(point_response_mutex);
            point_response = telescope->telescope->GetPointResponse(time_mjd_s);
        }
        std::complex<float> buffer[4];
        // Iterate over directions in the outer loop; the point-response object
        // caches the ITRF conversion of the last-used direction.
        for (size_t d = 0; d < num_directions; d++) {
            for (size_t s = 0; s < num_stations; s++) {
                for (size_t f = 0; f < num_freqs; f++) {
                    point_response->Response(telescope->beam_mode, buffer, ras[d], decs[d],
                                             freqs_hz[f], stations[s], field_id);
                    double *o = out + 8 * (s * station_stride + f * freq_stride + d);
                    for (size_t i = 0; i < 4; i++) {
                        o[2 * i] = buffer[i].real();
                        o[2 * i + 1] = buffer[i].imag();
                    }
                }
            }
        }
        return 0;
    } catch (const std::exception &e) {
        write_error(e.what(), err, err_len);
    } catch (...) {
        write_error("Unknown C++ exception when calculating EveryBeam responses", err, err_len);
    }
    return 1;
}

} // extern "C"
