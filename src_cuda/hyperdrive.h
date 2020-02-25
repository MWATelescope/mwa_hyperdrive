// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

/// A struct containing metadata on the observation.
typedef struct Context_s {
    /// The observation's frequency resolution [Hz]
    const double fine_channel_width;
    /// The base frequency of the observation [Hz]
    const double base_freq;
    /// The LST at the start of the observation [radians]
    const double base_lst;
} Context_s;

/// A struct containing all of the baselines for a given epoch. The (U,V,W)
/// coordinates must be dimensionless; this is expected to be done by dividing
/// (U,V,W) coordinates in metres by a wavelength.
typedef struct UVW_s {
    /// The number of baselines present.
    const unsigned int n_baselines;
    /// The number of elements in each array.
    const unsigned int n_elem;
    /// u-coordinates [dimensionless]
    const float *u;
    /// v-coordinates [dimensionless]
    const float *v;
    /// w-coordinates [dimensionless]
    const float *w;
} UVW_s;

/// A struct representing a source's components. Assumes that there is one
/// frequency and Stokes I, Q, U, V flux density per component, such that the
/// length of `point_l` is `n_points` and `point_fd` is 4x `n_points`.
typedef struct Source_s {
    /// The number of point source components.
    const unsigned int n_points;
    /// l-coordinates [dimensionless]
    const float *point_l;
    /// m-coordinates [dimensionless]
    const float *point_m;
    /// n-coordinates [dimensionless]
    const float *point_n;
    /// Flux densities of Stokes I, Q, U and V [Jy]. This array is four times
    /// larger than `n_points`.
    const float *point_fd;
} Source_s;

/// A struct representing a source and its various components. Assumes that
/// there is one frequency and Stokes I, Q, U, V flux density per component,
/// such that the length of `point_freq` is `n_points` and `point_fd` is 4x
/// `n_points`.
typedef struct Visibilities_s {
    /// The number of visibilities.
    const unsigned int n_visibilities;
    /// Real components of the visibilities.
    float *real;
    /// Imaginary components of the visibilities.
    float *imag;
} Visibilities_s;
