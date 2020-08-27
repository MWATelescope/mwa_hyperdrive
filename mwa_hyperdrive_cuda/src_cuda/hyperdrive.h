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
    /// The number of frequency channels (num. freq. bands * num. fine channels)
    /// present.
    const unsigned int n_channels;
    /// The number of elements (visibilities) in each array.
    const unsigned int n_vis;
    /// u-coordinates [dimensionless]
    const float *u;
    /// v-coordinates [dimensionless]
    const float *v;
    /// w-coordinates [dimensionless]
    const float *w;
} UVW_s;

/// A struct representing a source's components. Assumes that there is one
/// (l,m,n) per component, and `n_channels` Stokes I flux densities per
/// component.
typedef struct Source_s {
    /// The number of point source components.
    const unsigned int n_points;
    /// l-coordinates [dimensionless]
    const float *point_l;
    /// m-coordinates [dimensionless]
    const float *point_m;
    /// n-coordinates [dimensionless]
    const float *point_n;
    /// The number of frequency channels (num. freq. bands * num. fine channels)
    /// present.
    const unsigned int n_channels;
    /// The point-source flux densities [Jy]. The length of this array should be
    /// `n_points` * `n_channels`.
    const float *point_fd;
} Source_s;

/// A container struct for storing visibilities.
typedef struct Visibilities_s {
    /// The number of visibilities.
    const unsigned int n_visibilities;
    /// Real components of the visibilities.
    float *real;
    /// Imaginary components of the visibilities.
    float *imag;
} Visibilities_s;
