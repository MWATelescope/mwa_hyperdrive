// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

/// A struct containing direction-cosine coordinates for a single source
/// component.
typedef struct LMN_c {
    /// l-coordinate [dimensionless]
    const float l;
    /// m-coordinate [dimensionless]
    const float m;
    /// n-coordinate [dimensionless]
    const float n;
} LMN_c;

/// A struct containing UVW coordinates for a single baseline.
typedef struct UVW_c {
    /// u-coordinate [dimensionless]
    const float u;
    /// v-coordinate [dimensionless]
    const float v;
    /// w-coordinate [dimensionless]
    const float w;
} UVW_c;

/// A struct containing metadata on the observation.
typedef struct Context_c {
    /// The observation's frequency resolution [Hz]
    const double fine_channel_width;
    /// The base frequency of the observation [Hz]
    const double base_freq;
    /// The number of frequency channels (num. freq. bands * num. fine channels)
    /// present.
    const unsigned int n_channels;
    /// The number of baselines present.
    const unsigned int n_baselines;
} Context_c;

/// A struct representing a source's components. Assumes that there is one
/// (l,m,n) per component, and `n_channels` Stokes I flux densities per
/// component.
typedef struct Source_c {
    /// The number of point source components.
    const unsigned int n_points;
    /// LMN coordinates for each point-source component [dimensionless]
    const LMN_c *point_lmn;
    /// The point-source flux densities [Jy]. The length of this array should be
    /// `n_points` * `n_channels`.
    const float *point_fd;
    /// The number of frequency channels (num. freq. bands * num. fine channels)
    /// present.
    const unsigned int n_channels;
} Source_c;

/// A container struct for storing visibilities.
typedef struct Vis_c {
    /// The number of visibilities.
    const unsigned int n_vis;
    /// Real components of the visibilities.
    float *real;
    /// Imaginary components of the visibilities.
    float *imag;
} Vis_c;
