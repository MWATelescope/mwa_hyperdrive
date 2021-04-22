// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "hyperdrive.h"

void vis_gen(const UVW_c *uvw, const Source_c *src, struct Vis_c *vis, unsigned int n_channels,
             unsigned int n_baselines);
