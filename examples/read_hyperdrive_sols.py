#!/usr/bin/env python3

import sys
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

if len(sys.argv) == 1:
    filename = "hyperdrive_solutions.fits"
else:
    filename = sys.argv[1]

f = fits.open(filename)
data = f["SOLUTIONS"].data
num_tiles = data.shape[1]
num_tiles_per_row = num_tiles // 16

# Only looking at the first timeblock.
i_timeblock = 0
data = data[i_timeblock, :, :, ::2] + data[i_timeblock, :, :, 1::2] * 1j

# # Uncomment if you want to divide by a reference.
# i_tile_ref = -1
# refs = []
# for ref in data[i_tile_ref].reshape((-1, 2, 2)):
#     refs.append(np.linalg.inv(ref))
# refs = np.array(refs)
# j_div_ref = []
# for tile_j in data:
#     for (j, ref) in zip(tile_j, refs):
#         j_div_ref.append(j.reshape((2, 2)).dot(ref))
# data = np.array(j_div_ref).reshape(data.shape)

# Amps
amps = np.abs(data)

_, ax = plt.subplots(num_tiles_per_row, 16, sharex=True, sharey=True)
# Uncomment if you want to manually set the y-limit
# ax[0, 0].set_ylim(0, 2)
for i in range(num_tiles):
    ax[i // 16, i % 16].plot(amps[i, :, 0].flatten())  # XX
    ax[i // 16, i % 16].plot(amps[i, :, 3].flatten())  # YY
plt.show()

# Phases
phases = np.rad2deg(np.angle(data))

_, ax = plt.subplots(num_tiles_per_row, 16, sharex=True, sharey=True)
ax[0, 0].set_ylim(-180, 180)
for i in range(num_tiles):
    ax[i // 16, i % 16].plot(phases[i, :, 0].flatten())  # XX
    ax[i // 16, i % 16].plot(phases[i, :, 3].flatten())  # YY
plt.show()
