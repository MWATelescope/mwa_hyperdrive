#!/usr/bin/env python

# Originally created by Jack Line, edited by CHJ.
#
# The values used here *must* reflect what is used inside hyperdrive and
# corresponding shapelet sky-model sources.

import numpy as np
from scipy.special import factorial, eval_hermite


def gen_shape_basis(n, xlocs, beta=1.0):
    '''Generates the 1D basis function for a given n at xlocs.
    beta is the scale param - default scale to 1 radian'''

    gauss = np.exp(-0.5 * (np.array(xlocs) * np.array(xlocs)))
    # norm = 1.0 / sqrt(2**(n1+n2)*pi**2*b1*b2*factorial(n1)*factorial(n2))

    n = int(n)
    norm = np.sqrt(beta) / np.sqrt(2**n * factorial(n))

    h = eval_hermite(n, xlocs)
    return gauss * norm * h


max_x = 50
num_samps = 10001  # SBF_L
xlocs = np.linspace(-max_x, max_x, num_samps) #/ sqrt(2*pi)
num_basis = 101  # SBF_N (max_x * 2 + 1 ?)
basis_array = np.empty((num_basis, len(xlocs)), dtype=np.float64)

for n in np.arange(num_basis):
    print("Generating basis %d of %d" % (n, num_basis))
    basis_array[n, :] = gen_shape_basis(n, xlocs, beta=1.0)

with open("shapelet_basis_values.bin", "wb") as h:
    h.write(basis_array.tobytes())

with open("shapelet_basis_values.bin", "rb") as h:
    saved_data = np.frombuffer(h.read(), dtype=np.float64).reshape((num_basis, -1))
