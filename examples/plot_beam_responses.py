#!/usr/bin/env python3

import sys

import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) == 1:
    file = "beam_responses.tsv"
else:
    file = sys.argv[1]
data = np.genfromtxt(fname=file, delimiter="\t", skip_header=0)

fig, ax = plt.subplots(1, 2, subplot_kw=dict(projection="polar"))
p = ax[0].scatter(data[:, 0], data[:, 1], c=data[:, 2])
plt.colorbar(p)
p = ax[1].scatter(data[:, 0], data[:, 1], c=np.log10(data[:, 2]))
plt.colorbar(p)
plt.show()
