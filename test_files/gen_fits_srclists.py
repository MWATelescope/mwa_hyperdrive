#!/usr/bin/env python

import pandas as pd
from astropy.coordinates import Angle
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
import numpy as np
import re
import io

arrays_jack = {
    "UNQ_SOURCE_ID": [],
    "NAME": [],
    "RA": [],
    "DEC": [],
    "INT_FLX150": [],
    "MAJOR_DC": [],
    "MINOR_DC": [],
    "PA_DC": [],
    "MOD_TYPE": [],
    "NORM_COMP_PL": [],
    "ALPHA_PL": [],
    "NORM_COMP_CPL": [],
    "ALPHA_CPL": [],
    "CURVE_CPL": [],
}
arrays_gleam = {
    "Name": [],
    "RAJ2000": [],
    "DEJ2000": [],
    "S_200": [],
    "alpha": [],
    "beta": [],
    "a": [],
    "b": [],
    "pa": [],
}
for (i,(  name, cmp, ra, dec, fd, maj, min,   pa, sn1, sn2,  sv,  typ, alpha, q, )) in enumerate([
    ["point-list", 0, 0., 1., 1.,   0.,  0.,  0.,  0.,  0., 0.0, "nan",  0.0, 0.0 ],
    ["point-pl",   0, 1., 2., 2.,   0.,  0.,  0.,  0.,  0., 0.0, "pl",  -0.8, 0.0 ],
    ["point-cpl",  0, 3., 4., 3.,   0.,  0.,  0.,  0.,  0., 0.0, "cpl", -0.9, 0.2 ],
    ["gauss-list", 0, 0., 1., 1.,  20., 10., 75.,  0.,  0., 0.0, "nan",  0.0, 0.0 ],
    ["gauss-pl",   0, 1., 2., 2.,  20., 10., 75.,  0.,  0., 0.0, "pl",  -0.8, 0.0 ],
    ["gauss-cpl",  0, 3., 4., 3.,  20., 10., 75.,  0.,  0., 0.0, "cpl", -0.9, 0.2 ],
    # ["shape-list", 0, 0., 1., 1.,  20., 10., 75.,  0.,  1., 0.5, "nan",  0.0, 0.0 ],
    # ["shape-pl",   0, 1., 2., 1.,  20., 10., 75.,  0.,  1., 0.5, "pl",  -0.8, 0.0 ],
    # ["shape-cpl",  0, 3., 4., 1.,  20., 10., 75.,  0.,  1., 0.5, "cpl", -0.8, 0.2 ],
    # todo: shapelets
]):
    if typ == "nan":
        continue

    # i_ = f"{i:04d}"
    arrays_jack["UNQ_SOURCE_ID"].append(f"{name}")
    arrays_jack["NAME"].append(f"{name}_C{cmp}")
    arrays_jack["RA"].append(ra)
    arrays_jack["DEC"].append(dec)
    arrays_jack["INT_FLX150"].append(fd)
    arrays_jack["MAJOR_DC"].append(maj)
    arrays_jack["MINOR_DC"].append(min)
    arrays_jack["PA_DC"].append(pa)
    arrays_jack["MOD_TYPE"].append(typ)
    # arrays_jack["NORM_COMP_PL"].append(fd)
    # arrays_jack["ALPHA_PL"].append(alpha)

    if typ == "cpl":
        arrays_jack["NORM_COMP_PL"].append(0.0)
        arrays_jack["ALPHA_PL"].append(0.0)
        arrays_jack["NORM_COMP_CPL"].append(fd)
        arrays_jack["ALPHA_CPL"].append(alpha)
        arrays_jack["CURVE_CPL"].append(q)
    else:
        arrays_jack["NORM_COMP_PL"].append(fd)
        arrays_jack["ALPHA_PL"].append(alpha)
        arrays_jack["NORM_COMP_CPL"].append(0.0)
        arrays_jack["ALPHA_CPL"].append(0.0)
        arrays_jack["CURVE_CPL"].append(0.0)

    if typ == "cpl":
        continue

    arrays_gleam['Name'].append(f"{name}")
    arrays_gleam['RAJ2000'].append(ra)
    arrays_gleam['DEJ2000'].append(dec)
    arrays_gleam['S_200'].append(fd)
    arrays_gleam['alpha'].append(alpha)
    arrays_gleam['beta'].append(q)
    arrays_gleam['a'].append(maj * 3600)
    arrays_gleam['b'].append(min * 3600)
    arrays_gleam['pa'].append(pa)


table = Table(arrays_jack, )
table.write('test_files/jack.fits', overwrite=True)
df = table.to_pandas()
print("jack\n", df[[df.columns[0], *df.columns[2:]]].to_string(index=False))

table = Table(arrays_gleam, )
table.write('test_files/gleam.fits', overwrite=True)
df = table.to_pandas()
print("gleam\n", df.to_string(index=False))