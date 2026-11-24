#!/usr/bin/env python3

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Generate a tiny synthetic phased-array measurement set that EveryBeam can read.

The measurement set has a few stations, each with a small grid of dual-pol
elements described by a PHASED_ARRAY sub-table (as written by OSKAR). The
stations are rigidly rotated with respect to each other (both the element
layout and the dipole orientations, via COORDINATE_AXES), and some X elements
of station 2 are flagged, so that the stations have different beam responses. The visibilities are all
zero; this is only intended for testing beam code.

EveryBeam chooses its telescope model from TELESCOPE_NAME; this is set to
SKA-LOW to select a phased-array model that uses the PHASED_ARRAY sub-table.
Nothing else here is specific to SKA-Low.

Requires python-casacore (pip install python-casacore).

Usage: make_mini_ms.py <output.ms>
"""

import sys

import numpy as np
import casacore.tables as pt
from casacore.measures import measures
from casacore.quanta import quantity

# An arbitrary southern-hemisphere location (near the SKA-Low core).
LON_DEG = 116.7644482
LAT_DEG = -26.82472208
HEIGHT_M = 377.8

# Station centres (east, north) [m] relative to the array centre.
# The first three stations are close together; the rest make baselines long
# enough for calibration (hyperdrive's default minimum baseline is 50 lambda).
STATION_EN = [
    (0.0, 0.0),
    (100.0, 30.0),
    (-60.0, 90.0),
    (450.0, -120.0),
    (-380.0, -400.0),
    (820.0, 610.0),
    (-1150.0, 300.0),
    (200.0, -1400.0),
]
# Rigid rotation of each station (anticlockwise from East, viewed from above)
# [degrees].
STATION_ROTATIONS_DEG = [0.0, 30.0, 75.0, 15.0, 60.0, 105.0, 140.0, 45.0]
# Elements per side of each (square) station, and their spacing [m].
ELEMENTS_PER_SIDE = 4
ELEMENT_SPACING_M = 1.5

FREQS_HZ = np.array([100e6, 101e6, 102e6])
CHAN_WIDTH_HZ = 1e6
# UTC MJD seconds of the first timestep.
START_MJD_S = 60000.5 * 86400.0
INTEGRATION_S = 10.0
NUM_TIMES = 2


def geodetic_to_itrf(lon_rad, lat_rad, height_m):
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f * (2.0 - f)
    n = a / np.sqrt(1.0 - e2 * np.sin(lat_rad) ** 2)
    x = (n + height_m) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (n + height_m) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (n * (1.0 - e2) + height_m) * np.sin(lat_rad)
    return np.array([x, y, z])


def enu_axes(lon_rad, lat_rad):
    east = np.array([-np.sin(lon_rad), np.cos(lon_rad), 0.0])
    north = np.array(
        [
            -np.sin(lat_rad) * np.cos(lon_rad),
            -np.sin(lat_rad) * np.sin(lon_rad),
            np.cos(lat_rad),
        ]
    )
    up = np.array(
        [
            np.cos(lat_rad) * np.cos(lon_rad),
            np.cos(lat_rad) * np.sin(lon_rad),
            np.sin(lat_rad),
        ]
    )
    return east, north, up


def main(ms_name):
    lon = np.radians(LON_DEG)
    lat = np.radians(LAT_DEG)
    centre = geodetic_to_itrf(lon, lat, HEIGHT_M)
    east, north, up = enu_axes(lon, lat)
    station_xyz = np.array([centre + e * east + n * north for (e, n) in STATION_EN])
    num_stations = len(STATION_EN)

    # Point at zenith at the start of the observation.
    dm = measures()
    dm.do_frame(dm.epoch("UTC", quantity(START_MJD_S, "s")))
    dm.do_frame(dm.position("ITRF", *[quantity(v, "m") for v in centre]))
    zenith = dm.measure(dm.direction("AZEL", "0deg", "90deg"), "J2000")
    ra = zenith["m0"]["value"]
    dec = zenith["m1"]["value"]

    # Main table.
    baselines = [(a1, a2) for a1 in range(num_stations) for a2 in range(a1, num_stations)]
    num_rows = NUM_TIMES * len(baselines)
    num_chans = len(FREQS_HZ)
    desc = pt.maketabdesc(
        [
            pt.makearrcoldesc("DATA", 0j, shape=[num_chans, 4], valuetype="complex"),
            pt.makearrcoldesc("WEIGHT_SPECTRUM", 0.0, shape=[num_chans, 4], valuetype="float"),
        ]
    )
    ms = pt.default_ms(ms_name, desc)
    ms.addrows(num_rows)
    times = np.repeat(START_MJD_S + INTEGRATION_S * np.arange(NUM_TIMES), len(baselines))
    ant1 = np.tile([b[0] for b in baselines], NUM_TIMES)
    ant2 = np.tile([b[1] for b in baselines], NUM_TIMES)
    uvw = np.array([station_xyz[a2] - station_xyz[a1] for a1, a2 in zip(ant1, ant2)])
    ms.putcol("TIME", times)
    ms.putcol("TIME_CENTROID", times)
    ms.putcol("ANTENNA1", ant1)
    ms.putcol("ANTENNA2", ant2)
    ms.putcol("UVW", uvw)
    ms.putcol("INTERVAL", np.full(num_rows, INTEGRATION_S))
    ms.putcol("EXPOSURE", np.full(num_rows, INTEGRATION_S))
    ms.putcol("DATA", np.zeros((num_rows, num_chans, 4), dtype=np.complex64))
    ms.putcol("FLAG", np.zeros((num_rows, num_chans, 4), dtype=bool))
    ms.putcol("WEIGHT", np.ones((num_rows, 4), dtype=np.float32))
    ms.putcol("SIGMA", np.ones((num_rows, 4), dtype=np.float32))
    ms.putcol("WEIGHT_SPECTRUM", np.ones((num_rows, num_chans, 4), dtype=np.float32))

    # ANTENNA.
    ant = pt.table(ms.getkeyword("ANTENNA"), readonly=False, ack=False)
    ant.addrows(num_stations)
    ant.putcol("NAME", [f"S{i:02}" for i in range(num_stations)])
    ant.putcol("STATION", ["SKA-LOW"] * num_stations)
    ant.putcol("TYPE", ["GROUND-BASED"] * num_stations)
    ant.putcol("MOUNT", ["X-Y"] * num_stations)
    ant.putcol("POSITION", station_xyz)
    ant.putcol("DISH_DIAMETER", np.full(num_stations, 38.0))
    ant.close()

    # SPECTRAL_WINDOW.
    spw = pt.table(ms.getkeyword("SPECTRAL_WINDOW"), readonly=False, ack=False)
    spw.addrows(1)
    spw.putcell("NUM_CHAN", 0, num_chans)
    spw.putcell("CHAN_FREQ", 0, FREQS_HZ)
    spw.putcell("CHAN_WIDTH", 0, np.full(num_chans, CHAN_WIDTH_HZ))
    spw.putcell("EFFECTIVE_BW", 0, np.full(num_chans, CHAN_WIDTH_HZ))
    spw.putcell("RESOLUTION", 0, np.full(num_chans, CHAN_WIDTH_HZ))
    spw.putcell("REF_FREQUENCY", 0, FREQS_HZ[0])
    spw.putcell("TOTAL_BANDWIDTH", 0, CHAN_WIDTH_HZ * num_chans)
    spw.putcell("NAME", 0, "SPW0")
    spw.close()

    # POLARIZATION (XX, XY, YX, YY).
    pol = pt.table(ms.getkeyword("POLARIZATION"), readonly=False, ack=False)
    pol.addrows(1)
    pol.putcell("NUM_CORR", 0, 4)
    pol.putcell("CORR_TYPE", 0, np.array([9, 10, 11, 12], dtype=np.int32))
    pol.putcell("CORR_PRODUCT", 0, np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int32))
    pol.close()

    ddesc = pt.table(ms.getkeyword("DATA_DESCRIPTION"), readonly=False, ack=False)
    ddesc.addrows(1)
    ddesc.putcell("SPECTRAL_WINDOW_ID", 0, 0)
    ddesc.putcell("POLARIZATION_ID", 0, 0)
    ddesc.close()

    # FIELD.
    field = pt.table(ms.getkeyword("FIELD"), readonly=False, ack=False)
    field.addrows(1)
    direction = np.array([[ra, dec]])
    for col in ["PHASE_DIR", "DELAY_DIR", "REFERENCE_DIR"]:
        field.putcell(col, 0, direction)
    field.putcell("NAME", 0, "zenith")
    field.putcell("TIME", 0, START_MJD_S)
    field.close()

    # OBSERVATION.
    obs = pt.table(ms.getkeyword("OBSERVATION"), readonly=False, ack=False)
    obs.addrows(1)
    obs.putcell("TELESCOPE_NAME", 0, "SKA-LOW")
    obs.putcell(
        "TIME_RANGE", 0, np.array([START_MJD_S, START_MJD_S + NUM_TIMES * INTEGRATION_S])
    )
    obs.putcell("OBSERVER", 0, "hyperdrive")
    obs.putcell("PROJECT", 0, "everybeam-test")
    obs.close()

    # PHASED_ARRAY (element layout of each station).
    n = ELEMENTS_PER_SIDE
    offsets_en = [
        ((i - (n - 1) / 2) * ELEMENT_SPACING_M, (j - (n - 1) / 2) * ELEMENT_SPACING_M)
        for i in range(n)
        for j in range(n)
    ]
    station_flags = [np.zeros((n * n, 2), dtype=bool) for _ in range(num_stations)]
    station_flags[2][[0, 5, 10], 0] = True
    pa_desc = pt.maketabdesc(
        [
            pt.makearrcoldesc("POSITION", 0.0, shape=[3], valuetype="double"),
            pt.makearrcoldesc("COORDINATE_AXES", 0.0, shape=[3, 3], valuetype="double"),
            pt.makearrcoldesc("ELEMENT_OFFSET", 0.0, ndim=2, valuetype="double"),
            pt.makearrcoldesc("ELEMENT_FLAG", False, ndim=2, valuetype="boolean"),
        ]
    )
    pa = pt.table(f"{ms_name}/PHASED_ARRAY", pa_desc, nrow=num_stations, ack=False)
    for col in ["POSITION", "COORDINATE_AXES", "ELEMENT_OFFSET"]:
        pa.putcolkeyword(col, "QuantumUnits", np.array(["m"]))
    pa.putcolkeyword(
        "POSITION", "MEASINFO", {"type": "position", "Ref": "ITRF"}
    )
    # casacore arrays are column-major, so these are transposed relative to
    # the C++ view; i.e. COORDINATE_AXES(:, 0) is the p (east) axis.
    for i in range(num_stations):
        c = np.cos(np.radians(STATION_ROTATIONS_DEG[i]))
        s = np.sin(np.radians(STATION_ROTATIONS_DEG[i]))
        p_axis = c * east + s * north
        q_axis = -s * east + c * north
        axes = np.array([p_axis, q_axis, up])
        pa.putcell("POSITION", i, station_xyz[i])
        pa.putcell("COORDINATE_AXES", i, axes)
        # Element offsets are in the station's (rotated) frame.
        offsets = np.array([e * p_axis + nn * q_axis for (e, nn) in offsets_en])
        pa.putcell("ELEMENT_OFFSET", i, offsets)
        pa.putcell("ELEMENT_FLAG", i, station_flags[i])
    pa.close()
    ms.putkeyword("PHASED_ARRAY", f"Table: {ms_name}/PHASED_ARRAY")

    ms.close()
    print(f"Wrote {ms_name}: {num_stations} stations, zenith RA/Dec {np.degrees(ra):.4f} {np.degrees(dec):.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
