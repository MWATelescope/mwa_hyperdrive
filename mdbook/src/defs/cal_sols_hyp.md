# The `hyperdrive` calibration solutions format

Jones matrices are stored in a `fits` file as an "image" with 4 dimensions
(timeblock, tile, chanblock, float, in that order) in the "SOLUTIONS" HDU (which
is the second HDU). An element of the solutions is a 64-bit float (a.k.a.
double-precision float). The last dimension always has a length of 8; these
correspond to the real part of XX, the imaginary part of XX, then XY, YX and YY
(4 complex numbers to form a Jones matrix).

Tiles are ordered by antenna number, i.e. the second column in the observation's
corresponding metafits files labelled "Antenna". Times and frequencies are
sorted ascendingly.

> Note that in the context of the MWA, "antenna" and "tile" are used
> interchangeably.

## Metadata

Many metadata keys are stored in HDU 1. All keys (in fact, all metadata) are
optional.

`OBSID` describes the MWA observation ID, which is a GPS timestamp.

`SOFTWARE` reports the software used to write this `fits` file.

`CMDLINE` is the command-line call that produced this `fits` file.

### Calibration-specific

`MAXITER` is the maximum number of iterations allowed for each chanblock's
convergence.

`S_THRESH` is the stop threshold of calibration; chanblock iteration ceases once
its precision is better than this.

`M_THRESH` is the minimum threshold of calibration; if a chanblock reaches the
maximum number of iterations while calibrating and this minimum threshold has
not been reached, we say that the chanblock failed to calibrate.

`UVW_MIN` and `UVW_MAX` are the respective minimum and maximum UVW cutoffs in
metres. Any UVWs below or above these thresholds have baseline weights of 0
during calibration (meaning they effectively aren't used in calibration).
`UVW_MIN_L` and `UVW_MAX_L` correspond to `UVW_MIN` and `UVW_MAX`, but are in
wavelength units (the `L` stands for lambda).

Some MWA beam codes require a file for their calculations. `BEAMFILE` is the
path to this file.

### Raw MWA data corrections

`PFB` describes the [PFB gains flavour](mwa/corrections.md#pfb-gains) applied to
the raw MWA data. At the time of writing, this flavour is described as "jake",
"cotter2014", "empirical", "levine", or "none".

`D_GAINS` is "Y" if the [digital
gains](../defs/mwa/corrections.md#digital-gains) were applied to the raw MWA
data. "N" if they were not.

`CABLELEN` is "Y" if the [cable length
corrections](../defs/mwa/corrections.md#cable-lengths) were applied to the raw
MWA data. "N" if they were not.

`GEOMETRY` is "Y" if the [geometric delay
correction](../defs/mwa/corrections.md#geometric-correction-aka-phase-tracking)
was applied to the raw MWA data. "N" if they were not.

### Others

`MODELLER` describes what was used to generate model visibilities in
calibration. Currently, this is either `CPU` or `CUDA GPU`.

## Extra HDUs

More metadata are contained in HDUs other than the first one (that which
contains the metadata keys described above). Other than the first HDU and the
"SOLUTIONS" HDU (HDUs 1 and 2, respectfully), *all HDUs and their contents are
optional*.

### TIMEBLOCKS

See [blocks](blocks.md) for an explanation of what timeblocks are.

The "TIMEBLOCKS" HDU is a FITS table with three columns:

1. `Start`
2. `End`
3. `Average`

Each row represents a calibration timeblock, and there must be the same number
of rows as there are timeblocks in the calibration solutions (in the "SOLUTIONS"
HDU). Each of these times is a centroid GPS timestamp.

It is possible to have one or multiple columns without data; `cfitsio` will
write zeros for values, but `hyperdrive` will ignore columns with all zeros.

While average times are likely just the median of its corresponding start and
end times, it need not be so; in this case, it helps to clarify that some
timesteps in that calibration timeblock were not used. e.g. a start time of 10
and an end time of 16 probably has an average time of 13, but, if 3 of 4
timesteps in that timeblock are used, then the average time could be 12.666 or
13.333.

### TILES

The "TILES" HDU is a FITS table with up to five columns:

1. `Antenna`
2. `Flag`
3. `TileName`
4. `DipoleGains`
5. `DipoleDelays`

`Antenna` is the 0-N antenna index (where N is the total number of antennas in
the observation). These indices match the "Antenna" column of an [MWA
metafits](mwa/metafits.md) file.

`Flag` is a boolean indicating whether an antenna was flagged for calibration
(1) or not (0).

`TileName` is the... name of the tile. As with `Antenna`, this should match the
contents of an MWA metafits file.

`DipoleGains` contains the dipole gains used for each tile in calibration. There
are 32 values per tile; the first 16 are for the X dipoles and the second 16 are
for the Y dipoles. Typically, the values are either 0 (dead dipole) or 1.

`DipoleDelays` contains the dipole delays used for each tile in calibration.
There are 16 values per tile.

### CHANBLOCKS

See [blocks](blocks.md) for an explanation of what chanblocks are.

The "CHANBLOCKS" HDU is a FITS table with up to three columns:

1. `Index`
2. `Flag`
3. `Freq`

`Index` is the 0-N chanblock index (where N is the total number of chanblocks in
the observation). Note that this is not necessarily the same as the total number
of *channels* in the observation; channels may be averaged before calibration,
making the number of chanblocks less than the number of channels.

`Flag` indicates whether calibration was attempted (1) or not (0) on a chanblock
(boolean).

`Freq` is the centroid frequency of the chanblock (as a double-precision float).
If *any* of the frequencies is an NaN, then `hyperdrive` will not use the `Freq`
column.

### RESULTS (Calibration results)

The "RESULTS" HDU is a FITS image with two dimensions -- timeblock and
chanblock, in that order -- that describe the precision to which a chanblock
converged for that timeblock (as double-precision floats). If a chanblock was
flagged, NaN is provided for its precision. NaN is also listed for chanblocks
that completely failed to calibrate.

These calibration precisions must have the same number of timeblocks and
chanblocks described by the calibration solutions (in the "SOLUTIONS" HDU).

### BASELINES

The "BASELINES" HDU is a FITS image with one dimension. The values of the
"image" (let's call it an array) are the double-precision float baseline weights
used in calibration (controlled by UVW minimum and maximum cutoffs). The length
of the array is the total number of baselines (i.e. flagged and unflagged).
Flagged baselines have weights of NaN, e.g. baseline 0 is between antennas 0 and
1, but if antenna 1 is flagged, the weight of baseline 0 is NaN, but baseline 1
is between antennas 0 and 2 so it has a value other than NaN.

These baseline weights must have a non-NaN value for all tiles in the
observation (e.g. if there are 128 tiles in the calibration solutions, then
there must be 8128 baseline weights).

~~~admonish example title="Python code for reading"
A full example of reading and plotting solutions is
[here](https://github.com/MWATelescope/mwa_hyperdrive/blob/main/examples/read_hyperdrive_sols.py),
but simple examples of reading solutions and various metadata are below.

```python
#!/usr/bin/env python3

from astropy.io import fits

f = fits.open("hyperdrive_solutions.fits")
sols = f["SOLUTIONS"].data
num_timeblocks, num_tiles, num_chanblocks, _ = sols.shape

obsid = f[0].header["OBSID"]
pfb_flavour = f[0].header["PFB"]
start_times = f[0].header["S_TIMES"]

tile_names = [tile[1] for tile in f["TILES"].data]
tile_flags = [tile[2] for tile in f["TILES"].data]

freqs = [chan[1] for chan in f["CHANBLOCKS"].data]

cal_precisions_for_timeblock_0 = f["RESULTS"].data[0]
```
~~~
