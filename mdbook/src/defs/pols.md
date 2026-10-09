# Instrumental polarisations

In `hyperdrive` (and [`mwalib`](https://github.com/MWATelescope/mwalib) and
[`hyperbeam`](https://github.com/MWATelescope/mwa_hyperbeam)), the X
polarisation refers to the East-West dipoles and the Y refers to North-South.
Note that this contrasts with the IAU definition of X and Y, which is opposite
to this. However, this is consistent within the MWA.

MWA visibilities in raw data products are ordered XX, XY, YX, YY where X is
East-West and Y is North-South. `Birli` and `cotter` also write pre-processed
visibilities this way.

`wsclean` expects its input measurement sets to be in the IAU order, so MWA
data (and `hyperdrive`'s default outputs) need their Q, U and V signs
corrected when imaged with `wsclean`. `hyperdrive`'s outputs now record which
orientation they use (see [Instrument and software
conventions](conventions.md)), and sky models can be converted in the IAU
convention with `--pol-convention iau`.

We expect that any input data contains 4 cross-correlation polarisations (XX XY
YX YY), but `hyperdrive` is able to read the following combinations out of the
supported [input data types](./vis_formats_read.md):
- XX
- YY
- XX YY
- XX XY YY

In addition, uvfits files need not have a weight associated with each
polarisation.

# Stokes polarisations

By default (the `mwa` convention) `hyperdrive` maps sky-model Stokes parameters
onto instrumental polarisations as:
- \\( \text{XX} = \text{I} - \text{Q} \\)
- \\( \text{XY} = \text{U} - i\text{V} \\)
- \\( \text{YX} = \text{U} + i\text{V} \\)
- \\( \text{YY} = \text{I} + \text{Q} \\)

where \\( \text{I} \\), \\( \text{Q} \\), \\( \text{U} \\), \\( \text{V} \\) are
Stokes polarisations and \\( i \\) is the imaginary unit.

## Other conventions

Other instruments and software packages define X as north-south, so their
visibilities relate to Stokes parameters differently. The `--pol-convention`
argument (available wherever sky-model visibilities are generated:
`vis-simulate`, `di-calibrate`, `vis-subtract`, `peel`, `vis-utils simulate`)
selects the mapping that is used when converting the sky model to
instrumental polarisations, so that calibration models match the data and
simulated visibilities can be consumed by other software directly:

| `--pol-convention` | XX | XY | YX | YY | used by |
|---|---|---|---|---|---|
| `mwa` (`east`) | \\( \text{I} - \text{Q} \\) | \\( \text{U} - i\text{V} \\) | \\( \text{U} + i\text{V} \\) | \\( \text{I} + \text{Q} \\) | MWA (`cotter`, `Birli`, `hyperdrive`, the RTS); OSKAR |
| `iau` (`north`) | \\( \text{I} + \text{Q} \\) | \\( \text{U} + i\text{V} \\) | \\( \text{U} - i\text{V} \\) | \\( \text{I} - \text{Q} \\) | IAU/TMS; casacore, CASA, DP3 (LOFAR), WSClean |
| `askap` (`north/sum`) | \\( (\text{I} + \text{Q})/2 \\) | \\( (\text{U} + i\text{V})/2 \\) | \\( (\text{U} - i\text{V})/2 \\) | \\( (\text{I} - \text{Q})/2 \\) | ASKAP (ASKAPsoft; \\( \text{I} = \text{XX} + \text{YY} \\)) |

A convention is an *X orientation* (`east` or `north`) and a *Stokes
convention* (`avg`, where \\( \text{I} = (\text{XX} + \text{YY})/2 \\), or
`sum`, where \\( \text{I} = \text{XX} + \text{YY} \\)), the same two
quantities that the UVH5 format records as `x_orientation` and
`pol_convention`; `east/sum` is also accepted. `lofar`, `casacore` and
`wsclean` are aliases of `iau`, `oskar` of `mwa` (OSKAR's X element lies
along its station x, i.e. east, axis) and `askapsoft` of `askap`. The
argument can also be given in an arguments file (`pol_convention = "iau"` in
the `[model]` section).

When `--pol-convention` is not given, the convention recorded in or implied
by the input data is used (the MWA's for `vis-simulate`); see [Instrument
and software conventions](conventions.md) for how files record it, for the
`--convention` presets that also select the UVW frame, and for how
`hyperdrive` works it out from older files.

Only the sky-model conversion is affected: input visibilities are never
reordered, and the MWA FEE beam is defined with X east-west, so the other
conventions are meant for data from other instruments (where `--no-beam` or
`--beam-type none` is used). The veto threshold (`--veto-threshold`) is
always evaluated on XX+YY in the `mwa` convention.
