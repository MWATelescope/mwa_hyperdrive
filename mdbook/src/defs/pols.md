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

The two orientations of a pair of crossed dipoles, seen from above (north
up, east right). Position angles, like feed angles and polarisation angles,
are measured from north through east. A source with Stokes Q > 0 has its
electric field (green) along north-south, so it shows up in whichever
dipole points north-south: YY for the MWA, XX for the IAU. Stokes U (field
at 45°) is seen equally by both dipoles and lands in XY and YX; V's sign
in XY and YX flips with the orientation because swapping X and Y swaps XY
and YX.

<div style="text-align:center">
<svg viewBox="0 0 640 260" width="100%" style="max-width:640px" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif" font-size="14" fill="currentColor" role="img" aria-label="Dipole orientations in the mwa and iau conventions, seen from above with north up"><defs><marker id="pa" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#3aa655"/></marker><marker id="na" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="currentColor"/></marker></defs><line x1="30" y1="110" x2="30" y2="60" stroke="currentColor" stroke-width="2" marker-end="url(#na)"/><text x="30" y="52" text-anchor="middle">N</text><line x1="30" y1="110" x2="70" y2="110" stroke="currentColor" stroke-width="2" marker-end="url(#na)"/><text x="80" y="115">E</text><text x="200" y="24" text-anchor="middle" font-weight="bold">mwa (east)</text><line x1="140" y1="130" x2="260" y2="130" stroke="#e0782f" stroke-width="7" stroke-linecap="round"/><text x="270" y="135" fill="#e0782f" font-weight="bold">X</text><line x1="200" y1="70" x2="200" y2="190" stroke="#2f8fe0" stroke-width="7" stroke-linecap="round"/><text x="208" y="66" fill="#2f8fe0" font-weight="bold">Y</text><line x1="240" y1="80" x2="240" y2="180" stroke="#3aa655" stroke-width="2" stroke-dasharray="5 4" marker-start="url(#pa)" marker-end="url(#pa)"/><text x="246" y="172" fill="#3aa655" font-size="12">E, Q &gt; 0</text><text x="200" y="222" text-anchor="middle" font-size="13">feed angles (X, Y) = (90°, 0°)</text><text x="200" y="244" text-anchor="middle" font-size="13">XX = I − Q, YY = I + Q</text><text x="470" y="24" text-anchor="middle" font-weight="bold">iau (north)</text><line x1="470" y1="70" x2="470" y2="190" stroke="#e0782f" stroke-width="7" stroke-linecap="round"/><text x="478" y="66" fill="#e0782f" font-weight="bold">X</text><line x1="410" y1="130" x2="530" y2="130" stroke="#2f8fe0" stroke-width="7" stroke-linecap="round"/><text x="540" y="135" fill="#2f8fe0" font-weight="bold">Y</text><line x1="510" y1="80" x2="510" y2="180" stroke="#3aa655" stroke-width="2" stroke-dasharray="5 4" marker-start="url(#pa)" marker-end="url(#pa)"/><text x="516" y="172" fill="#3aa655" font-size="12">E, Q &gt; 0</text><text x="470" y="222" text-anchor="middle" font-size="13">feed angles (X, Y) = (0°, 90°)</text><text x="470" y="244" text-anchor="middle" font-size="13">XX = I + Q, YY = I − Q</text><line x1="335" y1="40" x2="335" y2="250" stroke="currentColor" stroke-opacity="0.3"/></svg>
</div>


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
