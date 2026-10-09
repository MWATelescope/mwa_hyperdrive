# Instrument and software conventions

`hyperdrive` was written for the MWA, and two of its conventions differ from
those of casacore-based software (CASA, DP3 and LOFAR, WSClean, ASKAPsoft)
and from OSKAR:

- the **polarisation convention**: how Stokes I, Q, U, V map onto the
  instrumental XX, XY, YX, YY (see [Polarisations](pols.md));
- the **UVW frame**: where the array is placed when the UVWs (and so the
  phases) are computed, and which way round a baseline is.

The types that name these live in
[Marlu](https://github.com/MWATelescope/Marlu) (`marlu::convention`), so that
`hyperdrive`, `Birli` and any other Marlu user agree on them and on how files
record them.

## Where the conventions come from

Every subcommand that models sky-model visibilities (`vis-simulate`,
`di-calibrate`, `vis-subtract`, `peel`, `vis-utils simulate`) models in a
polarisation convention and a UVW frame, chosen in this order:

1. `--pol-convention` and `--uvw-frame`, if given;
2. `--convention`, a preset for both;
3. the conventions recorded in, or implied by, the input data (see below);
4. the MWA's (`vis-simulate` has no input data, so this is its default).

The chosen conventions and where they came from are printed at startup. All
three arguments can also be given in an arguments file (`convention =
"lofar"`, `pol_convention = "iau"`, `uvw_frame = "casacore"` in the `[model]`
section).

| `--convention` | polarisations | UVW frame | for |
|---|---|---|---|
| `mwa` | `mwa` | `hyperdrive` | MWA data and `hyperdrive`'s own outputs |
| `iau` | `iau` | `hyperdrive` | IAU polarisations without the casacore frame |
| `lofar` | `iau` | `casacore` | LOFAR data, DP3 |
| `casacore` (aliases `casa`, `wsclean`, `uvh5`, `pyuvdata`) | `iau` | `casacore` | Measurement Sets made or consumed by casacore, CASA or WSClean; UVH5 files |
| `askap` (alias `askapsoft`) | `askap` | `casacore` | ASKAPsoft |
| `oskar` | `mwa` | `oskar` | OSKAR simulations |

`--pol-convention` accepts `mwa` (alias `oskar`), `iau` (aliases `lofar`,
`casacore`, `wsclean`), `askap` (alias `askapsoft`), or the two parts
spelled as UVH5 does: `east` or `north` for the X orientation, optionally
followed by `/avg` or `/sum` for the Stokes convention (`iau` is
`north/avg`, `askap` is `north/sum`).

`--uvw-frame` accepts `hyperdrive` (aliases `marlu`, `mwa`, `aips`),
`casacore` (aliases `lofar`, `casa`, `wsclean`, `askap`, `uvh5`, `pyuvdata`)
and `oskar`.

## UVW frames

Visibilities are always \\( V = \sum_s S_s \exp(2 \pi i (u l + v m + w (n-1)))
\\); the frames differ in how \\( (u, v, w) \\) relate to the antenna positions.

In the `hyperdrive` frame the array is precessed (and nutated) to J2000 and
rotated with the mean local sidereal time, and a baseline is antenna 1 minus
antenna 2, as `Birli` and `cotter` write and as the uvfits (AIPS) convention
has it.

The `casacore` frame reproduces the J2000 UVWs that casacore computes for a
Measurement Set (`MBaseline` ITRF → J2000), which DP3, WSClean, CASA,
ASKAPsoft and pyuvdata all use: the array is rotated with the *apparent*
sidereal time, the J2000 baselines are rotated by the rigid rotation that
takes the annual aberration of the phase centre out
(`MeasMath::deapplyAberration`), and a baseline is antenna 2 minus antenna
1, so the modelled visibilities are the complex conjugates of the
`hyperdrive` frame's. The first two effects are each about 20 arcsec; they
change the model of a source 1° from the phase centre by about 0.5% per 4 km
of baseline. Marlu's implementation matches python-casacore to better than
1 cm on 2 km baselines. The frame needs precession (`--no-precession`
disables it with a warning).

The `oskar` frame is what OSKAR uses with its default
`use_casa_phase_convention`: nothing is precessed - the sky-model
coordinates are taken as apparent coordinates of date and the array is
rotated with the apparent sidereal time - and a baseline is antenna 1 minus
antenna 2.

With `--convention lofar`, `vis-simulate` reproduces a DP3 `predict` of the
same sky model on a LOFAR-like array to 1 part in 10<sup>4</sup> (DP3's own
single-precision level), including an 80 km baseline, with no reordering or
conjugation.

## How files record conventions

Marlu's Measurement Set and uvfits writers (and so every `hyperdrive`
output) record the conventions the visibilities were made in, using each
format's own metadata where it has some:

| what | Measurement Set | uvfits | UVH5 |
|---|---|---|---|
| X orientation | `FEED` table `RECEPTOR_ANGLE` (east is X at 90°: `[π/2, 0]`; north is `[0, π/2]`) | `AIPS AN` `POLAA`/`POLAB` (degrees) | `feed_angle` (same angles; `x_orientation` is derived from them) |
| Stokes convention | main-table keyword `pyuvdata_polconv` (`avg` or `sum`) | primary-header keyword `POLCONV` | `pol_convention` |
| UVW frame | `UVW` column `MEASINFO` reference (`J2000`, or `APP` when not precessing) and `marlu_uvw_frame` | `marlu_uvw_frame` | `uvw_array` is antenna 2 minus antenna 1 (the casacore frame) |

The feed angles and the Stokes keyword are exactly what
[pyuvdata](https://github.com/RadioAstronomySoftwareGroup/pyuvdata) reads
and writes, so files made by either are understood by the other. The UVW
frame is the one thing no format can express; `marlu_uvw_frame` (a table
keyword in a Measurement Set, a `HIERARCH` keyword in a uvfits primary
header) is `hyperdrive`, `casacore` or `oskar`. Note that `cotter` recorded
the IAU feed angles (`[0, π/2]`) for MWA data even though the MWA's X is
east-west.

When reading, `hyperdrive` works out the input data's conventions in this
order and prints the result:

- **X orientation**: from the recorded feed angles (X within 45° of east is
  east-west, as pyuvdata classifies them), except that MWA data (a metafits
  was given, the MS has an `MWA_TILE_POINTING` table, or the uvfits
  `TELESCOP` is `MWA`) are east-west unless the file was written by
  something that records conventions (it has a Stokes-convention or UVW-frame
  keyword, or pyuvdata's `pyuvdata_has_feed`), because `cotter`, `Birli` and
  earlier `hyperdrive` wrote the IAU angles for MWA data. Non-MWA data with
  no feed angles are north-south.
- **Stokes convention**: the `pyuvdata_polconv` (MS) or `POLCONV` (uvfits)
  keyword, else `avg`.
- **UVW frame**: the `marlu_uvw_frame` keyword; else, when the first
  cross-correlation's UVWs could be compared against the antenna positions,
  `casacore` if they have the antenna 2 minus antenna 1 sign; else
  `hyperdrive` for MWA data; else `oskar` if the MS has OSKAR's
  `PHASED_ARRAY` table; else `hyperdrive` when the sign matched, or the
  format's own convention (`casacore` for an MS, `hyperdrive` for uvfits)
  when it could not be checked. A keyword that contradicts the UVW sign is
  used, with a warning.

Input visibilities are never reordered or conjugated; a file in the casacore
frame is modelled in the casacore frame instead. (Earlier versions of
`hyperdrive` conjugated such data on input.) `vis-convert` carries the
input's conventions through to its outputs.
