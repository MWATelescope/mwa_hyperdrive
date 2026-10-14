# The DP3/BBS (`dp3`) source list format

This is the text sky-model format read by
[DP3](https://dp3.readthedocs.io/en/latest/steps/Predict.html) (and BBS, via
LOFAR's `makesourcedb`), and written by WSClean (`-save-source-list`). Files
usually have a `.skymodel` or `.txt` extension. `hyperdrive` can read these
(`-i dp3`, or the aliases `skymodel` and `bbs`), but not write them. A file is
recognised automatically if its first non-comment line is a format line.

`hyperdrive` mirrors DP3's reader (`sky_model/ReadSkyModel.cc`) and spectral
model (`base/PointSource.cc`).

~~~admonish example
```plaintext
Format = Name, Type, Patch, Ra, Dec, I, Q, U, V, ReferenceFrequency, SpectralIndex, LogarithmicSI, MajorAxis, MinorAxis, Orientation

# A patch definition: empty name and type.
 , , FornaxA, 03:21:29.7, -37.08.37.1
FornaxA_0, GAUSSIAN, FornaxA, 03:21:29.74400, -37.08.37.07600, 198.59297, 0, 0, 0, 1.8879e+08, [-0.8], true, 822.529, 822.529, 0
FornaxA_1, POINT, FornaxA, 03:24:00.42500, -37.16.31.93900, 97.526513, 0, 0, 0, 1.8879e+08, [-0.8, 0.3], true, , ,
# A component without a patch is a source of its own.
J002549-260211, POINT, , 00:25:49.2, -26.02.13, 16.77, 0, 0, 0, 1e+08, [-0.83], true, , ,
```
~~~

## Format line

The first non-comment line should be `Format = ...` (case insensitive), or the
file should start with comments including the older `# (...) = format`. It
names the comma-separated fields on each line; names are case insensitive and
may be in any order, and any subset may be used (but a position is required). A
field may carry a default value in quotes, used when a line leaves the field
empty, e.g. WSClean writes

```plaintext
Format = Name, Type, Ra, Dec, I, SpectralIndex, LogarithmicSI, ReferenceFrequency='150000000.0', MajorAxis, MinorAxis, Orientation
```

`Name=fixed'value'` gives a value that is always used. Unknown fields are
ignored. Values may be quoted, and lists (`[...]`) may contain commas. Lines
starting with `#` are comments (as is anything after a `#` outside of quotes),
and blank lines are ignored.

If no format line is found before the first data line, then, like DP3, the
default format

```plaintext
Name, Type, Ra, Dec, I, Q, U, V, MajorAxis, MinorAxis, Orientation
```

is used and a warning is printed. This only happens when the source list type
is given explicitly (`-i dp3`); automatic detection of the file type requires a
format line, so that other kinds of files aren't mistaken for DP3 sky models.

## Sources and patches

A line with an empty `Name` (and `Type`) defines a patch, e.g.
` , , PatchName, 03:21:29.7, -37.08.37.1`. Each patch becomes one `hyperdrive`
source named after the patch, containing all components that name the patch
in their `Patch` column (a patch must be defined before it is used; patches
without components are ignored). Components without a patch become a source of
their own, named after the component's `Name`.

## Components

`Type` is `POINT` (the default when empty) or `GAUSSIAN`. DP3's `DISK` and
`SHAPELET` types are not supported, and cause an error. For Gaussians,
`MajorAxis` and `MinorAxis` are FWHM in arcseconds, and `Orientation` is the
position angle in degrees (east of north); these are the same as `hyperdrive`'s
own format. DP3 by default (`OrientationIsAbsolute=false`) evaluates Gaussians
in the phase-centre frame, as `hyperdrive` does; if `OrientationIsAbsolute=true`
is given, a warning is printed and the orientation is used as is.

Only J2000 positions (`RefType` empty or `J2000`) are supported. Rotation
measures (`RotationMeasure`, `PolarizedFraction`, `PolarizationAngle`,
`ReferenceWavelength`) are not supported, and cause an error.

## Positions

`Ra` and `Dec` are read like casacore's `MVAngle` (as DP3 does). The accepted
forms (`+` is an optional sign; separator letters are case insensitive) are:

| Form | Example | Meaning |
|------|---------|---------|
| `+h:m:s` or `+h:m` | `03:21:29.744` | hours, minutes, seconds |
| `+hHmMsS` | `03h21m29.744s` | hours, minutes, seconds |
| `+d.m.s` | `-37.08.37.076` | degrees, arcminutes, arcseconds |
| `+dDmMsS` | `-37d08m37.076s`, `10d` | degrees, arcminutes, arcseconds |
| number with a unit | `50.37deg`, `0.879rad`, `30arcmin` | `deg`, `rad`, `arcmin`, `arcsec`, `mas`, `'`, `"`, `h`, `min`, `s` |

Note that, as in casacore, a number without a unit is in **radians**, and an
RA written as `d.m.s` is in degrees. Colons are not allowed in declinations.
Unlike casacore, trailing characters that are not part of these forms are an
error. The older separate fields `rah`/`rad`, `ram`, `ras` and
`dech`/`decd`, `decm`, `decs` are also supported.

## Flux densities

`I`, `Q`, `U` and `V` are in Jy (empty `I` means 1 Jy, as in DP3; empty `Q`,
`U`, `V` are 0). `ReferenceFrequency` is in Hz. `SpectralIndex` is a list of
spectral terms `[c0, c1, ...]` (or a single value), and `LogarithmicSI`
(default `true`) selects how they are applied to Stokes I. With
`LogarithmicSI=true`,

\\[ S_\nu = I \left(\frac{\nu}{\nu_0}\right)^{c_0 + c_1 \log_{10}(\nu/\nu_0) + c_2 \log_{10}^2(\nu/\nu_0) + \dots}. \\]

- No terms is a flat spectrum (what DP3 predicts); it becomes a power law with
  spectral index 0.
- One term is a [power law](fd_types.md) with spectral index \\( c_0 \\).
- Two terms are a [curved power law](fd_types.md) with spectral index
  \\( c_0 \\) and \\( q = c_1 / \ln 10 \\) (exact, because
  \\( (\nu/\nu_0)^{c_1 \log_{10}(\nu/\nu_0)} = e^{(c_1 / \ln 10) \ln^2(\nu/\nu_0)} \\)).
- More terms (after dropping trailing zeros) can't be represented exactly; the
  spectrum is evaluated at 16 log-spaced frequencies between 70 and 300 MHz
  into a [list](fd_types.md), and a warning is printed.

With `LogarithmicSI=false` (WSClean's "ordinary polynomial" output),
\\( S_\nu = I + c_0 x + c_1 x^2 + \dots \\) with \\( x = \nu/\nu_0 - 1 \\); this
is also evaluated into a list (with a warning).

DP3 only applies spectral terms to Stokes I, but `hyperdrive`'s power laws
scale all Stokes parameters; a warning is printed if components have both
non-zero Q, U or V and spectral terms.
