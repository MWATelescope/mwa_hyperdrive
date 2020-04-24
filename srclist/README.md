# mwa_hyperdrive_srclist

Code to handle the `hyperdrive` source list format, as well as other supported
formats (currently RTS and WODEN).

## The `hyperdrive` source list format

A source list is a map of names associated sources. A source is a list of
components. A component has right ascension and declination coordinates in the
J2000 epoch, a component type, and a flux density type. The RA coordinates must
be between 0 and 360 degrees, and the declination must be between -90 and 90
degrees. Flux densities do not need to be positive, but the sum of all flux
densities for a source must be positive.

Supported component types are:
- `point`

- `gaussian`
  - Has major and minor axes specified in arcseconds, and a position angle in
    degrees.

- `shapelet`
  - Has major and minor axes specified in arcseconds, a position angle in
    degrees, and a list of shapelet coefficients. Each coefficient has an
    integer `n1` and `n2`, as well as a `coeff` float value.

Supported flux density types are:
- `list`
  - Multiple instances of a frequency (in Hz) associated with Stokes I, Q, U, V
    flux densities (in Jy).

- `power_law`
  - A spectral index associated with a single instance of a frequency (in Hz)
    associated with Stokes I, Q, U, V flux densities (in Jy).

- `curved_power_law`
  - A spectral index and curvature parameter associated with a single instance
    of a frequency (in Hz) associated with Stokes I, Q, U, V flux densities (in
    Jy). See [Callingham et al.
    2017](https://ui.adsabs.harvard.edu/abs/2017ApJ...836..174C/abstract),
    section 4.1 for more details.

Example `hyperdrive` source lists in the `yaml` and `json` formats are in the
`examples` directory. Python scripts to read and write `yaml` files are also in
the `examples` directory.

Don't forget to verify your source list with the `srclist` binary (see usage
below).

## Usage
### Verification
To check if a source list is compatible with `hyperdrive`, the following can be
used:

    srclist verify </path/to/srclist1> </path/to/srclist2>

More than one source list can be given at a time.

If a source list file's extension is `.txt`, then `srclist` will assume that it
is an `RTS`-style file. To specify another kind, use `-i`. Check the help text
for more details:

    srclist verify -h

### Conversion
To convert a source list into another format, the following can be used:

    srclist convert </path/to/input/srclist> </path/to/output/srclist>
    
`srclist convert` will do its best to know what the source list styles are based
on the file extensions. The `hyperdrive` format is in either `.yaml` or `.json`
files, whereas `RTS` and `WODEN` source lists are in `.txt` files. If a source
list file's extension is `.txt`, then `srclist` will assume that it is an
`RTS`-style file. To specify another kind, use `-i` or `-o`. Check the help text
for more details:

    srclist convert -h

## Installation
- `git clone` this repo
- `cd` into this directory
- Run `cargo build --release`.

The `srclist` tool will now be available at `./target/release/srclist`.

### Prerequisites
<details>

- A Rust compiler with a version >= 1.42.0

  `https://www.rust-lang.org/tools/install`

- [cfitsio](https://heasarc.gsfc.nasa.gov/docs/software/fitsio/)
  - Ubuntu: `libcfitsio-dev`
  - Arch: `cfitsio`
  - Library and include dirs can be specified manually with CFITSIO_LIB and
    CFITSIO_INC
  - If not specified, `pkg-config` is used to find the library.

- [ERFA](https://github.com/liberfa/erfa)
  - Ubuntu: `liberfa-dev`
  - Arch: AUR package `erfa`
  - Library and include dirs can be specified manually with ERFA_LIB and
    ERFA_INC
  - If not specified, `pkg-config` is used to find the library.

- libclang
  - Ubuntu: `libclang-dev`
  - Arch: `clang`

To compile a library statically, use e.g. `ERFA_STATIC=1`. To compile all
libraries statically, use `PKG_CONFIG_ALL_STATIC=1`. </details>

## FAQ
- Another source list format?

    Well, the others are really easy to write incorrectly, and difficult to read
    correctly! By using standard interchangeable formats, such as `yaml` or
    `json`, the files are still human-readable while being extremely easy for
    computers to read. Every programming language can read from and write to
    these formats.
