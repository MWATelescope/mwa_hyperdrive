# mwa_hyperdrive_core

Rust code to handle coordinate transformations and provide structures for flux
densities and source types.

Supported source component types are:
- `point`
- `gaussian`
- `shapelet`

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

See the README in the `mwa_hyperdrive_srclist` directory for more info about
these.

## Prerequisites
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
  - The library dir can be specified manually with ERFA_LIB
  - If not specified, `pkg-config` is used to find the library.
  - Use `--features=erfa-static` to build the library automatically. Requires a
    C compiler and `autoconf`.

- libclang
  - Ubuntu: `libclang-dev`
  - Arch: `clang`

To compile a library statically, use e.g. `ERFA_STATIC=1`. To compile all
libraries statically, use `PKG_CONFIG_ALL_STATIC=1`.
