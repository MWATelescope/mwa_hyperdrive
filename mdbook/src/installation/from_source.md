# Installing `hyperdrive` from source code

## Dependencies
`hyperdrive` depends on these C libraries:


```admonish example title="[cfitsio](https://heasarc.gsfc.nasa.gov/docs/software/fitsio/)"
- Ubuntu: `libcfitsio-dev`
- Arch: `cfitsio`
- Library and include dirs can be specified manually with `CFITSIO_LIB` and
  `CFITSIO_INC`
  - If not specified, `pkg-config` is used to find the library.
- Can compile statically; use the `cfitsio-static` or `all-static` features.
  - Requires a C compiler and `autoconf`.
```

```admonish example title="[ERFA](https://github.com/liberfa/erfa)"
- Ubuntu: `liberfa-dev`
- Arch: AUR package `erfa`
- The library dir can be specified manually with `ERFA_LIB`
  - If not specified, `pkg-config` is used to find the library.
- Can compile statically; use the `erfa-static` or `all-static` features.
  - Requires a C compiler and `autoconf`.
```

```admonish example title="[hdf5](https://www.hdfgroup.org/hdf5)"
- Ubuntu: `libhdf5-dev`
- Arch: `hdf5`
- The library dir can be specified manually with `HDF5_DIR`
  - If not specified, `pkg-config` is used to find the library.
- Can compile statically; use the `hdf5-static` or `all-static` features.
  - Requires `CMake` version 3.10 or higher.
```

### Optional dependencies

```admonish tip title="freetype2 (for calibration solutions plotting)"
- Only required if the `plotting` feature is enabled (which it is by default)
- Version must be `>=2.11.1`
- Arch: `pkg-config` `make` `cmake` `freetype2`
- Ubuntu: `libfreetype-dev` `libexpat1-dev`
- Installation may be eased by using the `fontconfig-dlopen` feature. This means
  that `libfontconfig` is used at runtime, and not found and linked at link
  time.
```

```admonish tip title="CUDA (for accelerated sky modelling)"
- Only required if either the `cuda` or `cuda-single` feature is enabled
- Requires a [CUDA-capable device](https://developer.nvidia.com/cuda-gpus)
- Arch: `cuda`
- Ubuntu and others: [Download link](https://developer.nvidia.com/cuda-zone)
- The library dir can be specified manually with `CUDA_LIB`
  - If not specified, `/usr/local/cuda` and `/opt/cuda` are searched.
- Can link statically; use the `cuda-static` or `all-static` features.
```

## Installing Rust

~~~admonish tip title="TL;DR"
```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
~~~

`hyperdrive` is written in [Rust](https://www.rust-lang.org/), so a Rust
environment is required. [The Rust
book](https://doc.rust-lang.org/book/ch01-01-installation.html) has excellent
information to do this. Similar, perhaps more direct information is
[here](https://www.rust-lang.org/tools/install).

**Do not** use `apt` to install Rust components.

## Cloning the `hyperdrive` repo

```shell
git clone https://github.com/MWATelescope/mwa_hyperdrive
```

Don't forget to `cd` into the repo:

```shell
cd mwa_hyperdrive
```

## Compiling `hyperdrive`

```shell
cargo install --path . --locked
```

This will install `hyperdrive` to `~/.cargo/bin/hyperdrive`. This binary can be
moved anywhere and it will still work. The installation destination can be
changed by setting `CARGO_HOME`.

~~~admonish danger title="CUDA"
Do you have a CUDA-capable NVIDIA GPU? Ensure you have installed
[CUDA](https://developer.nvidia.com/cuda-zone) (instructions are above), find
your CUDA device's compute capability
[here](https://developer.nvidia.com/cuda-gpus) (e.g. Geforce RTX 2070 is 7.5),
and set a variable with this information (note the lack of a period in the
number):

```shell
export HYPERDRIVE_CUDA_COMPUTE=75
```

Now you can compile `hyperdrive` with CUDA enabled (single-precision floats):

```shell
cargo install --path . --locked --features=cuda-single
```

If you're using "datacentre" products (e.g. a V100 available on the
Pawsey-hosted supercomputer "garrawarla"), you probably want double-precision
floats:

```shell
cargo install --path . --locked --features=cuda
```

You can still compile with double-precision on a desktop GPU, but it will be
much slower than single-precision.
~~~

~~~admonish tip title="Static dependencies"
The aforementioned C libraries can each be compiled by `cargo`. `all-static`
will statically-link all dependencies (including CUDA, if CUDA is enabled) such
that **you need not have these libraries available to use `hyperdrive`**.
Individual dependencies can be statically compiled and linked, e.g.
`cfitsio-static`. See the dependencies list above for more information.
~~~

~~~admonish info title="Multiple features"
`cargo` features can be chained in a comma-separated list:

```shell
cargo install --path . --locked --features=cuda,all-static
```
~~~

~~~admonish help title="Troubleshooting"
If you're having problems compiling, it's possible you have an older Rust
toolchain installed. Try updating it:

```shell
rustup update
```

If that doesn't help, try cleaning the local build directories:

```shell
cargo clean
```

and try compiling again. If you're still having problems, raise a [GitHub
issue](https://github.com/MWATelescope/mwa_hyperdrive/issues) describing your
system and what you've tried.
~~~
