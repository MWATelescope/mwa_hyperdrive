# Installing `hyperdrive` from source code

## Dependencies
`hyperdrive` needs a C and a C++ compiler (some of its dependencies, such as
the measurement set code, are compiled from source) and depends on these C
libraries:


```admonish example title="[cfitsio](https://heasarc.gsfc.nasa.gov/docs/software/fitsio/)"
- Ubuntu: `libcfitsio-dev`
- Arch: `cfitsio`
- Library and include dirs can be specified manually with `CFITSIO_LIB` and
  `CFITSIO_INC`
  - If not specified, `pkg-config` is used to find the library.
- Can compile statically; use the `cfitsio-static` or `all-static` features.
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

```admonish tip title="CUDA (for accelerated sky modelling with NVIDIA GPUs)"
- Only required if the `cuda` feature is enabled
- Requires a [CUDA-capable device](https://developer.nvidia.com/cuda-gpus)
- Arch: `cuda`
- Ubuntu and others: [Download link](https://developer.nvidia.com/cuda-zone)
- The installation dir(s) can be specified manually with `CUDA_LIBRARY_PATH`
  - `/opt/cuda`, `/usr/local/cuda` and `/usr/local/cuda-*` are also searched.
- Can link statically; use the `cuda-static` or `all-static` features.
```

```admonish tip title="HIP (for accelerated sky modelling with AMD GPUs)"
- Only required if the `hip` feature is enabled
- Requires a [HIP-capable device](https://docs.amd.com/en/latest/release/gpu_os_support.html) (N.B. This seems to be incomplete)
- Arch:
  - See [https://wiki.archlinux.org/title/GPGPU#ROCm](https://wiki.archlinux.org/title/GPGPU#ROCm)
  - It is possible to get pre-compiled products from the [arch4edu repo](https://github.com/arch4edu/arch4edu).
- Ubuntu and others: install [ROCm](https://rocm.docs.amd.com/), which
  includes HIP. ROCm 6 and 7 are known to work.
- The installation dir (the one containing `bin/hipcc`, or its `hip`
  subdirectory) can be specified manually with `HIP_PATH`
  - If not specified, the location found by the `hip-sys` crate is used.
- N.B. Despite HIP installations being able to run HIP code on NVIDIA GPUs,
  this is not supported by `hyperdrive`; please compile with the CUDA
  instructions above.
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

`hyperdrive` needs Rust version 1.88 or newer. `rustup update` gets the latest
version.

## Installing `hyperdrive` from crates.io

```shell
cargo install mwa_hyperdrive --locked
```

If you want to download the source code and install it yourself, read on.

## Manually installing from the `hyperdrive` repo

~~~admonish tip title="Optimal Performance"
(optional) Use native CPU features (not portable!)
```shell
export RUSTFLAGS="-C target-cpu=native"
```
~~~

Clone the git repo and point `cargo` to it:

```shell
git clone https://github.com/MWATelescope/mwa_hyperdrive
cargo install --path mwa_hyperdrive --locked
```

This will install `hyperdrive` to `~/.cargo/bin/hyperdrive`. This binary can be
moved anywhere and it will still work. The installation destination can be
changed by setting `CARGO_HOME`.

~~~admonish danger title="Further optimisation"
It is possible to compile with more optimisations if you give `--profile
production` to the `cargo install` command. This may make things a few percent
faster, but compilation will take much longer.
~~~

~~~admonish danger title="CUDA"
Do you have a CUDA-capable NVIDIA GPU? Ensure you have installed
[CUDA](https://developer.nvidia.com/cuda-zone) (instructions are above).

By default, the GPU code is compiled for every compute capability that your
CUDA installation supports. To compile for only your device (which is
quicker), find its compute capability
[here](https://developer.nvidia.com/cuda-gpus) (e.g. Geforce RTX 2070 is 7.5),
and set a variable with this information (note the lack of a period in the
number; more than one can be given, separated by commas):

```shell
export HYPERDRIVE_CUDA_COMPUTE=75
```

Now you can compile `hyperdrive` with CUDA enabled (single-precision floats):

```shell
cargo install --path . --locked --features=cuda,gpu-single
```

If you're using "datacentre" products (e.g. a V100 or an A100), you probably
want double-precision floats:

```shell
cargo install --path . --locked --features=cuda
```

You can still compile with double-precision on a desktop GPU, but it will be
much slower than single-precision.

If you get a compiler error, it may be due to a compiler mismatch. CUDA releases
are compatible with select versions of `gcc`, so it's important to keep the CUDA
compiler happy. You can select a custom C++ compiler with the `CXX` variable,
e.g. `CXX=/opt/cuda/bin/g++`. If `CXX` is not set and `$CUDA_PATH/bin/g++`
exists, it is used.
~~~

~~~admonish danger title="HIP"
Do you have a HIP-capable AMD GPU? Ensure you have installed HIP (instructions
are above), and compile with the `hip` feature (single-precision floats):

```shell
cargo install --path . --locked --features=hip,gpu-single
```

If you're using "datacentre" products (e.g. the GPUs on the "setonix"
supercomputer), you probably want double-precision floats:

```shell
cargo install --path . --locked --features=hip
```

You can still compile with double-precision on a desktop GPU, but it will be
much slower than single-precision.

If you are encountering problems, you may need to set your `HIP_PATH` variable.

The GPU architecture to compile for can be set with `HYPERDRIVE_HIP_ARCH`; its
value is given to `hipcc` as `--offload-arch`, e.g.

```shell
export HYPERDRIVE_HIP_ARCH=gfx90a
```
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

```admonish info title="Changes from older versions"
`hyperdrive` used to depend on the [ERFA](https://github.com/liberfa/erfa) C
library. It now uses a pure-Rust equivalent.
```
