# Installing with EveryBeam support

`hyperdrive` can optionally use
[EveryBeam](https://git.astron.nl/RD/EveryBeam) for beam responses. This
allows beam models other than the MWA FEE beam to be used, e.g. SKA-Low,
LOFAR and OSKAR-simulated arrays; see [EveryBeam beam
responses](../defs/beam.md#everybeam) for usage.

There are two ways to get EveryBeam support:

1. [Building from source with a vendored EveryBeam](#from-source-vendored-everybeam)
   (recommended); or
2. [Building from source against an installed EveryBeam](#from-source-installed-everybeam).

~~~admonish warning title="Licensing"
EveryBeam is licensed under the GPL-3.0, whereas `hyperdrive` is licensed under
the MPL-2.0. `hyperdrive` binaries that include EveryBeam are therefore subject
to the GPL-3.0. Binaries without EveryBeam (the default) are unaffected.
~~~

## From source: vendored EveryBeam

With the `everybeam-vendored` feature, the build downloads EveryBeam and
casacore (and some header-only dependencies), verifies them, builds them and
links them statically. Nothing needs to be installed except some common system
packages:

| Distribution | Packages |
| ------------ | -------- |
| Debian/Ubuntu | `build-essential cmake curl git gfortran flex bison libboost-dev libhdf5-dev libfftw3-dev libgsl-dev libblas-dev liblapack-dev` |
| Fedora/RHEL | `gcc-c++ gcc-gfortran cmake curl git flex bison boost-devel hdf5-devel fftw-devel gsl-devel blas-devel lapack-devel` |

(plus `hyperdrive`'s usual [dependencies](from_source.md)). On RHEL-like
systems, `hdf5-devel` comes from EPEL and the BLAS/LAPACK packages from the
PowerTools/CRB repository, and a newer GCC (e.g. `gcc-toolset-13`) is needed on
RHEL 8, because EveryBeam needs a C++20 compiler (GCC 10 or later). Then:

```shell
# From a clone of the hyperdrive repo:
cargo install --path . --locked --features everybeam-vendored
```

~~~admonish info title="Build time"
The first build compiles casacore and EveryBeam, which adds about 7 minutes on
4 cores (casacore is most of that; it's faster with more cores). These are
built inside cargo's target directory (using about 2.5 GB of disk space), so
later `cargo build`s don't rebuild them; see [Cleaning up](#cleaning-up-after-a-build).
To avoid rebuilding them for every `cargo install`:
- use a persistent target directory, e.g.
  `CARGO_TARGET_DIR=~/.cache/hyperdrive-target cargo install ...`; and/or
- use [sccache](https://github.com/mozilla/sccache) before building:
  ```shell
  # EveryBeam (and other C/C++ code compiled by cargo) uses RUSTC_WRAPPER;
  # casacore (built with CMake) uses the CMAKE_*_COMPILER_LAUNCHER variables.
  export RUSTC_WRAPPER=sccache CMAKE_C_COMPILER_LAUNCHER=sccache \
      CMAKE_CXX_COMPILER_LAUNCHER=sccache CMAKE_Fortran_COMPILER_LAUNCHER=sccache
  ```
~~~

For offline builds (or mirrors), any of the downloaded sources can instead be
supplied as an extracted directory with `EVERYBEAM_SYS_<NAME>_SRC`, where
`<NAME>` is one of `CASACORE`, `EVERYBEAM`, `AOCOMMON`, `SKA_SDP_FUNC`, `XTL`,
`XTENSOR` or `EIGEN`; see `crates/everybeam-sys/build.rs` for the versions.

EveryBeam's element-response coefficients (e.g. for SKA-Low and LOFAR element
models) are embedded in the binary, and are used automatically.

### Without root: conda

The system packages can instead come from
[conda-forge](https://conda-forge.org/) (e.g. with
[miniforge](https://github.com/conda-forge/miniforge)):

```shell
conda create -n hyperdrive-eb -c conda-forge \
    c-compiler cxx-compiler fortran-compiler cmake make pkg-config flex bison \
    libboost-devel "hdf5=1.14" fftw gsl libblas liblapack cfitsio \
    expat git curl
conda activate hyperdrive-eb

# Use conda's compilers and libraries (the compiler packages don't always set
# these on activation).
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++
export FC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gfortran
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=$CC
export HDF5_DIR=$CONDA_PREFIX PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig
# Find conda's libraries at runtime without LD_LIBRARY_PATH.
export RUSTFLAGS="-C link-arg=-Wl,-rpath,$CONDA_PREFIX/lib"

cargo install --path . --locked --features everybeam-vendored
```

HDF5 is pinned to 1.14 because the HDF5 bindings used by `hyperdrive` don't yet
support HDF5 2. Rust itself can be installed without root with
[rustup](https://rustup.rs/). The resulting binary uses libraries from the conda
environment, so keep the environment (it doesn't need to be activated to run
`hyperdrive`).

## From source: installed EveryBeam

If EveryBeam (>= 0.9) and casacore (>= 3.6) are already installed, use the
`everybeam` feature:

```shell
# Only needed if they aren't in $CONDA_PREFIX, /opt/everybeam, /opt/casacore,
# /usr/local or /usr:
export EVERYBEAM_DIR=/path/to/everybeam/prefix
export CASACORE_DIR=/path/to/casacore/prefix

cargo install --path . --locked --features everybeam
```

The binary records where the libraries are (an rpath), so `LD_LIBRARY_PATH`
isn't needed. Other build-time environment variables are `EVERYBEAM_CXXFLAGS`
(extra C++ compiler flags) and `EVERYBEAM_LIBS` (extra libraries to link).

<details>
<summary>Installing casacore and EveryBeam from source</summary>

At the time of writing, Ubuntu and Debian only package casacore 3.5, so
casacore must be built from source. EveryBeam needs a C++20 compiler (e.g.
GCC 10 or later).

```shell
sudo apt install -y build-essential cmake git wget gfortran flex bison \
    libboost-all-dev libhdf5-dev libfftw3-dev libblas-dev liblapack-dev \
    libcfitsio-dev wcslib-dev libgsl-dev casacore-data

git clone --depth 1 --branch v3.8.2 https://github.com/casacore/casacore.git
mkdir casacore/build && cd casacore/build
cmake .. -DCMAKE_INSTALL_PREFIX=/opt/casacore -DCMAKE_BUILD_TYPE=Release \
    -DMODULE=ms -DBUILD_PYTHON=OFF -DBUILD_PYTHON3=OFF -DBUILD_TESTING=OFF \
    -DBUILD_SISCO=OFF -DDATA_DIR=/usr/share/casacore/data
make -j"$(nproc)" install
cd ../..

git clone --recursive --branch v0.9.0 https://git.astron.nl/RD/EveryBeam.git
mkdir EveryBeam/build && cd EveryBeam/build
cmake .. -DCMAKE_INSTALL_PREFIX=/opt/everybeam -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=/opt/casacore
make -j"$(nproc)" install
```

See [EveryBeam's build
instructions](https://everybeam.readthedocs.io/en/latest/build-instructions.html)
for more options. Don't build against the `everybeam` Python wheels on PyPI;
they contain no headers and use a renamed casacore namespace.
</details>

## Cleaning up after a build

What can be removed depends on how `hyperdrive` was built:

- **`cargo install`** builds in a temporary directory that cargo deletes
  afterwards, so nothing is left behind (unless `CARGO_TARGET_DIR` was set; then
  delete that directory when it's no longer needed).
- **From a clone of the repo** (`cargo build`), the vendored casacore and
  EveryBeam builds live in the `target` directory (about 2.5 GB per build
  profile and feature set). Remove just those with
  ```shell
  cargo clean -p everybeam-sys
  ```
  or everything with `cargo clean`. The next `everybeam-vendored` build then
  downloads and compiles them again.
- **sccache**, if used, keeps compiled objects in its own cache directory
  (`~/.cache/sccache` by default); stop its server with `sccache --stop-server`
  and delete that directory to clear it.
- **An installed EveryBeam/casacore** (the manual route above): the source and
  `build` directories can be deleted once `make install` has finished. To
  uninstall, delete the installation prefixes (e.g. `/opt/everybeam` and
  `/opt/casacore`).
- **conda**: `conda clean --all` removes conda's download cache. The
  environment itself is needed at runtime by a `hyperdrive` built in it; remove
  it (`conda env remove -n hyperdrive-eb`) only along with that `hyperdrive`.

At runtime, `hyperdrive` may write a few small files to the temporary directory
(`$TMPDIR`, usually `/tmp`): `hyperdrive-everybeam-<version>-data*` (EveryBeam's
coefficients, for vendored builds). These can be deleted at any time; they're
recreated when needed.

## Runtime: casacore's measures data

casacore needs its "measures" data (leap seconds, Earth-orientation tables) to
convert coordinates; without it, EveryBeam fails with errors mentioning
`TAI_UTC` or `IERS`. Download
ASTRON's copy (no root needed) and tell casacore where it is:

```shell
mkdir -p ~/casacore-data
curl -fL https://www.astron.nl/iers/WSRT_Measures.ztar -o WSRT_Measures.ztar
tar -xf WSRT_Measures.ztar -C ~/casacore-data && rm WSRT_Measures.ztar
echo "measures.directory: $HOME/casacore-data" >> ~/.casarc
```

The Earth-orientation tables are updated regularly, so re-download them every
few months for the most accurate coordinate conversions.

~~~admonish tip title="Distribution packages"
Debian/Ubuntu's `casacore-data` package keeps the leap-second table separately
(in `/var/lib/casacore/data`), which casacore built from source (including
the vendored build) doesn't look for, so ASTRON's copy is simpler. A casacore
installed from a distribution or conda package knows where its own data are.
~~~

## Check that it worked

```shell
hyperdrive di-calibrate --help | grep -i everybeam
```

should list `everybeam` as a beam type, along with the `BEAM (EVERYBEAM)`
options. From a clone of the repo, the EveryBeam tests can be run with e.g.

```shell
cargo test --features everybeam-vendored --lib beam::everybeam
cargo test --features everybeam-vendored --test integration_tests everybeam
```
