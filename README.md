# mwa_hyperdrive

<div class="bg-gray-dark" align="center" style="background-color:#24292e">
<img src="doc/hyperdrive.png" height="200px" alt="hyperdrive logo">
<br/>
<a href="https://docs.rs/crate/mwa_hyperdrive"><img src="https://docs.rs/mwa_hyperdrive/badge.svg" alt="docs"></a>
<img src="https://github.com/MWATelescope/mwa_hyperdrive/workflows/Tests/badge.svg" alt="Linux%20tests">
<a href="https://codecov.io/gh/MWATelescope/mwa_hyperdrive">
  <img src="https://codecov.io/gh/MWATelescope/mwa_hyperdrive/branch/main/graph/badge.svg?token=FSRY8T0G0R"/>
</a>
</div>

Calibration software for the Murchison Widefield Array (MWA) radio telescope.
Aims to provide feature parity with and exceed the MWA Real-Time System (RTS).

Currently in heavy development. Major milestones are listed below:

- [x] Direction-independent calibration (on both CPUs and GPUs)
- [x] Reading visibilities from raw MWA data (legacy and MWAX), uvfits and
      measurement sets
- [x] Reading and writing RTS, AOcal and WODEN style sky-model source lists
- [x] Reading and writing RTS, AOcal and hyperdrive style calibration
      solutions
- [x] Writing visibilities to uvfits
- [x] Writing visibilities to measurement sets
- [ ] Writing to multiple uvfits or measurement sets for each MWA coarse channel
- [ ] Direction-dependent calibration

## Usage

A comprehensive use guide can be found on the [wiki](https://github.com/MWATelescope/mwa_hyperdrive/wiki), but below are some TL;DR examples.

Many `hyperdrive` functions require the beam code to function. The MWA
FEE beam HDF5 file can be obtained with:

`wget http://ws.mwatelescope.org/static/mwa_full_embedded_element_pattern.h5`

Move the `h5` file anywhere you like, and put the file path in `MWA_BEAM_FILE`:

`export MWA_BEAM_FILE=/path/to/mwa_full_embedded_element_pattern.h5`

See the README for [`hyperbeam`](https://github.com/MWATelescope/mwa_hyperbeam)
for more info.

### DI Calibration

<details>
By default, only calibration solutions are written out (to a default filename):

    # -d is short for --data

    # Raw MWA data (MWAX or legacy)
    hyperdrive di-calibrate -d *ch???*.fits *gpubox*.fits *.metafits -s srclist.yaml

    # Measurement sets
    hyperdrive di-calibrate -d *.ms *.metafits -s srclist.yaml

    # uvfits
    hyperdrive di-calibrate -d *.uvfits *.metafits -s srclist.yaml

The output solutions file can be customised, and even multiple files of
different types written:

    # Using an alias to help keep the examples clear
    alias HYP_CAL="hyperdrive di-calibrate -d *.ms *.metafits -s srclist.yaml"
    # -o is short for --outputs
    HYP_CAL -o hyp_sol.bin hyp_sol.fits

The output could also be calibrated visibilities (this does not mean the
solutions can't be written out too):

    HYP_CAL -o hyp_cal.uvfits hyp_sol.bin

Output calibrated visibilities can be averaged in multiples:

    HYP_CAL -o hyp_cal.uvfits \
            --output-vis-time-average 4
            --output-vis-freq-average 2

or to a target resolution:

    HYP_CAL -o hyp_cal.uvfits \
            --output-vis-time-average 8s
            --output-vis-freq-average 80kHz

</details>

### Source lists

<details>
A number of sky-model source list utilities are available. At the time of
writing, the following subcommands are available (output edited for clarity):

    $ hyperdrive
    hyperdrive 0.2.0-alpha9
    ...

    SUBCOMMANDS:
        ...
        srclist-by-beam      Reduce a sky-model source list to the top N brightest sources, given pointing information
        srclist-convert      Convert a sky-model source list from one format to another
        srclist-shift        Shift the sources in a source list. Useful to correct for the ionosphere. The shifts must
                             be detailed in a .json file, with source names as keys associated with an "ra" and "dec"
                             in degrees. Only the sources specified in the .json are written to the output source list
        srclist-verify       Verify that sky-model source lists can be read by hyperdrive
        ...

Each of these subcommands have their own associated help, e.g.

    hyperdrive srclist-by-beam --help

Perhaps the most common operation is `srclist-by-beam`. This routine effectively
reduces an existing source list to the top `n` brightest sources given a
pointing and target frequencies (determined by a metafits file):

    hyperdrive srclist-by-beam \
               srclist_pumav3_EoR0aegean_fixedEoR1pietro+ForA_phase1+2.txt \
               -m *.metafits \
               -n 1000 \
               srclist_1000.yaml

</details>

### Visibility Simulation

<details>

`hyperdrive` can generate visibilities from a sky-model source list (output
visibilities are saved to a default filename):

    hyperdrive vis-simulate \
               -s srclist.yaml \
               -m *.metafits

Many options are available, but perhaps some of the more interesting ones are
being able to filter specific kinds of sky-model sources (`--filter-gaussians`
also available):

    hyperdrive vis-simulate \
               -s srclist.yaml \
               -m *.metafits \
               --filter-points \
               --filter-shapelets \
               -o model_gaussians.uvfits

</details>

## Installation

<details>

### Prerequisites

<details>

- An NVIDIA GPU with compute capability >=2. See this
  [list](https://developer.nvidia.com/cuda-gpus) to determine what compute
  capability a GPU has.

- A Rust compiler with a version >= 1.58.0

  `https://www.rust-lang.org/tools/install`

- [cfitsio](https://heasarc.gsfc.nasa.gov/docs/software/fitsio/)

  - Can compile statically; use the `cfitsio-static` or `all-static` features.
  - Ubuntu: `libcfitsio-dev`
  - Arch: `cfitsio`
  - Library and include dirs can be specified manually with `CFITSIO_LIB` and
    `CFITSIO_INC`
    - If not specified, `pkg-config` is used to find the library.

- [ERFA](https://github.com/liberfa/erfa)

  - Can compile statically; use the `erfa-static` or `all-static` features.
    - Requires a C compiler and `autoconf`.
  - Ubuntu: `liberfa-dev`
  - Arch: AUR package `erfa`
  - The library dir can be specified manually with `ERFA_LIB`
    - If not specified, `pkg-config` is used to find the library.

- [hdf5](https://www.hdfgroup.org/hdf5)
  - Can compile statically; use the `hdf5-static` or `all-static` features.
    - Requires `CMake` version 3.10 or higher.
  - Ubuntu: `libhdf5-dev`
  - Arch: `hdf5`
  - The library dir can be specified manually with `HDF5_DIR`
    - If not specified, `pkg-config` is used to find the library.

#### Optional dependencies

- [CUDA](https://developer.nvidia.com/cuda-zone)

  - Only needed if either the `cuda` or `cuda-single` feature is enabled
  - Can link statically; use the `cuda-static` or `all-static` features.
  - Arch: `cuda`
  - The library dir can be specified manually with `CUDA_LIB`
    - If not specified, `/usr/local/cuda` and `/opt/cuda` are searched.

- Calibration solutions plotting
  - Only needed if the `plotting` feature is enabled (which it is by default)
  - Arch: `pkg-config` `make` `cmake` `freetype2`
  - Ubuntu: `libfreetype-dev` `libexpat1-dev`

System libraries can also be linked statically. Use e.g. `ERFA_STATIC=1`. To
link all libraries statically, use `PKG_CONFIG_ALL_STATIC=1`.

</details>

### Hyperdrive-specific instructions

- Specify your GPU's compute capability

  Export the `HYPERDRIVE_CUDA_COMPUTE` environment variable with your
  compute-capability number, e.g.

  `export HYPERDRIVE_CUDA_COMPUTE=75`

- Compile the source

  `cargo install --path . --locked`

  To enable CUDA functionality, add `--features=cuda` or
  `--features=cuda-single` to the above command. If you're using a desktop
  NVIDIA GPU, you probably want the `cuda-single` feature.

  You may need to specify additional compiler options, depending on your
  setup. For example, CUDA can only use certain versions of GCC, so the
  following might be needed before running `cargo install`:

  `export CXX=/usr/bin/g++-5`

  It's also possible to specify environment variables temporarily:

  `CXX=/usr/bin/g++-5 HYPERDRIVE_CUDA_COMPUTE=75 cargo install --path . --locked`

- Run the compiled binary (you may need to include it in your `PATH`; see the
  output of `cargo install`)

  `hyperdrive -h`

  A number of subcommands should present themselves, and the help text for
  each subcommand should clarify usage.

  On the same system, the `hyperdrive` binary can be copied and used
  anywhere you like!

</details>

## Troubleshooting

Run `hyperdrive` again, but this time with the debug build (i.e. `cargo build`):

    cargo build
    ./target/debug/hyperdrive <your settings here>

Report the version of the software used, your usage and the program output in a
new GitHub issue.

## FAQ

- Naming convention

  This project is called `mwa_hyperdrive` so that it cannot be confused with
  other projects involving the word `hyperdrive`, but the binary is called
  `hyperdrive` to help users' fingers.
