# mwa_hyperdrive

Calibration software for the Murchison Widefield Array (MWA) radio telescope.

Currently in heavy development. Aims to provide feature parity with and exceed
the MWA Real-Time System (RTS).

## Usage
The MWA_BEAM_FILE environment variable must be set and gives the path to the MWA
FEE beam HDF5 file. See the README for
[`hyperbeam`](https://github.com/MWATelescope/mwa_hyperbeam) for more info.

### Calibration
Work in progress!

### Source lists
See the README in the `srclist` directory.

### Visibility Simulation
<details>

`hyperdrive` can simulate MWA visibilities from a source catalogue, similar to
Jack Line's [`WODEN`](https://github.com/JLBLine/WODEN), although `WODEN` should
be instead of `hyperdrive` for this purpose.

The help text containing all possible options can be seen with:

    hyperdrive simulate-vis -h

While all options can be specified as command line arguments, they may also be
specified as `.toml` or `.json` parameter files, e.g.

`hyperdrive.toml`
```toml
source_list = "/home/chj/wodanlist_VLA-ForA.txt"
metafits = "/home/chj/1102865128.metafits"
ra = 50.67
dec = -37.20
fine_channel_width = 80
bands = [1,2,3]
steps = 14
time_res = 8.0
```

`hyperdrive.json`
```json
{
    "source_list": "/home/chj/wodanlist_VLA-ForA.txt",
    "metafits": "/home/chj/1102865128.metafits",
    "ra": 50.67,
    "dec": -37.20,
    "fine_channel_width": 80,
    "num_bands": 3,
    "steps": 14,
    "time_res": 8.0
}
```

Run with:

    hyperdrive simulate-vis </path/to/param/file.toml_or_json>

Any command-line arguments specified alongside a parameter file will *override*
the parameter file's settings.

#### Verification
To check if arguments could be used with `simulate-vis`, one should check with
the dry run option:

    hyperdrive simulate-vis --dry-run <args>

If valid, this routine will print out the args as well as the observation
context from the metafits file.

</details>

## Installation
<details>

### Prerequisites
<details>

- An NVIDIA GPU with compute capability >=2. See this
  [list](https://developer.nvidia.com/cuda-gpus) to determine what compute
  capability a GPU has.

- A Rust compiler with a version >= 1.53.0

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

- [CUDA](https://developer.nvidia.com/cuda-zone)
  - Arch: `cuda`
  - The library dir can be specified manually with CUDA_LIB
  - If not specified, `/usr/local/cuda` and `/opt/cuda` are searched.

To compile a library statically, use e.g. `ERFA_STATIC=1`. To compile all
libraries statically, use `PKG_CONFIG_ALL_STATIC=1`.

Memory requirements can't be specified yet, as the code is still in development.
</details>

### Hyperdrive-specific instructions
<details>

- Specify your GPU's compute capability

    Export the `HYPERDRIVE_CUDA_COMPUTE` environment variable with your
    compute-capability number, e.g.

    `export HYPERDRIVE_CUDA_COMPUTE=75`

- Compile the source

    `cargo build --release`

    You may need to specify additional compiler options, depending on your
    setup. For example, CUDA can only use certain versions of GCC, so the
    following might be needed before running `cargo build`:

    `export CXX=/usr/bin/g++-5`

    It's also possible to specify environment variables temporarily:

    `CXX=/usr/bin/g++-5 HYPERDRIVE_CUDA_COMPUTE=75 cargo build --release`

- Run the compiled binary

    `./target/release/hyperdrive -h`

    A number of subcommands should present themselves, and the help text for
    each command should clarify usage.

    On the same system, the `hyperdrive` binary can be copied and used
    anywhere you like!
</details>
</details>

## Troubleshooting

Run `hyperdrive` again, but this time with the debug build:

    cargo build
    ./target/debug/hyperdrive <your settings here>

Report the version of the software used, your usage and the program output in a
new GitHub issue.

## FAQ
- Naming convention

    This project is called `mwa_hyperdrive` so that it cannot be confused with
    other projects involving the word `hyperdrive`, but the binary is called
    `hyperdrive` to help users' fingers.

- `hyperdrive` source list format

    See the README in the `mwa_hyperdrive_srclist` directory.
