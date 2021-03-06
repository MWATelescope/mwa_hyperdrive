# mwa_hyperdrive

Calibration software for the Murchison Widefield Array (MWA) radio telescope.

Currently in heavy development. Aims to provide feature parity with and exceed
the MWA Real-Time System (RTS).

## Usage
### Visibility Simulation
`hyperdrive` can simulate MWA visibilities from a source catalogue, similar to
Jack Line's [WODEN](https://github.com/JLBLine/WODEN).

The help text containing all possible options can be seen with:

    mwa_hyperdrive simulate-vis -h

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

    mwa_hyperdrive simulate-vis </path/to/param/file.toml_or_json>

Any command-line arguments specified alongside a parameter file will *override*
the parameter file's settings.

### Source list validation
To check if a source list is compatible with `hyperdrive`, the following can be
used:

    mwa_hyperdrive verify-srclist </path/to/srclist1> </path/to/srclist2>

## Installation
- Prerequisites

    - A Rust compiler with a version >= 1.42.0
      - Instructions for installing Rust are below
    - [cfitsio](https://heasarc.gsfc.nasa.gov/docs/software/fitsio/)
    - [CUDA](https://developer.nvidia.com/cuda-zone)
      - As well as a GPU with compute capability >=2

    Memory requirements can't be specified yet, as the code is still in
    development.

- Install Rust

    `https://www.rust-lang.org/tools/install`

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

    `./target/release/mwa_hyperdrive -h`

    A number of subcommands should present themselves, and the help text for
    each command should clarify usage.

    On the same system, the `mwa_hyperdrive` binary can be copied and used
    anywhere you like!

## Troubleshooting

Run `mwa_hyperdrive` again, but this time with the debug build:

    cargo build
    ./target/debug/mwa_hyperdrive <your settings here>

Report the version of the software used, your usage and the program output in a
new GitHub issue.
