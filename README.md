# mwa_hyperdrive

Calibration software for the Murchison Widefield Array (MWA) radio telescope.

Currently in heavy development. Aims to provide feature parity with and exceed
the MWA Real-Time System (RTS).

## Installation
- Prerequisites

    - [cfitsio](https://heasarc.gsfc.nasa.gov/docs/software/fitsio/)
    - [CUDA](https://developer.nvidia.com/cuda-zone)
      - A GPU with compute capability >=2

    Memory requirements can't be specified yet, as the code is still in
    development.

- Install rust

    `https://www.rust-lang.org/tools/install`

- Specify your GPU's compute capability

    Export the `HYPERDRIVE_CUDA_COMPUTE` environment variable with your
    compute-capability number, e.g. `export HYPERDRIVE_CUDA_COMPUTE=75`

- Compile the source

    `cargo build --release`

    It's also possible to do:

    `HYPERDRIVE_CUDA_COMPUTE=75 cargo build --release`

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
