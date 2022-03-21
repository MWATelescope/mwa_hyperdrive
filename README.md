# mwa_hyperdrive

<div class="bg-gray-dark" align="center" style="background-color:#24292e">
<img src="mdbook/src/hyperdrive.png" height="200px" alt="hyperdrive logo">
<br/>
<img src="https://github.com/MWATelescope/mwa_hyperdrive/workflows/Tests/badge.svg" alt="Linux%20tests">
<a href="https://codecov.io/gh/MWATelescope/mwa_hyperdrive">
  <img src="https://codecov.io/gh/MWATelescope/mwa_hyperdrive/branch/main/graph/badge.svg?token=FSRY8T0G0R"/>
</a>
</div>

Calibration software for the Murchison Widefield Array (MWA) radio telescope.

Currently in heavy development. Major milestones are listed below:

- [x] Direction-independent calibration (on both CPUs and GPUs)
  - [x] Single timeblock
  - [x] Multiple timeblocks
  - [ ] Frequency averaging before calibration
  - [ ] Chunking for memory constrained calibration
- [x] Reading visibilities from raw MWA data (legacy and MWAX), uvfits and
      measurement sets
- [x] Reading and writing RTS, AOcal and WODEN style sky-model source lists
- [x] Reading and writing RTS, AOcal and hyperdrive style calibration
      solutions
- [x] Writing visibilities to uvfits and measurement sets
- [ ] Direction-dependent calibration
- [ ] Writing to multiple uvfits or measurement sets for each MWA coarse channel

## Installation
See the instructions
[here](https://MWATelescope.github.io/mwa_hyperdrive/installation/intro.html).

## Usage

A comprehensive guide for usage can be found
[here](https://MWATelescope.github.io/mwa_hyperdrive/user/intro.html).

## Troubleshooting

Run `hyperdrive` again, but this time with the debug build:

    cargo build
    ./target/debug/hyperdrive <your settings here>

See the instructions
[here](https://MWATelescope.github.io/mwa_hyperdrive/installation/from_source.html)
to build from source. Report the version of the software used, your usage and
the program output in a [new GitHub
issue](https://github.com/MWATelescope/mwa_hyperdrive/issues).

## FAQ

- Naming convention

  This project is called `mwa_hyperdrive` so that it cannot be confused with
  other projects involving the word `hyperdrive`, but the binary is called
  `hyperdrive` to help users' fingers.

- `hyperdrive`?

  [HYPERDRIVE!](https://youtu.be/Mf4_LB32M6Q)
