# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic
Versioning](https://semver.org/spec/v2.0.0.html).

## unreleased

### Fixed

- [#59](https://github.com/MWATelescope/mwa_hyperdrive/issues/59) multi-timeblock calibration.

### Removed

- deleted unused `io::read::uvfits::error::FitsError` type

### Changed

- vis-simulate now supports auto-correlations by default, use `--output-no-autos`
  to disable
- VisRead trait generalizes `read_{crosses,autos,crosses_and_autos}` with `read_inner_dispatch`

## [0.6.1] - 2025-07-29

### Fixed

- tapered weights were accidentally written out in peeling, flagging short baselines since v0.5.0

## [0.6.0] - 2025-07-28

### Added

- auto-correlation visibility simulation and writing support

### Changed

- add oversampled pfb gains from birli 0.18, these are not the default! For phase3 data, use `--pfb-flavour jake_oversampled`
- cli args `--sources-to-subtract` and `--invert` from vis-subtract are now
  included in all source list reading commands, behaviour is slighly different:
    - `--sources-to-subtract` renamed to `--named-sources`
    - `--invert` now requires `--named-sources` to be present
- remove default `--source-dist-cutoff`.

### Fixed

- [#29](https://github.com/MWATelescope/mwa_hyperdrive/issues/29) confusing syntax
  around named source filtering and inversion.
- [#47](https://github.com/MWATelescope/mwa_hyperdrive/issues/47) `--source-dist-cutoff`
  was too small, default removed.

## [0.5.1] - 2025-04-11

### Changed

- updated mwalib@1.8.7, marlu@0.16.1, birli@0.17.1 and mwa_hyperbeam@0.10.2

## [0.5.0] - 2025-02-05

### Added

- `peel` subcommand, see: <https://mwatelescope.github.io/mwa_hyperdrive/user/peel/intro.html>
- `SkyModeller::update_with_a_source`
- `CalVis::scale_by_weights`
- `DevicePointer::copy_to`
- `SourceList::search` and `search_asec`

### Changed

- `srclist::types::components::ComponentList::new` now takes an iterator over `SourceComponent` references
- `di_calibrate::calibrate_timeblock` is now public to the crate

## [0.4.2] - 2024-11-30

### Fixed

- fixed averaging issue #41

## [0.4.1] - 2024-07-31

### Added

- hyperbeam@0.9.3

### Fixed

- fix a compile error when specifying env `HIP_FLAGS` with `--features=hip`
- fix #34 , a compile error for non-x86 CPUs (thanks @cjordan )

## [0.4.0] - 2024-06-19

### Added

- fits sourcelist support (including shapelets for Jack-style fits)
- hyperbeam@0.9.2 built@0.7 marlu@0.11.0 mwalib@1.3.3 birli@0.11.0

### Fixed

- rocm6 support
- a bunch of really nasty segfaults that took a big toll on my sanity
- Huge thanks to @robotopia for fixing <https://github.com/MWATelescope/mwa_hyperbeam/issues/9>
  via hyperbeam 0.9.0
- performance optimizations in hyperbeam 0.9.2

## [0.3.0] - 2023-09-27

### Added

- Support for HIP, which allows AMD GPUs to be used instead of only NVIDIA GPUs
  via CUDA.
- Support for the "DipAmps" column in a metafits file. This allows users to
  control dipole gains in beam code.
- Support for averaging incoming visibilities in time and frequency *before*
  doing any work on them.
- When writing out visibilities, it is now possible to write out the smallest
  contiguous band of unflagged channels.
- Plots can be written to a specific directory, not only the CWD. Fixes #18.
- Support for visibilities using the ant2-ant1 ordering rather than ant1-ant2.
- Add new errors
  - If all baselines are flagged due to UVW cutoffs, then the program exits with
    this report.
  - If all sources are vetoed from a source list, then the program exits with
    this report.
- Benchmarks
  - Raw MWA, uvfits and measurement set reading.
  - More CUDA benchmarks for the modelling code.
- Support for "argument files". This is an advanced feature that most users
  probably should avoid. Previously, argument files were supported for the
  di-calibrate subcommand, but now it is more consistently supported among the
  "big" subcommands.

### Fixed

- When raw MWA data is missing gpubox files in what is otherwise a contiguous
  spectral window, it is no longer treated as a "picket fence" situation.
- Flux densities were not being correctly estimated for curved-power-law
  components with GPU code.
- AO source lists were always read like they were version 1.1; version 1.0 is
  now read properly.
- RTS source lists with multiple SHAPELET components are now removed properly
  (not to be confused with the favoured SHAPELET2 component type).
- Some MWA raw data observations were not handled correctly due to a
  floating-point error.
- MWA raw data observations with all flagged tiles weren't allowed to be used,
  even with the explicit "ignore input data tile flags".
- Some aspects of hyperdrive weren't using user-specified array positions
  correctly. The help text also indicated the wrong units.
- Fine-channel flags and fine-channel-per-coarse channel flags are now checked
  for validity.

### Changed

- The performance of CPU visibility modelling has been dramatically improved.
- The command-line interface has been overhauled. Some things may be different,
  but generally the options and flags are much more consistent between
  subcommands.
- The preamble to "big" subcommands, like di-calibrate, has been overhauled to
  be much easier to read.
- Plotting
  - Legend labels have changed to $g_x$, $D_x$, $D_y$, $g_y$ ($g$ for gain, $D$
    for leakage). Thanks Jack Line.
  - The way that rows and columns are distributed when there are more than 128
    tiles has changed, but this can now also be controlled manually.
- Internal source list types can now be (de)serialised directly. This makes
  interfacing with YAML and JSON files is ~7% faster and simplifies the internal
  code.
- More error checks have been added to RTS, AO and WODEN source list reading.

## [0.2.1] - 2022-10-20

### Added

- `hyperdrive` can now be installed from crates.io
- A --ignore-input-solutions-tile-flag argument for `solutions-apply`.
- Debug-level messages stating which tiles are being flagged and why in
  `solutions-apply`.

### Fixed

- Until now, `hyperdrive` only supported raw MWA data with a frequency
  resolution of 10, 20 or 40 kHz. It now supports any resolution.
- When reading from uvfits/measurement set, a metafits' dipole information was
  applied, but perhaps in the wrong order. Order checks are now in place, but
  tile names must be consistent between the metafits and uvfits/MS.
- Picket fence observations were previously detected, but not still handled
  correctly. Until they are fixed, hyperdrive will not attempt to do anything
  with a picket fence observation. A doc page has been added for picket fences.
- `solutions-apply` would incorrectly apply solutions when flagged tiles were
  present. Thorough testing suggests that this is no longer the case and cannot
  happen again.
- Various improvements when reading measurement sets:
  - Birli and possibly cotter measurement set metadata weren't being read
    correctly.
  - MSs can't provide the array location, so the code assumes that all MSs were
    situated at the MWA. It is now possible to use a user-supplied array
    location to get around this.
  - `hyperdrive`-written MSs weren't being handled properly (oops)
- Bugs were fixed surrounding the reading of RTS solutions.

### Changed

- The library used for logging has changed. The user experience should only be
  superficial, but piping `hyperdrive` output to a file (e.g. a job output on a
  supercomputer) should display a little better.
- The output of `dipole-gains` is now less confusing.
- When applying solutions, if a baseline/tile doesn't have an appropriate
  solution available, the output Jones matrix is all zeros. This is to avoid
  confusion on whether the Jones matrix has been calibrated but is just flagged
  (in this case, not true), or has not been calibrated and is otherwise flagged.
- Stop capping spectral indices less than -2. Results from estimating flux
  densities may change.
- RTS solutions may no longer be written out.
- Use "lto = "thin"". This makes performance slightly better and decreases the
  size of the binary.
- Internal crates have been "folded" in with the main code. This should
  hopefully make editing the code a bit simpler.
