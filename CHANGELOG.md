# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic
Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - Unreleased
### Added
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
