# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic
Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - Unreleased
### Added
- A --ignore-input-solutions-tile-flag argument for `solutions-apply`.
- Debug-level messages stating which tiles are being flagged and why in
  `solutions-apply`.

### Fixed
- `solutions-apply` would incorrectly apply solutions when flagged tiles were
  present. Thorough testing suggests that this is no longer the case and cannot
  happen again.
- Birli and possibly cotter measurement set metadata weren't being read
  correctly.
- Bugs were fixed surrounding the reading of RTS solutions.

### Changed
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
