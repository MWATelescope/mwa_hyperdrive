# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic
Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - Unreleased
### Changed
- Use "lto = "thin"". This makes performance slightly better and decreases the
  size of the binary.
- Internal crates have been "folded" in with the main code. This should
  hopefully make editing the code a bit simpler.
