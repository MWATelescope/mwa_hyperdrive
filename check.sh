#!/usr/bin/env bash
set -e

echo Cleaning
cargo clean

echo Upadting
cargo update --verbose

echo Cargo check...
cargo check --features=all-static,cuda,gpu-single  --all-targets

echo Cargo clippy...
cargo clippy --features=all-static,cuda,gpu-single --all-targets -- -D warnings

echo Cargo fmt...
cargo fmt --check

echo Done
