#!/bin/bash

set -eux

# This should be run in the mwa_hyperdrive_cuda project directory.

bindgen src_cuda/vis_gen.h > src/bindings.rs
