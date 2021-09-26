#!/bin/bash

# Update the Rust bindings to CUDA code (via a header). This script must be run
# whenever the CUDA code changes.

# This script requires bindgen. This can be provided by a package manager or
# installed with "cargo install bindgen".

# https://stackoverflow.com/questions/4774054/reliable-way-for-a-bash-script-to-get-the-full-path-to-itself
SCRIPTPATH="$(cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P)"

bindgen "${SCRIPTPATH}"/src_cuda/model.h \
    --allowlist-function "model_.*" \
    --blocklist-type "Addresses" \
    --blocklist-type "__int8_t" \
    --allowlist-type "RADec" \
    --allowlist-type "UVW" \
    --allowlist-type "LMN" \
    --allowlist-type "ShapeletCoeff" \
    --allowlist-type "ShapeletUV" \
    --allowlist-type "Jones.*" \
    --allowlist-type "Points" \
    --allowlist-type "Gaussians" \
    --allowlist-type "Shapelets" \
    --allowlist-var "POWER_LAW_FD_REF_FREQ" \
    --size_t-is-usize \
    > "${SCRIPTPATH}"/src/model_double.rs

bindgen "${SCRIPTPATH}"/src_cuda/model.h \
    --allowlist-function "model_.*" \
    --blocklist-type "Addresses" \
    --blocklist-type "__int8_t" \
    --allowlist-type "RADec" \
    --allowlist-type "UVW" \
    --allowlist-type "LMN" \
    --allowlist-type "ShapeletCoeff" \
    --allowlist-type "ShapeletUV" \
    --allowlist-type "Jones.*" \
    --allowlist-type "Points" \
    --allowlist-type "Gaussians" \
    --allowlist-type "Shapelets" \
    --allowlist-var "POWER_LAW_FD_REF_FREQ" \
    --size_t-is-usize \
    -- -D SINGLE \
    > "${SCRIPTPATH}"/src/model_single.rs

bindgen "${SCRIPTPATH}"/src_cuda/memory.h \
    --blocklist-function "model_.*" \
    --blocklist-type "__uint64_t" \
    --allowlist-function "init_model" \
    --allowlist-function "copy_vis" \
    --allowlist-function "clear_vis" \
    --allowlist-function "destroy" \
    --blocklist-type "UVW" \
    --blocklist-type "LMN" \
    --blocklist-type "JonesF.*" \
    --allowlist-type "Addresses" \
    --size_t-is-usize \
    > "${SCRIPTPATH}"/src/memory_double.rs

bindgen "${SCRIPTPATH}"/src_cuda/memory.h \
    --blocklist-function "model_.*" \
    --blocklist-type "__uint64_t" \
    --allowlist-function "init_model" \
    --allowlist-function "copy_vis" \
    --allowlist-function "clear_vis" \
    --allowlist-function "destroy" \
    --blocklist-type "UVW" \
    --blocklist-type "LMN" \
    --blocklist-type "JonesF.*" \
    --allowlist-type "Addresses" \
    --size_t-is-usize \
    -- -D SINGLE \
    > "${SCRIPTPATH}"/src/memory_single.rs
