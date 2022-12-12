#!/bin/bash

# Update the Rust bindings to CUDA code (via a header). This script must be run
# whenever the CUDA code changes.

# This script requires bindgen. This can be provided by a package manager or
# installed with "cargo install bindgen".

# https://stackoverflow.com/questions/4774054/reliable-way-for-a-bash-script-to-get-the-full-path-to-itself
SCRIPTPATH="$(
    cd -- "$(dirname "$0")" >/dev/null 2>&1
    pwd -P
)"

bindgen "${SCRIPTPATH}"/utils.h \
    --allowlist-function "get_cuda_device_info" \
    --size_t-is-usize \
    >"${SCRIPTPATH}"/utils_bindings.rs

bindgen "${SCRIPTPATH}"/model.h \
    --allowlist-function "model_.*" \
    --blocklist-type "__int8_t" \
    --allowlist-type "Addresses" \
    --allowlist-type "RADec" \
    --allowlist-type "XYZ" \
    --allowlist-type "UVW" \
    --allowlist-type "LmnRime" \
    --allowlist-type "ShapeletCoeff" \
    --allowlist-type "ShapeletUV" \
    --allowlist-type "Jones.*" \
    --allowlist-type "Points" \
    --allowlist-type "Gaussians" \
    --allowlist-type "Shapelets" \
    --allowlist-var "POWER_LAW_FD_REF_FREQ" \
    --size_t-is-usize \
    >"${SCRIPTPATH}"/model_double.rs

bindgen "${SCRIPTPATH}"/model.h \
    --allowlist-function "model_.*" \
    --blocklist-type "__int8_t" \
    --allowlist-type "Addresses" \
    --allowlist-type "RADec" \
    --allowlist-type "XYZ" \
    --allowlist-type "UVW" \
    --allowlist-type "LmnRime" \
    --allowlist-type "ShapeletCoeff" \
    --allowlist-type "ShapeletUV" \
    --allowlist-type "Jones.*" \
    --allowlist-type "Points" \
    --allowlist-type "Gaussians" \
    --allowlist-type "Shapelets" \
    --allowlist-var "POWER_LAW_FD_REF_FREQ" \
    --size_t-is-usize \
    -- -D SINGLE \
    >"${SCRIPTPATH}"/model_single.rs

bindgen "${SCRIPTPATH}"/peel.h \
    --allowlist-function "rotate_average" \
    --allowlist-function "xyzs_to_uvws" \
    --allowlist-function "iono_loop" \
    --allowlist-function "subtract_iono" \
    --blocklist-type "Addresses" \
    --blocklist-type "RADec" \
    --blocklist-type "XYZ" \
    --blocklist-type "UVW" \
    --blocklist-type "LmnRime" \
    --blocklist-type "ShapeletCoeff" \
    --blocklist-type "ShapeletUV" \
    --blocklist-type "Jones.*" \
    --blocklist-type "Points" \
    --blocklist-type "Gaussians" \
    --blocklist-type "Shapelets" \
    --size_t-is-usize \
    >"${SCRIPTPATH}"/peel_double.rs

bindgen "${SCRIPTPATH}"/peel.h \
    --allowlist-function "rotate_average" \
    --allowlist-function "xyzs_to_uvws" \
    --allowlist-function "iono_loop" \
    --allowlist-function "subtract_iono" \
    --blocklist-type "Addresses" \
    --blocklist-type "RADec" \
    --blocklist-type "XYZ" \
    --blocklist-type "UVW" \
    --blocklist-type "LmnRime" \
    --blocklist-type "ShapeletCoeff" \
    --blocklist-type "ShapeletUV" \
    --blocklist-type "Jones.*" \
    --blocklist-type "Points" \
    --blocklist-type "Gaussians" \
    --blocklist-type "Shapelets" \
    --size_t-is-usize \
    -- -D SINGLE \
    >"${SCRIPTPATH}"/peel_single.rs
