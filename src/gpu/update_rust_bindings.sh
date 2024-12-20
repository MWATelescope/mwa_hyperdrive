#!/bin/bash

# Update the Rust bindings to GPU code. This script must be run whenever the C
# headers for GPU code change.

# This script requires bindgen. This can be provided by a package manager or
# installed with "cargo install bindgen".

# https://stackoverflow.com/questions/4774054/reliable-way-for-a-bash-script-to-get-the-full-path-to-itself
SCRIPTPATH="$(cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P)"

bindgen "${SCRIPTPATH}"/utils.h \
    --allowlist-function "get_gpu_device_info" \
    > "${SCRIPTPATH}"/utils_bindings.rs

for PRECISION in SINGLE DOUBLE; do
    LOWER_CASE=$(echo "${PRECISION}" | tr '[:upper:]' '[:lower:]')
    echo "Generating bindings for ${LOWER_CASE}-precision GPU code"

    bindgen "${SCRIPTPATH}"/types.h \
            --ignore-functions \
            --blocklist-type "__int8_t" \
            --allowlist-type "UVW" \
            --allowlist-type "LmnRime" \
            --allowlist-type "ShapeletCoeff" \
            --allowlist-type "ShapeletUV" \
            --allowlist-type "Jones.*" \
            --allowlist-type "Addresses" \
            --allowlist-type "Points" \
            --allowlist-type "Gaussians" \
            --allowlist-type "Shapelets" \
            --allowlist-var "SBF_.*" \
            --with-derive-default \
            --with-derive-partialeq \
            --with-derive-eq \
            -- -D "${PRECISION}" \
            > "${SCRIPTPATH}/types_${LOWER_CASE}.rs"

    bindgen "${SCRIPTPATH}"/model.h \
            --allowlist-function "model_.*" \
            --allowlist-var "POWER_LAW_FD_REF_FREQ" \
            --blocklist-type ".*" \
            -- -D "${PRECISION}" \
            > "${SCRIPTPATH}/model_${LOWER_CASE}.rs"
done
