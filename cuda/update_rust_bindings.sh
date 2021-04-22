#!/bin/bash

# Requires a locally-installed bindgen.

# https://stackoverflow.com/questions/4774054/reliable-way-for-a-bash-script-to-get-the-full-path-to-itself
SCRIPTPATH="$(cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P)"

bindgen "${SCRIPTPATH}"/src_cuda/vis_gen.h > "${SCRIPTPATH}"/src/bindings.rs
