# Images are built by .github/workflows/docker.yml on release tags (or manual
# dispatch). Equivalent manual builds:

# -> cpu, multiplatform (birli:main already ships aoflagger)
# docker buildx build --platform=linux/amd64,linux/arm64 . \
#   --build-arg=FEATURES=bmetrics \
#   --tag=mwatelescope/hyperdrive:${HYP_VER} --push

# -> cuda, V100 (70) and A100 (80)
# export CUDA_VER=12.5.1
# docker build . \
#   --build-arg=BASE_IMAGE=nvidia/cuda:${CUDA_VER}-devel-ubuntu24.04 \
#   --build-arg=FEATURES=cuda,bmetrics \
#   --build-arg=CUDA_COMPUTE=70,80 \
#   --tag=mwatelescope/hyperdrive:${HYP_VER}-cuda${CUDA_VER}-ubuntu24.04 --push
# note: ubuntu20.04/22.04 bases have too old a python for pyuvdata>6 / mwax_mover

# -> rocm, setonix MI250X (gfx90a); use HIP_ARCH=gfx906 for MI50
# export ROCM_VER=6.3.3
# docker build . \
#   --build-arg=BASE_IMAGE=quay.io/pawsey/rocm-mpich-base:rocm${ROCM_VER}-mpich3.4.3-ubuntu24.04 \
#   --build-arg=FEATURES=hip,bmetrics \
#   --build-arg=HIP_ARCH=gfx90a \
#   --tag=mwatelescope/hyperdrive:${HYP_VER}-rocm${ROCM_VER}-ubuntu24.04 --push

# singularity pull docker://mwatelescope/hyperdrive:${HYP_VER}-cuda${CUDA_VER}-ubuntu24.04

ARG BASE_IMAGE=mwatelescope/birli:main
FROM $BASE_IMAGE

ENV DEBIAN_FRONTEND="noninteractive"
RUN apt-get update -y && \
    apt-get -y install \
    build-essential \
    clang \
    cmake \
    curl \
    git \
    lcov \
    libcfitsio-dev \
    libhdf5-dev \
    pkg-config \
    tzdata \
    && \
    apt-get clean all && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    apt-get -y autoremove

# Get Rust
ARG RUST_VERSION=stable
ENV RUSTUP_HOME=/opt/rust CARGO_HOME=/opt/cargo
ENV PATH="${CARGO_HOME}/bin:${PATH}"
# 2025-07-08: rustup is broken in gh actions ci if it's already installed
RUN if [ ! -f $RUSTUP_HOME/settings.toml ]; then \
        mkdir -pm755 $RUSTUP_HOME/tmp $CARGO_HOME && ( \
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | env RUSTUP_HOME=$RUSTUP_HOME CARGO_HOME=$CARGO_HOME TMPDIR=$RUSTUP_HOME/tmp \
        sh -s -- -y \
        --profile=minimal \
        --default-toolchain=${RUST_VERSION} \
        ) \
    fi

# optional, example: "70,80" for V100 and A100
ARG CUDA_COMPUTE=""
# optional, example: gfx90a for MI250X
ARG HIP_ARCH=""
# optional, example: "hip,bmetrics"
ARG FEATURES=""
# optional, example: "-C target-cpu=native"
ARG RUSTFLAGS=""

# bmetrics needs aoflagger; build it unless the base image already has it
ARG CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release"
ARG AOFLAGGER_BRANCH=v3.4.0
RUN if echo "$FEATURES" | grep -q "bmetrics" && ! ldconfig -p | grep -q libaoflagger; then \
        apt-get update -y && \
        apt-get -y install \
        casacore-dev \
        gfortran \
        libblas-dev \
        libboost-date-time-dev \
        libboost-filesystem-dev \
        libboost-program-options-dev \
        libboost-system-dev \
        libboost-test-dev \
        libfftw3-dev \
        liblapack-dev \
        liblua5.3-dev \
        libpng-dev \
        libxml2-dev \
        python3 \
        python3-dev \
        && \
        apt-get clean all && \
        rm -rf /var/lib/apt/lists/* && \
        git clone --depth 1 --branch=${AOFLAGGER_BRANCH} --recurse-submodules https://gitlab.com/aroffringa/aoflagger.git /aoflagger && \
        cd /aoflagger && \
        mkdir build && \
        cd build && \
        cmake $CMAKE_ARGS \
        -DENABLE_GUI=OFF \
        -DPORTABLE=True \
        .. && \
        make install -j && \
        ldconfig && \
        cd / && \
        rm -rf /aoflagger; \
    fi

ADD . /mwa_hyperdrive
WORKDIR /mwa_hyperdrive
# TODO: is this a cuda thing?
ENV CXX=/usr/bin/g++

# e.g. docker build . --build-arg=TEST_SHIM=MWA_BEAM_FILE=mwa_full_embedded_element_pattern.h5\ cargo\ test\ --release
ARG TEST_SHIM=""

# might need HIP_PATH for rocm5
# export HIP_PATH=/opt/rocm; \
#     if expr "${ROCM_VER}" : '5.*'; then export HIP_PATH=/opt/rocm/hip; fi; \

# HYPERBEAM_* is read by mwa_hyperbeam's build script too
RUN [ -z "$CUDA_COMPUTE" ] || export "HYPERDRIVE_CUDA_COMPUTE=${CUDA_COMPUTE}" "HYPERBEAM_CUDA_COMPUTE=${CUDA_COMPUTE}"; \
    [ -z "$HIP_ARCH" ] || export "HYPERDRIVE_HIP_ARCH=${HIP_ARCH}" "HYPERBEAM_HIP_ARCH=${HIP_ARCH}"; \
    [ -z "$RUSTFLAGS" ] || export "CARGO_BUILD_RUSTFLAGS=${RUSTFLAGS}"; \
    [ -z "$TEST_SHIM" ] || eval ${TEST_SHIM} && \
    RUSTFLAGS="${RUSTFLAGS} -C link-args=-Wl,-rpath,/usr/local/lib -L/usr/local/lib" \
    cargo install --path . --features=${FEATURES} --locked \
    && cargo clean

ENTRYPOINT ["hyperdrive"]
