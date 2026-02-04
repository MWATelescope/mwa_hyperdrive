# -> cpu-only, multiplatform with aoflagger
# docker buildx build --platform=arm64,amd64 . \
#   --build-arg=BASE_IMAGE=mwatelescope/birli:main \
#   --tag=mwatelescope/hyperdrive:cpu \
#   --push
# not implemented: --build-arg=FEATURES=aoflagger

# -> cuda, V100 or A100
# export CUDA_VER=12.5.1
# docker build . \
#   --build-arg=BASE_IMAGE=nvidia/cuda:${CUDA_VER}-devel-ubuntu24.04 \
#   --build-arg=FEATURES=cuda \
#   --build-arg=CUDA_COMPUTE=70,80 \
#   --tag=mwatelescope/hyperdrive:${HYP_VER}-cuda${CUDA_VER}-ubuntu24.04 --push
# module load singularity/default; singularity pull -F /data/curtin_mwaeor/singularity/hyperdrive_autos-dev_cuda12.5.1-ubuntu24.04.sif docker://mwatelescope/hyperdrive:cuda12.5.1-ubuntu24.04
# note: don't use nvidia/cuda:${CUDA_VER}-devel-ubuntu20.04, python3.8 is too old for pyuvdata>6
# note: don't use nvidia/cuda:${CUDA_VER}-devel-ubuntu22.04, python3.10 is too old for mwax_mover

# -> rocm, setonix MI250
# export ROCM_VER=6.3.3
# export HYP_VER=0.6.1-autos
# docker build . \
#   --build-arg=BASE_IMAGE=quay.io/pawsey/rocm-mpich-base:rocm${ROCM_VER}-mpich3.4.3-ubuntu24.04 \
#   --build-arg=FEATURES=hip \
#   --build-arg=HIP_ARCH=gfx90a \
#   --tag=mwatelescope/hyperdrive:${HYP_VER}-rocm${ROCM_VER}-ubuntu24.04 --push
# module load singularity/default; singularity pull -F /software/projects/mwaeor/singularity/hyperdrive_rocm6.3.1-ubuntu22.sif docker://mwatelescope/hyperdrive:0.6.1-autos-rocm6.3.1-ubuntu22

# -> dug MI50
# export ROCM_VER=6.0.2
# docker build . \
#   --build-arg=BASE_IMAGE=quay.io/pawsey/rocm-mpich-base:rocm${ROCM_VER}-mpich3.4.3-ubuntu22 \
#   --build-arg=FEATURES=hip \
#   --build-arg=HIP_ARCH=gfx906

# docker build --build-arg="ROCM_VER=${ROCM_VER}" --build-arg="HIP_ARCH=${HIP_ARCH}" .
ARG BASE_IMAGE=mwatelescope/birli:main
FROM $BASE_IMAGE

ENV DEBIAN_FRONTEND="noninteractive"
RUN apt-get update -y && \
    apt-get -y install \
    build-essential \
    clang \
    cmake \
    curl \
    fontconfig \
    git \
    lcov \
    libcfitsio-dev \
    libexpat1-dev \
    libfontconfig-dev \
    libfreetype-dev \
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
# optional, example: gfx90a for MI250
ARG HIP_ARCH=""
# optional, example: "hip" for ROCm, plotting is included by default, use " --"
ARG FEATURES=""
# optional, example: "-C target-cpu=native"
ARG RUSTFLAGS=""

ADD . /mwa_hyperdrive
WORKDIR /mwa_hyperdrive
# TODO: is this a cuda thing?
ENV CXX=/usr/bin/g++

# e.g. docker build . --build-arg=TEST_SHIM=MWA_BEAM_FILE=mwa_full_embedded_element_pattern.h5\ cargo\ test\ --release
ARG TEST_SHIM=""

# might need HIP_PATH for rocm5
# export HIP_PATH=/opt/rocm; \
#     if expr "${ROCM_VER}" : '5.*'; then export HIP_PATH=/opt/rocm/hip; fi; \

RUN [ -z "$CUDA_COMPUTE" ] || export "HYPERDRIVE_CUDA_COMPUTE=${CUDA_COMPUTE}"; \
    [ -z "$HIP_ARCH" ] || export "HYPERDRIVE_HIP_ARCH=${HIP_ARCH}"; \
    [ -z "$RUSTFLAGS" ] || export "CARGO_BUILD_RUSTFLAGS=${RUSTFLAGS}"; \
    [ -z "$TEST_SHIM" ] || eval ${TEST_SHIM} && \
    cargo install --path . --features=${FEATURES} --locked \
    && cargo clean
