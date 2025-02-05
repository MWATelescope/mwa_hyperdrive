# There are many ways you can build this Dockerfile

# -> cpu-only, multiplatform with aoflagger
# docker buildx build --platform=arm64,amd64 . \
#   --build-arg=BASE_IMAGE=mwatelescope/birli:main \
#   --build-arg=FEATURES=aoflagger

# -> cuda, V100 or A100
# export CUDA_VER=11.4.3
# docker build . \
#   --build-arg=BASE_IMAGE=nvidia/cuda:${CUDA_VER}-devel-ubuntu20.04 \
#   --build-arg=FEATURES=cuda \
#   --build-arg=CUDA_COMPUTE=70,80

# -> rocm, setonix MI250
# export ROCM_VER=6.3.1
# docker build . \
#   --build-arg=BASE_IMAGE=quay.io/pawsey/rocm-mpich-base:rocm${ROCM_VER}-mpich3.4.3-ubuntu22 \
#   --build-arg=FEATURES=hip \
#   --build-arg=HIP_ARCH=gfx90a

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
RUN mkdir -pm755 /opt/rust /opt/cargo
ENV RUSTUP_HOME=/opt/rust CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain=$RUST_VERSION

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