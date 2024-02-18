ARG NVIDIA_VERSION=11.4.3

FROM nvidia/cuda:${NVIDIA_VERSION}-devel-ubuntu20.04

ENV DEBIAN_FRONTEND="noninteractive"
RUN apt-get update -y && \
    apt-get -y install \
            tzdata \
            build-essential \
            pkg-config \
            cmake \
            curl \
            git \
            lcov \
            fontconfig \
            libfreetype-dev \
            libexpat1-dev \
            libcfitsio-dev \
            libhdf5-dev \
            clang \
            libfontconfig-dev \
            && apt-get clean all \
            && rm -rf /var/lib/apt/lists/*

ARG RUST_VERSION=1.72
ARG TARGET_CPU=x86-64
# example: 70
ARG CUDA_COMPUTE

# Get Rust
RUN mkdir -m755 /opt/rust /opt/cargo
ENV RUSTUP_HOME=/opt/rust CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:$PATH
# set minimal rust version here to use a newer stable version
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain=$RUST_VERSION

ADD . /mwa_hyperdrive
WORKDIR /mwa_hyperdrive
ENV CXX=/usr/bin/g++
ENV CARGO_BUILD_RUSTFLAGS="-C target-cpu=${TARGET_CPU}"
RUN [ -z "$CUDA_COMPUTE" ] || export HYPERDRIVE_CUDA_COMPUTE=${CUDA_COMPUTE}; \
    cargo install --path . --no-default-features --features=cuda,plotting --locked \
    && cargo clean