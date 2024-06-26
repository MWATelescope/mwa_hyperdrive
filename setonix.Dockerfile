ARG ROCM_VER=5.6.0
# options:
# - rocm5.4.6-mpich3.4.3-ubuntu22
# - rocm5.6.0-mpich3.4.3-ubuntu22
# - rocm5.6.1-mpich3.4.3-ubuntu22
# - rocm5.7.3-mpich3.4.3-ubuntu22
# - rocm6.0.2-mpich3.4.3-ubuntu22
# - rocm6.1-mpich3.4.3-ubuntu22

# for ROCM_VER in 6.1 5.7.3; do
#   export TAG=d3vnull0/hyperdrive:v0.4.0-setonix-rocm${ROCM_VER}
#   docker build -t ${TAG} -f setonix.Dockerfile --build-arg="ROCM_VER=${ROCM_VER}" . \
#   && docker push ${TAG}
# done
FROM quay.io/pawsey/rocm-mpich-base:rocm${ROCM_VER}-mpich3.4.3-ubuntu22

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
# example: gfx90a
ARG HIP_ARCH=gfx90a

# Get Rust
RUN mkdir -m755 /opt/rust /opt/cargo
ENV RUSTUP_HOME=/opt/rust CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:$PATH
# set minimal rust version here to use a newer stable version
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain=$RUST_VERSION

ADD . /mwa_hyperdrive
WORKDIR /mwa_hyperdrive
RUN export HYPERBEAM_HIP_ARCH="${HIP_ARCH}"; \
    export CARGO_BUILD_RUSTFLAGS="-C target-cpu=${TARGET_CPU}"; \
    export HIP_PATH=/opt/rocm; \
    if expr "${ROCM_VER}" : '5.*'; then export HIP_PATH=/opt/rocm/hip; fi; \
    cargo install --path . --no-default-features --features=hip,plotting --locked \
    && cargo clean