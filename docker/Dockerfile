# If you want to use CUDA for visibility subtraction (much faster!), you need
# to adjust this number to the NVIDIA driver on the host system.
ARG NVIDIA_VERSION=11.4.3

# CPU
FROM ubuntu:22.04
# CUDA version
# FROM nvidia/cuda:${NVIDIA_VERSION}-devel-ubuntu20.04

ENV DEBIAN_FRONTEND="noninteractive"
RUN apt-get update -y && \
    apt-get -y install \
            tzdata \
            build-essential \
            pkg-config \
            cmake \
            curl \
            git \
            autoconf \
            libtool \
            unzip \
            wget \
            zip \
            libcfitsio-dev \
            liberfa-dev \
            libstarlink-pal-dev \
            libgsl-dev \
            libfontconfig-dev \
            liblapack-dev \
            gfortran \
            && apt-get clean all \
            && rm -rf /var/lib/apt/lists/*

# Get Rust
RUN mkdir -m755 /opt/rust /opt/cargo
ENV RUSTUP_HOME=/opt/rust CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:$PATH
ENV RUST_VERSION=1.72
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain=$RUST_VERSION

# CPU
RUN git clone https://github.com/MWATelescope/mwa_hyperdrive -b sdc3 \
  && /opt/cargo/bin/cargo install --path mwa_hyperdrive --locked --features=cfitsio-static,hdf5-static \
  && rm -r mwa_hyperdrive
# # CUDA
# RUN git clone https://github.com/MWATelescope/mwa_hyperdrive -b sdc3 \
#   && /opt/cargo/bin/cargo install --path mwa_hyperdrive --locked --features=cfitsio-static,hdf5-static,cuda \
#   && rm -r mwa_hyperdrive

RUN git clone https://github.com/cjordan/sdc3_vis_convert \
  && /opt/cargo/bin/cargo install --path sdc3_vis_convert --locked \
  && rm -r sdc3_vis_convert

ADD chips /chips
RUN cd /chips \
  && ls \
  && make \
  && cd /
