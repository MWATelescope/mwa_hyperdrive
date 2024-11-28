FROM mwatelescope/birli:main

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
    clang \
    libfontconfig-dev \
    && \
    apt-get clean all && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    apt-get -y autoremove

ADD . /hyperdrive
WORKDIR /hyperdrive

ARG TEST_SHIM=""
RUN ${TEST_SHIM}

RUN cargo install --path . --no-default-features --features=plotting --locked \
    && cargo clean