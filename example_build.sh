#!/bin/bash

git clone https://github.com/MWATelescope/mwa_hyperdrive -b SDC3
cd mwa_hyperdrive/docker
docker build . -t chjordan/hyperdrive:SDC3 –build-arg NVIDIA_VERSION=<the version of your NVIDIA driver>

