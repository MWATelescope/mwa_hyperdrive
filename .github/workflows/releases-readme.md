This is a release of `mwa_hyperdrive`, calibration software for the Murchison
Widefield Array radio telescope, obtained from the [GitHub releases
page](https://github.com/MWATelescope/mwa_hyperdrive/releases).

Documentation on `hyperdrive` can be found
[here](https://mwatelescope.github.io/mwa_hyperdrive/index.html).

Many `hyperdrive` functions require the beam code to function. The MWA
FEE beam HDF5 file can be obtained with:

  `wget http://ws.mwatelescope.org/static/mwa_full_embedded_element_pattern.h5`

Move the `h5` file anywhere you like, and put the file path in `MWA_BEAM_FILE`:

  `export MWA_BEAM_FILE=/path/to/mwa_full_embedded_element_pattern.h5`

See the README for [`hyperbeam`](https://github.com/MWATelescope/mwa_hyperbeam)
for more info.

# Licensing

`hyperdrive` is licensed under the [Mozilla Public License 2.0 (MPL
2.0)](https://www.mozilla.org/en-US/MPL/2.0/). The LICENSE file is the relevant
copy.

Other licenses are included from the `hdf5` (COPYING-hdf5), `erfa`
(LICENSE-erfa) and `cfitsio` (LICENSE-cfitsio) libraries. These are included
because (as per the terms of the licenses) `hdf5`, `erfa` and `cfitsio` are
compiled inside the `hyperdrive` binary.

An NVIDIA license may also be included as, per the terms of the license,
`hyperdrive` utilises code that was modified from an existing CUDA example.

# x86-64-v3?

This is a [microarchitecture
level](https://en.wikipedia.org/wiki/X86-64#Microarchitecture_levels). By
default, Rust compiles for all x86-64 CPUs; this allows maximum compatibility,
but potentially limits the runtime performance because many modern CPU features
can't be used. Compiling at different levels allows the code to be optimised for
different classes of CPUs so users can get something that works best for them.

If your CPU does not support x86-64-v3, you will need to compile `hyperdrive`
from source.

# CUDA?

The releases with "CUDA" in the name are CUDA enabled. The `hyperdrive` binaries
have been dynamically linked against CUDA 11.2.0; to run them, a CUDA
installation on version 11 is required.

There is also a double- or single-precision version of `hyperdrive` provided. If
you're running a desktop NVIDIA GPU (e.g. RTX 2070), then you probably want the
single-precision version. This is because desktop GPUs have a lot less
double-precision computation capability. It is still possible to use the
double-precision version, but the extra precision comes at the expensive of
speed.

Other GPUs, like the V100s hosted by the Pawsey Supercomputing Centre, are
capable of running the double-precision code much faster, so there is little
incentive for running single-precision code on these GPUs.
