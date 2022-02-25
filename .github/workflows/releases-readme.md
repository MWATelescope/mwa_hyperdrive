This is a release of `mwa_hyperdrive`, calibration software for the Murchison
Widefield Array radio telescope, obtained from the [GitHub releases
page](https://github.com/MWATelescope/mwa_hyperdrive/releases).

# Licensing

`hyperdrive` is licensed under the [Mozilla Public License 2.0 (MPL
2.0)](https://www.mozilla.org/en-US/MPL/2.0/). The LICENSE file is the relevant
copy.

Other licenses are included from the `hdf5` (COPYING-hdf5), `erfa`
(LICENSE-erfa) and `cfitsio` (LICENSE-cfitsio) libraries. These are included
because (as per the terms of the licenses) `hdf5`, `erfa` and `cfitsio` are
compiled inside the `hyperdrive` binary.

# x86-64-v3?

This is a [microarchitecture
level](https://en.wikipedia.org/wiki/X86-64#Microarchitecture_levels). By
default, Rust compiles for all x86-64 CPUs; this allows maximum compatibility,
but potentially limits the runtime performance because many modern CPU features
can't be used. Compiling at different levels allows the code to be optimised for
different classes of CPUs so users can get something that works best for them.

If your CPU does not support x86-64-v3, you will need to compile `hyperdrive`
from source.
