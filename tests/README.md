## mwa_hyperdrive_tests

This directory contains a crate "mwa_hyperdrive_tests", which is used to
consolidate common testing code. In addition, it contains data to test against.
To enable the full test suite of hyperdrive, this directory should also contain
the gpubox and mwaf files associated with the observations:

    - 1065880128

## Details on the git-tracked data here

1065880128_broken.metafits has a dipole delay deliberately set to 32 in its
first RF input (antenna 75, pol Y).

1065880128_01.mwaf.gz and 1065880128_02.mwaf.gz are gzipped cotter flag files.
