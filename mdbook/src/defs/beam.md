# Beam responses

Beam responses are given by
[`mwa_hyperbeam`](https://github.com/MWATelescope/mwa_hyperbeam). At present,
only MWA beam code is used.

To function, MWA beam code needs a few things:

- The [dipole delays](mwa/delays.md);
- The [dipole gains](mwa/dead_dipoles.md) (usually dead dipoles are 0, others are 1);
- The direction we want the beam response as an Azimuth-Elevation coordinate; and
- A frequency.

In addition, the FEE beam code needs an HDF5 file to function. See the
[post-installation instructions](../installation/post.md) for information on
getting that set up.

## Errors

Beam code usually does not error, but if it does it's likely because:

1. There aren't exactly 16 dipole delays;
2. There aren't exactly 16 or 32 dipole gains per tile; or
3. There's something wrong with the FEE HDF5 file. The official file is well
   tested.