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

It is possible to use the "analytic" MWA beam, and this does not require an
additional file (but the FEE beam is the default selection). There are two
"flavours": `mwa_pb` and `RTS`, and these can be used by specifying `--beam-type
analytic-mwa_pb` and `--beam-type analytic-rts`, respectively. The differences
between the flavours is not huge, but I (CHJ) suggest the `RTS` flavour if in
doubt, as it seems to look a little better.

## Errors

Beam code usually does not error, but if it does it's likely because:

1. There aren't exactly 16 dipole delays;
2. There aren't exactly 16 or 32 dipole gains per tile; or
3. There's something wrong with the FEE HDF5 file. The official file is well
   tested.

## CRAM tile

[The CRAM tile](https://www.mwatelescope.org/science/eor/cram/) is a special
64-bowtie tile, as opposed to the usual 16-bowtie tiles. `hyperdrive` attempts
to detect and simulate the CRAM's beam response using the RTS-flavoured analytic
beam (using 64 bowties).

Currently, it is anticipated that the CRAM does not ever point away from zenith,
so no bowtie/dipole delays are accepted. Metafits files also have no way of
specifying dead dipoles in the CRAM (to my knowledge), so `hyperdrive` allows
users to specify bowtie gains on the command line with `--cram-dipole-gains`
(64 values, 1 for "alive", 0 for "dead", see [dead dipoles](mwa/dead_dipoles.md)
for more info). `hyperdrive` will automatically detect a tile named "CRAM"
and use the special beam responses for it, but a different tile can be elected
with `--cram-tile`. Finally, the presence of the CRAM tile can be ignored with
`--ignore-cram`, but this is not likely to do anything useful.
