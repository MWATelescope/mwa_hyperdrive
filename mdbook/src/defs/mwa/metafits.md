# Metafits files

The MWA tracks observation metadata with "metafits" files. Often these accompany
the raw visibilities in a download, but these could be old (such as the "PPD
metafits" files). **`hyperdrive` does not support PPD metafits files; only new
metafits files should be used.**

This command downloads a new metafits file for the specified observation ID:

~~~admonish example title="Download MWA metafits file"
```
OBSID=1090008640; wget "http://ws.mwatelescope.org/metadata/fits?obs_id=${OBSID}" -O "${OBSID}".metafits
```
~~~

~~~admonish info title="Why should I use a metafits file?"
Measurement sets and uvfits files do not contain MWA-specific information,
particularly dead dipole information. Calibration should perform better when
[dead dipoles](dead_dipoles.md) are taken into account. Measurement sets and
uvfits file may also lack [dipole delay](delays.md) information.
~~~

~~~admonish info title="Why are new metafits files better?"
The database of MWA metadata can change over time for observations conducted
even many years ago, and the engineering team may decide that some tiles/dipoles
for some observations should be retroactively flagged, or that digital gains
were wrong, etc. In addition, older metafits files may not have all the metadata
that is required to be present by
[`mwalib`](https://github.com/MWATelescope/mwalib), which is used by
`hyperdrive` when reading metafits files.
~~~

## Controlling dipole gains

If the "TILEDATA" HDU of a metafits contains a "DipAmps" column, each row
containing 16 double-precision values for bowties in the M&C order, these are
used as the dipole gains in beam calculations. If the "DipAmps" column isn't
available, the default behaviour is to use gains of 1.0 for all dipoles, except
those that have delays of 32 in the "Delays" column (they will have a gain of
0.0, and are considered [dead](dead_dipoles.md)).
