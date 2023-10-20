# Post installation instructions

~~~admonish info title="Setting up the beam"

Many `hyperdrive` functions require the beam code to function. The MWA FEE beam
HDF5 file can be obtained with:

```shell
wget http://ws.mwatelescope.org/static/mwa_full_embedded_element_pattern.h5
```

Move the `h5` file anywhere you like, and put the file path in the
`MWA_BEAM_FILE` environment variable:

```shell
export MWA_BEAM_FILE=/path/to/mwa_full_embedded_element_pattern.h5
```

It is possible to use the analytic beam instead of the FEE beam, meaning you
don't need the HDF5 file, but the FEE beam is probably better.

See the README for [`hyperbeam`](https://github.com/MWATelescope/mwa_hyperbeam)
for more info.
~~~
