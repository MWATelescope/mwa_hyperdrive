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

See the README for [`hyperbeam`](https://github.com/MWATelescope/mwa_hyperbeam)
for more info.
~~~
