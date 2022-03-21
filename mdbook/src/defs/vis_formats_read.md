# Supported visibility formats for reading

~~~admonish info title="Raw MWA data"
Raw "legacy" MWA data comes in "gpubox" files. "MWAX" data comes in a similar
format, and `*ch???*.fits` is a useful glob to identify them. Raw data can be
accessed from the [ASVO](https://asvo.mwatelescope.org/).

Here are examples of using each of these MWA formats with `di-calibrate`:

```shell
hyperdrive di-calibrate -d *gpubox*.fits *.metafits *.mwaf -s a_good_sky_model.yaml
hyperdrive di-calibrate -d *ch???*.fits *.metafits *.mwaf -s a_good_sky_model.yaml
```

Note that all visibility formats should probably be accompanied by a metafits
file. See [this page](mwa/metafits.md) for more info.

`mwaf` files indicate what visibilities should be flagged. See [this
page](mwa/mwaf.md) for more info.
~~~

~~~admonish info title="Measurement sets"
```shell
hyperdrive di-calibrate -d *.ms *.metafits -s a_good_sky_model.yaml
```

Measurement sets are typically made with
[`Birli`](https://github.com/MWATelescope/Birli) or
[`cotter`](https://github.com/MWATelescope/cotter)
([`Birli`](https://github.com/MWATelescope/Birli) preferred). Note that a
metafits is desirable but usually not required. At the time of writing,
MWA-formatted measurement sets do not contain dead dipole information, and so
calibration may not be as accurate as it could be.
~~~

~~~admonish info title="uvfits"
```shell
hyperdrive di-calibrate -d *.uvfits *.metafits -s a_good_sky_model.yaml
```

When reading uvfits, a metafits is not required *only* if the user has supplied
the MWA dipole delays. At the time of writing, MWA-formatted uvfits files do not
contain dipole delays or dead dipole information, and so avoiding a metafits
file when calibrating may mean it is not as accurate as it could be.

A copy of the uvfits standard is
[here](https://library.nrao.edu/public/memos/aips/memos/AIPSM_117.pdf).
~~~
