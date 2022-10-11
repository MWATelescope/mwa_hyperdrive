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

Measurement sets (MSs) are typically made with
[`Birli`](https://github.com/MWATelescope/Birli) or
[`cotter`](https://github.com/MWATelescope/cotter)
([`Birli`](https://github.com/MWATelescope/Birli) preferred). At the time of
writing, MWA-formatted measurement sets do not contain dead dipole information,
and so calibration may not be as accurate as it could be. To get around this, an
observation's [metafits](mwa/metafits.md) file can be supplied alongside the MS
to improve calibration. See
[below](./vis_formats_read.md#admonition-when-using-a-metafits) for more info.
~~~

~~~admonish info title="uvfits"
```shell
hyperdrive di-calibrate -d *.uvfits *.metafits -s a_good_sky_model.yaml
```

When reading uvfits, a metafits is not required *only* if the user has supplied
the MWA dipole delays. At the time of writing, MWA-formatted uvfits files do not
contain dipole delays or dead dipole information, and so avoiding a metafits
file when calibrating may mean it is not as accurate as it could be. See
[below](./vis_formats_read.md#admonition-when-using-a-metafits) for more info.


A copy of the uvfits standard is
[here](https://library.nrao.edu/public/memos/aips/memos/AIPSM_117.pdf).
~~~

~~~admonish warning title="When using a metafits"
When using a metafits file with a uvfits/MS, the tile names in the metafits and
uvfits/MS must *exactly* match. Only when they exactly match are the dipole
delays and dipole gains able to be applied properly. If they don't match, a
warning is issued.

MWA uvfits/MS files made with `Birli` or `cotter` will always match their
observation's metafits tile names, so this issue only applies to uvfits/MS files
created elsewhere.
~~~
