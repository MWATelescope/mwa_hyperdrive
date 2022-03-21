# Simple usage of DI calibrate

~~~admonish info
DI calibration is done with the `di-calibrate` subcommand, i.e.

```shell
hyperdrive di-calibrate
```

At the very least, this requires:
- Input data (with the flag `-d`)
  - [Supported formats](../../defs/vis_formats_read.md)
- A sky model (with the flag `-s`)
  - [Supported formats](../../defs/source_lists.md)
  - PUMA sky models suitable for EoR calibration (and perhaps other parts of the
    sky) can be obtained [here](https://github.com/JLBLine/srclists) (at the time of writing [srclist_pumav3_EoR0aegean_fixedEoR1pietro+ForA_phase1+2.txt](https://github.com/JLBLine/srclists/blob/master/srclist_pumav3_EoR0aegean_fixedEoR1pietro%2BForA_phase1%2B2.txt) is preferred)
~~~

## Examples

~~~admonish example title="Raw MWA data"
A [metafits](../../defs/mwa/metafits.md) file is always required when reading
raw MWA data. [`mwaf`](../../defs/mwa/mwaf.md) files are optional.

For "legacy" MWA data:

```shell
hyperdrive di-calibrate -d *gpubox*.fits *.metafits *.mwaf -s a_good_sky_model.yaml
```

or for MWAX:

```shell
hyperdrive di-calibrate -d *ch???*.fits *.metafits *.mwaf -s a_good_sky_model.yaml
```
~~~

~~~admonish example title="Measurement sets"
Note that a metafits may not be required, but is generally a good idea.

```shell
hyperdrive di-calibrate -d *.ms *.metafits -s a_good_sky_model.yaml
```
~~~

~~~admonish example title="uvfits"
Note that a metafits may not be required, but is generally a good idea.

```shell
hyperdrive di-calibrate -d *.uvfits *.metafits -s a_good_sky_model.yaml
```
~~~
