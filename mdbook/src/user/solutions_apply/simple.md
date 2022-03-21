# Simple usage of solutions apply

~~~admonish info
Use the `solutions-apply` subcommand, i.e.

```shell
hyperdrive solutions-apply
```

At the very least, this requires:
- Input data (with the flag `-d`)
  - [Supported formats](../../defs/vis_formats_read.md)
- Calibration solutions (with the flag `-s`)
  - [Supported formats](../../defs/cal_sols.md)
~~~

## Examples

~~~admonish example title="From raw MWA data"
```shell
hyperdrive solutions-apply -d *gpubox*.fits *.metafits *.mwaf -s hyp_sols.fits -o hyp_cal.ms
```
~~~

~~~admonish example title="From an uncalibrated measurement set"
```shell
hyperdrive solutions-apply -d *.ms -s hyp_sols.fits -o hyp_cal.ms
```
~~~

~~~admonish example title="From an uncalibrated uvfits"
```shell
hyperdrive solutions-apply -d *.uvfits -s hyp_sols.fits -o hyp_cal.ms
```
~~~

Generally the syntax is the same as [`di-calibrate`](../di_cal/simple.md).
