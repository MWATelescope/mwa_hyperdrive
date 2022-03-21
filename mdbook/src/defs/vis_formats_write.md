# Supported visibility formats for writing

The following examples illustrate how to produce each of the supported
visibility file formats with `solutions-apply`, but other aspects of
`hyperdrive` are also able to produce these file formats, and all aspects are
able to perform averaging and write to multiple outputs.

~~~admonish info title="Measurement sets"
```shell
hyperdrive solutions-apply \
    -d *gpubox*.fits *.metafits \
    -s hyp_sols.fits \
    -o hyp_cal.ms
```
~~~

~~~admonish info title="uvfits"
```shell
hyperdrive solutions-apply \
    -d *gpubox*.fits *.metafits \
    -s hyp_sols.fits \
    -o hyp_cal.uvfits
```

A copy of the uvfits standard is
[here](https://library.nrao.edu/public/memos/aips/memos/AIPSM_117.pdf).
~~~

~~~admonish tip title="Visibility averaging"
When writing out visibilities, they can be averaged in time and frequency. Units
can be given to these; e.g. using seconds and kiloHertz:

```shell
hyperdrive solutions-apply \
    -d *gpubox*.fits *.metafits *.mwaf \
    -s hyp_sols.fits \
    -o hyp_cal.ms \
    --time-average 8s \
    --freq-average 80kHz
```

Units are not required; in this case, these factors multiply the observation's
time and freq. resolutions:

```shell
hyperdrive solutions-apply \
    -d *gpubox*.fits *.metafits *.mwaf \
    -s hyp_sols.fits \
    -o hyp_cal.ms \
    --time-average 4 \
    --freq-average 2
```

If the same observation is used in both examples, with a time resolution of 2s
and a freq. resolution of 40kHz, then both commands will yield the same result.

See [this page](blocks.md) for information on how visibilities are averaged in
time and frequency.
~~~

~~~admonish tip title="Writing to multiple visibility outputs"
All aspects of `hyperdrive` that can write visibilities can write to multiple
outputs. Note that it probably does not make sense to write out more than one of
each kind (e.g. two uvfits files), as each of these files will be exactly the
same, and a simple `cp` from one to the other is probably faster than writing to
two files simultaneously from `hyperdrive`.

Example (a measurement set and uvfits):
```shell
hyperdrive solutions-apply \
    -d *gpubox*.fits *.metafits *.mwaf \
    -s hyp_sols.fits \
    -o hyp_cal.ms hyp_cal.uvfits \
    --time-average 4 \
    --freq-average 2
```
~~~
