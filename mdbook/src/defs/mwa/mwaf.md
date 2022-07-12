# mwaf flag files

`mwaf` files indicate what visibilities should be flagged, and should be made
with [`Birli`](https://github.com/MWATelescope/Birli) (which uses
[`AOFlagger`](https://gitlab.com/aroffringa/aoflagger)). They aren't necessary,
but may improve things by removing radio-frequency interference. An example of
producing them is:

```shell
birli *gpubox*.fits -m *.metafits -f birli_flag_%%.mwaf
```

At the time of writing, `hyperdrive` only utilises `mwaf` files when reading
visibilities from raw data.

~~~admonish danger title="cotter-produced mwaf files"
`cotter`-produced `mwaf` files are unreliable because
1. The start time of the flags is not written; and
2. The number of timesteps per mwaf file can vary, further confusing things.

Many MWA observations have pre-generated `mwaf` files that are stored in the
archive. These should be ignored and `mwaf` files should be made with `Birli`,
versions 0.7.0 or greater.
~~~
