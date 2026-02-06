# Convert visibilities

`vis-convert` reads in visibilities and writes them out, performing whatever
transformations were requested on the way (e.g. ignore autos, average to a
particular time resolution, flag some tiles, etc.).

Autos control: Auto-correlations are not read by default. Use **--autos** to include them when reading, and they will be written to the output. If `--autos` is not used (default), autos are not read and not written.

~~~admonish info title="Simple examples"
```shell
hyperdrive vis-convert \
    -d *gpubox* *.metafits \
    --tile-flags Tile011 Tile012 \
    -o hyp_converted.uvfits hyp_converted.ms
```

```shell
hyperdrive vis-convert \
    -d *.uvfits \
    --autos \
    -o hyp_converted.ms
```
~~~
