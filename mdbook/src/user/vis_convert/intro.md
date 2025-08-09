# Convert visibilities

`vis-convert` reads in visibilities and writes them out, performing whatever
transformations were requested on the way (e.g. ignore autos, average to a
particular time resolution, flag some tiles, etc.).

Autos control: **--no-autos** ignores auto-correlations

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
    --no-autos \
    -o hyp_converted.ms
```
~~~
