# Simulate visibilities

`vis-simulate` effectively turns a sky-model source list into visibilities.

~~~admonish info title="Simple example"
```shell
hyperdrive vis-simulate \
    -s srclist.yaml \
    -m *.metafits
```
~~~

## Considerations

### Disabling beam attenuation

`--no-beam`

### Dead dipoles

By default, [dead dipoles](../../defs/mwa/dead_dipoles.md) in the
[metafits](../../defs/mwa/metafits.md) are used. These will affect the generated
visibilities. You can disable them with `--unity-dipole-gains`.

### Vetoing

Source-list vetoing can do unexpected things. You can effectively disable it by
supplying `--veto-threshold 0`, although the veto routine will still:

1. Remove sources below the horizon; and
2. Sort the remaining sources by brightness based off of the centre frequencies
   MWA coarse channels.
