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

Vetoing removes sources that have components below the elevation limit,
components that are too far from the phase centre,
components with beam-attenuated flux densities less than the threshold,
and/or remove sources that aren't in the top N sources specified by `num_sources`.

This is important for calibration, because it is expensive to generate a sky
model, and using only dim sources would result in poor calibration.

Sources are vetoed if any of their components are further away from the
phase centre than `source_dist_cutoff_deg` or their beam attenuated flux
densities are less than `veto_threshold`.

Source-list vetoing can do unexpected things. You can effectively disable it by
supplying `--veto-threshold 0`, although the veto routine will still:

1. Remove sources below the horizon; and
2. Sort the remaining sources by brightness based off of the centre frequencies
   MWA coarse channels.

If there are fewer sources than that of `num_sources`, an error is returned;
it's up to the caller to handle this if they want to.

The frequencies to use for beam calculations are the coarse channel centers,
1.28 MHz apart on MWA.

### Auto-correlations

By default, visibilities include auto-correlations when simulated. To write only cross-correlations, use:

```shell
hyperdrive vis-simulate \
  -s srclist.yaml \
  -m *.metafits \
  --output-no-autos
```

Including autos can increase file size; disable them if you only need cross-correlations.
