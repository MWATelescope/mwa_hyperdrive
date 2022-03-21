# Modelling visibilities

`hyperdrive` uses a sky model when modelling/simulating visibilities. This means
that for every sky-model source, a visibility needs to be generated for each
observation time, baseline and frequency. Modelling visibilities for a source
can be broken down into three broad steps:
- [Estimating](estimating.md) a source's flux density at a particular frequency;
- Getting the baseline's beam response toward the source; and
- Applying these factors to the result of the [measurement equation](rime.md).

Beam responses are given by
[`mwa_hyperbeam`](https://github.com/MWATelescope/mwa_hyperbeam). See more info
on the beam [here](../beam.md).

The following pages go into further detail of how visibilities are modelled in
`hyperdrive`.
