# Varying solutions over time

~~~admonish tip
See [this page](../../../defs/blocks.md) for information on timeblocks.
~~~

By default, `di-calibrate` uses only one "timeblock", i.e. all data timesteps
are averaged together during calibration. This provides good signal-to-noise,
but it is possible that calibration is improved by taking time variations into
account. This is done with `--timesteps-per-timeblock` (`-t` for short).

If `--timesteps-per-timeblock` is given a value of 4, then every 4 timesteps are
calibrated together and written out as a timeblock. Values with time units (e.g.
`8s`) are also accepted; in this case, every 8 seconds worth of data are
averaged during calibration and written out as a timeblock.

Depending on the number of timesteps in the data, using `-t` could result in
*many* timeblocks written to the calibration solutions. Each solution timeblock
is plotted when these solutions are given to `solutions-plot`. For each timestep
in question, the best solution timeblock is used when running `solutions-apply`.

## Implementation

When multiple timeblocks are to be made, `hyperdrive` will do a pass of
calibration using *all* timesteps to provide each timeblock's calibration with a
good "initial guess" of what their solutions should be.
