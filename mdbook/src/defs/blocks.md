# Timeblocks

A timeblock is an averaged unit of timesteps. The number of timesteps per
timeblock is determined by the user, but it is always at least 1. An observation
may be calibrated in multiple timeblocks, e.g. 4 timesteps per timeblock. If the
same observation has more than 4 timesteps, then there are multiple calibration
timeblocks, and time-varying effects can be seen. Here's a representation of an
observation with 10 timesteps and 4 timesteps per timeblock:

```text
Timeblock 1    Timeblock 2   Timeblock 3
[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
```

Timeblocks do not need to be contiguous and can be sparse, e.g. for an
observation containing 10 timesteps (starting at timestep 0):

```text
    Timeblock 1            Timeblock 2
[_, [1, _, 3],  [_, _, _], [_, _, 9]]
```

is a valid representation of how the data would be averaged if there are 3
timesteps per timeblock. In this case, the **timestamps** of each timeblock
correspond to the timestamps of timesteps 2 and 8.

Timeblock are also used in writing out averaged visibilities. If there are 4
timesteps per timeblock, then the output visibilities might be 4x smaller than
the input visibilities (depending on how the timesteps align with the
timeblocks).

# Chanblocks

Similar to timeblocks, chanblocks are averaged units of channels. Frequency
averaging is currently only implemented when writing out visibilities, so there
is not much discussion needed here, yet.
