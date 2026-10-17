# Constraining solutions (DP3-style)

~~~admonish info title="Summary"
`--solve-mode` restricts the form of the Jones matrices (diagonal, scalar,
phase only, or a phase model across frequency) and `--smoothness-constraint`
smooths the solutions across frequency. Both are applied between iterations,
the way DP3's DDECal applies its constraints, and both use DP3's names and
units where DP3 has an equivalent.
~~~

By default `di-calibrate` solves for an unconstrained 2x2 Jones matrix per
tile, per chanblock, per timeblock, with every chanblock solved independently.
Two kinds of constraint change this:

- a **solve mode** (`--solve-mode`, DP3's `mode`) projects every tile's Jones
  matrix onto a simpler form after each iteration, and
- a **smoothness constraint** (`--smoothness-constraint`, DP3's
  `smoothnessconstraint`) replaces each tile's solutions with a kernel-smoothed
  version across frequency after each iteration.

When either is used, all chanblocks of a timeblock are iterated together: one
StefCal update is made for every chanblock, the constraints are applied across
the chanblocks, and the next update starts from the constrained solutions.
Iteration stops when every chanblock has converged (or `--max-iterations` is
reached), and the convergence thresholds mean the same thing as before. The
constraints are applied once more at the end so the written solutions satisfy
them exactly.

The constraints act on the gains in the sense of DP3 (`data ≈ G model Gᴴ`);
the solutions `hyperdrive` writes out are the inverses of these, as usual.

## Solve modes

| `--solve-mode`    | Jones matrix                                  | DP3 `mode`        |
|-------------------|-----------------------------------------------|-------------------|
| `fulljones`       | unconstrained (default)                       | `fulljones`       |
| `diagonal`        | XX and YY complex gains, no leakage           | `diagonal`        |
| `diagonalphase`   | XX and YY phases, unit amplitude              | `diagonalphase`   |
| `diagonalamplitude` | XX and YY real amplitudes                   | `diagonalamplitude` |
| `scalar`          | one complex gain for XX and YY                | `scalar`          |
| `scalarphase`     | one phase for XX and YY                       | `scalarphase`     |
| `scalaramplitude` | one amplitude for XX and YY                   | `scalaramplitude` |
| `tec`             | scalar phase `φ(ν) = a/ν`                     | `tec`             |
| `tec+phase`       | scalar phase `φ(ν) = a/ν + b`                 | `tec+phase`       |
| `tec+delay`       | scalar phase `φ(ν) = a/ν + cν`                | `tec+delay`       |
| `tec+phase+delay` | scalar phase `φ(ν) = a/ν + b + cν`            | `tec+phase+delay` |
| `delay`           | scalar phase `φ(ν) = cν` (linear in frequency) | (none)           |
| `phase+delay`     | scalar phase `φ(ν) = b + cν`                  | (none)            |

The phase-model modes fit the per-tile phases across all chanblocks of a
timeblock, as DP3's `TecConstraint`/`TecOffsetDelayConstraint` do: the phases
are first referenced to the first tile with enough unflagged chanblocks, then
the model is fitted with a grid search over the wrapping terms (`a` and `c`,
spanning `--max-tec-delay-wraps` turns across the band; DP3's
`max_tec_delay_wraps`, default 5) followed by weighted linear least squares on
the wrapped residuals. The fitted TEC (TECU), phase (rad) and delay (s) per
tile are logged at debug verbosity. Unlike DP3, residuals are re-centred on
their circular mean before each linear fit, so offsets near ±π are fitted
correctly.

The per-chanblock data weights of each tile (the summed weights of its
baselines) are used as the fit and smoothing weights, as in DP3; flagged
chanblocks and failed tiles get zero weight.

## Smoothing

`--smoothness-constraint <SIZE>` smooths each tile's XX, XY, YX and YY
solutions (XX and YY only with a diagonal or scalar mode) with a Gaussian
kernel across frequency. `SIZE` is DP3's `smoothnessconstraint`: the kernel
is `exp(-9 (Δν / SIZE)²)` (σ = SIZE / 3), truncated at the kernel size, and
accepts a unit (`2MHz`, `400kHz`, or Hz). The remaining options map onto
DP3's:

| hyperdrive                            | DP3                            | default |
|---------------------------------------|--------------------------------|---------|
| `--smoothness-ref-frequency <FREQ>`   | `smoothnessreffrequency`       | off     |
| `--smoothness-spectral-exponent <X>`  | `smoothnessspectralexponent`   | -1      |
| `--smoothness-ref-distance <METRES>`  | `smoothnessrefdistance`        | off     |
| `--smoothness-ref-tile <NAME>`        | `smoothnessrefantenna`         | first unflagged tile |
| `--no-smoothness-kernel-truncation`   | `smoothness_kernel_truncation=false` | truncated |

With a reference frequency, the kernel size scales as `(ν / ν_ref)` to the
power of minus the spectral exponent, i.e. with the default exponent of -1 the
kernel grows with frequency, matching the 1/ν behaviour of ionospheric phases.
With a reference distance, tiles at distance `d` from the reference tile use a
kernel scaled by `ref_distance / d` (closer tiles are smoothed more; the
reference tile gets the largest factor).

## Examples

Diagonal gains smoothed over 2 MHz:

```shell
hyperdrive di-calibrate -d *.uvfits *.metafits -s srclist.yaml \
    --solve-mode diagonal --smoothness-constraint 2MHz
```

A scalar phase per tile that is linear in frequency (a delay plus an offset),
for a self-calibration pass against a single direction:

```shell
hyperdrive di-calibrate -d *.uvfits *.metafits -s srclist.yaml \
    --named-sources J0444-2809 --solve-mode phase+delay
```

The equivalent DP3 settings for the first example are
`ddecal.mode=diagonal ddecal.smoothnessconstraint=2e6`; for the second,
DP3's closest mode is `tec+phase+delay`, which `hyperdrive` also provides.
