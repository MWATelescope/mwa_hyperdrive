# Solutions enhance

`solutions-enhance` takes the fine frequency structure of calibration solutions
from the auto-correlations of the data the solutions belong to, and writes out
new solutions. `enhance-solutions` can be used instead of `solutions-enhance`.

~~~admonish info title="Why?"
`di-calibrate` solves every frequency channel on its own, so the errors of an
incomplete sky model (and noise) go into the solutions as frequency structure.
When the solutions are applied, that structure moves foreground power to
delays (or \\( k_\parallel \\)) where there otherwise wouldn't be any.

The auto-correlation of a tile measures the same analogue signal chain as its
solutions (the gain squared, times the total power the tile receives) with much
more signal to noise than a per-channel solution, and without a sky model. The
total power is smooth in frequency, so the frequency structure of an
auto-correlation beyond a short delay is that of the tile's gain. This follows
[Li et al. (2019)](https://doi.org/10.3847/1538-4357/ab55e4), section 5.2.
~~~

## Usage

At the very least, this requires:
- Input data **containing auto-correlations** (with the flag `-d`)
  - [Supported formats](../defs/vis_formats_read.md)
- Calibration solutions for that data (with the flag `-s`)
  - [Supported formats](../defs/cal_sols.md)

~~~admonish example title="From an uncalibrated uvfits"
```shell
hyperdrive solutions-enhance -d obs.uvfits obs.metafits -s hyp_sols.fits -o hyp_sols_enhanced.fits
```
~~~

~~~admonish example title="From raw MWA data"
```shell
hyperdrive solutions-enhance -d *gpubox*.fits *.metafits *.mwaf -s hyp_sols.fits -o hyp_sols_enhanced.fits
```
~~~

The data must not be calibrated already, and should be the data that the
solutions were made from (the same observation and the same raw-data
corrections; when reading raw MWA data, the corrections listed in the solutions
file are used). The enhanced solutions are used like any others, e.g. with
[`solutions-apply`](solutions_apply/intro.md).

## What it does

For every tile, and for the XX and YY terms of its solutions separately:

```text
ln(solution) = lowpass[ ln(input solution) ] + highpass[ u + i v ]
```

- `u` is the log amplitude from the auto-correlation, \\( -\frac{1}{2}
  \ln(\text{auto}) \\) (`hyperdrive` solutions are the inverses of the gains).
  The auto-correlation used is the median over the timesteps of the solution
  timeblock (channels and timesteps flagged in the data are not used; flagged
  channels are bridged).
- `v` is the phase that goes with `u` for a causal response. An analogue chain
  (filters, cable reflections) only delays the signal, so the phase ripple that
  goes with an amplitude ripple is known (it is the Hilbert transform of `u`).
  Nothing is fitted to the phases of the input solutions for this, so there are
  no phase jumps from fitting wrapped phases. `--amplitude-only` leaves the
  phases of the input solutions as they are.
- The split is at `--delay-cut` (default 0.1 μs), the half-power point of a
  Gaussian low-pass filter in frequency (applied to what a cubic leaves).
  Below it, the input solutions are kept: their overall level and slope, and
  the slow structure in which the total power and the response to the sky
  differ (sky spectrum, beam, receiver noise, mutual coupling).

The cross terms (XY and YX) of the solutions are not changed. A phase that all
tiles share in a channel, which cancels in every visibility, is removed from
the solutions.

A tile's solutions are left as they are when its auto-correlation doesn't
describe them: when the fine structure of its solutions and of its
auto-correlation differ (averaged over 8 channels) by more than
`--max-mismatch` (default 5) times the median difference over tiles. This
happens for tiles with broken dipoles, for example, which receive a total
power with a different spectrum to their response towards the field. These
tiles are listed in the output. `--max-mismatch 0` enhances every tile.

For each timeblock and polarisation, the number of tiles enhanced is reported,
along with the size of the fine structure in the auto-correlations, the size of
the structure in the input solutions that isn't in the auto-correlations (i.e.
what is removed), and the slope of the fine phase structure of the input
solutions against that derived from the auto-correlations. This slope is about
1 for low-noise solutions and less (but positive) for noisier ones; a negative
slope means that the solutions don't belong to this data, and a warning is
given.

~~~admonish warning title="Limitations"
- The solutions should have the frequency resolution of the data or be coarser
  (auto-correlations are averaged over the channels of a solution chanblock),
  and need at least 16 channels.
- The phase derived from the auto-correlations is less accurate near the ends
  of the band (within a few tens of channels), where a ripple can't be
  followed past the edge.
- Structure that is in the auto-correlations but not in a tile's response to
  the sky, at delays above the cut, ends up in the solutions (e.g. narrow-band
  RFI that isn't flagged in the data).
~~~
