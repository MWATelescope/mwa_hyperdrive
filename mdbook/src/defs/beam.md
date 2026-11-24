# Beam responses

Beam responses are given by
[`mwa_hyperbeam`](https://github.com/MWATelescope/mwa_hyperbeam) (the default)
or, if `hyperdrive` was compiled with the `everybeam` feature,
[EveryBeam](#everybeam).

To function, MWA beam code needs a few things:

- The [dipole delays](mwa/delays.md);
- The [dipole gains](mwa/dead_dipoles.md) (usually dead dipoles are 0, others are 1);
- The direction we want the beam response as an Azimuth-Elevation coordinate; and
- A frequency.

In addition, the FEE beam code needs an HDF5 file to function. See the
[post-installation instructions](../installation/post.md) for information on
getting that set up.

It is possible to use the "analytic" MWA beam, and this does not require an
additional file (but the FEE beam is the default selection). There are two
"flavours": `mwa_pb` and `RTS`, and these can be used by specifying `--beam-type
analytic-mwa_pb` (alias `mwa_pb`) and `--beam-type analytic-rts` (alias `rts`),
respectively (beam type names are case-insensitive). The differences
between the flavours is not huge, but I (CHJ) suggest the `RTS` flavour if in
doubt, as it seems to look a little better.

## Frequency resolution

The FEE beam's coefficients are only defined at multiples of 1.28 MHz (the MWA
coarse-channel width, i.e. at coarse-channel centres). When modelling
visibilities, FEE beam responses are computed at the file frequencies spanning
the fine channels (from the one at or below the lowest channel to the one at or
above the highest), and each fine channel's response is interpolated from them
with a not-a-knot cubic spline through the real and imaginary parts of each
Jones element. This is the same interpolation scheme used to make the 10 kHz
FEE file `MWA_embedded_element_pattern_rev2_interp_167_197MHz.h5` (a not-a-knot
cubic spline through each spherical-wave coefficient, with knots spanning the
band). The two are not identical, though: hyperdrive splines the
zenith-normalised responses, whereas the dense file normalises after splining
the coefficients, and the normalisation is not linear in frequency. The
difference is about 1e-6 near zenith and 1e-4 at 25° elevation, and both
suppress the coarse-channel harmonics equally.

The analytic beams are defined at any frequency, but they are splined the same
way from knots on the 1.28 MHz grid (with two extra knots beyond each end of
the band), because the modellers' beam-response arrays scale with the number
of knots; evaluating at every fine channel would need 32 times the memory for
40 kHz channels. The beam is smooth enough that the spline is within ~1e-6 of
the beam evaluated at each channel. Likewise, if a beam file sampled more
finely than 1.28 MHz is used, knots at least 1.28 MHz apart are chosen from it.

Fine channels outside the beam file's frequency range use the beam at the
file's first or last frequency (a warning is logged); the beam is not
extrapolated.

(Previously, every fine channel used the beam response at the nearest multiple
of 1.28 MHz, for both beam types, so the beam, and with it the sky model and
calibration solutions, was constant within each coarse channel and stepped at
every coarse-channel edge. Those 1.28 MHz-periodic steps put power at
multiples of 781.25 ns in delay space.)

![Sky model with nearest-frequency vs interpolated beam
responses](beam_interp_model.png)

## Errors

Beam code usually does not error, but if it does it's likely because:

1. There aren't exactly 16 dipole delays;
2. There aren't exactly 16 or 32 dipole gains per tile; or
3. There's something wrong with the FEE HDF5 file. The official file is well
   tested.

## EveryBeam

If `hyperdrive` was compiled with the `everybeam` feature (see [Installing with
EveryBeam support](../installation/everybeam.md)), `--beam-type everybeam`
uses EveryBeam for beam responses. This supports any telescope that EveryBeam
supports (e.g. SKA-Low, LOFAR, OSKAR-simulated arrays). EveryBeam reads the
telescope's description (station positions, element layouts and orientations,
pointing) from a measurement set:

- If the input visibilities are a measurement set, it is used by default.
- Otherwise (or to use a different one), supply it with `--beam-ms`.

The number of stations in the measurement set must match the number of tiles
in the input data, and the station order is assumed to be the same.

~~~admonish warning title="Files written by hyperdrive"
When `hyperdrive` writes a measurement set (e.g. with `vis-subtract` or
`solutions-apply`) from one with a `PHASED_ARRAY` table (e.g. SKA-Low or
OSKAR data), it copies that table and the telescope name, so EveryBeam can use
the output directly. Other telescope descriptions (e.g. LOFAR's) aren't copied,
and uvfits files can't hold one; when processing such outputs, point
`--beam-ms` at the original measurement set.
~~~

```shell
hyperdrive di-calibrate -d obs.ms -s srclist.yaml --beam-type everybeam
```

### Options

| Option | Description |
| ------ | ----------- |
| `--beam-ms` | The measurement set describing the telescope. |
| `--everybeam-element-model` | The element response model, e.g. `default`, `oskar_dipole_cos`, `oskar_dipole`, `skala40_wave`, `skalow_feko`, `hamaker`, `lobes`. The default depends on the telescope; for SKA-Low/OSKAR it is `oskar_dipole_cos`, which is the best match to real SKA-Low stations. |
| `--everybeam-mode` | `full` (default), `array_factor` or `element`. |
| `--everybeam-normalisation` | `none` (default), `amplitude`, `full`, `preapplied` or `preapplied_or_full`. See below. |
| `--everybeam-coeff-path` | Path to element-response coefficients (telescope dependent). |
| `--everybeam-field-id` | The measurement set `FIELD` used for the beam pointing (default 0). |
| `--everybeam-data-column` | The data column used to check for a pre-applied beam (LOFAR only). |
| `--everybeam-subband-frequency` | Use the subband frequency for the station beamformer (LOFAR only). |
| `--everybeam-frequency-interpolation` | Interpolate the beam over frequency (MWA only). |
| `--everybeam-knot-spacing` | The spacing of the frequencies at which EveryBeam is evaluated (default `1.28MHz`); responses are splined to the fine channels. `0` evaluates every fine channel. See below. |

### Polarisation conventions and normalisation

`hyperdrive`'s sky model uses the MWA [polarisation](pols.md) convention (X is
East-West, Y is North-South), whereas the IAU convention has X as North-South.
Only the sky side (the columns of the beam Jones matrices) is converted to
`hyperdrive`'s convention. The feed side (the rows) is left in the basis of the
telescope's feeds, so that it matches the visibilities.

- `none` (the default) uses each station's unnormalised response, so
  calibration solutions don't contain the beam's spectral response. `amplitude`
  instead scales each station's response, separately at each time and
  frequency, to unit amplitude in the direction of the measurement set's
  `FIELD` `REFERENCE_DIR`.
- With `none` or `amplitude` normalisation, the rows are each
  station's own feeds. For example, SKA-Low stations are rigidly rotated with
  respect to each other (as described in the measurement set's `PHASED_ARRAY`
  table), so each station's X and Y feeds are generally not East-West and
  North-South. This is appropriate for data that have not had a beam
  correction applied.
- With `full` (or `preapplied`) normalisation, EveryBeam multiplies the
  response by the inverse of the response in the `REFERENCE_DIR` direction. The rows are
  then in the (North, East) sky basis, i.e. the IAU order. This is only
  appropriate for data that have already had the beam at the phase centre
  corrected (e.g. with DP3's `applybeam`).

### Performance

EveryBeam only runs on the CPU. When GPU modelling is used, EveryBeam beam
responses are calculated on the CPU and copied to the GPU, which is slower than
the MWA FEE beam's GPU code.

For each timestep, every direction is converted to ITRF once, and each station's
response in a direction is calculated for all frequencies in one EveryBeam call
(as DP3 does). This work is spread over all CPU threads.

As for the analytic beams, EveryBeam is evaluated at frequencies 1.28 MHz apart
(`--everybeam-knot-spacing`), spanning the band plus two more on each side, and
the responses are interpolated to each fine channel with a not-a-knot cubic
spline (see [frequency resolution](#frequency-resolution)). This is much faster
than evaluating every fine channel (by the number of fine channels per
1.28 MHz). `--everybeam-knot-spacing 0` evaluates every fine channel instead,
which is exact.
