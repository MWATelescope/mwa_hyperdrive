# Raw data corrections

A number of things can be done to "correct" or "pre-process" raw MWA data before
it is ready for calibration (or other analysis). These tasks are handled by
[`Birli`](https://github.com/MWATelescope/Birli), either as the `Birli`
executable itself, or internally in `hyperdrive`.
[`cotter`](https://github.com/MWATelescope/cotter) used to perform these tasks
but it has been superseded by `Birli`.

~~~admonish info title="Geometric correction (a.k.a. phase tracking)"
Many MWA observations do not apply a geometric correction despite having a
desired phase centre. This correction applies

\\[ e^{-2 \pi i w_f / \lambda} \\]

to each visibility; note the dependence on baseline \\( w \\) and frequency.

Not performing the geometric correction can have a dramatically adverse effect
on calibration!
~~~

~~~admonish info title="PFB gains"
The poly-phase filter bank used by the MWA affects visibilities before they get
saved to disk. Over time, a number of "flavours" of these gains have been used:

- "Jake Jones" (`jake`; 200 Hz)
- "cotter 2014" (`cotter2014`; 10 kHz)
- "RTS empirical" (`empirical`; 40 kHz)
- "Alan Levine" (`levine`; 40 kHz)

When correcting raw data, the "Jake Jones" gains are used by default. For each
flavour, the first item in the parentheses (e.g. `cotter2014`) indicates what
should be supplied to `hyperdrive` if you want to use those gains instead. There
is also a `none` "flavour" if you want to disable PFB gain correction.

In CHJ's experience, using different flavours have very little effect on
calibration quality.

Some more information on the PFB can be found
[here](https://wiki.mwatelescope.org/display/MP/RRI+Receiver+PFB+Filter).
~~~

~~~admonish info title="Cable lengths"
Each tile is connected by a cable, and that cable might have a different length
to others. This correction aims to better align the signals of each tile.
~~~

~~~admonish info title="Digital gains"
```shell
todo!()
```
~~~
