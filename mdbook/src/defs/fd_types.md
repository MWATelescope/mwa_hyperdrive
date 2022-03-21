# Flux-density types

This page describes supported flux-density types within `hyperdrive`. The
following pages detail their usage within sky-model source lists. [This
page](modelling/estimating.md) details how each type is estimated in modelling.

~~~admonish info title="Power laws and Curved power laws"
Most astrophysical sources are modelled as power laws. These are simply
described by a reference Stokes \\( \text{I} \\), \\( \text{Q} \\), \\( \text{U}
\\) and \\( \text{V} \\) flux density at a frequency \\( \nu \\) alongside a
spectral index \\( \alpha \\).

Curved power laws are formalised in Section 4.1 of [Callingham et al.
2017](https://ui.adsabs.harvard.edu/abs/2017ApJ...836..174C/abstract). These are
the same as power laws but with an additional "spectral curvature" parameter \\(
q \\).

Both kinds of power law flux-density representations are preferred in
`hyperdrive`.
~~~

~~~admonish info title="Flux density lists"
The list type is simply many instances of a Stokes \\( \text{I} \\), \\(
\text{Q} \\), \\( \text{U} \\) and \\( \text{V} \\) value at a frequency.
Example: this source (in the [RTS](source_list_rts.md) style) has 3 defined
frequencies for flux densities:

```plaintext
SOURCE J161720+151943 16.2889374 15.32883
FREQ 80.0e+6 1.45351 0 0 0
FREQ 100.0e+6 1.23465 0 0 0
FREQ 120.0e+6 1.07389 0 0 0
ENDSOURCE
```

In this case, Stokes \\( \text{Q} \\), \\( \text{U} \\) and \\( \text{V} \\) are
all 0 (this is typical), but Stokes \\( \text{I} \\) is 1.45351 Jy at 80 MHz,
1.23465 Jy at 100 MHz and 1.07389 Jy at 120 MHz. This information can be used to
estimate flux densities within the defined frequencies (\\( 80 <=
\nu_{\text{MHz}} <= 120 \\); interpolation) or outside the range (\\(
\nu_{\text{MHz}} < 80 \\) or \\( \nu_{\text{MHz}} > 120 \\); extrapolation).
~~~
