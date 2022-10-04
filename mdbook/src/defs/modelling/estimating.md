# Estimating flux densities

The algorithm used to estimate a sky-model component's flux density depends on
the [flux-density type](../fd_types.md).

Note that in the calculations below, flux densities are allowed to be negative.
It is expected, however, that a sky-model component with a negative flux density
belongs to a source with multiple components, and that the overall flux density
of that source at any frequency is positive. A source with a negative flux
density is not physical.

~~~admonish info title="Power laws and Curved power laws"
Both power-law and curved-power-law sources have a spectral index (\\( \alpha
\\)) and a reference flux density (\\( S_0 \\)) defined at a particular
frequency (\\( \nu_0 \\)). In addition to this, curved power laws have a
curvature term (\\( q \\)).

To estimate a flux density (\\( S \\)) at an arbitrary frequency (\\( \nu \\)), a
ratio is calculated:

\\[ r = \left(\frac{\nu}{\nu_0}\right)^\alpha \\]

For power laws, \\( S \\) is simply:

\\[ S = S_0 r \\]

whereas another term is needed for curved power laws:

\\[ c = \exp\left({q \ln\left(\frac{\nu}{\nu_0}\right)^2 }\right) \\]
\\[ S = S_0 r c \\]

\\( S \\) can represent a flux density for Stokes \\( \text{I} \\), \\( \text{Q}
\\), \\( \text{U} \\) or \\( \text{V} \\). The same \\( r \\) and \\( c \\)
values are used for each Stokes flux density.
~~~

<!-- ~~~admonish info title="Flux density lists" -->
To estimate a flux density (\\( S \\)) at an arbitrary frequency (\\( \nu \\)),
a number of considerations must be made.

In the case that a list only has one flux density, we must assume that it is a
power law, use a default spectral index (\\( -0.8 \\)) for it and follow the
algorithm above.

In all other cases, there are at least two flux densities in the list (\\( n >=
2 \\)). We find the two list frequencies (\\( \nu_i \\)) and (\\( \nu_j \\))
closest to \\( \nu \\) (these can both be smaller and larger than \\( \nu \\)).
If the flux densities \\( S_i \\) and \\( S_j \\) are both positive or both
negative, we proceed with the power law approach: A spectral index is calculated
with \\( \nu_i \\) and \\( \nu_j \\) (\\( \alpha \\)) and used to estimate a
flux density with the power law algorithm. If \\( \alpha < -2.0 \\), a
trace-level message is emitted, indicating that this is a very steep spectral
index.

If the signs of \\( S_i \\) and \\( S_j \\) are opposites, then we cannot fit a
spectral index. Instead, we fit a straight between \\( S_i \\) and \\( S_j \\)
and use the straight line to estimate \\( S \\).

No estimation is required when \\( \nu \\) is equal to any of the list
frequencies \\( \nu_i \\).
<!-- ~~~ -->


~~~admonish danger title="Concerns on list types"
When estimating flux densities from a list, it is feared that the "jagged" shape
of a component's spectral energy distribution introduces artefacts into an EoR
power spectrum.

It is relatively expensive to estimate flux densities from a list type. For all
these reasons, users are strongly encouraged to not use list types where
possible.
~~~
