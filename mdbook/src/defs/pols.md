# Instrumental polarisations

In `hyperdrive` (and [`mwalib`](https://github.com/MWATelescope/mwalib) and
[`hyperbeam`](https://github.com/MWATelescope/mwa_hyperbeam)), the X
polarisation refers to the East-West dipoles and the Y refers to North-South.
Note that this contrasts with the IAU definition of X and Y, which is opposite
to this. However, this is consistent within the MWA.

MWA visibilities are ordered XX, XY, YX, YY (using the above definitions of X
and Y).

# Stokes polarisations

In `hyperdrive`:
- \\( \text{XX} = \text{I} + \text{Q} \\)
- \\( \text{XY} = \text{U} + i\text{V} \\)
- \\( \text{YX} = \text{U} - i\text{V} \\)
- \\( \text{YY} = \text{I} - \text{Q} \\)

where \\( \text{I} \\), \\( \text{Q} \\), \\( \text{U} \\), \\( \text{V} \\) are
Stokes polarisations and \\( i \\) is the imaginary unit.
