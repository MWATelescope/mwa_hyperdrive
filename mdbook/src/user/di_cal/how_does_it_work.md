# How does it work?

`hyperdrive`'s direction-independent calibration is based off of a sky model.
That is, data visibilities are compared against "sky model" visibilities, and
the differences between the two are used to calculate antenna gains (a.k.a.
calibration solutions).

Here is the algorithm used to determine antenna gains in `hyperdrive`:

\\[ G_{p,i} = \frac{ \sum_{q,q \neq p} D_{pq} G_{q,i-1} M_{pq}^{H}}{ \sum_{q,q \neq p} (M_{pq} G_{q,i-1}^{H}) (M_{pq} G_{q,i-1}^{H})^{H} } \\]

where

- \\( p \\) and \\( q \\) are antenna indices;
- \\( G\_{p} \\) is the gain Jones matrix for an antenna \\( p \\);
- \\( D\_{pq} \\) is a "data" Jones matrix from baseline \\( pq \\);
- \\( M\_{pq} \\) is a "model" Jones matrix from baseline \\( pq \\);
- \\( i \\) is the current iteration index; and
- the \\( H \\) superscript denotes a [Hermitian
  transpose](https://en.wikipedia.org/wiki/Conjugate_transpose).

The maximum number of iterations can be changed at run time, as well as
thresholds of acceptable convergence (i.e. the amount of change in the gains
between iterations).

This iterative algorithm is done independently for every individual channel
supplied. This means that if, for whatever reason, part of the data's band is
problematic, the good parts of the band can be calibrated without issue.

## StefCal? MitchCal?

[It appears that](https://www.aoc.nrao.edu/~sbhatnag/misc/stefcal.pdf) StefCal
(as well as
[MitchCal](https://ui.adsabs.harvard.edu/abs/2008ISTSP...2..707M/abstract)) is
no different to "antsol" (i.e. the above equation).
