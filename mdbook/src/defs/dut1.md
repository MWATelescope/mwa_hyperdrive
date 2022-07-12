# DUT1

[DUT1](https://en.wikipedia.org/wiki/DUT1) is the difference between the UT1 and
UTC time frames. In short, using the DUT1 allows a better representation of the
local sidereal time within `hyperdrive`.

Since July 2022, [MWA metafits files](./mwa/metafits.md) contain a key `DUT1`
populated by `astropy`.

If available, uvfits files display the DUT1 with the `UT1UTC` key in the antenna
table HDU. However, the times in `hyperdrive`-written uvfits files will still be
in the UTC frame, as if there was no DUT1 value.

Measurement sets don't appear to have a way of displaying what the DUT1 value
is; when writing out measurement sets, `hyperdrive` will change the time frame
of the `TIME` and `TIME_CENTROID` columns from UTC to UT1 iff the DUT1 is non
zero.

## More explanation

A lot of good, easy-to-read information is
[here](https://lweb.cfa.harvard.edu/~jzhao/times.html).

UTC keeps track with TAI but only through the aid of leap seconds (both are
"atomic time frames"). UT1 is the "actual time", but the Earth's rate of
rotation is difficult to measure and predict. DUT1 is not allowed to be more
than -0.9 or 0.9 seconds; a leap second is introduced before that threshold is
reached.
