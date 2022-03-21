# Dipole delays

A tile's dipole delays control where it is "pointing". Delays are provided as
numbers, and this controls how long a dipole's response is delayed before its
response correlated with other dipoles. This effectively allows the MWA to be
more sensitive in a particular direction without any physical movement.

e.g. This set of dipole delays

```plaintext
 6  4  2  0
 8  6  4  2
10  8  6  4
12 10  8  6
```

has the North-East-most (top-right) dipole not being delayed, whereas all others
are delayed by some amount. See [this
page](https://wiki.mwatelescope.org/pages/viewpage.action?pageId=48005139) for
more info on dipole ordering.


Dipole delays are usually provided by [metafits](metafits.md) files, but can
also be supplied by command line arguments, e.g.

```shell
--delays 6 4 2 0 8 6 4 2 10 8 6 4 12 10 8 6
```

would correspond to the example above. Note that these user-supplied delays will
override delays that are otherwise provided.

Dipoles cannot be delayed by more than "31". "32" is code for ["dead
dipole"](dead_dipoles.md), which means that these dipoles should not be used
when modelling a tile's response.

## Ideal dipole delays

Most (all?) MWA observations use a single set of delays for all tiles. Dipole
delays are listed in two ways in a metafits file:

- In the `DELAYS` key in HDU 1; and
- For each tile in HDU 2.

The delays in HDU 1 are referred to as "ideal" dipole delays. A set of delays
are not ideal if any are "32" (i.e. dead).

However, the HDU 1 delays may all be "32". This is an indication from the
observatory that this observation is "bad" and should not be used. `hyperdrive`
will proceed with such observations but issue a warning. In this case, the ideal
delays are obtained by iterating over all tile delays until each delay is not
32.
