# Picket fence observations

A "picket fence" observation contains more than one "spectral window" (or
"SPW"). That is, not all the frequency channels in an observation are
continuous; there's at least one gap somewhere.

`hyperdrive` does not currently support picket fence observations, but it will
eventually support them properly. *However*, it is possible to calibrate a
single SPW of a picket fence obs. with `hyperdrive`; e.g. MWA observation
[1329828184](https://ws.mwatelescope.org/observation/obs/?obs_id=1329828184) has
12 SPWs. If all 24 raw data files are given to `hyperdrive`, it will refuse to
interact with the data. But, if you supply one of the SPWs, e.g. coarse channels
62 and 63, `hyperdrive` will calibrate and provide solutions for the provided
channels, i.e.

```shell
hyperdrive di-calibrate \
    -d *ch{062,063}*.fits *.metafits \
    -s srclist_pumav3_EoR0aegean_EoR1pietro+ForA_phase1+2_TGSSgalactic.txt \
    -n 100 \
    -o hyp_sols.fits
```

For this example, the output contains solutions for 256 channels, and only one
channel did not converge.
