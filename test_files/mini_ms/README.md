# mini.ms

A tiny synthetic (non-MWA) measurement set: 8 stations, 2 timesteps, and 32
channels of width 5425.35 Hz starting at 100 MHz, with all-zero visibilities.
The channel width is deliberately not a whole number of hertz, so that the
channel centres round to integer hertz with alternating spacings (5425 and
5426 Hz). It is used to test that such channel grids are read, grouped into
chanblocks, averaged and written correctly.

Regenerate it with `make_mini_ms.py` (needs python-casacore):

```
python3 make_mini_ms.py mini.ms
```

The generator is adapted from Alec Thomson's hyperdrive fork
(https://github.com/AlecThomson/mwa_hyperdrive, where it lives at
`test_files/everybeam/make_mini_ms.py` and makes a 1 MHz-channel version for
EveryBeam tests). The PHASED_ARRAY sub-table and SKA-LOW telescope name are
kept from there so the same file can serve EveryBeam; hyperdrive's own beams
ignore them.
