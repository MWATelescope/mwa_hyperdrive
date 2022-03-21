# Calibration solutions file formats

Calibration solutions are Jones matrices that, when applied to raw data,
"calibrate" the visibilities.

`hyperdrive` can convert between supported formats (see `solutions-convert`).
Soon it will also be able to apply them (but users can write out calibrated
visibilities as part of `di-calibrate`).

- [`hyperdrive` format](cal_sols_hyp.md)
- [André Offringa (`ao`) format](cal_sols_ao.md)
- [`RTS` format](cal_sols_rts.md)
