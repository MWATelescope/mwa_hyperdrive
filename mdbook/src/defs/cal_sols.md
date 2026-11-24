# Calibration solutions file formats

Calibration solutions are Jones matrices that, when applied to raw data,
"calibrate" the visibilities.

`hyperdrive` can convert between supported formats (see `solutions-convert`)
and apply them to visibilities (see
[`solutions-apply`](../user/solutions_apply/intro.md)).

- [`hyperdrive` format](cal_sols_hyp.md)
- [André Offringa (`ao`) format](cal_sols_ao.md)
- [`RTS` format](cal_sols_rts.md)
