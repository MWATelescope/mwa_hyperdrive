# Summary

[Introduction](README.md)

---

# Installation

- [How do I install hyperdrive?](installation/intro.md)
  - [Pre-compiled](installation/pre_compiled.md)
  - [From source](installation/from_source.md)
  - [Post installation](installation/post.md)

# User Guide

- [Introduction](user/intro.md)
- [Getting started](user/help.md)
- [DI calibration](user/di_cal/intro.md)
  - [Tutorial](user/di_cal/tutorial.md)
  - [Simple usage](user/di_cal/simple.md)
  - [Getting calibrated data](user/di_cal/out_calibrated.md)
  - [Advanced usage]()
    - [Varying solutions over time](user/di_cal/advanced/time_varying.md)
  - [Usage on garrawarla](user/di_cal/garrawarla.md)
  - [How does it work?](user/di_cal/how_does_it_work.md)
- [Apply solutions](user/solutions_apply/intro.md)
  - [Simple usage](user/solutions_apply/simple.md)
- [Plot solutions](user/plotting.md)
- [Convert visibilities](user/vis_convert/intro.md)
- [Simulate visibilities](user/vis_simulate/intro.md)
- [Subtract visibilities](user/vis_subtract/intro.md)
- [Get beam responses](user/beam.md)

---

# Definitions and Concepts

- [Polarisations](defs/pols.md)
- [Supported visibility formats]()
  - [Read](defs/vis_formats_read.md)
  - [Write](defs/vis_formats_write.md)
- [MWA-specific details]()
  - [Metafits files](defs/mwa/metafits.md)
  - [Dipole delays](defs/mwa/delays.md)
  - [Dead dipoles](defs/mwa/dead_dipoles.md)
  - [mwaf flag files](defs/mwa/mwaf.md)
  - [Raw data corrections](defs/mwa/corrections.md)
  - [Picket fence obs](defs/mwa/picket_fence.md)
  - [mwalib](defs/mwa/mwalib.md)
- [Sky-model source lists](defs/source_lists.md)
  - [Flux-density types](defs/fd_types.md)
  - [hyperdrive format](defs/source_list_hyperdrive.md)
  - [André Offringa (ao) format](defs/source_list_ao.md)
  - [RTS format](defs/source_list_rts.md)
  - [FITS format](defs/source_list_fits.md)
- [Calibration solutions file formats](defs/cal_sols.md)
  - [hyperdrive format](defs/cal_sols_hyp.md)
  - [André Offringa (ao) format](defs/cal_sols_ao.md)
  - [RTS format](defs/cal_sols_rts.md)
- [Beam responses](defs/beam.md)
- [Modelling visibilities](defs/modelling/intro.md)
  - [Measurement equation](defs/modelling/rime.md)
  - [Estimating flux densities](defs/modelling/estimating.md)
- [Coordinate systems](defs/coords.md)
- [DUT1](defs/dut1.md)
- [Terminology]()
  - [Timeblocks and chanblocks](defs/blocks.md)

---

# Developer Guide

- [Multiple-dimension arrays (ndarray)](dev/ndarray.md)
- [Non-empty vectors (vec1)](dev/vec1.md)
