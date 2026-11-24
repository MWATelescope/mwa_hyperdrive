# Usage at Pawsey

garrawarla was a supercomputer dedicated to MWA activities, hosted by the
Pawsey Supercomputing Centre. It has been retired, and MWA processing at Pawsey
now happens on setonix. [This MWA wiki
page](https://wiki.mwatelescope.org/pages/viewpage.action?pageId=52068764)
details how `hyperdrive` was used on garrawarla; the `hyperdrive` commands
themselves are the same anywhere.

garrawarla's GPUs were NVIDIA V100s, for which `hyperdrive` is compiled with
the `cuda` feature. setonix's GPUs are AMD, so `hyperdrive` must be compiled
with the `hip` feature there; see the [installation
instructions](../../installation/from_source.md).
