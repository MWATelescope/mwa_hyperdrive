# `mwalib`

[`mwalib`](https://github.com/MWATelescope/mwalib) is the official MWA
raw-data-reading library. `hyperdrive` users usually don't need to concern
themselves with it, but `mwalib` errors may arise.

## Errors

### Missing a key in the metafits file

`mwalib` does not support PPD metafits files; only new metafits files should be
used. See the [metafits](metafits.md) page for more info.

### Others

Hopefully the error message alone is clear enough! Please file a [GitHub
issue](https://github.com/MWATelescope/mwa_hyperdrive/issues) if something is
confusing.
