# Getting started

Do you want to do some calibration, but don't know how to start? Can't remember
what that command-line argument is called? If ever you're in doubt, consult the
help text:

```shell
# Top-level help
hyperdrive --help

# di-calibrate help
hyperdrive di-calibrate --help
```

`di-calibrate` is one of many subcommands. Subcommands are accessed by typing
them after `hyperdrive`. Each subcommand accepts `--help` (as well as `-h`).
Detailed usage information on each subcommand can be seen in the table of
contents of this book. More information on subcommands as a concept is below.

~~~admonish info title="Subcommands"

`hyperdrive` itself is split into many subcommands. These are simple to list:

```shell
hyperdrive -h
# OR
hyperdrive --help
```

Output (edited for brevity):

```plaintext
Commands:
  di-calibrate       Perform direction-independent calibration on the input MWA data.
  peel               Solve for and subtract ionospheric sky-model.
  vis-convert        Convert visibilities from one type to another.
  vis-simulate       Simulate visibilities of a sky-model source list.
  vis-subtract       Subtract sky-model sources from supplied visibilities.
  solutions-apply    Apply calibration solutions to input data.
  solutions-plot     Plot calibration solutions.
  crosstalk-plot     Plot per-baseline crosstalk scores over many observations
  solutions-qa       Plot QA metrics of calibration solutions over many observations
  solutions-convert  Convert between calibration solution file formats.
  srclist-by-beam    Reduce a sky-model source list to the top N brightest sources
  srclist-convert    Convert a sky-model source list from one format to another
  srclist-verify     Verify that sky-model source lists can be read by hyperdrive
  srclist-shift      Shift the sources in a source list
  dipole-gains       Print information on the dipole gains listed by a metafits file
  beam               Generate beam-response values.
```

The help text for these is accessible in a similar way:

```shell
hyperdrive solutions-plot -h
# OR
hyperdrive solutions-plot --help
```

Output (edited for brevity; `-h` gives a shorter version of the same text):

```plaintext
Plot calibration solutions.
https://mwatelescope.github.io/mwa_hyperdrive/user/plotting.html

Usage: hyperdrive solutions-plot [OPTIONS] [SOLUTIONS_FILES]...

Arguments:
  [SOLUTIONS_FILES]...

Options:
  -r, --ref-tile <REF_TILE>
          The reference tile to use. If this isn't specified, the best one from the end is used
  -n, --no-ref-tile
          Don't use a reference tile. Using this will ignore any input for `ref_tile`
      --ignore-cross-pols
          Don't plot the leakage polarisations (D_x and D_y)
      --gains
          Plot the instrumental gains, i.e. the inverses of the stored solutions
      --min-amp <MIN_AMP>
          The minimum y-range value on the amplitude gain plots
      --max-amp <MAX_AMP>
          The maximum y-range value on the amplitude gain plots
  -o, --output-directory <OUTPUT_DIRECTORY>
          The directory to write the plots into
  -m, --metafits <METAFITS>
          The metafits file associated with the solutions. This provides additional information on
          the plots, like the tile names
  -v, --verbosity...
          The verbosity of the program. Increase by specifying multiple times (e.g. -vv)
  -h, --help
          Print help
  -V, --version
          Print version
```
~~~

~~~admonish tip title="Options common to all subcommands"
`--dry-run` checks the arguments and prints what would be done without doing
it. `--save-toml` writes the arguments to a file. The subcommands that list an
`[ARGUMENTS_FILE]` in their help (e.g. `di-calibrate`) can be given such a file
in place of command-line arguments, to repeat a run.
`--no-progress-bars` is useful when the output is going to a log file.
~~~

~~~admonish tip title="Shortcuts"

It's possible to save keystrokes when subcommands aren't ambiguous, e.g. use
`solutions-p` as an alias for `solutions-plot`:

```shell
hyperdrive solutions-p
<help text for "solutions-plot">
```

This works because there is no other subcommand that `solutions-p` could refer
to. On the other hand, `solutions` won't be accepted because both
`solutions-plot`, `solutions-convert` and others exist.

`di-c` works for `di-calibrate`. Unfortunately this is not perfect; the `-` is
required even though `di` should be enough.
~~~
