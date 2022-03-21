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
SUBCOMMANDS:
    di-calibrate
    vis-simulate
    solutions-convert
    solutions-plot
    srclist-by-beam
```

The help text for these is accessible in a similar way:

```shell
hyperdrive solutions-plot -h
# OR
hyperdrive solutions-plot --help
```

```plaintext
hyperdrive-solutions-plot 0.2.0-alpha.11
Plot calibration solutions. Only available if compiled with the "plotting" feature.

USAGE:
    hyperdrive solutions-plot [OPTIONS] [SOLUTIONS_FILES]...

ARGS:
    <SOLUTIONS_FILES>...

OPTIONS:
    -r, --ref-tile <REF_TILE>    The reference tile to use. If this isn't specified, the best one from the end is used
    -n, --no-ref-tile            Don't use a reference tile. Using this will ignore any input for `ref_tile`
        --ignore-cross-pols      Don't plot XY and YX polarisations
        --min-amp <MIN_AMP>      The minimum y-range value on the amplitude gain plots
        --max-amp <MAX_AMP>      The maximum y-range value on the amplitude gain plots
    -m, --metafits <METAFITS>    The metafits file associated with the solutions. This provides additional information on the plots, like the tile names
    -v, --verbosity              The verbosity of the program. Increase by specifying multiple times (e.g. -vv). The default is to print only high-level information
    -h, --help                   Print help information
    -V, --version                Print version information
```
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
`solutions-plot` and `solutions-convert` exist.

`di-c` works for `di-calibrate`. Unfortunately this is not perfect; the `-` is
required even though `di` should be enough.
~~~
