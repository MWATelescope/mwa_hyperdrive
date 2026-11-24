# Other subcommands

The subcommands below don't have their own pages. As always, `--help` describes
all of their arguments.

## Sky-model source lists

See [this page](../defs/source_lists.md) for the supported source list formats.

~~~admonish example title="srclist-by-beam"
Reduce a source list to the `-n` brightest sources after beam attenuation, for
a pointing given by a metafits file (or by `--lst`, `--phase-centre`, `--freqs`
and `--delays`). The vetoing is the same as [is done before
modelling](vis_simulate/intro.md#vetoing).

```shell
hyperdrive srclist-by-beam -n 1000 -m *.metafits srclist.yaml srclist_1000.yaml
```
~~~

~~~admonish example title="srclist-convert"
Convert a source list from one format to another. The types are guessed from
the files where possible; otherwise use `-i` and `-o`.

```shell
hyperdrive srclist-convert srclist.yaml -o rts srclist_rts.txt
```
~~~

~~~admonish example title="srclist-verify"
Check that source lists can be read, and print their types and how many
sources and components they have.

```shell
hyperdrive srclist-verify srclist.yaml another_srclist.txt
```
~~~

~~~admonish example title="srclist-shift"
Shift the positions of sources, e.g. to correct for the ionosphere. The shifts
are given in a `.json` file with source names as keys, each with an `ra` and a
`dec` shift in degrees. Only the sources named in the `.json` file are written,
unless `--include-unshifted-sources` is given.

```shell
hyperdrive srclist-shift srclist.yaml shifts.json srclist_shifted.yaml
```
~~~

## Calibration solutions

~~~admonish example title="solutions-convert"
Convert calibration solutions from one [format](../defs/cal_sols.md) to
another. A directory is taken to be RTS solutions. A metafits file (`-m`) may
be needed, depending on the formats.

```shell
hyperdrive solutions-convert hyp_sols.fits ao_sols.bin
```
~~~

## MWA metadata

~~~admonish example title="dipole-gains"
Print the [dipole gains](../defs/mwa/dead_dipoles.md) listed by a metafits
file, i.e. which dipoles of which tiles are dead.

```shell
hyperdrive dipole-gains *.metafits
```
~~~
