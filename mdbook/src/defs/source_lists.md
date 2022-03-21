# Sky-model source lists

`hyperdrive` performs sky-model calibration. Sky-model source lists describe
what the sky looks like, and the closer the sky model matches the data to be
calibrated, the better the calibration quality.

A sky-model source list is composed of many sources, and each source is composed
of at least one component. Each component has a position, a component type and a
flux-density type. Within the code, a source list is a tree structure
associating a source name to a collection of components.

Source list file formats have historically been bespoke. In line with
`hyperdrive`'s goals, `hyperdrive` will read many source list formats, but also
presents its own preferred format (which has no limitations within this
software). Each supported format is detailed on the following documentation
pages.

`hyperdrive` can also convert between formats, although in a "lossy" way;
non-`hyperdrive` formats cannot represent all component and/or flux-density
types.

~~~admonish info title="Supported formats"
- [`hyperdrive` format](source_list_hyperdrive.md)
- [André Offringa (`ao`) format](source_list_ao.md)
- [`RTS` format](source_list_rts.md)
~~~

~~~admonish info title="Conversion"
`hyperdrive` can convert (as best it can) between different source list formats.
`hyperdrive srclist-convert` takes the path to input file, and the path to the
output file to be written. If it isn't specified, the type of the input file
will be guessed. Depending on the output file name, the output source list type
may need to be specified.
~~~

~~~admonish info title="Verification"
`hyperdrive` can be given many source lists in order to test that they are
correctly read. For each input file, `hyperdrive srclist-verify` will print out
what kind of source list the file represents (i.e. `hyperdrive`, `ao`, `rts`,
...) as well as how many sources and components are within the file.
~~~

~~~admonish info title="Component types"
Each component in a sky model is represented in one of three ways:

- point source
- Gaussian
- shapelet

Point sources are the simplest. Gaussian sources could be considered the same as
point sources, but have details on their structure (major- and minor-axes,
position angle). Finally, shapelets are described the same way as Gaussians but
additionally have multiple "shapelet components". Examples of each of these
components can be found on the following documentation pages and in the [examples
directory](https://github.com/MWATelescope/mwa_hyperdrive/tree/main/examples).
~~~
