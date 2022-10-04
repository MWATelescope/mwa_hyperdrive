# The `RTS` calibration solutions format

![](https://media.giphy.com/media/NsIwMll0rhfgpdQlzn/giphy.gif)

This format is extremely complicated and therefore its usage is discouraged.
However, it is possible to convert `RTS` solutions to one of the other supported
formats; a metafits file is required, and the *directory* containing the
solutions (i.e. `DI_JonesMatrices` and `BandpassCalibration` files) is supplied:

~~~admonish example title="Converting `RTS` solutions to another format"
```shell
hyperdrive solutions-convert /path/to/rts/solutions/ rts-as-hyp-solutions.fits -m /path/to/obs.metafits
```
~~~

Once in another format, the solutions can also be plotted.

An example of RTS solutions can be found in the `test_files` directory (as a
`.tar.gz` file). The code to read the solutions attempts to unpack and clarify
the format, but it is messy.

~~~admonish warning title="Writing `RTS` solutions"
I (CHJ) spent a very long time trying to make the writing of `RTS` solutions
possible, but ultimately gave up. One particularly difficult detail here is that
the RTS solutions contain a beam response; this could be either the MWA analytic
or FEE beam. But its application to the solutions is not clear and difficult to
reverse-engineer.

If you dare, there is incomplete commented-out code within `hyperdrive` that
attempts to write out the solutions.
~~~
