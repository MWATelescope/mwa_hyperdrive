# Get beam responses

The `beam` subcommand can be used to obtain beam responses from any of the
supported beam types. The output format is
[tab-separated values](https://en.wikipedia.org/wiki/Tab-separated_values) (`tsv`).

The responses are calculated by moving the zenith angle from 0 to the `--max-za`
in steps of `--step`, then for each of these zenith angles, moving from 0 to
\\( 2 \pi \\) in steps of `--step` for the azimuth. Using a smaller `--step`
will generate many more responses, so be aware that it might take a while.

~~~admonish danger title="CUDA/HIP"
If CUDA or HIP is available to you, the `--gpu` flag will generate the beam
responses on the GPU, vastly decreasing the time taken.
~~~

~~~admonish example title="`Python` example to plot beam responses"
```python
{{#include ../../../examples/plot_beam_responses.py}}
```
~~~
