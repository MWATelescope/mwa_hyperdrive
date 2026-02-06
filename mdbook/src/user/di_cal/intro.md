# DI calibration

Direction-Independent (DI) calibration "corrects" raw telescope data.
`hyperdrive` achieves this with "sky model calibration". This can work very
well, but relies on two key assumptions:

- The sky model is an accurate reflection of the input data; and
- The input data are not too contaminated (e.g. by radio-frequency
  interference).

A high-level overview of the steps in `di-calibrate` are below. Solid lines
indicate actions that always happen, dashed lines are optional:

```mermaid
%%{init: {'theme':'dark', 'themeVariables': {'fontsize': 20}}}%%
flowchart TD
    InputData[fa:fa-file Input data files]-->Args
    SkyModel[fa:fa-file Sky-model source-list file]-->Args
    Settings[fa:fa-cog Other settings]-.->Args

    Args[fa:fa-cog User arguments]-->Valid{fa:fa-code Valid?}
    Valid --> cal

    subgraph cal[For all timeblocks]
        Read[fa:fa-code Read a timestep\nof input data]
        Model["fa:fa-code Generate model vis\n (CPU or GPU)"]
        Model-.->WriteModelVis[fa:fa-save Write model visibilities]

        LSQ[fa:fa-code Calibrate via least squares]
        Read-->LSQ
        Model-->LSQ
        LSQ-->|Iterate|LSQ
        LSQ-->Sols[fa:fa-wrench Accumulate\ncalibration solutions]
    end

    cal-->WriteSols[fa:fa-save Write calibration solutions]
```

~~~admonish info title="Model visibility outputs"
If `--model-filenames` is supplied, model visibilities are written for inspection.
Auto-correlations are not read by default. Use `--autos` when reading input data to include them. Outputs match the input: if input data includes auto-correlations, they are written; if input data excludes them (default), they are not written.
~~~
