# Subtract visibilities

`vis-subtract` can subtract the sky-model visibilities from calibrated data
visibilities and write them out. This can be useful to see how well the sky
model agrees with the input data, although direction-dependent effects (e.g. the
ionosphere) may be present and produce "holes" in the visibilities, e.g.:

![](subtracted.jpg)

A high-level overview of the steps in `vis-subtract` are below. Solid lines
indicate actions that always happen, dashed lines are optional:

```mermaid
%%{init: {'theme':'dark', 'themeVariables': {'fontsize': 20}}}%%
flowchart TD
    InputData[/fa:fa-file Calibrated input data/]-->Args
    CalSols[/fa:fa-file Sky-model source-list file/]-->Args
    Settings[/fa:fa-cog Other settings/]-.->Args

    Args[fa:fa-cog User arguments]-->Valid{fa:fa-code Valid?}
    Valid --> subtract

    subgraph subtract[For all timesteps]
        Read["fa:fa-code Read a timestep
        of input data"]
        Simulate["fa:fa-code Generate model vis"]
        Read & Simulate-->Subtract["fa:fa-minus Subtract model vis from data"]
        Subtract-->Write[fa:fa-save Write timeblock
        visibilities]
    end
```
