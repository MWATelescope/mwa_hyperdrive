# Solutions apply

`solutions-apply` takes calibration solutions and applies them to input
visibilities before writing out visibilities. All input formats are supported,
however `hyperdrive`-style calibration solutions are preferred because they are
unambiguous when applying multiple timeblocks.

`apply-solutions` can be used instead of `solutions-apply`.

A high-level overview of the steps in `solutions-apply` are below. Solid lines
indicate actions that always happen, dashed lines are optional:

```mermaid
%%{init: {'theme':'dark', 'themeVariables': {'fontsize': 20}}}%%
flowchart TD
    InputData[fa:fa-file Input data files]-->Args
    CalSols[fa:fa-wrench Calibration\nsolutions]-->Args
    Settings[fa:fa-cog Other settings]-.->Args

    Args[fa:fa-cog User arguments]-->Valid{fa:fa-code Valid?}
    Valid --> apply

    subgraph apply[For all timesteps]
        Read[fa:fa-code Read a timestep\nof input data]
        Read-->Apply["fa:fa-code Apply calibration\nsolutions to timeblock"]
        Apply-->Write[fa:fa-save Write timeblock\nvisibilities]
    end
```
