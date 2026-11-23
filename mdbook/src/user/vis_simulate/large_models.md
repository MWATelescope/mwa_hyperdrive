# Tutorial: a large sky model on a GPU

A sky model with hundreds of thousands of components does not have to be split
into several source lists to fit on a GPU. `vis-simulate` (and anything else
that models visibilities on a GPU: `di-calibrate`, `vis-subtract`, `peel`)
works through the model in chunks of components that fit in the device memory
that is free, so the whole model can go through in one pass.

This tutorial simulates a diffuse model of 640,000 point sources, each with its
own flux density in 384 frequency channels, on one GPU of a Setonix GPU node
(an AMD MI250X with 64 GiB of memory). The same commands work on an NVIDIA
card with a CUDA build of `hyperdrive`.

## What takes up memory

| What                           | Where        | Size                                                      |
|:-------------------------------|:-------------|:----------------------------------------------------------|
| `list` flux densities          | host and GPU | components × channels × 64 bytes                          |
| Beam responses, per chunk      | GPU          | unique tiles × beam frequencies × chunk components × 64 bytes, twice that when the beam is interpolated in frequency |
| Shapelet UVs, per chunk        | GPU          | baselines × chunk components × 16 bytes                   |
| Visibilities, per timestep     | host and GPU | channels × baselines × 32 bytes                           |

Sizes are for the default double-precision GPU build; the `gpu-single` feature
halves the GPU sizes.

Only the per-chunk rows are chunked. Flux densities of
[`list` type](../../defs/fd_types.md) are kept on the GPU for every component
and channel for the whole run, so they have to fit: 640,000 components × 384
channels is 15.7 GB. Power laws and curved power laws need a few numbers per
component instead, so use them when the model allows it.

The number of unique tiles is 1 when all tiles have the same beam, and can be
as many as the number of tiles when [dead
dipoles](../../defs/mwa/dead_dipoles.md) make them differ. `--unity-dipole-gains`
ignores dead dipoles and brings it back to 1.

## The source list

Any [source-list format](../../defs/source_lists.md) works. For a model of this
size use the [FITS format](../../defs/source_list_fits.md), which is much
faster to read than the text formats. In this example it is
`srclist.fits`, a 2.0 GB file with one row per component and channel.

## Running it

~~~admonish example title="Slurm script for one Setonix GPU"
```shell
#!/bin/bash -l
#SBATCH --account=${PAWSEY_PROJECT}-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

module load singularity/4.3.5-nohost

singularity exec --rocm -B /scratch -B /software hyperdrive.sif \
    hyperdrive vis-simulate \
    -m 1443896376.metafits \
    -s srclist.fits --source-list-type fits \
    --beam-file mwa_full_embedded_element_pattern.h5 \
    --num-timesteps 15 --time-res 8 \
    --num-fine-channels 384 --freq-res 80 --middle-freq 182.42 \
    --veto-threshold=-1e9 --source-dist-cutoff 90 \
    --output-autos \
    --no-progress-bars \
    -o model.uvfits
```
~~~

`hyperdrive.sif` is a container image of a `hyperdrive` built with the `hip`
feature for the node's GPUs (`HIP_ARCH=gfx90a`); see [installing from
source](../../installation/from_source.md).

A few of these arguments matter for a large model:

- There is no `--mem`: a Setonix GPU job gets 29 GB of host memory for each
  GPU and can't ask for another amount. The source list is held in host memory
  while it is read and copied to the GPU, and this model peaks at 13 GiB.
  (Earlier versions of `hyperdrive` needed several times that for it, and the
  job was killed.)
- `--veto-threshold=-1e9 --source-dist-cutoff 90` keep every component above
  the horizon; without them the faintest sources are
  [vetoed](intro.md#vetoing).
- The number of sources is not given, so all of them are used.

`hyperdrive` reports the chunking when it is needed. On the 64 GiB MI250X:

```text
Modelling 640000 sky-model components in chunks of 487328 to fit in GPU memory
```

## Limiting the GPU memory used

By default a chunk is sized to use most of the device memory that is free when
modelling starts. If something else shares the GPU, or to leave headroom, give
a limit in GiB for the per-chunk buffers:

```shell
hyperdrive vis-simulate ... --gpu-mem-limit 8
```

Smaller chunks give the same visibilities; they only cost a little time.

## What to expect

This model with one timestep (`--num-timesteps 1`), in one pass:

| GPU                              | Chunks       | Peak host memory | Peak GPU memory | Wall time |
|:---------------------------------|:-------------|:-----------------|:----------------|:----------|
| AMD MI250X, 64 GiB (Setonix, HIP)| 2 of 487,328 | 12.8 GiB         | 63.1 GiB        | 17.9 min  |
| NVIDIA RTX 3090, 24 GiB (CUDA)   | 12 of 57,149 | 12.0 GiB         | 23.0 GiB        | 34.9 min  |

On Setonix, 6.5 of the 17.9 minutes are spent reading the source list and
another minute setting up, both only once; each timestep then takes about 10
minutes. The flux densities alone take 15 GiB of either card.

The visibilities agree with the sum of the same model simulated as 8 source
lists of 80,000 sources to single-precision rounding (a largest difference of
1.3 × 10⁻⁶ of the largest visibility). Splitting the list is not faster per
timestep; it only repeats the reading of the list.
