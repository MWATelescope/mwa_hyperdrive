your task is to implement di calibration in peeling.

the code in `src/params/peel/gpu.rs`, `peel_gpu` implements the algorithm described in the file `URSI_hyperdrive.tex`.

This is an approximation of peeling, that fits only the ionospheric constants ɑ, β, and g.

On the final iteration, it should solve for the DI Jones matrices towards the source.

this is almost working, but instead of returning the di solutions from peel, have peel_thread initialize the solutions, and pass them in to peel_gpu mutably. then, modify the tx_iono_consts channel to also accept those solutions, and write out the di solutions in the write thread, where it's currently writing out placeholder solutions.

NEVER USE THE GIT COMMAND

Test with

```bash
cargo run --release --features=cuda -- peel --data 1099487728.metafits hyp_1099487728_ssins_ggsm_75l_src8k_300it_8s_160kHz.uvfits --timesteps 4 --source-list GGSM_updated.fits --peel 2 --iono-sub 50 --num-sources 1000  --uvw-min 75l --uvw-max 1667l --di-per-source-dir .
```