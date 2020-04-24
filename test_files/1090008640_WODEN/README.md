output_band01.uvfits is a sky model simulated by
[WODEN](https://github.com/JLBLine/WODEN) commit e3c810f, corresponding to the
source list file "srclist_3x3_grid.txt" using obsid 1090008640. The settings of
note when WODEN was ran are:

    --time_res=4
    --num_time_steps=1
    --ra0=0
    --dec0=-27

These data have the MWA FEE beam applied, but a slightly different
implementation to hyperbeam. It is for this reason I suspect hyperdrive doesn't
calibrate instantly.
