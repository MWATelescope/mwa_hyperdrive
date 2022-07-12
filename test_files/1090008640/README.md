1090008640 is an EoR-0, high-band (centred ~184 MHz), zenith-pointed observation
with a relatively inactive ionosphere. It has no flagged tiles and all
non-edge-channels calibrate well.

The single gpubox file supplied here has been altered to contain only one
timestep (the HDU with (Unix) TIME 1405973441, corresponding to GPS time
1090008658). The measurement set and uvfits have been derived from this gpubox
file with Birli v0.5.0 (default settings, which also means no PFB gains
applied). There are no flags in the data as it comes out of Birli; for testing
purposes, a flag was added to baseline 12 (tile 0 and tile 12) on channel 2.

All mwaf files here have been altered contain only one timestep's worth of
flags. The timestep corresponds to the only timestep available in the lone
gpubox file.

`1090008640_01.mwaf` is a version 2.0 mwaf file, and all flags are 0 (i.e. "not
flagged") except for baseline 12 on channel 2.

`1090008640_01_cotter{,_offset_{forwards,backwards}}.mwaf.gz` contain version
1.0 mwaf files. The offset files indicate a `GPSTIME` that is offset from the
observation timesteps, both forwards and backwards by 1s. As with the version
2.0 file, only baseline 12 channel 2 is flagged.

The flags in `primes_01.mwaf` indicate prime numbers (i.e. index 1 => 0, 2 => 1,
3 => 1, 4 => 0).
