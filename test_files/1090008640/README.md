1090008640 is an EoR-0, high-band (centred ~184 MHz), zenith-pointed observation
with a relatively inactive ionosphere. It has no flagged tiles and all
non-edge-channels calibrate well.

The single gpubox file supplied here has been altered to contain only one
timestep (the HDU with (Unix) TIME 1405973441, corresponding to GPS time
1090008658). The measurement set and uvfits have been derived from this gpubox
file with Birli v0.5.0 (default settings, which also means no PFB gains
applied). There are no flags in the data as it comes out of Birli; for testing
purposes, a flag was added to baseline 12 (tile 0 and tile 12) on channel 2.

primes_01.mwaf has been altered contain only one timestep's worth of flags, and
those flags indicate prime numbers (i.e. index 1 => 0, 2 => 1, 3 => 1, 4 => 0).
The timestep corresponds to the only timestep available in the lone gpubox file.
