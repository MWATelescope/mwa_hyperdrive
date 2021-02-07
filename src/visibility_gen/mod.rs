// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

mod cuda;

use std::fs::File;
use std::io::BufWriter;
use std::io::Write;

use byteorder::{ByteOrder, LittleEndian};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;

use crate::*;
use mwa_hyperdrive_core::{constants::*, Source};

/// Parameters for visibility generation. Members of this struct might be
/// different from a `Context` struct (e.g. such as `time_resolution`), which
/// allows control on how visibilities are generated.
#[derive(Debug)]
pub struct TimeFreqParams {
    /// The number of time steps to iterate over.
    pub n_time_steps: usize,
    /// The amount of time per step [seconds].
    pub time_resolution: f64,
    /// The coarse-band channels to use.
    pub freq_bands: Vec<u8>,
    /// The number of fine channels per coarse band.
    pub n_fine_channels: u64,
    /// The resolution of a single fine-frequency channel [Hz].
    pub fine_channel_width: f64,
}

/// Generate visibilities for a sky model by iterating over each time step
/// defined in `params`, and write the results to a binary file per
/// coarse-frequency band, named "./hyperdrive_bandxx.bin" (this is the format
/// of uvfits files, and matches the output of WODEN). If specified, data is
/// also written to a text file named "./hyperdrive_bandxx.txt".
///
/// Currently only works with a single `Source` made from point-source
/// components.
pub fn vis_gen(
    context: &crate::Context,
    src: &Source,
    params: &TimeFreqParams,
    mut pc: PointingCentre,
    cuda: bool,
    text_file: bool,
) -> Result<(), anyhow::Error> {
    // Because WODEN writes all `u` coordinates before writing all `v`, then
    // `w`, etc., we need to store all data before it can be written out. An
    // alternative to this approach is to write out intermediate "per time, per
    // freq. band" files, but hopefully when we write directly to uvfits files,
    // this approach can go away.
    let num_coords = params.n_time_steps * context.xyz.len();
    let mut u = Vec::with_capacity(num_coords);
    let mut v = Vec::with_capacity(num_coords);
    let mut w = Vec::with_capacity(num_coords);
    let total_num_visibilities =
        num_coords as usize * params.freq_bands.len() * params.n_fine_channels as usize;
    let mut real = Vec::with_capacity(total_num_visibilities);
    let mut imag = Vec::with_capacity(total_num_visibilities);

    // Get the interpolated/extrapolated flux densities for each frequency. The
    // flux densities of the source components do not change with time, and we
    // know all of the frequencies in advance. Calculate them once here.
    let mut flux_densities = Vec::with_capacity(
        src.components.len() * params.freq_bands.len() * params.n_fine_channels as usize,
    );
    for band in &params.freq_bands {
        // Have to subtract 1, as we index MWA coarse bands from 1.
        let base_freq = (context.base_freq
            + (*band - 1) as u32 * context.mwalib.coarse_channel_width_hz)
            as f64;
        for fine_channel in 0..params.n_fine_channels {
            let freq = base_freq + params.fine_channel_width * fine_channel as f64;
            let mut fds = src
                .get_flux_estimates(freq)?
                .into_iter()
                .map(|fd| fd.i as _)
                .collect();
            flux_densities.append(&mut fds);
        }
    }
    // Ensure that no memory is dangling.
    flux_densities.shrink_to_fit();

    // Make a progress bar for the time steps.
    let pb = ProgressBar::new(params.n_time_steps as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{msg}{percent}% [{bar:40.cyan/blue}] {pos}/{len} [{elapsed_precise}<{eta_precise}]",
            )
            .progress_chars("#>-"),
    );

    // Iterate over time. This allows minimal re-calculation of the pointing
    // centre, and is more efficient at generating visibilities.
    for time_step in 0..params.n_time_steps {
        // Adjust the pointing centre by half a time step.
        pc.update(
            context.base_lst
                + (time_step as f64 + 0.5) * params.time_resolution * SOLAR2SIDEREAL * DS2R,
        );

        // Get the (l,m,n) coordinates for each source component.
        let lmn = src.get_lmn(&pc);

        // Get the UVW baselines with the new PC.
        let uvw_metres = UVW::get_baselines(&context.xyz, &pc);

        // For each fine channel, scale all of the UVW coordinates by
        // wavelength, and store the result in `uvw`.
        let mut uvw = Vec::with_capacity(
            params.freq_bands.len() * params.n_fine_channels as usize * context.xyz.len(),
        );
        for band in &params.freq_bands {
            // Have to subtract 1, as we index MWA coarse bands from 1.
            let base_freq = (context.base_freq
                + (*band - 1) as u32 * context.mwalib.coarse_channel_width_hz)
                as f64;
            for fine_channel in 0..params.n_fine_channels {
                let freq = base_freq + params.fine_channel_width * fine_channel as f64;
                let wavelength = VEL_C / freq;
                let mut uvw_scaled = uvw_metres
                    .par_iter()
                    .map(|v| *v / wavelength)
                    .collect::<Vec<_>>();
                uvw.append(&mut uvw_scaled);
            }
        }

        pb.set_position(time_step as u64);
        let (mut real_t, mut imag_t) = if cuda {
            if time_step == 0 {
                pb.println(format!(
                    r#"Running CUDA with:
    {uvw} UVW baselines ({bl} baselines * {fc} fine channels * {cb} coarse bands)
    {comps} source components"#,
                    uvw = params.freq_bands.len() as u64
                        * params.n_fine_channels
                        * context.xyz.len() as u64,
                    bl = context.xyz.len(),
                    fc = params.n_fine_channels,
                    cb = params.freq_bands.len(),
                    comps = src.components.len()
                ));
            }
            cuda::cuda_vis_gen(&context, &src, &params, &flux_densities, lmn, uvw)
        } else {
            cpu_vis_gen(&context, &src, &params, &flux_densities, lmn, uvw)
        };

        let (mut u_t, mut v_t, mut w_t) = UVW::decompose(uvw_metres);
        u.append(&mut u_t);
        v.append(&mut v_t);
        w.append(&mut w_t);
        real.append(&mut real_t);
        imag.append(&mut imag_t);
    }
    pb.finish();

    // Now that we have all of the data in memory, write it out per freq. band.
    for (band_num, band) in params.freq_bands.iter().enumerate() {
        let file = File::create(format!("hyperdrive_band{:02}.bin", band))?;
        let mut buf = BufWriter::new(file);

        // Write the coordinates.
        write_binary_uvw(
            params.n_time_steps,
            params.n_fine_channels,
            context.xyz.len() as u64,
            &mut buf,
            &u,
        )?;
        write_binary_uvw(
            params.n_time_steps,
            params.n_fine_channels,
            context.xyz.len() as u64,
            &mut buf,
            &v,
        )?;
        write_binary_uvw(
            params.n_time_steps,
            params.n_fine_channels,
            context.xyz.len() as u64,
            &mut buf,
            &w,
        )?;

        // Write the visibilities.
        write_binary_real_imag(
            params.n_time_steps,
            params.freq_bands.len() as u64,
            params.n_fine_channels,
            context.xyz.len() as u64,
            band_num as u64,
            &mut buf,
            &real,
        )?;
        write_binary_real_imag(
            params.n_time_steps,
            params.freq_bands.len() as u64,
            params.n_fine_channels,
            context.xyz.len() as u64,
            band_num as u64,
            &mut buf,
            &imag,
        )?;

        // Write a text file.
        if text_file {
            let file = File::create(format!("hyperdrive_band{:02}.txt", band))?;
            let mut buf = BufWriter::new(file);
            let unit = params.n_fine_channels as usize * context.xyz.len();
            for time_step in 0..params.n_time_steps as usize {
                let coord_offset = time_step * context.xyz.len();
                let vis_offset = time_step * params.freq_bands.len() * unit + band_num * unit;
                for i in 0..unit {
                    writeln!(
                        buf,
                        "{:.7} {:.7} {:.7} {:.7} {:.7}",
                        u[coord_offset + (i % context.xyz.len())],
                        v[coord_offset + (i % context.xyz.len())],
                        w[coord_offset + (i % context.xyz.len())],
                        real[vis_offset + i],
                        imag[vis_offset + i]
                    )?;
                }
            }
        }
    }

    Ok(())
}

/// Write a `u`, `v` or `w` coordinate to a file buffer the correct number of
/// times to match the WODEN format. `coord` should have a length of
/// `n_time_steps` * `n_baselines`.
fn write_binary_uvw(
    n_time_steps: usize,
    n_fine_channels: u64,
    n_baselines: u64,
    buf: &mut BufWriter<File>,
    coord: &[f32],
) -> Result<(), std::io::Error> {
    // Allocate a space for the bytes. 4 bytes per f32.
    let mut bytes = vec![0; 4 * n_baselines as usize];
    for time_step in 0..n_time_steps {
        let i_start = time_step * n_baselines as usize;
        let i_end = (time_step + 1) * n_baselines as usize;
        // Read in the data as bytes.
        LittleEndian::write_f32_into(&coord[i_start..i_end], &mut bytes);
        // For each fine channel, write out the bytes.
        for _ in 0..n_fine_channels {
            buf.write_all(&bytes)?;
        }
    }

    Ok(())
}

/// Write real or imaginary visibilities to a file buffer the correct sequence
/// to match the WODEN format. `vis` should have a length of `n_time_steps` *
/// `n_fine_channels` * `n_baselines` * the number of freq. bands.
fn write_binary_real_imag(
    n_time_steps: usize,
    n_freq_bands: u64,
    n_fine_channels: u64,
    n_baselines: u64,
    band_num: u64,
    buf: &mut BufWriter<File>,
    vis: &[f32],
) -> Result<(), std::io::Error> {
    let unit = (n_fine_channels * n_baselines) as usize;
    // Allocate a space for the bytes. 4 bytes per f32.
    let mut bytes = vec![0; 4 * (n_fine_channels * n_baselines) as usize];
    for time_step in 0..n_time_steps {
        let offset = time_step * n_freq_bands as usize * unit;
        let i_start = offset + band_num as usize * unit;
        let i_end = offset + (band_num + 1) as usize * unit;
        // Read in the data as bytes.
        LittleEndian::write_f32_into(&vis[i_start..i_end], &mut bytes);
        buf.write_all(&bytes)?;
    }

    Ok(())
}

/// Calculate the visibility equation for every (u,v,w) and (l,m,n)
/// combination. This is a CPU implementation of `cuda_vis_gen`, and does
/// calculations in parallel. Currently only works with a single `Source` made
/// from point-source components.
///
/// On my Ryzen 9 3900X (12 hyper-threaded cores), this function is
/// approximately 50x slower than `cuda_vis_gen` (using an NVIDIA GeForce RTX
/// 2070).
fn cpu_vis_gen(
    context: &Context,
    src: &Source,
    params: &TimeFreqParams,
    flux_densities: &[f32],
    lmn: Vec<LMN>,
    uvw: Vec<UVW>,
) -> (Vec<f32>, Vec<f32>) {
    // Perform the visibility equation over each UVW baseline and LMN triple.
    let n_baselines = context.xyz.len();
    let n_components = src.components.len();
    let n_visibilities = params.freq_bands.len() * params.n_fine_channels as usize * n_baselines;
    let mut real = Vec::with_capacity(n_visibilities);
    let mut imag = Vec::with_capacity(n_visibilities);

    uvw.par_iter()
        .enumerate()
        // `bl` for baseline
        .map(|(i_vis, bl)| {
            lmn.iter()
                .zip(
                    // There's a flux density for each LMN triple, but we
                    // need to get the right one.
                    flux_densities
                        .iter()
                        .skip(i_vis / n_baselines * n_components),
                )
                .fold(
                    (0.0, 0.0),
                    // `dc` for direction cosine, `fd` for flux density.
                    |acc, (dc, &fd)| {
                        let arg = TAU * (bl.u * dc.l + bl.v * dc.m + bl.w * (dc.n - 1.0));
                        let c = cexp(arg);
                        (acc.0 + c.re as f32 * fd, acc.1 + c.im as f32 * fd)
                    },
                )
        })
        .unzip_into_vecs(&mut real, &mut imag);
    (real, imag)
}
