// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

mod cuda;

use std::fs::File;
use std::io::BufWriter;
use std::io::Write;

use byteorder::{ByteOrder, LittleEndian};
use indicatif::{ProgressBar, ProgressStyle};

use crate::constants::*;
use crate::coord::*;
use crate::sourcelist::types::Source;

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
    pub n_fine_channels: usize,
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
    mut pc: PC,
    cuda: bool,
    text_file: bool,
) -> Result<(), std::io::Error> {
    // Because WODEN writes all `u` coordinates before writing all `v`, then
    // `w`, etc., we need to store all data before it can be written out. An
    // alternative to this approach is to write out intermediate "per time, per
    // freq. band" files, but hopefully when we write directly to uvfits files,
    // this approach can go away.
    let num_coords = params.n_time_steps * context.n_baselines;
    let mut u = Vec::with_capacity(num_coords);
    let mut v = Vec::with_capacity(num_coords);
    let mut w = Vec::with_capacity(num_coords);
    let total_num_visibilities = num_coords * params.freq_bands.len() * params.n_fine_channels;
    let mut real = Vec::with_capacity(total_num_visibilities);
    let mut imag = Vec::with_capacity(total_num_visibilities);

    // Make a progress bar for the time steps.
    let pb = ProgressBar::new(params.n_time_steps as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{msg}{percent}% [{bar:60.cyan/blue}] {pos}/{len} [{elapsed_precise}<{eta_precise}]",
            )
            .progress_chars("#>-"),
    );

    // Iterate over time. This allows minimal re-calculation of the pointing
    // centre, and is more efficient at generating visibilities.
    for time_step in 0..params.n_time_steps {
        // Adjust the pointing centre by half a time step.
        pc.update(
            context.base_lst
                + (time_step as f64 + 0.5) * params.time_resolution * *SOLAR2SIDEREAL * *DS2R,
        );
        // Get the UVW baselines with the new PC.
        let uvw_metres = UVW::get_baselines(&context.xyz, &pc);

        pb.set_position(time_step as u64);
        let (mut real_t, mut imag_t) = if cuda {
            if time_step == 0 {
                pb.println(format!(
                    r#"Running CUDA with:
    {uvw} UVW baselines ({bl} baselines * {fc} fine channels * {cb} coarse bands)
    {comps} source components"#,
                    uvw = params.freq_bands.len() * params.n_fine_channels * context.n_baselines,
                    bl = context.n_baselines,
                    fc = params.n_fine_channels,
                    cb = params.freq_bands.len(),
                    comps = src.components.len()
                ));
            }
            cuda::cuda_vis_gen(&context, &src, &params, &pc, &uvw_metres)
        } else {
            // TODO: Implement CPU visibility generation.
            unimplemented!("CPU-only visibility generation");
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
            context.n_baselines,
            &mut buf,
            &u,
        )?;
        write_binary_uvw(
            params.n_time_steps,
            params.n_fine_channels,
            context.n_baselines,
            &mut buf,
            &v,
        )?;
        write_binary_uvw(
            params.n_time_steps,
            params.n_fine_channels,
            context.n_baselines,
            &mut buf,
            &w,
        )?;

        // Write the visibilities.
        write_binary_real_imag(
            params.n_time_steps,
            params.freq_bands.len(),
            params.n_fine_channels,
            context.n_baselines,
            band_num,
            &mut buf,
            &real,
        )?;
        write_binary_real_imag(
            params.n_time_steps,
            params.freq_bands.len(),
            params.n_fine_channels,
            context.n_baselines,
            band_num,
            &mut buf,
            &imag,
        )?;

        // Write a text file.
        if text_file {
            let file = File::create(format!("hyperdrive_band{:02}.txt", band))?;
            let mut buf = BufWriter::new(file);
            let unit = params.n_fine_channels * context.n_baselines;
            for time_step in 0..params.n_time_steps {
                let coord_offset = time_step * context.n_baselines;
                let vis_offset = time_step * params.freq_bands.len() * unit + band_num * unit;
                for i in 0..unit {
                    writeln!(
                        buf,
                        "{:.7} {:.7} {:.7} {:.7} {:.7}",
                        u[coord_offset + (i % context.n_baselines)],
                        v[coord_offset + (i % context.n_baselines)],
                        w[coord_offset + (i % context.n_baselines)],
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
    n_fine_channels: usize,
    n_baselines: usize,
    buf: &mut BufWriter<File>,
    coord: &[f32],
) -> Result<(), std::io::Error> {
    // Allocate a space for the bytes. 4 bytes per f32.
    let mut bytes = vec![0; 4 * n_baselines];
    for time_step in 0..n_time_steps {
        let i_start = time_step * n_baselines;
        let i_end = (time_step + 1) * n_baselines;
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
    n_freq_bands: usize,
    n_fine_channels: usize,
    n_baselines: usize,
    band_num: usize,
    buf: &mut BufWriter<File>,
    vis: &[f32],
) -> Result<(), std::io::Error> {
    let unit = n_fine_channels * n_baselines;
    // Allocate a space for the bytes. 4 bytes per f32.
    let mut bytes = vec![0; 4 * n_fine_channels * n_baselines];
    for time_step in 0..n_time_steps {
        let offset = time_step * n_freq_bands * unit;
        let i_start = offset + band_num * unit;
        let i_end = offset + (band_num + 1) * unit;
        // Read in the data as bytes.
        LittleEndian::write_f32_into(&vis[i_start..i_end], &mut bytes);
        buf.write_all(&bytes)?;
    }

    Ok(())
}
