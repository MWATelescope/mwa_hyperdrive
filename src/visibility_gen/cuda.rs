// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Generate visibilities using CUDA.
 */

use rayon::prelude::*;

use super::*;
use crate::constants::*;
use crate::context::Context;
use crate::foreign::*;
use crate::sourcelist::{estimate::calc_flux_ratio, *};

/// For all coarse-band frequencies and their fine channels, generate
/// visibilities for a sky model with CUDA, using the C function `vis_gen`.
///
/// Currently only works with a single `Source` made from point-source
/// components.
///
/// Note that results will be slightly different between runs of this function
/// with the same settings; floating-point operations are not associative, and
/// CUDA will add floats in an effectively random order, resulting in
/// non-determinism.
#[allow(clippy::many_single_char_names)]
pub fn cuda_vis_gen(
    context: &Context,
    src: &Source,
    params: &TimeFreqParams,
    pc: &PC,
    uvw_metres: &[UVW],
) -> (Vec<f32>, Vec<f32>) {
    // Generate UVW baselines for each fine-frequency channel in each coarse
    // freq. band and calculate the expected flux densities at each frequency.
    let n_visibilities = params.freq_bands.len() * params.n_fine_channels * context.n_baselines;
    // Pre-allocate the arrays to be passed to C.
    let mut uvw = Vec::with_capacity(n_visibilities);
    let mut flux_densities = Vec::with_capacity(params.n_fine_channels * params.freq_bands.len());

    // For each fine channel, scale all of the UVW coordinates and calculate
    // the expected flux density.
    for band in &params.freq_bands {
        let mut uvw_scaled: Vec<Vec<UVW>> = (0..params.n_fine_channels)
            .into_par_iter()
            .map(|fine_channel| {
                // Calculate the wavelength for this fine channel, and scale
                // the UVW coords with it.
                let freq = (context.base_freq + *band as usize * context.coarse_channel_width)
                    as f64
                    + params.fine_channel_width * fine_channel as f64;

                let wavelength = *VEL_C / freq;
                uvw_metres.iter().map(|v| *v / wavelength).collect()
            })
            .collect();

        let mut fd_extrap: Vec<Vec<FluxDensity>> = (0..params.n_fine_channels)
            .into_par_iter()
            .map(|fine_channel| {
                let freq = (context.base_freq + *band as usize * context.coarse_channel_width)
                    as f64
                    + params.fine_channel_width * fine_channel as f64;

                src.components
                    .iter()
                    .map(|comp| {
                        comp.flux_densities
                            .iter()
                            .map(|fd| *fd * calc_flux_ratio(freq, fd.freq, *DEFAULT_SPEC_INDEX))
                            .collect::<Vec<FluxDensity>>()
                    })
                    .flatten()
                    .collect()
            })
            .collect();

        for (mut bl, mut fd) in uvw_scaled.drain(..).zip(fd_extrap.drain(..)) {
            uvw.append(&mut bl);
            for f in fd.drain(..) {
                flux_densities.push(f.i as f32);
                flux_densities.push(f.q as f32);
                flux_densities.push(f.u as f32);
                flux_densities.push(f.v as f32);
            }
        }
    }

    // Convert `uvw` to be C compatible.
    let (u, v, w) = UVW::decompose(uvw);
    let uvw_s = Box::into_raw(Box::new(UVW_s {
        n_baselines: context.n_baselines as u32,
        n_elem: n_visibilities as u32,
        u: u.as_ptr(),
        v: v.as_ptr(),
        w: w.as_ptr(),
    }));

    // Convert `src` to be C compatible.
    let (l, m, n) = LMN::decompose(src.get_lmn(&pc));
    let src_s = Box::into_raw(Box::new(Source_s {
        n_points: src.components.len() as u32,
        point_l: l.as_ptr(),
        point_m: m.as_ptr(),
        point_n: n.as_ptr(),
        point_fd: flux_densities.as_ptr(),
    }));

    // Create a `visibilities_s` struct to pass into C.
    let mut real = vec![0.0; n_visibilities];
    let mut imag = vec![0.0; n_visibilities];
    real.shrink_to_fit();
    imag.shrink_to_fit();
    let vis_s = Box::into_raw(Box::new(Visibilities_s {
        n_visibilities: n_visibilities as u32,
        real: real.as_mut_ptr(),
        imag: imag.as_mut_ptr(),
    }));

    // Call CUDA.
    unsafe {
        crate::foreign::vis_gen(uvw_s, src_s, vis_s);
        // Deallocate.
        Box::from_raw(uvw_s);
        Box::from_raw(src_s);
        Box::from_raw(vis_s);
    }

    (real, imag)
}
