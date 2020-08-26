// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Generate visibilities using CUDA.
 */

use super::*;
use crate::context::Context;
use crate::foreign::{Source_s, UVW_s, Visibilities_s};

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
    flux_densities: &[f32],
    lmn: Vec<LMN>,
    uvw: Vec<UVW>,
) -> (Vec<f32>, Vec<f32>) {
    let n_freq_bands = params.freq_bands.len() as u64;
    let n_channels = n_freq_bands * params.n_fine_channels;
    let n_visibilities = n_freq_bands * params.n_fine_channels * context.n_baselines;

    // Convert `uvw` to be C compatible.
    let (u, v, w) = UVW::decompose(uvw);
    let uvw_s = Box::into_raw(Box::new(UVW_s {
        n_baselines: context.n_baselines as u32,
        n_channels: n_channels as u32,
        n_vis: n_visibilities as u32,
        u: u.as_ptr(),
        v: v.as_ptr(),
        w: w.as_ptr(),
    }));

    // Convert `src` to be C compatible.
    let (l, m, n) = LMN::decompose(lmn);
    let src_s = Box::into_raw(Box::new(Source_s {
        n_points: src.components.len() as u32,
        point_l: l.as_ptr(),
        point_m: m.as_ptr(),
        point_n: n.as_ptr(),
        n_channels: n_channels as u32,
        point_fd: flux_densities.as_ptr(),
    }));

    // Create a `visibilities_s` struct to pass into C.
    let mut real = vec![0.0; n_visibilities as usize];
    let mut imag = vec![0.0; n_visibilities as usize];
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
