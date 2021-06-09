// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Generate visibilities using CUDA.

use super::context::Context;
use super::*;

use mwa_hyperdrive_cuda::{vis_gen, Context_c, LMN_c, Source_c, UVW_c, Vis_c};

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
    lmn: &[LMN],
    uvw: Vec<UVW>,
) -> (Vec<f32>, Vec<f32>) {
    let n_freq_bands = params.freq_bands.len() as u64;
    let n_channels = n_freq_bands * params.n_fine_channels;
    let n_visibilities = n_freq_bands * params.n_fine_channels * context.xyz.len() as u64;

    // Convert `uvw` to be C compatible.
    let uvw_c: Vec<_> = uvw
        .into_iter()
        .map(|uvw| UVW_c {
            u: uvw.u as _,
            v: uvw.v as _,
            w: uvw.w as _,
        })
        .collect();

    // Convert `lmn` to be C compatible.
    let lmn_c: Vec<_> = lmn
        .iter()
        .map(|lmn| LMN_c {
            l: lmn.l as _,
            m: lmn.m as _,
            n: lmn.n as _,
        })
        .collect();

    let src_c = Source_c {
        n_points: src.components.len() as _,
        point_lmn: lmn_c.as_ptr(),
        point_fd: flux_densities.as_ptr(),
        n_channels: n_channels as _,
    };

    // Create a `visibilities_s` struct to pass into C.
    let mut real = vec![0.0; n_visibilities as usize];
    let mut imag = vec![0.0; n_visibilities as usize];
    real.shrink_to_fit();
    imag.shrink_to_fit();
    let vis_c = Vis_c {
        n_vis: n_visibilities as _,
        real: real.as_mut_ptr(),
        imag: imag.as_mut_ptr(),
    };

    // Box the C structs so they can be used by C. This memory will be leaked by
    // Rust if we don't free it before the end of this function's scope.
    let src_c = Box::into_raw(Box::new(src_c));
    let vis_c = Box::into_raw(Box::new(vis_c));

    // Call CUDA.
    unsafe {
        vis_gen(
            uvw_c.as_ptr(),
            src_c,
            vis_c,
            n_channels as _,
            context.xyz.len() as _,
        );
        // Deallocate the leaked memory.
        Box::from_raw(src_c);
        Box::from_raw(vis_c);
    }

    (real, imag)
}
