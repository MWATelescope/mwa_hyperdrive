// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Code to predict visibilities, given a sky model and array.
 */

use std::{env, f64::consts::TAU};

use log::debug;
use ndarray::prelude::*;

use super::CalibrateError;
use crate::math::cexp;
use mwa_hyperdrive_core::{
    c64, constants::VEL_C, mwa_hyperbeam::fee::FEEBeamError, AzEl, InstrumentalStokes, Jones,
    RADec, SourceList, LMN, UVW,
};

/// For a range of times, frequencies and baselines, predict visibility values
/// for each sky-model source component.
///
/// `calc_jones`: A function that takes the horizon coordinates (i.e. `AzEl`) of
/// a source component and a frequency \[Hz\] returning a beam Jones matrix.
///
/// `lsts`: The local sidereal times that we have to predict over. This is a
/// proxy for time. Each LST is used with the pointing to make correct `AzEl`
/// coordinates for each source component.
///
/// `uvw`: The UVW coordinates of each baseline. Each row corresponds to a
/// unique time.
///
/// `lmn`: The LMN coordinates of all sky-model source components.
pub(crate) fn predict(
    calc_jones: fn(AzEl, f64) -> Result<Jones, FEEBeamError>,
    source_list: &SourceList,
    num_components: usize,
    pointing: &RADec,
    lsts: &[f64],
    freqs: &[f64],
    uvws: ArrayView2<UVW>,
    lmns: &[LMN],
) -> Result<Array2<[c64; 4]>, CalibrateError> {
    // Every element of `flux_densities` is a 4-element array of complex-valued
    // instrumental Stokes flux densities.
    let mut flux_densities = Array3::from_elem(
        (lsts.len(), freqs.len(), num_components),
        [c64::default(); 4],
    );
    // Every element of `model` is a 4-element array of complex visibilities.
    let mut model = Array2::from_elem((lsts.len(), freqs.len()), [c64::default(); 4]);

    debug!("started predicting flux densities");
    for (mut time_axis, lst) in flux_densities.outer_iter_mut().zip(lsts.iter()) {
        for (mut freq_axis, freq) in time_axis.outer_iter_mut().zip(freqs.iter()) {
            let mut beam_corrected_fds: Vec<[c64; 4]> = Vec::with_capacity(num_components);
            for (_, src) in source_list.iter() {
                for comp in &src.components {
                    let is: InstrumentalStokes = comp.flux_type.estimate_at_freq(*freq)?.into();
                    let azel = comp.radec.to_hadec(*lst).to_azel_mwa();
                    // `jones` is the beam-response Jones matrix.
                    let jones = calc_jones(azel, *freq)?;
                    // TODO: Use a Jones matrix from another tile!
                    let jones_h = jones.h();
                    beam_corrected_fds.push(Jones::outer_mul(jones, is.to_array(), jones_h));
                }
            }
            freq_axis.assign(&Array1::from(beam_corrected_fds));
        }
    }
    debug!("finished predicting flux densities, starting predicting model");

    for (((mut model_time_axis, fd_time_axis), lst), uvw) in model
        .outer_iter_mut()
        .zip(flux_densities.outer_iter())
        .zip(lsts.iter())
        .zip(uvws.iter())
    {
        let pointing = pointing.to_hadec(*lst);
        for ((mut model_freq_axis, fd_freq_axis), freq) in model_time_axis
            .outer_iter_mut()
            .zip(fd_time_axis.outer_iter())
            .zip(freqs.iter())
        {
            let uvw_lambda = *uvw / (VEL_C / freq);

            // let mut model_comp = c64::new(0.0, 0.0);
            for mut model_comp in model_freq_axis.iter_mut() {
                for (component_fd, lmn) in fd_freq_axis.outer_iter().zip(lmns.iter()) {
                    // TODO: When non-point-source components are being used,
                    // calculate envelope correctly.
                    let envelope = 1.0;
                    let arg = TAU * (uvw.u * lmn.l + uvw.v * lmn.m + uvw.w * (lmn.n - 1.0));
                    let phase = cexp(arg) * envelope;
                    // model_comp += component_fd
                    model_comp
                        .iter_mut()
                        .zip(component_fd.iter())
                        .map(|(&mut model, &comp_fd)| {
                            model += comp_fd * phase;
                        })
                }
            }
        }
    }
    debug!("finished predicting model");

    Ok(model)
}
