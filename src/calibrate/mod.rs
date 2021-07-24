// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to handle calibration.

pub mod args;
mod di;
mod error;
pub(crate) mod params;
pub mod solutions;

pub use error::CalibrateError;
use params::CalibrateParams;

pub fn calibrate(mut params: CalibrateParams) -> Result<(), CalibrateError> {
    // TODO: Remove this.
    // Remove the first x most significant sources for debugging.
    // for rs in params.ranked_sources.iter().take(30) {
    //     params.source_list.remove(&rs.name);
    // }
    // dbg!(params.source_list.len());

    // TODO: Allow the user to specify whether or not we should do DI
    // calibration.
    di::di_cal(&params)?;

    // How much time is available?
    //
    // Assume we're doing a DI step. How much data gets averaged together? Does
    // this depend on baseline length?

    // Work with a single "scan" for now.
    // Assume we start at "time 0".

    // Rotate all the sources.

    // So all of the sources have their (RA, Dec) coordinates read in.
    //     params.source_list.par_iter().map(|(src_name, src)| {
    //         let rotated_comps: Vec<_> = src.components.iter_mut().map(|comp| {
    //             let hd = comp.radec.to_hadec(params.get_lst());
    //             (hd, comp.comp_type, comp.flux_type)
    //         }).collect();
    //         Source
    // rotated_comps
    //     }).collect()
    // Line 1735 of the RTS

    // If we're not starting at "time 0", the RTS "resets the initial Jones
    // matrices"; for each tile, get the beam-response Jones matrix toward each
    // primary calibrator source (there are usually 5 used by the RTS) at the
    // centre frequency of the entire observation.

    // TODO: RTS SetSourceSpectra. Needed?

    // TODO: "start processing at"

    // TODO: RTS's rts_options.do_MWA_rx_corrections. PFB gains.

    // TODO: Load existing calibration solutions. This should be higher up; fail fast.

    // TODO: Set "AlignmentFluxDensity" and "NewDIMatrices" (line 1988
    // mwa_rts.c). Is this just the estimated Stokes I FD of all components? Do
    // I need to track all estimated FDs?

    // TODO: PrecessZenithtoJ2000
    // Jesus Christ. Do I need this? Surely I can just multiply the visibilities
    // by e^{2pi i w}?

    // CalcMinIntegTime
    // Might be useful. I think it actually calculates the max integration time,
    // not the min - classic.

    // TODO: Load data! Birli's job.
    // import_uvfits_single is where the RTS reads gpubox files.
    // _importuvfits_set_uvdata_visgroup is where the actual fits data gets read.
    // VI_FillVisibilityData accumulates the visibilities (?)

    // The XYZ coordinates of all of the baselines does not change with time for
    // the observation.
    // let xyz = XYZ::get_baselines_mwalib(&params.context.metafits_context);

    // for t in params.get_timesteps() {
    //     let lst = params.get_lst();
    //     let uvw = UVW::get_baselines(&xyz, params.get_pointing());
    // }

    Ok(())
}
