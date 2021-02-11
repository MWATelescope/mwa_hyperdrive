// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Code to handle calibration.
 */

pub mod args;
pub mod params;
pub mod veto;

use ndarray::Array1;

use crate::constants::*;
use mwa_hyperdrive_core::Jones;
use params::CalibrateParams;

pub fn calibrate(mut params: CalibrateParams) -> Result<(), anyhow::Error> {
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

    // mwa_rts.c, line 1719
    let gains = init_calibrator_gains(&params);

    todo!();
}

/// Get derivatives of the gain Jones matrices for each tile, each frequency,
/// each calibrator. Assumes that the latitude is the MWA site latitude.
///
/// The RTS calls this function "SetCalibratorMatrices".
pub(crate) fn init_calibrator_gains(params: &CalibrateParams) -> Array1<Jones> {
    // SetCalibratorMatrices in CalibratorMeasurements.c, line 1203

    // Use `dt` to determine the forward-difference derivative.
    let dt = params.time_res * DS2R * SOLAR2SIDEREAL;
    let lst = params.get_lst();

    // Get azimuth and zenith angle calibrator positions at the current LST, as well as LST + `dt`.
    let (az, za) = params.source_list.get_azza(lst);
    // Get the "forward" coords by altering the LST.
    let (az_forward, za_forward) = params.source_list.get_azza(lst + dt);

    // Iterate over all tiles...
    params
        .beam
        .calc_jones_array(az_rad, za_rad, freq_hz, delays, amps, true);

    // Get the "derivatives" of the Jones matrices. Because we're doing a
    // forward-difference derivative, the derivatives are most accurate at
    // (`lst` + `dt`) / 2, which is what we want.

    Array1::from(vec![Jones::identity()])
}
