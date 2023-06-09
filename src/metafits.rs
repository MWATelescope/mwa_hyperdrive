// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to handle reading from MWA metafits files.
//!
//! Anything here is to supplement mwalib.

use std::{collections::HashMap, ffi::CString};

use fitsio::errors::check_status as fits_check_status;
use log::debug;
use mwalib::{MetafitsContext, Pol};
use ndarray::prelude::*;

use crate::io::read::fits::{fits_open, fits_open_hdu, fits_read_cell_f64_array, FitsError};

/// The column to search for dipole gains.
const COL_NAME: &str = "DipAmps";

/// Get the delays for each tile's dipoles.
pub(crate) fn get_dipole_delays(context: &MetafitsContext) -> Array2<u32> {
    let mut all_tile_delays = Array2::zeros((context.num_ants, 16));
    for (i, mut tile_delays) in all_tile_delays.outer_iter_mut().enumerate() {
        let rf_input_1 = &context.rf_inputs[2 * i];
        let rf_input_2 = &context.rf_inputs[2 * i + 1];
        let delays: Vec<_> = rf_input_1
            .dipole_delays
            .iter()
            .zip(rf_input_2.dipole_delays.iter())
            .map(|(&d_1, &d_2)| {
                // The delays should be the same, modulo some being 32 (i.e.
                // that dipole's component is dead). This code will pick the
                // smaller delay of the two (delays are always <=32). If both
                // are 32, there's nothing else that can be done.
                d_1.min(d_2)
            })
            .collect();
        tile_delays.assign(&Array1::from(delays));
    }
    all_tile_delays
}

/// Get the gains for each tile's dipoles. If a dipole is "alive", its gain is
/// one, otherwise it is "dead" and has a gain of zero.
pub(crate) fn get_dipole_gains(context: &MetafitsContext) -> Array2<f64> {
    let mut dipole_gains = Array2::zeros((context.num_ants, 32));
    assert_eq!(context.rf_inputs[0].dipole_gains.len() * 2, 32);

    // Check for Aman's "dipole amps". Given that we've got a `MetafitsContext`,
    // we expect the associated metafits file to still exist and be accessible.
    let mut m = fits_open(&context.metafits_filename).unwrap();
    let hdu = fits_open_hdu(&mut m, 1).unwrap();
    let mut dip_amps_col_exists = false;

    unsafe {
        let col_name_cstr = CString::new(COL_NAME).unwrap().into_raw();
        let mut i_col = 0;
        let mut status = 0;
        // ffgcno = fits_get_colnum
        fitsio_sys::ffgcno(
            m.as_raw(),    /* I - FITS file pionter                       */
            1,             /* I - case sensitive string comparison? 0=no  */
            col_name_cstr, /* I - input name of column (w/wildcards)      */
            &mut i_col,    /* O - number of the named column; 1=first col */
            &mut status,   /* IO - error status                           */
        );
        match status {
            // 219 = named column not found
            219 => (),
            _ => {
                fits_check_status(status)
                    .map_err(|err| FitsError::ReadCellArray {
                        col_name: COL_NAME,
                        hdu_num: hdu.number + 1,
                        err: Box::new(err),
                    })
                    .unwrap();
                dip_amps_col_exists = true;
            }
        };
        // Don't leak the `CString`.
        drop(CString::from_raw(col_name_cstr));
    }

    // If we found the "DipAmps" column, use that for the dipole gains.
    if dip_amps_col_exists {
        debug!("Using Aman's '{COL_NAME}' dipole gains from the metafits");
        for rf_input in &context.rf_inputs {
            // These ranges account for the Y dipoles being before the X
            // dipoles, or vice versa.
            let range = if rf_input.pol == Pol::X {
                0..16
            } else {
                16..32
            };

            // `rf_input.ant` is the metafits antenna index, which is the same
            // order as `dipole_gains`. However, `rf_input.input` is the 0 to 2N
            // index in the metafits, where N is the number of tiles.
            dipole_gains
                .slice_mut(s![rf_input.ant as usize, range])
                .assign(&ArrayView1::from(
                    &fits_read_cell_f64_array::<16>(
                        &mut m,
                        &hdu,
                        COL_NAME,
                        i64::from(rf_input.input),
                    )
                    .unwrap(),
                ));
        }
    } else {
        for (i, mut dipole_gains_for_one_tile) in dipole_gains.outer_iter_mut().enumerate() {
            let rf_input_1 = &context.rf_inputs[2 * i];
            let rf_input_2 = &context.rf_inputs[2 * i + 1];
            let (rf_input_x, rf_input_y) = if rf_input_1.pol == Pol::X {
                (rf_input_1, rf_input_2)
            } else {
                (rf_input_2, rf_input_1)
            };
            dipole_gains_for_one_tile
                .slice_mut(s![..16])
                .assign(&ArrayView1::from(&rf_input_x.dipole_gains));
            dipole_gains_for_one_tile
                .slice_mut(s![16..])
                .assign(&ArrayView1::from(&rf_input_y.dipole_gains));
        }
    }
    dipole_gains
}

pub(crate) fn map_antenna_order(
    context: &MetafitsContext,
    antenna_names: &[String],
) -> Option<HashMap<usize, usize>> {
    let metafits_names: Vec<&str> = context
        .antennas
        .iter()
        .map(|a| a.tile_name.as_str())
        .collect();

    let mut hm = HashMap::with_capacity(antenna_names.len());
    // Innocent until proven guilty.
    let mut all_antennas_present = true;
    for (i_name, antenna_name) in antenna_names.iter().enumerate() {
        match metafits_names.iter().position(|&n| n == antenna_name) {
            Some(i) => hm.insert(i_name, i),
            None => {
                debug!("Could not find tile name '{antenna_name}' in the metafits file");
                all_antennas_present = false;
                break;
            }
        };
    }

    if all_antennas_present {
        Some(hm)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map_metafits_antenna_order() {
        let metafits = "test_files/1090008640/1090008640.metafits";
        let context = mwalib::MetafitsContext::new(metafits, None).unwrap();
        let antenna_names = ["Tile142".to_string(), "Tile083".to_string()];

        let map = map_antenna_order(&context, &antenna_names);
        assert!(map.is_some());
        let map = map.unwrap();
        assert_eq!(map[&0], 105);
        assert_eq!(map[&1], 58);
    }
}
