// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Data and variables for computations with shapelets.
 */

use std::io::Read;

use byteorder::{ByteOrder, LittleEndian};

use mwa_hyperdrive_core::c64;

pub(crate) const I_POWER_TABLE: [c64; 4] = [
    c64::new(1.0, 0.0),
    c64::new(0.0, 1.0),
    c64::new(-1.0, 0.0),
    c64::new(0.0, -1.0),
];

pub(crate) const SBF_C: f64 = 5000.0;
pub(crate) const SBF_L: usize = 10001;
pub(crate) const _SBF_N: usize = 101;
pub(crate) const SBF_DX: f64 = 0.01;

// Read from "shapelet_basis_functions.bin.gz" in hyperdrive's project src
// directory. Courtesy Jack Line.
lazy_static::lazy_static! {
    // TODO: Is this leaking?
    pub(crate) static ref SHAPELET_BASIS_VALUES: Vec<f64> = {
        // Read the compressed binary file from inside the hyperdrive binary.
        // This is loaded into the hyperdrive binary at compile time.
        let bytes = include_bytes!("shapelet_basis_values.bin.gz");
        // Create a reader to interface with the compressed bytes.
        let mut gz_reader = flate2::read::GzDecoder::new(bytes.as_ref());
        // Allocate a vector for the values. There are too many to sit on the
        // stack!
        let mut array = vec![0; 1010101 * 8];
        // Read the bytes to the array.
        gz_reader.read_to_end(&mut array).unwrap();
        // Re-interpret the bytes as floats.
        unsafe {
            std::mem::transmute::<Vec<u8>, Vec<f64>>(array)
        }
    };
}
