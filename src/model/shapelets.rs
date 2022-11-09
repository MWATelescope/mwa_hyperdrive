// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Data and variables for computations with shapelets.

use std::io::{BufReader, Read};

// These values correspond to what were used when
// "shapelet_basis_functions.bin.gz" was generated.
pub(crate) const SBF_C: f64 = 5000.0;
pub(crate) const SBF_L: usize = 10001;
pub(crate) const SBF_N: usize = 101;
pub(crate) const SBF_DX: f64 = 0.01;

// Read from "shapelet_basis_functions.bin.gz" in hyperdrive's project src
// directory. Courtesy Jack Line.
lazy_static::lazy_static! {
    pub(crate) static ref SHAPELET_BASIS_VALUES: &'static [f64] = {
        // Read the compressed binary file from inside the hyperdrive binary.
        // This is loaded into the hyperdrive binary at compile time.
        let bytes = include_bytes!("shapelet_basis_values.bin.gz");
        // Create a reader to interface with the compressed bytes.
        let gz_reader = BufReader::new(bytes.as_ref());
        let mut gz_reader = flate2::read::GzDecoder::new(gz_reader);
        // Allocate a vector for the bytes and the values. There are too many to
        // sit on the stack!
        let mut bytes_array: Vec<u8> = Vec::with_capacity(SBF_L * SBF_N * 8) ;
        // Read in the bytes...
        gz_reader.read_to_end(&mut bytes_array).unwrap();
        // ... ensure that the right number of bytes are there...
        assert_eq!(bytes_array.len(), SBF_L * SBF_N * 8);
        // ... and then leak the memory, so it looks like a global immutable
        // array, and cast the bytes to floats.
        bytemuck::cast_slice(bytes_array.leak())
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn shapelet_basis_values() {
        // Not all SHAPELET_BASIS_VALUES are equal to zero (or smaller than
        // epsilon).
        assert!(!SHAPELET_BASIS_VALUES.iter().all(|&v| v < f64::EPSILON));

        // The middle value has a value of 1.0.
        assert_abs_diff_eq!(SHAPELET_BASIS_VALUES[SBF_L / 2], 1.0, epsilon = 1e-10);

        // The middle + 1 value has a value of 0.99995.
        assert_abs_diff_eq!(
            SHAPELET_BASIS_VALUES[SBF_L / 2 + 1],
            0.99995,
            epsilon = 1e-8
        );
    }
}
