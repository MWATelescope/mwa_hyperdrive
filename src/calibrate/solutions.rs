// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to read and write calibration solutions.

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use byteorder::{ByteOrder, LittleEndian, WriteBytesExt};
use ndarray::ArrayView3;
use thiserror::Error;

use mwa_hyperdrive_core::{mwalib, Jones};
use mwalib::fitsio::{
    images::{ImageDescription, ImageType},
    FitsFile,
};

pub(super) fn write_solutions(
    file: &Path,
    di_jones: ArrayView3<Jones<f32>>,
    num_timeblocks: usize,
    total_num_tiles: usize,
    num_fine_freq_chans: usize,
    tile_flags: &HashSet<usize>,
    unflagged_fine_chans: &HashSet<usize>,
) -> Result<(), WriteSolutionsError> {
    match &file.extension().and_then(|os_str| os_str.to_str()) {
        Some("fits") => write_fits(
            file,
            di_jones,
            num_timeblocks,
            total_num_tiles,
            num_fine_freq_chans,
            tile_flags,
            unflagged_fine_chans,
        )?,
        Some("bin") => write_andre_binary(
            file,
            di_jones,
            num_timeblocks,
            total_num_tiles,
            num_fine_freq_chans,
            tile_flags,
            unflagged_fine_chans,
        )?,
        _ => panic!("Tried to write calibration solutions file with an unsupported format!"),
    }
    Ok(())
}

/// Write a "André-Offringa calibrate format" calibration solutions binary file.
fn write_andre_binary(
    bin_file: &Path,
    di_jones: ArrayView3<Jones<f32>>,
    num_timeblocks: usize,
    total_num_tiles: usize,
    num_fine_freq_chans: usize,
    tile_flags: &HashSet<usize>,
    unflagged_fine_chans: &HashSet<usize>,
) -> Result<(), WriteSolutionsError> {
    let num_polarisations = 4;

    let mut bin_file = BufWriter::new(File::create(bin_file)?);
    // 8 floats, 8 bytes per float.
    let mut buf = [0; 8 * 8];
    bin_file.write_all(b"MWAOCAL")?;
    bin_file.write_u8(0)?;
    bin_file.write_i32::<LittleEndian>(0)?;
    bin_file.write_i32::<LittleEndian>(0)?;
    bin_file.write_i32::<LittleEndian>(num_timeblocks as _)?;
    bin_file.write_i32::<LittleEndian>(total_num_tiles as _)?;
    bin_file.write_i32::<LittleEndian>(num_fine_freq_chans as _)?;
    bin_file.write_i32::<LittleEndian>(num_polarisations)?;
    // TODO: Use real timestamps.
    bin_file.write_f64::<LittleEndian>(0.0)?;
    bin_file.write_f64::<LittleEndian>(1.0)?;

    for di_jones_per_time in di_jones.outer_iter() {
        let mut unflagged_tile_index = 0;
        for tile_index in 0..total_num_tiles {
            let mut unflagged_chan_index = 0;
            for chan in 0..num_fine_freq_chans {
                if unflagged_fine_chans.contains(&chan) {
                    // Invert the Jones matrices so that they can be applied as
                    // J D J^H
                    let j = if tile_flags.contains(&tile_index) {
                        // This is a Jones matrix of all NaN.
                        Jones::default().inv()
                    } else {
                        di_jones_per_time[[unflagged_tile_index, unflagged_chan_index]].inv()
                    };

                    LittleEndian::write_f64_into(
                        &[
                            j[0].re as _,
                            j[0].im as _,
                            j[1].re as _,
                            j[1].im as _,
                            j[2].re as _,
                            j[2].im as _,
                            j[3].re as _,
                            j[3].im as _,
                        ],
                        &mut buf,
                    );
                    unflagged_chan_index += 1;
                } else {
                    LittleEndian::write_f64_into(&[f64::NAN; 8], &mut buf);
                }
                bin_file.write_all(&buf)?;
            }
            if !tile_flags.contains(&tile_index) {
                unflagged_tile_index += 1;
            };
        }
    }
    bin_file.flush()?;
    Ok(())
}

fn write_fits(
    fits_file: &Path,
    di_jones: ArrayView3<Jones<f32>>,
    num_timeblocks: usize,
    total_num_tiles: usize,
    num_fine_freq_chans: usize,
    tile_flags: &HashSet<usize>,
    unflagged_fine_chans: &HashSet<usize>,
) -> Result<(), WriteSolutionsError> {
    if fits_file.exists() {
        std::fs::remove_file(&fits_file)?;
    }
    let mut fptr = FitsFile::create(&fits_file).open()?;
    // Four elements for each Jones matrix, and we need to double the last axis,
    // because we can't write complex numbers directly to FITS files; instead,
    // we write each real and imag float as individual floats.
    let dim = [num_timeblocks, total_num_tiles, num_fine_freq_chans, 4 * 2];
    let image_description = ImageDescription {
        data_type: ImageType::Float,
        dimensions: &dim,
    };
    let hdu = fptr.create_image("SOLUTIONS".to_string(), &image_description)?;

    // Fill the fits file with NaN before overwriting with our solved solutions.
    // We have to be tricky with what gets written out, because `di_jones`
    // doesn't necessarily have the same shape as the output.
    let mut fits_image_data = vec![f32::NAN; dim.iter().product()];
    for (timeblock, di_jones_per_time) in di_jones.outer_iter().enumerate() {
        let mut unflagged_tile_index = 0;
        for tile in 0..total_num_tiles {
            let mut unflagged_chan_index = 0;
            for chan in 0..num_fine_freq_chans {
                if unflagged_fine_chans.contains(&chan) {
                    let one_dim_index = timeblock * dim[1] * dim[2] * dim[3]
                        + tile * dim[2] * dim[3]
                        + chan * dim[3];
                    // Invert the Jones matrices so that they can be applied as
                    // J D J^H
                    let j = if tile_flags.contains(&tile) {
                        // This is a Jones matrix of all NaN.
                        Jones::default().inv()
                    } else {
                        di_jones_per_time[[unflagged_tile_index, unflagged_chan_index]].inv()
                    };
                    fits_image_data[one_dim_index] = j[0].re;
                    fits_image_data[one_dim_index + 1] = j[0].im;
                    fits_image_data[one_dim_index + 2] = j[1].re;
                    fits_image_data[one_dim_index + 3] = j[1].im;
                    fits_image_data[one_dim_index + 4] = j[2].re;
                    fits_image_data[one_dim_index + 5] = j[2].im;
                    fits_image_data[one_dim_index + 6] = j[3].re;
                    fits_image_data[one_dim_index + 7] = j[3].im;
                    unflagged_chan_index += 1;
                }
            }
            if !tile_flags.contains(&tile) {
                unflagged_tile_index += 1;
            };
        }
    }
    hdu.write_image(&mut fptr, &fits_image_data)?;

    Ok(())
}

#[derive(Error, Debug)]
pub enum WriteSolutionsError {
    #[error("cfitsio error: {0}")]
    Fitsio(#[from] mwalib::fitsio::errors::Error),

    #[error("IO error: {0}")]
    IO(#[from] std::io::Error),
}
