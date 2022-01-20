// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to read and write hyperdrive-style calibration solutions.
//!
//! See for more info:
//! https://github.com/MWATelescope/mwa_hyperdrive/wiki/Calibration-solutions

use std::ffi::CString;
use std::path::Path;

use hifitime::Epoch;
use marlu::{
    time::{epoch_as_gps_seconds, gps_to_epoch},
    Jones,
};
use mwalib::{
    fitsio::{
        errors::check_status as fits_check_status,
        images::{ImageDescription, ImageType},
        FitsFile,
    },
    *,
};
use ndarray::prelude::*;

use super::{error::*, CalibrationSolutions};
use mwa_hyperdrive_common::{hifitime, marlu, mwalib, ndarray, Complex};

pub(super) fn read<T: AsRef<Path>>(file: T) -> Result<CalibrationSolutions, ReadSolutionsError> {
    let file = file.as_ref().to_path_buf();
    let mut fptr = fits_open!(&file)?;
    let hdu = fits_open_hdu!(&mut fptr, 0)?;
    let obsid = get_optional_fits_key!(&mut fptr, &hdu, "OBSID")?;
    let time_res = get_optional_fits_key!(&mut fptr, &hdu, "TIMERES")?;
    let start_timestamps: Vec<Epoch> = {
        let timestamps: Option<String> = get_optional_fits_key!(&mut fptr, &hdu, "TIMESTAMPS")?;
        // The retrieved string might be empty (or just have spaces for values)
        match timestamps.as_ref().map(|s| s.trim()) {
            None | Some("") => vec![],
            Some(s) => s
                .split(',')
                .map(|ss| match ss.trim().parse::<f64>() {
                    Ok(float) => Ok(gps_to_epoch(float)),
                    Err(_) => Err(ReadSolutionsError::Parse {
                        file: file.display().to_string(),
                        key: "TIMESTAMPS",
                        got: ss.to_string(),
                    }),
                })
                .collect::<Vec<Result<Epoch, ReadSolutionsError>>>()
                .into_iter()
                .collect::<Result<Vec<Epoch>, ReadSolutionsError>>()?,
        }
    };
    let hdu = fits_open_hdu!(&mut fptr, 1)?;
    let num_timeblocks: usize = get_required_fits_key!(&mut fptr, &hdu, "NAXIS4")?;
    let total_num_tiles: usize = get_required_fits_key!(&mut fptr, &hdu, "NAXIS3")?;
    let total_num_fine_freq_chans: usize = get_required_fits_key!(&mut fptr, &hdu, "NAXIS2")?;
    let num_polarisations: usize = {
        let p: usize = get_required_fits_key!(&mut fptr, &hdu, "NAXIS1")?;
        // There are two floats per polarisation; one for real, one for
        // imag.
        p / 2
    };

    let di_jones_vec: Vec<f64> = get_fits_image!(&mut fptr, &hdu)?;
    let di_jones_a4 = Array4::from_shape_vec(
        (
            num_timeblocks,
            total_num_tiles,
            total_num_fine_freq_chans,
            2 * num_polarisations,
        ),
        di_jones_vec,
    )
    .unwrap();
    let di_jones = di_jones_a4.map_axis(Axis(3), |view| {
        Jones::from([
            Complex::new(view[0], view[1]),
            Complex::new(view[2], view[3]),
            Complex::new(view[4], view[5]),
            Complex::new(view[6], view[7]),
        ])
    });

    Ok(CalibrationSolutions {
        di_jones,
        num_timeblocks,
        total_num_tiles,
        unflagged_tiles: todo!(),
        total_num_fine_freq_chans,
        unflagged_fine_channels: todo!(),
        start_timestamps,
        obsid,
        time_res,
    })
}

pub(super) fn write<T: AsRef<Path>>(
    sols: &CalibrationSolutions,
    file: T,
) -> Result<(), WriteSolutionsError> {
    if file.as_ref().exists() {
        std::fs::remove_file(&file)?;
    }
    let mut fptr = FitsFile::create(&file).open()?;
    let hdu = fits_open_hdu!(&mut fptr, 0)?;

    // Write the obsid if we can.
    if let Some(obsid) = sols.obsid {
        unsafe {
            let keyname = CString::new("OBSID").unwrap();
            let value = CString::new(obsid.to_string()).unwrap();
            let comment = CString::new("The MWA observation ID").unwrap();
            let mut status = 0;
            // ffpkys = fits_write_key_str
            fitsio_sys::ffpkys(
                fptr.as_raw(),    /* I - FITS file pointer        */
                keyname.as_ptr(), /* I - name of keyword to write */
                value.as_ptr(),   /* I - keyword value            */
                comment.as_ptr(), /* I - keyword comment          */
                &mut status,      /* IO - error status            */
            );
            fits_check_status(status)?;
        }
    }

    // Write the time resolution if we can.
    if let Some(time_res) = sols.time_res {
        unsafe {
            let keyname = CString::new("TIMERES").unwrap();
            let value = CString::new(format!("{:.2}", time_res)).unwrap();
            let comment = CString::new("The number of seconds per timeblock").unwrap();
            let mut status = 0;
            fitsio_sys::ffpkys(
                fptr.as_raw(),    /* I - FITS file pointer        */
                keyname.as_ptr(), /* I - name of keyword to write */
                value.as_ptr(),   /* I - keyword value            */
                comment.as_ptr(), /* I - keyword comment          */
                &mut status,      /* IO - error status            */
            );
            fits_check_status(status)?;
        }
    }

    // Write the start timestamps as a comma separated list in a string.
    let start_timestamps = sols
        .start_timestamps
        .iter()
        .map(|&t| format!("{:.2}", epoch_as_gps_seconds(t)))
        .collect::<Vec<_>>()
        .join(",");
    unsafe {
        let keyname = CString::new("TIMESTAMPS").unwrap();
        let value = CString::new(start_timestamps).unwrap();
        let comment =
            CString::new("Start GPS timesteps used for these solution timeblocks").unwrap();
        let mut status = 0;
        // ffpkls = fits_write_key_longstr
        fitsio_sys::ffpkls(
            fptr.as_raw(),    /* I - FITS file pointer        */
            keyname.as_ptr(), /* I - name of keyword to write */
            value.as_ptr(),   /* I - keyword value            */
            comment.as_ptr(), /* I - keyword comment          */
            &mut status,      /* IO - error status            */
        );
        fits_check_status(status)?;
    }
    hdu.write_key(
        &mut fptr,
        "SOFTWARE",
        format!(
            "Created by {} v{}",
            env!("CARGO_PKG_NAME"),
            env!("CARGO_PKG_VERSION")
        ),
    )?;

    // Four elements for each Jones matrix, and we need to double the last
    // axis, because we can't write complex numbers directly to FITS files;
    // instead, we write each real and imag float as individual floats.
    let dim = [
        sols.num_timeblocks,
        sols.total_num_tiles,
        sols.total_num_fine_freq_chans,
        4 * 2,
    ];
    let image_description = ImageDescription {
        data_type: ImageType::Float,
        dimensions: &dim,
    };
    let hdu = fptr.create_image("SOLUTIONS".to_string(), &image_description)?;

    // Fill the fits file with NaN before overwriting with our solved
    // solutions. We have to be tricky with what gets written out, because
    // `di_jones` doesn't necessarily have the same shape as the output.
    let mut fits_image_data = vec![f64::NAN; dim.iter().product()];
    for (timeblock, di_jones_per_time) in sols.di_jones.outer_iter().enumerate() {
        let mut unflagged_tile_index = 0;
        for tile in 0..sols.total_num_tiles {
            let mut unflagged_chan_index = 0;
            for chan in 0..sols.total_num_fine_freq_chans {
                if sols.unflagged_fine_channels.contains(&chan) {
                    let one_dim_index = timeblock * dim[1] * dim[2] * dim[3]
                        + tile * dim[2] * dim[3]
                        + chan * dim[3];
                    // Invert the Jones matrices so that they can be applied
                    // as J D J^H
                    let j = if sols.unflagged_tiles.contains(&tile) {
                        di_jones_per_time[(unflagged_tile_index, unflagged_chan_index)].inv()
                    } else {
                        Jones::nan() // This is a Jones matrix of all NaN.
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
            if sols.unflagged_tiles.contains(&tile) {
                unflagged_tile_index += 1;
            };
        }
    }
    hdu.write_image(&mut fptr, &fits_image_data)?;

    Ok(())
}
