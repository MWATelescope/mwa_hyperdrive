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
use itertools::Itertools;
use marlu::Jones;
use mwalib::{
    fitsio::{
        errors::check_status as fits_check_status,
        images::{ImageDescription, ImageType},
        FitsFile,
    },
    *,
};
use ndarray::prelude::*;
use rayon::prelude::*;

use super::{error::*, CalibrationSolutions};
use mwa_hyperdrive_common::{hifitime, itertools, marlu, mwalib, ndarray, rayon, Complex};

pub(super) fn read<P: AsRef<Path>>(file: P) -> Result<CalibrationSolutions, ReadSolutionsError> {
    fn inner(file: &Path) -> Result<CalibrationSolutions, ReadSolutionsError> {
        let mut fptr = fits_open!(&file)?;
        let hdu = fits_open_hdu!(&mut fptr, 0)?;
        let obsid = get_optional_fits_key!(&mut fptr, &hdu, "OBSID")?;
        let start_timestamps = {
            let timestamps: Option<String> =
                get_optional_fits_key_long_string!(&mut fptr, &hdu, "S_TIMES")?;
            // The retrieved string might be empty (or just have spaces for values)
            match timestamps.as_ref().map(|s| s.trim()) {
                None | Some("") => vec![],
                Some(s) => s
                    .split(',')
                    .map(|ss| match ss.trim().parse::<f64>() {
                        Ok(float) => Ok(Epoch::from_gpst_seconds(float)),
                        Err(_) => Err(ReadSolutionsError::Parse {
                            file: file.display().to_string(),
                            key: "S_TIMES",
                            got: ss.to_string(),
                        }),
                    })
                    .collect::<Vec<Result<Epoch, ReadSolutionsError>>>()
                    .into_iter()
                    .collect::<Result<Vec<Epoch>, ReadSolutionsError>>()?,
            }
        };
        let end_timestamps = {
            let timestamps: Option<String> =
                get_optional_fits_key_long_string!(&mut fptr, &hdu, "E_TIMES")?;
            // The retrieved string might be empty (or just have spaces for values)
            match timestamps.as_ref().map(|s| s.trim()) {
                None | Some("") => vec![],
                Some(s) => s
                    .split(',')
                    .map(|ss| match ss.trim().parse::<f64>() {
                        Ok(float) => Ok(Epoch::from_gpst_seconds(float)),
                        Err(_) => Err(ReadSolutionsError::Parse {
                            file: file.display().to_string(),
                            key: "E_TIMES",
                            got: ss.to_string(),
                        }),
                    })
                    .collect::<Vec<Result<Epoch, ReadSolutionsError>>>()
                    .into_iter()
                    .collect::<Result<Vec<Epoch>, ReadSolutionsError>>()?,
            }
        };
        let average_timestamps = {
            let timestamps: Option<String> =
                get_optional_fits_key_long_string!(&mut fptr, &hdu, "A_TIMES")?;
            // The retrieved string might be empty (or just have spaces for values)
            match timestamps.as_ref().map(|s| s.trim()) {
                None | Some("") => vec![],
                Some(s) => s
                    .split(',')
                    .map(|ss| match ss.trim().parse::<f64>() {
                        Ok(float) => Ok(Epoch::from_gpst_seconds(float)),
                        Err(_) => Err(ReadSolutionsError::Parse {
                            file: file.display().to_string(),
                            key: "A_TIMES",
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

        // Find any tiles containing only NaNs; these are flagged.
        let flagged_tiles = di_jones
            .axis_iter(Axis(1))
            .into_par_iter()
            .enumerate()
            .filter(|(_, di_jones)| di_jones.iter().all(|j| j.any_nan()))
            .map(|pair| pair.0)
            .collect();

        // Also find any flagged channels.
        let flagged_chanblocks = di_jones
            .axis_iter(Axis(2))
            .into_par_iter()
            .enumerate()
            .filter(|(_, di_jones)| di_jones.iter().all(|j| j.any_nan()))
            .map(|pair| pair.0.try_into().unwrap())
            .collect();

        Ok(CalibrationSolutions {
            di_jones,
            flagged_tiles,
            flagged_chanblocks,
            obsid,
            start_timestamps,
            end_timestamps,
            average_timestamps,
        })
    }
    inner(file.as_ref())
}

pub(super) fn write<P: AsRef<Path>>(
    sols: &CalibrationSolutions,
    file: P,
) -> Result<(), WriteSolutionsError> {
    fn inner(sols: &CalibrationSolutions, file: &Path) -> Result<(), WriteSolutionsError> {
        if file.exists() {
            std::fs::remove_file(&file)?;
        }
        let mut fptr = FitsFile::create(&file).open()?;
        let hdu = fits_open_hdu!(&mut fptr, 0)?;
        let mut status = 0;

        // Signal that we're using long strings.
        unsafe {
            fitsio_sys::ffplsw(
                fptr.as_raw(), /* I - FITS file pointer  */
                &mut status,   /* IO - error status       */
            );
            fits_check_status(status)?;
        }

        // Write the obsid if we can.
        if let Some(obsid) = sols.obsid {
            unsafe {
                let key_name = CString::new("OBSID").unwrap();
                let value = CString::new(obsid.to_string()).unwrap();
                let comment = CString::new("The MWA observation ID").unwrap();
                // ffpkys = fits_write_key_str
                fitsio_sys::ffpkys(
                    fptr.as_raw(),     /* I - FITS file pointer        */
                    key_name.as_ptr(), /* I - name of keyword to write */
                    value.as_ptr(),    /* I - keyword value            */
                    comment.as_ptr(),  /* I - keyword comment          */
                    &mut status,       /* IO - error status            */
                );
                fits_check_status(status)?;
            }
        }

        // Write the timestamps as a comma separated list in a string.
        for (key_name, key_type, timestamps) in [
            ("S_TIMES", "Start", &sols.start_timestamps),
            ("E_TIMES", "End", &sols.end_timestamps),
            ("A_TIMES", "Average", &sols.average_timestamps),
        ] {
            // There's an "intersperse" in unstable Rust, so use fully-qualified
            // syntax to appease clippy. Itertools might not be needed in this
            // module once std's intersperse becomes stable.
            let timestamps_str = timestamps
                .iter()
                .map(|&t| format!("{:.3}", t.as_gpst_seconds()));
            let timestamps_str =
                Itertools::intersperse(timestamps_str, ",".into()).collect::<String>();
            unsafe {
                let key_name = CString::new(key_name).unwrap();
                let value = CString::new(timestamps_str).unwrap();
                let comment = CString::new(format!(
                    "{key_type} GPS timesteps used for these solution timeblocks"
                ))
                .unwrap();
                let mut status = 0;
                // ffpkls = fits_write_key_longstr
                fitsio_sys::ffpkls(
                    fptr.as_raw(),     /* I - FITS file pointer        */
                    key_name.as_ptr(), /* I - name of keyword to write */
                    value.as_ptr(),    /* I - keyword value            */
                    comment.as_ptr(),  /* I - keyword comment          */
                    &mut status,       /* IO - error status            */
                );
                fits_check_status(status)?;
            }
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
        let (num_timeblocks, total_num_tiles, total_num_chanblocks) = sols.di_jones.dim();
        let dim = [num_timeblocks, total_num_tiles, total_num_chanblocks, 4 * 2];
        let image_description = ImageDescription {
            data_type: ImageType::Double,
            dimensions: &dim,
        };
        let hdu = fptr.create_image("SOLUTIONS".to_string(), &image_description)?;

        // Fill the fits file with NaN before overwriting with our solved
        // solutions. We have to be tricky with what gets written out, because
        // `di_jones` doesn't necessarily have the same shape as the output.
        let mut fits_image_data = vec![f64::NAN; dim.iter().product()];
        let mut one_dim_index = 0;
        for j in sols.di_jones.iter() {
            fits_image_data[one_dim_index] = j[0].re;
            fits_image_data[one_dim_index + 1] = j[0].im;
            fits_image_data[one_dim_index + 2] = j[1].re;
            fits_image_data[one_dim_index + 3] = j[1].im;
            fits_image_data[one_dim_index + 4] = j[2].re;
            fits_image_data[one_dim_index + 5] = j[2].im;
            fits_image_data[one_dim_index + 6] = j[3].re;
            fits_image_data[one_dim_index + 7] = j[3].im;

            one_dim_index += 8;
        }
        hdu.write_image(&mut fptr, &fits_image_data)?;

        Ok(())
    }
    inner(sols, file.as_ref())
}
