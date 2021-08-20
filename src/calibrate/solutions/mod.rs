// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to read and write calibration solutions.
//!
//! See for more info:
//! https://github.com/MWATelescope/mwa_hyperdrive/wiki/Calibration-solutions

mod error;

pub(crate) use error::*;

use std::collections::HashSet;
use std::ffi::CString;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use byteorder::{ByteOrder, LittleEndian, ReadBytesExt, WriteBytesExt};
use hifitime::Epoch;
use ndarray::prelude::*;

use mwa_rust_core::{mwalib, time::epoch_as_gps_seconds, Complex, Jones};
use mwalib::fitsio::{
    errors::check_status as fits_check_status,
    images::{ImageDescription, ImageType},
    FitsFile,
};
use mwalib::*;

pub struct CalibrationSolutions {
    pub file: PathBuf,
    pub di_jones: Array3<Jones<f64>>,
    pub num_timeblocks: usize,
    pub total_num_tiles: usize,
    pub total_num_fine_freq_chans: usize,

    /// The centroid timesteps used to produce these calibration solutions.
    pub timesteps: Vec<Epoch>,
    pub obsid: Option<u32>,

    /// The time resolution of the calibration solutions. Only really useful if
    /// there are multiple timeblocks.
    pub time_res: Option<f64>,
}

impl CalibrationSolutions {
    pub(super) fn write_solutions_from_ext(
        &self,
        tile_flags: &HashSet<usize>,
        unflagged_fine_chans: &HashSet<usize>,
    ) -> Result<(), WriteSolutionsError> {
        let ext = self.file.extension().and_then(|e| e.to_str());
        match ext {
            Some("fits") => self.write_hyperdrive_fits(tile_flags, unflagged_fine_chans),
            Some("bin") => self.write_andre_binary(tile_flags, unflagged_fine_chans),
            s => {
                let ext = s.unwrap_or("<no extension>").to_string();
                Err(WriteSolutionsError::UnsupportedExt { ext })
            }
        }
    }

    fn write_hyperdrive_fits(
        &self,
        tile_flags: &HashSet<usize>,
        unflagged_fine_chans: &HashSet<usize>,
    ) -> Result<(), WriteSolutionsError> {
        if self.file.exists() {
            std::fs::remove_file(&self.file)?;
        }
        let mut fptr = FitsFile::create(&self.file).open()?;
        let hdu = fits_open_hdu!(&mut fptr, 0)?;

        // Write the obsid if we can.
        if let Some(obsid) = self.obsid {
            unsafe {
                let keyname = CString::new("OBSID").unwrap();
                let value = CString::new(obsid.to_string()).unwrap();
                let comment = CString::new("The MWA observational ID").unwrap();
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
        if let Some(time_res) = self.time_res {
            unsafe {
                let keyname = CString::new("TIMERES").unwrap();
                let value = CString::new(format!("{:.2}", time_res)).unwrap();
                let comment = CString::new("The time resolution of the solutions").unwrap();
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

        // Write the timesteps as a comma separated list in a string.
        let timesteps = self
            .timesteps
            .iter()
            .map(|&t| format!("{:.2}", epoch_as_gps_seconds(t)))
            .collect::<Vec<_>>()
            .join(",");
        unsafe {
            let keyname = CString::new("TMESTEPS").unwrap();
            let value = CString::new(timesteps).unwrap();
            let comment = CString::new("GPS timesteps used for these solutions").unwrap();
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
            self.num_timeblocks,
            self.total_num_tiles,
            self.total_num_fine_freq_chans,
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
        for (timeblock, di_jones_per_time) in self.di_jones.outer_iter().enumerate() {
            let mut unflagged_tile_index = 0;
            for tile in 0..self.total_num_tiles {
                let mut unflagged_chan_index = 0;
                for chan in 0..self.total_num_fine_freq_chans {
                    if unflagged_fine_chans.contains(&chan) {
                        let one_dim_index = timeblock * dim[1] * dim[2] * dim[3]
                            + tile * dim[2] * dim[3]
                            + chan * dim[3];
                        // Invert the Jones matrices so that they can be applied
                        // as J D J^H
                        let j = if tile_flags.contains(&tile) {
                            // This is a Jones matrix of all NaN.
                            Jones::nan()
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

    /// Write a "André-Offringa calibrate format" calibration solutions binary file.
    fn write_andre_binary(
        &self,
        tile_flags: &HashSet<usize>,
        unflagged_fine_chans: &HashSet<usize>,
    ) -> Result<(), WriteSolutionsError> {
        let num_polarisations = 4;

        let mut bin_file = BufWriter::new(File::create(&self.file)?);
        // 8 floats, 8 bytes per float.
        let mut buf = [0; 8 * 8];
        bin_file.write_all(b"MWAOCAL")?;
        bin_file.write_u8(0)?;
        bin_file.write_u32::<LittleEndian>(0)?;
        bin_file.write_u32::<LittleEndian>(0)?;
        bin_file.write_u32::<LittleEndian>(self.num_timeblocks as _)?;
        bin_file.write_u32::<LittleEndian>(self.total_num_tiles as _)?;
        bin_file.write_u32::<LittleEndian>(self.total_num_fine_freq_chans as _)?;
        bin_file.write_u32::<LittleEndian>(num_polarisations)?;
        // André indicates that "AIPS time" should be used here. However, it
        // looks like only 0.0 is ever written. I don't know what AIPS time is,
        // and I hate leap seconds, so GPS it is.
        bin_file.write_f64::<LittleEndian>(
            self.timesteps
                .first()
                .map(|&t| epoch_as_gps_seconds(t))
                .unwrap_or(0.0),
        )?;
        bin_file.write_f64::<LittleEndian>(
            self.timesteps
                .last()
                .map(|&t| epoch_as_gps_seconds(t))
                .unwrap_or(0.0),
        )?;

        for di_jones_per_time in self.di_jones.outer_iter() {
            let mut unflagged_tile_index = 0;
            for tile_index in 0..self.total_num_tiles {
                let mut unflagged_chan_index = 0;
                for chan in 0..self.total_num_fine_freq_chans {
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

    /// Read in calibration solutions from `self.file`. The format of the file
    /// is determined by `self.ext`. Mostly useful for testing.    
    pub fn read_solutions_from_ext<T: AsRef<Path>>(file: T) -> Result<Self, ReadSolutionsError> {
        match file.as_ref().extension().and_then(|s| s.to_str()) {
            Some("fits") => Self::read_hyperdrive_fits(file),
            Some("bin") => Self::read_andre_binary(file),
            s => {
                let ext = s.unwrap_or("<no extension>").to_string();
                Err(ReadSolutionsError::UnsupportedExt { ext })
            }
        }
    }

    pub fn read_hyperdrive_fits<T: AsRef<Path>>(file: T) -> Result<Self, ReadSolutionsError> {
        let file = file.as_ref().to_path_buf();
        let mut fptr = fits_open!(&file)?;
        let hdu = fits_open_hdu!(&mut fptr, 0)?;
        let obsid = get_optional_fits_key!(&mut fptr, &hdu, "OBSID")?;
        let time_res = get_optional_fits_key!(&mut fptr, &hdu, "TIMERES")?;
        let timesteps: Vec<Epoch> = {
            let timesteps: Option<String> = get_optional_fits_key!(&mut fptr, &hdu, "TMESTEPS")?;
            match timesteps {
                None => vec![],
                Some(s) => s
                    .split(',')
                    .map(|ss| match ss.parse::<f64>() {
                        Ok(float) => Ok(Epoch::from_tai_seconds(float + 19.0)),
                        Err(_) => Err(ReadSolutionsError::Parse {
                            file: file.display().to_string(),
                            key: "TMESTEPS",
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

        Ok(Self {
            file,
            di_jones,
            num_timeblocks,
            total_num_tiles,
            total_num_fine_freq_chans,
            timesteps,
            obsid,
            time_res,
        })
    }

    pub fn read_andre_binary<T: AsRef<Path>>(file: T) -> Result<Self, ReadSolutionsError> {
        let file_str = file.as_ref().display().to_string();
        let mut bin_file = BufReader::new(File::open(&file)?);
        // The first 7 bytes should be ASCII "MWAOCAL".
        let mwaocal_str = String::from_utf8(vec![
            bin_file.read_u8()?,
            bin_file.read_u8()?,
            bin_file.read_u8()?,
            bin_file.read_u8()?,
            bin_file.read_u8()?,
            bin_file.read_u8()?,
            bin_file.read_u8()?,
        ])
        .unwrap();
        if mwaocal_str.as_str() != "MWAOCAL" {
            return Err(ReadSolutionsError::AndreBinaryStr {
                file: file_str,
                got: mwaocal_str,
            });
        }
        for _ in 0..9 {
            match bin_file.read_u8()? {
                0 => (),
                v => {
                    return Err(ReadSolutionsError::AndreBinaryVal {
                        file: file_str,
                        expected: "0",
                        got: v.to_string(),
                    })
                }
            }
        }
        let num_timeblocks = bin_file.read_u32::<LittleEndian>()? as usize;
        let total_num_tiles = bin_file.read_u32::<LittleEndian>()? as usize;
        let total_num_fine_freq_chans = bin_file.read_u32::<LittleEndian>()? as usize;
        let num_polarisations = bin_file.read_u32::<LittleEndian>()? as usize;
        let t = bin_file.read_f64::<LittleEndian>()?;
        let start_time = if t.abs() < f64::EPSILON {
            None
        } else {
            Some(Epoch::from_tai_seconds(t + 19.0))
        };
        let t = bin_file.read_f64::<LittleEndian>()?;
        let end_time = if t.abs() < f64::EPSILON {
            None
        } else {
            Some(Epoch::from_tai_seconds(t + 19.0))
        };
        let mut di_jones_vec = vec![
            0.0;
            num_timeblocks
                * total_num_tiles
                * total_num_fine_freq_chans
                // Real and imag for each polarisation.
                * 2 * num_polarisations
        ];
        bin_file.read_f64_into::<LittleEndian>(&mut di_jones_vec)?;
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

        Ok(Self {
            file: file.as_ref().to_path_buf(),
            di_jones,
            num_timeblocks,
            total_num_tiles,
            total_num_fine_freq_chans,
            timesteps: match (start_time, end_time) {
                (None, None) => vec![],
                (Some(t), None) => vec![t],
                (None, Some(t)) => vec![t],
                (Some(s), Some(e)) => vec![s, e],
            },
            obsid: None,
            time_res: None,
        })
    }
}
