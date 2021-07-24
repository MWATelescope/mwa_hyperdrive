// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to handle writing to uvfits files.

use std::collections::HashSet;
use std::ffi::CString;
use std::path::Path;

use erfa_sys::{ERFA_DJM0, ERFA_WGS84};
use fitsio::{errors::check_status as fits_check_status, FitsFile};
use hifitime::Epoch;
use ndarray::prelude::*;

use super::*;
use crate::math::{cross_correlation_baseline_to_tiles, num_tiles_from_num_baselines};
use mwa_hyperdrive_core::{
    constants::VEL_C, erfa_sys, mwalib, Jones, RADec, XyzGeocentric, XyzGeodetic, UVW,
};
use mwalib::{fitsio, fitsio_sys};

/// A helper struct to write out a uvfits file.
pub(crate) struct UvfitsWriter<'a> {
    /// The path to the uvifts file.
    path: &'a Path,

    /// The number of timesteps to be written.    
    num_timesteps: usize,

    /// The number of baselines per timestep to be written.
    num_baselines: usize,

    /// The number of fine channels per baseline to be written. uvfits has no
    /// notion of "fine channel" or "coarse channel".
    num_chans: usize,

    /// The number of uvfits rows. This is equal to `num_timesteps` *
    /// `num_baselines`.
    total_num_rows: usize,

    /// The number of uvfits rows that have currently been written.
    current_num_rows: usize,

    /// The frequency at the centre of the bandwidth \[Hz\].
    centre_freq: f64,

    /// A `hifitime` [Epoch] struct associated with the first timestep of the
    /// data.
    start_epoch: Epoch,
}

impl<'a> UvfitsWriter<'a> {
    /// Create a new uvfits file at the specified filename.
    pub(crate) fn new(
        filename: &'a Path,
        num_timesteps: usize,
        num_baselines: usize,
        num_chans: usize,
        start_epoch: Epoch,
        fine_chan_width_hz: f64,
        centre_freq_hz: f64,
        centre_freq_chan: usize,
        phase_centre: &RADec,
        obs_name: Option<&str>,
    ) -> Result<Self, UvfitsWriteError> {
        // Delete any file that already exists.
        if filename.exists() {
            std::fs::remove_file(&filename)?;
        }

        // Create a new fits file.
        let mut status = 0;
        let c_filename = CString::new(filename.to_str().unwrap())?;
        let mut fptr = std::ptr::null_mut();
        unsafe {
            fitsio_sys::ffinit(
                &mut fptr as *mut *mut _, /* O - FITS file pointer                   */
                c_filename.as_ptr(),      /* I - name of file to create              */
                &mut status,              /* IO - error status                       */
            );
        }
        fits_check_status(status)?;

        // Initialise the group header. Copied from cotter. -32 means FLOAT_IMG.
        let naxis = 6;
        let mut naxes = [0, 3, 4, num_chans as i64, 1, 1];
        let num_group_params = 5;
        let total_num_rows = num_timesteps * num_baselines;
        unsafe {
            fitsio_sys::ffphpr(
                fptr,                  /* I - FITS file pointer                        */
                1,                     /* I - does file conform to FITS standard? 1/0  */
                -32,                   /* I - number of bits per data value pixel      */
                naxis,                 /* I - number of axes in the data array         */
                naxes.as_mut_ptr(),    /* I - length of each data axis                 */
                num_group_params,      /* I - number of group parameters (usually 0)   */
                total_num_rows as i64, /* I - number of random groups (usually 1 or 0) */
                1,                     /* I - may FITS file have extensions?           */
                &mut status,           /* IO - error status                            */
            );
        }
        fits_check_status(status)?;

        // Finally close the file.
        unsafe {
            fitsio_sys::ffclos(fptr, &mut status);
        }
        fits_check_status(status)?;

        // Open the fits file with rust-fitsio.
        let mut u = FitsFile::edit(&filename)?;
        let hdu = u.hdu(0)?;
        hdu.write_key(&mut u, "BSCALE", 1.0)?;

        // Set header names and scales.
        for (i, &param) in ["UU", "VV", "WW", "BASELINE", "DATE"].iter().enumerate() {
            let ii = i + 1;
            hdu.write_key(&mut u, &format!("PTYPE{}", ii), param)?;
            hdu.write_key(&mut u, &format!("PSCAL{}", ii), 1.0)?;
            if param != "DATE" {
                hdu.write_key(&mut u, &format!("PZERO{}", ii), 0.0)?;
            } else {
                // Set the zero level for the DATE column.
                hdu.write_key(
                    &mut u,
                    &format!("PZERO{}", ii),
                    start_epoch.as_jde_utc_days().floor() + 0.5,
                )?;
            }
        }
        hdu.write_key(&mut u, "DATE-OBS", get_truncated_date_string(start_epoch))?;

        // Dimensions.
        hdu.write_key(&mut u, "CTYPE2", "COMPLEX")?;
        hdu.write_key(&mut u, "CRVAL2", 1.0)?;
        hdu.write_key(&mut u, "CRPIX2", 1.0)?;
        hdu.write_key(&mut u, "CDELT2", 1.0)?;

        // Linearly polarised.
        hdu.write_key(&mut u, "CTYPE3", "STOKES")?;
        hdu.write_key(&mut u, "CRVAL3", -5)?;
        hdu.write_key(&mut u, "CDELT3", -1)?;
        hdu.write_key(&mut u, "CRPIX3", 1.0)?;

        hdu.write_key(&mut u, "CTYPE4", "FREQ")?;
        hdu.write_key(&mut u, "CRVAL4", centre_freq_hz)?;
        hdu.write_key(&mut u, "CDELT4", fine_chan_width_hz)?;
        hdu.write_key(&mut u, "CRPIX4", centre_freq_chan as u64 + 1)?;

        hdu.write_key(&mut u, "CTYPE5", "RA")?;
        hdu.write_key(&mut u, "CRVAL5", phase_centre.ra.to_degrees())?;
        hdu.write_key(&mut u, "CDELT5", 1)?;
        hdu.write_key(&mut u, "CRPIX5", 1)?;

        hdu.write_key(&mut u, "CTYPE6", "DEC")?;
        hdu.write_key(&mut u, "CRVAL6", phase_centre.dec.to_degrees())?;
        hdu.write_key(&mut u, "CDELT6", 1)?;
        hdu.write_key(&mut u, "CRPIX6", 1)?;

        hdu.write_key(&mut u, "OBSRA", phase_centre.ra.to_degrees())?;
        hdu.write_key(&mut u, "OBSDEC", phase_centre.dec.to_degrees())?;
        hdu.write_key(&mut u, "EPOCH", 2000.0)?;

        hdu.write_key(&mut u, "OBJECT", obs_name.unwrap_or("Undefined"))?;
        hdu.write_key(&mut u, "TELESCOP", "MWA")?;
        hdu.write_key(&mut u, "INSTRUME", "MWA")?;

        // This is apparently required...
        let history = CString::new("AIPS WTSCAL =  1.0").unwrap();
        unsafe {
            fitsio_sys::ffphis(
                u.as_raw(),       /* I - FITS file pointer  */
                history.as_ptr(), /* I - history string     */
                &mut status,      /* IO - error status      */
            );
        }
        fits_check_status(status)?;

        // Add in version information
        let comment = CString::new(format!(
            "Created by {} v{}",
            env!("CARGO_PKG_NAME"),
            env!("CARGO_PKG_VERSION")
        ))
        .unwrap();
        unsafe {
            fitsio_sys::ffpcom(
                u.as_raw(),       /* I - FITS file pointer   */
                comment.as_ptr(), /* I - comment string      */
                &mut status,      /* IO - error status       */
            );
        }
        fits_check_status(status)?;

        hdu.write_key(&mut u, "SOFTWARE", env!("CARGO_PKG_NAME"))?;
        hdu.write_key(
            &mut u,
            "GITLABEL",
            format!("v{}", env!("CARGO_PKG_VERSION")),
        )?;

        Ok(Self {
            path: filename,
            num_timesteps,
            num_baselines,
            num_chans,
            total_num_rows,
            current_num_rows: 0,
            centre_freq: centre_freq_hz,
            start_epoch,
        })
    }

    /// Opens the associated uvfits file in edit mode, returning the [FitsFile]
    /// struct.
    pub(crate) fn open(&self) -> Result<FitsFile, fitsio::errors::Error> {
        let mut f = FitsFile::edit(&self.path)?;
        // Ensure HDU 0 is opened.
        f.hdu(0)?;
        Ok(f)
    }

    /// Write the antenna table to a uvfits file. Assumes that the array
    /// location is MWA.
    ///
    /// `centre_freq` is the centre frequency of the coarse band that this
    /// uvfits file pertains to. `positions` are the [XyzGeodetic] coordinates
    /// of the MWA tiles.
    ///
    /// `Self` must have only have a single HDU when this function is called
    /// (true when using methods only provided by `Self`).
    // Derived from cotter.
    pub(crate) fn write_uvfits_antenna_table<T: AsRef<str>>(
        self,
        antenna_names: &[T],
        positions: &[XyzGeodetic],
    ) -> Result<(), UvfitsWriteError> {
        if self.current_num_rows != self.total_num_rows {
            return Err(UvfitsWriteError::NotEnoughRowsWritten {
                current: self.current_num_rows,
                total: self.total_num_rows,
            });
        }

        let mut uvfits = self.open()?;

        // Stuff that a uvfits file always expects?
        let col_names = [
            "ANNAME", "STABXYZ", "NOSTA", "MNTSTA", "STAXOF", "POLTYA", "POLAA", "POLCALA",
            "POLTYB", "POLAB", "POLCALB",
        ];
        let col_formats = [
            "8A", "3D", "1J", "1J", "1E", "1A", "1E", "3E", "1A", "1E", "3E",
        ];
        let col_units = [
            "", "METERS", "", "", "METERS", "", "DEGREES", "", "", "DEGREES", "",
        ];
        let mut c_col_names = rust_strings_to_c_strings(&col_names)?;
        let mut c_col_formats = rust_strings_to_c_strings(&col_formats)?;
        let mut c_col_units = rust_strings_to_c_strings(&col_units)?;
        let extname = CString::new("AIPS AN").unwrap();

        // ffcrtb creates a new binary table in a new HDU. This should be the second
        // HDU, so there should only be one HDU before this function is called.
        let mut status = 0;
        unsafe {
            // BINARY_TBL is 2.
            fitsio_sys::ffcrtb(
                uvfits.as_raw(),            /* I - FITS file pointer                        */
                2,                          /* I - type of table to create                  */
                0,                          /* I - number of rows in the table              */
                11,                         /* I - number of columns in the table           */
                c_col_names.as_mut_ptr(),   /* I - name of each column                      */
                c_col_formats.as_mut_ptr(), /* I - value of TFORMn keyword for each column  */
                c_col_units.as_mut_ptr(),   /* I - value of TUNITn keyword for each column  */
                extname.as_ptr(),           /* I - value of EXTNAME keyword, if any         */
                &mut status,                /* IO - error status                            */
            );
        }
        fits_check_status(status)?;

        // Open the newly-created HDU.
        let hdu = uvfits.hdu(1)?;

        // Set ARRAYX, Y and Z to the MWA's coordinates in XYZ (geocentric). The
        // results here are slightly different to those given by cotter. This is
        // at least partly due to different constants (the altitude is
        // definitely slightly different), but possibly also because ERFA is
        // more accurate than cotter's "homebrewed" Geodetic2XYZ.
        let mut mwa_xyz: [f64; 3] = [0.0; 3];
        unsafe {
            status = erfa_sys::eraGd2gc(
                ERFA_WGS84 as i32,             // ellipsoid identifier (Note 1)
                mwalib::MWA_LONGITUDE_RADIANS, // longitude (radians, east +ve)
                mwalib::MWA_LATITUDE_RADIANS,  // latitude (geodetic, radians, Note 3)
                mwalib::MWA_ALTITUDE_METRES,   // height above ellipsoid (geodetic, Notes 2,3)
                mwa_xyz.as_mut_ptr(),          // geocentric vector (Note 2)
            );
        }
        if status != 0 {
            return Err(UvfitsWriteError::Erfa {
                source_file: file!(),
                source_line: line!(),
                status,
                function: "eraGd2gc",
            });
        }
        let mwa_xyz = XyzGeocentric {
            x: mwa_xyz[0],
            y: mwa_xyz[1],
            z: mwa_xyz[2],
        };

        hdu.write_key(&mut uvfits, "ARRAYX", mwa_xyz.x)?;
        hdu.write_key(&mut uvfits, "ARRAYY", mwa_xyz.y)?;
        hdu.write_key(&mut uvfits, "ARRAYZ", mwa_xyz.z)?;

        hdu.write_key(&mut uvfits, "FREQ", self.centre_freq)?;

        // Get the Greenwich apparent sidereal time from ERFA.
        let mjd = self.start_epoch.as_mjd_utc_days();
        let gst = unsafe { erfa_sys::eraGst06a(ERFA_DJM0, mjd.floor(), ERFA_DJM0, mjd.floor()) }
            .to_degrees();
        hdu.write_key(&mut uvfits, "GSTIA0", gst)?;
        hdu.write_key(&mut uvfits, "DEGPDY", 3.60985e2)?; // Earth's rotation rate

        let date_truncated = get_truncated_date_string(self.start_epoch);
        hdu.write_key(&mut uvfits, "RDATE", date_truncated)?;

        hdu.write_key(&mut uvfits, "POLARX", 0.0)?;
        hdu.write_key(&mut uvfits, "POLARY", 0.0)?;
        hdu.write_key(&mut uvfits, "UT1UTC", 0.0)?;
        hdu.write_key(&mut uvfits, "DATUTC", 0.0)?;

        hdu.write_key(&mut uvfits, "TIMSYS", "UTC")?;
        hdu.write_key(&mut uvfits, "ARRNAM", "MWA")?;
        hdu.write_key(&mut uvfits, "NUMORB", 0)?; // number of orbital parameters in table
        hdu.write_key(&mut uvfits, "NOPCAL", 3)?; // Nr pol calibration values / IF(N_pcal)
        hdu.write_key(&mut uvfits, "FREQID", -1)?; // Frequency setup number
        hdu.write_key(&mut uvfits, "IATUTC", 33.0)?;

        // Assume the station coordinates are "right handed".
        hdu.write_key(&mut uvfits, "XYZHAND", "RIGHT")?;

        let c_antenna_names = rust_strings_to_c_strings(antenna_names)?;

        // Write to the table row by row.
        for (i, pos) in positions.iter().enumerate() {
            let row = i as i64 + 1;
            unsafe {
                // ANNAME. ffpcls = fits_write_col_str
                fitsio_sys::ffpcls(
                    uvfits.as_raw(),                   /* I - FITS file pointer                       */
                    1,   /* I - number of column to write (1 = 1st col) */
                    row, /* I - first row to write (1 = 1st row)        */
                    1,   /* I - first vector element to write (1 = 1st) */
                    1,   /* I - number of strings to write              */
                    [c_antenna_names[i]].as_mut_ptr(), /* I - array of pointers to strings            */
                    &mut status, /* IO - error status                           */
                );
                fits_check_status(status)?;

                let mut c_xyz = [pos.x, pos.y, pos.z];
                // STABXYZ. ffpcld = fits_write_col_dbl
                fitsio_sys::ffpcld(
                    uvfits.as_raw(),    /* I - FITS file pointer                       */
                    2,                  /* I - number of column to write (1 = 1st col) */
                    row,                /* I - first row to write (1 = 1st row)        */
                    1,                  /* I - first vector element to write (1 = 1st) */
                    3,                  /* I - number of values to write               */
                    c_xyz.as_mut_ptr(), /* I - array of values to write                */
                    &mut status,        /* IO - error status                           */
                );
                fits_check_status(status)?;

                // NOSTA. ffpclk = fits_write_col_int
                fitsio_sys::ffpclk(
                    uvfits.as_raw(),           /* I - FITS file pointer                       */
                    3,                         /* I - number of column to write (1 = 1st col) */
                    row,                       /* I - first row to write (1 = 1st row)        */
                    1,                         /* I - first vector element to write (1 = 1st) */
                    1,                         /* I - number of values to write               */
                    [row as i32].as_mut_ptr(), /* I - array of values to write                */
                    &mut status,               /* IO - error status                           */
                );
                fits_check_status(status)?;

                // MNTSTA
                fitsio_sys::ffpclk(
                    uvfits.as_raw(),  /* I - FITS file pointer                       */
                    4,                /* I - number of column to write (1 = 1st col) */
                    row,              /* I - first row to write (1 = 1st row)        */
                    1,                /* I - first vector element to write (1 = 1st) */
                    1,                /* I - number of values to write               */
                    [0].as_mut_ptr(), /* I - array of values to write                */
                    &mut status,      /* IO - error status                           */
                );
                fits_check_status(status)?;

                // No row 5?
                // POLTYA
                fitsio_sys::ffpcls(
                    uvfits.as_raw(), /* I - FITS file pointer                       */
                    6,               /* I - number of column to write (1 = 1st col) */
                    row,             /* I - first row to write (1 = 1st row)        */
                    1,               /* I - first vector element to write (1 = 1st) */
                    1,               /* I - number of strings to write              */
                    [CString::new("X").unwrap().into_raw()].as_mut_ptr(), /* I - array of pointers to strings            */
                    &mut status, /* IO - error status                           */
                );
                fits_check_status(status)?;

                // POLAA. ffpcle = fits_write_col_flt
                fitsio_sys::ffpcle(
                    uvfits.as_raw(),    /* I - FITS file pointer                       */
                    7,                  /* I - number of column to write (1 = 1st col) */
                    row,                /* I - first row to write (1 = 1st row)        */
                    1,                  /* I - first vector element to write (1 = 1st) */
                    1,                  /* I - number of values to write               */
                    [0.0].as_mut_ptr(), /* I - array of values to write                */
                    &mut status,        /* IO - error status                           */
                );
                fits_check_status(status)?;

                // POL calA
                fitsio_sys::ffpcle(
                    uvfits.as_raw(),    /* I - FITS file pointer                       */
                    8,                  /* I - number of column to write (1 = 1st col) */
                    row,                /* I - first row to write (1 = 1st row)        */
                    1,                  /* I - first vector element to write (1 = 1st) */
                    1,                  /* I - number of values to write               */
                    [0.0].as_mut_ptr(), /* I - array of values to write                */
                    &mut status,        /* IO - error status                           */
                );
                fits_check_status(status)?;

                // POLTYB
                fitsio_sys::ffpcls(
                    uvfits.as_raw(), /* I - FITS file pointer                       */
                    9,               /* I - number of column to write (1 = 1st col) */
                    row,             /* I - first row to write (1 = 1st row)        */
                    1,               /* I - first vector element to write (1 = 1st) */
                    1,               /* I - number of strings to write              */
                    [CString::new("Y").unwrap().into_raw()].as_mut_ptr(), /* I - array of pointers to strings            */
                    &mut status, /* IO - error status                           */
                );
                fits_check_status(status)?;

                // POLAB.
                fitsio_sys::ffpcle(
                    uvfits.as_raw(),     /* I - FITS file pointer                       */
                    10,                  /* I - number of column to write (1 = 1st col) */
                    row,                 /* I - first row to write (1 = 1st row)        */
                    1,                   /* I - first vector element to write (1 = 1st) */
                    1,                   /* I - number of values to write               */
                    [90.0].as_mut_ptr(), /* I - array of values to write                */
                    &mut status,         /* IO - error status                           */
                );
                fits_check_status(status)?;

                // POL calB
                fitsio_sys::ffpcle(
                    uvfits.as_raw(),    /* I - FITS file pointer                       */
                    11,                 /* I - number of column to write (1 = 1st col) */
                    row,                /* I - first row to write (1 = 1st row)        */
                    1,                  /* I - first vector element to write (1 = 1st) */
                    1,                  /* I - number of values to write               */
                    [0.0].as_mut_ptr(), /* I - array of values to write                */
                    &mut status,        /* IO - error status                           */
                );
                fits_check_status(status)?;
            }
        }

        Ok(())
    }

    /// Write a visibility row into the uvfits file.
    ///
    /// `uvfits` must have been opened in write mode and currently have HDU 0
    /// open. The [FitsFile] must be supplied to this function to force the
    /// caller to think about calling this function efficiently; opening the
    /// file for every call would be a problem, and keeping the file open in
    /// [UvfitsWriter] would mean the struct is not thread safe.
    ///
    /// `tile_index1` and `tile_index2` are expected to be zero indexed; they
    /// are made one indexed by this function.
    // TODO: Assumes that all fine channels are written in `vis.` This needs to
    // be updated to add visibilities to an existing uvfits row.
    pub(crate) fn write_vis(
        &mut self,
        uvfits: &mut FitsFile,
        uvw: &UVW,
        tile_index1: usize,
        tile_index2: usize,
        epoch: Epoch,
        vis: &[f32],
    ) -> Result<(), UvfitsWriteError> {
        if self.current_num_rows + 1 > self.total_num_rows {
            return Err(UvfitsWriteError::BadRowNum {
                row_num: self.current_num_rows,
                num_rows: self.total_num_rows,
            });
        }

        let mut row = Vec::with_capacity(5 + vis.len());
        row.push((uvw.u / VEL_C) as f32);
        row.push((uvw.v / VEL_C) as f32);
        row.push((uvw.w / VEL_C) as f32);
        row.push(encode_uvfits_baseline(tile_index1 + 1, tile_index2 + 1) as f32);
        let jd_trunc = self.start_epoch.as_jde_utc_days().floor() + 0.5;
        let jd_frac = epoch.as_jde_utc_days() - jd_trunc;
        row.push(jd_frac as f32);
        for &v in vis {
            row.push(v);
        }

        let mut status = 0;
        unsafe {
            fitsio_sys::ffpgpe(
                uvfits.as_raw(),                  /* I - FITS file pointer                      */
                self.current_num_rows as i64 + 1, /* I - group to write(1 = 1st group)          */
                1,                                /* I - first vector element to write(1 = 1st) */
                row.len() as i64,                 /* I - number of values to write              */
                row.as_mut_ptr(),                 /* I - array of values that are written       */
                &mut status,                      /* IO - error status                          */
            );
        }
        fits_check_status(status)?;
        self.current_num_rows += 1;
        Ok(())
    }

    /// Assumes that `vis_array` has already had `weights` applied; these need
    /// to be undone locally by this function.
    // TODO: Assumes that all fine channels are written for all baselines in
    // `vis_array.`
    pub(crate) fn write_from_vis(
        &mut self,
        vis_array: ArrayView2<Jones<f32>>,
        weights: ArrayView2<f32>,
        uvws: &[UVW],
        epoch: Epoch,
        num_fine_chans: usize,
        fine_chan_flags: &HashSet<usize>,
    ) -> Result<(), UvfitsWriteError> {
        let num_unflagged_baselines = vis_array.len_of(Axis(0));
        let num_unflagged_tiles = num_tiles_from_num_baselines(num_unflagged_baselines);
        // Write out all the baselines of the timestep we received.
        let mut vis: Vec<f32> = Vec::with_capacity(12 * num_fine_chans);
        for (unflagged_bl, uvw) in (0..num_unflagged_baselines).into_iter().zip(uvws.iter()) {
            // uvfits expects the tile numbers to be from 1 to the total number
            // of tiles sequentially, so don't use the actual unflagged tile
            // numbers.
            let (tile1, tile2) =
                cross_correlation_baseline_to_tiles(num_unflagged_tiles, unflagged_bl);
            let mut unflagged_chan_index = 0;
            for fine_chan_index in 0..num_fine_chans {
                if fine_chan_flags.contains(&fine_chan_index) {
                    vis.extend_from_slice(&[0.0; 12])
                } else {
                    let weight = unsafe { weights.uget((unflagged_bl, unflagged_chan_index)) };
                    let jones = (unsafe { vis_array.uget((unflagged_bl, unflagged_chan_index)) })
                        .clone()
                        // Undo the weight.
                        * (1.0 / weight);
                    unflagged_chan_index += 1;
                    vis.extend_from_slice(&[
                        // XX
                        jones[0].re,
                        jones[0].im,
                        *weight,
                        // YY
                        jones[3].re,
                        jones[3].im,
                        *weight,
                        // XY
                        jones[1].re,
                        jones[1].im,
                        *weight,
                        // YX
                        jones[2].re,
                        jones[2].im,
                        *weight,
                    ]);
                };
            }
            let mut uvfits = self.open()?;
            self.write_vis(&mut uvfits, uvw, tile1, tile2, epoch, &vis)?;
            vis.clear();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::HIFITIME_GPS_FACTOR;
    use tempfile::NamedTempFile;

    #[test]
    // Make a tiny uvfits file. The result has been verified by CASA's
    // "importuvfits" function.
    fn test_new_uvfits_is_sensible() {
        let tmp_uvfits_file = NamedTempFile::new().unwrap();
        let num_timesteps = 1;
        let num_baselines = 3;
        let num_chans = 2;
        let obsid = 1065880128;
        let start_epoch = Epoch::from_tai_seconds(obsid as f64 + 19.0 + HIFITIME_GPS_FACTOR);

        let mut u = UvfitsWriter::new(
            tmp_uvfits_file.path(),
            num_timesteps,
            num_baselines,
            num_chans,
            start_epoch,
            40e3,
            170e6,
            3,
            &RADec::new_degrees(0.0, 60.0),
            Some("test"),
        )
        .unwrap();

        let mut f = u.open().unwrap();
        for _timestep_index in 0..num_timesteps {
            for baseline_index in 0..num_baselines {
                let (tile1, tile2) = match baseline_index {
                    0 => (0, 1),
                    1 => (0, 2),
                    2 => (1, 2),
                    _ => unreachable!(),
                };

                u.write_vis(
                    &mut f,
                    &UVW::default(),
                    tile1,
                    tile2,
                    start_epoch,
                    (baseline_index..baseline_index + num_chans)
                        .into_iter()
                        .map(|int| int as f32)
                        .collect::<Vec<_>>()
                        .as_slice(),
                )
                .unwrap();
            }
        }

        let names = ["Tile1", "Tile2", "Tile3"];
        let positions: Vec<XyzGeodetic> = (0..names.len())
            .into_iter()
            .map(|i| XyzGeodetic {
                x: i as f64,
                y: i as f64 * 2.0,
                z: i as f64 * 3.0,
            })
            .collect();
        u.write_uvfits_antenna_table(&names, &positions).unwrap();
    }
}
