// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to handle writing to uvfits files.

use std::collections::HashSet;
use std::ffi::CString;
use std::path::{Path, PathBuf};

use erfa_sys::{ERFA_DJM0, ERFA_WGS84};
use fitsio::{errors::check_status as fits_check_status, FitsFile};
use hifitime::Epoch;
use marlu::{erfa_sys, mwalib, Jones, RADec, XyzGeocentric, XyzGeodetic, UVW};
use mwalib::{fitsio, fitsio_sys};
use ndarray::prelude::*;

use super::*;
use crate::constants::*;

/// A helper struct to write out a uvfits file.
pub(crate) struct UvfitsWriter<'a> {
    /// The path to the uvifts file.
    path: PathBuf,

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

    /// Are autocorrelations accompanying the cross-correlation visibilities?
    autocorrelations_present: bool,

    /// A map from an unflagged cross-correlation baseline to its constituent
    /// antennas.
    unflagged_cross_baseline_to_ants_map: &'a HashMap<usize, (usize, usize)>,

    /// Unflagged tile indices. uvfits must internally refer to tiles in the
    /// antenna tables by their indices, not by MWA tile indices. Ascendingly
    /// sorted.
    unflagged_ants: Vec<usize>,

    /// A map between unflagged antenna indices and uvifts antenna indices.
    unflagged_antenna_map: HashMap<usize, usize>,

    /// A set of channels that are flagged; in the output file, these channels
    /// and will contain only zeroes.
    flagged_channels: &'a HashSet<usize>,
}

impl<'a> UvfitsWriter<'a> {
    /// Create a new uvfits file at the specified filename.
    ///
    /// If `fine_chan_width_hz` is unknown, then zero is written at the FREQ
    /// CDELT.
    pub(crate) fn new<T: AsRef<Path>>(
        filename: T,
        num_timesteps: usize,
        num_baselines: usize,
        num_chans: usize,
        autocorrelations_present: bool,
        start_epoch: Epoch,
        fine_chan_width_hz: Option<f64>,
        centre_freq_hz: f64,
        centre_freq_chan: usize,
        phase_centre: RADec,
        obs_name: Option<&str>,
        unflagged_cross_baseline_to_ants_map: &'a HashMap<usize, (usize, usize)>,
        flagged_channels: &'a HashSet<usize>,
    ) -> Result<UvfitsWriter<'a>, UvfitsWriteError> {
        // Delete any file that already exists.
        if filename.as_ref().exists() {
            std::fs::remove_file(&filename)?;
        }

        // Create a new fits file.
        let mut status = 0;
        let c_filename = CString::new(filename.as_ref().to_str().unwrap())?;
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
        let mut naxes = [0, 3, 4, num_chans as i64, 1, 1];
        let num_group_params = 5;
        let total_num_rows = num_timesteps * num_baselines;
        unsafe {
            fitsio_sys::ffphpr(
                fptr,                  /* I - FITS file pointer                        */
                1,                     /* I - does file conform to FITS standard? 1/0  */
                -32,                   /* I - number of bits per data value pixel      */
                naxes.len() as i32,    /* I - number of axes in the data array         */
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
        hdu.write_key(&mut u, "CDELT4", fine_chan_width_hz.unwrap_or(0.0))?;
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

        // Get the unflagged tile indices; get the unique tiles in the baseline
        // map.
        let mut unflagged_antenna_set = HashSet::new();
        for &(tile1, tile2) in unflagged_cross_baseline_to_ants_map.values() {
            unflagged_antenna_set.insert(tile1);
            unflagged_antenna_set.insert(tile2);
        }
        // Convert the set to a vector and sort it.
        let mut unflagged_ants: Vec<usize> = unflagged_antenna_set.into_iter().collect();
        unflagged_ants.sort_unstable();
        // Make a map between unflagged antennas and uvfits antennas.
        let unflagged_antenna_map: HashMap<usize, usize> = unflagged_ants
            .iter()
            .enumerate()
            .map(|(uvfits_ant, &ant)| (ant, uvfits_ant))
            .collect();

        Ok(Self {
            path: filename.as_ref().to_path_buf(),
            num_timesteps,
            num_baselines,
            num_chans,
            total_num_rows,
            current_num_rows: 0,
            centre_freq: centre_freq_hz,
            start_epoch,
            autocorrelations_present,
            unflagged_cross_baseline_to_ants_map,
            unflagged_ants,
            unflagged_antenna_map,
            flagged_channels,
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
                ERFA_WGS84 as i32,    // ellipsoid identifier (Note 1)
                MWA_LONG_RAD,         // longitude (radians, east +ve)
                MWA_LAT_RAD,          // latitude (geodetic, radians, Note 3)
                MWA_HEIGHT_M,         // height above ellipsoid (geodetic, Notes 2,3)
                mwa_xyz.as_mut_ptr(), // geocentric vector (Note 2)
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
    /// `tile_index1` and `tile_index2` are expected to be zero-indexed; they
    /// are made one-indexed by this function.
    // TODO: Assumes that all fine channels are written in `vis`. This needs to
    // be updated to add visibilities to an existing uvfits row.
    pub(super) fn write_vis(
        &mut self,
        uvfits: &mut FitsFile,
        uvw: UVW,
        tile_index1: usize,
        tile_index2: usize,
        epoch: Epoch,
        row: &mut [f32],
    ) -> Result<(), UvfitsWriteError> {
        if self.current_num_rows + 1 > self.total_num_rows {
            return Err(UvfitsWriteError::BadRowNum {
                row_num: self.current_num_rows,
                num_rows: self.total_num_rows,
            });
        }

        row[0] = (uvw.u / VEL_C) as f32;
        row[1] = (uvw.v / VEL_C) as f32;
        row[2] = (uvw.w / VEL_C) as f32;
        row[3] = encode_uvfits_baseline(tile_index1 + 1, tile_index2 + 1) as f32;
        let jd_trunc = self.start_epoch.as_jde_utc_days().floor() + 0.5;
        let jd_frac = epoch.as_jde_utc_days() - jd_trunc;
        row[4] = jd_frac as f32;

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

    /// Write cross-correlation visibilities contained in `cross_vis` to the
    /// uvfits file. The first axis of `cross_vis` corresponds to baselines, the
    /// second frequencies. This function assumes that visibilities do not
    /// already have weights applied to them.
    // TODO: Assumes that all baselines and fine channels are written for a
    // single timestep.
    pub(crate) fn write_cross_timestep_vis(
        &mut self,
        cross_vis: ArrayView2<Jones<f32>>,
        cross_weights: ArrayView2<f32>,
        uvws: &[UVW],
        epoch: Epoch,
    ) -> Result<(), UvfitsWriteError> {
        let mut uvfits = self.open()?;

        {
            let num_cross_baselines = cross_vis.len_of(Axis(0));
            let num_chans = cross_vis.len_of(Axis(1));
            debug_assert_eq!(num_cross_baselines, uvws.len());
            debug_assert_eq!(num_cross_baselines, self.num_baselines);
            debug_assert_eq!(num_chans, self.num_chans);
        }

        // Write out all the baselines of the timestep we received.
        let mut vis: Vec<f32> = Vec::with_capacity(5 + 12 * self.num_chans);
        // Ignore the first 5 elements; those get overwritten with group
        // parameters.
        vis.extend_from_slice(&[0.0; 5]);
        for (i_bl, uvw) in uvws.iter().enumerate() {
            let (ant1, ant2) = match self.unflagged_cross_baseline_to_ants_map.get(&i_bl) {
                Some(&(ant1, ant2)) => (ant1, ant2),
                None => continue,
            };
            self.unpack_freqs(
                &mut vis,
                cross_vis.index_axis(Axis(0), i_bl),
                cross_weights.index_axis(Axis(0), i_bl),
            );
            // Convert the antenna indices to uvfits antenna indices.
            let uvfits_ant1 = self.unflagged_antenna_map[&ant1];
            let uvfits_ant2 = self.unflagged_antenna_map[&ant2];
            self.write_vis(&mut uvfits, *uvw, uvfits_ant1, uvfits_ant2, epoch, &mut vis)?;
            vis.clear();
            vis.extend_from_slice(&[0.0; 5]);
        }
        Ok(())
    }

    /// Write cross- and auto-correlation visibilities (contained in
    /// `cross_vis` and `auto_vis`, respectively) to the uvfits file. The
    /// first axis of `cross_vis` corresponds to baselines, the second
    /// frequencies. This function assumes that visibilities do not already have
    /// weights applied to them.
    // TODO: Assumes that all baselines and fine channels are written for a
    // single timestep.
    pub(crate) fn write_cross_and_auto_timestep_vis(
        &mut self,
        cross_vis: ArrayView2<Jones<f32>>,
        cross_weights: ArrayView2<f32>,
        auto_vis: ArrayView2<Jones<f32>>,
        auto_weights: ArrayView2<f32>,
        uvws: &[UVW],
        epoch: Epoch,
    ) -> Result<(), UvfitsWriteError> {
        let mut uvfits = self.open()?;

        let num_cross_baselines = cross_vis.len_of(Axis(0));
        let num_chans = cross_vis.len_of(Axis(1));
        let num_antennas = auto_vis.len_of(Axis(0));
        debug_assert_eq!(self.unflagged_ants.len(), num_antennas);
        debug_assert_eq!((num_antennas * (num_antennas - 1)) / 2, num_cross_baselines);
        debug_assert_eq!(num_cross_baselines, uvws.len());
        debug_assert_eq!(num_cross_baselines, self.num_baselines - num_antennas);
        debug_assert_eq!(num_chans, self.num_chans);

        // Write out all the baselines of the timestep we received.
        let mut vis: Vec<f32> = Vec::with_capacity(5 + 12 * self.num_chans);
        // Ignore the first 5 elements; those get overwritten with group
        // parameters.
        vis.extend_from_slice(&[0.0; 5]);
        let mut auto_ant_index = 0;
        for (baseline, uvw) in (0..num_cross_baselines).into_iter().zip(uvws.iter()) {
            let (cross_ant1, cross_ant2) =
                match self.unflagged_cross_baseline_to_ants_map.get(&baseline) {
                    Some(&(ant1, ant2)) => (ant1, ant2),
                    None => continue,
                };

            // Before we write cross-correlations, write out the autos. We know
            // when this needs to be if `cross_ant1` has changed as we iterate
            // over cross-correlation baselines.
            while self.unflagged_ants[auto_ant_index] <= cross_ant1 {
                self.unpack_freqs(
                    &mut vis,
                    auto_vis.index_axis(Axis(0), auto_ant_index),
                    auto_weights.index_axis(Axis(0), auto_ant_index),
                );
                let auto_ant = self.unflagged_ants[auto_ant_index];
                self.write_vis(
                    &mut uvfits,
                    UVW::default(),
                    auto_ant,
                    auto_ant,
                    epoch,
                    &mut vis,
                )?;
                vis.clear();
                vis.extend_from_slice(&[0.0; 5]);
                auto_ant_index += 1;
            }

            // Write the cross-correlation visibilities for this baseline.
            self.unpack_freqs(
                &mut vis,
                cross_vis.index_axis(Axis(0), baseline),
                cross_weights.index_axis(Axis(0), baseline),
            );
            // Convert the cross antenna indices to uvfits antenna indices.
            let uvfits_ant1 = self.unflagged_antenna_map[&cross_ant1];
            let uvfits_ant2 = self.unflagged_antenna_map[&cross_ant2];
            self.write_vis(&mut uvfits, *uvw, uvfits_ant1, uvfits_ant2, epoch, &mut vis)?;
            vis.clear();
            vis.extend_from_slice(&[0.0; 5]);
        }

        // Don't forget the last set of auto-correlations.
        while auto_ant_index < num_antennas {
            self.unpack_freqs(
                &mut vis,
                auto_vis.index_axis(Axis(0), auto_ant_index),
                auto_weights.index_axis(Axis(0), auto_ant_index),
            );
            let auto_tile = self.unflagged_ants[auto_ant_index];
            self.write_vis(
                &mut uvfits,
                UVW::default(),
                auto_tile,
                auto_tile,
                epoch,
                &mut vis,
            )?;
            vis.clear();
            vis.extend_from_slice(&[0.0; 5]);
            auto_ant_index += 1;
        }

        Ok(())
    }

    fn unpack_freqs(
        &self,
        vis: &mut Vec<f32>,
        jones: ArrayView1<Jones<f32>>,
        weights: ArrayView1<f32>,
    ) {
        let mut unflagged_chan_index = 0;
        for fine_chan_index in 0..self.num_chans {
            if self.flagged_channels.contains(&fine_chan_index) {
                vis.extend_from_slice(&[0.0; 12])
            } else {
                let jones = unsafe { jones.uget(unflagged_chan_index) };
                let weight = unsafe { weights.uget(unflagged_chan_index) };
                vis.extend_from_slice(&[
                    // XX
                    jones[0].re,
                    jones[0].im,
                    if jones[0].re.is_nan() { 0.0 } else { *weight },
                    // YY
                    jones[3].re,
                    jones[3].im,
                    if jones[3].re.is_nan() { 0.0 } else { *weight },
                    // XY
                    jones[1].re,
                    jones[1].im,
                    if jones[1].re.is_nan() { 0.0 } else { *weight },
                    // YX
                    jones[2].re,
                    jones[2].im,
                    if jones[2].re.is_nan() { 0.0 } else { *weight },
                ]);
                unflagged_chan_index += 1;
            };
        }
    }
}
