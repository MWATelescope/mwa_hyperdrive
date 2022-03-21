// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to read and write hyperdrive-style calibration solutions.
//!
//! See for more info:
//! <https://mwatelescope.github.io/mwa_hyperdrive/defs/cal_sols_hyp.html>

use std::{
    ffi::CString,
    path::{Path, PathBuf},
};

use hifitime::Epoch;
use marlu::{constants::VEL_C, Jones};
use mwalib::{
    fitsio::{
        errors::check_status as fits_check_status,
        images::{ImageDescription, ImageType},
        tables::{ColumnDataType, ColumnDescription},
        FitsFile,
    },
    *,
};
use ndarray::prelude::*;
use rayon::prelude::*;
use vec1::Vec1;

use super::{error::*, CalibrationSolutions};
use crate::{pfb_gains::PfbFlavour, vis_io::read::RawDataCorrections};
use mwa_hyperdrive_common::{hifitime, marlu, mwalib, ndarray, rayon, vec1, Complex};

pub(super) fn read(file: &Path) -> Result<CalibrationSolutions, SolutionsReadError> {
    let mut fptr = fits_open!(&file)?;
    let hdu = fits_open_hdu!(&mut fptr, 0)?;
    let obsid = get_optional_fits_key!(&mut fptr, &hdu, "OBSID")?;

    let max_iterations: Option<u32> = get_optional_fits_key!(&mut fptr, &hdu, "MAXITER")?;
    let stop_threshold: Option<f64> = get_optional_fits_key!(&mut fptr, &hdu, "S_THRESH")?;
    let min_threshold: Option<f64> = get_optional_fits_key!(&mut fptr, &hdu, "M_THRESH")?;
    let uvw_min: Option<f64> = get_optional_fits_key!(&mut fptr, &hdu, "UVW_MIN")?;
    let uvw_max: Option<f64> = get_optional_fits_key!(&mut fptr, &hdu, "UVW_MAX")?;
    let freq_centroid: Option<f64> = {
        // We need one of the lambda values as well as it's paired cutoff.
        let uvw_min_l: Option<f64> = get_optional_fits_key!(&mut fptr, &hdu, "UVW_MIN_L")?;
        let uvw_max_l: Option<f64> = get_optional_fits_key!(&mut fptr, &hdu, "UVW_MAX_L")?;
        match (uvw_min, uvw_min_l, uvw_max, uvw_max_l) {
            (Some(uvw_min), Some(uvw_min_l), _, _) => {
                let lambda = uvw_min / uvw_min_l;
                Some(VEL_C / lambda)
            }
            (_, _, Some(uvw_max), Some(uvw_max_l)) => {
                let lambda = uvw_max / uvw_max_l;
                Some(VEL_C / lambda)
            }
            _ => None,
        }
    };
    let beam_file: Option<String> =
        get_optional_fits_key_long_string!(&mut fptr, &hdu, "BEAMFILE")?;

    let pfb_flavour: Option<String> = get_optional_fits_key!(&mut fptr, &hdu, "PFB")?;
    let digital_gains: Option<String> = get_optional_fits_key!(&mut fptr, &hdu, "D_GAINS")?;
    let cable_length: Option<String> = get_optional_fits_key!(&mut fptr, &hdu, "CABLELEN")?;
    let geometric: Option<String> = get_optional_fits_key!(&mut fptr, &hdu, "GEOMETRY")?;
    let raw_data_corrections = match (pfb_flavour, digital_gains, cable_length, geometric) {
        (Some(p), Some(d), Some(c), Some(g)) => {
            let pfb_flavour = PfbFlavour::parse(&p)?;
            let digital_gains = matches!(d.as_str(), "Y");
            let cable_length = matches!(c.as_str(), "Y");
            let geometric = matches!(g.as_str(), "Y");
            Some(RawDataCorrections {
                pfb_flavour,
                digital_gains,
                cable_length,
                geometric,
            })
        }
        (None, None, None, None) => None,
        // Some raw data corrections were defined, but not all. Issue a warning?
        _ => None,
    };

    let modeller = get_optional_fits_key!(&mut fptr, &hdu, "MODELLER")?;

    let hdu = fptr.hdu("SOLUTIONS")?;
    let num_timeblocks: usize = get_required_fits_key!(&mut fptr, &hdu, "NAXIS4")?;
    let total_num_tiles: usize = get_required_fits_key!(&mut fptr, &hdu, "NAXIS3")?;
    let total_num_chanblocks: usize = get_required_fits_key!(&mut fptr, &hdu, "NAXIS2")?;
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
            total_num_chanblocks,
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

    // Find any tiles containing only NaNs; these are flagged. Note that we
    // ignore the "Flag" column of the "TILES" HDU; it is possible that there
    // are more flagged tiles in the solutions than what was given to
    // calibration. e.g. An unflagged tile in calibration has all dead dipoles;
    // calibration effectively flags this tile even though we used it in
    // calibration, and this information may not be reported in the "TILES" HDU.
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

    // If available, open the "TIMEBLOCKS" HDU and get the start, end and
    // average times out.
    let (start_timestamps, end_timestamps, average_timestamps) = {
        match fptr.hdu("TIMEBLOCKS") {
            Err(e) => match e {
                // Status code 301 means "unavailable".
                fitsio::errors::Error::Fits(fitsio::errors::FitsError { status: 301, .. }) => {
                    (None, None, None)
                }
                _ => return Err(SolutionsReadError::Fitsio(e)),
            },
            Ok(hdu) => {
                let start = match hdu.read_col(&mut fptr, "Start") {
                    Ok(v) => {
                        // If all values are zeros, we say there are no values.
                        if v.iter().all(|t| *t == 0.0) {
                            None
                        } else {
                            // Complain if the number of timeblocks isn't right.
                            if v.len() != num_timeblocks {
                                return Err(SolutionsReadError::BadShape {
                                    thing: "the start timeblocks in TIMEBLOCKS",
                                    expected: num_timeblocks,
                                    actual: v.len(),
                                });
                            }
                            Some(v)
                        }
                    }
                    Err(e) => match &e {
                        fitsio::errors::Error::Message(m) => match m.as_str() {
                            "Cannot find column \"Start\"" => None,
                            _ => return Err(SolutionsReadError::Fitsio(e)),
                        },
                        _ => return Err(SolutionsReadError::Fitsio(e)),
                    },
                };
                let end: Option<Vec<f64>> = match hdu.read_col(&mut fptr, "End") {
                    Ok(v) => {
                        // If all values are zeros, we say there are no values.
                        if v.iter().all(|t| *t == 0.0) {
                            None
                        } else {
                            // Complain if the number of timeblocks isn't right.
                            if v.len() != num_timeblocks {
                                return Err(SolutionsReadError::BadShape {
                                    thing: "the end timeblocks in TIMEBLOCKS",
                                    expected: num_timeblocks,
                                    actual: v.len(),
                                });
                            }
                            Some(v)
                        }
                    }
                    Err(e) => match &e {
                        fitsio::errors::Error::Message(m) => match m.as_str() {
                            "Cannot find column \"End\"" => None,
                            _ => return Err(SolutionsReadError::Fitsio(e)),
                        },
                        _ => return Err(SolutionsReadError::Fitsio(e)),
                    },
                };
                let average = match hdu.read_col(&mut fptr, "Average") {
                    Ok(v) => {
                        // If all values are zeros, we say there are no values.
                        if v.iter().all(|t| *t == 0.0) {
                            None
                        } else {
                            // Complain if the number of timeblocks isn't right.
                            if v.len() != num_timeblocks {
                                return Err(SolutionsReadError::BadShape {
                                    thing: "the average timeblocks in TIMEBLOCKS",
                                    expected: num_timeblocks,
                                    actual: v.len(),
                                });
                            }
                            Some(v)
                        }
                    }
                    Err(e) => match &e {
                        fitsio::errors::Error::Message(m) => match m.as_str() {
                            "Cannot find column \"Average\"" => None,
                            _ => return Err(SolutionsReadError::Fitsio(e)),
                        },
                        _ => return Err(SolutionsReadError::Fitsio(e)),
                    },
                };
                (
                    start
                        .and_then(|ts| Vec1::try_from_vec(ts).ok())
                        .map(|ts| ts.mapped(Epoch::from_gpst_seconds)),
                    end.and_then(|ts| Vec1::try_from_vec(ts).ok())
                        .map(|ts| ts.mapped(Epoch::from_gpst_seconds)),
                    average
                        .and_then(|ts| Vec1::try_from_vec(ts).ok())
                        .map(|ts| ts.mapped(Epoch::from_gpst_seconds)),
                )
            }
        }
    };

    // If available, open the "CHANBLOCKS" HDU and get the frequencies out.
    let chanblock_freqs = {
        match fptr.hdu("CHANBLOCKS") {
            Ok(hdu) => match hdu.read_col(&mut fptr, "Freq") {
                Ok(v) => {
                    // Complain if the number of frequencies isn't right.
                    if v.len() != total_num_chanblocks {
                        return Err(SolutionsReadError::BadShape {
                            thing: "the number of chanblock frequencies in CHANBLOCKS",
                            expected: total_num_chanblocks,
                            actual: v.len(),
                        });
                    }

                    // If *any* frequencies are NaN, we say there are no
                    // frequencies.
                    if v.iter().any(|f: &f64| f.is_nan()) {
                        None
                    } else {
                        Vec1::try_from_vec(v).ok()
                    }
                }
                Err(e) => match &e {
                    fitsio::errors::Error::Message(m) => match m.as_str() {
                        "Cannot find column \"Freq\"" => None,
                        _ => return Err(SolutionsReadError::Fitsio(e)),
                    },
                    _ => return Err(SolutionsReadError::Fitsio(e)),
                },
            },
            Err(e) => match e {
                // Status code 301 means "unavailable".
                fitsio::errors::Error::Fits(fitsio::errors::FitsError { status: 301, .. }) => None,
                _ => return Err(SolutionsReadError::Fitsio(e)),
            },
        }
    };

    // If available, open the "TILES" HDU and get the tile names out.
    let (tile_names, dipole_gains, dipole_delays) = {
        match fptr.hdu("TILES") {
            Err(e) => match e {
                // Status code 301 means "unavailable".
                fitsio::errors::Error::Fits(fitsio::errors::FitsError { status: 301, .. }) => {
                    (None, None, None)
                }
                _ => return Err(SolutionsReadError::Fitsio(e)),
            },
            Ok(hdu) => {
                let tile_names = match hdu.read_col(&mut fptr, "TileName") {
                    Ok(v) => {
                        // Complain if the number of tiles isn't right.
                        if v.len() != total_num_tiles {
                            return Err(SolutionsReadError::BadShape {
                                thing: "the number of tiles in TILES",
                                expected: total_num_tiles,
                                actual: v.len(),
                            });
                        }

                        Vec1::try_from_vec(v).ok()
                    }
                    Err(e) => match &e {
                        fitsio::errors::Error::Message(m) => match m.as_str() {
                            "Cannot find column \"TileName\"" => None,
                            _ => return Err(SolutionsReadError::Fitsio(e)),
                        },
                        _ => return Err(SolutionsReadError::Fitsio(e)),
                    },
                };

                let dipole_gains = {
                    // It's more effort than I care to expend to read the
                    // array-in-a-column values via fitsio, so I'm using
                    // fitsio-sys.

                    let mut status = 0;
                    let mut array: Array1<f64> = Array1::zeros(32);
                    let mut dipole_gains = Some(Array2::zeros((total_num_tiles, 32)));

                    let i_col = unsafe {
                        let col_name = CString::new("DipoleGains").unwrap().into_raw();
                        let mut i_col = 0;

                        // ffgcno = fits_get_colnum
                        fitsio_sys::ffgcno(
                            fptr.as_raw(), /* I - FITS file pionter                       */
                            1,             /* I - case sensitive string comparison? 0=no  */
                            col_name,      /* I - input name of column (w/wildcards)      */
                            &mut i_col,    /* O - number of the named column; 1=first col */
                            &mut status,   /* IO - error status                           */
                        );
                        let i_col = match status {
                            // 219 = named column not found
                            219 => None,
                            _ => {
                                fits_check_status(status)?;
                                Some(i_col)
                            }
                        };

                        drop(CString::from_raw(col_name));
                        i_col
                    };

                    if let Some(i_col) = i_col {
                        for i_tile in 0..total_num_tiles {
                            unsafe {
                                // ffgcv = fits_read_col
                                fitsio_sys::ffgcv(
                                    fptr.as_raw(),
                                    82, // TDOUBLE (fitsio.h)
                                    i_col,
                                    i_tile as i64 + 1,
                                    1,
                                    32,
                                    std::ptr::null_mut(),
                                    array.as_mut_ptr().cast(),
                                    &mut 0,
                                    &mut status,
                                );
                            }
                            match status {
                                // Status code 301 means "unavailable". 302 is "column number < 1 or > tfields"; in other
                                // words, the column doesn't exist.
                                301 | 302 => {
                                    dipole_gains = None;
                                    break;
                                }
                                _ => fits_check_status(status)?,
                            }

                            dipole_gains
                                .as_mut()
                                .unwrap()
                                .slice_mut(s![i_tile, ..])
                                .assign(&array);
                        }
                    }
                    dipole_gains
                };

                let dipole_delays = {
                    let mut status = 0;
                    let mut array: Array1<u32> = Array1::zeros(16);
                    let mut dipole_delays = Some(Array2::zeros((total_num_tiles, 16)));

                    let i_col = unsafe {
                        let col_name = CString::new("DipoleDelays").unwrap().into_raw();
                        let mut i_col = 0;

                        // ffgcno = fits_get_colnum
                        fitsio_sys::ffgcno(
                            fptr.as_raw(), /* I - FITS file pionter                       */
                            1,             /* I - case sensitive string comparison? 0=no  */
                            col_name,      /* I - input name of column (w/wildcards)      */
                            &mut i_col,    /* O - number of the named column; 1=first col */
                            &mut status,   /* IO - error status                           */
                        );
                        let i_col = match status {
                            // 219 = named column not found
                            219 => None,
                            _ => {
                                fits_check_status(status)?;
                                Some(i_col)
                            }
                        };

                        drop(CString::from_raw(col_name));
                        i_col
                    };

                    if let Some(i_col) = i_col {
                        for i_tile in 0..total_num_tiles {
                            unsafe {
                                // ffgcv = fits_read_col
                                fitsio_sys::ffgcv(
                                    fptr.as_raw(),
                                    30, // TUINT (fitsio.h)
                                    i_col,
                                    i_tile as i64 + 1,
                                    1,
                                    16,
                                    std::ptr::null_mut(),
                                    array.as_mut_ptr().cast(),
                                    &mut 0,
                                    &mut status,
                                );
                            }
                            match status {
                                // Status code 301 means "unavailable". 302 is "column number < 1 or > tfields"; in other
                                // words, the column doesn't exist.
                                301 | 302 => {
                                    dipole_delays = None;
                                    break;
                                }
                                _ => fits_check_status(status)?,
                            }

                            dipole_delays
                                .as_mut()
                                .unwrap()
                                .slice_mut(s![i_tile, ..])
                                .assign(&array);
                        }
                    }
                    dipole_delays
                };

                (
                    tile_names,
                    dipole_gains.map(|a| a.into_shared()),
                    dipole_delays.map(|a| a.into_shared()),
                )
            }
        }
    };

    // If available, open the "RESULTS" HDU and get the calibration precisions
    // out.
    let calibration_results = {
        match fptr.hdu("RESULTS") {
            // If the HDU exists, we assume that it is sensible.
            Ok(hdu) => {
                let n_timeblocks: usize = get_required_fits_key!(&mut fptr, &hdu, "NAXIS2")?;
                let n_chanblocks: usize = get_required_fits_key!(&mut fptr, &hdu, "NAXIS1")?;

                // Complain if the number of timeblocks or chanblocks isn't
                // right.
                if n_timeblocks != num_timeblocks {
                    return Err(SolutionsReadError::BadShape {
                        thing: "the number of timeblocks in RESULTS",
                        expected: num_timeblocks,
                        actual: n_timeblocks,
                    });
                }
                if n_chanblocks != total_num_chanblocks {
                    return Err(SolutionsReadError::BadShape {
                        thing: "the number of chanblocks in RESULTS",
                        expected: total_num_chanblocks,
                        actual: n_chanblocks,
                    });
                }

                let results_vec: Vec<f64> = get_fits_image!(&mut fptr, &hdu)?;
                Some(Array2::from_shape_vec((n_timeblocks, n_chanblocks), results_vec).unwrap())
            }
            Err(e) => match e {
                // Status code 301 means "unavailable".
                fitsio::errors::Error::Fits(fitsio::errors::FitsError { status: 301, .. }) => None,
                _ => return Err(SolutionsReadError::Fitsio(e)),
            },
        }
    };

    let baseline_weights = {
        match fptr.hdu("BASELINES") {
            Ok(hdu) => {
                // Complain if the number of baselines isn't right.
                let num_baselines: usize = get_required_fits_key!(&mut fptr, &hdu, "NAXIS1")?;
                let expected_num_baselines = (total_num_tiles * (total_num_tiles - 1)) / 2;
                if num_baselines != expected_num_baselines {
                    return Err(SolutionsReadError::BadShape {
                        thing: "the number of baselines in BASELINES",
                        expected: expected_num_baselines,
                        actual: num_baselines,
                    });
                }

                Vec1::try_from_vec(get_fits_image!(&mut fptr, &hdu)?).ok()
            }
            Err(e) => match e {
                // Status code 301 means "unavailable".
                fitsio::errors::Error::Fits(fitsio::errors::FitsError { status: 301, .. }) => None,
                _ => return Err(SolutionsReadError::Fitsio(e)),
            },
        }
    };

    Ok(CalibrationSolutions {
        di_jones,
        flagged_tiles,
        flagged_chanblocks,
        chanblock_freqs,
        obsid,
        start_timestamps,
        end_timestamps,
        average_timestamps,
        max_iterations,
        stop_threshold,
        min_threshold,
        raw_data_corrections,
        tile_names,
        dipole_gains,
        dipole_delays,
        beam_file: beam_file.map(PathBuf::from),
        calibration_results,
        baseline_weights,
        uvw_min,
        uvw_max,
        freq_centroid,
        modeller,
    })
}

pub(crate) fn write(sols: &CalibrationSolutions, file: &Path) -> Result<(), SolutionsWriteError> {
    if file.exists() {
        std::fs::remove_file(&file)?;
    }
    let mut fptr = FitsFile::create(&file).open()?;
    let hdu = fits_open_hdu!(&mut fptr, 0)?;
    let mut status = 0;

    let CalibrationSolutions {
        di_jones,
        flagged_tiles,
        flagged_chanblocks,
        chanblock_freqs,
        obsid,
        start_timestamps,
        end_timestamps,
        average_timestamps,
        max_iterations,
        stop_threshold,
        min_threshold,
        raw_data_corrections,
        tile_names,
        dipole_gains,
        dipole_delays,
        beam_file,
        calibration_results,
        baseline_weights,
        uvw_min,
        uvw_max,
        freq_centroid,
        modeller,
    } = sols;

    // Signal that we're using long strings.
    unsafe {
        // ffplsw = fits_write_key_longwarn
        fitsio_sys::ffplsw(
            fptr.as_raw(), /* I - FITS file pointer  */
            &mut status,   /* IO - error status       */
        );
        fits_check_status(status)?;
    }

    // Write the documentation URL as a comment.
    unsafe {
        let comm = CString::new("The contents of this file are documented at:").unwrap();
        // ffpcom = fits_write_comment
        fitsio_sys::ffpcom(fptr.as_raw(), comm.as_ptr(), &mut status);
        fits_check_status(status)?;
        let comm =
            CString::new("https://mwatelescope.github.io/mwa_hyperdrive/defs/cal_sols_hyp.html")
                .unwrap();
        fitsio_sys::ffpcom(fptr.as_raw(), comm.as_ptr(), &mut status);
        fits_check_status(status)?;
    }

    // Write the obsid if we can.
    if let Some(obsid) = obsid {
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

    if let Some(raw_data_corrections) = raw_data_corrections {
        hdu.write_key(
            &mut fptr,
            "PFB",
            raw_data_corrections.pfb_flavour.to_string(),
        )?;
        hdu.write_key(
            &mut fptr,
            "D_GAINS",
            if raw_data_corrections.digital_gains {
                "Y"
            } else {
                "N"
            },
        )?;
        hdu.write_key(
            &mut fptr,
            "CABLELEN",
            if raw_data_corrections.cable_length {
                "Y"
            } else {
                "N"
            },
        )?;
        hdu.write_key(
            &mut fptr,
            "GEOMETRY",
            if raw_data_corrections.geometric {
                "Y"
            } else {
                "N"
            },
        )?;
    }
    if let Some(max_iterations) = max_iterations {
        hdu.write_key(&mut fptr, "MAXITER", *max_iterations)?;
    }
    if let Some(stop_threshold) = stop_threshold {
        hdu.write_key(&mut fptr, "S_THRESH", *stop_threshold)?;
    }
    if let Some(min_threshold) = min_threshold {
        hdu.write_key(&mut fptr, "M_THRESH", *min_threshold)?;
    }
    // UVW cutoffs can be infinite, and cfitsio doesn't know how to convert
    // these to strings...
    if let Some(uvw_min) = uvw_min {
        if uvw_min.is_infinite() {
            hdu.write_key(&mut fptr, "UVW_MIN", "inf")?;
        } else {
            hdu.write_key(&mut fptr, "UVW_MIN", *uvw_min)?;
        }
    }
    if let (Some(uvw_min), Some(freq_centroid)) = (uvw_min, freq_centroid) {
        if uvw_min.is_infinite() {
            hdu.write_key(&mut fptr, "UVW_MIN_L", "inf")?
        } else {
            hdu.write_key(&mut fptr, "UVW_MIN_L", *uvw_min * *freq_centroid / VEL_C)?
        }
    }
    if let Some(uvw_max) = uvw_max {
        if uvw_max.is_infinite() {
            hdu.write_key(&mut fptr, "UVW_MAX", "inf")?;
        } else {
            hdu.write_key(&mut fptr, "UVW_MAX", *uvw_max)?;
        }
    }
    if let (Some(uvw_max), Some(freq_centroid)) = (uvw_max, freq_centroid) {
        if uvw_max.is_infinite() {
            hdu.write_key(&mut fptr, "UVW_MAX_L", "inf")?
        } else {
            hdu.write_key(&mut fptr, "UVW_MAX_L", *uvw_max * *freq_centroid / VEL_C)?
        }
    }
    if let Some(beam_file) = beam_file {
        match CString::new(beam_file.display().to_string()) {
            // This represents failure to convert an argument to UTF-8.
            Err(_) => (),
            Ok(value) => unsafe {
                let key_name = CString::new("BEAMFILE").unwrap();
                let mut status = 0;
                // ffpkls = fits_write_key_longstr
                fitsio_sys::ffpkls(
                    fptr.as_raw(),     /* I - FITS file pointer        */
                    key_name.as_ptr(), /* I - name of keyword to write */
                    value.as_ptr(),    /* I - keyword value            */
                    std::ptr::null(),  /* I - keyword comment          */
                    &mut status,       /* IO - error status            */
                );
                fits_check_status(status)?;
            },
        }
    }

    if let Some(modeller) = modeller {
        hdu.write_key(&mut fptr, "MODELLER", modeller.as_str())?;
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

    // Write out the current command-line call ("CMDLINE").
    unsafe {
        // It's possible that the command-line call has invalid UTF-8. So use
        // args_os and attempt to convert to UTF-8 strings. If there are
        // problems on the way, don't bother trying to write the CMDLINE key.
        match std::env::args_os()
            .map(|a| a.into_string())
            .collect::<Result<Vec<String>, _>>()
            .and_then(|v| CString::new(v.join(" ")).map_err(|_| std::ffi::OsString::from("")))
        {
            // This represents failure to convert an argument to UTF-8.
            Err(_) => (),
            Ok(value) => {
                let key_name = CString::new("CMDLINE").unwrap();
                let comment = CString::new("Command-line call").unwrap();
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
    }

    // Four elements for each Jones matrix, and we need to double the last axis,
    // because we can't write complex numbers directly to FITS files; instead,
    // we write each real and imag float as individual floats.
    let (num_timeblocks, total_num_tiles, total_num_chanblocks) = di_jones.dim();
    let dim = [num_timeblocks, total_num_tiles, total_num_chanblocks, 4 * 2];
    let image_description = ImageDescription {
        data_type: ImageType::Double,
        dimensions: &dim,
    };
    let hdu = fptr.create_image("SOLUTIONS", &image_description)?;
    let mut fits_image_data = vec![f64::NAN; dim.iter().product()];
    let mut one_dim_index = 0;
    for j in di_jones.iter() {
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

    // Write the timeblock information ("TIMEBLOCKS" HDU).
    if start_timestamps.is_some() || end_timestamps.is_some() || average_timestamps.is_some() {
        let start_col = ColumnDescription::new("Start")
            .with_type(ColumnDataType::Double)
            .create()?;
        let end_col = ColumnDescription::new("End")
            .with_type(ColumnDataType::Double)
            .create()?;
        let average_col = ColumnDescription::new("Average")
            .with_type(ColumnDataType::Double)
            .create()?;
        let hdu = match (
            start_timestamps.is_some(),
            end_timestamps.is_some(),
            average_timestamps.is_some(),
        ) {
            (true, true, true) => {
                fptr.create_table("TIMEBLOCKS", &[start_col, end_col, average_col])
            }
            (true, true, false) => fptr.create_table("TIMEBLOCKS", &[start_col, end_col]),
            (true, false, true) => fptr.create_table("TIMEBLOCKS", &[start_col, average_col]),
            (false, true, true) => fptr.create_table("TIMEBLOCKS", &[end_col, average_col]),
            (true, false, false) => fptr.create_table("TIMEBLOCKS", &[start_col]),
            (false, true, false) => fptr.create_table("TIMEBLOCKS", &[end_col]),
            (false, false, true) => fptr.create_table("TIMEBLOCKS", &[average_col]),
            (false, false, false) => unreachable!(),
        }?;

        for (key_name, timestamps) in [
            ("Start", start_timestamps),
            ("End", end_timestamps),
            ("Average", average_timestamps),
        ] {
            if let Some(timestamps) = timestamps {
                hdu.write_col(
                    &mut fptr,
                    key_name,
                    &timestamps.mapped_ref(|e| e.as_gpst_seconds()),
                )?;
            }
        }
    }

    // Write tile information ("TILES" HDU).
    {
        let antenna_num_col = ColumnDescription::new("Antenna")
            .with_type(ColumnDataType::Int)
            .create()?;
        let flag_col = ColumnDescription::new("Flag")
            .with_type(ColumnDataType::Short)
            .create()?;
        let tile_name_col = ColumnDescription::new("TileName")
            .with_type(ColumnDataType::String)
            .that_repeats(8)
            .create()?;
        let dipole_gains_col = ColumnDescription::new("DipoleGains")
            .with_type(ColumnDataType::Double)
            .that_repeats(32)
            .create()?;
        let dipole_delays_col = ColumnDescription::new("DipoleDelays")
            .with_type(ColumnDataType::Int)
            .that_repeats(16)
            .create()?;
        let hdu = match (
            tile_names.is_some(),
            dipole_gains.is_some(),
            dipole_delays.is_some(),
        ) {
            (true, true, true) => fptr.create_table(
                "TILES",
                &[
                    antenna_num_col,
                    flag_col,
                    tile_name_col,
                    dipole_gains_col,
                    dipole_delays_col,
                ],
            )?,
            (true, true, false) => fptr.create_table(
                "TILES",
                &[antenna_num_col, flag_col, tile_name_col, dipole_gains_col],
            )?,
            (true, false, true) => fptr.create_table(
                "TILES",
                &[antenna_num_col, flag_col, tile_name_col, dipole_delays_col],
            )?,
            (false, true, true) => fptr.create_table(
                "TILES",
                &[
                    antenna_num_col,
                    flag_col,
                    dipole_gains_col,
                    dipole_delays_col,
                ],
            )?,
            (true, false, false) => {
                fptr.create_table("TILES", &[antenna_num_col, flag_col, tile_name_col])?
            }
            (false, true, false) => {
                fptr.create_table("TILES", &[antenna_num_col, flag_col, dipole_gains_col])?
            }
            (false, false, true) => {
                fptr.create_table("TILES", &[antenna_num_col, flag_col, dipole_delays_col])?
            }
            (false, false, false) => fptr.create_table("TILES", &[antenna_num_col, flag_col])?,
        };

        hdu.write_col(
            &mut fptr,
            "Antenna",
            &(0..total_num_tiles)
                .into_iter()
                .map(|i| i as u32)
                .collect::<Vec<_>>(),
        )?;
        hdu.write_col(
            &mut fptr,
            "Flag",
            &(0..total_num_tiles)
                .into_iter()
                .map(|i_tile| flagged_tiles.contains(&i_tile).into())
                .collect::<Vec<i32>>(),
        )?;
        if let Some(tile_names) = tile_names {
            assert_eq!(tile_names.len(), total_num_tiles);
            hdu.write_col(&mut fptr, "TileName", tile_names)?;
        };
        if let Some(dipole_gains) = dipole_gains {
            assert_eq!(dipole_gains.len_of(Axis(0)), total_num_tiles);
            assert_eq!(dipole_gains.len_of(Axis(1)), 32);
            hdu.write_col(&mut fptr, "DipoleGains", dipole_gains.as_slice().unwrap())?;
        }
        if let Some(dipole_delays) = dipole_delays {
            assert_eq!(dipole_delays.len_of(Axis(0)), total_num_tiles);
            assert_eq!(dipole_delays.len_of(Axis(1)), 16);
            hdu.write_col(&mut fptr, "DipoleDelays", dipole_delays.as_slice().unwrap())?;
        }
    }

    // Write chanblock information ("CHANBLOCK" HDU).
    {
        let index_col = ColumnDescription::new("Index")
            .with_type(ColumnDataType::Int)
            .create()?;
        let flag_col = ColumnDescription::new("Flag")
            .with_type(ColumnDataType::Bool)
            .create()?;
        let freq_col = ColumnDescription::new("Freq")
            .with_type(ColumnDataType::Double)
            .create()?;
        let hdu = fptr.create_table("CHANBLOCKS", &[index_col, flag_col, freq_col])?;
        hdu.write_col(
            &mut fptr,
            "Index",
            &(0..di_jones.len_of(Axis(2)))
                .into_iter()
                .map(|i_cb| i_cb as u32)
                .collect::<Vec<u32>>(),
        )?;
        hdu.write_col(
            &mut fptr,
            "Flag",
            &(0..di_jones.len_of(Axis(2)))
                .into_iter()
                .map(|i_cb| flagged_chanblocks.contains(&(i_cb as u16)).into())
                .collect::<Vec<i32>>(),
        )?;
        if let Some(chanblock_freqs) = chanblock_freqs {
            hdu.write_col(&mut fptr, "Freq", chanblock_freqs)?;
        } else {
            hdu.write_col(&mut fptr, "Freq", &vec![f64::NAN; di_jones.len_of(Axis(2))])?;
        }
    }

    // Write calibration result information ("RESULTS" HDU).
    if let Some(chanblock_results) = calibration_results {
        let (num_timeblocks, num_chanblocks) = chanblock_results.dim();
        let dim = [num_timeblocks, num_chanblocks];
        let image_description = ImageDescription {
            data_type: ImageType::Double,
            dimensions: &dim,
        };
        let hdu = fptr.create_image("RESULTS", &image_description)?;
        // Fill the fits image with NaN before overwriting with our precisions.
        let mut fits_image_data = vec![f64::NAN; dim.iter().product()];
        chanblock_results
            .iter()
            .zip(fits_image_data.iter_mut())
            .for_each(|(precision, fits_image_elem)| {
                *fits_image_elem = *precision;
            });
        hdu.write_image(&mut fptr, &fits_image_data)?;
    }

    // Write baseline information ("BASELINES" HDU).
    if let Some(baseline_weights) = baseline_weights {
        let image_description = ImageDescription {
            data_type: ImageType::Double,
            dimensions: &[baseline_weights.len()],
        };
        let hdu = fptr.create_image("BASELINES", &image_description)?;
        hdu.write_image(&mut fptr, baseline_weights)?;
    }

    Ok(())
}
