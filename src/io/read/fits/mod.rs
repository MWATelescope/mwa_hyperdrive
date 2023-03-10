// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Helper functions for reading FITS files.

mod error;

pub(crate) use error::FitsError;

use std::{
    ffi::{CStr, CString},
    fmt::Display,
    ptr,
};

use fitsio::{hdu::*, FitsFile};

/// Open a fits file.
#[track_caller]
pub(crate) fn fits_open<P: AsRef<std::path::Path>>(file: P) -> Result<FitsFile, FitsError> {
    FitsFile::open(file.as_ref()).map_err(|e| {
        let caller = std::panic::Location::caller();
        FitsError::Open {
            fits_error: Box::new(e),
            fits_filename: file.as_ref().to_path_buf().into_boxed_path(),
            source_file: caller.file(),
            source_line: caller.line(),
            source_column: caller.column(),
        }
    })
}

/// Open a fits file's HDU.
#[track_caller]
pub(crate) fn fits_open_hdu<T: DescribesHdu + Display + Copy>(
    fits_fptr: &mut FitsFile,
    hdu_description: T,
) -> Result<FitsHdu, FitsError> {
    fits_fptr.hdu(hdu_description).map_err(|e| {
        let caller = std::panic::Location::caller();
        FitsError::Fitsio {
            fits_error: Box::new(e),
            fits_filename: fits_fptr.filename.clone().into_boxed_path(),
            hdu_description: format!("{hdu_description}").into_boxed_str(),
            source_file: caller.file(),
            source_line: caller.line(),
            source_column: caller.column(),
        }
    })
}

/// Given a FITS file pointer, a HDU that belongs to it, and a keyword that may
/// or may not exist, pull out the value of the keyword, parsing it into the
/// desired type.
#[track_caller]
pub(crate) fn fits_get_optional_key<T: std::str::FromStr>(
    fits_fptr: &mut FitsFile,
    hdu: &FitsHdu,
    keyword: &str,
) -> Result<Option<T>, FitsError> {
    let unparsed_value: String = match hdu.read_key(fits_fptr, keyword) {
        Ok(key_value) => key_value,
        Err(e) => match &e {
            fitsio::errors::Error::Fits(fe) => match fe.status {
                202 | 204 => return Ok(None),
                _ => {
                    let caller = std::panic::Location::caller();
                    return Err(FitsError::Fitsio {
                        fits_error: Box::new(e),
                        fits_filename: fits_fptr.filename.clone().into_boxed_path(),
                        hdu_description: format!("{}", hdu.number + 1).into_boxed_str(),
                        source_file: caller.file(),
                        source_line: caller.line(),
                        source_column: caller.column(),
                    });
                }
            },
            _ => {
                let caller = std::panic::Location::caller();
                return Err(FitsError::Fitsio {
                    fits_error: Box::new(e),
                    fits_filename: fits_fptr.filename.clone().into_boxed_path(),
                    hdu_description: format!("{}", hdu.number + 1).into_boxed_str(),
                    source_file: caller.file(),
                    source_line: caller.line(),
                    source_column: caller.column(),
                });
            }
        },
    };

    match unparsed_value.parse() {
        Ok(parsed_value) => Ok(Some(parsed_value)),
        Err(_) => {
            let caller = std::panic::Location::caller();
            Err(FitsError::Parse {
                key: keyword.to_string().into_boxed_str(),
                fits_filename: fits_fptr.filename.clone().into_boxed_path(),
                hdu_num: hdu.number + 1,
                source_file: caller.file(),
                source_line: caller.line(),
                source_column: caller.column(),
            })
        }
    }
}

/// Given a FITS file pointer, a HDU that belongs to it, and a keyword, pull out
/// the value of the keyword, parsing it into the desired type.
#[track_caller]
pub(crate) fn fits_get_required_key<T: std::str::FromStr>(
    fits_fptr: &mut FitsFile,
    hdu: &FitsHdu,
    keyword: &str,
) -> Result<T, FitsError> {
    match fits_get_optional_key(fits_fptr, hdu, keyword) {
        Ok(Some(value)) => Ok(value),
        Ok(None) => {
            let caller = std::panic::Location::caller();
            Err(FitsError::MissingKey {
                key: keyword.to_string().into_boxed_str(),
                fits_filename: fits_fptr.filename.clone().into_boxed_path(),
                hdu_num: hdu.number + 1,
                source_file: caller.file(),
                source_line: caller.line(),
                source_column: caller.column(),
            })
        }
        Err(error) => Err(error),
    }
}

/// Get a column from a fits file's HDU.
#[track_caller]
pub(crate) fn fits_get_col<T: fitsio::tables::ReadsCol>(
    fits_fptr: &mut FitsFile,
    hdu: &FitsHdu,
    keyword: &str,
) -> Result<Vec<T>, FitsError> {
    hdu.read_col(fits_fptr, keyword).map_err(|e| {
        let caller = std::panic::Location::caller();
        FitsError::Fitsio {
            fits_error: Box::new(e),
            fits_filename: fits_fptr.filename.clone().into_boxed_path(),
            hdu_description: format!("{}", hdu.number + 1).into_boxed_str(),
            source_file: caller.file(),
            source_line: caller.line(),
            source_column: caller.column(),
        }
    })
}

/// Given a FITS file pointer, and a keyword to a long string keyword that may
/// or may not exist, pull out the long string of the keyword. This deals with
/// FITSs CONTINUE mechanism by calling a low level fits function.
#[track_caller]
pub(crate) fn fits_get_optional_key_long_string(
    fits_fptr: &mut fitsio::FitsFile,
    hdu: &FitsHdu,
    keyword: &str,
) -> Result<Option<String>, FitsError> {
    // Read the long string.
    let keyword_ffi = CString::new(keyword)
        .expect("fits_get_optional_fits_long_string: CString::new() failed for keyword");
    let long_string = unsafe {
        let mut status = 0;
        let mut long_string_ptr = ptr::null_mut();
        // ffgkls = fits_read_key_longstr
        fitsio_sys::ffgkls(
            fits_fptr.as_raw(),
            keyword_ffi.as_ptr(),
            &mut long_string_ptr,
            ptr::null_mut(),
            &mut status,
        );
        // Check the call worked!
        match status {
            0 => {
                let long_string = CStr::from_ptr(long_string_ptr)
                    .to_str()
                    .expect("fits_get_optional_key_long_string: reading C string as UTF-8 failed")
                    .to_string();
                // Free the cfitsio-allocated string. The status code passed
                // isn't useful (have a look at the source if you don't believe
                // me!)
                // fffree = fits_free_memory
                fitsio_sys::fffree(long_string_ptr.cast(), &mut 0);
                Some(long_string)
            }
            202 | 204 => None,
            _ => {
                let caller = std::panic::Location::caller();
                return Err(FitsError::LongString {
                    key: keyword.to_string().into_boxed_str(),
                    fits_filename: fits_fptr.filename.clone().into_boxed_path(),
                    hdu_num: hdu.number + 1,
                    source_file: caller.file(),
                    source_line: caller.line(),
                    source_column: caller.column(),
                });
            }
        }
    };

    Ok(long_string)
}

/// Given a FITS file pointer, and a keyword to a long string keyword, pull out
/// the long string of the keyword. This deals with FITSs CONTINUE mechanism by
/// calling a low level fits function.
#[track_caller]
pub(crate) fn _fits_get_required_key_long_string(
    fits_fptr: &mut FitsFile,
    hdu: &FitsHdu,
    keyword: &str,
) -> Result<String, FitsError> {
    match fits_get_optional_key_long_string(fits_fptr, hdu, keyword) {
        Ok(Some(value)) => Ok(value),
        Ok(None) => {
            let caller = std::panic::Location::caller();
            Err(FitsError::MissingKey {
                key: keyword.to_string().into_boxed_str(),
                fits_filename: fits_fptr.filename.clone().into_boxed_path(),
                hdu_num: hdu.number + 1,
                source_file: caller.file(),
                source_line: caller.line(),
                source_column: caller.column(),
            })
        }
        Err(error) => Err(error),
    }
}

/// Get the size of the image on the supplied FITS file pointer and HDU.
#[track_caller]
pub(crate) fn _fits_get_image_size<'a>(
    fits_fptr: &FitsFile,
    hdu: &'a FitsHdu,
) -> Result<&'a Vec<usize>, FitsError> {
    match &hdu.info {
        HduInfo::ImageInfo { shape, .. } => Ok(shape),
        _ => {
            let caller = std::panic::Location::caller();
            Err(FitsError::NotImage {
                fits_filename: fits_fptr.filename.clone().into_boxed_path(),
                hdu_num: hdu.number + 1,
                source_file: caller.file(),
                source_line: caller.line(),
                source_column: caller.column(),
            })
        }
    }
}

/// Given a FITS file pointer and a HDU, read the associated image.
#[track_caller]
pub(crate) fn fits_get_image<T: fitsio::images::ReadImage>(
    fits_fptr: &mut FitsFile,
    hdu: &FitsHdu,
) -> Result<T, FitsError> {
    match &hdu.info {
        HduInfo::ImageInfo { .. } => hdu.read_image(fits_fptr).map_err(|e| {
            let caller = std::panic::Location::caller();
            FitsError::Fitsio {
                fits_error: Box::new(e),
                fits_filename: fits_fptr.filename.clone().into_boxed_path(),
                hdu_description: format!("{}", hdu.number + 1).into_boxed_str(),
                source_file: caller.file(),
                source_line: caller.line(),
                source_column: caller.column(),
            }
        }),
        _ => {
            let caller = std::panic::Location::caller();
            Err(FitsError::NotImage {
                fits_filename: fits_fptr.filename.clone().into_boxed_path(),
                hdu_num: hdu.number + 1,
                source_file: caller.file(),
                source_line: caller.line(),
                source_column: caller.column(),
            })
        }
    }
}

/// Given a FITS file pointer and a HDU, read the associated float image.
#[track_caller]
pub(crate) fn _fits_get_float_image_into_buffer(
    fits_fptr: &mut FitsFile,
    hdu: &FitsHdu,
    buffer: &mut [f32],
) -> Result<(), FitsError> {
    unsafe {
        // Get raw ptr and length to user supplied buffer
        let buffer_len = buffer.len() as i64;
        let buffer_ptr = buffer.as_mut_ptr();

        // Call the underlying cfitsio read function for floats
        let mut status = 0;
        // ffgpv = fits_read_img
        fitsio_sys::ffgpv(
            fits_fptr.as_raw(),
            fitsio_sys::TFLOAT as _,
            1,
            buffer_len,
            ptr::null_mut(),
            buffer_ptr as *mut _,
            ptr::null_mut(),
            &mut status,
        );

        // Check fits call status
        match fitsio::errors::check_status(status) {
            Ok(_) => {}
            Err(e) => {
                let caller = std::panic::Location::caller();
                return Err(FitsError::Fitsio {
                    fits_error: Box::new(e),
                    fits_filename: fits_fptr.filename.clone().into_boxed_path(),
                    hdu_description: format!("{}", hdu.number + 1).into_boxed_str(),
                    source_file: caller.file(),
                    source_line: caller.line(),
                    source_column: caller.column(),
                });
            }
        }
    }

    Ok(())
}
