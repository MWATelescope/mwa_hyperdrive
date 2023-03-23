use fitsio::{
    hdu::{FitsHdu, HduInfo},
    FitsFile,
};

use crate::io::read::fits::FitsError;

/// Open a fits file for reading.
#[track_caller]
pub(crate) fn fits_edit<P: AsRef<std::path::Path>>(file: P) -> Result<FitsFile, FitsError> {
    FitsFile::edit(file.as_ref()).map_err(|e| {
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

/// Given a FITS file pointer and a HDU, write the image.
#[track_caller]
pub fn fits_write_image<T: fitsio::images::WriteImage>(
    fits_fptr: &mut FitsFile,
    hdu: &FitsHdu,
    data: &[T],
) -> Result<(), FitsError> {
    match &hdu.info {
        HduInfo::ImageInfo { .. } => hdu.write_image(fits_fptr, data).map_err(|e| {
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
