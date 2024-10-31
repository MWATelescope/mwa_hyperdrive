use fitsio::FitsFile;

use crate::io::read::fits::{fits_get_required_key, FitsError};

// check that the filename and key appear in the error struct
#[test]
fn test_fits_get_required_key() {
    let output = tempfile::NamedTempFile::new().unwrap();

    // create a fits file with only header A
    {
        let mut fitsfile = FitsFile::create(output.path()).overwrite().open().unwrap();
        let hdu = fitsfile.primary_hdu().unwrap();
        hdu.write_key(&mut fitsfile, "A", "1").unwrap();
    }

    let mut fitsfile = FitsFile::edit(output.path()).unwrap();
    let hdu = fitsfile.primary_hdu().unwrap();

    // read header B from the fits file (doesn't exist)
    let missing: Result<String, FitsError> = fits_get_required_key(&mut fitsfile, &hdu, "B");

    assert!(
        matches!(missing, Err(FitsError::MissingKey { key, fits_filename, .. }) if key == Box::from("B") && fits_filename == Box::from(output.path()))
    );
}
