// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Code to read in cotter flags in .mwaf files.
 */

use std::collections::BTreeMap;
use std::path::Path;

use super::errors::*;
use crate::*;

/// This monstrosity exists to nicely handle converting any type that can be
/// represented as a `Path` into a string slice. This is kind of a hack, but a
/// necessary one, because cfitsio can't handle UTF-8 characters.
fn cfitsio_path_to_str<T: AsRef<Path>>(filename: &T) -> &str {
    match filename.as_ref().to_str() {
        None => {
            error!("Filename contained invalid UTF-8, and cannot be used.");
            std::process::exit(1);
        }
        Some(s) => s,
    }
}

struct Mwaf {
    /// The number of channels as described by the mwaf file (NCHANS).
    num_channels: usize,
    /// The number of antennas as described by the mwaf file (NANTENNA).
    num_antennas: usize,
    /// The number of time steps as described by the mwaf file (NSCANS).
    num_time_steps: usize,
    /// The number of bytes per row in the binary table containing cotter flags
    /// (NAXIS1).
    bytes_per_row: usize,
    /// The number of rows in the binary table containing cotter flags (NAXIS2).
    num_rows: usize,
    /// The cotter flags. The bits are *not* unpacked into individual bytes.
    ///
    /// Example: Given a value of 192 (0b11000000), the first and second
    /// visibilities are flagged, and the following six visibilities are
    /// unflagged.
    ///
    /// The flags are listed in the "sensible" baseline order, e.g. ant1 ->
    /// ant1, ant1 -> ant2, etc. Time is the slowest axis, then baseline, then
    /// frequency.
    flags: Vec<u8>,
    /// The gpubox number that these flags apply to.
    gpubox_num: u8,
    /// The version of cotter used to write the flags.
    cotter_version: String,
    /// The date on which this cotter version was created.
    cotter_version_date: String,
}

impl Mwaf {
    /// A helper function to unpack and parse the contents of an mwaf file. It is
    /// not exposed publicly; use `CotterFlags::new_from_mwaf` to perform additional
    /// checks on the contents before returning to the caller.
    fn unpack<T: AsRef<Path>>(file: &T) -> Result<Self, FitsError> {
        // Get the metadata written with the flags.
        let s = cfitsio_path_to_str(file);
        debug!("Reading in {}", s);
        let mut fptr = fits_open!(file)?;
        let hdu = fits_open_hdu!(&mut fptr, 0)?;

        let num_channels = get_required_fits_key!(&mut fptr, &hdu, "NCHANS")?;
        let num_antennas = get_required_fits_key!(&mut fptr, &hdu, "NANTENNA")?;
        let num_time_steps = get_required_fits_key!(&mut fptr, &hdu, "NSCANS")?;
        let num_baselines = (num_antennas * (num_antennas + 1)) / 2;
        let gpubox_num = get_required_fits_key!(&mut fptr, &hdu, "GPUBOXNO")?;
        let cotter_version = get_required_fits_key!(&mut fptr, &hdu, "COTVER")?;
        let cotter_version_date = get_required_fits_key!(&mut fptr, &hdu, "COTVDATE")?;

        let hdu = fits_open_hdu!(&mut fptr, 1)?;
        let bytes_per_row = get_required_fits_key!(&mut fptr, &hdu, "NAXIS1")?;
        let num_rows = get_required_fits_key!(&mut fptr, &hdu, "NAXIS2")?;

        // Visibility flags are encoded as bits. rust-fitsio currently doesn't
        // read this data in correctly, so use cfitsio via fitsio-sys.
        debug!("Reading the FLAGS column in {}", s);
        let flags = {
            let mut flags: Vec<u8> = vec![0; num_baselines * num_time_steps * bytes_per_row];
            let mut status = 0;
            unsafe {
                fitsio_sys::ffgcvb(
                    fptr.as_raw(),      /* I - FITS file pointer                       */
                    1,                  /* I - number of column to read (1 = 1st col)  */
                    1,                  /* I - first row to read (1 = 1st row)         */
                    1,                  /* I - first vector element to read (1 = 1st)  */
                    flags.len() as i64, /* I - number of values to read                */
                    0,                  /* I - value for null pixels                   */
                    flags.as_mut_ptr(), /* O - array of values that are read           */
                    &mut 0,             /* O - set to 1 if any values are null; else 0 */
                    &mut status,        /* IO - error status                           */
                );
            }
            fitsio::errors::check_status(status).map_err(|e| FitsError::Fitsio {
                fits_error: e,
                fits_filename: s.to_string(),
                hdu_num: 1,
                source_file: file!().to_string(),
                source_line: line!(),
            })?;

            flags
        };

        Ok(Self {
            num_channels,
            num_antennas,
            num_time_steps,
            bytes_per_row,
            num_rows,
            flags,
            gpubox_num,
            cotter_version,
            cotter_version_date,
        })
    }
}

/// In an effort to keep things simple and make bad states impossible, use a
/// temp struct instead of `CotterFlags`, so we can represent a single mwaf file
/// instead of possibly many.
#[derive(Debug)]
struct CotterFlagsTemp {
    num_time_steps: usize,
    /// The number of fine channels per coarse channel.
    num_channels: usize,
    /// The number of baselines (auto- and cross-correlation).
    num_baselines: usize,
    /// The visibility flags. Flags are encoded as bits, i.e. 0 for unflagged, 1 for flagged.
    ///
    /// Example: Given a value of 192 (0b11000000), the first and second
    /// visibilities are flagged, and the following six visibilities are
    /// unflagged.
    flags: BTreeMap<u8, Vec<u8>>,
    /// The gpubox number that these flags apply to (usually between 1 and 24).
    gpubox_num: u8,
    /// The version of cotter used to write the flags.
    cotter_version: String,
    /// The date on which this cotter version was created.
    cotter_version_date: String,
}

#[derive(Debug)]
pub struct CotterFlags {
    pub num_time_steps: usize,
    /// The total number of fine channels over all coarse bands.
    pub num_channels: usize,
    /// The number of baselines (auto- and cross-correlation).
    pub num_baselines: usize,
    /// The visibility flags. These are separated by gpubox number. Flags are
    /// encoded as bits, i.e. 0 for unflagged, 1 for flagged.
    ///
    /// Example: Given a value of 192 (0b11000000), the first and second
    /// visibilities are flagged, and the following six visibilities are
    /// unflagged.
    pub flags: BTreeMap<u8, Vec<u8>>,
    /// The gpubox numbers that these flags apply to (usually between 1 and 24).
    /// The values here should be used as keys for `flags`.
    pub gpubox_nums: Vec<u8>,
    /// The version of cotter used to write the flags.
    pub cotter_version: String,
    /// The date on which this cotter version was created.
    pub cotter_version_date: String,
}

impl CotterFlags {
    pub fn new_from_mwaf<T: AsRef<Path>>(file: &T) -> Result<Self, MwafError> {
        let m = Mwaf::unpack(file)?;

        // Check that things are consistent.
        let num_baselines = m.num_antennas * (m.num_antennas + 1) / 2;

        if m.num_rows != m.num_time_steps * num_baselines {
            return Err(MwafError::Inconsistent {
                file: cfitsio_path_to_str(file).to_string(),
                expected: "NSCANS * NANTENNA * (NANTENNA+1) / 2 = NAXIS2".to_string(),
                found: format!("{} * {} = {}", m.num_time_steps, num_baselines, m.num_rows),
            });
        }

        if m.bytes_per_row * m.num_rows != m.flags.len() {
            return Err(MwafError::Inconsistent {
                file: cfitsio_path_to_str(file).to_string(),
                expected: "NAXIS1 * NAXIS2 = number of flags read".to_string(),
                found: format!("{} * {} = {}", m.bytes_per_row, m.num_rows, m.flags.len()),
            });
        }

        let mut flags = BTreeMap::new();
        flags.insert(m.gpubox_num, m.flags);
        Ok(Self {
            num_time_steps: m.num_time_steps,
            num_channels: m.num_channels,
            num_baselines,
            flags,
            gpubox_nums: vec![m.gpubox_num],
            cotter_version: m.cotter_version,
            cotter_version_date: m.cotter_version_date,
        })
    }

    /// From many mwaf files, return a single `CotterFlags` struct with all
    /// flags. The mwaf files are sorted by their internal gpubox file numbers,
    /// to (hopefully) get the flags in the struct's vector in the correct
    /// order.
    pub fn new_from_mwafs<T: AsRef<Path>>(files: &[T]) -> Result<Self, MwafMergeError> {
        if files.is_empty() {
            return Err(MwafMergeError::NoFilesGiven);
        }

        let mut unpacked = vec![];
        for f in files {
            let n = Self::new_from_mwaf(f)?;
            // In an effort to keep things simple and make bad states
            // impossible, use a temp struct to represent the gpubox numbers as
            // a number.
            unpacked.push(CotterFlagsTemp {
                gpubox_num: n.gpubox_nums[0],
                num_time_steps: n.num_time_steps,
                num_channels: n.num_channels,
                num_baselines: n.num_baselines,
                flags: n.flags,
                cotter_version: n.cotter_version,
                cotter_version_date: n.cotter_version_date,
            });
        }
        Ok(Self::merge(unpacked)?)
    }

    /// Merge several `CotterFlags` together. The structs are sorted by their
    /// internal gpubox file numbers, to (hopefully) get the flags in the
    /// struct's vector in the correct order.
    ///
    /// This function is private so it cannot be misused outside this module. If
    /// a user wants to flatten a bunch of mwaf files together, they should use
    /// `CotterFlags::new_from_mwafs`.
    fn merge(mut flags: Vec<CotterFlagsTemp>) -> Result<Self, MwafMergeError> {
        // Sort by the gpubox number. Because this function is private and only
        // called by `Self::new_from_mwafs`, we can be sure that each of these
        // gpubox_num vectors contains only a single number.
        flags.sort_unstable_by(|a, b| a.gpubox_num.cmp(&b.gpubox_num));

        // Take the last struct from the flags, and use it to compare with
        // everything else. If anything is inconsistent, we blow up.
        let last = flags.pop().unwrap();
        let mut all_flags = BTreeMap::new();
        let mut num_channels = 0;
        let mut gpubox_nums = Vec::with_capacity(flags.len());

        for f in flags.into_iter() {
            if f.num_time_steps != last.num_time_steps {
                return Err(MwafMergeError::Inconsistent {
                    gpubox1: f.gpubox_num,
                    gpubox2: last.gpubox_num,
                    expected: format!("Num. time steps = {}", f.num_time_steps),
                    found: format!("Num. time steps = {}", last.num_time_steps),
                });
            }

            if f.num_channels != last.num_channels {
                return Err(MwafMergeError::Inconsistent {
                    gpubox1: f.gpubox_num,
                    gpubox2: last.gpubox_num,
                    expected: format!("Num. channels = {}", f.num_channels),
                    found: format!("Num. channels = {}", last.num_channels),
                });
            }

            if f.num_baselines != last.num_baselines {
                return Err(MwafMergeError::Inconsistent {
                    gpubox1: f.gpubox_num,
                    gpubox2: last.gpubox_num,
                    expected: format!("Num. baselines = {}", f.num_baselines),
                    found: format!("Num. baselines = {}", last.num_baselines),
                });
            }

            if f.cotter_version != last.cotter_version {
                return Err(MwafMergeError::Inconsistent {
                    gpubox1: f.gpubox_num,
                    gpubox2: last.gpubox_num,
                    expected: format!("cotter version = {}", f.cotter_version),
                    found: format!("cotter version = {}", last.cotter_version),
                });
            }

            if f.cotter_version_date != last.cotter_version_date {
                return Err(MwafMergeError::Inconsistent {
                    gpubox1: f.gpubox_num,
                    gpubox2: last.gpubox_num,
                    expected: format!("cotter version date = {}", f.cotter_version_date),
                    found: format!("cotter version date = {}", last.cotter_version_date),
                });
            }

            if f.flags.len() != last.flags.len() {
                return Err(MwafMergeError::Inconsistent {
                    gpubox1: f.gpubox_num,
                    gpubox2: last.gpubox_num,
                    expected: format!("flags.len() = {}", f.flags.len()),
                    found: format!("flags.len() = {}", last.flags.len()),
                });
            }

            // Pull out the data from f and amalgamate it.
            all_flags.insert(f.gpubox_num, f.flags[&f.gpubox_num].clone());
            num_channels += f.num_channels;
            gpubox_nums.push(f.gpubox_num);
        }

        // Pull out data from the last struct.
        all_flags.insert(last.gpubox_num, last.flags[&last.gpubox_num].clone());
        num_channels += last.num_channels;
        gpubox_nums.push(last.gpubox_num);

        Ok(Self {
            num_time_steps: last.num_time_steps,
            num_channels,
            num_baselines: last.num_baselines,
            flags: all_flags,
            gpubox_nums,
            cotter_version: last.cotter_version,
            cotter_version_date: last.cotter_version_date,
        })
    }
}

impl std::fmt::Display for CotterFlags {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            r#"CotterFlags {{
    num_time_steps: {nts},
    num_channels: {nc},
    num_baselines: {nb},
    num_flags: {nf},
    cotter_version: {cv},
    cotter_version_date: {cvd},
}}
"#,
            nts = self.num_time_steps,
            nc = self.num_channels,
            nb = self.num_baselines,
            nf = self.flags.len() * 32,
            cv = self.cotter_version,
            cvd = self.cotter_version_date
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn deflate_gz<T: AsRef<Path>>(file: &T) -> NamedTempFile {
        let mut temp = NamedTempFile::new().unwrap();
        let mut gz = flate2::read::GzDecoder::new(std::fs::File::open(file).unwrap());
        std::io::copy(&mut gz, &mut temp).unwrap();
        temp
    }

    #[test]
    fn test_1065880128_01_mwaf() {
        // The mwaf file is gzipped to save space in git. gunzip it to a
        // temporary spot.
        let mwaf = deflate_gz(&"tests/1065880128_01.mwaf.gz");
        let result = CotterFlags::new_from_mwaf(&mwaf);
        assert!(result.is_ok());
        let m = result.unwrap();

        assert_eq!(m.num_time_steps, 224);
        assert_eq!(m.num_channels, 32);
        assert_eq!(m.num_baselines, 8256);
        assert_eq!(m.flags[&1].len(), 7397376);
        assert_eq!(m.gpubox_nums, vec![1]);

        assert_eq!(m.flags[&1][0], 0);
        assert_eq!(m.flags[&1][1], 0);
        assert_eq!(m.flags[&1][2], 0);
        assert_eq!(m.flags[&1][3], 0);
        // These are the first two channels, middle channel, and last two
        // channels flagged. 11000000 00000000 10000000 00000011
        assert_eq!(m.flags[&1][4], 192);
        assert_eq!(m.flags[&1][5], 0);
        assert_eq!(m.flags[&1][6], 128);
        assert_eq!(m.flags[&1][7], 3);

        // Add every third channel. There are 32 channels, and the flags are
        // stored as bytes (8 flags per byte), so we only want the first byte.
        let mut chan_occupancy = 0;
        for (i, &f) in m.flags[&1].iter().enumerate() {
            if i % 4 == 0 {
                chan_occupancy += ((f & 0x20) / 0x20) as u32;
            }
        }
        assert_eq!(chan_occupancy, 155462);
    }

    #[test]
    fn test_1065880128_02_mwaf() {
        let mwaf = deflate_gz(&"tests/1065880128_02.mwaf.gz");
        let result = CotterFlags::new_from_mwaf(&mwaf);
        assert!(result.is_ok());
        let m = result.unwrap();

        assert_eq!(m.num_time_steps, 224);
        assert_eq!(m.num_channels, 32);
        assert_eq!(m.num_baselines, 8256);
        assert_eq!(m.flags[&2].len(), 7397376);
        assert_eq!(m.gpubox_nums, vec![2]);

        assert_eq!(m.flags[&2][0], 0);
        assert_eq!(m.flags[&2][1], 0);
        assert_eq!(m.flags[&2][2], 0);
        assert_eq!(m.flags[&2][3], 0);
        assert_eq!(m.flags[&2][4], 192);
        assert_eq!(m.flags[&2][5], 0);
        assert_eq!(m.flags[&2][6], 128);
        assert_eq!(m.flags[&2][7], 3);

        let mut chan_occupancy = 0;
        for (i, &f) in m.flags[&2].iter().enumerate() {
            if i % 4 == 0 {
                chan_occupancy += ((f & 0x20) / 0x20) as u32;
            }
        }
        assert_eq!(chan_occupancy, 148897);
    }

    #[test]
    fn test_merging_1065880128_mwafs() {
        let result = CotterFlags::new_from_mwafs(&[
            deflate_gz(&"tests/1065880128_01.mwaf.gz"),
            deflate_gz(&"tests/1065880128_02.mwaf.gz"),
        ]);
        assert!(result.is_ok());
        let m = result.unwrap();

        assert_eq!(m.num_time_steps, 224);
        assert_eq!(m.num_channels, 64);
        assert_eq!(m.num_baselines, 8256);
        assert_eq!(m.flags[&1].len(), 7397376);
        assert_eq!(m.flags[&2].len(), 7397376);
        assert_eq!(m.gpubox_nums, vec![1, 2]);

        assert_ne!(m.flags[&1], m.flags[&2]);

        assert_eq!(m.flags[&1][4], 192);
        assert_eq!(m.flags[&1][5], 0);
        assert_eq!(m.flags[&1][6], 128);
        assert_eq!(m.flags[&1][7], 3);
        assert_eq!(m.flags[&2][4], 192);
        assert_eq!(m.flags[&2][5], 0);
        assert_eq!(m.flags[&2][6], 128);
        assert_eq!(m.flags[&2][7], 3);
    }
}
