// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Code to read in cotter flags in .mwaf files.
 */

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
    /// The number of antennas as described by the mwaf file.
    num_channels: usize,
    /// The number of antennas as described by the mwaf file.
    num_ants: usize,
    /// The number of antennas as described by the mwaf file.
    num_time_steps: usize,
    /// The number of bytes per row in the binary table containing cotter
    /// flags. AKA NAXIS1.
    bytes_per_row: usize,
    /// The number of rows in the binary table containing cotter flags. AKA
    /// NAXIS2.
    num_rows: usize,
    /// The cotter flags. The bits are *not* unpacked into individual bytes.
    ///
    /// Example: Given a value of 192 (0b11000000), the first and second
    /// visibilities are flagged, and the following six visibilities are
    /// unflagged.
    flags: Vec<u8>,
    /// The gpubox number(s) that these flags apply to.
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
        let num_ants = get_required_fits_key!(&mut fptr, &hdu, "NANTENNA")?;
        let num_time_steps = get_required_fits_key!(&mut fptr, &hdu, "NSCANS")?;
        let gpubox_num = get_required_fits_key!(&mut fptr, &hdu, "GPUBOXNO")?;
        let cotter_version = get_required_fits_key!(&mut fptr, &hdu, "COTVER")?;
        let cotter_version_date = get_required_fits_key!(&mut fptr, &hdu, "COTVDATE")?;

        let hdu = fits_open_hdu!(&mut fptr, 1)?;
        let bytes_per_row = get_required_fits_key!(&mut fptr, &hdu, "NAXIS1")?;
        let num_rows = get_required_fits_key!(&mut fptr, &hdu, "NAXIS2")?;

        // Visibility flags are encoded as bits. rust-fitsio only allows columns to
        // be read in bigger types (such as u32, but not u8), even though the values
        // will fit fine into u8s. So, read the column as u32, but promptly convert
        // it to u8.
        debug!("Reading the FLAGS column in {}", s);
        let flags: Vec<u8> = {
            let f: Vec<u32> = get_fits_col!(&mut fptr, &hdu, "FLAGS")?;
            f.into_iter().map(|u| u as u8).collect()
        };

        Ok(Self {
            num_channels,
            num_ants,
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
    flags: Vec<u8>,
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
    /// The total number of fine channels.
    pub num_channels: usize,
    /// The number of baselines (auto- and cross-correlation).
    pub num_baselines: usize,
    /// The visibility flags. Flags are encoded as bits, i.e. 0 for unflagged, 1 for flagged.
    ///
    /// Example: Given a value of 192 (0b11000000), the first and second
    /// visibilities are flagged, and the following six visibilities are
    /// unflagged.
    pub flags: Vec<u8>,
    /// The gpubox numbers that these flags apply to (usually between 1 and 24).
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
        let num_baselines = m.num_ants * (m.num_ants + 1) / 2;

        // Because we're using u32, there are 4x *fewer* bytes per row.
        if 4 * m.num_rows != m.bytes_per_row * m.num_time_steps * num_baselines {
            return Err(MwafError::Inconsistent {
                file: cfitsio_path_to_str(file).to_string(),
                expected: "NAXIS1 * NSCANS * NANTENNA * (NANTENNA+1) / 2 = NAXIS2".to_string(),
                found: format!(
                    "{} * {} * {} = {}",
                    m.bytes_per_row, m.num_time_steps, num_baselines, m.num_rows
                ),
            });
        }

        if m.bytes_per_row * m.num_rows / 4 != m.flags.len() {
            return Err(MwafError::Inconsistent {
                file: cfitsio_path_to_str(file).to_string(),
                expected: "NAXIS1 * NAXIS2 / 4 = number of flags read".to_string(),
                found: format!(
                    "{} * {} / 4 = {}",
                    m.bytes_per_row,
                    m.num_rows,
                    m.flags.len()
                ),
            });
        }

        Ok(Self {
            num_time_steps: m.num_time_steps,
            num_channels: m.num_channels,
            num_baselines,
            flags: m.flags,
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
        let mut last = flags.pop().unwrap();
        let mut all_flags: Vec<_> = Vec::with_capacity(flags.len() * last.flags.len());
        let mut num_channels = 0;
        let mut gpubox_nums = Vec::with_capacity(flags.len());

        for mut f in &mut flags.drain(..) {
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
            all_flags.append(&mut f.flags);
            num_channels += f.num_channels;
            gpubox_nums.push(f.gpubox_num);
        }

        // Pull out data from the last struct.
        all_flags.append(&mut last.flags);
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
