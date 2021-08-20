// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to read in AOFlagger flags in .mwaf files.

use std::collections::BTreeMap;
use std::convert::TryInto;
use std::path::Path;

use log::{debug, error};
use mwa_rust_core::mwalib;
use mwalib::*;

use super::error::*;

/// This monstrosity exists to nicely handle converting any type that can be
/// represented as a `Path` into a string slice. This is kind of a hack, but a
/// necessary one, because cfitsio can't handle UTF-8 characters.
// TODO: Use a result type, you animal.
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
    /// The start time of the observation as described by the mwaf file (GPSTIME).
    start_time_milli: u64,
    /// The number of fine channels as described by the mwaf file (NCHANS).
    num_channels: usize,
    /// The number of antennas as described by the mwaf file (NANTENNA).
    num_antennas: usize,
    /// The number of time steps as described by the mwaf file (NSCANS).
    num_time_steps: usize,
    /// The number of bytes per row in the binary table containing AOF flags
    /// (NAXIS1).
    bytes_per_row: usize,
    /// The number of rows in the binary table containing AOF flags (NAXIS2).
    num_rows: usize,
    /// The AOFlagger flags. The bits are *not* unpacked into individual bytes.
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
    /// not exposed publicly; use `AOFlags::new_from_mwaf` to perform additional
    /// checks on the contents before returning to the caller.
    fn unpack<T: AsRef<Path>>(file: &T) -> Result<Self, FitsError> {
        // Get the metadata written with the flags.
        let s = cfitsio_path_to_str(file);
        debug!("Reading in {}", s);
        let mut fptr = fits_open!(file)?;
        let hdu = fits_open_hdu!(&mut fptr, 0)?;

        // We assume that GPSTIME is the scheduled start time of the
        // observation, and that this is when the flags start.
        let start_time_milli = {
            let start_time: u64 = get_required_fits_key!(&mut fptr, &hdu, "GPSTIME")?;
            start_time * 1000
        };
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
                source_file: file!(),
                source_line: line!(),
            })?;

            flags
        };

        Ok(Self {
            start_time_milli,
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
/// temp struct instead of `AOFlags`, so we can represent a single mwaf file
/// instead of possibly many.
#[derive(Debug)]
struct AOFlagsTemp {
    /// The GPS time of the first scan.
    start_time_milli: u64,
    /// The number of time steps in the data (duration of observation /
    /// integration time).
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
    /// The fractional amount that each channel is flagged.
    ///
    /// Each key is a gpubox number. Each value (which is a vector) has
    /// `num_channels / flags.len()` elements (i.e. the number of fine channels
    /// per coarse band), and each of those elements is between 0 (0% flagged)
    /// and 1 (100% flagged).
    occupancy: BTreeMap<u8, Vec<f32>>,
    /// The gpubox number that these flags apply to (usually between 1 and 24).
    gpubox_num: u8,
    /// The version of cotter used to write the flags.
    cotter_version: String,
    /// The date on which this cotter version was created.
    cotter_version_date: String,
}

#[derive(Debug)]
pub(crate) struct AOFlags {
    /// The GPS time of the first scan \[milliseconds\].
    pub(crate) start_time_milli: u64,

    /// The number of time steps in the data (duration of observation /
    /// integration time).
    pub(crate) num_time_steps: usize,

    /// The total number of fine channels over all coarse bands.
    pub(crate) num_channels: usize,

    /// The number of baselines (auto- and cross-correlation).
    pub(crate) num_baselines: usize,

    /// The visibility flags. These are separated by gpubox number. Flags are
    /// encoded as bits, i.e. 0 for unflagged, 1 for flagged.
    ///
    /// Example: Given a value of 192 (0b11000000), the first and second
    /// visibilities are flagged, and the following six visibilities are
    /// unflagged.
    pub(crate) flags: BTreeMap<u8, Vec<u8>>,

    /// The fractional amount that each channel is flagged.
    ///
    /// Each key is a gpubox number. Each value (which is a vector) has
    /// `num_channels / flags.len()` elements (i.e. the number of fine channels
    /// per coarse band), and each of those elements is between 0 (0% flagged)
    /// and 1 (100% flagged).
    pub(crate) occupancy: BTreeMap<u8, Vec<f32>>,

    /// The gpubox numbers that these flags apply to (usually between 1 and 24).
    /// The values here should be used as keys for `flags`.
    pub(crate) gpubox_nums: Vec<u8>,

    /// The version of cotter used to write the flags.
    pub(crate) cotter_version: String,

    /// The date on which this cotter version was created.
    pub(crate) cotter_version_date: String,
}

impl AOFlags {
    /// Create a `AOFlags` struct from a cotter mwaf file. You should
    /// probably also run the `trim` function on this struct.
    pub(crate) fn new_from_mwaf<T: AsRef<Path>>(file: &T) -> Result<Self, MwafError> {
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

        let mut occupancy = BTreeMap::new();
        occupancy.insert(m.gpubox_num, get_occupancy(&m.flags, m.num_channels));

        let mut flags = BTreeMap::new();
        flags.insert(m.gpubox_num, m.flags);

        Ok(Self {
            start_time_milli: m.start_time_milli,
            num_time_steps: m.num_time_steps,
            num_channels: m.num_channels,
            num_baselines,
            flags,
            occupancy,
            gpubox_nums: vec![m.gpubox_num],
            cotter_version: m.cotter_version,
            cotter_version_date: m.cotter_version_date,
        })
    }

    /// From many mwaf files, return a single `AOFlags` struct with all
    /// flags. You should probably also run the `trim` function on this struct.
    pub(crate) fn new_from_mwafs<T: AsRef<Path>>(files: &[T]) -> Result<Self, MwafMergeError> {
        if files.is_empty() {
            return Err(MwafMergeError::NoFilesGiven);
        }

        let mut unpacked: Vec<AOFlagsTemp> = Vec::with_capacity(files.len());
        for f in files {
            let n = Self::new_from_mwaf(f)?;
            // In an effort to keep things simple and make bad states
            // impossible, use a temp struct to represent the gpubox numbers as
            // a number.
            unpacked.push(AOFlagsTemp {
                start_time_milli: n.start_time_milli,
                gpubox_num: n.gpubox_nums[0],
                num_time_steps: n.num_time_steps,
                num_channels: n.num_channels,
                num_baselines: n.num_baselines,
                flags: n.flags,
                occupancy: n.occupancy,
                cotter_version: n.cotter_version,
                cotter_version_date: n.cotter_version_date,
            })
        }
        Self::merge(unpacked)
    }

    /// Merge several `AOFlags` into a single struct.
    ///
    /// This function is private so it cannot be misused outside this module. If
    /// a user wants to flatten a bunch of mwaf files together, they should use
    /// `AOFlags::new_from_mwafs`.
    fn merge(mut flags: Vec<AOFlagsTemp>) -> Result<Self, MwafMergeError> {
        // Sort by the gpubox number. Because this function is private and only
        // called by `Self::new_from_mwafs`, we can be sure that each of these
        // gpubox_num vectors contains only a single number.
        flags.sort_unstable_by(|a, b| a.gpubox_num.cmp(&b.gpubox_num));

        // Take the last struct from the flags, and use it to compare with
        // everything else. If anything is inconsistent, we blow up.
        let last = flags.pop().unwrap();
        let mut all_flags = BTreeMap::new();
        let mut all_occupancies = BTreeMap::new();
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
            all_occupancies.insert(f.gpubox_num, f.occupancy[&f.gpubox_num].clone());
            num_channels += f.num_channels;
            gpubox_nums.push(f.gpubox_num);
        }

        // Pull out data from the last struct.
        all_flags.insert(last.gpubox_num, last.flags[&last.gpubox_num].clone());
        all_occupancies.insert(last.gpubox_num, last.occupancy[&last.gpubox_num].clone());
        num_channels += last.num_channels;
        gpubox_nums.push(last.gpubox_num);

        Ok(Self {
            start_time_milli: last.start_time_milli,
            num_time_steps: last.num_time_steps,
            num_channels,
            num_baselines: last.num_baselines,
            flags: all_flags,
            occupancy: all_occupancies,
            gpubox_nums,
            cotter_version: last.cotter_version,
            cotter_version_date: last.cotter_version_date,
        })
    }

    /// Trim the cotter flags to match the times that are available in the
    /// mwalib context.
    ///
    /// cotter appears to write flags for all integrations, even if no data was
    /// being collected. This routine discards flags from times that mwalib does
    /// not use (i.e. before OBSID+QUACKTIM and the last common time to all
    /// gpubox files).
    pub(crate) fn trim(&mut self, context: &MetafitsContext) {
        // Don't use "as i64", just in case something goes wrong.
        let to_i64 = |n: u64| -> i64 {
            n.try_into()
                .unwrap_or_else(|_| panic!("Could not convert {} to i64", n))
        };

        let mwalib_start_time_milli =
            to_i64((context.obs_id as u64 * 1000) + context.quack_time_duration_ms);
        let mwalib_duration_milli = to_i64(context.sched_duration_ms);
        let int_time_milli = to_i64(context.corr_int_time_ms);

        let cotter_start_time_milli = to_i64(self.start_time_milli);
        let cotter_duration_milli = to_i64(self.num_time_steps as u64) * int_time_milli;

        let start_trim_milli: i64 = mwalib_start_time_milli - cotter_start_time_milli;
        let num_start_scans_to_trim = (start_trim_milli / int_time_milli) as usize;

        let end_trim_milli = (cotter_start_time_milli + cotter_duration_milli)
            - (mwalib_start_time_milli + mwalib_duration_milli);
        let num_end_scans_to_trim = (end_trim_milli / int_time_milli) as usize;

        // Remove the extraneous flags.
        let step = self.num_baselines * self.num_channels / self.flags.len() / 8;
        for f in self.flags.values_mut() {
            // Remove flags from the start.
            f.drain(..step * num_start_scans_to_trim);
            // Remove flags from the end.
            f.drain(
                step * (self.num_time_steps - num_start_scans_to_trim - num_end_scans_to_trim)..,
            );
        }

        // Update the occupancy of the flags.
        for (&gpubox_num, f) in self.flags.iter() {
            self.occupancy.insert(
                gpubox_num,
                get_occupancy(f, self.num_channels / self.flags.len()),
            );
        }

        self.num_time_steps = self.num_time_steps - num_start_scans_to_trim - num_end_scans_to_trim;
        self.start_time_milli += start_trim_milli as u64;
    }
}

impl std::fmt::Display for AOFlags {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            r#"AOFlags {{
    num_time_steps: {nts},
    num_channels: {nc},
    num_baselines: {nb},
    num_flags: {nf},
    gpubox_nums: {gn:?},
    cotter_version: {cv},
    cotter_version_date: {cvd},
    occupancy: {occ:?}
}}
"#,
            nts = self.num_time_steps,
            nc = self.num_channels,
            nb = self.num_baselines,
            nf = self.flags.len() * 32,
            gn = self.gpubox_nums,
            cv = self.cotter_version,
            cvd = self.cotter_version_date,
            occ = self.occupancy,
        )
    }
}

/// Calculate the fraction that each channel is flagged. `num_channels` is the
/// number of fine channels per coarse band.
fn get_occupancy(flags: &[u8], num_channels: usize) -> Vec<f32> {
    // Collapse the flags into a total number of flags per channel.
    let mut total: Vec<u32> = vec![0; num_channels];
    // The number of bytes to cover all channels. e.g. If we have 32 channels,
    // then width should be 4, as there are 4 flag bytes.
    let width = num_channels / 8;

    // Inspired by Brian Crosse. Add each unique byte to a "histogram" of
    // bytes, then unpack the bits from the bytes.
    let mut histogram: [u32; 256];
    for s in 0..width {
        histogram = [0; 256];
        for f in flags.iter().skip(s).step_by(width) {
            histogram[*f as usize] += 1;
        }
        // Unpack the histogram.
        for (v, h) in histogram.iter().enumerate() {
            for bit in 0..8 {
                if ((v >> bit) & 0x01) == 0x01 {
                    total[7 * (s + 1) + s - bit] += h;
                }
            }
        }
    }

    // Now normalise the totals, so they can be analysed as a fraction.
    let total_samples = (flags.len() / width) as f32;
    total
        .into_iter()
        .map(|t| t as f32 / total_samples)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::*;

    #[test]
    fn test_1065880128_01_mwaf() {
        // The mwaf file is gzipped to save space in git. gunzip it to a
        // temporary spot.
        let mwaf =
            crate::tests::deflate_gz_into_tempfile(&"test_files/1065880128/1065880128_01.mwaf.gz");
        let result = AOFlags::new_from_mwaf(&mwaf);
        assert!(result.is_ok(), "{}", result.unwrap_err());
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

        let expected = vec![
            0.99999946,
            0.99999946,
            0.08406332,
            0.08242058,
            0.0813894,
            0.080897875,
            0.080609664,
            0.08064265,
            0.08067942,
            0.0807097,
            0.08074052,
            0.08076053,
            0.08071998,
            0.08084758,
            0.080910854,
            0.0810601,
            0.99999946,
            0.08099088,
            0.08082109,
            0.080699965,
            0.08065725,
            0.0805026,
            0.08050044,
            0.08046691,
            0.080403104,
            0.08041392,
            0.08044528,
            0.08078865,
            0.08164841,
            0.08251683,
            0.99999946,
            0.99999946,
        ];
        for (&res, &exp) in m.occupancy[&1].iter().zip(expected.iter()) {
            assert_abs_diff_eq!(res, exp);
        }
    }

    #[test]
    fn test_1065880128_02_mwaf() {
        let mwaf =
            crate::tests::deflate_gz_into_tempfile(&"test_files/1065880128/1065880128_02.mwaf.gz");
        let result = AOFlags::new_from_mwaf(&mwaf);
        assert!(result.is_ok(), "{}", result.unwrap_err());
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

        let expected = vec![
            0.99999946,
            0.99999946,
            0.08051342,
            0.07879118,
            0.0775913,
            0.077013254,
            0.07664555,
            0.07659635,
            0.07666232,
            0.07658445,
            0.076587155,
            0.07665853,
            0.07662068,
            0.076593645,
            0.076794796,
            0.0767391,
            0.99999946,
            0.0771652,
            0.07753344,
            0.07830885,
            0.07831534,
            0.07967312,
            0.08014517,
            0.08064157,
            0.08073079,
            0.08076215,
            0.080983855,
            0.08168518,
            0.082425445,
            0.08412713,
            0.99999946,
            0.99999946,
        ];
        for (&res, &exp) in m.occupancy[&2].iter().zip(expected.iter()) {
            assert_abs_diff_eq!(res, exp);
        }
    }

    #[test]
    fn test_merging_1065880128_mwafs() {
        let result = AOFlags::new_from_mwafs(&[
            deflate_gz_into_tempfile(&"test_files/1065880128/1065880128_01.mwaf.gz"),
            deflate_gz_into_tempfile(&"test_files/1065880128/1065880128_02.mwaf.gz"),
        ]);
        assert!(result.is_ok(), "{}", result.unwrap_err());
        let m = result.unwrap();

        assert_eq!(m.num_time_steps, 224);
        assert_eq!(m.num_channels, 64);
        assert_eq!(m.num_baselines, 8256);
        assert_eq!(m.flags[&1].len(), 7397376);
        assert_eq!(m.flags[&2].len(), 7397376);
        assert_eq!(m.gpubox_nums, vec![1, 2]);

        assert_ne!(m.flags[&1], m.flags[&2]);
        assert_ne!(m.occupancy[&1], m.occupancy[&2]);

        assert_eq!(m.flags[&1][4], 192);
        assert_eq!(m.flags[&1][5], 0);
        assert_eq!(m.flags[&1][6], 128);
        assert_eq!(m.flags[&1][7], 3);
        assert_eq!(m.flags[&2][4], 192);
        assert_eq!(m.flags[&2][5], 0);
        assert_eq!(m.flags[&2][6], 128);
        assert_eq!(m.flags[&2][7], 3);
    }

    #[test]
    fn test_trimming_1065880128_mwafs() {
        let mut m = AOFlags::new_from_mwafs(&[
            deflate_gz_into_tempfile(&"test_files/1065880128/1065880128_01.mwaf.gz"),
            deflate_gz_into_tempfile(&"test_files/1065880128/1065880128_02.mwaf.gz"),
        ])
        .unwrap();
        assert_eq!(m.num_time_steps, 224);
        assert_eq!(m.start_time_milli, 1065880128000);
        assert_eq!(m.flags[&1].len(), 7397376);
        assert_eq!(m.flags[&2].len(), 7397376);
        assert_eq!(
            m.flags[&2].len(),
            m.num_baselines * m.num_time_steps * m.num_channels / m.flags.len() / 8
        );

        let expected = vec![
            0.99999946,
            0.99999946,
            0.08051342,
            0.07879118,
            0.0775913,
            0.077013254,
            0.07664555,
            0.07659635,
            0.07666232,
            0.07658445,
            0.076587155,
            0.07665853,
            0.07662068,
            0.076593645,
            0.076794796,
            0.0767391,
            0.99999946,
            0.0771652,
            0.07753344,
            0.07830885,
            0.07831534,
            0.07967312,
            0.08014517,
            0.08064157,
            0.08073079,
            0.08076215,
            0.080983855,
            0.08168518,
            0.082425445,
            0.08412713,
            0.99999946,
            0.99999946,
        ];
        for (&res, &exp) in m.occupancy[&2].iter().zip(expected.iter()) {
            assert_abs_diff_eq!(res, exp);
        }

        let mut c = MetafitsContext::new(
            &"test_files/1065880128/1065880128.metafits",
            Some(MWAVersion::CorrLegacy),
        )
        .unwrap();
        // 1065880128 actually has 109s of data as opposed to the scheduled
        // 112s, but this is impossible to determine without its gpubox files.
        // Because I don't want to include the gpubox files in hyperdrive for
        // testing (at least for all tests), we'll just put this here.
        c.sched_duration_ms = 109000;

        m.trim(&c);

        assert_eq!(m.num_time_steps, 218);
        assert_eq!(m.start_time_milli, 1065880128500);
        assert_eq!(m.flags[&1].len(), 7199232);
        assert_eq!(m.flags[&2].len(), 7199232);
        assert_eq!(
            m.flags[&2].len(),
            m.num_baselines * m.num_time_steps * m.num_channels / m.flags.len() / 8
        );

        let expected = vec![
            1.0,
            1.0,
            0.0595558,
            0.057780053,
            0.05655659,
            0.05596708,
            0.055591486,
            0.055540368,
            0.055607043,
            0.05553148,
            0.055530924,
            0.055604264,
            0.055568706,
            0.05553648,
            0.055742055,
            0.05568594,
            1.0,
            0.056122653,
            0.056501582,
            0.057291113,
            0.057304446,
            0.058702372,
            0.059183534,
            0.059695814,
            0.059788045,
            0.05981249,
            0.060048074,
            0.060764816,
            0.061518785,
            0.06326064,
            1.0,
            1.0,
        ];
        for (&res, &exp) in m.occupancy[&2].iter().zip(expected.iter()) {
            assert_abs_diff_eq!(res, exp);
        }

        // Trimming again shouldn't do anything.
        m.trim(&c);

        assert_eq!(m.num_time_steps, 218);
        assert_eq!(m.start_time_milli, 1065880128500);
        assert_eq!(m.flags[&1].len(), 7199232);
        assert_eq!(m.flags[&2].len(), 7199232);
        assert_eq!(
            m.flags[&2].len(),
            m.num_baselines * m.num_time_steps * m.num_channels / m.flags.len() / 8
        );

        let expected = vec![
            1.0,
            1.0,
            0.0595558,
            0.057780053,
            0.05655659,
            0.05596708,
            0.055591486,
            0.055540368,
            0.055607043,
            0.05553148,
            0.055530924,
            0.055604264,
            0.055568706,
            0.05553648,
            0.055742055,
            0.05568594,
            1.0,
            0.056122653,
            0.056501582,
            0.057291113,
            0.057304446,
            0.058702372,
            0.059183534,
            0.059695814,
            0.059788045,
            0.05981249,
            0.060048074,
            0.060764816,
            0.061518785,
            0.06326064,
            1.0,
            1.0,
        ];
        for (&res, &exp) in m.occupancy[&2].iter().zip(expected.iter()) {
            assert_abs_diff_eq!(res, exp);
        }
    }
}
