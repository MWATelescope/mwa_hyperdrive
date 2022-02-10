// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to read and write "André Offringa style" calibration solutions.
//!
//! See for more info:
//! https://github.com/MWATelescope/mwa_hyperdrive/wiki/Calibration-solutions

use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use byteorder::{ByteOrder, LittleEndian, ReadBytesExt, WriteBytesExt};
use hifitime::Epoch;
use marlu::Jones;
use ndarray::prelude::*;
use rayon::prelude::*;

use super::{error::*, CalibrationSolutions};
use crate::math::average_epoch;
use mwa_hyperdrive_common::{hifitime, marlu, ndarray, rayon, Complex};

pub(super) fn read<T: AsRef<Path>>(file: T) -> Result<CalibrationSolutions, ReadSolutionsError> {
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
    let total_num_chanblocks = bin_file.read_u32::<LittleEndian>()? as usize;
    let num_polarisations = bin_file.read_u32::<LittleEndian>()? as usize;
    // If the start time (read in here as `t`) is 0, then we don't really have a
    // start time!
    let t = bin_file.read_f64::<LittleEndian>()?;
    let start_time = if t.abs() < f64::EPSILON {
        None
    } else {
        Some(Epoch::from_gpst_seconds(t))
    };
    // And similarly for the end time.
    let t = bin_file.read_f64::<LittleEndian>()?;
    let end_time = if t.abs() < f64::EPSILON {
        None
    } else {
        Some(Epoch::from_gpst_seconds(t))
    };

    // The rest of the binary is only Jones matrices.
    let mut di_jones_a4 = Array4::zeros((
        num_timeblocks,
        total_num_tiles,
        total_num_chanblocks,
        2 * num_polarisations,
    ));
    bin_file.read_f64_into::<LittleEndian>(di_jones_a4.as_slice_mut().unwrap())?;
    let di_jones = di_jones_a4.map_axis(Axis(3), |view| {
        Jones::from([
            Complex::new(view[0], view[1]),
            Complex::new(view[2], view[3]),
            Complex::new(view[4], view[5]),
            Complex::new(view[6], view[7]),
        ])
    });

    // Find any tiles containing only NaNs; these are flagged.
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

    // We'd really like to have the *actual* timestamps of each timeblock,
    // but this isn't possible with this format. Here is the best effort.
    let (start_timestamps, end_timestamps, average_timestamps) = match (start_time, end_time) {
        (Some(s), Some(e)) => (vec![s], vec![e], vec![average_epoch(&[s, e])]),
        (Some(s), None) => (vec![s], vec![], vec![]),
        (None, Some(e)) => (vec![], vec![e], vec![]),
        (None, None) => (vec![], vec![], vec![]),
    };

    Ok(CalibrationSolutions {
        di_jones,
        flagged_tiles,
        flagged_chanblocks,
        average_timestamps,
        start_timestamps,
        end_timestamps,
        obsid: None,
    })
}

/// Write a "André-Offringa calibrate format" calibration solutions binary file.
pub(super) fn write<T: AsRef<Path>>(
    sols: &CalibrationSolutions,
    file: T,
) -> Result<(), WriteSolutionsError> {
    let num_polarisations = 4;
    let (num_timeblocks, total_num_tiles, total_num_chanblocks) = sols.di_jones.dim();

    let mut bin_file = BufWriter::new(File::create(file)?);
    // 8 floats, 8 bytes per float.
    let mut buf = [0; 8 * 8];
    bin_file.write_all(b"MWAOCAL")?;
    bin_file.write_u8(0)?;
    bin_file.write_u32::<LittleEndian>(0)?;
    bin_file.write_u32::<LittleEndian>(0)?;
    bin_file.write_u32::<LittleEndian>(num_timeblocks as _)?;
    bin_file.write_u32::<LittleEndian>(total_num_tiles as _)?;
    bin_file.write_u32::<LittleEndian>(total_num_chanblocks as _)?;
    bin_file.write_u32::<LittleEndian>(num_polarisations)?;
    // André indicates that "AIPS time" should be used for the "start" and "end"
    // times here. However, it looks like only 0.0 is ever written by
    // mwa-reduce's calibrate. I don't know what AIPS time is, and I hate leap
    // seconds, so GPS it is.
    let (start, end) = match (
        sols.start_timestamps.as_slice(),
        sols.end_timestamps.as_slice(),
        sols.average_timestamps.as_slice(),
    ) {
        // One start and end time.
        ([s, ..], [.., e], _) => (s.as_gpst_seconds(), e.as_gpst_seconds()),
        // No start and end times, but averages. Sure, why not.
        ([], [], [a0, .., an]) => (a0.as_gpst_seconds(), an.as_gpst_seconds()),
        ([], [], [a0]) => (a0.as_gpst_seconds(), a0.as_gpst_seconds()),
        // Less-than-ideal amount of info.
        ([s, ..], [], _) => (s.as_gpst_seconds(), 0.0),
        ([], [.., e], _) => (0.0, e.as_gpst_seconds()),
        // Nothing, nothing.
        ([], [], []) => (0.0, 0.0),
    };
    bin_file.write_f64::<LittleEndian>(start)?;
    bin_file.write_f64::<LittleEndian>(end)?;

    for j in sols.di_jones.iter() {
        LittleEndian::write_f64_into(
            &[
                j[0].re, j[0].im, j[1].re, j[1].im, j[2].re, j[2].im, j[3].re, j[3].im,
            ],
            &mut buf,
        );
        bin_file.write_all(&buf)?;
    }
    bin_file.flush()?;
    Ok(())
}
