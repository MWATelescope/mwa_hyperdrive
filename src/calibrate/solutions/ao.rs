// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to read and write "André Offringa style" calibration solutions.
//!
//! See for more info:
//! https://github.com/MWATelescope/mwa_hyperdrive/wiki/Calibration-solutions

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use byteorder::{ByteOrder, LittleEndian, ReadBytesExt, WriteBytesExt};
use hifitime::{Duration, Epoch};
use marlu::{
    time::{epoch_as_gps_seconds, gps_to_epoch},
    Jones,
};
use ndarray::prelude::*;

use super::{error::*, CalibrationSolutions};
use mwa_hyperdrive_common::{hifitime, marlu, ndarray, Complex};

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
    let total_num_fine_freq_chans = bin_file.read_u32::<LittleEndian>()? as usize;
    let num_polarisations = bin_file.read_u32::<LittleEndian>()? as usize;
    let t = bin_file.read_f64::<LittleEndian>()?;
    let start_time = if t.abs() < f64::EPSILON {
        None
    } else {
        Some(gps_to_epoch(t))
    };
    let t = bin_file.read_f64::<LittleEndian>()?;
    let end_time = if t.abs() < f64::EPSILON {
        None
    } else {
        Some(gps_to_epoch(t))
    };
    let mut di_jones_vec = vec![
        0.0;
        num_timeblocks
            * total_num_tiles
            * total_num_fine_freq_chans
            // Real and imag for each polarisation.
            * 2 * num_polarisations
    ];
    bin_file.read_f64_into::<LittleEndian>(&mut di_jones_vec)?;
    let di_jones_a4 = Array4::from_shape_vec(
        (
            num_timeblocks,
            total_num_tiles,
            total_num_fine_freq_chans,
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

    Ok(CalibrationSolutions {
        di_jones,
        num_timeblocks,
        total_num_tiles,
        total_num_fine_freq_chans,
        // We'd really like to have the *actual* start times of each
        // timeblock, but this isn't possible with this format. Here is the
        // best effort.
        start_timestamps: match (start_time, end_time) {
            (None, None) => vec![],
            (Some(t), None) => vec![t],
            (None, Some(t)) => vec![t],
            (Some(s), Some(e)) => {
                let duration = e - s;
                let average_duration_per_timeblock = duration.in_seconds() / num_timeblocks as f64;
                let mut start_timestamps = vec![];
                for i_timeblock in 0..num_timeblocks {
                    let start: Epoch = s + i_timeblock as f64
                        * Duration::from_f64(
                            average_duration_per_timeblock,
                            hifitime::TimeUnit::Second,
                        );
                    start_timestamps.push(start);
                }
                start_timestamps
            }
        },
        obsid: None,
        time_res: None,
    })
}

/// Write a "André-Offringa calibrate format" calibration solutions binary file.
pub(super) fn write<T: AsRef<Path>>(
    sols: &CalibrationSolutions,
    file: T,
    tile_flags: &HashSet<usize>,
    unflagged_fine_chans: &HashSet<usize>,
) -> Result<(), WriteSolutionsError> {
    let num_polarisations = 4;

    let mut bin_file = BufWriter::new(File::create(file)?);
    // 8 floats, 8 bytes per float.
    let mut buf = [0; 8 * 8];
    bin_file.write_all(b"MWAOCAL")?;
    bin_file.write_u8(0)?;
    bin_file.write_u32::<LittleEndian>(0)?;
    bin_file.write_u32::<LittleEndian>(0)?;
    bin_file.write_u32::<LittleEndian>(sols.num_timeblocks as _)?;
    bin_file.write_u32::<LittleEndian>(sols.total_num_tiles as _)?;
    bin_file.write_u32::<LittleEndian>(sols.total_num_fine_freq_chans as _)?;
    bin_file.write_u32::<LittleEndian>(num_polarisations)?;
    // André indicates that "AIPS time" should be used here. However, it
    // looks like only 0.0 is ever written. I don't know what AIPS time is,
    // and I hate leap seconds, so GPS it is.
    bin_file.write_f64::<LittleEndian>(
        sols.start_timestamps
            .first()
            .map(|&t| epoch_as_gps_seconds(t))
            .unwrap_or(0.0),
    )?;
    bin_file.write_f64::<LittleEndian>(
        sols.start_timestamps
            .last()
            .map(|&t| epoch_as_gps_seconds(t))
            .unwrap_or(0.0),
    )?;

    for di_jones_per_time in sols.di_jones.outer_iter() {
        let mut unflagged_tile_index = 0;
        for tile_index in 0..sols.total_num_tiles {
            let mut unflagged_chan_index = 0;
            for chan in 0..sols.total_num_fine_freq_chans {
                if unflagged_fine_chans.contains(&chan) {
                    // Invert the Jones matrices so that they can be applied as
                    // J D J^H
                    let j = if tile_flags.contains(&tile_index) {
                        // This is a Jones matrix of all NaN.
                        Jones::default().inv()
                    } else {
                        di_jones_per_time[(unflagged_tile_index, unflagged_chan_index)].inv()
                    };

                    LittleEndian::write_f64_into(
                        &[
                            j[0].re, j[0].im, j[1].re, j[1].im, j[2].re, j[2].im, j[3].re, j[3].im,
                        ],
                        &mut buf,
                    );
                    unflagged_chan_index += 1;
                } else {
                    LittleEndian::write_f64_into(&[f64::NAN; 8], &mut buf);
                }
                bin_file.write_all(&buf)?;
            }
            if !tile_flags.contains(&tile_index) {
                unflagged_tile_index += 1;
            };
        }
    }
    bin_file.flush()?;
    Ok(())
}
