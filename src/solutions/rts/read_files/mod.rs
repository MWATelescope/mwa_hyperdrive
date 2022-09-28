// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to read and write RTS calibration solutions.

// Anakin: This is where the fun begins...
//
// We must make a lot of assumptions about the structure of the files here,
// because this "format" is laughably insane.
//
// RTS calibration solutions are stored in BandpassCalibration_node???.dat and
// DI_JonesMatrices_node???.dat files, one of each of these files for each
// coarse channel used in the DI run (typically 24), starting from node001. We
// assume that each, for the given directory `dir`, these belong to the same DI
// run.

#[cfg(test)]
mod tests;

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use marlu::Jones;
use ndarray::prelude::*;
use num_complex::Complex;
use thiserror::Error;
use vec1::Vec1;

// DI_JonesMatrices_node???.dat files are structured by lines as follows:
// 1) A single float for the "alignment flux density",
// 2) A Jones matrix for the "post-alignment matrix",
// 3-end) Each line is a Jones matrix representing a "pre-alignment matrix".
// Each of these Jones matrices corresponds to a tile.
//
// The Jones matrices are ", " separated floats and 8 floats are expected on
// each line (4 complex number pairs). It appears that there is a Jones matrix
// for flagged tiles, too.

/// RTS "DI Jones matrices".
///
/// More info: https://cira-pulsars-and-transients-group.github.io/vcsbeam/calibration.html
#[derive(Debug)]
pub(super) struct DiJm {
    /// The brightness of the calibrator source ("base source") used by the RTS.
    /// Not sure if it's ever used.
    pub(super) _alignment_flux_density: f64,

    /// The beam response toward the calibrator source ("base source") used by
    /// the RTS. When reading in these, it is inverted and applied to the
    /// pre-alignment matrices.
    pub(super) post_alignment_matrix: Jones<f64>,

    /// The length of this vector is the total number of tiles in the
    /// observation (flagged and unflagged).
    pub(super) pre_alignment_matrices: Vec1<Jones<f64>>,
}

impl DiJm {
    pub(super) fn read_file(path: &Path) -> Result<DiJm, ReadDiJmFileError> {
        let mut file = BufReader::new(File::open(path)?);
        Self::read(&mut file)
    }

    fn read<T: std::io::BufRead>(file: &mut T) -> Result<DiJm, ReadDiJmFileError> {
        let mut alignment_flux_density = 0.0;
        let mut post_alignment_matrix = Jones::default();
        let mut pre_alignment_matrices: Vec<Jones<f64>> = Vec::with_capacity(128);
        for (i_line, line) in file.lines().enumerate() {
            let line = line?;

            match i_line {
                0 => {
                    alignment_flux_density =
                        line.parse::<f64>()
                            .map_err(|_| ReadDiJmFileError::ParseFloat {
                                text: line.to_string(),
                                line_num: 1,
                            })?
                }

                1 => {
                    let mut num_floats = 0;
                    let mut post_alignment_matrix_floats = [0.0; 8];
                    for (float_str, post) in line
                        .split(", ")
                        .zip(post_alignment_matrix_floats.iter_mut())
                    {
                        num_floats += 1;
                        let float = float_str.parse::<f64>().map_err(|_| {
                            ReadDiJmFileError::ParseFloat {
                                text: float_str.to_string(),
                                line_num: i_line + 1,
                            }
                        })?;
                        *post = float;
                    }

                    if num_floats != 8 {
                        return Err(ReadDiJmFileError::BadFloatCount {
                            count: num_floats,
                            line_num: i_line + 1,
                        });
                    }
                    post_alignment_matrix = Jones::from([
                        post_alignment_matrix_floats[0],
                        post_alignment_matrix_floats[1],
                        post_alignment_matrix_floats[2],
                        post_alignment_matrix_floats[3],
                        post_alignment_matrix_floats[4],
                        post_alignment_matrix_floats[5],
                        post_alignment_matrix_floats[6],
                        post_alignment_matrix_floats[7],
                    ]);
                }

                _ => {
                    let mut pre_alignment_matrix_floats = [0.0; 8];
                    let mut num_floats = 0;
                    for (float_str, pre) in
                        line.split(", ").zip(pre_alignment_matrix_floats.iter_mut())
                    {
                        num_floats += 1;
                        let float = float_str.parse::<f64>().map_err(|_| {
                            ReadDiJmFileError::ParseFloat {
                                text: float_str.to_string(),
                                line_num: i_line + 1,
                            }
                        })?;
                        *pre = float;
                    }

                    if num_floats != 8 {
                        return Err(ReadDiJmFileError::BadFloatCount {
                            count: num_floats,
                            line_num: i_line + 1,
                        });
                    }
                    let pre_alignment_matrix = Jones::from([
                        pre_alignment_matrix_floats[0],
                        pre_alignment_matrix_floats[1],
                        pre_alignment_matrix_floats[2],
                        pre_alignment_matrix_floats[3],
                        pre_alignment_matrix_floats[4],
                        pre_alignment_matrix_floats[5],
                        pre_alignment_matrix_floats[6],
                        pre_alignment_matrix_floats[7],
                    ]);

                    pre_alignment_matrices.push(pre_alignment_matrix);
                }
            }
        }

        Ok(DiJm {
            _alignment_flux_density: alignment_flux_density,
            post_alignment_matrix,
            pre_alignment_matrices: Vec1::try_from_vec(pre_alignment_matrices)
                .map_err(|_| ReadDiJmFileError::NoPreAlignmentMatrices)?,
        })
    }
}

#[derive(Error, Debug)]
pub(crate) enum ReadDiJmFileError {
    #[error("Couldn't parse '{text}' as a float on line {line_num}")]
    ParseFloat { text: String, line_num: usize },

    #[error("Expected to find 8 floats on line {line_num}, but found {count} instead")]
    BadFloatCount { count: usize, line_num: usize },

    #[error("Not enough lines in the file to define any pre-alignment matrices")]
    NoPreAlignmentMatrices,

    #[error("IO error: {0}")]
    IO(#[from] std::io::Error),
}

// The first line of the BandpassCalibration files is a ", " separated list of
// unflagged fine channels, e.g.
//
// 0.080000, 0.120000, 0.160000, 0.200000, 0.240000, 0.280000, 0.320000, ...
//
// (but on one line). In this example, there are 7 (visible) unflagged fine
// channels. We assume that the smallest gap between any pair of numbers
// corresponds to the fine-frequency channel resolution in MHz. With that
// information, we can work out which channels are flagged (in the above
// example, 0 and 1 are flagged).
//
// The following lines then start with a tile number (starting from 1) and have
// N pairs of numbers following it (N being the number of unflagged channels),
// e.g.
//
// 1, +1.015465,+0.047708, +1.006940,+0.005763, +1.024352,+0.015657, ...
//
// (but on one line). Each pair corresponds to (amp, phase). There will be 8
// lines per tile; they represent:
// 1) PX_lsq
// 2) PX_fit
// 3) PY_lsq
// 4) PY_fit
// 5) QX_lsq
// 6) QX_fit
// 7) QY_lsq
// 8) QY_fit
//
// "lsq" lines are "measured values", whereas "fit" lines are "fitted values".
// Apparently only the measured data is interesting.

/// RTS "Bandpass calibration" Jones matrices.
///
/// More info: https://cira-pulsars-and-transients-group.github.io/vcsbeam/calibration.html
#[derive(Debug)]
pub(super) struct BpCal {
    /// The smallest gap between consecutive channels in this file \[Hz\]. This
    /// is `None` if there's only one unflagged channel.
    pub(super) fine_channel_resolution: Option<f64>,

    /// The unflagged fine channel indices for the coarse channel that this file
    /// corresponds to. Zero indexed.
    pub(super) unflagged_fine_channel_indices: Vec1<usize>,

    /// The unflagged RF input indices (*divided by 2*) for the coarse channel
    /// that this file corresponds to. Only one RF input is included per tile,
    /// so when there are 0 flagged tiles in a 128-tile observation, this vector
    /// has 128 elements. Zero indexed.
    pub(super) unflagged_rf_input_indices: Vec1<u8>,

    /// All of the Jones matrices in the file. The 1st dimension is per tile,
    /// the 2nd dimension is for lsq and fit (in that order) and always has a
    /// length of 2, and the 3rd dimension is a Jones matrix for each
    /// fine-frequency channel.
    pub(super) data: Array3<Jones<f64>>,
}

impl BpCal {
    pub(super) fn read_file(path: &Path) -> Result<BpCal, ReadBpCalFileError> {
        let mut file = BufReader::new(File::open(path)?);
        Self::read(&mut file)
    }

    /// We assume that the first line of the file is ascendingly sorted.
    fn read<T: std::io::BufRead>(file: &mut T) -> Result<BpCal, ReadBpCalFileError> {
        // Read the entire file into a string. We do this because we iterate over it
        // multiple times.
        let file = {
            let mut str = String::new();
            file.read_to_string(&mut str)?;
            str
        };

        // Go through the entire file to count how many tiles are there. Also do
        // some sanity checks.
        let num_tiles = {
            let line_count = file.lines().count();
            if line_count == 0 {
                return Err(ReadBpCalFileError::Empty);
            }
            if (line_count - 1) % 8 != 0 {
                return Err(ReadBpCalFileError::UnexpectedLineCount { got: line_count });
            }

            let num_tiles = (line_count - 1) / 8;
            if num_tiles == 0 {
                return Err(ReadBpCalFileError::NoTiles);
            }

            num_tiles
        };

        // Just read the first line to get the unflagged channels.
        let mut unflagged_fine_chan_freqs = vec![];
        if let Some(line) = file.lines().next() {
            for float_str in line.split(", ") {
                // Parse the float.
                let float =
                    float_str
                        .parse::<f64>()
                        .map_err(|_| ReadBpCalFileError::ParseFloat {
                            text: float_str.to_string(),
                            line_num: 1,
                        })?;
                // Convert to Hz and round. If there should be a fraction of a
                // Hz in the value, you have my sympathy.
                unflagged_fine_chan_freqs.push((float * 1e6).round());
            }
        }
        if unflagged_fine_chan_freqs.is_empty() {
            return Err(ReadBpCalFileError::NoUnflaggedChans);
        }

        let mut unflagged_rf_input_indices = Vec::with_capacity(128);
        let mut data: Array3<Jones<f64>> =
            Array3::zeros((num_tiles, 2, unflagged_fine_chan_freqs.len()));
        let mut i_tile = 0;
        for (i_line, line) in file.lines().skip(1).enumerate() {
            let mut elems = line.split(',').flat_map(|s| s.split_whitespace());
            let input_num_str = elems.next().ok_or(ReadBpCalFileError::NoTileNum {
                line_num: i_line + 2,
            })?;
            if i_line > 0 && i_line % 8 == 0 {
                i_tile += 1;
            }

            let input_num =
                input_num_str
                    .parse::<u8>()
                    .map_err(|_| ReadBpCalFileError::ParseInt {
                        text: input_num_str.to_string(),
                        line_num: i_line + 2,
                    })?
                    - 1; // convert to zero indexed
            if !unflagged_rf_input_indices.contains(&input_num) {
                unflagged_rf_input_indices.push(input_num);
            }

            let lsq_or_fit = i_line % 2;
            let px_py_qx_qy = (i_line % 8) / 2;

            let mut first = true;
            let mut pair = (0.0, 0.0);
            for (i_float, elem) in elems.enumerate() {
                let float = elem
                    .parse::<f64>()
                    .map_err(|_| ReadBpCalFileError::ParseFloat {
                        text: elem.to_string(),
                        line_num: i_line + 2,
                    })?;

                if first {
                    pair.0 = float;
                    first = false;
                } else {
                    pair.1 = float;
                    first = true;

                    data[(i_tile, lsq_or_fit, i_float / 2)][px_py_qx_qy] =
                        Complex::from_polar(pair.0, pair.1);
                }
            }
        }
        let unflagged_rf_input_indices = Vec1::try_from_vec(unflagged_rf_input_indices)
            .map_err(|_| ReadBpCalFileError::NoTiles)?;

        let fine_channel_resolution = if unflagged_fine_chan_freqs.len() == 1 {
            None
        } else {
            let smallest_gap =
                unflagged_fine_chan_freqs
                    .windows(2)
                    .fold(f64::INFINITY, |acc, pair| {
                        let diff = pair[1] - pair[0];
                        if diff < acc {
                            diff
                        } else {
                            acc
                        }
                    });
            Some(smallest_gap)
        };

        let unflagged_fine_channel_indices = match fine_channel_resolution {
            Some(res) => unflagged_fine_chan_freqs
                .iter()
                .map(|&f| {
                    let big_int = (f / res).round() as u32;
                    big_int.try_into().unwrap()
                })
                .collect(),

            None => vec![0],
        };
        let unflagged_fine_channel_indices = Vec1::try_from_vec(unflagged_fine_channel_indices)
            .map_err(|_| ReadBpCalFileError::NoUnflaggedChans)?;

        Ok(BpCal {
            fine_channel_resolution,
            unflagged_fine_channel_indices,
            unflagged_rf_input_indices,
            data,
        })
    }
}

#[derive(Error, Debug)]
pub(crate) enum ReadBpCalFileError {
    #[error("File is empty")]
    Empty,

    #[error("File contained no tile data")]
    NoTiles,

    #[error("Expected a 8N+1 lines (where N is an int), got {got} lines")]
    UnexpectedLineCount { got: usize },

    #[error("Couldn't parse '{text}' as a float on line {line_num}")]
    ParseFloat { text: String, line_num: usize },

    #[error("Couldn't parse '{text}' as an int on line {line_num}")]
    ParseInt { text: String, line_num: usize },

    #[error("The first line has no channel information")]
    NoUnflaggedChans,

    #[error("No integer tile number on line {line_num}")]
    NoTileNum { line_num: usize },

    #[error("IO error: {0}")]
    IO(#[from] std::io::Error),
}
