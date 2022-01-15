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

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use marlu::Jones;
use thiserror::Error;

use mwa_hyperdrive_common::{hifitime, marlu, ndarray, thiserror, Complex};

struct DI_JM {
    alignment_flux_density: f64,
    post_alignment_matrix: Jones<f64>,
    /// The length of this vector is the number of tiles.
    pre_alignment_matrices: Vec<Jones<f64>>,
}

fn read_di_jm_file(path: &Path) -> Result<DI_JM, ReadDiJmFileError> {
    let filename = path.display().to_string();
    let mut file = BufReader::new(File::open(path)?);
    read_di_jm(&mut file, filename)
}

fn read_di_jm<T: std::io::BufRead, S: AsRef<str>>(
    file: &mut T,
    filename: S,
) -> Result<DI_JM, ReadDiJmFileError> {
    let mut alignment_flux_density = 0.0;
    let mut post_alignment_matrix: Jones<f64> = Jones::default();
    let mut pre_alignment_matrices: Vec<Jones<f64>> = Vec::with_capacity(128);
    for (i_line, line) in file.lines().enumerate() {
        let line = line?;

        match i_line {
            0 => {
                alignment_flux_density =
                    line.parse::<f64>()
                        .map_err(|_| ReadDiJmFileError::BadAlignmentFluxDensity {
                            filename: filename.as_ref().to_string(),
                            line,
                        })?
            }

            1 => {
                let mut num_floats = 0;
                for (float_str, j) in line
                    .split(", ")
                    .zip(post_alignment_matrix.iter_mut().flat_map(|c| [c.re, c.im]))
                {
                    num_floats += 1;
                    dbg!(&float_str, num_floats);
                    let float =
                        float_str
                            .parse::<f64>()
                            .map_err(|_| ReadDiJmFileError::ParseFloat {
                                text: float_str.to_string(),
                                line_num: i_line + 1,
                            })?;
                    j = float;
                    // if num_floats % 2 == 1 {
                    //     j.re = float;
                    // } else {
                    //     j.im = float;
                    // }
                }

                if num_floats != 8 {
                    return Err(ReadDiJmFileError::BadFloatCount {
                        count: num_floats,
                        line_num: i_line + 1,
                    });
                }
            }

            _ => {
                let mut pre_alignment: Jones<f64> = Jones::default();
                let mut num_floats = 0;
                for (float_str, j) in line.split(", ").zip(pre_alignment.iter_mut()) {
                    num_floats += 1;
                    let float =
                        float_str
                            .parse::<f64>()
                            .map_err(|_| ReadDiJmFileError::ParseFloat {
                                text: float_str.to_string(),
                                line_num: i_line + 1,
                            })?;
                    if num_floats % 2 == 1 {
                        j.re = float;
                    } else {
                        j.im = float;
                    }
                }

                if num_floats != 8 {
                    return Err(ReadDiJmFileError::BadFloatCount {
                        count: num_floats,
                        line_num: i_line + 1,
                    });
                }

                pre_alignment_matrices.push(pre_alignment);
            }
        }
    }

    Ok(DI_JM {
        alignment_flux_density,
        post_alignment_matrix,
        pre_alignment_matrices,
    })
}

#[derive(Error, Debug)]
pub enum ReadDiJmFileError {
    #[error("The first line of {filename} should contain the alignment flux density, but it could not be parsed as a float ({line})")]
    BadAlignmentFluxDensity { filename: String, line: String },

    #[error("Couldn't parse '{text}' as a float on line {line_num}")]
    ParseFloat { text: String, line_num: usize },

    #[error("Expected to find 8 floats on line {line_num}, but found {} instead")]
    BadFloatCount { count: usize, line_num: usize },

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
// the number of unflagged fine channels of complex numbers following it, e.g.
//
// 1, +1.015465,+0.047708, +1.006940,+0.005763, +1.024352,+0.015657, ...
//
// (but on one line). There will be 8 lines per tile; they represent:
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
// Apparently only the fitted data is interesting.

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use approx::assert_abs_diff_eq;
    use indoc::indoc;

    use super::*;
    use crate::jones_test::TestJones;

    #[test]
    fn test_read_di_jm() {
        let mut contents = Cursor::new(indoc! {"
        16.990089
        -0.131782, -0.933494, +0.019562, +0.135049, -0.008773, -0.134161, -0.063910, -0.921045
        +0.446814, -0.203973, -0.071393, +0.031377, -0.027429, +0.060070, -0.153991, +0.446221
        -0.629691, -0.403369, +0.088931, +0.060120, -0.065456, -0.061017, -0.541366, -0.433762
        "});

        let result = read_di_jm(&mut contents, "dummy");
        let di_jm = match result {
            Ok(r) => r,
            Err(e) => panic!("{}", e),
        };

        assert_abs_diff_eq!(di_jm.alignment_flux_density, 16.990089);
        assert_abs_diff_eq!(
            TestJones::from(di_jm.post_alignment_matrix),
            TestJones::from([
                Complex::new(-0.131782, -0.933494),
                Complex::new(0.019562, 0.135049),
                Complex::new(-0.008773, -0.134161),
                Complex::new(-0.063910, -0.921045)
            ])
        );
        assert_eq!(di_jm.pre_alignment_matrices.len(), 2);
    }
}
