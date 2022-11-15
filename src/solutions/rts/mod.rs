// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to read and write RTS calibration solutions.
//!
//! See for more info:
//! <https://mwatelescope.github.io/mwa_hyperdrive/defs/cal_sols_rts.html>

// The RTS calls DI_JonesMatrices files "alignment files". The first two lines
// of these files correspond to "alignment flux density" (or "flux scale") and
// "post-alignment matrix". The reference line is structured the same as all
// those that follow it; 4 complex number pairs. All lines after the first two
// are "pre-alignment matrices". When reading in this data, the post-alignment
// matrix is inverted and applied to each pre-alignment matrix (as in, A*B where
// A is pre- and B- is inverse post).

mod read_files;
#[cfg(test)]
mod tests;

use read_files::*;

use std::{
    collections::{BTreeMap, HashSet},
    path::{Path, PathBuf},
};

use itertools::Itertools;
use log::{debug, trace, warn};
use marlu::Jones;
use mwalib::MetafitsContext;
use ndarray::prelude::*;
use regex::Regex;
use thiserror::Error;
use vec1::Vec1;

use super::CalibrationSolutions;
use crate::glob::get_all_matches_from_glob;

lazy_static::lazy_static! {
    static ref NODE_NUM: Regex = Regex::new(r"node(\d{3})\.dat$").unwrap();
}

pub(super) fn read<P: AsRef<Path>, P2: AsRef<Path>>(
    dir: P,
    metafits: P2,
) -> Result<CalibrationSolutions, RtsReadSolsError> {
    fn inner(dir: &Path, metafits: &Path) -> Result<CalibrationSolutions, RtsReadSolsError> {
        // Check `dir` exists and that it really is a directory.
        if !dir.exists() || !dir.is_dir() {
            return Err(RtsReadSolsError::NotADir(dir.to_path_buf()));
        }

        let context = MetafitsContext::new(metafits, None)?;

        // Search `dir` for DI_JonesMatrices_node???.dat and
        // BandpassCalibration_node???.dat files.
        let di_jm =
            get_all_matches_from_glob(&format!("{}/DI_JonesMatrices_node???.dat", dir.display()))?;
        if di_jm.is_empty() {
            return Err(RtsReadSolsError::NoDiJmFiles {
                dir: dir.display().to_string(),
            });
        }

        let bp_cal = get_all_matches_from_glob(&format!(
            "{}/BandpassCalibration_node???.dat",
            dir.display()
        ))?;
        if bp_cal.is_empty() {
            return Err(RtsReadSolsError::NoBpCalFiles {
                dir: dir.display().to_string(),
            });
        }

        debug!("Found RTS DI calibration files:");
        debug!("{:?}", &di_jm);
        debug!("{:?}", &bp_cal);

        // There should be an equal number of files for each type.
        if bp_cal.len() != di_jm.len() {
            return Err(RtsReadSolsError::UnequalFileCount {
                dir: dir.display().to_string(),
                di_jm_count: di_jm.len(),
                bp_cal_count: bp_cal.len(),
            });
        }

        // Make a map from the gpubox number (which is the same as the node???
        // number in the files) to the receiver channel, which is a proxy for the
        // sky frequency.
        let mut gpubox_to_receiver: BTreeMap<u8, u8> = BTreeMap::new();
        for cc in &context.metafits_coarse_chans {
            gpubox_to_receiver.insert(
                cc.gpubox_number.try_into().unwrap(),
                cc.rec_chan_number.try_into().unwrap(),
            );
        }
        debug!("RTS DI calibration file node num to receiver channel map:");
        debug!("{:?}", gpubox_to_receiver);

        // Unpack the files by receiver channel.
        let mut receiver_channel_to_data: BTreeMap<u8, (u8, Option<DiJm>, Option<BpCal>)> =
            BTreeMap::new();
        for di_jm in di_jm {
            trace!("Unpacking {di_jm:?}");
            let node_num = &NODE_NUM.captures(&di_jm.display().to_string()).unwrap()[1]
                .parse::<u8>()
                .unwrap();
            let receiver_number = gpubox_to_receiver.get(node_num).ok_or({
                RtsReadSolsError::NoGpuboxForNodeNum {
                    node_num: *node_num,
                }
            })?;
            let r = receiver_channel_to_data
                .entry(*receiver_number)
                .or_insert((*node_num, None, None));
            r.1 = Some(
                DiJm::read_file(&di_jm).map_err(|err| RtsReadSolsError::DiJmError {
                    filename: di_jm,
                    err,
                })?,
            );
        }
        for bp_cal in bp_cal {
            trace!("Unpacking {bp_cal:?}");
            let node_num = &NODE_NUM.captures(&bp_cal.display().to_string()).unwrap()[1]
                .parse::<u8>()
                .unwrap();
            let receiver_number = gpubox_to_receiver.get(node_num).ok_or({
                RtsReadSolsError::NoGpuboxForNodeNum {
                    node_num: *node_num,
                }
            })?;
            let r = receiver_channel_to_data
                .entry(*receiver_number)
                // Yeah, if we have to insert here, it meant there was no
                // corresponding DI_JM file. Handle that error soon.
                .or_insert((*node_num, None, None));
            r.2 = Some(
                BpCal::read_file(&bp_cal).map_err(|err| RtsReadSolsError::BpCalError {
                    filename: bp_cal,
                    err,
                })?,
            );
        }

        // Check if a receiver channel is missing a file. In the process, make a
        // more convenient data structure.
        let receiver_channel_to_data = {
            let mut out: BTreeMap<u8, (u8, DiJm, BpCal)> = BTreeMap::new();
            for (rec_chan, (gpubox_num, di_jm, bp_cal)) in receiver_channel_to_data {
                if di_jm.is_none() || bp_cal.is_none() {
                    return Err(RtsReadSolsError::MissingFileForGpuboxNum { num: gpubox_num });
                }
                out.insert(rec_chan, (gpubox_num, di_jm.unwrap(), bp_cal.unwrap()));
            }
            out
        };

        read_no_files(receiver_channel_to_data, &context)
    }
    inner(dir.as_ref(), metafits.as_ref())
}

fn read_no_files(
    receiver_channel_to_data: BTreeMap<u8, (u8, DiJm, BpCal)>,
    context: &MetafitsContext,
) -> Result<CalibrationSolutions, RtsReadSolsError> {
    // The number of files corresponds to the number of *provided* coarse
    // channels. We trust the metafits to represent that total number of coarse
    // channels so we can fill any gaps.
    let available_num_coarse_chans = receiver_channel_to_data.len();
    let total_num_coarse_chans = context.num_metafits_coarse_chans;
    if available_num_coarse_chans != total_num_coarse_chans {
        warn!("The number of coarse channels expected by the metafits ({total_num_coarse_chans})");
        warn!("    wasn't equal to the number of node files ({available_num_coarse_chans}).");
        warn!("    We will use NaNs for the missing coarse channels.");
    };

    // Check that the number of tiles is the same everywhere.
    let (_, some_di_jm, some_bp_cal) = receiver_channel_to_data.iter().next().unwrap().1;
    let total_num_tiles = some_di_jm.pre_alignment_matrices.len();
    let num_unflagged_tiles = some_bp_cal.unflagged_rf_input_indices.len();
    for (gpubox_num, di_jm, bp_cal) in receiver_channel_to_data.values() {
        if di_jm.pre_alignment_matrices.len() < total_num_tiles {
            return Err(RtsReadSolsError::UnequalTileCountDiJm {
                expected: total_num_tiles,
                got: di_jm.pre_alignment_matrices.len(),
                gpubox_num: *gpubox_num,
            });
        }
        if bp_cal.unflagged_rf_input_indices.len() != num_unflagged_tiles {
            return Err(RtsReadSolsError::UnequalTileCountBpCal {
                expected: num_unflagged_tiles,
                got: bp_cal.unflagged_rf_input_indices.len(),
                gpubox_num: *gpubox_num,
            });
        }
    }

    let antenna_to_rf_input_on_2_map: BTreeMap<_, _> = (0..total_num_tiles)
        .map(|i_tile| {
            let i_input = context
                .rf_inputs
                .iter()
                .find(|rf| rf.ant == i_tile as u32)
                .unwrap()
                .input
                / 2;
            (i_tile, i_input)
        })
        .collect();
    debug!("Antenna number to RF input on 2 map:");
    debug!("{antenna_to_rf_input_on_2_map:?}",);

    // Get the flagged tile indices from the flagged RF inputs.
    let flagged_tiles: Vec<_> = (0..total_num_tiles)
        .filter(|i_tile| {
            let i_input = context
                .rf_inputs
                .iter()
                .find(|rf| rf.ant == *i_tile as u32)
                .unwrap()
                .input
                / 2;
            !some_bp_cal
                .unflagged_rf_input_indices
                .contains(&(i_input.try_into().unwrap()))
        })
        .collect();
    debug!("Flagged tiles: {flagged_tiles:?}");
    let flagged_rf_input_indices: Vec<_> = flagged_tiles
        .iter()
        .map(|i_tile| {
            context
                .rf_inputs
                .iter()
                .find(|rf| rf.ant == *i_tile as u32)
                .unwrap()
                .input
                / 2
        })
        .collect();

    // Try to work out the total number of fine frequency channels.
    let num_fine_chans_per_coarse_chan = {
        let smallest_fine_chan_res = receiver_channel_to_data
            .iter()
            .fold(f64::INFINITY, |acc, (_, (_, _, bp))| {
                acc.min(bp.fine_channel_resolution.unwrap_or(f64::INFINITY))
            });

        // Only handling legacy correlator settings for now.
        if (smallest_fine_chan_res - 40e3).abs() < f64::EPSILON {
            32
        } else if (smallest_fine_chan_res - 20e3).abs() < f64::EPSILON {
            64
        } else if (smallest_fine_chan_res - 10e3).abs() < f64::EPSILON {
            128
        } else {
            return Err(RtsReadSolsError::UnhandledFreqRes(smallest_fine_chan_res));
        }
    };
    debug!("Number of fine channels per coarse channel: {num_fine_chans_per_coarse_chan}");
    let total_num_fine_freq_chans = num_fine_chans_per_coarse_chan * total_num_coarse_chans;
    debug!("Total number of fine freq. channels: {total_num_fine_freq_chans}");
    // Get the flagged fine channels. Start by checking available channels.
    let unflagged_fine_chans = receiver_channel_to_data
        .values()
        .flat_map(|(i_gpubox, _, bp)| {
            let i_cc = usize::from(*i_gpubox - 1);
            let offset = i_cc * num_fine_chans_per_coarse_chan;
            bp.unflagged_fine_channel_indices
                .iter()
                .map(move |&i_chan| i_chan + offset)
        })
        .collect::<HashSet<_>>();
    let mut flagged_fine_channels: HashSet<u16> = (0..total_num_fine_freq_chans)
        .filter(|i_chan| !unflagged_fine_chans.contains(i_chan))
        .map(|i_chan| {
            let i_chan: u16 = i_chan.try_into().unwrap();
            i_chan
        })
        .collect();
    // Now flag all unavailable channels.
    for i_chan in 0..total_num_fine_freq_chans {
        if !unflagged_fine_chans.contains(&i_chan) {
            flagged_fine_channels.insert(i_chan.try_into().unwrap());
        }
    }
    let flagged_fine_channels = flagged_fine_channels.into_iter().sorted().collect();

    let mut di_jones = Array3::from_elem(
        (
            1, // RTS solutions don't change over time.
            total_num_tiles,
            total_num_fine_freq_chans,
        ),
        Jones::nan(),
    );
    // Iterating over the BTreeMap gives the solutions in the correct order,
    // becauase the map's keys are ascendingly sorted and correspond to
    // ascending sky frequency (which is how we want the data).
    for (i_cc, (_, (_, di_jm, mut bp_cal))) in receiver_channel_to_data.into_iter().enumerate() {
        debug!("Reading coarse channel {i_cc}");

        // Apply di_jm to the bp_cal data. Modify the bp_cal data in place, then
        // put it in the outgoing di_jones.
        let mut bp_cal_data = bp_cal.data.slice_mut(s![
            ..,
            1, // Throw away the "lsq" data; only use "fit".
            ..
        ]);
        let post_inv = Jones::from([
            di_jm.post_alignment_matrix[3],
            di_jm.post_alignment_matrix[2],
            di_jm.post_alignment_matrix[1],
            di_jm.post_alignment_matrix[0],
        ])
        .inv();
        bp_cal_data
            .outer_iter_mut()
            .zip(
                di_jm
                    .pre_alignment_matrices
                    .iter()
                    .enumerate()
                    .filter(|(i_input, _)| {
                        !flagged_rf_input_indices.contains(&((*i_input).try_into().unwrap()))
                    })
                    .map(|pair| pair.1),
            )
            .for_each(|(mut bp_cal, &di_jm_pre)| {
                let di_jm = Jones::from([
                    di_jm_pre[0].re,
                    -di_jm_pre[0].im,
                    di_jm_pre[1].re,
                    -di_jm_pre[1].im,
                    di_jm_pre[2].re,
                    -di_jm_pre[2].im,
                    di_jm_pre[3].re,
                    -di_jm_pre[3].im,
                ])
                .inv()
                    * post_inv;

                bp_cal.iter_mut().for_each(|bp_cal| {
                    *bp_cal = di_jm * *bp_cal;

                    // These Jones matrices are currently in [PX, PY, QX, QY].
                    // Map them to [XX, XY, YX, YY].
                    *bp_cal = Jones::from([bp_cal[3], bp_cal[2], bp_cal[1], bp_cal[0]]);
                });
            });

        // Put unflagged data into the output di_jones. di_jones tiles are
        // ordered by metafits antenna number, whereas RTS data is ordered by
        // metafits input number. Flagged tiles and channels already have NaN
        // written.
        for (i_tile, mut di_jones) in di_jones
            .slice_mut(s![0, .., ..])
            .outer_iter_mut()
            .enumerate()
        {
            trace!("i_tile: {i_tile}");
            // Get the input number for this tile.
            let mut i_input = context
                .rf_inputs
                .iter()
                .find(|rf| rf.ant == i_tile as u32)
                .map(|rf| rf.input / 2)
                .unwrap();
            trace!("i_input: {i_input}");
            // Is it unflagged?
            if !flagged_rf_input_indices.contains(&i_input) {
                // Now adjust i_input based on how many inputs were flagged
                // before it, so that we can index the array of data (it doesn't
                // include flagged data).
                let mut flagged_count = 0;
                for &flagged_input in &flagged_rf_input_indices {
                    if flagged_input <= i_input {
                        flagged_count += 1;
                    }
                }
                i_input -= flagged_count;

                let bp_cal_data = bp_cal_data.slice(s![i_input as usize, ..]);

                // Unpack the unflagged channels.
                let offset = i_cc * num_fine_chans_per_coarse_chan;
                for ((_, di_jones), bp_cal_data) in di_jones
                    .iter_mut()
                    .skip(offset)
                    .enumerate()
                    .filter(|(i_chan, _)| bp_cal.unflagged_fine_channel_indices.contains(i_chan))
                    .zip(bp_cal_data.iter())
                {
                    *di_jones = *bp_cal_data;
                }
            }
        }
    }

    let tile_names: Vec<String> = context
        .antennas
        .iter()
        .map(|a| a.tile_name.clone())
        .collect();
    let tile_names = Vec1::try_from_vec(tile_names).ok();

    Ok(CalibrationSolutions {
        di_jones,
        flagged_tiles,
        flagged_chanblocks: flagged_fine_channels,
        // We assume that the metafits file is the right one.
        obsid: Some(context.obs_id),
        tile_names,
        // lmao
        ..Default::default()
    })
}

// Writing RTS solutions is perhaps more art than science. Hopefully no one ever
// needs this, but some commented code is better than nothing.

// pub(super) fn write<P: AsRef<Path>, P2: AsRef<Path>>(
//     sols: &CalibrationSolutions,
//     dir: P,
//     metafits: P2,
// ) -> Result<(), RtsWriteSolsError> {
//     fn inner(
//         sols: &CalibrationSolutions,
//         dir: &Path,
//         metafits: &Path,
//     ) -> Result<(), RtsWriteSolsError> {
//         if dir.exists() {
//             // If it exists, check that `dir` really is a directory.
//             if !dir.is_dir() {
//                 return Err(RtsWriteSolsError::NotADir(dir.to_path_buf()));
//             }
//         } else {
//             // Make the specified directory and all its parents if `dir` doesn't
//             // exist.
//             std::fs::create_dir_all(dir).map_err(|e| RtsWriteSolsError::CouldntMakeDir {
//                 dir: dir.to_path_buf(),
//                 err: e,
//             })?;
//         }

//         let context = MetafitsContext::new(metafits, None)?;
//         let num_fine_chans_per_coarse_chan = context.num_corr_fine_chans_per_coarse;
//         let freq_res = context.corr_fine_chan_width_hz as f64;

//         if sols.di_jones.len_of(Axis(0)) > 1 {
//             warn!("Multiple timeblocks of solutions aren't supported by the RTS; using only the first one");
//         }

//         // Make a map from the receiver channel (a proxy for the sky frequency) to
//         // the gpubox number (which is the same as the node??? number in the files).
//         let mut receiver_to_gpubox: BTreeMap<usize, usize> = BTreeMap::new();
//         for cc in &context.metafits_coarse_chans {
//             receiver_to_gpubox.insert(cc.rec_chan_number, cc.gpubox_number);
//         }
//         debug!("Receiver channel map to RTS DI calibration file node num:");
//         debug!("{:?}", receiver_to_gpubox);
//         // And a map from the RF input number divided by 2 to the tile (a.k.a.
//         // ANTENNA) number.
//         let mut input_to_tile: BTreeMap<usize, usize> = BTreeMap::new();
//         for rf in &context.rf_inputs {
//             input_to_tile.insert(rf.input as usize / 2, rf.ant as usize);
//         }
//         debug!("RF input / 2 map to tile index:");
//         debug!("{:?}", input_to_tile);

//         for (i_cc, (_, i_gpubox)) in receiver_to_gpubox.into_iter().enumerate() {
//             // Create the RTS files.
//             let di_jm_fp = format!("{}/DI_JonesMatrices_node{:03}.dat", dir.display(), i_gpubox);
//             debug!("Writing to {di_jm_fp}");
//             let mut di_jm_file = BufWriter::new(File::create(di_jm_fp)?);
//             let bp_cal_fp = format!(
//                 "{}/BandpassCalibration_node{:03}.dat",
//                 dir.display(),
//                 i_gpubox
//             );
//             debug!("Writing to {bp_cal_fp}");
//             let mut bp_cal_file = BufWriter::new(File::create(bp_cal_fp)?);

//             // Isolate the applicable data.
//             let chan_offset = i_cc * num_fine_chans_per_coarse_chan;
//             let chan_range = chan_offset..((i_cc + 1) * num_fine_chans_per_coarse_chan);
//             let data = sols.di_jones.slice(s![0, .., chan_range]);

//             // Write the useless alignment flux density...
//             writeln!(&mut di_jm_file, "{:.6}", 1.0)?;
//             // ... and make the post-aligment matrix identity.
//             write!(&mut di_jm_file, "{:+.6}, ", 1.0)?;
//             write!(&mut di_jm_file, "{:+.6}, ", 0.0)?;
//             write!(&mut di_jm_file, "{:+.6}, ", 0.0)?;
//             write!(&mut di_jm_file, "{:+.6}, ", 0.0)?;
//             write!(&mut di_jm_file, "{:+.6}, ", 0.0)?;
//             write!(&mut di_jm_file, "{:+.6}, ", 0.0)?;
//             write!(&mut di_jm_file, "{:+.6}, ", 1.0)?;
//             writeln!(&mut di_jm_file, "{:+.6}", 0.0)?;

//             // Write the unflagged fine channel frequencies in MHz.
//             let mut unflagged_chan_line = String::new();
//             for i_chan in 0..num_fine_chans_per_coarse_chan {
//                 if !sols
//                     .flagged_chanblocks
//                     .contains(&((i_chan + chan_offset).try_into().unwrap()))
//                 {
//                     unflagged_chan_line.push_str(&format!(
//                         "{:.6}, ",
//                         (i_chan as f64 * freq_res).round() / 1e6
//                     ));
//                 }
//             }
//             writeln!(
//                 &mut bp_cal_file,
//                 "{}",
//                 unflagged_chan_line.trim_end_matches(", ")
//             )?;

//             // And now write tile Jones matrices.
//             for (&i_input, &i_tile) in &input_to_tile {
//                 let flagged = sols.flagged_tiles.contains(&i_tile);

//                 // Find the "average" Jones matrix for the DI JM file.
//                 let data = data.slice(s![i_tile, ..]);
//                 let average = if flagged {
//                     Jones::default()
//                 } else {
//                     let (sum, length) = data.iter().enumerate().fold(
//                         (Jones::default(), 0),
//                         |(acc_sum, acc_length), (i_chan_in_cc, j)| {
//                             if !sols
//                                 .flagged_chanblocks
//                                 .contains(&((i_chan_in_cc + chan_offset) as _))
//                             {
//                                 (acc_sum + *j, acc_length + 1)
//                             } else {
//                                 (acc_sum, acc_length)
//                             }
//                         },
//                     );
//                     if length > 0 {
//                         sum / length as f64
//                     } else {
//                         sum
//                     }
//                 };
//                 writeln!(
//                     &mut di_jm_file,
//                     "{:+.6}, {:+.6}, {:+.6}, {:+.6}, {:+.6}, {:+.6}, {:+.6}, {:+.6}",
//                     // Don't forget to reorder into RTS PX, PY, QX, QY.
//                     average[3].re,
//                     average[3].im,
//                     average[2].re,
//                     average[2].im,
//                     average[1].re,
//                     average[1].im,
//                     average[0].re,
//                     average[0].im
//                 )?;

//                 // BP cal files don't include flagged tiles.
//                 if flagged {
//                     continue;
//                 }

//                 // With the average, find the bandpass Jones matrices.
//                 let avg_inv = average.inv();
//                 let bp_data = data.mapv(|j| avg_inv * j);
//                 // Write "fit" data the same as "lsq". No one cares...
//                 let mut bp_cal_line = String::new();
//                 for i in 0..8 {
//                     let i_jones_elem = match i / 2 {
//                         0 => 3,
//                         1 => 2,
//                         2 => 1,
//                         3 => 0,
//                         _ => unreachable!(),
//                     };

//                     bp_cal_line.push_str(&format!("{}, ", i_input + 1));
//                     for (_, j) in bp_data.iter().enumerate().filter(|(i_chan, _)| {
//                         !sols
//                             .flagged_chanblocks
//                             .contains(&((i_chan + chan_offset) as _))
//                     }) {
//                         bp_cal_line.push_str(&format!(
//                             "{:+.6},{:+.6}, ",
//                             j[i_jones_elem].norm(),
//                             j[i_jones_elem].arg()
//                         ));
//                     }
//                     writeln!(&mut bp_cal_file, "{}", bp_cal_line.trim_end_matches(", "))?;
//                     bp_cal_line.clear();
//                 }
//             }
//             di_jm_file.flush()?;
//             bp_cal_file.flush()?;
//         }

//         Ok(())
//     }
//     inner(sols, dir.as_ref(), metafits.as_ref())
// }

#[derive(Error, Debug)]
pub(crate) enum RtsReadSolsError {
    #[error("Attempted to read RTS solutions from '{0}', but this isn't a directory")]
    NotADir(PathBuf),

    #[error("Found no RTS DI_JonesMatrices_node???.dat files in directory {dir}")]
    NoDiJmFiles { dir: String },

    #[error("Found no RTS BandpassCalibration_node???.dat files in directory {dir}")]
    NoBpCalFiles { dir: String },

    #[error("In directory {dir}, found {di_jm_count} DI_JonesMatrices_node???.dat files, but {bp_cal_count} BandpassCalibration_node???.dat files.\nThere must be an equal number of these files.")]
    UnequalFileCount {
        dir: String,
        di_jm_count: usize,
        bp_cal_count: usize,
    },

    #[error("Either a BandpassCalibration or DI_JonesMatrices file is missing for node{num:03}")]
    MissingFileForGpuboxNum { num: u8 },

    #[error("Node number {node_num} didn't correspond to a gpubox number")]
    NoGpuboxForNodeNum { node_num: u8 },

    #[error("Expected {expected} tiles in all files, but got {got} tiles in DI_JonesMatrices_node{gpubox_num:03}.dat")]
    UnequalTileCountDiJm {
        expected: usize,
        got: usize,
        gpubox_num: u8,
    },

    #[error("Expected {expected} tiles in all files, but got {got} tiles in BandpassCalibration_node{gpubox_num:03}.dat")]
    UnequalTileCountBpCal {
        expected: usize,
        got: usize,
        gpubox_num: u8,
    },

    #[error("Unhandled RTS frequency resolution: {0} Hz")]
    UnhandledFreqRes(f64),

    #[error("Error when reading {filename}: {err}")]
    DiJmError {
        filename: PathBuf,
        err: ReadDiJmFileError,
    },

    #[error("Error when reading {filename}: {err}")]
    BpCalError {
        filename: PathBuf,
        err: ReadBpCalFileError,
    },

    #[error(transparent)]
    Glob(#[from] crate::glob::GlobError),

    #[error("Error when reading metafits: {0}")]
    Mwalib(#[from] mwalib::MwalibError),
}

// #[derive(Error, Debug)]
// pub(crate) enum RtsWriteSolsError {
//     #[error("Attempted to write RTS solutions into '{0}', but this isn't a directory")]
//     NotADir(PathBuf),

//     #[error("Couldn't create directory '{dir}' (or its parents): {err}")]
//     CouldntMakeDir { dir: PathBuf, err: std::io::Error },

//     #[error("Error when reading metafits: {0}")]
//     Mwalib(#[from] mwalib::MwalibError),

//     #[error(transparent)]
//     IO(#[from] std::io::Error),
// }
