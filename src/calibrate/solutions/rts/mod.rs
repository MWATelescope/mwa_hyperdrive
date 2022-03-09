// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to read and write RTS calibration solutions.
//!
//! See for more info:
//! https://github.com/MWATelescope/mwa_hyperdrive/wiki/Calibration-solutions

// The RTS calls DI_JonesMatrices files "alignment files". The first two lines
// of these files correspond to "alignment flux density" (or "flux scale") and
// "post-alignment matrix". The reference line is structured the same as all
// those that follow it; 4 complex number pairs. All lines after the first two
// are "pre-alignment matrices". When reading in this data, the post-alignment
// matrix is inverted and applied to each pre-alignment matrix (as in, A*B where
// A is pre- and B- is inverse post).

mod read_files;

use birli::mwalib::MetafitsContext;
use read_files::*;

use std::collections::{BTreeMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use log::{debug, warn};
use marlu::{c64, Jones, RADec};
use ndarray::prelude::*;
use regex::Regex;
use thiserror::Error;

use super::CalibrationSolutions;
use crate::glob::get_all_matches_from_glob;
use mwa_hyperdrive_common::{lazy_static, log, marlu, mwalib, ndarray, thiserror};

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

        let context = mwalib::MetafitsContext::new(&metafits, None)?;

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
            let node_num = &NODE_NUM.captures(&di_jm.display().to_string()).unwrap()[1]
                .parse::<u8>()
                .unwrap();
            let receiver_number = gpubox_to_receiver.get(node_num).ok_or({
                RtsReadSolsError::NoGpuboxForNodeNum {
                    node_num: *node_num,
                }
            })?;
            let mut r = receiver_channel_to_data
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
            let node_num = &NODE_NUM.captures(&bp_cal.display().to_string()).unwrap()[1]
                .parse::<u8>()
                .unwrap();
            let receiver_number = gpubox_to_receiver.get(node_num).ok_or({
                RtsReadSolsError::NoGpuboxForNodeNum {
                    node_num: *node_num,
                }
            })?;
            let mut r = receiver_channel_to_data
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

    // Get the flagged tile indices from the flagged RF inputs.
    let flagged_tiles = (0..total_num_tiles)
        .into_iter()
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
    let total_num_fine_freq_chans = num_fine_chans_per_coarse_chan * total_num_coarse_chans;
    // Get the flagged fine channels. Start by checking available channels.
    let unflagged_fine_chans = receiver_channel_to_data
        .values()
        .flat_map(|(_, _, bp)| &bp.unflagged_fine_channel_indices)
        .collect::<HashSet<_>>();
    let mut flagged_fine_channels: Vec<u16> = (0..total_num_fine_freq_chans)
        .into_iter()
        .filter(|i_chan| !unflagged_fine_chans.contains(i_chan))
        .map(|i_chan| i_chan.try_into().unwrap())
        .collect();
    // Now flag all unavailable channels.
    for i_chan in 0..total_num_fine_freq_chans {
        if !unflagged_fine_chans.contains(&i_chan) {
            flagged_fine_channels.push(i_chan.try_into().unwrap());
        }
    }
    flagged_fine_channels.sort_unstable();

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
        let unflagged_rf_input_indices = bp_cal.unflagged_rf_input_indices;

        // Apply di_jm to the bp_cal data. Modify the bp_cal data in place, then
        // put it in the outgoing di_jones.
        let mut bp_cal_data = bp_cal.data.slice_mut(s![
            ..,
            1, // Throw away the "lsq" data; only use "fit".
            ..
        ]);
        let post_inv = di_jm.post_alignment_matrix.inv();
        bp_cal_data
            .outer_iter_mut()
            .zip(
                di_jm
                    .pre_alignment_matrices
                    .iter()
                    .enumerate()
                    .filter(|(i_input, _)| {
                        unflagged_rf_input_indices.contains(&((*i_input).try_into().unwrap()))
                    })
                    .map(|pair| pair.1),
            )
            .for_each(|(mut bp_cal, &di_jm_pre)| {
                dbg!(di_jm_pre);
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
                dbg!(di_jm);
                bp_cal.iter_mut().for_each(|bp_cal| {
                    *bp_cal = di_jm * *bp_cal;

                    // These Jones matrices are currently in [PX, PY, QX, QY].
                    // Map them to [XX, XY, YX, YY].
                    *bp_cal = Jones::from([bp_cal[3], bp_cal[2], bp_cal[1], bp_cal[0]]);
                    dbg!(bp_cal);
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
            // Get the input number for this tile.
            let mut i_input = context
                .rf_inputs
                .iter()
                .find(|rf| rf.ant == i_tile as u32)
                .map(|rf| rf.input / 2)
                .unwrap()
                .try_into()
                .unwrap();
            // Is it unflagged?
            if unflagged_rf_input_indices.contains(&i_input) {
                // Now adjust i_input based on how many inputs were flagged
                // before it, so that we can index the array of data (it doesn't
                // include flagged data).
                let mut last = 0;
                let mut flagged_count = 0;
                for (i_unflagged_input, &unflagged_input) in
                    unflagged_rf_input_indices.iter().enumerate()
                {
                    if i_unflagged_input == 0 {
                        flagged_count += unflagged_input;
                        last = unflagged_input;
                        continue;
                    }

                    if unflagged_input > i_input {
                        break;
                    }
                    if unflagged_input - last > 1 {
                        flagged_count += 1;
                    }
                    last = unflagged_input;
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

    Ok(CalibrationSolutions {
        di_jones,
        flagged_tiles,
        flagged_chanblocks: flagged_fine_channels,
        obsid: Some(context.obs_id),
        // lmao
        start_timestamps: vec![],
        end_timestamps: vec![],
        average_timestamps: vec![],
    })
}

pub(super) fn write<P: AsRef<Path>, P2: AsRef<Path>, P3: AsRef<Path>>(
    sols: &CalibrationSolutions,
    dir: P,
    metafits: P2,
    fee_beam_file: Option<P3>,
    num_decimal_points: Option<u8>,
) -> Result<(), RtsWriteSolsError> {
    fn inner(
        sols: &CalibrationSolutions,
        dir: &Path,
        metafits: &Path,
        fee_beam_file: Option<&Path>,
        num_decimal_points: Option<u8>,
    ) -> Result<(), RtsWriteSolsError> {
        if dir.exists() {
            // If it exists, check that `dir` really is a directory.
            if !dir.is_dir() {
                return Err(RtsWriteSolsError::NotADir(dir.to_path_buf()));
            }
        } else {
            // Make the specified directory and all its parents if `dir` doesn't
            // exist.
            std::fs::create_dir_all(dir).map_err(|e| RtsWriteSolsError::CouldntMakeDir {
                dir: dir.to_path_buf(),
                err: e,
            })?;
        }

        let context = mwalib::MetafitsContext::new(&metafits, None)?;
        let num_fine_chans_per_coarse_chan = context.num_corr_fine_chans_per_coarse;
        let freq_res = context.corr_fine_chan_width_hz as f64;
        let phase_centre_radec = RADec::new_degrees(
            context
                .ra_phase_center_degrees
                .unwrap_or(context.ra_tile_pointing_degrees),
            context
                .dec_phase_center_degrees
                .unwrap_or(context.dec_tile_pointing_degrees),
        );
        let phase_centre = phase_centre_radec.to_hadec(context.lst_rad).to_azel_mwa();

        let fee_beam = mwa_hyperdrive_beam::create_fee_beam_object(
            fee_beam_file,
            1,
            mwa_hyperdrive_beam::Delays::Partial(
                crate::data_formats::metafits::get_ideal_dipole_delays(&context),
            ),
            None,
        )
        .unwrap();

        if sols.di_jones.len_of(Axis(0)) > 1 {
            warn!("Multiple timeblocks of solutions aren't supported by the RTS; using only the first one");
        }

        // Make a map from the receiver channel (a proxy for the sky frequency) to
        // the gpubox number (which is the same as the node??? number in the files).
        let mut receiver_to_gpubox: BTreeMap<usize, usize> = BTreeMap::new();
        for cc in &context.metafits_coarse_chans {
            receiver_to_gpubox.insert(cc.rec_chan_number, cc.gpubox_number);
        }
        debug!("Receiver channel map to RTS DI calibration file node num:");
        debug!("{:?}", receiver_to_gpubox);
        // And a map from the RF input number divided by 2 to the tile (a.k.a.
        // ANTENNA) number.
        let mut input_to_tile: BTreeMap<usize, usize> = BTreeMap::new();
        for rf in &context.rf_inputs {
            input_to_tile.insert(rf.input as usize / 2, rf.ant as usize);
        }
        debug!("RF input / 2 map to tile index:");
        debug!("{:?}", input_to_tile);

        let num_decimal_points = num_decimal_points.unwrap_or(12) as usize;

        let smallest_rec_chan = context
            .metafits_coarse_chans
            .iter()
            .fold(usize::MAX, |acc, cc| acc.min(cc.rec_chan_number));
        for (i_cc, (i_recv, i_gpubox)) in receiver_to_gpubox.into_iter().enumerate() {
            let offset = i_recv - smallest_rec_chan;

            // Create the RTS files.
            let di_jm_fp = format!("{}/DI_JonesMatrices_node{:03}.dat", dir.display(), i_gpubox);
            debug!("Writing to {di_jm_fp}");
            let mut di_jm_file = BufWriter::new(File::create(di_jm_fp)?);
            let bp_cal_fp = format!(
                "{}/BandpassCalibration_node{:03}.dat",
                dir.display(),
                i_gpubox
            );
            debug!("Writing to {bp_cal_fp}");
            let mut bp_cal_file = BufWriter::new(File::create(bp_cal_fp)?);

            // Isolate the applicable data.
            let chan_offset = offset * num_fine_chans_per_coarse_chan;
            let chan_range = chan_offset..((offset + 1) * num_fine_chans_per_coarse_chan);
            let data = sols.di_jones.slice(s![0, .., chan_range]);

            // Write the useless alignment flux density...
            writeln!(&mut di_jm_file, "{1:.0$}", num_decimal_points, 1.0)?;
            // ... and write the beam response for this coarse channel.
            // let freq_hz = i_recv as f64 * 1.28e6;
            // let beam_response = fee_beam.calc_jones(phase_centre, freq_hz, 0).unwrap();
            let beam_response = Jones::from(match i_gpubox {
                1 => [
                    0.823644, 0.005932, -0.049899, -0.000282, 0.047816, -0.000096, 0.800999,
                    -0.003228,
                ],
                2 => [
                    0.825069, 0.006439, -0.049982, -0.000314, 0.047960, -0.000097, 0.803325,
                    -0.003212,
                ],
                3 => [
                    0.826439, 0.006867, -0.050071, -0.000342, 0.048094, -0.000097, 0.805604,
                    -0.003191,
                ],
                4 => [
                    0.827763, 0.007216, -0.050143, -0.000360, 0.048230, -0.000094, 0.807838,
                    -0.003176,
                ],
                5 => [
                    0.829058, 0.007474, -0.050211, -0.000376, 0.048354, -0.000095, 0.810031,
                    -0.003172,
                ],
                6 => [
                    0.830318, 0.007647, -0.050299, -0.000387, 0.048486, -0.000094, 0.812185,
                    -0.003191,
                ],
                7 => [
                    0.831571, 0.007722, -0.050371, -0.000379, 0.048612, -0.000094, 0.814312,
                    -0.003237,
                ],
                8 => [
                    0.832825, 0.007704, -0.050451, -0.000377, 0.048734, -0.000098, 0.816416,
                    -0.003308,
                ],
                9 => [
                    0.834089, 0.007598, -0.050526, -0.000367, 0.048853, -0.000097, 0.818507,
                    -0.003411,
                ],
                10 => [
                    0.835377, 0.007404, -0.050611, -0.000357, 0.048982, -0.000100, 0.820593,
                    -0.003540,
                ],
                11 => [
                    0.836703, 0.007132, -0.050699, -0.000335, 0.049095, -0.000107, 0.822678,
                    -0.003693,
                ],
                12 => [
                    0.838512, 0.006706, -0.050832, -0.000300, 0.049224, -0.000121, 0.825254,
                    -0.003946,
                ],
                13 => [
                    0.839887, 0.006343, -0.050911, -0.000282, 0.049333, -0.000135, 0.827323,
                    -0.004102,
                ],
                14 => [
                    0.841318, 0.005931, -0.051015, -0.000253, 0.049451, -0.000145, 0.829395,
                    -0.004272,
                ],
                15 => [
                    0.842840, 0.005466, -0.051112, -0.000226, 0.049583, -0.000156, 0.831477,
                    -0.004436,
                ],
                16 => [
                    0.844344, 0.004993, -0.051200, -0.000196, 0.049703, -0.000167, 0.833564,
                    -0.004628,
                ],
                17 => [
                    0.845931, 0.004462, -0.051348, -0.000126, 0.049781, -0.000135, 0.835694,
                    -0.004822,
                ],
                18 => [
                    0.847601, 0.003947, -0.051454, -0.000096, 0.049896, -0.000150, 0.837831,
                    -0.005004,
                ],
                19 => [
                    0.849314, 0.003428, -0.051570, -0.000075, 0.050021, -0.000167, 0.839929,
                    -0.005176,
                ],
                20 => [
                    0.851081, 0.002916, -0.051679, -0.000048, 0.050142, -0.000184, 0.842027,
                    -0.005346,
                ],
                21 => [
                    0.852896, 0.002418, -0.051794, -0.000015, 0.050247, -0.000192, 0.844125,
                    -0.005507,
                ],
                22 => [
                    0.854748, 0.001938, -0.051908, 0.000001, 0.050377, -0.000206, 0.846222,
                    -0.005661,
                ],
                23 => [
                    0.856636, 0.001483, -0.052033, 0.000029, 0.050498, -0.000225, 0.848318,
                    -0.005806,
                ],
                24 => [
                    0.858554, 0.001054, -0.052150, 0.000047, 0.050618, -0.000236, 0.850410,
                    -0.005945,
                ],
                _ => unreachable!(),
            });
            let beam_response = Jones::from([
                beam_response[3],
                beam_response[2],
                beam_response[1],
                beam_response[0],
            ]);
            write!(
                &mut di_jm_file,
                "{1:+.0$}, ",
                num_decimal_points, beam_response[3].re
            )?;
            write!(
                &mut di_jm_file,
                "{1:+.0$}, ",
                num_decimal_points, beam_response[3].im
            )?;
            write!(
                &mut di_jm_file,
                "{1:+.0$}, ",
                num_decimal_points, beam_response[2].re
            )?;
            write!(
                &mut di_jm_file,
                "{1:+.0$}, ",
                num_decimal_points, beam_response[2].im
            )?;
            write!(
                &mut di_jm_file,
                "{1:+.0$}, ",
                num_decimal_points, beam_response[1].re
            )?;
            write!(
                &mut di_jm_file,
                "{1:+.0$}, ",
                num_decimal_points, beam_response[1].im
            )?;
            write!(
                &mut di_jm_file,
                "{1:+.0$}, ",
                num_decimal_points, beam_response[0].re
            )?;
            writeln!(
                &mut di_jm_file,
                "{1:+.0$}",
                num_decimal_points, beam_response[0].im
            )?;

            // Write the unflagged fine channel frequencies in MHz.
            let mut unflagged_chan_line = String::new();
            for i_chan in 0..num_fine_chans_per_coarse_chan {
                if !sols
                    .flagged_chanblocks
                    .contains(&((i_chan + chan_offset).try_into().unwrap()))
                {
                    unflagged_chan_line.push_str(&format!(
                        "{1:.0$}, ",
                        num_decimal_points,
                        (i_chan as f64 * freq_res).round() / 1e6
                    ));
                }
            }
            writeln!(
                &mut bp_cal_file,
                "{}",
                unflagged_chan_line.trim_end_matches(", ")
            )?;

            // And now write tile Jones matrices.
            for (&i_input, &i_tile) in &input_to_tile {
                let flagged = sols.flagged_tiles.contains(&i_tile);

                // Find the "average" Jones matrix for the DI JM file.
                let data = data.slice(s![i_tile, ..]);

                let average = if flagged {
                    Jones::default()
                } else {
                    let (sum, length) = data.iter().enumerate().fold(
                        (Jones::default(), 0),
                        |(acc_sum, acc_length), (i_chan_in_cc, j)| {
                            if !sols
                                .flagged_chanblocks
                                .contains(&((i_chan_in_cc + chan_offset) as _))
                            {
                                (acc_sum + *j, acc_length + 1)
                            } else {
                                (acc_sum, acc_length)
                            }
                        },
                    );
                    let average = if length > 0 { sum / length as f64 } else { sum };
                    // average
                    // average.inv()
                    let average = Jones::from([
                        average[0].re,
                        -average[0].im,
                        average[1].re,
                        -average[1].im,
                        average[2].re,
                        -average[2].im,
                        average[3].re,
                        -average[3].im,
                    ]);
                    // ]) * beam_response.inv())
                    // .inv()
                    // dbg!(average, beam_response, (average * beam_response).inv());
                    dbg!(average, average.inv());
                    (average * beam_response).inv()
                    // average.inv()
                };
                if !flagged {
                    dbg!(i_input, i_tile, average);
                }
                writeln!(
                    &mut di_jm_file,
                    "{1:+.0$}, {2:+.0$}, {3:+.0$}, {4:+.0$}, {5:+.0$}, {6:+.0$}, {7:+.0$}, {8:+.0$}",
                    num_decimal_points,
                    // Don't forget to reorder into RTS PX, PY, QX, QY.
                    average[3].re,
                    average[3].im,
                    average[2].re,
                    average[2].im,
                    average[1].re,
                    average[1].im,
                    average[0].re,
                    average[0].im
                )?;

                // BP cal files don't include flagged tiles.
                if flagged {
                    continue;
                }

                // With the average, find the bandpass Jones matrices.
                let average = Jones::from([
                    average[0].re,
                    -average[0].im,
                    average[1].re,
                    -average[1].im,
                    average[2].re,
                    -average[2].im,
                    average[3].re,
                    -average[3].im,
                ]);
                let avg_inv = (average.inv() * beam_response.inv()).inv();
                // let avg_inv = average.inv() * beam_response;
                let bp_data = data.mapv(|j| avg_inv * j);
                // Write "fit" data the same as "lsq". No one cares...
                let mut bp_cal_line = String::new();
                for i in 0..8 {
                    let i_jones_elem = match i / 2 {
                        0 => 3,
                        1 => 2,
                        2 => 1,
                        3 => 0,
                        _ => unreachable!(),
                    };

                    bp_cal_line.push_str(&format!("{}, ", i_input + 1));
                    for (_, j) in bp_data.iter().enumerate().filter(|(i_chan, _)| {
                        !sols
                            .flagged_chanblocks
                            .contains(&((i_chan + chan_offset) as _))
                    }) {
                        bp_cal_line.push_str(&format!(
                            "{1:+.0$},{2:+.0$}, ",
                            num_decimal_points,
                            j[i_jones_elem].norm(),
                            j[i_jones_elem].arg()
                        ));
                    }
                    writeln!(&mut bp_cal_file, "{}", bp_cal_line.trim_end_matches(", "))?;
                    bp_cal_line.clear();
                }
            }
            di_jm_file.flush()?;
            bp_cal_file.flush()?;
        }

        Ok(())
    }
    inner(
        sols,
        dir.as_ref(),
        metafits.as_ref(),
        fee_beam_file.as_ref().map(|f| f.as_ref()),
        num_decimal_points,
    )
}

#[derive(Error, Debug)]
pub enum RtsReadSolsError {
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

    #[error("The DI_JonesMatrices files say that there are {got} tiles, but the metafits has {expected}. Unsure how to continue.")]
    UnequalMetafitsTileCount { got: usize, expected: usize },

    #[error("Unhandled RTS frequency resolution: {0} Hz")]
    UnhandledFreqRes(f64),

    #[error("Error when reading {filename}: {0}")]
    DiJmError {
        filename: PathBuf,
        err: ReadDiJmFileError,
    },

    #[error("Error when reading {filename}: {0}")]
    BpCalError {
        filename: PathBuf,
        err: ReadBpCalFileError,
    },

    #[error("{0}")]
    Glob(#[from] crate::glob::GlobError),

    #[error("Error when reading metafits: {0}")]
    Mwalib(#[from] mwalib::MwalibError),
}

#[derive(Error, Debug)]
pub enum RtsWriteSolsError {
    #[error("Attempted to write RTS solutions into '{0}', but this isn't a directory")]
    NotADir(PathBuf),

    #[error("Couldn't create directory '{dir}' (or its parents): {err}")]
    CouldntMakeDir { dir: PathBuf, err: std::io::Error },

    #[error("Error when reading metafits: {0}")]
    Mwalib(#[from] mwalib::MwalibError),

    #[error("{0}")]
    IO(#[from] std::io::Error),
}
