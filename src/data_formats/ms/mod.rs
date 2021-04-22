// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Code to interface with CASA measurement sets.
 */

pub(crate) mod error;
mod helpers;

pub use error::*;
use helpers::*;

use std::collections::BTreeSet;
use std::f64::consts::TAU;
use std::path::{Path, PathBuf};

use log::{debug, trace, warn};
use ndarray::prelude::*;

use super::*;
use crate::constants::HIFITIME_GPS_FACTOR;
use crate::context::{FreqContext, ObsContext};
use crate::glob::get_single_match_from_glob;
use mwa_hyperdrive_core::{c32, erfa_sys, mwalib, RADec, XYZ};

pub(crate) struct MS {
    /// Observation metadata.
    obs_context: ObsContext,

    /// Frequency metadata.
    freq_context: FreqContext,

    // MS-specific things follow.
    /// The path to the measurement set on disk.
    pub(crate) ms: PathBuf,

    /// The "stride" of the data, i.e. the number of rows (baselines) before the
    /// time index changes.
    step: usize,
}

impl MS {
    /// Verify and populate metadata associated with this measurement set. TODO:
    /// Use the metafits to get dead dipole info.
    ///
    /// The measurement set is expected to be formatted in the way that
    /// cotter/Birli write measurement sets.
    pub(crate) fn new<T: AsRef<Path>>(ms: &T, metafits: Option<&T>) -> Result<Self, NewMSError> {
        // The ms argument could be a glob. If the specified argument can't be
        // found as a file, treat it as a glob and expand it to find a match.
        let ms = {
            let pb = PathBuf::from(ms.as_ref());
            if pb.exists() {
                pb
            } else {
                get_single_match_from_glob(ms.as_ref().to_str().unwrap())?
            }
        };
        debug!("Using measurement set: {}", ms.display());
        if !ms.exists() {
            return Err(NewMSError::BadFile(ms));
        }

        let mut main_table = read_table(&ms, None)?;
        if main_table.n_rows() == 0 {
            return Err(NewMSError::Empty);
        }

        // This currently only returns table names. Maybe that's this function's
        // intention, but there should be a way to read the "String" types, not
        // just "Table" types from the table keywords.
        // Was this measurement set created by cotter?
        let cotter = {
            // TODO: Allow rubbl to read table keywords, otherwise this will
            // match Birli too.
            match read_table(&ms, Some("MWA_TILE_POINTING")) {
                Ok(_) => true,
                Err(_) => false,
            }
        };
        // let table_keywords = main_table.table_keyword_names().unwrap();
        // let cotter = table_keywords.contains(&"MWA_COTTER_VERSION".to_string());

        // Get the antenna XYZ positions (geocentric, not geodetic).
        let mut antenna_table = read_table(&ms, Some("ANTENNA"))?;
        let mut casacore_positions = Array2::zeros((antenna_table.n_rows() as usize, 3));
        let mut i = 0;
        antenna_table
            .for_each_row(|row| {
                // TODO: Kill the failure crate, and all unwraps!!
                let pos: Vec<f64> = row.get_cell("POSITION").unwrap();
                casacore_positions
                    .slice_mut(s![i, ..])
                    .assign(&Array1::from(pos));
                i += 1;
                Ok(())
            })
            .unwrap();
        let tile_xyz = casacore_positions_to_local_xyz(casacore_positions.view())?;
        let total_num_tiles = tile_xyz.len();
        debug!("There are {} total tiles", total_num_tiles);

        // Get the observation's flagged tiles. cotter doesn't populate the
        // ANTENNA table with this information; it looks like all tiles are
        // unflagged there. But, the flagged tiles don't appear in the main
        // table of baselines. Take the first n baeslines (where n is the length
        // of `xyz` above; which is the number of tiles) from the main table,
        // and find any missing antennas; these are the flagged tiles.
        // TODO: Handle autos not being present.
        let mut autocorrelations_present = false;
        let tile_flags: Vec<usize> = {
            let mut present_tiles = std::collections::HashSet::new();
            // N.B. The following method doesn't work if the antenna1 number
            // increases faster than antenna2.
            let mut first_antenna1 = None;
            for i in 0..total_num_tiles {
                let antenna1: i32 = main_table.get_cell("ANTENNA1", i as u64).unwrap();
                if first_antenna1.is_none() {
                    first_antenna1 = Some(antenna1);
                    present_tiles.insert(antenna1 as usize);
                }
                // We concern ourselves only with baselines with the first
                // antenna.
                if antenna1 != first_antenna1.unwrap() {
                    break;
                }
                let antenna2: i32 = main_table.get_cell("ANTENNA2", i as u64).unwrap();
                if antenna1 == antenna2 {
                    // TODO: Verify that this happens if cotter is told to not write
                    // autocorrelations.
                    autocorrelations_present = true;
                }
                present_tiles.insert(antenna2 as usize);
            }
            (0..total_num_tiles)
                .into_iter()
                .filter(|ant| !present_tiles.contains(ant))
                .collect()
        };
        debug!("Flagged tiles in the MS: {:?}", tile_flags);
        debug!("Autocorrelations present: {}", autocorrelations_present);

        // Now that we have the number of flagged tiles in the measurement set,
        // we can work out the first and last good timesteps. This is important
        // because cotter can pad the observation's data with visibilities that
        // should all be flagged, and we are not interested in using any of
        // those data. We work out the first and last good timesteps by
        // inspecting the flags at each timestep. TODO: Autocorrelations?
        let num_unflagged_tiles = total_num_tiles - tile_flags.len();
        let step = num_unflagged_tiles * (num_unflagged_tiles - 1) / 2
            + if autocorrelations_present {
                num_unflagged_tiles
            } else {
                0
            };
        trace!("MS step: {}", step);
        let timestep_indices = {
            // The first and last good timestep indicies.
            let mut first: Option<usize> = None;
            let mut last: Option<usize> = None;

            for i in 0..(main_table.n_rows() as usize + 1) / step {
                let vis_flags: Vec<bool> = main_table
                    .get_cell_as_vec(
                        "FLAG",
                        // Auto-correlations are more likely to be flagged than
                        // cross-correlations, so ignore the autos (if present).
                        (i * step + if autocorrelations_present { 1 } else { 0 }) as u64,
                    )
                    .unwrap();
                match (first, last, vis_flags.into_iter().all(|f| f == true)) {
                    (None, _, false) => first = Some(i),
                    (Some(_), None, true) => last = Some(i),
                    _ => (),
                }
            }

            // Did the indices get set correctly?
            match (first, last) {
                (Some(f), Some(l)) => f..l,
                // If there weren't any flags at the end of the MS, then the
                // last timestep is fine.
                (Some(f), None) => f..main_table.n_rows() as usize / step,
                _ => return Err(NewMSError::AllFlagged),
            }
        };
        debug!("MS timestep indices (exclusive): {:?}", timestep_indices);

        // Get the unique times in the MS.
        let utc_times: Vec<f64> = main_table.get_col_as_vec("TIME").unwrap();
        let mut time_set: BTreeSet<u64> = BTreeSet::new();
        for utc_time in utc_times {
            // Avoid float precision errors by multiplying by 1e3 and converting
            // to an int.
            time_set.insert((utc_time * 1e3).round() as _);
        }
        // Assume the timesteps are contiguous, i.e. the span of time between
        // two consequtive timesteps is the same between all consequtive
        // timesteps.
        let mut utc_timesteps: Vec<u64> = time_set
            .into_iter()
            .enumerate()
            .filter(|(i, _)| timestep_indices.contains(i))
            .map(|(_, t)| t)
            .collect();
        // Sort the timesteps ascendingly.
        utc_timesteps.sort_unstable_by(|a, b| a.partial_cmp(&b).unwrap());

        // Get the observation's native time resolution. There is a possibility
        // that the MS contains only one timestep.
        let time_res = if cotter && utc_timesteps.len() == 1 {
            warn!("Only one timestep is present in the data; can't determine the observation's native time resolution.");
            0.0
        } else {
            // Undo the multiply by 1e3.
            (utc_timesteps[1] - utc_timesteps[0]) as f64 / 1e3
        };

        let timesteps: Vec<hifitime::Epoch> = utc_timesteps
            .into_iter()
            // casacore keeps the stores the times as centroids, i.e. if data is
            // accessed for a 2s slice, then the time associated with the data
            // is at +1s (the middle of the slice). Also undo the multiply by
            // 1e3.
            .map(|utc| casacore_utc_to_epoch(utc as f64 / 1e3 - time_res / 2.0))
            .collect();
        // unwrap should be fine here, because we insist that the measurement
        // set is not empty at the start of this function.
        debug!(
            "First good GPS timestep: {}",
            // Need to remove a number from the result of .as_gpst_seconds(), as
            // it goes from the 1900 epoch, not the expected 1980 epoch.
            timesteps.first().unwrap().as_gpst_seconds() - HIFITIME_GPS_FACTOR
        );
        debug!(
            "Last good GPS timestep:  {}",
            timesteps.iter().last().unwrap().as_gpst_seconds() - HIFITIME_GPS_FACTOR
        );

        // Now that we have the timesteps, we can get the first LST. As
        // measurement sets populate centroids, the first timestep does not need
        // to be adjusted according to timewidth.
        let first_timestep_mjd = timesteps[0].as_mjd_utc_days();
        let lst0 = unsafe {
            let gmst = erfa_sys::eraGmst06(
                erfa_sys::ERFA_DJM0,
                first_timestep_mjd,
                erfa_sys::ERFA_DJM0,
                first_timestep_mjd,
            );
            (gmst + mwalib::MWA_LONGITUDE_RADIANS) % TAU
        };

        // Get the observation phase centre.
        let mut field_table = read_table(&ms, Some("FIELD"))?;
        let pointing_vec = field_table.get_cell_as_vec("PHASE_DIR", 0).unwrap();
        let pointing = RADec::new(pointing_vec[0], pointing_vec[1]);

        // Get the frequency information.
        let mut spectral_window_table = read_table(&ms, Some("SPECTRAL_WINDOW"))?;
        let fine_chan_freqs_hz: Vec<f64> = spectral_window_table
            .get_cell_as_vec("CHAN_FREQ", 0)
            .unwrap();
        // Assume that `total_bandwidth_hz` is the total bandwidth inside the
        // measurement set, which is not necessarily the whole observation.
        let total_bandwidth_hz: f64 = spectral_window_table
            .get_cell("TOTAL_BANDWIDTH", 0)
            .unwrap();
        debug!("MS total bandwidth: {} Hz", total_bandwidth_hz);

        // Note the "subband" is CASA nomenclature. MWA tends to use coarse
        // channel instead.
        // TODO: I think cotter always writes 24 coarse channels here. Hopefully
        // Birli is better...
        let coarse_chan_nums: Vec<u32> = {
            // If MWA_SUBBAND doesn't exist, then we must assume that this
            // measurement set only contains one coarse channel.
            match read_table(&ms, Some("MWA_SUBBAND")) {
                Err(_) => vec![1],
                Ok(mut mwa_subband_table) => {
                    let zero_indexed_coarse_chans: Vec<i32> =
                        mwa_subband_table.get_col_as_vec("NUMBER").unwrap();
                    zero_indexed_coarse_chans
                        .into_iter()
                        .map(|cc_num| (cc_num + 1) as _)
                        .collect()
                }
            }
        };
        debug!("MS coarse channels: {:?}", &coarse_chan_nums);

        // Get other metadata.
        let obsid: Option<u32> = {
            let mut observation_table = read_table(&ms, Some("OBSERVATION"))?;
            match observation_table.get_cell::<f64>("MWA_GPS_TIME", 0) {
                Err(_) => {
                    debug!("MS obsid not available (no MWA_GPS_TIME in OBSERVATION table)");
                    None
                }
                Ok(obsid_float) => {
                    let obsid_int = obsid_float as _;
                    debug!("MS obsid: {}", obsid_int);
                    Some(obsid_int)
                }
            }
        };

        let delays: Vec<u32> = {
            match read_table(&ms, Some("MWA_TILE_POINTING")) {
                // TODO: Use the phase centre to work out the sweet spot, and
                // get the delays that way.
                Err(_) => {
                    warn!("Assuming all dipole delays are 0 - TODO for Chris!");
                    vec![0; 16]
                }
                Ok(mut mwa_tile_pointing_table) => {
                    let delays_signed: Vec<i32> = mwa_tile_pointing_table
                        .get_cell_as_vec("DELAYS", 0)
                        .unwrap();
                    delays_signed.into_iter().map(|d| d as _).collect()
                }
            }
        };
        debug!("MS dipole delays: {:?}", &delays);

        let coarse_chan_width = total_bandwidth_hz / coarse_chan_nums.len() as f64;
        let native_fine_chan_width = if fine_chan_freqs_hz.len() == 1 {
            coarse_chan_width
        } else {
            fine_chan_freqs_hz[1] - fine_chan_freqs_hz[0]
        };
        let num_fine_chans_per_coarse_chan =
            (total_bandwidth_hz / coarse_chan_nums.len() as f64 / native_fine_chan_width).round()
                as _;
        let coarse_chan_freqs: Vec<f64> = fine_chan_freqs_hz
            .chunks_exact(num_fine_chans_per_coarse_chan)
            // round is OK because these values are Hz, and we're not ever
            // looking at sub-Hz resolution.
            .map(|chunk| chunk[chunk.len() / 2].round())
            .collect();
        let fine_chan_range = 0..fine_chan_freqs_hz.len();
        let freq_context = FreqContext {
            coarse_chan_nums,
            coarse_chan_freqs,
            coarse_chan_width,
            total_bandwidth: total_bandwidth_hz,
            fine_chan_range,
            fine_chan_freqs: fine_chan_freqs_hz,
            num_fine_chans_per_coarse_chan,
            native_fine_chan_width,
        };

        // Get the observation's flagged channels per coarse band.
        // TODO: Detect Birli.
        let fine_chan_flags_per_coarse_band = if cotter {
            // cotter doesn't list this conveniently. It's possible to inspect
            // the FLAG column in the main ms table, but, that would use huge
            // amount of IO, and there's no guarantee that any cell of that
            // column contains only default fine-channel flags; there may also
            // be RFI flags. So, we inspect the command-line options! (Sorry. At
            // least I wrote this comment.) cotter flags 80 kHz at the edges by
            // default, as well as the centre channel.
            let mut history_table = read_table(&ms, Some("HISTORY"))?;
            // For whatever reason, the CLI_COMMAND column needs to be accessed as a
            // vector, even though it only has one element.
            let cli_command: Vec<String> = history_table.get_cell_as_vec("CLI_COMMAND", 0).unwrap();
            debug!("cotter CLI command: {:?}", cli_command);

            // cotter CLI args are *always* split by whitespace; no equals
            // symbol allowed.
            let mut str_iter = cli_command[0].split_whitespace();
            match str_iter.next() {
                Some(exe) => exe.contains("cotter"),
                // TODO
                None => panic!("measurement set not from cotter!"),
            };

            // If -edgewidth is specified, then a custom amount of channels is
            // flagged at the coarse channel edges. The default is 80 kHz.
            let mut ew_iter = str_iter.clone();
            let edgewidth = match ew_iter.find(|&s| s.contains("-edgewidth")) {
                Some(_) => ew_iter.next().unwrap().parse::<f64>().unwrap() * 1e3, // kHz -> Hz;
                None => 80e3,
            };
            debug!("cotter -edgewidth (Hz): {}", edgewidth);
            // If -noflagdcchannels is specified, then the centre channel of
            // each coarse channel is not flagged. Without this flag, the centre
            // channels are flagged.
            let noflagdcchannels = match str_iter.find(|&s| s.contains("-noflagdcchannels")) {
                Some(_) => true,
                None => false,
            };
            debug!("cotter -noflagdcchannels: {}", noflagdcchannels);

            let num_edge_channels = (edgewidth / freq_context.native_fine_chan_width).round() as _;
            let mut fine_chan_flags = vec![];
            for ec in 0..num_edge_channels {
                fine_chan_flags.push(ec);
                fine_chan_flags.push(freq_context.num_fine_chans_per_coarse_chan - ec - 1);
            }
            if !noflagdcchannels {
                fine_chan_flags.push(freq_context.num_fine_chans_per_coarse_chan / 2);
            }
            fine_chan_flags.sort_unstable();
            fine_chan_flags
        } else {
            warn!("Assuming no fine channel flags - TODO on Chris!");
            vec![]
        };

        let baseline_xyz = XYZ::get_baselines(&tile_xyz);

        // Get dead dipole information. When interacting with beam code, use a
        // gain of 0 for dead dipoles, and 1 for all others. cotter doesn't
        // supply this information; if the user provided a metafits file, we can
        // use that, otherwise we must assume all dipoles are alive.
        let dipole_gains: Array2<f64> = match (cotter, metafits) {
            (true, None) => {
                warn!("cotter does not supply dead dipole information.");
                warn!("Without a metafits file, we must assume all dipoles are alive.");
                warn!("This will make beam Jones matrices inaccurate in sky-model prediction.");
                Array2::from_elem((num_unflagged_tiles, 16), 1.0)
            }

            (false, None) => {
                warn!("Without a metafits file, we must assume all dipoles are alive.");
                warn!("This will make beam Jones matrices inaccurate in sky-model prediction.");
                Array2::from_elem((num_unflagged_tiles, 16), 1.0)
            }

            (_, Some(m)) => {
                let mwalib = mwalib::MetafitsContext::new(m)?;
                let mut dipole_gains = Array2::from_elem(
                    (
                        mwalib.rf_inputs.len() / 2,
                        mwalib.rf_inputs[0].dipole_gains.len(),
                    ),
                    1.0,
                );
                for (mut dipole_gains_for_one_tile, rf_input) in dipole_gains.outer_iter_mut().zip(
                    mwalib
                        .rf_inputs
                        .iter()
                        .filter(|rf_input| rf_input.pol == mwalib::Pol::Y),
                ) {
                    dipole_gains_for_one_tile.assign(&ArrayView1::from(&rf_input.dipole_gains));
                }
                dipole_gains
            }
        };

        let num_unflagged_tiles = tile_xyz.len() - tile_flags.len();
        let obs_context = ObsContext {
            obsid,
            timesteps,
            timestep_indices,
            lst0,
            time_res,
            pointing,
            delays,
            tile_xyz,
            baseline_xyz,
            tile_flags,
            fine_chan_flags_per_coarse_band,
            num_unflagged_tiles,
            num_unflagged_baselines: num_unflagged_tiles * (num_unflagged_tiles - 1) / 2,
            dipole_gains,
        };

        Ok(Self {
            obs_context,
            freq_context,
            ms,
            step,
        })
    }
}

impl InputData for MS {
    fn get_obs_context(&self) -> &ObsContext {
        &self.obs_context
    }

    fn get_freq_context(&self) -> &FreqContext {
        &self.freq_context
    }

    fn read(
        &self,
        mut data_array: ArrayViewMut2<Vis<f32>>,
        timestep: usize,
        tile_to_baseline_map: &HashMap<(usize, usize), usize>,
        flagged_fine_chans: &HashSet<usize>,
    ) -> Result<(Vec<UVW>, Array2<f32>), ReadInputDataError> {
        // When reading in a new timestep's data, these indices should be
        // multiplied by `step` to get the amount of rows to stride in the main
        // table.
        let row_range_start = timestep * self.step;
        let row_range_end = (timestep + 1) * self.step;
        let row_range = row_range_start as u64..row_range_end as u64;
        trace!(
            "Reading timestep {} (row range {:?}) from the MS",
            timestep,
            row_range
        );

        let mut uvws = Vec::with_capacity(row_range_end - row_range_start);
        let mut out_weights = Array2::from_elem(data_array.dim(), 0.0);

        let mut main_table = read_table(&self.ms, None).unwrap();
        main_table
            .for_each_row_in_range(row_range, |row| {
                // Antenna numbers are zero indexed.
                let ant1: i32 = row.get_cell("ANTENNA1").unwrap();
                let ant2: i32 = row.get_cell("ANTENNA2").unwrap();
                // If this ant1-ant2 pair is in the baseline map, then the
                // baseline is not flagged, and we should proceed.
                if let Some(&bl) = tile_to_baseline_map.get(&(ant1 as usize, ant2 as usize)) {
                    // TODO: Filter on UVW lengths, as specified by the user.
                    let uvw: Vec<f64> = row.get_cell("UVW").unwrap();
                    // Use unsafe to stop Rust from doing bounds checks on uvw
                    // indices. If uvw has less than 3 elements, then something
                    // is seriously wrong with the MS.
                    unsafe {
                        uvws.push(UVW {
                            u: *uvw.get_unchecked(0),
                            v: *uvw.get_unchecked(1),
                            w: *uvw.get_unchecked(2),
                        });
                    }
                    // The data array is arranged [frequency][instrumental_pol].
                    let data: Array2<c32> = row.get_cell("DATA").unwrap();
                    // The weight array is arranged
                    // [frequency][instrumental_pol], however, the weights for
                    // all instrumental polarisations of a visibility are all
                    // the same. There isn't a way to just read one axis of the
                    // data, though.
                    let data_weights: Array2<f32> = row.get_cell("WEIGHT_SPECTRUM").unwrap();
                    // The flag array is arranged [frequency][instrumental_pol].
                    // As with the weights, the polarisation doesn't matter.
                    let flags: Array2<bool> = row.get_cell("FLAG").unwrap();

                    // Put the data and weights into the shared arrays outside
                    // this function. Before we can do this, we need to remove
                    // any globally-flagged fine channels. Use an int to index
                    // unflagged fine channel (outer_freq_chan_index).
                    let mut outer_freq_chan_index: usize = 0;
                    data.outer_iter()
                        .zip(data_weights.outer_iter())
                        .zip(flags.outer_iter())
                        .enumerate()
                        .for_each(
                            |(
                                freq_chan,
                                ((data_freq_axis, weights_freq_axis), flags_freq_axis),
                            )| {
                                if !flagged_fine_chans.contains(&freq_chan) {
                                    // If we're running in debug mode, assert that the
                                    // lengths of our arrays are sensible.
                                    debug_assert_eq!(data_freq_axis.len(), 4);
                                    debug_assert_eq!(weights_freq_axis.len(), 4);
                                    debug_assert_eq!(flags_freq_axis.len(), 4);
                                    debug_assert!(data_array.len_of(Axis(0)) >= bl);
                                    debug_assert!(
                                        data_array.len_of(Axis(1)) >= outer_freq_chan_index
                                    );

                                    // Skip bounds checks again.
                                    unsafe {
                                        // This is a reference to the visibility in the
                                        // output data array.
                                        let data_array_elem =
                                            data_array.uget_mut((bl, outer_freq_chan_index));
                                        // These are the components of the input data's
                                        // visibility.
                                        let data_xx_elem = data_freq_axis.uget(0);
                                        let data_xy_elem = data_freq_axis.uget(1);
                                        let data_yx_elem = data_freq_axis.uget(2);
                                        let data_yy_elem = data_freq_axis.uget(3);
                                        // This is the corresponding weight of the
                                        // visibility. It is the same for all
                                        // polarisations.
                                        let weight = weights_freq_axis.uget(0);
                                        // Get the element of the output weights
                                        // array, and write to it.
                                        let weight_elem =
                                            out_weights.uget_mut((bl, outer_freq_chan_index));
                                        *weight_elem = *weight;
                                        // The corresponding flag.
                                        let flag = flags_freq_axis.uget(0);
                                        // Adjust the weight by the flag.
                                        if *flag {
                                            *weight_elem = 0.0
                                        };
                                        // Multiply the input data visibility by the
                                        // weight and mutate the output data array.
                                        data_array_elem.xx = *data_xx_elem * *weight_elem;
                                        data_array_elem.xy = *data_xy_elem * *weight_elem;
                                        data_array_elem.yx = *data_yx_elem * *weight_elem;
                                        data_array_elem.yy = *data_yy_elem * *weight_elem;
                                    }
                                    outer_freq_chan_index += 1;
                                }
                            },
                        );
                }
                Ok(())
            })
            .unwrap();
        Ok((uvws, out_weights))
    }
}
