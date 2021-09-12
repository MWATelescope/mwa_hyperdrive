// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to interface with CASA measurement sets.

pub(crate) mod error;
mod helpers;

pub use error::*;
use helpers::*;

use std::collections::{BTreeSet, HashSet};
use std::path::{Path, PathBuf};

use log::{debug, trace, warn};
use ndarray::prelude::*;
use rayon::prelude::*;

use super::*;
use crate::{
    context::{FreqContext, ObsContext},
    data_formats::metafits,
    glob::get_single_match_from_glob,
};
use mwa_hyperdrive_beam::Delays;
use mwa_rust_core::{
    c32,
    constants::{
        COTTER_MWA_HEIGHT_METRES, COTTER_MWA_LATITUDE_RADIANS, COTTER_MWA_LONGITUDE_RADIANS,
    },
    time::{casacore_utc_to_epoch, epoch_as_gps_seconds},
    Jones, RADec, XyzGeocentric,
};

const COTTER_DEFAULT_EDGEWIDTH: f64 = 80e3;
const TIMESTEP_AS_INT_FACTOR: f64 = 1e6;

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
    /// Verify and populate metadata associated with this measurement set.
    ///
    /// The measurement set is expected to be formatted in the way that
    /// cotter/Birli write measurement sets.
    // TODO: Handle multiple measurement sets.
    pub(crate) fn new<T: AsRef<Path>>(
        ms: &T,
        metafits: Option<&T>,
        dipole_delays: &mut Delays,
    ) -> Result<Self, NewMSError> {
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
            read_table(&ms, Some("MWA_TILE_POINTING")).is_ok()
        };
        // let table_keywords = main_table.table_keyword_names().unwrap();
        // let cotter = table_keywords.contains(&"MWA_COTTER_VERSION".to_string());

        // Get the tile names and XYZ positions.
        let mut antenna_table = read_table(&ms, Some("ANTENNA"))?;
        let tile_names: Vec<String> = antenna_table.get_col_as_vec("NAME").unwrap();
        let mut casacore_positions = Vec::with_capacity(antenna_table.n_rows() as usize);
        antenna_table
            .for_each_row(|row| {
                // TODO: Kill the failure crate, and all unwraps!!
                let pos: Vec<f64> = row.get_cell("POSITION").unwrap();
                let pos_xyz = XyzGeocentric {
                    x: pos[0],
                    y: pos[1],
                    z: pos[2],
                };
                casacore_positions.push(pos_xyz);
                Ok(())
            })
            .unwrap();
        let tile_xyzs = if cotter {
            casacore_positions_to_local_xyz(
                &casacore_positions,
                COTTER_MWA_LONGITUDE_RADIANS,
                COTTER_MWA_LATITUDE_RADIANS,
                COTTER_MWA_HEIGHT_METRES,
            )?
        } else {
            // TODO: Get actual array coordinates.
            casacore_positions_to_local_xyz_mwa(&casacore_positions)?
        };
        let total_num_tiles = tile_xyzs.len();
        trace!("There are {} total tiles", total_num_tiles);

        // Get the observation's flagged tiles. cotter doesn't populate the
        // ANTENNA table with this information; it looks like all tiles are
        // unflagged there. But, the flagged tiles don't appear in the main
        // table of baselines. Take the first n baeslines (where n is the length
        // of `xyz` above; which is the number of tiles) from the main table,
        // and find any missing antennas; these are the flagged tiles.
        // TODO: Handle autos not being present.
        let mut autocorrelations_present = false;
        let tile_flags: Vec<usize> = {
            let mut present_tiles = HashSet::new();
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
        let num_unflagged_tiles = total_num_tiles - tile_flags.len();
        debug!("Flagged tiles in the MS: {:?}", tile_flags);
        debug!("Autocorrelations present: {}", autocorrelations_present);

        // Get the observation phase centre.
        let phase_centre = {
            let mut field_table = read_table(&ms, Some("FIELD"))?;
            let phase_vec = field_table.get_cell_as_vec("PHASE_DIR", 0).unwrap();
            RADec::new(phase_vec[0], phase_vec[1])
        };

        // Now that we have the number of flagged tiles in the measurement set,
        // we can work out the first and last good timesteps. This is important
        // because cotter can pad the observation's data with visibilities that
        // should all be flagged, and we are not interested in using any of
        // those data. We work out the first and last good timesteps by
        // inspecting the flags at each timestep. TODO: Autocorrelations?
        let step = num_unflagged_tiles * (num_unflagged_tiles - 1) / 2
            + if autocorrelations_present {
                num_unflagged_tiles
            } else {
                0
            };
        trace!("MS step: {}", step);
        let unflagged_timestep_indices = {
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
                match (first, last, vis_flags.into_iter().all(|f| f)) {
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

        // Get the unique times in the MS.
        let utc_times: Vec<f64> = main_table.get_col_as_vec("TIME").unwrap();
        let mut time_set: BTreeSet<u64> = BTreeSet::new();
        for utc_time in utc_times {
            // Avoid float precision errors by multiplying by a big number and
            // converting to an int.
            time_set.insert((utc_time * TIMESTEP_AS_INT_FACTOR).round() as _);
        }
        // Assume the timesteps are contiguous, i.e. the span of time between
        // two consecutive timesteps is the same between all consecutive
        // timesteps.
        let mut utc_timesteps: Vec<u64> = time_set.into_iter().collect();
        // Sort the timesteps ascendingly.
        utc_timesteps.sort_unstable();

        // Get the observation's native time resolution. There is a possibility
        // that the MS contains only one timestep.
        let time_res = if utc_timesteps.len() == 1 {
            warn!("Only one timestep is present in the data; can't determine the observation's native time resolution.");
            None
        } else {
            // Undo the multiply by the big number.
            let tr = (utc_timesteps[1] - utc_timesteps[0]) as f64 / TIMESTEP_AS_INT_FACTOR;
            Some(tr)
        };

        let timesteps: Vec<hifitime::Epoch> = utc_timesteps
            .into_par_iter()
            // casacore keeps the stores the times as centroids, so no
            // correction is needed. Undo the multiply by a big number from
            // above.
            .map(|utc| casacore_utc_to_epoch(utc as f64 / TIMESTEP_AS_INT_FACTOR))
            .collect();
        if let Some(time_res) = time_res {
            debug!(
                "First good timestep (GPS): {:.2}",
                // Need to remove a number from the result of .as_gpst_seconds(), as
                // it goes from the 1900 epoch, not the expected 1980 epoch. Also we
                // expect GPS timestamps to be "leading edge", not centroids.
                epoch_as_gps_seconds(timesteps[unflagged_timestep_indices.start]) - time_res / 2.0
            );
            debug!(
                "Last good timestep  (GPS): {:.2}",
                epoch_as_gps_seconds(timesteps[unflagged_timestep_indices.end - 1])
                    - time_res / 2.0
            );
        } else {
            // No time resolution; just print out the first GPS timestep.
            debug!(
                "Only timestep (GPS): {:.2}",
                epoch_as_gps_seconds(timesteps[0])
            );
        }

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

        // If a metafits file was provided, we _may_ use it. Get a _potential_
        // mwalib object ready.
        let mut mwalib = None;

        // Populate the dipole delays if we need to, and get the pointing centre
        // if we can.
        let pointing_centre: Option<RADec> =
            match (read_table(&ms, Some("MWA_TILE_POINTING")), metafits) {
                (Err(_), None) => {
                    // MWA_TILE_POINTING doesn't exist and no metafits file was
                    // provided; no changes to the delays can be made here. We also
                    // know nothing about the pointing centre.
                    None
                }

                // MWA_TILE_POINTING exists - use this over the metafits
                // file even if it's provided.
                (Ok(mut mwa_tile_pointing_table), _) => {
                    // Only use the measurement set delays if the delays struct
                    // provided to this function was empty.
                    match dipole_delays {
                        Delays::Available(_) | Delays::NotNecessary => (),
                        Delays::None => {
                            debug!("Using MWA_TILE_POINTING for dipole delays");
                            let delays_signed: Vec<i32> = mwa_tile_pointing_table
                                .get_cell_as_vec("DELAYS", 0)
                                .unwrap();
                            let delays_unsigned: Vec<u32> =
                                delays_signed.into_iter().map(|d| d as u32).collect();
                            *dipole_delays = Delays::Available(delays_unsigned);
                        }
                    }
                    let pointing_vec: Vec<f64> = mwa_tile_pointing_table
                        .get_cell_as_vec("DIRECTION", 0)
                        .unwrap();
                    Some(RADec::new(pointing_vec[0], pointing_vec[1]))
                }

                // Use the metafits file.
                (Err(_), Some(meta)) => {
                    // TODO: Let the user supply the MWA version
                    let context = metafits::populate_metafits_context(&mut mwalib, meta, None)?;
                    // Only use the metafits delays if none were provided to
                    // this function.
                    match dipole_delays {
                        Delays::Available(_) | Delays::NotNecessary => (),
                        Delays::None => {
                            debug!("Using metafits for dipole delays");
                            let metafits_delays = metafits::get_true_delays(context);
                            *dipole_delays = Delays::Available(metafits_delays);
                        }
                    }
                    Some(RADec::new_degrees(
                        context.ra_tile_pointing_degrees,
                        context.dec_tile_pointing_degrees,
                    ))
                }
            };
        match &dipole_delays {
            Delays::Available(d) => debug!("Dipole delays: {:?}", d),
            Delays::NotNecessary => {
                debug!("Dipole delays weren't searched for in input data; not necessary")
            }
            Delays::None => warn!("Dipole delays not provided and not available in input data!"),
        }

        // Get dipole information. When interacting with beam code, use a gain
        // of 0 for dead dipoles, and 1 for all others. cotter doesn't supply
        // this information; if the user provided a metafits file, we can use
        // that, otherwise we must assume all dipoles are alive.
        let dipole_gains: Array2<f64> = match metafits {
            None => {
                if cotter {
                    warn!("cotter does not supply dead dipole information.");
                }
                warn!("Without a metafits file, we must assume all dipoles are alive.");
                warn!("This will make beam Jones matrices inaccurate in sky-model generation.");
                Array2::ones((num_unflagged_tiles, 16))
            }

            Some(meta) => {
                let context = metafits::populate_metafits_context(&mut mwalib, meta, None)?;
                metafits::get_dipole_gains(context)
            }
        };

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
            native_fine_chan_width: Some(native_fine_chan_width),
        };

        // Get the observation's flagged channels per coarse band.
        // TODO: Detect Birli.
        let fine_chan_flags_per_coarse_chan = if cotter {
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
                Some(_) => {
                    let ew_str = ew_iter.next().unwrap();
                    // The string may be quoted; remove those so it can be
                    // parsed as a float.
                    let ew_str = ew_str.trim_matches('\"');
                    let ew = ew_str.parse::<f64>().unwrap() * 1e3;
                    debug!("cotter -edgewidth: {} kHz", ew);
                    ew
                }
                None => {
                    debug!(
                        "Assuming cotter is using default edgewidth of {} kHz",
                        COTTER_DEFAULT_EDGEWIDTH / 1e3
                    );
                    COTTER_DEFAULT_EDGEWIDTH
                }
            };
            // If -noflagdcchannels is specified, then the centre channel of
            // each coarse channel is not flagged. Without this flag, the centre
            // channels are flagged.
            let noflagdcchannels = str_iter.any(|s| s.contains("-noflagdcchannels"));
            debug!("cotter -noflagdcchannels: {}", noflagdcchannels);

            let num_edge_channels = (edgewidth / native_fine_chan_width).round() as _;
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

        let obs_context = ObsContext {
            obsid,
            timesteps,
            unflagged_timestep_indices,
            phase_centre,
            pointing_centre,
            tile_names,
            tile_xyzs,
            tile_flags,
            fine_chan_flags_per_coarse_chan,
            dipole_gains,
            time_res,
            // TODO
            array_longitude_rad: None,
            array_latitude_rad: None,
        };

        let ms = Self {
            obs_context,
            freq_context,
            ms,
            step,
        };
        Ok(ms)
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
        mut data_array: ArrayViewMut2<Jones<f32>>,
        timestep: usize,
        tile_to_unflagged_baseline_map: &HashMap<(usize, usize), usize>,
        flagged_fine_chans: &HashSet<usize>,
    ) -> Result<Array2<f32>, ReadInputDataError> {
        // When reading in a new timestep's data, these indices should be
        // multiplied by `step` to get the amount of rows to stride in the main
        // table.
        let row_range_start = timestep * self.step;
        let row_range_end = (timestep + 1) * self.step;
        let row_range = row_range_start as u64..row_range_end as u64;

        let mut out_weights = Array2::zeros(data_array.dim());

        let mut main_table = read_table(&self.ms, None).unwrap();
        let mut row_index = row_range.start;
        main_table
            .for_each_row_in_range(row_range, |row| {
                // Antenna numbers are zero indexed.
                let ant1: i32 = row.get_cell("ANTENNA1").unwrap();
                let ant2: i32 = row.get_cell("ANTENNA2").unwrap();
                // If this ant1-ant2 pair is in the baseline map, then the
                // baseline is not flagged, and we should proceed.
                if let Some(&bl) =
                    tile_to_unflagged_baseline_map.get(&(ant1 as usize, ant2 as usize))
                {
                    // TODO: Filter on UVW lengths, as specified by the user.
                    let uvw: Vec<f64> = row.get_cell("UVW").unwrap();
                    if uvw.len() < 3 {
                        return Err(MSError::NotThreeUVW { row_index }.into());
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
                    // this scope. Before we can do this, we need to remove any
                    // globally-flagged fine channels. Use an int to index
                    // unflagged fine channel (outer_freq_chan_index).
                    let mut outer_freq_chan_index: usize = 0;
                    for (freq_chan, ((data_freq_axis, weights_freq_axis), flags_freq_axis)) in data
                        .outer_iter()
                        .zip(data_weights.outer_iter())
                        .zip(flags.outer_iter())
                        .enumerate()
                    {
                        if !flagged_fine_chans.contains(&freq_chan) {
                            // Ensure that all arrays have appropriate
                            // sizes.
                            // We have to panic here because of the way Rubbl
                            // does error handling.
                            if data_freq_axis.len() != 4 {
                                panic!(
                                    "{}",
                                    MSError::BadArraySize {
                                        array_type: "data",
                                        row_index,
                                        expected_len: 4,
                                        axis_num: 1,
                                    }
                                );
                            }
                            if weights_freq_axis.len() != 4 {
                                panic!(
                                    "{}",
                                    MSError::BadArraySize {
                                        array_type: "weights",
                                        row_index,
                                        expected_len: 4,
                                        axis_num: 1,
                                    }
                                );
                            }
                            if flags_freq_axis.len() != 4 {
                                panic!(
                                    "{}",
                                    MSError::BadArraySize {
                                        array_type: "flags",
                                        row_index,
                                        expected_len: 4,
                                        axis_num: 1,
                                    }
                                );
                            }
                            if data_array.len_of(Axis(0)) < bl {
                                panic!(
                                    "{}",
                                    ReadInputDataError::BadArraySize {
                                        array_type: "data",
                                        expected_len: bl,
                                        axis_num: 0,
                                    }
                                );
                            }
                            if data_array.len_of(Axis(1)) < outer_freq_chan_index {
                                panic!(
                                    "{}",
                                    ReadInputDataError::BadArraySize {
                                        array_type: "data",
                                        expected_len: outer_freq_chan_index,
                                        axis_num: 1,
                                    }
                                );
                            }

                            // This is a reference to the visibility in
                            // the output data array.
                            let data_array_elem =
                                data_array.get_mut((bl, outer_freq_chan_index)).unwrap();
                            // These are the components of the input
                            // data's visibility.
                            let data_xx_elem = data_freq_axis.get(0).unwrap();
                            let data_xy_elem = data_freq_axis.get(1).unwrap();
                            let data_yx_elem = data_freq_axis.get(2).unwrap();
                            let data_yy_elem = data_freq_axis.get(3).unwrap();
                            // This is the corresponding weight of the
                            // visibility. It is the same for all polarisations.
                            let weight = weights_freq_axis.get(0).unwrap();
                            // Get the element of the output weights array, and
                            // write to it.
                            let weight_elem =
                                out_weights.get_mut((bl, outer_freq_chan_index)).unwrap();
                            *weight_elem = *weight;
                            // The corresponding flag.
                            let flag = flags_freq_axis.get(0).unwrap();
                            // Adjust the weight by the flag.
                            if *flag {
                                *weight_elem = 0.0
                            };
                            // Multiply the input data visibility by the weight
                            // and mutate the output data array.
                            data_array_elem[0] = *data_xx_elem * *weight_elem;
                            data_array_elem[1] = *data_xy_elem * *weight_elem;
                            data_array_elem[2] = *data_yx_elem * *weight_elem;
                            data_array_elem[3] = *data_yy_elem * *weight_elem;

                            outer_freq_chan_index += 1;
                        }
                    }
                }
                row_index += 1;
                Ok(())
            })
            .unwrap();
        Ok(out_weights)
    }
}
