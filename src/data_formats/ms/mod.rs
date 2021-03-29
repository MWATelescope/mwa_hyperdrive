// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Code to interface with CASA measurement sets.
 */

pub mod error;
mod helpers;

pub use error::*;
use helpers::*;

use std::collections::BTreeSet;
use std::ops::Range;
use std::path::{Path, PathBuf};

use log::{debug, warn};
use ndarray::prelude::*;

use super::{error::*, *};
use crate::constants::HIFITIME_GPS_FACTOR;
use crate::context::{Context, FreqContext};
use crate::glob::get_single_match_from_glob;
use mwa_hyperdrive_core::{c32, RADec, XyzBaseline, XYZ};

pub(crate) struct MS {
    /// The path to the measurement set on disk.
    ms: PathBuf,

    /// Metadata on the observation.
    context: Context,

    // MS-specific things follow.
    /// The timestep indices of the MS that contain not-totally-flagged data.
    /// When reading in a new timestep's data, these indices should be
    /// multiplied by `step` to get the amount of rows to stride in the main
    /// table.
    timestep_indices: Range<usize>,

    /// The "stride" of the data, i.e. num. baselines.
    step: usize,
}

impl MS {
    /// Create a new instance of the `InputData` trait with a measurement set.
    ///
    /// The measurement set is expected to be formatted in the way that
    /// cotter/Birli write measurement sets.
    // pub(crate) fn new<T: AsRef<Path>>(ms: &T) -> Result<impl InputData, NewMSError> {
    pub(crate) fn new<T: AsRef<Path>>(ms: &T) -> Result<Self, NewMSError> {
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
        let cotter = true;
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
        let xyz = casacore_positions_to_local_xyz(casacore_positions.view())?;
        let total_num_tiles = xyz.len();
        debug!("There are {} total tiles", xyz.len());

        // Get the observation's flagged tiles. cotter doesn't populate the
        // ANTENNA table with this information; it looks like all tiles are
        // unflagged there. But, the flagged tiles don't appear in the main
        // table of baselines. Take the first n baeslines (where n is the length
        // of `xyz` above; which is the number of tiles) from the main table,
        // and find any missing antennas; these are the flagged tiles.
        // TODO: Handle autos not being present.
        let mut tile_flags = vec![];
        let mut autocorrelations_present = false;
        let mut last_antenna_num = -1;
        for i in 0..xyz.len() {
            let antenna1_cell: i32 = main_table.get_cell("ANTENNA1", i as u64).unwrap();
            if antenna1_cell > 0 {
                break;
            }
            let antenna2_cell: i32 = main_table.get_cell("ANTENNA2", i as u64).unwrap();
            if antenna1_cell == antenna2_cell {
                // TODO: Verify that this happens if cotter is told to not write
                // autocorrelations.
                autocorrelations_present = true;
            }
            if antenna2_cell != last_antenna_num + 1 {
                tile_flags.push(i + tile_flags.len());
            }
            last_antenna_num = antenna2_cell;
        }
        tile_flags.sort_unstable();
        debug!("Flagged tiles in the MS: {:?}", tile_flags);

        // Now we have the number of flagged tiles in the measurement set, we
        // can work out the first and last good timesteps. This is important
        // because cotter can pad the observation's data with visibilities that
        // should all be flagged, and we are not interested in using any of
        // those data. We work out the first and last good timesteps by
        // inspecting the flags at each timestep.
        // TODO: Autocorrelations?
        let num_unflagged_tiles = total_num_tiles - tile_flags.len();
        let step = num_unflagged_tiles * (num_unflagged_tiles + 1) / 2;
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
        let native_time_res = if cotter && utc_timesteps.len() == 1 {
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
            .map(|utc| casacore_utc_to_epoch(utc as f64 / 1e3 - native_time_res / 2.0))
            .collect();
        // unwrap should be fine here, because we insist that the measurement
        // set is not empty at the start of this function.
        debug!(
            "First good GPS timestep: {}",
            timesteps.first().unwrap().as_gpst_seconds() - HIFITIME_GPS_FACTOR
        );
        debug!(
            "Last good GPS timestep: {}",
            timesteps.iter().last().unwrap().as_gpst_seconds() - HIFITIME_GPS_FACTOR
        );

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

        // Note the "subband" is CASA nomenclature. MWA tends to use coarse
        // channel instead.
        // TODO: I think cotter always writes 24 coarse channels here. Hopefully
        // Birli is better...
        let mut mwa_subband_table = read_table(&ms, Some("MWA_SUBBAND"))?;
        let coarse_chan_nums: Vec<u32> = {
            let zero_indexed_coarse_chans: Vec<i32> =
                mwa_subband_table.get_col_as_vec("NUMBER").unwrap();
            zero_indexed_coarse_chans
                .into_iter()
                .map(|cc_num| (cc_num + 1) as _)
                .collect()
        };

        // Get other metadata.
        let mut observation_table = read_table(&ms, Some("OBSERVATION"))?;
        let obsid: u32 = observation_table
            .get_cell::<f64>("MWA_GPS_TIME", 0)
            .unwrap() as _;

        let mut mwa_tile_pointing_table = read_table(&ms, Some("MWA_TILE_POINTING"))?;
        let delays: Vec<u32> = {
            let delays_signed: Vec<i32> = mwa_tile_pointing_table
                .get_cell_as_vec("DELAYS", 0)
                .unwrap();
            delays_signed.into_iter().map(|d| d as _).collect()
        };

        let mut context = Context::new(
            obsid,
            timesteps,
            native_time_res,
            pointing,
            delays,
            xyz,
            tile_flags,
            vec![], // Populate this properly soon!
            coarse_chan_nums,
            total_bandwidth_hz,
            fine_chan_freqs_hz,
        );

        // Get the observation's flagged channels.
        // TODO: Detect Birli.
        let fine_chan_flags = if cotter {
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
            // If -noflagdcchannels is specified, then the centre channel of
            // each coarse channel is not flagged. Without this flag, the centre
            // channels are flagged.
            let noflagdcchannels = match str_iter.find(|&s| s.contains("-noflagdcchannels")) {
                Some(_) => true,
                None => false,
            };

            let freq_context = context.get_freq_context();
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
            panic!("Not using a cotter MS!");
        };
        context.fine_chan_flags = fine_chan_flags;

        Ok(Self {
            ms,
            context,
            timestep_indices,
            step,
        })
    }
}

impl InputData for MS {
    fn read(&self, time_range: Range<usize>) -> Result<Vec<Visibilities>, ReadInputDataError> {
        let row_range = (time_range.start * self.step) as u64..(time_range.end * self.step) as u64;
        debug!("Reading row range {:?} from the MS", row_range);
        let mut main_table = read_table(&self.ms, None).unwrap();
        // TODO: Work out .with_capacity.
        let mut visibilities = vec![];
        main_table
            .for_each_row_in_range(row_range, |row| {
                let ant1: i32 = row.get_cell("ANTENNA1").unwrap();
                let ant2: i32 = row.get_cell("ANTENNA2").unwrap();
                if ant1 != ant2 {
                    let uvw: Vec<f64> = row.get_cell("UVW").unwrap();
                }
                Ok(())
            })
            .unwrap();
        Ok(visibilities)
    }

    fn get_obsid(&self) -> u32 {
        self.context.get_obsid()
    }

    fn get_timesteps(&self) -> &[hifitime::Epoch] {
        self.context.get_timesteps()
    }

    fn get_timestep_indices(&self) -> &Range<usize> {
        &self.timestep_indices
    }

    fn get_native_time_res(&self) -> f64 {
        self.context.get_native_time_res()
    }

    fn get_pointing(&self) -> &RADec {
        self.context.get_pointing()
    }

    fn get_freq_context(&self) -> &FreqContext {
        self.context.get_freq_context()
    }

    fn get_tile_xyz(&self) -> &[XYZ] {
        self.context.get_tile_xyz()
    }

    fn get_baseline_xyz(&self) -> &[XyzBaseline] {
        self.context.get_baseline_xyz()
    }

    fn get_ideal_delays(&self) -> &[u32] {
        self.context.get_delays()
    }

    fn get_tile_flags(&self) -> &[usize] {
        &self.context.get_tile_flags()
    }

    fn get_fine_chan_flags(&self) -> &[usize] {
        &self.context.get_fine_chan_flags()
    }
}
