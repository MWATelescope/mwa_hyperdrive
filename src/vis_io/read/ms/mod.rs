// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to interface with CASA measurement sets.

mod error;
mod helpers;
#[cfg(test)]
mod tests;

pub(crate) use error::*;
use helpers::*;

use std::collections::{BTreeSet, HashSet};
use std::path::{Path, PathBuf};

use hifitime::{Duration, Epoch, Unit};
use log::{debug, trace};
use marlu::{
    c32,
    constants::{
        COTTER_MWA_HEIGHT_METRES, COTTER_MWA_LATITUDE_RADIANS, COTTER_MWA_LONGITUDE_RADIANS,
    },
    rubbl_casatables, Jones, LatLngHeight, RADec, XyzGeocentric,
};
use ndarray::prelude::*;
use rubbl_casatables::Table;

use super::*;
use crate::{context::ObsContext, metafits, time::round_hundredths_of_a_second};
use mwa_hyperdrive_beam::Delays;
use mwa_hyperdrive_common::{hifitime, log, marlu, mwalib, ndarray};

pub struct MsReader {
    /// Input data metadata.
    obs_context: ObsContext,

    /// The path to the measurement set on disk.
    pub(crate) ms: PathBuf,

    /// The "stride" of the data, i.e. the number of rows (baselines) before the
    /// time index changes.
    step: usize,

    /// The [`mwalib::MetafitsContext`] used when [`MsReader`] was created.
    metafits_context: Option<MetafitsContext>,
}

pub(crate) enum MsFlavour {
    /// Birli before version 0.2.0
    Birli,

    /// Anything that writes ms with the marlu library
    Marlu,

    Cotter,

    /// Generic?
    Casa,
}

impl MsReader {
    /// Verify and populate metadata associated with this measurement set.
    ///
    /// The measurement set is expected to be formatted in the way that
    /// cotter/Birli write measurement sets.
    // TODO: Handle multiple measurement sets.
    pub(crate) fn new<P: AsRef<Path>, P2: AsRef<Path>>(
        ms: P,
        metafits: Option<P2>,
    ) -> Result<MsReader, VisReadError> {
        fn inner(ms: &Path, metafits: Option<&Path>) -> Result<MsReader, MsReadError> {
            debug!("Using measurement set: {}", ms.display());
            if !ms.exists() {
                return Err(MsReadError::BadFile(ms.to_path_buf()));
            }

            // If a metafits file was provided, get an mwalib object ready.
            let mwalib_context = match metafits {
                None => None,
                // TODO: Let the user supply the MWA version
                Some(m) => Some(mwalib::MetafitsContext::new(&m, None)?),
            };

            let mut main_table = read_table(ms, None)?;
            if main_table.n_rows() == 0 {
                return Err(MsReadError::Empty);
            }

            // This currently only returns table names. Maybe that's this function's
            // intention, but there should be a way to read the "String" types, not
            // just "Table" types from the table keywords.
            // Was this measurement set created by cotter?
            let flavour = {
                let mut history_table = read_table(ms, Some("HISTORY"))?;
                let app: String = history_table.get_cell("APPLICATION", 0).unwrap();
                let app = app.to_uppercase();
                let app_name = if app.starts_with("BIRLI") {
                    Some(MsFlavour::Birli)
                } else if app.starts_with("MARLU") {
                    Some(MsFlavour::Marlu)
                } else if app.starts_with("COTTER") {
                    Some(MsFlavour::Cotter)
                } else {
                    None
                };

                // If there wasn't an app in the "APPLICATION" column, see if we
                // can get more information out of the "MESSAGE" column.
                app_name.unwrap_or_else(|| {
                    let messages: Vec<String> = history_table.get_col_as_vec("MESSAGE").unwrap();
                    let mut app = None;
                    for message in messages {
                        let upper = message.to_uppercase();
                        if upper.contains("MARLU") {
                            app = Some(MsFlavour::Marlu);
                            break;
                        } else if upper.contains("BIRLI") {
                            app = Some(MsFlavour::Birli);
                            break;
                        }
                    }
                    // If we *still* don't know what the app is, fallback on
                    // "Casa".
                    app.unwrap_or(MsFlavour::Casa)
                })
            };

            // Get the tile names and XYZ positions.
            let mut antenna_table = read_table(ms, Some("ANTENNA"))?;
            let tile_names: Vec<String> = antenna_table.get_col_as_vec("NAME").unwrap();
            let tile_names = Vec1::try_from_vec(tile_names).map_err(|_| MsReadError::Empty)?;

            let get_casacore_positions = |antenna_table: &mut Table, flavour: &MsFlavour| {
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
                let array_pos = match flavour {
                    MsFlavour::Birli | MsFlavour::Marlu => LatLngHeight::new_mwa(),
                    MsFlavour::Cotter => LatLngHeight {
                        longitude_rad: COTTER_MWA_LONGITUDE_RADIANS,
                        latitude_rad: COTTER_MWA_LATITUDE_RADIANS,
                        height_metres: COTTER_MWA_HEIGHT_METRES,
                    },
                    MsFlavour::Casa => todo!(),
                };
                casacore_positions_to_local_xyz(&casacore_positions, array_pos)
            };
            let tile_xyzs = match (&flavour, mwalib_context.as_ref()) {
                // If possible, use the metafits positions because even if the
                // MS's positions are derived from the same metafits values,
                // they're stored as geocentric positions, but we want geodetic
                // positions. The additional transform done for the MS means
                // that they're slightly less accurate.
                (MsFlavour::Birli | MsFlavour::Marlu, Some(context)) => {
                    marlu::XyzGeodetic::get_tiles_mwa(context)
                }
                (MsFlavour::Birli | MsFlavour::Marlu, None) => {
                    get_casacore_positions(&mut antenna_table, &flavour)?
                }
                (MsFlavour::Cotter, _) => get_casacore_positions(&mut antenna_table, &flavour)?,
                (MsFlavour::Casa, Some(context)) => marlu::XyzGeodetic::get_tiles_mwa(context),
                (MsFlavour::Casa, None) => todo!(),
            };
            let tile_xyzs = Vec1::try_from_vec(tile_xyzs).map_err(|_| MsReadError::Empty)?;
            let total_num_tiles = tile_xyzs.len();
            trace!("There are {} total tiles", total_num_tiles);

            // Get the observation's flagged tiles. cotter doesn't populate the
            // ANTENNA table with this information; it looks like all tiles are
            // unflagged there. But, the flagged tiles don't appear in the main
            // table of baselines. Take the first n baeslines (where n is the length
            // of `xyz` above, which is the number of tiles) from the main table,
            // and find any missing antennas; these are the flagged tiles.
            let mut autocorrelations_present = false;
            let flagged_tiles: Vec<usize> = {
                let mut present_tiles = HashSet::new();
                // N.B. The following method doesn't work if the antenna1 number
                // increases faster than antenna2.
                let mut first_antenna1 = -999;
                for i in 0..total_num_tiles {
                    let antenna1: i32 = main_table.get_cell("ANTENNA1", i as u64).unwrap();
                    if first_antenna1 == -999 {
                        first_antenna1 = antenna1;
                        present_tiles.insert(antenna1 as usize);
                    }
                    // We concern ourselves only with baselines with the first
                    // antenna.
                    if antenna1 != first_antenna1 {
                        break;
                    }
                    let antenna2: i32 = main_table.get_cell("ANTENNA2", i as u64).unwrap();
                    if !autocorrelations_present && antenna1 == antenna2 {
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
            let num_unflagged_tiles = total_num_tiles - flagged_tiles.len();
            debug!("Flagged tiles in the MS: {:?}", flagged_tiles);
            debug!("Autocorrelations present: {}", autocorrelations_present);

            // Get the observation phase centre.
            let phase_centre = {
                let mut field_table = read_table(ms, Some("FIELD"))?;
                let phase_vec = field_table.get_cell_as_vec("PHASE_DIR", 0).unwrap();
                RADec::new(phase_vec[0], phase_vec[1])
            };

            // Now that we have the number of flagged tiles in the measurement set,
            // we can work out the first and last good timesteps. This is important
            // because cotter can pad the observation's data with visibilities that
            // should all be flagged, and we are not interested in using any of
            // those data. We work out the first and last good timesteps by
            // inspecting the flags at each timestep.
            let step = num_unflagged_tiles * (num_unflagged_tiles - 1) / 2
                + if autocorrelations_present {
                    num_unflagged_tiles
                } else {
                    0
                };
            trace!("MS step: {}", step);
            let unflagged_timesteps: Vec<usize> = {
                // The first and last good timestep indices.
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
                    _ => return Err(MsReadError::AllFlagged),
                }
            }
            .into_iter()
            .collect();

            // Get the unique times in the MS.
            let utc_times: Vec<f64> = main_table.get_col_as_vec("TIME").unwrap();
            let mut utc_time_set: BTreeSet<u64> = BTreeSet::new();
            let mut timestamps = vec![];
            for utc_time in utc_times {
                let bits = utc_time.to_bits();
                if !utc_time_set.contains(&bits) {
                    utc_time_set.insert(bits);

                    // casacore stores the times as centroids, so no correction
                    // is needed.
                    let e = Epoch::from_utc_seconds(
                        // casacore stores the times as UTC seconds... but with an
                        // offset.
                        utc_time - hifitime::J1900_OFFSET * hifitime::SECONDS_PER_DAY,
                    );
                    // The values can be slightly off of their intended values;
                    // round them to the nearest hundredth.
                    timestamps.push(round_hundredths_of_a_second(e));
                }
            }
            let timestamps =
                Vec1::try_from_vec(timestamps).map_err(|_| MsReadError::NoTimesteps {
                    file: ms.display().to_string(),
                })?;
            match timestamps.as_slice() {
                // Handled above; measurement sets aren't allowed to be empty.
                [] => unreachable!(),
                [t] => debug!("Only timestep (GPS): {:.2}", t.as_gpst_seconds()),
                [t0, .., tn] => {
                    debug!("First good timestep (GPS): {:.2}", t0.as_gpst_seconds());
                    debug!("Last good timestep  (GPS): {:.2}", tn.as_gpst_seconds());
                }
            }

            // Get the data's time resolution. There is a possibility that the MS
            // contains only one timestep.
            let time_res = if timestamps.len() == 1 {
                debug!("Only one timestep is present in the data; can't determine the data's time resolution.");
                None
            } else {
                // Find the minimum gap between two consecutive timestamps.
                let time_res = timestamps.windows(2).fold(
                    Duration::from_f64(f64::INFINITY, Unit::Second),
                    |acc, ts| acc.min(ts[1] - ts[0]),
                );
                trace!("Time resolution: {}s", time_res.in_seconds());
                Some(time_res)
            };

            let all_timesteps = (0..timestamps.len()).into_iter().collect();
            let all_timesteps =
                Vec1::try_from_vec(all_timesteps).map_err(|_| MsReadError::NoTimesteps {
                    file: ms.display().to_string(),
                })?;

            // Get the frequency information.
            let mut spectral_window_table = read_table(ms, Some("SPECTRAL_WINDOW"))?;
            let fine_chan_freqs = {
                let fine_chan_freqs_hz: Vec<f64> = spectral_window_table
                    .get_cell_as_vec("CHAN_FREQ", 0)
                    .unwrap();
                let fine_chan_freqs = fine_chan_freqs_hz
                    .into_iter()
                    .map(|f| f.round() as u64)
                    .collect();
                Vec1::try_from_vec(fine_chan_freqs).map_err(|_| MsReadError::NoChannelFreqs)?
            };
            // Assume that `total_bandwidth_hz` is the total bandwidth inside the
            // measurement set, which is not necessarily the whole observation.
            let total_bandwidth_hz: f64 = spectral_window_table
                .get_cell("TOTAL_BANDWIDTH", 0)
                .unwrap();
            debug!("MS total bandwidth: {} Hz", total_bandwidth_hz);

            // Note the "subband" is CASA nomenclature. MWA tends to use "coarse
            // channel" instead.
            // TODO: I think cotter always writes 24 coarse channels here. Hopefully
            // Birli is better...
            let coarse_chan_nums: Vec<u32> = {
                // If MWA_SUBBAND doesn't exist, then we must assume that this
                // measurement set only contains one coarse channel.
                match read_table(ms, Some("MWA_SUBBAND")) {
                    Err(_) => vec![1],
                    Ok(mut mwa_subband_table) => {
                        let zero_indexed_coarse_chans: Vec<i32> =
                            mwa_subband_table.get_col_as_vec("NUMBER").unwrap();
                        let one_indexed_coarse_chans: Vec<u32> = zero_indexed_coarse_chans
                            .into_iter()
                            .map(|cc_num| (cc_num + 1) as _)
                            .collect();
                        if one_indexed_coarse_chans.is_empty() {
                            vec![1]
                        } else {
                            one_indexed_coarse_chans
                        }
                    }
                }
            };
            debug!("Coarse channel numbers: {:?}", coarse_chan_nums);
            let num_coarse_chans = coarse_chan_nums.len();

            // Get other metadata.
            let obsid: Option<u32> = {
                let mut observation_table = read_table(ms, Some("OBSERVATION"))?;
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

            // Populate the dipole delays and the pointing centre if we can.
            let mut dipole_delays: Option<Delays> = None;
            let mut pointing_centre: Option<RADec> = None;

            match (read_table(ms, Some("MWA_TILE_POINTING")), &mwalib_context) {
                // MWA_TILE_POINTING doesn't exist and no metafits file was
                // provided; we have no information on the delays. We also know
                // nothing about the pointing centre.
                (Err(_), None) => (),

                // Use the metafits file.
                (_, Some(context)) => {
                    debug!("Using metafits for dipole delays and pointing centre");
                    dipole_delays = Some(Delays::Full(metafits::get_dipole_delays(context)));
                    pointing_centre = Some(RADec::new_degrees(
                        context.ra_tile_pointing_degrees,
                        context.dec_tile_pointing_degrees,
                    ));
                }

                // MWA_TILE_POINTING exists.
                (Ok(mut mwa_tile_pointing_table), _) => {
                    debug!("Using MWA_TILE_POINTING for dipole delays and pointing centre");
                    let table_delays_signed: Vec<i32> = mwa_tile_pointing_table
                        .get_cell_as_vec("DELAYS", 0)
                        .unwrap();
                    let delays_unsigned: Vec<u32> = table_delays_signed
                        .into_iter()
                        .map(|d| d as u32)
                        .collect::<Vec<_>>();
                    // TODO: Error handling, check there are 16 delays, print a
                    // warning that only one set of delays are given?
                    dipole_delays = Some(Delays::Partial(delays_unsigned));

                    let pointing_vec: Vec<f64> = mwa_tile_pointing_table
                        .get_cell_as_vec("DIRECTION", 0)
                        .unwrap();
                    pointing_centre = Some(RADec::new(pointing_vec[0], pointing_vec[1]));
                }
            }

            // Get dipole gain information from the metafits, if it exists.
            // Measurement sets cannot supply dipole information.
            let dipole_gains = mwalib_context.as_ref().map(metafits::get_dipole_gains);

            // Round the values in here because sometimes they have a fractional
            // component, for some reason. We're unlikely to ever have a fraction of
            // a Hz as the channel resolution.
            let freq_res = {
                let all_widths: Vec<f64> = spectral_window_table
                    .get_cell_as_vec("CHAN_WIDTH", 0)
                    .unwrap();
                let width = *all_widths.get(0).ok_or(MsReadError::NoChanWidths)?;
                // Make sure all the widths all the same.
                for w in all_widths.iter().skip(1) {
                    if (w - width).abs() > f64::EPSILON {
                        return Err(MsReadError::ChanWidthsUnequal);
                    }
                }
                width
            };

            let num_fine_chans_per_coarse_chan =
                (total_bandwidth_hz / coarse_chan_nums.len() as f64 / freq_res).round() as _;
            let coarse_chan_freqs: Vec<f64> = match mwalib_context.as_ref() {
                Some(context) => {
                    // Get the coarse channel information out of the metafits
                    // file, but only the ones aligned with the frequencies in
                    // the uvfits file.
                    let cc_width = f64::from(context.coarse_chan_width_hz);

                    context
                        .metafits_coarse_chans
                        .iter()
                        .map(|cc| f64::from(cc.chan_centre_hz))
                        .filter(|cc_freq| {
                            fine_chan_freqs
                                .iter()
                                .any(|f| (*f as f64 - *cc_freq).abs() < cc_width / 2.0)
                        })
                        .collect()
                }
                None => {
                    // Divide each chunk of fine channels per coarse channel
                    fine_chan_freqs
                        .chunks_exact(num_fine_chans_per_coarse_chan)
                        .map(|chunk| {
                            if chunk.len() % 2 == 0 {
                                // We round the coarse channel freqs hoping
                                // there isn't any sub-Hz structure.
                                ((chunk[chunk.len() / 2 - 1] + chunk[chunk.len() / 2]) / 2) as f64
                            } else {
                                chunk[chunk.len() / 2] as f64
                            }
                        })
                        .collect()
                }
            };
            debug!(
                "Coarse channel centre frequencies [Hz]: {:?}",
                coarse_chan_freqs
            );

            // Get the observation's flagged channels per coarse band.
            let flagged_fine_chans: Vec<bool> = {
                // We assume here that the main_table contains a FLAG table.

                // Get the first unflagged timestep. If there aren't any, get
                // the middle one.
                let timestep = *unflagged_timesteps
                    .first()
                    .unwrap_or(&all_timesteps[all_timesteps.len() / 2]);

                // In this first unflagged timestep, get all the channel flags and
                // logically AND them together. If an entire channel is flagged due
                // to RFI, then we unfortunately will flag it for all timesteps.
                let row_range = (timestep * step) as u64..((timestep + 1) * step) as u64;
                let mut flagged_fine_chans: Vec<bool> = {
                    // The flags need to be read in as a 1D array, but there's
                    // actually 4 values per channel, because there's a flag for
                    // each pol. We don't care about individual pol flags; if any
                    // are flagged, flag the whole channel.
                    let flagged_fine_chans: Vec<bool> =
                        main_table.get_cell_as_vec("FLAG", row_range.start).unwrap();
                    flagged_fine_chans
                        .chunks_exact(4)
                        .map(|pol_flags| pol_flags.iter().any(|f| *f))
                        .collect()
                };
                main_table
                    .for_each_row_in_range(row_range, |row| {
                        let row_flagged_fine_chans: Array2<bool> = row.get_cell("FLAG").unwrap();
                        flagged_fine_chans
                            .iter_mut()
                            .zip(row_flagged_fine_chans.outer_iter())
                            .for_each(|(f1, f2)| {
                                let any_flagged = f2.iter().any(|f| *f);
                                *f1 &= any_flagged;
                            });
                        Ok(())
                    })
                    .unwrap();
                flagged_fine_chans
            };

            let flagged_fine_chans_per_coarse_chan = {
                let mut flagged_fine_chans_per_coarse_chan = vec![];
                for i_chan in 0..num_fine_chans_per_coarse_chan {
                    let mut chan_is_flagged = true;
                    for i_cc in 0..num_coarse_chans {
                        if !flagged_fine_chans[i_cc * num_fine_chans_per_coarse_chan + i_chan] {
                            chan_is_flagged = false;
                            break;
                        }
                    }
                    if chan_is_flagged {
                        flagged_fine_chans_per_coarse_chan.push(i_chan);
                    }
                }
                flagged_fine_chans_per_coarse_chan
            };
            let flagged_fine_chans = flagged_fine_chans
                .into_iter()
                .enumerate()
                .filter(|(_, f)| *f)
                .map(|(i, _)| i)
                .collect();

            let obs_context = ObsContext {
                obsid,
                timestamps,
                all_timesteps,
                unflagged_timesteps,
                phase_centre,
                pointing_centre,
                tile_names,
                tile_xyzs,
                flagged_tiles,
                autocorrelations_present,
                dipole_delays,
                dipole_gains,
                time_res,
                // XXX(Dev): no way to get array_pos from MS AFAIK
                array_position: None,
                coarse_chan_nums,
                coarse_chan_freqs,
                num_fine_chans_per_coarse_chan,
                freq_res: Some(freq_res),
                fine_chan_freqs,
                flagged_fine_chans,
                flagged_fine_chans_per_coarse_chan,
            };

            let ms = MsReader {
                obs_context,
                ms: ms.to_path_buf(),
                step,
                metafits_context: mwalib_context,
            };
            Ok(ms)
        }
        inner(ms.as_ref(), metafits.as_ref().map(|f| f.as_ref())).map_err(VisReadError::from)
    }

    /// An internal method for reading visibilities. Cross- and/or
    /// auto-correlation visibilities and weights are written to the supplied
    /// arrays.
    fn read_inner(
        &self,
        mut crosses: Option<CrossData>,
        mut autos: Option<AutoData>,
        timestep: usize,
        flagged_fine_chans: &HashSet<usize>,
    ) -> Result<(), VisReadError> {
        // When reading in a new timestep's data, these indices should be
        // multiplied by `step` to get the amount of rows to stride in the main
        // table.
        let row_range_start = timestep * self.step;
        let row_range_end = (timestep + 1) * self.step;
        let row_range = row_range_start as u64..row_range_end as u64;

        let mut main_table = read_table(&self.ms, None).unwrap();
        let mut row_index = row_range.start;
        main_table
            .for_each_row_in_range(row_range, |row| {
                // Antenna numbers are zero indexed.
                let ant1: i32 = row.get_cell("ANTENNA1").unwrap();
                let ant2: i32 = row.get_cell("ANTENNA2").unwrap();

                // Read this row if the baseline is unflagged.
                if let Some(crosses) = crosses.as_mut() {
                    if let Some(bl) = crosses
                        .tile_to_unflagged_baseline_map
                        .get(&(ant1 as usize, ant2 as usize))
                        .cloned()
                    {
                        // The data array is arranged [frequency][instrumental_pol].
                        let ms_data: Array2<c32> = row.get_cell("DATA").unwrap();
                        // The weight array is arranged
                        // [frequency][instrumental_pol], however, we assume the
                        // weights for all instrumental visibility polarisations
                        // are all the same. There isn't a way to just read one
                        // axis of the data.
                        let ms_weights: Array2<f32> = row.get_cell("WEIGHT_SPECTRUM").unwrap();
                        // The flag array is arranged
                        // [frequency][instrumental_pol]. As with the weights,
                        // the polarisation doesn't matter.
                        let flags: Array2<bool> = row.get_cell("FLAG").unwrap();

                        // Ensure that all arrays have appropriate sizes. We
                        // have to panic here because of the way Rubbl does
                        // error handling.
                        if ms_data.len_of(Axis(1)) != 4 {
                            panic!(
                                "{}",
                                MsReadError::BadArraySize {
                                    array_type: "ms_data",
                                    row_index,
                                    expected_len: 4,
                                    axis_num: 1,
                                }
                            );
                        }
                        if ms_weights.len_of(Axis(1)) != 4 {
                            panic!(
                                "{}",
                                MsReadError::BadArraySize {
                                    array_type: "weights",
                                    row_index,
                                    expected_len: 4,
                                    axis_num: 1,
                                }
                            );
                        }
                        if flags.len_of(Axis(1)) != 4 {
                            panic!(
                                "{}",
                                MsReadError::BadArraySize {
                                    array_type: "flags",
                                    row_index,
                                    expected_len: 4,
                                    axis_num: 1,
                                }
                            );
                        }
                        assert_eq!(ms_data.dim(), ms_weights.dim());
                        assert_eq!(ms_weights.dim(), flags.dim());
                        if crosses.data_array.len_of(Axis(0)) < bl {
                            panic!(
                                "{}",
                                VisReadError::BadArraySize {
                                    array_type: "data_array",
                                    expected_len: bl,
                                    axis_num: 0,
                                }
                            );
                        }
                        if crosses.data_array.len_of(Axis(1)) > ms_data.len_of(Axis(0)) {
                            panic!(
                                "{}",
                                VisReadError::BadArraySize {
                                    array_type: "data_array",
                                    expected_len: ms_data.len_of(Axis(0)),
                                    axis_num: 1,
                                }
                            );
                        }

                        // Put the data and weights into the shared arrays
                        // outside this scope. Before we can do this, we need to
                        // remove any globally-flagged fine channels.
                        let mut out_vis = crosses.data_array.slice_mut(s![bl, ..]);
                        ms_data
                            .outer_iter()
                            .enumerate()
                            .filter(|(i_chan, _)| !flagged_fine_chans.contains(i_chan))
                            .zip(out_vis.iter_mut())
                            .for_each(|((_, ms_data), out_vis)| {
                                *out_vis =
                                    Jones::from([ms_data[0], ms_data[1], ms_data[2], ms_data[3]]);
                            });

                        // Apply the flags to the weights (negate if flagged),
                        // and throw away 3 of the 4 weights; there are 4
                        // weights (for XX XY YX YY) and we assume that the
                        // first weight is the same as the others.
                        let mut out_weights = crosses.weights_array.slice_mut(s![bl, ..]);
                        ms_weights
                            .into_iter()
                            .step_by(4)
                            .zip(flags.into_iter().step_by(4))
                            .enumerate()
                            .filter(|(i_chan, _)| !flagged_fine_chans.contains(i_chan))
                            .zip(out_weights.iter_mut())
                            .for_each(|((_, (weight, flag)), out_weight)| {
                                *out_weight = if flag { -weight.abs() } else { weight };
                            });
                    }
                }

                if let Some(autos) = autos.as_mut() {
                    if ant1 == ant2 && !autos.flagged_tiles.contains(&(ant1 as usize)) {
                        let mut ant = ant1 as usize;
                        let ms_data: Array2<c32> = row.get_cell("DATA").unwrap();
                        let ms_weights: Array2<f32> = row.get_cell("WEIGHT_SPECTRUM").unwrap();
                        let flags: Array2<bool> = row.get_cell("FLAG").unwrap();

                        if ms_data.len_of(Axis(1)) != 4 {
                            panic!(
                                "{}",
                                MsReadError::BadArraySize {
                                    array_type: "ms_data",
                                    row_index,
                                    expected_len: 4,
                                    axis_num: 1,
                                }
                            );
                        }
                        if ms_weights.len_of(Axis(1)) != 4 {
                            panic!(
                                "{}",
                                MsReadError::BadArraySize {
                                    array_type: "weights",
                                    row_index,
                                    expected_len: 4,
                                    axis_num: 1,
                                }
                            );
                        }
                        if flags.len_of(Axis(1)) != 4 {
                            panic!(
                                "{}",
                                MsReadError::BadArraySize {
                                    array_type: "flags",
                                    row_index,
                                    expected_len: 4,
                                    axis_num: 1,
                                }
                            );
                        }
                        assert_eq!(ms_data.dim(), ms_weights.dim());
                        assert_eq!(ms_weights.dim(), flags.dim());

                        // `ant` needs to be adjusted based on how many flagged
                        // tiles were before it; it is then a valid index into
                        // passed-in arrays.
                        ant -= (0..ant)
                            .filter(|i_ant| autos.flagged_tiles.contains(i_ant))
                            .count();

                        if autos.data_array.len_of(Axis(0)) < ant {
                            panic!(
                                "{}",
                                VisReadError::BadArraySize {
                                    array_type: "data_array",
                                    expected_len: ant,
                                    axis_num: 0,
                                }
                            );
                        }
                        if autos.data_array.len_of(Axis(1)) > ms_data.len_of(Axis(0)) {
                            panic!(
                                "{}",
                                VisReadError::BadArraySize {
                                    array_type: "data_array",
                                    expected_len: ms_data.len_of(Axis(0)),
                                    axis_num: 1,
                                }
                            );
                        }

                        let mut out_vis = autos.data_array.slice_mut(s![ant, ..]);
                        ms_data
                            .outer_iter()
                            .enumerate()
                            .filter(|(i_chan, _)| !flagged_fine_chans.contains(i_chan))
                            .zip(out_vis.iter_mut())
                            .for_each(|((_, ms_data), out_vis)| {
                                *out_vis =
                                    Jones::from([ms_data[0], ms_data[1], ms_data[2], ms_data[3]]);
                            });

                        let mut out_weights = autos.weights_array.slice_mut(s![ant, ..]);
                        ms_weights
                            .into_iter()
                            .step_by(4)
                            .zip(flags.into_iter().step_by(4))
                            .enumerate()
                            .filter(|(i_chan, _)| !flagged_fine_chans.contains(i_chan))
                            .zip(out_weights.iter_mut())
                            .for_each(|((_, (weight, flag)), out_weight)| {
                                *out_weight = if flag { -weight.abs() } else { weight };
                            });
                    }
                }

                row_index += 1;
                Ok(())
            })
            .unwrap();

        Ok(())
    }
}

impl VisRead for MsReader {
    fn get_obs_context(&self) -> &ObsContext {
        &self.obs_context
    }

    fn get_input_data_type(&self) -> VisInputType {
        VisInputType::MeasurementSet
    }

    fn get_metafits_context(&self) -> Option<&MetafitsContext> {
        self.metafits_context.as_ref()
    }

    fn read_crosses_and_autos(
        &self,
        cross_data_array: ArrayViewMut2<Jones<f32>>,
        cross_weights_array: ArrayViewMut2<f32>,
        auto_data_array: ArrayViewMut2<Jones<f32>>,
        auto_weights_array: ArrayViewMut2<f32>,
        timestep: usize,
        tile_to_unflagged_baseline_map: &HashMap<(usize, usize), usize>,
        flagged_tiles: &[usize],
        flagged_fine_chans: &HashSet<usize>,
    ) -> Result<(), VisReadError> {
        self.read_inner(
            Some(CrossData {
                data_array: cross_data_array,
                weights_array: cross_weights_array,
                tile_to_unflagged_baseline_map,
            }),
            Some(AutoData {
                data_array: auto_data_array,
                weights_array: auto_weights_array,
                flagged_tiles,
            }),
            timestep,
            flagged_fine_chans,
        )
    }

    fn read_crosses(
        &self,
        data_array: ArrayViewMut2<Jones<f32>>,
        weights_array: ArrayViewMut2<f32>,
        timestep: usize,
        tile_to_unflagged_baseline_map: &HashMap<(usize, usize), usize>,
        flagged_fine_chans: &HashSet<usize>,
    ) -> Result<(), VisReadError> {
        self.read_inner(
            Some(CrossData {
                data_array,
                weights_array,
                tile_to_unflagged_baseline_map,
            }),
            None,
            timestep,
            flagged_fine_chans,
        )
    }

    fn read_autos(
        &self,
        data_array: ArrayViewMut2<Jones<f32>>,
        weights_array: ArrayViewMut2<f32>,
        timestep: usize,
        flagged_tiles: &[usize],
        flagged_fine_chans: &HashSet<usize>,
    ) -> Result<(), VisReadError> {
        self.read_inner(
            None,
            Some(AutoData {
                data_array,
                weights_array,
                flagged_tiles,
            }),
            timestep,
            flagged_fine_chans,
        )
    }
}
