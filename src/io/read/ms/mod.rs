// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to interface with CASA measurement sets.
//!
//! More info: https://casa.nrao.edu/Memos/229.html#SECTION00060000000000000000

mod error;
#[cfg(test)]
mod tests;

pub(crate) use error::*;

use std::{
    collections::{BTreeSet, HashMap},
    path::{Path, PathBuf},
};

use hifitime::{Duration, Epoch};
use log::{debug, trace, warn};
use marlu::{
    c32,
    constants::{
        COTTER_MWA_HEIGHT_METRES, COTTER_MWA_LATITUDE_RADIANS, COTTER_MWA_LONGITUDE_RADIANS,
    },
    rubbl_casatables, Jones, LatLngHeight, RADec, XyzGeocentric,
};
use ndarray::prelude::*;
use rayon::prelude::*;
use rubbl_casatables::{Table, TableOpenMode};

use super::*;
use crate::{beam::Delays, context::ObsContext, metafits, misc::round_hundredths_of_a_second};

const SUPPORTED_WEIGHT_COL_NAMES: [&str; 2] = ["WEIGHT_SPECTRUM", "WEIGHT"];

pub(crate) enum MsFlavour {
    Hyperdrive,

    /// Birli before version 0.2.0 and after 0.7.0
    Birli,

    /// Anything that writes ms with the Marlu library without specifying the
    /// appropriate history
    Marlu,

    Cotter,

    /// Generic?
    Casa,
}

/// Open a measurement set table read only. If `table` is `None`, then open the
/// base table.
pub(super) fn read_table(ms: &Path, table: Option<&str>) -> Result<Table, MsReadError> {
    let t = Table::open(
        format!("{}/{}", ms.display(), table.unwrap_or("")),
        TableOpenMode::Read,
    )?;
    Ok(t)
}

/// Attempt to determine who/what created this measurement set.
fn get_ms_flavour(history_table: &mut Table) -> Result<MsFlavour, MsReadError> {
    let app = history_table
        .get_cell::<String>("APPLICATION", 0)?
        .to_uppercase();
    if app.starts_with("MWA_HYPERDRIVE") {
        return Ok(MsFlavour::Hyperdrive);
    } else if app.starts_with("BIRLI") {
        return Ok(MsFlavour::Birli);
    } else if app.starts_with("MARLU") {
        return Ok(MsFlavour::Marlu);
    } else if app.starts_with("COTTER") {
        return Ok(MsFlavour::Cotter);
    };

    // If there wasn't an app in the "APPLICATION" column, see if we
    // can get more information out of the "MESSAGE" column.
    let messages: Vec<String> = history_table.get_col_as_vec("MESSAGE")?;
    for message in messages {
        let upper = message.to_uppercase();
        if upper.contains("HYPERDRIVE") {
            return Ok(MsFlavour::Hyperdrive);
        } else if upper.contains("MARLU") {
            return Ok(MsFlavour::Marlu);
        } else if upper.contains("BIRLI") {
            return Ok(MsFlavour::Birli);
        }
    }

    // If we *still* don't know what the app is, fallback on
    // "Casa".
    Ok(MsFlavour::Casa)
}

pub(crate) struct MsReader {
    /// Input data metadata.
    obs_context: ObsContext,

    /// The path to the measurement set on disk.
    pub(crate) ms: PathBuf,

    /// The "stride" of the data, i.e. the number of rows (baselines) before the
    /// time index changes.
    step: usize,

    /// The [`mwalib::MetafitsContext`] used when [`MsReader`] was created.
    metafits_context: Option<MetafitsContext>,

    /// MSs may not number their antennas 0 to the total number of antennas.
    /// This map converts a MS antenna number to a 0-to-total index.
    tile_map: HashMap<i32, usize>,

    /// The name of the column to be used containing visibility data in the main
    /// column.
    data_col_name: String,

    /// The name of the column containing visibility weights in the main column.
    // Some measurement sets use WEIGHT_SPECTRUM as their weights column, others
    // use WEIGHT. Maybe there are probably even more variants.
    weight_col_name: &'static str,

    /// Is the weight column two-dimensional? We track this here because it
    /// could be 1D or 2D.
    weight_col_is_2d: bool,
}

impl MsReader {
    /// Verify and populate metadata associated with this measurement set.
    ///
    /// The measurement set is expected to be formatted in the way that
    /// cotter/Birli write measurement sets. There's a difference between a
    /// flagged antenna and an antenna which has no data. The former may be
    /// used, but its flagged status hints that maybe it shouldn't be used.
    // TODO: Handle multiple measurement sets.
    pub(crate) fn new<P: AsRef<Path>, P2: AsRef<Path>>(
        ms: P,
        data_column_name: Option<String>,
        metafits: Option<P2>,
        array_pos: Option<LatLngHeight>,
    ) -> Result<MsReader, VisReadError> {
        fn inner(
            ms: &Path,
            data_column_name: Option<String>,
            metafits: Option<&Path>,
            array_pos: Option<LatLngHeight>,
        ) -> Result<MsReader, MsReadError> {
            debug!("Using measurement set: {}", ms.display());
            if !ms.exists() {
                return Err(MsReadError::BadFile(ms.to_path_buf()));
            }

            // If a metafits file was provided, get an mwalib object ready.
            let mwalib_context = match metafits {
                None => None,
                // TODO: Let the user supply the MWA version
                Some(m) => Some(mwalib::MetafitsContext::new(m, None)?),
            };

            let mut main_table = read_table(ms, None)?;
            if main_table.n_rows() == 0 {
                return Err(MsReadError::MainTableEmpty);
            }
            let col_names = main_table.column_names()?;
            let data_col_name = data_column_name.unwrap_or_else(|| "DATA".to_string());
            // Validate the data column name, specified or not.
            if !col_names.contains(&data_col_name) {
                return Err(MsReadError::NoDataCol { col: data_col_name });
            }
            let weight_col_name = {
                let mut weight_col_name_to_use = None;
                for name in SUPPORTED_WEIGHT_COL_NAMES {
                    if col_names.iter().any(|col_name| col_name == name) {
                        weight_col_name_to_use = Some(name);
                        break;
                    }
                }
                weight_col_name_to_use.ok_or(MsReadError::NoWeightCol)?
            };
            drop(col_names);

            // The weights array can be 1D or 2D (upside-down
            // emoji). This is documented, but, I don't trust people
            // to be sensible here. So, if the weights array is 2D,
            // we assume that the second axis is a weight per
            // polarisation, otherwise a single weight for all pols
            // of a visibility.
            let weight_col_is_2d = {
                // Try the 2D read, and report whether it succeeded or not.
                let mut weight_col_is_2d = true;
                main_table.for_each_row_in_range(0..1, |row| {
                    let array2: Result<Array2<f32>, _> = row.get_cell(weight_col_name);
                    if array2.is_ok() {
                        // // Until the need arises, we complain if there aren't 4
                        // // polarisations present.
                        // if array2?.len_of(Axis(1)) != 4 {
                        //     panic!(
                        //         "{}",
                        //         MsReadError::BadArraySize {
                        //             array_type: "weights",
                        //             row_index: 0,
                        //             expected_len: 4,
                        //             axis_num: 1,
                        //         }
                        //     );
                        // }
                    } else {
                        weight_col_is_2d = false;
                    }
                    Ok(())
                })?;
                weight_col_is_2d
            };

            // Verify that the dimensions of the data and flag columns are
            // sensible. We're assuming that the dimensions of the arrays pulled
            // out from each column doesn't change with row, but that's not
            // possible with a MS, right?
            main_table.for_each_row_in_range(0..1, |row| {
                let data: Result<Array2<c32>, _> = row.get_cell(&data_col_name);
                // If there was an error here, the reason might be that we
                // attempted to read complex numbers out of a column that
                // doesn't contain them.
                if let Err(err) = data {
                    panic!(
                        "{}",
                        MsReadError::MainTableColReadError {
                            column: data_col_name.clone(),
                            err,
                        }
                    );
                }
                let data = data?;
                // We assume here that the main_table contains a FLAG table.
                let flags: Array2<bool> = row.get_cell("FLAG")?;

                // // Until the need arises, we complain if there aren't 4
                // // polarisations present.
                // if data.len_of(Axis(1)) != 4 {
                //     panic!(
                //         "{}",
                //         MsReadError::BadArraySize {
                //             array_type: "data",
                //             row_index: 0,
                //             expected_len: 4,
                //             axis_num: 1,
                //         }
                //     );
                // }
                // if flags.len_of(Axis(1)) != 4 {
                //     panic!(
                //         "{}",
                //         MsReadError::BadArraySize {
                //             array_type: "flags",
                //             row_index: 0,
                //             expected_len: 4,
                //             axis_num: 1,
                //         }
                //     );
                // }
                assert_eq!(data.dim(), flags.dim());
                Ok(())
            })?;

            // What created this measurement set?
            let mut history_table = read_table(ms, Some("HISTORY"))?;
            let flavour = get_ms_flavour(&mut history_table)?;

            // Get the tile names and XYZ positions.
            let mut antenna_table = read_table(ms, Some("ANTENNA"))?;
            let tile_names: Vec<String> = antenna_table.get_col_as_vec("NAME")?;
            trace!("There are {} tile names", tile_names.len());
            let tile_names =
                Vec1::try_from_vec(tile_names).map_err(|_| MsReadError::AntennaTableEmpty)?;

            let (tile_xyzs, array_pos): (Vec<marlu::XyzGeodetic>, LatLngHeight) = {
                let mut casacore_positions = Vec::with_capacity(antenna_table.n_rows() as usize);
                antenna_table.for_each_row(|row| {
                    let pos: Vec<f64> = row.get_cell("POSITION")?;
                    let pos_xyz = XyzGeocentric {
                        x: pos[0],
                        y: pos[1],
                        z: pos[2],
                    };
                    casacore_positions.push(pos_xyz);
                    Ok(())
                })?;
                // XXX(Dev): no way to get array_pos from MS AFAIK
                let array_pos = match (array_pos, flavour) {
                    (Some(p), _) => p,
                    (None, MsFlavour::Hyperdrive | MsFlavour::Birli | MsFlavour::Marlu) => {
                        warn!("Assuming that this measurement set's array position is the MWA");
                        LatLngHeight::mwa()
                    }
                    (None, MsFlavour::Cotter) => {
                        warn!("Assuming that this measurement set's array position is cotter's MWA position");
                        LatLngHeight {
                            longitude_rad: COTTER_MWA_LONGITUDE_RADIANS,
                            latitude_rad: COTTER_MWA_LATITUDE_RADIANS,
                            height_metres: COTTER_MWA_HEIGHT_METRES,
                        }
                    }
                    (None, MsFlavour::Casa) => return Err(MsReadError::NoArrayPos),
                };

                let vec = XyzGeocentric::get_geocentric_vector(array_pos);
                let (s_long, c_long) = array_pos.longitude_rad.sin_cos();
                let tile_xyzs = casacore_positions
                    .par_iter()
                    .map(|xyz| xyz.to_geodetic_inner(vec, s_long, c_long))
                    .collect();

                (tile_xyzs, array_pos)
            };
            trace!("There are positions for {} tiles", tile_xyzs.len());
            // Not sure if this is even possible, but we'll handle it anyway.
            if tile_xyzs.len() != tile_names.len() {
                return Err(MsReadError::MismatchNumNamesNumXyzs);
            }
            let tile_xyzs =
                Vec1::try_from_vec(tile_xyzs).map_err(|_| MsReadError::AntennaTableEmpty)?;
            let total_num_tiles = tile_xyzs.len();

            // Analyse the antenna numbers in the main table. We need to ensure
            // that there aren't more antennas here than there are antenna names
            // or XYZs. We also need to identify antenna numbers that have no
            // associated data ("unavailable tiles"). Iterate over the baselines
            // (i.e. main table rows) until we've seen all available antennas.
            let mut autocorrelations_present = false;
            let (tile_map, unavailable_tiles): (HashMap<i32, usize>, Vec<usize>) = {
                let antenna1: Vec<i32> = main_table.get_col_as_vec("ANTENNA1")?;
                let antenna2: Vec<i32> = main_table.get_col_as_vec("ANTENNA2")?;

                let mut present_tiles = HashSet::with_capacity(total_num_tiles);
                for (&antenna1, &antenna2) in antenna1.iter().zip(antenna2.iter()) {
                    present_tiles.insert(antenna1);
                    present_tiles.insert(antenna2);

                    if !autocorrelations_present && antenna1 == antenna2 {
                        autocorrelations_present = true;
                    }
                }

                // Ensure there aren't more tiles here than in the names or XYZs
                // (names and XYZs are checked to be the same above).
                if present_tiles.len() > tile_xyzs.len() {
                    return Err(MsReadError::MismatchNumMainTableNumXyzs {
                        main: present_tiles.len(),
                        xyzs: tile_xyzs.len(),
                    });
                }

                // Ensure all MS antenna indices are positive and none are
                // bigger than the number of XYZs.
                for &i in &present_tiles {
                    if i < 0 {
                        return Err(MsReadError::AntennaNumNegative(i));
                    }
                    if i as usize >= tile_xyzs.len() {
                        return Err(MsReadError::AntennaNumTooBig(i));
                    }
                }

                let mut tile_map = HashMap::with_capacity(present_tiles.len());
                let mut unavailable_tiles =
                    Vec::with_capacity(total_num_tiles - present_tiles.len());
                for i_tile in 0..total_num_tiles {
                    if let Some(v) = present_tiles.get(&(i_tile as i32)) {
                        tile_map.insert(*v, i_tile);
                    } else {
                        unavailable_tiles.push(i_tile);
                    }
                }
                (tile_map, unavailable_tiles)
            };
            debug!("Autocorrelations present: {autocorrelations_present}");
            debug!("Unavailable tiles: {unavailable_tiles:?}");

            // This is the number of main table rows (i.e. baselines) per
            // timestep.
            let num_available_tiles = total_num_tiles - unavailable_tiles.len();
            let step = num_available_tiles * (num_available_tiles - 1) / 2
                + if autocorrelations_present {
                    num_available_tiles
                } else {
                    0
                };
            trace!("MS step: {}", step);

            // Work out the first and last good timesteps. This is important
            // because the observation's data may start and end with
            // visibilities that are all flagged, and (by default) we are not
            // interested in using any of those data. We work out the first and
            // last good timesteps by inspecting the flags at each timestep.
            let unflagged_timesteps: Vec<usize> = {
                // The first and last good timestep indices.
                let mut first: Option<usize> = None;
                let mut last: Option<usize> = None;

                trace!("Searching for unflagged timesteps in the MS");
                for i_step in 0..(main_table.n_rows() as usize) / step {
                    trace!("Reading timestep {i_step}");
                    let mut all_rows_for_step_flagged = true;
                    for i_row in 0..step {
                        let vis_flags: Vec<bool> =
                            main_table.get_cell_as_vec("FLAG", (i_step * step + i_row) as u64)?;
                        let all_flagged = vis_flags.into_iter().all(|f| f);
                        if !all_flagged {
                            all_rows_for_step_flagged = false;
                            if first.is_none() {
                                first = Some(i_step);
                                debug!("First good timestep: {i_step}");
                            }
                            break;
                        }
                    }
                    if all_rows_for_step_flagged && first.is_some() {
                        last = Some(i_step);
                        debug!("Last good timestep: {}", i_step - 1);
                        break;
                    }
                }

                // Did the indices get set correctly?
                match (first, last) {
                    (Some(f), Some(l)) => f..l,
                    // If there weren't any flags at the end of the MS, then the
                    // last timestep is fine.
                    (Some(f), None) => f..main_table.n_rows() as usize / step,
                    // All timesteps are flagged. The user can still use the MS,
                    // but they must specify some amount of flagged timesteps.
                    _ => 0..0,
                }
            }
            .collect();

            // Neither Birli nor cotter utilise the "FLAG_ROW" column of the
            // antenna table. This is the best (only?) way to unambiguously
            // identify flagged tiles. I (CHJ) have investigated determining
            // flagged tiles from the main table, but (1) only Birli uses the
            // "FLAG_ROW" column, (2) baselines could be flagged independent of
            // tile flags, (3) it can be difficult to determine/ambiguous if a
            // baseline is flagged because the whole timestep is flagged. For
            // these reasons, we say all tiles are unflagged (except those that
            // are unavailable). When reading visibilities, flags and weights
            // will be applied, so truly flagged tiles won't be directly used in
            // calibration, but their data is still uselessly kept in memory.
            // TODO: Use "FLAG_ROW" in Birli's antenna table.
            let flagged_tiles = unavailable_tiles.clone();
            debug!("Flagged tiles in the MS: {:?}", flagged_tiles);

            // Get the unique times in the MS.
            let utc_times: Vec<f64> = main_table.get_col_as_vec("TIME")?;
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
                [t] => debug!("Only timestep (GPS): {:.2}", t.to_gpst_seconds()),
                [t0, .., tn] => {
                    debug!("First good timestep (GPS): {:.2}", t0.to_gpst_seconds());
                    debug!("Last good timestep  (GPS): {:.2}", tn.to_gpst_seconds());
                }
            }

            // Get the data's time resolution. There is a possibility that the MS
            // contains only one timestep.
            let time_res = if timestamps.len() == 1 {
                debug!("Only one timestep is present in the data; can't determine the data's time resolution.");
                None
            } else {
                // Find the minimum gap between two consecutive timestamps.
                let time_res = timestamps
                    .windows(2)
                    .fold(Duration::from_seconds(f64::INFINITY), |acc, ts| {
                        acc.min(ts[1] - ts[0])
                    });
                trace!("Time resolution: {}s", time_res.to_seconds());
                Some(time_res)
            };

            let all_timesteps = (0..timestamps.len()).collect();
            let all_timesteps =
                Vec1::try_from_vec(all_timesteps).map_err(|_| MsReadError::NoTimesteps {
                    file: ms.display().to_string(),
                })?;

            // Get the frequency information.
            let mut spectral_window_table = read_table(ms, Some("SPECTRAL_WINDOW"))?;
            let fine_chan_freqs = {
                let fine_chan_freqs_hz: Vec<f64> =
                    spectral_window_table.get_cell_as_vec("CHAN_FREQ", 0)?;
                let fine_chan_freqs = fine_chan_freqs_hz
                    .into_iter()
                    .map(|f| f.round() as u64)
                    .collect();
                Vec1::try_from_vec(fine_chan_freqs).map_err(|_| MsReadError::NoChannelFreqs)?
            };
            // Assume that `total_bandwidth_hz` is the total bandwidth inside the
            // measurement set, which is not necessarily the whole observation.
            let total_bandwidth_hz: f64 = spectral_window_table.get_cell("TOTAL_BANDWIDTH", 0)?;
            debug!("MS total bandwidth: {} Hz", total_bandwidth_hz);

            // Note the "subband" is CASA nomenclature. MWA tends to use "coarse
            // channel" instead.
            let coarse_chan_nums: Vec<u32> = {
                // If MWA_SUBBAND doesn't exist, then we must assume that this
                // measurement set only contains one coarse channel.
                match read_table(ms, Some("MWA_SUBBAND")) {
                    Err(_) => vec![1],
                    Ok(mut mwa_subband_table) => {
                        let zero_indexed_coarse_chans: Vec<i32> =
                            mwa_subband_table.get_col_as_vec("NUMBER")?;
                        let one_indexed_coarse_chans: Vec<u32> = zero_indexed_coarse_chans
                            .into_iter()
                            .map(|cc_num| {
                                if cc_num < 0 {
                                    Err(MsReadError::NegativeSubband { num: cc_num })
                                } else {
                                    Ok((cc_num + 1) as _)
                                }
                            })
                            .collect::<Result<_, MsReadError>>()?;
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

            // Get the observation phase centre.
            let phase_centre = {
                let mut field_table = read_table(ms, Some("FIELD"))?;
                let phase_vec = field_table.get_cell_as_vec("PHASE_DIR", 0)?;
                RADec::from_radians(phase_vec[0], phase_vec[1])
            };

            // Populate the dipole delays, gains and the pointing centre if we
            // can.
            let mut dipole_delays: Option<Delays> = None;
            let mut dipole_gains: Option<_> = None;
            let mut pointing_centre: Option<RADec> = None;

            match (&mwalib_context, read_table(ms, Some("MWA_TILE_POINTING"))) {
                // No metafits file was provided and MWA_TILE_POINTING doesn't
                // exist; we have no information on the dipole delays or gains.
                // We also know nothing about the pointing centre.
                (None, Err(_)) => {
                    debug!(
                        "No dipole delays, dipole gains or pointing centre information available"
                    );
                }

                // Use the metafits file. The MWA_TILE_POINTING table can only
                // supply ideal dipole delays, so it's always better to use the
                // metafits.
                (Some(context), _) => {
                    debug!("Using metafits for dipole delays, dipole gains and pointing centre");
                    let delays = metafits::get_dipole_delays(context);
                    let gains = metafits::get_dipole_gains(context);
                    pointing_centre = Some(RADec::from_degrees(
                        context.ra_tile_pointing_degrees,
                        context.dec_tile_pointing_degrees,
                    ));

                    // Re-order the tile delays and gains according to the
                    // uvfits order, if possible.
                    if let Some(map) = metafits::map_antenna_order(context, &tile_names) {
                        let mut delays2 = delays.clone();
                        let mut gains2 = gains.clone();
                        for i in 0..tile_names.len() {
                            let j = map[&i];
                            delays2
                                .slice_mut(s![i, ..])
                                .assign(&delays.slice(s![j, ..]));
                            gains2.slice_mut(s![i, ..]).assign(&gains.slice(s![j, ..]));
                        }
                        dipole_delays = Some(Delays::Full(delays2));
                        dipole_gains = Some(gains2);
                    } else {
                        // We have no choice but to leave the order as is.
                        warn!(
                            "The MS antenna names are different to those supplied in the metafits."
                        );
                        warn!("Dipole delays/gains may be incorrectly mapped to MS antennas.");
                        dipole_delays = Some(Delays::Full(delays));
                        dipole_gains = Some(gains);
                    }
                }

                // MWA_TILE_POINTING exists.
                (_, Ok(mut mwa_tile_pointing_table)) => {
                    debug!("Using MWA_TILE_POINTING for dipole delays and pointing centre");
                    let table_delays: Vec<i32> =
                        mwa_tile_pointing_table.get_cell_as_vec("DELAYS", 0)?;
                    if table_delays.len() != 16 {
                        return Err(MsReadError::WrongNumDipoleDelays {
                            num: table_delays.len(),
                        });
                    }
                    let delays: Vec<u32> = table_delays
                        .into_iter()
                        .map(|delay| {
                            if !(0..=32).contains(&delay) {
                                Err(MsReadError::InvalidDelay { delay })
                            } else {
                                Ok(delay as u32)
                            }
                        })
                        .collect::<Result<_, MsReadError>>()?;
                    dipole_delays = Some(Delays::Partial(delays));

                    let pointing_vec: Vec<f64> =
                        mwa_tile_pointing_table.get_cell_as_vec("DIRECTION", 0)?;
                    pointing_centre = Some(RADec::from_radians(pointing_vec[0], pointing_vec[1]));
                }
            }

            // Round the values in here because sometimes they have a fractional
            // component, for some reason. We're unlikely to ever have a fraction of
            // a Hz as the channel resolution.
            let freq_res = {
                let all_widths: Vec<f64> =
                    spectral_window_table.get_cell_as_vec("CHAN_WIDTH", 0)?;
                let width = *all_widths.first().ok_or(MsReadError::NoChanWidths)?;
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
                        main_table.get_cell_as_vec("FLAG", row_range.start)?;
                    // If there are 4x as many flags as there are fine channels,
                    // then we assume its because there's a flag specified for
                    // each polarisation. Which is dumb. If any of the 4 flags
                    // for a channel are flagged, we consider the channel
                    // flagged.
                    if (flagged_fine_chans.len() / fine_chan_freqs.len()) % 4 == 0 {
                        flagged_fine_chans
                            .chunks_exact(4)
                            .map(|pol_flags| pol_flags.iter().any(|f| *f))
                            .collect()
                    } else {
                        flagged_fine_chans
                    }
                };
                main_table.for_each_row_in_range(row_range, |row| {
                    let row_flagged_fine_chans: Array2<bool> = row.get_cell("FLAG")?;
                    flagged_fine_chans
                        .iter_mut()
                        .zip(row_flagged_fine_chans.outer_iter())
                        .for_each(|(f1, f2)| {
                            let any_flagged = f2.iter().any(|f| *f);
                            *f1 &= any_flagged;
                        });
                    Ok(())
                })?;
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

            // Measurement sets don't appear to have an official way to supply
            // the DUT1. Marlu 0.9.0 writes UT1UTC into the main table's
            // keywords, so pick it up if it's there, otherwise use the
            // metafits.
            let dut1 = match (
                main_table
                    .get_keyword_record()?
                    .get_field::<f64>("UT1UTC")
                    .ok(),
                mwalib_context.as_ref(),
            ) {
                // If the MS has the key, then use it, even if we have a
                // metafits.
                (Some(dut1), _) => {
                    debug!("MS has no UT1UTC");
                    Some(dut1)
                }

                // Use the value in the metafits.
                (None, Some(c)) => {
                    debug!("MS has no UT1UTC");
                    match c.dut1 {
                        Some(dut1) => debug!("metafits DUT1: {dut1}"),
                        None => debug!("metafits has no DUT1"),
                    }
                    c.dut1
                }

                // We have no DUT1.
                (None, None) => {
                    debug!("MS has no UT1UTC");
                    debug!("metafits has no DUT1");
                    None
                }
            }
            .map(Duration::from_seconds);

            let obs_context = ObsContext {
                obsid,
                timestamps,
                all_timesteps,
                unflagged_timesteps,
                phase_centre,
                pointing_centre,
                array_position: Some(array_pos),
                dut1,
                tile_names,
                tile_xyzs,
                flagged_tiles,
                unavailable_tiles,
                autocorrelations_present,
                dipole_delays,
                dipole_gains,
                time_res,
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
                tile_map,
                data_col_name,
                weight_col_name,
                weight_col_is_2d,
            };
            Ok(ms)
        }
        inner(
            ms.as_ref(),
            data_column_name,
            metafits.as_ref().map(|f| f.as_ref()),
            array_pos,
        )
        .map_err(VisReadError::from)
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

        let mut main_table = read_table(&self.ms, None)?;
        let mut row_index = row_range.start;
        main_table
            .for_each_row_in_range(row_range, |row| {
                // Antenna numbers are zero indexed.
                let ant1: i32 = row.get_cell("ANTENNA1")?;
                let ant2: i32 = row.get_cell("ANTENNA2")?;
                // Use our map.
                let ant1 = self.tile_map[&ant1];
                let ant2 = self.tile_map[&ant2];

                // Read this row if the baseline is unflagged.
                if let Some(crosses) = crosses.as_mut() {
                    if let Some(bl) = crosses
                        .tile_baseline_flags
                        .tile_to_unflagged_cross_baseline_map
                        .get(&(ant1, ant2))
                        .copied()
                    {
                        // The data array is arranged [frequency][instrumental_pol].
                        let ms_data: Array2<c32> = row.get_cell(&self.data_col_name)?;
                        let ms_weights: Vec<f32> = {
                            if self.weight_col_is_2d {
                                // The weight array is arranged
                                // [frequency][instrumental_pol].
                                let ms_weights: Array2<f32> = row.get_cell(self.weight_col_name)?;
                                // Collapse the weights into a single number per
                                // frequency; having a weight per polarisation
                                // is not useful.
                                ms_weights
                                    .exact_chunks((1, 4))
                                    .into_iter()
                                    .map(|weights| {
                                        weights.iter().copied().reduce(f32::min).expect("not empty")
                                    })
                                    .collect()
                            } else {
                                // One weight per frequency.
                                row.get_cell(self.weight_col_name)?
                            }
                        };
                        // The flag array is arranged
                        // [frequency][instrumental_pol]. As with the weights,
                        // we ignore the per polarisation values.
                        let flags: Array2<bool> = row.get_cell("FLAG")?;

                        if crosses.data_array.len_of(Axis(1)) < bl {
                            panic!(
                                "{}",
                                VisReadError::BadArraySize {
                                    array_type: "data_array",
                                    expected_len: bl,
                                    axis_num: 1,
                                }
                            );
                        }
                        if crosses.data_array.len_of(Axis(0)) > ms_data.len_of(Axis(0)) {
                            panic!(
                                "{}",
                                VisReadError::BadArraySize {
                                    array_type: "data_array",
                                    expected_len: ms_data.len_of(Axis(0)),
                                    axis_num: 0,
                                }
                            );
                        }

                        // Put the data and weights into the shared arrays
                        // outside this scope. Before we can do this, we need to
                        // remove any globally-flagged fine channels.
                        let mut out_vis = crosses.data_array.slice_mut(s![.., bl]);
                        ms_data
                            .outer_iter()
                            .enumerate()
                            .filter(|(i_chan, _)| !flagged_fine_chans.contains(i_chan))
                            .zip(out_vis.iter_mut())
                            .for_each(|((_, ms_data), out_vis)| {
                                *out_vis = if ms_data.len() == 4 {
                                    Jones::from([ms_data[0], ms_data[1], ms_data[2], ms_data[3]])
                                } else if ms_data.len() == 1 {
                                    Jones::from([
                                        ms_data[0],
                                        c32::default(),
                                        c32::default(),
                                        ms_data[0],
                                    ])
                                } else {
                                    panic!("Number of data polarisations wasn't 4 or 1");
                                };
                            });

                        // Apply the flags to the weights (negate if flagged),
                        // and throw away 3 of the 4 weights; there are 4
                        // weights (for XX XY YX YY) and we assume that the
                        // first weight is the same as the others.
                        let mut out_weights = crosses.weights_array.slice_mut(s![.., bl]);
                        ms_weights
                            .into_iter()
                            .zip(flags.outer_iter())
                            .enumerate()
                            .filter(|(i_chan, _)| !flagged_fine_chans.contains(i_chan))
                            .zip(out_weights.iter_mut())
                            .for_each(|((_, (weight, flag)), out_weight)| {
                                // Collapse the multiple flag values into a
                                // single one by finding any that are true (i.e.
                                // at least one polarisation is marked as
                                // flagged, so flag the whole visibility).
                                let flag = flag.into_iter().copied().any(|f| f);
                                *out_weight = if flag { -weight.abs() } else { weight };
                            });
                    }
                }

                if let Some(autos) = autos.as_mut() {
                    if ant1 == ant2 {
                        if let Some(i_ant) = autos
                            .tile_baseline_flags
                            .tile_to_unflagged_auto_index_map
                            .get(&ant1)
                            .copied()
                        {
                            let ms_data: Array2<c32> = row.get_cell(&self.data_col_name)?;
                            let ms_weights: Vec<f32> = {
                                if self.weight_col_is_2d {
                                    let ms_weights: Array2<f32> =
                                        row.get_cell(self.weight_col_name)?;
                                    ms_weights
                                        .exact_chunks((1, 4))
                                        .into_iter()
                                        .map(|weights| {
                                            weights
                                                .iter()
                                                .copied()
                                                .reduce(f32::min)
                                                .expect("not empty")
                                        })
                                        .collect()
                                } else {
                                    row.get_cell(self.weight_col_name)?
                                }
                            };
                            let flags: Array2<bool> = row.get_cell("FLAG")?;

                            if autos.data_array.len_of(Axis(1)) < i_ant {
                                panic!(
                                    "{}",
                                    VisReadError::BadArraySize {
                                        array_type: "data_array",
                                        expected_len: i_ant,
                                        axis_num: 1,
                                    }
                                );
                            }
                            if autos.data_array.len_of(Axis(0)) > ms_data.len_of(Axis(0)) {
                                panic!(
                                    "{}",
                                    VisReadError::BadArraySize {
                                        array_type: "data_array",
                                        expected_len: ms_data.len_of(Axis(0)),
                                        axis_num: 0,
                                    }
                                );
                            }

                            let mut out_vis = autos.data_array.slice_mut(s![.., i_ant]);
                            ms_data
                                .outer_iter()
                                .enumerate()
                                .filter(|(i_chan, _)| !flagged_fine_chans.contains(i_chan))
                                .zip(out_vis.iter_mut())
                                .for_each(|((_, ms_data), out_vis)| {
                                    *out_vis = Jones::from([
                                        ms_data[0], ms_data[1], ms_data[2], ms_data[3],
                                    ]);
                                });

                            let mut out_weights = autos.weights_array.slice_mut(s![.., i_ant]);
                            ms_weights
                                .into_iter()
                                .zip(flags.outer_iter())
                                .enumerate()
                                .filter(|(i_chan, _)| !flagged_fine_chans.contains(i_chan))
                                .zip(out_weights.iter_mut())
                                .for_each(|((_, (weight, flag)), out_weight)| {
                                    let flag = flag.into_iter().copied().any(|f| f);
                                    *out_weight = if flag { -weight.abs() } else { weight };
                                });
                        }
                    }
                }

                row_index += 1;
                Ok(())
            })
            .map_err(MsReadError::from)?;

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

    fn get_flags(&self) -> Option<&MwafFlags> {
        None
    }

    fn read_crosses_and_autos(
        &self,
        cross_data_array: ArrayViewMut2<Jones<f32>>,
        cross_weights_array: ArrayViewMut2<f32>,
        auto_data_array: ArrayViewMut2<Jones<f32>>,
        auto_weights_array: ArrayViewMut2<f32>,
        timestep: usize,
        tile_baseline_flags: &TileBaselineFlags,
        flagged_fine_chans: &HashSet<usize>,
    ) -> Result<(), VisReadError> {
        self.read_inner(
            Some(CrossData {
                data_array: cross_data_array,
                weights_array: cross_weights_array,
                tile_baseline_flags,
            }),
            Some(AutoData {
                data_array: auto_data_array,
                weights_array: auto_weights_array,
                tile_baseline_flags,
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
        tile_baseline_flags: &TileBaselineFlags,
        flagged_fine_chans: &HashSet<usize>,
    ) -> Result<(), VisReadError> {
        self.read_inner(
            Some(CrossData {
                data_array,
                weights_array,
                tile_baseline_flags,
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
        tile_baseline_flags: &TileBaselineFlags,
        flagged_fine_chans: &HashSet<usize>,
    ) -> Result<(), VisReadError> {
        self.read_inner(
            None,
            Some(AutoData {
                data_array,
                weights_array,
                tile_baseline_flags,
            }),
            timestep,
            flagged_fine_chans,
        )
    }
}
