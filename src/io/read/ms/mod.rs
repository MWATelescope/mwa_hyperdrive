// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to interface with CASA measurement sets.
//!
//! More info:
//! <https://casa.nrao.edu/Memos/229.html#SECTION00060000000000000000>

// TODO: Detect mismatched time resolutions
// TODO: Inside `read_inner`, check the timestamp value of each row, verifying
// that is what is expected.

mod error;
#[cfg(test)]
mod tests;

pub(crate) use error::*;

use std::{
    collections::{BTreeSet, HashMap},
    num::NonZeroU16,
    path::{Path, PathBuf},
};

use hifitime::{Duration, Epoch, TimeUnits};
use log::{debug, trace};
use marlu::{c32, rubbl_casatables, Jones, LatLngHeight, RADec, XyzGeocentric, XyzGeodetic};
use ndarray::prelude::*;
use rubbl_casatables::{Table, TableError, TableOpenMode};

use super::*;
use crate::{
    beam::Delays,
    cli::Warn,
    constants::DEFAULT_MS_DATA_COL_NAME,
    context::{ObsContext, Polarisations},
    metafits,
};

const SUPPORTED_WEIGHT_COL_NAMES: [&str; 2] = ["WEIGHT_SPECTRUM", "WEIGHT"];
lazy_static::lazy_static! {
    /// Round all MS timestamps to this.
    static ref ROUND_TO: Duration = 10.microseconds();
}

/// Open a measurement set table read only. If `table` is `None`, then open the
/// base table.
pub(super) fn read_table(ms: &Path, table: Option<&str>) -> Result<Table, TableError> {
    if let Some(table) = table {
        Table::open(ms.join(table), TableOpenMode::Read)
    } else {
        Table::open(ms, TableOpenMode::Read)
    }
}

// Get the data's time resolution. We use the INTERVAL or EXPOSURE columns, if
// they're available, in that order. Failing that we need to fall back on the
// timesteps. This function assumes that the `main_table` and `timestamps` are
// not empty.
fn get_time_resolution(main_table: &mut Table, timestamps: &[Epoch]) -> Option<Duration> {
    if let Ok(s) = main_table.get_cell("INTERVAL", 0) {
        debug!("Using the INTERVAL column for the time resolution ({s} seconds)");
        return Some(Duration::from_seconds(s));
    }
    if let Ok(s) = main_table.get_cell("EXPOSURE", 0) {
        debug!("Using the EXPOSURE column for the time resolution ({s} seconds)");
        return Some(Duration::from_seconds(s));
    }
    if timestamps.len() == 1 {
        debug!(
            "Only one timestep is present in the data; can't determine the data's time resolution."
        );
        None
    } else {
        debug!("Using the timestamps to determine the time resolution");
        // Find the minimum gap between two consecutive timestamps.
        let time_res = timestamps
            .windows(2)
            .fold(Duration::from_seconds(f64::INFINITY), |acc, ts| {
                acc.min(ts[1] - ts[0])
            });
        Some(time_res)
    }
}

pub struct MsReader {
    /// Input data metadata.
    obs_context: ObsContext,

    /// The path to the measurement set on disk.
    ms: PathBuf,

    /// The "stride" of the data, i.e. the number of rows (baselines) before the
    /// time index changes.
    step: usize,

    /// The [`mwalib::MetafitsContext`] used when [`MsReader`] was created.
    metafits_context: Option<MetafitsContext>,

    /// MSs may not number their antennas 0 to the total number of antennas.
    /// This map converts a MS antenna number to a 0-to-total index.
    tile_map: HashMap<i32, usize>,

    /// The timestamps in the MS. Only used as a safety check.
    ms_timestamps: Vec1<f64>,

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

    /// If the incoming data uses ant2-ant1 UVWs instead of ant1-ant2 UVWs, we
    /// need to conjugate the visibilities to match what will be modelled.
    conjugate_vis: bool,
}

impl MsReader {
    /// Verify and populate metadata associated with this measurement set.
    ///
    /// The measurement set is expected to be formatted in the way that
    /// cotter/Birli write measurement sets. There's a difference between a
    /// flagged antenna and an antenna which has no data. The former may be
    /// used, but its flagged status hints that maybe it shouldn't be used.
    // TODO: Handle multiple measurement sets.
    pub fn new(
        ms: PathBuf,
        data_column_name: Option<String>,
        metafits: Option<&Path>,
        array_position: Option<LatLngHeight>,
    ) -> Result<MsReader, MsReadError> {
        debug!("Using measurement set: {}", ms.display());
        if !ms.exists() {
            return Err(MsReadError::BadFile(ms));
        }

        // If a metafits file was provided, get an mwalib object ready.
        let mwalib_context = match metafits {
            None => None,
            // TODO: Let the user supply the MWA version
            Some(m) => Some(mwalib::MetafitsContext::new(m, None)?),
        };

        let mut main_table = read_table(&ms, None)?;
        if main_table.n_rows() == 0 {
            return Err(MsReadError::MainTableEmpty);
        }
        let col_names = main_table.column_names()?;
        let data_col_name =
            data_column_name.unwrap_or_else(|| DEFAULT_MS_DATA_COL_NAME.to_string());
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

        // Verify that the dimensions of the data and flag columns are sensible.
        // We're assuming that the dimensions of the arrays pulled out from each
        // column doesn't change with row, but that's not possible with a MS,
        // right?
        let pols = {
            let mut num_pols = 0;
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

                assert_eq!(data.dim(), flags.dim());
                num_pols = data.len_of(Axis(1));
                Ok(())
            })?;

            assert_ne!(num_pols, 0, "Couldn't detect the number of polarisations");
            assert!(
                num_pols <= 4,
                "Detected {num_pols} polarisations; this is not supported"
            );

            // Verify the polarisation count with the POLARIZATION table,
            // and determine what pols are actually available.
            let mut pol_table = read_table(&ms, Some("POLARIZATION"))?;
            let corr_type: Vec<i32> = pol_table.get_cell_as_vec("CORR_TYPE", 0)?;
            // We only support instrumental pols for now.
            for &corr_type in &corr_type {
                if ![9, 10, 11, 12].contains(&corr_type) {
                    return Err(MsReadError::UnsupportedCorrType { corr_type });
                }
            }

            let (pols, pol_table_num_pols) = match corr_type.as_slice() {
                [9, 10, 11, 12] => (Polarisations::XX_XY_YX_YY, 4),
                [9] => (Polarisations::XX, 1),
                [12] => (Polarisations::YY, 1),
                [9, 12] => (Polarisations::XX_YY, 2),
                [9, 10, 12] => (Polarisations::XX_YY_XY, 3),
                _ => return Err(MsReadError::UnsupportedPols { corr_type }),
            };
            if num_pols != pol_table_num_pols {
                return Err(MsReadError::InconsistentPolCount {
                    data: num_pols,
                    pol_table: pol_table_num_pols,
                });
            }

            pols
        };
        debug!("Polarisations: {pols}");

        // The weights array can be 1D or 2D (upside-down emoji). This is
        // documented, but, I don't trust people to be sensible here. So, if the
        // weights array is 2D, we assume that the second axis is a weight per
        // polarisation, otherwise a single weight for all pols of a visibility.
        let weight_col_is_2d = {
            // Try the 2D read, and report whether it succeeded or not.
            let mut weight_col_is_2d = true;
            main_table.for_each_row_in_range(0..1, |row| {
                let array2: Result<Array2<f32>, _> = row.get_cell(weight_col_name);
                if array2.is_ok() {
                    // Complain if there aren't the expected number of
                    // polarisations present.
                    if array2?.len_of(Axis(1)) != usize::from(pols.num_pols()) {
                        panic!(
                            "{}",
                            MsReadError::BadArraySize {
                                array_type: "weights",
                                row_index: 0,
                                expected_len: usize::from(pols.num_pols()),
                                axis_num: 1,
                            }
                        );
                    }
                } else {
                    weight_col_is_2d = false;
                }
                Ok(())
            })?;
            weight_col_is_2d
        };

        // Get the tile names and XYZ positions.
        let mut antenna_table = read_table(&ms, Some("ANTENNA"))?;
        let tile_names: Vec<String> = antenna_table.get_col_as_vec("NAME")?;
        trace!("There are {} tile names", tile_names.len());
        let tile_names =
            Vec1::try_from_vec(tile_names).map_err(|_| MsReadError::AntennaTableEmpty)?;

        let geocentric_tile_xyzs: Vec<XyzGeocentric> = {
            let mut xyzs = Vec::with_capacity(antenna_table.n_rows() as usize);
            antenna_table.for_each_row(|row| {
                let pos: Vec<f64> = row.get_cell("POSITION")?;
                let pos_xyz = XyzGeocentric {
                    x: pos[0],
                    y: pos[1],
                    z: pos[2],
                };
                xyzs.push(pos_xyz);
                Ok(())
            })?;
            xyzs
        };

        let mut obs_table = read_table(&ms, Some("OBSERVATION"))?;
        let supplied_array_position = {
            // SKA SDC3 data had the array position in the OBSERVATION table,
            // ARRAY_CENTER. It doesn't appear to be standard, but if we're
            // here, give it a go, because it'll be more precise than getting it
            // from the antenna positions.
            let maybe_pos = match obs_table.get_cell_as_vec::<f64>("ARRAY_CENTER", 0) {
                Ok(v) => {
                    assert!(
                        v.len() >= 3,
                        "ARRAY_CENTER in OBSERVATION didn't have at least 3 elements"
                    );
                    let xyz = XyzGeocentric {
                        x: v[0],
                        y: v[1],
                        z: v[2],
                    };
                    Some(xyz.to_earth_wgs84())
                }

                Err(_) => None,
            };

            if let Some(pos) = maybe_pos {
                pos
            } else {
                // Get the array position from the average antenna position.
                let mut average_xyz = XyzGeocentric::default();
                for XyzGeocentric { x, y, z } in &geocentric_tile_xyzs {
                    average_xyz.x += x;
                    average_xyz.y += y;
                    average_xyz.z += z;
                }
                average_xyz.x /= geocentric_tile_xyzs.len() as f64;
                average_xyz.y /= geocentric_tile_xyzs.len() as f64;
                average_xyz.z /= geocentric_tile_xyzs.len() as f64;
                average_xyz.to_earth_wgs84()
            }
        };
        let array_position = array_position.unwrap_or(supplied_array_position);

        let tile_xyzs: Vec<XyzGeodetic> = {
            let vec = XyzGeocentric::get_geocentric_vector(array_position);
            let (s_long, c_long) = array_position.longitude_rad.sin_cos();
            geocentric_tile_xyzs
                .iter()
                .map(|xyz| xyz.to_geodetic_inner(vec, s_long, c_long))
                .collect()
        };
        trace!("There are positions for {} tiles", tile_xyzs.len());
        // Not sure if this is even possible, but we'll handle it anyway.
        if tile_xyzs.len() != tile_names.len() {
            return Err(MsReadError::MismatchNumNamesNumXyzs);
        }
        let tile_xyzs =
            Vec1::try_from_vec(tile_xyzs).map_err(|_| MsReadError::AntennaTableEmpty)?;
        let total_num_tiles = tile_xyzs.len();

        // Analyse the antenna numbers in the main table. We need to ensure that
        // there aren't more antennas here than there are antenna names or XYZs.
        // We also need to identify antenna numbers that have no associated data
        // ("unavailable tiles"). Iterate over the baselines (i.e. main table
        // rows) until we've seen all available antennas.
        let mut autocorrelations_present = false;
        let (tile_map, unavailable_tiles): (HashMap<i32, usize>, Vec<usize>) =
            crate::misc::expensive_op(
                || {
                    let mut main_table = read_table(&ms, None)?;
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

                    // Ensure all MS antenna indices are positive and none are bigger
                    // than the number of XYZs.
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
                    Ok::<_, MsReadError>((tile_map, unavailable_tiles))
                },
                "Still waiting to determine MS antenna metadata",
            )?;
        debug!("Autocorrelations present: {autocorrelations_present}");
        debug!("Unavailable tiles: {unavailable_tiles:?}");

        let num_available_tiles = total_num_tiles - unavailable_tiles.len();
        // This is the number of main table rows (i.e. baselines) per timestep.
        let step = num_available_tiles * (num_available_tiles - 1) / 2
            + if autocorrelations_present {
                num_available_tiles
            } else {
                0
            };
        trace!("MS step: {}", step);
        assert!(
            main_table.n_rows() % step as u64 == 0,
            "There are a variable number of baselines per timestep, which is not supported"
        );

        // Work out the first and last good timesteps. This is important because
        // the observation's data may start and end with visibilities that are
        // all flagged, and (by default) we are not interested in using any of
        // those data. We work out the first and last good timesteps by
        // inspecting the flags at each timestep.
        let unflagged_timesteps: Vec<usize> = crate::misc::expensive_op(
            || {
                // The first and last good timestep indices.
                let mut first: Option<usize> = None;
                let mut last: Option<usize> = None;

                trace!("Searching for unflagged timesteps in the MS");
                let mut main_table = read_table(&ms, None)?;
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
                let timesteps = match (first, last) {
                    (Some(f), Some(l)) => f..l,
                    // If there weren't any flags at the end of the MS, then the
                    // last timestep is fine.
                    (Some(f), None) => f..main_table.n_rows() as usize / step,
                    // All timesteps are flagged. The user can still use the MS, but
                    // they must specify some amount of flagged timesteps.
                    _ => 0..0,
                }
                .collect();
                Ok::<_, TableError>(timesteps)
            },
            "Still waiting to determine MS timesteps",
        )?;

        // Neither Birli nor cotter utilise the "FLAG_ROW" column of the antenna
        // table. This is the best (only?) way to unambiguously identify flagged
        // tiles. I (CHJ) have investigated determining flagged tiles from the
        // main table, but (1) only Birli uses the "FLAG_ROW" column, (2)
        // baselines could be flagged independent of tile flags, (3) it can be
        // difficult to determine/ambiguous if a baseline is flagged because the
        // whole timestep is flagged. For these reasons, we say all tiles are
        // unflagged (except those that are unavailable). When reading
        // visibilities, flags and weights will be applied, so truly flagged
        // tiles won't be directly used in calibration, but their data is still
        // uselessly kept in memory.
        // TODO: Use "FLAG_ROW" in Birli's antenna table.
        let flagged_tiles = unavailable_tiles.clone();
        debug!("Flagged tiles in the MS: {:?}", flagged_tiles);

        // Get the unique times in the MS.
        let utc_times: Vec<f64> = main_table.get_col_as_vec("TIME")?;
        let mut utc_time_set: BTreeSet<u64> = BTreeSet::new();
        let mut ms_timestamps = vec![];
        for utc_time in utc_times {
            let bits = utc_time.to_bits();
            if !utc_time_set.contains(&bits) {
                utc_time_set.insert(bits);

                // casacore stores the times as centroids, so no correction is
                // needed for that.
                ms_timestamps.push(utc_time);
            }
        }
        let ms_timestamps =
            Vec1::try_from_vec(ms_timestamps).map_err(|_| MsReadError::NoTimesteps {
                file: ms.display().to_string(),
            })?;
        // The values can be slightly off of their intended values; round them.
        // Also, casacore stores the times as UTC seconds with an offset.
        let timestamps = ms_timestamps.mapped_ref(|utc_time| {
            let e = hifitime::J1900_OFFSET.mul_add(-hifitime::SECONDS_PER_DAY, *utc_time);
            Epoch::from_utc_seconds(e).round(*ROUND_TO)
        });
        match timestamps.as_slice() {
            [] => unreachable!("vec1 cannot be empty"),
            [t] => debug!("Only timestep (GPS): {:.2}", t.to_gpst_seconds()),
            [t0, .., tn] => {
                debug!("First good timestep (GPS): {:.2}", t0.to_gpst_seconds());
                debug!("Last good timestep  (GPS): {:.2}", tn.to_gpst_seconds());
            }
        }

        let time_res = get_time_resolution(&mut main_table, &timestamps);
        trace!("Time resolution: {:?}", time_res);

        let all_timesteps = (0..timestamps.len()).collect();
        let all_timesteps =
            Vec1::try_from_vec(all_timesteps).map_err(|_| MsReadError::NoTimesteps {
                file: ms.display().to_string(),
            })?;

        // Get the frequency information.
        let mut spectral_window_table = read_table(&ms, Some("SPECTRAL_WINDOW"))?;
        let fine_chan_freqs_f64: Vec<f64> =
            spectral_window_table.get_cell_as_vec("CHAN_FREQ", 0)?;
        let fine_chan_freqs = {
            let fine_chan_freqs = fine_chan_freqs_f64
                .iter()
                .map(|f| f.round() as u64)
                .collect();
            Vec1::try_from_vec(fine_chan_freqs).map_err(|_| MsReadError::NoChannelFreqs)?
        };
        // Assume that `total_bandwidth_hz` is the total bandwidth inside the
        // measurement set, which is not necessarily the whole observation.
        let total_bandwidth_hz: f64 = spectral_window_table.get_cell("TOTAL_BANDWIDTH", 0)?;
        debug!("MS total bandwidth: {} Hz", total_bandwidth_hz);

        // Round the values in here because sometimes they have a fractional
        // component, for some reason. We're unlikely to ever have a fraction of
        // a Hz as the channel resolution.
        let freq_res = {
            let all_widths: Vec<f64> = spectral_window_table.get_cell_as_vec("CHAN_WIDTH", 0)?;
            let width = *all_widths.first().ok_or(MsReadError::NoChanWidths)?;
            // Make sure all the widths all the same.
            for w in all_widths.iter().skip(1) {
                if (w - width).abs() > f64::EPSILON {
                    return Err(MsReadError::ChanWidthsUnequal);
                }
            }
            width
        };

        // The MWA_SUBBAND table is supposed to contain information on MWA
        // coarse channels ("subband" is CASA nomenclature, MWA tends to use
        // "coarse channel" instead). Instead, it contains nothing useful. So,
        // we determine what would be MWA coarse channel numbers from the
        // available frequencies.
        let mwa_coarse_chan_nums = match mwalib_context.as_ref() {
            Some(c) => {
                // Get the coarse channel information out of the metafits file,
                // but only the ones aligned with the frequencies in the uvfits
                // file.
                let cc_width = f64::from(c.coarse_chan_width_hz);
                let mut cc_nums: Vec<u32> = c
                    .metafits_coarse_chans
                    .iter()
                    .filter_map(|cc| {
                        let cc_num =
                            u32::try_from(cc.rec_chan_number).expect("not bigger than u32::MAX");
                        let cc_centre = f64::from(cc.chan_centre_hz);
                        for &f in &fine_chan_freqs_f64 {
                            if (f - cc_centre).abs() < cc_width / 2.0 {
                                return Some(cc_num);
                            }
                        }
                        None
                    })
                    .collect();
                cc_nums.sort_unstable();
                debug!("Found corresponding MWA coarse channel numbers from the metafits and MS frequencies");
                Vec1::try_from_vec(cc_nums).ok()
            }

            None => {
                debug!("Assuming MWA coarse channel numbers from MS frequencies");

                // Find all multiples of 1.28 MHz within our bandwidth.
                let mut cc_nums = fine_chan_freqs
                    .iter()
                    .map(|&f| (f as f64 / 1.28e6).round() as u32)
                    .collect::<Vec<_>>();
                cc_nums.sort_unstable();
                cc_nums.dedup();
                Vec1::try_from_vec(cc_nums).ok()
            }
        };

        let num_fine_chans_per_coarse_chan = {
            let n = (1.28e6 / freq_res).round() as u16;
            Some(NonZeroU16::new(n).expect("is not 0"))
        };

        match (
            mwa_coarse_chan_nums.as_ref(),
            num_fine_chans_per_coarse_chan,
        ) {
            (Some(mwa_ccs), Some(n)) => {
                debug!("MWA coarse channel numbers: {mwa_ccs:?}");
                debug!("num_fine_chans_per_coarse_chan: {n}");
            }
            _ => debug!("This doesn't appear to be MWA data; no MWA coarse channels described"),
        }

        // Get other metadata.
        let obsid: Option<u32> = {
            match obs_table.get_cell::<f64>("MWA_GPS_TIME", 0) {
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
            let mut field_table = read_table(&ms, Some("FIELD"))?;
            let phase_vec = field_table.get_cell_as_vec("PHASE_DIR", 0)?;
            RADec::from_radians(phase_vec[0], phase_vec[1])
        };

        // Populate the dipole delays, gains and the pointing centre if we can.
        let mut dipole_delays: Option<Delays> = None;
        let mut dipole_gains: Option<_> = None;
        let mut pointing_centre: Option<RADec> = None;

        match (&mwalib_context, read_table(&ms, Some("MWA_TILE_POINTING"))) {
            // No metafits file was provided and MWA_TILE_POINTING doesn't
            // exist; we have no information on the dipole delays or gains. We
            // also know nothing about the pointing centre.
            (None, Err(_)) => {
                debug!("No dipole delays, dipole gains or pointing centre information available");
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

                // Re-order the tile delays and gains according to the uvfits
                // order, if possible.
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
                    [
                        "The MS antenna names are different to those supplied in the metafits."
                            .into(),
                        "Dipole delays/gains may be incorrectly mapped to MS antennas.".into(),
                    ]
                    .warn();
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

        // Get the observation's flagged channels per coarse band.
        let flagged_fine_chans: Vec<bool> = {
            // Get the first unflagged timestep. If there aren't any, get the
            // middle one.
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
                // then we assume its because there's a flag specified for each
                // polarisation. Which is dumb. If any of the 4 flags for a
                // channel are flagged, we consider the channel flagged.
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

            if let Some(num_fine_chans_per_coarse_chan) =
                num_fine_chans_per_coarse_chan.map(|n| n.get())
            {
                // Because we allow data to come in that might have channels
                // missing at the edges, there might be a different number of
                // fine channels in each coarse channel. Match the channel freqs
                // with what we think their index should be inside a coarse
                // channel.
                let chan_info = fine_chan_freqs_f64
                    .iter()
                    .enumerate()
                    .map(|(i_chan, &freq)| {
                        // Get the coarse channel number.
                        let cc_num = (freq / 1.28e6).round();
                        // Find the offset from the coarse channel centre.
                        let offset_hz = freq - cc_num * 1.28e6;
                        let i_offset: u16 = ((offset_hz / freq_res).round() as i32
                            + i32::from(num_fine_chans_per_coarse_chan) / 2)
                            .try_into()
                            .expect("smaller than u16::MAX");
                        (i_chan, i_offset, cc_num as u32)
                    })
                    .collect::<Vec<_>>();

                for i_chan in 0..num_fine_chans_per_coarse_chan {
                    let mut chan_is_flagged = true;
                    let mut chan_is_present = false;
                    for (full_index, _, _) in chan_info
                        .iter()
                        .filter(|(_, this_chan, _)| *this_chan == i_chan)
                    {
                        chan_is_present = true;
                        if !flagged_fine_chans[*full_index] {
                            chan_is_flagged = false;
                            break;
                        }
                    }
                    if chan_is_flagged && chan_is_present {
                        flagged_fine_chans_per_coarse_chan.push(i_chan);
                    }
                }
            }

            Vec1::try_from_vec(flagged_fine_chans_per_coarse_chan).ok()
        };
        let flagged_fine_chans = (0..)
            .zip(flagged_fine_chans)
            .filter(|(_, f)| *f)
            .map(|(i, _)| i)
            .collect();

        // Measurement sets don't appear to have an official way to supply the
        // DUT1. Marlu 0.9.0 writes UT1UTC into the main table's keywords, so
        // pick it up if it's there, otherwise use the metafits.
        let dut1 = match (
            main_table
                .get_keyword_record()?
                .get_field::<f64>("UT1UTC")
                .ok(),
            mwalib_context.as_ref(),
        ) {
            // If the MS has the key, then use it, even if we have a metafits.
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

        // Compare the first cross-correlation row's UVWs against UVWs that we
        // would make with the existing tiles. If they're negative of one
        // another, we need to negate our XYZs to match the UVWs the data use.
        let conjugate_vis = {
            let mut data_uvw = None;
            let mut tile1_xyz = None;
            let mut tile2_xyz = None;

            for i_row in 0.. {
                let uvw: Vec<f64> = main_table.get_cell_as_vec("UVW", i_row)?;
                let ant1: i32 = main_table.get_cell("ANTENNA1", i_row)?;
                let ant2: i32 = main_table.get_cell("ANTENNA2", i_row)?;

                // We need to use a cross-correlation baseline.
                if ant1 == ant2 {
                    continue;
                }

                data_uvw = Some(UVW {
                    u: uvw[0],
                    v: uvw[1],
                    w: uvw[2],
                });
                tile1_xyz = Some(tile_xyzs[ant1 as usize]);
                tile2_xyz = Some(tile_xyzs[ant2 as usize]);
                break;
            }

            if baseline_convention_is_different(
                // If this data somehow has no cross-correlation data, using
                // default values won't affect anything.
                data_uvw.unwrap_or_default(),
                tile1_xyz.unwrap_or_default(),
                tile2_xyz.unwrap_or_default(),
                array_position,
                phase_centre,
                *timestamps.first(),
                dut1,
            ) {
                "MS UVWs use the other baseline convention; will conjugate incoming visibilities"
                    .warn();
                true
            } else {
                false
            }
        };

        let obs_context = ObsContext {
            input_data_type: VisInputType::MeasurementSet,
            obsid,
            timestamps,
            all_timesteps,
            unflagged_timesteps,
            phase_centre,
            pointing_centre,
            array_position,
            supplied_array_position,
            dut1,
            tile_names,
            tile_xyzs,
            flagged_tiles,
            unavailable_tiles,
            autocorrelations_present,
            dipole_delays,
            dipole_gains,
            time_res,
            mwa_coarse_chan_nums,
            num_fine_chans_per_coarse_chan,
            freq_res: Some(freq_res),
            fine_chan_freqs,
            flagged_fine_chans,
            flagged_fine_chans_per_coarse_chan,
            polarisations: pols,
        };

        let ms = MsReader {
            obs_context,
            ms,
            step,
            metafits_context: mwalib_context,
            tile_map,
            ms_timestamps,
            data_col_name,
            weight_col_name,
            weight_col_is_2d,
            conjugate_vis,
        };
        Ok(ms)
    }

    /// An internal method for reading visibilities. Cross- and/or
    /// auto-correlation visibilities and weights are written to the supplied
    /// arrays.
    pub fn read_inner<const NUM_POLS: usize>(
        &self,
        mut crosses: Option<CrossData>,
        mut autos: Option<AutoData>,
        timestep: usize,
        flagged_fine_chans: &HashSet<u16>,
    ) -> Result<(), VisReadError> {
        // When reading in a new timestep's data, these indices should be
        // multiplied by `step` to get the amount of rows to stride in the main
        // table.
        let row_range_start = timestep * self.step;
        let row_range_end = (timestep + 1) * self.step;
        let row_range = row_range_start as u64..row_range_end as u64;

        let mut main_table = read_table(&self.ms, None).map_err(MsReadError::from)?;
        let chan_flags = (0..self.obs_context.fine_chan_freqs.len())
            .map(|i_chan| flagged_fine_chans.contains(&(i_chan as u16)))
            .collect::<Vec<_>>();
        main_table
            .for_each_row_in_range(row_range, |row| {
                // Check that the time associated with this row matches the
                // specified timestep.
                let this_timestamp = row.get_cell::<f64>("TIME")?;
                assert_eq!(
                    this_timestamp,
                    self.ms_timestamps[timestep],
                    "A MS row for timestep {timestep} did not match the expected timestamp value. This means the MS has a variable number of baselines per timestep, or the times are inexplicably changing, or there's a bug in the code."
                );

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
                                let (ms_weights_raw, ms_weights_offset) = ms_weights.into_raw_vec_and_offset();
                                assert!(ms_weights_offset.unwrap_or(0) == 0);
                                ms_weights_raw
                                    .chunks_exact(NUM_POLS)
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
                        let ms_flags: Array2<bool> = row.get_cell("FLAG")?;

                        if crosses.vis_fb.len_of(Axis(1)) < bl {
                            panic!(
                                "{}",
                                VisReadError::BadArraySize {
                                    array_type: "data_array",
                                    expected_len: bl,
                                    axis_num: 1,
                                }
                            );
                        }
                        if crosses.vis_fb.len_of(Axis(0)) > ms_data.len_of(Axis(0)) {
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
                        let mut out_vis = crosses.vis_fb.slice_mut(s![.., bl]);
                        let (ms_data_raw, ms_data_offset) = ms_data.into_raw_vec_and_offset();
                        assert!(ms_data_offset.unwrap_or(0) == 0);
                        ms_data_raw
                            .chunks_exact(NUM_POLS)
                            .zip(chan_flags.iter())
                            .filter(|(_, &chan_flag)| !chan_flag)
                            .zip(out_vis.iter_mut())
                            .for_each(|((ms_data, _chan_flag), out_vis)| {
                                let mut vis = Jones::default();
                                if NUM_POLS > 0 {
                                    vis[0] = ms_data[0];
                                }
                                if NUM_POLS > 1 {
                                    vis[1] = ms_data[1];
                                }
                                if NUM_POLS > 2 {
                                    vis[2] = ms_data[2];
                                }
                                if NUM_POLS > 3 {
                                    vis[3] = ms_data[3];
                                }
                                *out_vis = vis;
                            });

                        // Apply the flags to the weights (negate if flagged),
                        // and throw away 3 of the 4 weights; there are 4
                        // weights (for XX XY YX YY) and we assume that the
                        // first weight is the same as the others.
                        let mut out_weights = crosses.weights_fb.slice_mut(s![.., bl]);
                        let (ms_flags_raw, ms_flags_offset) = ms_flags.into_raw_vec_and_offset();
                        assert!(ms_flags_offset.unwrap_or(0) == 0);
                        ms_weights
                            .into_iter()
                            .zip(ms_flags_raw.chunks_exact(NUM_POLS))
                            .zip(chan_flags.iter())
                            .filter(|((_, _), &chan_flag)| !chan_flag)
                            .zip(out_weights.iter_mut())
                            .for_each(|(((weight, flags), _chan_flag), out_weight)| {
                                // Collapse the multiple flag values into a
                                // single one by finding any that are true (i.e.
                                // at least one polarisation is marked as
                                // flagged, so flag the whole visibility).
                                let flag = flags.iter().any(|f| *f);
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
                                    let (ms_weights_raw, ms_weights_offset) = ms_weights.into_raw_vec_and_offset();
                                    assert!(ms_weights_offset.unwrap_or(0) == 0);
                                    ms_weights_raw
                                        .chunks_exact(NUM_POLS)
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

                            if autos.vis_fb.len_of(Axis(1)) < i_ant {
                                panic!(
                                    "{}",
                                    VisReadError::BadArraySize {
                                        array_type: "data_array",
                                        expected_len: i_ant,
                                        axis_num: 1,
                                    }
                                );
                            }
                            if autos.vis_fb.len_of(Axis(0)) > ms_data.len_of(Axis(0)) {
                                panic!(
                                    "{}",
                                    VisReadError::BadArraySize {
                                        array_type: "data_array",
                                        expected_len: ms_data.len_of(Axis(0)),
                                        axis_num: 0,
                                    }
                                );
                            }

                            let mut out_vis = autos.vis_fb.slice_mut(s![.., i_ant]);
                            let (ms_data_raw, ms_data_offset) = ms_data.into_raw_vec_and_offset();
                            assert!(ms_data_offset.unwrap_or(0) == 0);
                            ms_data_raw
                                .chunks_exact(4)
                                .zip(chan_flags.iter())
                                .filter(|(_, &chan_flag)| !chan_flag)
                                .zip(out_vis.iter_mut())
                                .for_each(|((ms_data, _chan_flag), out_vis)| {
                                    let mut vis = Jones::default();
                                    if NUM_POLS > 0 {
                                        vis[0] = ms_data[0];
                                    }
                                    if NUM_POLS > 1 {
                                        vis[1] = ms_data[1];
                                    }
                                    if NUM_POLS > 2 {
                                        vis[2] = ms_data[2];
                                    }
                                    if NUM_POLS > 3 {
                                        vis[3] = ms_data[3];
                                    }
                                    *out_vis = vis;
                                });

                            let mut out_weights = autos.weights_fb.slice_mut(s![.., i_ant]);
                            let (flags_raw, flags_offset) = flags.into_raw_vec_and_offset();
                            assert!(flags_offset.unwrap_or(0) == 0);
                            ms_weights
                                .into_iter()
                                .zip(flags_raw.chunks_exact(4))
                                .zip(chan_flags.iter())
                                .filter(|((_, _), &chan_flag)| !chan_flag)
                                .zip(out_weights.iter_mut())
                                .for_each(|(((weight, flags), _chan_flag), out_weight)| {
                                    let flag = flags.iter().any(|f| *f);
                                    *out_weight = if flag { -weight.abs() } else { weight };
                                });
                        }
                    }
                }

                Ok(())
            })
            .map_err(MsReadError::from)?;

        // Transform the data, depending on what the actual polarisations are.
        if let Some(crosses) = crosses.as_mut() {
            let c0 = num_complex::Complex32::default();
            match (self.conjugate_vis, self.obs_context.polarisations) {
                // These pols are all handled correctly.
                (false, Polarisations::XX_XY_YX_YY) => (),
                (false, Polarisations::XX) => (),
                // Just conjugate.
                (true, Polarisations::XX_XY_YX_YY | Polarisations::XX) => {
                    crosses.vis_fb.mapv_inplace(|j| {
                        Jones::from([j[0].conj(), j[1].conj(), j[2].conj(), j[3].conj()])
                    })
                }

                // Because we read in one polarisation, it was treated as XX,
                // but this is actually YY.
                (false, Polarisations::YY) => crosses
                    .vis_fb
                    .mapv_inplace(|j| Jones::from([c0, c0, c0, j[0]])),
                (true, Polarisations::YY) => crosses
                    .vis_fb
                    .mapv_inplace(|j| Jones::from([c0, c0, c0, j[0].conj()])),

                // What looks like XY is YY.
                (false, Polarisations::XX_YY) => crosses
                    .vis_fb
                    .mapv_inplace(|j| Jones::from([j[0], c0, c0, j[1]])),
                (true, Polarisations::XX_YY) => crosses
                    .vis_fb
                    .mapv_inplace(|j| Jones::from([j[0].conj(), c0, c0, j[1].conj()])),

                // What looks like YX is YY.
                (false, Polarisations::XX_YY_XY) => crosses
                    .vis_fb
                    .mapv_inplace(|j| Jones::from([j[0], j[1], c0, j[2]])),
                (true, Polarisations::XX_YY_XY) => crosses
                    .vis_fb
                    .mapv_inplace(|j| Jones::from([j[0].conj(), j[1].conj(), c0, j[2].conj()])),
            }
        }
        if let Some(autos) = autos.as_mut() {
            let c0 = num_complex::Complex32::default();
            match (self.conjugate_vis, self.obs_context.polarisations) {
                // These pols are all handled correctly.
                (false, Polarisations::XX_XY_YX_YY) => (),
                (false, Polarisations::XX) => (),
                // Just conjugate.
                (true, Polarisations::XX_XY_YX_YY | Polarisations::XX) => {
                    autos.vis_fb.mapv_inplace(|j| {
                        Jones::from([j[0].conj(), j[1].conj(), j[2].conj(), j[3].conj()])
                    })
                }

                // Because we read in one polarisation, it was treated as XX,
                // but this is actually YY.
                (false, Polarisations::YY) => autos
                    .vis_fb
                    .mapv_inplace(|j| Jones::from([c0, c0, c0, j[0]])),
                (true, Polarisations::YY) => autos
                    .vis_fb
                    .mapv_inplace(|j| Jones::from([c0, c0, c0, j[0].conj()])),

                // What looks like XY is YY.
                (false, Polarisations::XX_YY) => autos
                    .vis_fb
                    .mapv_inplace(|j| Jones::from([j[0], c0, c0, j[1]])),
                (true, Polarisations::XX_YY) => autos
                    .vis_fb
                    .mapv_inplace(|j| Jones::from([j[0].conj(), c0, c0, j[1].conj()])),

                // What looks like YX is YY.
                (false, Polarisations::XX_YY_XY) => autos
                    .vis_fb
                    .mapv_inplace(|j| Jones::from([j[0], j[1], c0, j[2]])),
                (true, Polarisations::XX_YY_XY) => autos
                    .vis_fb
                    .mapv_inplace(|j| Jones::from([j[0].conj(), j[1].conj(), c0, j[2].conj()])),
            }
        }

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

    fn get_raw_data_corrections(&self) -> Option<RawDataCorrections> {
        None
    }

    fn set_raw_data_corrections(&mut self, _: RawDataCorrections) {}

    fn read_crosses_and_autos(
        &self,
        cross_vis_fb: ArrayViewMut2<Jones<f32>>,
        cross_weights_fb: ArrayViewMut2<f32>,
        auto_vis_fb: ArrayViewMut2<Jones<f32>>,
        auto_weights_fb: ArrayViewMut2<f32>,
        timestep: usize,
        tile_baseline_flags: &TileBaselineFlags,
        flagged_fine_chans: &HashSet<u16>,
    ) -> Result<(), VisReadError> {
        let cross_data = Some(CrossData {
            vis_fb: cross_vis_fb,
            weights_fb: cross_weights_fb,
            tile_baseline_flags,
        });
        let auto_data = Some(AutoData {
            vis_fb: auto_vis_fb,
            weights_fb: auto_weights_fb,
            tile_baseline_flags,
        });

        match self.obs_context.polarisations.num_pols() {
            4 => self.read_inner::<4>(cross_data, auto_data, timestep, flagged_fine_chans),
            3 => self.read_inner::<3>(cross_data, auto_data, timestep, flagged_fine_chans),
            2 => self.read_inner::<2>(cross_data, auto_data, timestep, flagged_fine_chans),
            1 => self.read_inner::<1>(cross_data, auto_data, timestep, flagged_fine_chans),
            _ => {
                unimplemented!("num pols must be 1-4")
            }
        }
    }

    fn read_crosses(
        &self,
        vis_fb: ArrayViewMut2<Jones<f32>>,
        weights_fb: ArrayViewMut2<f32>,
        timestep: usize,
        tile_baseline_flags: &TileBaselineFlags,
        flagged_fine_chans: &HashSet<u16>,
    ) -> Result<(), VisReadError> {
        let cross_data = Some(CrossData {
            vis_fb,
            weights_fb,
            tile_baseline_flags,
        });

        match self.obs_context.polarisations.num_pols() {
            4 => self.read_inner::<4>(cross_data, None, timestep, flagged_fine_chans),
            3 => self.read_inner::<3>(cross_data, None, timestep, flagged_fine_chans),
            2 => self.read_inner::<2>(cross_data, None, timestep, flagged_fine_chans),
            1 => self.read_inner::<1>(cross_data, None, timestep, flagged_fine_chans),
            _ => {
                unimplemented!("num pols must be 1-4")
            }
        }
    }

    fn read_autos(
        &self,
        vis_fb: ArrayViewMut2<Jones<f32>>,
        weights_fb: ArrayViewMut2<f32>,
        timestep: usize,
        tile_baseline_flags: &TileBaselineFlags,
        flagged_fine_chans: &HashSet<u16>,
    ) -> Result<(), VisReadError> {
        let auto_data = Some(AutoData {
            vis_fb,
            weights_fb,
            tile_baseline_flags,
        });

        match self.obs_context.polarisations.num_pols() {
            4 => self.read_inner::<4>(None, auto_data, timestep, flagged_fine_chans),
            3 => self.read_inner::<3>(None, auto_data, timestep, flagged_fine_chans),
            2 => self.read_inner::<2>(None, auto_data, timestep, flagged_fine_chans),
            1 => self.read_inner::<1>(None, auto_data, timestep, flagged_fine_chans),
            _ => {
                unimplemented!("num pols must be 1-4")
            }
        }
    }

    fn get_marlu_mwa_info(&self) -> Option<MarluMwaObsContext> {
        self.get_metafits_context()
            .map(MarluMwaObsContext::from_mwalib)
    }
}
