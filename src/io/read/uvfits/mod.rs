// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to handle reading from uvfits files.

mod error;
#[cfg(test)]
mod tests;

pub(crate) use error::*;

use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    os::raw::c_char,
    path::{Path, PathBuf},
};

use fitsio::{errors::check_status as fits_check_status, hdu::FitsHdu, FitsFile};
use hifitime::{Duration, Epoch};
use log::{debug, trace, warn};
use marlu::{
    constants::{
        COTTER_MWA_HEIGHT_METRES, COTTER_MWA_LATITUDE_RADIANS, COTTER_MWA_LONGITUDE_RADIANS,
    },
    io::uvfits::decode_uvfits_baseline,
    Jones, LatLngHeight, RADec, XyzGeocentric, XyzGeodetic,
};
use mwalib::MetafitsContext;
use ndarray::prelude::*;

use super::*;
use crate::io::read::fits::{fits_get_col, fits_get_optional_key, fits_get_required_key};
use crate::{
    beam::Delays,
    context::ObsContext,
    io::read::{
        fits::{fits_open, fits_open_hdu},
        VisRead, VisReadError,
    },
    metafits::{get_dipole_delays, get_dipole_gains, map_antenna_order},
    misc::quantize_duration,
};

pub(crate) struct UvfitsReader {
    /// Observation metadata.
    pub(super) obs_context: ObsContext,

    // uvfits-specific things follow.
    /// The path to the uvfits on disk.
    pub(crate) uvfits: PathBuf,

    /// The uvfits-specific metadata, like which indices contain which
    /// parameters.
    metadata: UvfitsMetadata,

    /// The "stride" of the data, i.e. the number of rows (baselines) before the
    /// time index changes.
    step: usize,

    /// The [`mwalib::MetafitsContext`] used when [`MsReader`] was created.
    metafits_context: Option<MetafitsContext>,

    /// uvfits files may number their tiles according to the antenna table, and
    /// not necessarily from 0 to the total number of tiles. This map converts a
    /// uvfits tile number to a 0-to-total index.
    tile_map: HashMap<usize, usize>,
}

impl UvfitsReader {
    /// Verify and populate metadata associated with this measurement set.
    ///
    /// The measurement set is expected to be formatted in the way that
    /// cotter/Birli write measurement sets.
    pub(crate) fn new<P: AsRef<Path>, P2: AsRef<Path>>(
        uvfits: P,
        metafits: Option<P2>,
        array_pos: Option<LatLngHeight>,
    ) -> Result<UvfitsReader, VisReadError> {
        fn inner(
            uvfits: &Path,
            metafits: Option<&Path>,
            array_position: Option<LatLngHeight>,
        ) -> Result<UvfitsReader, UvfitsReadError> {
            // If a metafits file was provided, get an mwalib object ready.
            // TODO: Let the user supply the MWA version.
            let mwalib_context = match metafits {
                None => None,
                Some(m) => Some(MetafitsContext::new(m, None)?),
            };

            debug!("Using uvfits file: {}", uvfits.display());
            if !uvfits.exists() {
                return Err(UvfitsReadError::BadFile(uvfits.to_path_buf()));
            }

            // Get the tile names, XYZ positions and antenna numbers.
            let mut uvfits_fptr = fits_open(uvfits)?;
            let primary_hdu = fits_open_hdu(&mut uvfits_fptr, 0)?;
            let antenna_table_hdu = fits_open_hdu(&mut uvfits_fptr, "AIPS AN")?;

            let tile_names: Vec<String> =
                fits_get_col(&mut uvfits_fptr, &antenna_table_hdu, "ANNAME")?;
            let tile_names =
                Vec1::try_from_vec(tile_names).map_err(|_| UvfitsReadError::AnnameEmpty)?;
            let total_num_tiles = tile_names.len();

            // Set up the tile map.
            let tile_nums: Vec<u32> = fits_get_col(&mut uvfits_fptr, &antenna_table_hdu, "NOSTA")?;
            let tile_map: HashMap<usize, usize> = tile_nums
                .into_iter()
                .zip(0..total_num_tiles)
                .map(|(a, b)| (a.try_into().expect("not larger than usize::MAX"), b))
                .collect();

            // Determine the array position from the uvfits file, if the user
            // didn't supply a position.
            let array_position = match array_position {
                Some(p) => Some(p),
                None => {
                    let frame: Option<String> =
                        fits_get_optional_key(&mut uvfits_fptr, &antenna_table_hdu, "FRAME")?;
                    // The uvfits standard only defines one frame (ITRF). So warn
                    // the user if this isn't explicit, but we assume this is always
                    // used.
                    let itrf_frame_warning = match frame.as_ref().map(|s| s.trim()) {
                        Some("ITRF") => None,
                        _ => Some("Assuming that the uvfits antenna coordinate system is ITRF"),
                    };
                    let array_x: Option<f64> =
                        fits_get_optional_key(&mut uvfits_fptr, &antenna_table_hdu, "ARRAYX")?;
                    let array_y: Option<f64> =
                        fits_get_optional_key(&mut uvfits_fptr, &antenna_table_hdu, "ARRAYY")?;
                    let array_z: Option<f64> =
                        fits_get_optional_key(&mut uvfits_fptr, &antenna_table_hdu, "ARRAYZ")?;
                    let array_position = match (array_x, array_y, array_z) {
                        (Some(x), Some(y), Some(z)) => {
                            debug!("uvfits ARRAYX: {x}");
                            debug!("uvfits ARRAYY: {y}");
                            debug!("uvfits ARRAYZ: {z}");
                            if let Some(itrf_frame_warning) = itrf_frame_warning {
                                warn!("{itrf_frame_warning}");
                            }
                            Some(XyzGeocentric { x, y, z }.to_earth_wgs84())
                        }
                        (None, None, None) => None,
                        _ => {
                            warn!("Only a subset of uvfits ARRAYX, ARRAYY, ARRAYZ is available; ignoring present values");
                            None
                        }
                    };

                    // If we couldn't find the position based off of
                    // ARRAY{X,Y,Z}, then we can guess based off the SOFTWARE
                    // key.
                    match array_position {
                        Some(p) => Some(p),
                        None => {
                            let software: Option<UvfitsFlavour> = fits_get_optional_key::<String>(
                                &mut uvfits_fptr,
                                &primary_hdu,
                                "SOFTWARE",
                            )?
                            .map(|s| get_uvfits_flavour(&s));
                            match software {
                                Some(
                                    UvfitsFlavour::Hyperdrive
                                    | UvfitsFlavour::Birli
                                    | UvfitsFlavour::Marlu,
                                ) => {
                                    warn!("Assuming that this uvfits's array position is the MWA");
                                    Some(LatLngHeight::mwa())
                                }

                                Some(UvfitsFlavour::Cotter) => {
                                    warn!("Assuming that this measurement set's array position is cotter's MWA position");
                                    Some(LatLngHeight {
                                        longitude_rad: COTTER_MWA_LONGITUDE_RADIANS,
                                        latitude_rad: COTTER_MWA_LATITUDE_RADIANS,
                                        height_metres: COTTER_MWA_HEIGHT_METRES,
                                    })
                                }

                                Some(UvfitsFlavour::Other) | None => None,
                            }
                        }
                    }
                }
            };

            let tile_xyzs = {
                let mut tile_xyzs: Vec<XyzGeocentric> = Vec::with_capacity(total_num_tiles);
                for i in 0..total_num_tiles {
                    let fits_xyz = read_cell_array(
                        &mut uvfits_fptr,
                        &antenna_table_hdu,
                        "STABXYZ",
                        i.try_into().expect("not larger than i64::MAX"),
                        3,
                    )?;
                    tile_xyzs.push(XyzGeocentric {
                        x: fits_xyz[0],
                        y: fits_xyz[1],
                        z: fits_xyz[2],
                    });
                }

                // Attempt to detect whether these coordinates are actually
                // geocentric (as they should be!).
                if detect_geocentric_antenna_positions(&tile_xyzs) {
                    // These are geocentric positions, but we need geodetic
                    // ones. To convert, we need the array position. If we
                    // couldn't determine it, we need to bail out here.
                    let array_position = array_position.ok_or(UvfitsReadError::NoArrayPos)?;

                    let vec = XyzGeocentric::get_geocentric_vector(array_position);
                    let (s_long, c_long) = array_position.longitude_rad.sin_cos();
                    tile_xyzs
                        .into_iter()
                        .map(|geocentric| geocentric.to_geodetic_inner(vec, s_long, c_long))
                        .collect()
                } else {
                    // Just convert the type to what it should be.
                    warn!("Detected geodetic antenna positions in the uvfits, when it should provide geocentric ones. Converting.");
                    tile_xyzs
                        .into_iter()
                        .map(|false_geocentric| XyzGeodetic {
                            x: false_geocentric.x,
                            y: false_geocentric.y,
                            z: false_geocentric.z,
                        })
                        .collect()
                }
            };
            let tile_xyzs = Vec1::try_from_vec(tile_xyzs)
                .expect("can't be empty, non-empty tile names verified above");

            let metadata = UvfitsMetadata::new(&mut uvfits_fptr, &primary_hdu)?;
            debug!("Number of rows in the uvfits:   {}", metadata.num_rows);
            debug!("PCOUNT:                         {}", metadata.pcount);
            debug!("Number of polarisations:        {}", metadata.num_pols);
            debug!(
                "Floats per polarisation:        {}",
                metadata.num_floats_per_pol
            );
            debug!(
                "Number of fine frequency chans: {}",
                metadata.num_fine_freq_chans
            );
            debug!("UU index:       {}", metadata.indices.u);
            debug!("VV index:       {}", metadata.indices.v);
            debug!("WW index:       {}", metadata.indices.w);
            match metadata.indices.baseline_or_antennas {
                BaselineOrAntennas::Baseline { index } => debug!("BASELINE index: {index}"),
                BaselineOrAntennas::Antennas { index1, index2 } => {
                    debug!("ANTENNA1 index: {index1}");
                    debug!("ANTENNA2 index: {index2}");
                }
            }
            debug!("DATE index:     {}", metadata.indices.date1);
            if let Some(d2) = metadata.indices.date2 {
                debug!("(Second) DATE index: {}", d2);
            }
            debug!("COMPLEX index:  {}", metadata.indices.complex);
            debug!("STOKES index:   {}", metadata.indices.stokes);
            debug!("FREQ index:     {}", metadata.indices.freq);
            debug!("RA index:       {}", metadata.indices.ra);
            debug!("DEC index:      {}", metadata.indices.dec);

            if metadata.num_rows == 0 {
                return Err(UvfitsReadError::Empty(uvfits.to_path_buf()));
            }

            // The phase centre is described by RA and DEC if there is no SOURCE
            // table (as per the standard).
            // TODO: Check that there is no SOURCE table!
            let phase_centre = {
                let ra = fits_get_required_key(
                    &mut uvfits_fptr,
                    &primary_hdu,
                    &format!("CRVAL{}", metadata.indices.ra),
                )?;
                let dec = fits_get_required_key(
                    &mut uvfits_fptr,
                    &primary_hdu,
                    &format!("CRVAL{}", metadata.indices.dec),
                )?;
                RADec::from_degrees(ra, dec)
            };

            // Populate the dipole delays, gains and the pointing centre if we
            // can.
            let mut dipole_delays: Option<Delays> = None;
            let mut dipole_gains: Option<_> = None;
            let mut pointing_centre: Option<RADec> = None;
            if let Some(context) = &mwalib_context {
                debug!("Using metafits for dipole delays, gains and pointing centre");
                let delays = get_dipole_delays(context);
                let gains = get_dipole_gains(context);
                pointing_centre = Some(RADec::from_degrees(
                    context.ra_tile_pointing_degrees,
                    context.dec_tile_pointing_degrees,
                ));

                // Re-order the tile delays and gains according to the uvfits
                // order, if possible.
                if let Some(map) = map_antenna_order(context, &tile_names) {
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
                        "The uvfits antenna names are different to those supplied in the metafits."
                    );
                    warn!("Dipole delays/gains may be incorrectly mapped to uvfits antennas.");
                    dipole_delays = Some(Delays::Full(delays));
                    dipole_gains = Some(gains);
                }
            }

            // Work out which tiles are unavailable.
            let mut present_tiles_set: HashSet<usize> = HashSet::new();
            metadata.uvfits_baselines.iter().for_each(|&uvfits_bl| {
                let (uvfits_ant1, uvfits_ant2) = decode_uvfits_baseline(uvfits_bl);
                present_tiles_set.insert(uvfits_ant1);
                present_tiles_set.insert(uvfits_ant2);
            });
            metadata.uvfits_antennas_set.iter().for_each(|&uvfits_ant| {
                present_tiles_set.insert(uvfits_ant);
            });
            let unavailable_tiles = {
                let mut v = tile_map
                    .iter()
                    .filter(|(i, _)| !present_tiles_set.contains(*i))
                    .map(|(_, v)| *v)
                    .collect::<Vec<_>>();
                v.sort_unstable();
                v
            };
            // Similar to the measurement set situation, we have no way(?) to
            // identify tiles in the uvfits file that have data but shouldn't be
            // used (i.e. they are flagged). So, we assume all tiles are
            // unflagged, except those that are "unavailable" (determined
            // above).
            let flagged_tiles = unavailable_tiles.clone();

            // Work out the timestamp epochs. The file tells us what time standard
            // is being used (probably UTC). If this is false, then we assume TAI.
            let uses_utc_time = {
                let timsys: Option<String> =
                    fits_get_optional_key(&mut uvfits_fptr, &antenna_table_hdu, "TIMSYS")?;
                match timsys {
                    None => {
                        debug!("No TIMSYS present; assuming UTC");
                        true
                    }
                    Some(timsys) => {
                        if timsys.starts_with("UTC") {
                            true
                        } else if timsys.starts_with("IAT") || timsys.starts_with("TAI") {
                            false
                        } else {
                            return Err(UvfitsReadError::UnknownTimsys(timsys));
                        }
                    }
                }
            };

            // uvfits timestamps are in the middle of their respective integration
            // periods, so no adjustment is needed here.
            let jd_zero = if uses_utc_time {
                Epoch::from_jde_utc(metadata.jd_zero)
            } else {
                Epoch::from_jde_tai(metadata.jd_zero)
            };

            // the number of nanoseconds to quantize to.
            let q = 10_000_000.;

            let (all_timesteps, timestamps): (Vec<usize>, Vec<Epoch>) = metadata
                .jd_frac_timestamps
                .iter()
                .enumerate()
                .map(|(i, &frac)| {
                    let jd_offset = Duration::from_days(frac);
                    (i, jd_zero + quantize_duration(jd_offset, q))
                })
                .unzip();
            // TODO: Determine flagging!
            let unflagged_timesteps = all_timesteps.clone();
            let all_timesteps =
                Vec1::try_from_vec(all_timesteps).map_err(|_| UvfitsReadError::NoTimesteps {
                    file: uvfits_fptr.filename.clone(),
                })?;
            let timestamps =
                Vec1::try_from_vec(timestamps).map_err(|_| UvfitsReadError::NoTimesteps {
                    file: uvfits_fptr.filename.clone(),
                })?;

            // Get the data's time resolution. There is a possibility that the file
            // contains only one timestep.
            let time_res = {
                // If it's available, the integration time should be written out
                // as a single-precision float. Reading as a double-precision
                // float means cfitsio either promotes the single or returns the
                // double that was written. I really don't mind if the key
                // breaks the standard here...
                let int_time: Option<f64> =
                    fits_get_optional_key(&mut uvfits_fptr, &primary_hdu, "INTTIM")?;
                match int_time {
                    Some(t) => {
                        let d = Duration::from_seconds(t);
                        trace!("Time resolution from INTTIM: {}s", d.to_seconds());
                        Some(d)
                    }
                    None => {
                        if timestamps.len() == 1 {
                            debug!("Only one timestep is present in the data; can't determine the data's time resolution.");
                            None
                        } else {
                            // Find the minimum gap between two consecutive
                            // timestamps.
                            let time_res = timestamps
                                .windows(2)
                                .fold(Duration::from_seconds(f64::INFINITY), |acc, ts| {
                                    acc.min(ts[1] - ts[0])
                                });
                            trace!(
                                "Time resolution from smallest gap: {}s",
                                time_res.to_seconds()
                            );
                            Some(time_res)
                        }
                    }
                }
            };
            match timestamps.as_slice() {
                // Handled above; uvfits files aren't allowed to be empty.
                [] => unreachable!(),
                [t] => debug!("Only timestep (GPS): {:.2}", t.to_gpst_seconds()),
                [t0, .., tn] => {
                    debug!("First good timestep (GPS): {:.2}", t0.to_gpst_seconds());
                    debug!("Last good timestep  (GPS): {:.2}", tn.to_gpst_seconds());
                }
            }

            debug!("Unavailable tiles in the uvfits: {unavailable_tiles:?}");
            debug!("Flagged tiles in the uvfits: {flagged_tiles:?}");
            debug!(
                "Autocorrelations present: {}",
                metadata.autocorrelations_present
            );

            // Get the obsid. There is an "obs. name" in the "object" key, but
            // that's not the same thing.
            let obsid = mwalib_context.as_ref().map(|context| context.obs_id);

            let step = metadata.num_rows / timestamps.len();

            let freq_val_str = format!("CRVAL{}", metadata.indices.freq);
            let base_freq_str: String =
                fits_get_required_key(&mut uvfits_fptr, &primary_hdu, &freq_val_str)?;
            let base_freq: f64 = match base_freq_str.parse() {
                Ok(p) => p,
                Err(e) => {
                    return Err(UvfitsReadError::Parse {
                        key: Cow::from(freq_val_str),
                        value: base_freq_str,
                        parse_error: e.to_string(),
                    })
                }
            };
            let base_index: isize = {
                // CRPIX might be a float. Parse it as one, then make it an int.
                let freq_val_str = format!("CRPIX{}", metadata.indices.freq);
                let f_str: String =
                    fits_get_required_key(&mut uvfits_fptr, &primary_hdu, &freq_val_str)?;
                let f: f64 = match f_str.parse() {
                    Ok(p) => p,
                    Err(e) => {
                        return Err(UvfitsReadError::Parse {
                            key: Cow::from(freq_val_str),
                            value: f_str,
                            parse_error: e.to_string(),
                        })
                    }
                };
                f.round() as _
            };
            let freq_val_str = format!("CDELT{}", metadata.indices.freq);
            let fine_chan_width_str: String =
                fits_get_required_key(&mut uvfits_fptr, &primary_hdu, &freq_val_str)?;
            let freq_res: f64 = match fine_chan_width_str.parse() {
                Ok(p) => p,
                Err(e) => {
                    return Err(UvfitsReadError::Parse {
                        key: Cow::from(freq_val_str),
                        value: fine_chan_width_str,
                        parse_error: e.to_string(),
                    })
                }
            };

            let mut fine_chan_freqs = Vec::with_capacity(metadata.num_fine_freq_chans);
            for i in 0..metadata.num_fine_freq_chans {
                fine_chan_freqs.push(
                    (base_freq + (i as isize - base_index + 1) as f64 * freq_res).round() as u64,
                );
            }
            let fine_chan_freqs = Vec1::try_from_vec(fine_chan_freqs).unwrap();

            let (coarse_chan_nums, coarse_chan_freqs) = match mwalib_context.as_ref() {
                Some(context) => {
                    // Get the coarse channel information out of the metafits
                    // file, but only the ones aligned with the frequencies in
                    // the uvfits file.
                    let cc_width = f64::from(context.coarse_chan_width_hz);

                    context
                        .metafits_coarse_chans
                        .iter()
                        .map(|cc| {
                            let gpubox_num: u32 = cc
                                .gpubox_number
                                .try_into()
                                .expect("not larger than u32::MAX");
                            (gpubox_num, f64::from(cc.chan_centre_hz))
                        })
                        .filter(|(_, cc_freq)| {
                            fine_chan_freqs
                                .iter()
                                .any(|f| (*f as f64 - *cc_freq).abs() < cc_width / 2.0)
                        })
                        .unzip()
                }
                None => {
                    // Divide each chunk of fine channels per coarse channel
                    let coarse_chan_freqs = fine_chan_freqs
                        // We assume 1.28 MHz per coarse channel.
                        .chunks_exact(1_280_000 / freq_res as usize)
                        .map(|chunk| {
                            if chunk.len() % 2 == 0 {
                                // We round the coarse channel freqs hoping
                                // there isn't any sub-Hz structure.
                                ((chunk[chunk.len() / 2 - 1] + chunk[chunk.len() / 2]) / 2) as f64
                            } else {
                                chunk[chunk.len() / 2] as f64
                            }
                        })
                        .collect::<Vec<_>>();
                    let coarse_chan_nums = (1..coarse_chan_freqs.len() + 1)
                        .map(|n| n.try_into().expect("not larger than u32::MAX"))
                        .collect();

                    (coarse_chan_nums, coarse_chan_freqs)
                }
            };
            debug!("Coarse channel numbers: {:?}", coarse_chan_nums);
            debug!(
                "Coarse channel centre frequencies [Hz]: {:?}",
                coarse_chan_freqs
            );

            let dut1 = {
                let antenna_table_hdu = fits_open_hdu(&mut uvfits_fptr, "AIPS AN")?;
                let uvfits_dut1: Option<f64> =
                    fits_get_optional_key(&mut uvfits_fptr, &antenna_table_hdu, "UT1UTC")?;
                match uvfits_dut1 {
                    Some(dut1) => debug!("uvfits DUT1: {dut1}"),
                    None => debug!("uvfits has no DUT1 (UT1UTC key)"),
                }

                let metafits_dut1 = match mwalib_context.as_ref() {
                    Some(c) => match c.dut1 {
                        Some(dut1) => {
                            debug!("metafits DUT1: {dut1}");
                            Some(dut1)
                        }
                        None => {
                            debug!("metafits has no DUT1");
                            None
                        }
                    },
                    None => None,
                };

                if metafits_dut1.is_some() && uvfits_dut1.is_some() {
                    debug!("Preferring metafits DUT1 over uvfits DUT1");
                }
                metafits_dut1.or(uvfits_dut1).map(Duration::from_seconds)
            };

            let obs_context = ObsContext {
                obsid,
                timestamps,
                all_timesteps,
                unflagged_timesteps,
                phase_centre,
                pointing_centre,
                array_position,
                dut1,
                tile_names,
                tile_xyzs,
                flagged_tiles,
                unavailable_tiles,
                autocorrelations_present: metadata.autocorrelations_present,
                dipole_delays,
                dipole_gains,
                time_res,
                coarse_chan_nums,
                coarse_chan_freqs,
                num_fine_chans_per_coarse_chan: metadata.num_fine_freq_chans,
                freq_res: Some(freq_res),
                fine_chan_freqs,
                // TODO: Get flagging right. I think that info is in an optional table.
                flagged_fine_chans: vec![],
                flagged_fine_chans_per_coarse_chan: vec![],
            };

            Ok(UvfitsReader {
                obs_context,
                uvfits: uvfits.to_path_buf(),
                metadata,
                step,
                metafits_context: mwalib_context,
                tile_map,
            })
        }
        inner(
            uvfits.as_ref(),
            metafits.as_ref().map(|f| f.as_ref()),
            array_pos,
        )
        .map_err(VisReadError::from)
    }

    fn read_inner(
        &self,
        mut crosses: Option<CrossData>,
        mut autos: Option<AutoData>,
        timestep: usize,
        flagged_fine_chans: &HashSet<usize>,
    ) -> Result<(), VisReadError> {
        let row_range_start = timestep * self.step;
        let row_range_end = (timestep + 1) * self.step;

        let mut uvfits = fits_open(&self.uvfits).map_err(UvfitsReadError::from)?;
        fits_open_hdu(&mut uvfits, 0).map_err(UvfitsReadError::from)?;
        let mut group_params: Vec<f32> = vec![0.0; self.metadata.pcount];
        let mut uvfits_vis: Vec<f32> = vec![
            0.0;
            self.metadata.num_fine_freq_chans
                * self.metadata.num_pols
                * usize::from(self.metadata.num_floats_per_pol)
        ];
        for row in row_range_start..row_range_end {
            // Read in the row's group parameters.
            let mut status = 0;
            unsafe {
                // ffggpe = fits_read_grppar_flt
                fitsio_sys::ffggpe(
                    uvfits.as_raw(), /* I - FITS file pointer                       */
                    (1 + row).try_into().expect("not larger than i64::MAX"), /* I - group to read (1 = 1st group)           */
                    1, /* I - first vector element to read (1 = 1st)  */
                    group_params
                        .len()
                        .try_into()
                        .expect("not larger than i64::MAX"), /* I - number of values to read                */
                    group_params.as_mut_ptr(), /* O - array of values that are returned       */
                    &mut status,               /* IO - error status                           */
                );
                fits_check_status(status).map_err(|err| UvfitsReadError::ReadVis {
                    row_num: row + 1,
                    err,
                })?;
            };

            let (uvfits_ant1, uvfits_ant2) = match self.metadata.indices.baseline_or_antennas {
                BaselineOrAntennas::Baseline { index } => {
                    let uvfits_bl = group_params[usize::from(index - 1)];
                    decode_uvfits_baseline(uvfits_bl as usize)
                }
                BaselineOrAntennas::Antennas { index1, index2 } => (
                    group_params[usize::from(index1 - 1)] as usize,
                    group_params[usize::from(index2 - 1)] as usize,
                ),
            };
            let (ant1, ant2) = (self.tile_map[&uvfits_ant1], self.tile_map[&uvfits_ant2]);

            if let Some(crosses) = crosses.as_mut() {
                if let Some(i_baseline) = crosses
                    .tile_baseline_flags
                    .tile_to_unflagged_cross_baseline_map
                    .get(&(ant1, ant2))
                    .copied()
                {
                    unsafe {
                        // ffgpve = fits_read_img_flt
                        fitsio_sys::ffgpve(
                            uvfits.as_raw(), /* I - FITS file pointer                       */
                            (1 + row).try_into().expect("not larger than i64::MAX"), /* I - group to read (1 = 1st group)           */
                            1, /* I - first vector element to read (1 = 1st)  */
                            uvfits_vis
                                .len()
                                .try_into()
                                .expect("not larger than i64::MAX"), /* I - number of values to read                */
                            0.0, /* I - value for undefined pixels              */
                            uvfits_vis.as_mut_ptr(), /* O - array of values that are returned       */
                            &mut 0,      /* O - set to 1 if any values are null; else 0 */
                            &mut status, /* IO - error status                           */
                        );
                    }
                    fits_check_status(status).map_err(|err| UvfitsReadError::ReadVis {
                        row_num: row + 1,
                        err,
                    })?;

                    // Put the data and weights into the shared arrays outside this
                    // scope. Before we can do this, we need to remove any
                    // globally-flagged fine channels.
                    let mut out_vis = crosses.data_array.slice_mut(s![.., i_baseline]);
                    let mut out_weights = crosses.weights_array.slice_mut(s![.., i_baseline]);

                    // The following is the closest I can do to a constexpr.
                    match (self.metadata.num_pols, self.metadata.num_floats_per_pol) {
                        (4, 3) => {
                            uvfits_vis
                                .chunks_exact(12)
                                .enumerate()
                                .filter(|(i_chan, _)| !flagged_fine_chans.contains(i_chan))
                                .zip(out_vis.iter_mut())
                                .zip(out_weights.iter_mut())
                                .for_each(|(((_, in_data), out_vis), out_weight)| {
                                    *out_vis = Jones::from([
                                        in_data[0],  // XX real
                                        in_data[1],  // XX imag
                                        in_data[6],  // XY real
                                        in_data[7],  // XY imag
                                        in_data[9],  // YX real
                                        in_data[10], // YX imag
                                        in_data[3],  // YY real
                                        in_data[4],  // YY imag
                                    ]);
                                    // Using f32::min is slower than doing a
                                    // normal comparison (i.e. > or <), because
                                    // it also checks for NaN. But, the
                                    // performance hit seems negligable compared
                                    // to writing out the data to a transposed
                                    // array.
                                    *out_weight = [in_data[2], in_data[5], in_data[8], in_data[11]]
                                        .into_iter()
                                        .reduce(f32::min)
                                        .expect("not empty");
                                });
                        }
                        (4, 2) => {
                            uvfits_vis
                                .chunks_exact(8)
                                .enumerate()
                                .filter(|(i_chan, _)| !flagged_fine_chans.contains(i_chan))
                                .zip(out_vis.iter_mut())
                                .zip(out_weights.iter_mut())
                                .for_each(|(((_, in_data), out_vis), out_weight)| {
                                    *out_vis = Jones::from([
                                        in_data[0], // XX real
                                        in_data[1], // XX imag
                                        in_data[4], // XY real
                                        in_data[5], // XY imag
                                        in_data[6], // YX real
                                        in_data[7], // YX imag
                                        in_data[2], // YY real
                                        in_data[3], // YY imag
                                    ]);
                                    *out_weight = 1.0;
                                });
                        }
                        (3, 3) => {
                            uvfits_vis
                                .chunks_exact(9)
                                .enumerate()
                                .filter(|(i_chan, _)| !flagged_fine_chans.contains(i_chan))
                                .zip(out_vis.iter_mut())
                                .zip(out_weights.iter_mut())
                                .for_each(|(((_, in_data), out_vis), out_weight)| {
                                    *out_vis = Jones::from([
                                        in_data[0], // XX real
                                        in_data[1], // XX imag
                                        in_data[6], // XY real
                                        in_data[7], // XY imag
                                        0.0,        // YX real
                                        0.0,        // YX imag
                                        in_data[3], // YY real
                                        in_data[4], // YY imag
                                    ]);
                                    *out_weight = [in_data[2], in_data[5], in_data[8]]
                                        .into_iter()
                                        .reduce(f32::min)
                                        .expect("not empty");
                                });
                        }
                        (3, 2) => {
                            uvfits_vis
                                .chunks_exact(6)
                                .enumerate()
                                .filter(|(i_chan, _)| !flagged_fine_chans.contains(i_chan))
                                .zip(out_vis.iter_mut())
                                .zip(out_weights.iter_mut())
                                .for_each(|(((_, in_data), out_vis), out_weight)| {
                                    *out_vis = Jones::from([
                                        in_data[0], // XX real
                                        in_data[1], // XX imag
                                        in_data[4], // XY real
                                        in_data[5], // XY imag
                                        0.0,        // YX real
                                        0.0,        // YX imag
                                        in_data[2], // YY real
                                        in_data[3], // YY imag
                                    ]);
                                    *out_weight = 1.0;
                                });
                        }
                        (2, 3) => {
                            uvfits_vis
                                .chunks_exact(6)
                                .enumerate()
                                .filter(|(i_chan, _)| !flagged_fine_chans.contains(i_chan))
                                .zip(out_vis.iter_mut())
                                .zip(out_weights.iter_mut())
                                .for_each(|(((_, in_data), out_vis), out_weight)| {
                                    *out_vis = Jones::from([
                                        in_data[0], // XX real
                                        in_data[1], // XX imag
                                        0.0,        // XY real
                                        0.0,        // XY imag
                                        0.0,        // YX real
                                        0.0,        // YX imag
                                        in_data[3], // YY real
                                        in_data[4], // YY imag
                                    ]);
                                    *out_weight = [in_data[2], in_data[5]]
                                        .into_iter()
                                        .reduce(f32::min)
                                        .expect("not empty");
                                });
                        }
                        (2, 2) => {
                            uvfits_vis
                                .chunks_exact(4)
                                .enumerate()
                                .filter(|(i_chan, _)| !flagged_fine_chans.contains(i_chan))
                                .zip(out_vis.iter_mut())
                                .zip(out_weights.iter_mut())
                                .for_each(|(((_, in_data), out_vis), out_weight)| {
                                    *out_vis = Jones::from([
                                        in_data[0], // XX real
                                        in_data[1], // XX imag
                                        0.0,        // XY real
                                        0.0,        // XY imag
                                        0.0,        // YX real
                                        0.0,        // YX imag
                                        in_data[2], // YY real
                                        in_data[3], // YY imag
                                    ]);
                                    *out_weight = 1.0;
                                });
                        }
                        (1, 3) => {
                            uvfits_vis
                                .chunks_exact(3)
                                .enumerate()
                                .filter(|(i_chan, _)| !flagged_fine_chans.contains(i_chan))
                                .zip(out_vis.iter_mut())
                                .zip(out_weights.iter_mut())
                                .for_each(|(((_, in_data), out_vis), out_weight)| {
                                    *out_vis = Jones::from([
                                        in_data[0], // XX real
                                        in_data[1], // XX imag
                                        0.0,        // XY real
                                        0.0,        // XY imag
                                        0.0,        // YX real
                                        0.0,        // YX imag
                                        0.0,        // YY real
                                        0.0,        // YY imag
                                    ]);
                                    *out_weight = in_data[2];
                                });
                        }
                        (1, 2) => {
                            uvfits_vis
                                .chunks_exact(2)
                                .enumerate()
                                .filter(|(i_chan, _)| !flagged_fine_chans.contains(i_chan))
                                .zip(out_vis.iter_mut())
                                .zip(out_weights.iter_mut())
                                .for_each(|(((_, in_data), out_vis), out_weight)| {
                                    *out_vis = Jones::from([
                                        in_data[0], // XX real
                                        in_data[1], // XX imag
                                        0.0,        // XY real
                                        0.0,        // XY imag
                                        0.0,        // YX real
                                        0.0,        // YX imag
                                        0.0,        // YY real
                                        0.0,        // YY imag
                                    ]);
                                    *out_weight = 1.0;
                                });
                        }
                        _ => unreachable!(),
                    }
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
                        unsafe {
                            // ffgpve = fits_read_img_flt
                            fitsio_sys::ffgpve(
                                uvfits.as_raw(), /* I - FITS file pointer                       */
                                (1 + row).try_into().expect("not larger than i64::MAX"), /* I - group to read (1 = 1st group)           */
                                1, /* I - first vector element to read (1 = 1st)  */
                                uvfits_vis
                                    .len()
                                    .try_into()
                                    .expect("not larger than i64::MAX"), /* I - number of values to read                */
                                0.0, /* I - value for undefined pixels              */
                                uvfits_vis.as_mut_ptr(), /* O - array of values that are returned       */
                                &mut 0,      /* O - set to 1 if any values are null; else 0 */
                                &mut status, /* IO - error status                           */
                            );
                        }
                        fits_check_status(status).map_err(|err| UvfitsReadError::ReadVis {
                            row_num: row + 1,
                            err,
                        })?;

                        let mut out_vis = autos.data_array.slice_mut(s![.., i_ant]);
                        let mut out_weights = autos.weights_array.slice_mut(s![.., i_ant]);
                        // Auto-correlations are a lower priority in hyperdrive,
                        // so we don't do the big match statement as with the
                        // cross-correlations. This means that unpacking the
                        // autos is slower. At least there are many fewer of
                        // them!
                        uvfits_vis
                            .chunks_exact(
                                self.metadata.num_pols
                                    * usize::from(self.metadata.num_floats_per_pol),
                            )
                            .enumerate()
                            .filter(|(i_chan, _)| !flagged_fine_chans.contains(i_chan))
                            .zip(out_vis.iter_mut())
                            .zip(out_weights.iter_mut())
                            .for_each(|(((_, in_data), out_vis), out_weight)| {
                                let mut out_vis_tmp = [0.0; 8];
                                let mut out_weight_tmp = f32::MAX;

                                in_data
                                    .chunks_exact(usize::from(self.metadata.num_floats_per_pol))
                                    .zip(out_vis_tmp.chunks_exact_mut(2))
                                    .for_each(|(in_data, out_data)| {
                                        out_data[0] = in_data[0];
                                        out_data[1] = in_data[1];
                                        if in_data.len() == 3 {
                                            if in_data[2] < out_weight_tmp {
                                                out_weight_tmp = in_data[2];
                                            }
                                        } else {
                                            out_weight_tmp = 1.0;
                                        };
                                    });

                                *out_vis = Jones::from([
                                    out_vis_tmp[0], // XX real
                                    out_vis_tmp[1], // XX imag
                                    out_vis_tmp[4], // XY real
                                    out_vis_tmp[5], // XY imag
                                    out_vis_tmp[6], // YX real
                                    out_vis_tmp[7], // YX imag
                                    out_vis_tmp[2], // YY real
                                    out_vis_tmp[3], // YY imag
                                ]);
                                *out_weight = out_weight_tmp;
                            });
                    }
                }
            }
        }

        Ok(())
    }
}

impl VisRead for UvfitsReader {
    fn get_obs_context(&self) -> &ObsContext {
        &self.obs_context
    }

    fn get_input_data_type(&self) -> VisInputType {
        VisInputType::Uvfits
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

struct UvfitsMetadata {
    /// The number of rows in the metafits file (hopefully equal to the number
    /// of timesteps * the number of baselines).
    num_rows: usize,

    /// The number of parameters are in each uvfits group (PCOUNT).
    pcount: usize,

    /// The number of polarisations (probably 4, but allowed to be no less than
    /// 1 and no greater than 4).
    num_pols: usize,

    /// The "type" of polarisation present. Values of 1 through 4 are assigned
    /// to Stokes I, Q, U, and V, values of -1 through -4 are assigned to RR,
    /// LL, RL, and LR polarization products, respectively. Values -5 through -8
    /// are assigned to XX, YY, XY, and YX polarization products, respectively.
    _pol_type: i8,

    /// The number of floats associated with a polarisation. If this value is 3,
    /// these are the real part of the pol, imag part of the pol, and the
    /// weight, respectively. If this value is 2, then it's the same as 3,
    /// except the weight is always 1.0.
    num_floats_per_pol: u8,

    /// The... number of fine channel frequencies.
    num_fine_freq_chans: usize,

    /// The Julian date at midnight of the first day of the observation, as per
    /// the uvfits standard.
    jd_zero: f64,

    /// The indices of various parameters (e.g. BASELINE is PTYPE4, DATE is
    /// PTYPE5, etc.)
    indices: Indices,

    /// Unique collection of baselines (uvfits formatted, i.e. need to be
    /// decoded).
    uvfits_baselines: Vec<usize>,

    /// Unique collection of antennas (uvfits formatted, i.e. need to be
    /// decoded).
    uvfits_antennas_set: HashSet<usize>,

    /// Unique collection of JD fractions for timestamps.
    jd_frac_timestamps: Vec<f64>,

    /// Are auto-correlations present?
    autocorrelations_present: bool,
}

impl UvfitsMetadata {
    /// Get metadata on the supplied uvfits file.
    ///
    /// This function assumes the correct HDU has already been opened (should be
    /// HDU 1, index 0).
    fn new(uvfits: &mut FitsFile, hdu: &FitsHdu) -> Result<Self, UvfitsReadError> {
        let indices = Indices::new(uvfits, hdu)?;

        // GCOUNT tells us how many visibilities are in the file.
        let num_rows_str: String = fits_get_required_key(uvfits, hdu, "GCOUNT")?;
        let num_rows: usize = match num_rows_str.parse() {
            Ok(p) => p,
            Err(e) => {
                return Err(UvfitsReadError::Parse {
                    key: Cow::from("GCOUNT"),
                    value: num_rows_str,
                    parse_error: e.to_string(),
                })
            }
        };

        // PCOUNT tells us how many parameters are in each uvfits group.
        let pcount_str: String = fits_get_required_key::<String>(uvfits, hdu, "PCOUNT")?;
        let pcount = pcount_str
            .parse::<usize>()
            .map_err(|e| UvfitsReadError::Parse {
                key: Cow::from("PCOUNT"),
                value: pcount_str,
                parse_error: e.to_string(),
            })?;

        // We expect the COMPLEX index to be 2 (mandated by the standard), the
        // STOKES index to be 3, and the FREQ index to be 4. The order of these
        // indices determines the shape of the array of visibilities, and we
        // currently only support this one particular order.
        if indices.complex != 2 && indices.stokes != 3 && indices.freq != 4 {
            return Err(UvfitsReadError::WrongDataOrder {
                complex: indices.complex,
                stokes: indices.stokes,
                freq: indices.freq,
            });
        }

        // NAXIS2 (COMPLEX) is how many floats are associated with a
        // polarisation. It must be either 2 or 3, as per the standard. The
        // first two floats represent the real and imag part of a complex
        // number, respectively, and the optional third is the weight. If there
        // are only 2 floats, the weight is set to 1.
        let num_floats_per_pol_str: String = fits_get_required_key(uvfits, hdu, "NAXIS2")?;
        let num_floats_per_pol =
            num_floats_per_pol_str
                .parse::<u8>()
                .map_err(|e| UvfitsReadError::Parse {
                    key: Cow::from("NAXIS2"),
                    value: num_floats_per_pol_str,
                    parse_error: e.to_string(),
                })?;
        match num_floats_per_pol {
            2 | 3 => (),
            _ => return Err(UvfitsReadError::WrongFloatsPerPolCount(num_floats_per_pol)),
        }

        // The number of polarisations is described by the NAXIS key associated
        // with STOKES.
        let stokes_naxis_str = format!("NAXIS{}", indices.stokes);
        let num_pols_str: String = fits_get_required_key(uvfits, hdu, &stokes_naxis_str)?;
        let num_pols = num_pols_str
            .parse::<usize>()
            .map_err(|e| UvfitsReadError::Parse {
                key: Cow::from(stokes_naxis_str),
                value: num_pols_str,
                parse_error: e.to_string(),
            })?;

        // The pol type is described by the CRVAL key associated with STOKES.
        let stokes_crval_str = format!("CRVAL{}", indices.stokes);
        let pol_type_str: String = fits_get_required_key(uvfits, hdu, &stokes_crval_str)?;
        let pol_type: i8 = match pol_type_str.parse::<f32>() {
            Ok(pol_type) => {
                // Convert the float to an int.
                if pol_type.abs() > 127.0 {
                    panic!(
                        "STOKES {stokes_crval_str} has an unsupported value (absolute value > 127)"
                    );
                }
                let pol_type = pol_type.round() as i8;
                // We currently only support a "pol type" of -5, i.e. XX.
                if pol_type != -5 {
                    return Err(UvfitsReadError::UnsupportedPolType {
                        key: Cow::from(stokes_crval_str),
                        value: pol_type,
                    });
                }
                pol_type
            }
            Err(e) => {
                return Err(UvfitsReadError::Parse {
                    key: Cow::from(stokes_crval_str),
                    value: pol_type_str,
                    parse_error: e.to_string(),
                })
            }
        };

        // The number of fine-frequency channels is described by the NAXIS key
        // associated with FREQ.
        let freq_naxis_str = format!("NAXIS{}", indices.freq);
        let num_fine_freq_chans_str: String = fits_get_required_key(uvfits, hdu, &freq_naxis_str)?;
        let num_fine_freq_chans =
            num_fine_freq_chans_str
                .parse::<usize>()
                .map_err(|e| UvfitsReadError::Parse {
                    key: Cow::from(freq_naxis_str),
                    value: num_fine_freq_chans_str,
                    parse_error: e.to_string(),
                })?;

        // "JD zero" refers to the Julian date at midnight of the first day of
        // the observation, as per the uvfits standard.
        let jd_zero_val_str = format!("PZERO{}", indices.date1);
        let jd_zero_str: String = fits_get_required_key(uvfits, hdu, &jd_zero_val_str)?;
        // We expect that the PZERO corresponding to the second date (if
        // available) is 0.
        if let Some(d2) = indices.date2 {
            let pzero = format!("PZERO{d2}");
            let key: Option<String> = fits_get_optional_key(uvfits, hdu, &pzero)?;
            match key {
                Some(key) => match key.parse::<f32>() {
                    Ok(n) => {
                        if n.abs() > f32::EPSILON {
                            warn!("{pzero}, corresponding to the second DATE, was not 0; ignoring it anyway")
                        }
                    }
                    Err(std::num::ParseFloatError { .. }) => {
                        warn!("Could not parse {pzero} as a float")
                    }
                },
                None => warn!("{pzero} does not exist, corresponding to the second DATE"),
            }
        }
        let jd_zero = jd_zero_str
            .parse::<f64>()
            .map_err(|e| UvfitsReadError::Parse {
                key: Cow::from(jd_zero_val_str),
                value: jd_zero_str,
                parse_error: e.to_string(),
            })?;

        // Read unique group parameters (timestamps and baselines/antennas).
        let mut uvfits_baselines_set = HashSet::new();
        let mut uvfits_antennas_set = HashSet::new();
        let mut uvfits_baselines = vec![];
        let mut autocorrelations_present = false;
        let mut jd_frac_timestamp_set = HashSet::new();
        let mut jd_frac_timestamps = vec![];

        let mut group_params = Array2::zeros((num_rows, pcount));
        unsafe {
            let mut status = 0;
            // ffggpe = fits_read_grppar_flt
            fitsio_sys::ffggpe(
                uvfits.as_raw(), /* I - FITS file pointer                       */
                1,               /* I - group to read (1 = 1st group)           */
                1,               /* I - first vector element to read (1 = 1st)  */
                (pcount * num_rows)
                    .try_into()
                    .expect("not larger than i64::MAX"), /* I - number of values to read                */
                group_params.as_mut_ptr(), /* O - array of values that are returned       */
                &mut status,               /* IO - error status                           */
            );
            // Check the status.
            fits_check_status(status).map_err(UvfitsReadError::Metadata)?;
        }

        for params in group_params.outer_iter() {
            let (ant1, ant2) = match indices.baseline_or_antennas {
                BaselineOrAntennas::Baseline { index } => {
                    let uvfits_bl = params[usize::from(index) - 1] as usize;

                    // Don't just push into a set; we want the order of the baselines as
                    // they come out of the uvfits file, and this isn't necessarily
                    // sorted.
                    if !uvfits_baselines_set.contains(&uvfits_bl) {
                        uvfits_baselines_set.insert(uvfits_bl);
                        uvfits_baselines.push(uvfits_bl);
                    }

                    decode_uvfits_baseline(uvfits_bl)
                }
                BaselineOrAntennas::Antennas { index1, index2 } => {
                    let uvfits_ant1 = params[usize::from(index1) - 1] as usize;
                    let uvfits_ant2 = params[usize::from(index2) - 1] as usize;
                    uvfits_antennas_set.insert(uvfits_ant1);
                    uvfits_antennas_set.insert(uvfits_ant2);
                    (uvfits_ant1, uvfits_ant2)
                }
            };

            if !autocorrelations_present && (ant1 == ant2) {
                autocorrelations_present = true;
            }

            let jd = {
                let mut t = params[usize::from(indices.date1) - 1] as f64;
                // Use the second date, if it's there.
                if let Some(d2) = indices.date2 {
                    t += params[usize::from(d2) - 1] as f64;
                }
                t
            };
            // Floats can't be hashed. Hash the bits!
            let jd_bits = jd.to_bits();
            if !jd_frac_timestamp_set.contains(&jd_bits) {
                jd_frac_timestamp_set.insert(jd_bits);
                jd_frac_timestamps.push(jd);
            }
        }

        Ok(UvfitsMetadata {
            num_rows,
            pcount,
            num_pols,
            _pol_type: pol_type,
            num_floats_per_pol,
            num_fine_freq_chans,
            jd_zero,
            indices,
            uvfits_baselines,
            uvfits_antennas_set,
            jd_frac_timestamps,
            autocorrelations_present,
        })
    }
}

#[derive(Debug)]
enum BaselineOrAntennas {
    Baseline { index: u8 },

    Antennas { index1: u8, index2: u8 },
}

#[derive(Debug)]
struct Indices {
    /// PTYPE
    u: u8,
    /// PTYPE
    v: u8,
    /// PTYPE
    w: u8,
    /// PTYPE
    baseline_or_antennas: BaselineOrAntennas,
    /// PTYPE
    date1: u8,
    /// PTYPE
    date2: Option<u8>,
    /// CTYPE
    complex: u8,
    /// CTYPE
    stokes: u8,
    /// CTYPE
    freq: u8,
    /// CTYPE
    ra: u8,
    /// CTYPE
    dec: u8,
}

impl Indices {
    /// Find the 1-indexed indices of "PTYPE" and "CTYPE" keys we require (e.g.
    /// "UU", "VV", "WW", "RA", "DEC"). "BASELINE" will be in most uvfits files,
    /// but "ANTENNA1" and "ANTENNA2" may be used instead; exactly one of the
    /// two is ensured to be present. A second "DATE"/"_DATE" key may also be
    /// present but does not have to be.
    fn new(uvfits: &mut FitsFile, hdu: &FitsHdu) -> Result<Self, UvfitsReadError> {
        // Accumulate the "PTYPE" keys.
        let mut ptypes = Vec::with_capacity(12);
        for i in 1.. {
            let ptype: Option<String> = fits_get_optional_key(uvfits, hdu, &format!("PTYPE{i}"))?;
            match ptype {
                Some(ptype) => ptypes.push(ptype),

                // We've found the last PTYPE.
                None => break,
            }
        }

        // We only care about UVWs, baselines and dates.
        let mut u_index = None;
        let mut v_index = None;
        let mut w_index = None;
        let mut baseline_index = None;
        let mut antenna1_index = None;
        let mut antenna2_index = None;
        let mut date1_index = None;
        let mut date2_index = None;

        for (i, key) in ptypes.into_iter().enumerate() {
            let ii = (i + 1) as u8;
            match key.as_str() {
                "UU" => {
                    if u_index.is_none() {
                        u_index = Some(ii)
                    } else {
                        warn!("Found another UU key -- only using the first");
                    }
                }
                "VV" => {
                    if v_index.is_none() {
                        v_index = Some(ii)
                    } else {
                        warn!("Found another VV key -- only using the first");
                    }
                }
                "WW" => {
                    if w_index.is_none() {
                        w_index = Some(ii)
                    } else {
                        warn!("Found another WW key -- only using the first");
                    }
                }
                "BASELINE" => {
                    if baseline_index.is_none() {
                        baseline_index = Some(ii)
                    } else {
                        warn!("Found another BASELINE key -- only using the first");
                    }
                }
                "ANTENNA1" => {
                    if antenna1_index.is_none() {
                        antenna1_index = Some(ii)
                    } else {
                        warn!("Found another ANTENNA1 key -- only using the first");
                    }
                }
                "ANTENNA2" => {
                    if antenna2_index.is_none() {
                        antenna2_index = Some(ii)
                    } else {
                        warn!("Found another ANTENNA1 key -- only using the first");
                    }
                }
                "DATE" | "_DATE" => match (date1_index, date2_index) {
                    (None, None) => date1_index = Some(ii),
                    (Some(_), None) => date2_index = Some(ii),
                    (Some(_), Some(_)) => {
                        warn!("Found more than 2 DATE/_DATE keys -- only using the first two")
                    }
                    (None, Some(_)) => unreachable!(),
                },
                _ => (),
            }
        }

        // Handle problems surrounding some combination of BASELINE and
        // ANTENNA1/ANTENNA2.
        let baseline_or_antennas = match (baseline_index, antenna1_index, antenna2_index) {
            // These are OK.
            (Some(index), None, None) => BaselineOrAntennas::Baseline { index },
            (None, Some(index1), Some(index2)) => BaselineOrAntennas::Antennas { index1, index2 },
            // These are not.
            (Some(_), Some(_), _) | (Some(_), _, Some(_)) => {
                return Err(UvfitsReadError::BaselineAndAntennas)
            }
            (None, Some(_), None) => return Err(UvfitsReadError::Antenna1ButNotAntenna2),
            (None, None, Some(_)) => return Err(UvfitsReadError::Antenna2ButNotAntenna1),
            (None, None, None) => return Err(UvfitsReadError::NoBaselineInfo),
        };

        let (u, v, w, date1) = match (u_index, v_index, w_index, date1_index) {
            (Some(u), Some(v), Some(w), Some(date1)) => (u, v, w, date1),
            (None, _, _, _) => {
                return Err(UvfitsReadError::MissingKey {
                    key: "UU",
                    hdu: hdu.number + 1,
                })
            }
            (_, None, _, _) => {
                return Err(UvfitsReadError::MissingKey {
                    key: "VV",
                    hdu: hdu.number + 1,
                })
            }
            (_, _, None, _) => {
                return Err(UvfitsReadError::MissingKey {
                    key: "WW",
                    hdu: hdu.number + 1,
                })
            }
            (_, _, _, None) => {
                return Err(UvfitsReadError::MissingKey {
                    key: "DATE",
                    hdu: hdu.number + 1,
                })
            }
        };

        // Now find CTYPEs.
        let mut ctypes = Vec::with_capacity(12);
        for i in 2.. {
            let ctype: Option<String> = fits_get_optional_key(uvfits, hdu, &format!("CTYPE{i}"))?;
            match ctype {
                Some(ctype) => ctypes.push(ctype),

                // We've found the last CTYPE.
                None => break,
            }
        }

        let mut complex_index = None;
        let mut stokes_index = None;
        let mut freq_index = None;
        let mut ra_index = None;
        let mut dec_index = None;

        for (i, key) in ctypes.into_iter().enumerate() {
            let ii = (i + 2) as u8;
            match key.as_str() {
                "COMPLEX" => complex_index = Some(ii),
                "STOKES" => stokes_index = Some(ii),
                "FREQ" => freq_index = Some(ii),
                "RA" => ra_index = Some(ii),
                "DEC" => dec_index = Some(ii),
                _ => (),
            }
        }

        let (complex, stokes, freq, ra, dec) =
            match (complex_index, stokes_index, freq_index, ra_index, dec_index) {
                (Some(complex), Some(stokes), Some(freq), Some(ra), Some(dec)) => {
                    (complex, stokes, freq, ra, dec)
                }
                (None, _, _, _, _) => {
                    return Err(UvfitsReadError::MissingKey {
                        key: "COMPLEX",
                        hdu: hdu.number + 1,
                    })
                }
                (_, None, _, _, _) => {
                    return Err(UvfitsReadError::MissingKey {
                        key: "STOKES",
                        hdu: hdu.number + 1,
                    })
                }
                (_, _, None, _, _) => {
                    return Err(UvfitsReadError::MissingKey {
                        key: "FREQ",
                        hdu: hdu.number + 1,
                    })
                }
                (_, _, _, None, _) => {
                    return Err(UvfitsReadError::MissingKey {
                        key: "RA",
                        hdu: hdu.number + 1,
                    })
                }
                (_, _, _, _, None) => {
                    return Err(UvfitsReadError::MissingKey {
                        key: "DEC",
                        hdu: hdu.number + 1,
                    })
                }
            };

        Ok(Indices {
            u,
            v,
            w,
            baseline_or_antennas,
            date1,
            date2: date2_index,
            complex,
            stokes,
            freq,
            ra,
            dec,
        })
    }
}

/// Pull out fits array-in-a-cell values; useful for e.g. STABXYZ. This function
/// assumes that the output datatype is f64, and that the fits datatype is
/// TDOUBLE, so it is not to be used generally!
fn read_cell_array(
    fits_ptr: &mut fitsio::FitsFile,
    hdu: &fitsio::hdu::FitsHdu,
    col_name: &'static str,
    row: i64,
    n_elem: i64,
) -> Result<Vec<f64>, UvfitsReadError> {
    unsafe {
        // With the column name, get the column number.
        let mut status = 0;
        let mut col_num = -1;
        let keyword = std::ffi::CString::new(col_name).expect("CString::new failed");
        // ffgcno = fits_get_colnum
        fitsio_sys::ffgcno(
            fits_ptr.as_raw(),
            0,
            keyword.as_ptr() as *mut c_char,
            &mut col_num,
            &mut status,
        );
        // Check the status.
        fits_check_status(status).map_err(|err| UvfitsReadError::ReadCellArray {
            col_name,
            hdu_num: hdu.number + 1,
            err,
        })?;

        // Now get the specified row from that column.
        let mut array: Vec<f64> = vec![0.0; n_elem as usize];
        // ffgcv = fits_read_col
        fitsio_sys::ffgcv(
            fits_ptr.as_raw(),
            82, // TDOUBLE (fitsio.h)
            col_num,
            row + 1,
            1,
            n_elem,
            std::ptr::null_mut(),
            array.as_mut_ptr().cast(),
            &mut 0,
            &mut status,
        );
        fits_check_status(status).map_err(|err| UvfitsReadError::ReadCellArray {
            col_name,
            hdu_num: hdu.number + 1,
            err,
        })?;

        Ok(array)
    }
}

enum UvfitsFlavour {
    Hyperdrive,

    /// Birli before version 0.2.0 and after 0.7.0
    Birli,

    /// Anything that writes ms with the Marlu library without specifying the
    /// appropriate history
    Marlu,

    Cotter,

    /// Generic?
    Other,
}

fn get_uvfits_flavour(software: &str) -> UvfitsFlavour {
    let app = software.to_uppercase();
    if app.starts_with("MWA_HYPERDRIVE") {
        return UvfitsFlavour::Hyperdrive;
    } else if app.starts_with("BIRLI") {
        return UvfitsFlavour::Birli;
    } else if app.starts_with("MARLU") {
        return UvfitsFlavour::Marlu;
    } else if app.starts_with("COTTER") {
        return UvfitsFlavour::Cotter;
    };

    // If we still don't know what the app is, fallback on "Other".
    UvfitsFlavour::Other
}

/// uvfits files should always be writing out [`XyzGeocentric`] coordinates for
/// their antenna positions (because FRAME is ITRF). However, MWA files
/// historically use [`XyzGeodetic`], perhaps because it's more useful. This
/// function attempts to detect whether the coordinates passed in are indeed
/// geodetic or geocentric (`true` if we think they're geocentric).
fn detect_geocentric_antenna_positions(xyzs: &[XyzGeocentric]) -> bool {
    // The heights for geocentric coordinates are (usually?) order 10^6 (~Earth
    // radius).
    let average = xyzs
        .iter()
        .copied()
        .map(|XyzGeocentric { x: _, y: _, z }| z.abs())
        .sum::<f64>()
        / xyzs.len() as f64;
    average > 1e5
}
