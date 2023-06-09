// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to handle reading from uvfits files.
//!
//! The uvfits standard can be found here:
//! <https://library.nrao.edu/public/memos/aips/memos/AIPSM_117.pdf>

mod error;
#[cfg(test)]
mod tests;

pub(crate) use error::*;

use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    num::NonZeroU16,
    path::{Path, PathBuf},
};

use fitsio::{errors::check_status as fits_check_status, hdu::FitsHdu, FitsFile};
use hifitime::{Duration, Epoch, TimeUnits};
use log::{debug, trace, warn};
use marlu::{
    constants::VEL_C, io::uvfits::decode_uvfits_baseline, Jones, LatLngHeight, RADec,
    XyzGeocentric, XyzGeodetic, UVW,
};
use mwalib::MetafitsContext;
use ndarray::prelude::*;
use num_complex::Complex;

use super::*;
use crate::{
    beam::Delays,
    cli::Warn,
    context::{ObsContext, Polarisations},
    io::read::{
        fits::{
            fits_get_col, fits_get_optional_key, fits_get_required_key, fits_open, fits_open_hdu,
            fits_read_cell_f64_array,
        },
        VisRead, VisReadError,
    },
    metafits::{get_dipole_delays, get_dipole_gains, map_antenna_order},
};

pub struct UvfitsReader {
    /// Observation metadata.
    obs_context: ObsContext,

    // uvfits-specific things follow.
    /// The path to the uvfits on disk.
    uvfits: PathBuf,

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

    /// If the incoming data uses ant2-ant1 UVWs instead of ant1-ant2 UVWs, we
    /// need to conjugate the visibilities to match what will be modelled.
    conjugate_vis: bool,
}

impl UvfitsReader {
    /// Verify and populate metadata associated with this uvfits file.
    ///
    /// The uvfits file is expected to generally be compliant with AIPS 117
    /// ("the uvfits standard"), but the code here is a little less rigorous
    /// than the standard mandates.
    pub fn new(
        uvfits: PathBuf,
        metafits: Option<&Path>,
        array_position: Option<LatLngHeight>,
    ) -> Result<UvfitsReader, UvfitsReadError> {
        // If a metafits file was provided, get an mwalib object ready.
        // TODO: Let the user supply the MWA version.
        let mwalib_context = match metafits {
            None => None,
            Some(m) => Some(MetafitsContext::new(m, None).map_err(Box::new)?),
        };

        debug!("Using uvfits file: {}", uvfits.display());
        if !uvfits.exists() {
            return Err(UvfitsReadError::BadFile(uvfits));
        }

        // Get the tile names, XYZ positions and antenna numbers.
        let mut uvfits_fptr = fits_open(&uvfits)?;
        let primary_hdu = fits_open_hdu(&mut uvfits_fptr, 0)?;
        let antenna_table_hdu = fits_open_hdu(&mut uvfits_fptr, "AIPS AN")?;

        let tile_names: Vec<String> = fits_get_col(&mut uvfits_fptr, &antenna_table_hdu, "ANNAME")?;
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

        // ARRAY{X,Y,Z} describes the array position.
        let array_x: f64 = fits_get_required_key(&mut uvfits_fptr, &antenna_table_hdu, "ARRAYX")?;
        let array_y: f64 = fits_get_required_key(&mut uvfits_fptr, &antenna_table_hdu, "ARRAYY")?;
        let array_z: f64 = fits_get_required_key(&mut uvfits_fptr, &antenna_table_hdu, "ARRAYZ")?;
        let mut supplied_array_position = XyzGeocentric {
            x: array_x,
            y: array_y,
            z: array_z,
        }
        .to_earth_wgs84();
        // It seems that CASA/casacore products incorrectly set these to 0, so
        // if we're in this situation, we have to alter our logic below.
        let wrong_array_xyz = array_x.abs() < f64::EPSILON
            && array_y.abs() < f64::EPSILON
            && array_z.abs() < f64::EPSILON;
        // Get the tile positions from the uvfits file. Update the supplied
        // array position if ARRAY{X,Y,Z} are wrong.
        let tile_xyzs = {
            // The uvfits standard only defines one frame (ITRF). So warn the
            // user if this isn't explicit, but we assume this is always used.
            // FRAME is supposed to be mandatory, but we'll be kind here.
            let frame: Option<String> =
                fits_get_optional_key(&mut uvfits_fptr, &antenna_table_hdu, "FRAME")?;
            if !matches!(frame.as_deref(), Some("ITRF")) {
                "Assuming that the uvfits antenna coordinate system is ITRF".warn();
            }

            // Because ARRAY{X,Y,Z} are defined to be the array position, the
            // STABXYZ positions are relative to it, and the frame is always
            // ITRF, the STABXYZ positions are geodetic.
            let mut tile_xyzs: Vec<XyzGeodetic> = Vec::with_capacity(total_num_tiles);
            let mut average_xyz = XyzGeocentric::default();
            for i in 0..total_num_tiles {
                let fits_xyz = fits_read_cell_f64_array::<3>(
                    &mut uvfits_fptr,
                    &antenna_table_hdu,
                    "STABXYZ",
                    i.try_into().expect("not larger than i64::MAX"),
                )?;
                if wrong_array_xyz {
                    tile_xyzs.push(XyzGeodetic {
                        x: fits_xyz[0],
                        y: fits_xyz[1],
                        z: fits_xyz[2],
                    });
                    average_xyz.x += fits_xyz[0];
                    average_xyz.y += fits_xyz[1];
                    average_xyz.z += fits_xyz[2];
                } else {
                    tile_xyzs.push(XyzGeodetic {
                        x: fits_xyz[0],
                        y: fits_xyz[1],
                        z: fits_xyz[2],
                    });
                }
            }

            if wrong_array_xyz {
                "It seems this uvfits file's antenna positions has been blessed by casacore. Unblessing.".warn();
                // Get the supplied array position from the average tile
                // position.
                average_xyz.x /= tile_xyzs.len() as f64;
                average_xyz.y /= tile_xyzs.len() as f64;
                average_xyz.z /= tile_xyzs.len() as f64;
                supplied_array_position = average_xyz.to_earth_wgs84();
                // If the user supplied an array position, use that instead of
                // the data's.
                let array_position = array_position.unwrap_or(supplied_array_position);

                // Convert the geocentric positions to geodetic.
                let vec = XyzGeocentric::get_geocentric_vector(array_position);
                let (s_long, c_long) = array_position.longitude_rad.sin_cos();
                tile_xyzs.iter_mut().for_each(|geocentric| {
                    // `geocentric` is typed as `XyzGeodetic`; just convert the
                    // type so we can use the `to_geodetic_inner` method.
                    let gc = XyzGeocentric {
                        x: geocentric.x,
                        y: geocentric.y,
                        z: geocentric.z,
                    };
                    *geocentric = gc.to_geodetic_inner(vec, s_long, c_long);
                });
            }

            tile_xyzs
        };
        // If the user supplied an array position, use that instead of the
        // data's.
        let array_position = array_position.unwrap_or(supplied_array_position);
        let tile_xyzs = Vec1::try_from_vec(tile_xyzs)
            .expect("can't be empty, non-empty tile names verified above");

        let metadata = crate::misc::expensive_op(
            || {
                let mut uvfits_fptr = fits_open(&uvfits)?;
                let hdu = fits_open_hdu(&mut uvfits_fptr, 0)?;
                UvfitsMetadata::new(&mut uvfits_fptr, &hdu)
            },
            "Still waiting to inspect all uvfits metadata",
        )?;
        // Make a nice little string for user display. uvfits always puts YY
        // before cross pols so we have to use some logic here.
        let pol_str = match metadata.pols {
            Polarisations::XX_XY_YX_YY => "4 [XX YY XY YX]",
            Polarisations::XX => "1 [XX]",
            Polarisations::YY => "1 [YY]",
            Polarisations::XX_YY => "2 [XX YY]",
            Polarisations::XX_YY_XY => "3 [XX YY XY]",
        };

        debug!("Number of rows in the uvfits:   {}", metadata.num_rows);
        debug!("PCOUNT:                         {}", metadata.pcount);
        debug!("Number of polarisations:        {pol_str}");
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
        if let Some(inttim) = metadata.indices.inttim {
            debug!("INTTIM index:        {}", inttim);
        }
        debug!("COMPLEX index:  {}", metadata.indices.complex);
        debug!("STOKES index:   {}", metadata.indices.stokes);
        debug!("FREQ index:     {}", metadata.indices.freq);
        debug!("RA index:       {}", metadata.indices.ra);
        debug!("DEC index:      {}", metadata.indices.dec);

        if metadata.num_rows == 0 {
            return Err(UvfitsReadError::Empty(uvfits));
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
                [
                    "The uvfits antenna names are different to those supplied in the metafits."
                        .into(),
                    "Dipole delays/gains may be incorrectly mapped to uvfits antennas.".into(),
                ]
                .warn();
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
        present_tiles_set.extend(metadata.uvfits_antennas.iter());
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

        // Work out the timestamp epochs.
        let (all_timesteps, timestamps): (Vec<usize>, Vec<Epoch>) = metadata
            .jd_frac_timestamps
            .iter()
            .enumerate()
            .map(|(i, &jd_frac)| {
                // uvfits timestamps are in the middle of their respective
                // integration periods (centroids), so no adjustment for
                // half the integration time is needed here.
                let e = metadata.jd_zero + jd_frac;
                // Round to the nearest 10 milliseconds to avoid float
                // precision issues.
                (i, e.round(10.milliseconds()))
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
        let time_res = match metadata.time_res {
            Some(r) => Some(Duration::from_seconds(r)),
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
        };
        // Verify that all timestamps are spaced by a multiple of the time
        // resolution.
        if let Some(time_res) = time_res {
            for (i_pair, window) in timestamps.windows(2).enumerate() {
                let diff = window[1] - window[0];
                if diff.total_nanoseconds() % time_res.total_nanoseconds() != 0 {
                    return Err(UvfitsReadError::IrregularTimestamps {
                        what_we_think_is_the_time_res: time_res,
                        gap_found: diff,
                        pair: i_pair,
                    });
                }
            }
        }
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
        debug!("Number of baselines per timestep: {step}");

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

        let mut fine_chan_freqs_f64 = Vec::with_capacity(metadata.num_fine_freq_chans);
        let mut fine_chan_freqs = Vec::with_capacity(metadata.num_fine_freq_chans);
        for i in 0..metadata.num_fine_freq_chans {
            let freq = (base_freq + (i as isize - base_index + 1) as f64 * freq_res).round();
            fine_chan_freqs_f64.push(freq);
            fine_chan_freqs.push(freq.round() as u64);
        }
        let fine_chan_freqs_f64 = Vec1::try_from_vec(fine_chan_freqs_f64).unwrap();
        let fine_chan_freqs = Vec1::try_from_vec(fine_chan_freqs).unwrap();

        let mwa_coarse_chan_nums = match mwalib_context.as_ref() {
            Some(c) => {
                // Get the coarse channel information out of the metafits
                // file, but only the ones aligned with the frequencies in
                // the uvfits file.
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
                debug!("Found corresponding MWA coarse channel numbers from the metafits and uvfits frequencies");
                Vec1::try_from_vec(cc_nums).ok()
            }

            None => {
                debug!("Assuming MWA coarse channel numbers from uvfits frequencies");

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

        // Compare the first cross-correlation row's UVWs against UVWs that we
        // would make with the existing tiles. If they're negative of one
        // another, we need to negate our XYZs to match the UVWs the data use.
        let conjugate_vis = {
            let mut first_cross_bl_and_uvw = None;
            let mut group_params = vec![0.0; metadata.pcount];
            // Ensure we're on the data-containing HDU.
            let _ = uvfits_fptr.primary_hdu()?;
            for i_row in 0..metadata.num_rows {
                unsafe {
                    let mut status = 0;
                    // ffggpe = fits_read_grppar_flt
                    fitsio_sys::ffggpe(
                        uvfits_fptr.as_raw(), /* I - FITS file pointer                       */
                        (i_row + 1).try_into().expect("not larger than i64::MAX"), /* I - group to read (1 = 1st group)           */
                        1, /* I - first vector element to read (1 = 1st)  */
                        group_params
                            .len()
                            .try_into()
                            .expect("not larger than i64::MAX"), /* I - number of values to read                */
                        group_params.as_mut_ptr(), /* O - array of values that are returned       */
                        &mut status,               /* IO - error status                           */
                    );
                    fits_check_status(status).map_err(|err| UvfitsReadError::ReadVis {
                        row_num: i_row + 1,
                        err,
                    })?;
                }
                let (uvfits_ant1, uvfits_ant2) = match metadata.indices.baseline_or_antennas {
                    BaselineOrAntennas::Baseline { index } => {
                        let uvfits_bl = group_params[usize::from(index - 1)];
                        decode_uvfits_baseline(uvfits_bl as usize)
                    }
                    BaselineOrAntennas::Antennas { index1, index2 } => (
                        group_params[usize::from(index1 - 1)] as usize,
                        group_params[usize::from(index2 - 1)] as usize,
                    ),
                };
                let (ant1, ant2) = (tile_map[&uvfits_ant1], tile_map[&uvfits_ant2]);
                if ant1 != ant2 {
                    let indices = &metadata.indices;
                    first_cross_bl_and_uvw = Some((
                        ant1,
                        ant2,
                        UVW {
                            u: f64::from(group_params[usize::from(indices.u - 1)]),
                            v: f64::from(group_params[usize::from(indices.v - 1)]),
                            w: f64::from(group_params[usize::from(indices.w - 1)]),
                        },
                    ));
                    break;
                }
            }
            // If this data somehow has no cross-correlation data, using
            // default values won't affect anything.
            let (ant1, ant2, data_uvw) = first_cross_bl_and_uvw.unwrap_or_default();
            let tile1_xyz = tile_xyzs[ant1];
            let tile2_xyz = tile_xyzs[ant2];

            if baseline_convention_is_different(
                data_uvw * VEL_C,
                tile1_xyz,
                tile2_xyz,
                array_position,
                phase_centre,
                *timestamps.first(),
                dut1,
            ) {
                "uvfits UVWs use the other baseline convention; will conjugate incoming visibilities".warn();
                true
            } else {
                false
            }
        };

        let obs_context = ObsContext {
            input_data_type: VisInputType::Uvfits,
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
            autocorrelations_present: metadata.autocorrelations_present,
            dipole_delays,
            dipole_gains,
            time_res,
            mwa_coarse_chan_nums,
            num_fine_chans_per_coarse_chan,
            freq_res: Some(freq_res),
            fine_chan_freqs,
            // TODO: Get flagging right. I think that info is in an optional table.
            flagged_fine_chans: vec![],
            flagged_fine_chans_per_coarse_chan: None,
            polarisations: metadata.pols,
        };

        Ok(UvfitsReader {
            obs_context,
            uvfits,
            metadata,
            step,
            metafits_context: mwalib_context,
            tile_map,
            conjugate_vis,
        })
    }

    /// This function is intended to be private, but is public so that it may be
    /// benchmarked. The const parameters should be equal to whatever is in the
    /// `self.metadata`; using them means that many branch conditions can be
    /// avoided.
    pub fn read_inner<const NUM_POLS: usize, const NUM_FLOATS_PER_POL: usize>(
        &self,
        mut crosses: Option<CrossData>,
        mut autos: Option<AutoData>,
        timestep: usize,
        flagged_fine_chans: &HashSet<u16>,
    ) -> Result<(), VisReadError> {
        let row_range_start = timestep * self.step;
        let row_range_end = (timestep + 1) * self.step;

        let mut uvfits = fits_open(&self.uvfits).map_err(UvfitsReadError::from)?;
        fits_open_hdu(&mut uvfits, 0).map_err(UvfitsReadError::from)?;
        let mut group_params: Vec<f32> = vec![0.0; self.metadata.pcount];
        let mut uvfits_vis: Vec<f32> =
            vec![0.0; self.metadata.num_fine_freq_chans * NUM_POLS * NUM_FLOATS_PER_POL];
        let flags = (0..self.metadata.num_fine_freq_chans)
            .map(|i_chan| flagged_fine_chans.contains(&(i_chan as u16)))
            .collect::<Vec<_>>();
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

            // Check that the time associated with these group parameters is
            // correct.
            let this_timestamp = {
                let mut jd_frac = Duration::from_days(f64::from(
                    group_params[usize::from(self.metadata.indices.date1 - 1)],
                ));
                if let Some(date2) = self.metadata.indices.date2 {
                    jd_frac += Duration::from_days(f64::from(group_params[usize::from(date2 - 1)]));
                }
                jd_frac
            };
            // Verify that this timestamp is what we expect it to be.
            if this_timestamp != self.metadata.jd_frac_timestamps[timestep] {
                return Err(UvfitsReadError::MismatchedTimestamps {
                    timestep,
                    expected_timestamp: self.metadata.jd_zero
                        + self.metadata.jd_frac_timestamps[timestep],
                    got: self.metadata.jd_zero + this_timestamp,
                    uvfits_row: row,
                }
                .into());
            }

            let (uvfits_ant1, uvfits_ant2) = match self.metadata.indices.baseline_or_antennas {
                BaselineOrAntennas::Baseline { index } => {
                    let uvfits_bl = group_params[usize::from(index - 1)];
                    assert!(self
                        .metadata
                        .uvfits_baselines
                        .contains(&(uvfits_bl as usize)));
                    decode_uvfits_baseline(uvfits_bl as usize)
                }
                BaselineOrAntennas::Antennas { index1, index2 } => {
                    let a1 = group_params[usize::from(index1 - 1)] as usize;
                    let a2 = group_params[usize::from(index2 - 1)] as usize;
                    assert!(self.metadata.uvfits_antennas.contains(&a1));
                    assert!(self.metadata.uvfits_antennas.contains(&a2));
                    (a1, a2)
                }
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
                    let mut out_vis = crosses.vis_fb.slice_mut(s![.., i_baseline]);
                    let mut out_weights = crosses.weights_fb.slice_mut(s![.., i_baseline]);

                    let mut i_unflagged_chan = 0;
                    uvfits_vis
                        .chunks_exact(NUM_POLS * NUM_FLOATS_PER_POL)
                        .zip(flags.iter())
                        .filter(|(_, &flag)| !flag)
                        .for_each(|(in_data, _flag)| {
                            let (vis, weight) = match NUM_FLOATS_PER_POL {
                                3 => {
                                    let mut vis = Jones::default();
                                    let mut weight = -0.0;
                                    if NUM_POLS > 0 {
                                        // XX
                                        vis[0] = Complex::new(in_data[0], in_data[1]);
                                        weight = in_data[2];
                                    }
                                    if NUM_POLS > 1 {
                                        // YY
                                        vis[3] = Complex::new(in_data[3], in_data[4]);
                                        if in_data[5] < weight {
                                            weight = in_data[5];
                                        };
                                    }
                                    if NUM_POLS > 2 {
                                        // XY
                                        vis[1] = Complex::new(in_data[6], in_data[7]);
                                        if in_data[8] < weight {
                                            weight = in_data[8];
                                        };
                                    }
                                    if NUM_POLS > 3 {
                                        // YX
                                        vis[2] = Complex::new(in_data[9], in_data[10]);
                                        if in_data[11] < weight {
                                            weight = in_data[11];
                                        };
                                    }
                                    (vis, weight)
                                }

                                2 => {
                                    let mut vis = Jones::default();
                                    if NUM_POLS > 0 {
                                        // XX
                                        vis[0] = Complex::new(in_data[0], in_data[1]);
                                    }
                                    if NUM_POLS > 1 {
                                        // YY
                                        vis[3] = Complex::new(in_data[2], in_data[3]);
                                    }
                                    if NUM_POLS > 2 {
                                        // XY
                                        vis[1] = Complex::new(in_data[4], in_data[5]);
                                    }
                                    if NUM_POLS > 3 {
                                        // YX
                                        vis[2] = Complex::new(in_data[6], in_data[7]);
                                    }
                                    (vis, 1.0)
                                }

                                _ => unreachable!("NUM_FLOATS_PER_POL must be 2 or 3"),
                            };

                            *out_vis.get_mut(i_unflagged_chan).expect("is in range") = vis;
                            *out_weights.get_mut(i_unflagged_chan).expect("is in range") = weight;
                            i_unflagged_chan += 1;
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

                        let mut out_vis = autos.vis_fb.slice_mut(s![.., i_ant]);
                        let mut out_weights = autos.weights_fb.slice_mut(s![.., i_ant]);
                        // Auto-correlations are a lower priority in hyperdrive,
                        // so we don't do the big match statement as with the
                        // cross-correlations. This means that unpacking the
                        // autos is slower. At least there are many fewer of
                        // them!
                        uvfits_vis
                            .chunks_exact(NUM_POLS * NUM_FLOATS_PER_POL)
                            .enumerate()
                            .filter(|(i_chan, _)| !flagged_fine_chans.contains(&(*i_chan as u16)))
                            .zip(out_vis.iter_mut())
                            .zip(out_weights.iter_mut())
                            .for_each(|(((_, in_data), out_vis), out_weight)| {
                                let mut out_vis_tmp = [0.0; 8];
                                let mut out_weight_tmp = f32::MAX;

                                in_data
                                    .chunks_exact(NUM_FLOATS_PER_POL)
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

        // Transform the data, depending on what the actual polarisations are
        // and if we need to conjugate.
        if let Some(crosses) = crosses.as_mut() {
            let c0 = num_complex::Complex32::default();
            match (self.conjugate_vis, self.metadata.pols) {
                // These pols are all handled correctly.
                (false, Polarisations::XX_XY_YX_YY) => (),
                (false, Polarisations::XX) => (),
                (false, Polarisations::XX_YY) => (),
                (false, Polarisations::XX_YY_XY) => (),
                // Just conjugate.
                (
                    true,
                    Polarisations::XX_XY_YX_YY
                    | Polarisations::XX
                    | Polarisations::XX_YY
                    | Polarisations::XX_YY_XY,
                ) => crosses.vis_fb.mapv_inplace(|j| {
                    Jones::from([j[0].conj(), j[1].conj(), j[2].conj(), j[3].conj()])
                }),

                // Because we read in one polarisation, it was treated as XX,
                // but this is actually YY.
                (false, Polarisations::YY) => crosses
                    .vis_fb
                    .mapv_inplace(|j| Jones::from([c0, c0, c0, j[0]])),
                (true, Polarisations::YY) => crosses
                    .vis_fb
                    .mapv_inplace(|j| Jones::from([c0, c0, c0, j[0].conj()])),
            }
        }
        if let Some(autos) = autos.as_mut() {
            let c0 = num_complex::Complex32::default();
            match (self.conjugate_vis, self.metadata.pols) {
                // These pols are all handled correctly.
                (false, Polarisations::XX_XY_YX_YY) => (),
                (false, Polarisations::XX) => (),
                (false, Polarisations::XX_YY) => (),
                (false, Polarisations::XX_YY_XY) => (),
                // Just conjugate.
                (
                    true,
                    Polarisations::XX_XY_YX_YY
                    | Polarisations::XX
                    | Polarisations::XX_YY
                    | Polarisations::XX_YY_XY,
                ) => autos.vis_fb.mapv_inplace(|j| {
                    Jones::from([j[0].conj(), j[1].conj(), j[2].conj(), j[3].conj()])
                }),

                // Because we read in one polarisation, it was treated as XX,
                // but this is actually YY.
                (false, Polarisations::YY) => autos
                    .vis_fb
                    .mapv_inplace(|j| Jones::from([c0, c0, c0, j[0]])),
                (true, Polarisations::YY) => autos
                    .vis_fb
                    .mapv_inplace(|j| Jones::from([c0, c0, c0, j[0].conj()])),
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

        match (
            self.metadata.pols.num_pols(),
            self.metadata.num_floats_per_pol,
        ) {
            (4, 3) => self.read_inner::<4, 3>(cross_data, auto_data, timestep, flagged_fine_chans),
            (3, 3) => self.read_inner::<3, 3>(cross_data, auto_data, timestep, flagged_fine_chans),
            (2, 3) => self.read_inner::<2, 3>(cross_data, auto_data, timestep, flagged_fine_chans),
            (1, 3) => self.read_inner::<1, 3>(cross_data, auto_data, timestep, flagged_fine_chans),
            (4, 2) => self.read_inner::<4, 2>(cross_data, auto_data, timestep, flagged_fine_chans),
            (3, 2) => self.read_inner::<3, 2>(cross_data, auto_data, timestep, flagged_fine_chans),
            (2, 2) => self.read_inner::<2, 2>(cross_data, auto_data, timestep, flagged_fine_chans),
            (1, 2) => self.read_inner::<1, 2>(cross_data, auto_data, timestep, flagged_fine_chans),
            _ => {
                unimplemented!("uvfits num pols must be 1-4 and num floats per pol must be 2 or 3")
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
        match (
            self.metadata.pols.num_pols(),
            self.metadata.num_floats_per_pol,
        ) {
            (4, 3) => self.read_inner::<4, 3>(cross_data, None, timestep, flagged_fine_chans),
            (3, 3) => self.read_inner::<3, 3>(cross_data, None, timestep, flagged_fine_chans),
            (2, 3) => self.read_inner::<2, 3>(cross_data, None, timestep, flagged_fine_chans),
            (1, 3) => self.read_inner::<1, 3>(cross_data, None, timestep, flagged_fine_chans),
            (4, 2) => self.read_inner::<4, 2>(cross_data, None, timestep, flagged_fine_chans),
            (3, 2) => self.read_inner::<3, 2>(cross_data, None, timestep, flagged_fine_chans),
            (2, 2) => self.read_inner::<2, 2>(cross_data, None, timestep, flagged_fine_chans),
            (1, 2) => self.read_inner::<1, 2>(cross_data, None, timestep, flagged_fine_chans),
            _ => {
                unimplemented!("uvfits num pols must be 1-4 and num floats per pol must be 2 or 3")
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
        match (
            self.metadata.pols.num_pols(),
            self.metadata.num_floats_per_pol,
        ) {
            (4, 3) => self.read_inner::<4, 3>(None, auto_data, timestep, flagged_fine_chans),
            (3, 3) => self.read_inner::<3, 3>(None, auto_data, timestep, flagged_fine_chans),
            (2, 3) => self.read_inner::<2, 3>(None, auto_data, timestep, flagged_fine_chans),
            (1, 3) => self.read_inner::<1, 3>(None, auto_data, timestep, flagged_fine_chans),
            (4, 2) => self.read_inner::<4, 2>(None, auto_data, timestep, flagged_fine_chans),
            (3, 2) => self.read_inner::<3, 2>(None, auto_data, timestep, flagged_fine_chans),
            (2, 2) => self.read_inner::<2, 2>(None, auto_data, timestep, flagged_fine_chans),
            (1, 2) => self.read_inner::<1, 2>(None, auto_data, timestep, flagged_fine_chans),
            _ => {
                unimplemented!("uvfits num pols must be 1-4 and num floats per pol must be 2 or 3")
            }
        }
    }

    fn get_marlu_mwa_info(&self) -> Option<MarluMwaObsContext> {
        self.get_metafits_context()
            .map(MarluMwaObsContext::from_mwalib)
    }
}

struct UvfitsMetadata {
    /// The number of rows in the metafits file (hopefully equal to the number
    /// of timesteps * the number of baselines).
    num_rows: usize,

    /// The number of parameters are in each uvfits group (PCOUNT).
    pcount: usize,

    /// The available polarisations.
    pols: Polarisations,

    /// The number of floats associated with a polarisation. If this value is 3,
    /// these are the real part of the pol, imag part of the pol, and the
    /// weight, respectively. If this value is 2, then it's the same as 3,
    /// except the weight is always 1.0.
    num_floats_per_pol: u8,

    /// The... number of fine channel frequencies.
    num_fine_freq_chans: usize,

    /// The Julian date at midnight of the first day of the observation, as per
    /// the uvfits standard.
    jd_zero: Epoch,

    /// The time resolution \[seconds\] determined by INTTIM (if it's
    /// available). We don't support multiple values of INTTIM, so this one
    /// value is true for all data.
    time_res: Option<f64>,

    /// The indices of various parameters (e.g. BASELINE is PTYPE4, DATE is
    /// PTYPE5, etc.)
    indices: Indices,

    /// Unique collection of baselines (uvfits formatted, i.e. need to be
    /// decoded).
    uvfits_baselines: HashSet<usize>,

    /// Unique collection of antennas (uvfits formatted, i.e. need to be
    /// decoded).
    uvfits_antennas: HashSet<usize>,

    /// Unique collection of JD fractions for timestamps.
    jd_frac_timestamps: Vec<Duration>,

    /// Are auto-correlations present?
    autocorrelations_present: bool,
}

impl UvfitsMetadata {
    /// Get metadata on the supplied uvfits file.
    ///
    /// This function assumes the correct HDU has already been opened (should be
    /// HDU "AIPS AN").
    fn new(uvfits: &mut FitsFile, hdu: &FitsHdu) -> Result<Self, UvfitsReadError> {
        let indices = Indices::new(uvfits, hdu)?;

        // The file tells us what time standard is being used (probably UTC). If
        // this is false, then we assume TAI.
        let uses_utc_time = {
            let timsys: Option<String> = fits_get_optional_key(uvfits, hdu, "TIMSYS")?;
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
        let stokes_naxis_str: Cow<str> = format!("NAXIS{}", indices.stokes).into();
        let num_pols_str: String = fits_get_required_key(uvfits, hdu, stokes_naxis_str.as_ref())?;
        let num_pols = match num_pols_str.parse::<u8>() {
            Ok(n) => n,
            Err(e) => {
                return Err(UvfitsReadError::Parse {
                    key: stokes_naxis_str,
                    value: num_pols_str,
                    parse_error: e.to_string(),
                })
            }
        };

        // The pol type is described by the CRVAL key associated with STOKES.
        let stokes_crval_str = format!("CRVAL{}", indices.stokes);
        let pol_type_str: String = fits_get_required_key(uvfits, hdu, &stokes_crval_str)?;
        let pols = match pol_type_str.parse::<f32>() {
            Ok(pol_type) => {
                // Convert the float to an int.
                if pol_type.abs() > 127.0 {
                    panic!(
                        "STOKES {stokes_crval_str} has an unsupported value (absolute value > 127)"
                    );
                }
                let pol_type = pol_type.round() as i8;

                // We currently only support a "pol type" of -5 or -6, i.e. XX or YY.
                match (pol_type, num_pols) {
                    (-5, 1) => Polarisations::XX,
                    (-5, 2) => Polarisations::XX_YY,
                    (-5, 3) => Polarisations::XX_YY_XY,
                    (-5, 4) => Polarisations::XX_XY_YX_YY,
                    (-5, _) => {
                        return Err(UvfitsReadError::UnsupportedPols {
                            crval: Cow::from(stokes_crval_str),
                            naxis: stokes_naxis_str,
                            pol_type,
                            num_pols,
                        })
                    }
                    (-6, 1) => Polarisations::YY,
                    (-6, _) => {
                        return Err(UvfitsReadError::UnsupportedPols {
                            crval: Cow::from(stokes_crval_str),
                            naxis: stokes_naxis_str,
                            pol_type,
                            num_pols,
                        })
                    }
                    _ => {
                        return Err(UvfitsReadError::UnsupportedPolType {
                            key: Cow::from(stokes_crval_str),
                            value: pol_type,
                        })
                    }
                }
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
                            format!("uvfits {pzero}, corresponding to the second DATE, was not 0; ignoring it anyway").warn()
                        }
                    }
                    Err(std::num::ParseFloatError { .. }) => {
                        format!("Could not parse uvfits {pzero} as a float").warn()
                    }
                },
                None => format!("uvfits {pzero} does not exist, corresponding to the second DATE")
                    .warn(),
            }
        }
        let jd_zero = jd_zero_str
            .parse::<f64>()
            .map_err(|e| UvfitsReadError::Parse {
                key: Cow::from(jd_zero_val_str),
                value: jd_zero_str,
                parse_error: e.to_string(),
            })?;
        // Given what JD zero is supposed to represent, we can round to the
        // nearest hour; doing this helps ward off float precision issues.
        let jd_zero = {
            let e = if uses_utc_time {
                Epoch::from_jde_utc(jd_zero)
            } else {
                Epoch::from_jde_tai(jd_zero)
            };

            // Don't round if the value is 0. Sigh.
            if jd_zero.abs() < f64::EPSILON {
                format!("uvfits PZERO{} is supposed to be non-zero!", indices.date1).warn();
                e
            } else {
                e.round(1.hours())
            }
        };

        // Read unique group parameters (timestamps and baselines/antennas).
        let mut uvfits_baselines = HashSet::new();
        let mut uvfits_antennas = HashSet::new();
        let mut autocorrelations_present = false;
        let mut jd_frac_timestamp_set = HashSet::new();
        let mut jd_frac_timestamps = vec![];
        let mut time_res = None;

        // Determine the number of baselines present. We assume that all
        // baselines of a timestep are grouped together, so the number of
        // baselines is found when the timestamp changes.
        let mut group_params = Array2::zeros((num_rows.min(8257), pcount)); // 8257 is likely to get all MWA baselines in a single pass.
        let num_cross_and_auto_baselines = {
            let mut num_cross_and_auto_baselines = None;
            let mut i_row = 0;
            'outer: while i_row < num_rows {
                let num_rows_to_iterate =
                    group_params.len_of(Axis(0)).min(num_rows.abs_diff(i_row));
                unsafe {
                    let mut status = 0;
                    // ffggpe = fits_read_grppar_flt
                    fitsio_sys::ffggpe(
                        uvfits.as_raw(), /* I - FITS file pointer                       */
                        (i_row + 1).try_into().expect("not larger than i64::MAX"), /* I - group to read (1 = 1st group)           */
                        1, /* I - first vector element to read (1 = 1st)  */
                        (num_rows_to_iterate * pcount)
                            .try_into()
                            .expect("not larger than i64::MAX"), /* I - number of values to read                */
                        group_params.as_mut_ptr(), /* O - array of values that are returned       */
                        &mut status,               /* IO - error status                           */
                    );
                    // Check the status.
                    fits_check_status(status).map_err(UvfitsReadError::Metadata)?;
                }

                for params in group_params.outer_iter().take(num_rows_to_iterate) {
                    // Track information on the timestamps.
                    let jd_frac = {
                        let mut t =
                            Duration::from_days(f64::from(params[usize::from(indices.date1) - 1]));
                        // Use the second date, if it's there.
                        if let Some(d2) = indices.date2 {
                            t += Duration::from_days(f64::from(params[usize::from(d2) - 1]));
                        }
                        t
                    };
                    let nanos = jd_frac.total_nanoseconds();
                    // If our set is empty, it's because this is the first
                    // timestamp.
                    if jd_frac_timestamp_set.is_empty() {
                        jd_frac_timestamp_set.insert(nanos);
                        jd_frac_timestamps.push(jd_frac);
                    }
                    // If the set doesn't contain this timestamp, we've hit the
                    // next timestep, and we've therefore found the number of
                    // baselines per timestep.
                    if !jd_frac_timestamp_set.contains(&nanos) {
                        num_cross_and_auto_baselines = Some(i_row);
                        break 'outer;
                    }

                    // Track information on the baseline/antennas.
                    let (ant1, ant2) = match indices.baseline_or_antennas {
                        BaselineOrAntennas::Baseline { index } => {
                            let uvfits_bl = params[usize::from(index) - 1] as usize;
                            if !uvfits_baselines.contains(&uvfits_bl) {
                                uvfits_baselines.insert(uvfits_bl);
                            }
                            decode_uvfits_baseline(uvfits_bl)
                        }
                        BaselineOrAntennas::Antennas { index1, index2 } => {
                            let uvfits_ant1 = params[usize::from(index1) - 1] as usize;
                            let uvfits_ant2 = params[usize::from(index2) - 1] as usize;
                            uvfits_antennas.insert(uvfits_ant1);
                            uvfits_antennas.insert(uvfits_ant2);
                            (uvfits_ant1, uvfits_ant2)
                        }
                    };

                    if !autocorrelations_present && (ant1 == ant2) {
                        autocorrelations_present = true;
                    }

                    // Get/check time resolution.
                    if let Some(i_inttim) = indices.inttim {
                        let inttim = params[usize::from(i_inttim) - 1] as f64;
                        if let Some(time_res) = time_res {
                            assert_eq!(time_res, inttim);
                        } else if time_res.is_none() {
                            time_res = Some(inttim);
                        }
                    }

                    i_row += 1;
                }
            }
            num_cross_and_auto_baselines.unwrap_or(i_row)
        };

        // Now get the timestamps out of all the other timesteps.
        let mut i_row = num_cross_and_auto_baselines;
        assert!(
            num_rows % num_cross_and_auto_baselines == 0,
            "There are a variable number of baselines per timestep, which is not supported"
        );
        while i_row < num_rows {
            unsafe {
                let mut status = 0;
                // ffggpe = fits_read_grppar_flt
                fitsio_sys::ffggpe(
                    uvfits.as_raw(), /* I - FITS file pointer                       */
                    (i_row + 1).try_into().expect("not larger than i64::MAX"), /* I - group to read (1 = 1st group)           */
                    1, /* I - first vector element to read (1 = 1st)  */
                    pcount.try_into().expect("not larger than i64::MAX"), /* I - number of values to read                */
                    group_params.as_mut_ptr(), /* O - array of values that are returned       */
                    &mut status,               /* IO - error status                           */
                );
                // Check the status.
                fits_check_status(status).map_err(UvfitsReadError::Metadata)?;
            }

            // Take the metadata out of read-in group parameters.
            let jd_frac = {
                let mut t = Duration::from_days(f64::from(
                    group_params[(0, usize::from(indices.date1) - 1)],
                ));
                // Use the second date, if it's there.
                if let Some(d2) = indices.date2 {
                    t += Duration::from_days(f64::from(group_params[(0, usize::from(d2) - 1)]));
                }
                t
            };
            let nanos = jd_frac.total_nanoseconds();
            if !jd_frac_timestamp_set.contains(&nanos) {
                jd_frac_timestamp_set.insert(nanos);
                jd_frac_timestamps.push(jd_frac);
            }

            match indices.baseline_or_antennas {
                BaselineOrAntennas::Baseline { index } => {
                    let uvfits_bl = group_params[(0, usize::from(index) - 1)] as usize;
                    if !uvfits_baselines.contains(&uvfits_bl) {
                        // TODO: error
                    }
                }
                BaselineOrAntennas::Antennas { index1, index2 } => {
                    let uvfits_ant1 = group_params[(0, usize::from(index1) - 1)] as usize;
                    let uvfits_ant2 = group_params[(0, usize::from(index2) - 1)] as usize;
                    if !uvfits_antennas.contains(&uvfits_ant1)
                        || !uvfits_antennas.contains(&uvfits_ant2)
                    {
                        // TODO: Error
                    }
                }
            }

            i_row += num_cross_and_auto_baselines;
        }

        Ok(UvfitsMetadata {
            num_rows,
            pcount,
            pols,
            num_floats_per_pol,
            num_fine_freq_chans,
            jd_zero,
            time_res,
            indices,
            uvfits_baselines,
            uvfits_antennas,
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
    /// PTYPE
    inttim: Option<u8>,
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
        let mut inttim_index = None;

        for (i, key) in ptypes.into_iter().enumerate() {
            let ii = (i + 1) as u8;
            match key.as_str() {
                "UU" => {
                    if u_index.is_none() {
                        u_index = Some(ii)
                    } else {
                        "Found another uvfits UU key -- only using the first".warn();
                    }
                }
                "VV" => {
                    if v_index.is_none() {
                        v_index = Some(ii)
                    } else {
                        "Found another uvfits VV key -- only using the first".warn();
                    }
                }
                "WW" => {
                    if w_index.is_none() {
                        w_index = Some(ii)
                    } else {
                        "Found another uvfits WW key -- only using the first".warn();
                    }
                }
                "BASELINE" => {
                    if baseline_index.is_none() {
                        baseline_index = Some(ii)
                    } else {
                        "Found another uvfits BASELINE key -- only using the first".warn();
                    }
                }
                "ANTENNA1" => {
                    if antenna1_index.is_none() {
                        antenna1_index = Some(ii)
                    } else {
                        "Found another uvfits ANTENNA1 key -- only using the first".warn();
                    }
                }
                "ANTENNA2" => {
                    if antenna2_index.is_none() {
                        antenna2_index = Some(ii)
                    } else {
                        "Found another uvfits ANTENNA1 key -- only using the first".warn();
                    }
                }
                "DATE" | "_DATE" => match (date1_index, date2_index) {
                    (None, None) => date1_index = Some(ii),
                    (Some(_), None) => date2_index = Some(ii),
                    (Some(_), Some(_)) => {
                        "Found more than 2 uvfits DATE/_DATE keys -- only using the first two"
                            .warn()
                    }
                    (None, Some(_)) => unreachable!(),
                },
                "INTTIM" => {
                    if inttim_index.is_none() {
                        inttim_index = Some(ii)
                    } else {
                        warn!("Found another INTTIM key -- only using the first");
                    }
                }
                _ => (),
            }
        }

        // Handle problems surrounding some combination of BASELINE and
        // ANTENNA1/ANTENNA2.
        let baseline_or_antennas = match (baseline_index, antenna1_index, antenna2_index) {
            // These are OK.
            (Some(index), None, None) => BaselineOrAntennas::Baseline { index },
            (None, Some(index1), Some(index2)) => BaselineOrAntennas::Antennas { index1, index2 },
            (Some(index), Some(_), _) | (Some(index), _, Some(_)) => {
                "Found both uvfits BASELINE and ANTENNA keys; only using BASELINE".warn();
                BaselineOrAntennas::Baseline { index }
            }
            // These are not.
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
            inttim: inttim_index,
        })
    }
}
