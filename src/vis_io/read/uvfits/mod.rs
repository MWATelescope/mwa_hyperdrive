// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to handle reading from uvfits files.

mod error;
#[cfg(test)]
mod tests;

pub(crate) use error::*;

use std::collections::{HashMap, HashSet};
use std::os::raw::c_char;
use std::path::{Path, PathBuf};

use hifitime::{Duration, Epoch, Unit};
use log::{debug, trace, warn};
use marlu::{io::uvfits::decode_uvfits_baseline, Jones, RADec, XyzGeocentric, XyzGeodetic};
use mwalib::{
    fitsio::{errors::check_status as fits_check_status, hdu::FitsHdu, FitsFile},
    *,
};
use ndarray::prelude::*;

use super::*;
use crate::{
    beam::Delays,
    context::ObsContext,
    metafits,
    misc::quantize_duration,
    vis_io::read::{VisRead, VisReadError},
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
    ) -> Result<UvfitsReader, VisReadError> {
        fn inner(uvfits: &Path, metafits: Option<&Path>) -> Result<UvfitsReader, UvfitsReadError> {
            // If a metafits file was provided, get an mwalib object ready.
            // TODO: Let the user supply the MWA version.
            let mwalib_context = match metafits {
                None => None,
                Some(m) => Some(mwalib::MetafitsContext::new(m, None)?),
            };

            debug!("Using uvfits file: {}", uvfits.display());
            if !uvfits.exists() {
                return Err(UvfitsReadError::BadFile(uvfits.to_path_buf()));
            }

            // Get the tile names, XYZ positions and antenna numbers.
            let mut uvfits_fptr = fits_open!(&uvfits)?;
            let antenna_table_hdu = fits_open_hdu!(&mut uvfits_fptr, 1)?;

            let tile_names: Vec<String> =
                get_fits_col!(&mut uvfits_fptr, &antenna_table_hdu, "ANNAME")?;
            let tile_names =
                Vec1::try_from_vec(tile_names).map_err(|_| UvfitsReadError::AnnameEmpty)?;
            let total_num_tiles = tile_names.len();

            let tile_xyzs = {
                let mut tile_xyzs: Vec<XyzGeodetic> = Vec::with_capacity(total_num_tiles);
                for i in 0..total_num_tiles {
                    let fits_xyz = read_cell_array(
                        &mut uvfits_fptr,
                        &antenna_table_hdu,
                        "STABXYZ",
                        i as _,
                        3,
                    )?;
                    tile_xyzs.push(XyzGeodetic {
                        x: fits_xyz[0],
                        y: fits_xyz[1],
                        z: fits_xyz[2],
                    });
                }
                tile_xyzs
            };
            let tile_xyzs = Vec1::try_from_vec(tile_xyzs).unwrap();

            // Set up the tile map.
            let tile_nums: Vec<u32> = get_fits_col!(&mut uvfits_fptr, &antenna_table_hdu, "NOSTA")?;
            let tile_map: HashMap<usize, usize> = tile_nums
                .into_iter()
                .zip(0..total_num_tiles)
                .map(|(a, b)| (a as usize, b))
                .collect();

            let array_position = {
                let frame: Option<String> =
                    get_optional_fits_key!(&mut uvfits_fptr, &antenna_table_hdu, "FRAME")?;
                // The uvfits standard only defines one frame (ITRF). So warn
                // the user if this isn't explicit, but we assume this is always
                // used.
                let itrf_frame_warning = match frame.as_ref().map(|s| s.trim()) {
                    Some("ITRF") => None,
                    _ => Some("Assuming that the uvfits antenna coordinate system is ITRF"),
                };
                let array_x: Option<f64> =
                    get_optional_fits_key!(&mut uvfits_fptr, &antenna_table_hdu, "ARRAYX")?;
                let array_y: Option<f64> =
                    get_optional_fits_key!(&mut uvfits_fptr, &antenna_table_hdu, "ARRAYY")?;
                let array_z: Option<f64> =
                    get_optional_fits_key!(&mut uvfits_fptr, &antenna_table_hdu, "ARRAYZ")?;
                match (array_x, array_y, array_z) {
                    (Some(x), Some(y), Some(z)) => {
                        if let Some(itrf_frame_warning) = itrf_frame_warning {
                            warn!("{itrf_frame_warning}");
                        }
                        Some(XyzGeocentric { x, y, z }.to_earth_wgs84()?)
                    }
                    (None, None, None) => None,
                    _ => {
                        warn!("Only a subset of uvfits ARRAYX, ARRAYY, ARRAYZ is available; ignoring present values");
                        None
                    }
                }
            };

            let hdu = fits_open_hdu!(&mut uvfits_fptr, 0)?;
            let metadata = UvfitsMetadata::new(&mut uvfits_fptr, &hdu)?;
            debug!("Number of rows in the uvfits: {}", metadata.num_rows);
            debug!("PCOUNT: {}", metadata.pcount);
            debug!("Number of cross polarisations: {}", metadata.num_pols);
            debug!("Floats per polarisation: {}", metadata.floats_per_pol);
            debug!(
                "Number of fine frequency chans: {}",
                metadata.num_fine_freq_chans
            );
            debug!("UU index:       {}", metadata.indices.u);
            debug!("VV index:       {}", metadata.indices.v);
            debug!("WW index:       {}", metadata.indices.w);
            debug!("BASELINE index: {}", metadata.indices.baseline);
            debug!("DATE index:     {}", metadata.indices.date1);
            if let Some(d2) = metadata.indices.date2 {
                debug!("(Second) DATE index: {}", d2);
            }
            debug!("RA index:       {}", metadata.indices.ra);
            debug!("DEC index:      {}", metadata.indices.dec);
            debug!("FREQ index:     {}", metadata.indices.freq);

            if metadata.num_rows == 0 {
                return Err(UvfitsReadError::Empty(uvfits.to_path_buf()));
            }

            // The phase centre is described by RA and DEC if there is no SOURCE
            // table (as per the standard).
            // TODO: Check that there is no SOURCE table!
            let phase_centre = {
                let ra = get_required_fits_key!(
                    &mut uvfits_fptr,
                    &hdu,
                    &format!("CRVAL{}", metadata.indices.ra)
                )?;
                let dec = get_required_fits_key!(
                    &mut uvfits_fptr,
                    &hdu,
                    &format!("CRVAL{}", metadata.indices.dec)
                )?;
                RADec::new_degrees(ra, dec)
            };

            // Populate the dipole delays and the pointing centre if we can.
            let mut dipole_delays: Option<Delays> = None;
            let mut pointing_centre: Option<RADec> = None;
            if let Some(context) = &mwalib_context {
                debug!("Using metafits for dipole delays and pointing centre");
                dipole_delays = Some(Delays::Full(metafits::get_dipole_delays(context)));
                pointing_centre = Some(RADec::new_degrees(
                    context.ra_tile_pointing_degrees,
                    context.dec_tile_pointing_degrees,
                ));
            }

            // Work out which tiles are unavailable.
            let mut present_tiles_set: HashSet<usize> = HashSet::new();
            metadata.uvfits_baselines.iter().for_each(|&uvfits_bl| {
                let (ant1, ant2) = decode_uvfits_baseline(uvfits_bl);
                present_tiles_set.insert(ant1);
                present_tiles_set.insert(ant2);
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
                    get_optional_fits_key!(&mut uvfits_fptr, &hdu, "TIMSYS")?;
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
                    let jd_offset = Duration::from_f64(frac, Unit::Day);
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
                    get_optional_fits_key!(&mut uvfits_fptr, &hdu, "INTTIM")?;
                match int_time {
                    Some(t) => {
                        let d = Duration::from_f64(t, Unit::Second);
                        trace!("Time resolution from INTTIM: {}s", d.in_seconds());
                        Some(d)
                    }
                    None => {
                        if timestamps.len() == 1 {
                            debug!("Only one timestep is present in the data; can't determine the data's time resolution.");
                            None
                        } else {
                            // Find the minimum gap between two consecutive
                            // timestamps.
                            let time_res = timestamps.windows(2).fold(
                                Duration::from_f64(f64::INFINITY, Unit::Second),
                                |acc, ts| acc.min(ts[1] - ts[0]),
                            );
                            trace!(
                                "Time resolution from smallest gap: {}s",
                                time_res.in_seconds()
                            );
                            Some(time_res)
                        }
                    }
                }
            };
            match timestamps.as_slice() {
                // Handled above; uvfits files aren't allowed to be empty.
                [] => unreachable!(),
                [t] => debug!("Only timestep (GPS): {:.2}", t.as_gpst_seconds()),
                [t0, .., tn] => {
                    debug!("First good timestep (GPS): {:.2}", t0.as_gpst_seconds());
                    debug!("Last good timestep  (GPS): {:.2}", tn.as_gpst_seconds());
                }
            }

            debug!("Unavailable tiles in the uvfits: {unavailable_tiles:?}");
            debug!("Flagged tiles in the uvfits: {flagged_tiles:?}");
            debug!(
                "Autocorrelations present: {}",
                metadata.autocorrelations_present
            );

            // Get the dipole gains. Only available with a metafits.
            let dipole_gains = mwalib_context.as_ref().map(metafits::get_dipole_gains);

            // Get the obsid. There is an "obs. name" in the "object" filed, but
            // that's not the same thing.
            let obsid = mwalib_context.as_ref().map(|context| context.obs_id);

            let step = metadata.num_rows / timestamps.len();

            let freq_val_str = format!("CRVAL{}", metadata.indices.freq);
            let base_freq_str: String =
                get_required_fits_key!(&mut uvfits_fptr, &hdu, &freq_val_str)?;
            let base_freq: f64 = match base_freq_str.parse() {
                Ok(p) => p,
                Err(_) => {
                    return Err(UvfitsReadError::Parse {
                        key: freq_val_str,
                        value: base_freq_str,
                    })
                }
            };
            let base_index: isize = {
                // CRPIX might be a float. Parse it as one, then make it an int.
                let freq_val_str = format!("CRPIX{}", metadata.indices.freq);
                let f_str: String = get_required_fits_key!(&mut uvfits_fptr, &hdu, &freq_val_str)?;
                let f: f64 = match f_str.parse() {
                    Ok(p) => p,
                    Err(_) => {
                        return Err(UvfitsReadError::Parse {
                            key: freq_val_str,
                            value: f_str,
                        })
                    }
                };
                f.round() as _
            };
            let freq_val_str = format!("CDELT{}", metadata.indices.freq);
            let fine_chan_width_str: String =
                get_required_fits_key!(&mut uvfits_fptr, &hdu, &freq_val_str)?;
            let freq_res: f64 = match fine_chan_width_str.parse() {
                Ok(p) => p,
                Err(_) => {
                    return Err(UvfitsReadError::Parse {
                        key: freq_val_str,
                        value: fine_chan_width_str,
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
                        .map(|cc| (cc.gpubox_number as u32, f64::from(cc.chan_centre_hz)))
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
                        .into_iter()
                        .map(|n| n as u32)
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
                // TODO: Don't assume the "AIPS AN" HDU is HDU 2.
                let antenna_table_hdu = fits_open_hdu!(&mut uvfits_fptr, 1)?;
                let uvfits_dut1: Option<f64> =
                    get_optional_fits_key!(&mut uvfits_fptr, &antenna_table_hdu, "UT1UTC")?;
                match uvfits_dut1 {
                    Some(dut1) => debug!("uvfits DUT1: {dut1}"),
                    None => debug!("uvfits has no DUT1 (UT1UTC key)"),
                }

                let metafits_dut1 = mwalib_context.as_ref().and_then(|c| c.dut1);
                match metafits_dut1 {
                    Some(dut1) => debug!("metafits DUT1: {dut1}"),
                    None => debug!("metafits has no DUT1"),
                }

                if metafits_dut1.is_some() && uvfits_dut1.is_some() {
                    debug!("Preferring metafits DUT1 over uvfits DUT1");
                }
                metafits_dut1
                    .or(uvfits_dut1)
                    .map(|dut1| Duration::from_f64(dut1, Unit::Second))
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
        inner(uvfits.as_ref(), metafits.as_ref().map(|f| f.as_ref())).map_err(VisReadError::from)
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

        let mut uvfits = fits_open!(&self.uvfits).map_err(UvfitsReadError::from)?;
        fits_open_hdu!(&mut uvfits, 0).map_err(UvfitsReadError::from)?;
        let mut group_params: Vec<f32> = vec![0.0; self.metadata.pcount];
        let mut uvfits_vis: Array3<f32> = Array3::zeros((
            self.metadata.num_fine_freq_chans,
            self.metadata.num_pols,
            self.metadata.floats_per_pol,
        ));
        for row in row_range_start..row_range_end {
            // Read in the row's group parameters.
            let mut status = 0;
            let uvfits_bl = unsafe {
                // ffggpe = fits_read_grppar_flt
                fitsio_sys::ffggpe(
                    uvfits.as_raw(),           /* I - FITS file pointer                       */
                    1 + row as i64,            /* I - group to read (1 = 1st group)           */
                    1,                         /* I - first vector element to read (1 = 1st)  */
                    group_params.len() as i64, /* I - number of values to read                */
                    group_params.as_mut_ptr(), /* O - array of values that are returned       */
                    &mut status,               /* IO - error status                           */
                );
                fits_check_status(status).map_err(|err| UvfitsReadError::ReadVis {
                    row_num: row + 1,
                    err,
                })?;

                group_params[(self.metadata.indices.baseline - 1) as usize]
            };

            let (ant1, ant2) = decode_uvfits_baseline(uvfits_bl.round() as usize);
            let (ant1, ant2) = (self.tile_map[&ant1], self.tile_map[&ant2]);

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
                            uvfits.as_raw(),         /* I - FITS file pointer                       */
                            1 + row as i64, /* I - group to read (1 = 1st group)           */
                            1,              /* I - first vector element to read (1 = 1st)  */
                            uvfits_vis.len() as i64, /* I - number of values to read                */
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
                    let mut out_vis = crosses.data_array.slice_mut(s![i_baseline, ..]);
                    let mut out_weights = crosses.weights_array.slice_mut(s![i_baseline, ..]);
                    uvfits_vis
                        .outer_iter()
                        .enumerate()
                        .filter(|(i_chan, _)| !flagged_fine_chans.contains(i_chan))
                        .zip(out_vis.iter_mut())
                        .zip(out_weights.iter_mut())
                        .for_each(|(((_, data_pol_axis), out_vis), out_weight)| {
                            // These are the components of the input data's
                            // visibilities.
                            let data_xx = data_pol_axis.index_axis(Axis(0), 0);
                            let data_yy = data_pol_axis.index_axis(Axis(0), 1);
                            let data_xy = data_pol_axis.index_axis(Axis(0), 2);
                            let data_yx = data_pol_axis.index_axis(Axis(0), 3);

                            // Write to the output weights array. We assume that
                            // each polarisation weight is equal.
                            *out_weight = data_xx[2];

                            // Write the input data visibility components to the
                            // output data array.
                            *out_vis = Jones::from([
                                data_xx[0], data_xx[1], data_xy[0], data_xy[1], data_yx[0],
                                data_yx[1], data_yy[0], data_yy[1],
                            ]);
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
                                uvfits.as_raw(),         /* I - FITS file pointer                       */
                                1 + row as i64, /* I - group to read (1 = 1st group)           */
                                1,              /* I - first vector element to read (1 = 1st)  */
                                uvfits_vis.len() as i64, /* I - number of values to read                */
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

                        let mut out_vis = autos.data_array.slice_mut(s![i_ant, ..]);
                        let mut out_weights = autos.weights_array.slice_mut(s![i_ant, ..]);
                        uvfits_vis
                            .outer_iter()
                            .enumerate()
                            .filter(|(i_chan, _)| !flagged_fine_chans.contains(i_chan))
                            .zip(out_vis.iter_mut())
                            .zip(out_weights.iter_mut())
                            .for_each(|(((_, data_pol_axis), out_vis), out_weight)| {
                                let data_xx = data_pol_axis.index_axis(Axis(0), 0);
                                let data_yy = data_pol_axis.index_axis(Axis(0), 1);
                                let data_xy = data_pol_axis.index_axis(Axis(0), 2);
                                let data_yx = data_pol_axis.index_axis(Axis(0), 3);

                                // We assume that weights are all equal for these
                                // visibilities.
                                *out_weight = data_xx[2];
                                *out_vis = Jones::from([
                                    data_xx[0], data_xx[1], data_xy[0], data_xy[1], data_yx[0],
                                    data_yx[1], data_yy[0], data_yy[1],
                                ]);
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
    num_rows: usize,
    /// The number of parameters are in each uvfits group (PCOUNT).
    pcount: usize,
    /// The number of cross polarisations (probably 4).
    num_pols: usize,
    /// The number of floats are associated with a cross pol (probably 3; real
    /// part of visibilitiy, imag part of visibility, weight).
    floats_per_pol: usize,
    num_fine_freq_chans: usize,
    // The Julian date at midnight of the first day of the observation, as per
    // the uvfits standard.
    jd_zero: f64,
    /// The indices of various parameters (e.g. BASELINE is PTYPE4, DATE is
    /// PTYPE5, etc.)
    indices: Indices,

    /// Unique collection of baselines (uvfits formatted, i.e. need to be
    /// decoded).
    uvfits_baselines: Vec<usize>,

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
        let num_rows_str: String = get_required_fits_key!(uvfits, hdu, "GCOUNT")?;
        let num_rows: usize = match num_rows_str.parse() {
            Ok(p) => p,
            Err(_) => {
                return Err(UvfitsReadError::Parse {
                    key: "GCOUNT".to_string(),
                    value: num_rows_str,
                })
            }
        };
        // PCOUNT tells us how many parameters are in each uvfits group.
        let pcount_str: String = get_required_fits_key!(uvfits, hdu, "PCOUNT")?;
        let pcount: usize = match pcount_str.parse() {
            Ok(p) => p,
            Err(_) => {
                return Err(UvfitsReadError::Parse {
                    key: "PCOUNT".to_string(),
                    value: pcount_str,
                })
            }
        };
        // NAXIS2 is how many floats are associated with a cross pol (probably 3; real
        // part of visibilitiy, imag part of visibility, weight).
        let floats_per_pol_str: String = get_required_fits_key!(uvfits, hdu, "NAXIS2")?;
        let floats_per_pol: usize = match floats_per_pol_str.parse() {
            Ok(p) => p,
            Err(_) => {
                return Err(UvfitsReadError::Parse {
                    key: "NAXIS2".to_string(),
                    value: floats_per_pol_str,
                })
            }
        };
        // NAXIS3 is the number of cross pols.
        let num_pols_str: String = get_required_fits_key!(uvfits, hdu, "NAXIS3")?;
        let num_pols: usize = match num_pols_str.parse() {
            Ok(p) => p,
            Err(_) => {
                return Err(UvfitsReadError::Parse {
                    key: "NAXIS3".to_string(),
                    value: num_pols_str,
                })
            }
        };

        // NAXIS4 is the number of fine-frequency channels.
        let num_fine_freq_chans_str: String = get_required_fits_key!(uvfits, hdu, "NAXIS4")?;
        let num_fine_freq_chans: usize = match num_fine_freq_chans_str.parse() {
            Ok(p) => p,
            Err(_) => {
                return Err(UvfitsReadError::Parse {
                    key: "NAXIS4".to_string(),
                    value: num_fine_freq_chans_str,
                })
            }
        };

        // "JD zero" refers to the Julian date at midnight of the first day of
        // the observation, as per the uvfits standard.
        let jd_zero_val_str = format!("PZERO{}", indices.date1);
        let jd_zero_str: String = get_required_fits_key!(uvfits, hdu, &jd_zero_val_str)?;
        // We expect that the PZERO corresponding to the second date (if
        // available) is 0.
        if let Some(d2) = indices.date2 {
            let pzero = format!("PZERO{}", d2);
            let key: Option<String> = get_optional_fits_key!(uvfits, hdu, &pzero)?;
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
        let jd_zero: f64 = match jd_zero_str.parse() {
            Ok(p) => p,
            Err(_) => {
                return Err(UvfitsReadError::Parse {
                    key: jd_zero_val_str,
                    value: jd_zero_str,
                })
            }
        };

        // Read unique group parameters (timestamps and baselines).
        let mut uvfits_baselines_set = HashSet::new();
        let mut uvfits_baselines = vec![];
        let mut autocorrelations_present = false;
        let mut jd_frac_timestamp_set = HashSet::new();
        let mut jd_frac_timestamps = vec![];

        let mut group_params = Array2::zeros((num_rows, pcount));
        unsafe {
            let mut status = 0;
            // ffggpe = fits_read_grppar_flt
            fitsio_sys::ffggpe(
                uvfits.as_raw(),           /* I - FITS file pointer                       */
                1,                         /* I - group to read (1 = 1st group)           */
                1,                         /* I - first vector element to read (1 = 1st)  */
                (pcount * num_rows) as _,  /* I - number of values to read                */
                group_params.as_mut_ptr(), /* O - array of values that are returned       */
                &mut status,               /* IO - error status                           */
            );
            // Check the status.
            fits_check_status(status).map_err(UvfitsReadError::Metadata)?;
        }

        for params in group_params.outer_iter() {
            let uvfits_bl = params[indices.baseline as usize - 1] as usize;
            let (ant1, ant2) = decode_uvfits_baseline(uvfits_bl);
            if !autocorrelations_present && (ant1 == ant2) {
                autocorrelations_present = true;
            }
            // Don't just push into a set; we want the order of the baselines as
            // they come out of the uvfits file, and this isn't necessarily
            // sorted.
            if !uvfits_baselines_set.contains(&uvfits_bl) {
                uvfits_baselines_set.insert(uvfits_bl);
                uvfits_baselines.push(uvfits_bl);
            }

            let jd = {
                let mut t = params[indices.date1 as usize - 1] as f64;
                // Use the second date, if it's there.
                if let Some(d2) = indices.date2 {
                    t += params[d2 as usize - 1] as f64;
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
            floats_per_pol,
            num_fine_freq_chans,
            jd_zero,
            indices,
            uvfits_baselines,
            jd_frac_timestamps,
            autocorrelations_present,
        })
    }
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
    baseline: u8,
    /// PTYPE
    date1: u8,
    /// PTYPE
    date2: Option<u8>,
    /// CTYPE
    ra: u8,
    /// CTYPE
    dec: u8,
    /// CTYPE
    freq: u8,
}

impl Indices {
    /// "UU", "VV", "WW", "BASELINE" and "DATE" indices could be on any index 1
    /// to 5. These are identified by the "PTYPE" key. Check they all exist, and
    /// return the indices. Do the same for "RA", "DEC" and "FREQ" in the ctypes.
    ///
    /// This function assumes the correct HDU has already been opened (should be
    /// HDU 1, index 0).
    fn new(uvfits: &mut FitsFile, hdu: &FitsHdu) -> Result<Self, UvfitsReadError> {
        // Accumulate the "PTYPE" keys.
        let mut ptypes = Vec::with_capacity(12);
        for i in 1.. {
            let ptype: Option<String> =
                get_optional_fits_key!(uvfits, hdu, &format!("PTYPE{}", i))?;
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
        let mut date1_index = None;
        let mut date2_index = None;

        for (i, key) in ptypes.into_iter().enumerate() {
            let ii = (i + 1) as u8;
            match key.as_ref() {
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
                "DATE" => match (date1_index, date2_index) {
                    (None, None) => date1_index = Some(ii),
                    (Some(_), None) => date2_index = Some(ii),
                    (Some(_), Some(_)) => {
                        warn!("Found more than 2 DATE keys -- only using the first two")
                    }
                    (None, Some(_)) => unreachable!(),
                },
                _ => (),
            }
        }

        let (u, v, w, baseline, date1) =
            match (u_index, v_index, w_index, baseline_index, date1_index) {
                (Some(u), Some(v), Some(w), Some(baseline), Some(date1)) => {
                    (u, v, w, baseline, date1)
                }
                (None, _, _, _, _) => {
                    return Err(UvfitsReadError::MissingKey {
                        key: "UU",
                        hdu: hdu.number + 1,
                    })
                }
                (_, None, _, _, _) => {
                    return Err(UvfitsReadError::MissingKey {
                        key: "VV",
                        hdu: hdu.number + 1,
                    })
                }
                (_, _, None, _, _) => {
                    return Err(UvfitsReadError::MissingKey {
                        key: "WW",
                        hdu: hdu.number + 1,
                    })
                }
                (_, _, _, None, _) => {
                    return Err(UvfitsReadError::MissingKey {
                        key: "BASELINE",
                        hdu: hdu.number + 1,
                    })
                }
                (_, _, _, _, None) => {
                    return Err(UvfitsReadError::MissingKey {
                        key: "DATE",
                        hdu: hdu.number + 1,
                    })
                }
            };

        let ctype2: Option<String> = get_optional_fits_key!(uvfits, hdu, "CTYPE2")?;
        let ctype3: Option<String> = get_optional_fits_key!(uvfits, hdu, "CTYPE3")?;
        let ctype4: Option<String> = get_optional_fits_key!(uvfits, hdu, "CTYPE4")?;
        let ctype5: Option<String> = get_optional_fits_key!(uvfits, hdu, "CTYPE5")?;
        let ctype6: Option<String> = get_optional_fits_key!(uvfits, hdu, "CTYPE6")?;
        let ctype7: Option<String> = get_optional_fits_key!(uvfits, hdu, "CTYPE7")?;

        let mut ra_index = None;
        let mut dec_index = None;
        let mut freq_index = None;

        for (i, key) in [ctype2, ctype3, ctype4, ctype5, ctype6, ctype7]
            .iter()
            .enumerate()
        {
            let ii = (i + 2) as u8;
            match key.as_deref() {
                Some("RA") => ra_index = Some(ii),
                Some("DEC") => dec_index = Some(ii),
                Some("FREQ") => freq_index = Some(ii),
                _ => (),
            }
        }

        let (ra, dec, freq) = match (ra_index, dec_index, freq_index) {
            (Some(ra), Some(dec), Some(freq)) => (ra, dec, freq),
            (None, _, _) => {
                return Err(UvfitsReadError::MissingKey {
                    key: "RA",
                    hdu: hdu.number + 1,
                })
            }
            (_, None, _) => {
                return Err(UvfitsReadError::MissingKey {
                    key: "DEC",
                    hdu: hdu.number + 1,
                })
            }
            (_, _, None) => {
                return Err(UvfitsReadError::MissingKey {
                    key: "FREQ",
                    hdu: hdu.number + 1,
                })
            }
        };

        Ok(Indices {
            u,
            v,
            w,
            baseline,
            date1,
            date2: date2_index,
            ra,
            dec,
            freq,
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
        let keyword = std::ffi::CString::new(col_name).unwrap();
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
