// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to handle reading from uvfits files.

use std::collections::{HashMap, HashSet};
use std::os::raw::c_char;
use std::path::{Path, PathBuf};

use log::{debug, trace, warn};
use marlu::{c32, Jones, RADec, XyzGeodetic};
use mwalib::{
    fitsio::{errors::check_status as fits_check_status, hdu::FitsHdu, FitsFile},
    *,
};
use ndarray::prelude::*;

use super::*;
use crate::{
    context::ObsContext,
    data_formats::{metafits, InputData, ReadInputDataError},
    time::round_hundredths_of_a_second,
};
use mwa_hyperdrive_beam::Delays;
use mwa_hyperdrive_common::{log, marlu, mwalib, ndarray};

pub(crate) struct UvfitsReader {
    /// Observation metadata.
    obs_context: ObsContext,

    // uvfits-specific things follow.
    /// The path to the uvfits on disk.
    pub(crate) uvfits: PathBuf,

    /// The uvfits-specific metadata, like which indices contain which
    /// parameters.    
    metadata: UvfitsMetadata,

    /// The "stride" of the data, i.e. the number of rows (baselines) before the
    /// time index changes.
    step: usize,
}

impl UvfitsReader {
    /// Verify and populate metadata associated with this measurement set. TODO:
    /// Use the metafits to get dead dipole info.
    ///
    /// The measurement set is expected to be formatted in the way that
    /// cotter/Birli write measurement sets.
    pub(crate) fn new<P: AsRef<Path>, P2: AsRef<Path>>(
        uvfits: P,
        metafits: Option<P2>,
        dipole_delays: &mut Delays,
    ) -> Result<UvfitsReader, UvfitsReadError> {
        fn inner(
            uvfits: &Path,
            metafits: Option<&Path>,
            dipole_delays: &mut Delays,
        ) -> Result<UvfitsReader, UvfitsReadError> {
            // If a metafits file was provided, get an mwalib object ready.
            // TODO: Let the user supply the MWA version.
            let mwalib_context = match metafits {
                None => None,
                Some(m) => Some(mwalib::MetafitsContext::new(&m, None)?),
            };

            debug!("Using uvfits file: {}", uvfits.display());
            if !uvfits.exists() {
                return Err(UvfitsReadError::BadFile(uvfits.to_path_buf()));
            }

            // Get the tile names and XYZ positions.
            let mut uvfits_fptr = fits_open!(&uvfits)?;
            let hdu = fits_open_hdu!(&mut uvfits_fptr, 1)?;

            let tile_names: Vec<String> = get_fits_col!(&mut uvfits_fptr, &hdu, "ANNAME")?;
            let tile_names =
                Vec1::try_from_vec(tile_names).map_err(|_| UvfitsReadError::AnnameEmpty)?;
            let total_num_tiles = tile_names.len();

            let tile_xyzs = {
                let mut tile_xyzs: Vec<XyzGeodetic> = Vec::with_capacity(total_num_tiles);
                for i in 0..total_num_tiles {
                    let fits_xyz = read_cell_array(&mut uvfits_fptr, &hdu, "STABXYZ", i as _, 3)?;
                    tile_xyzs.push(XyzGeodetic {
                        x: fits_xyz[0],
                        y: fits_xyz[1],
                        z: fits_xyz[2],
                    });
                }
                tile_xyzs
            };
            let tile_xyzs = Vec1::try_from_vec(tile_xyzs).unwrap();

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
            debug!("DATE index:     {}", metadata.indices.date);
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

            // Get the dipole delays and the pointing centre (if possible).
            let pointing_centre: Option<RADec> = match &mwalib_context {
                Some(context) => {
                    // Only use the metafits delays if none were provided to
                    // this function.
                    match dipole_delays {
                        Delays::Full(_) | Delays::Partial(_) | Delays::NotNecessary => (),
                        Delays::None => {
                            debug!("Using metafits for dipole delays");
                            *dipole_delays = Delays::Full(metafits::get_dipole_delays(context));
                        }
                    }
                    Some(RADec::new_degrees(
                        context.ra_tile_pointing_degrees,
                        context.dec_tile_pointing_degrees,
                    ))
                }

                None => None,
            };
            match &dipole_delays {
                Delays::Full(d) => debug!("Dipole delays: {:?}", d),
                Delays::Partial(d) => debug!("Dipole delays: {:?}", d),
                Delays::NotNecessary => {
                    debug!("Dipole delays weren't searched for in input data; not necessary")
                }
                Delays::None => {
                    warn!("Dipole delays not provided and not available in input data!")
                }
            }

            // Work out the tile flags.
            let mut present_tiles_set: HashSet<usize> = HashSet::new();
            metadata.uvfits_baselines.iter().for_each(|&uvfits_bl| {
                let (ant1, ant2) = decode_uvfits_baseline(uvfits_bl);
                // Don't forget to subtract one from the uvfits-formatted baselines.
                present_tiles_set.insert(ant1 - 1);
                present_tiles_set.insert(ant2 - 1);
            });
            let flagged_tiles = (0..total_num_tiles)
                .into_iter()
                .filter(|i| !present_tiles_set.contains(i))
                .collect();

            // Work out the timestamp epochs. The file tells us what time standard
            // is being used (probably UTC). If this is false, then we assume TAI.
            let uses_utc_time = {
                let timsys: Option<String> =
                    get_optional_fits_key!(&mut uvfits_fptr, &hdu, "TIMSYS")?;
                match timsys {
                    None => {
                        warn!("No TIMSYS present; assuming UTC");
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
            let (all_timesteps, timestamps): (Vec<usize>, Vec<Epoch>) = metadata
                .jd_frac_timestamps
                .iter()
                .enumerate()
                .map(|(i, &frac)| {
                    let jd_days = metadata.jd_zero + (frac as f64);
                    let e = if uses_utc_time {
                        Epoch::from_jde_utc(jd_days)
                    } else {
                        Epoch::from_jde_tai(jd_days)
                    };
                    // Here's why you don't store your times in a stupid format (JD)
                    // in a single-precision float -- they come out wrong. Check how
                    // this epoch is represented in GPS; if it's close to an int,
                    // round to an int.
                    (i, round_hundredths_of_a_second(e))
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
            let time_res = if timestamps.len() == 1 {
                warn!("Only one timestep is present in the data; can't determine the data's time resolution.");
                None
            } else {
                // Assume the timestamps are contiguous, i.e. the span of time
                // between two consecutive timestamps is the same between all
                // consecutive timestamps.
                let time_res = (timestamps[1] - timestamps[0]).in_seconds();
                trace!("Time resolution: {}s", time_res);
                Some(time_res)
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

            debug!("Flagged tiles in the uvfits: {:?}", flagged_tiles);
            debug!(
                "Autocorrelations present: {}",
                metadata.autocorrelations_present
            );

            // Get the dipole gains. Only available with a metafits.
            let dipole_gains: Option<Array2<f64>> = match &mwalib_context {
                Some(context) => Some(metafits::get_dipole_gains(context)),
                None => {
                    warn!("Without a metafits file, we must assume all dipoles are alive.");
                    warn!("This will make beam Jones matrices inaccurate in sky-model generation.");
                    None
                }
            };

            // Get the obsid. There is an "obs. name" in the "object" filed, but
            // that's not the same thing.
            let obsid = mwalib_context.map(|context| context.obs_id);

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
                    (base_freq + (i as isize - base_index + 1) as f64 * freq_res).round() as _,
                );
            }
            let fine_chan_freqs = Vec1::try_from_vec(fine_chan_freqs).unwrap();

            let total_bandwidth =
                (*fine_chan_freqs.last() - *fine_chan_freqs.first()) as f64 + freq_res;

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
                autocorrelations_present: metadata.autocorrelations_present,
                dipole_gains,
                time_res,
                // TODO: Where does this live in a uvfits?
                array_longitude_rad: None,
                array_latitude_rad: None,
                // TODO - populate properly. The values don't matter until we want to
                // use coarse channel information.
                coarse_chan_nums: vec![1],
                coarse_chan_freqs: vec![150e6],
                coarse_chan_width: 40e3 * 32.0,
                num_fine_chans_per_coarse_chan: metadata.num_fine_freq_chans,
                total_bandwidth,
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
            })
        }
        inner(
            uvfits.as_ref(),
            metafits.as_ref().map(|f| f.as_ref()),
            dipole_delays,
        )
    }
}

impl InputData for UvfitsReader {
    fn get_obs_context(&self) -> &ObsContext {
        &self.obs_context
    }

    fn get_input_data_type(&self) -> VisInputType {
        VisInputType::Uvfits
    }

    fn read_crosses_and_autos(
        &self,
        _cross_data_array: ArrayViewMut2<Jones<f32>>,
        _cross_weights_array: ArrayViewMut2<f32>,
        _auto_data_array: ArrayViewMut2<Jones<f32>>,
        _auto_weights_array: ArrayViewMut2<f32>,
        _timestep: usize,
        _tile_to_unflagged_baseline_map: &HashMap<(usize, usize), usize>,
        _flagged_tiles: &[usize],
        _flagged_fine_chans: &HashSet<usize>,
    ) -> Result<(), ReadInputDataError> {
        todo!()
    }

    fn read_crosses(
        &self,
        mut data_array: ArrayViewMut2<Jones<f32>>,
        mut weights_array: ArrayViewMut2<f32>,
        timestep_index: usize,
        tile_to_unflagged_baseline_map: &HashMap<(usize, usize), usize>,
        flagged_fine_chans: &HashSet<usize>,
    ) -> Result<(), ReadInputDataError> {
        let row_range_start = timestep_index * self.step;
        let row_range_end = (timestep_index + 1) * self.step;

        let mut uvfits = fits_open!(&self.uvfits).map_err(UvfitsReadError::from)?;
        fits_open_hdu!(&mut uvfits, 0).map_err(UvfitsReadError::from)?;
        let mut group_params: Vec<f32> = vec![0.0; self.metadata.pcount];
        let mut vis: Vec<f32> = vec![
            0.0;
            self.metadata.num_fine_freq_chans
                * self.metadata.num_pols
                * self.metadata.floats_per_pol
        ];
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
                // TODO: Handle the errors nicely; the error messages aren't helpful
                // right now.
                fits_check_status(status).map_err(UvfitsReadError::from)?;

                group_params[(self.metadata.indices.baseline - 1) as usize]
            };

            let (ant1, ant2) = decode_uvfits_baseline(uvfits_bl.round() as usize);
            if let Some(i_baseline) = tile_to_unflagged_baseline_map
                .get(&(ant1 - 1, ant2 - 1))
                .cloned()
            {
                unsafe {
                    // ffgpve = fits_read_img_flt
                    fitsio_sys::ffgpve(
                        uvfits.as_raw(),  /* I - FITS file pointer                       */
                        1 + row as i64,   /* I - group to read (1 = 1st group)           */
                        1,                /* I - first vector element to read (1 = 1st)  */
                        vis.len() as i64, /* I - number of values to read                */
                        0.0,              /* I - value for undefined pixels              */
                        vis.as_mut_ptr(), /* O - array of values that are returned       */
                        &mut 0,           /* O - set to 1 if any values are null; else 0 */
                        &mut status,      /* IO - error status                           */
                    );
                }
                fits_check_status(status).map_err(UvfitsReadError::from)?;

                let vis_array = ArrayView3::from_shape(
                    (
                        self.metadata.num_fine_freq_chans,
                        self.metadata.num_pols,
                        self.metadata.floats_per_pol,
                    ),
                    vis.as_slice(),
                )
                .unwrap();
                // Put the data and weights into the shared arrays outside this
                // scope. Before we can do this, we need to remove any
                // globally-flagged fine channels.
                for (i_chan, data_pol_axis) in vis_array
                    .outer_iter()
                    .enumerate()
                    .filter(|(i_chan, _)| !flagged_fine_chans.contains(i_chan))
                {
                    // Skip boundary checks to improve performance.
                    // TODO: How much does this actually help?
                    unsafe {
                        // These are references to the visibilities and weights
                        // in the output arrays.
                        let data_array_elem = data_array.uget_mut((i_baseline, i_chan));
                        let weight_elem = weights_array.uget_mut((i_baseline, i_chan));

                        // These are the components of the input data's
                        // visibilities.
                        let data_xx = data_pol_axis.index_axis(Axis(0), 0);
                        let data_yy = data_pol_axis.index_axis(Axis(0), 1);
                        let data_xy = data_pol_axis.index_axis(Axis(0), 2);
                        let data_yx = data_pol_axis.index_axis(Axis(0), 3);

                        // Write to the output weights array. We assume that
                        // each polarisation weight is equal.
                        *weight_elem = data_xx[2];

                        // Write the input data visibility components to the
                        // output data array.
                        data_array_elem[0] = c32::new(data_xx[0], data_xx[1]);
                        data_array_elem[1] = c32::new(data_xy[0], data_xy[1]);
                        data_array_elem[2] = c32::new(data_yx[0], data_yx[1]);
                        data_array_elem[3] = c32::new(data_yy[0], data_yy[1]);
                    }
                }
            }
        }

        Ok(())
    }

    fn read_autos(
        &self,
        mut data_array: ArrayViewMut2<Jones<f32>>,
        mut weights_array: ArrayViewMut2<f32>,
        timestep_index: usize,
        flagged_tiles: &[usize],
        flagged_fine_chans: &HashSet<usize>,
    ) -> Result<(), ReadInputDataError> {
        let row_range_start = timestep_index * self.step;
        let row_range_end = (timestep_index + 1) * self.step;

        let mut uvfits = fits_open!(&self.uvfits).map_err(UvfitsReadError::from)?;
        fits_open_hdu!(&mut uvfits, 0).map_err(UvfitsReadError::from)?;
        let mut group_params: Vec<f32> = vec![0.0; self.metadata.pcount];
        let mut vis: Vec<f32> = vec![
            0.0;
            self.metadata.num_fine_freq_chans
                * self.metadata.num_pols
                * self.metadata.floats_per_pol
        ];
        let mut auto_array_index = 0;
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
                // TODO: Handle the errors nicely; the error messages aren't helpful
                // right now.
                fits_check_status(status).map_err(UvfitsReadError::from)?;

                group_params[(self.metadata.indices.baseline - 1) as usize]
            };

            let (ant1, ant2) = decode_uvfits_baseline(uvfits_bl.round() as usize);
            if ant1 == ant2 && !flagged_tiles.contains(&ant1) {
                unsafe {
                    // ffgpve = fits_read_img_flt
                    fitsio_sys::ffgpve(
                        uvfits.as_raw(),  /* I - FITS file pointer                       */
                        1 + row as i64,   /* I - group to read (1 = 1st group)           */
                        1,                /* I - first vector element to read (1 = 1st)  */
                        vis.len() as i64, /* I - number of values to read                */
                        0.0,              /* I - value for undefined pixels              */
                        vis.as_mut_ptr(), /* O - array of values that are returned       */
                        &mut 0,           /* O - set to 1 if any values are null; else 0 */
                        &mut status,      /* IO - error status                           */
                    );
                }
                fits_check_status(status).map_err(UvfitsReadError::from)?;

                let vis_array = ArrayView3::from_shape(
                    (
                        self.metadata.num_fine_freq_chans,
                        self.metadata.num_pols,
                        self.metadata.floats_per_pol,
                    ),
                    vis.as_slice(),
                )
                .unwrap();
                // Put the data and weights into the shared arrays outside this
                // scope. Before we can do this, we need to remove any
                // globally-flagged fine channels.
                for (i_chan, data_pol_axis) in vis_array
                    .outer_iter()
                    .enumerate()
                    .filter(|(i_chan, _)| !flagged_fine_chans.contains(i_chan))
                {
                    // Skip boundary checks to improve performance.
                    // TODO: How much does this actually help?
                    unsafe {
                        // These are references to the visibilities and weights
                        // in the output arrays.
                        let data_array_elem = data_array.uget_mut((auto_array_index, i_chan));
                        let weight_elem = weights_array.uget_mut((auto_array_index, i_chan));

                        // These are the components of the input data's
                        // visibilities.
                        let data_xx = data_pol_axis.index_axis(Axis(0), 0);
                        let data_yy = data_pol_axis.index_axis(Axis(0), 1);
                        let data_xy = data_pol_axis.index_axis(Axis(0), 2);
                        let data_yx = data_pol_axis.index_axis(Axis(0), 3);

                        // Get the element of the output weights array, and
                        // write to it. We assume that weights are all equal for
                        // these visibilities.
                        *weight_elem = data_xx[2];

                        // Write the input data visibility components to the
                        // output data array, also multiplying by the weight.
                        data_array_elem[0] = c32::new(data_xx[0], data_xx[1]) * *weight_elem;
                        data_array_elem[1] = c32::new(data_xy[0], data_xy[1]) * *weight_elem;
                        data_array_elem[2] = c32::new(data_yx[0], data_yx[1]) * *weight_elem;
                        data_array_elem[3] = c32::new(data_yy[0], data_yy[1]) * *weight_elem;
                    }
                }
                auto_array_index += 1;
            }
        }

        Ok(())
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
    jd_frac_timestamps: Vec<f32>,

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
        let jd_zero_val_str = format!("PZERO{}", indices.date);
        let jd_zero_str: String = get_required_fits_key!(uvfits, hdu, &jd_zero_val_str)?;
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

        let mut status = 0;
        unsafe {
            let mut group_params = vec![0.0; pcount];
            for row_num in 0..num_rows {
                // ffggpe = fits_read_grppar_flt
                fitsio_sys::ffggpe(
                    uvfits.as_raw(),           /* I - FITS file pointer                       */
                    (row_num + 1) as _,        /* I - group to read (1 = 1st group)           */
                    1,                         /* I - first vector element to read (1 = 1st)  */
                    pcount as _,               /* I - number of values to read                */
                    group_params.as_mut_ptr(), /* O - array of values that are returned       */
                    &mut status,               /* IO - error status                           */
                );
                // Check the status.
                // TODO: Handle the errors nicely; the error messages aren't helpful
                // right now.
                fits_check_status(status)?;

                // Floats can't be hashed. Hash the bits!
                let uvfits_bl = group_params[indices.baseline as usize - 1] as usize;
                let (ant1, ant2) = decode_uvfits_baseline(uvfits_bl);
                if !autocorrelations_present && (ant1 == ant2) {
                    autocorrelations_present = true;
                }
                if !uvfits_baselines_set.contains(&uvfits_bl) {
                    uvfits_baselines_set.insert(uvfits_bl);
                    uvfits_baselines.push(uvfits_bl);
                }

                let timestamp = group_params[indices.date as usize - 1];
                let timestamp_bits = timestamp.to_bits();
                if !jd_frac_timestamp_set.contains(&timestamp_bits) {
                    jd_frac_timestamp_set.insert(timestamp_bits);
                    jd_frac_timestamps.push(timestamp);
                }
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
    date: u8,
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
        let ptype1: String = get_required_fits_key!(uvfits, hdu, "PTYPE1")?;
        let ptype2: String = get_required_fits_key!(uvfits, hdu, "PTYPE2")?;
        let ptype3: String = get_required_fits_key!(uvfits, hdu, "PTYPE3")?;
        let ptype4: String = get_required_fits_key!(uvfits, hdu, "PTYPE4")?;
        let ptype5: String = get_required_fits_key!(uvfits, hdu, "PTYPE5")?;

        let mut u_index = None;
        let mut v_index = None;
        let mut w_index = None;
        let mut baseline_index = None;
        let mut date_index = None;

        for (i, key) in [ptype1, ptype2, ptype3, ptype4, ptype5].iter().enumerate() {
            let ii = (i + 1) as u8;
            match key.as_ref() {
                "UU" => u_index = Some(ii),
                "VV" => v_index = Some(ii),
                "WW" => w_index = Some(ii),
                "BASELINE" => baseline_index = Some(ii),
                "DATE" => date_index = Some(ii),
                _ => (),
            }
        }

        let (u, v, w, baseline, date) =
            match (u_index, v_index, w_index, baseline_index, date_index) {
                (Some(u), Some(v), Some(w), Some(baseline), Some(date)) => {
                    (u, v, w, baseline, date)
                }
                (None, _, _, _, _) => {
                    return Err(UvfitsReadError::MissingKey {
                        key: "UU",
                        hdu: hdu.number,
                    })
                }
                (_, None, _, _, _) => {
                    return Err(UvfitsReadError::MissingKey {
                        key: "VV",
                        hdu: hdu.number,
                    })
                }
                (_, _, None, _, _) => {
                    return Err(UvfitsReadError::MissingKey {
                        key: "WW",
                        hdu: hdu.number,
                    })
                }
                (_, _, _, None, _) => {
                    return Err(UvfitsReadError::MissingKey {
                        key: "BASELINE",
                        hdu: hdu.number,
                    })
                }
                (_, _, _, _, None) => {
                    return Err(UvfitsReadError::MissingKey {
                        key: "DATE",
                        hdu: hdu.number,
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
                    hdu: hdu.number,
                })
            }
            (_, None, _) => {
                return Err(UvfitsReadError::MissingKey {
                    key: "DEC",
                    hdu: hdu.number,
                })
            }
            (_, _, None) => {
                return Err(UvfitsReadError::MissingKey {
                    key: "FREQ",
                    hdu: hdu.number,
                })
            }
        };

        Ok(Indices {
            u,
            v,
            w,
            baseline,
            date,
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
    _hdu: &fitsio::hdu::FitsHdu,
    col_name: &str,
    row: i64,
    n_elem: i64,
) -> Result<Vec<f64>, UvfitsReadError> {
    unsafe {
        // With the column name, get the column number.
        let mut status = 0;
        let mut col_num = -1;
        let keyword = std::ffi::CString::new(col_name).unwrap();
        fitsio_sys::ffgcno(
            fits_ptr.as_raw(),
            0,
            keyword.as_ptr() as *mut c_char,
            &mut col_num,
            &mut status,
        );
        // Check the status.
        // TODO: Handle the errors nicely; the error messages aren't helpful
        // right now.
        fits_check_status(status)?;

        // Now get the specified row from that column.
        let mut array: Vec<f64> = vec![0.0; n_elem as usize];
        fitsio_sys::ffgcv(
            fits_ptr.as_raw(),
            82, // TDOUBLE (fitsio.h)
            col_num,
            row + 1,
            1,
            n_elem,
            std::ptr::null_mut(),
            array.as_mut_ptr() as _,
            &mut 0,
            &mut status,
        );
        // TODO: As above.
        fits_check_status(status)?;

        Ok(array)
    }
}
