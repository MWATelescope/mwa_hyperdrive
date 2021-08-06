// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to handle reading from uvfits files.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::os::raw::c_char;
use std::path::{Path, PathBuf};

use log::{debug, trace, warn};
use mwa_hyperdrive_core::XyzGeodetic;
use mwalib::{
    fitsio::{errors::check_status as fits_check_status, hdu::FitsHdu, FitsFile},
    *,
};
use ndarray::prelude::*;

use super::*;
use crate::context::{FreqContext, ObsContext};
use crate::data_formats::{metafits, InputData, ReadInputDataError};
use crate::glob::get_single_match_from_glob;
use crate::time::{epoch_as_gps_seconds, jd_to_epoch};
use mwa_hyperdrive_core::{beam::Delays, c32, mwalib, Jones, RADec};

const TIMESTEP_AS_INT_FACTOR: f64 = 1e18;

pub(crate) struct Uvfits {
    /// Observation metadata.
    obs_context: ObsContext,

    /// Frequency metadata.
    freq_context: FreqContext,

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

impl Uvfits {
    /// Verify and populate metadata associated with this measurement set. TODO:
    /// Use the metafits to get dead dipole info.
    ///
    /// The measurement set is expected to be formatted in the way that
    /// cotter/Birli write measurement sets.
    pub(crate) fn new<T: AsRef<Path>>(
        uvfits: &T,
        metafits: Option<&T>,
        dipole_delays: &mut Delays,
    ) -> Result<Self, UvfitsReadError> {
        let mut mwalib = match metafits {
            None => return Err(UvfitsReadError::NoMetafits),
            // If a metafits file was provided, we _may_ use it. Get a _potential_
            // mwalib object ready.
            _ => None,
        };

        // The uvfits argument could be a glob. If the specified argument can't
        // be found as a file, treat it as a glob and expand it to find a match.
        let uvfits_pb = {
            let pb = PathBuf::from(uvfits.as_ref());
            if pb.exists() {
                pb
            } else {
                get_single_match_from_glob(uvfits.as_ref().to_str().unwrap())?
            }
        };
        debug!("Using uvfits file: {}", uvfits_pb.display());
        if !uvfits_pb.exists() {
            return Err(UvfitsReadError::BadFile(uvfits_pb));
        }

        // Get the tile names and XYZ positions.
        let mut uvfits = fits_open!(&uvfits_pb)?;
        let hdu = fits_open_hdu!(&mut uvfits, 1)?;

        let tile_names: Vec<String> = get_fits_col!(&mut uvfits, &hdu, "ANNAME")?;
        let total_num_tiles = tile_names.len();
        let tile_xyzs = {
            let mut tile_xyzs: Vec<XyzGeodetic> = Vec::with_capacity(total_num_tiles);
            for i in 0..total_num_tiles {
                let fits_xyz = read_cell_array(&mut uvfits, &hdu, "STABXYZ", i as _, 3)?;
                tile_xyzs.push(XyzGeodetic {
                    x: fits_xyz[0],
                    y: fits_xyz[1],
                    z: fits_xyz[2],
                });
            }
            tile_xyzs
        };

        let hdu = fits_open_hdu!(&mut uvfits, 0)?;
        let metadata = UvfitsMetadata::new(&mut uvfits, &hdu)?;
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
            return Err(UvfitsReadError::Empty(uvfits_pb));
        }

        let freq_context = get_freq_context(&mut uvfits, &hdu, &metadata)?;

        // The phase centre is described by RA and DEC if there is no SOURCE
        // table (as per the standard).
        // TODO: Check that there is no SOURCE table!
        let phase_centre = {
            let ra = get_required_fits_key!(
                &mut uvfits,
                &hdu,
                &format!("CRVAL{}", metadata.indices.ra)
            )?;
            let dec = get_required_fits_key!(
                &mut uvfits,
                &hdu,
                &format!("CRVAL{}", metadata.indices.dec)
            )?;
            RADec::new_degrees(ra, dec)
        };

        // TODO: Properly determine MWA version.
        let mwa_version = MWAVersion::CorrLegacy;

        // Get the dipole delays and the pointing centre (if possible).
        // TODO: Decide on a key that uvfits can optionally provide for dipole
        // delays. Until this available, a metafits file is always necessary.
        let pointing_centre: Option<RADec> = match metafits {
            Some(meta) => {
                // Populate `mwalib` if it isn't already populated.
                let context = metafits::populate_metafits_context(&mut mwalib, meta, mwa_version)?;
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
            // TODO: Unreachable while we require a metafits file.
            None => unreachable!(),
        };
        match &dipole_delays {
            Delays::Available(d) => debug!("Dipole delays: {:?}", d),
            Delays::NotNecessary => {
                debug!("Dipole delays weren't searched for in input data; not necessary")
            }
            Delays::None => warn!("Dipole delays not provided and not available in input data!"),
        }

        // Get the timesteps. uvfits timesteps are in the middle of their
        // respective integration periods, so no adjustment is needed here.
        let jd_frac_timesteps = get_jd_frac_timesteps(&mut uvfits, &metadata)?;
        let timesteps: Vec<Epoch> = jd_frac_timesteps
            .into_iter()
            .map(|frac| {
                let jd = metadata.jd_zero + (frac as f64) / TIMESTEP_AS_INT_FACTOR;
                jd_to_epoch(jd)
            })
            .collect();
        // TODO: Determine flagging!
        let unflagged_timestep_indices = 0..timesteps.len();
        let time_res: Option<f64> = match (timesteps.first(), timesteps.get(1)) {
            (Some(&first), Some(&second)) => {
                let time_res = (second - first).in_unit_f64(hifitime::TimeUnit::Second);
                trace!("Time resolution: {}s", time_res);
                debug!(
                    "First good GPS timestep: {:.2}",
                    // Need to remove a number from the result of .as_gpst_seconds(), as
                    // it goes from the 1900 epoch, not the expected 1980 epoch. Also we
                    // expect GPS timestamps to be "leading edge", not centroids.
                    epoch_as_gps_seconds(timesteps[unflagged_timestep_indices.start])
                        - time_res / 2.0
                );
                debug!(
                    "Last good GPS timestep:  {:.2}",
                    epoch_as_gps_seconds(timesteps[unflagged_timestep_indices.end - 1])
                        - time_res / 2.0
                );
                Some(time_res)
            }
            _ => {
                warn!("Only one timestep is present in the data; can't determine the observation's native time resolution.");
                debug!(
                    "Only GPS timestep: {:.2}",
                    epoch_as_gps_seconds(timesteps[0])
                );
                None
            }
        };

        // TODO: Determine if autocorrelations are present.
        let autocorrelations_present = false;
        // TODO: Determine tile flags.
        let tile_flags: Vec<usize> = vec![];
        let num_unflagged_tiles = total_num_tiles - tile_flags.len();
        debug!("Flagged tiles in the uvfits: {:?}", tile_flags);
        debug!("Autocorrelations present: {}", autocorrelations_present);

        // Get the dipole gains. Only available with a metafits.
        let dipole_gains: Array2<f64> = match metafits {
            Some(meta) => {
                // Populate `mwalib` if it isn't already populated.
                let context = metafits::populate_metafits_context(&mut mwalib, meta, mwa_version)?;
                metafits::get_dipole_gains(context)
            }
            None => {
                warn!("Without a metafits file, we must assume all dipoles are alive.");
                warn!("This will make beam Jones matrices inaccurate in sky-model generation.");
                Array2::from_elem((num_unflagged_tiles, 16), 1.0)
            }
        };

        // Get the obsid.
        let obsid = match metafits {
            Some(meta) => {
                // Populate `mwalib` if it isn't already populated.
                let context = metafits::populate_metafits_context(&mut mwalib, meta, mwa_version)?;
                Some(context.obs_id)
            }
            // How does a uvfits file advertise the obsid? There is an "obs.
            // name" in the "object" filed, but that's not the same thing.
            None => None,
        };

        // TODO: Get flagging right. I think that info is in an optional table.
        let fine_chan_flags_per_coarse_chan = vec![];
        let step = metadata.num_rows / timesteps.len();

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
            // TODO: Where does this live?
            array_longitude_rad: None,
            array_latitude_rad: None,
        };

        Ok(Self {
            obs_context,
            freq_context,
            uvfits: uvfits_pb,
            metadata,
            step,
        })
    }
}

impl InputData for Uvfits {
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
        let row_range_start = timestep * self.step;
        let row_range_end = (timestep + 1) * self.step;

        let mut out_weights = Array2::from_elem(data_array.dim(), 0.0);

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
            // Read in the row's group parameters. We only read the first `pcount`
            // elements, but make the vector bigger for writing later.

            let mut status = 0;
            let uvfits_bl = unsafe {
                // ffggpe = fits_read_grppar_flt
                fitsio_sys::ffggpe(
                    uvfits.as_raw(),             /* I - FITS file pointer                       */
                    1 + row as i64,              /* I - group to read (1 = 1st group)           */
                    1,                           /* I - first vector element to read (1 = 1st)  */
                    self.metadata.pcount as i64, /* I - number of values to read                */
                    group_params.as_mut_ptr(),   /* O - array of values that are returned       */
                    &mut status,                 /* IO - error status                           */
                );
                // TODO: Handle the errors nicely; the error messages aren't helpful
                // right now.
                fits_check_status(status).map_err(UvfitsReadError::from)?;

                *group_params.get_unchecked((self.metadata.indices.baseline - 1) as usize)
            };

            let (ant1, ant2) = decode_uvfits_baseline(uvfits_bl.round() as usize);
            if let Some(&bl) = tile_to_unflagged_baseline_map.get(&(ant1 - 1, ant2 - 1)) {
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
                // globally-flagged fine channels. Use an int to index unflagged
                // fine channel (outer_freq_chan_index).
                let mut outer_freq_chan_index: usize = 0;
                for (freq_chan, data_pol_axis) in vis_array.outer_iter().enumerate() {
                    if !flagged_fine_chans.contains(&freq_chan) {
                        // This is a reference to the visibilities in the output
                        // data array.
                        let data_array_elem =
                            data_array.get_mut((bl, outer_freq_chan_index)).unwrap();

                        // These are the components of the input data's
                        // visibilities.
                        let data_xx = data_pol_axis.index_axis(Axis(0), 0);
                        let data_yy = data_pol_axis.index_axis(Axis(0), 1);
                        let data_xy = data_pol_axis.index_axis(Axis(0), 2);
                        let data_yx = data_pol_axis.index_axis(Axis(0), 3);

                        // Get the element of the output weights array, and
                        // write to it. We assume that weights are all equal for
                        // these visibilities.
                        let weight_elem = out_weights.get_mut((bl, outer_freq_chan_index)).unwrap();
                        *weight_elem = data_xx[2];

                        // Write the input data visibility components to the
                        // output data array, also multiplying by the weight.
                        data_array_elem[0] = c32::new(data_xx[0], data_xx[1]) * *weight_elem;
                        data_array_elem[1] = c32::new(data_xy[0], data_xy[1]) * *weight_elem;
                        data_array_elem[2] = c32::new(data_yx[0], data_yx[1]) * *weight_elem;
                        data_array_elem[3] = c32::new(data_yy[0], data_yy[1]) * *weight_elem;

                        outer_freq_chan_index += 1;
                    }
                }
            }
        }
        Ok(out_weights)
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

        Ok(Self {
            num_rows,
            pcount,
            num_pols,
            floats_per_pol,
            num_fine_freq_chans,
            jd_zero,
            indices,
        })
    }
}

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

        Ok(Self {
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

/// Get each of the Julian date fractions out of the uvfits file's rows.
///
/// Why return a set? This prevents a some memory being uselessly allocated.
/// However, this method assumes that all timesteps are in ascending order.
fn get_jd_frac_timesteps(
    uvfits: &mut FitsFile,
    metadata: &UvfitsMetadata,
) -> Result<BTreeSet<i64>, UvfitsReadError> {
    let mut timesteps = BTreeSet::new();
    let mut status = 0;
    unsafe {
        let mut timestep = [0.0];
        for row_num in 0..metadata.num_rows {
            // ffggpd = fits_read_grppar_dbl
            fitsio_sys::ffggpd(
                uvfits.as_raw(),            /* I - FITS file pointer                       */
                (row_num + 1) as _,         /* I - group to read (1 = 1st group)           */
                metadata.indices.date as _, /* I - first vector element to read (1 = 1st)  */
                1,                          /* I - number of values to read                */
                timestep.as_mut_ptr(),      /* O - array of values that are returned       */
                &mut status,                /* IO - error status                           */
            );
            // Check the status.
            // TODO: Handle the errors nicely; the error messages aren't helpful
            // right now.
            fits_check_status(status)?;

            // Floats can't be compared nicely. Multiply by a big number and
            // round to an int.
            let timestep_as_int = (timestep[0] * TIMESTEP_AS_INT_FACTOR) as i64;
            timesteps.insert(timestep_as_int);
        }
    }
    Ok(timesteps)
}

fn get_freq_context(
    uvfits: &mut FitsFile,
    hdu: &FitsHdu,
    metadata: &UvfitsMetadata,
) -> Result<FreqContext, UvfitsReadError> {
    let freq_val_str = format!("CRVAL{}", metadata.indices.freq);
    let base_freq_str: String = get_required_fits_key!(uvfits, hdu, &freq_val_str)?;
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
        let f_str: String = get_required_fits_key!(uvfits, hdu, &freq_val_str)?;
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
    let fine_chan_width_str: String = get_required_fits_key!(uvfits, hdu, &freq_val_str)?;
    let fine_chan_width: f64 = match fine_chan_width_str.parse() {
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
        fine_chan_freqs
            .push((base_freq + (i as isize - base_index + 1) as f64 * fine_chan_width) as _);
    }

    Ok(FreqContext {
        // TODO
        coarse_chan_nums: vec![1],
        coarse_chan_freqs: vec![150e6],
        coarse_chan_width: 40e3 * 32.0,
        total_bandwidth: fine_chan_freqs[metadata.num_fine_freq_chans - 1] - fine_chan_freqs[0]
            + fine_chan_width,
        fine_chan_range: 0..metadata.num_fine_freq_chans,
        fine_chan_freqs,
        num_fine_chans_per_coarse_chan: metadata.num_fine_freq_chans,
        native_fine_chan_width: fine_chan_width,
    })
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
