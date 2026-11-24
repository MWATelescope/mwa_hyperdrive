// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code for EveryBeam beam calculations.
//!
//! EveryBeam (<https://git.astron.nl/RD/EveryBeam>) is a C++ library that
//! provides beam models for many telescopes (e.g. SKA-Low, LOFAR, MWA,
//! OSKAR-simulated arrays). Telescopes are described by a measurement set.
//!
//! EveryBeam needs the absolute time and the J2000 (RA, Dec) of each
//! direction; see [`BeamTime`]. EveryBeam can only calculate beam responses on
//! the CPU; if GPU modelling is used, EveryBeam beam responses are calculated
//! on the CPU and copied to the GPU.
//!
//! # Polarisation conventions
//!
//! EveryBeam's Jones matrices have columns in the (North, East) sky basis
//! (with respect to the J2000 celestial pole). hyperdrive's sky model uses the
//! MWA convention (X is East-West, Y is North-South; see
//! `FluxDensity::to_inst_stokes`), i.e. the sky basis is (East, North), so the
//! columns are swapped.
//!
//! The rows are left alone, so that they continue to match the basis of the
//! visibilities:
//! - With "none" (the default) or "amplitude" (scalar) normalisation, the rows
//!   are each station's own feeds, in the station's (possibly rotated) frame,
//!   as described by the measurement set (e.g. SKA-Low stations are rigidly
//!   rotated with respect to each other, and this is described by the
//!   PHASED_ARRAY table). This is appropriate for data that have not had a beam
//!   correction applied.
//! - With "full" (or "preapplied") normalisation, EveryBeam left-multiplies the
//!   response by the inverse of the response in the FIELD table's
//!   REFERENCE_DIR, so the rows are in the (North, East) sky basis (i.e. the
//!   IAU order). This is only appropriate for data that have had the beam at
//!   the phase centre corrected (e.g. by DP3's applybeam).

#[cfg(test)]
pub(crate) mod reference;
#[cfg(test)]
mod tests;

use std::{
    ffi::{c_char, CStr, CString},
    os::unix::ffi::OsStrExt,
    path::{Path, PathBuf},
    ptr::NonNull,
    sync::{Arc, Mutex},
};

use log::debug;
use marlu::{AzEl, Jones, RADec};
use ndarray::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use everybeam_sys as ffi;

use super::{
    beam_knots_on_grid_with_spacing, Beam, BeamError, BeamTime, BeamType, BEAM_KNOT_SPACING_HZ,
};

#[cfg(any(feature = "cuda", feature = "hip"))]
use super::{BeamGpu, DevicePointer, GpuFloat};

/// The size of the buffers used to receive error messages from the shim.
const ERR_LEN: usize = 1024;

/// The number of directions and frequencies given to EveryBeam per call (for
/// one station). Each call is independent and is given to a rayon thread, and
/// makes its own point-response object (which takes a lock and converts
/// coordinates for the time), so calls shouldn't be too small.
const DIRECTION_CHUNK_SIZE: usize = 64;
const FREQ_CHUNK_SIZE: usize = 16;

/// The number of beam knots beyond each end of the band (see
/// [`Beam::get_beam_knots`]), so that the spline's end conditions are applied
/// away from the channels being modelled.
const NUM_GUARD_KNOTS: usize = 2;

/// Options for EveryBeam. Any options that aren't set use EveryBeam's defaults.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EveryBeamOptions {
    /// The element response model, e.g. "default", "hamaker", "lobes",
    /// "oskar_dipole", "oskar_spherical_wave", "skala40_wave", "skalow_feko".
    pub element_response_model: Option<String>,

    /// The beam mode, e.g. "full", "array_factor", "element".
    pub beam_mode: Option<String>,

    /// The beam normalisation mode, e.g. "none", "full", "amplitude",
    /// "preapplied".
    pub beam_normalisation_mode: Option<String>,

    /// The path to element-response coefficients (telescope dependent).
    pub coeff_path: Option<PathBuf>,

    /// The data column to use for LOFAR "preapplied" beam information.
    pub data_column_name: Option<String>,

    /// The measurement set field ID to use for the beam pointing.
    pub field_id: usize,

    /// Use the subband (i.e. reference) frequency for the station
    /// beamformer, rather than each channel's frequency (LOFAR only).
    pub use_subband_frequency: bool,

    /// Interpolate the beam over frequency (MWA only).
    pub frequency_interpolation: bool,

    /// The spacing \[Hz\] of the frequencies at which EveryBeam is evaluated
    /// when modelling; the responses are splined to the fine channels (see
    /// [`Beam::get_beam_knots`]). `None` means the default
    /// ([`BEAM_KNOT_SPACING_HZ`]); 0 means every fine channel (no splining).
    #[serde(default)]
    pub knot_spacing_hz: Option<f64>,
}

/// casacore's table system is not thread safe, so telescopes are loaded (and
/// freed) one at a time.
static CASACORE_LOCK: Mutex<()> = Mutex::new(());

/// An owned handle to a telescope loaded by EveryBeam.
struct Telescope(NonNull<ffi::eb_telescope>);

// The shim only calls const methods on the telescope, and each call to
// `eb_point_responses` uses its own (non-shared) point-response object.
unsafe impl Send for Telescope {}
unsafe impl Sync for Telescope {}

impl Drop for Telescope {
    fn drop(&mut self) {
        let _lock = CASACORE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe { ffi::eb_free(self.0.as_ptr()) }
    }
}

/// A wrapper around a telescope loaded by EveryBeam that implements the
/// [`Beam`] trait.
pub(crate) struct EveryBeam {
    inner: Arc<EveryBeamInner>,
}

struct EveryBeamInner {
    telescope: Telescope,
    num_stations: usize,
    ms: PathBuf,
    options: EveryBeamOptions,
    /// If this beam is unnormalised, an "amplitude"-normalised copy, used to
    /// veto sources (see [`Beam::veto_beam`]); unnormalised responses have an
    /// arbitrary scale.
    veto: Option<EveryBeam>,
}

fn opt_cstring(s: Option<&str>) -> Result<Option<CString>, BeamError> {
    s.map(|s| CString::new(s).map_err(|_| BeamError::EveryBeam(format!("Invalid string '{s}'"))))
        .transpose()
}

fn cstring_ptr(s: &Option<CString>) -> *const c_char {
    s.as_ref().map(|s| s.as_ptr()).unwrap_or(std::ptr::null())
}

fn err_to_string(err: &[c_char]) -> String {
    unsafe { CStr::from_ptr(err.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

/// A suffix to make shared temporary files user-specific.
#[cfg(feature = "everybeam-vendored")]
fn user_suffix() -> String {
    match std::env::var("USER") {
        Ok(user) if !user.is_empty() => format!("-{user}"),
        _ => String::new(),
    }
}

/// With a vendored EveryBeam, tell EveryBeam where its data files
/// (element-response coefficients) are: `EVERYBEAM_DATADIR` if set, otherwise
/// the build-time directory if it still exists, otherwise the data embedded in
/// the binary, written to a temporary directory.
#[cfg(feature = "everybeam-vendored")]
fn set_vendored_data_dir() -> Result<(), BeamError> {
    use std::sync::OnceLock;

    static DATA_DIR: OnceLock<Result<CString, String>> = OnceLock::new();
    let dir = DATA_DIR.get_or_init(|| {
        let dir = if let Some(dir) = std::env::var_os("EVERYBEAM_DATADIR") {
            PathBuf::from(dir)
        } else if Path::new(ffi::data::BUILD_DATA_DIR).exists() {
            PathBuf::from(ffi::data::BUILD_DATA_DIR)
        } else {
            let dir = std::env::temp_dir().join(format!(
                "hyperdrive-everybeam-{}-data{}",
                ffi::data::EVERYBEAM_VERSION,
                user_suffix()
            ));
            write_embedded_data(&dir).map_err(|e| {
                format!(
                    "Couldn't write EveryBeam's data files to {}: {e}",
                    dir.display()
                )
            })?;
            dir
        };
        debug!("Using EveryBeam data directory {}", dir.display());
        CString::new(dir.as_os_str().as_bytes()).map_err(|e| e.to_string())
    });
    match dir {
        Ok(dir) => {
            unsafe { ffi::eb_set_data_dir(dir.as_ptr()) };
            Ok(())
        }
        Err(e) => Err(BeamError::EveryBeam(e.clone())),
    }
}

#[cfg(feature = "everybeam-vendored")]
fn write_embedded_data(dir: &Path) -> std::io::Result<()> {
    for (name, contents) in ffi::data::FILES {
        let path = dir.join(name);
        let up_to_date = path
            .metadata()
            .map(|m| m.len() == contents.len() as u64)
            .unwrap_or(false);
        if up_to_date {
            continue;
        }
        std::fs::create_dir_all(path.parent().expect("has a parent"))?;
        // Write atomically, in case of concurrent hyperdrive processes.
        let tmp = path.with_extension(format!("tmp{}", std::process::id()));
        std::fs::write(&tmp, contents)?;
        std::fs::rename(&tmp, &path)?;
    }
    Ok(())
}

/// Convert an EveryBeam Jones matrix with columns in the (North, East) sky
/// basis to hyperdrive's convention (columns in the (East, North) sky basis).
/// The rows are unchanged.
fn swap_columns(j: Jones<f64>) -> Jones<f64> {
    Jones::from([j[1], j[0], j[3], j[2]])
}

impl EveryBeam {
    /// Load the telescope described by the measurement set `ms`. If
    /// `num_tiles` is supplied, it is checked against the number of stations
    /// in the measurement set.
    pub(crate) fn new(
        ms: &Path,
        num_tiles: Option<usize>,
        options: EveryBeamOptions,
    ) -> Result<EveryBeam, BeamError> {
        debug!("Loading EveryBeam telescope from {}", ms.display());
        let ms_c = CString::new(ms.as_os_str().as_bytes())
            .map_err(|_| BeamError::EveryBeam(format!("Invalid path {}", ms.display())))?;
        let element_response_model = opt_cstring(options.element_response_model.as_deref())?;
        let beam_mode = opt_cstring(options.beam_mode.as_deref())?;
        let beam_normalisation_mode = opt_cstring(options.beam_normalisation_mode.as_deref())?;
        let coeff_path = opt_cstring(
            options
                .coeff_path
                .as_ref()
                .map(|p| {
                    p.to_str().ok_or_else(|| {
                        BeamError::EveryBeam(format!("Invalid path {}", p.display()))
                    })
                })
                .transpose()?,
        )?;
        let data_column_name = opt_cstring(options.data_column_name.as_deref())?;
        let eb_options = ffi::eb_options {
            element_response_model: cstring_ptr(&element_response_model),
            beam_mode: cstring_ptr(&beam_mode),
            beam_normalisation_mode: cstring_ptr(&beam_normalisation_mode),
            coeff_path: cstring_ptr(&coeff_path),
            data_column_name: cstring_ptr(&data_column_name),
            use_channel_frequency: (!options.use_subband_frequency).into(),
            frequency_interpolation: options.frequency_interpolation.into(),
        };

        #[cfg(feature = "everybeam-vendored")]
        set_vendored_data_dir()?;

        let mut err = [0 as c_char; ERR_LEN];
        let telescope = {
            let _lock = CASACORE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            unsafe { ffi::eb_load(ms_c.as_ptr(), &eb_options, err.as_mut_ptr(), ERR_LEN) }
        };
        let telescope = NonNull::new(telescope).ok_or_else(|| {
            let msg = err_to_string(&err);
            // casacore needs its "measures" data to convert coordinates.
            let hint = if ["TAI_UTC", "IERS", "leap second", "measures"]
                .iter()
                .any(|s| msg.contains(s))
            {
                "\ncasacore's measures data couldn't be found; see https://mwatelescope.github.io/mwa_hyperdrive/installation/everybeam.html"
            } else {
                ""
            };
            BeamError::EveryBeam(format!(
                "Couldn't load telescope from {}: {msg}{hint}",
                ms.display(),
            ))
        })?;
        let telescope = Telescope(telescope);
        let num_stations = unsafe { ffi::eb_num_stations(telescope.0.as_ptr()) };
        // EveryBeam's default normalisation is "none".
        let veto = match options.beam_normalisation_mode.as_deref() {
            None | Some("none") => Some(EveryBeam::new(
                ms,
                None,
                EveryBeamOptions {
                    beam_normalisation_mode: Some("amplitude".to_string()),
                    ..options.clone()
                },
            )?),
            Some(_) => None,
        };
        let beam = EveryBeam {
            inner: Arc::new(EveryBeamInner {
                telescope,
                num_stations,
                ms: ms.to_path_buf(),
                options,
                veto,
            }),
        };
        debug!("EveryBeam telescope has {num_stations} stations");

        if let Some(num_tiles) = num_tiles {
            if num_tiles != num_stations {
                return Err(BeamError::StationCountMismatch {
                    ms: num_stations,
                    tiles: num_tiles,
                });
            }
        }

        Ok(beam)
    }
}

impl EveryBeamInner {
    /// Calculate beam responses for (RA, Dec) directions, at all specified
    /// frequencies and stations. The results are written into `results`, which
    /// must have dimensions (stations, freqs, directions). The responses are in
    /// hyperdrive's polarisation convention.
    pub(crate) fn calc_jones_radec(
        &self,
        radecs: &[RADec],
        freqs_hz: &[f64],
        stations: &[usize],
        time: BeamTime,
        mut results: ArrayViewMut3<Jones<f64>>,
    ) -> Result<(), BeamError> {
        assert_eq!(
            results.dim(),
            (stations.len(), freqs_hz.len(), radecs.len())
        );
        if let Some(&s) = stations.iter().find(|&&s| s >= self.num_stations) {
            return Err(BeamError::BadTileIndex {
                got: s,
                num_tiles: self.num_stations,
            });
        }
        if radecs.is_empty() || freqs_hz.is_empty() || stations.is_empty() {
            return Ok(());
        }

        let (ras, decs): (Vec<f64>, Vec<f64>) = radecs.iter().map(|r| (r.ra, r.dec)).unzip();
        let time_mjd_s = time.epoch.to_mjd_utc_seconds();
        let num_directions = radecs.len();
        let station_stride = freqs_hz.len() * num_directions;
        let freq_stride = num_directions;

        // Each chunk of directions writes to disjoint parts of `results`.
        struct SendPtr(*mut f64);
        unsafe impl Send for SendPtr {}
        unsafe impl Sync for SendPtr {}
        let out = SendPtr(
            results
                .as_slice_mut()
                .expect("is contiguous")
                .as_mut_ptr()
                .cast::<f64>(),
        );

        // Work is split into blocks of (directions, one station,
        // frequencies), each given to EveryBeam in one call (which makes its
        // own point-response object) and run in parallel. Blocks write to
        // disjoint parts of `results`. (With few directions, e.g. one
        // source, parallelising over directions alone used one thread.)
        let num_dir_chunks = num_directions.div_ceil(DIRECTION_CHUNK_SIZE);
        let num_freq_chunks = freqs_hz.len().div_ceil(FREQ_CHUNK_SIZE);
        let num_blocks = num_dir_chunks * stations.len() * num_freq_chunks;
        (0..num_blocks).into_par_iter().try_for_each(|i_block| {
            let out = &out;
            let i_dir_chunk = i_block % num_dir_chunks;
            let i_freq_chunk = (i_block / num_dir_chunks) % num_freq_chunks;
            let i_station = i_block / (num_dir_chunks * num_freq_chunks);
            let d0 = i_dir_chunk * DIRECTION_CHUNK_SIZE;
            let d1 = (d0 + DIRECTION_CHUNK_SIZE).min(num_directions);
            let f0 = i_freq_chunk * FREQ_CHUNK_SIZE;
            let f1 = (f0 + FREQ_CHUNK_SIZE).min(freqs_hz.len());
            let mut err = [0 as c_char; ERR_LEN];
            let status = unsafe {
                ffi::eb_point_responses(
                    self.telescope.0.as_ptr(),
                    time_mjd_s,
                    self.options.field_id,
                    ras[d0..d1].as_ptr(),
                    decs[d0..d1].as_ptr(),
                    d1 - d0,
                    freqs_hz[f0..f1].as_ptr(),
                    f1 - f0,
                    stations[i_station..].as_ptr(),
                    1,
                    out.0
                        .add(8 * (i_station * station_stride + f0 * freq_stride + d0)),
                    station_stride,
                    freq_stride,
                    err.as_mut_ptr(),
                    ERR_LEN,
                )
            };
            if status == 0 {
                Ok(())
            } else {
                Err(BeamError::EveryBeam(err_to_string(&err)))
            }
        })?;

        results.mapv_inplace(swap_columns);
        Ok(())
    }

    /// Like [`EveryBeam::calc_jones_radec`], but with [`AzEl`] directions.
    /// `time.lst_rad` and `latitude_rad` are used to convert the directions to
    /// (RA, Dec).
    fn calc_jones_azel(
        &self,
        azels: &[AzEl],
        freqs_hz: &[f64],
        stations: &[usize],
        latitude_rad: f64,
        time: Option<BeamTime>,
        results: ArrayViewMut3<Jones<f64>>,
    ) -> Result<(), BeamError> {
        let time = time.ok_or(BeamError::NeedsTime)?;
        let radecs: Vec<RADec> = azels
            .iter()
            .map(|azel| azel.to_hadec(latitude_rad).to_radec(time.lst_rad))
            .collect();
        self.calc_jones_radec(&radecs, freqs_hz, stations, time, results)
    }
}

impl Beam for EveryBeam {
    fn get_beam_type(&self) -> BeamType {
        BeamType::EveryBeam
    }

    fn get_num_tiles(&self) -> usize {
        self.inner.num_stations
    }

    fn get_dipole_delays(&self) -> Option<ArcArray<u32, Dim<[usize; 2]>>> {
        None
    }

    fn get_ideal_dipole_delays(&self) -> Option<[u32; 16]> {
        None
    }

    fn get_dipole_gains(&self) -> Option<ArcArray<f64, Dim<[usize; 2]>>> {
        None
    }

    fn get_beam_file(&self) -> Option<&Path> {
        Some(&self.inner.ms)
    }

    fn veto_beam(&self) -> Option<&dyn Beam> {
        self.inner.veto.as_ref().map(|b| b as &dyn Beam)
    }

    fn calc_jones(
        &self,
        azel: AzEl,
        freq_hz: f64,
        tile_index: Option<usize>,
        latitude_rad: f64,
        time: Option<BeamTime>,
    ) -> Result<Jones<f64>, BeamError> {
        let mut result = [Jones::default()];
        self.calc_jones_array_inner(
            &[azel],
            freq_hz,
            tile_index,
            latitude_rad,
            time,
            &mut result,
        )?;
        Ok(result[0])
    }

    fn calc_jones_array(
        &self,
        azels: &[AzEl],
        freq_hz: f64,
        tile_index: Option<usize>,
        latitude_rad: f64,
        time: Option<BeamTime>,
    ) -> Result<Vec<Jones<f64>>, BeamError> {
        let mut results = vec![Jones::default(); azels.len()];
        self.calc_jones_array_inner(azels, freq_hz, tile_index, latitude_rad, time, &mut results)?;
        Ok(results)
    }

    fn calc_jones_array_inner(
        &self,
        azels: &[AzEl],
        freq_hz: f64,
        tile_index: Option<usize>,
        latitude_rad: f64,
        time: Option<BeamTime>,
        results: &mut [Jones<f64>],
    ) -> Result<(), BeamError> {
        // EveryBeam stations are not "ideal"; if no tile is specified, use the
        // first station.
        let station = tile_index.unwrap_or(0);
        let results = ArrayViewMut3::from_shape((1, 1, azels.len()), results)
            .expect("results has the same length as azels");
        self.inner
            .calc_jones_azel(azels, &[freq_hz], &[station], latitude_rad, time, results)
    }

    fn calc_jones_tiles_freqs(
        &self,
        azels: &[AzEl],
        freqs_hz: &[f64],
        tile_indices: &[usize],
        latitude_rad: f64,
        time: Option<BeamTime>,
        results: ArrayViewMut3<Jones<f64>>,
    ) -> Result<(), BeamError> {
        self.inner
            .calc_jones_azel(azels, freqs_hz, tile_indices, latitude_rad, time, results)
    }

    fn find_closest_freq(&self, desired_freq_hz: f64) -> f64 {
        desired_freq_hz
    }

    fn get_beam_knots(&self, freqs_hz: &[f64]) -> Vec<f64> {
        // EveryBeam is defined at any frequency, but evaluating it at every
        // fine channel is expensive (and the modellers' beam-response arrays
        // scale with the number of knots), so evaluate it on a grid and spline
        // to the fine channels, as for the analytic beams. A spacing of 0
        // evaluates every fine channel.
        match self
            .inner
            .options
            .knot_spacing_hz
            .unwrap_or(BEAM_KNOT_SPACING_HZ)
        {
            s if s > 0.0 => beam_knots_on_grid_with_spacing(freqs_hz, NUM_GUARD_KNOTS, s),
            _ => {
                let mut knots = freqs_hz.to_vec();
                knots.sort_unstable_by(f64::total_cmp);
                knots.dedup();
                knots
            }
        }
    }

    fn empty_coeff_cache(&self) {}

    fn get_unique_tiles(
        &self,
        total_num_tiles: usize,
        flagged_tiles: &std::collections::HashSet<usize>,
    ) -> (Vec<usize>, Vec<usize>) {
        // Every station is assumed to have a unique response.
        let mut unique_tiles = Vec::with_capacity(total_num_tiles);
        let mut map = Vec::with_capacity(total_num_tiles);
        for i_tile in 0..total_num_tiles {
            if flagged_tiles.contains(&i_tile) {
                map.push(0);
            } else {
                map.push(unique_tiles.len());
                unique_tiles.push(i_tile);
            }
        }
        (unique_tiles, map)
    }

    #[cfg(any(feature = "cuda", feature = "hip"))]
    fn prepare_gpu_beam(&self, freqs_hz: &[u32]) -> Result<Box<dyn BeamGpu>, BeamError> {
        // All stations and frequencies are treated as unique.
        let tile_map: Vec<i32> = (0..self.inner.num_stations as i32).collect();
        let freq_map: Vec<i32> = (0..freqs_hz.len() as i32).collect();
        Ok(Box::new(EveryBeamGpu {
            beam: Arc::clone(&self.inner),
            freqs_hz: freqs_hz.iter().map(|&f| f as f64).collect(),
            tile_map: DevicePointer::copy_to_device(&tile_map)?,
            freq_map: DevicePointer::copy_to_device(&freq_map)?,
        }))
    }
}

/// An EveryBeam "GPU" beam. EveryBeam can't run on a GPU, so beam responses
/// are calculated on the CPU and copied to the device.
#[cfg(any(feature = "cuda", feature = "hip"))]
struct EveryBeamGpu {
    beam: Arc<EveryBeamInner>,
    freqs_hz: Vec<f64>,
    tile_map: DevicePointer<i32>,
    freq_map: DevicePointer<i32>,
}

#[cfg(any(feature = "cuda", feature = "hip"))]
impl BeamGpu for EveryBeamGpu {
    unsafe fn calc_jones_pair(
        &self,
        az_rad: &[GpuFloat],
        za_rad: &[GpuFloat],
        latitude_rad: f64,
        time: Option<BeamTime>,
        d_jones: *mut std::ffi::c_void,
    ) -> Result<(), BeamError> {
        #[cfg(feature = "cuda")]
        use cuda_runtime_sys::{
            cudaMemcpy as gpuMemcpy,
            cudaMemcpyKind::cudaMemcpyHostToDevice as gpuMemcpyHostToDevice,
        };
        #[cfg(feature = "hip")]
        use hip_sys::hiprt::{
            hipMemcpy as gpuMemcpy, hipMemcpyKind::hipMemcpyHostToDevice as gpuMemcpyHostToDevice,
        };

        let beam = &self.beam;
        // `GpuFloat` is f32 with "gpu-single", otherwise f64.
        #[allow(clippy::unnecessary_cast)]
        let azels: Vec<AzEl> = az_rad
            .iter()
            .zip(za_rad.iter())
            .map(|(&az, &za)| {
                AzEl::from_radians(az as f64, std::f64::consts::FRAC_PI_2 - za as f64)
            })
            .collect();
        let stations: Vec<usize> = (0..beam.num_stations).collect();
        let mut results = Array3::default((stations.len(), self.freqs_hz.len(), azels.len()));
        beam.calc_jones_azel(
            &azels,
            &self.freqs_hz,
            &stations,
            latitude_rad,
            time,
            results.view_mut(),
        )?;

        // The results are ordered tile, frequency, direction, slowest to
        // fastest, matching what the GPU code expects.
        let results: Vec<Jones<GpuFloat>> = results
            .into_iter()
            .map(|j| {
                Jones::from([
                    num_complex::Complex::new(j[0].re as GpuFloat, j[0].im as GpuFloat),
                    num_complex::Complex::new(j[1].re as GpuFloat, j[1].im as GpuFloat),
                    num_complex::Complex::new(j[2].re as GpuFloat, j[2].im as GpuFloat),
                    num_complex::Complex::new(j[3].re as GpuFloat, j[3].im as GpuFloat),
                ])
            })
            .collect();
        gpuMemcpy(
            d_jones,
            results.as_ptr().cast(),
            results.len() * std::mem::size_of::<Jones<GpuFloat>>(),
            gpuMemcpyHostToDevice,
        );
        crate::gpu::check_for_errors(crate::gpu::GpuCall::CopyToDevice)?;
        Ok(())
    }

    fn get_beam_type(&self) -> BeamType {
        BeamType::EveryBeam
    }

    fn get_tile_map(&self) -> *const i32 {
        self.tile_map.get()
    }

    fn get_freq_map(&self) -> *const i32 {
        self.freq_map.get()
    }

    fn get_num_unique_tiles(&self) -> i32 {
        self.beam.num_stations as i32
    }

    fn get_num_unique_freqs(&self) -> i32 {
        self.freqs_hz.len() as i32
    }
}
