// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to generate sky-model visibilities.

mod cpu;
#[cfg(feature = "cuda")]
mod cuda;
#[cfg(test)]
mod integration_tests;
#[cfg(test)]
mod tests;

use cpu::SkyModellerCpu;
#[cfg(feature = "cuda")]
use cuda::SkyModellerCuda;

use hifitime::Epoch;
use marlu::{Jones, RADec, XyzGeodetic, UVW};
use ndarray::prelude::*;

use mwa_hyperdrive_beam::{Beam, BeamError};
use mwa_hyperdrive_common::{cfg_if, hifitime, marlu, ndarray};
use mwa_hyperdrive_srclist::{ComponentList, SourceList};

#[derive(Debug, Clone)]
pub(crate) enum ModellerInfo {
    /// The CPU is used for modelling. This always uses double-precision floats
    /// when modelling.
    Cpu,

    /// A CUDA-capable device is used for modelling. The precision depends on
    /// the compile features used.
    #[cfg(feature = "cuda")]
    Cuda {
        device_info: mwa_hyperdrive_cuda::CudaDeviceInfo,
        driver_info: mwa_hyperdrive_cuda::CudaDriverInfo,
    },
}

/// An object that simulates sky-model visibilities.
pub trait SkyModeller<'a> {
    /// Generate sky-model visibilities for a single timestep. The [UVW]
    /// coordinates used in generating the visibilities are returned.
    ///
    /// `vis_model_slice`: A mutable view into an `ndarray`. Rather than
    /// returning an array from this function, modelled visibilities are written
    /// into this array. This slice *must* have dimensions `[n1][n2]`, where
    /// `n1` is number of unflagged cross correlation baselines and `n2` is the
    /// number of unflagged frequencies.
    ///
    /// `timestamp`: The [hifitime::Epoch] struct used to determine what this
    /// timestep corresponds to.
    ///
    /// # Errors
    ///
    /// This function will return an error if there was a problem with
    /// beam-response calculation.
    fn model_timestep(
        &self,
        vis_model_slice: ArrayViewMut2<Jones<f32>>,
        timestamp: Epoch,
    ) -> Result<Vec<UVW>, BeamError>;

    /// Model only the point sources for a timestep. If other types of sources
    /// will also be modelled, it is more efficient to use `model_timestep`. The
    /// [UVW] coordinates used in generating the visibilities are returned.
    ///
    /// `vis_model_slice`: A mutable view into an `ndarray`. Rather than
    /// returning an array from this function, modelled visibilities are written
    /// into this array. This slice *must* have dimensions `[n1][n2]`, where
    /// `n1` is number of unflagged cross correlation baselines and `n2` is the
    /// number of unflagged frequencies.
    ///
    /// `timestamp`: The [hifitime::Epoch] struct used to determine what this
    /// timestep corresponds to.
    ///
    /// # Errors
    ///
    /// This function will return an error if there was a problem with
    /// beam-response calculation.
    fn model_points(
        &self,
        vis_model_slice: ArrayViewMut2<Jones<f32>>,
        timestamp: Epoch,
    ) -> Result<Vec<UVW>, BeamError>;

    /// Model only the Gaussian sources for a timestep. If other types of
    /// sources will also be modelled, it is more efficient to use
    /// `model_timestep`. The [UVW] coordinates used in generating the
    /// visibilities are returned.
    ///
    /// `vis_model_slice`: A mutable view into an `ndarray`. Rather than
    /// returning an array from this function, modelled visibilities are written
    /// into this array. This slice *must* have dimensions `[n1][n2]`, where
    /// `n1` is number of unflagged cross correlation baselines and `n2` is the
    /// number of unflagged frequencies.
    ///
    /// `timestamp`: The [hifitime::Epoch] struct used to determine what this
    /// timestep corresponds to.
    ///
    /// # Errors
    ///
    /// This function will return an error if there was a problem with
    /// beam-response calculation.
    fn model_gaussians(
        &self,
        vis_model_slice: ArrayViewMut2<Jones<f32>>,
        timestamp: Epoch,
    ) -> Result<Vec<UVW>, BeamError>;

    /// Model only the shapelet sources for a timestep. If other types of
    /// sources will also be modelled, it is more efficient to use
    /// `model_timestep`. The [UVW] coordinates used in generating the
    /// visibilities are returned.
    ///
    /// `vis_model_slice`: A mutable view into an `ndarray`. Rather than
    /// returning an array from this function, modelled visibilities are written
    /// into this array. This slice *must* have dimensions `[n1][n2]`, where
    /// `n1` is number of unflagged cross correlation baselines and `n2` is the
    /// number of unflagged frequencies.
    ///
    /// `timestamp`: The [hifitime::Epoch] struct used to determine what this
    /// timestep corresponds to.
    ///
    /// # Errors
    ///
    /// This function will return an error if there was a problem with
    /// beam-response calculation.
    fn model_shapelets(
        &self,
        vis_model_slice: ArrayViewMut2<Jones<f32>>,
        timestamp: Epoch,
    ) -> Result<Vec<UVW>, BeamError>;
}

/// Create a [SkyModeller] trait object that generates sky-model visibilities on
/// the CPU in parallel.
///
/// It is expected that the number of unflagged `XYZ`s plus the number of
/// flagged tiles is the total number of tiles in the observation. The
/// frequencies should have units of \[Hz\].
#[allow(clippy::too_many_arguments)]
pub fn new_cpu_sky_modeller<'a>(
    beam: &'a dyn Beam,
    source_list: &SourceList,
    unflagged_tile_xyzs: &'a [XyzGeodetic],
    unflagged_fine_chan_freqs: &'a [f64],
    flagged_tiles: &'a [usize],
    phase_centre: RADec,
    array_longitude_rad: f64,
    array_latitude_rad: f64,
    apply_precession: bool,
) -> Box<dyn SkyModeller<'a> + 'a> {
    Box::new(new_cpu_sky_modeller_inner(
        beam,
        source_list,
        unflagged_tile_xyzs,
        unflagged_fine_chan_freqs,
        flagged_tiles,
        phase_centre,
        array_longitude_rad,
        array_latitude_rad,
        apply_precession,
    ))
}

/// Create a [SkyModeller] trait object that generates sky-model visibilities on
/// CUDA-compatible GPU.
///
/// It is expected that the number of unflagged `XYZ`s plus the number of
/// flagged tiles is the total number of tiles in the observation. The
/// frequencies should have units of \[Hz\].
///
/// # Errors
///
/// This function will return an error if CUDA mallocs and copies can't be
/// executed, or if there was a problem in setting up a `BeamCUDA`.
///
/// # Safety
///
/// This function interfaces directly with the CUDA API. Rust errors attempt to
/// catch problems but there are no guarantees.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn new_cuda_sky_modeller<'a>(
    beam: &'a dyn Beam,
    source_list: &SourceList,
    unflagged_tile_xyzs: &'a [XyzGeodetic],
    unflagged_fine_chan_freqs: &'a [f64],
    flagged_tiles: &'a [usize],
    phase_centre: RADec,
    array_longitude_rad: f64,
    array_latitude_rad: f64,
    apply_precession: bool,
) -> Result<Box<dyn SkyModeller<'a> + 'a>, BeamError> {
    let modeller = new_cuda_sky_modeller_inner(
        beam,
        source_list,
        unflagged_tile_xyzs,
        unflagged_fine_chan_freqs,
        flagged_tiles,
        phase_centre,
        array_longitude_rad,
        array_latitude_rad,
        apply_precession,
    )?;
    Ok(Box::new(modeller))
}

/// Create a [SkyModeller] trait object that generates sky-model visibilities on
/// either the CPU or a CUDA-compatible GPU. This function conveniently uses
/// either [new_cpu_sky_modeller] or [new_cuda_sky_modeller] depending on how
/// `hyperdrive` was compiled and the `use_cpu_for_modelling` flag.
///
/// # Errors
///
/// This function will return an error if CUDA mallocs and copies can't be
/// executed, or if there was a problem in setting up a `BeamCUDA`.
///
/// # Safety
///
/// If not using the CPU modeller, this function wraps an `unsafe` call, which
/// interfaces directly with the CUDA API. Rust errors attempt to catch problems
/// but there are no guarantees.
#[allow(clippy::too_many_arguments)]
pub(crate) fn new_sky_modeller<'a>(
    #[cfg(feature = "cuda")] use_cpu_for_modelling: bool,
    beam: &'a dyn Beam,
    source_list: &SourceList,
    unflagged_tile_xyzs: &'a [XyzGeodetic],
    unflagged_fine_chan_freqs: &'a [f64],
    flagged_tiles: &'a [usize],
    phase_centre: RADec,
    array_longitude_rad: f64,
    array_latitude_rad: f64,
    apply_precession: bool,
) -> Result<Box<dyn SkyModeller<'a> + 'a>, BeamError> {
    cfg_if::cfg_if! {
        if #[cfg(feature = "cuda")] {
            if use_cpu_for_modelling {
                Ok(new_cpu_sky_modeller(beam,
                    source_list,
                    unflagged_tile_xyzs,
                    unflagged_fine_chan_freqs,
                    flagged_tiles,
                    phase_centre,
                    array_longitude_rad,
                    array_latitude_rad,
                    apply_precession,
                ))
            } else {
                unsafe {
                new_cuda_sky_modeller(
                    beam,
                    source_list,
                    unflagged_tile_xyzs,
                    unflagged_fine_chan_freqs,
                    flagged_tiles,
                    phase_centre,
                    array_longitude_rad,
                    array_latitude_rad,
                    apply_precession,
                )}
            }
        } else {
            Ok(new_cpu_sky_modeller(beam,
                source_list,
                unflagged_tile_xyzs,
                unflagged_fine_chan_freqs,
                flagged_tiles,
                phase_centre,
                array_longitude_rad,
                array_latitude_rad,
                apply_precession,
            ))
        }
    }
}

/// Create a [SkyModellerCpu] struct that generates sky-model visibilities on
/// the CPU in parallel. This function is mostly provided for testing.
///
/// It is expected that the number of unflagged `XYZ`s plus the number of
/// flagged tiles is the total number of tiles in the observation. The
/// frequencies should have units of \[Hz\].
#[allow(clippy::too_many_arguments)]
pub(super) fn new_cpu_sky_modeller_inner<'a>(
    beam: &'a dyn Beam,
    source_list: &SourceList,
    unflagged_tile_xyzs: &'a [XyzGeodetic],
    unflagged_fine_chan_freqs: &'a [f64],
    flagged_tiles: &'a [usize],
    phase_centre: RADec,
    array_longitude_rad: f64,
    array_latitude_rad: f64,
    apply_precession: bool,
) -> SkyModellerCpu<'a> {
    let components = ComponentList::new(source_list, unflagged_fine_chan_freqs, phase_centre);
    let maps = crate::math::TileBaselineMaps::new(
        unflagged_tile_xyzs.len() + flagged_tiles.len(),
        flagged_tiles,
    );

    SkyModellerCpu {
        beam,
        phase_centre,
        array_longitude: array_longitude_rad,
        array_latitude: array_latitude_rad,
        precess: apply_precession,
        unflagged_fine_chan_freqs,
        unflagged_tile_xyzs,
        unflagged_baseline_to_tile_map: maps.unflagged_cross_baseline_to_tile_map,
        components,
    }
}

/// Create a [SkyModellerCuda] struct that generates sky-model visibilities on a
/// CUDA-compatible GPU. This function is mostly provided for testing.
///
/// It is expected that the number of unflagged `XYZ`s plus the number of
/// flagged tiles is the total number of tiles in the observation. The
/// frequencies should have units of \[Hz\].
///
/// # Errors
///
/// This function will return an error if CUDA mallocs and copies can't be
/// executed, or if there was a problem in setting up a `BeamCUDA`.
///
/// # Safety
///
/// This function interfaces directly with the CUDA API. Rust errors attempt to
/// catch problems but there are no guarantees.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn new_cuda_sky_modeller_inner<'a>(
    beam: &'a dyn Beam,
    source_list: &SourceList,
    unflagged_tile_xyzs: &'a [XyzGeodetic],
    unflagged_fine_chan_freqs: &'a [f64],
    flagged_tiles: &'a [usize],
    phase_centre: RADec,
    array_longitude_rad: f64,
    array_latitude_rad: f64,
    apply_precession: bool,
) -> Result<SkyModellerCuda<'a>, BeamError> {
    let modeller = SkyModellerCuda::new(
        beam,
        source_list,
        unflagged_tile_xyzs,
        unflagged_fine_chan_freqs,
        flagged_tiles,
        phase_centre,
        array_longitude_rad,
        array_latitude_rad,
        apply_precession,
    )?;
    Ok(modeller)
}
