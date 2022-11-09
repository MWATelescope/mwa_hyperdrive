// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to generate sky-model visibilities with CUDA.

use std::collections::HashSet;

use hifitime::{Duration, Epoch};
use log::debug;
use marlu::{
    cuda::{cuda_status_to_error, DevicePointer, ERROR_STR_LENGTH as CUDA_ERROR_STR_LENGTH},
    cuda_runtime_sys,
    pos::xyz::xyzs_to_cross_uvws_parallel,
    precession::{get_lmst, precess_time},
    Jones, LmnRime, RADec, XyzGeodetic, UVW,
};
use ndarray::prelude::*;
use rayon::prelude::*;

use crate::{
    beam::{Beam, BeamCUDA, BeamError},
    cuda::{self, CudaFloat, CudaJones},
    shapelets,
    srclist::{
        get_instrumental_flux_densities, ComponentType, FluxDensityType, ShapeletCoeff, SourceList,
    },
};

/// The first axis of `*_list_fds` is unflagged fine channel frequency, the
/// second is the source component. The length of `hadecs`, `lmns`,
/// `*_list_fds`'s second axis are the same.
pub(crate) struct SkyModellerCuda<'a> {
    cuda_beam: Box<dyn BeamCUDA>,

    /// The phase centre used for all modelling.
    phase_centre: RADec,
    /// The longitude of the array we're using \[radians\].
    array_longitude: f64,
    /// The *unprecessed* latitude of the array we're using \[radians\]. If we
    /// are precessing, this latitude isn't used when calculating [`AzEl`]s.
    array_latitude: f64,
    /// The UT1 - UTC offset. If this is 0, effectively UT1 == UTC, which is a
    /// wrong assumption by up to 0.9s. We assume the this value does not change
    /// over the timestamps given to this `SkyModellerCuda`.
    dut1: Duration,
    /// Shift baselines, LSTs and array latitudes back to J2000.
    precess: bool,

    freqs: Vec<CudaFloat>,

    /// The [XyzGeodetic] positions of each of the unflagged tiles.
    unflagged_tile_xyzs: &'a [XyzGeodetic],
    /// The number of cross-correlation baselines given the number of unflagged
    /// tiles.
    num_baselines: usize,

    /// A simple map from an absolute tile index into an unflagged tile index.
    /// This is important because CUDA will use tile indices from 0 to the
    /// length of `unflagged_tile_xyzs`, but the beam code has dipole delays and
    /// dipole gains available for *all* tiles. So if tile 32 is flagged, any
    /// CUDA thread with a tile index of 32 would naively get the flagged beam
    /// info. This map would make tile index go to the next unflagged tile,
    /// perhaps 33.
    tile_index_to_unflagged_tile_index_map: DevicePointer<i32>,

    sbf_l: i32,
    sbf_n: i32,
    sbf_c: CudaFloat,
    sbf_dx: CudaFloat,

    d_vis: DevicePointer<f32>,
    d_freqs: DevicePointer<CudaFloat>,
    d_shapelet_basis_values: DevicePointer<CudaFloat>,

    point_power_law_radecs: Vec<RADec>,
    point_power_law_lmns: DevicePointer<cuda::LmnRime>,
    /// Instrumental flux densities calculated at 150 MHz.
    point_power_law_fds: DevicePointer<CudaJones>,
    /// Spectral indices.
    point_power_law_sis: DevicePointer<CudaFloat>,

    point_curved_power_law_radecs: Vec<RADec>,
    point_curved_power_law_lmns: DevicePointer<cuda::LmnRime>,
    point_curved_power_law_fds: DevicePointer<CudaJones>,
    point_curved_power_law_sis: DevicePointer<CudaFloat>,
    point_curved_power_law_qs: DevicePointer<CudaFloat>,

    point_list_radecs: Vec<RADec>,
    point_list_lmns: DevicePointer<cuda::LmnRime>,
    /// Instrumental (i.e. XX, XY, YX, XX).
    point_list_fds: DevicePointer<CudaJones>,

    gaussian_power_law_radecs: Vec<RADec>,
    gaussian_power_law_lmns: DevicePointer<cuda::LmnRime>,
    /// Instrumental flux densities calculated at 150 MHz.
    gaussian_power_law_fds: DevicePointer<CudaJones>,
    /// Spectral indices.
    gaussian_power_law_sis: DevicePointer<CudaFloat>,
    gaussian_power_law_gps: DevicePointer<cuda::GaussianParams>,

    gaussian_curved_power_law_radecs: Vec<RADec>,
    gaussian_curved_power_law_lmns: DevicePointer<cuda::LmnRime>,
    gaussian_curved_power_law_fds: DevicePointer<CudaJones>,
    gaussian_curved_power_law_sis: DevicePointer<CudaFloat>,
    gaussian_curved_power_law_qs: DevicePointer<CudaFloat>,
    gaussian_curved_power_law_gps: DevicePointer<cuda::GaussianParams>,

    gaussian_list_radecs: Vec<RADec>,
    gaussian_list_lmns: DevicePointer<cuda::LmnRime>,
    /// Instrumental (i.e. XX, XY, YX, XX).
    gaussian_list_fds: DevicePointer<CudaJones>,
    gaussian_list_gps: DevicePointer<cuda::GaussianParams>,

    shapelet_power_law_radecs: Vec<RADec>,
    shapelet_power_law_lmns: DevicePointer<cuda::LmnRime>,
    /// Instrumental flux densities calculated at 150 MHz.
    shapelet_power_law_fds: DevicePointer<CudaJones>,
    /// Spectral indices.
    shapelet_power_law_sis: DevicePointer<CudaFloat>,
    shapelet_power_law_gps: DevicePointer<cuda::GaussianParams>,
    shapelet_power_law_coeffs: DevicePointer<cuda::ShapeletCoeff>,
    shapelet_power_law_coeff_lens: DevicePointer<usize>,

    shapelet_curved_power_law_radecs: Vec<RADec>,
    shapelet_curved_power_law_lmns: DevicePointer<cuda::LmnRime>,
    shapelet_curved_power_law_fds: DevicePointer<CudaJones>,
    shapelet_curved_power_law_sis: DevicePointer<CudaFloat>,
    shapelet_curved_power_law_qs: DevicePointer<CudaFloat>,
    shapelet_curved_power_law_gps: DevicePointer<cuda::GaussianParams>,
    shapelet_curved_power_law_coeffs: DevicePointer<cuda::ShapeletCoeff>,
    shapelet_curved_power_law_coeff_lens: DevicePointer<usize>,

    shapelet_list_radecs: Vec<RADec>,
    shapelet_list_lmns: DevicePointer<cuda::LmnRime>,
    /// Instrumental (i.e. XX, XY, YX, XX).
    shapelet_list_fds: DevicePointer<CudaJones>,
    shapelet_list_gps: DevicePointer<cuda::GaussianParams>,
    shapelet_list_coeffs: DevicePointer<cuda::ShapeletCoeff>,
    shapelet_list_coeff_lens: DevicePointer<usize>,
}

impl<'a> SkyModellerCuda<'a> {
    /// Given a source list, split the components into each component type (e.g.
    /// points, shapelets) and by each flux density type (e.g. list, power law),
    /// then copy them to a GPU ready for modelling. Where possible, list flux
    /// density types should be converted to power laws before calling this
    /// function, because using power laws is more efficient and probably more
    /// accurate.
    ///
    /// # Safety
    ///
    /// This function interfaces directly with the CUDA API. Rust errors attempt
    /// to catch problems but there are no guarantees.
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn new(
        beam: &dyn Beam,
        source_list: &SourceList,
        unflagged_tile_xyzs: &'a [XyzGeodetic],
        unflagged_fine_chan_freqs: &[f64],
        flagged_tiles: &HashSet<usize>,
        phase_centre: RADec,
        array_longitude_rad: f64,
        array_latitude_rad: f64,
        dut1: Duration,
        apply_precession: bool,
    ) -> Result<SkyModellerCuda<'a>, BeamError> {
        let mut point_power_law_radecs: Vec<RADec> = vec![];
        let mut point_power_law_lmns: Vec<cuda::LmnRime> = vec![];
        let mut point_power_law_fds: Vec<_> = vec![];
        let mut point_power_law_sis: Vec<_> = vec![];

        let mut point_curved_power_law_radecs: Vec<RADec> = vec![];
        let mut point_curved_power_law_lmns: Vec<cuda::LmnRime> = vec![];
        let mut point_curved_power_law_fds: Vec<_> = vec![];
        let mut point_curved_power_law_sis: Vec<_> = vec![];
        let mut point_curved_power_law_qs: Vec<_> = vec![];

        let mut point_list_radecs: Vec<RADec> = vec![];
        let mut point_list_lmns: Vec<cuda::LmnRime> = vec![];
        let mut point_list_fds: Vec<FluxDensityType> = vec![];

        let mut gaussian_power_law_radecs: Vec<RADec> = vec![];
        let mut gaussian_power_law_lmns: Vec<cuda::LmnRime> = vec![];
        let mut gaussian_power_law_fds: Vec<_> = vec![];
        let mut gaussian_power_law_sis: Vec<_> = vec![];
        let mut gaussian_power_law_gps: Vec<cuda::GaussianParams> = vec![];

        let mut gaussian_curved_power_law_radecs: Vec<RADec> = vec![];
        let mut gaussian_curved_power_law_lmns: Vec<cuda::LmnRime> = vec![];
        let mut gaussian_curved_power_law_fds: Vec<_> = vec![];
        let mut gaussian_curved_power_law_sis: Vec<_> = vec![];
        let mut gaussian_curved_power_law_qs: Vec<_> = vec![];
        let mut gaussian_curved_power_law_gps: Vec<cuda::GaussianParams> = vec![];

        let mut gaussian_list_radecs: Vec<RADec> = vec![];
        let mut gaussian_list_lmns: Vec<cuda::LmnRime> = vec![];
        let mut gaussian_list_fds: Vec<FluxDensityType> = vec![];
        let mut gaussian_list_gps: Vec<cuda::GaussianParams> = vec![];

        let mut shapelet_power_law_radecs: Vec<RADec> = vec![];
        let mut shapelet_power_law_lmns: Vec<cuda::LmnRime> = vec![];
        let mut shapelet_power_law_fds: Vec<_> = vec![];
        let mut shapelet_power_law_sis: Vec<_> = vec![];
        let mut shapelet_power_law_gps: Vec<cuda::GaussianParams> = vec![];
        let mut shapelet_power_law_coeffs: Vec<Vec<ShapeletCoeff>> = vec![];

        let mut shapelet_curved_power_law_radecs: Vec<RADec> = vec![];
        let mut shapelet_curved_power_law_lmns: Vec<cuda::LmnRime> = vec![];
        let mut shapelet_curved_power_law_fds: Vec<_> = vec![];
        let mut shapelet_curved_power_law_sis: Vec<_> = vec![];
        let mut shapelet_curved_power_law_qs: Vec<_> = vec![];
        let mut shapelet_curved_power_law_gps: Vec<cuda::GaussianParams> = vec![];
        let mut shapelet_curved_power_law_coeffs: Vec<Vec<ShapeletCoeff>> = vec![];

        let mut shapelet_list_radecs: Vec<RADec> = vec![];
        let mut shapelet_list_lmns: Vec<cuda::LmnRime> = vec![];
        let mut shapelet_list_fds: Vec<FluxDensityType> = vec![];
        let mut shapelet_list_gps: Vec<cuda::GaussianParams> = vec![];
        let mut shapelet_list_coeffs: Vec<Vec<ShapeletCoeff>> = vec![];

        cfg_if::cfg_if! {
            if #[cfg(feature = "cuda-single")] {
                let jones_to_cuda_jones = |j: Jones<f64>| -> cuda::JonesF32 {
                    cuda::JonesF32 {
                        xx_re: j[0].re as f32,
                        xx_im: j[0].im as f32,
                        xy_re: j[1].re as f32,
                        xy_im: j[1].im as f32,
                        yx_re: j[2].re as f32,
                        yx_im: j[2].im as f32,
                        yy_re: j[3].re as f32,
                        yy_im: j[3].im as f32,
                    }
                };
            } else {
                let jones_to_cuda_jones = |j: Jones<f64>| -> cuda::JonesF64 {
                    cuda::JonesF64 {
                        xx_re: j[0].re,
                        xx_im: j[0].im,
                        xy_re: j[1].re,
                        xy_im: j[1].im,
                        yx_re: j[2].re,
                        yx_im: j[2].im,
                        yy_re: j[3].re,
                        yy_im: j[3].im,
                    }
                };
            }
        }

        // Reverse the source list; if the source list has been sorted
        // (brightest sources first), reversing makes the dimmest sources get
        // used first. This is good because floating-point precision errors are
        // smaller when similar values are accumulated. Accumulating into a
        // float starting from the brightest component means that the
        // floating-point precision errors are greater as we work through the
        // source list.
        for comp in source_list
            .iter()
            .rev()
            .flat_map(|(_, src)| &src.components)
        {
            let radec = comp.radec;
            let LmnRime { l, m, n } = comp.radec.to_lmn(phase_centre).prepare_for_rime();
            let lmn = cuda::LmnRime {
                l: l as CudaFloat,
                m: m as CudaFloat,
                n: n as CudaFloat,
            };
            match &comp.comp_type {
                ComponentType::Point => match comp.flux_type {
                    FluxDensityType::PowerLaw { si, .. } => {
                        point_power_law_radecs.push(radec);
                        point_power_law_lmns.push(lmn);
                        let fd_at_150mhz = comp.estimate_at_freq(cuda::POWER_LAW_FD_REF_FREQ as _);
                        let inst_fd: Jones<f64> = fd_at_150mhz.to_inst_stokes();
                        let cuda_inst_fd = jones_to_cuda_jones(inst_fd);
                        point_power_law_fds.push(cuda_inst_fd);
                        point_power_law_sis.push(si as CudaFloat);
                    }

                    FluxDensityType::CurvedPowerLaw { si, q, .. } => {
                        point_curved_power_law_radecs.push(radec);
                        point_curved_power_law_lmns.push(lmn);
                        let fd_at_150mhz = comp.estimate_at_freq(cuda::POWER_LAW_FD_REF_FREQ as _);
                        let inst_fd: Jones<f64> = fd_at_150mhz.to_inst_stokes();
                        let cuda_inst_fd = jones_to_cuda_jones(inst_fd);
                        point_curved_power_law_fds.push(cuda_inst_fd);
                        point_curved_power_law_qs.push(q as CudaFloat);
                        point_curved_power_law_sis.push(si as CudaFloat);
                    }

                    FluxDensityType::List { .. } => {
                        point_list_radecs.push(radec);
                        point_list_lmns.push(lmn);
                        point_list_fds.push(comp.flux_type.clone());
                    }
                },

                ComponentType::Gaussian { maj, min, pa } => {
                    let gp = cuda::GaussianParams {
                        maj: *maj as CudaFloat,
                        min: *min as CudaFloat,
                        pa: *pa as CudaFloat,
                    };
                    match comp.flux_type {
                        FluxDensityType::PowerLaw { si, .. } => {
                            gaussian_power_law_radecs.push(radec);
                            gaussian_power_law_lmns.push(lmn);
                            let fd_at_150mhz =
                                comp.estimate_at_freq(cuda::POWER_LAW_FD_REF_FREQ as _);
                            let inst_fd: Jones<f64> = fd_at_150mhz.to_inst_stokes();
                            let cuda_inst_fd = jones_to_cuda_jones(inst_fd);
                            gaussian_power_law_fds.push(cuda_inst_fd);
                            gaussian_power_law_sis.push(si as CudaFloat);
                            gaussian_power_law_gps.push(gp);
                        }

                        FluxDensityType::CurvedPowerLaw { si, q, .. } => {
                            gaussian_curved_power_law_radecs.push(radec);
                            gaussian_curved_power_law_lmns.push(lmn);
                            let fd_at_150mhz =
                                comp.estimate_at_freq(cuda::POWER_LAW_FD_REF_FREQ as _);
                            let inst_fd: Jones<f64> = fd_at_150mhz.to_inst_stokes();
                            let cuda_inst_fd = jones_to_cuda_jones(inst_fd);
                            gaussian_curved_power_law_fds.push(cuda_inst_fd);
                            gaussian_curved_power_law_qs.push(q as CudaFloat);
                            gaussian_curved_power_law_sis.push(si as CudaFloat);
                            gaussian_curved_power_law_gps.push(gp);
                        }

                        FluxDensityType::List { .. } => {
                            gaussian_list_radecs.push(radec);
                            gaussian_list_lmns.push(lmn);
                            gaussian_list_fds.push(comp.flux_type.clone());
                            gaussian_list_gps.push(gp);
                        }
                    };
                }

                ComponentType::Shapelet {
                    maj,
                    min,
                    pa,
                    coeffs,
                } => {
                    let gp = cuda::GaussianParams {
                        maj: *maj as CudaFloat,
                        min: *min as CudaFloat,
                        pa: *pa as CudaFloat,
                    };
                    match comp.flux_type {
                        FluxDensityType::PowerLaw { si, .. } => {
                            shapelet_power_law_radecs.push(radec);
                            shapelet_power_law_lmns.push(lmn);
                            let fd_at_150mhz = comp
                                .flux_type
                                .estimate_at_freq(cuda::POWER_LAW_FD_REF_FREQ as _);
                            let inst_fd: Jones<f64> = fd_at_150mhz.to_inst_stokes();
                            let cuda_inst_fd = jones_to_cuda_jones(inst_fd);
                            shapelet_power_law_fds.push(cuda_inst_fd);
                            shapelet_power_law_sis.push(si as CudaFloat);
                            shapelet_power_law_gps.push(gp);
                            shapelet_power_law_coeffs.push(coeffs.clone());
                        }

                        FluxDensityType::CurvedPowerLaw { si, q, .. } => {
                            shapelet_curved_power_law_radecs.push(radec);
                            shapelet_curved_power_law_lmns.push(lmn);
                            let fd_at_150mhz =
                                comp.estimate_at_freq(cuda::POWER_LAW_FD_REF_FREQ as _);
                            let inst_fd: Jones<f64> = fd_at_150mhz.to_inst_stokes();
                            let cuda_inst_fd = jones_to_cuda_jones(inst_fd);
                            shapelet_curved_power_law_fds.push(cuda_inst_fd);
                            shapelet_curved_power_law_qs.push(q as CudaFloat);
                            shapelet_curved_power_law_sis.push(si as CudaFloat);
                            shapelet_curved_power_law_gps.push(gp);
                            shapelet_curved_power_law_coeffs.push(coeffs.clone());
                        }

                        FluxDensityType::List { .. } => {
                            shapelet_list_radecs.push(radec);
                            shapelet_list_lmns.push(lmn);
                            shapelet_list_fds.push(comp.flux_type.clone());
                            shapelet_list_gps.push(gp);
                            shapelet_list_coeffs.push(coeffs.clone());
                        }
                    }
                }
            }
        }

        let point_list_fds =
            get_instrumental_flux_densities(&point_list_fds, unflagged_fine_chan_freqs)
                .mapv(jones_to_cuda_jones);
        let gaussian_list_fds =
            get_instrumental_flux_densities(&gaussian_list_fds, unflagged_fine_chan_freqs)
                .mapv(jones_to_cuda_jones);
        let shapelet_list_fds =
            get_instrumental_flux_densities(&shapelet_list_fds, unflagged_fine_chan_freqs)
                .mapv(jones_to_cuda_jones);

        let (shapelet_power_law_coeffs, shapelet_power_law_coeff_lens) =
            get_flattened_coeffs(shapelet_power_law_coeffs);
        let (shapelet_curved_power_law_coeffs, shapelet_curved_power_law_coeff_lens) =
            get_flattened_coeffs(shapelet_curved_power_law_coeffs);
        let (shapelet_list_coeffs, shapelet_list_coeff_lens) =
            get_flattened_coeffs(shapelet_list_coeffs);

        // Variables for CUDA. They're made flexible in their types for
        // whichever precision is being used in the CUDA code.
        let (unflagged_fine_chan_freqs_ints, unflagged_fine_chan_freqs_floats): (Vec<_>, Vec<_>) =
            unflagged_fine_chan_freqs
                .iter()
                .map(|&f| (f as u32, f as CudaFloat))
                .unzip();
        let shapelet_basis_values: Vec<CudaFloat> = shapelets::SHAPELET_BASIS_VALUES
            .iter()
            .map(|&f| f as CudaFloat)
            .collect();

        let n = unflagged_tile_xyzs.len();
        let num_baselines = (n * (n - 1)) / 2;

        let d_vis: DevicePointer<f32> = DevicePointer::malloc(
            num_baselines * unflagged_fine_chan_freqs.len() * std::mem::size_of::<Jones<f32>>(),
        )?;
        // Ensure the visibilities are zero'd.
        cuda_runtime_sys::cudaMemset(
            d_vis.get_mut().cast(),
            0,
            num_baselines * unflagged_fine_chan_freqs.len() * std::mem::size_of::<Jones<f32>>(),
        );
        cuda_runtime_sys::cudaDeviceSynchronize();

        let d_freqs = DevicePointer::copy_to_device(&unflagged_fine_chan_freqs_floats)?;
        let d_shapelet_basis_values = DevicePointer::copy_to_device(&shapelet_basis_values)?;

        let mut tile_index_to_unflagged_tile_index_map: Vec<i32> =
            Vec::with_capacity(unflagged_tile_xyzs.len());
        let mut i_unflagged_tile = 0;
        for i_tile in 0..unflagged_tile_xyzs.len() + flagged_tiles.len() {
            if flagged_tiles.contains(&i_tile) {
                i_unflagged_tile += 1;
                continue;
            }
            tile_index_to_unflagged_tile_index_map.push(i_unflagged_tile);
            i_unflagged_tile += 1;
        }
        let d_tile_index_to_unflagged_tile_index_map =
            DevicePointer::copy_to_device(&tile_index_to_unflagged_tile_index_map)?;

        Ok(SkyModellerCuda {
            cuda_beam: beam.prepare_cuda_beam(&unflagged_fine_chan_freqs_ints)?,

            phase_centre,
            array_longitude: array_longitude_rad,
            array_latitude: array_latitude_rad,
            dut1,
            precess: apply_precession,

            freqs: unflagged_fine_chan_freqs_floats,

            unflagged_tile_xyzs,
            num_baselines,

            tile_index_to_unflagged_tile_index_map: d_tile_index_to_unflagged_tile_index_map,

            sbf_l: shapelets::SBF_L.try_into().unwrap(),
            sbf_n: shapelets::SBF_N.try_into().unwrap(),
            sbf_c: shapelets::SBF_C as CudaFloat,
            sbf_dx: shapelets::SBF_DX as CudaFloat,

            d_vis,
            d_freqs,
            d_shapelet_basis_values,

            point_power_law_radecs,
            point_power_law_lmns: DevicePointer::copy_to_device(&point_power_law_lmns)?,
            point_power_law_fds: DevicePointer::copy_to_device(&point_power_law_fds)?,
            point_power_law_sis: DevicePointer::copy_to_device(&point_power_law_sis)?,

            point_curved_power_law_radecs,
            point_curved_power_law_lmns: DevicePointer::copy_to_device(
                &point_curved_power_law_lmns,
            )?,
            point_curved_power_law_fds: DevicePointer::copy_to_device(&point_curved_power_law_fds)?,
            point_curved_power_law_sis: DevicePointer::copy_to_device(&point_curved_power_law_sis)?,
            point_curved_power_law_qs: DevicePointer::copy_to_device(&point_curved_power_law_qs)?,

            point_list_radecs,
            point_list_lmns: DevicePointer::copy_to_device(&point_list_lmns)?,
            point_list_fds: DevicePointer::copy_to_device(point_list_fds.as_slice().unwrap())?,

            gaussian_power_law_radecs,
            gaussian_power_law_lmns: DevicePointer::copy_to_device(&gaussian_power_law_lmns)?,
            gaussian_power_law_fds: DevicePointer::copy_to_device(&gaussian_power_law_fds)?,
            gaussian_power_law_sis: DevicePointer::copy_to_device(&gaussian_power_law_sis)?,
            gaussian_power_law_gps: DevicePointer::copy_to_device(&gaussian_power_law_gps)?,

            gaussian_curved_power_law_radecs,
            gaussian_curved_power_law_lmns: DevicePointer::copy_to_device(
                &gaussian_curved_power_law_lmns,
            )?,
            gaussian_curved_power_law_fds: DevicePointer::copy_to_device(
                &gaussian_curved_power_law_fds,
            )?,
            gaussian_curved_power_law_sis: DevicePointer::copy_to_device(
                &gaussian_curved_power_law_sis,
            )?,
            gaussian_curved_power_law_qs: DevicePointer::copy_to_device(
                &gaussian_curved_power_law_qs,
            )?,
            gaussian_curved_power_law_gps: DevicePointer::copy_to_device(
                &gaussian_curved_power_law_gps,
            )?,

            gaussian_list_radecs,
            gaussian_list_lmns: DevicePointer::copy_to_device(&gaussian_list_lmns)?,
            gaussian_list_fds: DevicePointer::copy_to_device(
                gaussian_list_fds.as_slice().unwrap(),
            )?,
            gaussian_list_gps: DevicePointer::copy_to_device(&gaussian_list_gps)?,

            shapelet_power_law_radecs,
            shapelet_power_law_lmns: DevicePointer::copy_to_device(&shapelet_power_law_lmns)?,
            shapelet_power_law_fds: DevicePointer::copy_to_device(&shapelet_power_law_fds)?,
            shapelet_power_law_sis: DevicePointer::copy_to_device(&shapelet_power_law_sis)?,
            shapelet_power_law_gps: DevicePointer::copy_to_device(&shapelet_power_law_gps)?,
            shapelet_power_law_coeffs: DevicePointer::copy_to_device(&shapelet_power_law_coeffs)?,
            shapelet_power_law_coeff_lens: DevicePointer::copy_to_device(
                &shapelet_power_law_coeff_lens,
            )?,

            shapelet_curved_power_law_radecs,
            shapelet_curved_power_law_lmns: DevicePointer::copy_to_device(
                &shapelet_curved_power_law_lmns,
            )?,
            shapelet_curved_power_law_fds: DevicePointer::copy_to_device(
                &shapelet_curved_power_law_fds,
            )?,
            shapelet_curved_power_law_sis: DevicePointer::copy_to_device(
                &shapelet_curved_power_law_sis,
            )?,
            shapelet_curved_power_law_qs: DevicePointer::copy_to_device(
                &shapelet_curved_power_law_qs,
            )?,
            shapelet_curved_power_law_gps: DevicePointer::copy_to_device(
                &shapelet_curved_power_law_gps,
            )?,
            shapelet_curved_power_law_coeffs: DevicePointer::copy_to_device(
                &shapelet_curved_power_law_coeffs,
            )?,
            shapelet_curved_power_law_coeff_lens: DevicePointer::copy_to_device(
                &shapelet_curved_power_law_coeff_lens,
            )?,

            shapelet_list_radecs,
            shapelet_list_lmns: DevicePointer::copy_to_device(&shapelet_list_lmns)?,
            shapelet_list_fds: DevicePointer::copy_to_device(
                shapelet_list_fds.as_slice().unwrap(),
            )?,
            shapelet_list_gps: DevicePointer::copy_to_device(&shapelet_list_gps)?,
            shapelet_list_coeffs: DevicePointer::copy_to_device(&shapelet_list_coeffs)?,
            shapelet_list_coeff_lens: DevicePointer::copy_to_device(&shapelet_list_coeff_lens)?,
        })
    }

    /// For a single timestep, over the already-provided baselines and
    /// frequencies, generate visibilities for each specified sky-model
    /// point-source component.
    ///
    /// `vis_model_slice`: A mutable `ndarray` view of the model of all
    /// visibilities. The first axis is unflagged fine channel, the second
    /// unflagged baseline.
    ///
    /// `uvws`: The [UVW] coordinates of each baseline \[metres\]. This should
    /// be the same length as `vis_model_slice`'s first axis.
    ///
    /// `lst_rad`: The local sidereal time in \[radians\].
    pub(super) unsafe fn model_points_inner(
        &self,
        d_uvws: &DevicePointer<cuda::UVW>,
        lst_rad: f64,
        array_latitude_rad: f64,
    ) -> Result<(), BeamError> {
        if self.point_power_law_radecs.is_empty()
            && self.point_curved_power_law_radecs.is_empty()
            && self.point_list_radecs.is_empty()
        {
            return Ok(());
        }

        let point_beam_jones = {
            let (azs, zas): (Vec<CudaFloat>, Vec<CudaFloat>) = self
                .point_power_law_radecs
                .par_iter()
                .chain(self.point_curved_power_law_radecs.par_iter())
                .chain(self.point_list_radecs.par_iter())
                .map(|radec| {
                    let azel = radec.to_hadec(lst_rad).to_azel(array_latitude_rad);
                    (azel.az as CudaFloat, azel.za() as CudaFloat)
                })
                .unzip();
            self.cuda_beam
                .calc_jones_pair(&azs, &zas, array_latitude_rad)?
        };

        let cuda_status = cuda::model_points(
            &cuda::Points {
                num_power_law_points: self.point_power_law_radecs.len(),
                power_law_lmns: self.point_power_law_lmns.get_mut(),
                power_law_fds: self.point_power_law_fds.get_mut(),
                power_law_sis: self.point_power_law_sis.get_mut(),
                num_curved_power_law_points: self.point_curved_power_law_radecs.len(),
                curved_power_law_lmns: self.point_curved_power_law_lmns.get_mut(),
                curved_power_law_fds: self.point_curved_power_law_fds.get_mut(),
                curved_power_law_sis: self.point_curved_power_law_sis.get_mut(),
                curved_power_law_qs: self.point_curved_power_law_qs.get_mut(),
                num_list_points: self.point_list_radecs.len(),
                list_lmns: self.point_list_lmns.get_mut(),
                list_fds: self.point_list_fds.get_mut(),
            },
            &self.get_addresses(),
            d_uvws.get(),
            point_beam_jones.get().cast(),
        );
        let error_str =
            std::ffi::CString::from_vec_unchecked(vec![0; CUDA_ERROR_STR_LENGTH]).into_raw();
        cuda_status_to_error(cuda_status, error_str)?;

        Ok(())
    }

    /// For a single timestep, over the already-provided baselines and
    /// frequencies, generate visibilities for each specified sky-model
    /// Gaussian-source component.
    ///
    /// `vis_model_slice`: A mutable `ndarray` view of the model of all
    /// visibilities. The first axis is unflagged fine channel, the second
    /// unflagged baseline.
    ///
    /// `uvws`: The [UVW] coordinates of each baseline \[metres\]. This should
    /// be the same length as `vis_model_slice`'s first axis.
    ///
    /// `lst_rad`: The local sidereal time in \[radians\].
    pub(super) unsafe fn model_gaussians_inner(
        &self,
        d_uvws: &DevicePointer<cuda::UVW>,
        lst_rad: f64,
        array_latitude_rad: f64,
    ) -> Result<(), BeamError> {
        if self.gaussian_power_law_radecs.is_empty()
            && self.gaussian_curved_power_law_radecs.is_empty()
            && self.gaussian_list_radecs.is_empty()
        {
            return Ok(());
        }

        let gaussian_beam_jones = {
            let (azs, zas): (Vec<CudaFloat>, Vec<CudaFloat>) = self
                .gaussian_power_law_radecs
                .par_iter()
                .chain(self.gaussian_curved_power_law_radecs.par_iter())
                .chain(self.gaussian_list_radecs.par_iter())
                .map(|radec| {
                    let azel = radec.to_hadec(lst_rad).to_azel(array_latitude_rad);
                    (azel.az as CudaFloat, azel.za() as CudaFloat)
                })
                .unzip();
            self.cuda_beam
                .calc_jones_pair(&azs, &zas, array_latitude_rad)?
        };

        let cuda_status = cuda::model_gaussians(
            &cuda::Gaussians {
                num_power_law_gaussians: self.gaussian_power_law_radecs.len(),
                power_law_lmns: self.gaussian_power_law_lmns.get_mut(),
                power_law_fds: self.gaussian_power_law_fds.get_mut(),
                power_law_sis: self.gaussian_power_law_sis.get_mut(),
                power_law_gps: self.gaussian_power_law_gps.get_mut(),
                num_curved_power_law_gaussians: self.gaussian_curved_power_law_radecs.len(),
                curved_power_law_lmns: self.gaussian_curved_power_law_lmns.get_mut(),
                curved_power_law_fds: self.gaussian_curved_power_law_fds.get_mut(),
                curved_power_law_sis: self.gaussian_curved_power_law_sis.get_mut(),
                curved_power_law_qs: self.gaussian_curved_power_law_qs.get_mut(),
                curved_power_law_gps: self.gaussian_curved_power_law_gps.get_mut(),
                num_list_gaussians: self.gaussian_list_radecs.len(),
                list_lmns: self.gaussian_list_lmns.get_mut(),
                list_fds: self.gaussian_list_fds.get_mut(),
                list_gps: self.gaussian_list_gps.get_mut(),
            },
            &self.get_addresses(),
            d_uvws.get(),
            gaussian_beam_jones.get().cast(),
        );
        let error_str =
            std::ffi::CString::from_vec_unchecked(vec![0; CUDA_ERROR_STR_LENGTH]).into_raw();
        cuda_status_to_error(cuda_status, error_str)?;

        Ok(())
    }

    /// For a single timestep, over the already-provided baselines and
    /// frequencies, generate visibilities for each specified sky-model
    /// Gaussian-source component.
    ///
    /// `vis_model_slice`: A mutable `ndarray` view of the model of all
    /// visibilities. The first axis is unflagged fine channel, the second
    /// unflagged baseline.
    ///
    /// `uvws`: The [UVW] coordinates of each baseline \[metres\]. This should
    /// be the same length as `vis_model_slice`'s first axis.
    ///
    /// `shapelet_uvws` are special UVWs generated as if each shapelet component
    /// was at the phase centre \[metres\]. The first axis is unflagged
    /// baseline, the second shapelet component.
    ///
    /// `lst_rad`: The local sidereal time in \[radians\].
    pub(super) unsafe fn model_shapelets_inner(
        &self,
        d_uvws: &DevicePointer<cuda::UVW>,
        lst_rad: f64,
        array_latitude_rad: f64,
    ) -> Result<(), BeamError> {
        if self.shapelet_power_law_radecs.is_empty()
            && self.shapelet_curved_power_law_radecs.is_empty()
            && self.shapelet_list_radecs.is_empty()
        {
            return Ok(());
        }

        let shapelet_beam_jones = {
            let (azs, zas): (Vec<CudaFloat>, Vec<CudaFloat>) = self
                .shapelet_power_law_radecs
                .par_iter()
                .chain(self.shapelet_curved_power_law_radecs.par_iter())
                .chain(self.shapelet_list_radecs.par_iter())
                .map(|radec| {
                    let azel = radec.to_hadec(lst_rad).to_azel(array_latitude_rad);
                    (azel.az as CudaFloat, azel.za() as CudaFloat)
                })
                .unzip();
            self.cuda_beam
                .calc_jones_pair(&azs, &zas, array_latitude_rad)?
        };

        let uvs = self.get_shapelet_uvs(lst_rad);
        let power_law_uvs = DevicePointer::copy_to_device(uvs.power_law.as_slice().unwrap())?;
        let curved_power_law_uvs =
            DevicePointer::copy_to_device(uvs.curved_power_law.as_slice().unwrap())?;
        let list_uvs = DevicePointer::copy_to_device(uvs.list.as_slice().unwrap())?;

        let cuda_status = cuda::model_shapelets(
            &cuda::Shapelets {
                num_power_law_shapelets: self.shapelet_power_law_radecs.len(),
                power_law_lmns: self.shapelet_power_law_lmns.get_mut(),
                power_law_fds: self.shapelet_power_law_fds.get_mut(),
                power_law_sis: self.shapelet_power_law_sis.get_mut(),
                power_law_gps: self.shapelet_power_law_gps.get_mut(),
                power_law_shapelet_uvs: power_law_uvs.get_mut(),
                power_law_shapelet_coeffs: self.shapelet_power_law_coeffs.get_mut(),
                power_law_num_shapelet_coeffs: self.shapelet_power_law_coeff_lens.get_mut(),
                num_curved_power_law_shapelets: self.shapelet_curved_power_law_radecs.len(),
                curved_power_law_lmns: self.shapelet_curved_power_law_lmns.get_mut(),
                curved_power_law_fds: self.shapelet_curved_power_law_fds.get_mut(),
                curved_power_law_sis: self.shapelet_curved_power_law_sis.get_mut(),
                curved_power_law_qs: self.shapelet_curved_power_law_qs.get_mut(),
                curved_power_law_gps: self.shapelet_curved_power_law_gps.get_mut(),
                curved_power_law_shapelet_uvs: curved_power_law_uvs.get_mut(),
                curved_power_law_shapelet_coeffs: self.shapelet_curved_power_law_coeffs.get_mut(),
                curved_power_law_num_shapelet_coeffs: self
                    .shapelet_curved_power_law_coeff_lens
                    .get_mut(),
                num_list_shapelets: self.shapelet_list_radecs.len(),
                list_lmns: self.shapelet_list_lmns.get_mut(),
                list_fds: self.shapelet_list_fds.get_mut(),
                list_gps: self.shapelet_list_gps.get_mut(),
                list_shapelet_uvs: list_uvs.get_mut(),
                list_shapelet_coeffs: self.shapelet_list_coeffs.get_mut(),
                list_num_shapelet_coeffs: self.shapelet_list_coeff_lens.get_mut(),
            },
            &self.get_addresses(),
            d_uvws.get(),
            shapelet_beam_jones.get().cast(),
        );
        let error_str =
            std::ffi::CString::from_vec_unchecked(vec![0; CUDA_ERROR_STR_LENGTH]).into_raw();
        cuda_status_to_error(cuda_status, error_str)?;

        Ok(())
    }

    /// Get a populated [cuda::Addresses]. This should never outlive `self`.
    fn get_addresses(&self) -> cuda::Addresses {
        let n = self.unflagged_tile_xyzs.len();
        let num_baselines = (n * (n - 1)) / 2;

        cuda::Addresses {
            num_freqs: self.freqs.len() as _,
            num_vis: (num_baselines * self.freqs.len()) as _,
            num_tiles: n as _,
            sbf_l: self.sbf_l,
            sbf_n: self.sbf_n,
            sbf_c: self.sbf_c,
            sbf_dx: self.sbf_dx,
            d_freqs: self.d_freqs.get_mut(),
            d_shapelet_basis_values: self.d_shapelet_basis_values.get_mut(),
            num_unique_beam_freqs: self.cuda_beam.get_num_unique_freqs(),
            d_tile_map: self.cuda_beam.get_tile_map(),
            d_freq_map: self.cuda_beam.get_freq_map(),
            d_tile_index_to_unflagged_tile_index_map: self
                .tile_index_to_unflagged_tile_index_map
                .get(),
            d_vis: self.d_vis.get_mut().cast(),
        }
    }

    /// Copy visibilities from the CUDA device (`d_vis` in the [SkyModellerCuda]
    /// struct) into the provided `ndarray` slice. The visibilities on the
    /// device are overwritten with zeros after the copy.
    ///
    /// # Safety
    ///
    /// This function interfaces directly with the CUDA API. Rust errors attempt
    /// to catch problems but there are no guarantees.
    pub(super) unsafe fn copy_and_reset_vis(&self, mut vis_model_slice: ArrayViewMut2<Jones<f32>>) {
        // Rust's strict typing means that we can't neatly call `copy_from_device`
        // on `d_vis` into `vis_model_slice`. Do the copy manually.
        cuda_runtime_sys::cudaMemcpy(
            vis_model_slice.as_mut_ptr().cast(),
            self.d_vis.get().cast(),
            self.num_baselines * self.freqs.len() * std::mem::size_of::<Jones<f32>>(),
            cuda_runtime_sys::cudaMemcpyKind::cudaMemcpyDeviceToHost,
        );
        // Clear the device visibilities.
        cuda_runtime_sys::cudaMemset(
            self.d_vis.get_mut().cast(),
            0,
            self.num_baselines * self.freqs.len() * std::mem::size_of::<Jones<f32>>(),
        );
        cuda_runtime_sys::cudaDeviceSynchronize();
    }

    /// Shapelets need their own special kind of UVW coordinates. Each shapelet
    /// component's position is treated as the phase centre. This function uses
    /// the FFI type [cuda::ShapeletUV]; the W isn't actually used in
    /// computation, and omitting it is hopefully a little more efficient.
    ///
    /// The returned arrays have baseline as the first axis and component as the
    /// second.
    pub(super) fn get_shapelet_uvs(&self, lst_rad: f64) -> ShapeletUVs {
        ShapeletUVs {
            power_law: get_shapelet_uvs_inner(
                &self.shapelet_power_law_radecs,
                lst_rad,
                self.unflagged_tile_xyzs,
            ),
            curved_power_law: get_shapelet_uvs_inner(
                &self.shapelet_curved_power_law_radecs,
                lst_rad,
                self.unflagged_tile_xyzs,
            ),
            list: get_shapelet_uvs_inner(
                &self.shapelet_list_radecs,
                lst_rad,
                self.unflagged_tile_xyzs,
            ),
        }
    }
}

impl<'a> super::SkyModeller<'a> for SkyModellerCuda<'a> {
    fn model_timestep(
        &self,
        vis_model_slice: ArrayViewMut2<Jones<f32>>,
        timestamp: Epoch,
    ) -> Result<Vec<UVW>, BeamError> {
        let (uvws, lst, latitude) = if self.precess {
            let precession_info = precess_time(
                self.array_longitude,
                self.array_latitude,
                self.phase_centre,
                timestamp,
                self.dut1,
            );
            // Apply precession to the tile XYZ positions.
            let precessed_tile_xyzs =
                precession_info.precess_xyz_parallel(self.unflagged_tile_xyzs);
            let uvws = xyzs_to_cross_uvws_parallel(
                &precessed_tile_xyzs,
                self.phase_centre.to_hadec(precession_info.lmst_j2000),
            );
            debug!(
                "Modelling GPS timestamp {}, LMST {}°, J2000 LMST {}°",
                timestamp.as_gpst_seconds(),
                precession_info.lmst.to_degrees(),
                precession_info.lmst_j2000.to_degrees()
            );
            (
                uvws,
                precession_info.lmst_j2000,
                precession_info.array_latitude_j2000,
            )
        } else {
            let lst = get_lmst(self.array_longitude, timestamp, self.dut1);
            let uvws = xyzs_to_cross_uvws_parallel(
                self.unflagged_tile_xyzs,
                self.phase_centre.to_hadec(lst),
            );
            debug!(
                "Modelling GPS timestamp {}, LMST {}°",
                timestamp.as_gpst_seconds(),
                lst.to_degrees()
            );
            (uvws, lst, self.array_latitude)
        };

        let cuda_uvws: Vec<cuda::UVW> = uvws
            .iter()
            .map(|&uvw| cuda::UVW {
                u: uvw.u as CudaFloat,
                v: uvw.v as CudaFloat,
                w: uvw.w as CudaFloat,
            })
            .collect();

        unsafe {
            let d_uvws = DevicePointer::copy_to_device(&cuda_uvws)?;

            self.model_points_inner(&d_uvws, lst, latitude)?;
            self.model_gaussians_inner(&d_uvws, lst, latitude)?;
            self.model_shapelets_inner(&d_uvws, lst, latitude)?;

            self.copy_and_reset_vis(vis_model_slice);
        }

        Ok(uvws)
    }

    fn model_points(
        &self,
        vis_model_slice: ArrayViewMut2<Jones<f32>>,
        timestamp: Epoch,
    ) -> Result<Vec<UVW>, BeamError> {
        let (uvws, lst, latitude) = if self.precess {
            let precession_info = precess_time(
                self.array_longitude,
                self.array_latitude,
                self.phase_centre,
                timestamp,
                self.dut1,
            );
            // Apply precession to the tile XYZ positions.
            let precessed_tile_xyzs =
                precession_info.precess_xyz_parallel(self.unflagged_tile_xyzs);
            let uvws = xyzs_to_cross_uvws_parallel(
                &precessed_tile_xyzs,
                self.phase_centre.to_hadec(precession_info.lmst_j2000),
            );
            (
                uvws,
                precession_info.lmst_j2000,
                precession_info.array_latitude_j2000,
            )
        } else {
            let lst = get_lmst(self.array_longitude, timestamp, self.dut1);
            let uvws = xyzs_to_cross_uvws_parallel(
                self.unflagged_tile_xyzs,
                self.phase_centre.to_hadec(lst),
            );
            (uvws, lst, self.array_latitude)
        };

        let cuda_uvws: Vec<cuda::UVW> = uvws
            .iter()
            .map(|uvw| cuda::UVW {
                u: uvw.u as CudaFloat,
                v: uvw.v as CudaFloat,
                w: uvw.w as CudaFloat,
            })
            .collect();

        unsafe {
            let d_uvws = DevicePointer::copy_to_device(&cuda_uvws)?;

            self.model_points_inner(&d_uvws, lst, latitude)?;
            self.copy_and_reset_vis(vis_model_slice);
        }

        Ok(uvws)
    }

    /// Model only the Gaussian sources. If other types of sources will also be
    /// modelled, it is more efficient to use `model_timestep`.
    ///
    /// # Safety
    ///
    /// This function interfaces directly with the CUDA API. Rust errors attempt
    /// to catch problems but there are no guarantees.
    fn model_gaussians(
        &self,
        vis_model_slice: ArrayViewMut2<Jones<f32>>,
        timestamp: Epoch,
    ) -> Result<Vec<UVW>, BeamError> {
        let (uvws, lst, latitude) = if self.precess {
            let precession_info = precess_time(
                self.array_longitude,
                self.array_latitude,
                self.phase_centre,
                timestamp,
                self.dut1,
            );
            // Apply precession to the tile XYZ positions.
            let precessed_tile_xyzs =
                precession_info.precess_xyz_parallel(self.unflagged_tile_xyzs);
            let uvws = xyzs_to_cross_uvws_parallel(
                &precessed_tile_xyzs,
                self.phase_centre.to_hadec(precession_info.lmst_j2000),
            );
            (
                uvws,
                precession_info.lmst_j2000,
                precession_info.array_latitude_j2000,
            )
        } else {
            let lst = get_lmst(self.array_longitude, timestamp, self.dut1);
            let uvws = xyzs_to_cross_uvws_parallel(
                self.unflagged_tile_xyzs,
                self.phase_centre.to_hadec(lst),
            );
            (uvws, lst, self.array_latitude)
        };

        let cuda_uvws: Vec<cuda::UVW> = uvws
            .iter()
            .map(|uvw| cuda::UVW {
                u: uvw.u as CudaFloat,
                v: uvw.v as CudaFloat,
                w: uvw.w as CudaFloat,
            })
            .collect();

        unsafe {
            let d_uvws = DevicePointer::copy_to_device(&cuda_uvws)?;

            self.model_gaussians_inner(&d_uvws, lst, latitude)?;
            self.copy_and_reset_vis(vis_model_slice);
        }

        Ok(uvws)
    }

    /// Model only the shapelet sources. If other types of sources will also be
    /// modelled, it is more efficient to use `model_timestep`.
    ///
    /// # Safety
    ///
    /// This function interfaces directly with the CUDA API. Rust errors attempt
    /// to catch problems but there are no guarantees.
    fn model_shapelets(
        &self,
        vis_model_slice: ArrayViewMut2<Jones<f32>>,
        timestamp: Epoch,
    ) -> Result<Vec<UVW>, BeamError> {
        let (uvws, lst, latitude) = if self.precess {
            let precession_info = precess_time(
                self.array_longitude,
                self.array_latitude,
                self.phase_centre,
                timestamp,
                self.dut1,
            );
            // Apply precession to the tile XYZ positions.
            let precessed_tile_xyzs =
                precession_info.precess_xyz_parallel(self.unflagged_tile_xyzs);
            let uvws = xyzs_to_cross_uvws_parallel(
                &precessed_tile_xyzs,
                self.phase_centre.to_hadec(precession_info.lmst_j2000),
            );
            (
                uvws,
                precession_info.lmst_j2000,
                precession_info.array_latitude_j2000,
            )
        } else {
            let lst = get_lmst(self.array_longitude, timestamp, self.dut1);
            let uvws = xyzs_to_cross_uvws_parallel(
                self.unflagged_tile_xyzs,
                self.phase_centre.to_hadec(lst),
            );
            (uvws, lst, self.array_latitude)
        };

        let cuda_uvws: Vec<cuda::UVW> = uvws
            .iter()
            .map(|uvw| cuda::UVW {
                u: uvw.u as CudaFloat,
                v: uvw.v as CudaFloat,
                w: uvw.w as CudaFloat,
            })
            .collect();

        unsafe {
            let d_uvws = DevicePointer::copy_to_device(&cuda_uvws)?;

            self.model_shapelets_inner(&d_uvws, lst, latitude)?;
            self.copy_and_reset_vis(vis_model_slice);
        }

        Ok(uvws)
    }
}

impl std::fmt::Debug for SkyModellerCuda<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SkyModellerCuda").finish()
    }
}

/// The return type of [SkyModellerCuda::get_shapelet_uvs]. These arrays have
/// baseline as the first axis and component as the second.
pub(super) struct ShapeletUVs {
    power_law: Array2<cuda::ShapeletUV>,
    curved_power_law: Array2<cuda::ShapeletUV>,
    pub(super) list: Array2<cuda::ShapeletUV>,
}

fn get_shapelet_uvs_inner(
    radecs: &[RADec],
    lst_rad: f64,
    tile_xyzs: &[XyzGeodetic],
) -> Array2<cuda::ShapeletUV> {
    let n = tile_xyzs.len();
    let num_baselines = (n * (n - 1)) / 2;

    let mut shapelet_uvs: Array2<cuda::ShapeletUV> = Array2::from_elem(
        (num_baselines, radecs.len()),
        cuda::ShapeletUV { u: 0.0, v: 0.0 },
    );
    shapelet_uvs
        .axis_iter_mut(Axis(1))
        .into_par_iter()
        .zip(radecs.par_iter())
        .for_each(|(mut baseline_uv, radec)| {
            let hadec = radec.to_hadec(lst_rad);
            let shapelet_uvs: Vec<cuda::ShapeletUV> = xyzs_to_cross_uvws_parallel(tile_xyzs, hadec)
                .into_iter()
                .map(|uvw| cuda::ShapeletUV {
                    u: uvw.u as CudaFloat,
                    v: uvw.v as CudaFloat,
                })
                .collect();
            baseline_uv.assign(&Array1::from(shapelet_uvs));
        });
    shapelet_uvs
}

/// There are a variable number of shapelet coefficients for each shapelet
/// component. To avoid excessive dereferencing on GPUs (expensive), this
/// method flattens the coefficients into a single array (lengths of the
/// array-of-arrays).
fn get_flattened_coeffs(
    shapelet_coeffs: Vec<Vec<ShapeletCoeff>>,
) -> (Vec<cuda::ShapeletCoeff>, Vec<usize>) {
    let mut coeffs: Vec<cuda::ShapeletCoeff> = vec![];
    let mut coeff_lengths = Vec::with_capacity(coeffs.len());

    for coeffs_for_comp in shapelet_coeffs {
        coeff_lengths.push(coeffs_for_comp.len());
        for coeff in coeffs_for_comp {
            coeffs.push(cuda::ShapeletCoeff {
                n1: coeff.n1,
                n2: coeff.n2,
                value: coeff.value as CudaFloat,
            })
        }
    }

    coeffs.shrink_to_fit();
    (coeffs, coeff_lengths)
}
