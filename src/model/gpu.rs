// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to generate sky-model visibilities with CUDA/HIP.

use std::{borrow::Cow, collections::HashSet};

use hifitime::{Duration, Epoch};
use log::debug;
use marlu::{
    pos::xyz::xyzs_to_cross_uvws,
    precession::{get_lmst, precess_time},
    Jones, LmnRime, RADec, XyzGeodetic, UVW,
};
use ndarray::prelude::*;

use super::{mask_pols, shapelets, ModelError, SkyModeller};
use crate::{
    beam::{Beam, BeamGpu},
    context::Polarisations,
    gpu::{self, gpu_kernel_call, DevicePointer, GpuError, GpuFloat, GpuJones},
    srclist::{
        get_instrumental_flux_densities, ComponentType, FluxDensityType, ShapeletCoeff, SourceList,
    },
};

/// The first axis of `*_list_fds` is unflagged fine channel frequency, the
/// second is the source component. The length of `hadecs`, `lmns`,
/// `*_list_fds`'s second axis are the same.
pub struct SkyModellerGpu<'a> {
    /// The trait object to use for beam calculations.
    gpu_beam: Box<dyn BeamGpu>,

    /// The phase centre used for all modelling.
    phase_centre: RADec,
    /// The longitude of the array we're using \[radians\].
    array_longitude: f64,
    /// The *unprecessed* latitude of the array we're using \[radians\]. If we
    /// are precessing, this latitude isn't used when calculating [`AzEl`]s.
    array_latitude: f64,
    /// The UT1 - UTC offset. If this is 0, effectively UT1 == UTC, which is a
    /// wrong assumption by up to 0.9s. We assume the this value does not change
    /// over the timestamps given to this `SkyModellerGpu`.
    dut1: Duration,
    /// Shift baselines, LSTs and array latitudes back to J2000.
    precess: bool,

    /// The *unprecessed* [`XyzGeodetic`] positions of each of the unflagged
    /// tiles.
    unflagged_tile_xyzs: &'a [XyzGeodetic],
    num_baselines: i32,
    num_freqs: i32,

    pols: Polarisations,

    /// A simple map from an absolute tile index into an unflagged tile index.
    /// This is important because CUDA/HIP will use tile indices from 0 to the
    /// length of `unflagged_tile_xyzs`, but the beam code has dipole delays and
    /// dipole gains available for *all* tiles. So if tile 32 is flagged, any
    /// CUDA/HIP thread with a tile index of 32 would naively get the flagged
    /// beam info. This map would make tile index go to the next unflagged tile,
    /// perhaps 33.
    tile_index_to_unflagged_tile_index_map: DevicePointer<i32>,

    d_freqs: DevicePointer<GpuFloat>,
    d_shapelet_basis_values: DevicePointer<GpuFloat>,

    point_power_law_radecs: Vec<RADec>,
    point_power_law_lmns: DevicePointer<gpu::LmnRime>,
    /// Instrumental flux densities calculated at 150 MHz.
    point_power_law_fds: DevicePointer<GpuJones>,
    /// Spectral indices.
    point_power_law_sis: DevicePointer<GpuFloat>,

    point_curved_power_law_radecs: Vec<RADec>,
    point_curved_power_law_lmns: DevicePointer<gpu::LmnRime>,
    pub(super) point_curved_power_law_fds: DevicePointer<GpuJones>,
    pub(super) point_curved_power_law_sis: DevicePointer<GpuFloat>,
    point_curved_power_law_qs: DevicePointer<GpuFloat>,

    point_list_radecs: Vec<RADec>,
    point_list_lmns: DevicePointer<gpu::LmnRime>,
    /// Instrumental (i.e. XX, XY, YX, XX).
    point_list_fds: DevicePointer<GpuJones>,

    gaussian_power_law_radecs: Vec<RADec>,
    gaussian_power_law_lmns: DevicePointer<gpu::LmnRime>,
    /// Instrumental flux densities calculated at 150 MHz.
    gaussian_power_law_fds: DevicePointer<GpuJones>,
    /// Spectral indices.
    gaussian_power_law_sis: DevicePointer<GpuFloat>,
    gaussian_power_law_gps: DevicePointer<gpu::GaussianParams>,

    gaussian_curved_power_law_radecs: Vec<RADec>,
    gaussian_curved_power_law_lmns: DevicePointer<gpu::LmnRime>,
    gaussian_curved_power_law_fds: DevicePointer<GpuJones>,
    gaussian_curved_power_law_sis: DevicePointer<GpuFloat>,
    gaussian_curved_power_law_qs: DevicePointer<GpuFloat>,
    gaussian_curved_power_law_gps: DevicePointer<gpu::GaussianParams>,

    gaussian_list_radecs: Vec<RADec>,
    gaussian_list_lmns: DevicePointer<gpu::LmnRime>,
    /// Instrumental (i.e. XX, XY, YX, XX).
    gaussian_list_fds: DevicePointer<GpuJones>,
    gaussian_list_gps: DevicePointer<gpu::GaussianParams>,

    shapelet_power_law_radecs: Vec<RADec>,
    shapelet_power_law_lmns: DevicePointer<gpu::LmnRime>,
    /// Instrumental flux densities calculated at 150 MHz.
    shapelet_power_law_fds: DevicePointer<GpuJones>,
    /// Spectral indices.
    shapelet_power_law_sis: DevicePointer<GpuFloat>,
    shapelet_power_law_gps: DevicePointer<gpu::GaussianParams>,
    shapelet_power_law_coeffs: DevicePointer<gpu::ShapeletCoeff>,
    shapelet_power_law_coeff_lens: DevicePointer<i32>,

    shapelet_curved_power_law_radecs: Vec<RADec>,
    shapelet_curved_power_law_lmns: DevicePointer<gpu::LmnRime>,
    shapelet_curved_power_law_fds: DevicePointer<GpuJones>,
    shapelet_curved_power_law_sis: DevicePointer<GpuFloat>,
    shapelet_curved_power_law_qs: DevicePointer<GpuFloat>,
    shapelet_curved_power_law_gps: DevicePointer<gpu::GaussianParams>,
    shapelet_curved_power_law_coeffs: DevicePointer<gpu::ShapeletCoeff>,
    shapelet_curved_power_law_coeff_lens: DevicePointer<i32>,

    shapelet_list_radecs: Vec<RADec>,
    shapelet_list_lmns: DevicePointer<gpu::LmnRime>,
    /// Instrumental (i.e. XX, XY, YX, XX).
    shapelet_list_fds: DevicePointer<GpuJones>,
    shapelet_list_gps: DevicePointer<gpu::GaussianParams>,
    shapelet_list_coeffs: DevicePointer<gpu::ShapeletCoeff>,
    shapelet_list_coeff_lens: DevicePointer<i32>,
}

impl<'a> SkyModellerGpu<'a> {
    /// Given a source list, split the components into each component type (e.g.
    /// points, shapelets) and by each flux density type (e.g. list, power law),
    /// then copy them to a GPU ready for modelling. Where possible, list flux
    /// density types should be converted to power laws before calling this
    /// function, because using power laws is more efficient and probably more
    /// accurate.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        beam: &dyn Beam,
        source_list: &SourceList,
        pols: Polarisations,
        unflagged_tile_xyzs: &'a [XyzGeodetic],
        unflagged_fine_chan_freqs: &[f64],
        flagged_tiles: &HashSet<usize>,
        phase_centre: RADec,
        array_longitude_rad: f64,
        array_latitude_rad: f64,
        dut1: Duration,
        apply_precession: bool,
    ) -> Result<SkyModellerGpu<'a>, ModelError> {
        let mut point_power_law_radecs: Vec<RADec> = vec![];
        let mut point_power_law_lmns: Vec<gpu::LmnRime> = vec![];
        let mut point_power_law_fds: Vec<_> = vec![];
        let mut point_power_law_sis: Vec<_> = vec![];

        let mut point_curved_power_law_radecs: Vec<RADec> = vec![];
        let mut point_curved_power_law_lmns: Vec<gpu::LmnRime> = vec![];
        let mut point_curved_power_law_fds: Vec<_> = vec![];
        let mut point_curved_power_law_sis: Vec<_> = vec![];
        let mut point_curved_power_law_qs: Vec<_> = vec![];

        let mut point_list_radecs: Vec<RADec> = vec![];
        let mut point_list_lmns: Vec<gpu::LmnRime> = vec![];
        let mut point_list_fds: Vec<&FluxDensityType> = vec![];

        let mut gaussian_power_law_radecs: Vec<RADec> = vec![];
        let mut gaussian_power_law_lmns: Vec<gpu::LmnRime> = vec![];
        let mut gaussian_power_law_fds: Vec<_> = vec![];
        let mut gaussian_power_law_sis: Vec<_> = vec![];
        let mut gaussian_power_law_gps: Vec<gpu::GaussianParams> = vec![];

        let mut gaussian_curved_power_law_radecs: Vec<RADec> = vec![];
        let mut gaussian_curved_power_law_lmns: Vec<gpu::LmnRime> = vec![];
        let mut gaussian_curved_power_law_fds: Vec<_> = vec![];
        let mut gaussian_curved_power_law_sis: Vec<_> = vec![];
        let mut gaussian_curved_power_law_qs: Vec<_> = vec![];
        let mut gaussian_curved_power_law_gps: Vec<gpu::GaussianParams> = vec![];

        let mut gaussian_list_radecs: Vec<RADec> = vec![];
        let mut gaussian_list_lmns: Vec<gpu::LmnRime> = vec![];
        let mut gaussian_list_fds: Vec<&FluxDensityType> = vec![];
        let mut gaussian_list_gps: Vec<gpu::GaussianParams> = vec![];

        let mut shapelet_power_law_radecs: Vec<RADec> = vec![];
        let mut shapelet_power_law_lmns: Vec<gpu::LmnRime> = vec![];
        let mut shapelet_power_law_fds: Vec<_> = vec![];
        let mut shapelet_power_law_sis: Vec<_> = vec![];
        let mut shapelet_power_law_gps: Vec<gpu::GaussianParams> = vec![];
        let mut shapelet_power_law_coeffs: Vec<&[ShapeletCoeff]> = vec![];

        let mut shapelet_curved_power_law_radecs: Vec<RADec> = vec![];
        let mut shapelet_curved_power_law_lmns: Vec<gpu::LmnRime> = vec![];
        let mut shapelet_curved_power_law_fds: Vec<_> = vec![];
        let mut shapelet_curved_power_law_sis: Vec<_> = vec![];
        let mut shapelet_curved_power_law_qs: Vec<_> = vec![];
        let mut shapelet_curved_power_law_gps: Vec<gpu::GaussianParams> = vec![];
        let mut shapelet_curved_power_law_coeffs: Vec<&[ShapeletCoeff]> = vec![];

        let mut shapelet_list_radecs: Vec<RADec> = vec![];
        let mut shapelet_list_lmns: Vec<gpu::LmnRime> = vec![];
        let mut shapelet_list_fds: Vec<&FluxDensityType> = vec![];
        let mut shapelet_list_gps: Vec<gpu::GaussianParams> = vec![];
        let mut shapelet_list_coeffs: Vec<&[ShapeletCoeff]> = vec![];

        let jones_to_gpu_jones = |j: Jones<f64>| -> GpuJones {
            GpuJones {
                j00_re: j[0].re as GpuFloat,
                j00_im: j[0].im as GpuFloat,
                j01_re: j[1].re as GpuFloat,
                j01_im: j[1].im as GpuFloat,
                j10_re: j[2].re as GpuFloat,
                j10_im: j[2].im as GpuFloat,
                j11_re: j[3].re as GpuFloat,
                j11_im: j[3].im as GpuFloat,
            }
        };

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
            .flat_map(|(_, src)| src.components.iter())
        {
            let radec = comp.radec;
            let LmnRime { l, m, n } = comp.radec.to_lmn(phase_centre).prepare_for_rime();
            let lmn = gpu::LmnRime {
                l: l as GpuFloat,
                m: m as GpuFloat,
                n: n as GpuFloat,
            };
            match &comp.flux_type {
                FluxDensityType::PowerLaw { si, fd: _ } => {
                    // Rather than using this PL's reference freq, use a pre-
                    // defined one, so the the GPU code doesn't need to keep
                    // track of all reference freqs.
                    let fd_at_150mhz = comp.estimate_at_freq(gpu::POWER_LAW_FD_REF_FREQ as _);
                    let inst_fd: Jones<f64> = fd_at_150mhz.to_inst_stokes();
                    let gpu_inst_fd = jones_to_gpu_jones(inst_fd);

                    match &comp.comp_type {
                        ComponentType::Point => {
                            point_power_law_radecs.push(radec);
                            point_power_law_lmns.push(lmn);
                            point_power_law_fds.push(gpu_inst_fd);
                            point_power_law_sis.push(*si as GpuFloat);
                        }

                        ComponentType::Gaussian { maj, min, pa } => {
                            let gp = gpu::GaussianParams {
                                maj: *maj as GpuFloat,
                                min: *min as GpuFloat,
                                pa: *pa as GpuFloat,
                            };
                            gaussian_power_law_radecs.push(radec);
                            gaussian_power_law_lmns.push(lmn);
                            gaussian_power_law_gps.push(gp);
                            gaussian_power_law_fds.push(gpu_inst_fd);
                            gaussian_power_law_sis.push(*si as GpuFloat);
                        }

                        ComponentType::Shapelet {
                            maj,
                            min,
                            pa,
                            coeffs,
                        } => {
                            let gp = gpu::GaussianParams {
                                maj: *maj as GpuFloat,
                                min: *min as GpuFloat,
                                pa: *pa as GpuFloat,
                            };
                            shapelet_power_law_radecs.push(radec);
                            shapelet_power_law_lmns.push(lmn);
                            shapelet_power_law_gps.push(gp);
                            shapelet_power_law_coeffs.push(coeffs);
                            shapelet_power_law_fds.push(gpu_inst_fd);
                            shapelet_power_law_sis.push(*si as GpuFloat);
                        }
                    };
                }

                FluxDensityType::CurvedPowerLaw { si, fd, q } => {
                    let fd_at_150mhz = comp.estimate_at_freq(gpu::POWER_LAW_FD_REF_FREQ as _);
                    let inst_fd: Jones<f64> = fd_at_150mhz.to_inst_stokes();
                    let gpu_inst_fd = jones_to_gpu_jones(inst_fd);

                    // A new SI is needed when changing the reference freq.
                    // Thanks Jack.
                    #[allow(clippy::unnecessary_cast)]
                    let si = if fd.freq == gpu::POWER_LAW_FD_REF_FREQ as f64 {
                        *si
                    } else {
                        #[allow(clippy::unnecessary_cast)]
                        let logratio = (fd.freq / gpu::POWER_LAW_FD_REF_FREQ as f64).ln();
                        ((fd.i / fd_at_150mhz.i).ln() - q * logratio.powi(2)) / logratio
                    };

                    match &comp.comp_type {
                        ComponentType::Point => {
                            point_curved_power_law_radecs.push(radec);
                            point_curved_power_law_lmns.push(lmn);
                            point_curved_power_law_fds.push(gpu_inst_fd);
                            point_curved_power_law_sis.push(si as GpuFloat);
                            point_curved_power_law_qs.push(*q as GpuFloat);
                        }

                        ComponentType::Gaussian { maj, min, pa } => {
                            let gp = gpu::GaussianParams {
                                maj: *maj as GpuFloat,
                                min: *min as GpuFloat,
                                pa: *pa as GpuFloat,
                            };
                            gaussian_curved_power_law_radecs.push(radec);
                            gaussian_curved_power_law_lmns.push(lmn);
                            gaussian_curved_power_law_gps.push(gp);
                            gaussian_curved_power_law_fds.push(gpu_inst_fd);
                            gaussian_curved_power_law_sis.push(si as GpuFloat);
                            gaussian_curved_power_law_qs.push(*q as GpuFloat);
                        }

                        ComponentType::Shapelet {
                            maj,
                            min,
                            pa,
                            coeffs,
                        } => {
                            let gp = gpu::GaussianParams {
                                maj: *maj as GpuFloat,
                                min: *min as GpuFloat,
                                pa: *pa as GpuFloat,
                            };
                            shapelet_curved_power_law_radecs.push(radec);
                            shapelet_curved_power_law_lmns.push(lmn);
                            shapelet_curved_power_law_gps.push(gp);
                            shapelet_curved_power_law_coeffs.push(coeffs);
                            shapelet_curved_power_law_fds.push(gpu_inst_fd);
                            shapelet_curved_power_law_sis.push(si as GpuFloat);
                            shapelet_curved_power_law_qs.push(*q as GpuFloat);
                        }
                    };
                }

                FluxDensityType::List(_) => match &comp.comp_type {
                    ComponentType::Point => {
                        point_list_radecs.push(radec);
                        point_list_lmns.push(lmn);
                        point_list_fds.push(&comp.flux_type);
                    }

                    ComponentType::Gaussian { maj, min, pa } => {
                        let gp = gpu::GaussianParams {
                            maj: *maj as GpuFloat,
                            min: *min as GpuFloat,
                            pa: *pa as GpuFloat,
                        };
                        gaussian_list_radecs.push(radec);
                        gaussian_list_lmns.push(lmn);
                        gaussian_list_gps.push(gp);
                        gaussian_list_fds.push(&comp.flux_type);
                    }

                    ComponentType::Shapelet {
                        maj,
                        min,
                        pa,
                        coeffs,
                    } => {
                        let gp = gpu::GaussianParams {
                            maj: *maj as GpuFloat,
                            min: *min as GpuFloat,
                            pa: *pa as GpuFloat,
                        };
                        shapelet_list_radecs.push(radec);
                        shapelet_list_lmns.push(lmn);
                        shapelet_list_gps.push(gp);
                        shapelet_list_coeffs.push(coeffs);
                        shapelet_list_fds.push(&comp.flux_type);
                    }
                },
            }
        }

        let point_list_fds =
            get_instrumental_flux_densities(&point_list_fds, unflagged_fine_chan_freqs)
                .mapv(jones_to_gpu_jones);
        let gaussian_list_fds =
            get_instrumental_flux_densities(&gaussian_list_fds, unflagged_fine_chan_freqs)
                .mapv(jones_to_gpu_jones);
        let shapelet_list_fds =
            get_instrumental_flux_densities(&shapelet_list_fds, unflagged_fine_chan_freqs)
                .mapv(jones_to_gpu_jones);

        let (shapelet_power_law_coeffs, shapelet_power_law_coeff_lens) =
            get_flattened_coeffs(shapelet_power_law_coeffs);
        let (shapelet_curved_power_law_coeffs, shapelet_curved_power_law_coeff_lens) =
            get_flattened_coeffs(shapelet_curved_power_law_coeffs);
        let (shapelet_list_coeffs, shapelet_list_coeff_lens) =
            get_flattened_coeffs(shapelet_list_coeffs);

        // Variables for CUDA/HIP. They're made flexible in their types for
        // whichever precision is being used.
        let (unflagged_fine_chan_freqs_ints, unflagged_fine_chan_freqs_floats): (Vec<_>, Vec<_>) =
            unflagged_fine_chan_freqs
                .iter()
                .map(|&f| (f as u32, f as GpuFloat))
                .unzip();
        let shapelet_basis_values: Vec<GpuFloat> = shapelets::SHAPELET_BASIS_VALUES
            .iter()
            .map(|&f| f as GpuFloat)
            .collect();

        let num_baselines = (unflagged_tile_xyzs.len() * (unflagged_tile_xyzs.len() - 1)) / 2;
        let num_freqs = unflagged_fine_chan_freqs.len();

        let d_freqs = DevicePointer::copy_to_device(&unflagged_fine_chan_freqs_floats)?;
        let d_shapelet_basis_values = DevicePointer::copy_to_device(&shapelet_basis_values)?;

        let gpu_beam = beam.prepare_gpu_beam(&unflagged_fine_chan_freqs_ints)?;

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

        Ok(SkyModellerGpu {
            gpu_beam,

            phase_centre,
            array_longitude: array_longitude_rad,
            array_latitude: array_latitude_rad,
            dut1,
            precess: apply_precession,

            unflagged_tile_xyzs,
            num_baselines: num_baselines.try_into().expect("not bigger than i32::MAX"),
            num_freqs: num_freqs.try_into().expect("not bigger than i32::MAX"),

            pols,

            tile_index_to_unflagged_tile_index_map: d_tile_index_to_unflagged_tile_index_map,

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
            point_list_fds: DevicePointer::copy_to_device(
                point_list_fds.as_slice().expect("is contiguous"),
            )?,

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
                gaussian_list_fds.as_slice().expect("is contiguous"),
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
                shapelet_list_fds.as_slice().expect("is contiguous"),
            )?,
            shapelet_list_gps: DevicePointer::copy_to_device(&shapelet_list_gps)?,
            shapelet_list_coeffs: DevicePointer::copy_to_device(&shapelet_list_coeffs)?,
            shapelet_list_coeff_lens: DevicePointer::copy_to_device(&shapelet_list_coeff_lens)?,
        })
    }

    /// This function is mostly used for testing. For a single timestep, over
    /// the already-provided baselines and frequencies, generate visibilities
    /// for each specified sky-model point-source component. The
    /// `SkyModellerGpu` object *must* already have its UVW coordinates set; see
    /// [`SkyModellerGpu::set_uvws`].
    ///
    /// `lst_rad`: The local sidereal time in \[radians\].
    ///
    /// `array_latitude_rad`: The latitude of the array/telescope/interferometer
    /// in \[radians\].
    pub(super) unsafe fn model_points(
        &self,
        lst_rad: f64,
        array_latitude_rad: f64,
        d_uvws: &DevicePointer<gpu::UVW>,
        d_beam_jones: &mut DevicePointer<GpuJones>,
        d_vis_fb: &mut DevicePointer<Jones<f32>>,
    ) -> Result<(), ModelError> {
        if self.point_power_law_radecs.is_empty()
            && self.point_curved_power_law_radecs.is_empty()
            && self.point_list_radecs.is_empty()
        {
            return Ok(());
        }

        {
            let (azs, zas): (Vec<GpuFloat>, Vec<GpuFloat>) = self
                .point_power_law_radecs
                .iter()
                .chain(self.point_curved_power_law_radecs.iter())
                .chain(self.point_list_radecs.iter())
                .map(|radec| {
                    let azel = radec.to_hadec(lst_rad).to_azel(array_latitude_rad);
                    (azel.az as GpuFloat, azel.za() as GpuFloat)
                })
                .unzip();
            d_beam_jones.realloc(
                self.gpu_beam.get_num_unique_tiles() as usize
                    * self.gpu_beam.get_num_unique_freqs() as usize
                    * azs.len()
                    * std::mem::size_of::<GpuJones>(),
            )?;
            let d_azs = DevicePointer::copy_to_device(&azs)?;
            let d_zas = DevicePointer::copy_to_device(&zas)?;

            self.gpu_beam
                .calc_jones_pair(&d_azs, &d_zas, array_latitude_rad, d_beam_jones)?;
        }

        gpu_kernel_call!(
            gpu::model_points,
            &gpu::Points {
                num_power_laws: self
                    .point_power_law_radecs
                    .len()
                    .try_into()
                    .expect("not bigger than i32::MAX"),
                power_law_lmns: self.point_power_law_lmns.get(),
                power_law_fds: self.point_power_law_fds.get(),
                power_law_sis: self.point_power_law_sis.get(),
                num_curved_power_laws: self
                    .point_curved_power_law_radecs
                    .len()
                    .try_into()
                    .expect("not bigger than i32::MAX"),
                curved_power_law_lmns: self.point_curved_power_law_lmns.get(),
                curved_power_law_fds: self.point_curved_power_law_fds.get(),
                curved_power_law_sis: self.point_curved_power_law_sis.get(),
                curved_power_law_qs: self.point_curved_power_law_qs.get(),
                num_lists: self
                    .point_list_radecs
                    .len()
                    .try_into()
                    .expect("not bigger than i32::MAX"),
                list_lmns: self.point_list_lmns.get(),
                list_fds: self.point_list_fds.get(),
            },
            &self.get_addresses(),
            d_uvws.get(),
            d_beam_jones.get(),
            d_vis_fb.get_mut().cast(),
        )?;

        Ok(())
    }

    /// This function is mostly used for testing. For a single timestep, over
    /// the already-provided baselines and frequencies, generate visibilities
    /// for each specified sky-model Gaussian-source component. The
    /// `SkyModellerGpu` object *must* already have its UVW coordinates set; see
    /// [`SkyModellerGpu::set_uvws`].
    ///
    /// `lst_rad`: The local sidereal time in \[radians\].
    ///
    /// `array_latitude_rad`: The latitude of the array/telescope/interferometer
    /// in \[radians\].
    pub(super) unsafe fn model_gaussians(
        &self,
        lst_rad: f64,
        array_latitude_rad: f64,
        d_uvws: &DevicePointer<gpu::UVW>,
        d_beam_jones: &mut DevicePointer<GpuJones>,
        d_vis_fb: &mut DevicePointer<Jones<f32>>,
    ) -> Result<(), ModelError> {
        if self.gaussian_power_law_radecs.is_empty()
            && self.gaussian_curved_power_law_radecs.is_empty()
            && self.gaussian_list_radecs.is_empty()
        {
            return Ok(());
        }

        {
            let (azs, zas): (Vec<GpuFloat>, Vec<GpuFloat>) = self
                .gaussian_power_law_radecs
                .iter()
                .chain(self.gaussian_curved_power_law_radecs.iter())
                .chain(self.gaussian_list_radecs.iter())
                .map(|radec| {
                    let azel = radec.to_hadec(lst_rad).to_azel(array_latitude_rad);
                    (azel.az as GpuFloat, azel.za() as GpuFloat)
                })
                .unzip();
            d_beam_jones.realloc(
                self.gpu_beam.get_num_unique_tiles() as usize
                    * self.gpu_beam.get_num_unique_freqs() as usize
                    * azs.len()
                    * std::mem::size_of::<GpuJones>(),
            )?;
            let d_azs = DevicePointer::copy_to_device(&azs)?;
            let d_zas = DevicePointer::copy_to_device(&zas)?;
            self.gpu_beam
                .calc_jones_pair(&d_azs, &d_zas, array_latitude_rad, d_beam_jones)?;
        }

        gpu_kernel_call!(
            gpu::model_gaussians,
            &gpu::Gaussians {
                num_power_laws: self
                    .gaussian_power_law_radecs
                    .len()
                    .try_into()
                    .expect("not bigger than i32::MAX"),
                power_law_lmns: self.gaussian_power_law_lmns.get(),
                power_law_fds: self.gaussian_power_law_fds.get(),
                power_law_sis: self.gaussian_power_law_sis.get(),
                power_law_gps: self.gaussian_power_law_gps.get(),
                num_curved_power_laws: self
                    .gaussian_curved_power_law_radecs
                    .len()
                    .try_into()
                    .expect("not bigger than i32::MAX"),
                curved_power_law_lmns: self.gaussian_curved_power_law_lmns.get(),
                curved_power_law_fds: self.gaussian_curved_power_law_fds.get(),
                curved_power_law_sis: self.gaussian_curved_power_law_sis.get(),
                curved_power_law_qs: self.gaussian_curved_power_law_qs.get(),
                curved_power_law_gps: self.gaussian_curved_power_law_gps.get(),
                num_lists: self
                    .gaussian_list_radecs
                    .len()
                    .try_into()
                    .expect("not bigger than i32::MAX"),
                list_lmns: self.gaussian_list_lmns.get(),
                list_fds: self.gaussian_list_fds.get(),
                list_gps: self.gaussian_list_gps.get(),
            },
            &self.get_addresses(),
            d_uvws.get(),
            d_beam_jones.get(),
            d_vis_fb.get_mut().cast(),
        )?;

        Ok(())
    }

    /// This function is mostly used for testing. For a single timestep, over
    /// the already-provided baselines and frequencies, generate visibilities
    /// for each specified sky-model Gaussian-source component. The
    /// `SkyModellerGpu` object *must* already have its UVW coordinates set; see
    /// [`SkyModellerGpu::set_uvws`].
    ///
    /// `lst_rad`: The local sidereal time in \[radians\].
    ///
    /// `array_latitude_rad`: The latitude of the array/telescope/interferometer
    /// in \[radians\].
    pub(super) unsafe fn model_shapelets(
        &self,
        lst_rad: f64,
        array_latitude_rad: f64,
        d_uvws: &DevicePointer<gpu::UVW>,
        d_beam_jones: &mut DevicePointer<GpuJones>,
        d_vis_fb: &mut DevicePointer<Jones<f32>>,
    ) -> Result<(), ModelError> {
        if self.shapelet_power_law_radecs.is_empty()
            && self.shapelet_curved_power_law_radecs.is_empty()
            && self.shapelet_list_radecs.is_empty()
        {
            return Ok(());
        }

        {
            let (azs, zas): (Vec<GpuFloat>, Vec<GpuFloat>) = self
                .shapelet_power_law_radecs
                .iter()
                .chain(self.shapelet_curved_power_law_radecs.iter())
                .chain(self.shapelet_list_radecs.iter())
                .map(|radec| {
                    let azel = radec.to_hadec(lst_rad).to_azel(array_latitude_rad);
                    (azel.az as GpuFloat, azel.za() as GpuFloat)
                })
                .unzip();
            d_beam_jones.realloc(
                self.gpu_beam.get_num_unique_tiles() as usize
                    * self.gpu_beam.get_num_unique_freqs() as usize
                    * azs.len()
                    * std::mem::size_of::<GpuJones>(),
            )?;
            let d_azs = DevicePointer::copy_to_device(&azs)?;
            let d_zas = DevicePointer::copy_to_device(&zas)?;
            self.gpu_beam
                .calc_jones_pair(&d_azs, &d_zas, array_latitude_rad, d_beam_jones)?
        };

        let uvs = self.get_shapelet_uvs(lst_rad);
        let power_law_uvs =
            DevicePointer::copy_to_device(uvs.power_law.as_slice().expect("is contiguous"))?;
        let curved_power_law_uvs =
            DevicePointer::copy_to_device(uvs.curved_power_law.as_slice().expect("is contiguous"))?;
        let list_uvs = DevicePointer::copy_to_device(uvs.list.as_slice().expect("is contiguous"))?;

        gpu_kernel_call!(
            gpu::model_shapelets,
            &gpu::Shapelets {
                num_power_laws: self
                    .shapelet_power_law_radecs
                    .len()
                    .try_into()
                    .expect("not bigger than i32::MAX"),
                power_law_lmns: self.shapelet_power_law_lmns.get(),
                power_law_fds: self.shapelet_power_law_fds.get(),
                power_law_sis: self.shapelet_power_law_sis.get(),
                power_law_gps: self.shapelet_power_law_gps.get(),
                power_law_shapelet_uvs: power_law_uvs.get(),
                power_law_shapelet_coeffs: self.shapelet_power_law_coeffs.get(),
                power_law_num_shapelet_coeffs: self.shapelet_power_law_coeff_lens.get(),
                num_curved_power_laws: self
                    .shapelet_curved_power_law_radecs
                    .len()
                    .try_into()
                    .expect("not bigger than i32::MAX"),
                curved_power_law_lmns: self.shapelet_curved_power_law_lmns.get(),
                curved_power_law_fds: self.shapelet_curved_power_law_fds.get(),
                curved_power_law_sis: self.shapelet_curved_power_law_sis.get(),
                curved_power_law_qs: self.shapelet_curved_power_law_qs.get(),
                curved_power_law_gps: self.shapelet_curved_power_law_gps.get(),
                curved_power_law_shapelet_uvs: curved_power_law_uvs.get(),
                curved_power_law_shapelet_coeffs: self.shapelet_curved_power_law_coeffs.get(),
                curved_power_law_num_shapelet_coeffs: self
                    .shapelet_curved_power_law_coeff_lens
                    .get(),
                num_lists: self
                    .shapelet_list_radecs
                    .len()
                    .try_into()
                    .expect("not bigger than i32::MAX"),
                list_lmns: self.shapelet_list_lmns.get(),
                list_fds: self.shapelet_list_fds.get(),
                list_gps: self.shapelet_list_gps.get(),
                list_shapelet_uvs: list_uvs.get(),
                list_shapelet_coeffs: self.shapelet_list_coeffs.get(),
                list_num_shapelet_coeffs: self.shapelet_list_coeff_lens.get(),
            },
            &self.get_addresses(),
            d_uvws.get(),
            d_beam_jones.get(),
            d_vis_fb.get_mut().cast(),
        )?;

        Ok(())
    }

    /// This is a "specialised" version of [`SkyModeller::model_timestep_with`];
    /// it accepts GPU buffers directly, saving some allocations. Unlike the
    /// aforementioned function, the incoming visibilities *are not* cleared;
    /// visibilities are accumulated.
    fn model_timestep_with(
        &self,
        lst_rad: f64,
        array_latitude_rad: f64,
        d_uvws: &DevicePointer<gpu::UVW>,
        d_beam_jones: &mut DevicePointer<GpuJones>,
        d_vis_fb: &mut DevicePointer<Jones<f32>>,
    ) -> Result<(), ModelError> {
        unsafe {
            self.model_points(lst_rad, array_latitude_rad, d_uvws, d_beam_jones, d_vis_fb)?;
            self.model_gaussians(lst_rad, array_latitude_rad, d_uvws, d_beam_jones, d_vis_fb)?;
            self.model_shapelets(lst_rad, array_latitude_rad, d_uvws, d_beam_jones, d_vis_fb)?;
        }

        Ok(())
    }

    /// For a timestamp, get the LST, [`UVW`]s and array latitude. These things
    /// depend on whether we're precessing, so rather than copy+pasting this
    /// code around the place, put it in one spot. The [`UVW`]s are stored in
    /// the supplied buffer, but are also returned in a new vector.
    fn get_lst_uvws_latitude(
        &self,
        timestamp: Epoch,
        d_uvws: &mut DevicePointer<gpu::UVW>,
    ) -> Result<(f64, Vec<UVW>, f64), GpuError> {
        let (lst, xyzs, latitude) = if self.precess {
            let precession_info = precess_time(
                self.array_longitude,
                self.array_latitude,
                self.phase_centre,
                timestamp,
                self.dut1,
            );
            // Apply precession to the tile XYZ positions.
            let precessed_tile_xyzs = precession_info.precess_xyz(self.unflagged_tile_xyzs);
            debug!(
                "Modelling GPS timestamp {}, LMST {}°, J2000 LMST {}°",
                timestamp.to_gpst_seconds(),
                precession_info.lmst.to_degrees(),
                precession_info.lmst_j2000.to_degrees()
            );
            (
                precession_info.lmst_j2000,
                Cow::from(precessed_tile_xyzs),
                precession_info.array_latitude_j2000,
            )
        } else {
            let lst = get_lmst(self.array_longitude, timestamp, self.dut1);
            debug!(
                "Modelling GPS timestamp {}, LMST {}°",
                timestamp.to_gpst_seconds(),
                lst.to_degrees()
            );
            (
                lst,
                Cow::from(self.unflagged_tile_xyzs),
                self.array_latitude,
            )
        };

        let uvws = xyzs_to_cross_uvws(&xyzs, self.phase_centre.to_hadec(lst));
        let gpu_uvws: Vec<gpu::UVW> = uvws
            .iter()
            .map(|&uvw| gpu::UVW {
                u: uvw.u as GpuFloat,
                v: uvw.v as GpuFloat,
                w: uvw.w as GpuFloat,
            })
            .collect();
        d_uvws.overwrite(&gpu_uvws)?;

        Ok((lst, uvws, latitude))
    }

    /// Get a populated [`gpu::Addresses`]. This should never outlive `self`.
    fn get_addresses(&self) -> gpu::Addresses {
        gpu::Addresses {
            num_freqs: self.num_freqs,
            num_vis: self.num_baselines * self.num_freqs,
            num_baselines: self.num_baselines,
            d_freqs: self.d_freqs.get(),
            d_shapelet_basis_values: self.d_shapelet_basis_values.get(),
            num_unique_beam_freqs: self.gpu_beam.get_num_unique_freqs(),
            d_tile_map: self.gpu_beam.get_tile_map(),
            d_freq_map: self.gpu_beam.get_freq_map(),
            d_tile_index_to_unflagged_tile_index_map: self
                .tile_index_to_unflagged_tile_index_map
                .get(),
        }
    }

    /// Shapelets need their own special kind of UVW coordinates. Each shapelet
    /// component's position is treated as the phase centre. This function uses
    /// the FFI type [`gpu::ShapeletUV`]; the W isn't actually used in
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

impl<'a> SkyModeller<'a> for SkyModellerGpu<'a> {
    fn model_timestep(
        &self,
        timestamp: Epoch,
    ) -> Result<(Array2<Jones<f32>>, Vec<UVW>), ModelError> {
        // The device buffers will automatically be resized.
        let mut d_uvws = DevicePointer::default();
        let (lst, uvws, latitude) = self.get_lst_uvws_latitude(timestamp, &mut d_uvws)?;

        let mut vis_fb = Array2::zeros((self.num_freqs as usize, self.num_baselines as usize));
        let mut d_vis_fb =
            DevicePointer::copy_to_device(vis_fb.as_slice().expect("is contiguous"))?;
        let mut d_beam_jones = DevicePointer::default();

        self.model_timestep_with(lst, latitude, &d_uvws, &mut d_beam_jones, &mut d_vis_fb)?;
        d_vis_fb.copy_from_device(vis_fb.as_slice_mut().expect("is contiguous"))?;

        Ok((vis_fb, uvws))
    }

    fn model_timestep_with(
        &self,
        timestamp: Epoch,
        mut vis_fb: ArrayViewMut2<Jones<f32>>,
    ) -> Result<Vec<UVW>, ModelError> {
        // The device buffers will automatically be resized.
        let mut d_uvws = DevicePointer::default();
        let (lst, uvws, latitude) = self.get_lst_uvws_latitude(timestamp, &mut d_uvws)?;

        let mut d_vis_fb =
            DevicePointer::copy_to_device(vis_fb.as_slice().expect("is contiguous"))?;
        let mut d_beam_jones = DevicePointer::default();

        self.model_timestep_with(lst, latitude, &d_uvws, &mut d_beam_jones, &mut d_vis_fb)?;
        d_vis_fb.copy_from_device(vis_fb.as_slice_mut().expect("is contiguous"))?;

        // Mask any unavailable polarisations.
        mask_pols(vis_fb, self.pols);

        Ok(uvws)
    }
}

/// The return type of [SkyModellerGpu::get_shapelet_uvs]. These arrays have
/// baseline as the first axis and component as the second.
pub(super) struct ShapeletUVs {
    power_law: Array2<gpu::ShapeletUV>,
    curved_power_law: Array2<gpu::ShapeletUV>,
    pub(super) list: Array2<gpu::ShapeletUV>,
}

fn get_shapelet_uvs_inner(
    radecs: &[RADec],
    lst_rad: f64,
    tile_xyzs: &[XyzGeodetic],
) -> Array2<gpu::ShapeletUV> {
    let n = tile_xyzs.len();
    let num_baselines = (n * (n - 1)) / 2;

    let mut shapelet_uvs: Array2<gpu::ShapeletUV> = Array2::from_elem(
        (num_baselines, radecs.len()),
        gpu::ShapeletUV { u: 0.0, v: 0.0 },
    );
    shapelet_uvs
        .axis_iter_mut(Axis(1))
        .zip(radecs.iter())
        .for_each(|(mut baseline_uv, radec)| {
            let hadec = radec.to_hadec(lst_rad);
            let shapelet_uvs: Vec<gpu::ShapeletUV> = xyzs_to_cross_uvws(tile_xyzs, hadec)
                .into_iter()
                .map(|uvw| gpu::ShapeletUV {
                    u: uvw.u as GpuFloat,
                    v: uvw.v as GpuFloat,
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
    shapelet_coeffs: Vec<&[ShapeletCoeff]>,
) -> (Vec<gpu::ShapeletCoeff>, Vec<i32>) {
    let mut coeffs: Vec<gpu::ShapeletCoeff> = vec![];
    let mut coeff_lengths = Vec::with_capacity(coeffs.len());

    for coeffs_for_comp in shapelet_coeffs {
        coeff_lengths.push(
            coeffs_for_comp
                .len()
                .try_into()
                .expect("not bigger than i32::MAX"),
        );
        for &ShapeletCoeff { n1, n2, value } in coeffs_for_comp {
            coeffs.push(gpu::ShapeletCoeff {
                n1,
                n2,
                value: value as GpuFloat,
            })
        }
    }

    coeffs.shrink_to_fit();
    (coeffs, coeff_lengths)
}
