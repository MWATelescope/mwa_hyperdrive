// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to generate sky-model visibilities with CUDA/HIP.

use std::{
    collections::HashSet,
    ops::Range,
    sync::atomic::{AtomicBool, Ordering},
};

use hifitime::{Duration, Epoch};
use marlu::{pos::xyz::xyzs_to_cross_uvws, Jones, LmnRime, RADec, XyzGeodetic, UVW};
use ndarray::prelude::*;

use super::{mask_pols, shapelets, ModelError, SkyModeller};
use crate::model::frame::{geometry_at, Geometry};
use crate::{
    beam::{Beam, BeamFreqSpline, BeamGpu},
    context::Polarisations,
    gpu::{self, gpu_kernel_call, DevicePointer, GpuError, GpuFloat, GpuJones},
    srclist::{ComponentType, FluxDensityType, ShapeletCoeff, Source, SourceComponent, SourceList},
};
use marlu::{PolConvention, UvwFrame};

/// The first axis of `*_list_fds` is the source component, the second is
/// unflagged fine channel frequency. The length of `*_radecs`, `*_lmns` and
/// `*_list_fds`'s first axis are the same.
///
/// The device buffers that grow with the number of components and that are
/// only needed while modelling (beam responses and shapelet UVs) are made for
/// one chunk of components at a time, so that sky models with many components
/// fit in device memory; see [`SkyModellerGpu::comp_chunks`].
pub struct SkyModellerGpu<'a> {
    /// The trait object to use for beam calculations.
    gpu_beam: Box<dyn BeamGpu>,
    /// For each unflagged fine channel, the indices of the beam knot
    /// frequencies (as `gpu_beam` orders them) at or below and at or above it,
    /// and its four cubic-spline weights (see [`BeamFreqSpline`]). These let
    /// the beam responses be splined in frequency instead of stepping every
    /// coarse channel.
    d_beam_freq_lo: DevicePointer<i32>,
    d_beam_freq_hi: DevicePointer<i32>,
    d_beam_freq_weights: DevicePointer<GpuFloat>,
    /// The spline's second-derivative matrix, in `gpu_beam`'s frequency order,
    /// if any fine channel is between knots. Beam buffers then hold the
    /// knots' second derivatives after the responses.
    d_beam_spline_s: Option<DevicePointer<GpuFloat>>,

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
    /// How sky-model Stokes parameters map onto XX, XY, YX, YY.
    pol_convention: PolConvention,
    /// The J2000 frame (and baseline sign) of the modelled visibilities.
    uvw_frame: UvwFrame,

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

    freqs: &'a [f64],
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

    /// For each kind of shapelet, where each component's coefficients start
    /// in its flattened coefficient array, followed by the array's length.
    shapelet_power_law_coeff_offsets: Vec<usize>,
    shapelet_curved_power_law_coeff_offsets: Vec<usize>,
    shapelet_list_coeff_offsets: Vec<usize>,

    /// If set, model this many components per chunk instead of working it out
    /// from the free device memory. Only for testing.
    chunk_comps_override: Option<usize>,
    /// Has the user been told that the sky model is being chunked?
    chunking_reported: AtomicBool,
    /// Has the beam kernel been run once, so that the device memory it keeps
    /// for itself is already missing from the free memory?
    beam_workspace_reserved: AtomicBool,
}

// You're not re-using pointers after they've been sent to another thread,
// right?
unsafe impl Send for SkyModellerGpu<'_> {}

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
        unflagged_fine_chan_freqs: &'a [f64],
        flagged_tiles: &HashSet<usize>,
        phase_centre: RADec,
        array_longitude_rad: f64,
        array_latitude_rad: f64,
        dut1: Duration,
        apply_precession: bool,
        pol_convention: PolConvention,
        uvw_frame: UvwFrame,
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
                    let inst_fd: Jones<f64> = fd_at_150mhz.to_inst_stokes(pol_convention);
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
                    let inst_fd: Jones<f64> = fd_at_150mhz.to_inst_stokes(pol_convention);
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

        // Variables for CUDA/HIP. They're made flexible in their types for
        // whichever precision is being used.
        let unflagged_fine_chan_freqs_floats: Vec<GpuFloat> = unflagged_fine_chan_freqs
            .iter()
            .map(|&f| f as GpuFloat)
            .collect();
        let shapelet_basis_values: Vec<GpuFloat> = shapelets::SHAPELET_BASIS_VALUES
            .iter()
            .map(|&f| f as GpuFloat)
            .collect();

        let num_baselines = (unflagged_tile_xyzs.len() * (unflagged_tile_xyzs.len() - 1)) / 2;
        let num_freqs = unflagged_fine_chan_freqs.len();

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

        // The beam frequencies ("knots") to spline beam responses from, so
        // that they don't step every coarse channel. The GPU beam's frequency
        // map tells us where each knot landed in its beam responses.
        let spline = BeamFreqSpline::new(
            beam.get_beam_knots(unflagged_fine_chan_freqs),
            unflagged_fine_chan_freqs,
        );
        let knots_ints: Vec<u32> = spline.knots_hz.iter().map(|f| f.round() as u32).collect();
        let gpu_beam = beam.prepare_gpu_beam(&knots_ints)?;
        let knot_cols: Vec<usize> =
            unsafe { gpu::copy_raw_from_device::<i32>(gpu_beam.get_freq_map(), knots_ints.len())? }
                .into_iter()
                .map(|c| c as usize)
                .collect();
        let num_knots = knot_cols.len();
        let d_beam_spline_s = match &spline.second_derivs {
            Some(s) => {
                // Each knot must have its own column, in the order of the
                // GPU beam's columns.
                assert_eq!(gpu_beam.get_num_unique_freqs() as usize, num_knots);
                let mut s_cols = vec![0.0 as GpuFloat; num_knots * num_knots];
                for (k, &c_k) in knot_cols.iter().enumerate() {
                    for (j, &c_j) in knot_cols.iter().enumerate() {
                        s_cols[c_k * num_knots + c_j] = s[(k, j)] as GpuFloat;
                    }
                }
                Some(DevicePointer::copy_to_device(&s_cols)?)
            }
            None => None,
        };
        let mut beam_freq_lo = Vec::with_capacity(spline.interp.len());
        let mut beam_freq_hi = Vec::with_capacity(spline.interp.len());
        let mut beam_freq_weights = Vec::with_capacity(4 * spline.interp.len());
        for &(lo, hi, w) in &spline.interp {
            beam_freq_lo.push(knot_cols[lo] as i32);
            beam_freq_hi.push(knot_cols[hi] as i32);
            beam_freq_weights.extend(w.map(|w| w as GpuFloat));
        }
        let d_beam_freq_lo = DevicePointer::copy_to_device(&beam_freq_lo)?;
        let d_beam_freq_hi = DevicePointer::copy_to_device(&beam_freq_hi)?;
        let d_beam_freq_weights = DevicePointer::copy_to_device(&beam_freq_weights)?;

        let mut modeller = SkyModellerGpu {
            gpu_beam,
            d_beam_freq_lo,
            d_beam_freq_hi,
            d_beam_freq_weights,
            d_beam_spline_s,

            phase_centre,
            array_longitude: array_longitude_rad,
            array_latitude: array_latitude_rad,
            dut1,
            precess: apply_precession,
            pol_convention,
            uvw_frame,

            unflagged_tile_xyzs,
            num_baselines: num_baselines.try_into().expect("not bigger than i32::MAX"),
            num_freqs: num_freqs.try_into().expect("not bigger than i32::MAX"),

            pols,

            tile_index_to_unflagged_tile_index_map: d_tile_index_to_unflagged_tile_index_map,

            freqs: unflagged_fine_chan_freqs,
            d_freqs,
            d_shapelet_basis_values,

            point_power_law_radecs: vec![],
            point_power_law_lmns: DevicePointer::default(),
            point_power_law_fds: DevicePointer::default(),
            point_power_law_sis: DevicePointer::default(),

            point_curved_power_law_radecs: vec![],
            point_curved_power_law_lmns: DevicePointer::default(),
            point_curved_power_law_fds: DevicePointer::default(),
            point_curved_power_law_sis: DevicePointer::default(),
            point_curved_power_law_qs: DevicePointer::default(),

            point_list_radecs: vec![],
            point_list_lmns: DevicePointer::default(),
            point_list_fds: DevicePointer::default(),

            gaussian_power_law_radecs: vec![],
            gaussian_power_law_lmns: DevicePointer::default(),
            gaussian_power_law_fds: DevicePointer::default(),
            gaussian_power_law_sis: DevicePointer::default(),
            gaussian_power_law_gps: DevicePointer::default(),

            gaussian_curved_power_law_radecs: vec![],
            gaussian_curved_power_law_lmns: DevicePointer::default(),
            gaussian_curved_power_law_fds: DevicePointer::default(),
            gaussian_curved_power_law_sis: DevicePointer::default(),
            gaussian_curved_power_law_qs: DevicePointer::default(),
            gaussian_curved_power_law_gps: DevicePointer::default(),

            gaussian_list_radecs: vec![],
            gaussian_list_lmns: DevicePointer::default(),
            gaussian_list_fds: DevicePointer::default(),
            gaussian_list_gps: DevicePointer::default(),

            shapelet_power_law_radecs: vec![],
            shapelet_power_law_lmns: DevicePointer::default(),
            shapelet_power_law_fds: DevicePointer::default(),
            shapelet_power_law_sis: DevicePointer::default(),
            shapelet_power_law_gps: DevicePointer::default(),
            shapelet_power_law_coeffs: DevicePointer::default(),
            shapelet_power_law_coeff_lens: DevicePointer::default(),

            shapelet_curved_power_law_radecs: vec![],
            shapelet_curved_power_law_lmns: DevicePointer::default(),
            shapelet_curved_power_law_fds: DevicePointer::default(),
            shapelet_curved_power_law_sis: DevicePointer::default(),
            shapelet_curved_power_law_qs: DevicePointer::default(),
            shapelet_curved_power_law_gps: DevicePointer::default(),
            shapelet_curved_power_law_coeffs: DevicePointer::default(),
            shapelet_curved_power_law_coeff_lens: DevicePointer::default(),

            shapelet_list_radecs: vec![],
            shapelet_list_lmns: DevicePointer::default(),
            shapelet_list_fds: DevicePointer::default(),
            shapelet_list_gps: DevicePointer::default(),
            shapelet_list_coeffs: DevicePointer::default(),
            shapelet_list_coeff_lens: DevicePointer::default(),

            shapelet_power_law_coeff_offsets: vec![0],
            shapelet_curved_power_law_coeff_offsets: vec![0],
            shapelet_list_coeff_offsets: vec![0],

            chunk_comps_override: None,
            chunking_reported: AtomicBool::new(false),
            beam_workspace_reserved: AtomicBool::new(false),
        };
        modeller.update_source_list(
            source_list
                .values()
                .rev()
                .flat_map(|src| src.components.iter()),
            phase_centre,
        )?;
        Ok(modeller)
    }

    fn update_source_list<'b, I>(
        &mut self,
        components: I,
        phase_centre: RADec,
    ) -> Result<(), ModelError>
    where
        I: IntoIterator<Item = &'b SourceComponent>,
    {
        self.phase_centre = phase_centre;

        self.point_power_law_radecs.clear();
        let mut point_power_law_lmns: Vec<gpu::LmnRime> = vec![];
        let mut point_power_law_fds: Vec<_> = vec![];
        let mut point_power_law_sis: Vec<_> = vec![];

        self.point_curved_power_law_radecs.clear();
        let mut point_curved_power_law_lmns: Vec<gpu::LmnRime> = vec![];
        let mut point_curved_power_law_fds: Vec<_> = vec![];
        let mut point_curved_power_law_sis: Vec<_> = vec![];
        let mut point_curved_power_law_qs: Vec<_> = vec![];

        self.point_list_radecs.clear();
        let mut point_list_lmns: Vec<gpu::LmnRime> = vec![];
        let mut point_list_fds: Vec<&FluxDensityType> = vec![];

        self.gaussian_power_law_radecs.clear();
        let mut gaussian_power_law_lmns: Vec<gpu::LmnRime> = vec![];
        let mut gaussian_power_law_fds: Vec<_> = vec![];
        let mut gaussian_power_law_sis: Vec<_> = vec![];
        let mut gaussian_power_law_gps: Vec<gpu::GaussianParams> = vec![];

        self.gaussian_curved_power_law_radecs.clear();
        let mut gaussian_curved_power_law_lmns: Vec<gpu::LmnRime> = vec![];
        let mut gaussian_curved_power_law_fds: Vec<_> = vec![];
        let mut gaussian_curved_power_law_sis: Vec<_> = vec![];
        let mut gaussian_curved_power_law_qs: Vec<_> = vec![];
        let mut gaussian_curved_power_law_gps: Vec<gpu::GaussianParams> = vec![];

        self.gaussian_list_radecs.clear();
        let mut gaussian_list_lmns: Vec<gpu::LmnRime> = vec![];
        let mut gaussian_list_fds: Vec<&FluxDensityType> = vec![];
        let mut gaussian_list_gps: Vec<gpu::GaussianParams> = vec![];

        self.shapelet_power_law_radecs.clear();
        let mut shapelet_power_law_lmns: Vec<gpu::LmnRime> = vec![];
        let mut shapelet_power_law_fds: Vec<_> = vec![];
        let mut shapelet_power_law_sis: Vec<_> = vec![];
        let mut shapelet_power_law_gps: Vec<gpu::GaussianParams> = vec![];
        let mut shapelet_power_law_coeffs: Vec<&[ShapeletCoeff]> = vec![];

        self.shapelet_curved_power_law_radecs.clear();
        let mut shapelet_curved_power_law_lmns: Vec<gpu::LmnRime> = vec![];
        let mut shapelet_curved_power_law_fds: Vec<_> = vec![];
        let mut shapelet_curved_power_law_sis: Vec<_> = vec![];
        let mut shapelet_curved_power_law_qs: Vec<_> = vec![];
        let mut shapelet_curved_power_law_gps: Vec<gpu::GaussianParams> = vec![];
        let mut shapelet_curved_power_law_coeffs: Vec<&[ShapeletCoeff]> = vec![];

        self.shapelet_list_radecs.clear();
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
        for comp in components {
            let radec = comp.radec;
            let LmnRime { l, m, n } = radec.to_lmn(phase_centre).prepare_for_rime();
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
                    let inst_fd: Jones<f64> = fd_at_150mhz.to_inst_stokes(self.pol_convention);
                    let gpu_inst_fd = jones_to_gpu_jones(inst_fd);

                    match &comp.comp_type {
                        ComponentType::Point => {
                            self.point_power_law_radecs.push(radec);
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
                            self.gaussian_power_law_radecs.push(radec);
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
                            self.shapelet_power_law_radecs.push(radec);
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
                    let inst_fd: Jones<f64> = fd_at_150mhz.to_inst_stokes(self.pol_convention);
                    let gpu_inst_fd = jones_to_gpu_jones(inst_fd);

                    // A new SI is needed when changing the reference freq.
                    // Thanks Jack.
                    #[cfg(feature = "gpu-single")]
                    let power_law_fd_ref_freq: f64 = gpu::POWER_LAW_FD_REF_FREQ.into();
                    #[cfg(not(feature = "gpu-single"))]
                    let power_law_fd_ref_freq: f64 = gpu::POWER_LAW_FD_REF_FREQ;
                    let si = if fd.freq == power_law_fd_ref_freq {
                        *si
                    } else {
                        let logratio = (fd.freq / power_law_fd_ref_freq).ln();
                        ((fd.i / fd_at_150mhz.i).ln() - q * logratio.powi(2)) / logratio
                    };

                    match &comp.comp_type {
                        ComponentType::Point => {
                            self.point_curved_power_law_radecs.push(radec);
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
                            self.gaussian_curved_power_law_radecs.push(radec);
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
                            self.shapelet_curved_power_law_radecs.push(radec);
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
                        self.point_list_radecs.push(radec);
                        point_list_lmns.push(lmn);
                        point_list_fds.push(&comp.flux_type);
                    }

                    ComponentType::Gaussian { maj, min, pa } => {
                        let gp = gpu::GaussianParams {
                            maj: *maj as GpuFloat,
                            min: *min as GpuFloat,
                            pa: *pa as GpuFloat,
                        };
                        self.gaussian_list_radecs.push(radec);
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
                        self.shapelet_list_radecs.push(radec);
                        shapelet_list_lmns.push(lmn);
                        shapelet_list_gps.push(gp);
                        shapelet_list_coeffs.push(coeffs);
                        shapelet_list_fds.push(&comp.flux_type);
                    }
                },
            }
        }

        let (shapelet_power_law_coeffs, shapelet_power_law_coeff_lens) =
            get_flattened_coeffs(shapelet_power_law_coeffs);
        let (shapelet_curved_power_law_coeffs, shapelet_curved_power_law_coeff_lens) =
            get_flattened_coeffs(shapelet_curved_power_law_coeffs);
        let (shapelet_list_coeffs, shapelet_list_coeff_lens) =
            get_flattened_coeffs(shapelet_list_coeffs);

        self.point_power_law_lmns.overwrite(&point_power_law_lmns)?;
        self.point_power_law_fds.overwrite(&point_power_law_fds)?;
        self.point_power_law_sis.overwrite(&point_power_law_sis)?;

        self.point_curved_power_law_lmns
            .overwrite(&point_curved_power_law_lmns)?;
        self.point_curved_power_law_fds
            .overwrite(&point_curved_power_law_fds)?;
        self.point_curved_power_law_sis
            .overwrite(&point_curved_power_law_sis)?;
        self.point_curved_power_law_qs
            .overwrite(&point_curved_power_law_qs)?;

        self.point_list_lmns.overwrite(&point_list_lmns)?;
        list_fds_to_device(
            &mut self.point_list_fds,
            &point_list_fds,
            self.freqs,
            self.pol_convention,
        )?;

        self.gaussian_power_law_lmns
            .overwrite(&gaussian_power_law_lmns)?;
        self.gaussian_power_law_fds
            .overwrite(&gaussian_power_law_fds)?;
        self.gaussian_power_law_sis
            .overwrite(&gaussian_power_law_sis)?;
        self.gaussian_power_law_gps
            .overwrite(&gaussian_power_law_gps)?;

        self.gaussian_curved_power_law_lmns
            .overwrite(&gaussian_curved_power_law_lmns)?;
        self.gaussian_curved_power_law_fds
            .overwrite(&gaussian_curved_power_law_fds)?;
        self.gaussian_curved_power_law_sis
            .overwrite(&gaussian_curved_power_law_sis)?;
        self.gaussian_curved_power_law_qs
            .overwrite(&gaussian_curved_power_law_qs)?;
        self.gaussian_curved_power_law_gps
            .overwrite(&gaussian_curved_power_law_gps)?;

        self.gaussian_list_lmns.overwrite(&gaussian_list_lmns)?;
        list_fds_to_device(
            &mut self.gaussian_list_fds,
            &gaussian_list_fds,
            self.freqs,
            self.pol_convention,
        )?;
        self.gaussian_list_gps.overwrite(&gaussian_list_gps)?;

        self.shapelet_power_law_lmns
            .overwrite(&shapelet_power_law_lmns)?;
        self.shapelet_power_law_fds
            .overwrite(&shapelet_power_law_fds)?;
        self.shapelet_power_law_sis
            .overwrite(&shapelet_power_law_sis)?;
        self.shapelet_power_law_gps
            .overwrite(&shapelet_power_law_gps)?;
        self.shapelet_power_law_coeffs
            .overwrite(&shapelet_power_law_coeffs)?;
        self.shapelet_power_law_coeff_lens
            .overwrite(&shapelet_power_law_coeff_lens)?;

        self.shapelet_curved_power_law_lmns
            .overwrite(&shapelet_curved_power_law_lmns)?;
        self.shapelet_curved_power_law_fds
            .overwrite(&shapelet_curved_power_law_fds)?;
        self.shapelet_curved_power_law_sis
            .overwrite(&shapelet_curved_power_law_sis)?;
        self.shapelet_curved_power_law_qs
            .overwrite(&shapelet_curved_power_law_qs)?;
        self.shapelet_curved_power_law_gps
            .overwrite(&shapelet_curved_power_law_gps)?;
        self.shapelet_curved_power_law_coeffs
            .overwrite(&shapelet_curved_power_law_coeffs)?;
        self.shapelet_curved_power_law_coeff_lens
            .overwrite(&shapelet_curved_power_law_coeff_lens)?;

        self.shapelet_list_lmns.overwrite(&shapelet_list_lmns)?;
        list_fds_to_device(
            &mut self.shapelet_list_fds,
            &shapelet_list_fds,
            self.freqs,
            self.pol_convention,
        )?;
        self.shapelet_list_gps.overwrite(&shapelet_list_gps)?;
        self.shapelet_list_coeffs.overwrite(&shapelet_list_coeffs)?;
        self.shapelet_list_coeff_lens
            .overwrite(&shapelet_list_coeff_lens)?;

        self.shapelet_power_law_coeff_offsets = coeff_offsets(&shapelet_power_law_coeff_lens);
        self.shapelet_curved_power_law_coeff_offsets =
            coeff_offsets(&shapelet_curved_power_law_coeff_lens);
        self.shapelet_list_coeff_offsets = coeff_offsets(&shapelet_list_coeff_lens);

        Ok(())
    }

    /// Model this many components per chunk, rather than as many as fit in
    /// the free device memory.
    #[cfg(test)]
    pub(super) fn set_chunk_comps(&mut self, num_comps: usize) {
        self.chunk_comps_override = Some(num_comps);
    }

    /// Split one component type's components (`counts` power laws, curved
    /// power laws and lists) into chunks to be modelled one after the other.
    /// A chunk is as many components as have their modelling buffers fit in
    /// the device memory that is free now (see [`gpu::model_chunk_budget`]),
    /// but at least one. `bytes_per_comp` is the device memory those buffers
    /// need for each component, and `reusable_bytes` the size of buffers that
    /// are already allocated for them.
    fn comp_chunks(
        &self,
        counts: [usize; 3],
        bytes_per_comp: usize,
        reusable_bytes: usize,
    ) -> Result<Vec<CompChunk>, ModelError> {
        let num_comps: usize = counts.iter().sum();
        if num_comps == 0 {
            return Ok(vec![]);
        }
        let comps_per_chunk = match self.chunk_comps_override {
            Some(n) => n,
            None => {
                self.reserve_beam_workspace()?;
                gpu::model_chunk_budget(reusable_bytes)? / bytes_per_comp.max(1)
            }
        }
        .clamp(1, num_comps);
        if comps_per_chunk < num_comps && !self.chunking_reported.swap(true, Ordering::Relaxed) {
            log::info!(
                "Modelling {num_comps} sky-model components in chunks of {comps_per_chunk} to fit in GPU memory"
            );
        }
        Ok(split_into_chunks(counts, comps_per_chunk))
    }

    /// Run the beam kernel on a single direction, the first time this is
    /// called.
    ///
    /// A beam kernel can need device memory of its own that is only taken when
    /// it first runs, and is then kept. The FEE beam kernel has tens of
    /// kilobytes of local arrays per thread, and CUDA reserves that much stack
    /// for every thread the device can run at once, which is gigabytes.
    /// Running the kernel before the free device memory is measured keeps that
    /// memory out of the budget for a chunk of components.
    fn reserve_beam_workspace(&self) -> Result<(), ModelError> {
        if self.beam_workspace_reserved.swap(true, Ordering::Relaxed) {
            return Ok(());
        }
        let mut d_beam_jones = DevicePointer::default();
        unsafe { self.calc_beam_rows(&[0.0], &[0.0], 0.0, &mut d_beam_jones) }
    }

    /// The device memory \[bytes\] needed for the beam responses of each
    /// component: [`Self::num_beam_rows`] responses, and its azimuth and
    /// zenith angle.
    fn beam_bytes_per_comp(&self) -> usize {
        self.num_beam_rows() * std::mem::size_of::<GpuJones>() + 2 * std::mem::size_of::<GpuFloat>()
    }

    fn point_counts(&self) -> [usize; 3] {
        [
            self.point_power_law_radecs.len(),
            self.point_curved_power_law_radecs.len(),
            self.point_list_radecs.len(),
        ]
    }

    fn gaussian_counts(&self) -> [usize; 3] {
        [
            self.gaussian_power_law_radecs.len(),
            self.gaussian_curved_power_law_radecs.len(),
            self.gaussian_list_radecs.len(),
        ]
    }

    fn shapelet_counts(&self) -> [usize; 3] {
        [
            self.shapelet_power_law_radecs.len(),
            self.shapelet_curved_power_law_radecs.len(),
            self.shapelet_list_radecs.len(),
        ]
    }

    /// The kernels' description of a chunk of the point components.
    fn points(&self, c: &CompChunk) -> gpu::Points {
        gpu::Points {
            num_power_laws: to_i32(c.power_law.len()),
            power_law_lmns: at(&self.point_power_law_lmns, c.power_law.start),
            power_law_fds: at(&self.point_power_law_fds, c.power_law.start),
            power_law_sis: at(&self.point_power_law_sis, c.power_law.start),
            num_curved_power_laws: to_i32(c.curved_power_law.len()),
            curved_power_law_lmns: at(&self.point_curved_power_law_lmns, c.curved_power_law.start),
            curved_power_law_fds: at(&self.point_curved_power_law_fds, c.curved_power_law.start),
            curved_power_law_sis: at(&self.point_curved_power_law_sis, c.curved_power_law.start),
            curved_power_law_qs: at(&self.point_curved_power_law_qs, c.curved_power_law.start),
            num_lists: to_i32(c.list.len()),
            list_lmns: at(&self.point_list_lmns, c.list.start),
            list_fds: at(&self.point_list_fds, c.list.start * self.freqs.len()),
        }
    }

    /// The kernels' description of a chunk of the Gaussian components.
    fn gaussians(&self, c: &CompChunk) -> gpu::Gaussians {
        gpu::Gaussians {
            num_power_laws: to_i32(c.power_law.len()),
            power_law_lmns: at(&self.gaussian_power_law_lmns, c.power_law.start),
            power_law_fds: at(&self.gaussian_power_law_fds, c.power_law.start),
            power_law_sis: at(&self.gaussian_power_law_sis, c.power_law.start),
            power_law_gps: at(&self.gaussian_power_law_gps, c.power_law.start),
            num_curved_power_laws: to_i32(c.curved_power_law.len()),
            curved_power_law_lmns: at(
                &self.gaussian_curved_power_law_lmns,
                c.curved_power_law.start,
            ),
            curved_power_law_fds: at(
                &self.gaussian_curved_power_law_fds,
                c.curved_power_law.start,
            ),
            curved_power_law_sis: at(
                &self.gaussian_curved_power_law_sis,
                c.curved_power_law.start,
            ),
            curved_power_law_qs: at(&self.gaussian_curved_power_law_qs, c.curved_power_law.start),
            curved_power_law_gps: at(
                &self.gaussian_curved_power_law_gps,
                c.curved_power_law.start,
            ),
            num_lists: to_i32(c.list.len()),
            list_lmns: at(&self.gaussian_list_lmns, c.list.start),
            list_fds: at(&self.gaussian_list_fds, c.list.start * self.freqs.len()),
            list_gps: at(&self.gaussian_list_gps, c.list.start),
        }
    }

    /// The kernels' description of a chunk of the shapelet components. `uvs`
    /// are the chunk's shapelet UVs (power laws, curved power laws, lists).
    fn shapelets(&self, c: &CompChunk, uvs: [*const gpu::ShapeletUV; 3]) -> gpu::Shapelets {
        let [power_law_uvs, curved_power_law_uvs, list_uvs] = uvs;
        gpu::Shapelets {
            num_power_laws: to_i32(c.power_law.len()),
            power_law_lmns: at(&self.shapelet_power_law_lmns, c.power_law.start),
            power_law_fds: at(&self.shapelet_power_law_fds, c.power_law.start),
            power_law_sis: at(&self.shapelet_power_law_sis, c.power_law.start),
            power_law_gps: at(&self.shapelet_power_law_gps, c.power_law.start),
            power_law_shapelet_uvs: power_law_uvs,
            power_law_shapelet_coeffs: at(
                &self.shapelet_power_law_coeffs,
                self.shapelet_power_law_coeff_offsets[c.power_law.start],
            ),
            power_law_num_shapelet_coeffs: at(
                &self.shapelet_power_law_coeff_lens,
                c.power_law.start,
            ),
            num_curved_power_laws: to_i32(c.curved_power_law.len()),
            curved_power_law_lmns: at(
                &self.shapelet_curved_power_law_lmns,
                c.curved_power_law.start,
            ),
            curved_power_law_fds: at(
                &self.shapelet_curved_power_law_fds,
                c.curved_power_law.start,
            ),
            curved_power_law_sis: at(
                &self.shapelet_curved_power_law_sis,
                c.curved_power_law.start,
            ),
            curved_power_law_qs: at(&self.shapelet_curved_power_law_qs, c.curved_power_law.start),
            curved_power_law_gps: at(
                &self.shapelet_curved_power_law_gps,
                c.curved_power_law.start,
            ),
            curved_power_law_shapelet_uvs: curved_power_law_uvs,
            curved_power_law_shapelet_coeffs: at(
                &self.shapelet_curved_power_law_coeffs,
                self.shapelet_curved_power_law_coeff_offsets[c.curved_power_law.start],
            ),
            curved_power_law_num_shapelet_coeffs: at(
                &self.shapelet_curved_power_law_coeff_lens,
                c.curved_power_law.start,
            ),
            num_lists: to_i32(c.list.len()),
            list_lmns: at(&self.shapelet_list_lmns, c.list.start),
            list_fds: at(&self.shapelet_list_fds, c.list.start * self.freqs.len()),
            list_gps: at(&self.shapelet_list_gps, c.list.start),
            list_shapelet_uvs: list_uvs,
            list_shapelet_coeffs: at(
                &self.shapelet_list_coeffs,
                self.shapelet_list_coeff_offsets[c.list.start],
            ),
            list_num_shapelet_coeffs: at(&self.shapelet_list_coeff_lens, c.list.start),
        }
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
        self.model_points_inner(
            lst_rad,
            array_latitude_rad,
            d_uvws,
            d_beam_jones,
            None,
            d_vis_fb,
        )
    }

    /// [`Self::model_points`], optionally with beam responses that were already
    /// computed (see [`Self::model_timestep_with_beams`]).
    unsafe fn model_points_inner(
        &self,
        lst_rad: f64,
        array_latitude_rad: f64,
        d_uvws: &DevicePointer<gpu::UVW>,
        d_beam_jones: &mut DevicePointer<GpuJones>,
        beam: Option<*const GpuJones>,
        d_vis_fb: &mut DevicePointer<Jones<f32>>,
    ) -> Result<(), ModelError> {
        let counts = self.point_counts();
        let chunks = match beam {
            // The beam responses given are of all the components.
            Some(_) => split_into_chunks(counts, usize::MAX),
            None => {
                self.comp_chunks(counts, self.beam_bytes_per_comp(), d_beam_jones.get_size())?
            }
        };

        for c in &chunks {
            let d_beam = self.beam_responses(
                self.point_power_law_radecs[c.power_law.clone()]
                    .iter()
                    .chain(&self.point_curved_power_law_radecs[c.curved_power_law.clone()])
                    .chain(&self.point_list_radecs[c.list.clone()]),
                lst_rad,
                array_latitude_rad,
                d_beam_jones,
                beam,
            )?;

            gpu_kernel_call!(
                gpu::model_points,
                &self.points(c),
                &self.get_addresses(),
                d_uvws.get(),
                d_beam,
                d_vis_fb.get_mut().cast(),
            )?;
        }

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
        self.model_gaussians_inner(
            lst_rad,
            array_latitude_rad,
            d_uvws,
            d_beam_jones,
            None,
            d_vis_fb,
        )
    }

    /// [`Self::model_gaussians`], optionally with beam responses that were already
    /// computed (see [`Self::model_timestep_with_beams`]).
    unsafe fn model_gaussians_inner(
        &self,
        lst_rad: f64,
        array_latitude_rad: f64,
        d_uvws: &DevicePointer<gpu::UVW>,
        d_beam_jones: &mut DevicePointer<GpuJones>,
        beam: Option<*const GpuJones>,
        d_vis_fb: &mut DevicePointer<Jones<f32>>,
    ) -> Result<(), ModelError> {
        let counts = self.gaussian_counts();
        let chunks = match beam {
            // The beam responses given are of all the components.
            Some(_) => split_into_chunks(counts, usize::MAX),
            None => {
                self.comp_chunks(counts, self.beam_bytes_per_comp(), d_beam_jones.get_size())?
            }
        };

        for c in &chunks {
            let d_beam = self.beam_responses(
                self.gaussian_power_law_radecs[c.power_law.clone()]
                    .iter()
                    .chain(&self.gaussian_curved_power_law_radecs[c.curved_power_law.clone()])
                    .chain(&self.gaussian_list_radecs[c.list.clone()]),
                lst_rad,
                array_latitude_rad,
                d_beam_jones,
                beam,
            )?;

            gpu_kernel_call!(
                gpu::model_gaussians,
                &self.gaussians(c),
                &self.get_addresses(),
                d_uvws.get(),
                d_beam,
                d_vis_fb.get_mut().cast(),
            )?;
        }

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
        self.model_shapelets_inner(
            lst_rad,
            array_latitude_rad,
            d_uvws,
            d_beam_jones,
            None,
            d_vis_fb,
        )
    }

    /// [`Self::model_shapelets`], optionally with beam responses that were already
    /// computed (see [`Self::model_timestep_with_beams`]).
    unsafe fn model_shapelets_inner(
        &self,
        lst_rad: f64,
        array_latitude_rad: f64,
        d_uvws: &DevicePointer<gpu::UVW>,
        d_beam_jones: &mut DevicePointer<GpuJones>,
        beam: Option<*const GpuJones>,
        d_vis_fb: &mut DevicePointer<Jones<f32>>,
    ) -> Result<(), ModelError> {
        let counts = self.shapelet_counts();
        let chunks = match beam {
            // The beam responses given are of all the components.
            Some(_) => split_into_chunks(counts, usize::MAX),
            // As well as its beam responses, a shapelet component needs UVs
            // for every baseline.
            None => self.comp_chunks(
                counts,
                self.beam_bytes_per_comp()
                    + self.num_baselines as usize * std::mem::size_of::<gpu::ShapeletUV>(),
                d_beam_jones.get_size(),
            )?,
        };

        for c in &chunks {
            let d_beam = self.beam_responses(
                self.shapelet_power_law_radecs[c.power_law.clone()]
                    .iter()
                    .chain(&self.shapelet_curved_power_law_radecs[c.curved_power_law.clone()])
                    .chain(&self.shapelet_list_radecs[c.list.clone()]),
                lst_rad,
                array_latitude_rad,
                d_beam_jones,
                beam,
            )?;

            let uvs = self.get_shapelet_uvs_of_chunk(lst_rad, c);
            let power_law_uvs =
                DevicePointer::copy_to_device(uvs.power_law.as_slice().expect("is contiguous"))?;
            let curved_power_law_uvs = DevicePointer::copy_to_device(
                uvs.curved_power_law.as_slice().expect("is contiguous"),
            )?;
            let list_uvs =
                DevicePointer::copy_to_device(uvs.list.as_slice().expect("is contiguous"))?;

            gpu_kernel_call!(
                gpu::model_shapelets,
                &self.shapelets(
                    c,
                    [
                        power_law_uvs.get(),
                        curved_power_law_uvs.get(),
                        list_uvs.get()
                    ]
                ),
                &self.get_addresses(),
                d_uvws.get(),
                d_beam,
                d_vis_fb.get_mut().cast(),
            )?;
        }

        Ok(())
    }

    /// This is a "specialised" version of [`SkyModeller::model_timestep_with`];
    /// it accepts GPU buffers directly, saving some allocations. Unlike the
    /// aforementioned function, the incoming visibilities *are not* cleared;
    /// visibilities are accumulated.
    pub(crate) fn model_timestep_with(
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

    /// [`Self::model_timestep_with`], but with the beam responses of the
    /// points, Gaussians and shapelets already on the device (in that order),
    /// each laid out as [`Self::calc_beam_jones`] lays them out for
    /// [`Self::component_radecs`]. Peeling models one source many times at
    /// the same timesteps, so it computes these once instead of on every call.
    ///
    /// # Safety
    ///
    /// Each pointer must hold [`Self::num_beam_rows`] times the number of
    /// components of that type `GpuJones`.
    pub(crate) unsafe fn model_timestep_with_beams(
        &self,
        lst_rad: f64,
        array_latitude_rad: f64,
        d_uvws: &DevicePointer<gpu::UVW>,
        beams: [*const GpuJones; 3],
        d_vis_fb: &mut DevicePointer<Jones<f32>>,
    ) -> Result<(), ModelError> {
        // Not used when the beam responses are provided.
        let mut d_unused = DevicePointer::default();
        let [points, gaussians, shapelets] = beams;
        self.model_points_inner(
            lst_rad,
            array_latitude_rad,
            d_uvws,
            &mut d_unused,
            Some(points),
            d_vis_fb,
        )?;
        self.model_gaussians_inner(
            lst_rad,
            array_latitude_rad,
            d_uvws,
            &mut d_unused,
            Some(gaussians),
            d_vis_fb,
        )?;
        self.model_shapelets_inner(
            lst_rad,
            array_latitude_rad,
            d_uvws,
            &mut d_unused,
            Some(shapelets),
            d_vis_fb,
        )?;
        Ok(())
    }

    /// The directions of the point, Gaussian and shapelet components, in the
    /// order the model kernels expect their beam responses.
    pub(crate) fn component_radecs(&self) -> [Vec<RADec>; 3] {
        let chain = |pl: &[RADec], cpl: &[RADec], list: &[RADec]| -> Vec<RADec> {
            pl.iter().chain(cpl).chain(list).copied().collect()
        };
        [
            chain(
                &self.point_power_law_radecs,
                &self.point_curved_power_law_radecs,
                &self.point_list_radecs,
            ),
            chain(
                &self.gaussian_power_law_radecs,
                &self.gaussian_curved_power_law_radecs,
                &self.gaussian_list_radecs,
            ),
            chain(
                &self.shapelet_power_law_radecs,
                &self.shapelet_curved_power_law_radecs,
                &self.shapelet_list_radecs,
            ),
        ]
    }

    /// The number of (unique tile, beam knot) rows of a beam-response buffer,
    /// doubled when the knots' second derivatives follow them. Each row holds
    /// one response per direction.
    pub(crate) fn num_beam_rows(&self) -> usize {
        let rows = self.gpu_beam.get_num_unique_tiles() as usize
            * self.gpu_beam.get_num_unique_freqs() as usize;
        if self.d_beam_spline_s.is_some() {
            2 * rows
        } else {
            rows
        }
    }

    /// Compute the beam responses towards `azs` and `zas` into `d_beam_jones`
    /// (re-allocated if necessary), followed by their second derivatives in
    /// frequency if the spline needs them.
    unsafe fn calc_beam_rows(
        &self,
        azs: &[GpuFloat],
        zas: &[GpuFloat],
        array_latitude_rad: f64,
        d_beam_jones: &mut DevicePointer<GpuJones>,
    ) -> Result<(), ModelError> {
        d_beam_jones.realloc(self.num_beam_rows() * azs.len() * std::mem::size_of::<GpuJones>())?;
        self.gpu_beam.calc_jones_pair(
            azs,
            zas,
            array_latitude_rad,
            d_beam_jones.get_mut().cast(),
        )?;
        if let Some(d_s) = &self.d_beam_spline_s {
            gpu_kernel_call!(
                gpu::beam_second_derivs,
                d_beam_jones.get_mut().cast(),
                self.gpu_beam.get_num_unique_tiles(),
                self.gpu_beam.get_num_unique_freqs(),
                azs.len().try_into().expect("not bigger than i32::MAX"),
                d_s.get(),
            )?;
        }
        Ok(())
    }

    /// Compute the beam responses towards `radecs` into `d_beam_jones`
    /// (re-allocated if necessary), [`Self::num_beam_rows`] rows of
    /// `radecs.len()` responses.
    pub(crate) fn calc_beam_jones(
        &self,
        radecs: &[RADec],
        lst_rad: f64,
        array_latitude_rad: f64,
        d_beam_jones: &mut DevicePointer<GpuJones>,
    ) -> Result<(), ModelError> {
        unsafe {
            self.beam_responses(
                radecs.iter(),
                lst_rad,
                array_latitude_rad,
                d_beam_jones,
                None,
            )?;
        }
        Ok(())
    }

    /// The device beam responses towards `radecs`: `beam` if given, otherwise
    /// computed into `d_beam_jones`.
    unsafe fn beam_responses<'r>(
        &self,
        radecs: impl Iterator<Item = &'r RADec>,
        lst_rad: f64,
        array_latitude_rad: f64,
        d_beam_jones: &mut DevicePointer<GpuJones>,
        beam: Option<*const GpuJones>,
    ) -> Result<*const GpuJones, ModelError> {
        if let Some(beam) = beam {
            return Ok(beam);
        }
        let (azs, zas): (Vec<GpuFloat>, Vec<GpuFloat>) = radecs
            .map(|radec| {
                let azel = radec.to_hadec(lst_rad).to_azel(array_latitude_rad);
                (azel.az as GpuFloat, azel.za() as GpuFloat)
            })
            .unzip();
        self.calc_beam_rows(&azs, &zas, array_latitude_rad, d_beam_jones)?;
        Ok(d_beam_jones.get())
    }

    /// This is a "specialised" version of [`SkyModeller::model_timestep_autos_with`];
    /// it accepts GPU buffers directly, saving some allocations. Unlike the
    /// aforementioned function, the incoming visibilities *are not* cleared;
    /// visibilities are accumulated.
    pub(crate) fn model_timestep_autos_with_inner(
        &self,
        lst_rad: f64,
        array_latitude_rad: f64,
        d_beam_jones: &mut DevicePointer<GpuJones>,
        d_vis_fb: &mut DevicePointer<Jones<f32>>,
    ) -> Result<(), ModelError> {
        let num_tiles = self.unflagged_tile_xyzs.len();
        let num_freqs = self.freqs.len();

        // The envelopes of Gaussians and shapelets are 1 for auto-correlations,
        // so all component types are modelled the same way. Model each type's
        // components in chunks; the kernel takes all three types at once, so
        // the other two are empty.
        let no_chunk = CompChunk::default();
        let no_uvs = [std::ptr::null(); 3]; // Not needed for autos
        let types: [(&[RADec], &[RADec], &[RADec]); 3] = [
            (
                self.point_power_law_radecs.as_slice(),
                self.point_curved_power_law_radecs.as_slice(),
                self.point_list_radecs.as_slice(),
            ),
            (
                self.gaussian_power_law_radecs.as_slice(),
                self.gaussian_curved_power_law_radecs.as_slice(),
                self.gaussian_list_radecs.as_slice(),
            ),
            (
                self.shapelet_power_law_radecs.as_slice(),
                self.shapelet_curved_power_law_radecs.as_slice(),
                self.shapelet_list_radecs.as_slice(),
            ),
        ];
        for (i_type, (power_law_radecs, curved_power_law_radecs, list_radecs)) in
            types.into_iter().enumerate()
        {
            let chunks = self.comp_chunks(
                [
                    power_law_radecs.len(),
                    curved_power_law_radecs.len(),
                    list_radecs.len(),
                ],
                self.beam_bytes_per_comp(),
                d_beam_jones.get_size(),
            )?;
            for c in &chunks {
                let d_beam = unsafe {
                    self.beam_responses(
                        power_law_radecs[c.power_law.clone()]
                            .iter()
                            .chain(&curved_power_law_radecs[c.curved_power_law.clone()])
                            .chain(&list_radecs[c.list.clone()]),
                        lst_rad,
                        array_latitude_rad,
                        d_beam_jones,
                        None,
                    )?
                };

                let chunk_of = |i: usize| if i == i_type { c } else { &no_chunk };
                gpu_kernel_call!(
                    gpu::model_autos,
                    num_tiles as i32,
                    num_freqs as i32,
                    self.d_freqs.get(),
                    &self.points(chunk_of(0)),
                    &self.gaussians(chunk_of(1)),
                    &self.shapelets(chunk_of(2), no_uvs),
                    d_beam,
                    self.gpu_beam.get_tile_map(),
                    self.d_beam_freq_lo.get(),
                    self.d_beam_freq_hi.get(),
                    self.d_beam_freq_weights.get(),
                    self.gpu_beam.get_num_unique_freqs(),
                    self.gpu_beam.get_num_unique_tiles(),
                    self.tile_index_to_unflagged_tile_index_map.get(),
                    d_vis_fb.get_mut().cast(),
                )?;
            }
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
        let Geometry {
            lst,
            xyzs,
            latitude,
            phase_centre_hadec,
        } = geometry_at(
            self.uvw_frame,
            self.precess,
            self.array_longitude,
            self.array_latitude,
            self.phase_centre,
            timestamp,
            self.dut1,
            self.unflagged_tile_xyzs,
        );
        let uvws = xyzs_to_cross_uvws(&xyzs, phase_centre_hadec);
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
            num_unique_tiles: self.gpu_beam.get_num_unique_tiles(),
            d_tile_map: self.gpu_beam.get_tile_map(),
            d_freq_map: self.d_beam_freq_lo.get(),
            d_tile_index_to_unflagged_tile_index_map: self
                .tile_index_to_unflagged_tile_index_map
                .get(),
            d_freq_map_hi: self.d_beam_freq_hi.get(),
            d_freq_weights: self.d_beam_freq_weights.get(),
        }
    }

    /// Shapelets need their own special kind of UVW coordinates. Each shapelet
    /// component's position is treated as the phase centre. This function uses
    /// the FFI type [`gpu::ShapeletUV`]; the W isn't actually used in
    /// computation, and omitting it is hopefully a little more efficient.
    ///
    /// The returned arrays have baseline as the first axis and component as the
    /// second.
    #[cfg(test)]
    pub(super) fn get_shapelet_uvs(&self, lst_rad: f64) -> ShapeletUVs {
        self.get_shapelet_uvs_of_chunk(
            lst_rad,
            &CompChunk::new(0..usize::MAX, self.shapelet_counts()),
        )
    }

    /// [`Self::get_shapelet_uvs`] for a chunk of the shapelet components.
    fn get_shapelet_uvs_of_chunk(&self, lst_rad: f64, c: &CompChunk) -> ShapeletUVs {
        ShapeletUVs {
            power_law: get_shapelet_uvs_inner(
                &self.shapelet_power_law_radecs[c.power_law.clone()],
                lst_rad,
                self.unflagged_tile_xyzs,
            ),
            curved_power_law: get_shapelet_uvs_inner(
                &self.shapelet_curved_power_law_radecs[c.curved_power_law.clone()],
                lst_rad,
                self.unflagged_tile_xyzs,
            ),
            list: get_shapelet_uvs_inner(
                &self.shapelet_list_radecs[c.list.clone()],
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

    fn update_with_a_source(
        &mut self,
        source: &Source,
        phase_centre: RADec,
    ) -> Result<(), ModelError> {
        self.update_source_list(source.components.iter(), phase_centre)
    }

    fn model_timestep_autos_with(
        &self,
        timestamp: Epoch,
        mut vis_fb: ArrayViewMut2<Jones<f32>>,
    ) -> Result<(), ModelError> {
        // The device buffers will automatically be resized.
        let mut d_uvws = DevicePointer::default();
        let (lst, _, latitude) = self.get_lst_uvws_latitude(timestamp, &mut d_uvws)?;

        let mut d_vis_fb =
            DevicePointer::copy_to_device(vis_fb.as_slice().expect("is contiguous"))?;
        let mut d_beam_jones = DevicePointer::default();

        self.model_timestep_autos_with_inner(lst, latitude, &mut d_beam_jones, &mut d_vis_fb)?;
        d_vis_fb.copy_from_device(vis_fb.as_slice_mut().expect("is contiguous"))?;

        // Mask any unavailable polarisations.
        mask_pols(vis_fb, self.pols);

        Ok(())
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

/// Where each shapelet component's coefficients start in the flattened
/// coefficient array of [`get_flattened_coeffs`], followed by the array's
/// length.
fn coeff_offsets(coeff_lens: &[i32]) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(coeff_lens.len() + 1);
    let mut offset = 0;
    offsets.push(offset);
    for &len in coeff_lens {
        offset += len as usize;
        offsets.push(offset);
    }
    offsets
}

/// How much host memory \[bytes\] [`list_fds_to_device`] uses at a time.
const LIST_FD_BLOCK_BYTES: usize = 64 * 1024 * 1024;

/// Evaluate the instrumental flux densities of list-type components at
/// `freqs` and put them in `d_fds` (re-allocated if necessary), with component
/// as the first axis and frequency as the second. Each component's flux
/// densities are then contiguous, so a run of components is too.
///
/// There is one flux density per component per frequency, which is many GB
/// for a big sky model, so they are evaluated and copied to the device a
/// block of components at a time; the host never holds all of them.
fn list_fds_to_device(
    d_fds: &mut DevicePointer<GpuJones>,
    comp_fds: &[&FluxDensityType],
    freqs: &[f64],
    pol_convention: PolConvention,
) -> Result<(), GpuError> {
    let num_fds = comp_fds.len() * freqs.len();
    if num_fds == 0 {
        return Ok(());
    }
    d_fds.realloc(num_fds * std::mem::size_of::<GpuJones>())?;

    let bytes_per_comp = freqs.len() * std::mem::size_of::<GpuJones>();
    let comps_per_block = (LIST_FD_BLOCK_BYTES / bytes_per_comp).max(1);
    let mut block: Vec<GpuJones> = Vec::with_capacity(comps_per_block * freqs.len());
    for (i_block, block_fds) in comp_fds.chunks(comps_per_block).enumerate() {
        block.clear();
        for comp_fd in block_fds {
            for &freq in freqs {
                let j: Jones<f64> = comp_fd
                    .estimate_at_freq(freq)
                    .to_inst_stokes(pol_convention);
                block.push(GpuJones {
                    j00_re: j[0].re as GpuFloat,
                    j00_im: j[0].im as GpuFloat,
                    j01_re: j[1].re as GpuFloat,
                    j01_im: j[1].im as GpuFloat,
                    j10_re: j[2].re as GpuFloat,
                    j10_im: j[2].im as GpuFloat,
                    j11_re: j[3].re as GpuFloat,
                    j11_im: j[3].im as GpuFloat,
                });
            }
        }
        d_fds.overwrite_at(i_block * comps_per_block * freqs.len(), &block)?;
    }
    Ok(())
}

/// A pointer to element `offset` of a device buffer. An empty buffer's (null)
/// pointer is only ever offset by 0.
fn at<T>(d_ptr: &DevicePointer<T>, offset: usize) -> *const T {
    d_ptr.get().wrapping_add(offset)
}

fn to_i32(n: usize) -> i32 {
    n.try_into().expect("not bigger than i32::MAX")
}

/// A run of one component type's components, in the order that the kernels
/// index them (power laws, then curved power laws, then lists): the range of
/// each flux-density type's components that are in the run.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct CompChunk {
    pub(super) power_law: Range<usize>,
    pub(super) curved_power_law: Range<usize>,
    pub(super) list: Range<usize>,
}

impl CompChunk {
    /// The components in `range`, which indexes all of a component type's
    /// components: `counts` power laws, curved power laws and lists.
    fn new(range: Range<usize>, counts: [usize; 3]) -> CompChunk {
        let [num_power_laws, num_curved_power_laws, num_lists] = counts;
        // The part of `range` that is in the `num` components after the first
        // `offset`, relative to `offset`.
        let part = |offset: usize, num: usize| -> Range<usize> {
            let start = range.start.clamp(offset, offset + num);
            let end = range.end.clamp(start, offset + num);
            start - offset..end - offset
        };
        CompChunk {
            power_law: part(0, num_power_laws),
            curved_power_law: part(num_power_laws, num_curved_power_laws),
            list: part(num_power_laws + num_curved_power_laws, num_lists),
        }
    }
}

/// Split one component type's components (`counts` power laws, curved power
/// laws and lists) into runs of up to `comps_per_chunk` components.
pub(super) fn split_into_chunks(counts: [usize; 3], comps_per_chunk: usize) -> Vec<CompChunk> {
    let num_comps: usize = counts.iter().sum();
    let comps_per_chunk = comps_per_chunk.max(1);
    (0..num_comps)
        .step_by(comps_per_chunk)
        .map(|start| CompChunk::new(start..start.saturating_add(comps_per_chunk), counts))
        .collect()
}
