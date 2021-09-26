// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Sky-model component types for FFI.

use std::convert::TryInto;

use super::*;
use mwa_hyperdrive_cuda as cuda;

/// [ComponentListFFI] is yet another alternative to [SourceList] where each of
/// the components and their parameters are arranged into vectors, but differs
/// from [ComponentList] in that [FluxDensityType]s are also split.
///
/// This is particularly useful for GPUs; empirically, it is much cheaper for
/// the GPU to estimate flux densities rather than read them from a large array
/// on the GPU (!?). For this reason, splitting power laws from "lists" allows
/// the GPU code to run much more efficiently.
///
/// The first axis of `*_list_fds` is unflagged fine channel frequency, the
/// second is the source component. The length of `hadecs`, `lmns`,
/// `*_list_fds`'s second axis are the same.
// TODO: Curved power laws.
#[derive(Clone, Debug)]
pub struct ComponentListFFI {
    #[cfg(all(feature = "cuda", not(feature = "cuda-single")))]
    pub unflagged_fine_chan_freqs: Vec<f64>,
    #[cfg(feature = "cuda-single")]
    pub unflagged_fine_chan_freqs: Vec<f32>,

    #[cfg(all(feature = "cuda", not(feature = "cuda-single")))]
    pub shapelet_basis_values: Vec<f64>,
    #[cfg(feature = "cuda-single")]
    pub shapelet_basis_values: Vec<f32>,
    pub sbf_l: i32,
    pub sbf_n: i32,
    #[cfg(all(feature = "cuda", not(feature = "cuda-single")))]
    pub sbf_c: f64,
    #[cfg(feature = "cuda-single")]
    pub sbf_c: f32,
    #[cfg(all(feature = "cuda", not(feature = "cuda-single")))]
    pub sbf_dx: f64,
    #[cfg(feature = "cuda-single")]
    pub sbf_dx: f32,

    pub point_power_law_radecs: Vec<RADec>,
    pub point_power_law_lmns: Vec<cuda::LMN>,
    /// Instrumental flux densities calculated at 150 MHz.
    #[cfg(all(feature = "cuda", not(feature = "cuda-single")))]
    pub point_power_law_fds: Vec<cuda::JonesF64>,
    #[cfg(feature = "cuda-single")]
    pub point_power_law_fds: Vec<cuda::JonesF32>,
    /// Spectral indices.
    #[cfg(all(feature = "cuda", not(feature = "cuda-single")))]
    pub point_power_law_sis: Vec<f64>,
    #[cfg(feature = "cuda-single")]
    pub point_power_law_sis: Vec<f32>,

    pub point_list_radecs: Vec<RADec>,
    pub point_list_lmns: Vec<cuda::LMN>,
    /// Instrumental (i.e. XX, XY, YX, XX).
    #[cfg(all(feature = "cuda", not(feature = "cuda-single")))]
    pub point_list_fds: Array2<cuda::JonesF64>,
    #[cfg(feature = "cuda-single")]
    pub point_list_fds: Array2<cuda::JonesF32>,

    pub gaussian_power_law_radecs: Vec<RADec>,
    pub gaussian_power_law_lmns: Vec<cuda::LMN>,
    /// Instrumental flux densities calculated at 150 MHz.
    #[cfg(all(feature = "cuda", not(feature = "cuda-single")))]
    pub gaussian_power_law_fds: Vec<cuda::JonesF64>,
    #[cfg(feature = "cuda-single")]
    pub gaussian_power_law_fds: Vec<cuda::JonesF32>,
    /// Spectral indices.
    #[cfg(all(feature = "cuda", not(feature = "cuda-single")))]
    pub gaussian_power_law_sis: Vec<f64>,
    #[cfg(feature = "cuda-single")]
    pub gaussian_power_law_sis: Vec<f32>,
    pub gaussian_power_law_gps: Vec<cuda::GaussianParams>,

    pub gaussian_list_radecs: Vec<RADec>,
    pub gaussian_list_lmns: Vec<cuda::LMN>,
    /// Instrumental (i.e. XX, XY, YX, XX).
    #[cfg(all(feature = "cuda", not(feature = "cuda-single")))]
    pub gaussian_list_fds: Array2<cuda::JonesF64>,
    #[cfg(feature = "cuda-single")]
    pub gaussian_list_fds: Array2<cuda::JonesF32>,
    pub gaussian_list_gps: Vec<cuda::GaussianParams>,

    pub shapelet_power_law_radecs: Vec<RADec>,
    pub shapelet_power_law_lmns: Vec<cuda::LMN>,
    /// Instrumental flux densities calculated at 150 MHz.
    #[cfg(all(feature = "cuda", not(feature = "cuda-single")))]
    pub shapelet_power_law_fds: Vec<cuda::JonesF64>,
    #[cfg(feature = "cuda-single")]
    pub shapelet_power_law_fds: Vec<cuda::JonesF32>,
    /// Spectral indices.
    #[cfg(all(feature = "cuda", not(feature = "cuda-single")))]
    pub shapelet_power_law_sis: Vec<f64>,
    #[cfg(feature = "cuda-single")]
    pub shapelet_power_law_sis: Vec<f32>,
    pub shapelet_power_law_gps: Vec<cuda::GaussianParams>,
    pub shapelet_power_law_coeffs: Vec<cuda::ShapeletCoeff>,
    pub shapelet_power_law_coeff_lens: Vec<usize>,

    pub shapelet_list_radecs: Vec<RADec>,
    pub shapelet_list_lmns: Vec<cuda::LMN>,
    /// Instrumental (i.e. XX, XY, YX, XX).
    #[cfg(all(feature = "cuda", not(feature = "cuda-single")))]
    pub shapelet_list_fds: Array2<cuda::JonesF64>,
    #[cfg(feature = "cuda-single")]
    pub shapelet_list_fds: Array2<cuda::JonesF32>,
    pub shapelet_list_gps: Vec<cuda::GaussianParams>,
    pub shapelet_list_coeffs: Vec<cuda::ShapeletCoeff>,
    pub shapelet_list_coeff_lens: Vec<usize>,
}

impl ComponentListFFI {
    /// Given a source list, split the components into each [ComponentType] and
    /// by each [FluxDensityType]. Any list [FluxDensityType]s should be
    /// converted to power laws before calling this function.
    ///
    /// These parameters don't change over time, so it's ideal to run this
    /// function once.
    pub fn new(
        source_list: SourceList,
        unflagged_fine_chan_freqs: &[f64],
        phase_centre: RADec,
        shapelet_basis_values: &[f64],
        sbf_l: usize,
        sbf_n: usize,
        sbf_c: f64,
        sbf_dx: f64,
    ) -> ComponentListFFI {
        let mut point_power_law_radecs: Vec<RADec> = vec![];
        let mut point_power_law_lmns: Vec<cuda::LMN> = vec![];
        let mut point_power_law_fds: Vec<_> = vec![];
        let mut point_power_law_sis: Vec<_> = vec![];

        let mut point_list_radecs: Vec<RADec> = vec![];
        let mut point_list_lmns: Vec<cuda::LMN> = vec![];
        let mut point_list_fds: Vec<FluxDensityType> = vec![];

        let mut gaussian_power_law_radecs: Vec<RADec> = vec![];
        let mut gaussian_power_law_lmns: Vec<cuda::LMN> = vec![];
        let mut gaussian_power_law_fds: Vec<_> = vec![];
        let mut gaussian_power_law_sis: Vec<_> = vec![];
        let mut gaussian_power_law_gps: Vec<cuda::GaussianParams> = vec![];

        let mut gaussian_list_radecs: Vec<RADec> = vec![];
        let mut gaussian_list_lmns: Vec<cuda::LMN> = vec![];
        let mut gaussian_list_fds: Vec<FluxDensityType> = vec![];
        let mut gaussian_list_gps: Vec<cuda::GaussianParams> = vec![];

        let mut shapelet_power_law_radecs: Vec<RADec> = vec![];
        let mut shapelet_power_law_lmns: Vec<cuda::LMN> = vec![];
        let mut shapelet_power_law_fds: Vec<_> = vec![];
        let mut shapelet_power_law_sis: Vec<_> = vec![];
        let mut shapelet_power_law_gps: Vec<cuda::GaussianParams> = vec![];
        let mut shapelet_power_law_coeffs: Vec<Vec<ShapeletCoeff>> = vec![];

        let mut shapelet_list_radecs: Vec<RADec> = vec![];
        let mut shapelet_list_lmns: Vec<cuda::LMN> = vec![];
        let mut shapelet_list_fds: Vec<FluxDensityType> = vec![];
        let mut shapelet_list_gps: Vec<cuda::GaussianParams> = vec![];
        let mut shapelet_list_coeffs: Vec<Vec<ShapeletCoeff>> = vec![];

        #[cfg(all(feature = "cuda", not(feature = "cuda-single")))]
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
        #[cfg(feature = "cuda-single")]
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

        for comp in source_list.into_iter().flat_map(|(_, src)| src.components) {
            let radec = comp.radec;
            let LMN { l, m, n } = comp.radec.to_lmn(phase_centre).prepare_for_rime();
            let lmn = cuda::LMN {
                l: l as _,
                m: m as _,
                n: n as _,
            };
            match comp.comp_type {
                ComponentType::Point => match comp.flux_type {
                    FluxDensityType::PowerLaw { si, .. } => {
                        point_power_law_radecs.push(radec);
                        point_power_law_lmns.push(lmn);
                        let fd_at_150mhz = comp.estimate_at_freq(cuda::POWER_LAW_FD_REF_FREQ as _);
                        let inst_fd: Jones<f64> = fd_at_150mhz.to_inst_stokes();
                        let cuda_inst_fd = jones_to_cuda_jones(inst_fd);
                        point_power_law_fds.push(cuda_inst_fd);
                        point_power_law_sis.push(si as _);
                    }

                    FluxDensityType::CurvedPowerLaw { .. } => todo!(),

                    FluxDensityType::List { .. } => {
                        point_list_radecs.push(radec);
                        point_list_lmns.push(lmn);
                        point_list_fds.push(comp.flux_type);
                    }
                },

                ComponentType::Gaussian { maj, min, pa } => {
                    let gp = cuda::GaussianParams {
                        maj: maj as _,
                        min: min as _,
                        pa: pa as _,
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
                            gaussian_power_law_sis.push(si as _);
                            gaussian_power_law_gps.push(gp);
                        }

                        FluxDensityType::CurvedPowerLaw { .. } => todo!(),

                        FluxDensityType::List { .. } => {
                            gaussian_list_radecs.push(radec);
                            gaussian_list_lmns.push(lmn);
                            gaussian_list_fds.push(comp.flux_type);
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
                        maj: maj as _,
                        min: min as _,
                        pa: pa as _,
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
                            shapelet_power_law_sis.push(si as _);
                            shapelet_power_law_gps.push(gp);
                            shapelet_power_law_coeffs.push(coeffs);
                        }

                        FluxDensityType::CurvedPowerLaw { .. } => todo!(),

                        FluxDensityType::List { .. } => {
                            shapelet_list_radecs.push(radec);
                            shapelet_list_lmns.push(lmn);
                            shapelet_list_fds.push(comp.flux_type);
                            shapelet_list_gps.push(gp);
                            shapelet_list_coeffs.push(coeffs);
                        }
                    }
                }
            }
        }

        let point_list_fds =
            get_instrumental_flux_densities(&point_list_fds, unflagged_fine_chan_freqs)
                .mapv(|j| jones_to_cuda_jones(j));
        let gaussian_list_fds =
            get_instrumental_flux_densities(&gaussian_list_fds, unflagged_fine_chan_freqs)
                .mapv(|j| jones_to_cuda_jones(j));
        let shapelet_list_fds =
            get_instrumental_flux_densities(&shapelet_list_fds, unflagged_fine_chan_freqs)
                .mapv(|j| jones_to_cuda_jones(j));

        let (shapelet_power_law_coeffs, shapelet_power_law_coeff_lens) =
            Self::get_flattened_coeffs(shapelet_power_law_coeffs);
        let (shapelet_list_coeffs, shapelet_list_coeff_lens) =
            Self::get_flattened_coeffs(shapelet_list_coeffs);

        // Variables for CUDA. They're made flexible in their types for
        // whichever precision is being used in the CUDA code.
        let unflagged_fine_chan_freqs: Vec<_> =
            unflagged_fine_chan_freqs.iter().map(|&f| f as _).collect();
        let shapelet_basis_values: Vec<_> = shapelet_basis_values.iter().map(|&f| f as _).collect();

        Self {
            unflagged_fine_chan_freqs,
            shapelet_basis_values,
            sbf_l: sbf_l.try_into().unwrap(),
            sbf_n: sbf_n.try_into().unwrap(),
            sbf_c: sbf_c as _,
            sbf_dx: sbf_dx as _,

            point_power_law_radecs,
            point_power_law_lmns,
            point_power_law_fds,
            point_power_law_sis,
            point_list_radecs,
            point_list_lmns,
            point_list_fds,

            gaussian_power_law_radecs,
            gaussian_power_law_lmns,
            gaussian_power_law_fds,
            gaussian_power_law_sis,
            gaussian_power_law_gps,
            gaussian_list_radecs,
            gaussian_list_lmns,
            gaussian_list_fds,
            gaussian_list_gps,

            shapelet_power_law_radecs,
            shapelet_power_law_lmns,
            shapelet_power_law_fds,
            shapelet_power_law_sis,
            shapelet_power_law_gps,
            shapelet_power_law_coeffs,
            shapelet_power_law_coeff_lens,
            shapelet_list_radecs,
            shapelet_list_lmns,
            shapelet_list_fds,
            shapelet_list_gps,
            shapelet_list_coeffs,
            shapelet_list_coeff_lens,
        }
    }

    /// Generate model visibilities for a single timestep on the GPU.
    ///
    /// TODO: Do the minimal amount of copying, rather than copying everything
    /// for every timestep.
    pub fn model_timestep(
        &self,
        beam: &dyn Beam,
        lst_rad: f64,
        tile_xyzs: &[XyzGeodetic],
        num_baselines: usize,
        uvws: *const cuda::UVW,
        vis: *mut cuda::JonesF32,
    ) -> i32 {
        // Expose all the struct fields to ensure they're all used.
        let Self {
            unflagged_fine_chan_freqs,
            shapelet_basis_values,
            sbf_l,
            sbf_n,
            sbf_c,
            sbf_dx,

            point_power_law_radecs,
            point_power_law_lmns,
            point_power_law_fds,
            point_power_law_sis,
            point_list_radecs,
            point_list_lmns,
            point_list_fds,
            gaussian_power_law_radecs,
            gaussian_power_law_lmns,
            gaussian_power_law_fds,
            gaussian_power_law_sis,
            gaussian_power_law_gps,
            gaussian_list_radecs,
            gaussian_list_lmns,
            gaussian_list_fds,
            gaussian_list_gps,
            shapelet_power_law_radecs,
            shapelet_power_law_lmns,
            shapelet_power_law_fds,
            shapelet_power_law_sis,
            shapelet_power_law_gps,
            shapelet_power_law_coeffs,
            shapelet_power_law_coeff_lens,
            shapelet_list_radecs,
            shapelet_list_lmns,
            shapelet_list_fds,
            shapelet_list_gps,
            shapelet_list_coeffs,
            shapelet_list_coeff_lens,
        } = self;

        let to_hadecs = |x: &[RADec]| -> (Vec<_>, Vec<_>) {
            x.iter()
                .map(|radec| {
                    let rust_hadec = radec.to_hadec(lst_rad);
                    cfg_if::cfg_if! {
                        if #[cfg(not(feature = "cuda"))] {
                            (rust_hadec.ha as f64, rust_hadec.dec as f64)
                        } else {
                            (rust_hadec.ha as f32, rust_hadec.dec as f32)
                        }
                    }
                })
                .unzip()
        };

        let (mut point_power_law_has, mut point_power_law_decs) =
            to_hadecs(&point_power_law_radecs);
        let (mut point_list_has, mut point_list_decs) = to_hadecs(&point_list_radecs);
        point_power_law_has.append(&mut point_list_has);
        point_power_law_decs.append(&mut point_list_decs);
        let mut points = cuda::Points {
            has: point_power_law_has.as_ptr() as _,
            decs: point_power_law_decs.as_ptr() as _,
            num_power_law_points: point_power_law_radecs.len(),
            power_law_lmns: point_power_law_lmns.as_ptr() as _,
            power_law_fds: point_power_law_fds.as_ptr() as _,
            power_law_sis: point_power_law_sis.as_ptr() as _,
            num_list_points: point_list_radecs.len(),
            list_lmns: point_list_lmns.as_ptr() as _,
            list_fds: point_list_fds.as_ptr() as _,
        };

        let (mut gaussian_power_law_has, mut gaussian_power_law_decs) =
            to_hadecs(&gaussian_power_law_radecs);
        let (mut gaussian_list_has, mut gaussian_list_decs) = to_hadecs(&gaussian_list_radecs);
        gaussian_power_law_has.append(&mut gaussian_list_has);
        gaussian_power_law_decs.append(&mut gaussian_list_decs);
        let mut gaussians = cuda::Gaussians {
            has: gaussian_power_law_has.as_ptr() as _,
            decs: gaussian_power_law_decs.as_ptr() as _,
            num_power_law_gaussians: gaussian_power_law_radecs.len(),
            power_law_lmns: gaussian_power_law_lmns.as_ptr() as _,
            power_law_fds: gaussian_power_law_fds.as_ptr() as _,
            power_law_sis: gaussian_power_law_sis.as_ptr() as _,
            power_law_gps: gaussian_power_law_gps.as_ptr() as _,
            num_list_gaussians: gaussian_list_radecs.len(),
            list_lmns: gaussian_list_lmns.as_ptr() as _,
            list_fds: gaussian_list_fds.as_ptr() as _,
            list_gps: gaussian_list_gps.as_ptr() as _,
        };

        let (mut shapelet_power_law_has, mut shapelet_power_law_decs) =
            to_hadecs(&shapelet_power_law_radecs);
        let (mut shapelet_list_has, mut shapelet_list_decs) = to_hadecs(&shapelet_list_radecs);
        shapelet_power_law_has.append(&mut shapelet_list_has);
        shapelet_power_law_decs.append(&mut shapelet_list_decs);
        let shapelet_uvs = self.get_shapelet_uvs(lst_rad, tile_xyzs);
        let mut shapelets = cuda::Shapelets {
            has: shapelet_power_law_has.as_ptr() as _,
            decs: shapelet_power_law_decs.as_ptr() as _,
            num_power_law_shapelets: shapelet_power_law_radecs.len(),
            power_law_lmns: shapelet_power_law_lmns.as_ptr() as _,
            power_law_fds: shapelet_power_law_fds.as_ptr() as _,
            power_law_sis: shapelet_power_law_sis.as_ptr() as _,
            power_law_gps: shapelet_power_law_gps.as_ptr() as _,
            power_law_shapelet_uvs: shapelet_uvs.power_law.as_ptr() as _,
            power_law_shapelet_coeffs: shapelet_power_law_coeffs.as_ptr() as _,
            power_law_num_shapelet_coeffs: shapelet_power_law_coeff_lens.as_ptr() as _,
            num_list_shapelets: shapelet_list_radecs.len(),
            list_lmns: shapelet_list_lmns.as_ptr() as _,
            list_fds: shapelet_list_fds.as_ptr() as _,
            list_gps: shapelet_list_gps.as_ptr() as _,
            list_shapelet_uvs: shapelet_uvs.list.as_ptr() as _,
            list_shapelet_coeffs: shapelet_list_coeffs.as_ptr() as _,
            list_num_shapelet_coeffs: shapelet_list_coeff_lens.as_ptr() as _,
        };

        unsafe {
            match beam.get_beam_type() {
                mwa_hyperdrive_beam::BeamType::FEE => {
                    let freq_ints: Vec<u32> = unflagged_fine_chan_freqs
                        .iter()
                        .map(|&f| f as u32)
                        .collect();
                    let device_pointers = beam.get_device_pointers(&freq_ints).unwrap();
                    cuda::model_timestep_fee_beam(
                        num_baselines.try_into().unwrap(),
                        unflagged_fine_chan_freqs.len().try_into().unwrap(),
                        tile_xyzs.len().try_into().unwrap(),
                        uvws as _,
                        unflagged_fine_chan_freqs.as_ptr() as _,
                        &mut points,
                        &mut gaussians,
                        &mut shapelets,
                        shapelet_basis_values.as_ptr() as _,
                        *sbf_l,
                        *sbf_n,
                        *sbf_c,
                        *sbf_dx,
                        [device_pointers.as_cuda_type()].as_ptr() as _,
                        device_pointers.num_coeffs,
                        device_pointers.num_tiles,
                        device_pointers.num_freqs,
                        (*device_pointers.d_coeff_map).cast(),
                        (*(device_pointers.d_norm_jones.unwrap())).cast(),
                        vis,
                    )
                }

                mwa_hyperdrive_beam::BeamType::None => cuda::model_timestep_no_beam(
                    num_baselines.try_into().unwrap(),
                    unflagged_fine_chan_freqs.len().try_into().unwrap(),
                    uvws as _,
                    unflagged_fine_chan_freqs.as_ptr() as _,
                    &mut points,
                    &mut gaussians,
                    &mut shapelets,
                    shapelet_basis_values.as_ptr() as _,
                    *sbf_l,
                    *sbf_n,
                    *sbf_c,
                    *sbf_dx,
                    vis,
                ),
            }
        }
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
                let shapelet_uvs: Vec<cuda::ShapeletUV> =
                    xyz::xyzs_to_cross_uvws_parallel(tile_xyzs, hadec)
                        .into_iter()
                        .map(|uvw| cuda::ShapeletUV {
                            u: uvw.u as _,
                            v: uvw.v as _,
                        })
                        .collect();
                baseline_uv.assign(&Array1::from(shapelet_uvs));
            });
        shapelet_uvs
    }

    /// Shapelets need their own special kind of UVW coordinates. Each shapelet
    /// component's position is treated as the phase centre. This function uses
    /// the FFI type [cuda::ShapeletUV]; the W isn't actually used in
    /// computation, and omitting it is hopefully a little more efficient.
    ///
    /// The returned arrays have baseline as the first axis and component as the
    /// second.
    pub fn get_shapelet_uvs(&self, lst_rad: f64, tile_xyzs: &[XyzGeodetic]) -> ShapeletUVs {
        ShapeletUVs {
            power_law: Self::get_shapelet_uvs_inner(
                &self.shapelet_power_law_radecs,
                lst_rad,
                tile_xyzs,
            ),
            list: Self::get_shapelet_uvs_inner(&self.shapelet_list_radecs, lst_rad, tile_xyzs),
        }
    }

    /// This function is intended for FFI with GPUs. There are a variable number
    /// of shapelet coefficients for each shapelet component. To avoid excessive
    /// dereferencing on GPUs (expensive), this method flattens the coefficients
    /// into a single array (lengths of the array-of-arrays).
    pub fn get_flattened_coeffs(
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
                    value: coeff.value as _,
                })
            }
        }

        coeffs.shrink_to_fit();
        (coeffs, coeff_lengths)
    }
}

/// The return type of [ComponentListFFI::get_shapelet_uvws]. These arrays have
/// baseline as the first axis and component as the second.
pub struct ShapeletUVs {
    pub power_law: Array2<cuda::ShapeletUV>,
    pub list: Array2<cuda::ShapeletUV>,
}
