// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Sky-model component types for FFI.

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
/// second is the source component. The length of `radecs`, `lmns`,
/// `*_list_fds`'s second axis are the same.
// TODO: Curved power laws.
#[derive(Clone, Debug)]
pub struct ComponentListFFI {
    pub point_power_law_radecs: Vec<cuda::RADec>,
    pub point_power_law_lmns: Vec<cuda::LMN>,
    /// Instrumental flux densities calculated at 150 MHz.
    #[cfg(feature = "cuda-double")]
    pub point_power_law_fds: Vec<cuda::JonesF64>,
    #[cfg(feature = "cuda-single")]
    pub point_power_law_fds: Vec<cuda::JonesF32>,
    /// Spectral indices.
    #[cfg(feature = "cuda-double")]
    pub point_power_law_sis: Vec<f64>,
    #[cfg(feature = "cuda-single")]
    pub point_power_law_sis: Vec<f32>,

    pub point_list_radecs: Vec<cuda::RADec>,
    pub point_list_lmns: Vec<cuda::LMN>,
    /// Instrumental (i.e. XX, XY, YX, XX).
    #[cfg(feature = "cuda-double")]
    pub point_list_fds: Array2<cuda::JonesF64>,
    #[cfg(feature = "cuda-single")]
    pub point_list_fds: Array2<cuda::JonesF32>,

    pub gaussian_power_law_radecs: Vec<cuda::RADec>,
    pub gaussian_power_law_lmns: Vec<cuda::LMN>,
    /// Instrumental flux densities calculated at 150 MHz.
    #[cfg(feature = "cuda-double")]
    pub gaussian_power_law_fds: Vec<cuda::JonesF64>,
    #[cfg(feature = "cuda-single")]
    pub gaussian_power_law_fds: Vec<cuda::JonesF32>,
    /// Spectral indices.
    #[cfg(feature = "cuda-double")]
    pub gaussian_power_law_sis: Vec<f64>,
    #[cfg(feature = "cuda-single")]
    pub gaussian_power_law_sis: Vec<f32>,
    pub gaussian_power_law_gps: Vec<cuda::GaussianParams>,

    pub gaussian_list_radecs: Vec<cuda::RADec>,
    pub gaussian_list_lmns: Vec<cuda::LMN>,
    /// Instrumental (i.e. XX, XY, YX, XX).
    #[cfg(feature = "cuda-double")]
    pub gaussian_list_fds: Array2<cuda::JonesF64>,
    #[cfg(feature = "cuda-single")]
    pub gaussian_list_fds: Array2<cuda::JonesF32>,
    pub gaussian_list_gps: Vec<cuda::GaussianParams>,

    pub shapelet_power_law_radecs: Vec<cuda::RADec>,
    pub shapelet_power_law_lmns: Vec<cuda::LMN>,
    /// Instrumental flux densities calculated at 150 MHz.
    #[cfg(feature = "cuda-double")]
    pub shapelet_power_law_fds: Vec<cuda::JonesF64>,
    #[cfg(feature = "cuda-single")]
    pub shapelet_power_law_fds: Vec<cuda::JonesF32>,
    /// Spectral indices.
    #[cfg(feature = "cuda-double")]
    pub shapelet_power_law_sis: Vec<f64>,
    #[cfg(feature = "cuda-single")]
    pub shapelet_power_law_sis: Vec<f32>,
    pub shapelet_power_law_gps: Vec<cuda::GaussianParams>,
    pub shapelet_power_law_coeffs: Vec<cuda::ShapeletCoeff>,
    pub shapelet_power_law_coeff_lens: Vec<usize>,

    pub shapelet_list_radecs: Vec<cuda::RADec>,
    pub shapelet_list_lmns: Vec<cuda::LMN>,
    /// Instrumental (i.e. XX, XY, YX, XX).
    #[cfg(feature = "cuda-double")]
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
    ) -> Self {
        let mut point_power_law_radecs: Vec<cuda::RADec> = vec![];
        let mut point_power_law_lmns: Vec<cuda::LMN> = vec![];
        let mut point_power_law_fds: Vec<_> = vec![];
        let mut point_power_law_sis: Vec<_> = vec![];

        let mut point_list_radecs: Vec<cuda::RADec> = vec![];
        let mut point_list_lmns: Vec<cuda::LMN> = vec![];
        let mut point_list_fds: Vec<FluxDensityType> = vec![];

        let mut gaussian_power_law_radecs: Vec<cuda::RADec> = vec![];
        let mut gaussian_power_law_lmns: Vec<cuda::LMN> = vec![];
        let mut gaussian_power_law_fds: Vec<_> = vec![];
        let mut gaussian_power_law_sis: Vec<_> = vec![];
        let mut gaussian_power_law_gps: Vec<cuda::GaussianParams> = vec![];

        let mut gaussian_list_radecs: Vec<cuda::RADec> = vec![];
        let mut gaussian_list_lmns: Vec<cuda::LMN> = vec![];
        let mut gaussian_list_fds: Vec<FluxDensityType> = vec![];
        let mut gaussian_list_gps: Vec<cuda::GaussianParams> = vec![];

        let mut shapelet_power_law_radecs: Vec<cuda::RADec> = vec![];
        let mut shapelet_power_law_lmns: Vec<cuda::LMN> = vec![];
        let mut shapelet_power_law_fds: Vec<_> = vec![];
        let mut shapelet_power_law_sis: Vec<_> = vec![];
        let mut shapelet_power_law_gps: Vec<cuda::GaussianParams> = vec![];
        let mut shapelet_power_law_coeffs: Vec<Vec<ShapeletCoeff>> = vec![];

        let mut shapelet_list_radecs: Vec<cuda::RADec> = vec![];
        let mut shapelet_list_lmns: Vec<cuda::LMN> = vec![];
        let mut shapelet_list_fds: Vec<FluxDensityType> = vec![];
        let mut shapelet_list_gps: Vec<cuda::GaussianParams> = vec![];
        let mut shapelet_list_coeffs: Vec<Vec<ShapeletCoeff>> = vec![];

        #[cfg(feature = "cuda-double")]
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
            let radec = cuda::RADec {
                ra: comp.radec.ra as _,
                dec: comp.radec.dec as _,
            };
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

        Self {
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

    /// Make this [ComponentListFFI] available to CUDA.
    ///
    /// If the optional arguments aren't given, then the returned
    /// [cuda::Shapelets] is defective and shouldn't be used.
    pub fn to_c_types(
        &self,
        shapelet_uvs: Option<&ShapeletUVs>,
    ) -> (cuda::Points, cuda::Gaussians, cuda::Shapelets) {
        // Expose all the struct fields to ensure they're all used.
        let Self {
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

        let points = cuda::Points {
            num_power_law_points: point_power_law_radecs.len(),
            power_law_radecs: point_power_law_radecs.as_ptr(),
            power_law_lmns: point_power_law_lmns.as_ptr(),
            power_law_fds: point_power_law_fds.as_ptr(),
            power_law_sis: point_power_law_sis.as_ptr(),
            num_list_points: point_list_radecs.len(),
            list_radecs: point_list_radecs.as_ptr(),
            list_lmns: point_list_lmns.as_ptr(),
            list_fds: point_list_fds.as_ptr(),
        };

        let gaussians = cuda::Gaussians {
            num_power_law_gaussians: gaussian_power_law_radecs.len(),
            power_law_radecs: gaussian_power_law_radecs.as_ptr(),
            power_law_lmns: gaussian_power_law_lmns.as_ptr(),
            power_law_fds: gaussian_power_law_fds.as_ptr(),
            power_law_sis: gaussian_power_law_sis.as_ptr(),
            power_law_gps: gaussian_power_law_gps.as_ptr(),
            num_list_gaussians: gaussian_list_radecs.len(),
            list_radecs: gaussian_list_radecs.as_ptr(),
            list_lmns: gaussian_list_lmns.as_ptr(),
            list_fds: gaussian_list_fds.as_ptr(),
            list_gps: gaussian_list_gps.as_ptr(),
        };

        let shapelets = cuda::Shapelets {
            num_power_law_shapelets: match shapelet_uvs {
                Some(_) => shapelet_power_law_radecs.len(),
                None => 0,
            },
            power_law_radecs: shapelet_power_law_radecs.as_ptr(),
            power_law_lmns: shapelet_power_law_lmns.as_ptr(),
            power_law_fds: shapelet_power_law_fds.as_ptr(),
            power_law_sis: shapelet_power_law_sis.as_ptr(),
            power_law_gps: shapelet_power_law_gps.as_ptr(),
            power_law_shapelet_uvs: match shapelet_uvs {
                Some(uvs) => uvs.power_law.as_ptr(),
                None => std::ptr::null(),
            },
            power_law_shapelet_coeffs: shapelet_power_law_coeffs.as_ptr(),
            power_law_num_shapelet_coeffs: shapelet_power_law_coeff_lens.as_ptr(),
            num_list_shapelets: match shapelet_uvs {
                Some(_) => shapelet_list_radecs.len(),
                None => 0,
            },
            list_radecs: shapelet_list_radecs.as_ptr(),
            list_lmns: shapelet_list_lmns.as_ptr(),
            list_fds: shapelet_list_fds.as_ptr(),
            list_gps: shapelet_list_gps.as_ptr(),
            list_shapelet_uvs: match shapelet_uvs {
                Some(uvs) => uvs.list.as_ptr(),
                None => std::ptr::null(),
            },
            list_shapelet_coeffs: shapelet_list_coeffs.as_ptr(),
            list_num_shapelet_coeffs: shapelet_list_coeff_lens.as_ptr(),
        };

        (points, gaussians, shapelets)
    }

    fn get_shapelet_uvs_inner(
        radecs: &[cuda::RADec],
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
                // Convert from the cuda RADec type to the Rust one (which has
                // methods). Hopefully this is a no-op because they should be
                // the exact same type...
                let radec = RADec {
                    ra: radec.ra as _,
                    dec: radec.dec as _,
                };
                let hadec = radec.to_hadec(lst_rad);
                let shapelet_uvs: Vec<cuda::ShapeletUV> = xyz::xyzs_to_uvws(tile_xyzs, hadec)
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
