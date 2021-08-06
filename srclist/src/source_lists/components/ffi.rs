// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Sky-model component types for FFI.

use super::*;
use mwa_hyperdrive_cuda as cuda;

/// [ComponentListSplit] is yet another alternative to [SourceList] where each
/// of the components and their parameters are arranged into vectors, but
/// differs from [ComponentList] in that [FluxDensityType]s are also split.
///
/// This is particularly useful for FFI with GPUs; empirically, it is much
/// cheaper for the GPU to estimate flux densities rather than read them from a
/// large array on the GPU (!?). For this reason, splitting power laws from
/// "lists" allows the GPU code to run much more efficiently.
///
/// The first axis of `*_list_fds` is unflagged fine channel frequency, the
/// second is the source component. The length of `radecs`, `lmns`,
/// `*_list_fds`'s second axis are the same.
// TODO: Curved power laws.
#[derive(Clone, Debug)]
pub struct ComponentListSplit {
    pub point_power_law_radecs: Vec<RADec>,
    pub point_power_law_lmns: Vec<cuda::LMN>,
    /// Instrumental flux densities calculated at 150 MHz.
    pub point_power_law_fds: Vec<cuda::JonesF64>,
    /// Spectral indices.
    pub point_power_law_sis: Vec<f64>,

    pub point_list_radecs: Vec<RADec>,
    pub point_list_lmns: Vec<cuda::LMN>,
    /// Instrumental (i.e. XX, XY, YX, XX).
    pub point_list_fds: Array2<cuda::JonesF64>,

    pub gaussian_power_law_radecs: Vec<RADec>,
    pub gaussian_power_law_lmns: Vec<cuda::LMN>,
    /// Instrumental flux densities calculated at 150 MHz.
    pub gaussian_power_law_fds: Vec<cuda::JonesF64>,
    /// Spectral indices.
    pub gaussian_power_law_sis: Vec<f64>,
    pub gaussian_power_law_gaussian_params: Vec<cuda::GaussianParams>,

    pub gaussian_list_radecs: Vec<RADec>,
    pub gaussian_list_lmns: Vec<cuda::LMN>,
    /// Instrumental (i.e. XX, XY, YX, XX).
    pub gaussian_list_fds: Array2<cuda::JonesF64>,
    pub gaussian_list_gaussian_params: Vec<cuda::GaussianParams>,

    pub shapelet_power_law_radecs: Vec<RADec>,
    pub shapelet_power_law_lmns: Vec<cuda::LMN>,
    /// Instrumental flux densities calculated at 150 MHz.
    pub shapelet_power_law_fds: Vec<cuda::JonesF64>,
    /// Spectral indices.
    pub shapelet_power_law_sis: Vec<f64>,
    pub shapelet_power_law_gaussian_params: Vec<cuda::GaussianParams>,
    pub shapelet_power_law_coeffs: Vec<cuda::ShapeletCoeff>,
    pub shapelet_power_law_coeff_lens: Vec<usize>,

    pub shapelet_list_radecs: Vec<RADec>,
    pub shapelet_list_lmns: Vec<cuda::LMN>,
    /// Instrumental (i.e. XX, XY, YX, XX).
    pub shapelet_list_fds: Array2<cuda::JonesF64>,
    pub shapelet_list_gaussian_params: Vec<cuda::GaussianParams>,
    pub shapelet_list_coeffs: Vec<cuda::ShapeletCoeff>,
    pub shapelet_list_coeff_lens: Vec<usize>,
}

impl ComponentListSplit {
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
        let mut point_power_law_radecs: Vec<RADec> = vec![];
        let mut point_power_law_lmns: Vec<cuda::LMN> = vec![];
        let mut point_power_law_fds: Vec<cuda::JonesF64> = vec![];
        let mut point_power_law_sis: Vec<f64> = vec![];

        let mut point_list_radecs: Vec<RADec> = vec![];
        let mut point_list_lmns: Vec<cuda::LMN> = vec![];
        let mut point_list_fds: Vec<FluxDensityType> = vec![];

        let mut gaussian_power_law_radecs: Vec<RADec> = vec![];
        let mut gaussian_power_law_lmns: Vec<cuda::LMN> = vec![];
        let mut gaussian_power_law_fds: Vec<cuda::JonesF64> = vec![];
        let mut gaussian_power_law_sis: Vec<f64> = vec![];
        let mut gaussian_power_law_gaussian_params: Vec<cuda::GaussianParams> = vec![];

        let mut gaussian_list_radecs: Vec<RADec> = vec![];
        let mut gaussian_list_lmns: Vec<cuda::LMN> = vec![];
        let mut gaussian_list_gaussian_params: Vec<cuda::GaussianParams> = vec![];
        let mut gaussian_list_fds: Vec<FluxDensityType> = vec![];

        let mut shapelet_power_law_radecs: Vec<RADec> = vec![];
        let mut shapelet_power_law_lmns: Vec<cuda::LMN> = vec![];
        let mut shapelet_power_law_fds: Vec<cuda::JonesF64> = vec![];
        let mut shapelet_power_law_sis: Vec<f64> = vec![];
        let mut shapelet_power_law_gaussian_params: Vec<cuda::GaussianParams> = vec![];
        let mut shapelet_power_law_coeffs: Vec<Vec<ShapeletCoeff>> = vec![];

        let mut shapelet_list_radecs: Vec<RADec> = vec![];
        let mut shapelet_list_lmns: Vec<cuda::LMN> = vec![];
        let mut shapelet_list_gaussian_params: Vec<cuda::GaussianParams> = vec![];
        let mut shapelet_list_fds: Vec<FluxDensityType> = vec![];
        let mut shapelet_list_coeffs: Vec<Vec<ShapeletCoeff>> = vec![];

        for comp in source_list.into_iter().flat_map(|(_, src)| src.components) {
            let comp_lmn = comp.radec.to_lmn(phase_centre).prepare_for_rime();
            let comp_cuda_lmn = cuda::LMN {
                l: comp_lmn.l,
                m: comp_lmn.m,
                n: comp_lmn.n,
            };
            match comp.comp_type {
                ComponentType::Point => match comp.flux_type {
                    FluxDensityType::PowerLaw { si, .. } => {
                        point_power_law_radecs.push(comp.radec);
                        point_power_law_lmns.push(comp_cuda_lmn);
                        let fd_at_150mhz = comp.estimate_at_freq(cuda::POWER_LAW_FD_REF_FREQ);
                        let inst_fd: Jones<f64> = fd_at_150mhz.into();
                        let cuda_inst_fd: cuda::JonesF64 = cuda::JonesF64 {
                            xx_re: inst_fd[0].re,
                            xx_im: inst_fd[0].im,
                            xy_re: inst_fd[1].re,
                            xy_im: inst_fd[1].im,
                            yx_re: inst_fd[2].re,
                            yx_im: inst_fd[2].im,
                            yy_re: inst_fd[3].re,
                            yy_im: inst_fd[3].im,
                        };
                        point_power_law_fds.push(cuda_inst_fd);
                        point_power_law_sis.push(si);
                    }

                    FluxDensityType::CurvedPowerLaw { .. } => todo!(),

                    FluxDensityType::List { .. } => {
                        point_list_radecs.push(comp.radec);
                        point_list_lmns.push(comp_cuda_lmn);
                        point_list_fds.push(comp.flux_type);
                    }
                },

                ComponentType::Gaussian { maj, min, pa } => {
                    let gp = cuda::GaussianParams { maj, min, pa };
                    match comp.flux_type {
                        FluxDensityType::PowerLaw { si, .. } => {
                            gaussian_power_law_radecs.push(comp.radec);
                            gaussian_power_law_lmns.push(comp_cuda_lmn);
                            let fd_at_150mhz = comp.estimate_at_freq(cuda::POWER_LAW_FD_REF_FREQ);
                            let inst_fd: Jones<f64> = fd_at_150mhz.into();
                            let cuda_inst_fd: cuda::JonesF64 = cuda::JonesF64 {
                                xx_re: inst_fd[0].re,
                                xx_im: inst_fd[0].im,
                                xy_re: inst_fd[1].re,
                                xy_im: inst_fd[1].im,
                                yx_re: inst_fd[2].re,
                                yx_im: inst_fd[2].im,
                                yy_re: inst_fd[3].re,
                                yy_im: inst_fd[3].im,
                            };
                            gaussian_power_law_fds.push(cuda_inst_fd);
                            gaussian_power_law_sis.push(si);
                            gaussian_power_law_gaussian_params.push(gp);
                        }

                        FluxDensityType::CurvedPowerLaw { .. } => todo!(),

                        FluxDensityType::List { .. } => {
                            gaussian_list_radecs.push(comp.radec);
                            gaussian_list_lmns.push(comp_cuda_lmn);
                            gaussian_list_fds.push(comp.flux_type);
                            gaussian_list_gaussian_params.push(gp);
                        }
                    };
                }

                ComponentType::Shapelet {
                    maj,
                    min,
                    pa,
                    coeffs,
                } => {
                    let gp = cuda::GaussianParams { maj, min, pa };
                    match comp.flux_type {
                        FluxDensityType::PowerLaw { si, .. } => {
                            shapelet_power_law_radecs.push(comp.radec);
                            shapelet_power_law_lmns.push(comp_cuda_lmn);
                            let fd_at_150mhz =
                                comp.flux_type.estimate_at_freq(cuda::POWER_LAW_FD_REF_FREQ);
                            let inst_fd: Jones<f64> = fd_at_150mhz.into();
                            let cuda_inst_fd: cuda::JonesF64 = cuda::JonesF64 {
                                xx_re: inst_fd[0].re,
                                xx_im: inst_fd[0].im,
                                xy_re: inst_fd[1].re,
                                xy_im: inst_fd[1].im,
                                yx_re: inst_fd[2].re,
                                yx_im: inst_fd[2].im,
                                yy_re: inst_fd[3].re,
                                yy_im: inst_fd[3].im,
                            };
                            shapelet_power_law_fds.push(cuda_inst_fd);
                            shapelet_power_law_sis.push(si);
                            shapelet_power_law_gaussian_params.push(gp);
                            shapelet_power_law_coeffs.push(coeffs);
                        }

                        FluxDensityType::CurvedPowerLaw { .. } => todo!(),

                        FluxDensityType::List { .. } => {
                            shapelet_list_radecs.push(comp.radec);
                            shapelet_list_lmns.push(comp_cuda_lmn);
                            shapelet_list_fds.push(comp.flux_type);
                            shapelet_list_gaussian_params.push(gp);
                            shapelet_list_coeffs.push(coeffs);
                        }
                    }
                }
            }
        }

        let point_list_fds =
            get_instrumental_flux_densities(&point_list_fds, unflagged_fine_chan_freqs).mapv(|j| {
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
            });
        let gaussian_list_fds =
            get_instrumental_flux_densities(&gaussian_list_fds, unflagged_fine_chan_freqs).mapv(
                |j| cuda::JonesF64 {
                    xx_re: j[0].re,
                    xx_im: j[0].im,
                    xy_re: j[1].re,
                    xy_im: j[1].im,
                    yx_re: j[2].re,
                    yx_im: j[2].im,
                    yy_re: j[3].re,
                    yy_im: j[3].im,
                },
            );
        let shapelet_list_fds =
            get_instrumental_flux_densities(&shapelet_list_fds, unflagged_fine_chan_freqs).mapv(
                |j| cuda::JonesF64 {
                    xx_re: j[0].re,
                    xx_im: j[0].im,
                    xy_re: j[1].re,
                    xy_im: j[1].im,
                    yx_re: j[2].re,
                    yx_im: j[2].im,
                    yy_re: j[3].re,
                    yy_im: j[3].im,
                },
            );

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
            gaussian_power_law_gaussian_params,
            gaussian_list_radecs,
            gaussian_list_lmns,
            gaussian_list_fds,
            gaussian_list_gaussian_params,
            shapelet_power_law_radecs,
            shapelet_power_law_lmns,
            shapelet_power_law_fds,
            shapelet_power_law_sis,
            shapelet_power_law_gaussian_params,
            shapelet_power_law_coeffs,
            shapelet_power_law_coeff_lens,
            shapelet_list_radecs,
            shapelet_list_lmns,
            shapelet_list_fds,
            shapelet_list_gaussian_params,
            shapelet_list_coeffs,
            shapelet_list_coeff_lens,
        }
    }

    /// Shapelets need their own special kind of UVW coordinates. Each shapelet
    /// component's position is treated as the phase centre. This function uses
    /// the FFI type [cuda::ShapeletUV]; the W isn't actually used in
    /// computation, and omitting it is hopefully a little more efficient.
    ///
    /// The returned array has baseline as the first axis and component as the
    /// second.
    pub fn get_shapelet_uvs(
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
                let shapelet_uvs: Vec<cuda::ShapeletUV> = xyz::xyzs_to_uvws(tile_xyzs, hadec)
                    .into_iter()
                    .map(|uvw| cuda::ShapeletUV { u: uvw.u, v: uvw.v })
                    .collect();
                baseline_uv.assign(&Array1::from(shapelet_uvs));
            });
        shapelet_uvs
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
                    value: coeff.value,
                })
            }
        }

        coeffs.shrink_to_fit();
        (coeffs, coeff_lengths)
    }
}
