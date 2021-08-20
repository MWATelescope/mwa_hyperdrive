// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Tests on generating sky-model visibilities.

use approx::assert_abs_diff_eq;
use ndarray::prelude::*;

use super::*;
use mwa_hyperdrive_srclist::{
    constants::DEFAULT_SPEC_INDEX, ComponentList, ComponentType, FluxDensity, FluxDensityType,
    Source, SourceComponent, SourceList,
};
use mwa_rust_core::{pos::xyz, Jones, RADec, XyzGeodetic};

#[cfg(feature = "cuda")]
use mwa_hyperdrive_cuda as cuda;
#[cfg(feature = "cuda")]
use mwa_hyperdrive_srclist::ComponentListFFI;

fn get_simple_point(pos: RADec, flux_density_scale: FluxDensityType) -> SourceComponent {
    SourceComponent {
        radec: pos,
        comp_type: ComponentType::Point,
        flux_type: flux_density_scale,
    }
}

fn get_simple_gaussian(pos: RADec, flux_density_scale: FluxDensityType) -> SourceComponent {
    SourceComponent {
        radec: pos,
        comp_type: ComponentType::Gaussian {
            maj: 0.0,
            min: 0.0,
            pa: 0.0,
        },
        flux_type: flux_density_scale,
    }
}

fn get_simple_shapelet(pos: RADec, flux_density_scale: FluxDensityType) -> SourceComponent {
    SourceComponent {
        radec: pos,
        comp_type: ComponentType::Shapelet {
            maj: 0.0,
            min: 0.0,
            pa: 0.0,
            coeffs: vec![ShapeletCoeff {
                n1: 0,
                n2: 0,
                value: 1.0,
            }],
        },
        flux_type: flux_density_scale,
    }
}

struct ObsParams {
    phase_centre: RADec,
    freqs: Vec<f64>,
    flux_density_scale: FluxDensityType,
    lst: f64,
    xyzs: Vec<XyzGeodetic>,
    uvws: Vec<UVW>,
}

impl ObsParams {
    fn power_law() -> Self {
        let phase_centre = RADec::new_degrees(0.0, -27.0);
        let freqs = vec![150e6, 175e6, 200e6];

        let flux_density_scale = FluxDensityType::PowerLaw {
            si: DEFAULT_SPEC_INDEX,
            fd: FluxDensity {
                freq: 150e6,
                i: 1.0,
                ..Default::default()
            },
        };

        let lst = 0.0;
        let xyzs = vec![
            XyzGeodetic {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            },
            XyzGeodetic {
                x: 0.0,
                y: -1.0,
                z: 0.0,
            },
            XyzGeodetic {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
        ];
        let uvws = xyz::xyzs_to_uvws(&xyzs, phase_centre.to_hadec(lst));

        Self {
            phase_centre,
            freqs,
            flux_density_scale,
            lst,
            xyzs,
            uvws,
        }
    }

    fn list() -> Self {
        let phase_centre = RADec::new_degrees(0.0, -27.0);
        let freqs = vec![150e6, 175e6, 200e6];

        let flux_density_scale = FluxDensityType::List {
            fds: vec![
                FluxDensity {
                    freq: 150e6,
                    i: 1.0,
                    ..Default::default()
                },
                FluxDensity {
                    freq: 175e6,
                    i: 3.0,
                    ..Default::default()
                },
                FluxDensity {
                    freq: 200e6,
                    i: 2.0,
                    ..Default::default()
                },
            ],
        };

        let lst = 0.0;
        let xyzs = vec![
            XyzGeodetic {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            },
            XyzGeodetic {
                x: 0.0,
                y: -1.0,
                z: 0.0,
            },
            XyzGeodetic {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
        ];
        let uvws = xyz::xyzs_to_uvws(&xyzs, phase_centre.to_hadec(lst));

        Self {
            phase_centre,
            freqs,
            flux_density_scale,
            lst,
            xyzs,
            uvws,
        }
    }
}

// Put a single point source at zenith.
#[test]
fn point_zenith_cpu() {
    let obs = ObsParams::list();

    let mut srclist = SourceList::new();
    srclist.insert(
        "zenith".to_string(),
        Source {
            components: vec![get_simple_point(
                obs.phase_centre.clone(),
                obs.flux_density_scale.clone(),
            )],
        },
    );
    // Get the component parameters via `ComponentList`.
    let comps = ComponentList::new(&srclist, &obs.freqs, obs.phase_centre);

    // Ignore applying the beam.
    let mut visibilities = Array2::from_elem((obs.xyzs.len(), obs.freqs.len()), Jones::default());
    model_points(
        visibilities.view_mut(),
        &comps.points.lmns,
        comps.points.flux_densities.view(),
        &obs.uvws,
        &obs.freqs,
    );
    // All LMN values are (0, 0, 1). This means that the Fourier transform to
    // make a visibility from LMN and UVW will always just be the input flux
    // density. To compare the flux densities against the visibilities, we need
    // to change the types (just demote the precision of the FDs).
    let fds: Array2<Jones<f32>> = comps.points.flux_densities.mapv(|v| v.into());
    for (fd, vis) in fds.iter().zip(visibilities.slice(s![0, ..])) {
        assert_abs_diff_eq!(fd, vis, epsilon = 1e-10);
    }
    assert_abs_diff_eq!(visibilities[[0, 0]][0].re, 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(visibilities[[0, 0]][0].im, 0.0, epsilon = 1e-10);
    assert_abs_diff_eq!(visibilities[[1, 1]][0].re, 3.0, epsilon = 1e-10);
    assert_abs_diff_eq!(visibilities[[1, 1]][0].im, 0.0, epsilon = 1e-10);
    assert_abs_diff_eq!(visibilities[[2, 2]][0].re, 2.0, epsilon = 1e-10);
    assert_abs_diff_eq!(visibilities[[2, 2]][0].im, 0.0, epsilon = 1e-10);
}

// Put a single point source just off zenith.
#[test]
fn point_off_zenith_cpu() {
    let obs = ObsParams::list();
    let pos = RADec::new_degrees(1.0, -27.0);

    let mut srclist = SourceList::new();
    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec![get_simple_point(
                pos.clone(),
                obs.flux_density_scale.clone(),
            )],
        },
    );
    let comps = ComponentList::new(&srclist, &obs.freqs, obs.phase_centre);

    let mut visibilities = Array2::from_elem((obs.xyzs.len(), obs.freqs.len()), Jones::default());
    model_points(
        visibilities.view_mut(),
        &comps.points.lmns,
        comps.points.flux_densities.view(),
        &obs.uvws,
        &obs.freqs,
    );
    // This time, all LMN values should be close to (but not the same as) (0, 0,
    // 1). This means that the visibilities should be somewhat close to the
    // input flux densities.
    let fds: Array2<Jones<f32>> = comps.points.flux_densities.mapv(|v| v.into());
    for (fd, vis) in fds.iter().zip(visibilities.slice(s![0, ..])) {
        assert_abs_diff_eq!(fd[0].re, vis[0].re, epsilon = 0.05);
        assert_abs_diff_eq!(fd[0].im, vis[0].im, epsilon = 0.35);
        assert_abs_diff_eq!(fd[3].re, vis[3].re, epsilon = 0.05);
        assert_abs_diff_eq!(fd[3].im, vis[3].im, epsilon = 0.35);
        assert_abs_diff_eq!(fd[1], vis[1], epsilon = 1e-10);
        assert_abs_diff_eq!(fd[2], vis[2], epsilon = 1e-10);
    }
    assert_abs_diff_eq!(visibilities[[0, 0]][0].re, 0.99522406, epsilon = 1e-10);
    assert_abs_diff_eq!(visibilities[[0, 0]][0].im, 0.09761678, epsilon = 1e-10);
    assert_abs_diff_eq!(visibilities[[1, 1]][0].re, 2.9950366, epsilon = 1e-10);
    assert_abs_diff_eq!(visibilities[[1, 1]][0].im, 0.17249982, epsilon = 1e-10);
    assert_abs_diff_eq!(visibilities[[2, 2]][0].re, 1.9958266, epsilon = 1e-10);
    assert_abs_diff_eq!(visibilities[[2, 2]][0].im, -0.12913574, epsilon = 1e-10);
}

// Put a single Gaussian source at zenith.
#[test]
fn gaussian_zenith_cpu() {
    let obs = ObsParams::list();

    let mut srclist = SourceList::new();
    srclist.insert(
        "zenith".to_string(),
        Source {
            components: vec![get_simple_gaussian(
                obs.phase_centre.clone(),
                obs.flux_density_scale.clone(),
            )],
        },
    );
    let comps = ComponentList::new(&srclist, &obs.freqs, obs.phase_centre);

    let mut visibilities = Array2::from_elem((obs.xyzs.len(), obs.freqs.len()), Jones::default());
    model_gaussians(
        visibilities.view_mut(),
        &comps.gaussians.lmns,
        &comps.gaussians.gaussian_params,
        comps.gaussians.flux_densities.view(),
        &obs.uvws,
        &obs.freqs,
    );
    // All LMN values are (0, 0, 1). This means that the Fourier transform to
    // make a visibility from LMN and UVW will always just be the input flux
    // density. To compare the flux densities against the visibilities, we need
    // to change the types (just demote the precision of the FDs).
    let fds: Array2<Jones<f32>> = comps.gaussians.flux_densities.mapv(|v| v.into());
    for (fd, vis) in fds.iter().zip(visibilities.slice(s![0, ..])) {
        assert_abs_diff_eq!(fd, vis, epsilon = 1e-10);
    }
    assert_abs_diff_eq!(visibilities[[0, 0]][0].re, 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(visibilities[[0, 0]][0].im, 0.0, epsilon = 1e-10);
    assert_abs_diff_eq!(visibilities[[1, 1]][0].re, 3.0, epsilon = 1e-10);
    assert_abs_diff_eq!(visibilities[[1, 1]][0].im, 0.0, epsilon = 1e-10);
    assert_abs_diff_eq!(visibilities[[2, 2]][0].re, 2.0, epsilon = 1e-10);
    assert_abs_diff_eq!(visibilities[[2, 2]][0].im, 0.0, epsilon = 1e-10);
}

// Put a single Gaussian source just off zenith.
#[test]
fn gaussian_off_zenith_cpu() {
    let obs = ObsParams::list();
    let pos = RADec::new_degrees(1.0, -27.0);

    let mut srclist = SourceList::new();
    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec![get_simple_gaussian(
                pos.clone(),
                obs.flux_density_scale.clone(),
            )],
        },
    );
    let comps = ComponentList::new(&srclist, &obs.freqs, obs.phase_centre);

    let mut visibilities = Array2::from_elem((obs.xyzs.len(), obs.freqs.len()), Jones::default());
    model_gaussians(
        visibilities.view_mut(),
        &comps.gaussians.lmns,
        &comps.gaussians.gaussian_params,
        comps.gaussians.flux_densities.view(),
        &obs.uvws,
        &obs.freqs,
    );
    // This time, all LMN values should be close to (but not the same as) (0, 0,
    // 1). This means that the visibilities should be somewhat close to the
    // input flux densities.
    let fds: Array2<Jones<f32>> = comps.gaussians.flux_densities.mapv(|v| v.into());
    for (fd, vis) in fds.iter().zip(visibilities.slice(s![0, ..])) {
        assert_abs_diff_eq!(fd[0].re, vis[0].re, epsilon = 0.05);
        assert_abs_diff_eq!(fd[0].im, vis[0].im, epsilon = 0.35);
        assert_abs_diff_eq!(fd[3].re, vis[3].re, epsilon = 0.05);
        assert_abs_diff_eq!(fd[3].im, vis[3].im, epsilon = 0.35);
        assert_abs_diff_eq!(fd[1], vis[1], epsilon = 1e-10);
        assert_abs_diff_eq!(fd[2], vis[2], epsilon = 1e-10);
    }
    assert_abs_diff_eq!(visibilities[[0, 0]][0].re, 0.99522406, epsilon = 1e-10);
    assert_abs_diff_eq!(visibilities[[0, 0]][0].im, 0.09761678, epsilon = 1e-10);
    assert_abs_diff_eq!(visibilities[[1, 1]][0].re, 2.9950366, epsilon = 1e-10);
    assert_abs_diff_eq!(visibilities[[1, 1]][0].im, 0.17249982, epsilon = 1e-10);
    assert_abs_diff_eq!(visibilities[[2, 2]][0].re, 1.9958266, epsilon = 1e-10);
    assert_abs_diff_eq!(visibilities[[2, 2]][0].im, -0.12913574, epsilon = 1e-10);
}

// Put a single shapelet source at zenith.
#[test]
fn shapelet_zenith_cpu() {
    let obs = ObsParams::list();

    let mut srclist = SourceList::new();
    srclist.insert(
        "zenith".to_string(),
        Source {
            components: vec![get_simple_shapelet(
                obs.phase_centre.clone(),
                obs.flux_density_scale.clone(),
            )],
        },
    );
    let comps = ComponentList::new(&srclist, &obs.freqs, obs.phase_centre);

    let mut visibilities = Array2::from_elem((obs.xyzs.len(), obs.freqs.len()), Jones::default());
    let shapelet_uvws = comps.shapelets.get_shapelet_uvws(obs.lst, &obs.xyzs);
    model_shapelets(
        visibilities.view_mut(),
        &comps.shapelets.lmns,
        &comps.shapelets.gaussian_params,
        &comps.shapelets.shapelet_coeffs,
        shapelet_uvws.view(),
        comps.shapelets.flux_densities.view(),
        &obs.uvws,
        &obs.freqs,
    );
    // All LMN values are (0, 0, 1). This means that the Fourier transform to
    // make a visibility from LMN and UVW will always just be the input flux
    // density. To compare the flux densities against the visibilities, we need
    // to change the types (just demote the precision of the FDs).
    let fds: Array2<Jones<f32>> = comps.shapelets.flux_densities.mapv(|v| v.into());
    for (fd, vis) in fds.iter().zip(visibilities.slice(s![0, ..])) {
        assert_abs_diff_eq!(fd, vis, epsilon = 1e-10);
    }
    assert_abs_diff_eq!(visibilities[[0, 0]][0].re, 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(visibilities[[0, 0]][0].im, 0.0, epsilon = 1e-10);
    assert_abs_diff_eq!(visibilities[[1, 1]][0].re, 3.0, epsilon = 1e-10);
    assert_abs_diff_eq!(visibilities[[1, 1]][0].im, 0.0, epsilon = 1e-10);
    assert_abs_diff_eq!(visibilities[[2, 2]][0].re, 2.0, epsilon = 1e-10);
    assert_abs_diff_eq!(visibilities[[2, 2]][0].im, 0.0, epsilon = 1e-10);
}

// Put a single shapelet source just off zenith.
#[test]
fn shapelet_off_zenith_cpu() {
    let obs = ObsParams::list();
    let pos = RADec::new_degrees(1.0, -27.0);

    let mut srclist = SourceList::new();
    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec![get_simple_shapelet(
                pos.clone(),
                obs.flux_density_scale.clone(),
            )],
        },
    );
    let comps = ComponentList::new(&srclist, &obs.freqs, obs.phase_centre);

    let mut visibilities = Array2::from_elem((obs.xyzs.len(), obs.freqs.len()), Jones::default());
    let shapelet_uvws = comps.shapelets.get_shapelet_uvws(obs.lst, &obs.xyzs);
    model_shapelets(
        visibilities.view_mut(),
        &comps.shapelets.lmns,
        &comps.shapelets.gaussian_params,
        &comps.shapelets.shapelet_coeffs,
        shapelet_uvws.view(),
        comps.shapelets.flux_densities.view(),
        &obs.uvws,
        &obs.freqs,
    );
    // This time, all LMN values should be close to (but not the same as) (0, 0,
    // 1). This means that the visibilities should be somewhat close to the
    // input flux densities.
    let fds: Array2<Jones<f32>> = comps.shapelets.flux_densities.mapv(|v| v.into());
    for (fd, vis) in fds.iter().zip(visibilities.slice(s![0, ..])) {
        assert_abs_diff_eq!(fd[0].re, vis[0].re, epsilon = 0.05);
        assert_abs_diff_eq!(fd[0].im, vis[0].im, epsilon = 0.35);
        assert_abs_diff_eq!(fd[3].re, vis[3].re, epsilon = 0.05);
        assert_abs_diff_eq!(fd[3].im, vis[3].im, epsilon = 0.35);
        assert_abs_diff_eq!(fd[1], vis[1], epsilon = 1e-10);
        assert_abs_diff_eq!(fd[2], vis[2], epsilon = 1e-10);
    }
    assert_abs_diff_eq!(visibilities[[0, 0]][0].re, 0.99522406, epsilon = 1e-10);
    assert_abs_diff_eq!(visibilities[[0, 0]][0].im, 0.09761678, epsilon = 1e-10);
    assert_abs_diff_eq!(visibilities[[1, 1]][0].re, 2.9950366, epsilon = 1e-10);
    assert_abs_diff_eq!(visibilities[[1, 1]][0].im, 0.17249982, epsilon = 1e-10);
    assert_abs_diff_eq!(visibilities[[2, 2]][0].re, 1.9958266, epsilon = 1e-10);
    assert_abs_diff_eq!(visibilities[[2, 2]][0].im, -0.12913574, epsilon = 1e-10);
}

// Put a single point source at zenith.
#[test]
#[cfg(feature = "cuda")]
fn point_zenith_gpu_list() {
    let obs = ObsParams::list();

    let mut srclist = SourceList::new();
    srclist.insert(
        "zenith".to_string(),
        Source {
            components: vec![get_simple_point(
                obs.phase_centre.clone(),
                obs.flux_density_scale.clone(),
            )],
        },
    );
    // Get the component parameters via `ComponentListFFI`.
    let comps = ComponentListFFI::new(srclist, &obs.freqs, obs.phase_centre);
    let (points, _, _) = comps.to_c_types(None, None);

    // Ignore applying the beam.
    let mut visibilities = Array2::from_elem((obs.xyzs.len(), obs.freqs.len()), Jones::default());
    unsafe {
        let addresses = cuda::init_model(
            obs.uvws.len(),
            obs.freqs.len(),
            crate::shapelets::SBF_L,
            crate::shapelets::SBF_N,
            crate::shapelets::SBF_C,
            crate::shapelets::SBF_DX,
            obs.uvws.as_ptr() as _,
            obs.freqs.as_ptr(),
            crate::shapelets::SHAPELET_BASIS_VALUES.as_ptr(),
            visibilities.as_mut_ptr() as _,
        );
        cuda::model_points(&points, &addresses);
        cuda::copy_vis(&addresses);
        cuda::destroy(&addresses);
    }
    // All LMN values are (0, 0, 1). This means that the Fourier transform to
    // make a visibility from LMN and UVW will always just be the input flux
    // density. To compare the flux densities against the visibilities, we need
    // to change the types (just demote the precision of the FDs).
    let fds: Array2<Jones<f32>> = comps.point_list_fds.mapv(|v| {
        Jones::from([
            Complex::new(v.xx_re, v.xx_im),
            Complex::new(v.xy_re, v.xy_im),
            Complex::new(v.yx_re, v.yx_im),
            Complex::new(v.yy_re, v.yy_im),
        ])
        .into()
    });
    for (fd, vis) in fds.iter().zip(visibilities.slice(s![0, ..])) {
        assert_abs_diff_eq!(fd, vis, epsilon = 1e-10);
    }
    assert_abs_diff_eq!(visibilities[[0, 0]][0].re, 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(visibilities[[0, 0]][0].im, 0.0, epsilon = 1e-10);
    assert_abs_diff_eq!(visibilities[[1, 1]][0].re, 3.0, epsilon = 1e-10);
    assert_abs_diff_eq!(visibilities[[1, 1]][0].im, 0.0, epsilon = 1e-10);
    assert_abs_diff_eq!(visibilities[[2, 2]][0].re, 2.0, epsilon = 1e-10);
    assert_abs_diff_eq!(visibilities[[2, 2]][0].im, 0.0, epsilon = 1e-10);
}

// Put a single point source just off zenith.
#[test]
#[cfg(feature = "cuda")]
fn point_off_zenith_gpu_list() {
    let obs = ObsParams::list();
    let pos = RADec::new_degrees(1.0, -27.0);

    let mut srclist = SourceList::new();
    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec![get_simple_point(
                pos.clone(),
                obs.flux_density_scale.clone(),
            )],
        },
    );
    let comps = ComponentListFFI::new(srclist, &obs.freqs, obs.phase_centre);
    let (points, _, _) = comps.to_c_types(None, None);

    // Ignore applying the beam.
    let mut visibilities = Array2::from_elem((obs.xyzs.len(), obs.freqs.len()), Jones::default());
    unsafe {
        let addresses = cuda::init_model(
            obs.uvws.len(),
            obs.freqs.len(),
            crate::shapelets::SBF_L,
            crate::shapelets::SBF_N,
            crate::shapelets::SBF_C,
            crate::shapelets::SBF_DX,
            obs.uvws.as_ptr() as _,
            obs.freqs.as_ptr(),
            crate::shapelets::SHAPELET_BASIS_VALUES.as_ptr(),
            visibilities.as_mut_ptr() as _,
        );
        cuda::model_points(&points, &addresses);
        cuda::copy_vis(&addresses);
        cuda::destroy(&addresses);
    }
    // This time, all LMN values should be close to (but not the same as) (0, 0,
    // 1). This means that the visibilities should be somewhat close to the
    // input flux densities.
    let fds: Array2<Jones<f32>> = comps.point_list_fds.mapv(|v| {
        Jones::from([
            Complex::new(v.xx_re, v.xx_im),
            Complex::new(v.xy_re, v.xy_im),
            Complex::new(v.yx_re, v.yx_im),
            Complex::new(v.yy_re, v.yy_im),
        ])
        .into()
    });
    for (fd, vis) in fds.iter().zip(visibilities.slice(s![0, ..])) {
        assert_abs_diff_eq!(fd[0].re, vis[0].re, epsilon = 0.05);
        assert_abs_diff_eq!(fd[0].im, vis[0].im, epsilon = 0.35);
        assert_abs_diff_eq!(fd[3].re, vis[3].re, epsilon = 0.05);
        assert_abs_diff_eq!(fd[3].im, vis[3].im, epsilon = 0.35);
        assert_abs_diff_eq!(fd[1], vis[1], epsilon = 1e-10);
        assert_abs_diff_eq!(fd[2], vis[2], epsilon = 1e-10);
    }
    assert_abs_diff_eq!(visibilities[[0, 0]][0].re, 0.99522406, epsilon = 1e-10);
    assert_abs_diff_eq!(visibilities[[0, 0]][0].im, 0.09761678, epsilon = 1e-10);
    assert_abs_diff_eq!(visibilities[[1, 1]][0].re, 2.9950366, epsilon = 1e-10);
    assert_abs_diff_eq!(visibilities[[1, 1]][0].im, 0.17249982, epsilon = 1e-10);
    assert_abs_diff_eq!(visibilities[[2, 2]][0].re, 1.9958266, epsilon = 1e-10);
    assert_abs_diff_eq!(visibilities[[2, 2]][0].im, -0.12913574, epsilon = 1e-10);
}

// Put a single Gaussian source at zenith.
#[test]
#[cfg(feature = "cuda")]
fn gaussian_zenith_gpu_list() {
    let obs = ObsParams::list();

    let mut srclist = SourceList::new();
    srclist.insert(
        "zenith".to_string(),
        Source {
            components: vec![get_simple_gaussian(
                obs.phase_centre.clone(),
                obs.flux_density_scale.clone(),
            )],
        },
    );
    // Get the component parameters via `ComponentListFFI`.
    let comps = ComponentListFFI::new(srclist, &obs.freqs, obs.phase_centre);
    let (_, gaussians, _) = comps.to_c_types(None, None);

    // Ignore applying the beam.
    let mut visibilities = Array2::from_elem((obs.xyzs.len(), obs.freqs.len()), Jones::default());
    unsafe {
        let addresses = cuda::init_model(
            obs.uvws.len(),
            obs.freqs.len(),
            crate::shapelets::SBF_L,
            crate::shapelets::SBF_N,
            crate::shapelets::SBF_C,
            crate::shapelets::SBF_DX,
            obs.uvws.as_ptr() as _,
            obs.freqs.as_ptr(),
            crate::shapelets::SHAPELET_BASIS_VALUES.as_ptr(),
            visibilities.as_mut_ptr() as _,
        );
        cuda::model_gaussians(&gaussians, &addresses);
        cuda::copy_vis(&addresses);
        cuda::destroy(&addresses);
    }
    // All LMN values are (0, 0, 1). This means that the Fourier transform to
    // make a visibility from LMN and UVW will always just be the input flux
    // density. To compare the flux densities against the visibilities, we need
    // to change the types (just demote the precision of the FDs).
    let fds: Array2<Jones<f32>> = comps.point_list_fds.mapv(|v| {
        Jones::from([
            Complex::new(v.xx_re, v.xx_im),
            Complex::new(v.xy_re, v.xy_im),
            Complex::new(v.yx_re, v.yx_im),
            Complex::new(v.yy_re, v.yy_im),
        ])
        .into()
    });
    for (fd, vis) in fds.iter().zip(visibilities.slice(s![0, ..])) {
        assert_abs_diff_eq!(fd, vis, epsilon = 1e-10);
    }
    assert_abs_diff_eq!(visibilities[[0, 0]][0].re, 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(visibilities[[0, 0]][0].im, 0.0, epsilon = 1e-10);
    assert_abs_diff_eq!(visibilities[[1, 1]][0].re, 3.0, epsilon = 1e-10);
    assert_abs_diff_eq!(visibilities[[1, 1]][0].im, 0.0, epsilon = 1e-10);
    assert_abs_diff_eq!(visibilities[[2, 2]][0].re, 2.0, epsilon = 1e-10);
    assert_abs_diff_eq!(visibilities[[2, 2]][0].im, 0.0, epsilon = 1e-10);
}

// Put a single Gaussian source just off zenith.
#[test]
#[cfg(feature = "cuda")]
fn gaussian_off_zenith_gpu_list() {
    let obs = ObsParams::list();
    let pos = RADec::new_degrees(1.0, -27.0);

    let mut srclist = SourceList::new();
    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec![get_simple_gaussian(
                pos.clone(),
                obs.flux_density_scale.clone(),
            )],
        },
    );
    let comps = ComponentListFFI::new(srclist, &obs.freqs, obs.phase_centre);
    let (_, gaussians, _) = comps.to_c_types(None, None);

    // Ignore applying the beam.
    let mut visibilities = Array2::from_elem((obs.xyzs.len(), obs.freqs.len()), Jones::default());
    unsafe {
        let addresses = cuda::init_model(
            obs.uvws.len(),
            obs.freqs.len(),
            crate::shapelets::SBF_L,
            crate::shapelets::SBF_N,
            crate::shapelets::SBF_C,
            crate::shapelets::SBF_DX,
            obs.uvws.as_ptr() as _,
            obs.freqs.as_ptr(),
            crate::shapelets::SHAPELET_BASIS_VALUES.as_ptr(),
            visibilities.as_mut_ptr() as _,
        );
        cuda::model_gaussians(&gaussians, &addresses);
        cuda::copy_vis(&addresses);
        cuda::destroy(&addresses);
    }
    // This time, all LMN values should be close to (but not the same as) (0, 0,
    // 1). This means that the visibilities should be somewhat close to the
    // input flux densities.
    let fds: Array2<Jones<f32>> = comps.point_list_fds.mapv(|v| {
        Jones::from([
            Complex::new(v.xx_re, v.xx_im),
            Complex::new(v.xy_re, v.xy_im),
            Complex::new(v.yx_re, v.yx_im),
            Complex::new(v.yy_re, v.yy_im),
        ])
        .into()
    });
    for (fd, vis) in fds.iter().zip(visibilities.slice(s![0, ..])) {
        assert_abs_diff_eq!(fd[0].re, vis[0].re, epsilon = 0.05);
        assert_abs_diff_eq!(fd[0].im, vis[0].im, epsilon = 0.35);
        assert_abs_diff_eq!(fd[3].re, vis[3].re, epsilon = 0.05);
        assert_abs_diff_eq!(fd[3].im, vis[3].im, epsilon = 0.35);
        assert_abs_diff_eq!(fd[1], vis[1], epsilon = 1e-10);
        assert_abs_diff_eq!(fd[2], vis[2], epsilon = 1e-10);
    }
    assert_abs_diff_eq!(visibilities[[0, 0]][0].re, 0.99522406, epsilon = 1e-10);
    assert_abs_diff_eq!(visibilities[[0, 0]][0].im, 0.09761678, epsilon = 1e-10);
    assert_abs_diff_eq!(visibilities[[1, 1]][0].re, 2.9950366, epsilon = 1e-10);
    assert_abs_diff_eq!(visibilities[[1, 1]][0].im, 0.17249982, epsilon = 1e-10);
    assert_abs_diff_eq!(visibilities[[2, 2]][0].re, 1.9958266, epsilon = 1e-10);
    assert_abs_diff_eq!(visibilities[[2, 2]][0].im, -0.12913574, epsilon = 1e-10);
}

// // The same as above, but on the GPU.
// #[test]
// #[cfg(feature = "cuda")]
// fn simple_zenith_gpu_list() {
//     let obs = ObsParams::list();

//     // One point, Gaussian and shapelet source all at zenith (which is also the
//     // phase centre).
//     let mut srclist = SourceList::new();
//     srclist.insert(
//         "zenith".to_string(),
//         Source {
//             components: vec![
//                 get_simple_point(obs.phase_centre.clone(), obs.flux_density_scale.clone()),
//                 get_simple_gaussian(obs.phase_centre.clone(), obs.flux_density_scale.clone()),
//                 get_simple_shapelet(obs.phase_centre.clone(), obs.flux_density_scale.clone()),
//             ],
//         },
//     );
//     // Get the component parameters via `ComponentList`.
//     let comps = ComponentList::new(srclist, &obs.freqs, obs.phase_centre);

//     let mut visibilities = Array2::from_elem((obs.xyzs.len(), obs.freqs.len()), Jones::default());
//     let addresses = unsafe {
//         cuda::init_model(
//             obs.uvws.len(),
//             obs.freqs.len(),
//             shapelets::SBF_L,
//             shapelets::SBF_N,
//             shapelets::SBF_C,
//             shapelets::SBF_DX,
//             obs.uvws.as_ptr() as _,
//             obs.freqs.as_ptr(),
//             shapelets::SHAPELET_BASIS_VALUES.as_ptr(),
//             visibilities.as_ptr() as _,
//         )
//     };
//     unsafe {
//         let result = cuda::model_list_points(
//             comps.points.radecs.len(),
//             comps.points.lmns.as_ptr() as _,
//             comps.points.flux_densities.as_ptr() as _,
//             &addresses,
//         );
//         assert_eq!(result, 0);
//         cuda::copy_vis(&addresses);
//     }

//     let fds: Array2<Jones<f32>> = comps.points.flux_densities.mapv(|v| v.into());
//     for (fd, vis) in fds.iter().zip(visibilities.slice(s![0, ..])) {
//         assert_abs_diff_eq!(fd, vis, epsilon = 1e-10);
//     }
//     let point_vis = visibilities.clone();
//     visibilities.fill(Jones::default());
//     unsafe {
//         cuda::clear_vis(&addresses);
//     }

//     unsafe {
//         let result = cuda::model_list_gaussians(
//             comps.gaussians.radecs.len(),
//             comps.gaussians.lmns.as_ptr() as _,
//             comps.gaussians.flux_densities.as_ptr() as _,
//             comps.gaussians.gaussian_params.as_ptr() as _,
//             addresses,
//         );
//         assert_eq!(result, 0);
//         cuda::copy_vis(&addresses);
//     }

//     let fds: Array2<Jones<f32>> = comps.gaussians.flux_densities.mapv(|v| v.into());
//     for (fd, vis) in fds.iter().zip(visibilities.slice(s![0, ..])) {
//         assert_abs_diff_eq!(fd, vis, epsilon = 1e-10);
//     }
//     for (point_vis, vis) in point_vis.iter().zip(visibilities.slice(s![0, ..])) {
//         assert_abs_diff_eq!(point_vis, vis, epsilon = 1e-10);
//     }
//     visibilities.fill(Jones::default());
//     unsafe {
//         cuda::clear_vis(&addresses);
//     }

//     let shapelet_uvs = comps.shapelets.get_shapelet_uvs_gpu(obs.lst, &obs.xyzs);
//     let (shapelet_coeffs, num_shapelet_coeffs) = comps.shapelets.get_flattened_coeffs();
//     unsafe {
//         let result = cuda::model_shapelets(
//             comps.shapelets.radecs.len(),
//             comps.shapelets.lmns.as_ptr() as _,
//             comps.shapelets.flux_densities.as_ptr() as _,
//             comps.shapelets.gaussian_params.as_ptr() as _,
//             shapelet_uvs.as_ptr(),
//             shapelet_coeffs.as_ptr(),
//             num_shapelet_coeffs.as_ptr(),
//             addresses,
//         );
//         assert_eq!(result, 0);
//         cuda::copy_vis(&addresses);
//     }

//     let fds: Array2<Jones<f32>> = comps.shapelets.flux_densities.mapv(|v| v.into());
//     for (fd, vis) in fds.iter().zip(visibilities.slice(s![0, ..])) {
//         assert_abs_diff_eq!(fd, vis, epsilon = 1e-10);
//     }
//     for (point_vis, vis) in point_vis.iter().zip(visibilities.slice(s![0, ..])) {
//         assert_abs_diff_eq!(point_vis, vis, epsilon = 1e-10);
//     }

//     unsafe { cuda::destroy(&addresses) };
// }
