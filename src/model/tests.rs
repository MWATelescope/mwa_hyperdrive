// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Tests on generating sky-model visibilities.

use approx::assert_abs_diff_eq;
use ndarray::prelude::*;

use super::*;
use crate::shapelets;
use mwa_hyperdrive_core::{xyz, Jones, RADec, XyzGeodetic};
#[cfg(feature = "cuda")]
use mwa_hyperdrive_cuda as cuda;
use mwa_hyperdrive_srclist::{
    ComponentList, ComponentType, FluxDensity, FluxDensityType, Source, SourceComponent, SourceList,
};

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
    fn basic() -> Self {
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
        let uvws = xyz::xyzs_to_uvws(&xyzs, &phase_centre.to_hadec(lst));

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
    let obs = ObsParams::basic();

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
    let comps = ComponentList::new(srclist, &obs.freqs, &obs.phase_centre);

    // Ignore applying the beam.
    let mut visibilities = Array2::from_elem((obs.xyzs.len(), obs.freqs.len()), Jones::default());
    model_points(
        visibilities.view_mut(),
        &comps.points.lmns,
        comps.points.instrumental_flux_densities.view(),
        &obs.uvws,
        &obs.freqs,
    );
    // All LMN values are (0, 0, 1). This means that the Fourier transform to
    // make a visibility from LMN and UVW will always just be the input flux
    // density. To compare the flux densities against the visibilities, we need
    // to change the types (just demote the precision of the FDs).
    let fds: Array2<Jones<f32>> = comps.points.instrumental_flux_densities.mapv(|v| v.into());
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
    let obs = ObsParams::basic();
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
    let comps = ComponentList::new(srclist, &obs.freqs, &obs.phase_centre);

    let mut visibilities = Array2::from_elem((obs.xyzs.len(), obs.freqs.len()), Jones::default());
    model_points(
        visibilities.view_mut(),
        &comps.points.lmns,
        comps.points.instrumental_flux_densities.view(),
        &obs.uvws,
        &obs.freqs,
    );
    // This time, all LMN values should be close to (but not the same as) (0, 0,
    // 1). This means that the visibilities should be somewhat close to the
    // input flux densities.
    let fds: Array2<Jones<f32>> = comps.points.instrumental_flux_densities.mapv(|v| v.into());
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
    let obs = ObsParams::basic();

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
    let comps = ComponentList::new(srclist, &obs.freqs, &obs.phase_centre);

    let mut visibilities = Array2::from_elem((obs.xyzs.len(), obs.freqs.len()), Jones::default());
    model_gaussians(
        visibilities.view_mut(),
        &comps.gaussians.lmns,
        &comps.gaussians.gaussian_params,
        comps.gaussians.instrumental_flux_densities.view(),
        &obs.uvws,
        &obs.freqs,
    );
    // All LMN values are (0, 0, 1). This means that the Fourier transform to
    // make a visibility from LMN and UVW will always just be the input flux
    // density. To compare the flux densities against the visibilities, we need
    // to change the types (just demote the precision of the FDs).
    let fds: Array2<Jones<f32>> = comps
        .gaussians
        .instrumental_flux_densities
        .mapv(|v| v.into());
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
    let obs = ObsParams::basic();
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
    let comps = ComponentList::new(srclist, &obs.freqs, &obs.phase_centre);

    let mut visibilities = Array2::from_elem((obs.xyzs.len(), obs.freqs.len()), Jones::default());
    model_gaussians(
        visibilities.view_mut(),
        &comps.gaussians.lmns,
        &comps.gaussians.gaussian_params,
        comps.gaussians.instrumental_flux_densities.view(),
        &obs.uvws,
        &obs.freqs,
    );
    // This time, all LMN values should be close to (but not the same as) (0, 0,
    // 1). This means that the visibilities should be somewhat close to the
    // input flux densities.
    let fds: Array2<Jones<f32>> = comps
        .gaussians
        .instrumental_flux_densities
        .mapv(|v| v.into());
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
    let obs = ObsParams::basic();

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
    let comps = ComponentList::new(srclist, &obs.freqs, &obs.phase_centre);

    let mut visibilities = Array2::from_elem((obs.xyzs.len(), obs.freqs.len()), Jones::default());
    let shapelet_uvws = comps.shapelets.get_shapelet_uvws(obs.lst, &obs.xyzs);
    model_shapelets(
        visibilities.view_mut(),
        &comps.shapelets.lmns,
        &comps.shapelets.gaussian_params,
        &comps.shapelets.shapelet_coeffs,
        shapelet_uvws.view(),
        comps.shapelets.instrumental_flux_densities.view(),
        &obs.uvws,
        &obs.freqs,
    );
    // All LMN values are (0, 0, 1). This means that the Fourier transform to
    // make a visibility from LMN and UVW will always just be the input flux
    // density. To compare the flux densities against the visibilities, we need
    // to change the types (just demote the precision of the FDs).
    let fds: Array2<Jones<f32>> = comps
        .shapelets
        .instrumental_flux_densities
        .mapv(|v| v.into());
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
    let obs = ObsParams::basic();
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
    let comps = ComponentList::new(srclist, &obs.freqs, &obs.phase_centre);

    let mut visibilities = Array2::from_elem((obs.xyzs.len(), obs.freqs.len()), Jones::default());
    let shapelet_uvws = comps.shapelets.get_shapelet_uvws(obs.lst, &obs.xyzs);
    model_shapelets(
        visibilities.view_mut(),
        &comps.shapelets.lmns,
        &comps.shapelets.gaussian_params,
        &comps.shapelets.shapelet_coeffs,
        shapelet_uvws.view(),
        comps.shapelets.instrumental_flux_densities.view(),
        &obs.uvws,
        &obs.freqs,
    );
    // This time, all LMN values should be close to (but not the same as) (0, 0,
    // 1). This means that the visibilities should be somewhat close to the
    // input flux densities.
    let fds: Array2<Jones<f32>> = comps
        .shapelets
        .instrumental_flux_densities
        .mapv(|v| v.into());
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

// This test makes a simple sky model and array to generate visibilities.
#[test]
fn simple_zenith_cpu() {
    let obs = ObsParams::basic();

    // One point, Gaussian and shapelet source all at zenith (which is also the
    // phase centre).
    let mut srclist = SourceList::new();
    srclist.insert(
        "zenith".to_string(),
        Source {
            components: vec![
                get_simple_point(obs.phase_centre.clone(), obs.flux_density_scale.clone()),
                get_simple_gaussian(obs.phase_centre.clone(), obs.flux_density_scale.clone()),
                get_simple_shapelet(obs.phase_centre.clone(), obs.flux_density_scale.clone()),
            ],
        },
    );
    // Get the component parameters via `ComponentList`.
    let comps = ComponentList::new(srclist, &obs.freqs, &obs.phase_centre);

    // Generate the visibilities for each component type. Ignore applying the
    // beam.
    let mut visibilities = Array2::from_elem((obs.xyzs.len(), obs.freqs.len()), Jones::default());
    model_points(
        visibilities.view_mut(),
        &comps.points.lmns,
        comps.points.instrumental_flux_densities.view(),
        &obs.uvws,
        &obs.freqs,
    );
    // All LMN values are (0, 0, 1). This means that the Fourier transform to
    // make a visibility from LMN and UVW will always just be the input flux
    // density. To compare the flux densities against the visibilities, we need
    // to change the types (just demote the precision of the FDs).
    let fds: Array2<Jones<f32>> = comps.points.instrumental_flux_densities.mapv(|v| v.into());
    for (fd, vis) in fds.iter().zip(visibilities.slice(s![0, ..])) {
        assert_abs_diff_eq!(fd, vis, epsilon = 1e-10);
    }
    // Compare these visibilities with the other types; they should be the same.
    // (Gaussians/shapelets with no sizes are point sources!)
    let point_vis = visibilities.clone();
    // Reset the visibilities before comparing with other component types.
    visibilities.fill(Jones::default());

    model_gaussians(
        visibilities.view_mut(),
        &comps.gaussians.lmns,
        &comps.gaussians.gaussian_params,
        comps.gaussians.instrumental_flux_densities.view(),
        &obs.uvws,
        &obs.freqs,
    );
    let fds: Array2<Jones<f32>> = comps
        .gaussians
        .instrumental_flux_densities
        .mapv(|v| v.into());
    for (fd, vis) in fds.iter().zip(visibilities.slice(s![0, ..])) {
        assert_abs_diff_eq!(fd, vis, epsilon = 1e-10);
    }
    for (point_vis, vis) in point_vis.iter().zip(visibilities.slice(s![0, ..])) {
        assert_abs_diff_eq!(point_vis, vis, epsilon = 1e-10);
    }
    visibilities.fill(Jones::default());

    let shapelet_uvws = comps.shapelets.get_shapelet_uvws(obs.lst, &obs.xyzs);
    model_shapelets(
        visibilities.view_mut(),
        &comps.shapelets.lmns,
        &comps.shapelets.gaussian_params,
        &comps.shapelets.shapelet_coeffs,
        shapelet_uvws.view(),
        comps.shapelets.instrumental_flux_densities.view(),
        &obs.uvws,
        &obs.freqs,
    );
    let fds: Array2<Jones<f32>> = comps
        .shapelets
        .instrumental_flux_densities
        .mapv(|v| v.into());
    for (fd, vis) in fds.iter().zip(visibilities.slice(s![0, ..])) {
        assert_abs_diff_eq!(fd, vis, epsilon = 1e-10);
    }
    for (point_vis, vis) in point_vis.iter().zip(visibilities.slice(s![0, ..])) {
        assert_abs_diff_eq!(point_vis, vis, epsilon = 1e-10);
    }
}

// The same as above, but on the GPU.
#[test]
#[cfg(feature = "cuda")]
fn simple_zenith_gpu() {
    let obs = ObsParams::basic();

    // One point, Gaussian and shapelet source all at zenith (which is also the
    // phase centre).
    let mut srclist = SourceList::new();
    srclist.insert(
        "zenith".to_string(),
        Source {
            components: vec![
                get_simple_point(obs.phase_centre.clone(), obs.flux_density_scale.clone()),
                get_simple_gaussian(obs.phase_centre.clone(), obs.flux_density_scale.clone()),
                get_simple_shapelet(obs.phase_centre.clone(), obs.flux_density_scale.clone()),
            ],
        },
    );
    // Get the component parameters via `ComponentList`.
    let comps = ComponentList::new(srclist, &obs.freqs, &obs.phase_centre);

    let mut visibilities = Array2::from_elem((obs.xyzs.len(), obs.freqs.len()), Jones::default());
    let addresses = unsafe {
        cuda::init_model(
            obs.uvws.len(),
            obs.freqs.len(),
            shapelets::SBF_L,
            shapelets::SBF_N,
            shapelets::SBF_C,
            shapelets::SBF_DX,
            obs.uvws.as_ptr() as _,
            obs.freqs.as_ptr(),
            shapelets::SHAPELET_BASIS_VALUES.as_ptr(),
            visibilities.as_ptr() as _,
        )
    };
    unsafe {
        let result = cuda::model_points(
            comps.points.radecs.len(),
            comps.points.lmns.as_ptr() as _,
            comps.points.instrumental_flux_densities.as_ptr() as _,
            &addresses,
        );
        assert_eq!(result, 0);
        cuda::copy_vis(&addresses);
    }

    let fds: Array2<Jones<f32>> = comps.points.instrumental_flux_densities.mapv(|v| v.into());
    for (fd, vis) in fds.iter().zip(visibilities.slice(s![0, ..])) {
        assert_abs_diff_eq!(fd, vis, epsilon = 1e-10);
    }
    let point_vis = visibilities.clone();
    visibilities.fill(Jones::default());
    unsafe {
        cuda::clear_vis(&addresses);
    }

    unsafe {
        let result = cuda::model_gaussians(
            comps.gaussians.radecs.len(),
            comps.gaussians.lmns.as_ptr() as _,
            comps.gaussians.instrumental_flux_densities.as_ptr() as _,
            comps.gaussians.gaussian_params.as_ptr() as _,
            addresses,
        );
        assert_eq!(result, 0);
        cuda::copy_vis(&addresses);
    }

    let fds: Array2<Jones<f32>> = comps
        .gaussians
        .instrumental_flux_densities
        .mapv(|v| v.into());
    for (fd, vis) in fds.iter().zip(visibilities.slice(s![0, ..])) {
        assert_abs_diff_eq!(fd, vis, epsilon = 1e-10);
    }
    for (point_vis, vis) in point_vis.iter().zip(visibilities.slice(s![0, ..])) {
        assert_abs_diff_eq!(point_vis, vis, epsilon = 1e-10);
    }
    visibilities.fill(Jones::default());
    unsafe {
        cuda::clear_vis(&addresses);
    }

    let shapelet_uvs = comps.shapelets.get_shapelet_uvs_gpu(obs.lst, &obs.xyzs);
    let (shapelet_coeffs, num_shapelet_coeffs) = comps.shapelets.get_flattened_coeffs();
    unsafe {
        let result = cuda::model_shapelets(
            comps.shapelets.radecs.len(),
            comps.shapelets.lmns.as_ptr() as _,
            comps.shapelets.instrumental_flux_densities.as_ptr() as _,
            comps.shapelets.gaussian_params.as_ptr() as _,
            shapelet_uvs.as_ptr(),
            shapelet_coeffs.as_ptr(),
            num_shapelet_coeffs.as_ptr(),
            addresses,
        );
        assert_eq!(result, 0);
        cuda::copy_vis(&addresses);
    }

    let fds: Array2<Jones<f32>> = comps
        .shapelets
        .instrumental_flux_densities
        .mapv(|v| v.into());
    for (fd, vis) in fds.iter().zip(visibilities.slice(s![0, ..])) {
        assert_abs_diff_eq!(fd, vis, epsilon = 1e-10);
    }
    for (point_vis, vis) in point_vis.iter().zip(visibilities.slice(s![0, ..])) {
        assert_abs_diff_eq!(point_vis, vis, epsilon = 1e-10);
    }

    unsafe { cuda::destroy(&addresses) };
}

// // This time put the sources off zenith.
// #[test]
// fn simple_off_zenith_cpu() {
//     let obs = ObsParams::basic();
//     let pos = RADec::new_degrees(1.0, -27.0);

//     // One point, Gaussian and shapelet source all just off zenith.
//     let mut srclist = SourceList::new();
//     srclist.insert(
//         "off_zenith".to_string(),
//         Source {
//             components: vec![
//                 get_simple_point(pos.clone(), obs.flux_density_scale.clone()),
//                 get_simple_gaussian(pos.clone(), obs.flux_density_scale.clone()),
//                 get_simple_shapelet(pos.clone(), obs.flux_density_scale.clone()),
//             ],
//         },
//     );
//     let comps = ComponentList::new(srclist, &obs.freqs, &obs.phase_centre).unwrap();

//     // Generate the visibilities for each component type. Ignore applying the
//     // beam.
//     let mut visibilities = Array2::from_elem((obs.xyzs.len(), obs.freqs.len()), Jones::default());
//     model_points(
//         visibilities.view_mut(),
//         &comps.points.lmns,
//         comps.points.instrumental_flux_densities.view(),
//         &obs.uvws,
//         &obs.freqs,
//     );
//     dbg!(&visibilities);
//     // This time, all LMN values should be close to (but not the same as) (0, 0,
//     // 1). This means that the visibilities should be somewhat close to the
//     // input flux densities.
//     let fds: Array2<Jones<f32>> = comps.points.instrumental_flux_densities.mapv(|v| v.into());
//     for (fd, vis) in fds.iter().zip(visibilities.slice(s![0, ..])) {
//         assert_abs_diff_eq!(fd, vis, epsilon = 0.12);
//     }
//     // Compare these visibilities with the other types; they should be the
//     // same. (Gaussians/shapelets with no sizes are point sources!)
//     let point_vis = visibilities.clone();
//     // Reset the visibilities before comparing with other component types.
//     visibilities.fill(Jones::default());

//     model_gaussians(
//         visibilities.view_mut(),
//         &comps.gaussians.lmns,
//         &comps.gaussians.gaussian_params,
//         comps.gaussians.instrumental_flux_densities.view(),
//         &obs.uvws,
//         &obs.freqs,
//     );
//     let fds: Array2<Jones<f32>> = comps
//         .gaussians
//         .instrumental_flux_densities
//         .mapv(|v| v.into());
//     for (fd, vis) in fds.iter().zip(visibilities.slice(s![0, ..])) {
//         assert_abs_diff_eq!(fd, vis, epsilon = 0.12);
//     }
//     for (point_vis, vis) in point_vis.iter().zip(visibilities.slice(s![0, ..])) {
//         assert_abs_diff_eq!(point_vis, vis, epsilon = 1e-10);
//     }
//     visibilities.fill(Jones::default());

//     let mut shapelet_uvws: Array2<UVW> = Array2::from_elem(
//         (visibilities.len_of(Axis(0)), comps.shapelets.radecs.len()),
//         UVW::default(),
//     );
//     shapelet_uvws
//         .axis_iter_mut(Axis(1))
//         .into_par_iter()
//         .zip(comps.shapelets.radecs.par_iter())
//         .for_each(|(mut baseline_uvw, radec)| {
//             let hadec = radec.to_hadec(obs.lst);
//             let shapelet_uvws = xyz::xyzs_to_uvws(&obs.xyzs, &hadec);
//             baseline_uvw.assign(&Array1::from(shapelet_uvws));
//         });
//     model_shapelets(
//         visibilities.view_mut(),
//         &comps.shapelets.lmns,
//         &comps.shapelets.gaussian_params,
//         &comps.shapelets.shapelet_coeffs,
//         shapelet_uvws.view(),
//         comps.shapelets.instrumental_flux_densities.view(),
//         &obs.uvws,
//         &obs.freqs,
//     );
//     let fds: Array2<Jones<f32>> = comps
//         .shapelets
//         .instrumental_flux_densities
//         .mapv(|v| v.into());
//     for (fd, vis) in fds.iter().zip(visibilities.slice(s![0, ..])) {
//         assert_abs_diff_eq!(fd, vis, epsilon = 0.12);
//     }
//     for (point_vis, vis) in point_vis.iter().zip(visibilities.slice(s![0, ..])) {
//         assert_abs_diff_eq!(point_vis, vis, epsilon = 1e-10);
//     }
// }

// // The same as above, but on the GPU.
// #[test]
// #[cfg(feature = "cuda")]
// fn simple_off_zenith_gpu() {
//     let obs = ObsParams::basic();
//     let pos = RADec::new_degrees(1.0, -27.0);

//     // Unlike the above test, just test one source at a time.
//     let mut srclist = SourceList::new();
//     srclist.insert(
//         "off_zenith".to_string(),
//         Source {
//             components: vec![get_simple_point(
//                 pos.clone(),
//                 obs.flux_density_scale.clone(),
//             )],
//         },
//     );
//     let comps = ComponentList::new(srclist.clone(), &obs.freqs, &obs.phase_centre).unwrap();

//     let mut visibilities = Array2::from_elem((obs.xyzs.len(), obs.freqs.len()), Jones::default());

//     unsafe {
//         let result = cuda::model_timestep(
//             obs.uvws.len(),
//             obs.freqs.len(),
//             comps.points.radecs.len(),
//             comps.gaussians.radecs.len(),
//             comps.shapelets.radecs.len(),
//             obs.uvws.as_ptr() as _,
//             obs.freqs.as_ptr(),
//             comps.points.lmns.as_ptr() as _,
//             comps.points.instrumental_flux_densities.as_ptr() as _,
//             comps.gaussians.lmns.as_ptr() as _,
//             comps.gaussians.instrumental_flux_densities.as_ptr() as _,
//             comps.gaussians.gaussian_params.as_ptr() as _,
//             comps.shapelets.lmns.as_ptr() as _,
//             comps.shapelets.instrumental_flux_densities.as_ptr() as _,
//             comps.shapelets.gaussian_params.as_ptr() as _,
//             std::ptr::null(),
//             comps.shapelets.shapelet_coeffs.as_ptr() as _,
//             std::ptr::null(),
//             visibilities.as_ptr() as _,
//         );
//         assert_eq!(result, 0);
//     }
//     dbg!(&visibilities);

//     let fds: Array2<Jones<f32>> = comps.points.instrumental_flux_densities.mapv(|v| v.into());
//     for (fd, vis) in fds.iter().zip(visibilities.slice(s![0, ..])) {
//         assert_abs_diff_eq!(fd, vis, epsilon = 0.12);
//     }
//     let point_vis = visibilities.clone();
//     visibilities.fill(Jones::default());

//     // Replace the source list with a Gaussian component, then a shapelet
//     // component.
//     srclist.insert(
//         "off_zenith".to_string(),
//         Source {
//             components: vec![get_simple_gaussian(
//                 pos.clone(),
//                 obs.flux_density_scale.clone(),
//             )],
//         },
//     );
//     let comps = ComponentList::new(srclist.clone(), &obs.freqs, &obs.phase_centre).unwrap();

//     unsafe {
//         let result = cuda::model_timestep(
//             obs.uvws.len(),
//             obs.freqs.len(),
//             comps.points.radecs.len(),
//             comps.gaussians.radecs.len(),
//             comps.shapelets.radecs.len(),
//             obs.uvws.as_ptr() as _,
//             obs.freqs.as_ptr(),
//             comps.points.lmns.as_ptr() as _,
//             comps.points.instrumental_flux_densities.as_ptr() as _,
//             comps.gaussians.lmns.as_ptr() as _,
//             comps.gaussians.instrumental_flux_densities.as_ptr() as _,
//             comps.gaussians.gaussian_params.as_ptr() as _,
//             comps.shapelets.lmns.as_ptr() as _,
//             comps.shapelets.instrumental_flux_densities.as_ptr() as _,
//             comps.shapelets.gaussian_params.as_ptr() as _,
//             std::ptr::null(),
//             comps.shapelets.shapelet_coeffs.as_ptr() as _,
//             std::ptr::null(),
//             visibilities.as_ptr() as _,
//         );
//         assert_eq!(result, 0);
//     }

//     let fds: Array2<Jones<f32>> = comps
//         .gaussians
//         .instrumental_flux_densities
//         .mapv(|v| v.into());
//     for (fd, vis) in fds.iter().zip(visibilities.slice(s![0, ..])) {
//         assert_abs_diff_eq!(fd, vis, epsilon = 0.12);
//     }
//     for (point_vis, vis) in point_vis.iter().zip(visibilities.slice(s![0, ..])) {
//         assert_abs_diff_eq!(point_vis, vis, epsilon = 1e-10);
//     }
//     visibilities.fill(Jones::default());

//     srclist.insert(
//         "off_zenith".to_string(),
//         Source {
//             components: vec![get_simple_shapelet(
//                 pos.clone(),
//                 obs.flux_density_scale.clone(),
//             )],
//         },
//     );
//     let comps = ComponentList::new(srclist, &obs.freqs, &obs.phase_centre).unwrap();
//     let shapelet_uvws = comps.shapelets.get_shapelet_uvws(obs.lst, &obs.xyzs);
//     let num_shapelet_coeffs = comps.shapelets.get_num_shapelet_coeffs();

//     unsafe {
//         let result = cuda::model_timestep(
//             obs.uvws.len(),
//             obs.freqs.len(),
//             comps.points.radecs.len(),
//             comps.gaussians.radecs.len(),
//             comps.shapelets.radecs.len(),
//             obs.uvws.as_ptr() as _,
//             obs.freqs.as_ptr(),
//             comps.points.lmns.as_ptr() as _,
//             comps.points.instrumental_flux_densities.as_ptr() as _,
//             comps.gaussians.lmns.as_ptr() as _,
//             comps.gaussians.instrumental_flux_densities.as_ptr() as _,
//             comps.gaussians.gaussian_params.as_ptr() as _,
//             comps.shapelets.lmns.as_ptr() as _,
//             comps.shapelets.instrumental_flux_densities.as_ptr() as _,
//             comps.shapelets.gaussian_params.as_ptr() as _,
//             shapelet_uvws.as_ptr() as _,
//             comps.shapelets.shapelet_coeffs.as_ptr() as _,
//             num_shapelet_coeffs.as_ptr() as _,
//             visibilities.as_ptr() as _,
//         );
//         assert_eq!(result, 0);
//     }

//     let fds: Array2<Jones<f32>> = comps
//         .shapelets
//         .instrumental_flux_densities
//         .mapv(|v| v.into());
//     for (fd, vis) in fds.iter().zip(visibilities.slice(s![0, ..])) {
//         assert_abs_diff_eq!(fd, vis, epsilon = 0.12);
//     }
//     for (point_vis, vis) in point_vis.iter().zip(visibilities.slice(s![0, ..])) {
//         assert_abs_diff_eq!(point_vis, vis, epsilon = 1e-10);
//     }
// }
