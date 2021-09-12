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

#[cfg(any(feature = "cuda-double", feature = "cuda-single"))]
use mwa_hyperdrive_cuda as cuda;
#[cfg(any(feature = "cuda-double", feature = "cuda-single"))]
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
}

fn assert_list_zenith_visibilities(vis: ArrayView2<Jones<f32>>, fds: ArrayView2<Jones<f64>>) {
    // All LMN values are (0, 0, 1). This means that the Fourier transform to
    // make a visibility from LMN and UVW will always just be the input flux
    // density.

    // First column: flux density at first frequency (i.e. 1 Jy XX and YY).
    assert_abs_diff_eq!(
        vis.slice(s![.., 0]),
        Array1::from_elem(vis.dim().0, Jones::identity()),
    );
    // 3 Jy XX and YY.
    assert_abs_diff_eq!(
        vis.slice(s![.., 1]),
        Array1::from_elem(vis.dim().0, Jones::identity() * 3.0),
    );
    // 2 Jy XX and YY.
    assert_abs_diff_eq!(
        vis.slice(s![.., 2]),
        Array1::from_elem(vis.dim().0, Jones::identity() * 2.0),
    );

    // To compare the flux densities against the visibilities, we need to change
    // the types (just demote the precision of the FDs).
    let fds_single: Array2<Jones<f32>> = fds.mapv(|v| v.into());
    let fds_broadcasted = fds_single.broadcast(vis.dim()).unwrap();
    assert_abs_diff_eq!(fds_broadcasted.t(), vis);
}

fn assert_list_off_zenith_visibilities(vis: ArrayView2<Jones<f32>>, fds: ArrayView2<Jones<f64>>) {
    // This time, all LMN values should be close to, but not the same as, (0, 0,
    // 1). This means that the visibilities should be somewhat close to the
    // input flux densities.

    assert_abs_diff_eq!(
        vis.slice(s![.., 0]),
        array![
            Jones::from([
                Complex::new(0.99522406, 0.09761678),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.99522406, 0.09761678),
            ]),
            Jones::from([
                Complex::new(0.99878436, 0.049292877),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.99878436, 0.049292877),
            ]),
            Jones::from([
                Complex::new(0.9988261, -0.04844065),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.9988261, -0.04844065),
            ]),
        ],
    );

    assert_abs_diff_eq!(
        vis.slice(s![.., 1]),
        array![
            Jones::from([
                Complex::new(2.980504, 0.34146205),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(2.980504, 0.34146205),
            ]),
            Jones::from([
                Complex::new(2.9950366, 0.17249982),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(2.9950366, 0.17249982),
            ]),
            Jones::from([
                Complex::new(2.9952068, -0.1695183),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(2.9952068, -0.1695183),
            ]),
        ],
    );

    assert_abs_diff_eq!(
        vis.slice(s![.., 2]),
        array![
            Jones::from([
                Complex::new(1.9830295, 0.25998875),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(1.9830295, 0.25998875),
            ]),
            Jones::from([
                Complex::new(1.9956784, 0.13140623),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(1.9956784, 0.13140623),
            ]),
            Jones::from([
                Complex::new(1.9958266, -0.12913574),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(1.9958266, -0.12913574),
            ]),
        ],
    );

    let fds_single: Array2<Jones<f32>> = fds.mapv(|v| v.into());
    let fds_broadcasted = fds_single.broadcast(vis.dim()).unwrap();
    for (fd, vis) in fds_broadcasted.t().iter().zip(vis.iter()) {
        assert_abs_diff_eq!(fd[0].re, vis[0].re, epsilon = 0.02);
        assert_abs_diff_eq!(fd[0].im, vis[0].im, epsilon = 0.342);
        assert_abs_diff_eq!(fd[3].re, vis[3].re, epsilon = 0.02);
        assert_abs_diff_eq!(fd[3].im, vis[3].im, epsilon = 0.342);
        assert_abs_diff_eq!(fd[1], vis[1]);
        assert_abs_diff_eq!(fd[2], vis[2]);
    }
}

#[cfg(feature = "cuda-double")]
fn cuda_jones_to_jones(a: ArrayView2<cuda::JonesF64>) -> Array2<Jones<f64>> {
    a.mapv(|j| {
        Jones::from([
            Complex::new(j.xx_re, j.xx_im),
            Complex::new(j.xy_re, j.xy_im),
            Complex::new(j.yx_re, j.yx_im),
            Complex::new(j.yy_re, j.yy_im),
        ])
    })
}
#[cfg(feature = "cuda-single")]
fn cuda_jones_to_jones(a: ArrayView2<cuda::JonesF32>) -> Array2<Jones<f64>> {
    a.mapv(|j| {
        Jones::from([
            Complex::new(j.xx_re as f64, j.xx_im as f64),
            Complex::new(j.xy_re as f64, j.xy_im as f64),
            Complex::new(j.yx_re as f64, j.yx_im as f64),
            Complex::new(j.yy_re as f64, j.yy_im as f64),
        ])
    })
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
                obs.phase_centre,
                obs.flux_density_scale.clone(),
            )],
        },
    );
    // Get the component parameters via `ComponentList`.
    let comps = ComponentList::new(&srclist, &obs.freqs, obs.phase_centre);

    // Ignore applying the beam.
    let mut visibilities = Array2::from_elem((obs.uvws.len(), obs.freqs.len()), Jones::default());
    model_points(
        visibilities.view_mut(),
        &comps.points.lmns,
        comps.points.flux_densities.view(),
        &obs.uvws,
        &obs.freqs,
    );
    assert_list_zenith_visibilities(visibilities.view(), comps.points.flux_densities.view());
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
            components: vec![get_simple_point(pos, obs.flux_density_scale.clone())],
        },
    );
    let comps = ComponentList::new(&srclist, &obs.freqs, obs.phase_centre);

    let mut visibilities = Array2::from_elem((obs.uvws.len(), obs.freqs.len()), Jones::default());
    model_points(
        visibilities.view_mut(),
        &comps.points.lmns,
        comps.points.flux_densities.view(),
        &obs.uvws,
        &obs.freqs,
    );
    assert_list_off_zenith_visibilities(visibilities.view(), comps.points.flux_densities.view());
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
                obs.phase_centre,
                obs.flux_density_scale.clone(),
            )],
        },
    );
    let comps = ComponentList::new(&srclist, &obs.freqs, obs.phase_centre);

    let mut visibilities = Array2::from_elem((obs.uvws.len(), obs.freqs.len()), Jones::default());
    model_gaussians(
        visibilities.view_mut(),
        &comps.gaussians.lmns,
        &comps.gaussians.gaussian_params,
        comps.gaussians.flux_densities.view(),
        &obs.uvws,
        &obs.freqs,
    );
    assert_list_zenith_visibilities(visibilities.view(), comps.gaussians.flux_densities.view());
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
            components: vec![get_simple_gaussian(pos, obs.flux_density_scale.clone())],
        },
    );
    let comps = ComponentList::new(&srclist, &obs.freqs, obs.phase_centre);

    let mut visibilities = Array2::from_elem((obs.uvws.len(), obs.freqs.len()), Jones::default());
    model_gaussians(
        visibilities.view_mut(),
        &comps.gaussians.lmns,
        &comps.gaussians.gaussian_params,
        comps.gaussians.flux_densities.view(),
        &obs.uvws,
        &obs.freqs,
    );
    assert_list_off_zenith_visibilities(visibilities.view(), comps.gaussians.flux_densities.view());
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
                obs.phase_centre,
                obs.flux_density_scale.clone(),
            )],
        },
    );
    let comps = ComponentList::new(&srclist, &obs.freqs, obs.phase_centre);

    let mut visibilities = Array2::from_elem((obs.uvws.len(), obs.freqs.len()), Jones::default());
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
    assert_list_zenith_visibilities(visibilities.view(), comps.shapelets.flux_densities.view());
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
            components: vec![get_simple_shapelet(pos, obs.flux_density_scale.clone())],
        },
    );
    let comps = ComponentList::new(&srclist, &obs.freqs, obs.phase_centre);

    let mut visibilities = Array2::from_elem((obs.uvws.len(), obs.freqs.len()), Jones::default());
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
    assert_list_off_zenith_visibilities(visibilities.view(), comps.shapelets.flux_densities.view());
}

// Put a single point source at zenith.
#[test]
#[cfg(any(feature = "cuda-double", feature = "cuda-single"))]
fn point_zenith_gpu_double_list() {
    let obs = ObsParams::list();

    let mut srclist = SourceList::new();
    srclist.insert(
        "zenith".to_string(),
        Source {
            components: vec![get_simple_point(
                obs.phase_centre,
                obs.flux_density_scale.clone(),
            )],
        },
    );
    // Get the component parameters via `ComponentListFFI`.
    let comps = ComponentListFFI::new(srclist, &obs.freqs, obs.phase_centre);
    let (points, _, _) = comps.to_c_types(None);

    let uvws: Vec<cuda::UVW> = obs
        .uvws
        .iter()
        .map(|uvw| cuda::UVW {
            u: uvw.u as _,
            v: uvw.v as _,
            w: uvw.w as _,
        })
        .collect();
    let freqs: Vec<_> = obs.freqs.iter().map(|&f| f as _).collect();
    let shapelet_basis_values: Vec<_> = crate::shapelets::SHAPELET_BASIS_VALUES
        .iter()
        .map(|&f| f as _)
        .collect();

    // Ignore applying the beam.
    let mut visibilities = Array2::from_elem((obs.uvws.len(), obs.freqs.len()), Jones::default());
    unsafe {
        let addresses = cuda::init_model(
            obs.uvws.len(),
            obs.freqs.len(),
            crate::shapelets::SBF_L,
            crate::shapelets::SBF_N,
            crate::shapelets::SBF_C as _,
            crate::shapelets::SBF_DX as _,
            uvws.as_ptr(),
            freqs.as_ptr(),
            shapelet_basis_values.as_ptr(),
            visibilities.as_mut_ptr() as _,
        );
        cuda::model_points(&points, &addresses);
        cuda::copy_vis(&addresses);
        cuda::destroy(&addresses);
    }
    let fds = cuda_jones_to_jones(comps.point_list_fds.view());
    assert_list_zenith_visibilities(visibilities.view(), fds.view());
}

// Put a single point source just off zenith.
#[test]
#[cfg(any(feature = "cuda-double", feature = "cuda-single"))]
fn point_off_zenith_gpu_double_list() {
    let obs = ObsParams::list();
    let pos = RADec::new_degrees(1.0, -27.0);

    let mut srclist = SourceList::new();
    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec![get_simple_point(pos, obs.flux_density_scale.clone())],
        },
    );
    let comps = ComponentListFFI::new(srclist, &obs.freqs, obs.phase_centre);
    let (points, _, _) = comps.to_c_types(None);

    let uvws: Vec<cuda::UVW> = obs
        .uvws
        .iter()
        .map(|uvw| cuda::UVW {
            u: uvw.u as _,
            v: uvw.v as _,
            w: uvw.w as _,
        })
        .collect();
    let freqs: Vec<_> = obs.freqs.iter().map(|&f| f as _).collect();
    let shapelet_basis_values: Vec<_> = crate::shapelets::SHAPELET_BASIS_VALUES
        .iter()
        .map(|&f| f as _)
        .collect();

    // Ignore applying the beam.
    let mut visibilities = Array2::from_elem((obs.uvws.len(), obs.freqs.len()), Jones::default());
    unsafe {
        let addresses = cuda::init_model(
            obs.uvws.len(),
            obs.freqs.len(),
            crate::shapelets::SBF_L,
            crate::shapelets::SBF_N,
            crate::shapelets::SBF_C as _,
            crate::shapelets::SBF_DX as _,
            uvws.as_ptr(),
            freqs.as_ptr(),
            shapelet_basis_values.as_ptr(),
            visibilities.as_mut_ptr() as _,
        );
        cuda::model_points(&points, &addresses);
        cuda::copy_vis(&addresses);
        cuda::destroy(&addresses);
    }
    let fds = cuda_jones_to_jones(comps.point_list_fds.view());
    assert_list_off_zenith_visibilities(visibilities.view(), fds.view());
}

// Put a single Gaussian source at zenith.
#[test]
#[cfg(any(feature = "cuda-double", feature = "cuda-single"))]
fn gaussian_zenith_gpu_double_list() {
    let obs = ObsParams::list();

    let mut srclist = SourceList::new();
    srclist.insert(
        "zenith".to_string(),
        Source {
            components: vec![get_simple_gaussian(
                obs.phase_centre,
                obs.flux_density_scale.clone(),
            )],
        },
    );
    // Get the component parameters via `ComponentListFFI`.
    let comps = ComponentListFFI::new(srclist, &obs.freqs, obs.phase_centre);
    let (_, gaussians, _) = comps.to_c_types(None);

    let uvws: Vec<cuda::UVW> = obs
        .uvws
        .iter()
        .map(|uvw| cuda::UVW {
            u: uvw.u as _,
            v: uvw.v as _,
            w: uvw.w as _,
        })
        .collect();
    let freqs: Vec<_> = obs.freqs.iter().map(|&f| f as _).collect();
    let shapelet_basis_values: Vec<_> = crate::shapelets::SHAPELET_BASIS_VALUES
        .iter()
        .map(|&f| f as _)
        .collect();

    // Ignore applying the beam.
    let mut visibilities = Array2::from_elem((obs.uvws.len(), obs.freqs.len()), Jones::default());
    unsafe {
        let addresses = cuda::init_model(
            obs.uvws.len(),
            obs.freqs.len(),
            crate::shapelets::SBF_L,
            crate::shapelets::SBF_N,
            crate::shapelets::SBF_C as _,
            crate::shapelets::SBF_DX as _,
            uvws.as_ptr(),
            freqs.as_ptr(),
            shapelet_basis_values.as_ptr(),
            visibilities.as_mut_ptr() as _,
        );
        cuda::model_gaussians(&gaussians, &addresses);
        cuda::copy_vis(&addresses);
        cuda::destroy(&addresses);
    }
    let fds = cuda_jones_to_jones(comps.gaussian_list_fds.view());
    assert_list_zenith_visibilities(visibilities.view(), fds.view());
}

// Put a single Gaussian source just off zenith.
#[test]
#[cfg(any(feature = "cuda-double", feature = "cuda-single"))]
fn gaussian_off_zenith_gpu_double_list() {
    let obs = ObsParams::list();
    let pos = RADec::new_degrees(1.0, -27.0);

    let mut srclist = SourceList::new();
    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec![get_simple_gaussian(pos, obs.flux_density_scale.clone())],
        },
    );
    let comps = ComponentListFFI::new(srclist, &obs.freqs, obs.phase_centre);
    let (_, gaussians, _) = comps.to_c_types(None);

    let uvws: Vec<cuda::UVW> = obs
        .uvws
        .iter()
        .map(|uvw| cuda::UVW {
            u: uvw.u as _,
            v: uvw.v as _,
            w: uvw.w as _,
        })
        .collect();
    let freqs: Vec<_> = obs.freqs.iter().map(|&f| f as _).collect();
    let shapelet_basis_values: Vec<_> = crate::shapelets::SHAPELET_BASIS_VALUES
        .iter()
        .map(|&f| f as _)
        .collect();

    // Ignore applying the beam.
    let mut visibilities = Array2::from_elem((obs.uvws.len(), obs.freqs.len()), Jones::default());
    unsafe {
        let addresses = cuda::init_model(
            obs.uvws.len(),
            obs.freqs.len(),
            crate::shapelets::SBF_L,
            crate::shapelets::SBF_N,
            crate::shapelets::SBF_C as _,
            crate::shapelets::SBF_DX as _,
            uvws.as_ptr(),
            freqs.as_ptr(),
            shapelet_basis_values.as_ptr(),
            visibilities.as_mut_ptr() as _,
        );
        cuda::model_gaussians(&gaussians, &addresses);
        cuda::copy_vis(&addresses);
        cuda::destroy(&addresses);
    }
    let fds = cuda_jones_to_jones(comps.gaussian_list_fds.view());
    assert_list_off_zenith_visibilities(visibilities.view(), fds.view());
}

// Put a single shapelet source at zenith.
#[test]
#[cfg(any(feature = "cuda-double", feature = "cuda-single"))]
fn shapelet_zenith_gpu_list() {
    let obs = ObsParams::list();

    let mut srclist = SourceList::new();
    srclist.insert(
        "zenith".to_string(),
        Source {
            components: vec![get_simple_shapelet(
                obs.phase_centre,
                obs.flux_density_scale.clone(),
            )],
        },
    );
    // Get the component parameters via `ComponentListFFI`.
    let comps = ComponentListFFI::new(srclist, &obs.freqs, obs.phase_centre);
    let shapelet_uvs = comps.get_shapelet_uvs(obs.lst, &obs.xyzs);
    let (_, _, shapelets) = comps.to_c_types(Some(&shapelet_uvs));

    let uvws: Vec<cuda::UVW> = obs
        .uvws
        .iter()
        .map(|uvw| cuda::UVW {
            u: uvw.u as _,
            v: uvw.v as _,
            w: uvw.w as _,
        })
        .collect();
    let freqs: Vec<_> = obs.freqs.iter().map(|&f| f as _).collect();
    let shapelet_basis_values: Vec<_> = crate::shapelets::SHAPELET_BASIS_VALUES
        .iter()
        .map(|&f| f as _)
        .collect();

    // Ignore applying the beam.
    let mut visibilities = Array2::from_elem((obs.uvws.len(), obs.freqs.len()), Jones::default());
    unsafe {
        let addresses = cuda::init_model(
            obs.uvws.len(),
            obs.freqs.len(),
            crate::shapelets::SBF_L,
            crate::shapelets::SBF_N,
            crate::shapelets::SBF_C as _,
            crate::shapelets::SBF_DX as _,
            uvws.as_ptr(),
            freqs.as_ptr(),
            shapelet_basis_values.as_ptr(),
            visibilities.as_mut_ptr() as _,
        );
        cuda::model_shapelets(&shapelets, &addresses);
        cuda::copy_vis(&addresses);
        cuda::destroy(&addresses);
    }
    let fds = cuda_jones_to_jones(comps.shapelet_list_fds.view());
    assert_list_zenith_visibilities(visibilities.view(), fds.view());
}

// Put a single shapelet source just off zenith.
#[test]
#[cfg(any(feature = "cuda-double", feature = "cuda-single"))]
fn shapelet_off_zenith_gpu_list() {
    let obs = ObsParams::list();
    let pos = RADec::new_degrees(1.0, -27.0);

    let mut srclist = SourceList::new();
    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec![get_simple_shapelet(pos, obs.flux_density_scale.clone())],
        },
    );
    // Get the component parameters via `ComponentListFFI`.
    let comps = ComponentListFFI::new(srclist, &obs.freqs, obs.phase_centre);
    let shapelet_uvs = comps.get_shapelet_uvs(obs.lst, &obs.xyzs);
    let (_, _, shapelets) = comps.to_c_types(Some(&shapelet_uvs));

    let uvws: Vec<cuda::UVW> = obs
        .uvws
        .iter()
        .map(|uvw| cuda::UVW {
            u: uvw.u as _,
            v: uvw.v as _,
            w: uvw.w as _,
        })
        .collect();
    let freqs: Vec<_> = obs.freqs.iter().map(|&f| f as _).collect();
    let shapelet_basis_values: Vec<_> = crate::shapelets::SHAPELET_BASIS_VALUES
        .iter()
        .map(|&f| f as _)
        .collect();

    // Ignore applying the beam.
    let mut visibilities = Array2::from_elem((obs.uvws.len(), obs.freqs.len()), Jones::default());
    unsafe {
        let addresses = cuda::init_model(
            obs.uvws.len(),
            obs.freqs.len(),
            crate::shapelets::SBF_L,
            crate::shapelets::SBF_N,
            crate::shapelets::SBF_C as _,
            crate::shapelets::SBF_DX as _,
            uvws.as_ptr(),
            freqs.as_ptr(),
            shapelet_basis_values.as_ptr(),
            visibilities.as_mut_ptr() as _,
        );
        cuda::model_shapelets(&shapelets, &addresses);
        cuda::copy_vis(&addresses);
        cuda::destroy(&addresses);
    }
    let fds = cuda_jones_to_jones(comps.shapelet_list_fds.view());
    assert_list_off_zenith_visibilities(visibilities.view(), fds.view());
}
