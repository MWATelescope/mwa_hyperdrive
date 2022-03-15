// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Tests on generating sky-model visibilities with CUDA.
//!
//! These tests use the same expected values as the CPU tests. For this reason,
//! tests using the FEE have to be restricted to double-precision CUDA, as
//! single-precision CUDA will not match.

use approx::{assert_abs_diff_eq, assert_abs_diff_ne};
use marlu::cuda::DevicePointer;
use ndarray::prelude::*;
#[cfg(not(feature = "cuda-single"))]
use serial_test::serial;
use vec1::vec1;

use super::*;
use crate::model::cuda::CudaFloat;
use mwa_hyperdrive_common::{marlu, ndarray, vec1};
use mwa_hyperdrive_cuda as cuda;
use mwa_hyperdrive_srclist::Source;

/// Helper function to copy [UVW]s to the device.
fn copy_uvws(uvws: &[UVW]) -> DevicePointer<cuda::UVW> {
    unsafe {
        let cuda_uvws = uvws
            .iter()
            .map(|&uvw| cuda::UVW {
                u: uvw.u as CudaFloat,
                v: uvw.v as CudaFloat,
                w: uvw.w as CudaFloat,
            })
            .collect::<Vec<_>>();
        DevicePointer::copy_to_device(&cuda_uvws).unwrap()
    }
}

// Put a single point source at zenith.
#[test]
fn point_zenith_gpu() {
    let obs = ObsParams::new(true);
    let mut srclist = SourceList::new();
    srclist.insert(
        "zenith".to_string(),
        Source {
            components: vec1![get_simple_point(obs.phase_centre, FluxType::List,)],
        },
    );
    let mut visibilities = Array2::zeros((obs.uvws.len(), obs.freqs.len()));
    let modeller = obs.get_gpu_modeller(&srclist);
    unsafe {
        let d_uvws = copy_uvws(&obs.uvws);
        let result = modeller.model_points_inner(&d_uvws, obs.lst);
        assert!(result.is_ok());
        result.unwrap();
        modeller.copy_and_reset_vis(visibilities.view_mut());
    }
    assert_list_zenith_visibilities(visibilities.view(), "Failed on flux-type list");

    srclist.insert(
        "zenith".to_string(),
        Source {
            components: vec1![get_simple_point(obs.phase_centre, FluxType::PowerLaw)],
        },
    );
    visibilities.fill(Jones::default());
    let modeller = obs.get_gpu_modeller(&srclist);
    unsafe {
        let d_uvws = copy_uvws(&obs.uvws);
        let result = modeller.model_points_inner(&d_uvws, obs.lst);
        assert!(result.is_ok());
        result.unwrap();
        modeller.copy_and_reset_vis(visibilities.view_mut());
    }
    assert_power_law_zenith_visibilities(visibilities.view(), "Failed on flux-type power law");

    srclist.insert(
        "zenith".to_string(),
        Source {
            components: vec1![get_simple_point(obs.phase_centre, FluxType::CurvedPowerLaw)],
        },
    );
    visibilities.fill(Jones::default());
    let modeller = obs.get_gpu_modeller(&srclist);
    unsafe {
        let d_uvws = copy_uvws(&obs.uvws);
        let result = modeller.model_points_inner(&d_uvws, obs.lst);
        assert!(result.is_ok());
        result.unwrap();
        modeller.copy_and_reset_vis(visibilities.view_mut());
    }
    assert_curved_power_law_zenith_visibilities(
        visibilities.view(),
        "Failed on flux-type curved power law",
    );
}

#[test]
fn point_off_zenith_gpu() {
    let obs = ObsParams::new(true);
    let mut srclist = SourceList::new();
    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec1![get_simple_point(
                RADec::new_degrees(1.0, -27.0),
                FluxType::List
            )],
        },
    );
    let mut visibilities = Array2::zeros((obs.uvws.len(), obs.freqs.len()));
    let modeller = obs.get_gpu_modeller(&srclist);
    unsafe {
        let d_uvws = copy_uvws(&obs.uvws);
        let result = modeller.model_points_inner(&d_uvws, obs.lst);
        assert!(result.is_ok());
        result.unwrap();
        modeller.copy_and_reset_vis(visibilities.view_mut());
    }
    assert_list_off_zenith_visibilities(visibilities.view(), "Failed on flux-type list");

    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec1![get_simple_point(
                RADec::new_degrees(1.0, -27.0),
                FluxType::PowerLaw
            )],
        },
    );
    visibilities.fill(Jones::default());
    let modeller = obs.get_gpu_modeller(&srclist);
    unsafe {
        let d_uvws = copy_uvws(&obs.uvws);
        let result = modeller.model_points_inner(&d_uvws, obs.lst);
        assert!(result.is_ok());
        result.unwrap();
        modeller.copy_and_reset_vis(visibilities.view_mut());
    }
    assert_power_law_off_zenith_visibilities(visibilities.view(), "Failed on flux-type power law");

    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec1![get_simple_point(
                RADec::new_degrees(1.0, -27.0),
                FluxType::CurvedPowerLaw
            )],
        },
    );
    visibilities.fill(Jones::default());
    let modeller = obs.get_gpu_modeller(&srclist);
    unsafe {
        let d_uvws = copy_uvws(&obs.uvws);
        let result = modeller.model_points_inner(&d_uvws, obs.lst);
        assert!(result.is_ok());
        result.unwrap();
        modeller.copy_and_reset_vis(visibilities.view_mut());
    }
    assert_curved_power_law_off_zenith_visibilities(
        visibilities.view(),
        "Failed on flux-type curved power law",
    );
}

// Put a single Gaussian source at zenith.
#[test]
fn gaussian_zenith_gpu() {
    let obs = ObsParams::new(true);
    let mut srclist = SourceList::new();
    srclist.insert(
        "zenith".to_string(),
        Source {
            components: vec1![get_simple_gaussian(obs.phase_centre, FluxType::List)],
        },
    );
    let mut visibilities = Array2::zeros((obs.uvws.len(), obs.freqs.len()));
    let modeller = obs.get_gpu_modeller(&srclist);
    unsafe {
        let cuda_uvws = obs
            .uvws
            .iter()
            .map(|&uvw| cuda::UVW {
                u: uvw.u as CudaFloat,
                v: uvw.v as CudaFloat,
                w: uvw.w as CudaFloat,
            })
            .collect::<Vec<_>>();
        let d_uvws = DevicePointer::copy_to_device(&cuda_uvws).unwrap();
        let result = modeller.model_gaussians_inner(&d_uvws, obs.lst);
        assert!(result.is_ok());
        result.unwrap();
        modeller.copy_and_reset_vis(visibilities.view_mut());
    }
    assert_list_zenith_visibilities(visibilities.view(), "Failed on flux-type list");

    srclist.insert(
        "zenith".to_string(),
        Source {
            components: vec1![get_simple_gaussian(obs.phase_centre, FluxType::PowerLaw)],
        },
    );
    visibilities.fill(Jones::default());
    let modeller = obs.get_gpu_modeller(&srclist);
    unsafe {
        let cuda_uvws = obs
            .uvws
            .iter()
            .map(|&uvw| cuda::UVW {
                u: uvw.u as CudaFloat,
                v: uvw.v as CudaFloat,
                w: uvw.w as CudaFloat,
            })
            .collect::<Vec<_>>();
        let d_uvws = DevicePointer::copy_to_device(&cuda_uvws).unwrap();
        let result = modeller.model_gaussians_inner(&d_uvws, obs.lst);
        assert!(result.is_ok());
        result.unwrap();
        modeller.copy_and_reset_vis(visibilities.view_mut());
    }
    assert_power_law_zenith_visibilities(visibilities.view(), "Failed on flux-type power law");

    srclist.insert(
        "zenith".to_string(),
        Source {
            components: vec1![get_simple_gaussian(
                obs.phase_centre,
                FluxType::CurvedPowerLaw
            )],
        },
    );
    visibilities.fill(Jones::default());
    let modeller = obs.get_gpu_modeller(&srclist);
    unsafe {
        let cuda_uvws = obs
            .uvws
            .iter()
            .map(|&uvw| cuda::UVW {
                u: uvw.u as CudaFloat,
                v: uvw.v as CudaFloat,
                w: uvw.w as CudaFloat,
            })
            .collect::<Vec<_>>();
        let d_uvws = DevicePointer::copy_to_device(&cuda_uvws).unwrap();
        let result = modeller.model_gaussians_inner(&d_uvws, obs.lst);
        assert!(result.is_ok());
        result.unwrap();
        modeller.copy_and_reset_vis(visibilities.view_mut());
    }
    assert_curved_power_law_zenith_visibilities(
        visibilities.view(),
        "Failed on flux-type curved power law",
    );
}

// Put a single Gaussian source just off zenith.
#[test]
fn gaussian_off_zenith_gpu() {
    let obs = ObsParams::new(true);
    let mut srclist = SourceList::new();
    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec1![get_simple_gaussian(
                RADec::new_degrees(1.0, -27.0),
                FluxType::List
            )],
        },
    );
    let mut visibilities = Array2::zeros((obs.uvws.len(), obs.freqs.len()));
    let modeller = obs.get_gpu_modeller(&srclist);
    unsafe {
        let d_uvws = copy_uvws(&obs.uvws);
        let result = modeller.model_gaussians_inner(&d_uvws, obs.lst);
        assert!(result.is_ok());
        result.unwrap();
        modeller.copy_and_reset_vis(visibilities.view_mut());
    }
    assert_list_off_zenith_visibilities(visibilities.view(), "Failed on flux-type list");

    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec1![get_simple_gaussian(
                RADec::new_degrees(1.0, -27.0),
                FluxType::PowerLaw
            )],
        },
    );
    visibilities.fill(Jones::default());
    let modeller = obs.get_gpu_modeller(&srclist);
    unsafe {
        let d_uvws = copy_uvws(&obs.uvws);
        let result = modeller.model_gaussians_inner(&d_uvws, obs.lst);
        assert!(result.is_ok());
        result.unwrap();
        modeller.copy_and_reset_vis(visibilities.view_mut());
    }
    assert_power_law_off_zenith_visibilities(visibilities.view(), "Failed on flux-type power law");

    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec1![get_simple_gaussian(
                RADec::new_degrees(1.0, -27.0),
                FluxType::CurvedPowerLaw
            )],
        },
    );
    visibilities.fill(Jones::default());
    let modeller = obs.get_gpu_modeller(&srclist);
    unsafe {
        let d_uvws = copy_uvws(&obs.uvws);
        let result = modeller.model_gaussians_inner(&d_uvws, obs.lst);
        assert!(result.is_ok());
        result.unwrap();
        modeller.copy_and_reset_vis(visibilities.view_mut());
    }
    assert_curved_power_law_off_zenith_visibilities(
        visibilities.view(),
        "Failed on flux-type curved power law",
    );
}

// Put a single shapelet source at zenith.
#[test]
fn shapelet_zenith_gpu() {
    let obs = ObsParams::new(true);
    let mut srclist = SourceList::new();
    srclist.insert(
        "zenith".to_string(),
        Source {
            components: vec1![get_simple_shapelet(obs.phase_centre, FluxType::List)],
        },
    );
    let mut visibilities = Array2::zeros((obs.uvws.len(), obs.freqs.len()));
    let modeller = obs.get_gpu_modeller(&srclist);
    unsafe {
        let d_uvws = copy_uvws(&obs.uvws);
        let result = modeller.model_shapelets_inner(&d_uvws, obs.lst);
        assert!(result.is_ok());
        result.unwrap();
        modeller.copy_and_reset_vis(visibilities.view_mut());
    }
    assert_list_zenith_visibilities(visibilities.view(), "Failed on flux-type list");

    srclist.insert(
        "zenith".to_string(),
        Source {
            components: vec1![get_simple_shapelet(obs.phase_centre, FluxType::PowerLaw)],
        },
    );
    visibilities.fill(Jones::default());
    let modeller = obs.get_gpu_modeller(&srclist);
    unsafe {
        let d_uvws = copy_uvws(&obs.uvws);
        let result = modeller.model_shapelets_inner(&d_uvws, obs.lst);
        assert!(result.is_ok());
        result.unwrap();
        modeller.copy_and_reset_vis(visibilities.view_mut());
    }
    assert_power_law_zenith_visibilities(visibilities.view(), "Failed on flux-type power law");

    srclist.insert(
        "zenith".to_string(),
        Source {
            components: vec1![get_simple_shapelet(
                obs.phase_centre,
                FluxType::CurvedPowerLaw
            )],
        },
    );
    visibilities.fill(Jones::default());
    let modeller = obs.get_gpu_modeller(&srclist);
    unsafe {
        let d_uvws = copy_uvws(&obs.uvws);
        let result = modeller.model_shapelets_inner(&d_uvws, obs.lst);
        assert!(result.is_ok());
        result.unwrap();
        modeller.copy_and_reset_vis(visibilities.view_mut());
    }
    assert_curved_power_law_zenith_visibilities(
        visibilities.view(),
        "Failed on flux-type curved power law",
    );
}

// Put a single shapelet source just off zenith.
#[test]
fn shapelet_off_zenith_gpu() {
    let obs = ObsParams::new(true);
    let mut srclist = SourceList::new();
    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec1![get_simple_shapelet(
                RADec::new_degrees(1.0, -27.0),
                FluxType::List
            )],
        },
    );
    let mut visibilities = Array2::zeros((obs.uvws.len(), obs.freqs.len()));
    let modeller = obs.get_gpu_modeller(&srclist);
    unsafe {
        let d_uvws = copy_uvws(&obs.uvws);
        let result = modeller.model_shapelets_inner(&d_uvws, obs.lst);
        assert!(result.is_ok());
        result.unwrap();
        modeller.copy_and_reset_vis(visibilities.view_mut());
    }
    assert_list_off_zenith_visibilities(visibilities.view(), "Failed on flux-type list");

    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec1![get_simple_shapelet(
                RADec::new_degrees(1.0, -27.0),
                FluxType::PowerLaw
            )],
        },
    );
    visibilities.fill(Jones::default());
    let modeller = obs.get_gpu_modeller(&srclist);
    unsafe {
        let d_uvws = copy_uvws(&obs.uvws);
        let result = modeller.model_shapelets_inner(&d_uvws, obs.lst);
        assert!(result.is_ok());
        result.unwrap();
        modeller.copy_and_reset_vis(visibilities.view_mut());
    }
    assert_power_law_off_zenith_visibilities(visibilities.view(), "Failed on flux-type power law");

    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec1![get_simple_shapelet(
                RADec::new_degrees(1.0, -27.0),
                FluxType::CurvedPowerLaw
            )],
        },
    );
    visibilities.fill(Jones::default());
    let modeller = obs.get_gpu_modeller(&srclist);
    unsafe {
        let d_uvws = copy_uvws(&obs.uvws);
        let result = modeller.model_shapelets_inner(&d_uvws, obs.lst);
        assert!(result.is_ok());
        result.unwrap();
        modeller.copy_and_reset_vis(visibilities.view_mut());
    }
    assert_curved_power_law_off_zenith_visibilities(
        visibilities.view(),
        "Failed on flux-type curved power law",
    );
}

// Put a single point source at zenith.
#[test]
#[serial]
#[cfg(not(feature = "cuda-single"))]
fn point_zenith_gpu_fee() {
    let obs = ObsParams::new(false);
    let mut srclist = SourceList::new();
    srclist.insert(
        "zenith".to_string(),
        Source {
            components: vec1![get_simple_point(obs.phase_centre, FluxType::List,)],
        },
    );
    let mut visibilities = Array2::zeros((obs.uvws.len(), obs.freqs.len()));
    let modeller = obs.get_gpu_modeller(&srclist);
    unsafe {
        let d_uvws = copy_uvws(&obs.uvws);
        let result = modeller.model_points_inner(&d_uvws, obs.lst);
        assert!(result.is_ok());
        result.unwrap();
        modeller.copy_and_reset_vis(visibilities.view_mut());
    }
    assert_list_zenith_visibilities_fee(visibilities.view(), "Failed on flux-type list");

    srclist.insert(
        "zenith".to_string(),
        Source {
            components: vec1![get_simple_point(obs.phase_centre, FluxType::PowerLaw)],
        },
    );
    visibilities.fill(Jones::default());
    let modeller = obs.get_gpu_modeller(&srclist);
    unsafe {
        let d_uvws = copy_uvws(&obs.uvws);
        let result = modeller.model_points_inner(&d_uvws, obs.lst);
        assert!(result.is_ok());
        result.unwrap();
        modeller.copy_and_reset_vis(visibilities.view_mut());
    }
    assert_power_law_zenith_visibilities_fee(visibilities.view(), "Failed on flux-type power law");

    srclist.insert(
        "zenith".to_string(),
        Source {
            components: vec1![get_simple_point(obs.phase_centre, FluxType::CurvedPowerLaw)],
        },
    );
    visibilities.fill(Jones::default());
    let modeller = obs.get_gpu_modeller(&srclist);
    unsafe {
        let d_uvws = copy_uvws(&obs.uvws);
        let result = modeller.model_points_inner(&d_uvws, obs.lst);
        assert!(result.is_ok());
        result.unwrap();
        modeller.copy_and_reset_vis(visibilities.view_mut());
    }
    assert_curved_power_law_zenith_visibilities_fee(
        visibilities.view(),
        "Failed on flux-type curved power law",
    );
}

#[test]
#[serial]
#[cfg(not(feature = "cuda-single"))]
fn point_off_zenith_gpu_fee() {
    let obs = ObsParams::new(false);
    let mut srclist = SourceList::new();
    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec1![get_simple_point(
                RADec::new_degrees(1.0, -27.0),
                FluxType::List
            )],
        },
    );
    let mut visibilities = Array2::zeros((obs.uvws.len(), obs.freqs.len()));
    let modeller = obs.get_gpu_modeller(&srclist);
    unsafe {
        let d_uvws = copy_uvws(&obs.uvws);
        let result = modeller.model_points_inner(&d_uvws, obs.lst);
        assert!(result.is_ok());
        result.unwrap();
        modeller.copy_and_reset_vis(visibilities.view_mut());
    }
    assert_list_off_zenith_visibilities_fee(visibilities.view(), "Failed on flux-type list");

    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec1![get_simple_point(
                RADec::new_degrees(1.0, -27.0),
                FluxType::PowerLaw
            )],
        },
    );
    visibilities.fill(Jones::default());
    let modeller = obs.get_gpu_modeller(&srclist);
    unsafe {
        let d_uvws = copy_uvws(&obs.uvws);
        let result = modeller.model_points_inner(&d_uvws, obs.lst);
        assert!(result.is_ok());
        result.unwrap();
        modeller.copy_and_reset_vis(visibilities.view_mut());
    }
    assert_power_law_off_zenith_visibilities_fee(
        visibilities.view(),
        "Failed on flux-type power law",
    );

    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec1![get_simple_point(
                RADec::new_degrees(1.0, -27.0),
                FluxType::CurvedPowerLaw
            )],
        },
    );
    visibilities.fill(Jones::default());
    let modeller = obs.get_gpu_modeller(&srclist);
    unsafe {
        let d_uvws = copy_uvws(&obs.uvws);
        let result = modeller.model_points_inner(&d_uvws, obs.lst);
        assert!(result.is_ok());
        result.unwrap();
        modeller.copy_and_reset_vis(visibilities.view_mut());
    }
    assert_curved_power_law_off_zenith_visibilities_fee(
        visibilities.view(),
        "Failed on flux-type curved power law",
    );
}

// Put a single Gaussian source at zenith.
#[test]
#[serial]
#[cfg(not(feature = "cuda-single"))]
fn gaussian_zenith_gpu_fee() {
    let obs = ObsParams::new(false);
    let mut srclist = SourceList::new();
    srclist.insert(
        "zenith".to_string(),
        Source {
            components: vec1![get_simple_gaussian(obs.phase_centre, FluxType::List)],
        },
    );
    let mut visibilities = Array2::zeros((obs.uvws.len(), obs.freqs.len()));
    let modeller = obs.get_gpu_modeller(&srclist);
    unsafe {
        let cuda_uvws = obs
            .uvws
            .iter()
            .map(|&uvw| cuda::UVW {
                u: uvw.u as CudaFloat,
                v: uvw.v as CudaFloat,
                w: uvw.w as CudaFloat,
            })
            .collect::<Vec<_>>();
        let d_uvws = DevicePointer::copy_to_device(&cuda_uvws).unwrap();
        let result = modeller.model_gaussians_inner(&d_uvws, obs.lst);
        assert!(result.is_ok());
        result.unwrap();
        modeller.copy_and_reset_vis(visibilities.view_mut());
    }
    assert_list_zenith_visibilities_fee(visibilities.view(), "Failed on flux-type list");

    srclist.insert(
        "zenith".to_string(),
        Source {
            components: vec1![get_simple_gaussian(obs.phase_centre, FluxType::PowerLaw)],
        },
    );
    visibilities.fill(Jones::default());
    let modeller = obs.get_gpu_modeller(&srclist);
    unsafe {
        let cuda_uvws = obs
            .uvws
            .iter()
            .map(|&uvw| cuda::UVW {
                u: uvw.u as CudaFloat,
                v: uvw.v as CudaFloat,
                w: uvw.w as CudaFloat,
            })
            .collect::<Vec<_>>();
        let d_uvws = DevicePointer::copy_to_device(&cuda_uvws).unwrap();
        let result = modeller.model_gaussians_inner(&d_uvws, obs.lst);
        assert!(result.is_ok());
        result.unwrap();
        modeller.copy_and_reset_vis(visibilities.view_mut());
    }
    assert_power_law_zenith_visibilities_fee(visibilities.view(), "Failed on flux-type power law");

    srclist.insert(
        "zenith".to_string(),
        Source {
            components: vec1![get_simple_gaussian(
                obs.phase_centre,
                FluxType::CurvedPowerLaw
            )],
        },
    );
    visibilities.fill(Jones::default());
    let modeller = obs.get_gpu_modeller(&srclist);
    unsafe {
        let cuda_uvws = obs
            .uvws
            .iter()
            .map(|&uvw| cuda::UVW {
                u: uvw.u as CudaFloat,
                v: uvw.v as CudaFloat,
                w: uvw.w as CudaFloat,
            })
            .collect::<Vec<_>>();
        let d_uvws = DevicePointer::copy_to_device(&cuda_uvws).unwrap();
        let result = modeller.model_gaussians_inner(&d_uvws, obs.lst);
        assert!(result.is_ok());
        result.unwrap();
        modeller.copy_and_reset_vis(visibilities.view_mut());
    }
    assert_curved_power_law_zenith_visibilities_fee(
        visibilities.view(),
        "Failed on flux-type curved power law",
    );
}

// Put a single Gaussian source just off zenith.
#[test]
#[serial]
#[cfg(not(feature = "cuda-single"))]
fn gaussian_off_zenith_gpu_fee() {
    let obs = ObsParams::new(false);
    let mut srclist = SourceList::new();
    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec1![get_simple_gaussian(
                RADec::new_degrees(1.0, -27.0),
                FluxType::List
            )],
        },
    );
    let mut visibilities = Array2::zeros((obs.uvws.len(), obs.freqs.len()));
    let modeller = obs.get_gpu_modeller(&srclist);
    unsafe {
        let d_uvws = copy_uvws(&obs.uvws);
        let result = modeller.model_gaussians_inner(&d_uvws, obs.lst);
        assert!(result.is_ok());
        result.unwrap();
        modeller.copy_and_reset_vis(visibilities.view_mut());
    }
    assert_list_off_zenith_visibilities_fee(visibilities.view(), "Failed on flux-type list");

    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec1![get_simple_gaussian(
                RADec::new_degrees(1.0, -27.0),
                FluxType::PowerLaw
            )],
        },
    );
    visibilities.fill(Jones::default());
    let modeller = obs.get_gpu_modeller(&srclist);
    unsafe {
        let d_uvws = copy_uvws(&obs.uvws);
        let result = modeller.model_gaussians_inner(&d_uvws, obs.lst);
        assert!(result.is_ok());
        result.unwrap();
        modeller.copy_and_reset_vis(visibilities.view_mut());
    }
    assert_power_law_off_zenith_visibilities_fee(
        visibilities.view(),
        "Failed on flux-type power law",
    );

    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec1![get_simple_gaussian(
                RADec::new_degrees(1.0, -27.0),
                FluxType::CurvedPowerLaw
            )],
        },
    );
    visibilities.fill(Jones::default());
    let modeller = obs.get_gpu_modeller(&srclist);
    unsafe {
        let d_uvws = copy_uvws(&obs.uvws);
        let result = modeller.model_gaussians_inner(&d_uvws, obs.lst);
        assert!(result.is_ok());
        result.unwrap();
        modeller.copy_and_reset_vis(visibilities.view_mut());
    }
    assert_curved_power_law_off_zenith_visibilities_fee(
        visibilities.view(),
        "Failed on flux-type curved power law",
    );
}

// Put a single shapelet source at zenith.
#[test]
#[serial]
#[cfg(not(feature = "cuda-single"))]
fn shapelet_zenith_gpu_fee() {
    let obs = ObsParams::new(false);
    let mut srclist = SourceList::new();
    srclist.insert(
        "zenith".to_string(),
        Source {
            components: vec1![get_simple_shapelet(obs.phase_centre, FluxType::List)],
        },
    );
    let mut visibilities = Array2::zeros((obs.uvws.len(), obs.freqs.len()));
    let modeller = obs.get_gpu_modeller(&srclist);
    unsafe {
        let d_uvws = copy_uvws(&obs.uvws);
        let result = modeller.model_shapelets_inner(&d_uvws, obs.lst);
        assert!(result.is_ok());
        result.unwrap();
        modeller.copy_and_reset_vis(visibilities.view_mut());
    }
    assert_list_zenith_visibilities_fee(visibilities.view(), "Failed on flux-type list");

    srclist.insert(
        "zenith".to_string(),
        Source {
            components: vec1![get_simple_shapelet(obs.phase_centre, FluxType::PowerLaw)],
        },
    );
    visibilities.fill(Jones::default());
    let modeller = obs.get_gpu_modeller(&srclist);
    unsafe {
        let d_uvws = copy_uvws(&obs.uvws);
        let result = modeller.model_shapelets_inner(&d_uvws, obs.lst);
        assert!(result.is_ok());
        result.unwrap();
        modeller.copy_and_reset_vis(visibilities.view_mut());
    }
    assert_power_law_zenith_visibilities_fee(visibilities.view(), "Failed on flux-type power law");

    srclist.insert(
        "zenith".to_string(),
        Source {
            components: vec1![get_simple_shapelet(
                obs.phase_centre,
                FluxType::CurvedPowerLaw
            )],
        },
    );
    visibilities.fill(Jones::default());
    let modeller = obs.get_gpu_modeller(&srclist);
    unsafe {
        let d_uvws = copy_uvws(&obs.uvws);
        let result = modeller.model_shapelets_inner(&d_uvws, obs.lst);
        assert!(result.is_ok());
        result.unwrap();
        modeller.copy_and_reset_vis(visibilities.view_mut());
    }
    assert_curved_power_law_zenith_visibilities_fee(
        visibilities.view(),
        "Failed on flux-type curved power law",
    );
}

// Put a single shapelet source just off zenith.
#[test]
#[serial]
#[cfg(not(feature = "cuda-single"))]
fn shapelet_off_zenith_gpu_fee() {
    let obs = ObsParams::new(false);
    let mut srclist = SourceList::new();
    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec1![get_simple_shapelet(
                RADec::new_degrees(1.0, -27.0),
                FluxType::List
            )],
        },
    );
    let mut visibilities = Array2::zeros((obs.uvws.len(), obs.freqs.len()));
    let modeller = obs.get_gpu_modeller(&srclist);
    unsafe {
        let d_uvws = copy_uvws(&obs.uvws);
        let result = modeller.model_shapelets_inner(&d_uvws, obs.lst);
        assert!(result.is_ok());
        result.unwrap();
        modeller.copy_and_reset_vis(visibilities.view_mut());
    }
    assert_list_off_zenith_visibilities_fee(visibilities.view(), "Failed on flux-type list");

    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec1![get_simple_shapelet(
                RADec::new_degrees(1.0, -27.0),
                FluxType::PowerLaw
            )],
        },
    );
    visibilities.fill(Jones::default());
    let modeller = obs.get_gpu_modeller(&srclist);
    unsafe {
        let d_uvws = copy_uvws(&obs.uvws);
        let result = modeller.model_shapelets_inner(&d_uvws, obs.lst);
        assert!(result.is_ok());
        result.unwrap();
        modeller.copy_and_reset_vis(visibilities.view_mut());
    }
    assert_power_law_off_zenith_visibilities_fee(
        visibilities.view(),
        "Failed on flux-type power law",
    );

    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec1![get_simple_shapelet(
                RADec::new_degrees(1.0, -27.0),
                FluxType::CurvedPowerLaw
            )],
        },
    );
    visibilities.fill(Jones::default());
    let modeller = obs.get_gpu_modeller(&srclist);
    unsafe {
        let d_uvws = copy_uvws(&obs.uvws);
        let result = modeller.model_shapelets_inner(&d_uvws, obs.lst);
        assert!(result.is_ok());
        result.unwrap();
        modeller.copy_and_reset_vis(visibilities.view_mut());
    }
    assert_curved_power_law_off_zenith_visibilities_fee(
        visibilities.view(),
        "Failed on flux-type curved power law",
    );
}

/// This test checks that beam responses are applied properly. The CUDA code
/// previously had a bug where the wrong beam response *might* have been applied
/// to the wrong component. Put multiple components with different flux types in
/// a source list and model it.
#[test]
#[serial]
#[cfg(not(feature = "cuda-single"))]
fn beam_responses_apply_properly_power_law_and_list() {
    let obs = ObsParams::new(false);
    let mut srclist = SourceList::new();
    srclist.insert(
        "mixed".to_string(),
        Source {
            components: vec1![
                get_simple_point(obs.phase_centre, FluxType::PowerLaw),
                get_simple_point(RADec::new_degrees(45.0, 18.0), FluxType::List)
            ],
        },
    );
    let mut visibilities = Array2::zeros((obs.uvws.len(), obs.freqs.len()));
    let modeller = obs.get_gpu_modeller(&srclist);
    unsafe {
        let d_uvws = copy_uvws(&obs.uvws);
        let result = modeller.model_points_inner(&d_uvws, obs.lst);
        assert!(result.is_ok());
        result.unwrap();
        modeller.copy_and_reset_vis(visibilities.view_mut());
    }

    // The visibilities should be very similar to having only the zenith
    // power-law component, because the list component is far from the pointing
    // centre. The expected values are taken from
    // `assert_power_law_zenith_visibilities_fee`.
    assert_abs_diff_eq!(
        TestJones::from(visibilities[(0, 0)]),
        TestJones::from([
            Complex::new(9.995525e-1, 0.0),
            Complex::new(-5.405832e-4, -5.00542e-6),
            Complex::new(-5.405832e-4, 5.00542e-6),
            Complex::new(9.9958146e-1, 0.0)
        ]),
        epsilon = 8e-4
    );

    srclist.insert(
        "mixed".to_string(),
        Source {
            components: vec1![
                // Every component type needs to be checked.
                get_simple_gaussian(obs.phase_centre, FluxType::PowerLaw),
                get_simple_gaussian(RADec::new_degrees(45.0, 18.0), FluxType::List)
            ],
        },
    );
    let mut visibilities = Array2::zeros((obs.uvws.len(), obs.freqs.len()));
    let modeller = obs.get_gpu_modeller(&srclist);
    unsafe {
        let d_uvws = copy_uvws(&obs.uvws);
        let result = modeller.model_gaussians_inner(&d_uvws, obs.lst);
        assert!(result.is_ok());
        result.unwrap();
        modeller.copy_and_reset_vis(visibilities.view_mut());
    }

    assert_abs_diff_eq!(
        TestJones::from(visibilities[(0, 0)]),
        TestJones::from([
            Complex::new(9.995525e-1, 0.0),
            Complex::new(-5.405832e-4, -5.00542e-6),
            Complex::new(-5.405832e-4, 5.00542e-6),
            Complex::new(9.9958146e-1, 0.0)
        ]),
        epsilon = 8e-4
    );

    srclist.insert(
        "mixed".to_string(),
        Source {
            components: vec1![
                get_simple_shapelet(obs.phase_centre, FluxType::PowerLaw),
                get_simple_shapelet(RADec::new_degrees(45.0, 18.0), FluxType::List)
            ],
        },
    );
    let mut visibilities = Array2::zeros((obs.uvws.len(), obs.freqs.len()));
    let modeller = obs.get_gpu_modeller(&srclist);
    unsafe {
        let d_uvws = copy_uvws(&obs.uvws);
        let result = modeller.model_shapelets_inner(&d_uvws, obs.lst);
        assert!(result.is_ok());
        result.unwrap();
        modeller.copy_and_reset_vis(visibilities.view_mut());
    }

    assert_abs_diff_eq!(
        TestJones::from(visibilities[(0, 0)]),
        TestJones::from([
            Complex::new(9.995525e-1, 0.0),
            Complex::new(-5.405832e-4, -5.00542e-6),
            Complex::new(-5.405832e-4, 5.00542e-6),
            Complex::new(9.9958146e-1, 0.0)
        ]),
        epsilon = 8e-4
    );
}

/// Similar to above.
#[test]
#[serial]
#[cfg(not(feature = "cuda-single"))]
fn beam_responses_apply_properly_power_law_and_curved_power_law() {
    let obs = ObsParams::new(false);
    let mut srclist = SourceList::new();
    srclist.insert(
        "mixed".to_string(),
        Source {
            components: vec1![
                get_simple_point(obs.phase_centre, FluxType::PowerLaw),
                get_simple_point(RADec::new_degrees(45.0, 18.0), FluxType::CurvedPowerLaw)
            ],
        },
    );
    let mut visibilities = Array2::zeros((obs.uvws.len(), obs.freqs.len()));
    let modeller = obs.get_gpu_modeller(&srclist);
    unsafe {
        let d_uvws = copy_uvws(&obs.uvws);
        let result = modeller.model_points_inner(&d_uvws, obs.lst);
        assert!(result.is_ok());
        result.unwrap();
        modeller.copy_and_reset_vis(visibilities.view_mut());
    }

    // The visibilities should be very similar to having only the zenith
    // power-law component, because the list component is far from the pointing
    // centre. The expected values are taken from
    // `assert_power_law_zenith_visibilities_fee`.
    assert_abs_diff_eq!(
        TestJones::from(visibilities[(0, 0)]),
        TestJones::from([
            Complex::new(9.995525e-1, 0.0),
            Complex::new(-5.405832e-4, -5.00542e-6),
            Complex::new(-5.405832e-4, 5.00542e-6),
            Complex::new(9.9958146e-1, 0.0)
        ]),
        epsilon = 8e-4
    );

    srclist.insert(
        "mixed".to_string(),
        Source {
            components: vec1![
                // Every component type needs to be checked.
                get_simple_gaussian(obs.phase_centre, FluxType::PowerLaw),
                get_simple_gaussian(RADec::new_degrees(45.0, 18.0), FluxType::CurvedPowerLaw)
            ],
        },
    );
    let mut visibilities = Array2::zeros((obs.uvws.len(), obs.freqs.len()));
    let modeller = obs.get_gpu_modeller(&srclist);
    unsafe {
        let d_uvws = copy_uvws(&obs.uvws);
        let result = modeller.model_gaussians_inner(&d_uvws, obs.lst);
        assert!(result.is_ok());
        result.unwrap();
        modeller.copy_and_reset_vis(visibilities.view_mut());
    }

    assert_abs_diff_eq!(
        TestJones::from(visibilities[(0, 0)]),
        TestJones::from([
            Complex::new(9.995525e-1, 0.0),
            Complex::new(-5.405832e-4, -5.00542e-6),
            Complex::new(-5.405832e-4, 5.00542e-6),
            Complex::new(9.9958146e-1, 0.0)
        ]),
        epsilon = 8e-4
    );

    srclist.insert(
        "mixed".to_string(),
        Source {
            components: vec1![
                get_simple_shapelet(obs.phase_centre, FluxType::PowerLaw),
                get_simple_shapelet(RADec::new_degrees(45.0, 18.0), FluxType::CurvedPowerLaw)
            ],
        },
    );
    let mut visibilities = Array2::zeros((obs.uvws.len(), obs.freqs.len()));
    let modeller = obs.get_gpu_modeller(&srclist);
    unsafe {
        let d_uvws = copy_uvws(&obs.uvws);
        let result = modeller.model_shapelets_inner(&d_uvws, obs.lst);
        assert!(result.is_ok());
        result.unwrap();
        modeller.copy_and_reset_vis(visibilities.view_mut());
    }

    assert_abs_diff_eq!(
        TestJones::from(visibilities[(0, 0)]),
        TestJones::from([
            Complex::new(9.995525e-1, 0.0),
            Complex::new(-5.405832e-4, -5.00542e-6),
            Complex::new(-5.405832e-4, 5.00542e-6),
            Complex::new(9.9958146e-1, 0.0)
        ]),
        epsilon = 8e-4
    );
}

/// Similar to above.
#[test]
#[serial]
#[cfg(not(feature = "cuda-single"))]
fn beam_responses_apply_properly_curved_power_law_and_list() {
    let obs = ObsParams::new(false);
    let mut srclist = SourceList::new();
    srclist.insert(
        "mixed".to_string(),
        Source {
            components: vec1![
                get_simple_point(obs.phase_centre, FluxType::CurvedPowerLaw),
                get_simple_point(RADec::new_degrees(45.0, 18.0), FluxType::List)
            ],
        },
    );
    let mut visibilities = Array2::zeros((obs.uvws.len(), obs.freqs.len()));
    let modeller = obs.get_gpu_modeller(&srclist);
    unsafe {
        let d_uvws = copy_uvws(&obs.uvws);
        let result = modeller.model_points_inner(&d_uvws, obs.lst);
        assert!(result.is_ok());
        result.unwrap();
        modeller.copy_and_reset_vis(visibilities.view_mut());
    }

    // The visibilities should be very similar to having only the zenith
    // power-law component, because the list component is far from the pointing
    // centre. The expected values are taken from
    // `assert_power_law_zenith_visibilities_fee`.
    assert_abs_diff_eq!(
        TestJones::from(visibilities[(0, 0)]),
        TestJones::from([
            Complex::new(9.995525e-1, 0.0),
            Complex::new(-5.405832e-4, -5.00542e-6),
            Complex::new(-5.405832e-4, 5.00542e-6),
            Complex::new(9.9958146e-1, 0.0)
        ]),
        epsilon = 8e-4
    );

    srclist.insert(
        "mixed".to_string(),
        Source {
            components: vec1![
                // Every component type needs to be checked.
                get_simple_gaussian(obs.phase_centre, FluxType::CurvedPowerLaw),
                get_simple_gaussian(RADec::new_degrees(45.0, 18.0), FluxType::List)
            ],
        },
    );
    let mut visibilities = Array2::zeros((obs.uvws.len(), obs.freqs.len()));
    let modeller = obs.get_gpu_modeller(&srclist);
    unsafe {
        let d_uvws = copy_uvws(&obs.uvws);
        let result = modeller.model_gaussians_inner(&d_uvws, obs.lst);
        assert!(result.is_ok());
        result.unwrap();
        modeller.copy_and_reset_vis(visibilities.view_mut());
    }

    assert_abs_diff_eq!(
        TestJones::from(visibilities[(0, 0)]),
        TestJones::from([
            Complex::new(9.995525e-1, 0.0),
            Complex::new(-5.405832e-4, -5.00542e-6),
            Complex::new(-5.405832e-4, 5.00542e-6),
            Complex::new(9.9958146e-1, 0.0)
        ]),
        epsilon = 8e-4
    );

    srclist.insert(
        "mixed".to_string(),
        Source {
            components: vec1![
                get_simple_shapelet(obs.phase_centre, FluxType::CurvedPowerLaw),
                get_simple_shapelet(RADec::new_degrees(45.0, 18.0), FluxType::List)
            ],
        },
    );
    let mut visibilities = Array2::zeros((obs.uvws.len(), obs.freqs.len()));
    let modeller = obs.get_gpu_modeller(&srclist);
    unsafe {
        let d_uvws = copy_uvws(&obs.uvws);
        let result = modeller.model_shapelets_inner(&d_uvws, obs.lst);
        assert!(result.is_ok());
        result.unwrap();
        modeller.copy_and_reset_vis(visibilities.view_mut());
    }

    assert_abs_diff_eq!(
        TestJones::from(visibilities[(0, 0)]),
        TestJones::from([
            Complex::new(9.995525e-1, 0.0),
            Complex::new(-5.405832e-4, -5.00542e-6),
            Complex::new(-5.405832e-4, 5.00542e-6),
            Complex::new(9.9958146e-1, 0.0)
        ]),
        epsilon = 8e-4
    );
}

// Test that all visibilities get cleared after doing a copy.
#[test]
fn copy_reset_cuda_vis_works() {
    let obs = ObsParams::new(true);
    let mut srclist = SourceList::new();
    srclist.insert(
        "zenith".to_string(),
        Source {
            components: vec1![get_simple_point(obs.phase_centre, FluxType::List)],
        },
    );
    let mut visibilities = Array2::zeros((obs.uvws.len(), obs.freqs.len()));
    let modeller = obs.get_gpu_modeller(&srclist);
    unsafe {
        let cuda_uvws = obs
            .uvws
            .iter()
            .map(|&uvw| cuda::UVW {
                u: uvw.u as CudaFloat,
                v: uvw.v as CudaFloat,
                w: uvw.w as CudaFloat,
            })
            .collect::<Vec<_>>();
        let d_uvws = DevicePointer::copy_to_device(&cuda_uvws).unwrap();
        let result = modeller.model_points_inner(&d_uvws, obs.lst);
        assert!(result.is_ok());
        result.unwrap();

        // Copy the visibilities; these are not all zero.
        modeller.copy_and_reset_vis(visibilities.view_mut());
        assert_abs_diff_ne!(
            visibilities.mapv(TestJones::from),
            Array2::from_elem((obs.uvws.len(), obs.freqs.len()), Jones::default())
                .mapv(TestJones::from)
        );
        // (In fact, none are zero.)
        visibilities.iter().for_each(|&v| {
            assert_abs_diff_ne!(TestJones::from(v), TestJones::from(Jones::default()));
        });

        // Copy the visibilities again; because they've been reset before, these
        // are all zero.
        modeller.copy_and_reset_vis(visibilities.view_mut());
        assert_abs_diff_eq!(
            visibilities.mapv(TestJones::from),
            Array2::from_elem((obs.uvws.len(), obs.freqs.len()), Jones::default())
                .mapv(TestJones::from)
        );
    }
}
