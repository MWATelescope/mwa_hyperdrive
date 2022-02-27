// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Tests on generating sky-model visibilities.

use approx::assert_abs_diff_ne;
use marlu::cuda::DevicePointer;
use ndarray::prelude::*;

use super::*;
use crate::model::cuda::CudaFloat;
use mwa_hyperdrive_common::{marlu, ndarray};
use mwa_hyperdrive_cuda as cuda;

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
fn point_zenith_gpu_list() {
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
    let mut visibilities = Array2::zeros((obs.uvws.len(), obs.freqs.len()));
    let modeller = obs.get_gpu_modeller(&srclist);
    unsafe {
        let d_uvws = copy_uvws(&obs.uvws);
        let result = modeller.model_points_inner(&d_uvws, obs.lst);
        assert!(result.is_ok());
        result.unwrap();
        modeller.copy_and_reset_vis(visibilities.view_mut());
    }
    assert_list_zenith_visibilities(visibilities.view());
}

#[test]
fn point_off_zenith_gpu_list() {
    let obs = ObsParams::list();
    let pos = RADec::new_degrees(1.0, -27.0);
    let mut srclist = SourceList::new();
    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec![get_simple_point(pos, obs.flux_density_scale.clone())],
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
    assert_list_off_zenith_visibilities(visibilities.view());
}

// Put a single Gaussian source at zenith.
#[test]
fn gaussian_zenith_gpu_list() {
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
    assert_list_zenith_visibilities(visibilities.view());
}

// Put a single Gaussian source just off zenith.
#[test]
fn gaussian_off_zenith_gpu_list() {
    let obs = ObsParams::list();
    let pos = RADec::new_degrees(1.0, -27.0);
    let mut srclist = SourceList::new();
    srclist.insert(
        "off_zenith".to_string(),
        Source {
            components: vec![get_simple_gaussian(pos, obs.flux_density_scale.clone())],
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
    assert_list_off_zenith_visibilities(visibilities.view());
}

// Put a single shapelet source at zenith.
#[test]
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
    let mut visibilities = Array2::zeros((obs.uvws.len(), obs.freqs.len()));
    let modeller = obs.get_gpu_modeller(&srclist);
    unsafe {
        let d_uvws = copy_uvws(&obs.uvws);
        let result = modeller.model_shapelets_inner(&d_uvws, obs.lst);
        assert!(result.is_ok());
        result.unwrap();
        modeller.copy_and_reset_vis(visibilities.view_mut());
    }
    assert_list_zenith_visibilities(visibilities.view());
}

// Put a single shapelet source just off zenith.
#[test]
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
    let mut visibilities = Array2::zeros((obs.uvws.len(), obs.freqs.len()));
    let modeller = obs.get_gpu_modeller(&srclist);
    unsafe {
        let d_uvws = copy_uvws(&obs.uvws);
        let result = modeller.model_shapelets_inner(&d_uvws, obs.lst);
        assert!(result.is_ok());
        result.unwrap();
        modeller.copy_and_reset_vis(visibilities.view_mut());
    }
    assert_list_off_zenith_visibilities(visibilities.view());
}

// Test that all visibilities get cleared after doing a copy.
#[test]
fn copy_reset_cuda_vis_works() {
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
