// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Tests on generating sky-model visibilities.

use cuda::SkyModellerCuda;
use mwa_hyperdrive_cuda as cuda;

use super::*;

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
                obs.phase_centre,
                obs.flux_density_scale.clone(),
            )],
        },
    );
    let modeller = obs.get_gpu_modeller(srclist);

    // Ignore applying the beam.
    let mut visibilities = Array2::from_elem((obs.uvws.len(), obs.freqs.len()), Jones::default());
    unsafe {
        let result = modeller.model_timestep(visibilities.view_mut(), obs.lst, &obs.uvws);
        assert!(result.is_ok());
        result.unwrap();
    }
    assert_list_zenith_visibilities(visibilities.view());
}

#[test]
#[cfg(feature = "cuda")]
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
    let modeller = unsafe {
        let result = SkyModellerCuda::new(
            obs.beam.deref(),
            &srclist,
            &obs.freqs,
            &obs.xyzs,
            &[],
            obs.phase_centre,
            obs.array_latitude_rad,
            &crate::shapelets::SHAPELET_BASIS_VALUES,
            crate::shapelets::SBF_L,
            crate::shapelets::SBF_N,
            crate::shapelets::SBF_C,
            crate::shapelets::SBF_DX,
        );
        assert!(result.is_ok());
        result.unwrap()
    };

    // Ignore applying the beam.
    let mut visibilities = Array2::from_elem((obs.uvws.len(), obs.freqs.len()), Jones::default());
    unsafe {
        let result = modeller.model_timestep(visibilities.view_mut(), obs.lst, &obs.uvws);
        assert!(result.is_ok());
        result.unwrap();
    }
    assert_list_off_zenith_visibilities(visibilities.view());
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
                obs.phase_centre,
                obs.flux_density_scale.clone(),
            )],
        },
    );
    // Get the component parameters via `SkyModellerCuda`.
    let modeller = unsafe {
        let result = SkyModellerCuda::new(
            obs.beam.deref(),
            &srclist,
            &obs.freqs,
            &obs.xyzs,
            &[],
            obs.phase_centre,
            obs.array_latitude_rad,
            &crate::shapelets::SHAPELET_BASIS_VALUES,
            crate::shapelets::SBF_L,
            crate::shapelets::SBF_N,
            crate::shapelets::SBF_C,
            crate::shapelets::SBF_DX,
        );
        assert!(result.is_ok());
        result.unwrap()
    };

    // Ignore applying the beam.
    let mut visibilities = Array2::from_elem((obs.uvws.len(), obs.freqs.len()), Jones::default());
    unsafe {
        let result = modeller.model_timestep(visibilities.view_mut(), obs.lst, &obs.uvws);
        assert!(result.is_ok());
        result.unwrap();
    }
    assert_list_zenith_visibilities(visibilities.view());
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
            components: vec![get_simple_gaussian(pos, obs.flux_density_scale.clone())],
        },
    );
    let modeller = unsafe {
        let result = SkyModellerCuda::new(
            obs.beam.deref(),
            &srclist,
            &obs.freqs,
            &obs.xyzs,
            &[],
            obs.phase_centre,
            obs.array_latitude_rad,
            &crate::shapelets::SHAPELET_BASIS_VALUES,
            crate::shapelets::SBF_L,
            crate::shapelets::SBF_N,
            crate::shapelets::SBF_C,
            crate::shapelets::SBF_DX,
        );
        assert!(result.is_ok());
        result.unwrap()
    };

    // Ignore applying the beam.
    let mut visibilities = Array2::from_elem((obs.uvws.len(), obs.freqs.len()), Jones::default());
    unsafe {
        let result = modeller.model_timestep(visibilities.view_mut(), obs.lst, &obs.uvws);
        assert!(result.is_ok());
        result.unwrap();
    }
    assert_list_off_zenith_visibilities(visibilities.view());
}

// Put a single shapelet source at zenith.
#[test]
#[cfg(feature = "cuda")]
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
    // Get the component parameters via `SkyModellerCuda`.
    let modeller = unsafe {
        let result = SkyModellerCuda::new(
            obs.beam.deref(),
            &srclist,
            &obs.freqs,
            &obs.xyzs,
            &[],
            obs.phase_centre,
            obs.array_latitude_rad,
            &crate::shapelets::SHAPELET_BASIS_VALUES,
            crate::shapelets::SBF_L,
            crate::shapelets::SBF_N,
            crate::shapelets::SBF_C,
            crate::shapelets::SBF_DX,
        );
        assert!(result.is_ok());
        result.unwrap()
    };

    // Ignore applying the beam.
    let mut visibilities = Array2::from_elem((obs.uvws.len(), obs.freqs.len()), Jones::default());
    unsafe {
        let result = modeller.model_timestep(visibilities.view_mut(), obs.lst, &obs.uvws);
        assert!(result.is_ok());
        result.unwrap();
    }
    assert_list_zenith_visibilities(visibilities.view());
}

// Put a single shapelet source just off zenith.
#[test]
#[cfg(feature = "cuda")]
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
    // Get the component parameters via `SkyModellerCuda`.
    let modeller = unsafe {
        let result = SkyModellerCuda::new(
            obs.beam.deref(),
            &srclist,
            &obs.freqs,
            &obs.xyzs,
            &[],
            obs.phase_centre,
            obs.array_latitude_rad,
            &crate::shapelets::SHAPELET_BASIS_VALUES,
            crate::shapelets::SBF_L,
            crate::shapelets::SBF_N,
            crate::shapelets::SBF_C,
            crate::shapelets::SBF_DX,
        );
        assert!(result.is_ok());
        result.unwrap()
    };

    // Ignore applying the beam.
    let mut visibilities = Array2::from_elem((obs.uvws.len(), obs.freqs.len()), Jones::default());
    unsafe {
        let result = modeller.model_timestep(visibilities.view_mut(), obs.lst, &obs.uvws);
        assert!(result.is_ok());
        result.unwrap();
    }
    assert_list_off_zenith_visibilities(visibilities.view());
}
