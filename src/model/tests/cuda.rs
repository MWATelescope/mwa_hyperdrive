// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Tests on generating sky-model visibilities with CUDA.
//!
//! These tests use the same expected values as the CPU tests.

use ndarray::prelude::*;
use vec1::vec1;

use super::*;
use crate::{
    cuda::{self, CudaFloat, DevicePointer},
    srclist::Source,
};

/// Test with a bunch of parameters (are we using a beam?, what function are we
/// using for modelling, what source list, what test function).
macro_rules! test_modelling {
    ($no_beam:expr, $model_fn:expr,
        $list_srclist:expr, $power_law_srclist:expr, $curved_power_law_srclist:expr,
        $list_test_fn:expr, $power_law_test_fn:expr, $curved_power_law_test_fn:expr) => {{
        let obs = ObsParams::new($no_beam);
        let mut visibilities = Array2::zeros((obs.freqs.len(), obs.uvws.len()));
        let mut d_vis_fb = DevicePointer::copy_to_device(visibilities.as_slice().unwrap()).unwrap();
        let (modeller, d_uvws) = obs.get_gpu_modeller($list_srclist);
        // The device buffer will automatically be resized.
        let mut d_beam_jones = DevicePointer::default();
        unsafe {
            $model_fn(
                &modeller,
                obs.lst,
                obs.array_latitude_rad,
                &d_uvws,
                &mut d_beam_jones,
                &mut d_vis_fb,
            )
            .unwrap();
        }
        d_vis_fb
            .copy_from_device(visibilities.as_slice_mut().unwrap())
            .unwrap();
        #[cfg(not(feature = "cuda-single"))]
        let epsilon = if $no_beam { 0.0 } else { 1e-15 };
        #[cfg(feature = "cuda-single")]
        let epsilon = if $no_beam { 6e-8 } else { 2e-3 };
        $list_test_fn(visibilities.view(), epsilon);
        d_vis_fb.clear();

        let (modeller, d_uvws) = obs.get_gpu_modeller($power_law_srclist);
        unsafe {
            $model_fn(
                &modeller,
                obs.lst,
                obs.array_latitude_rad,
                &d_uvws,
                &mut d_beam_jones,
                &mut d_vis_fb,
            )
            .unwrap();
        }
        d_vis_fb
            .copy_from_device(visibilities.as_slice_mut().unwrap())
            .unwrap();
        $power_law_test_fn(visibilities.view(), epsilon);
        d_vis_fb.clear();

        let (modeller, d_uvws) = obs.get_gpu_modeller($curved_power_law_srclist);
        unsafe {
            $model_fn(
                &modeller,
                obs.lst,
                obs.array_latitude_rad,
                &d_uvws,
                &mut d_beam_jones,
                &mut d_vis_fb,
            )
            .unwrap();
        }
        d_vis_fb
            .copy_from_device(visibilities.as_slice_mut().unwrap())
            .unwrap();
        $curved_power_law_test_fn(visibilities.view(), epsilon);
    }};
}

#[test]
fn point_zenith_gpu() {
    test_modelling!(
        true,
        SkyModellerCuda::model_points,
        &POINT_ZENITH_LIST,
        &POINT_ZENITH_POWER_LAW,
        &POINT_ZENITH_CURVED_POWER_LAW,
        test_list_zenith_visibilities,
        test_power_law_zenith_visibilities,
        test_curved_power_law_zenith_visibilities
    );
}

#[test]
fn point_off_zenith_gpu() {
    test_modelling!(
        true,
        SkyModellerCuda::model_points,
        &POINT_OFF_ZENITH_LIST,
        &POINT_OFF_ZENITH_POWER_LAW,
        &POINT_OFF_ZENITH_CURVED_POWER_LAW,
        test_list_off_zenith_visibilities,
        test_power_law_off_zenith_visibilities,
        test_curved_power_law_off_zenith_visibilities
    );
}

#[test]
fn gaussian_zenith_gpu() {
    test_modelling!(
        true,
        SkyModellerCuda::model_gaussians,
        &GAUSSIAN_ZENITH_LIST,
        &GAUSSIAN_ZENITH_POWER_LAW,
        &GAUSSIAN_ZENITH_CURVED_POWER_LAW,
        test_list_zenith_visibilities,
        test_power_law_zenith_visibilities,
        test_curved_power_law_zenith_visibilities
    );
}

#[test]
fn gaussian_off_zenith_gpu() {
    test_modelling!(
        true,
        SkyModellerCuda::model_gaussians,
        &GAUSSIAN_OFF_ZENITH_LIST,
        &GAUSSIAN_OFF_ZENITH_POWER_LAW,
        &GAUSSIAN_OFF_ZENITH_CURVED_POWER_LAW,
        test_list_off_zenith_visibilities,
        test_power_law_off_zenith_visibilities,
        test_curved_power_law_off_zenith_visibilities
    );
}

#[test]
fn shapelet_zenith_gpu() {
    test_modelling!(
        true,
        SkyModellerCuda::model_shapelets,
        &SHAPELET_ZENITH_LIST,
        &SHAPELET_ZENITH_POWER_LAW,
        &SHAPELET_ZENITH_CURVED_POWER_LAW,
        test_list_zenith_visibilities,
        test_power_law_zenith_visibilities,
        test_curved_power_law_zenith_visibilities
    );
}

#[test]
fn shapelet_off_zenith_gpu() {
    test_modelling!(
        true,
        SkyModellerCuda::model_shapelets,
        &SHAPELET_OFF_ZENITH_LIST,
        &SHAPELET_OFF_ZENITH_POWER_LAW,
        &SHAPELET_OFF_ZENITH_CURVED_POWER_LAW,
        test_list_off_zenith_visibilities,
        test_power_law_off_zenith_visibilities,
        test_curved_power_law_off_zenith_visibilities
    );
}

#[test]
fn point_zenith_gpu_fee() {
    test_modelling!(
        false,
        SkyModellerCuda::model_points,
        &POINT_ZENITH_LIST,
        &POINT_ZENITH_POWER_LAW,
        &POINT_ZENITH_CURVED_POWER_LAW,
        test_list_zenith_visibilities_fee,
        test_power_law_zenith_visibilities_fee,
        test_curved_power_law_zenith_visibilities_fee
    );
}

#[test]
fn point_off_zenith_gpu_fee() {
    test_modelling!(
        false,
        SkyModellerCuda::model_points,
        &POINT_OFF_ZENITH_LIST,
        &POINT_OFF_ZENITH_POWER_LAW,
        &POINT_OFF_ZENITH_CURVED_POWER_LAW,
        test_list_off_zenith_visibilities_fee,
        test_power_law_off_zenith_visibilities_fee,
        test_curved_power_law_off_zenith_visibilities_fee
    );
}

#[test]
fn gaussian_zenith_gpu_fee() {
    test_modelling!(
        false,
        SkyModellerCuda::model_gaussians,
        &GAUSSIAN_ZENITH_LIST,
        &GAUSSIAN_ZENITH_POWER_LAW,
        &GAUSSIAN_ZENITH_CURVED_POWER_LAW,
        test_list_zenith_visibilities_fee,
        test_power_law_zenith_visibilities_fee,
        test_curved_power_law_zenith_visibilities_fee
    );
}

#[test]
fn gaussian_off_zenith_gpu_fee() {
    test_modelling!(
        false,
        SkyModellerCuda::model_gaussians,
        &GAUSSIAN_OFF_ZENITH_LIST,
        &GAUSSIAN_OFF_ZENITH_POWER_LAW,
        &GAUSSIAN_OFF_ZENITH_CURVED_POWER_LAW,
        test_list_off_zenith_visibilities_fee,
        test_power_law_off_zenith_visibilities_fee,
        test_curved_power_law_off_zenith_visibilities_fee
    );
}

#[test]
fn shapelet_zenith_gpu_fee() {
    test_modelling!(
        false,
        SkyModellerCuda::model_shapelets,
        &SHAPELET_ZENITH_LIST,
        &SHAPELET_ZENITH_POWER_LAW,
        &SHAPELET_ZENITH_CURVED_POWER_LAW,
        test_list_zenith_visibilities_fee,
        test_power_law_zenith_visibilities_fee,
        test_curved_power_law_zenith_visibilities_fee
    );
}

#[test]
fn shapelet_off_zenith_gpu_fee() {
    test_modelling!(
        false,
        SkyModellerCuda::model_shapelets,
        &SHAPELET_OFF_ZENITH_LIST,
        &SHAPELET_OFF_ZENITH_POWER_LAW,
        &SHAPELET_OFF_ZENITH_CURVED_POWER_LAW,
        test_list_off_zenith_visibilities_fee,
        test_power_law_off_zenith_visibilities_fee,
        test_curved_power_law_off_zenith_visibilities_fee
    );
}

#[test]
fn non_trivial_gaussian() {
    test_modelling!(
        false,
        SkyModellerCuda::model_gaussians,
        &SourceList::from([(
            "list".to_string(),
            Source {
                components: vec1![get_gaussian2(*OFF_PHASE_CENTRE, FluxType::List)]
            }
        )]),
        &SourceList::from([(
            "power_law".to_string(),
            Source {
                components: vec1![get_gaussian2(*OFF_PHASE_CENTRE, FluxType::PowerLaw)]
            }
        )]),
        &SourceList::from([(
            "curved_power_law".to_string(),
            Source {
                components: vec1![get_gaussian2(*OFF_PHASE_CENTRE, FluxType::CurvedPowerLaw)]
            }
        )]),
        test_non_trivial_gaussian_list,
        test_non_trivial_gaussian_power_law,
        test_non_trivial_gaussian_curved_power_law
    );
}

/// This test checks that beam responses are applied properly. The CUDA code
/// previously had a bug where the wrong beam response *might* have been applied
/// to the wrong component. Put multiple components with different flux types in
/// a source list and model it.
macro_rules! test_beam_applies_to_first_component {
    ($flux_type1:expr, $flux_type2:expr) => {{
        let obs = ObsParams::new(false);
        let mut srclist = SourceList::new();
        srclist.insert(
            "mixed".to_string(),
            Source {
                components: vec1![
                    get_point(obs.phase_centre, $flux_type1),
                    get_point(RADec::new_degrees(45.0, 18.0), $flux_type2)
                ],
            },
        );
        let mut visibilities = Array2::zeros((obs.freqs.len(), obs.uvws.len()));
        let mut d_vis_fb = DevicePointer::copy_to_device(visibilities.as_slice().unwrap()).unwrap();
        let (modeller, d_uvws) = obs.get_gpu_modeller(&srclist);
        // The device buffer will automatically be resized.
        let mut d_beam_jones = DevicePointer::default();
        unsafe {
            modeller
                .model_points(
                    obs.lst,
                    obs.array_latitude_rad,
                    &d_uvws,
                    &mut d_beam_jones,
                    &mut d_vis_fb,
                )
                .unwrap();
        }
        d_vis_fb
            .copy_from_device(visibilities.as_slice_mut().unwrap())
            .unwrap();
        d_vis_fb.clear();

        // The visibilities should be very similar to having only the zenith
        // component, because the other component is far from the pointing
        // centre. The expected values are taken from
        // `assert_power_law_zenith_visibilities_fee`.
        assert_abs_diff_eq!(
            visibilities[(0, 0)],
            Jones::from([
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
                    get_gaussian(obs.phase_centre, $flux_type1),
                    get_gaussian(RADec::new_degrees(45.0, 18.0), $flux_type2)
                ],
            },
        );
        let (modeller, d_uvws) = obs.get_gpu_modeller(&srclist);
        unsafe {
            modeller
                .model_gaussians(
                    obs.lst,
                    obs.array_latitude_rad,
                    &d_uvws,
                    &mut d_beam_jones,
                    &mut d_vis_fb,
                )
                .unwrap();
        }
        d_vis_fb
            .copy_from_device(visibilities.as_slice_mut().unwrap())
            .unwrap();
        d_vis_fb.clear();

        assert_abs_diff_eq!(
            visibilities[(0, 0)],
            Jones::from([
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
                    get_shapelet(obs.phase_centre, $flux_type1),
                    get_shapelet(RADec::new_degrees(45.0, 18.0), $flux_type2)
                ],
            },
        );
        let (modeller, d_uvws) = obs.get_gpu_modeller(&srclist);
        unsafe {
            modeller
                .model_shapelets(
                    obs.lst,
                    obs.array_latitude_rad,
                    &d_uvws,
                    &mut d_beam_jones,
                    &mut d_vis_fb,
                )
                .unwrap();
        }
        d_vis_fb
            .copy_from_device(visibilities.as_slice_mut().unwrap())
            .unwrap();
        d_vis_fb.clear();

        assert_abs_diff_eq!(
            visibilities[(0, 0)],
            Jones::from([
                Complex::new(9.995525e-1, 0.0),
                Complex::new(-5.405832e-4, -5.00542e-6),
                Complex::new(-5.405832e-4, 5.00542e-6),
                Complex::new(9.9958146e-1, 0.0)
            ]),
            epsilon = 8e-4
        );
    }};
}

#[test]
fn beam_responses_apply_properly_power_law2() {
    test_beam_applies_to_first_component!(FluxType::PowerLaw, FluxType::PowerLaw);
}

#[test]
fn beam_responses_apply_properly_power_law_and_curved_power_law() {
    test_beam_applies_to_first_component!(FluxType::PowerLaw, FluxType::CurvedPowerLaw);
}

#[test]
fn beam_responses_apply_properly_power_law_and_list() {
    test_beam_applies_to_first_component!(FluxType::PowerLaw, FluxType::List);
}

#[test]
fn beam_responses_apply_properly_curved_power_law_and_power_law() {
    test_beam_applies_to_first_component!(FluxType::CurvedPowerLaw, FluxType::PowerLaw);
}

#[test]
fn beam_responses_apply_properly_curved_power_law2() {
    test_beam_applies_to_first_component!(FluxType::CurvedPowerLaw, FluxType::CurvedPowerLaw);
}

#[test]
fn beam_responses_apply_properly_curved_power_law_and_list() {
    test_beam_applies_to_first_component!(FluxType::CurvedPowerLaw, FluxType::List);
}

#[test]
fn beam_responses_apply_properly_list_and_power_law() {
    test_beam_applies_to_first_component!(FluxType::List, FluxType::PowerLaw);
}

#[test]
fn beam_responses_apply_properly_list_and_curved_power_law() {
    test_beam_applies_to_first_component!(FluxType::List, FluxType::CurvedPowerLaw);
}

#[test]
fn beam_responses_apply_properly_list2() {
    test_beam_applies_to_first_component!(FluxType::List, FluxType::List);
}

#[test]
fn gaussian_multiple_components() {
    let obs = ObsParams::new(true);
    let mut srclist = SourceList::new();
    srclist.insert(
        "gaussians".to_string(),
        Source {
            components: vec1![
                get_gaussian(RADec::new_degrees(1.0, -27.0), FluxType::List),
                get_gaussian(RADec::new_degrees(1.1, -27.0), FluxType::List)
            ],
        },
    );
    let (modeller, d_uvws) = obs.get_gpu_modeller(&srclist);

    let mut visibilities = Array2::zeros((obs.freqs.len(), obs.uvws.len()));
    let mut d_vis_fb = DevicePointer::copy_to_device(visibilities.as_slice().unwrap()).unwrap();
    // The device buffer will automatically be resized.
    let mut d_beam_jones = DevicePointer::default();
    unsafe {
        modeller
            .model_gaussians(
                obs.lst,
                obs.array_latitude_rad,
                &d_uvws,
                &mut d_beam_jones,
                &mut d_vis_fb,
            )
            .unwrap();
    };
    d_vis_fb
        .copy_from_device(visibilities.as_slice_mut().unwrap())
        .unwrap();

    #[cfg(not(feature = "cuda-single"))]
    test_multiple_gaussian_components(visibilities.view(), 0.0);
    #[cfg(feature = "cuda-single")]
    test_multiple_gaussian_components(visibilities.view(), 5e-7);
}

#[test]
fn shapelet_multiple_components() {
    let obs = ObsParams::new(true);
    let mut srclist = SourceList::new();
    srclist.insert(
        "shapelets".to_string(),
        Source {
            components: vec1![
                get_shapelet(RADec::new_degrees(1.0, -27.0), FluxType::List),
                get_shapelet(RADec::new_degrees(1.1, -27.0), FluxType::List)
            ],
        },
    );
    let (modeller, d_uvws) = obs.get_gpu_modeller(&srclist);

    let mut visibilities = Array2::zeros((obs.freqs.len(), obs.uvws.len()));
    let mut d_vis_fb = DevicePointer::copy_to_device(visibilities.as_slice().unwrap()).unwrap();
    // The device buffer will automatically be resized.
    let mut d_beam_jones = DevicePointer::default();
    unsafe {
        modeller
            .model_shapelets(
                obs.lst,
                obs.array_latitude_rad,
                &d_uvws,
                &mut d_beam_jones,
                &mut d_vis_fb,
            )
            .unwrap();
    };
    d_vis_fb
        .copy_from_device(visibilities.as_slice_mut().unwrap())
        .unwrap();

    // Compare the shapelet UVs, but convert to UVWs so we can test an entire
    // array of values.
    let shapelet_uvs = modeller
        .get_shapelet_uvs(obs.lst)
        .list
        .map(|&cuda::ShapeletUV { u, v }| UVW {
            u: CudaFloat::into(u),
            v: CudaFloat::into(v),
            w: 0.0,
        });

    #[cfg(not(feature = "cuda-single"))]
    test_multiple_shapelet_components(visibilities.view(), shapelet_uvs.view(), 0.0, 0.0);
    #[cfg(feature = "cuda-single")]
    test_multiple_shapelet_components(visibilities.view(), shapelet_uvs.view(), 5e-7, 3e-8);
}

#[test]
/// With Jack's help, we found that the old behaviour was incorrect. All of the
/// above tests don't expose the incorrect behaviour because (currently) they
/// use a curved-power-law source with a reference freq of 150 MHz, which
/// matches the target freq. Using a different ref freq tests the conversion
/// logic.
fn test_curved_power_law_changing_ref_freq() {
    let mut srclist = POINT_ZENITH_CURVED_POWER_LAW.clone();
    {
        let src = srclist.get_mut("zenith").expect("exists");
        assert_eq!(src.components.len(), 1);
        src.components
            .iter_mut()
            .for_each(|comp| match &mut comp.flux_type {
                FluxDensityType::CurvedPowerLaw { fd, si, .. } => {
                    // Set a different reference frequency.
                    fd.freq = 200e6;
                    assert_abs_diff_eq!(*si, -0.8);
                }
                _ => unreachable!(),
            });
    }

    // Now that we have a different ref freq, test the values inside of a new
    // sky modeller.
    let mut obs = ObsParams::new(true);
    obs.freqs.clear();
    obs.freqs.push(150e6);
    let (modeller, _) = obs.get_gpu_modeller(&srclist);
    let mut modeller_fds = [crate::cuda::CudaJones::default(); 1];
    let mut modeller_sis = [0.0; 1];
    modeller
        .point_curved_power_law_fds
        .copy_from_device(&mut modeller_fds)
        .unwrap();
    modeller
        .point_curved_power_law_sis
        .copy_from_device(&mut modeller_sis)
        .unwrap();

    assert_abs_diff_eq!(modeller_fds[0].j00_re, 1.261912575563708);
    // The SI has changed from -0.8, which was what we started with.
    assert_abs_diff_eq!(modeller_sis[0], -0.8172609243471072);
}
