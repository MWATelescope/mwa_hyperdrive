// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Tests on generating sky-model visibilities.
//!
//! There are no actual test functions in this module; the functions test input
//! visibilities against expected values. The `track_caller` annotation is
//! useful here because it allows the responsible caller to be highlighted if
//! there's an error, rather than an assert in one of these functions being
//! highlighted.

mod cpu;
#[cfg(any(feature = "cuda", feature = "hip"))]
mod gpu;

use approx::assert_abs_diff_eq;
use marlu::{
    constants::{MWA_LAT_RAD, MWA_LONG_RAD},
    pos::xyz::xyzs_to_cross_uvws,
    Jones, RADec, XyzGeodetic,
};
use ndarray::prelude::*;
use num_complex::Complex;
use vec1::vec1;

use super::*;
#[cfg(any(feature = "cuda", feature = "hip"))]
use crate::gpu::DevicePointer;
#[cfg(any(feature = "cuda", feature = "hip"))]
use crate::model::gpu::SkyModellerGpu;
use crate::{
    beam::{create_beam_object, Delays},
    srclist::{
        ComponentType, FluxDensity, FluxDensityType, ShapeletCoeff, Source, SourceComponent,
        SourceList,
    },
};

lazy_static::lazy_static! {
    static ref PHASE_CENTRE: RADec = RADec::from_degrees(0.0, -27.0);
    static ref OFF_PHASE_CENTRE: RADec = RADec::from_degrees(1.0, -27.0);

    static ref POINT_ZENITH_LIST: SourceList = SourceList::from([
        ("zenith".to_string(),
        Source {components: vec![get_point(*PHASE_CENTRE, FluxType::List)].into_boxed_slice()})
    ]);
    static ref POINT_ZENITH_POWER_LAW: SourceList = SourceList::from([
        ("zenith".to_string(),
        Source {components: vec![get_point(*PHASE_CENTRE, FluxType::PowerLaw)].into_boxed_slice()})
    ]);
    static ref POINT_ZENITH_CURVED_POWER_LAW: SourceList = SourceList::from([
        ("zenith".to_string(),
        Source {components: vec![get_point(*PHASE_CENTRE, FluxType::CurvedPowerLaw)].into_boxed_slice()})
    ]);

    static ref POINT_OFF_ZENITH_LIST: SourceList = SourceList::from([
        ("off_zenith".to_string(),
        Source {components: vec![get_point(*OFF_PHASE_CENTRE, FluxType::List)].into_boxed_slice()})
    ]);
    static ref POINT_OFF_ZENITH_POWER_LAW: SourceList = SourceList::from([
        ("off_zenith".to_string(),
        Source {components: vec![get_point(*OFF_PHASE_CENTRE, FluxType::PowerLaw)].into_boxed_slice()})
    ]);
    static ref POINT_OFF_ZENITH_CURVED_POWER_LAW: SourceList = SourceList::from([
        ("off_zenith".to_string(),
        Source {components: vec![get_point(*OFF_PHASE_CENTRE, FluxType::CurvedPowerLaw)].into_boxed_slice()})
    ]);

    static ref GAUSSIAN_ZENITH_LIST: SourceList = SourceList::from([
        ("zenith".to_string(),
        Source {components: vec![get_gaussian(*PHASE_CENTRE, FluxType::List)].into_boxed_slice()})
    ]);
    static ref GAUSSIAN_ZENITH_POWER_LAW: SourceList = SourceList::from([
        ("zenith".to_string(),
        Source {components: vec![get_gaussian(*PHASE_CENTRE, FluxType::PowerLaw)].into_boxed_slice()})
    ]);
    static ref GAUSSIAN_ZENITH_CURVED_POWER_LAW: SourceList = SourceList::from([
        ("zenith".to_string(),
        Source {components: vec![get_gaussian(*PHASE_CENTRE, FluxType::CurvedPowerLaw)].into_boxed_slice()})
    ]);

    static ref GAUSSIAN_OFF_ZENITH_LIST: SourceList = SourceList::from([
        ("off_zenith".to_string(),
        Source {components: vec![get_gaussian(*OFF_PHASE_CENTRE, FluxType::List)].into_boxed_slice()})
    ]);
    static ref GAUSSIAN_OFF_ZENITH_POWER_LAW: SourceList = SourceList::from([
        ("off_zenith".to_string(),
        Source {components: vec![get_gaussian(*OFF_PHASE_CENTRE, FluxType::PowerLaw)].into_boxed_slice()})
    ]);
    static ref GAUSSIAN_OFF_ZENITH_CURVED_POWER_LAW: SourceList = SourceList::from([
        ("off_zenith".to_string(),
        Source {components: vec![get_gaussian(*OFF_PHASE_CENTRE, FluxType::CurvedPowerLaw)].into_boxed_slice()})
    ]);

    static ref SHAPELET_ZENITH_LIST: SourceList = SourceList::from([
        ("zenith".to_string(),
        Source {components: vec![get_shapelet(*PHASE_CENTRE, FluxType::List)].into_boxed_slice()})
    ]);
    static ref SHAPELET_ZENITH_POWER_LAW: SourceList = SourceList::from([
        ("zenith".to_string(),
        Source {components: vec![get_shapelet(*PHASE_CENTRE, FluxType::PowerLaw)].into_boxed_slice()})
    ]);
    static ref SHAPELET_ZENITH_CURVED_POWER_LAW: SourceList = SourceList::from([
        ("zenith".to_string(),
        Source {components: vec![get_shapelet(*PHASE_CENTRE, FluxType::CurvedPowerLaw)].into_boxed_slice()})
    ]);


    static ref SHAPELET_OFF_ZENITH_LIST: SourceList = SourceList::from([
        ("off_zenith".to_string(),
        Source {components: vec![get_shapelet(*OFF_PHASE_CENTRE, FluxType::List)].into_boxed_slice()})
    ]);
    static ref SHAPELET_OFF_ZENITH_POWER_LAW: SourceList = SourceList::from([
        ("off_zenith".to_string(),
        Source {components: vec![get_shapelet(*OFF_PHASE_CENTRE, FluxType::PowerLaw)].into_boxed_slice()})
    ]);
    static ref SHAPELET_OFF_ZENITH_CURVED_POWER_LAW: SourceList = SourceList::from([
        ("off_zenith".to_string(),
        Source {components: vec![get_shapelet(*OFF_PHASE_CENTRE, FluxType::CurvedPowerLaw)].into_boxed_slice()})
    ]);

}

fn get_list() -> FluxDensityType {
    FluxDensityType::List(vec1![
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
    ])
}

fn get_power_law() -> FluxDensityType {
    FluxDensityType::PowerLaw {
        si: -0.8,
        fd: FluxDensity {
            freq: 150e6,
            i: 1.0,
            ..Default::default()
        },
    }
}

fn get_curved_power_law() -> FluxDensityType {
    FluxDensityType::CurvedPowerLaw {
        si: -0.8,
        fd: FluxDensity {
            freq: 150e6,
            i: 1.0,
            ..Default::default()
        },
        q: 0.03,
    }
}

#[derive(Clone, Copy)]
enum FluxType {
    List,
    PowerLaw,
    CurvedPowerLaw,
}

fn get_point(pos: RADec, flux_type: FluxType) -> SourceComponent {
    SourceComponent {
        radec: pos,
        comp_type: ComponentType::Point,
        flux_type: match flux_type {
            FluxType::List => get_list(),
            FluxType::PowerLaw => get_power_law(),
            FluxType::CurvedPowerLaw => get_curved_power_law(),
        },
    }
}

fn get_gaussian(pos: RADec, flux_type: FluxType) -> SourceComponent {
    SourceComponent {
        radec: pos,
        comp_type: ComponentType::Gaussian {
            maj: 0.0,
            min: 0.0,
            pa: 0.0,
        },
        flux_type: match flux_type {
            FluxType::List => get_list(),
            FluxType::PowerLaw => get_power_law(),
            FluxType::CurvedPowerLaw => get_curved_power_law(),
        },
    }
}

fn get_gaussian2(pos: RADec, flux_type: FluxType) -> SourceComponent {
    SourceComponent {
        radec: pos,
        comp_type: ComponentType::Gaussian {
            maj: 1.0,
            min: 0.5,
            pa: 0.25,
        },
        flux_type: match flux_type {
            FluxType::List => get_list(),
            FluxType::PowerLaw => get_power_law(),
            FluxType::CurvedPowerLaw => get_curved_power_law(),
        },
    }
}

fn get_shapelet(pos: RADec, flux_type: FluxType) -> SourceComponent {
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
            }]
            .into_boxed_slice(),
        },
        flux_type: match flux_type {
            FluxType::List => get_list(),
            FluxType::PowerLaw => get_power_law(),
            FluxType::CurvedPowerLaw => get_curved_power_law(),
        },
    }
}

struct ObsParams {
    phase_centre: RADec,
    freqs: Vec<f64>,
    lst: f64,
    xyzs: Vec<XyzGeodetic>,
    uvws: Vec<UVW>,
    /// Per-tile UVWs, for the factorised CPU kernels.
    tile_uvws: Vec<UVW>,
    beam: Box<dyn Beam>,
    flagged_tiles: HashSet<usize>,
    array_longitude_rad: f64,
    array_latitude_rad: f64,
}

impl ObsParams {
    fn new(no_beam: bool) -> ObsParams {
        let phase_centre = RADec::from_degrees(0.0, -27.0);
        let freqs = vec![150e6, 175e6, 200e6];

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
        let uvws = xyzs_to_cross_uvws(&xyzs, phase_centre.to_hadec(lst));
        let tile_uvws = xyzs
            .iter()
            .map(|xyz| UVW::from_xyz(*xyz, phase_centre.to_hadec(lst)))
            .collect();
        let beam = create_beam_object(
            Some(if no_beam { "none" } else { "fee" }),
            xyzs.len(),
            Delays::Partial(vec![0; 16]),
        )
        .unwrap();
        let flagged_tiles = HashSet::new();
        let array_longitude_rad = MWA_LONG_RAD;
        let array_latitude_rad = MWA_LAT_RAD;

        ObsParams {
            phase_centre,
            freqs,
            lst,
            xyzs,
            uvws,
            tile_uvws,
            beam,
            flagged_tiles,
            array_longitude_rad,
            array_latitude_rad,
        }
    }

    /// The same as [`ObsParams::new`], but with a caller-supplied beam. The
    /// beam must be set up for [`ObsParams::xyzs`]'s number of tiles.
    #[cfg(any(feature = "cuda", feature = "hip"))]
    fn new_with_beam(beam: Box<dyn Beam>) -> ObsParams {
        let mut obs = ObsParams::new(true);
        assert_eq!(beam.get_num_tiles(), obs.xyzs.len());
        obs.beam = beam;
        obs
    }

    fn get_cpu_modeller(&self, srclist: &SourceList) -> SkyModellerCpu<'_> {
        SkyModellerCpu::new(
            &*self.beam,
            srclist,
            Polarisations::default(),
            &self.xyzs,
            &self.freqs,
            &self.flagged_tiles,
            self.phase_centre,
            self.array_longitude_rad,
            self.array_latitude_rad,
            Duration::default(),
            true,
        )
    }

    #[cfg(any(feature = "cuda", feature = "hip"))]
    #[track_caller]
    fn get_gpu_modeller(
        &self,
        srclist: &SourceList,
    ) -> (SkyModellerGpu<'_>, DevicePointer<crate::gpu::UVW>) {
        let m = SkyModellerGpu::new(
            &*self.beam,
            srclist,
            Polarisations::default(),
            &self.xyzs,
            &self.freqs,
            &self.flagged_tiles,
            self.phase_centre,
            self.array_longitude_rad,
            self.array_latitude_rad,
            Duration::default(),
            true,
        )
        .unwrap();
        let gpu_uvws = self
            .uvws
            .iter()
            .map(|&uvw| crate::gpu::UVW {
                u: uvw.u as crate::gpu::GpuFloat,
                v: uvw.v as crate::gpu::GpuFloat,
                w: uvw.w as crate::gpu::GpuFloat,
            })
            .collect::<Vec<_>>();
        let d_uvws = DevicePointer::copy_to_device(&gpu_uvws).unwrap();
        (m, d_uvws)
    }
}

#[track_caller]
fn test_list_zenith_visibilities(vis: ArrayView2<Jones<f32>>, epsilon: f32) {
    // All LMN values are (0, 0, 1). This means that the Fourier transform to
    // make a visibility from LMN and UVW will always just be the input flux
    // density.

    // First row: flux density at first frequency (i.e. 1 Jy XX and YY).
    let expected_1st_row = Array1::from_elem(vis.dim().0, Jones::identity());
    let result = vis.slice(s![0, ..]);
    assert_abs_diff_eq!(expected_1st_row, result, epsilon = epsilon);
    // 3 Jy XX and YY.
    let expected_2nd_row = Array1::from_elem(vis.dim().0, Jones::identity() * 3.0);
    let result = vis.slice(s![1, ..]);
    assert_abs_diff_eq!(expected_2nd_row, result, epsilon = epsilon);
    // 2 Jy XX and YY.
    let expected_3rd_row = Array1::from_elem(vis.dim().0, Jones::identity() * 2.0);
    let result = vis.slice(s![2, ..]);
    assert_abs_diff_eq!(expected_3rd_row, result, epsilon = epsilon);
}

#[track_caller]
fn test_list_off_zenith_visibilities(vis: ArrayView2<Jones<f32>>, epsilon: f32) {
    // This time, all LMN values should be close to, but not the same as, (0, 0,
    // 1). This means that the visibilities should be somewhat close to the
    // input flux densities.

    let expected_1st_row = array![
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
    ];
    let result = vis.slice(s![0, ..]);
    assert_abs_diff_eq!(expected_1st_row, result, epsilon = epsilon);

    let expected_2nd_row = array![
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
    ];
    let result = vis.slice(s![1, ..]);
    assert_abs_diff_eq!(expected_2nd_row, result, epsilon = epsilon);

    let expected_3rd_row = array![
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
    ];
    let result = vis.slice(s![2, ..]);
    assert_abs_diff_eq!(expected_3rd_row, result, epsilon = epsilon);
}

#[track_caller]
fn test_list_zenith_visibilities_fee(vis: ArrayView2<Jones<f32>>, epsilon: f32) {
    // All LMN values are (0, 0, 1). This means that the Fourier transform to
    // make a visibility from LMN and UVW will always just be the input flux
    // density *multiplied by the beam response*.

    // First row: flux density at first frequency (i.e. ~1 Jy XX and YY).
    let expected_1st_row = array![
        Jones::from([
            Complex::new(9.9958014e-1, 0.0),
            Complex::new(-5.4407195e-4, 4.5348115e-6),
            Complex::new(-5.4407195e-4, -4.5348115e-6),
            Complex::new(9.995511e-1, 0.0),
        ]),
        Jones::from([
            Complex::new(9.9958014e-1, 0.0),
            Complex::new(-5.4407195e-4, 4.5348115e-6),
            Complex::new(-5.4407195e-4, -4.5348115e-6),
            Complex::new(9.995511e-1, 0.0),
        ]),
        Jones::from([
            Complex::new(9.9958014e-1, 0.0),
            Complex::new(-5.4407195e-4, 4.5348115e-6),
            Complex::new(-5.4407195e-4, -4.5348115e-6),
            Complex::new(9.995511e-1, 0.0),
        ]),
    ];
    let result = vis.slice(s![0, ..]);
    assert_abs_diff_eq!(expected_1st_row, result, epsilon = epsilon);
    // ~3 Jy XX and YY.
    let expected_2nd_row = array![
        Jones::from([
            Complex::new(2.9983048e0, 1.110223e-16),
            Complex::new(-1.4281868e-3, -5.550485e-6),
            Complex::new(-1.4281868e-3, 5.550485e-6),
            Complex::new(2.998183e0, 0.0),
        ]),
        Jones::from([
            Complex::new(2.9983048e0, 1.110223e-16),
            Complex::new(-1.4281868e-3, -5.550485e-6),
            Complex::new(-1.4281868e-3, 5.550485e-6),
            Complex::new(2.998183e0, 0.0),
        ]),
        Jones::from([
            Complex::new(2.9983048e0, 1.110223e-16),
            Complex::new(-1.4281868e-3, -5.550485e-6),
            Complex::new(-1.4281868e-3, 5.550485e-6),
            Complex::new(2.998183e0, 0.0),
        ]),
    ];
    let result = vis.slice(s![1, ..]);
    assert_abs_diff_eq!(expected_2nd_row, result, epsilon = epsilon);
    // ~2 Jy XX and YY.
    let expected_3rd_row = array![
        Jones::from([
            Complex::new(1.9986477e0, 0.0),
            Complex::new(-6.9518393e-4, 6.8433046e-6),
            Complex::new(-6.9518393e-4, -6.8433046e-6),
            Complex::new(1.9984435e0, 0.0),
        ]),
        Jones::from([
            Complex::new(1.9986477e0, 0.0),
            Complex::new(-6.9518393e-4, 6.8433046e-6),
            Complex::new(-6.9518393e-4, -6.8433046e-6),
            Complex::new(1.9984435e0, 0.0),
        ]),
        Jones::from([
            Complex::new(1.9986477e0, 0.0),
            Complex::new(-6.9518393e-4, 6.8433046e-6),
            Complex::new(-6.9518393e-4, -6.8433046e-6),
            Complex::new(1.9984435e0, 0.0),
        ]),
    ];
    let result = vis.slice(s![2, ..]);
    assert_abs_diff_eq!(expected_3rd_row, result, epsilon = epsilon);
}

#[track_caller]
fn test_list_off_zenith_visibilities_fee(vis: ArrayView2<Jones<f32>>, epsilon: f32) {
    // This time, all LMN values should be close to, but not the same as, (0, 0,
    // 1). This means that the visibilities should be somewhat close to the
    // input flux densities.

    let expected_1st_row = array![
        Jones::from([
            Complex::new(9.907028e-1, 9.717331e-2),
            Complex::new(-4.7281972e-4, -4.143827e-5),
            Complex::new(-4.718602e-4, -5.122087e-5),
            Complex::new(9.90933e-1, 9.7195886e-2),
        ]),
        Jones::from([
            Complex::new(9.942469e-1, 4.9068943e-2),
            Complex::new(-4.7427195e-4, -1.8485927e-5),
            Complex::new(-4.7378743e-4, -2.8303528e-5),
            Complex::new(9.944779e-1, 4.9080346e-2),
        ]),
        Jones::from([
            Complex::new(9.9428844e-1, -4.8220586e-2),
            Complex::new(-4.738114e-4, 2.789926e-5),
            Complex::new(-4.7428755e-4, 1.808125e-5),
            Complex::new(9.945195e-1, -4.823179e-2),
        ]),
    ];
    let result = vis.slice(s![0, ..]);
    assert_abs_diff_eq!(expected_1st_row, result, epsilon = epsilon);

    let expected_2nd_row = array![
        Jones::from([
            Complex::new(2.9622474e0, 3.393705e-1),
            Complex::new(-1.2511797e-3, -1.4764839e-4),
            Complex::new(-1.2521538e-3, -1.3914655e-4),
            Complex::new(2.963214e0, 3.3948123e-1),
        ]),
        Jones::from([
            Complex::new(2.976691e0, 1.7144321e-1),
            Complex::new(-1.2575238e-3, -7.671319e-5),
            Complex::new(-1.2580157e-3, -6.8169895e-5),
            Complex::new(2.9776623e0, 1.7149915e-1),
        ]),
        Jones::from([
            Complex::new(2.97686e0, -1.6847996e-1),
            Complex::new(-1.258083e-3, 6.691756e-5),
            Complex::new(-1.2575994e-3, 7.546134e-5),
            Complex::new(2.9778316e0, -1.6853493e-1),
        ]),
    ];
    let result = vis.slice(s![1, ..]);
    assert_abs_diff_eq!(expected_2nd_row, result, epsilon = epsilon);

    let expected_3rd_row = array![
        Jones::from([
            Complex::new(1.9675336e0, 2.5795713e-1),
            Complex::new(-5.953736e-4, -7.018217e-5),
            Complex::new(-5.9334346e-4, -8.566678e-5),
            Complex::new(1.9691538e0, 2.5816956e-1),
        ]),
        Jones::from([
            Complex::new(1.9800837e0, 1.303794e-1),
            Complex::new(-5.9866277e-4, -3.1593714e-5),
            Complex::new(-5.976367e-4, -4.7177098e-5),
            Complex::new(1.9817142e0, 1.3048676e-1),
        ]),
        Jones::from([
            Complex::new(1.9802307e0, -1.2812665e-1),
            Complex::new(-5.9768994e-4, 4.649716e-5),
            Complex::new(-5.9869833e-4, 3.0912615e-5),
            Complex::new(1.9818615e0, -1.2823215e-1),
        ]),
    ];
    let result = vis.slice(s![2, ..]);
    assert_abs_diff_eq!(expected_3rd_row, result, epsilon = epsilon);
}

#[track_caller]
fn test_power_law_zenith_visibilities(vis: ArrayView2<Jones<f32>>, epsilon: f32) {
    // All LMN values are (0, 0, 1). This means that the Fourier transform to
    // make a visibility from LMN and UVW will always just be the input flux
    // density.

    // First row: flux density at first frequency (i.e. 1 Jy XX and YY).
    let expected_1st_row = Array1::from_elem(vis.dim().0, Jones::identity());
    let result = vis.slice(s![0, ..]);
    assert_abs_diff_eq!(expected_1st_row, result, epsilon = epsilon);

    let expected_2nd_row = array![
        Jones::from([
            Complex::new(8.839803e-1, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(8.839803e-1, 0.0)
        ]),
        Jones::from([
            Complex::new(8.839803e-1, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(8.839803e-1, 0.0)
        ]),
        Jones::from([
            Complex::new(8.839803e-1, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(8.839803e-1, 0.0)
        ]),
    ];
    let result = vis.slice(s![1, ..]);
    assert_abs_diff_eq!(expected_2nd_row, result, epsilon = epsilon);

    let expected_3rd_row = array![
        Jones::from([
            Complex::new(7.9441786e-1, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(7.9441786e-1, 0.0)
        ]),
        Jones::from([
            Complex::new(7.9441786e-1, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(7.9441786e-1, 0.0)
        ]),
        Jones::from([
            Complex::new(7.9441786e-1, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(7.9441786e-1, 0.0)
        ]),
    ];
    let result = vis.slice(s![2, ..]);
    assert_abs_diff_eq!(expected_3rd_row, result, epsilon = epsilon);
}

#[track_caller]
fn test_power_law_off_zenith_visibilities(vis: ArrayView2<Jones<f32>>, epsilon: f32) {
    // This time, all LMN values should be close to, but not the same as, (0, 0,
    // 1). This means that the visibilities should be somewhat close to the
    // input flux densities.

    let expected_1st_row = array![
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
    ];
    let result = vis.slice(s![0, ..]);
    assert_abs_diff_eq!(expected_1st_row, result, epsilon = epsilon);

    let expected_2nd_row = array![
        Jones::from([
            Complex::new(8.782355e-1, 1.0061524e-1),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(8.782355e-1, 1.0061524e-1),
        ]),
        Jones::from([
            Complex::new(8.8251776e-1, 5.0828815e-2),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(8.8251776e-1, 5.0828815e-2),
        ]),
        Jones::from([
            Complex::new(8.825679e-1, -4.995028e-2),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(8.825679e-1, -4.995028e-2),
        ]),
    ];
    let result = vis.slice(s![1, ..]);
    assert_abs_diff_eq!(expected_2nd_row, result, epsilon = epsilon);

    let expected_3rd_row = array![
        Jones::from([
            Complex::new(7.8767705e-1, 1.0326985e-1),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(7.8767705e-1, 1.0326985e-1),
        ]),
        Jones::from([
            Complex::new(7.927013e-1, 5.219573e-2),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(7.927013e-1, 5.219573e-2),
        ]),
        Jones::from([
            Complex::new(7.927602e-1, -5.1293872e-2),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(7.927602e-1, -5.1293872e-2),
        ]),
    ];
    let result = vis.slice(s![2, ..]);
    assert_abs_diff_eq!(expected_3rd_row, result, epsilon = epsilon);
}

#[track_caller]
fn test_power_law_zenith_visibilities_fee(vis: ArrayView2<Jones<f32>>, epsilon: f32) {
    // All LMN values are (0, 0, 1). This means that the Fourier transform to
    // make a visibility from LMN and UVW will always just be the input flux
    // density *multiplied by the beam response*.

    // First row: flux density at first frequency (i.e. ~1 Jy XX and YY).
    let expected_1st_row = array![
        Jones::from([
            Complex::new(9.9958014e-1, 0.0),
            Complex::new(-5.4407195e-4, 4.5348115e-6),
            Complex::new(-5.4407195e-4, -4.5348115e-6),
            Complex::new(9.995511e-1, 0.0),
        ]),
        Jones::from([
            Complex::new(9.9958014e-1, 0.0),
            Complex::new(-5.4407195e-4, 4.5348115e-6),
            Complex::new(-5.4407195e-4, -4.5348115e-6),
            Complex::new(9.995511e-1, 0.0),
        ]),
        Jones::from([
            Complex::new(9.9958014e-1, 0.0),
            Complex::new(-5.4407195e-4, 4.5348115e-6),
            Complex::new(-5.4407195e-4, -4.5348115e-6),
            Complex::new(9.995511e-1, 0.0),
        ]),
    ];
    let result = vis.slice(s![0, ..]);
    assert_abs_diff_eq!(expected_1st_row, result, epsilon = epsilon);

    let expected_2nd_row = array![
        Jones::from([
            Complex::new(8.834808e-1, 0.0),
            Complex::new(-4.2082966e-4, -1.6355065e-6),
            Complex::new(-4.2082966e-4, 1.6355065e-6),
            Complex::new(8.8344485e-1, 0.0),
        ]),
        Jones::from([
            Complex::new(8.834808e-1, 0.0),
            Complex::new(-4.2082966e-4, -1.6355065e-6),
            Complex::new(-4.2082966e-4, 1.6355065e-6),
            Complex::new(8.8344485e-1, 0.0),
        ]),
        Jones::from([
            Complex::new(8.834808e-1, 0.0),
            Complex::new(-4.2082966e-4, -1.6355065e-6),
            Complex::new(-4.2082966e-4, 1.6355065e-6),
            Complex::new(8.8344485e-1, 0.0),
        ]),
    ];
    let result = vis.slice(s![1, ..]);
    assert_abs_diff_eq!(expected_2nd_row, result, epsilon = epsilon);

    let expected_3rd_row = array![
        Jones::from([
            Complex::new(7.938807e-1, 3.3087225e-24),
            Complex::new(-2.7613327e-4, 2.7182218e-6),
            Complex::new(-2.7613327e-4, -2.7182218e-6),
            Complex::new(7.9379964e-1, 3.3087225e-24),
        ]),
        Jones::from([
            Complex::new(7.938807e-1, 3.3087225e-24),
            Complex::new(-2.7613327e-4, 2.7182218e-6),
            Complex::new(-2.7613327e-4, -2.7182218e-6),
            Complex::new(7.9379964e-1, 3.3087225e-24),
        ]),
        Jones::from([
            Complex::new(7.938807e-1, 3.3087225e-24),
            Complex::new(-2.7613327e-4, 2.7182218e-6),
            Complex::new(-2.7613327e-4, -2.7182218e-6),
            Complex::new(7.9379964e-1, 3.3087225e-24),
        ]),
    ];
    let result = vis.slice(s![2, ..]);
    assert_abs_diff_eq!(expected_3rd_row, result, epsilon = epsilon);
}

#[track_caller]
fn test_power_law_off_zenith_visibilities_fee(vis: ArrayView2<Jones<f32>>, epsilon: f32) {
    // This time, all LMN values should be close to, but not the same as, (0, 0,
    // 1). This means that the visibilities should be somewhat close to the
    // input flux densities.

    let expected_1st_row = array![
        Jones::from([
            Complex::new(9.907028e-1, 9.717331e-2),
            Complex::new(-4.7281972e-4, -4.143827e-5),
            Complex::new(-4.718602e-4, -5.122087e-5),
            Complex::new(9.90933e-1, 9.7195886e-2),
        ]),
        Jones::from([
            Complex::new(9.942469e-1, 4.9068943e-2),
            Complex::new(-4.7427195e-4, -1.8485927e-5),
            Complex::new(-4.7378743e-4, -2.8303528e-5),
            Complex::new(9.944779e-1, 4.9080346e-2),
        ]),
        Jones::from([
            Complex::new(9.9428844e-1, -4.8220586e-2),
            Complex::new(-4.738114e-4, 2.789926e-5),
            Complex::new(-4.7428755e-4, 1.808125e-5),
            Complex::new(9.945195e-1, -4.823179e-2),
        ]),
    ];
    let result = vis.slice(s![0, ..]);
    assert_abs_diff_eq!(expected_1st_row, result, epsilon = epsilon);

    let expected_2nd_row = array![
        Jones::from([
            Complex::new(8.728561e-1, 9.9998936e-2),
            Complex::new(-3.6867274e-4, -4.3506087e-5),
            Complex::new(-3.6895974e-4, -4.1000934e-5),
            Complex::new(8.731409e-1, 1.0003157e-1),
        ]),
        Jones::from([
            Complex::new(8.7711203e-1, 5.051747e-2),
            Complex::new(-3.7054205e-4, -2.2604316e-5),
            Complex::new(-3.7068705e-4, -2.0086947e-5),
            Complex::new(8.773982e-1, 5.0533954e-2),
        ]),
        Jones::from([
            Complex::new(8.7716186e-1, -4.9644317e-2),
            Complex::new(-3.7070684e-4, 1.9717934e-5),
            Complex::new(-3.7056438e-4, 2.2235447e-5),
            Complex::new(8.774481e-1, -4.966052e-2),
        ]),
    ];
    let result = vis.slice(s![1, ..]);
    assert_abs_diff_eq!(expected_2nd_row, result, epsilon = epsilon);

    let expected_3rd_row = array![
        Jones::from([
            Complex::new(7.815219e-1, 1.0246288e-1),
            Complex::new(-2.3648771e-4, -2.7876984e-5),
            Complex::new(-2.3568133e-4, -3.4027613e-5),
            Complex::new(7.8216547e-1, 1.0254725e-1),
        ]),
        Jones::from([
            Complex::new(7.8650695e-1, 5.178786e-2),
            Complex::new(-2.377942e-4, -1.2549305e-5),
            Complex::new(-2.3738663e-4, -1.8739165e-5),
            Complex::new(7.871546e-1, 5.1830508e-2),
        ]),
        Jones::from([
            Complex::new(7.8656536e-1, -5.089305e-2),
            Complex::new(-2.3740779e-4, 1.8469087e-5),
            Complex::new(-2.3780833e-4, 1.2278767e-5),
            Complex::new(7.872131e-1, -5.093496e-2),
        ]),
    ];
    let result = vis.slice(s![2, ..]);
    assert_abs_diff_eq!(expected_3rd_row, result, epsilon = epsilon);
}

#[track_caller]
fn test_curved_power_law_zenith_visibilities(vis: ArrayView2<Jones<f32>>, epsilon: f32) {
    // All LMN values are (0, 0, 1). This means that the Fourier transform to
    // make a visibility from LMN and UVW will always just be the input flux
    // density.

    // First row: flux density at first frequency (i.e. 1 Jy XX and YY).
    let expected_1st_row = Array1::from_elem(vis.dim().0, Jones::identity());
    let result = vis.slice(s![0, ..]);
    assert_abs_diff_eq!(expected_1st_row, result, epsilon = epsilon);

    let expected_2nd_row = array![
        Jones::from([
            Complex::new(8.8461065e-1, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(8.8461065e-1, 0.0),
        ]),
        Jones::from([
            Complex::new(8.8461065e-1, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(8.8461065e-1, 0.0),
        ]),
        Jones::from([
            Complex::new(8.8461065e-1, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(8.8461065e-1, 0.0),
        ]),
    ];
    let result = vis.slice(s![1, ..]);
    assert_abs_diff_eq!(expected_2nd_row, result, epsilon = epsilon);

    let expected_3rd_row = array![
        Jones::from([
            Complex::new(7.9639274e-1, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(7.9639274e-1, 0.0),
        ]),
        Jones::from([
            Complex::new(7.9639274e-1, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(7.9639274e-1, 0.0),
        ]),
        Jones::from([
            Complex::new(7.9639274e-1, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(7.9639274e-1, 0.0),
        ]),
    ];
    let result = vis.slice(s![2, ..]);
    assert_abs_diff_eq!(expected_3rd_row, result, epsilon = epsilon);
}

#[track_caller]
fn test_curved_power_law_off_zenith_visibilities(vis: ArrayView2<Jones<f32>>, epsilon: f32) {
    // This time, all LMN values should be close to, but not the same as, (0, 0,
    // 1). This means that the visibilities should be somewhat close to the
    // input flux densities.

    let expected_1st_row = array![
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
    ];
    let result = vis.slice(s![0, ..]);
    assert_abs_diff_eq!(expected_1st_row, result, epsilon = epsilon);

    let expected_2nd_row = array![
        Jones::from([
            Complex::new(8.7886184e-1, 1.0068699e-1),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(8.7886184e-1, 1.0068699e-1),
        ]),
        Jones::from([
            Complex::new(8.8314706e-1, 5.086506e-2),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(8.8314706e-1, 5.086506e-2),
        ]),
        Jones::from([
            Complex::new(8.8319725e-1, -4.99859e-2),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(8.8319725e-1, -4.99859e-2),
        ]),
    ];
    let result = vis.slice(s![1, ..]);
    assert_abs_diff_eq!(expected_2nd_row, result, epsilon = epsilon);

    let expected_3rd_row = array![
        Jones::from([
            Complex::new(7.896351e-1, 1.0352658e-1),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(7.896351e-1, 1.0352658e-1),
        ]),
        Jones::from([
            Complex::new(7.946719e-1, 5.2325487e-2),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(7.946719e-1, 5.2325487e-2),
        ]),
        Jones::from([
            Complex::new(7.947309e-1, -5.1421385e-2),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(7.947309e-1, -5.1421385e-2),
        ]),
    ];
    let result = vis.slice(s![2, ..]);
    assert_abs_diff_eq!(expected_3rd_row, result, epsilon = epsilon);
}

#[track_caller]
fn test_curved_power_law_zenith_visibilities_fee(vis: ArrayView2<Jones<f32>>, epsilon: f32) {
    // All LMN values are (0, 0, 1). This means that the Fourier transform to
    // make a visibility from LMN and UVW will always just be the input flux
    // density *multiplied by the beam response*.

    // First row: flux density at first frequency (i.e. ~1 Jy XX and YY).
    let expected_1st_row = array![
        Jones::from([
            Complex::new(9.9958014e-1, 0.0),
            Complex::new(-5.4407195e-4, 4.5348115e-6),
            Complex::new(-5.4407195e-4, -4.5348115e-6),
            Complex::new(9.995511e-1, 0.0),
        ]),
        Jones::from([
            Complex::new(9.9958014e-1, 0.0),
            Complex::new(-5.4407195e-4, 4.5348115e-6),
            Complex::new(-5.4407195e-4, -4.5348115e-6),
            Complex::new(9.995511e-1, 0.0),
        ]),
        Jones::from([
            Complex::new(9.9958014e-1, 0.0),
            Complex::new(-5.4407195e-4, 4.5348115e-6),
            Complex::new(-5.4407195e-4, -4.5348115e-6),
            Complex::new(9.995511e-1, 0.0),
        ]),
    ];
    let result = vis.slice(s![0, ..]);
    assert_abs_diff_eq!(expected_1st_row, result, epsilon = epsilon);

    let expected_2nd_row = array![
        Jones::from([
            Complex::new(8.841108e-1, -1.6543612e-24),
            Complex::new(-4.2112975e-4, -1.6366728e-6),
            Complex::new(-4.2112975e-4, 1.6366728e-6),
            Complex::new(8.8407487e-1, -2.7755574e-17),
        ]),
        Jones::from([
            Complex::new(8.841108e-1, -1.6543612e-24),
            Complex::new(-4.2112975e-4, -1.6366728e-6),
            Complex::new(-4.2112975e-4, 1.6366728e-6),
            Complex::new(8.8407487e-1, -2.7755574e-17),
        ]),
        Jones::from([
            Complex::new(8.841108e-1, -1.6543612e-24),
            Complex::new(-4.2112975e-4, -1.6366728e-6),
            Complex::new(-4.2112975e-4, 1.6366728e-6),
            Complex::new(8.8407487e-1, -2.7755574e-17),
        ]),
    ];
    let result = vis.slice(s![1, ..]);
    assert_abs_diff_eq!(expected_2nd_row, result, epsilon = epsilon);

    let expected_3rd_row = array![
        Jones::from([
            Complex::new(7.958543e-1, -6.938894e-18),
            Complex::new(-2.7681972e-4, 2.7249791e-6),
            Complex::new(-2.7681972e-4, -2.7249791e-6),
            Complex::new(7.9577297e-1, 0.0),
        ]),
        Jones::from([
            Complex::new(7.958543e-1, -6.938894e-18),
            Complex::new(-2.7681972e-4, 2.7249791e-6),
            Complex::new(-2.7681972e-4, -2.7249791e-6),
            Complex::new(7.9577297e-1, 0.0),
        ]),
        Jones::from([
            Complex::new(7.958543e-1, -6.938894e-18),
            Complex::new(-2.7681972e-4, 2.7249791e-6),
            Complex::new(-2.7681972e-4, -2.7249791e-6),
            Complex::new(7.9577297e-1, 0.0),
        ]),
    ];
    let result = vis.slice(s![2, ..]);
    assert_abs_diff_eq!(expected_3rd_row, result, epsilon = epsilon);
}

#[track_caller]
fn test_curved_power_law_off_zenith_visibilities_fee(vis: ArrayView2<Jones<f32>>, epsilon: f32) {
    // This time, all LMN values should be close to, but not the same as, (0, 0,
    // 1). This means that the visibilities should be somewhat close to the
    // input flux densities.

    let expected_1st = array![
        Jones::from([
            Complex::new(9.907028e-1, 9.717331e-2),
            Complex::new(-4.7281972e-4, -4.143827e-5),
            Complex::new(-4.718602e-4, -5.122087e-5),
            Complex::new(9.90933e-1, 9.7195886e-2),
        ]),
        Jones::from([
            Complex::new(9.942469e-1, 4.9068943e-2),
            Complex::new(-4.7427195e-4, -1.8485927e-5),
            Complex::new(-4.7378743e-4, -2.8303528e-5),
            Complex::new(9.944779e-1, 4.9080346e-2),
        ]),
        Jones::from([
            Complex::new(9.9428844e-1, -4.8220586e-2),
            Complex::new(-4.738114e-4, 2.789926e-5),
            Complex::new(-4.7428755e-4, 1.808125e-5),
            Complex::new(9.945195e-1, -4.823179e-2),
        ]),
    ];
    let result = vis.slice(s![0, ..]);
    assert_abs_diff_eq!(expected_1st, result, epsilon = epsilon);

    let expected_2nd_row = array![
        Jones::from([
            Complex::new(8.7347853e-1, 1.0007025e-1),
            Complex::new(-3.6893564e-4, -4.353711e-5),
            Complex::new(-3.6922286e-4, -4.1030173e-5),
            Complex::new(8.7376356e-1, 1.001029e-1),
        ]),
        Jones::from([
            Complex::new(8.777375e-1, 5.0553497e-2),
            Complex::new(-3.708063e-4, -2.2620436e-5),
            Complex::new(-3.709514e-4, -2.0101272e-5),
            Complex::new(8.780239e-1, 5.0569993e-2),
        ]),
        Jones::from([
            Complex::new(8.777874e-1, -4.9679723e-2),
            Complex::new(-3.7097122e-4, 1.9731995e-5),
            Complex::new(-3.7082864e-4, 2.2251303e-5),
            Complex::new(8.780738e-1, -4.969593e-2),
        ]),
    ];
    let result = vis.slice(s![1, ..]);
    assert_abs_diff_eq!(expected_2nd_row, result, epsilon = epsilon);

    let expected_3rd_row = array![
        Jones::from([
            Complex::new(7.834647e-1, 1.0271759e-1),
            Complex::new(-2.3707561e-4, -2.7946284e-5),
            Complex::new(-2.3626721e-4, -3.4112203e-5),
            Complex::new(7.841099e-1, 1.0280218e-1),
        ]),
        Jones::from([
            Complex::new(7.8846216e-1, 5.1916603e-2),
            Complex::new(-2.3838534e-4, -1.2580502e-5),
            Complex::new(-2.3797675e-4, -1.878575e-5),
            Complex::new(7.8911144e-1, 5.1959354e-2),
        ]),
        Jones::from([
            Complex::new(7.885207e-1, -5.1019564e-2),
            Complex::new(-2.3799797e-4, 1.8515e-5),
            Complex::new(-2.383995e-4, 1.2309291e-5),
            Complex::new(7.8917e-1, -5.1061578e-2),
        ]),
    ];
    let result = vis.slice(s![2, ..]);
    assert_abs_diff_eq!(expected_3rd_row, result, epsilon = epsilon);
}

// I discovered that the code that handles a Gaussian's major and minor axes was
// not tested by other test functions. So these functions exist to be used with
// "gaussian2".
#[track_caller]
fn test_non_trivial_gaussian_list(vis: ArrayView2<Jones<f32>>, epsilon: f32) {
    let expected_1st = array![
        Jones::from([
            Complex::new(3.4502435e-1, 3.3841792e-2),
            Complex::new(-1.6466524e-4, -1.44313835e-5),
            Complex::new(-1.6433107e-4, -1.7838294e-5),
            Complex::new(3.4510452e-1, 3.3849653e-2),
        ]),
        Jones::from([
            Complex::new(7.4138254e-1, 3.6589358e-2),
            Complex::new(-3.5365153e-4, -1.3784447e-5),
            Complex::new(-3.5329023e-4, -2.1105161e-5),
            Complex::new(7.415548e-1, 3.6597863e-2),
        ]),
        Jones::from([
            Complex::new(5.542473e-1, -2.6879657e-2),
            Complex::new(-2.6411723e-4, 1.5551916e-5),
            Complex::new(-2.6438266e-4, 1.0079052e-5),
            Complex::new(5.543761e-1, -2.6885903e-2),
        ]),
    ];
    let result = vis.slice(s![0, ..]);
    assert_abs_diff_eq!(expected_1st, result, epsilon = epsilon);

    let expected_2nd_row = array![
        Jones::from([
            Complex::new(7.048634e-1, 8.075283e-2),
            Complex::new(-2.9771682e-4, -3.5132765e-5),
            Complex::new(-2.9794857e-4, -3.310976e-5),
            Complex::new(7.050934e-1, 8.077917e-2),
        ]),
        Jones::from([
            Complex::new(1.9964459e0, 1.14985764e-1),
            Complex::new(-8.434124e-4, -5.1451003e-5),
            Complex::new(-8.4374245e-4, -4.5721074e-5),
            Complex::new(1.9970974e0, 1.15023285e-1),
        ]),
        Jones::from([
            Complex::new(1.3436819e0, -7.604774e-2),
            Complex::new(-5.6786794e-4, 3.0204952e-5),
            Complex::new(-5.6764967e-4, 3.4061406e-5),
            Complex::new(1.3441204e0, -7.607255e-2),
        ]),
    ];
    let result = vis.slice(s![1, ..]);
    assert_abs_diff_eq!(expected_2nd_row, result, epsilon = epsilon);

    let expected_3rd_row = array![
        Jones::from([
            Complex::new(3.016709e-1, 3.9551124e-2),
            Complex::new(-9.128531e-5, -1.0760639e-5),
            Complex::new(-9.0974034e-5, -1.3134809e-5),
            Complex::new(3.0191934e-1, 3.9583694e-2),
        ]),
        Jones::from([
            Complex::new(1.1751746e0, 7.737984e-2),
            Complex::new(-3.5530483e-4, -1.8750789e-5),
            Complex::new(-3.5469583e-4, -2.7999487e-5),
            Complex::new(1.1761423e0, 7.744356e-2),
        ]),
        Jones::from([
            Complex::new(7.006485e-1, -4.533398e-2),
            Complex::new(-2.1147564e-4, 1.64517e-5),
            Complex::new(-2.1183242e-4, 1.0937552e-5),
            Complex::new(7.0122546e-1, -4.5371313e-2),
        ]),
    ];
    let result = vis.slice(s![2, ..]);
    assert_abs_diff_eq!(expected_3rd_row, result, epsilon = epsilon);
}

#[track_caller]
fn test_non_trivial_gaussian_power_law(vis: ArrayView2<Jones<f32>>, epsilon: f32) {
    let expected_1st = array![
        Jones::from([
            Complex::new(3.4502435e-1, 3.3841792e-2),
            Complex::new(-1.6466524e-4, -1.44313835e-5),
            Complex::new(-1.6433107e-4, -1.7838294e-5),
            Complex::new(3.4510452e-1, 3.3849653e-2),
        ]),
        Jones::from([
            Complex::new(7.4138254e-1, 3.6589358e-2),
            Complex::new(-3.5365153e-4, -1.3784447e-5),
            Complex::new(-3.5329023e-4, -2.1105161e-5),
            Complex::new(7.415548e-1, 3.6597863e-2),
        ]),
        Jones::from([
            Complex::new(5.542473e-1, -2.6879657e-2),
            Complex::new(-2.6411723e-4, 1.5551916e-5),
            Complex::new(-2.6438266e-4, 1.0079052e-5),
            Complex::new(5.543761e-1, -2.6885903e-2),
        ]),
    ];
    let result = vis.slice(s![0, ..]);
    assert_abs_diff_eq!(expected_1st, result, epsilon = epsilon);

    let expected_2nd_row = array![
        Jones::from([
            Complex::new(2.0769511e-1, 2.3794634e-2),
            Complex::new(-8.772526e-5, -1.0352224e-5),
            Complex::new(-8.779355e-5, -9.756126e-6),
            Complex::new(2.0776288e-1, 2.38024e-2),
        ]),
        Jones::from([
            Complex::new(5.882729e-1, 3.3881716e-2),
            Complex::new(-2.4852e-4, -1.5160557e-5),
            Complex::new(-2.4861723e-4, -1.3472176e-5),
            Complex::new(5.884649e-1, 3.3892773e-2),
        ]),
        Jones::from([
            Complex::new(3.9592943e-1, -2.2408232e-2),
            Complex::new(-1.6732802e-4, 8.900194e-6),
            Complex::new(-1.672637e-4, 1.0036537e-5),
            Complex::new(3.9605865e-1, -2.2415545e-2),
        ]),
    ];
    let result = vis.slice(s![1, ..]);
    assert_abs_diff_eq!(expected_2nd_row, result, epsilon = epsilon);

    let expected_3rd_row = array![
        Jones::from([
            Complex::new(1.19826384e-1, 1.571006e-2),
            Complex::new(-3.625934e-5, -4.274222e-6),
            Complex::new(-3.6135698e-5, -5.217263e-6),
            Complex::new(1.1992506e-1, 1.5722997e-2),
        ]),
        Jones::from([
            Complex::new(4.6678987e-1, 3.0735964e-2),
            Complex::new(-1.4113025e-4, -7.4479804e-6),
            Complex::new(-1.4088835e-4, -1.1121647e-5),
            Complex::new(4.6717426e-1, 3.0761274e-2),
        ]),
        Jones::from([
            Complex::new(2.7830383e-1, -1.8007062e-2),
            Complex::new(-8.4000014e-5, 6.5347626e-6),
            Complex::new(-8.4141735e-5, 4.3444934e-6),
            Complex::new(2.78533e-1, -1.802189e-2),
        ]),
    ];
    let result = vis.slice(s![2, ..]);
    assert_abs_diff_eq!(expected_3rd_row, result, epsilon = epsilon);
}

#[track_caller]
fn test_non_trivial_gaussian_curved_power_law(vis: ArrayView2<Jones<f32>>, epsilon: f32) {
    let expected_1st = array![
        Jones::from([
            Complex::new(3.4502435e-1, 3.3841792e-2),
            Complex::new(-1.6466524e-4, -1.44313835e-5),
            Complex::new(-1.6433107e-4, -1.7838294e-5),
            Complex::new(3.4510452e-1, 3.3849653e-2),
        ]),
        Jones::from([
            Complex::new(7.4138254e-1, 3.6589358e-2),
            Complex::new(-3.5365153e-4, -1.3784447e-5),
            Complex::new(-3.5329023e-4, -2.1105161e-5),
            Complex::new(7.415548e-1, 3.6597863e-2),
        ]),
        Jones::from([
            Complex::new(5.542473e-1, -2.6879657e-2),
            Complex::new(-2.6411723e-4, 1.5551916e-5),
            Complex::new(-2.6438266e-4, 1.0079052e-5),
            Complex::new(5.543761e-1, -2.6885903e-2),
        ]),
    ];
    let result = vis.slice(s![0, ..]);
    assert_abs_diff_eq!(expected_1st, result, epsilon = epsilon);

    let expected_2nd_row = array![
        Jones::from([
            Complex::new(2.0784323e-1, 2.3811603e-2),
            Complex::new(-8.778782e-5, -1.0359607e-5),
            Complex::new(-8.785616e-5, -9.763083e-6),
            Complex::new(2.0791104e-1, 2.3819372e-2),
        ]),
        Jones::from([
            Complex::new(5.886924e-1, 3.390588e-2),
            Complex::new(-2.486972e-4, -1.5171368e-5),
            Complex::new(-2.4879453e-4, -1.3481783e-5),
            Complex::new(5.8888453e-1, 3.3916943e-2),
        ]),
        Jones::from([
            Complex::new(3.9621177e-1, -2.2424214e-2),
            Complex::new(-1.6744735e-4, 8.906541e-6),
            Complex::new(-1.6738298e-4, 1.0043695e-5),
            Complex::new(3.963411e-1, -2.243153e-2),
        ]),
    ];
    let result = vis.slice(s![1, ..]);
    assert_abs_diff_eq!(expected_2nd_row, result, epsilon = epsilon);

    let expected_3rd_row = array![
        Jones::from([
            Complex::new(1.20124266e-1, 1.5749114e-2),
            Complex::new(-3.6349476e-5, -4.2848474e-6),
            Complex::new(-3.622553e-5, -5.230233e-6),
            Complex::new(1.2022318e-1, 1.5762083e-2),
        ]),
        Jones::from([
            Complex::new(4.6795025e-1, 3.0812372e-2),
            Complex::new(-1.4148108e-4, -7.4664954e-6),
            Complex::new(-1.4123859e-4, -1.1149295e-5),
            Complex::new(4.683356e-1, 3.0837744e-2),
        ]),
        Jones::from([
            Complex::new(2.789957e-1, -1.8051825e-2),
            Complex::new(-8.4208834e-5, 6.5510076e-6),
            Complex::new(-8.4350904e-5, 4.3552936e-6),
            Complex::new(2.7922544e-1, -1.8066691e-2),
        ]),
    ];
    let result = vis.slice(s![2, ..]);
    assert_abs_diff_eq!(expected_3rd_row, result, epsilon = epsilon);
}

#[track_caller]
fn test_multiple_gaussian_components(vis: ArrayView2<Jones<f32>>, epsilon: f32) {
    let expected = array![
        [
            Jones::from([
                Complex::new(1.9894463e0, 2.0495814e-1),
                Complex::new(0e0, 0e0),
                Complex::new(0e0, 0e0),
                Complex::new(1.9894463e0, 2.0495814e-1),
            ]),
            Jones::from([
                Complex::new(1.997311e0, 1.03556715e-1),
                Complex::new(0e0, 0e0),
                Complex::new(0e0, 0e0),
                Complex::new(1.997311e0, 1.03556715e-1),
            ]),
            Jones::from([
                Complex::new(1.9974082e0, -1.0167356e-1),
                Complex::new(0e0, 0e0),
                Complex::new(0e0, 0e0),
                Complex::new(1.9974082e0, -1.0167356e-1),
            ]),
        ],
        [
            Jones::from([
                Complex::new(5.9569197e0, 7.1689516e-1),
                Complex::new(0e0, 0e0),
                Complex::new(0e0, 0e0),
                Complex::new(5.9569197e0, 7.1689516e-1),
            ]),
            Jones::from([
                Complex::new(5.989021e0, 3.6238956e-1),
                Complex::new(0e0, 0e0),
                Complex::new(0e0, 0e0),
                Complex::new(5.989021e0, 3.6238956e-1),
            ]),
            Jones::from([
                Complex::new(5.9894176e0, -3.5580167e-1),
                Complex::new(0e0, 0e0),
                Complex::new(0e0, 0e0),
                Complex::new(5.9894176e0, -3.5580167e-1),
            ]),
        ],
        [
            Jones::from([
                Complex::new(3.9625018e0, 5.4580307e-1),
                Complex::new(0e0, 0e0),
                Complex::new(0e0, 0e0),
                Complex::new(3.9625018e0, 5.4580307e-1),
            ]),
            Jones::from([
                Complex::new(3.9904408e0, 2.760545e-1),
                Complex::new(0e0, 0e0),
                Complex::new(0e0, 0e0),
                Complex::new(3.9904408e0, 2.760545e-1),
            ]),
            Jones::from([
                Complex::new(3.9907863e0, -2.7103797e-1),
                Complex::new(0e0, 0e0),
                Complex::new(0e0, 0e0),
                Complex::new(3.9907863e0, -2.7103797e-1),
            ]),
        ]
    ];
    assert_abs_diff_eq!(expected, vis, epsilon = epsilon);
}

#[track_caller]
fn test_multiple_shapelet_components(
    vis: ArrayView2<Jones<f32>>,
    shapelet_uvws: ArrayView2<UVW>,
    epsilon1: f32,
    epsilon2: f64,
) {
    let expected = array![
        [
            Jones::from([
                Complex::new(1.9894463e0, 2.0495814e-1),
                Complex::new(0e0, 0e0),
                Complex::new(0e0, 0e0),
                Complex::new(1.9894463e0, 2.0495814e-1),
            ]),
            Jones::from([
                Complex::new(1.997311e0, 1.03556715e-1),
                Complex::new(0e0, 0e0),
                Complex::new(0e0, 0e0),
                Complex::new(1.997311e0, 1.03556715e-1),
            ]),
            Jones::from([
                Complex::new(1.9974082e0, -1.0167356e-1),
                Complex::new(0e0, 0e0),
                Complex::new(0e0, 0e0),
                Complex::new(1.9974082e0, -1.0167356e-1),
            ]),
        ],
        [
            Jones::from([
                Complex::new(5.9569197e0, 7.1689516e-1),
                Complex::new(0e0, 0e0),
                Complex::new(0e0, 0e0),
                Complex::new(5.9569197e0, 7.1689516e-1),
            ]),
            Jones::from([
                Complex::new(5.989021e0, 3.6238956e-1),
                Complex::new(0e0, 0e0),
                Complex::new(0e0, 0e0),
                Complex::new(5.989021e0, 3.6238956e-1),
            ]),
            Jones::from([
                Complex::new(5.9894176e0, -3.5580167e-1),
                Complex::new(0e0, 0e0),
                Complex::new(0e0, 0e0),
                Complex::new(5.9894176e0, -3.5580167e-1),
            ]),
        ],
        [
            Jones::from([
                Complex::new(3.9625018e0, 5.4580307e-1),
                Complex::new(0e0, 0e0),
                Complex::new(0e0, 0e0),
                Complex::new(3.9625018e0, 5.4580307e-1),
            ]),
            Jones::from([
                Complex::new(3.9904408e0, 2.760545e-1),
                Complex::new(0e0, 0e0),
                Complex::new(0e0, 0e0),
                Complex::new(3.9904408e0, 2.760545e-1),
            ]),
            Jones::from([
                Complex::new(3.9907863e0, -2.7103797e-1),
                Complex::new(0e0, 0e0),
                Complex::new(0e0, 0e0),
                Complex::new(3.9907863e0, -2.7103797e-1),
            ]),
        ]
    ];
    assert_abs_diff_eq!(expected, vis, epsilon = epsilon1);

    let expected = array![
        [
            UVW {
                u: 1.9996953903127825,
                v: 0.01584645344024005,
                w: 0.0,
            },
            UVW {
                u: 1.9996314242432884,
                v: 0.017430912937512553,
                w: 0.0,
            }
        ],
        [
            UVW {
                u: 1.0173001015936747,
                v: -0.44599812806736405,
                w: 0.0,
            },
            UVW {
                u: 1.019013154521334,
                v: -0.4451913783247998,
                w: 0.0,
            }
        ],
        [
            UVW {
                u: -0.9823952887191078,
                v: -0.4618445815076041,
                w: 0.0,
            },
            UVW {
                u: -0.9806182697219545,
                v: -0.46262229126231236,
                w: 0.0,
            }
        ]
    ];
    assert_abs_diff_eq!(expected, shapelet_uvws, epsilon = epsilon2);
}

fn test_model_timestep_autos_with_point(vis: ArrayView2<Jones<f32>>, epsilon: f32) {
    let expected = array![
        [
            Jones::identity() * 1e0,
            Jones::identity() * 1e0,
            Jones::identity() * 1e0,
        ],
        [
            Jones::identity() * 3e0,
            Jones::identity() * 3e0,
            Jones::identity() * 3e0,
        ],
        [
            Jones::identity() * 2e0,
            Jones::identity() * 2e0,
            Jones::identity() * 2e0,
        ]
    ];
    assert_abs_diff_eq!(expected, vis, epsilon = epsilon);
}

fn test_model_timestep_autos_with_gaussian(vis: ArrayView2<Jones<f32>>, epsilon: f32) {
    // the gaussian test srclist generates identical autocorrelations to the point test
    test_model_timestep_autos_with_point(vis, epsilon);
}

fn test_model_timestep_autos_with_shapelet(vis: ArrayView2<Jones<f32>>, epsilon: f32) {
    // the shapelet test srclist generates identical autocorrelations to the point test
    test_model_timestep_autos_with_point(vis, epsilon);
}
