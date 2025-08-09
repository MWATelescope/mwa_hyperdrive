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
            beam,
            flagged_tiles,
            array_longitude_rad,
            array_latitude_rad,
        }
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
            Complex::new(9.9958146e-1, 0.0),
            Complex::new(-5.405832e-4, 5.00542e-6),
            Complex::new(-5.405832e-4, -5.00542e-6),
            Complex::new(9.995525e-1, 0.0),
        ]),
        Jones::from([
            Complex::new(9.9958146e-1, 0.0),
            Complex::new(-5.405832e-4, 5.00542e-6),
            Complex::new(-5.405832e-4, -5.00542e-6),
            Complex::new(9.995525e-1, 0.0),
        ]),
        Jones::from([
            Complex::new(9.9958146e-1, 0.0),
            Complex::new(-5.405832e-4, 5.00542e-6),
            Complex::new(-5.405832e-4, -5.00542e-6),
            Complex::new(9.995525e-1, 0.0),
        ]),
    ];
    let result = vis.slice(s![0, ..]);
    assert_abs_diff_eq!(expected_1st_row, result, epsilon = epsilon);
    // ~3 Jy XX and YY.
    let expected_2nd_row = array![
        Jones::from([
            Complex::new(2.9982994e0, 0.0),
            Complex::new(-1.4202512e-3, -1.0991403e-5),
            Complex::new(-1.4202512e-3, 1.0991403e-5),
            Complex::new(2.998176e0, -1.110223e-16),
        ]),
        Jones::from([
            Complex::new(2.9982994e0, 0.0),
            Complex::new(-1.4202512e-3, -1.0991403e-5),
            Complex::new(-1.4202512e-3, 1.0991403e-5),
            Complex::new(2.998176e0, -1.110223e-16),
        ]),
        Jones::from([
            Complex::new(2.9982994e0, 0.0),
            Complex::new(-1.4202512e-3, -1.0991403e-5),
            Complex::new(-1.4202512e-3, 1.0991403e-5),
            Complex::new(2.998176e0, -1.110223e-16),
        ]),
    ];
    let result = vis.slice(s![1, ..]);
    assert_abs_diff_eq!(expected_2nd_row, result, epsilon = epsilon);
    // ~2 Jy XX and YY.
    let expected_3rd_row = array![
        Jones::from([
            Complex::new(1.9986509e0, 0.0),
            Complex::new(-6.9212046e-4, 8.06261e-6),
            Complex::new(-6.9212046e-4, -8.06261e-6),
            Complex::new(1.9984484e0, 0.0),
        ]),
        Jones::from([
            Complex::new(1.9986509e0, 0.0),
            Complex::new(-6.9212046e-4, 8.06261e-6),
            Complex::new(-6.9212046e-4, -8.06261e-6),
            Complex::new(1.9984484e0, 0.0),
        ]),
        Jones::from([
            Complex::new(1.9986509e0, 0.0),
            Complex::new(-6.9212046e-4, 8.06261e-6),
            Complex::new(-6.9212046e-4, -8.06261e-6),
            Complex::new(1.9984484e0, 0.0),
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
            Complex::new(9.907169e-1, 9.717469e-2),
            Complex::new(-4.694063e-4, -4.0636343e-5),
            Complex::new(-4.6835604e-4, -5.1344286e-5),
            Complex::new(9.9094635e-1, 9.71972e-2),
        ]),
        Jones::from([
            Complex::new(9.942611e-1, 4.906964e-2),
            Complex::new(-4.7082372e-4, -1.785029e-5),
            Complex::new(-4.7029337e-4, -2.8596542e-5),
            Complex::new(9.9449134e-1, 4.9081005e-2),
        ]),
        Jones::from([
            Complex::new(9.943026e-1, -4.822127e-2),
            Complex::new(-4.7031758e-4, 2.8195254e-5),
            Complex::new(-4.7083877e-4, 1.7448556e-5),
            Complex::new(9.945328e-1, -4.823244e-2),
        ]),
    ];
    let result = vis.slice(s![0, ..]);
    assert_abs_diff_eq!(expected_1st_row, result, epsilon = epsilon);

    let expected_2nd_row = array![
        Jones::from([
            Complex::new(2.962178e0, 3.3936253e-1),
            Complex::new(-1.2438099e-3, -1.5220148e-4),
            Complex::new(-1.2460046e-3, -1.3304464e-4),
            Complex::new(2.9631553e0, 3.394745e-1),
        ]),
        Jones::from([
            Complex::new(2.9766212e0, 1.7143919e-1),
            Complex::new(-1.2504229e-3, -8.1675455e-5),
            Complex::new(-1.2515316e-3, -6.2425206e-5),
            Complex::new(2.9776032e0, 1.7149575e-1),
        ]),
        Jones::from([
            Complex::new(2.9767904e0, -1.6847602e-1),
            Complex::new(-1.2515932e-3, 6.117933e-5),
            Complex::new(-1.2505036e-3, 8.043067e-5),
            Complex::new(2.9777725e0, -1.685316e-1),
        ]),
    ];
    let result = vis.slice(s![1, ..]);
    assert_abs_diff_eq!(expected_2nd_row, result, epsilon = epsilon);

    let expected_3rd_row = array![
        Jones::from([
            Complex::new(1.9675823e0, 2.5796354e-1),
            Complex::new(-5.92617e-4, -6.803491e-5),
            Complex::new(-5.901265e-4, -8.7030865e-5),
            Complex::new(1.969188e0, 2.5817403e-1),
        ]),
        Jones::from([
            Complex::new(1.9801328e0, 1.3038263e-1),
            Complex::new(-5.957733e-4, -2.9628924e-5),
            Complex::new(-5.945145e-4, -4.874605e-5),
            Complex::new(1.9817487e0, 1.3048902e-1),
        ]),
        Jones::from([
            Complex::new(1.9802799e0, -1.2812982e-1),
            Complex::new(-5.945696e-4, 4.8069658e-5),
            Complex::new(-5.9580663e-4, 2.8951115e-5),
            Complex::new(1.9818958e0, -1.2823439e-1),
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
            Complex::new(9.9958146e-1, 0.0),
            Complex::new(-5.405832e-4, 5.00542e-6),
            Complex::new(-5.405832e-4, -5.00542e-6),
            Complex::new(9.995525e-1, 0.0),
        ]),
        Jones::from([
            Complex::new(9.9958146e-1, 0.0),
            Complex::new(-5.405832e-4, 5.00542e-6),
            Complex::new(-5.405832e-4, -5.00542e-6),
            Complex::new(9.995525e-1, 0.0),
        ]),
        Jones::from([
            Complex::new(9.9958146e-1, 0.0),
            Complex::new(-5.405832e-4, 5.00542e-6),
            Complex::new(-5.405832e-4, -5.00542e-6),
            Complex::new(9.995525e-1, 0.0),
        ]),
    ];
    let result = vis.slice(s![0, ..]);
    assert_abs_diff_eq!(expected_1st_row, result, epsilon = epsilon);

    let expected_2nd_row = array![
        Jones::from([
            Complex::new(8.834791e-1, 1.6543612e-24),
            Complex::new(-4.1849137e-4, -3.2387275e-6),
            Complex::new(-4.1849137e-4, 3.2387275e-6),
            Complex::new(8.834428e-1, 0.0),
        ]),
        Jones::from([
            Complex::new(8.834791e-1, 1.6543612e-24),
            Complex::new(-4.1849137e-4, -3.2387275e-6),
            Complex::new(-4.1849137e-4, 3.2387275e-6),
            Complex::new(8.834428e-1, 0.0),
        ]),
        Jones::from([
            Complex::new(8.834791e-1, 1.6543612e-24),
            Complex::new(-4.1849137e-4, -3.2387275e-6),
            Complex::new(-4.1849137e-4, 3.2387275e-6),
            Complex::new(8.834428e-1, 0.0),
        ]),
    ];
    let result = vis.slice(s![1, ..]);
    assert_abs_diff_eq!(expected_2nd_row, result, epsilon = epsilon);

    let expected_3rd_row = array![
        Jones::from([
            Complex::new(7.93882e-1, 0.0),
            Complex::new(-2.7491644e-4, 3.2025407e-6),
            Complex::new(-2.7491644e-4, -3.2025407e-6),
            Complex::new(7.9380155e-1, 3.3087225e-24),
        ]),
        Jones::from([
            Complex::new(7.93882e-1, 0.0),
            Complex::new(-2.7491644e-4, 3.2025407e-6),
            Complex::new(-2.7491644e-4, -3.2025407e-6),
            Complex::new(7.9380155e-1, 3.3087225e-24),
        ]),
        Jones::from([
            Complex::new(7.93882e-1, 0.0),
            Complex::new(-2.7491644e-4, 3.2025407e-6),
            Complex::new(-2.7491644e-4, -3.2025407e-6),
            Complex::new(7.9380155e-1, 3.3087225e-24),
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
            Complex::new(9.907169e-1, 9.717469e-2),
            Complex::new(-4.694063e-4, -4.0636343e-5),
            Complex::new(-4.6835604e-4, -5.1344286e-5),
            Complex::new(9.9094635e-1, 9.71972e-2),
        ]),
        Jones::from([
            Complex::new(9.942611e-1, 4.906964e-2),
            Complex::new(-4.7082372e-4, -1.785029e-5),
            Complex::new(-4.7029337e-4, -2.8596542e-5),
            Complex::new(9.9449134e-1, 4.9081005e-2),
        ]),
        Jones::from([
            Complex::new(9.943026e-1, -4.822127e-2),
            Complex::new(-4.7031758e-4, 2.8195254e-5),
            Complex::new(-4.7083877e-4, 1.7448556e-5),
            Complex::new(9.945328e-1, -4.823244e-2),
        ]),
    ];
    let result = vis.slice(s![0, ..]);
    assert_abs_diff_eq!(expected_1st_row, result, epsilon = epsilon);

    let expected_2nd_row = array![
        Jones::from([
            Complex::new(8.7283564e-1, 9.99966e-2),
            Complex::new(-3.6650113e-4, -4.4847704e-5),
            Complex::new(-3.6714782e-4, -3.9202947e-5),
            Complex::new(8.731236e-1, 1.0002959e-1),
        ]),
        Jones::from([
            Complex::new(8.7709147e-1, 5.0516285e-2),
            Complex::new(-3.6844972e-4, -2.4066496e-5),
            Complex::new(-3.6877644e-4, -1.8394216e-5),
            Complex::new(8.7738085e-1, 5.0532952e-2),
        ]),
        Jones::from([
            Complex::new(8.771413e-1, -4.9643155e-2),
            Complex::new(-3.6879454e-4, 1.8027107e-5),
            Complex::new(-3.684735e-4, 2.3699708e-5),
            Complex::new(8.7743074e-1, -4.9659535e-2),
        ]),
    ];
    let result = vis.slice(s![1, ..]);
    assert_abs_diff_eq!(expected_2nd_row, result, epsilon = epsilon);

    let expected_3rd_row = array![
        Jones::from([
            Complex::new(7.815413e-1, 1.0246542e-1),
            Complex::new(-2.3539278e-4, -2.7024074e-5),
            Complex::new(-2.3440353e-4, -3.456944e-5),
            Complex::new(7.8217906e-1, 1.0254903e-1),
        ]),
        Jones::from([
            Complex::new(7.8652644e-1, 5.1789146e-2),
            Complex::new(-2.3664648e-4, -1.1768873e-5),
            Complex::new(-2.3614647e-4, -1.9362365e-5),
            Complex::new(7.8716826e-1, 5.183141e-2),
        ]),
        Jones::from([
            Complex::new(7.8658485e-1, -5.0894313e-2),
            Complex::new(-2.3616836e-4, 1.9093699e-5),
            Complex::new(-2.366597e-4, 1.1499642e-5),
            Complex::new(7.8722674e-1, -5.0935842e-2),
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
            Complex::new(9.9958146e-1, 0.0),
            Complex::new(-5.405832e-4, 5.00542e-6),
            Complex::new(-5.405832e-4, -5.00542e-6),
            Complex::new(9.995525e-1, 0.0),
        ]),
        Jones::from([
            Complex::new(9.9958146e-1, 0.0),
            Complex::new(-5.405832e-4, 5.00542e-6),
            Complex::new(-5.405832e-4, -5.00542e-6),
            Complex::new(9.995525e-1, 0.0),
        ]),
        Jones::from([
            Complex::new(9.9958146e-1, 0.0),
            Complex::new(-5.405832e-4, 5.00542e-6),
            Complex::new(-5.405832e-4, -5.00542e-6),
            Complex::new(9.995525e-1, 0.0),
        ]),
    ];
    let result = vis.slice(s![0, ..]);
    assert_abs_diff_eq!(expected_1st_row, result, epsilon = epsilon);

    let expected_2nd_row = array![
        Jones::from([
            Complex::new(8.8410914e-1, 1.6543612e-24),
            Complex::new(-4.187898e-4, -3.2410371e-6),
            Complex::new(-4.187898e-4, 3.2410371e-6),
            Complex::new(8.8407284e-1, 3.3087225e-24),
        ]),
        Jones::from([
            Complex::new(8.8410914e-1, 1.6543612e-24),
            Complex::new(-4.187898e-4, -3.2410371e-6),
            Complex::new(-4.187898e-4, 3.2410371e-6),
            Complex::new(8.8407284e-1, 3.3087225e-24),
        ]),
        Jones::from([
            Complex::new(8.8410914e-1, 1.6543612e-24),
            Complex::new(-4.187898e-4, -3.2410371e-6),
            Complex::new(-4.187898e-4, 3.2410371e-6),
            Complex::new(8.8407284e-1, 3.3087225e-24),
        ]),
    ];
    let result = vis.slice(s![1, ..]);
    assert_abs_diff_eq!(expected_2nd_row, result, epsilon = epsilon);

    let expected_3rd_row = array![
        Jones::from([
            Complex::new(7.958555e-1, -6.938894e-18),
            Complex::new(-2.7559986e-4, 3.210502e-6),
            Complex::new(-2.7559986e-4, -3.210502e-6),
            Complex::new(7.957749e-1, 6.938894e-18),
        ]),
        Jones::from([
            Complex::new(7.958555e-1, -6.938894e-18),
            Complex::new(-2.7559986e-4, 3.210502e-6),
            Complex::new(-2.7559986e-4, -3.210502e-6),
            Complex::new(7.957749e-1, 6.938894e-18),
        ]),
        Jones::from([
            Complex::new(7.958555e-1, -6.938894e-18),
            Complex::new(-2.7559986e-4, 3.210502e-6),
            Complex::new(-2.7559986e-4, -3.210502e-6),
            Complex::new(7.957749e-1, 6.938894e-18),
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
            Complex::new(9.907169e-1, 9.717469e-2),
            Complex::new(-4.694063e-4, -4.0636343e-5),
            Complex::new(-4.6835604e-4, -5.1344286e-5),
            Complex::new(9.9094635e-1, 9.71972e-2),
        ]),
        Jones::from([
            Complex::new(9.942611e-1, 4.906964e-2),
            Complex::new(-4.7082372e-4, -1.785029e-5),
            Complex::new(-4.7029337e-4, -2.8596542e-5),
            Complex::new(9.9449134e-1, 4.9081005e-2),
        ]),
        Jones::from([
            Complex::new(9.943026e-1, -4.822127e-2),
            Complex::new(-4.7031758e-4, 2.8195254e-5),
            Complex::new(-4.7083877e-4, 1.7448556e-5),
            Complex::new(9.945328e-1, -4.823244e-2),
        ]),
    ];
    let result = vis.slice(s![0, ..]);
    assert_abs_diff_eq!(expected_1st, result, epsilon = epsilon);

    let expected_2nd_row = array![
        Jones::from([
            Complex::new(8.73458e-1, 1.00067906e-1),
            Complex::new(-3.667625e-4, -4.4879685e-5),
            Complex::new(-3.6740967e-4, -3.92309e-5),
            Complex::new(8.737462e-1, 1.0010092e-1),
        ]),
        Jones::from([
            Complex::new(8.7771696e-1, 5.055231e-2),
            Complex::new(-3.6871247e-4, -2.4083658e-5),
            Complex::new(-3.6903942e-4, -1.8407334e-5),
            Complex::new(8.780065e-1, 5.056899e-2),
        ]),
        Jones::from([
            Complex::new(8.7776685e-1, -4.9678557e-2),
            Complex::new(-3.6905755e-4, 1.8039962e-5),
            Complex::new(-3.6873628e-4, 2.371661e-5),
            Complex::new(8.780564e-1, -4.9694948e-2),
        ]),
    ];
    let result = vis.slice(s![1, ..]);
    assert_abs_diff_eq!(expected_2nd_row, result, epsilon = epsilon);

    let expected_3rd_row = array![
        Jones::from([
            Complex::new(7.8348416e-1, 1.0272014e-1),
            Complex::new(-2.3597795e-4, -2.7091255e-5),
            Complex::new(-2.3498623e-4, -3.4655375e-5),
            Complex::new(7.841235e-1, 1.0280396e-1),
        ]),
        Jones::from([
            Complex::new(7.884817e-1, 5.1917892e-2),
            Complex::new(-2.3723475e-4, -1.179813e-5),
            Complex::new(-2.3673351e-4, -1.94105e-5),
            Complex::new(7.891251e-1, 5.1960256e-2),
        ]),
        Jones::from([
            Complex::new(7.8854024e-1, -5.102083e-2),
            Complex::new(-2.3675544e-4, 1.9141164e-5),
            Complex::new(-2.3724802e-4, 1.1528229e-5),
            Complex::new(7.8918374e-1, -5.1062465e-2),
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
            Complex::new(3.4502926e-1, 3.3842273e-2),
            Complex::new(-1.6347648e-4, -1.4152103e-5),
            Complex::new(-1.631107e-4, -1.7881275e-5),
            Complex::new(3.4510916e-1, 3.385011e-2),
        ]),
        Jones::from([
            Complex::new(7.413931e-1, 3.658988e-2),
            Complex::new(-3.510803e-4, -1.331047e-5),
            Complex::new(-3.506848e-4, -2.1323653e-5),
            Complex::new(7.415648e-1, 3.6598355e-2),
        ]),
        Jones::from([
            Complex::new(5.542552e-1, -2.6880039e-2),
            Complex::new(-2.6216966e-4, 1.5716912e-5),
            Complex::new(-2.624602e-4, 9.726368e-6),
            Complex::new(5.543836e-1, -2.6886264e-2),
        ]),
    ];
    let result = vis.slice(s![0, ..]);
    assert_abs_diff_eq!(expected_1st, result, epsilon = epsilon);

    let expected_2nd_row = array![
        Jones::from([
            Complex::new(7.0484686e-1, 8.0750935e-2),
            Complex::new(-2.9596317e-4, -3.621617e-5),
            Complex::new(-2.9648538e-4, -3.1657823e-5),
            Complex::new(7.0507944e-1, 8.077758e-2),
        ]),
        Jones::from([
            Complex::new(1.9963992e0, 1.14983074e-1),
            Complex::new(-8.3865e-4, -5.4779157e-5),
            Complex::new(-8.393936e-4, -4.1868152e-5),
            Complex::new(1.9970578e0, 1.1502101e-1),
        ]),
        Jones::from([
            Complex::new(1.3436505e0, -7.604595e-2),
            Complex::new(-5.649386e-4, 2.7614855e-5),
            Complex::new(-5.644468e-4, 3.630444e-5),
            Complex::new(1.3440938e0, -7.607105e-2),
        ]),
    ];
    let result = vis.slice(s![1, ..]);
    assert_abs_diff_eq!(expected_2nd_row, result, epsilon = epsilon);

    let expected_3rd_row = array![
        Jones::from([
            Complex::new(3.016784e-1, 3.9552104e-2),
            Complex::new(-9.086265e-5, -1.0431412e-5),
            Complex::new(-9.04808e-5, -1.3343956e-5),
            Complex::new(3.019246e-1, 3.958438e-2),
        ]),
        Jones::from([
            Complex::new(1.1752038e0, 7.738176e-2),
            Complex::new(-3.5358992e-4, -1.758469e-5),
            Complex::new(-3.5284285e-4, -2.8930655e-5),
            Complex::new(1.1761627e0, 7.74449e-2),
        ]),
        Jones::from([
            Complex::new(7.006659e-1, -4.5335107e-2),
            Complex::new(-2.1037158e-4, 1.7008086e-5),
            Complex::new(-2.1080927e-4, 1.02435315e-5),
            Complex::new(7.012376e-1, -4.53721e-2),
        ]),
    ];
    let result = vis.slice(s![2, ..]);
    assert_abs_diff_eq!(expected_3rd_row, result, epsilon = epsilon);
}

#[track_caller]
fn test_non_trivial_gaussian_power_law(vis: ArrayView2<Jones<f32>>, epsilon: f32) {
    let expected_1st = array![
        Jones::from([
            Complex::new(3.4502926e-1, 3.3842273e-2),
            Complex::new(-1.6347648e-4, -1.4152103e-5),
            Complex::new(-1.631107e-4, -1.7881275e-5),
            Complex::new(3.4510916e-1, 3.385011e-2),
        ]),
        Jones::from([
            Complex::new(7.413931e-1, 3.658988e-2),
            Complex::new(-3.510803e-4, -1.331047e-5),
            Complex::new(-3.506848e-4, -2.1323653e-5),
            Complex::new(7.415648e-1, 3.6598355e-2),
        ]),
        Jones::from([
            Complex::new(5.542552e-1, -2.6880039e-2),
            Complex::new(-2.6216966e-4, 1.5716912e-5),
            Complex::new(-2.624602e-4, 9.726368e-6),
            Complex::new(5.543836e-1, -2.6886264e-2),
        ]),
    ];
    let result = vis.slice(s![0, ..]);
    assert_abs_diff_eq!(expected_1st, result, epsilon = epsilon);

    let expected_2nd_row = array![
        Jones::from([
            Complex::new(2.0769024e-1, 2.3794077e-2),
            Complex::new(-8.720853e-5, -1.067146e-5),
            Complex::new(-8.736241e-5, -9.328297e-6),
            Complex::new(2.0775877e-1, 2.3801927e-2),
        ]),
        Jones::from([
            Complex::new(5.8825916e-1, 3.3880923e-2),
            Complex::new(-2.4711667e-4, -1.6141232e-5),
            Complex::new(-2.473358e-4, -1.2336873e-5),
            Complex::new(5.8845323e-1, 3.3892103e-2),
        ]),
        Jones::from([
            Complex::new(3.9592016e-1, -2.2407709e-2),
            Complex::new(-1.6646486e-4, 8.136995e-6),
            Complex::new(-1.6631994e-4, 1.0697469e-5),
            Complex::new(3.9605078e-1, -2.2415102e-2),
        ]),
    ];
    let result = vis.slice(s![1, ..]);
    assert_abs_diff_eq!(expected_2nd_row, result, epsilon = epsilon);

    let expected_3rd_row = array![
        Jones::from([
            Complex::new(1.1982936e-1, 1.5710449e-2),
            Complex::new(-3.6091456e-5, -4.14345e-6),
            Complex::new(-3.593978e-5, -5.3003387e-6),
            Complex::new(1.1992714e-1, 1.572327e-2),
        ]),
        Jones::from([
            Complex::new(4.6680143e-1, 3.0736726e-2),
            Complex::new(-1.4044908e-4, -6.9847965e-6),
            Complex::new(-1.4015233e-4, -1.1491515e-5),
            Complex::new(4.6718237e-1, 3.0761808e-2),
        ]),
        Jones::from([
            Complex::new(2.7831075e-1, -1.800751e-2),
            Complex::new(-8.356148e-5, 6.755764e-6),
            Complex::new(-8.373533e-5, 4.0688224e-6),
            Complex::new(2.7853787e-1, -1.8022204e-2),
        ]),
    ];
    let result = vis.slice(s![2, ..]);
    assert_abs_diff_eq!(expected_3rd_row, result, epsilon = epsilon);
}

#[track_caller]
fn test_non_trivial_gaussian_curved_power_law(vis: ArrayView2<Jones<f32>>, epsilon: f32) {
    let expected_1st = array![
        Jones::from([
            Complex::new(3.4502926e-1, 3.3842273e-2),
            Complex::new(-1.6347648e-4, -1.4152103e-5),
            Complex::new(-1.631107e-4, -1.7881275e-5),
            Complex::new(3.4510916e-1, 3.385011e-2),
        ]),
        Jones::from([
            Complex::new(7.413931e-1, 3.658988e-2),
            Complex::new(-3.510803e-4, -1.331047e-5),
            Complex::new(-3.506848e-4, -2.1323653e-5),
            Complex::new(7.415648e-1, 3.6598355e-2),
        ]),
        Jones::from([
            Complex::new(5.542552e-1, -2.6880039e-2),
            Complex::new(-2.6216966e-4, 1.5716912e-5),
            Complex::new(-2.624602e-4, 9.726368e-6),
            Complex::new(5.543836e-1, -2.6886264e-2),
        ]),
    ];
    let result = vis.slice(s![0, ..]);
    assert_abs_diff_eq!(expected_1st, result, epsilon = epsilon);

    let expected_2nd_row = array![
        Jones::from([
            Complex::new(2.0783836e-1, 2.3811044e-2),
            Complex::new(-8.7270724e-5, -1.067907e-5),
            Complex::new(-8.742471e-5, -9.334949e-6),
            Complex::new(2.0790693e-1, 2.38189e-2),
        ]),
        Jones::from([
            Complex::new(5.8867866e-1, 3.3905085e-2),
            Complex::new(-2.472929e-4, -1.6152742e-5),
            Complex::new(-2.4751216e-4, -1.2345671e-5),
            Complex::new(5.888729e-1, 3.391627e-2),
        ]),
        Jones::from([
            Complex::new(3.962025e-1, -2.2423686e-2),
            Complex::new(-1.6658356e-4, 8.142798e-6),
            Complex::new(-1.6643855e-4, 1.0705098e-5),
            Complex::new(3.9633322e-1, -2.2431085e-2),
        ]),
    ];
    let result = vis.slice(s![1, ..]);
    assert_abs_diff_eq!(expected_2nd_row, result, epsilon = epsilon);

    let expected_3rd_row = array![
        Jones::from([
            Complex::new(1.20127246e-1, 1.5749505e-2),
            Complex::new(-3.6181176e-5, -4.1537505e-6),
            Complex::new(-3.6029123e-5, -5.313515e-6),
            Complex::new(1.20225266e-1, 1.5762357e-2),
        ]),
        Jones::from([
            Complex::new(4.6796188e-1, 3.0813135e-2),
            Complex::new(-1.4079822e-4, -7.00216e-6),
            Complex::new(-1.4050074e-4, -1.1520082e-5),
            Complex::new(4.6834373e-1, 3.0838279e-2),
        ]),
        Jones::from([
            Complex::new(2.790026e-1, -1.8052274e-2),
            Complex::new(-8.37692e-5, 6.772558e-6),
            Complex::new(-8.394349e-5, 4.078937e-6),
            Complex::new(2.7923027e-1, -1.8067004e-2),
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
