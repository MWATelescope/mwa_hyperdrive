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
            Complex::new(9.9957865e-1, 0.0),
            Complex::new(-5.7808764e-4, 6.5317404e-6),
            Complex::new(-5.7808764e-4, -6.5317404e-6),
            Complex::new(9.9954975e-1, 0.0),
        ]),
        Jones::from([
            Complex::new(9.9957865e-1, 0.0),
            Complex::new(-5.7808764e-4, 6.5317404e-6),
            Complex::new(-5.7808764e-4, -6.5317404e-6),
            Complex::new(9.9954975e-1, 0.0),
        ]),
        Jones::from([
            Complex::new(9.9957865e-1, 0.0),
            Complex::new(-5.7808764e-4, 6.5317404e-6),
            Complex::new(-5.7808764e-4, -6.5317404e-6),
            Complex::new(9.9954975e-1, 0.0),
        ]),
    ];
    let result = vis.slice(s![0, ..]);
    assert_abs_diff_eq!(expected_1st_row, result, epsilon = epsilon);
    // ~3 Jy XX and YY.
    let expected_2nd_row = array![
        Jones::from([
            Complex::new(2.9983046e0, -6.617445e-24),
            Complex::new(-1.4190843e-3, -7.828336e-6),
            Complex::new(-1.4190843e-3, 7.828336e-6),
            Complex::new(2.998183e0, 6.617445e-24),
        ]),
        Jones::from([
            Complex::new(2.9983046e0, -6.617445e-24),
            Complex::new(-1.4190843e-3, -7.828336e-6),
            Complex::new(-1.4190843e-3, 7.828336e-6),
            Complex::new(2.998183e0, 6.617445e-24),
        ]),
        Jones::from([
            Complex::new(2.9983046e0, -6.617445e-24),
            Complex::new(-1.4190843e-3, -7.828336e-6),
            Complex::new(-1.4190843e-3, 7.828336e-6),
            Complex::new(2.998183e0, 6.617445e-24),
        ]),
    ];
    let result = vis.slice(s![1, ..]);
    assert_abs_diff_eq!(expected_2nd_row, result, epsilon = epsilon);
    // ~2 Jy XX and YY.
    let expected_3rd_row = array![
        Jones::from([
            Complex::new(1.9986478e0, 0.0),
            Complex::new(-6.9266825e-4, 7.9946185e-6),
            Complex::new(-6.9266825e-4, -7.9946185e-6),
            Complex::new(1.9984438e0, 0.0),
        ]),
        Jones::from([
            Complex::new(1.9986478e0, 0.0),
            Complex::new(-6.9266825e-4, 7.9946185e-6),
            Complex::new(-6.9266825e-4, -7.9946185e-6),
            Complex::new(1.9984438e0, 0.0),
        ]),
        Jones::from([
            Complex::new(1.9986478e0, 0.0),
            Complex::new(-6.9266825e-4, 7.9946185e-6),
            Complex::new(-6.9266825e-4, -7.9946185e-6),
            Complex::new(1.9984438e0, 0.0),
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
            Complex::new(9.9070144e-1, 9.717317e-2),
            Complex::new(-5.068446e-4, -4.277074e-5),
            Complex::new(-5.054955e-4, -5.6524867e-5),
            Complex::new(9.909318e-1, 9.719577e-2),
        ]),
        Jones::from([
            Complex::new(9.942455e-1, 4.9068876e-2),
            Complex::new(-5.0832145e-4, -1.8168646e-5),
            Complex::new(-5.076402e-4, -3.197198e-5),
            Complex::new(9.9447674e-1, 4.9080286e-2),
        ]),
        Jones::from([
            Complex::new(9.94287e-1, -4.822052e-2),
            Complex::new(-5.076673e-4, 3.1538824e-5),
            Complex::new(-5.0833676e-4, 1.7734916e-5),
            Complex::new(9.945183e-1, -4.8231732e-2),
        ]),
    ];
    let result = vis.slice(s![0, ..]);
    assert_abs_diff_eq!(expected_1st_row, result, epsilon = epsilon);

    let expected_2nd_row = array![
        Jones::from([
            Complex::new(2.9622476e0, 3.3937052e-1),
            Complex::new(-1.2422496e-3, -1.4903258e-4),
            Complex::new(-1.243768e-3, -1.3577854e-4),
            Complex::new(2.9632137e0, 3.3948117e-1),
        ]),
        Jones::from([
            Complex::new(2.9766912e0, 1.7144322e-1),
            Complex::new(-1.248686e-3, -7.859979e-5),
            Complex::new(-1.2494532e-3, -6.5281114e-5),
            Complex::new(2.9776618e0, 1.7149913e-1),
        ]),
        Jones::from([
            Complex::new(2.9768605e0, -1.6847998e-1),
            Complex::new(-1.2495174e-3, 6.403731e-5),
            Complex::new(-1.2487636e-3, 7.735674e-5),
            Complex::new(2.9778311e0, -1.6853492e-1),
        ]),
    ];
    let result = vis.slice(s![1, ..]);
    assert_abs_diff_eq!(expected_2nd_row, result, epsilon = epsilon);

    let expected_3rd_row = array![
        Jones::from([
            Complex::new(1.9675331e0, 2.5795707e-1),
            Complex::new(-5.9294904e-4, -6.823252e-5),
            Complex::new(-5.9049827e-4, -8.692553e-5),
            Complex::new(1.9691546e0, 2.5816965e-1),
        ]),
        Jones::from([
            Complex::new(1.9800832e0, 1.3037936e-1),
            Complex::new(-5.961174e-4, -2.980468e-5),
            Complex::new(-5.948787e-4, -4.861693e-5),
            Complex::new(1.9817151e0, 1.3048682e-1),
        ]),
        Jones::from([
            Complex::new(1.9802303e0, -1.2812662e-1),
            Complex::new(-5.9493363e-4, 4.7940128e-5),
            Complex::new(-5.9615093e-4, 2.9126479e-5),
            Complex::new(1.9818622e0, -1.2823221e-1),
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
            Complex::new(9.9957865e-1, 0.0),
            Complex::new(-5.7808764e-4, 6.5317404e-6),
            Complex::new(-5.7808764e-4, -6.5317404e-6),
            Complex::new(9.9954975e-1, 0.0),
        ]),
        Jones::from([
            Complex::new(9.9957865e-1, 0.0),
            Complex::new(-5.7808764e-4, 6.5317404e-6),
            Complex::new(-5.7808764e-4, -6.5317404e-6),
            Complex::new(9.9954975e-1, 0.0),
        ]),
        Jones::from([
            Complex::new(9.9957865e-1, 0.0),
            Complex::new(-5.7808764e-4, 6.5317404e-6),
            Complex::new(-5.7808764e-4, -6.5317404e-6),
            Complex::new(9.9954975e-1, 0.0),
        ]),
    ];
    let result = vis.slice(s![0, ..]);
    assert_abs_diff_eq!(expected_1st_row, result, epsilon = epsilon);

    let expected_2nd_row = array![
        Jones::from([
            Complex::new(8.834807e-1, -2.7755576e-17),
            Complex::new(-4.1814748e-4, -2.306698e-6),
            Complex::new(-4.1814748e-4, 2.306698e-6),
            Complex::new(8.8344485e-1, 2.7755576e-17),
        ]),
        Jones::from([
            Complex::new(8.834807e-1, -2.7755576e-17),
            Complex::new(-4.1814748e-4, -2.306698e-6),
            Complex::new(-4.1814748e-4, 2.306698e-6),
            Complex::new(8.8344485e-1, 2.7755576e-17),
        ]),
        Jones::from([
            Complex::new(8.834807e-1, -2.7755576e-17),
            Complex::new(-4.1814748e-4, -2.306698e-6),
            Complex::new(-4.1814748e-4, 2.306698e-6),
            Complex::new(8.8344485e-1, 2.7755576e-17),
        ]),
    ];
    let result = vis.slice(s![1, ..]);
    assert_abs_diff_eq!(expected_2nd_row, result, epsilon = epsilon);

    let expected_3rd_row = array![
        Jones::from([
            Complex::new(7.9388076e-1, 3.3087225e-24),
            Complex::new(-2.7513402e-4, 3.175534e-6),
            Complex::new(-2.7513402e-4, -3.175534e-6),
            Complex::new(7.9379976e-1, -1.38777845e-17),
        ]),
        Jones::from([
            Complex::new(7.9388076e-1, 3.3087225e-24),
            Complex::new(-2.7513402e-4, 3.175534e-6),
            Complex::new(-2.7513402e-4, -3.175534e-6),
            Complex::new(7.9379976e-1, -1.38777845e-17),
        ]),
        Jones::from([
            Complex::new(7.9388076e-1, 3.3087225e-24),
            Complex::new(-2.7513402e-4, 3.175534e-6),
            Complex::new(-2.7513402e-4, -3.175534e-6),
            Complex::new(7.9379976e-1, -1.38777845e-17),
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
            Complex::new(9.9070144e-1, 9.717317e-2),
            Complex::new(-5.068446e-4, -4.277074e-5),
            Complex::new(-5.054955e-4, -5.6524867e-5),
            Complex::new(9.909318e-1, 9.719577e-2),
        ]),
        Jones::from([
            Complex::new(9.942455e-1, 4.9068876e-2),
            Complex::new(-5.0832145e-4, -1.8168646e-5),
            Complex::new(-5.076402e-4, -3.197198e-5),
            Complex::new(9.9447674e-1, 4.9080286e-2),
        ]),
        Jones::from([
            Complex::new(9.94287e-1, -4.822052e-2),
            Complex::new(-5.076673e-4, 3.1538824e-5),
            Complex::new(-5.0833676e-4, 1.7734916e-5),
            Complex::new(9.945183e-1, -4.8231732e-2),
        ]),
    ];
    let result = vis.slice(s![0, ..]);
    assert_abs_diff_eq!(expected_1st_row, result, epsilon = epsilon);

    let expected_2nd_row = array![
        Jones::from([
            Complex::new(8.7285614e-1, 9.999895e-2),
            Complex::new(-3.6604138e-4, -4.3913955e-5),
            Complex::new(-3.664888e-4, -4.0008515e-5),
            Complex::new(8.7314075e-1, 1.00031555e-1),
        ]),
        Jones::from([
            Complex::new(8.771121e-1, 5.0517477e-2),
            Complex::new(-3.6793793e-4, -2.316022e-5),
            Complex::new(-3.6816395e-4, -1.9235738e-5),
            Complex::new(8.7739813e-1, 5.0533947e-2),
        ]),
        Jones::from([
            Complex::new(8.77162e-1, -4.9644325e-2),
            Complex::new(-3.6818293e-4, 1.886924e-5),
            Complex::new(-3.679608e-4, 2.2793944e-5),
            Complex::new(8.7744796e-1, -4.966051e-2),
        ]),
    ];
    let result = vis.slice(s![1, ..]);
    assert_abs_diff_eq!(expected_2nd_row, result, epsilon = epsilon);

    let expected_3rd_row = array![
        Jones::from([
            Complex::new(7.8152174e-1, 1.0246286e-1),
            Complex::new(-2.3552467e-4, -2.7102566e-5),
            Complex::new(-2.3455119e-4, -3.45276e-5),
            Complex::new(7.821658e-1, 1.02547295e-1),
        ]),
        Jones::from([
            Complex::new(7.865068e-1, 5.1787853e-2),
            Complex::new(-2.3678315e-4, -1.1838685e-5),
            Complex::new(-2.3629113e-4, -1.9311079e-5),
            Complex::new(7.871549e-1, 5.183053e-2),
        ]),
        Jones::from([
            Complex::new(7.865652e-1, -5.089304e-2),
            Complex::new(-2.3631295e-4, 1.9042247e-5),
            Complex::new(-2.3679648e-4, 1.1569298e-5),
            Complex::new(7.872134e-1, -5.0934978e-2),
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
            Complex::new(9.9957865e-1, 0.0),
            Complex::new(-5.7808764e-4, 6.5317404e-6),
            Complex::new(-5.7808764e-4, -6.5317404e-6),
            Complex::new(9.9954975e-1, 0.0),
        ]),
        Jones::from([
            Complex::new(9.9957865e-1, 0.0),
            Complex::new(-5.7808764e-4, 6.5317404e-6),
            Complex::new(-5.7808764e-4, -6.5317404e-6),
            Complex::new(9.9954975e-1, 0.0),
        ]),
        Jones::from([
            Complex::new(9.9957865e-1, 0.0),
            Complex::new(-5.7808764e-4, 6.5317404e-6),
            Complex::new(-5.7808764e-4, -6.5317404e-6),
            Complex::new(9.9954975e-1, 0.0),
        ]),
    ];
    let result = vis.slice(s![0, ..]);
    assert_abs_diff_eq!(expected_1st_row, result, epsilon = epsilon);

    let expected_2nd_row = array![
        Jones::from([
            Complex::new(8.8411075e-1, 2.7755576e-17),
            Complex::new(-4.1844568e-4, -2.308343e-6),
            Complex::new(-4.1844568e-4, 2.308343e-6),
            Complex::new(8.8407487e-1, -2.7755576e-17),
        ]),
        Jones::from([
            Complex::new(8.8411075e-1, 2.7755576e-17),
            Complex::new(-4.1844568e-4, -2.308343e-6),
            Complex::new(-4.1844568e-4, 2.308343e-6),
            Complex::new(8.8407487e-1, -2.7755576e-17),
        ]),
        Jones::from([
            Complex::new(8.8411075e-1, 2.7755576e-17),
            Complex::new(-4.1844568e-4, -2.308343e-6),
            Complex::new(-4.1844568e-4, 2.308343e-6),
            Complex::new(8.8407487e-1, -2.7755576e-17),
        ]),
    ];
    let result = vis.slice(s![1, ..]);
    assert_abs_diff_eq!(expected_2nd_row, result, epsilon = epsilon);

    let expected_3rd_row = array![
        Jones::from([
            Complex::new(7.958543e-1, -6.938894e-18),
            Complex::new(-2.75818e-4, 3.1834281e-6),
            Complex::new(-2.75818e-4, -3.1834281e-6),
            Complex::new(7.957731e-1, -3.3087225e-24),
        ]),
        Jones::from([
            Complex::new(7.958543e-1, -6.938894e-18),
            Complex::new(-2.75818e-4, 3.1834281e-6),
            Complex::new(-2.75818e-4, -3.1834281e-6),
            Complex::new(7.957731e-1, -3.3087225e-24),
        ]),
        Jones::from([
            Complex::new(7.958543e-1, -6.938894e-18),
            Complex::new(-2.75818e-4, 3.1834281e-6),
            Complex::new(-2.75818e-4, -3.1834281e-6),
            Complex::new(7.957731e-1, -3.3087225e-24),
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
            Complex::new(9.9070144e-1, 9.717317e-2),
            Complex::new(-5.068446e-4, -4.277074e-5),
            Complex::new(-5.054955e-4, -5.6524867e-5),
            Complex::new(9.909318e-1, 9.719577e-2),
        ]),
        Jones::from([
            Complex::new(9.942455e-1, 4.9068876e-2),
            Complex::new(-5.0832145e-4, -1.8168646e-5),
            Complex::new(-5.076402e-4, -3.197198e-5),
            Complex::new(9.9447674e-1, 4.9080286e-2),
        ]),
        Jones::from([
            Complex::new(9.94287e-1, -4.822052e-2),
            Complex::new(-5.076673e-4, 3.1538824e-5),
            Complex::new(-5.0833676e-4, 1.7734916e-5),
            Complex::new(9.945183e-1, -4.8231732e-2),
        ]),
    ];
    let result = vis.slice(s![0, ..]);
    assert_abs_diff_eq!(expected_1st, result, epsilon = epsilon);

    let expected_2nd_row = array![
        Jones::from([
            Complex::new(8.734786e-1, 1.0007026e-1),
            Complex::new(-3.663024e-4, -4.394527e-5),
            Complex::new(-3.6675014e-4, -4.0037048e-5),
            Complex::new(8.7376344e-1, 1.00102894e-1),
        ]),
        Jones::from([
            Complex::new(8.777376e-1, 5.05535e-2),
            Complex::new(-3.6820033e-4, -2.3176737e-5),
            Complex::new(-3.6842652e-4, -1.9249457e-5),
            Complex::new(8.780238e-1, 5.0569985e-2),
        ]),
        Jones::from([
            Complex::new(8.777875e-1, -4.9679726e-2),
            Complex::new(-3.684455e-4, 1.8882694e-5),
            Complex::new(-3.682232e-4, 2.2810198e-5),
            Complex::new(8.780737e-1, -4.9695928e-2),
        ]),
    ];
    let result = vis.slice(s![1, ..]);
    assert_abs_diff_eq!(expected_2nd_row, result, epsilon = epsilon);

    let expected_3rd_row = array![
        Jones::from([
            Complex::new(7.8346455e-1, 1.0271757e-1),
            Complex::new(-2.3611017e-4, -2.7169941e-5),
            Complex::new(-2.3513427e-4, -3.4613433e-5),
            Complex::new(7.841102e-1, 1.0280222e-1),
        ]),
        Jones::from([
            Complex::new(7.88462e-1, 5.191659e-2),
            Complex::new(-2.3737177e-4, -1.18681155e-5),
            Complex::new(-2.3687854e-4, -1.9359084e-5),
            Complex::new(7.8911173e-1, 5.1959377e-2),
        ]),
        Jones::from([
            Complex::new(7.885205e-1, -5.1019553e-2),
            Complex::new(-2.369004e-4, 1.9089584e-5),
            Complex::new(-2.3738513e-4, 1.1598058e-5),
            Complex::new(7.891703e-1, -5.10616e-2),
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
            Complex::new(3.4502387e-1, 3.3841744e-2),
            Complex::new(-1.7651482e-4, -1.4895431e-5),
            Complex::new(-1.7604498e-4, -1.9685476e-5),
            Complex::new(3.451041e-1, 3.3849612e-2),
        ]),
        Jones::from([
            Complex::new(7.413815e-1, 3.658931e-2),
            Complex::new(-3.790413e-4, -1.3547859e-5),
            Complex::new(-3.7853332e-4, -2.3840625e-5),
            Complex::new(7.415539e-1, 3.6597818e-2),
        ]),
        Jones::from([
            Complex::new(5.5424654e-1, -2.6879618e-2),
            Complex::new(-2.8298955e-4, 1.7580724e-5),
            Complex::new(-2.8336272e-4, 9.885995e-6),
            Complex::new(5.5437547e-1, -2.6885869e-2),
        ]),
    ];
    let result = vis.slice(s![0, ..]);
    assert_abs_diff_eq!(expected_1st, result, epsilon = epsilon);

    let expected_2nd_row = array![
        Jones::from([
            Complex::new(7.048635e-1, 8.0752835e-2),
            Complex::new(-2.955919e-4, -3.5462137e-5),
            Complex::new(-2.9595318e-4, -3.230835e-5),
            Complex::new(7.050933e-1, 8.0779165e-2),
        ]),
        Jones::from([
            Complex::new(1.9964461e0, 1.1498578e-1),
            Complex::new(-8.374851e-4, -5.2716332e-5),
            Complex::new(-8.379995e-4, -4.378359e-5),
            Complex::new(1.9970971e0, 1.1502327e-1),
        ]),
        Jones::from([
            Complex::new(1.343682e0, -7.604775e-2),
            Complex::new(-5.640017e-4, 2.8904877e-5),
            Complex::new(-5.636614e-4, 3.4916942e-5),
            Complex::new(1.3441203e0, -7.6072544e-2),
        ]),
    ];
    let result = vis.slice(s![1, ..]);
    assert_abs_diff_eq!(expected_2nd_row, result, epsilon = epsilon);

    let expected_3rd_row = array![
        Jones::from([
            Complex::new(3.0167085e-1, 3.9551117e-2),
            Complex::new(-9.091356e-5, -1.046171e-5),
            Complex::new(-9.05378e-5, -1.3327805e-5),
            Complex::new(3.0191946e-1, 3.958371e-2),
        ]),
        Jones::from([
            Complex::new(1.1751744e0, 7.737982e-2),
            Complex::new(-3.5379414e-4, -1.7689e-5),
            Complex::new(-3.5305897e-4, -2.8854023e-5),
            Complex::new(1.1761428e0, 7.744359e-2),
        ]),
        Jones::from([
            Complex::new(7.0064837e-1, -4.533397e-2),
            Complex::new(-2.105004e-4, 1.6962254e-5),
            Complex::new(-2.1093109e-4, 1.0305579e-5),
            Complex::new(7.0122576e-1, -4.537133e-2),
        ]),
    ];
    let result = vis.slice(s![2, ..]);
    assert_abs_diff_eq!(expected_3rd_row, result, epsilon = epsilon);
}

#[track_caller]
fn test_non_trivial_gaussian_power_law(vis: ArrayView2<Jones<f32>>, epsilon: f32) {
    let expected_1st = array![
        Jones::from([
            Complex::new(3.4502387e-1, 3.3841744e-2),
            Complex::new(-1.7651482e-4, -1.4895431e-5),
            Complex::new(-1.7604498e-4, -1.9685476e-5),
            Complex::new(3.451041e-1, 3.3849612e-2),
        ]),
        Jones::from([
            Complex::new(7.413815e-1, 3.658931e-2),
            Complex::new(-3.790413e-4, -1.3547859e-5),
            Complex::new(-3.7853332e-4, -2.3840625e-5),
            Complex::new(7.415539e-1, 3.6597818e-2),
        ]),
        Jones::from([
            Complex::new(5.5424654e-1, -2.6879618e-2),
            Complex::new(-2.8298955e-4, 1.7580724e-5),
            Complex::new(-2.8336272e-4, 9.885995e-6),
            Complex::new(5.5437547e-1, -2.6885869e-2),
        ]),
    ];
    let result = vis.slice(s![0, ..]);
    assert_abs_diff_eq!(expected_1st, result, epsilon = epsilon);

    let expected_2nd_row = array![
        Jones::from([
            Complex::new(2.0769514e-1, 2.3794636e-2),
            Complex::new(-8.709913e-5, -1.04492765e-5),
            Complex::new(-8.72056e-5, -9.519981e-6),
            Complex::new(2.0776285e-1, 2.3802396e-2),
        ]),
        Jones::from([
            Complex::new(5.88273e-1, 3.388172e-2),
            Complex::new(-2.4677342e-4, -1.5533398e-5),
            Complex::new(-2.4692502e-4, -1.2901277e-5),
            Complex::new(5.884648e-1, 3.389277e-2),
        ]),
        Jones::from([
            Complex::new(3.959295e-1, -2.2408236e-2),
            Complex::new(-1.6618878e-4, 8.517113e-6),
            Complex::new(-1.6608852e-4, 1.0288629e-5),
            Complex::new(3.960586e-1, -2.2415543e-2),
        ]),
    ];
    let result = vis.slice(s![1, ..]);
    assert_abs_diff_eq!(expected_2nd_row, result, epsilon = epsilon);

    let expected_3rd_row = array![
        Jones::from([
            Complex::new(1.1982636e-1, 1.5710056e-2),
            Complex::new(-3.611168e-5, -4.155485e-6),
            Complex::new(-3.596242e-5, -5.2939236e-6),
            Complex::new(1.1992511e-1, 1.5723003e-2),
        ]),
        Jones::from([
            Complex::new(4.6678978e-1, 3.0735958e-2),
            Complex::new(-1.405302e-4, -7.0262295e-6),
            Complex::new(-1.4023819e-4, -1.1461076e-5),
            Complex::new(4.6717444e-1, 3.0761287e-2),
        ]),
        Jones::from([
            Complex::new(2.783038e-1, -1.8007059e-2),
            Complex::new(-8.3612635e-5, 6.737559e-6),
            Complex::new(-8.3783714e-5, 4.093468e-6),
            Complex::new(2.7853313e-1, -1.8021898e-2),
        ]),
    ];
    let result = vis.slice(s![2, ..]);
    assert_abs_diff_eq!(expected_3rd_row, result, epsilon = epsilon);
}

#[track_caller]
fn test_non_trivial_gaussian_curved_power_law(vis: ArrayView2<Jones<f32>>, epsilon: f32) {
    let expected_1st = array![
        Jones::from([
            Complex::new(3.4502387e-1, 3.3841744e-2),
            Complex::new(-1.7651482e-4, -1.4895431e-5),
            Complex::new(-1.7604498e-4, -1.9685476e-5),
            Complex::new(3.451041e-1, 3.3849612e-2),
        ]),
        Jones::from([
            Complex::new(7.413815e-1, 3.658931e-2),
            Complex::new(-3.790413e-4, -1.3547859e-5),
            Complex::new(-3.7853332e-4, -2.3840625e-5),
            Complex::new(7.415539e-1, 3.6597818e-2),
        ]),
        Jones::from([
            Complex::new(5.5424654e-1, -2.6879618e-2),
            Complex::new(-2.8298955e-4, 1.7580724e-5),
            Complex::new(-2.8336272e-4, 9.885995e-6),
            Complex::new(5.5437547e-1, -2.6885869e-2),
        ]),
    ];
    let result = vis.slice(s![0, ..]);
    assert_abs_diff_eq!(expected_1st, result, epsilon = epsilon);

    let expected_2nd_row = array![
        Jones::from([
            Complex::new(2.0784324e-1, 2.3811605e-2),
            Complex::new(-8.716124e-5, -1.0456728e-5),
            Complex::new(-8.7267785e-5, -9.526771e-6),
            Complex::new(2.0791101e-1, 2.381937e-2),
        ]),
        Jones::from([
            Complex::new(5.886925e-1, 3.3905882e-2),
            Complex::new(-2.469494e-4, -1.5544476e-5),
            Complex::new(-2.471011e-4, -1.29104765e-5),
            Complex::new(5.888845e-1, 3.391694e-2),
        ]),
        Jones::from([
            Complex::new(3.9621183e-1, -2.2424215e-2),
            Complex::new(-1.6630729e-4, 8.523187e-6),
            Complex::new(-1.6620696e-4, 1.0295967e-5),
            Complex::new(3.9634103e-1, -2.2431528e-2),
        ]),
    ];
    let result = vis.slice(s![1, ..]);
    assert_abs_diff_eq!(expected_2nd_row, result, epsilon = epsilon);

    let expected_3rd_row = array![
        Jones::from([
            Complex::new(1.20124236e-1, 1.574911e-2),
            Complex::new(-3.620145e-5, -4.165815e-6),
            Complex::new(-3.605182e-5, -5.307084e-6),
            Complex::new(1.2022323e-1, 1.5762089e-2),
        ]),
        Jones::from([
            Complex::new(4.6795017e-1, 3.0812364e-2),
            Complex::new(-1.4087955e-4, -7.043696e-6),
            Complex::new(-1.4058681e-4, -1.1489567e-5),
            Complex::new(4.683358e-1, 3.0837758e-2),
        ]),
        Jones::from([
            Complex::new(2.7899563e-1, -1.8051822e-2),
            Complex::new(-8.382049e-5, 6.7543083e-6),
            Complex::new(-8.3991996e-5, 4.103644e-6),
            Complex::new(2.7922556e-1, -1.8066699e-2),
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
