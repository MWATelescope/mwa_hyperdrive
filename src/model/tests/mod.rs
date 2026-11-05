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
            Complex::new(9.995392e-1, 0.0),
            Complex::new(-5.4398115e-4, 4.531473e-6),
            Complex::new(-5.4398115e-4, -4.531473e-6),
            Complex::new(9.995102e-1, 0.0),
        ]),
        Jones::from([
            Complex::new(9.995392e-1, 0.0),
            Complex::new(-5.4398115e-4, 4.531473e-6),
            Complex::new(-5.4398115e-4, -4.531473e-6),
            Complex::new(9.995102e-1, 0.0),
        ]),
        Jones::from([
            Complex::new(9.995392e-1, 0.0),
            Complex::new(-5.4398115e-4, 4.531473e-6),
            Complex::new(-5.4398115e-4, -4.531473e-6),
            Complex::new(9.995102e-1, 0.0),
        ]),
    ];
    let result = vis.slice(s![0, ..]);
    assert_abs_diff_eq!(expected_1st_row, result, epsilon = epsilon);
    // ~3 Jy XX and YY.
    let expected_2nd_row = array![
        Jones::from([
            Complex::new(2.9982355e0, 0.0),
            Complex::new(-1.4275565e-3, -5.36277e-6),
            Complex::new(-1.4275565e-3, 5.36277e-6),
            Complex::new(2.9981132e0, 1.11022296e-16),
        ]),
        Jones::from([
            Complex::new(2.9982355e0, 0.0),
            Complex::new(-1.4275565e-3, -5.36277e-6),
            Complex::new(-1.4275565e-3, 5.36277e-6),
            Complex::new(2.9981132e0, 1.11022296e-16),
        ]),
        Jones::from([
            Complex::new(2.9982355e0, 0.0),
            Complex::new(-1.4275565e-3, -5.36277e-6),
            Complex::new(-1.4275565e-3, 5.36277e-6),
            Complex::new(2.9981132e0, 1.11022296e-16),
        ]),
    ];
    let result = vis.slice(s![1, ..]);
    assert_abs_diff_eq!(expected_2nd_row, result, epsilon = epsilon);
    // ~2 Jy XX and YY.
    let expected_3rd_row = array![
        Jones::from([
            Complex::new(1.9984084e0, 0.0),
            Complex::new(-6.94998e-4, 6.7894125e-6),
            Complex::new(-6.94998e-4, -6.7894125e-6),
            Complex::new(1.9982046e0, 0.0),
        ]),
        Jones::from([
            Complex::new(1.9984084e0, 0.0),
            Complex::new(-6.94998e-4, 6.7894125e-6),
            Complex::new(-6.94998e-4, -6.7894125e-6),
            Complex::new(1.9982046e0, 0.0),
        ]),
        Jones::from([
            Complex::new(1.9984084e0, 0.0),
            Complex::new(-6.94998e-4, 6.7894125e-6),
            Complex::new(-6.94998e-4, -6.7894125e-6),
            Complex::new(1.9982046e0, 0.0),
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
            Complex::new(9.906622e-1, 9.7169325e-2),
            Complex::new(-4.7287444e-4, -4.1443327e-5),
            Complex::new(-4.7191486e-4, -5.1226547e-5),
            Complex::new(9.9089247e-1, 9.7191915e-2),
        ]),
        Jones::from([
            Complex::new(9.942062e-1, 4.906693e-2),
            Complex::new(-4.7432687e-4, -1.8488328e-5),
            Complex::new(-4.7384232e-4, -2.8306546e-5),
            Complex::new(9.944373e-1, 4.9078338e-2),
        ]),
        Jones::from([
            Complex::new(9.942477e-1, -4.821861e-2),
            Complex::new(-4.7386627e-4, 2.7902232e-5),
            Complex::new(-4.7434247e-4, 1.8083603e-5),
            Complex::new(9.944788e-1, -4.822982e-2),
        ]),
    ];
    let result = vis.slice(s![0, ..]);
    assert_abs_diff_eq!(expected_1st_row, result, epsilon = epsilon);

    let expected_2nd_row = array![
        Jones::from([
            Complex::new(2.9621787e0, 3.3936265e-1),
            Complex::new(-1.2509684e-3, -1.476209e-4),
            Complex::new(-1.2519417e-3, -1.3912552e-4),
            Complex::new(2.963145e0, 3.3947334e-1),
        ]),
        Jones::from([
            Complex::new(2.976622e0, 1.7143923e-1),
            Complex::new(-1.2573112e-3, -7.669769e-5),
            Complex::new(-1.2578028e-3, -6.816089e-5),
            Complex::new(2.977593e0, 1.7149515e-1),
        ]),
        Jones::from([
            Complex::new(2.9767911e0, -1.6847606e-1),
            Complex::new(-1.2578701e-3, 6.690877e-5),
            Complex::new(-1.2573869e-3, 7.5446056e-5),
            Complex::new(2.9777622e0, -1.6853102e-1),
        ]),
    ];
    let result = vis.slice(s![1, ..]);
    assert_abs_diff_eq!(expected_2nd_row, result, epsilon = epsilon);

    let expected_3rd_row = array![
        Jones::from([
            Complex::new(1.967298e0, 2.5792626e-1),
            Complex::new(-5.9508363e-4, -7.013924e-5),
            Complex::new(-5.930522e-4, -8.56335e-5),
            Complex::new(1.9689182e0, 2.5813866e-1),
        ]),
        Jones::from([
            Complex::new(1.9798467e0, 1.3036379e-1),
            Complex::new(-5.983706e-4, -3.1569605e-5),
            Complex::new(-5.973439e-4, -4.716269e-5),
            Complex::new(1.9814771e0, 1.3047114e-1),
        ]),
        Jones::from([
            Complex::new(1.9799937e0, -1.281113e-1),
            Complex::new(-5.9739716e-4, 4.6483085e-5),
            Complex::new(-5.984061e-4, 3.0888838e-5),
            Complex::new(1.9816242e0, -1.282168e-1),
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
            Complex::new(9.995392e-1, 0.0),
            Complex::new(-5.4398115e-4, 4.531473e-6),
            Complex::new(-5.4398115e-4, -4.531473e-6),
            Complex::new(9.995102e-1, 0.0),
        ]),
        Jones::from([
            Complex::new(9.995392e-1, 0.0),
            Complex::new(-5.4398115e-4, 4.531473e-6),
            Complex::new(-5.4398115e-4, -4.531473e-6),
            Complex::new(9.995102e-1, 0.0),
        ]),
        Jones::from([
            Complex::new(9.995392e-1, 0.0),
            Complex::new(-5.4398115e-4, 4.531473e-6),
            Complex::new(-5.4398115e-4, -4.531473e-6),
            Complex::new(9.995102e-1, 0.0),
        ]),
    ];
    let result = vis.slice(s![0, ..]);
    assert_abs_diff_eq!(expected_1st_row, result, epsilon = epsilon);

    let expected_2nd_row = array![
        Jones::from([
            Complex::new(8.8346034e-1, 0.0),
            Complex::new(-4.2064392e-4, -1.5801943e-6),
            Complex::new(-4.2064392e-4, 1.5801943e-6),
            Complex::new(8.834243e-1, 0.0),
        ]),
        Jones::from([
            Complex::new(8.8346034e-1, 0.0),
            Complex::new(-4.2064392e-4, -1.5801943e-6),
            Complex::new(-4.2064392e-4, 1.5801943e-6),
            Complex::new(8.834243e-1, 0.0),
        ]),
        Jones::from([
            Complex::new(8.8346034e-1, 0.0),
            Complex::new(-4.2064392e-4, -1.5801943e-6),
            Complex::new(-4.2064392e-4, 1.5801943e-6),
            Complex::new(8.834243e-1, 0.0),
        ]),
    ];
    let result = vis.slice(s![1, ..]);
    assert_abs_diff_eq!(expected_2nd_row, result, epsilon = epsilon);

    let expected_3rd_row = array![
        Jones::from([
            Complex::new(7.937857e-1, -3.3087225e-24),
            Complex::new(-2.760594e-4, 2.6968153e-6),
            Complex::new(-2.760594e-4, -2.6968153e-6),
            Complex::new(7.9370475e-1, 3.3087225e-24),
        ]),
        Jones::from([
            Complex::new(7.937857e-1, -3.3087225e-24),
            Complex::new(-2.760594e-4, 2.6968153e-6),
            Complex::new(-2.760594e-4, -2.6968153e-6),
            Complex::new(7.9370475e-1, 3.3087225e-24),
        ]),
        Jones::from([
            Complex::new(7.937857e-1, -3.3087225e-24),
            Complex::new(-2.760594e-4, 2.6968153e-6),
            Complex::new(-2.760594e-4, -2.6968153e-6),
            Complex::new(7.9370475e-1, 3.3087225e-24),
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
            Complex::new(9.906622e-1, 9.7169325e-2),
            Complex::new(-4.7287444e-4, -4.1443327e-5),
            Complex::new(-4.7191486e-4, -5.1226547e-5),
            Complex::new(9.9089247e-1, 9.7191915e-2),
        ]),
        Jones::from([
            Complex::new(9.942062e-1, 4.906693e-2),
            Complex::new(-4.7432687e-4, -1.8488328e-5),
            Complex::new(-4.7384232e-4, -2.8306546e-5),
            Complex::new(9.944373e-1, 4.9078338e-2),
        ]),
        Jones::from([
            Complex::new(9.942477e-1, -4.821861e-2),
            Complex::new(-4.7386627e-4, 2.7902232e-5),
            Complex::new(-4.7434247e-4, 1.8083603e-5),
            Complex::new(9.944788e-1, -4.822982e-2),
        ]),
    ];
    let result = vis.slice(s![0, ..]);
    assert_abs_diff_eq!(expected_1st_row, result, epsilon = epsilon);

    let expected_2nd_row = array![
        Jones::from([
            Complex::new(8.728359e-1, 9.999663e-2),
            Complex::new(-3.6861046e-4, -4.349799e-5),
            Complex::new(-3.6889725e-4, -4.099474e-5),
            Complex::new(8.7312055e-1, 1.0002924e-1),
        ]),
        Jones::from([
            Complex::new(8.770917e-1, 5.05163e-2),
            Complex::new(-3.7047942e-4, -2.2599748e-5),
            Complex::new(-3.706243e-4, -2.0084293e-5),
            Complex::new(8.773778e-1, 5.0532777e-2),
        ]),
        Jones::from([
            Complex::new(8.771416e-1, -4.964317e-2),
            Complex::new(-3.7064412e-4, 1.9715344e-5),
            Complex::new(-3.7050174e-4, 2.223094e-5),
            Complex::new(8.7742764e-1, -4.9659364e-2),
        ]),
    ];
    let result = vis.slice(s![1, ..]);
    assert_abs_diff_eq!(expected_2nd_row, result, epsilon = epsilon);

    let expected_3rd_row = array![
        Jones::from([
            Complex::new(7.814284e-1, 1.0245062e-1),
            Complex::new(-2.3637254e-4, -2.7859935e-5),
            Complex::new(-2.3556563e-4, -3.4014392e-5),
            Complex::new(7.820719e-1, 1.0253498e-1),
        ]),
        Jones::from([
            Complex::new(7.8641284e-1, 5.178166e-2),
            Complex::new(-2.3767815e-4, -1.2539729e-5),
            Complex::new(-2.3727033e-4, -1.8733443e-5),
            Complex::new(7.8706044e-1, 5.1824305e-2),
        ]),
        Jones::from([
            Complex::new(7.864712e-1, -5.088696e-2),
            Complex::new(-2.3729149e-4, 1.8463497e-5),
            Complex::new(-2.3769226e-4, 1.2269323e-5),
            Complex::new(7.8711885e-1, -5.0928865e-2),
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
            Complex::new(9.995392e-1, 0.0),
            Complex::new(-5.4398115e-4, 4.531473e-6),
            Complex::new(-5.4398115e-4, -4.531473e-6),
            Complex::new(9.995102e-1, 0.0),
        ]),
        Jones::from([
            Complex::new(9.995392e-1, 0.0),
            Complex::new(-5.4398115e-4, 4.531473e-6),
            Complex::new(-5.4398115e-4, -4.531473e-6),
            Complex::new(9.995102e-1, 0.0),
        ]),
        Jones::from([
            Complex::new(9.995392e-1, 0.0),
            Complex::new(-5.4398115e-4, 4.531473e-6),
            Complex::new(-5.4398115e-4, -4.531473e-6),
            Complex::new(9.995102e-1, 0.0),
        ]),
    ];
    let result = vis.slice(s![0, ..]);
    assert_abs_diff_eq!(expected_1st_row, result, epsilon = epsilon);

    let expected_2nd_row = array![
        Jones::from([
            Complex::new(8.8409036e-1, 0.0),
            Complex::new(-4.209439e-4, -1.5813213e-6),
            Complex::new(-4.209439e-4, 1.5813213e-6),
            Complex::new(8.8405424e-1, -2.7755576e-17),
        ]),
        Jones::from([
            Complex::new(8.8409036e-1, 0.0),
            Complex::new(-4.209439e-4, -1.5813213e-6),
            Complex::new(-4.209439e-4, 1.5813213e-6),
            Complex::new(8.8405424e-1, -2.7755576e-17),
        ]),
        Jones::from([
            Complex::new(8.8409036e-1, 0.0),
            Complex::new(-4.209439e-4, -1.5813213e-6),
            Complex::new(-4.209439e-4, 1.5813213e-6),
            Complex::new(8.8405424e-1, -2.7755576e-17),
        ]),
    ];
    let result = vis.slice(s![1, ..]);
    assert_abs_diff_eq!(expected_2nd_row, result, epsilon = epsilon);

    let expected_3rd_row = array![
        Jones::from([
            Complex::new(7.9575896e-1, 3.3087225e-24),
            Complex::new(-2.7674568e-4, 2.7035194e-6),
            Complex::new(-2.7674568e-4, -2.7035194e-6),
            Complex::new(7.9567784e-1, 3.3087225e-24),
        ]),
        Jones::from([
            Complex::new(7.9575896e-1, 3.3087225e-24),
            Complex::new(-2.7674568e-4, 2.7035194e-6),
            Complex::new(-2.7674568e-4, -2.7035194e-6),
            Complex::new(7.9567784e-1, 3.3087225e-24),
        ]),
        Jones::from([
            Complex::new(7.9575896e-1, 3.3087225e-24),
            Complex::new(-2.7674568e-4, 2.7035194e-6),
            Complex::new(-2.7674568e-4, -2.7035194e-6),
            Complex::new(7.9567784e-1, 3.3087225e-24),
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
            Complex::new(9.906622e-1, 9.7169325e-2),
            Complex::new(-4.7287444e-4, -4.1443327e-5),
            Complex::new(-4.7191486e-4, -5.1226547e-5),
            Complex::new(9.9089247e-1, 9.7191915e-2),
        ]),
        Jones::from([
            Complex::new(9.942062e-1, 4.906693e-2),
            Complex::new(-4.7432687e-4, -1.8488328e-5),
            Complex::new(-4.7384232e-4, -2.8306546e-5),
            Complex::new(9.944373e-1, 4.9078338e-2),
        ]),
        Jones::from([
            Complex::new(9.942477e-1, -4.821861e-2),
            Complex::new(-4.7386627e-4, 2.7902232e-5),
            Complex::new(-4.7434247e-4, 1.8083603e-5),
            Complex::new(9.944788e-1, -4.822982e-2),
        ]),
    ];
    let result = vis.slice(s![0, ..]);
    assert_abs_diff_eq!(expected_1st, result, epsilon = epsilon);

    let expected_2nd_row = array![
        Jones::from([
            Complex::new(8.734583e-1, 1.00067936e-1),
            Complex::new(-3.6887333e-4, -4.352901e-5),
            Complex::new(-3.6916032e-4, -4.1023974e-5),
            Complex::new(8.737432e-1, 1.0010058e-1),
        ]),
        Jones::from([
            Complex::new(8.777172e-1, 5.0552323e-2),
            Complex::new(-3.7074363e-4, -2.2615865e-5),
            Complex::new(-3.708886e-4, -2.0098616e-5),
            Complex::new(8.780035e-1, 5.0568815e-2),
        ]),
        Jones::from([
            Complex::new(8.777671e-1, -4.967857e-2),
            Complex::new(-3.7090844e-4, 1.9729403e-5),
            Complex::new(-3.7076595e-4, 2.2246795e-5),
            Complex::new(8.7805337e-1, -4.9694777e-2),
        ]),
    ];
    let result = vis.slice(s![1, ..]);
    assert_abs_diff_eq!(expected_2nd_row, result, epsilon = epsilon);

    let expected_3rd_row = array![
        Jones::from([
            Complex::new(7.83371e-1, 1.027053e-1),
            Complex::new(-2.3696013e-4, -2.7929193e-5),
            Complex::new(-2.3615123e-4, -3.409895e-5),
            Complex::new(7.840161e-1, 1.0278988e-1),
        ]),
        Jones::from([
            Complex::new(7.8836775e-1, 5.191039e-2),
            Complex::new(-2.38269e-4, -1.2570901e-5),
            Complex::new(-2.3786016e-4, -1.8780012e-5),
            Complex::new(7.8901696e-1, 5.1953137e-2),
        ]),
        Jones::from([
            Complex::new(7.884263e-1, -5.101346e-2),
            Complex::new(-2.3788137e-4, 1.8509396e-5),
            Complex::new(-2.3828314e-4, 1.22998235e-5),
            Complex::new(7.8907555e-1, -5.105547e-2),
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
            Complex::new(3.4501022e-1, 3.3840403e-2),
            Complex::new(-1.646843e-4, -1.4433144e-5),
            Complex::new(-1.6435012e-4, -1.784027e-5),
            Complex::new(3.4509042e-1, 3.384827e-2),
        ]),
        Jones::from([
            Complex::new(7.4135214e-1, 3.658786e-2),
            Complex::new(-3.5369248e-4, -1.3786237e-5),
            Complex::new(-3.5333115e-4, -2.1107411e-5),
            Complex::new(7.415245e-1, 3.6596365e-2),
        ]),
        Jones::from([
            Complex::new(5.542246e-1, -2.6878554e-2),
            Complex::new(-2.6414782e-4, 1.5553573e-5),
            Complex::new(-2.6441325e-4, 1.0080363e-5),
            Complex::new(5.543535e-1, -2.6884804e-2),
        ]),
    ];
    let result = vis.slice(s![0, ..]);
    assert_abs_diff_eq!(expected_1st, result, epsilon = epsilon);

    let expected_2nd_row = array![
        Jones::from([
            Complex::new(7.048471e-1, 8.075096e-2),
            Complex::new(-2.9766653e-4, -3.5126228e-5),
            Complex::new(-2.978981e-4, -3.310476e-5),
            Complex::new(7.05077e-1, 8.0777295e-2),
        ]),
        Jones::from([
            Complex::new(1.9963998e0, 1.14983104e-1),
            Complex::new(-8.4326987e-4, -5.144061e-5),
            Complex::new(-8.435997e-4, -4.5715035e-5),
            Complex::new(1.9970509e0, 1.1502061e-1),
        ]),
        Jones::from([
            Complex::new(1.3436508e0, -7.6045975e-2),
            Complex::new(-5.6777184e-4, 3.0200983e-5),
            Complex::new(-5.6755374e-4, 3.4054505e-5),
            Complex::new(1.3440892e0, -7.607078e-2),
        ]),
    ];
    let result = vis.slice(s![1, ..]);
    assert_abs_diff_eq!(expected_2nd_row, result, epsilon = epsilon);

    let expected_3rd_row = array![
        Jones::from([
            Complex::new(3.0163482e-1, 3.954639e-2),
            Complex::new(-9.124084e-5, -1.0754058e-5),
            Complex::new(-9.0929374e-5, -1.3129706e-5),
            Complex::new(3.0188322e-1, 3.9578956e-2),
        ]),
        Jones::from([
            Complex::new(1.1750339e0, 7.737058e-2),
            Complex::new(-3.5513143e-4, -1.8736479e-5),
            Complex::new(-3.5452205e-4, -2.7990936e-5),
            Complex::new(1.1760015e0, 7.7434294e-2),
        ]),
        Jones::from([
            Complex::new(7.005646e-1, -4.5328554e-2),
            Complex::new(-2.1137204e-4, 1.6446722e-5),
            Complex::new(-2.1172904e-4, 1.09291395e-5),
            Complex::new(7.0114154e-1, -4.536588e-2),
        ]),
    ];
    let result = vis.slice(s![2, ..]);
    assert_abs_diff_eq!(expected_3rd_row, result, epsilon = epsilon);
}

#[track_caller]
fn test_non_trivial_gaussian_power_law(vis: ArrayView2<Jones<f32>>, epsilon: f32) {
    let expected_1st = array![
        Jones::from([
            Complex::new(3.4501022e-1, 3.3840403e-2),
            Complex::new(-1.646843e-4, -1.4433144e-5),
            Complex::new(-1.6435012e-4, -1.784027e-5),
            Complex::new(3.4509042e-1, 3.384827e-2),
        ]),
        Jones::from([
            Complex::new(7.4135214e-1, 3.658786e-2),
            Complex::new(-3.5369248e-4, -1.3786237e-5),
            Complex::new(-3.5333115e-4, -2.1107411e-5),
            Complex::new(7.415245e-1, 3.6596365e-2),
        ]),
        Jones::from([
            Complex::new(5.542246e-1, -2.6878554e-2),
            Complex::new(-2.6414782e-4, 1.5553573e-5),
            Complex::new(-2.6441325e-4, 1.0080363e-5),
            Complex::new(5.543535e-1, -2.6884804e-2),
        ]),
    ];
    let result = vis.slice(s![0, ..]);
    assert_abs_diff_eq!(expected_1st, result, epsilon = epsilon);

    let expected_2nd_row = array![
        Jones::from([
            Complex::new(2.0769031e-1, 2.3794083e-2),
            Complex::new(-8.771045e-5, -1.0350297e-5),
            Complex::new(-8.777869e-5, -9.754652e-6),
            Complex::new(2.0775805e-1, 2.3801845e-2),
        ]),
        Jones::from([
            Complex::new(5.8825934e-1, 3.388093e-2),
            Complex::new(-2.4847797e-4, -1.5157494e-5),
            Complex::new(-2.4857515e-4, -1.3470396e-5),
            Complex::new(5.884512e-1, 3.3891983e-2),
        ]),
        Jones::from([
            Complex::new(3.9592028e-1, -2.2407714e-2),
            Complex::new(-1.672997e-4, 8.899025e-6),
            Complex::new(-1.6723544e-4, 1.0034504e-5),
            Complex::new(3.960494e-1, -2.2415023e-2),
        ]),
    ];
    let result = vis.slice(s![1, ..]);
    assert_abs_diff_eq!(expected_2nd_row, result, epsilon = epsilon);

    let expected_3rd_row = array![
        Jones::from([
            Complex::new(1.1981204e-1, 1.570818e-2),
            Complex::new(-3.624168e-5, -4.271608e-6),
            Complex::new(-3.6117963e-5, -5.2152363e-6),
            Complex::new(1.1991071e-1, 1.5721114e-2),
        ]),
        Jones::from([
            Complex::new(4.66734e-1, 3.0732285e-2),
            Complex::new(-1.4106138e-4, -7.442297e-6),
            Complex::new(-1.4081933e-4, -1.111825e-5),
            Complex::new(4.6711835e-1, 3.0757593e-2),
        ]),
        Jones::from([
            Complex::new(2.7827054e-1, -1.8004907e-2),
            Complex::new(-8.395886e-5, 6.5327845e-6),
            Complex::new(-8.410067e-5, 4.341152e-6),
            Complex::new(2.784997e-1, -1.8019734e-2),
        ]),
    ];
    let result = vis.slice(s![2, ..]);
    assert_abs_diff_eq!(expected_3rd_row, result, epsilon = epsilon);
}

#[track_caller]
fn test_non_trivial_gaussian_curved_power_law(vis: ArrayView2<Jones<f32>>, epsilon: f32) {
    let expected_1st = array![
        Jones::from([
            Complex::new(3.4501022e-1, 3.3840403e-2),
            Complex::new(-1.646843e-4, -1.4433144e-5),
            Complex::new(-1.6435012e-4, -1.784027e-5),
            Complex::new(3.4509042e-1, 3.384827e-2),
        ]),
        Jones::from([
            Complex::new(7.4135214e-1, 3.658786e-2),
            Complex::new(-3.5369248e-4, -1.3786237e-5),
            Complex::new(-3.5333115e-4, -2.1107411e-5),
            Complex::new(7.415245e-1, 3.6596365e-2),
        ]),
        Jones::from([
            Complex::new(5.542246e-1, -2.6878554e-2),
            Complex::new(-2.6414782e-4, 1.5553573e-5),
            Complex::new(-2.6441325e-4, 1.0080363e-5),
            Complex::new(5.543535e-1, -2.6884804e-2),
        ]),
    ];
    let result = vis.slice(s![0, ..]);
    assert_abs_diff_eq!(expected_1st, result, epsilon = epsilon);

    let expected_2nd_row = array![
        Jones::from([
            Complex::new(2.0783842e-1, 2.3811052e-2),
            Complex::new(-8.777299e-5, -1.0357679e-5),
            Complex::new(-8.784128e-5, -9.761608e-6),
            Complex::new(2.079062e-1, 2.3818819e-2),
        ]),
        Jones::from([
            Complex::new(5.8867884e-1, 3.3905093e-2),
            Complex::new(-2.4865518e-4, -1.5168303e-5),
            Complex::new(-2.4875242e-4, -1.3480002e-5),
            Complex::new(5.888708e-1, 3.3916153e-2),
        ]),
        Jones::from([
            Complex::new(3.9620262e-1, -2.2423694e-2),
            Complex::new(-1.67419e-4, 8.90537e-6),
            Complex::new(-1.673547e-4, 1.0041659e-5),
            Complex::new(3.9633185e-1, -2.2431009e-2),
        ]),
    ];
    let result = vis.slice(s![1, ..]);
    assert_abs_diff_eq!(expected_2nd_row, result, epsilon = epsilon);

    let expected_3rd_row = array![
        Jones::from([
            Complex::new(1.20109886e-1, 1.5747229e-2),
            Complex::new(-3.633177e-5, -4.2822267e-6),
            Complex::new(-3.6207748e-5, -5.228201e-6),
            Complex::new(1.202088e-1, 1.5760196e-2),
        ]),
        Jones::from([
            Complex::new(4.6789426e-1, 3.0808683e-2),
            Complex::new(-1.4141205e-4, -7.460798e-6),
            Complex::new(-1.411694e-4, -1.1145889e-5),
            Complex::new(4.6827957e-1, 3.0834055e-2),
        ]),
        Jones::from([
            Complex::new(2.7896228e-1, -1.8049665e-2),
            Complex::new(-8.416758e-5, 6.5490244e-6),
            Complex::new(-8.430974e-5, 4.351944e-6),
            Complex::new(2.7919203e-1, -1.8064529e-2),
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
