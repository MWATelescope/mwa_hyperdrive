// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Tests on generating sky-model visibilities.

mod cpu;
#[cfg(feature = "cuda")]
mod cuda;

use std::ops::Deref;

use approx::abs_diff_eq;
use marlu::{
    constants::{MWA_LAT_RAD, MWA_LONG_RAD},
    pos::xyz::xyzs_to_cross_uvws_parallel,
    Complex, Jones, RADec, XyzGeodetic,
};
use ndarray::prelude::*;
use vec1::vec1;

use super::*;
#[cfg(feature = "cuda")]
use crate::model::cuda::SkyModellerCuda;
use mwa_hyperdrive_beam::{create_fee_beam_object, create_no_beam_object};
use mwa_hyperdrive_common::{marlu, ndarray, vec1};
use mwa_hyperdrive_srclist::{
    ComponentType, FluxDensity, FluxDensityType, ShapeletCoeff, SourceComponent, SourceList,
};

fn get_list() -> FluxDensityType {
    FluxDensityType::List {
        fds: vec1![
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
    }
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

fn get_simple_point(pos: RADec, flux_type: FluxType) -> SourceComponent {
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

fn get_simple_gaussian(pos: RADec, flux_type: FluxType) -> SourceComponent {
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

fn get_simple_shapelet(pos: RADec, flux_type: FluxType) -> SourceComponent {
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
    flagged_tiles: Vec<usize>,
    array_longitude_rad: f64,
    array_latitude_rad: f64,
}

impl ObsParams {
    fn new(no_beam: bool) -> ObsParams {
        let phase_centre = RADec::new_degrees(0.0, -27.0);
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
        let uvws = xyzs_to_cross_uvws_parallel(&xyzs, phase_centre.to_hadec(lst));
        let beam = if no_beam {
            create_no_beam_object(xyzs.len())
        } else {
            let beam_file: Option<&str> = None;
            create_fee_beam_object(
                beam_file,
                xyzs.len(),
                mwa_hyperdrive_beam::Delays::Partial(vec![0; 16]),
                None,
            )
            .unwrap()
        };
        let flagged_tiles = vec![];
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

    fn get_cpu_modeller(&self, srclist: &SourceList) -> SkyModellerCpu {
        new_cpu_sky_modeller_inner(
            self.beam.deref(),
            srclist,
            &self.xyzs,
            &self.freqs,
            &self.flagged_tiles,
            self.phase_centre,
            self.array_longitude_rad,
            self.array_latitude_rad,
            Duration::from_total_nanoseconds(0),
            true,
        )
    }

    #[cfg(feature = "cuda")]
    fn get_gpu_modeller(&self, srclist: &SourceList) -> SkyModellerCuda {
        unsafe {
            super::new_cuda_sky_modeller_inner(
                self.beam.deref(),
                srclist,
                &self.xyzs,
                &self.freqs,
                &self.flagged_tiles,
                self.phase_centre,
                self.array_longitude_rad,
                self.array_latitude_rad,
                Duration::from_total_nanoseconds(0),
                true,
            )
            .unwrap()
        }
    }
}

fn assert_list_zenith_visibilities(vis: ArrayView2<Jones<f32>>, fail_str: &str) {
    // All LMN values are (0, 0, 1). This means that the Fourier transform to
    // make a visibility from LMN and UVW will always just be the input flux
    // density.

    // First column: flux density at first frequency (i.e. 1 Jy XX and YY).
    let expected = Array1::from_elem(vis.dim().0, Jones::identity());
    let result = vis.slice(s![.., 0]);
    assert!(
        abs_diff_eq!(expected, result),
        "{fail_str}:\nexpected: {expected:?}\ngot:      {result:?}\n",
    );
    // 3 Jy XX and YY.
    let expected = Array1::from_elem(vis.dim().0, Jones::identity() * 3.0);
    let result = vis.slice(s![.., 1]);
    assert!(
        abs_diff_eq!(expected, result),
        "{fail_str}:\nexpected: {expected:?}\ngot:      {result:?}\n",
    );
    // 2 Jy XX and YY.
    let expected = Array1::from_elem(vis.dim().0, Jones::identity() * 2.0);
    let result = vis.slice(s![.., 2]);
    assert!(
        abs_diff_eq!(expected, result),
        "{fail_str}:\nexpected: {expected:?}\ngot:      {result:?}\n",
    );
}

fn assert_list_off_zenith_visibilities(vis: ArrayView2<Jones<f32>>, fail_str: &str) {
    // This time, all LMN values should be close to, but not the same as, (0, 0,
    // 1). This means that the visibilities should be somewhat close to the
    // input flux densities.

    let expected = array![
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
    let result = vis.slice(s![.., 0]);
    assert!(
        abs_diff_eq!(expected, result),
        "{fail_str}:\nexpected: {expected:?}\ngot:      {result:?}\n",
    );

    let expected = array![
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
    let result = vis.slice(s![.., 1]);
    assert!(
        abs_diff_eq!(expected, result),
        "{fail_str}:\nexpected: {expected:?}\ngot:      {result:?}\n",
    );

    let expected = array![
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
    let result = vis.slice(s![.., 2]);
    assert!(
        abs_diff_eq!(expected, result),
        "{fail_str}:\nexpected: {expected:?}\ngot:      {result:?}\n",
    );
}

fn assert_list_zenith_visibilities_fee(vis: ArrayView2<Jones<f32>>, fail_str: &str) {
    // All LMN values are (0, 0, 1). This means that the Fourier transform to
    // make a visibility from LMN and UVW will always just be the input flux
    // density *multiplied by the beam response*.

    // First column: flux density at first frequency (i.e. ~1 Jy XX and YY).
    let expected = array![
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
    let result = vis.slice(s![.., 0]);
    assert!(
        abs_diff_eq!(expected, result),
        "{fail_str}:\nexpected: {expected:?}\ngot:      {result:?}\n",
    );
    // ~3 Jy XX and YY.
    let expected = array![
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
    let result = vis.slice(s![.., 1]);
    assert!(
        abs_diff_eq!(expected, result),
        "{fail_str}:\nexpected: {expected:?}\ngot:      {result:?}\n",
    );
    // ~2 Jy XX and YY.
    let expected = array![
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
    let result = vis.slice(s![.., 2]);
    assert!(
        abs_diff_eq!(expected, result),
        "{fail_str}:\nexpected: {expected:?}\ngot:      {result:?}\n",
    );
}

fn assert_list_off_zenith_visibilities_fee(vis: ArrayView2<Jones<f32>>, fail_str: &str) {
    // This time, all LMN values should be close to, but not the same as, (0, 0,
    // 1). This means that the visibilities should be somewhat close to the
    // input flux densities.

    let expected = array![
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
    let result = vis.slice(s![.., 0]);
    assert!(
        abs_diff_eq!(expected, result),
        "{fail_str}:\nexpected: {expected:?}\ngot:      {result:?}\n",
    );

    let expected = array![
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
    let result = vis.slice(s![.., 1]);
    assert!(
        abs_diff_eq!(expected, result),
        "{fail_str}:\nexpected: {expected:?}\ngot:      {result:?}\n",
    );

    let expected = array![
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
    let result = vis.slice(s![.., 2]);
    assert!(
        abs_diff_eq!(expected, result),
        "{fail_str}:\nexpected: {expected:?}\ngot:      {result:?}\n",
    );
}

fn assert_power_law_zenith_visibilities(vis: ArrayView2<Jones<f32>>, fail_str: &str) {
    // All LMN values are (0, 0, 1). This means that the Fourier transform to
    // make a visibility from LMN and UVW will always just be the input flux
    // density.

    // First column: flux density at first frequency (i.e. 1 Jy XX and YY).
    let expected = Array1::from_elem(vis.dim().0, Jones::identity());
    let result = vis.slice(s![.., 0]);
    assert!(
        abs_diff_eq!(expected, result),
        "{fail_str}:\nexpected: {expected:?}\ngot:      {result:?}\n",
    );

    let expected = array![
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
    let result = vis.slice(s![.., 1]);
    assert!(
        abs_diff_eq!(expected, result),
        "{fail_str}:\nexpected: {expected:?}\ngot:      {result:?}\n",
    );

    let expected = array![
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
    let result = vis.slice(s![.., 2]);
    assert!(
        abs_diff_eq!(expected, result),
        "{fail_str}:\nexpected: {expected:?}\ngot:      {result:?}\n",
    );
}

fn assert_power_law_off_zenith_visibilities(vis: ArrayView2<Jones<f32>>, fail_str: &str) {
    // This time, all LMN values should be close to, but not the same as, (0, 0,
    // 1). This means that the visibilities should be somewhat close to the
    // input flux densities.

    let expected = array![
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
    let result = vis.slice(s![.., 0]);
    assert!(
        abs_diff_eq!(expected, result),
        "{fail_str}:\nexpected: {expected:?}\ngot:      {result:?}\n",
    );

    let expected = array![
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
    let result = vis.slice(s![.., 1]);
    assert!(
        abs_diff_eq!(expected, result),
        "{fail_str}:\nexpected: {expected:?}\ngot:      {result:?}\n",
    );

    let expected = array![
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
    let result = vis.slice(s![.., 2]);
    assert!(
        abs_diff_eq!(expected, result),
        "{fail_str}:\nexpected: {expected:?}\ngot:      {result:?}\n",
    );
}

fn assert_power_law_zenith_visibilities_fee(vis: ArrayView2<Jones<f32>>, fail_str: &str) {
    // All LMN values are (0, 0, 1). This means that the Fourier transform to
    // make a visibility from LMN and UVW will always just be the input flux
    // density *multiplied by the beam response*.

    // First column: flux density at first frequency (i.e. ~1 Jy XX and YY).
    let expected = array![
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
    let result = vis.slice(s![.., 0]);
    assert!(
        abs_diff_eq!(expected, result),
        "{fail_str}:\nexpected: {expected:?}\ngot:      {result:?}\n",
    );

    let expected = array![
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
    let result = vis.slice(s![.., 1]);
    assert!(
        abs_diff_eq!(expected, result),
        "{fail_str}:\nexpected: {expected:?}\ngot:      {result:?}\n",
    );

    let expected = array![
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
    let result = vis.slice(s![.., 2]);
    assert!(
        abs_diff_eq!(expected, result),
        "{fail_str}:\nexpected: {expected:?}\ngot:      {result:?}\n",
    );
}

fn assert_power_law_off_zenith_visibilities_fee(vis: ArrayView2<Jones<f32>>, fail_str: &str) {
    // This time, all LMN values should be close to, but not the same as, (0, 0,
    // 1). This means that the visibilities should be somewhat close to the
    // input flux densities.

    let expected = array![
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
    let result = vis.slice(s![.., 0]);
    assert!(
        abs_diff_eq!(expected, result),
        "{fail_str}:\nexpected: {expected:?}\ngot:      {result:?}\n",
    );

    let expected = array![
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
    let result = vis.slice(s![.., 1]);
    assert!(
        abs_diff_eq!(expected, result),
        "{fail_str}:\nexpected: {expected:?}\ngot:      {result:?}\n",
    );

    let expected = array![
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
    let result = vis.slice(s![.., 2]);
    assert!(
        abs_diff_eq!(expected, result),
        "{fail_str}:\nexpected: {expected:?}\ngot:      {result:?}\n",
    );
}

fn assert_curved_power_law_zenith_visibilities(vis: ArrayView2<Jones<f32>>, fail_str: &str) {
    // All LMN values are (0, 0, 1). This means that the Fourier transform to
    // make a visibility from LMN and UVW will always just be the input flux
    // density.

    // First column: flux density at first frequency (i.e. 1 Jy XX and YY).
    let expected = Array1::from_elem(vis.dim().0, Jones::identity());
    let result = vis.slice(s![.., 0]);
    assert!(
        abs_diff_eq!(expected, result),
        "{fail_str}:\nexpected: {expected:?}\ngot:      {result:?}\n",
    );

    let expected = array![
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
    let result = vis.slice(s![.., 1]);
    assert!(
        abs_diff_eq!(expected, result),
        "{fail_str}:\nexpected: {expected:?}\ngot:      {result:?}\n",
    );

    let expected = array![
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
    let result = vis.slice(s![.., 2]);
    assert!(
        abs_diff_eq!(expected, result),
        "{fail_str}:\nexpected: {expected:?}\ngot:      {result:?}\n",
    );
}

fn assert_curved_power_law_off_zenith_visibilities(vis: ArrayView2<Jones<f32>>, fail_str: &str) {
    // This time, all LMN values should be close to, but not the same as, (0, 0,
    // 1). This means that the visibilities should be somewhat close to the
    // input flux densities.

    let expected = array![
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
    let result = vis.slice(s![.., 0]);
    assert!(
        abs_diff_eq!(expected, result),
        "{fail_str}:\nexpected: {expected:?}\ngot:      {result:?}\n",
    );

    let expected = array![
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
    let result = vis.slice(s![.., 1]);
    assert!(
        abs_diff_eq!(expected, result),
        "{fail_str}:\nexpected: {expected:?}\ngot:      {result:?}\n",
    );

    let expected = array![
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
    let result = vis.slice(s![.., 2]);
    assert!(
        abs_diff_eq!(expected, result),
        "{fail_str}:\nexpected: {expected:?}\ngot:      {result:?}\n",
    );
}

fn assert_curved_power_law_zenith_visibilities_fee(vis: ArrayView2<Jones<f32>>, fail_str: &str) {
    // All LMN values are (0, 0, 1). This means that the Fourier transform to
    // make a visibility from LMN and UVW will always just be the input flux
    // density *multiplied by the beam response*.

    // First column: flux density at first frequency (i.e. ~1 Jy XX and YY).
    let expected = array![
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
    let result = vis.slice(s![.., 0]);
    assert!(
        abs_diff_eq!(expected, result),
        "{fail_str}:\nexpected: {expected:?}\ngot:      {result:?}\n",
    );

    let expected = array![
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
    let result = vis.slice(s![.., 1]);
    assert!(
        abs_diff_eq!(expected, result),
        "{fail_str}:\nexpected: {expected:?}\ngot:      {result:?}\n",
    );

    let expected = array![
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
    let result = vis.slice(s![.., 2]);
    assert!(
        abs_diff_eq!(expected, result),
        "{fail_str}:\nexpected: {expected:?}\ngot:      {result:?}\n",
    );
}

fn assert_curved_power_law_off_zenith_visibilities_fee(
    vis: ArrayView2<Jones<f32>>,
    fail_str: &str,
) {
    // This time, all LMN values should be close to, but not the same as, (0, 0,
    // 1). This means that the visibilities should be somewhat close to the
    // input flux densities.

    let expected = array![
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
    let result = vis.slice(s![.., 0]);
    assert!(
        abs_diff_eq!(expected, result),
        "{fail_str}:\nexpected: {expected:?}\ngot:      {result:?}\n",
    );

    let expected = array![
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
    let result = vis.slice(s![.., 1]);
    assert!(
        abs_diff_eq!(expected, result),
        "{fail_str}:\nexpected: {expected:?}\ngot:      {result:?}\n",
    );

    let expected = array![
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
    let result = vis.slice(s![.., 2]);
    assert!(
        abs_diff_eq!(expected, result),
        "{fail_str}:\nexpected: {expected:?}\ngot:      {result:?}\n",
    );
}
