// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::f64::consts::{FRAC_PI_4, LOG2_10, PI};

use approx::{assert_abs_diff_eq, assert_abs_diff_ne};
use marlu::RADec;
use vec1::vec1;

use crate::srclist::{
    calc_flux_ratio,
    types::{ComponentType, FluxDensity, FluxDensityType, Source, SourceComponent},
};

fn get_list_source_1() -> Source {
    Source {
        components: vec![SourceComponent {
            radec: RADec::from_radians(PI, FRAC_PI_4),
            comp_type: ComponentType::Point,
            flux_type: FluxDensityType::List(vec1![
                FluxDensity {
                    freq: 100.0,
                    i: 10.0,
                    q: 7.0,
                    u: 6.0,
                    v: 1.0,
                },
                FluxDensity {
                    freq: 150.0,
                    i: 8.0,
                    q: 5.0,
                    u: 4.0,
                    v: 0.25,
                },
                FluxDensity {
                    freq: 200.0,
                    i: 6.0,
                    q: 4.0,
                    u: 3.0,
                    v: 0.1,
                },
            ]),
        }]
        .into_boxed_slice(),
    }
}

// TODO: Write more tests using this source.
// This is an extreme example.
fn get_list_source_2() -> Source {
    Source {
        components: vec![SourceComponent {
            radec: RADec::from_radians(PI - 0.82, -FRAC_PI_4),
            comp_type: ComponentType::Point,
            flux_type: FluxDensityType::List(vec1![
                FluxDensity {
                    freq: 100.0,
                    i: 10.0,
                    q: 7.0,
                    u: 6.0,
                    v: 1.0,
                },
                FluxDensity {
                    freq: 200.0,
                    i: 1.0,
                    q: 2.0,
                    u: 3.0,
                    v: 4.1,
                },
            ]),
        }]
        .into_boxed_slice(),
    }
}

#[test]
fn list_calc_spec_index_source_1() {
    let source = get_list_source_1();
    let fds = match &source.components[0].flux_type {
        FluxDensityType::List(fds) => fds,
        _ => unreachable!(),
    };

    let si = fds[0].calc_spec_index(&fds[1]);
    assert_abs_diff_eq!(si, -0.5503397132132084, epsilon = 1e-10);

    let si = fds[1].calc_spec_index(&fds[2]);
    assert_abs_diff_eq!(si, -1.0000000000000002, epsilon = 1e-10);

    let si = fds[0].calc_spec_index(&fds[2]);
    assert_abs_diff_eq!(si, -0.7369655941662062, epsilon = 1e-10);

    // Despite using the same two flux densities in the opposite way, the SI
    // comes out the same.
    let si = fds[2].calc_spec_index(&fds[0]);
    let expected = -0.7369655941662062;
    assert_abs_diff_eq!(si, expected, epsilon = 1e-10);
}

#[test]
fn list_calc_spec_index_source_2() {
    let source = get_list_source_2();
    let fds = match &source.components[0].flux_type {
        FluxDensityType::List(fds) => fds,
        _ => unreachable!(),
    };

    let si = fds[0].calc_spec_index(&fds[1]);
    assert_abs_diff_eq!(si, -LOG2_10, epsilon = 1e-10);
}

#[test]
#[should_panic]
fn list_estimate_flux_density_at_freq_no_comps() {
    let mut source = get_list_source_1();
    // Delete all flux densities. This will panic on the unwrap because `fds` needs to have at least 1 elements.
    match &mut source.components[0].flux_type {
        FluxDensityType::List(fds) => fds.drain(..).unwrap(),
        _ => unreachable!(),
    };
}
#[test]
fn list_estimate_flux_density_at_freq_extrapolation_single_comp1() {
    let mut source = get_list_source_1();
    match &mut source.components[0].flux_type {
        FluxDensityType::List(fds) => fds.drain(1..).unwrap(),
        _ => unreachable!(),
    };

    let fd = source.components[0].flux_type.estimate_at_freq(90.0);
    assert_abs_diff_eq!(fd.i, 10.879426248455298, epsilon = 1e-10);
    assert_abs_diff_eq!(fd.q, 7.615598373918708, epsilon = 1e-10);
    assert_abs_diff_eq!(fd.u, 6.527655749073178, epsilon = 1e-10);
    assert_abs_diff_eq!(fd.v, 1.0879426248455297, epsilon = 1e-10);

    let fd = source.components[0].flux_type.estimate_at_freq(110.0);
    assert_abs_diff_eq!(fd.i, 9.265862513558696, epsilon = 1e-10);
    assert_abs_diff_eq!(fd.q, 6.486103759491087, epsilon = 1e-10);
    assert_abs_diff_eq!(fd.u, 5.559517508135217, epsilon = 1e-10);
    assert_abs_diff_eq!(fd.v, 0.9265862513558696, epsilon = 1e-10);
}

#[test]
fn list_estimate_flux_density_at_freq_extrapolation_source_1() {
    let source = get_list_source_1();
    let fd = source.components[0].flux_type.estimate_at_freq(90.0);
    assert_abs_diff_eq!(fd.i, 10.596981209124532, epsilon = 1e-10);
    assert_abs_diff_eq!(fd.q, 7.417886846387172, epsilon = 1e-10);
    assert_abs_diff_eq!(fd.u, 6.358188725474719, epsilon = 1e-10);
    assert_abs_diff_eq!(fd.v, 1.0596981209124532, epsilon = 1e-10);

    let fd = source.components[0].flux_type.estimate_at_freq(210.0);
    assert_abs_diff_eq!(fd.i, 5.7142857142857135, epsilon = 1e-10);
    assert_abs_diff_eq!(fd.q, 3.8095238095238093, epsilon = 1e-10);
    assert_abs_diff_eq!(fd.u, 2.8571428571428568, epsilon = 1e-10);
    assert_abs_diff_eq!(fd.v, 0.09523809523809523, epsilon = 1e-10);
}

#[test]
fn list_estimate_flux_density_at_freq_source_2() {
    let source = get_list_source_2();
    let fds = match &source.components[0].flux_type {
        FluxDensityType::List(fds) => fds,
        _ => unreachable!(),
    };

    let desired_freq = 210.0;
    let fd = source.components[0]
        .flux_type
        .estimate_at_freq(desired_freq);
    let si = (fds[1].i / fds[0].i).ln() / (fds[1].freq / fds[0].freq).ln();
    assert_abs_diff_eq!(si, -LOG2_10, epsilon = 1e-10);
    let freq_ratio = calc_flux_ratio(desired_freq, fds[1].freq, si);
    let expected = fds[1] * freq_ratio;
    assert_abs_diff_eq!(fd.i, expected.i, epsilon = 1e-10);
    assert_abs_diff_eq!(fd.q, expected.q, epsilon = 1e-10);
    assert_abs_diff_eq!(fd.u, expected.u, epsilon = 1e-10);
    assert_abs_diff_eq!(fd.v, expected.v, epsilon = 1e-10);
}

#[test]
#[should_panic]
fn unsorted_list_error() {
    let mut source = get_list_source_2();
    match &mut source.components[0].flux_type {
        FluxDensityType::List(fds) => {
            fds.reverse();
        }
        _ => unreachable!(),
    };
    let _fd = source.components[0].flux_type.estimate_at_freq(210.0);
}

#[test]
fn test_power_law() {
    let mut source = get_list_source_1();
    source.components[0].flux_type = FluxDensityType::PowerLaw {
        si: -0.6,
        fd: FluxDensity {
            freq: 150.0,
            i: 10.0,
            q: 7.0,
            u: 6.0,
            v: 1.0,
        },
    };
    let new_fd = source.components[0].flux_type.estimate_at_freq(200.0);
    assert_abs_diff_eq!(new_fd.i, 8.414663590846496, epsilon = 1e-10);
    assert_abs_diff_eq!(new_fd.q, 5.890264513592547, epsilon = 1e-10);
    assert_abs_diff_eq!(new_fd.u, 5.048798154507898, epsilon = 1e-10);
    assert_abs_diff_eq!(new_fd.v, 0.8414663590846496, epsilon = 1e-10);

    let new_fd = source.components[0].flux_type.estimate_at_freq(100.0);
    assert_abs_diff_eq!(new_fd.i, 12.754245006257909, epsilon = 1e-10);
    assert_abs_diff_eq!(new_fd.q, 8.927971504380537, epsilon = 1e-10);
    assert_abs_diff_eq!(new_fd.u, 7.652547003754746, epsilon = 1e-10);
    assert_abs_diff_eq!(new_fd.v, 1.275424500625791, epsilon = 1e-10);
}

#[test]
fn test_curved_power_law() {
    let mut source = get_list_source_1();
    source.components[0].flux_type = FluxDensityType::CurvedPowerLaw {
        si: -0.6,
        fd: FluxDensity {
            freq: 150.0,
            i: 10.0,
            q: 7.0,
            u: 6.0,
            v: 1.0,
        },
        q: 0.5,
    };
    let new_fd = source.components[0].flux_type.estimate_at_freq(200.0);
    assert_abs_diff_eq!(new_fd.i, 8.770171284543176, epsilon = 1e-10);
    assert_abs_diff_eq!(new_fd.q, 6.139119899180223, epsilon = 1e-10);
    assert_abs_diff_eq!(new_fd.u, 5.262102770725906, epsilon = 1e-10);
    assert_abs_diff_eq!(new_fd.v, 0.8770171284543176, epsilon = 1e-10);

    let new_fd = source.components[0].flux_type.estimate_at_freq(100.0);
    assert_abs_diff_eq!(new_fd.i, 13.846951980528223, epsilon = 1e-10);
    assert_abs_diff_eq!(new_fd.q, 9.692866386369756, epsilon = 1e-10);
    assert_abs_diff_eq!(new_fd.u, 8.308171188316935, epsilon = 1e-10);
    assert_abs_diff_eq!(new_fd.v, 1.3846951980528224, epsilon = 1e-10);

    // Just ensure that the results coming out of a curved power law don't
    // match those of a normal power law.
    source.components[0].flux_type = FluxDensityType::PowerLaw {
        si: -0.6,
        fd: FluxDensity {
            freq: 150.0,
            i: 10.0,
            q: 7.0,
            u: 6.0,
            v: 1.0,
        },
    };
    let norm_power_law = source.components[0].flux_type.estimate_at_freq(200.0);
    assert_abs_diff_ne!(new_fd.i, norm_power_law.i);
    assert_abs_diff_ne!(new_fd.q, norm_power_law.q);
    assert_abs_diff_ne!(new_fd.u, norm_power_law.u);
    assert_abs_diff_ne!(new_fd.v, norm_power_law.v);
}
