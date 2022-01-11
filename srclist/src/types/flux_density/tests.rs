// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use approx::assert_abs_diff_eq;
use marlu::c64;

use super::*;
use crate::jones_test::TestJones;

#[test]
fn calc_freq_ratio_1() {
    let desired_freq = 160.0;
    let cat_freq = 150.0;
    let spec_index = -0.6;
    let ratio = calc_flux_ratio(desired_freq, cat_freq, spec_index);
    let expected = 0.9620170425907598;
    assert_abs_diff_eq!(ratio, expected, epsilon = 1e-10);
}

#[test]
fn calc_freq_ratio_2() {
    let desired_freq = 140.0;
    let cat_freq = 150.0;
    let spec_index = -0.6;
    let ratio = calc_flux_ratio(desired_freq, cat_freq, spec_index);
    let expected = 1.0422644718599143;
    assert_abs_diff_eq!(ratio, expected, epsilon = 1e-10);
}

#[test]
fn estimate_with_negative_fds() {
    // MLT011814-4027 from LoBES
    let fdt = FluxDensityType::List {
        fds: vec![
            FluxDensity {
                freq: 130e6,
                i: -0.006830723490566015,
                ..Default::default()
            },
            FluxDensity {
                freq: 143e6,
                i: -0.027053141966462135,
                ..Default::default()
            },
            FluxDensity {
                freq: 151e6,
                i: -0.038221485912799835,
                ..Default::default()
            },
            FluxDensity {
                freq: 166e6,
                i: 0.08616726100444794,
                ..Default::default()
            },
            FluxDensity {
                freq: 174e6,
                i: 0.11915085464715958,
                ..Default::default()
            },
            FluxDensity {
                freq: 181e6,
                i: 0.06860895454883575,
                ..Default::default()
            },
        ],
    };
    let desired_freq = 136.5e6;
    let result = fdt.estimate_at_freq(desired_freq);
    let si = (0.027053141966462135_f64 / 0.006830723490566015).ln() / (143e6_f64 / 130e6).ln();
    let flux_ratio = calc_flux_ratio(desired_freq, 130e6, si);
    let expected = FluxDensity {
        freq: desired_freq,
        i: -0.004744315639,
        ..Default::default()
    } * flux_ratio;
    assert_abs_diff_eq!(result, expected, epsilon = 1e-10);

    let desired_freq = 158.5e6;
    let result = fdt.estimate_at_freq(desired_freq);
    let si = (0.08616726100444794_f64 / 0.038221485912799835).ln() / (166e6_f64 / 151e6).ln();
    let flux_ratio = calc_flux_ratio(desired_freq, 151e6, si);
    let expected = FluxDensity {
        freq: desired_freq,
        i: -0.013818502583985523,
        ..Default::default()
    } * flux_ratio;
    assert_abs_diff_eq!(result, expected, epsilon = 1e-10);
}

fn get_fdt() -> FluxDensityType {
    // J034844-125505 from
    // srclist_pumav3_EoR0aegean_fixedEoR1pietro+ForA_phase1+2.txt
    FluxDensityType::List {
        fds: vec![
            FluxDensity {
                freq: 120e6,
                i: 8.00841,
                ..Default::default()
            },
            FluxDensity {
                freq: 140e6,
                i: 6.80909,
                ..Default::default()
            },
            FluxDensity {
                freq: 160e6,
                i: 5.91218,
                ..Default::default()
            },
            FluxDensity {
                freq: 180e6,
                i: 5.21677,
                ..Default::default()
            },
        ],
    }
}

#[test]
#[should_panic]
fn test_none_convert_list_to_power_law() {
    let mut fdt = get_fdt();
    // This is definitely a list.
    assert!(matches!(fdt, FluxDensityType::List { .. }));
    // Empty the flux densities. Our function will panic.
    match &mut fdt {
        FluxDensityType::List { fds } => *fds = vec![],
        _ => unreachable!(),
    }
    fdt.convert_list_to_power_law();
}

#[test]
fn test_one_convert_list_to_power_law() {
    let mut fdt = get_fdt();
    // This is definitely a list.
    assert!(matches!(fdt, FluxDensityType::List { .. }));
    // Leave one flux density.
    match &mut fdt {
        FluxDensityType::List { fds } => *fds = vec![fds[0].clone()],
        _ => unreachable!(),
    }
    fdt.convert_list_to_power_law();
    // It's been converted to a power law.
    assert!(matches!(fdt, FluxDensityType::PowerLaw { .. }));
    // We're using the default SI.
    match fdt {
        FluxDensityType::PowerLaw { si, .. } => assert_abs_diff_eq!(si, DEFAULT_SPEC_INDEX),
        _ => unreachable!(),
    }
}

#[test]
fn test_two_convert_list_to_power_law() {
    let mut fdt = get_fdt();
    // This is definitely a list.
    assert!(matches!(fdt, FluxDensityType::List { .. }));
    // Leave two flux densities.
    match &mut fdt {
        FluxDensityType::List { fds } => *fds = vec![fds[0].clone(), fds[1].clone()],
        _ => unreachable!(),
    }
    fdt.convert_list_to_power_law();
    // It's been converted to a power law.
    assert!(matches!(fdt, FluxDensityType::PowerLaw { .. }));
    // We're using the SI between the only two FDs.
    match fdt {
        FluxDensityType::PowerLaw { si, .. } => assert_abs_diff_eq!(si, -1.0524361973093983),
        _ => unreachable!(),
    }
}

#[test]
fn test_many_convert_list_to_power_law() {
    let mut fdt = get_fdt();
    // This is definitely a list.
    assert!(matches!(fdt, FluxDensityType::List { .. }));
    fdt.convert_list_to_power_law();
    // It's been converted to a power law.
    assert!(matches!(fdt, FluxDensityType::PowerLaw { .. }));
    // We're using the SI between the middle two FDs.
    match fdt {
        FluxDensityType::PowerLaw { si, fd } => {
            let expected_fd = FluxDensity {
                freq: 160e6,
                i: 5.910484034862892,
                q: 0.0,
                u: 0.0,
                v: 0.0,
            };
            assert_abs_diff_eq!(si, -1.0570227720845136);
            assert_abs_diff_eq!(fd, expected_fd);
        }
        _ => unreachable!(),
    }
}

#[test]
fn test_many_convert_bad_list_to_power_law() {
    // This source is deliberately poorly conditioned; it should not be
    // converted.
    let mut fdt = FluxDensityType::List {
        fds: vec![
            FluxDensity {
                freq: 120e6,
                i: 1.0,
                ..Default::default()
            },
            FluxDensity {
                freq: 140e6,
                i: 2.0,
                ..Default::default()
            },
            FluxDensity {
                freq: 160e6,
                i: 0.5,
                ..Default::default()
            },
            FluxDensity {
                freq: 180e6,
                i: 3.0,
                ..Default::default()
            },
        ],
    };

    // This is definitely a list.
    assert!(matches!(fdt, FluxDensityType::List { .. }));
    fdt.convert_list_to_power_law();
    // It's not been converted to a power law.
    assert!(matches!(fdt, FluxDensityType::List { .. }));
}

#[test]
fn test_to_jones() {
    let fd = FluxDensity {
        freq: 170e6,
        i: 0.058438801501144624,
        q: -0.3929914018344019,
        u: -0.3899498110659575,
        v: -0.058562589895788,
    };
    let fd2 = fd.clone();
    let result = fd.to_inst_stokes();
    assert_abs_diff_eq!(
        TestJones::from(result),
        TestJones::from([
            c64::new(fd2.i + fd2.q, 0.0),
            c64::new(fd2.u, fd2.v),
            c64::new(fd2.u, -fd2.v),
            c64::new(fd2.i - fd2.q, 0.0),
        ])
    );
}
