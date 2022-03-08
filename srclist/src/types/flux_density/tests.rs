// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use approx::assert_abs_diff_eq;
use marlu::c64;
use vec1::vec1;

use super::*;
use crate::jones_test::TestJones;
use mwa_hyperdrive_common::vec1;

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
#[ignore]
// TODO: Fix!
fn estimate_with_negative_fds() {
    // MLT011814-4027 from LoBES
    let fdt = FluxDensityType::List {
        fds: vec1![
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
