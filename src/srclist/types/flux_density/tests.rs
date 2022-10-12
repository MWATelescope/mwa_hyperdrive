// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use approx::assert_abs_diff_eq;
use marlu::c64;
use vec1::vec1;

use super::*;

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
    // All negative to start with.
    let fdt = FluxDensityType::List(vec1![
        FluxDensity {
            freq: 150e6,
            i: -1.0,
            ..Default::default()
        },
        FluxDensity {
            freq: 175e6,
            i: -3.0,
            ..Default::default()
        },
        FluxDensity {
            freq: 200e6,
            i: -2.0,
            ..Default::default()
        }
    ]);
    assert_abs_diff_eq!(
        fdt.estimate_at_freq(140e6),
        FluxDensity {
            freq: 140e6,
            // Using the FDs with 150 and 175 MHz, SI = 0.611.
            i: -0.611583722518741,
            ..Default::default()
        }
    );
    assert_abs_diff_eq!(
        fdt.estimate_at_freq(150e6),
        FluxDensity {
            freq: 150e6,
            // Exact match.
            i: -1.0,
            ..Default::default()
        }
    );
    assert_abs_diff_eq!(
        fdt.estimate_at_freq(160e6),
        FluxDensity {
            freq: 160e6,
            // Spec index 7.126.
            i: -1.584007188344133,
            ..Default::default()
        }
    );
    assert_abs_diff_eq!(
        fdt.estimate_at_freq(190e6),
        FluxDensity {
            freq: 190e6,
            // Spec index -3.036.
            i: -2.3370702845538984,
            ..Default::default()
        }
    );
    assert_abs_diff_eq!(
        fdt.estimate_at_freq(210e6),
        FluxDensity {
            freq: 210e6,
            // Spectral index -3.036.
            i: -1.72460308893938,
            ..Default::default()
        }
    );

    // One negative, one positive.
    let fdt = FluxDensityType::List(vec1![
        FluxDensity {
            freq: 100e6,
            i: -1.0,
            ..Default::default()
        },
        FluxDensity {
            freq: 200e6,
            i: 1.0,
            ..Default::default()
        },
    ]);
    let fds = match &fdt {
        FluxDensityType::List(fds) => fds,
        _ => unreachable!(),
    };
    let desired_freq = 90e6;
    let result = fdt.estimate_at_freq(desired_freq);
    let expected = FluxDensity {
        freq: desired_freq,
        // IQUV are increased/decreased with the straight line fit between the
        // positive and negative FDs.
        i: fds[0].i - 0.2,
        q: 0.0,
        u: 0.0,
        v: 0.0,
    };
    assert_abs_diff_eq!(result, expected);

    let desired_freq = 210e6;
    let result = fdt.estimate_at_freq(desired_freq);
    let expected = FluxDensity {
        freq: desired_freq,
        i: fds[1].i + 0.2,
        q: 0.0,
        u: 0.0,
        v: 0.0,
    };
    assert_abs_diff_eq!(result, expected);

    let desired_freq = 150e6;
    let result = fdt.estimate_at_freq(desired_freq);
    let expected = FluxDensity {
        freq: desired_freq,
        i: 0.0,
        q: 0.0,
        u: 0.0,
        v: 0.0,
    };
    assert_abs_diff_eq!(result, expected);

    // Two negative, one positive.
    let fdt = FluxDensityType::List(vec1![
        FluxDensity {
            freq: 100e6,
            i: -1.0,
            ..Default::default()
        },
        FluxDensity {
            freq: 150e6,
            i: -0.5,
            ..Default::default()
        },
        FluxDensity {
            freq: 200e6,
            i: 1.0,
            ..Default::default()
        },
    ]);
    let fds = match &fdt {
        FluxDensityType::List(fds) => fds,
        _ => unreachable!(),
    };
    let desired_freq = 90e6;
    let result = fdt.estimate_at_freq(desired_freq);
    // A spectral index is used for frequencies < 150e6.
    let spec_index = (fds[1].i / fds[0].i).ln() / (fds[1].freq / fds[0].freq).ln();
    let ratio = calc_flux_ratio(desired_freq, fds[0].freq, spec_index);
    let expected = FluxDensity {
        freq: desired_freq,
        ..fds[0]
    } * ratio;
    assert_abs_diff_eq!(result, expected);
    assert_abs_diff_eq!(result.i, -1.1973550404744007);

    let desired_freq = 145e6;
    let result = fdt.estimate_at_freq(desired_freq);
    let ratio = calc_flux_ratio(desired_freq, fds[0].freq, spec_index);
    let expected = FluxDensity {
        freq: desired_freq,
        ..fds[0]
    } * ratio;
    assert_abs_diff_eq!(result, expected);
    assert_abs_diff_eq!(result.i, -0.5298337000434852);

    // The straight line is used again > 150e6.
    let desired_freq = 155e6;
    let result = fdt.estimate_at_freq(desired_freq);
    let expected = FluxDensity {
        freq: desired_freq,
        i: fds[1].i + 1.5 / 50.0 * 5.0,
        q: 0.0,
        u: 0.0,
        v: 0.0,
    };
    assert_abs_diff_eq!(result, expected);

    let desired_freq = 210e6;
    let result = fdt.estimate_at_freq(desired_freq);
    let expected = FluxDensity {
        freq: desired_freq,
        i: fds[2].i + 1.5 / 50.0 * 10.0,
        q: 0.0,
        u: 0.0,
        v: 0.0,
    };
    assert_abs_diff_eq!(result, expected);
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
    let result = fd.to_inst_stokes();
    assert_abs_diff_eq!(
        result,
        Jones::from([
            c64::new(fd.i - fd.q, 0.0),
            c64::new(fd.u, -fd.v),
            c64::new(fd.u, fd.v),
            c64::new(fd.i + fd.q, 0.0),
        ])
    );
}
