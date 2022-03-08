// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::f64::consts::*;

use approx::*;
use vec1::vec1;

use super::*;
use crate::{FluxDensity, FluxDensityType};
use mwa_hyperdrive_common::vec1;

#[test]
// Test that the (Az, El) coordinates retrieved from the
// `.get_azel_mwa_parallel()` method of `SourceList` are correct and always
// in the same order.
fn test_get_azel_mwa() {
    let mut sl = SourceList::new();
    // Use a common component. Only the `radec` part needs to be modified.
    let comp = SourceComponent {
        radec: RADec::new(PI, FRAC_PI_4),
        comp_type: ComponentType::Point,
        flux_type: FluxDensityType::PowerLaw {
            si: -0.8,
            fd: FluxDensity {
                freq: 100.0,
                i: 10.0,
                q: 7.0,
                u: 6.0,
                v: 1.0,
            },
        },
    };
    // Don't modify the first component.
    let mut s = Source {
        components: vec1![comp.clone()],
    };

    // Modify the coordinates of other components.
    s.components.push(comp.clone());
    s.components.last_mut().radec = RADec::new(PI - 0.1, FRAC_PI_4 + 0.1);

    s.components.push(comp.clone());
    s.components.last_mut().radec = RADec::new(PI + 0.1, FRAC_PI_4 - 0.1);

    // Push "source_1".
    sl.insert("source_1".to_string(), s);

    let mut s = Source {
        components: vec1![comp.clone()],
    };

    s.components.push(comp.clone());
    s.components.last_mut().radec = RADec::new(PI - 0.1, FRAC_PI_4 + 0.1);

    s.components.push(comp.clone());
    s.components.last_mut().radec = RADec::new(PI + 0.1, FRAC_PI_4 - 0.1);

    sl.insert("source_2".to_string(), s);

    let mut s = Source {
        components: vec1![comp.clone()],
    };
    s.components.last_mut().radec = RADec::new(FRAC_PI_2, PI);

    s.components.push(comp);
    s.components.last_mut().radec = RADec::new(FRAC_PI_2 - 0.1, PI + 0.2);

    sl.insert("source_3".to_string(), s);

    let lst = 3.0 * FRAC_PI_4;
    let azels = sl.get_azel_mwa_parallel(lst);
    let az_expected = [
        0.5284641294204054,
        0.4140207507698987,
        0.6516588664580675,
        0.5284641294204054,
        0.4140207507698987,
        0.6516588664580675,
        1.9931268490084542,
        2.1121964836053806,
    ];
    let za_expected = [
        1.4415169467014715,
        1.4807939480563403,
        1.416863456467004,
        1.4415169467014715,
        1.4807939480563403,
        1.416863456467004,
        2.254528351516936,
        2.0543439118454256,
    ];
    for ((azel, &expected_az), &expected_za) in
        azels.iter().zip(az_expected.iter()).zip(za_expected.iter())
    {
        assert_abs_diff_eq!(azel.az, expected_az, epsilon = 1e-10);
        assert_abs_diff_eq!(azel.za(), expected_za, epsilon = 1e-10);
    }
}
