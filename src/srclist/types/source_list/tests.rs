// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::f64::consts::*;

use approx::*;
use marlu::RADec;

use super::*;
use crate::srclist::{FluxDensity, FluxDensityType};

#[test]
// Test that the (Az, El) coordinates retrieved from the
// `.get_azel_mwa_parallel()` method of `SourceList` are correct and always
// in the same order.
fn test_get_azel_mwa() {
    let mut sl = SourceList::new();
    // Use a common component. Only the `radec` part needs to be modified.
    let comp = SourceComponent {
        radec: RADec::from_radians(PI, FRAC_PI_4),
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
    let mut comps = vec![comp.clone()];

    // Modify the coordinates of other components.
    let mut comp2 = comp.clone();
    comp2.radec = RADec::from_radians(PI - 0.1, FRAC_PI_4 + 0.1);
    comps.push(comp2);

    let mut comp2 = comp.clone();
    comp2.radec = RADec::from_radians(PI + 0.1, FRAC_PI_4 - 0.1);
    comps.push(comp2);

    // Push "source_1".
    sl.insert(
        "source_1".to_string(),
        Source {
            components: comps.clone().into_boxed_slice(),
        },
    );
    comps.clear();

    comps.push(comp.clone());
    let mut comp2 = comp.clone();
    comp2.radec = RADec::from_radians(PI - 0.1, FRAC_PI_4 + 0.1);
    comps.push(comp2);

    let mut comp2 = comp.clone();
    comp2.radec = RADec::from_radians(PI + 0.1, FRAC_PI_4 - 0.1);
    comps.push(comp2);

    sl.insert(
        "source_2".to_string(),
        Source {
            components: comps.clone().into_boxed_slice(),
        },
    );
    comps.clear();

    let mut comp2 = comp.clone();
    comp2.radec = RADec::from_radians(FRAC_PI_2, PI);
    comps.push(comp2);

    let mut comp2 = comp;
    comp2.radec = RADec::from_radians(FRAC_PI_2 - 0.1, PI + 0.2);
    comps.push(comp2);

    sl.insert(
        "source_3".to_string(),
        Source {
            components: comps.into_boxed_slice(),
        },
    );

    let lst = 3.0 * FRAC_PI_4;
    let (azs, zas): (Vec<_>, Vec<_>) = sl
        .iter()
        .flat_map(|(_, src)| src.components.iter())
        .map(|comp| {
            let azel = comp.radec.to_hadec(lst).to_azel_mwa();
            (azel.az, azel.za())
        })
        .unzip();
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
    assert_abs_diff_eq!(azs.as_slice(), az_expected.as_slice(), epsilon = 1e-10);
    assert_abs_diff_eq!(zas.as_slice(), za_expected.as_slice(), epsilon = 1e-10);
}
