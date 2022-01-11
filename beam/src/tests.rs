// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use approx::assert_abs_diff_eq;
use mwa_hyperbeam::fee::FEEBeam;
use serial_test::serial;

use super::*;
use crate::jones_test::TestJones;

#[test]
fn no_beam_means_no_beam() {
    let azels = [
        AzEl { az: 0.0, el: 0.0 },
        AzEl { az: 1.0, el: 0.1 },
        AzEl { az: -1.0, el: 0.2 },
    ];
    let beam = create_no_beam_object(1);
    for azel in azels {
        let j = beam.calc_jones(azel, 150e6, 0).unwrap();

        let j = TestJones::from(j);
        let expected = TestJones::from(Jones::identity());
        assert_abs_diff_eq!(j, expected);
    }
}

#[test]
#[serial]
fn fee_beam_values_are_sensible() {
    let delays = vec![0; 16];
    let amps = [1.0; 16];
    let freq = 150e6;
    let azels = [
        AzEl { az: 0.0, el: 0.0 },
        AzEl { az: 1.0, el: 0.1 },
        AzEl { az: -1.0, el: 0.2 },
    ];
    let (azs, zas): (Vec<_>, Vec<_>) = azels.iter().map(|azel| (azel.az, azel.za())).unzip();

    // Get the beam values right out of hyperbeam.
    let hyperbeam = FEEBeam::new_from_env().unwrap();
    let hyperbeam_values = hyperbeam
        .calc_jones_array(&azs, &zas, freq as u32, &delays, &amps, true)
        .unwrap();

    // Compare these with the hyperdrive `Beam` trait.
    let gains = array![amps];
    let hyperdrive = super::FEEBeam::new_from_env(1, Delays::Partial(delays), Some(gains)).unwrap();
    let hyperdrive_values: Vec<Jones<f64>> = azels
        .iter()
        .map(|&azel| hyperdrive.calc_jones(azel, freq, 0).unwrap())
        .collect();

    let hyperdrive_values = Array1::from(hyperdrive_values).mapv(TestJones::from);
    let hyperbeam_values =
        Array1::from(hyperbeam_values).mapv(|j| TestJones::from([j[0], j[1], j[2], j[3]]));
    assert_abs_diff_eq!(hyperdrive_values, hyperbeam_values);
}

#[test]
#[serial]
#[cfg(feature = "cuda")]
fn fee_cuda_beam_values_are_sensible() {
    let delays = Array2::zeros((1, 16));
    let amps = Array2::ones((1, 16));
    let freqs = [150e6 as u32];
    let azels = [
        AzEl { az: 0.0, el: 0.0 },
        AzEl { az: 1.0, el: 0.1 },
        AzEl { az: -1.0, el: 0.2 },
    ];
    let (azs, zas): (Vec<_>, Vec<_>) = azels
        .iter()
        .map(|azel| (azel.az as CudaFloat, azel.za() as CudaFloat))
        .unzip();

    // Get the beam values right out of hyperbeam.
    let hyperbeam = FEEBeam::new_from_env().unwrap();
    let hyperbeam =
        unsafe { hyperbeam.cuda_prepare(&freqs, delays.view(), amps.view(), true) }.unwrap();
    let hyperbeam_values = hyperbeam.calc_jones(&azs, &zas, true).unwrap();

    // Compare these with the hyperdrive `Beam` trait.
    let hyperdrive = super::FEEBeam::new_from_env(1, Delays::Full(delays), Some(amps)).unwrap();
    let hyperdrive = unsafe { hyperdrive.prepare_cuda_beam(&freqs).unwrap() };
    let hyperdrive_values_device = unsafe { hyperdrive.calc_jones(&azels).unwrap() };
    let mut hyperdrive_values = vec![Jones::default(); hyperbeam_values.len()];
    unsafe {
        hyperdrive_values_device
            .copy_from_device(&mut hyperdrive_values)
            .unwrap();
    }

    let hyperdrive_values = Array3::from_shape_vec(hyperbeam_values.dim(), hyperdrive_values)
        .unwrap()
        .mapv(TestJones::from);
    let hyperbeam_values = hyperbeam_values.mapv(|j| TestJones::from([j[0], j[1], j[2], j[3]]));
    assert_abs_diff_eq!(hyperdrive_values, hyperbeam_values);
}
