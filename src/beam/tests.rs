// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use approx::assert_abs_diff_eq;
use marlu::{constants::MWA_LAT_RAD, AzEl, Jones};
use mwa_hyperbeam::fee::FEEBeam;
use ndarray::prelude::*;
use serial_test::serial;

use super::*;

#[test]
fn no_beam_means_no_beam() {
    let azels = [
        AzEl { az: 0.0, el: 0.0 },
        AzEl { az: 1.0, el: 0.1 },
        AzEl { az: -1.0, el: 0.2 },
    ];
    let beam = create_no_beam_object(1);
    for azel in azels {
        let j = beam.calc_jones(azel, 150e6, None, MWA_LAT_RAD).unwrap();

        let expected = Jones::identity();
        assert_abs_diff_eq!(j, expected);
    }
}

#[test]
#[serial]
fn fee_beam_values_are_sensible() {
    let delays = [0; 16];
    let amps = [1.0; 16];
    let freq = 150e6;
    let azels = [
        AzEl { az: 0.0, el: 0.0 },
        AzEl { az: 1.0, el: 0.1 },
        AzEl { az: -1.0, el: 0.2 },
    ];
    let (azs, zas): (Vec<f64>, Vec<f64>) =
        azels.into_iter().map(|azel| (azel.az, azel.za())).unzip();

    // Get the beam values right out of hyperbeam.
    let hyperbeam = FEEBeam::new_from_env().unwrap();
    let hyperbeam_values = hyperbeam
        .calc_jones_array_pair(
            &azs,
            &zas,
            freq as u32,
            &delays,
            &amps,
            true,
            Some(MWA_LAT_RAD),
            false,
        )
        .unwrap();

    // Compare these with the hyperdrive `Beam` trait.
    let gains = array![amps];
    let hyperdrive =
        super::fee::FEEBeam::new_from_env(1, Delays::Partial(delays.to_vec()), Some(gains))
            .unwrap();
    let hyperdrive_values: Vec<Jones<f64>> = azels
        .iter()
        .map(|&azel| {
            hyperdrive
                .calc_jones(azel, freq, None, MWA_LAT_RAD)
                .unwrap()
        })
        .collect();

    assert_abs_diff_eq!(&hyperdrive_values[..], &hyperbeam_values[..]);
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
        unsafe { hyperbeam.gpu_prepare(&freqs, delays.view(), amps.view(), true) }.unwrap();
    let hyperbeam_values = hyperbeam
        .calc_jones_pair(&azs, &zas, Some(MWA_LAT_RAD), false)
        .unwrap();

    // Compare these with the hyperdrive `Beam` trait.
    let hyperdrive =
        super::fee::FEEBeam::new_from_env(1, Delays::Full(delays), Some(amps)).unwrap();
    let hyperdrive = hyperdrive.prepare_cuda_beam(&freqs).unwrap();
    let hyperdrive_values_device = unsafe {
        let mut hyperdrive_values_device: DevicePointer<Jones<CudaFloat>> = DevicePointer::malloc(
            hyperdrive.get_num_unique_tiles() as usize
                * hyperdrive.get_num_unique_freqs() as usize
                * azs.len()
                * std::mem::size_of::<Jones<CudaFloat>>(),
        )
        .unwrap();
        hyperdrive
            .calc_jones_pair(
                &azs,
                &zas,
                MWA_LAT_RAD,
                hyperdrive_values_device.get_mut().cast(),
            )
            .unwrap();
        hyperdrive_values_device
    };
    let mut hyperdrive_values = vec![Jones::default(); hyperbeam_values.len()];
    hyperdrive_values_device
        .copy_from_device(&mut hyperdrive_values)
        .unwrap();

    let hyperdrive_values =
        Array3::from_shape_vec(hyperbeam_values.dim(), hyperdrive_values).unwrap();
    assert_abs_diff_eq!(hyperdrive_values, hyperbeam_values);
}

#[test]
fn set_delays_to_ideal() {
    let v = vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 32, 3];
    let d = Delays::Partial(v);
    let mut d2 = d.clone();
    d2.set_to_ideal_delays();
    // Despite having a 32, the values are the same, because there's no
    // information on what 32 should be replaced with.
    match (d, d2) {
        (Delays::Partial(d), Delays::Partial(d2)) => assert_eq!(d, d2),
        _ => unreachable!(),
    }

    let v = vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3];
    let d = Delays::Partial(v);
    let mut d2 = d.clone();
    d2.set_to_ideal_delays();
    // Already-ideal delays are left as is too.
    match (d, d2) {
        (Delays::Partial(d), Delays::Partial(d2)) => assert_eq!(d, d2),
        _ => unreachable!(),
    }

    let mut a = Array2::ones((2, 16));
    a[(0, 15)] = 32;
    let d = Delays::Full(a);
    let mut d2 = d.clone();
    d2.set_to_ideal_delays();
    match (d, d2) {
        (Delays::Full(d), Delays::Full(d2)) => {
            // The delays are not equal, because the 32 has been replaced.
            assert_ne!(d, d2);
            for i in 0..2 {
                for j in 0..15 {
                    assert_eq!(d[(i, j)], d2[(i, j)]);
                }
            }
            assert_ne!(d[(0, 15)], d2[(0, 15)]);
        }
        _ => unreachable!(),
    }

    let a = Array2::ones((2, 16));
    let d = Delays::Full(a);
    let mut d2 = d.clone();
    d2.set_to_ideal_delays();
    match (d, d2) {
        (Delays::Full(d), Delays::Full(d2)) => {
            // The delays are equal because nothing needs to be done.
            assert_eq!(d, d2);
        }
        _ => unreachable!(),
    }
}
