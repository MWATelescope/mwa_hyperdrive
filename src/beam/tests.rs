// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use approx::assert_abs_diff_eq;
#[cfg(any(feature = "cuda", feature = "hip"))]
use approx::assert_abs_diff_ne;
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
    let beam = NoBeam { num_tiles: 1 };
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

    // Exercise the array path (rewritten to call calc_jones_array_inner).
    let hyperdrive_array = hyperdrive
        .calc_jones_array(&azels, freq, None, MWA_LAT_RAD)
        .unwrap();
    assert_abs_diff_eq!(&hyperdrive_array[..], &hyperbeam_values[..]);
    let hyperdrive_tile0 = hyperdrive
        .calc_jones_array(&azels, freq, Some(0), MWA_LAT_RAD)
        .unwrap();
    assert_eq!(hyperdrive_tile0.len(), azels.len());
}

#[test]
#[serial]
#[cfg(any(feature = "cuda", feature = "hip"))]
fn fee_gpu_beam_values_are_sensible() {
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
        .map(|azel| (azel.az as GpuFloat, azel.za() as GpuFloat))
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
    let hyperdrive = hyperdrive.prepare_gpu_beam(&freqs).unwrap();
    let hyperdrive_values_device = unsafe {
        let mut hyperdrive_values_device: DevicePointer<Jones<GpuFloat>> = DevicePointer::malloc(
            hyperdrive.get_num_unique_tiles() as usize
                * hyperdrive.get_num_unique_freqs() as usize
                * azs.len()
                * std::mem::size_of::<Jones<GpuFloat>>(),
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

#[test]
fn create_analytic_beam_objects() {
    let delays = Delays::Partial(vec![0; 16]);
    let mwa_pb = create_beam_object(Some("analytic-mwa_pb"), 3, delays.clone()).unwrap();
    assert_eq!(mwa_pb.get_beam_type(), BeamType::AnalyticMwaPb);
    assert_eq!(mwa_pb.get_num_tiles(), 3);

    let rts = create_beam_object(Some("analytic-rts"), 1, delays.clone()).unwrap();
    assert_eq!(rts.get_beam_type(), BeamType::AnalyticRts);

    // Beam type names are case-insensitive.
    for alias in ["mwa_pb", "MWA_PB", "analytic-mwa_pb", "Analytic-MWA_pb"] {
        assert_eq!(
            create_beam_object(Some(alias), 1, delays.clone())
                .unwrap()
                .get_beam_type(),
            BeamType::AnalyticMwaPb
        );
    }
    for alias in ["rts", "RTS", "analytic-rts", "Analytic-RTS"] {
        assert_eq!(
            create_beam_object(Some(alias), 1, delays.clone())
                .unwrap()
                .get_beam_type(),
            BeamType::AnalyticRts
        );
    }
    for alias in ["none", "None", "NONE"] {
        assert_eq!(
            create_beam_object(Some(alias), 1, delays.clone())
                .unwrap()
                .get_beam_type(),
            BeamType::None
        );
    }
    assert_eq!(BeamType::from_str("FEE").unwrap(), BeamType::FEE);

    assert!(matches!(
        create_beam_object(Some("not-a-beam"), 1, Delays::Partial(vec![0; 16])),
        Err(BeamError::Unrecognised(_))
    ));
}

#[test]
fn analytic_beam_values_match_hyperbeam() {
    use mwa_hyperbeam::analytic::{AnalyticBeam as HbAnalytic, AnalyticType};

    let delays = [0u32, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6];
    let amps = [1.0; 16];
    let freq = 180e6;
    let azels = [
        AzEl { az: 0.0, el: 1.2 },
        AzEl { az: 1.0, el: 0.8 },
        AzEl { az: -0.5, el: 0.5 },
    ];

    let hyperbeam = HbAnalytic::new_custom(
        AnalyticType::MwaPb,
        AnalyticType::MwaPb.get_default_dipole_height(),
        4,
    );
    let hyperbeam_values = hyperbeam
        .calc_jones_array(&azels, freq as _, &delays, &amps, MWA_LAT_RAD, true)
        .unwrap();

    let hyperdrive = super::analytic::AnalyticBeam::new(
        AnalyticType::MwaPb,
        1,
        Delays::Partial(delays.to_vec()),
        None,
    )
    .unwrap();
    let hyperdrive_values = hyperdrive
        .calc_jones_array(&azels, freq, None, MWA_LAT_RAD)
        .unwrap();
    assert_abs_diff_eq!(&hyperdrive_values[..], &hyperbeam_values[..]);
}

#[test]
fn hyperbeam_analytic_error_maps_to_beam() {
    let err = BeamError::HyperbeamAnalytic(
        mwa_hyperbeam::analytic::AnalyticBeamError::IncorrectAmpsLength {
            got: 1,
            expected1: 16,
            expected2: 32,
        },
    );
    let mapped = crate::HyperdriveError::from(err);
    let s = mapped.to_string();
    assert!(s.contains("hyperbeam analytic"), "{s}");
}

/// Ensure that the GPU analytic beam gives the same results as hyperbeam's own
/// GPU code. Multiple tiles (with distinct delays and dipole gains) and
/// multiple frequencies are used, so that the tile and frequency maps are
/// exercised.
#[test]
#[cfg(any(feature = "cuda", feature = "hip"))]
fn analytic_gpu_beam_values_are_sensible() {
    use mwa_hyperbeam::analytic::{AnalyticBeam as HbAnalytic, AnalyticType};

    let mut delays = Array2::zeros((2, 16));
    delays.slice_mut(s![1, ..]).fill(2);
    let mut amps = Array2::ones((2, 32));
    // Kill a dipole on the second tile.
    amps[(1, 5)] = 0.0;
    let freqs = [149.76e6 as u32, 175.36e6 as u32, 199.68e6 as u32];
    let azels = [
        AzEl { az: 0.0, el: 1.5 },
        AzEl { az: 1.0, el: 0.1 },
        AzEl { az: -1.0, el: 0.2 },
    ];
    let (azs, zas): (Vec<_>, Vec<_>) = azels
        .iter()
        .map(|azel| (azel.az as GpuFloat, azel.za() as GpuFloat))
        .unzip();

    for (analytic_type, beam_type) in [
        (AnalyticType::MwaPb, BeamType::AnalyticMwaPb),
        (AnalyticType::Rts, BeamType::AnalyticRts),
    ] {
        // Get the beam values right out of hyperbeam.
        let hyperbeam =
            HbAnalytic::new_custom(analytic_type, analytic_type.get_default_dipole_height(), 4);
        let hyperbeam = unsafe { hyperbeam.gpu_prepare(delays.view(), amps.view()) }.unwrap();
        let hyperbeam_values = hyperbeam
            .calc_jones_pair(&azs, &zas, &freqs, MWA_LAT_RAD as GpuFloat, true)
            .unwrap();

        // Compare these with the hyperdrive `Beam` trait.
        let hyperdrive = super::analytic::AnalyticBeam::new(
            analytic_type,
            2,
            Delays::Full(delays.clone()),
            Some(amps.clone()),
        )
        .unwrap();
        let hyperdrive = hyperdrive.prepare_gpu_beam(&freqs).unwrap();
        assert_eq!(hyperdrive.get_beam_type(), beam_type);
        // The frequencies are in distinct coarse channels, so none are
        // de-duplicated.
        assert_eq!(hyperdrive.get_num_unique_freqs(), freqs.len() as i32);
        // The two tiles are distinct, so there's no de-duplication and the tile
        // map is the identity map; the device results can be compared directly
        // against hyperbeam's "expanded" results.
        assert_eq!(hyperdrive.get_num_unique_tiles(), 2);

        let hyperdrive_values_device = unsafe {
            let mut d: DevicePointer<Jones<GpuFloat>> = DevicePointer::malloc(
                hyperdrive.get_num_unique_tiles() as usize
                    * hyperdrive.get_num_unique_freqs() as usize
                    * azs.len()
                    * std::mem::size_of::<Jones<GpuFloat>>(),
            )
            .unwrap();
            hyperdrive
                .calc_jones_pair(&azs, &zas, MWA_LAT_RAD, d.get_mut().cast())
                .unwrap();
            d
        };
        let mut hyperdrive_values = vec![Jones::default(); hyperbeam_values.len()];
        hyperdrive_values_device
            .copy_from_device(&mut hyperdrive_values)
            .unwrap();
        let hyperdrive_values =
            Array3::from_shape_vec(hyperbeam_values.dim(), hyperdrive_values).unwrap();
        assert_abs_diff_eq!(hyperdrive_values, hyperbeam_values);

        // The values should not be trivial. (Note that j00 is zero for an
        // azimuth of 0, so all of the Jones elements are checked.)
        assert!(
            hyperdrive_values
                .iter()
                .any(|j| (0..4).any(|i| j[i].norm() > 0.1)),
            "all of the analytic beam responses were ~zero"
        );
    }
}

/// The GPU analytic beam should agree with the CPU analytic beam, including
/// which tile and frequency the responses belong to.
#[test]
#[cfg(any(feature = "cuda", feature = "hip"))]
fn analytic_gpu_beam_matches_cpu() {
    // Give each tile its own delays so that no tiles are de-duplicated; this
    // means that the device results are laid out per tile in the same order as
    // the tiles given here.
    let mut delays = Array2::zeros((3, 16));
    delays.slice_mut(s![1, ..]).fill(2);
    delays.slice_mut(s![2, ..]).fill(4);
    let mut amps = Array2::ones((3, 32));
    amps[(2, 0)] = 0.0;
    let freqs = [149.76e6 as u32, 199.68e6 as u32];
    // Two of the directions are below the horizon; the CPU gives them a zero
    // response and the GPU must too.
    let azels = [
        AzEl { az: 0.0, el: 1.5 },
        AzEl { az: 0.5, el: -0.1 },
        AzEl { az: 1.0, el: 0.3 },
        AzEl { az: -2.0, el: 0.6 },
        AzEl { az: 2.5, el: -1.0 },
        AzEl { az: 3.0, el: 0.9 },
    ];
    let below_horizon = [1, 4];
    let (azs, zas): (Vec<_>, Vec<_>) = azels
        .iter()
        .map(|azel| (azel.az as GpuFloat, azel.za() as GpuFloat))
        .unzip();
    // The same directions without the below-horizon ones, to check that
    // zeroing the below-horizon responses leaves the others alone.
    let (above_azs, above_zas): (Vec<_>, Vec<_>) = azels
        .iter()
        .enumerate()
        .filter(|(i, _)| !below_horizon.contains(i))
        .map(|(_, azel)| (azel.az as GpuFloat, azel.za() as GpuFloat))
        .unzip();

    #[cfg(not(feature = "gpu-single"))]
    let epsilon = 1e-10;
    #[cfg(feature = "gpu-single")]
    let epsilon = 2e-4;

    for beam_type in [BeamType::AnalyticMwaPb, BeamType::AnalyticRts] {
        let delays = Delays::Full(delays.clone());
        let amps = Some(amps.clone());
        let cpu = AnalyticBeam::new(beam_type.analytic_type().unwrap(), 3, delays, amps).unwrap();
        let gpu = cpu.prepare_gpu_beam(&freqs).unwrap();
        assert_eq!(gpu.get_num_unique_tiles(), 3);
        assert_eq!(gpu.get_num_unique_freqs(), 2);

        let mut d_jones: DevicePointer<Jones<GpuFloat>> = DevicePointer::malloc(
            gpu.get_num_unique_tiles() as usize
                * gpu.get_num_unique_freqs() as usize
                * azs.len()
                * std::mem::size_of::<Jones<GpuFloat>>(),
        )
        .unwrap();
        unsafe {
            gpu.calc_jones_pair(&azs, &zas, MWA_LAT_RAD, d_jones.get_mut().cast())
                .unwrap();
        }
        let mut gpu_values = vec![Jones::default(); 3 * freqs.len() * azels.len()];
        d_jones.copy_from_device(&mut gpu_values).unwrap();
        let gpu_values = Array3::from_shape_vec((3, freqs.len(), azels.len()), gpu_values).unwrap();

        // The below-horizon responses are exactly zero...
        for i_dir in below_horizon {
            for j in gpu_values.slice(s![.., .., i_dir]) {
                assert_eq!(
                    *j,
                    Jones::default(),
                    "direction {i_dir} is below the horizon"
                );
            }
        }
        // ... and the other responses are exactly what the GPU gives when the
        // below-horizon directions aren't present.
        let mut d_above_jones: DevicePointer<Jones<GpuFloat>> = DevicePointer::malloc(
            3 * freqs.len() * above_azs.len() * std::mem::size_of::<Jones<GpuFloat>>(),
        )
        .unwrap();
        unsafe {
            gpu.calc_jones_pair(
                &above_azs,
                &above_zas,
                MWA_LAT_RAD,
                d_above_jones.get_mut().cast(),
            )
            .unwrap();
        }
        let mut above_values = vec![Jones::default(); 3 * freqs.len() * above_azs.len()];
        d_above_jones.copy_from_device(&mut above_values).unwrap();
        let above_values =
            Array3::from_shape_vec((3, freqs.len(), above_azs.len()), above_values).unwrap();
        for (i_above, i_dir) in (0..azels.len())
            .filter(|i| !below_horizon.contains(i))
            .enumerate()
        {
            assert_eq!(
                gpu_values.slice(s![.., .., i_dir]),
                above_values.slice(s![.., .., i_above])
            );
        }

        for i_tile in 0..3 {
            for (i_freq, &freq) in freqs.iter().enumerate() {
                let cpu_values = cpu
                    .calc_jones_array(&azels, f64::from(freq), Some(i_tile), MWA_LAT_RAD)
                    .unwrap();
                for (i_dir, cpu_j) in cpu_values.iter().enumerate() {
                    let gpu_j = gpu_values[(i_tile, i_freq, i_dir)];
                    let cpu_j: Jones<GpuFloat> = Jones::from([
                        cpu_j[0].re as GpuFloat,
                        cpu_j[0].im as GpuFloat,
                        cpu_j[1].re as GpuFloat,
                        cpu_j[1].im as GpuFloat,
                        cpu_j[2].re as GpuFloat,
                        cpu_j[2].im as GpuFloat,
                        cpu_j[3].re as GpuFloat,
                        cpu_j[3].im as GpuFloat,
                    ]);
                    assert_abs_diff_eq!(gpu_j, cpu_j, epsilon = epsilon);
                }
            }
        }

        assert!(
            gpu_values.iter().any(|j| (0..4).any(|i| j[i].norm() > 0.1)),
            "all of the analytic beam responses were ~zero"
        );
    }
}

/// Copy a [`BeamGpu`]'s frequency map back from the device.
#[cfg(any(feature = "cuda", feature = "hip"))]
fn get_host_freq_map(gpu: &dyn BeamGpu, num_freqs: usize) -> Vec<i32> {
    #[cfg(feature = "cuda")]
    use cuda_runtime_sys::{
        cudaMemcpy as gpuMemcpy, cudaMemcpyKind::cudaMemcpyDeviceToHost as gpuMemcpyDeviceToHost,
    };
    #[cfg(feature = "hip")]
    use hip_sys::hiprt::{
        hipMemcpy as gpuMemcpy, hipMemcpyKind::hipMemcpyDeviceToHost as gpuMemcpyDeviceToHost,
    };

    let mut freq_map = vec![-1_i32; num_freqs];
    unsafe {
        gpuMemcpy(
            freq_map.as_mut_ptr().cast(),
            gpu.get_freq_map().cast(),
            std::mem::size_of_val(freq_map.as_slice()),
            gpuMemcpyDeviceToHost,
        );
    }
    freq_map
}

/// The GPU analytic beam is evaluated at exactly the supplied frequencies (the
/// modellers supply their spline knots); only duplicate frequencies share a
/// response.
#[test]
#[cfg(any(feature = "cuda", feature = "hip"))]
fn analytic_gpu_beam_freq_map_dedups_exact_freqs() {
    let beam =
        AnalyticBeam::new(AnalyticType::MwaPb, 1, Delays::Partial(vec![0; 16]), None).unwrap();

    // Two different fine channels in the same coarse channel get their own
    // responses.
    let freqs = [150.00e6 as u32, 150.04e6 as u32];
    let gpu = beam.prepare_gpu_beam(&freqs).unwrap();
    assert_eq!(gpu.get_num_unique_freqs(), 2);
    assert_eq!(get_host_freq_map(&*gpu, freqs.len()), [0, 1]);

    // Duplicates are de-duplicated, and the map points at the first-seen
    // unique entry.
    let freqs = [
        150.00e6 as u32, // index 0
        151.04e6 as u32, // index 1
        150.00e6 as u32, // index 0
        151.04e6 as u32, // index 1
    ];
    let gpu = beam.prepare_gpu_beam(&freqs).unwrap();
    assert_eq!(gpu.get_num_unique_freqs(), 2);
    assert_eq!(get_host_freq_map(&*gpu, freqs.len()), [0, 1, 0, 1]);

    // The responses at the two unique frequencies are what the CPU gives at
    // those frequencies, and they differ from one another.
    let azels = [AzEl { az: 1.0, el: 0.3 }, AzEl { az: -2.0, el: 0.6 }];
    let (azs, zas): (Vec<_>, Vec<_>) = azels
        .iter()
        .map(|azel| (azel.az as GpuFloat, azel.za() as GpuFloat))
        .unzip();
    let mut d_jones: DevicePointer<Jones<GpuFloat>> =
        DevicePointer::malloc(2 * azels.len() * std::mem::size_of::<Jones<GpuFloat>>()).unwrap();
    unsafe {
        gpu.calc_jones_pair(&azs, &zas, MWA_LAT_RAD, d_jones.get_mut().cast())
            .unwrap();
    }
    let mut gpu_values = vec![Jones::default(); 2 * azels.len()];
    d_jones.copy_from_device(&mut gpu_values).unwrap();
    let gpu_values = Array2::from_shape_vec((2, azels.len()), gpu_values).unwrap();

    #[cfg(not(feature = "gpu-single"))]
    let epsilon = 1e-10;
    #[cfg(feature = "gpu-single")]
    let epsilon = 2e-4;
    for (i_freq, grid_freq) in [150.00e6, 151.04e6].into_iter().enumerate() {
        let cpu_values = beam
            .calc_jones_array(&azels, grid_freq, None, MWA_LAT_RAD)
            .unwrap();
        for (i_dir, cpu_j) in cpu_values.iter().enumerate() {
            let cpu_j: Jones<GpuFloat> = Jones::from([
                cpu_j[0].re as GpuFloat,
                cpu_j[0].im as GpuFloat,
                cpu_j[1].re as GpuFloat,
                cpu_j[1].im as GpuFloat,
                cpu_j[2].re as GpuFloat,
                cpu_j[2].im as GpuFloat,
                cpu_j[3].re as GpuFloat,
                cpu_j[3].im as GpuFloat,
            ]);
            assert_abs_diff_eq!(gpu_values[(i_freq, i_dir)], cpu_j, epsilon = epsilon);
        }
    }
    assert_abs_diff_ne!(
        gpu_values[(0, 0)],
        gpu_values[(1, 0)],
        epsilon = 1e-6 as GpuFloat
    );
}

#[test]
fn spanning_freqs_bracket_and_clamp() {
    let freqs = [100.0, 200.0, 300.0, 400.0];
    assert_eq!(spanning_freqs(&freqs, &[]), Vec::<f64>::new());
    assert_eq!(spanning_freqs(&freqs, &[150.0]), vec![100.0, 200.0]);
    assert_eq!(spanning_freqs(&freqs, &[200.0]), vec![200.0]);
    assert_eq!(spanning_freqs(&freqs, &[200.0, 300.0]), vec![200.0, 300.0]);
    assert_eq!(
        spanning_freqs(&freqs, &[120.0, 310.0]),
        vec![100.0, 200.0, 300.0, 400.0]
    );
    assert_eq!(
        spanning_freqs(&freqs, &[50.0, 250.0]),
        vec![100.0, 200.0, 300.0]
    );
    assert_eq!(spanning_freqs(&freqs, &[350.0, 500.0]), vec![300.0, 400.0]);
    assert_eq!(spanning_freqs(&freqs, &[500.0]), vec![400.0]);
    assert_eq!(spanning_freqs(&[150.0], &[160.0]), vec![150.0]);
}

#[test]
#[serial]
fn fee_beam_knots_span_fine_channels() {
    let beam = super::FEEBeam::new_from_env(1, Delays::Partial(vec![0; 16]), None).unwrap();
    // 40 kHz channels over 167.04--197.76 MHz (24 coarse channels): the knots
    // are the FEE file's 1.28 MHz frequencies from 166.40 to 198.40 MHz.
    let freqs: Vec<f64> = (0..768).map(|i| 167.06e6 + 40e3 * i as f64).collect();
    let knots = beam.get_beam_knots(&freqs);
    assert_eq!(knots.len(), 26);
    assert_abs_diff_eq!(knots[0], 166.40e6);
    assert_abs_diff_eq!(knots[25], 198.40e6);
    for w in knots.windows(2) {
        assert_abs_diff_eq!(w[1] - w[0], 1.28e6);
    }
    // A channel exactly at a file frequency needs only that one.
    assert_eq!(beam.get_beam_knots(&[167.68e6]), vec![167.68e6]);
}

#[test]
fn thin_freqs_keeps_ends_and_spacing() {
    // Already at least 1.28 MHz apart: unchanged.
    let coarse: Vec<f64> = (0..5).map(|i| 166.40e6 + 1.28e6 * f64::from(i)).collect();
    assert_eq!(thin_freqs(&coarse, 1.28e6), coarse);
    // 10 kHz spacing (like the interpolated FEE file) from 167.04 MHz: every
    // 128th frequency, and the last one even though it's closer than that.
    let dense: Vec<f64> = (0..=3000).map(|i| 167.04e6 + 10e3 * f64::from(i)).collect();
    let thinned = thin_freqs(&dense, 1.28e6);
    assert_eq!(thinned.len(), 25);
    for (i, t) in thinned.iter().take(24).enumerate() {
        assert_abs_diff_eq!(*t, 167.04e6 + 1.28e6 * i as f64, epsilon = 1e-3);
    }
    assert_abs_diff_eq!(thinned[24], 197.04e6, epsilon = 1e-3);
    // Degenerate inputs.
    assert_eq!(thin_freqs(&[], 1.28e6), Vec::<f64>::new());
    assert_eq!(thin_freqs(&[150e6], 1.28e6), vec![150e6]);
    assert_eq!(
        thin_freqs(&[150e6, 150.01e6], 1.28e6),
        vec![150e6, 150.01e6]
    );
}

#[test]
fn beam_knots_from_file_freqs_thins_dense_files() {
    // A beam file sampled every 10 kHz over 166.40--197.12 MHz. 40 kHz
    // channels within it get knots about a coarse channel apart, not one per
    // channel.
    let file: Vec<f64> = (0..=3072).map(|i| 166.40e6 + 10e3 * f64::from(i)).collect();
    let freqs: Vec<f64> = (0..768).map(|i| 167.06e6 + 40e3 * f64::from(i)).collect();
    let knots = beam_knots_from_file_freqs(&file, &freqs);
    assert!(knots.len() <= 27, "{} knots", knots.len());
    assert_abs_diff_eq!(knots[0], 167.06e6, epsilon = 1e-3);
    assert_abs_diff_eq!(knots[knots.len() - 1], 197.12e6, epsilon = 1e-3);
    for w in knots.windows(2) {
        assert!(w[1] - w[0] >= 1.28e6 - 1e-3 || w[1] == 197.12e6);
    }
    // The standard 1.28 MHz file is unchanged by thinning.
    let file: Vec<f64> = (0..=24).map(|i| 166.40e6 + 1.28e6 * f64::from(i)).collect();
    assert_eq!(beam_knots_from_file_freqs(&file, &freqs), file);
}

#[test]
fn beam_knots_on_grid_span_with_guards() {
    assert_eq!(beam_knots_on_grid(&[], 2), Vec::<f64>::new());
    // A single channel on the grid: that knot plus the guards.
    let knots = beam_knots_on_grid(&[167.68e6], 2);
    assert_eq!(knots.len(), 5);
    assert_abs_diff_eq!(knots[2], 167.68e6);
    // Channels within one coarse channel: its two bracketing knots plus two
    // guards each side, all multiples of 1.28 MHz.
    let knots = beam_knots_on_grid(&[150e6, 150.04e6, 150.08e6], 2);
    assert_eq!(knots.len(), 6);
    assert_abs_diff_eq!(knots[0], 115.0 * 1.28e6);
    assert_abs_diff_eq!(knots[5], 120.0 * 1.28e6);
    // No guards.
    let knots = beam_knots_on_grid(&[150e6, 150.04e6, 150.08e6], 0);
    assert_eq!(knots, vec![149.76e6, 151.04e6]);
    // Never at or below zero.
    assert_abs_diff_eq!(beam_knots_on_grid(&[1e6], 2)[0], 1.28e6);
}
