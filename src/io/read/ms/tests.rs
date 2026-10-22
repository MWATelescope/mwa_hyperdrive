// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::ffi::CString;

use approx::{assert_abs_diff_eq, assert_abs_diff_ne};
use fitsio::errors::check_status as fits_check_status;
use hifitime::Duration;
use marlu::{
    Jones, MeasurementSetWriter, ObsContext as MarluObsContext, UvwFrame, VisContext, VisWrite,
    XyzGeodetic,
};
use serial_test::serial; // Need to test serially because casacore is a steaming pile.
use tempfile::tempdir;

use super::*;
use crate::tests::{deflate_gz_into_tempfile, get_reduced_1090008640_ms_pbs, DataAsPathBufs};

#[test]
#[serial]
fn test_1090008640_cross_vis() {
    let DataAsPathBufs {
        metafits, mut vis, ..
    } = get_reduced_1090008640_ms_pbs();
    let ms_reader = MsReader::new(vis.swap_remove(0), None, Some(&metafits), None).unwrap();

    let obs_context = ms_reader.get_obs_context();
    let total_num_tiles = obs_context.get_total_num_tiles();
    let num_baselines = (total_num_tiles * (total_num_tiles - 1)) / 2;
    let num_chans = usize::from(obs_context.num_fine_chans_per_coarse_chan.unwrap().get());
    let tile_baseline_flags = TileBaselineFlags::new(total_num_tiles, HashSet::new());

    assert_abs_diff_eq!(
        obs_context.timestamps.first().to_gpst_seconds(),
        1090008658.0
    );

    let mut vis = Array2::zeros((num_chans, num_baselines));
    let mut vis_weights = Array2::zeros((num_chans, num_baselines));
    let result = ms_reader.read_crosses(
        vis.view_mut(),
        vis_weights.view_mut(),
        *obs_context.all_timesteps.first(),
        &tile_baseline_flags,
        &HashSet::new(),
    );
    assert!(result.is_ok(), "{}", result.unwrap_err());

    // These values are exactly the same as the raw data when all corrections
    // (except the PFB gains) are turned on. See the
    // read_1090008640_cross_vis_with_corrections test.
    assert_abs_diff_eq!(
        vis[(0, 0)],
        Jones::from([
            c32::new(-1.2564129e2, -1.497961e1),
            c32::new(8.207059e1, -1.4936417e2),
            c32::new(-7.306871e1, 2.36177e2),
            c32::new(-5.5305626e1, -2.3209404e1)
        ])
    );
    assert_abs_diff_eq!(
        vis[(16, 10)],
        Jones::from([
            c32::new(-4.138127e1, -2.638188e2),
            c32::new(5.220332e2, -2.6055228e2),
            c32::new(4.854074e2, -1.9634505e2),
            c32::new(1.6101791e1, -4.4489478e2),
        ])
    );

    // PFB gains will affect weights, but these weren't in Birli when it made
    // this MS; all but one weight are 8.0 (it's flagged).
    assert_abs_diff_eq!(vis_weights[(2, 11)], -8.0);
    // Undo the flag and test all values.
    vis_weights[(2, 11)] = 8.0;
    assert_abs_diff_eq!(vis_weights, Array2::ones(vis_weights.dim()) * 8.0);
}

#[test]
#[serial]
fn read_1090008640_auto_vis() {
    let DataAsPathBufs {
        metafits, mut vis, ..
    } = get_reduced_1090008640_ms_pbs();
    let ms_reader = MsReader::new(vis.swap_remove(0), None, Some(&metafits), None).unwrap();

    let obs_context = ms_reader.get_obs_context();
    let total_num_tiles = obs_context.get_total_num_tiles();
    let num_chans = usize::from(obs_context.num_fine_chans_per_coarse_chan.unwrap().get());
    let tile_baseline_flags = TileBaselineFlags::new(total_num_tiles, HashSet::new());

    assert_abs_diff_eq!(
        obs_context.timestamps.first().to_gpst_seconds(),
        1090008658.0
    );

    let mut vis = Array2::zeros((num_chans, total_num_tiles));
    let mut vis_weights = Array2::zeros((num_chans, total_num_tiles));
    let result = ms_reader.read_autos(
        vis.view_mut(),
        vis_weights.view_mut(),
        *obs_context.all_timesteps.first(),
        &tile_baseline_flags,
        &HashSet::new(),
    );
    assert!(result.is_ok(), "{}", result.unwrap_err());

    assert_abs_diff_eq!(
        vis[(0, 0)],
        Jones::from([
            5.3557855e4,
            4.3092007e-7,
            -7.420802e2,
            1.3296518e2,
            -7.420802e2,
            -1.3296518e2,
            5.084874e4,
            -6.6132475e-7
        ])
    );
    assert_abs_diff_eq!(
        vis[(2, 0)],
        Jones::from([
            7.1403125e4,
            -1.3957654e-6,
            -1.0667509e3,
            1.01154564e2,
            -1.0667509e3,
            -1.01154564e2,
            6.7373945e4,
            -7.004146e-7
        ])
    );
    assert_abs_diff_eq!(
        vis[(16, 0)],
        Jones::from([
            1.07272586e5,
            1.9233863e-8,
            -1.0756711e3,
            -4.4097336e1,
            -1.0756711e3,
            4.4097336e1,
            1.0141891e5,
            -9.5129735e-7
        ])
    );
    assert_abs_diff_eq!(
        vis[(16, 10)],
        Jones::from([
            1.0766406e5,
            1.5415758e-6,
            -1.334196e3,
            -1.8926495e3,
            -1.334196e3,
            1.8926495e3,
            1.21874336e5,
            2.0712553e-6
        ])
    );

    assert_abs_diff_eq!(vis_weights, Array2::from_elem(vis_weights.dim(), 8.0));
}

#[test]
#[serial]
fn read_1090008640_auto_vis_with_flags() {
    let DataAsPathBufs {
        metafits, mut vis, ..
    } = get_reduced_1090008640_ms_pbs();
    let ms_reader = MsReader::new(vis.swap_remove(0), None, Some(&metafits), None).unwrap();

    let obs_context = ms_reader.get_obs_context();
    let total_num_tiles = obs_context.get_total_num_tiles();
    let num_chans = usize::from(obs_context.num_fine_chans_per_coarse_chan.unwrap().get());
    let tile_flags = HashSet::from([1, 9]);
    let num_unflagged_tiles = total_num_tiles - tile_flags.len();
    let chan_flags = HashSet::from([1]);
    let num_unflagged_chans = num_chans - chan_flags.len();
    let tile_baseline_flags = TileBaselineFlags::new(total_num_tiles, tile_flags);

    assert_abs_diff_eq!(
        obs_context.timestamps.first().to_gpst_seconds(),
        1090008658.0
    );

    let mut vis = Array2::zeros((num_unflagged_chans, num_unflagged_tiles));
    let mut vis_weights = Array2::zeros((num_unflagged_chans, num_unflagged_tiles));
    let result = ms_reader.read_autos(
        vis.view_mut(),
        vis_weights.view_mut(),
        *obs_context.all_timesteps.first(),
        &tile_baseline_flags,
        &chan_flags,
    );
    assert!(result.is_ok(), "{}", result.unwrap_err());

    // Use the same values as the test above, adjusting only the indices.
    assert_abs_diff_eq!(
        vis[(0, 0)],
        Jones::from([
            5.3557855e4,
            4.3092007e-7,
            -7.420802e2,
            1.3296518e2,
            -7.420802e2,
            -1.3296518e2,
            5.084874e4,
            -6.6132475e-7
        ])
    );
    assert_abs_diff_eq!(
        // Channel 2 -> 1
        vis[(1, 0)],
        Jones::from([
            7.1403125e4,
            -1.3957654e-6,
            -1.0667509e3,
            1.01154564e2,
            -1.0667509e3,
            -1.01154564e2,
            6.7373945e4,
            -7.004146e-7
        ])
    );
    assert_abs_diff_eq!(
        // Channel 16 -> 15
        vis[(15, 0)],
        Jones::from([
            1.07272586e5,
            1.9233863e-8,
            -1.0756711e3,
            -4.4097336e1,
            -1.0756711e3,
            4.4097336e1,
            1.0141891e5,
            -9.5129735e-7
        ])
    );
    assert_abs_diff_eq!(
        // Two flagged tiles before tile 10; use index 8. Channel 16 -> 15.
        vis[(15, 8)],
        Jones::from([
            1.0766406e5,
            1.5415758e-6,
            -1.334196e3,
            -1.8926495e3,
            -1.334196e3,
            1.8926495e3,
            1.21874336e5,
            2.0712553e-6
        ])
    );

    assert_abs_diff_eq!(vis_weights, Array2::from_elem(vis_weights.dim(), 8.0));
}

#[test]
#[serial]
fn read_1090008640_cross_and_auto_vis() {
    let DataAsPathBufs {
        metafits, mut vis, ..
    } = get_reduced_1090008640_ms_pbs();
    let ms_reader = MsReader::new(vis.swap_remove(0), None, Some(&metafits), None).unwrap();

    let obs_context = ms_reader.get_obs_context();
    let total_num_tiles = obs_context.get_total_num_tiles();
    let num_baselines = (total_num_tiles * (total_num_tiles - 1)) / 2;
    let num_chans = usize::from(obs_context.num_fine_chans_per_coarse_chan.unwrap().get());
    let tile_baseline_flags = TileBaselineFlags::new(total_num_tiles, HashSet::new());

    assert_abs_diff_eq!(
        obs_context.timestamps.first().to_gpst_seconds(),
        1090008658.0
    );

    assert_abs_diff_eq!(
        obs_context.timestamps.first().to_gpst_seconds(),
        1090008658.0
    );

    let mut cross_vis = Array2::zeros((num_chans, num_baselines));
    let mut cross_vis_weights = Array2::zeros((num_chans, num_baselines));
    let mut auto_vis = Array2::zeros((num_chans, total_num_tiles));
    let mut auto_vis_weights = Array2::zeros((num_chans, total_num_tiles));
    let result = ms_reader.read_crosses_and_autos(
        cross_vis.view_mut(),
        cross_vis_weights.view_mut(),
        auto_vis.view_mut(),
        auto_vis_weights.view_mut(),
        *obs_context.all_timesteps.first(),
        &tile_baseline_flags,
        &HashSet::new(),
    );
    assert!(result.is_ok(), "{}", result.unwrap_err());

    assert_abs_diff_eq!(
        cross_vis[(0, 0)],
        Jones::from([
            c32::new(-1.2564129e2, -1.497961e1),
            c32::new(8.207059e1, -1.4936417e2),
            c32::new(-7.306871e1, 2.36177e2),
            c32::new(-5.5305626e1, -2.3209404e1)
        ])
    );
    assert_abs_diff_eq!(
        cross_vis[(16, 10)],
        Jones::from([
            c32::new(-4.138127e1, -2.638188e2),
            c32::new(5.220332e2, -2.6055228e2),
            c32::new(4.854074e2, -1.9634505e2),
            c32::new(1.6101791e1, -4.4489478e2),
        ])
    );

    assert_abs_diff_eq!(cross_vis_weights[(2, 11)], -8.0);
    cross_vis_weights[(2, 11)] = 8.0;
    assert_abs_diff_eq!(
        cross_vis_weights,
        Array2::ones(cross_vis_weights.dim()) * 8.0
    );

    assert_abs_diff_eq!(
        auto_vis[(0, 0)],
        Jones::from([
            5.3557855e4,
            4.3092007e-7,
            -7.420802e2,
            1.3296518e2,
            -7.420802e2,
            -1.3296518e2,
            5.084874e4,
            -6.6132475e-7
        ])
    );
    assert_abs_diff_eq!(
        auto_vis[(2, 0)],
        Jones::from([
            7.1403125e4,
            -1.3957654e-6,
            -1.0667509e3,
            1.01154564e2,
            -1.0667509e3,
            -1.01154564e2,
            6.7373945e4,
            -7.004146e-7
        ])
    );
    assert_abs_diff_eq!(
        auto_vis[(16, 0)],
        Jones::from([
            1.07272586e5,
            1.9233863e-8,
            -1.0756711e3,
            -4.4097336e1,
            -1.0756711e3,
            4.4097336e1,
            1.0141891e5,
            -9.5129735e-7
        ])
    );
    assert_abs_diff_eq!(
        auto_vis[(16, 10)],
        Jones::from([
            1.0766406e5,
            1.5415758e-6,
            -1.334196e3,
            -1.8926495e3,
            -1.334196e3,
            1.8926495e3,
            1.21874336e5,
            2.0712553e-6
        ])
    );

    assert_abs_diff_eq!(
        auto_vis_weights,
        Array2::from_elem(auto_vis_weights.dim(), 8.0)
    );
}

#[test]
#[serial]
fn test_timestep_reading() {
    let temp_dir = tempdir().expect("Couldn't make temp dir");
    let vis_path = temp_dir.path().join("vis.ms");

    let num_timesteps = 10;
    let num_channels = 10;
    let ant_pairs = vec![(0, 1), (0, 2), (1, 2)];

    let obsid = 1090000000;

    let vis_ctx = VisContext {
        num_sel_timesteps: num_timesteps,
        start_timestamp: Epoch::from_gpst_seconds(obsid as f64),
        int_time: Duration::from_seconds(1.),
        num_sel_chans: num_channels,
        start_freq_hz: 128_000_000.,
        freq_resolution_hz: 10_000.,
        sel_baselines: ant_pairs,
        avg_time: 1,
        avg_freq: 1,
        num_vis_pols: 4,
    };

    let shape = vis_ctx.sel_dims();

    let vis_data = Array3::<Jones<f32>>::from_shape_fn(shape, |(t, c, b)| {
        let (ant1, ant2) = vis_ctx.sel_baselines[b];
        Jones::from([t as f32, c as f32, ant1 as f32, ant2 as f32, 0., 0., 0., 0.])
    });

    let weight_data = Array3::<f32>::from_elem(shape, 1.);

    let phase_centre = RADec::from_degrees(0., -27.);
    let array_pos = LatLngHeight::mwa();
    #[rustfmt::skip]
    let tile_xyzs = [
        XyzGeodetic { x: 0., y: 0., z: 0., },
        XyzGeodetic { x: 1., y: 0., z: 0., },
        XyzGeodetic { x: 0., y: 1., z: 0., },
    ];
    let tile_names = ["tile_0_0", "tile_1_0", "tile_0_1"];

    let marlu_obs_ctx = MarluObsContext {
        sched_start_timestamp: Epoch::from_gpst_seconds(obsid as f64),
        sched_duration: ((num_timesteps + 1) as f64 * vis_ctx.int_time),
        name: Some(format!("MWA obsid {obsid}")),
        phase_centre,
        pointing_centre: Some(phase_centre),
        array_pos,
        ant_positions_enh: tile_xyzs
            .iter()
            .map(|xyz| xyz.to_enh(array_pos.latitude_rad))
            .collect(),
        ant_names: tile_names.iter().map(|&s| String::from(s)).collect(),
        field_name: None,
        project_id: None,
        observer: None,
    };
    let (s_lat, c_lat) = array_pos.latitude_rad.sin_cos();
    let ant_positions_xyz = marlu_obs_ctx
        .ant_positions_enh
        .iter()
        .map(|enh| enh.to_xyz_inner(s_lat, c_lat))
        .collect();
    let mut writer = MeasurementSetWriter::new(
        &vis_path,
        phase_centre,
        array_pos,
        ant_positions_xyz,
        Duration::from_seconds(0.0),
        true,
    );
    writer.initialize(&vis_ctx, &marlu_obs_ctx, None).unwrap();

    writer
        .write_vis(vis_data.view(), weight_data.view(), &vis_ctx)
        .unwrap();

    let ms_reader = MsReader::new(vis_path, None, None, None).unwrap();
    let ms_ctx = ms_reader.get_obs_context();

    let expected_timestamps = (0..num_timesteps)
        .map(|t| Epoch::from_gpst_seconds((obsid + t) as f64 + 0.5))
        .collect::<Vec<_>>();
    assert_eq!(
        ms_ctx
            .timestamps
            .iter()
            .map(|t| t.to_gpst_seconds())
            .collect::<Vec<_>>(),
        expected_timestamps
            .iter()
            .map(|t| t.to_gpst_seconds())
            .collect::<Vec<_>>()
    );
}

#[test]
#[serial]
fn test_map_metafits_antenna_order() {
    // First, check the delays and gains of the existing test data. Because this
    // MS has its tiles in the same order as the "metafits order", the delays
    // and gains are already correct without re-ordering.
    let DataAsPathBufs {
        metafits, mut vis, ..
    } = get_reduced_1090008640_ms_pbs();
    let ms_pb = vis.swap_remove(0);
    let ms = MsReader::new(ms_pb.clone(), None, Some(&metafits), None).unwrap();
    let obs_context = ms.get_obs_context();
    let delays = match obs_context.dipole_delays.as_ref() {
        Some(Delays::Full(d)) => d,
        _ => unreachable!(),
    };
    // All delays should be 0.
    assert_eq!(delays, Array2::from_elem(delays.dim(), 0));
    // Keep the true gains for later.
    let gains = match obs_context.dipole_gains.as_ref() {
        Some(g) => g,
        _ => unreachable!(),
    };

    // Test that the dipole delays/gains get mapped correctly. As the test MS is
    // already in the same order as the metafits file, the easiest thing to do
    // is to modify the metafits file.
    let metafits_tmp = tempfile::NamedTempFile::new().expect("couldn't make a temp file");
    std::fs::copy(&metafits, metafits_tmp.path()).unwrap();
    unsafe {
        let metafits_c_str = CString::new(metafits_tmp.path().display().to_string())
            .unwrap()
            .into_raw();
        let mut fptr = std::ptr::null_mut();
        let mut status = 0;

        // ffopen = fits_open_file
        fitsio_sys::ffopen(
            &mut fptr,      /* O - FITS file pointer                   */
            metafits_c_str, /* I - full name of file to open           */
            1,              /* I - 0 = open readonly; 1 = read/write   */
            &mut status,    /* IO - error status                       */
        );
        fits_check_status(status).unwrap();
        drop(CString::from_raw(metafits_c_str));
        // ffmahd = fits_movabs_hdu
        fitsio_sys::ffmahd(
            fptr,                 /* I - FITS file pointer             */
            2,                    /* I - number of the HDU to move to  */
            std::ptr::null_mut(), /* O - type of extension, 0, 1, or 2 */
            &mut status,          /* IO - error status                 */
        );
        fits_check_status(status).unwrap();

        // Swap Tile011 (rows 87 and 88) with Tile017 (rows 91 and 92), as
        // Tile017 has a dead dipole but Tile011 doesn't.
        let mut tile_name = CString::new("Tile017").unwrap().into_raw();
        // ffpcls = fits_write_col_str
        fitsio_sys::ffpcls(
            fptr,           /* I - FITS file pointer                       */
            4,              /* I - number of column to write (1 = 1st col) */
            87,             /* I - first row to write (1 = 1st row)        */
            1,              /* I - first vector element to write (1 = 1st) */
            1,              /* I - number of strings to write              */
            &mut tile_name, /* I - array of pointers to strings            */
            &mut status,    /* IO - error status                           */
        );
        fits_check_status(status).unwrap();
        fitsio_sys::ffpcls(
            fptr,           /* I - FITS file pointer                       */
            4,              /* I - number of column to write (1 = 1st col) */
            88,             /* I - first row to write (1 = 1st row)        */
            1,              /* I - first vector element to write (1 = 1st) */
            1,              /* I - number of strings to write              */
            &mut tile_name, /* I - array of pointers to strings            */
            &mut status,    /* IO - error status                           */
        );
        fits_check_status(status).unwrap();
        drop(CString::from_raw(tile_name));

        let mut tile_name = CString::new("Tile011").unwrap().into_raw();
        fitsio_sys::ffpcls(
            fptr,           /* I - FITS file pointer                       */
            4,              /* I - number of column to write (1 = 1st col) */
            91,             /* I - first row to write (1 = 1st row)        */
            1,              /* I - first vector element to write (1 = 1st) */
            1,              /* I - number of strings to write              */
            &mut tile_name, /* I - array of pointers to strings            */
            &mut status,    /* IO - error status                           */
        );
        fits_check_status(status).unwrap();
        fitsio_sys::ffpcls(
            fptr,           /* I - FITS file pointer                       */
            4,              /* I - number of column to write (1 = 1st col) */
            92,             /* I - first row to write (1 = 1st row)        */
            1,              /* I - first vector element to write (1 = 1st) */
            1,              /* I - number of strings to write              */
            &mut tile_name, /* I - array of pointers to strings            */
            &mut status,    /* IO - error status                           */
        );
        fits_check_status(status).unwrap();
        drop(CString::from_raw(tile_name));

        // ffclos = fits_close_file
        fitsio_sys::ffclos(fptr, &mut status);
        fits_check_status(status).unwrap();
    }

    let ms = MsReader::new(ms_pb.clone(), None, Some(metafits_tmp.path()), None).unwrap();
    let obs_context = ms.get_obs_context();
    let delays = match obs_context.dipole_delays.as_ref() {
        Some(Delays::Full(d)) => d,
        _ => unreachable!(),
    };
    // All delays should be 0.
    assert_eq!(delays, Array2::from_elem(delays.dim(), 0));
    let mut perturbed_gains = match obs_context.dipole_gains.as_ref() {
        Some(g) => g.clone(),
        _ => unreachable!(),
    };

    // If the gains are mapped correctly, the before and after gains are
    // different.
    assert_abs_diff_ne!(gains, &perturbed_gains);

    // The first tile's gains of the perturbed metafits
    // (corresponding to Tile017) will have a dead dipole.
    assert_eq!(
        perturbed_gains.slice(s![0, ..]).as_slice().unwrap(),
        [
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
        ]
    );
    // Tile011 is all 1s.
    assert_eq!(
        perturbed_gains.slice(s![6, ..]).as_slice().unwrap(),
        &[1.0; 32]
    );

    // Confirm that the gains are equal when the gain rows are swapped.
    let row1 = perturbed_gains.slice(s![0, ..]).into_owned();
    let row2 = perturbed_gains.slice(s![6, ..]).into_owned();
    perturbed_gains.slice_mut(s![0, ..]).assign(&row2);
    perturbed_gains.slice_mut(s![6, ..]).assign(&row1);
    assert_abs_diff_eq!(gains, &perturbed_gains);

    // Test that the dipole delays/gains aren't mapped when an unknown tile name
    // is encountered.
    let metafits_tmp2 = tempfile::NamedTempFile::new().expect("couldn't make a temp file");
    std::fs::copy(metafits, metafits_tmp2.path()).unwrap();
    unsafe {
        let metafits_c_str = CString::new(metafits_tmp2.path().display().to_string())
            .unwrap()
            .into_raw();
        let mut fptr = std::ptr::null_mut();
        let mut status = 0;

        // ffopen = fits_open_file
        fitsio_sys::ffopen(
            &mut fptr,      /* O - FITS file pointer                   */
            metafits_c_str, /* I - full name of file to open           */
            1,              /* I - 0 = open readonly; 1 = read/write   */
            &mut status,    /* IO - error status                       */
        );
        fits_check_status(status).unwrap();
        drop(CString::from_raw(metafits_c_str));
        // ffmahd = fits_movabs_hdu
        fitsio_sys::ffmahd(
            fptr,                 /* I - FITS file pointer             */
            2,                    /* I - number of the HDU to move to  */
            std::ptr::null_mut(), /* O - type of extension, 0, 1, or 2 */
            &mut status,          /* IO - error status                 */
        );
        fits_check_status(status).unwrap();

        let mut tile_name = CString::new("darkness").unwrap().into_raw();
        // ffpcls = fits_write_col_str
        fitsio_sys::ffpcls(
            fptr,           /* I - FITS file pointer                       */
            4,              /* I - number of column to write (1 = 1st col) */
            87,             /* I - first row to write (1 = 1st row)        */
            1,              /* I - first vector element to write (1 = 1st) */
            1,              /* I - number of strings to write              */
            &mut tile_name, /* I - array of pointers to strings            */
            &mut status,    /* IO - error status                           */
        );
        fits_check_status(status).unwrap();
        fitsio_sys::ffpcls(
            fptr,           /* I - FITS file pointer                       */
            4,              /* I - number of column to write (1 = 1st col) */
            88,             /* I - first row to write (1 = 1st row)        */
            1,              /* I - first vector element to write (1 = 1st) */
            1,              /* I - number of strings to write              */
            &mut tile_name, /* I - array of pointers to strings            */
            &mut status,    /* IO - error status                           */
        );
        fits_check_status(status).unwrap();
        drop(CString::from_raw(tile_name));

        // ffclos = fits_close_file
        fitsio_sys::ffclos(fptr, &mut status);
        fits_check_status(status).unwrap();
    }

    let ms = MsReader::new(ms_pb, None, Some(metafits_tmp2.path()), None).unwrap();
    let obs_context = ms.get_obs_context();
    let delays = match obs_context.dipole_delays.as_ref() {
        Some(Delays::Full(d)) => d,
        _ => unreachable!(),
    };
    // All delays should be 0.
    assert_eq!(delays, Array2::from_elem(delays.dim(), 0));
    let perturbed_gains = match obs_context.dipole_gains.as_ref() {
        Some(g) => g,
        _ => unreachable!(),
    };
    // The gains should be the same as the unaltered-metafits case.
    assert_abs_diff_eq!(gains, perturbed_gains);
}

#[test]
#[serial]
fn test_trunc_data() {
    let expected_num_tiles = 128;
    let expected_unavailable_tiles = (2..128).collect::<Vec<usize>>();

    let result = MsReader::new(
        PathBuf::from("test_files/1090008640/1090008640_cotter_trunc_autos.ms"),
        None,
        None,
        None,
    );
    assert!(result.is_ok(), "{:?}", result.err());
    let reader = result.unwrap();
    let obs_context = reader.get_obs_context();
    let total_num_tiles = obs_context.get_total_num_tiles();
    let num_unflagged_tiles = total_num_tiles - obs_context.flagged_tiles.len();
    assert!(obs_context.autocorrelations_present);
    assert_eq!(total_num_tiles, expected_num_tiles);
    assert_eq!(num_unflagged_tiles, 2);
    assert_eq!(&obs_context.unavailable_tiles, &expected_unavailable_tiles);
    assert_eq!(&obs_context.flagged_tiles, &expected_unavailable_tiles);
    assert_eq!(&obs_context.all_timesteps, &[0, 1, 2]);
    assert_eq!(&obs_context.unflagged_timesteps, &[2]);

    let result = MsReader::new(
        PathBuf::from("test_files/1090008640/1090008640_cotter_trunc_noautos.ms"),
        None,
        None,
        None,
    );
    assert!(result.is_ok(), "{:?}", result.err());
    let reader = result.unwrap();
    let obs_context = reader.get_obs_context();
    let total_num_tiles = obs_context.get_total_num_tiles();
    let num_unflagged_tiles = total_num_tiles - obs_context.flagged_tiles.len();
    assert!(!obs_context.autocorrelations_present);
    assert_eq!(total_num_tiles, expected_num_tiles);
    assert_eq!(num_unflagged_tiles, 2);
    assert_eq!(&obs_context.unavailable_tiles, &expected_unavailable_tiles);
    assert_eq!(&obs_context.flagged_tiles, &expected_unavailable_tiles);
    assert_eq!(&obs_context.all_timesteps, &[0, 1, 2]);
    assert_eq!(&obs_context.unflagged_timesteps, &[2]);

    let result = MsReader::new(
        PathBuf::from("test_files/1090008640/1090008640_birli_trunc.ms"),
        None,
        None,
        None,
    );
    assert!(result.is_ok(), "{:?}", result.err());
    let reader = result.unwrap();
    let obs_context = reader.get_obs_context();
    let total_num_tiles = obs_context.get_total_num_tiles();
    let num_unflagged_tiles = total_num_tiles - obs_context.flagged_tiles.len();
    assert!(obs_context.autocorrelations_present);
    assert_eq!(total_num_tiles, expected_num_tiles);
    assert_eq!(num_unflagged_tiles, 2);
    assert_eq!(&obs_context.unavailable_tiles, &expected_unavailable_tiles);
    assert_eq!(&obs_context.flagged_tiles, &expected_unavailable_tiles);
    assert_eq!(&obs_context.all_timesteps, &[0, 1, 2]);
    assert_eq!(&obs_context.unflagged_timesteps, &[1, 2]);

    // Test that attempting to use all tiles still results in only 2 tiles being available.
    let ms_reader = MsReader::new(
        PathBuf::from("test_files/1090008640/1090008640_birli_trunc.ms"),
        None,
        None,
        None,
    )
    .unwrap();
    assert_eq!(
        ms_reader.get_obs_context().flagged_tiles.len(),
        expected_unavailable_tiles.len()
    );
}

#[test]
fn test_sdc3() {
    let ms = tempdir().unwrap();
    // I hate measurement sets.
    let ms_tarball_temppath = deflate_gz_into_tempfile("test_files/sdc3/sdc3_0000.ms.tar.gz");
    let mut archive = tar::Archive::new(std::fs::File::open(ms_tarball_temppath).unwrap());
    for entry in archive.entries().unwrap() {
        let mut entry = entry.unwrap();
        let path = ms.path().join(entry.header().path().unwrap());
        if path.to_string_lossy().ends_with('/') {
            std::fs::create_dir_all(path).unwrap();
            continue;
        }
        entry.unpack(path).unwrap();
    }

    let ms = MsReader::new(ms.path().join("sdc3_0000.ms"), None, None, None).unwrap();
    let obs_context = ms.get_obs_context();
    assert_eq!(obs_context.timestamps.len(), 1);
    assert_eq!(obs_context.fine_chan_freqs.len(), 1);
    let supplied_array_position = obs_context.supplied_array_position;
    assert_abs_diff_eq!(
        supplied_array_position.longitude_rad.to_degrees(),
        116.76444819999999,
        epsilon = 1e-8
    );
    assert_abs_diff_eq!(
        supplied_array_position.latitude_rad.to_degrees(),
        -26.824722078221356,
        epsilon = 1e-8
    );
    assert_abs_diff_eq!(
        supplied_array_position.height_metres,
        -4.979552228708624e-5,
        epsilon = 1e-8
    );
    assert_eq!(obs_context.polarisations, Polarisations::XX);

    let mut cross_vis = Array2::zeros((1, 1));
    let mut cross_vis_weights = Array2::zeros((1, 1));
    let tile_baseline_flags = TileBaselineFlags::new(512, (2..512).collect());
    ms.read_crosses(
        cross_vis.view_mut(),
        cross_vis_weights.view_mut(),
        0,
        &tile_baseline_flags,
        &HashSet::new(),
    )
    .unwrap();

    // The SDC3 data use ant2-ant1 (casacore) UVWs. The reader no longer
    // conjugates the visibilities to compensate; instead the UVW frame is
    // inferred from the file and the modeller is expected to honour it.
    assert_eq!(obs_context.uvw_frame, UvwFrame::Casacore);
    assert_abs_diff_eq!(
        cross_vis[(0, 0)],
        Jones::from([-8.517027, 7.5777674, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    );
    assert_abs_diff_eq!(cross_vis_weights[(0, 0)], 1.0);
}

/// Write a small, non-MWA measurement set (ASKAP-like 1 MHz channels starting
/// at 744 MHz) without a recorded UVW frame, and then scale its UVW column by
/// `uvw_factor`.
fn write_non_mwa_ms(vis_path: &Path, uvw_factor: f64) {
    let num_timesteps = 2;
    let num_channels = 10;
    let ant_pairs = vec![(0, 1), (0, 2), (1, 2)];
    let start_timestamp = Epoch::from_gpst_seconds(1_400_000_000.0);

    let vis_ctx = VisContext {
        num_sel_timesteps: num_timesteps,
        start_timestamp,
        int_time: Duration::from_seconds(10.),
        num_sel_chans: num_channels,
        start_freq_hz: 744e6,
        freq_resolution_hz: 1e6,
        sel_baselines: ant_pairs,
        avg_time: 1,
        avg_freq: 1,
        num_vis_pols: 4,
    };
    let shape = vis_ctx.sel_dims();
    let vis_data = Array3::from_elem(shape, Jones::identity());
    let weight_data = Array3::<f32>::from_elem(shape, 1.);

    let phase_centre = RADec::from_degrees(98.2119, -30.861);
    let array_pos = LatLngHeight {
        longitude_rad: 116.637_f64.to_radians(),
        latitude_rad: (-26.697_f64).to_radians(),
        height_metres: 360.0,
    };
    #[rustfmt::skip]
    let tile_xyzs = vec![
        XyzGeodetic { x: 0., y: 0., z: 0., },
        XyzGeodetic { x: 100., y: 20., z: 0., },
        XyzGeodetic { x: 30., y: 150., z: 10., },
    ];
    let marlu_obs_ctx = MarluObsContext {
        sched_start_timestamp: start_timestamp,
        sched_duration: num_timesteps as f64 * vis_ctx.int_time,
        name: None,
        phase_centre,
        pointing_centre: Some(phase_centre),
        array_pos,
        ant_positions_enh: tile_xyzs
            .iter()
            .map(|xyz| xyz.to_enh(array_pos.latitude_rad))
            .collect(),
        ant_names: vec!["ak01".into(), "ak02".into(), "ak03".into()],
        field_name: None,
        project_id: None,
        observer: None,
    };
    let mut writer = MeasurementSetWriter::new(
        vis_path,
        phase_centre,
        array_pos,
        tile_xyzs,
        Duration::from_seconds(0.0),
        true,
    );
    writer.initialize(&vis_ctx, &marlu_obs_ctx, None).unwrap();
    writer
        .write_vis(vis_data.view(), weight_data.view(), &vis_ctx)
        .unwrap();

    let mut main_table = Table::open(vis_path, TableOpenMode::ReadWrite).unwrap();
    for i_row in 0..main_table.n_rows() {
        let uvw: Vec<f64> = main_table.get_cell_as_vec("UVW", i_row).unwrap();
        let uvw: Vec<f64> = uvw.into_iter().map(|x| x * uvw_factor).collect();
        main_table.put_cell("UVW", i_row, &uvw).unwrap();
    }
    // Don't record the UVW frame, like MSs that weren't written by Marlu, so
    // the reader has to infer it.
    main_table
        .put_keyword("marlu_uvw_frame", &String::new())
        .unwrap();
}

#[test]
#[serial]
fn test_non_mwa_ms_with_zero_uvws() {
    let temp_dir = tempdir().expect("Couldn't make temp dir");
    let vis_path = temp_dir.path().join("vis.ms");
    write_non_mwa_ms(&vis_path, 0.0);

    // This used to panic, because non-MWA frequencies don't sit nicely in MWA
    // coarse channels.
    let ms_reader = MsReader::new(vis_path, None, None, None).unwrap();
    let obs_context = ms_reader.get_obs_context();
    assert!(obs_context.mwa_coarse_chan_nums.is_none());
    assert!(obs_context.num_fine_chans_per_coarse_chan.is_none());
    assert!(obs_context.flagged_fine_chans_per_coarse_chan.is_none());
    assert_eq!(obs_context.fine_chan_freqs.len(), 10);
    // The UVWs are all zero, so their baseline sign couldn't be checked; the
    // MS default is used.
    assert_eq!(obs_context.uvw_frame, UvwFrame::Casacore);
}

#[test]
#[serial]
fn test_ms_baseline_convention_detection() {
    let temp_dir = tempdir().expect("Couldn't make temp dir");

    let vis_path = temp_dir.path().join("same.ms");
    write_non_mwa_ms(&vis_path, 1.0);
    let ms_reader = MsReader::new(vis_path, None, None, None).unwrap();
    assert_eq!(ms_reader.get_obs_context().uvw_frame, UvwFrame::Hyperdrive);

    let vis_path = temp_dir.path().join("other.ms");
    write_non_mwa_ms(&vis_path, -1.0);
    let ms_reader = MsReader::new(vis_path, None, None, None).unwrap();
    assert_eq!(ms_reader.get_obs_context().uvw_frame, UvwFrame::Casacore);
}

/// A small synthetic non-MWA measurement set whose channel width (5425.35 Hz)
/// is not a whole number of hertz; see `test_files/mini_ms/make_mini_ms.py`.
const MINI_MS: &str = "test_files/mini_ms/mini.ms";

fn copy_dir(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).unwrap();
    for entry in std::fs::read_dir(src).unwrap() {
        let entry = entry.unwrap();
        let to = dst.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_dir(&entry.path(), &to);
        } else {
            std::fs::copy(entry.path(), &to).unwrap();
        }
    }
}

/// The channel width of a measurement set is read as-is, even when it is not
/// a whole number of hertz, and the channel centres are stored to the nearest
/// hertz.
#[test]
#[serial]
fn test_fractional_chan_width() {
    let reader = MsReader::new(PathBuf::from(MINI_MS), None, None, None).unwrap();
    let obs_context = reader.get_obs_context();
    assert_eq!(obs_context.timestamps.len(), 2);
    assert_eq!(obs_context.tile_xyzs.len(), 8);
    assert_eq!(obs_context.fine_chan_freqs.len(), 32);
    assert_abs_diff_eq!(obs_context.freq_res.unwrap(), 5425.35);
    // 1.28 MHz / 5425.35 Hz isn't a whole number, so this isn't MWA data; no
    // coarse channels are inferred (previously, coarse channel 78 with 236
    // fine channels was assumed).
    assert!(obs_context.mwa_coarse_chan_nums.is_none());
    assert!(obs_context.num_fine_chans_per_coarse_chan.is_none());
    assert!(obs_context.flagged_fine_chans_per_coarse_chan.is_none());
    // The UVWs are ant1 - ant2, and the PHASED_ARRAY table makes this OSKAR's
    // frame.
    assert_eq!(obs_context.uvw_frame, UvwFrame::Oskar);
    for (i, &freq) in obs_context.fine_chan_freqs.iter().enumerate() {
        assert_eq!(freq, (100e6 + 5425.35 * i as f64).round() as u64);
    }
    // Successive rounded channel centres are alternately 5425 and 5426 Hz
    // apart; this used to be taken as a "picket fence".
    let diffs: HashSet<u64> = obs_context
        .fine_chan_freqs
        .windows(2)
        .map(|w| w[1] - w[0])
        .collect();
    assert_eq!(diffs, HashSet::from([5425, 5426]));
}

/// Channel widths that differ only by floating-point noise are equal, but
/// genuinely different widths are still an error.
#[test]
#[serial]
fn test_chan_width_noise() {
    let set_widths = |ms: &Path, perturb: f64| {
        let mut t = Table::open(ms.join("SPECTRAL_WINDOW"), TableOpenMode::ReadWrite).unwrap();
        let mut widths: Vec<f64> = t.get_cell_as_vec("CHAN_WIDTH", 0).unwrap();
        for (i, w) in widths.iter_mut().enumerate() {
            *w += perturb * i as f64;
        }
        t.put_cell("CHAN_WIDTH", 0, &widths).unwrap();
    };

    let dir = tempdir().unwrap();
    let noisy = dir.path().join("noisy.ms");
    copy_dir(Path::new(MINI_MS), &noisy);
    set_widths(&noisy, 3e-8);
    assert!(MsReader::new(noisy, None, None, None).is_ok());

    let unequal = dir.path().join("unequal.ms");
    copy_dir(Path::new(MINI_MS), &unequal);
    set_widths(&unequal, 1000.0);
    assert!(matches!(
        MsReader::new(unequal, None, None, None),
        Err(MsReadError::ChanWidthsUnequal)
    ));
}

/// Copy `test_files/mini_ms/mini.ms` and multiply its UVW column by
/// `uvw_factor`.
fn copy_mini_ms_with_uvw_factor(dst: &Path, uvw_factor: f64) {
    copy_dir(Path::new(MINI_MS), dst);
    let mut main_table = Table::open(dst, TableOpenMode::ReadWrite).unwrap();
    for i_row in 0..main_table.n_rows() {
        let uvw: Vec<f64> = main_table.get_cell_as_vec("UVW", i_row).unwrap();
        let uvw: Vec<f64> = uvw.into_iter().map(|x| x * uvw_factor).collect();
        main_table.put_cell("UVW", i_row, &uvw).unwrap();
    }
}

/// The baseline convention of a real (non-MWA) measurement set is detected
/// from its UVWs: negated UVWs are the casacore frame, and all-zero UVWs (as
/// some simulators write) can't be checked, so the PHASED_ARRAY table decides
/// (OSKAR's frame, which isn't conjugated).
#[test]
#[serial]
fn test_mini_ms_uvw_conventions() {
    let temp_dir = tempdir().expect("Couldn't make temp dir");

    let other = temp_dir.path().join("other.ms");
    copy_mini_ms_with_uvw_factor(&other, -1.0);
    let reader = MsReader::new(other, None, None, None).unwrap();
    assert_eq!(reader.get_obs_context().uvw_frame, UvwFrame::Casacore);

    let zero = temp_dir.path().join("zero.ms");
    copy_mini_ms_with_uvw_factor(&zero, 0.0);
    let reader = MsReader::new(zero, None, None, None).unwrap();
    assert_eq!(reader.get_obs_context().uvw_frame, UvwFrame::Oskar);
    assert_eq!(reader.get_obs_context().fine_chan_freqs.len(), 32);
}

/// A fully-flagged timestep between unflagged ones must not truncate the
/// timesteps; only leading and trailing flagged timesteps are excluded.
#[test]
#[serial]
fn test_flagged_timestep_in_the_middle() {
    let dir = tempdir().unwrap();
    let ms = dir.path().join("flagged.ms");
    copy_dir(
        Path::new("test_files/1090008640/1090008640_cotter_trunc_noautos.ms"),
        &ms,
    );
    // This MS has one baseline (row) per timestep.
    {
        let mut t = Table::open(&ms, TableOpenMode::ReadWrite).unwrap();
        for (row, flagged) in [(0, false), (1, true), (2, false)] {
            let num_flags = t.get_cell_as_vec::<bool>("FLAG", row).unwrap().len() / 4;
            t.put_cell("FLAG", row, &Array2::from_elem((num_flags, 4), flagged))
                .unwrap();
        }
    }

    let reader = MsReader::new(ms, None, None, None).unwrap();
    assert_eq!(&reader.get_obs_context().unflagged_timesteps, &[0, 1, 2]);
}

/// The visibilities read from a measurement set with fewer than 4
/// polarisations are put in the right place of the Jones matrix.
#[test]
fn test_pol_transform() {
    let c = |re: f32, im: f32| num_complex::Complex32::new(re, im);
    let c0 = c(0.0, 0.0);
    // The read visibilities are packed into the first elements.
    let j = Jones::from([c(1.0, 1.0), c(2.0, 2.0), c(3.0, 3.0), c(4.0, 4.0)]);
    for (pols, expected) in [
        (Polarisations::XX_XY_YX_YY, j),
        (
            Polarisations::XX,
            j, // only the first element is meaningful
        ),
        (Polarisations::YY, Jones::from([c0, c0, c0, j[0]])),
        (Polarisations::XX_YY, Jones::from([j[0], c0, c0, j[1]])),
        (Polarisations::XX_YY_XY, Jones::from([j[0], j[1], c0, j[2]])),
    ] {
        assert_eq!(PolTransform::new(pols).apply(j), expected);
    }
}

#[test]
#[serial]
fn test_ms_without_receptor_angles() {
    let temp_dir = tempdir().expect("Couldn't make temp dir");
    let vis_path = temp_dir.path().join("vis.ms");
    write_non_mwa_ms(&vis_path, 1.0);
    let with_angles = MsReader::new(vis_path.clone(), None, None, None)
        .unwrap()
        .get_obs_context()
        .pol_convention;

    Table::open(vis_path.join("FEED"), TableOpenMode::ReadWrite)
        .unwrap()
        .remove_column("RECEPTOR_ANGLE")
        .unwrap();
    let without_angles = MsReader::new(vis_path, None, None, None)
        .unwrap()
        .get_obs_context()
        .pol_convention;
    // Without the recorded feed angles, non-MWA data default to X north-south.
    assert_eq!(with_angles, PolConvention::MWA);
    assert_eq!(without_angles, PolConvention::IAU);
}
