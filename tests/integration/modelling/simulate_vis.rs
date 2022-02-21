// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Integration tests for sky-model visibilities generated by the "simulate-vis"
//! subcommand of hyperdrive.

use approx::assert_abs_diff_eq;
use marlu::{XyzGeodetic, ENH};
use mwalib::{fitsio_sys, *};
use serial_test::serial;

use crate::*;
use mwa_hyperdrive_common::{cfg_if, marlu, mwalib};

fn read_uvfits_stabxyz(
    fptr: &mut fitsio::FitsFile,
    _hdu: &fitsio::hdu::FitsHdu,
    num_tiles: usize,
) -> Vec<XyzGeodetic> {
    unsafe {
        // With the column name, get the column number.
        let mut status = 0;
        let mut col_num = -1;
        let keyword = std::ffi::CString::new("STABXYZ").unwrap();
        fitsio_sys::ffgcno(
            fptr.as_raw(),
            0,
            keyword.as_ptr() as *mut std::os::raw::c_char,
            &mut col_num,
            &mut status,
        );
        assert_eq!(status, 0, "Status wasn't 0");

        // Now get the column data.
        let mut array = vec![XyzGeodetic::default(); num_tiles];
        let array_ptr = array.as_mut_ptr();
        fitsio_sys::ffgcv(
            fptr.as_raw(),
            82, // TDOUBLE
            col_num,
            1,
            1,
            (num_tiles * 3) as i64,
            std::ptr::null_mut(),
            array_ptr as *mut core::ffi::c_void,
            &mut 0,
            &mut status,
        );
        assert_eq!(status, 0, "Status wasn't 0");
        array
    }
}

#[test]
#[serial]
fn test_1090008640_simulate_vis() {
    let num_timesteps = 2;
    let num_chans = 10;

    let mut output_path = TempDir::new().expect("couldn't make tmp dir").into_path();
    output_path.push("model.uvfits");
    let args = get_reduced_1090008640(true, false);
    let metafits = args.data.as_ref().unwrap()[0].clone();

    let cmd = hyperdrive()
        .args(&[
            "simulate-vis",
            "--metafits",
            &metafits,
            "--source-list",
            &args.source_list.unwrap(),
            "--output-model-file",
            &format!("{}", output_path.display()),
            "--num-timesteps",
            &format!("{}", num_timesteps),
            "--num-fine-channels",
            &format!("{}", num_chans),
        ])
        .ok();
    assert!(cmd.is_ok(), "{:?}", get_cmd_output(cmd));

    // Test some metadata. Compare with the input metafits file.
    let metafits = MetafitsContext::new(&metafits, None).unwrap();
    let mut uvfits = fits_open!(&output_path).unwrap();
    let hdu = fits_open_hdu!(&mut uvfits, 0).unwrap();
    let gcount: String = get_required_fits_key!(&mut uvfits, &hdu, "GCOUNT").unwrap();
    let pcount: String = get_required_fits_key!(&mut uvfits, &hdu, "PCOUNT").unwrap();
    let floats_per_pol: String = get_required_fits_key!(&mut uvfits, &hdu, "NAXIS2").unwrap();
    let num_pols: String = get_required_fits_key!(&mut uvfits, &hdu, "NAXIS3").unwrap();
    let num_fine_freq_chans: String = get_required_fits_key!(&mut uvfits, &hdu, "NAXIS4").unwrap();
    let jd_zero: String = get_required_fits_key!(&mut uvfits, &hdu, "PZERO5").unwrap();
    let ptype4: String = get_required_fits_key!(&mut uvfits, &hdu, "PTYPE4").unwrap();

    assert_eq!(gcount.parse::<i32>().unwrap(), 16256);
    assert_eq!(pcount.parse::<i32>().unwrap(), 5);
    assert_eq!(floats_per_pol.parse::<i32>().unwrap(), 3);
    assert_eq!(num_pols.parse::<i32>().unwrap(), 4);
    assert_eq!(num_fine_freq_chans.parse::<i32>().unwrap(), 10);
    let jd_zero = jd_zero.parse::<f64>().unwrap();
    assert_abs_diff_eq!(jd_zero, 2.456860500E+06);
    assert_eq!(ptype4, "BASELINE");

    let hdu = fits_open_hdu!(&mut uvfits, 1).unwrap();
    let tile_names: Vec<String> = get_fits_col!(&mut uvfits, &hdu, "ANNAME").unwrap();
    assert_eq!(tile_names.len(), 128);
    assert_eq!(tile_names[0], "Tile011");
    assert_eq!(tile_names[1], "Tile012");
    assert_eq!(tile_names[127], "Tile168");
    for (i, (tile_name, metafits_tile_name)) in tile_names
        .iter()
        .zip(
            metafits
                .rf_inputs
                .iter()
                .filter(|rf| rf.pol == Pol::X)
                .map(|rf| &rf.tile_name),
        )
        .enumerate()
    {
        assert_eq!(tile_name, metafits_tile_name, "Wrong for tile {i}");
    }

    let tile_positions = read_uvfits_stabxyz(&mut uvfits, &hdu, 128);
    assert_abs_diff_eq!(tile_positions[0].x, 456.2500494643639);
    assert_abs_diff_eq!(tile_positions[0].y, -149.78500366210938);
    assert_abs_diff_eq!(tile_positions[0].z, 68.04598669887378);
    assert_abs_diff_eq!(tile_positions[10].x, 464.8409142556812);
    assert_abs_diff_eq!(tile_positions[10].y, -123.66699981689453);
    assert_abs_diff_eq!(tile_positions[10].z, 85.0377637878737);
    for (tile_pos, metafits_tile_pos) in
        tile_positions
            .into_iter()
            .zip(
                metafits
                    .rf_inputs
                    .iter()
                    .filter(|rf| rf.pol == Pol::X)
                    .map(|rf| {
                        ENH {
                            e: rf.east_m,
                            n: rf.north_m,
                            h: rf.height_m,
                        }
                        .to_xyz_mwa()
                    }),
            )
    {
        assert_abs_diff_eq!(tile_pos.x, metafits_tile_pos.x);
        assert_abs_diff_eq!(tile_pos.y, metafits_tile_pos.y);
        assert_abs_diff_eq!(tile_pos.z, metafits_tile_pos.z);
    }

    // Test visibility values.
    fits_open_hdu!(&mut uvfits, 0).unwrap();
    let mut group_params = [0.0; 5];
    let mut vis: Vec<f32> = vec![0.0; 10 * 4 * 3];
    let mut status = 0;
    unsafe {
        fitsio_sys::ffggpe(
            uvfits.as_raw(),           /* I - FITS file pointer                       */
            1,                         /* I - group to read (1 = 1st group)           */
            1,                         /* I - first vector element to read (1 = 1st)  */
            group_params.len() as i64, /* I - number of values to read                */
            group_params.as_mut_ptr(), /* O - array of values that are returned       */
            &mut status,               /* IO - error status                           */
        );
        assert_eq!(status, 0, "Status wasn't 0");
        fitsio_sys::ffgpve(
            uvfits.as_raw(),  /* I - FITS file pointer                       */
            1,                /* I - group to read (1 = 1st group)           */
            1,                /* I - first vector element to read (1 = 1st)  */
            vis.len() as i64, /* I - number of values to read                */
            0.0,              /* I - value for undefined pixels              */
            vis.as_mut_ptr(), /* O - array of values that are returned       */
            &mut 0,           /* O - set to 1 if any values are null; else 0 */
            &mut status,      /* IO - error status                           */
        );
        assert_eq!(status, 0, "Status wasn't 0");
    };

    assert_abs_diff_eq!(group_params[0], -1.8128954e-7);
    assert_abs_diff_eq!(group_params[1], -1.6615635e-8);
    assert_abs_diff_eq!(group_params[2], -4.8240993e-9);
    assert_abs_diff_eq!(group_params[3], 258.0);
    assert_abs_diff_eq!(group_params[4], -0.15944445);
    assert_abs_diff_eq!(group_params[4] as f64 + jd_zero, 2456860.3405555487);

    // The values of the visibilities changes slightly depending on the precision.
    cfg_if::cfg_if! {
        if #[cfg(feature = "cuda-single")] {
            assert_abs_diff_eq!(vis[0], 36.808147);
            assert_abs_diff_eq!(vis[1], -37.754723);
            assert_abs_diff_eq!(vis[3], 36.533882);
            assert_abs_diff_eq!(vis[4], -37.974045);
            assert_abs_diff_eq!(vis[6], 0.12816785);
            assert_abs_diff_eq!(vis[7], -0.07708679);
            assert_abs_diff_eq!(vis[9], 0.13573173);
            assert_abs_diff_eq!(vis[10], -0.052073088);
            assert_abs_diff_eq!(vis[12], 36.744892);
            assert_abs_diff_eq!(vis[13], -37.803474);
            assert_abs_diff_eq!(vis[15], 36.480362);
            assert_abs_diff_eq!(vis[16], -38.029716);
            assert_abs_diff_eq!(vis[18], 0.13181247);
            assert_abs_diff_eq!(vis[19], -0.07537043);
            assert_abs_diff_eq!(vis[21], 0.13933058);
            assert_abs_diff_eq!(vis[22], -0.050386995);
            assert_abs_diff_eq!(vis[24], 36.678135);
            assert_abs_diff_eq!(vis[25], -37.850082);
            assert_abs_diff_eq!(vis[27], 36.42348);
            assert_abs_diff_eq!(vis[28], -38.083344);
        } else {
            assert_abs_diff_eq!(vis[0], 36.808372);
            assert_abs_diff_eq!(vis[1], -37.75457);
            assert_abs_diff_eq!(vis[3], 36.53414);
            assert_abs_diff_eq!(vis[4], -37.973904);
            assert_abs_diff_eq!(vis[6], 0.12817207);
            assert_abs_diff_eq!(vis[7], -0.07708947);
            assert_abs_diff_eq!(vis[9], 0.1357368);
            assert_abs_diff_eq!(vis[10], -0.052075468);
            assert_abs_diff_eq!(vis[12], 36.745102);
            assert_abs_diff_eq!(vis[13], -37.803284);
            assert_abs_diff_eq!(vis[15], 36.480606);
            assert_abs_diff_eq!(vis[16], -38.02954);
            assert_abs_diff_eq!(vis[18], 0.13181655);
            assert_abs_diff_eq!(vis[19], -0.07537329);
            assert_abs_diff_eq!(vis[21], 0.13933551);
            assert_abs_diff_eq!(vis[22], -0.050389454);
            assert_abs_diff_eq!(vis[24], 36.67836);
            assert_abs_diff_eq!(vis[25], -37.84991);
            assert_abs_diff_eq!(vis[27], 36.423744);
            assert_abs_diff_eq!(vis[28], -38.08319);
        }
    }
    // Every third value (a weight) should be 1.
    for (i, vis) in vis.iter().enumerate() {
        if i % 3 == 2 {
            assert_abs_diff_eq!(*vis, 1.0);
        }
    }

    unsafe {
        fitsio_sys::ffggpe(
            uvfits.as_raw(),           /* I - FITS file pointer                       */
            8129,                      /* I - group to read (1 = 1st group)           */
            1,                         /* I - first vector element to read (1 = 1st)  */
            group_params.len() as i64, /* I - number of values to read                */
            group_params.as_mut_ptr(), /* O - array of values that are returned       */
            &mut status,               /* IO - error status                           */
        );
        assert_eq!(status, 0, "Status wasn't 0");
        fitsio_sys::ffgpve(
            uvfits.as_raw(),  /* I - FITS file pointer                       */
            8129,             /* I - group to read (1 = 1st group)           */
            1,                /* I - first vector element to read (1 = 1st)  */
            vis.len() as i64, /* I - number of values to read                */
            0.0,              /* I - value for undefined pixels              */
            vis.as_mut_ptr(), /* O - array of values that are returned       */
            &mut 0,           /* O - set to 1 if any values are null; else 0 */
            &mut status,      /* IO - error status                           */
        );
        assert_eq!(status, 0, "Status wasn't 0");
    };

    assert_abs_diff_eq!(group_params[0], -1.8129641e-7);
    assert_abs_diff_eq!(group_params[1], -1.6567755e-8);
    assert_abs_diff_eq!(group_params[2], -4.729797e-9);
    assert_abs_diff_eq!(group_params[3], 258.0);
    assert_abs_diff_eq!(group_params[4], -0.15935186);
    assert_abs_diff_eq!(group_params[4] as f64 + jd_zero, 2456860.3406481445);

    cfg_if::cfg_if! {
        if #[cfg(feature = "cuda-single")] {
            assert_abs_diff_eq!(vis[0], 36.86735);
            assert_abs_diff_eq!(vis[1], -37.659954);
            assert_abs_diff_eq!(vis[3], 36.58934);
            assert_abs_diff_eq!(vis[4], -37.86888);
            assert_abs_diff_eq!(vis[6], 0.12899087);
            assert_abs_diff_eq!(vis[7], -0.0773094);
            assert_abs_diff_eq!(vis[9], 0.13671538);
            assert_abs_diff_eq!(vis[10], -0.052378107);
            assert_abs_diff_eq!(vis[12], 36.801903);
            assert_abs_diff_eq!(vis[13], -37.709015);
            assert_abs_diff_eq!(vis[15], 36.533394);
            assert_abs_diff_eq!(vis[16], -37.92491);
            assert_abs_diff_eq!(vis[18], 0.13262466);
            assert_abs_diff_eq!(vis[19], -0.075590976);
            assert_abs_diff_eq!(vis[21], 0.14030261);
            assert_abs_diff_eq!(vis[22], -0.050690085);
            assert_abs_diff_eq!(vis[24], 36.732975);
            assert_abs_diff_eq!(vis[25], -37.755962);
            assert_abs_diff_eq!(vis[27], 36.47412);
            assert_abs_diff_eq!(vis[28], -37.978943);
        } else {
            assert_abs_diff_eq!(vis[0], 36.867413);
            assert_abs_diff_eq!(vis[1], -37.65978);
            assert_abs_diff_eq!(vis[3], 36.589394);
            assert_abs_diff_eq!(vis[4], -37.868717);
            assert_abs_diff_eq!(vis[6], 0.12899569);
            assert_abs_diff_eq!(vis[7], -0.07731515);
            assert_abs_diff_eq!(vis[9], 0.13671657);
            assert_abs_diff_eq!(vis[10], -0.05238175);
            assert_abs_diff_eq!(vis[12], 36.801952);
            assert_abs_diff_eq!(vis[13], -37.7088);
            assert_abs_diff_eq!(vis[15], 36.533443);
            assert_abs_diff_eq!(vis[16], -37.924717);
            assert_abs_diff_eq!(vis[18], 0.13262925);
            assert_abs_diff_eq!(vis[19], -0.07559694);
            assert_abs_diff_eq!(vis[21], 0.1403035);
            assert_abs_diff_eq!(vis[22], -0.050693996);
            assert_abs_diff_eq!(vis[24], 36.733006);
            assert_abs_diff_eq!(vis[25], -37.755737);
            assert_abs_diff_eq!(vis[27], 36.474148);
            assert_abs_diff_eq!(vis[28], -37.978733);
        }
    }
    for (i, vis) in vis.iter().enumerate() {
        if i % 3 == 2 {
            assert_abs_diff_eq!(*vis, 1.0);
        }
    }
}

// Ensure that visibilities generated by double-precision CUDA and the CPU are
// exactly the same.
#[test]
#[serial]
#[cfg(all(feature = "cuda", not(feature = "cuda-single")))]
fn test_1090008640_simulate_vis_cpu_gpu_match() {
    let num_timesteps = 2;
    let num_chans = 10;

    let mut output_path = TempDir::new().expect("couldn't make tmp dir").into_path();
    output_path.push("model.uvfits");
    let args = get_reduced_1090008640(true, false);
    let metafits = args.data.as_ref().unwrap()[0].clone();
    let cmd = hyperdrive()
        .args(&[
            "simulate-vis",
            "--metafits",
            &metafits,
            "--source-list",
            &args.source_list.unwrap(),
            "--output-model-file",
            &format!("{}", output_path.display()),
            "--num-timesteps",
            &format!("{}", num_timesteps),
            "--num-fine-channels",
            &format!("{}", num_chans),
            "--cpu",
        ])
        .ok();
    assert!(cmd.is_ok(), "{:?}", get_cmd_output(cmd));

    let mut uvfits = fits_open!(&output_path).unwrap();
    let hdu = fits_open_hdu!(&mut uvfits, 0).unwrap();

    let mut group_params = [0.0; 5];
    let mut vis_cpu: Vec<f32> = vec![0.0; 10 * 4 * 3];
    let mut status = 0;
    unsafe {
        fitsio_sys::ffggpe(
            uvfits.as_raw(),           /* I - FITS file pointer                       */
            1,                         /* I - group to read (1 = 1st group)           */
            1,                         /* I - first vector element to read (1 = 1st)  */
            group_params.len() as i64, /* I - number of values to read                */
            group_params.as_mut_ptr(), /* O - array of values that are returned       */
            &mut status,               /* IO - error status                           */
        );
        assert_eq!(status, 0, "Status wasn't 0");
        fitsio_sys::ffgpve(
            uvfits.as_raw(),      /* I - FITS file pointer                       */
            1,                    /* I - group to read (1 = 1st group)           */
            1,                    /* I - first vector element to read (1 = 1st)  */
            vis_cpu.len() as i64, /* I - number of values to read                */
            0.0,                  /* I - value for undefined pixels              */
            vis_cpu.as_mut_ptr(), /* O - array of values that are returned       */
            &mut 0,               /* O - set to 1 if any values are null; else 0 */
            &mut status,          /* IO - error status                           */
        );
        assert_eq!(status, 0, "Status wasn't 0");
    };
    drop(hdu);
    drop(uvfits);

    let args = get_reduced_1090008640(true, false);
    let metafits = args.data.as_ref().unwrap()[0].clone();
    let cmd = hyperdrive()
        .args(&[
            "simulate-vis",
            "--metafits",
            &metafits,
            "--source-list",
            &args.source_list.unwrap(),
            "--output-model-file",
            &format!("{}", output_path.display()),
            "--num-timesteps",
            &format!("{}", num_timesteps),
            "--num-fine-channels",
            &format!("{}", num_chans),
        ])
        .ok();
    assert!(cmd.is_ok(), "{:?}", get_cmd_output(cmd));

    let mut uvfits = fits_open!(&output_path).unwrap();
    let hdu = fits_open_hdu!(&mut uvfits, 0).unwrap();

    let mut vis_gpu: Vec<f32> = vec![0.0; 10 * 4 * 3];
    unsafe {
        fitsio_sys::ffggpe(
            uvfits.as_raw(),           /* I - FITS file pointer                       */
            1,                         /* I - group to read (1 = 1st group)           */
            1,                         /* I - first vector element to read (1 = 1st)  */
            group_params.len() as i64, /* I - number of values to read                */
            group_params.as_mut_ptr(), /* O - array of values that are returned       */
            &mut status,               /* IO - error status                           */
        );
        assert_eq!(status, 0, "Status wasn't 0");
        fitsio_sys::ffgpve(
            uvfits.as_raw(),      /* I - FITS file pointer                       */
            1,                    /* I - group to read (1 = 1st group)           */
            1,                    /* I - first vector element to read (1 = 1st)  */
            vis_gpu.len() as i64, /* I - number of values to read                */
            0.0,                  /* I - value for undefined pixels              */
            vis_gpu.as_mut_ptr(), /* O - array of values that are returned       */
            &mut 0,               /* O - set to 1 if any values are null; else 0 */
            &mut status,          /* IO - error status                           */
        );
        assert_eq!(status, 0, "Status wasn't 0");
    };
    drop(hdu);
    drop(uvfits);

    for (cpu, gpu) in vis_cpu.into_iter().zip(vis_gpu) {
        assert_abs_diff_eq!(cpu, gpu)
    }
}