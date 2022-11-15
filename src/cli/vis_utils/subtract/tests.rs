// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::{
    fs::File,
    io::{BufWriter, Write},
};

use approx::assert_abs_diff_eq;
use clap::Parser;
use marlu::RADec;
use mwalib::{_open_fits, _open_hdu, fits_open, fits_open_hdu};
use tempfile::TempDir;
use vec1::vec1;

use super::*;
use crate::{
    cli::vis_utils::simulate::VisSimulateArgs,
    srclist::{ComponentType, FluxDensity, FluxDensityType, Source, SourceComponent, SourceList},
    tests::reduced_obsids::get_reduced_1090008640,
};

/// Simulate visibilities for two points sources, then test vis-subtract by
/// subtracting one and both of them.
#[test]
fn test_1090008640_vis_subtract() {
    let num_timesteps = 2;
    let num_chans = 5;

    let temp_dir = TempDir::new().expect("couldn't make tmp dir");
    let subtracted = temp_dir.path().join("subtracted.uvfits");

    let mut args = get_reduced_1090008640(false, false);
    args.no_beam = true;
    let metafits = args.data.as_ref().unwrap()[0].as_str();
    let mut srclist = SourceList::new();
    srclist.insert(
        "src1".to_string(),
        Source {
            components: vec1![SourceComponent {
                radec: RADec::from_degrees(0.0, -27.0),
                comp_type: ComponentType::Point,
                flux_type: FluxDensityType::List {
                    fds: vec1![FluxDensity {
                        freq: 150e6,
                        i: 1.0,
                        ..Default::default()
                    }],
                },
            }],
        },
    );
    // Write out this 1-source source list for comparison.
    let source_list_1 = temp_dir.path().join("srclist_1.yaml");
    let mut f = BufWriter::new(File::create(&source_list_1).unwrap());
    crate::srclist::hyperdrive::source_list_to_yaml(&mut f, &srclist, None).unwrap();
    f.flush().unwrap();
    let model_1 = temp_dir.path().join("model_1.uvfits");

    srclist.insert(
        "src2".to_string(),
        Source {
            components: vec1![SourceComponent {
                radec: RADec::from_degrees(1.0, -27.0),
                comp_type: ComponentType::Point,
                flux_type: FluxDensityType::List {
                    fds: vec1![FluxDensity {
                        freq: 150e6,
                        i: 1.0,
                        ..Default::default()
                    }],
                },
            }],
        },
    );
    let source_list_2 = temp_dir.path().join("srclist_2.yaml");
    let mut f = BufWriter::new(File::create(&source_list_2).unwrap());
    crate::srclist::hyperdrive::source_list_to_yaml(&mut f, &srclist, None).unwrap();
    f.flush().unwrap();
    let model_2 = temp_dir.path().join("model_2.uvfits");

    // Generate visibilities for the 1- and 2-source source lists.
    #[rustfmt::skip]
    let sim_args = VisSimulateArgs::parse_from([
        "vis-simulate",
        "--metafits", metafits,
        "--source-list", &format!("{}", source_list_1.display()),
        "--output-model-files", &format!("{}", model_1.display()),
        "--num-timesteps", &format!("{num_timesteps}"),
        "--num-fine-channels", &format!("{num_chans}"),
        "--no-progress-bars"
    ]);
    let result = sim_args.run(false);
    assert!(result.is_ok(), "result={:?} not ok", result.err().unwrap());

    #[rustfmt::skip]
    let sim_args = VisSimulateArgs::parse_from([
        "vis-simulate",
        "--metafits", metafits,
        "--source-list", &format!("{}", source_list_2.display()),
        "--output-model-files", &format!("{}", model_2.display()),
        "--num-timesteps", &format!("{num_timesteps}"),
        "--num-fine-channels", &format!("{num_chans}"),
        "--no-progress-bars"
    ]);
    let result = sim_args.run(false);
    assert!(result.is_ok(), "result={:?} not ok", result.err().unwrap());

    {
        // We expect that when we subtract src2 from model_2, we get
        // visibilities similar to model_1.
        #[rustfmt::skip]
        let sub_args = VisSubtractArgs::parse_from([
            "vis-subtract",
            "--data", metafits, &format!("{}", model_2.display()),
            "--outputs", &format!("{}", subtracted.display()),
            "--source-list", &format!("{}", source_list_2.display()),
            "--sources-to-subtract", "src2",
            "--no-progress-bars",
        ]);
        let result = sub_args.run(false);
        assert!(result.is_ok(), "result={:?} not ok", result.err().unwrap());

        let mut uvfits_1 = fits_open!(&model_1).unwrap();
        fits_open_hdu!(&mut uvfits_1, 0).unwrap();
        let mut group_params = [0.0; 5];
        let mut vis_1: Vec<f32> = vec![0.0; num_chans * 4 * 3];
        let mut status = 0;
        unsafe {
            // ffggpe = fits_read_grppar_flt
            fitsio_sys::ffggpe(
                uvfits_1.as_raw(),         /* I - FITS file pointer                       */
                1,                         /* I - group to read (1 = 1st group)           */
                1,                         /* I - first vector element to read (1 = 1st)  */
                group_params.len() as i64, /* I - number of values to read                */
                group_params.as_mut_ptr(), /* O - array of values that are returned       */
                &mut status,               /* IO - error status                           */
            );
            assert_eq!(status, 0, "Status wasn't 0");
            // ffgpve = fits_read_img_flt
            fitsio_sys::ffgpve(
                uvfits_1.as_raw(),  /* I - FITS file pointer                       */
                1,                  /* I - group to read (1 = 1st group)           */
                1,                  /* I - first vector element to read (1 = 1st)  */
                vis_1.len() as i64, /* I - number of values to read                */
                0.0,                /* I - value for undefined pixels              */
                vis_1.as_mut_ptr(), /* O - array of values that are returned       */
                &mut 0,             /* O - set to 1 if any values are null; else 0 */
                &mut status,        /* IO - error status                           */
            );
            assert_eq!(status, 0, "Status wasn't 0");
        };

        let mut uvfits_2 = fits_open!(&subtracted).unwrap();
        fits_open_hdu!(&mut uvfits_2, 0).unwrap();
        let mut vis_2: Vec<f32> = vec![0.0; num_chans * 4 * 3];
        unsafe {
            // ffggpe = fits_read_grppar_flt
            fitsio_sys::ffggpe(
                uvfits_2.as_raw(),         /* I - FITS file pointer                       */
                1,                         /* I - group to read (1 = 1st group)           */
                1,                         /* I - first vector element to read (1 = 1st)  */
                group_params.len() as i64, /* I - number of values to read                */
                group_params.as_mut_ptr(), /* O - array of values that are returned       */
                &mut status,               /* IO - error status                           */
            );
            assert_eq!(status, 0, "Status wasn't 0");
            // ffgpve = fits_read_img_flt
            fitsio_sys::ffgpve(
                uvfits_2.as_raw(),  /* I - FITS file pointer                       */
                1,                  /* I - group to read (1 = 1st group)           */
                1,                  /* I - first vector element to read (1 = 1st)  */
                vis_2.len() as i64, /* I - number of values to read                */
                0.0,                /* I - value for undefined pixels              */
                vis_2.as_mut_ptr(), /* O - array of values that are returned       */
                &mut 0,             /* O - set to 1 if any values are null; else 0 */
                &mut status,        /* IO - error status                           */
            );
            assert_eq!(status, 0, "Status wasn't 0");
        };

        assert_abs_diff_eq!(&vis_1[..], &vis_2[..]);
    }

    {
        // We should also have zeros for visibilities if we subtract both sources.
        #[rustfmt::skip]
        let sub_args = VisSubtractArgs::parse_from([
            "vis-subtract",
            "--data", metafits, &format!("{}", model_2.display()),
            "--outputs", &format!("{}", subtracted.display()),
            "--source-list", &format!("{}", source_list_2.display()),
            "--sources-to-subtract", "src1", "src2",
            "--no-progress-bars",
        ]);
        let result = sub_args.run(false);
        assert!(result.is_ok(), "result={:?} not ok", result.err().unwrap());

        let mut uvfits = fits_open!(&subtracted).unwrap();
        fits_open_hdu!(&mut uvfits, 0).unwrap();
        let mut group_params = [0.0; 5];
        let mut vis: Vec<f32> = vec![0.0; num_chans * 4 * 3];
        let mut status = 0;
        unsafe {
            // ffggpe = fits_read_grppar_flt
            fitsio_sys::ffggpe(
                uvfits.as_raw(),           /* I - FITS file pointer                       */
                1,                         /* I - group to read (1 = 1st group)           */
                1,                         /* I - first vector element to read (1 = 1st)  */
                group_params.len() as i64, /* I - number of values to read                */
                group_params.as_mut_ptr(), /* O - array of values that are returned       */
                &mut status,               /* IO - error status                           */
            );
            assert_eq!(status, 0, "Status wasn't 0");
            // ffgpve = fits_read_img_flt
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

        let mut expected = vec![0.0; num_chans * 4 * 3];
        expected.chunks_exact_mut(3).for_each(|c| c[2] = 64.0);
        assert_abs_diff_eq!(&vis[..], &expected[..]);
    }
}
