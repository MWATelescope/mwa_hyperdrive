// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::{
    num::NonZeroU16,
    path::{Path, PathBuf},
};

use clap::Parser;
use tempfile::TempDir;

use super::VisConvertArgs;
use crate::{
    io::read::VisRead,
    tests::{get_reduced_1090008640_raw, DataAsStrings},
    MsReader, UvfitsReader,
};

#[test]
fn test_per_coarse_chan_flags_and_smallest_contiguous_band_writing() {
    let temp_dir = TempDir::new().expect("couldn't make tmp dir");
    let uvfits_converted = temp_dir.path().join("converted.uvfits");
    let ms_converted = temp_dir.path().join("converted.ms");

    fn get_data_object(
        output: &Path,
        per_coarse_chan_flags: Option<Vec<String>>,
        output_smallest_contiguous_band: bool,
    ) -> Box<dyn VisRead> {
        let DataAsStrings {
            metafits,
            vis,
            mwafs: _,
            srclist: _,
        } = get_reduced_1090008640_raw();
        let metafits_pb = PathBuf::from(&metafits);

        let output_string = output.display().to_string();
        #[rustfmt::skip]
        let mut args = vec![
            "vis-convert",
            "--data", &vis[0], &metafits,
            "--outputs", &output_string
        ];
        if output_smallest_contiguous_band {
            args.push("--output-smallest-contiguous-band");
        }
        if let Some(per_coarse_chan_flags) = per_coarse_chan_flags.as_ref() {
            args.push("--fine-chan-flags-per-coarse-chan");
            for f in per_coarse_chan_flags {
                args.push(f.as_str());
            }
        }
        let vis_convert_args = VisConvertArgs::parse_from(args);
        vis_convert_args.run(false).unwrap();

        match output.extension().and_then(|os_str| os_str.to_str()) {
            Some("uvfits") => {
                Box::new(UvfitsReader::new(output.to_path_buf(), Some(&metafits_pb), None).unwrap())
            }
            Some("ms") => Box::new(
                MsReader::new(output.to_path_buf(), None, Some(&metafits_pb), None).unwrap(),
            ),
            _ => unreachable!(),
        }
    }

    for output_smallest_contiguous_band in [false, true] {
        for output in [&uvfits_converted, &ms_converted] {
            let data = get_data_object(output, None, output_smallest_contiguous_band);
            let obs_context = data.get_obs_context();
            if output_smallest_contiguous_band {
                assert_eq!(obs_context.fine_chan_freqs.len(), 28);
            } else {
                assert_eq!(obs_context.fine_chan_freqs.len(), 32);
            };
            assert_eq!(
                obs_context.num_fine_chans_per_coarse_chan,
                Some(NonZeroU16::new(32).unwrap())
            );
            assert_eq!(
                obs_context.mwa_coarse_chan_nums.as_deref(),
                Some([154].as_slice())
            );
            match output.extension().and_then(|os_str| os_str.to_str()) {
                Some("uvfits") => {
                    // uvfits currently doesn't try to determine this.
                    assert_eq!(obs_context.flagged_fine_chans_per_coarse_chan, None);
                }
                Some("ms") => {
                    assert_eq!(
                        obs_context.flagged_fine_chans_per_coarse_chan.as_deref(),
                        if output_smallest_contiguous_band {
                            // The MS reader can't tell if the edge channels are
                            // flagged or not; they're not available.
                            Some([16].as_slice())
                        } else {
                            Some([0, 1, 16, 30, 31].as_slice())
                        }
                    );
                }
                _ => unreachable!(),
            }
        }

        // Now test with additional per-coarse-chan flags.
        for output in [&uvfits_converted, &ms_converted] {
            let data = get_data_object(
                output,
                Some(vec!["2".to_string(), "5".to_string()]),
                output_smallest_contiguous_band,
            );
            let obs_context = data.get_obs_context();
            if output_smallest_contiguous_band {
                assert_eq!(obs_context.fine_chan_freqs.len(), 27);
            } else {
                assert_eq!(obs_context.fine_chan_freqs.len(), 32);
            };
            assert_eq!(
                obs_context.num_fine_chans_per_coarse_chan,
                Some(NonZeroU16::new(32).unwrap())
            );
            assert_eq!(
                obs_context.mwa_coarse_chan_nums.as_deref(),
                Some([154].as_slice())
            );
            match output.extension().and_then(|os_str| os_str.to_str()) {
                Some("uvfits") => {
                    // uvfits currently doesn't try to determine this.
                    assert_eq!(
                        obs_context.flagged_fine_chans_per_coarse_chan.as_deref(),
                        None
                    );
                }
                Some("ms") => {
                    assert_eq!(
                        obs_context.flagged_fine_chans_per_coarse_chan.as_deref(),
                        if output_smallest_contiguous_band {
                            // The MS reader can't tell if the edge channels are
                            // flagged or not; they're not available.
                            Some([5, 16].as_slice())
                        } else {
                            Some([0, 1, 2, 5, 16, 30, 31].as_slice())
                        }
                    );
                }
                _ => unreachable!(),
            }
        }
    }
}
