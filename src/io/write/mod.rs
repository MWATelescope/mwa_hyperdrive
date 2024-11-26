// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to handle writing out visibilities.

mod error;
#[cfg(test)]
mod tests;
pub(crate) use error::{FileWriteError, VisWriteError};

use std::{
    collections::HashSet,
    num::NonZeroUsize,
    path::{Path, PathBuf},
};

use crossbeam_channel::Receiver;
use crossbeam_utils::atomic::AtomicCell;
use hifitime::{Duration, Epoch};
use indicatif::ProgressBar;
use itertools::Itertools;
use log::{debug, trace};
use marlu::{
    math::num_tiles_from_num_baselines, History, Jones, LatLngHeight, MeasurementSetWriter,
    MwaObsContext as MarluMwaObsContext, ObsContext as MarluObsContext, RADec, UvfitsWriter,
    VisContext, VisWrite, XyzGeodetic,
};
use ndarray::{prelude::*, ArcArray2};
use strum::IntoEnumIterator;
use strum_macros::{Display, EnumIter, EnumString};
use vec1::{vec1, Vec1};

use crate::{
    averaging::{Spw, Timeblock},
    cli::Warn,
};

#[derive(Debug, Display, EnumIter, EnumString, Clone, Copy)]
/// All write-supported visibility formats.
pub(crate) enum VisOutputType {
    #[strum(serialize = "uvfits")]
    Uvfits,
    #[strum(serialize = "ms")]
    MeasurementSet,
}

lazy_static::lazy_static! {
    pub(crate) static ref VIS_OUTPUT_EXTENSIONS: String = VisOutputType::iter().join(", ");
}

/// A struct to carry all of the visibilities of a timestep.
pub(crate) struct VisTimestep {
    /// Cross-correlation visibilities ([channel][baseline]).
    pub(crate) cross_data_fb: ArcArray<Jones<f32>, Ix2>,

    /// Cross-correlation weights (1:1 with the visibilities).
    pub(crate) cross_weights_fb: ArcArray<f32, Ix2>,

    /// Visibilities followed by weights ([channel][tile]).
    pub(crate) autos: Option<(ArcArray2<Jones<f32>>, ArcArray2<f32>)>,

    /// The timestamp corresponding to these visibilities.
    pub(crate) timestamp: Epoch,
}

/// Create the specified visibility outputs and receive visibilities to write to
/// them. This function is intended to be run concurrently with other async
/// threads, and must receive the timesteps of data as specified by `timesteps`.
///
/// # Arguments
///
/// * `outputs` - each of the output files to be written, paired with the output
///   type.
/// * `array_pos` - the position of the array for the incoming visibilities.
/// * `phase_centre` - the phase centre used for the incoming visibilities.
/// * `pointing_centre` - the pointing centre used for the incoming
///   visibilities.
/// * `tile_positions` - the *un-precessed* positions of the tiles used for the
///   incoming visibilities (flagged and unflagged).
/// * `tile_names` - the names of the tiles used for the incoming visibilities
///   (flagged and unflagged).
/// * `obsid` - the MWA observation ID. If provided, it is used as the scheduled
///   start time of the observation and as an identifier. If not provided, the
///   first timestep is used as the scheduled start time and a placeholder will
///   be used for the identifier.
/// * `timeblocks` - the details on all incoming timestamps and how to combine
///   them.
/// * `time_res` - the time resolution of the incoming visibilities.
/// * `dut1` - the DUT1 to use in the UVWs of the outgoing visibilities.
/// * `spw` - the spectral window information of the outgoing visibilities.
/// * `unflagged_baseline_tile_pairs` - the tile indices corresponding to
///   unflagged baselines. This includes auto-correlation "baselines" if they
///   are unflagged.
/// * `time_average_factor` - the time average factor (i.e. average this many
///   visibilities in time before writing out).
/// * `freq_average_factor` - the frequency average factor (i.e. average this
///   many channels before writing out).
/// * `marlu_mwa_obs_context` - a tuple of [`marlu::MwaObsContext`] and a range
///   of MWA coarse channel indices. Kept optional because they're not strictly
///   needed.
/// * `rx` - the channel to receive visibilities from.
/// * `error` - a thread-safe [`bool`] to indicate if an error has occurred.
///   Receiving `true` signals that we should not continue, as another thread
///   has experienced an error.
/// * `progress_bar` - an optional progress bar to increment with writing
///   progress.
///
/// # Returns
///
/// * A neatly-formatted string reporting all of the files that got written out.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_vis(
    outputs: &Vec1<(PathBuf, VisOutputType)>,
    array_pos: LatLngHeight,
    phase_centre: RADec,
    pointing_centre: Option<RADec>,
    tile_positions: &[XyzGeodetic],
    tile_names: &[String],
    obsid: Option<u32>,
    timeblocks: &Vec1<Timeblock>,
    time_res: Duration,
    dut1: Duration,
    spw: &Spw,
    unflagged_baseline_tile_pairs: &[(usize, usize)],
    time_average_factor: NonZeroUsize,
    freq_average_factor: NonZeroUsize,
    marlu_mwa_obs_context: Option<&MarluMwaObsContext>,
    write_smallest_contiguous_band: bool,
    rx: Receiver<VisTimestep>,
    error: &AtomicCell<bool>,
    progress_bar: Option<ProgressBar>,
) -> Result<String, VisWriteError> {
    // Ensure our timestamps are regularly spaced in terms of `time_res`.
    for t in timeblocks {
        let diff = (t.median - timeblocks.first().median).total_nanoseconds();
        if diff % time_res.total_nanoseconds() > 0 {
            return Err(VisWriteError::IrregularTimestamps {
                first: timeblocks.first().median.to_gpst_seconds(),
                bad: t.median.to_gpst_seconds(),
                time_res: time_res.to_seconds(),
            });
        }
    }

    // When writing out visibility data, the frequency axis *must* be
    // contiguous. But, the incoming visibility data might not be contiguous due
    // to flags. Set up the outgoing frequencies and set a flag so we know if
    // the incoming data needs to be padded.
    let chanblock_freqs = if write_smallest_contiguous_band {
        match spw.chanblocks.as_slice() {
            [] => panic!("There weren't any unflagged chanblocks in the SPW"),
            [c] => vec1![c.freq],
            [c1, .., cn] => {
                let first_freq = c1.freq;
                let last_freq = cn.freq;
                let mut v = Array1::range(first_freq, last_freq, spw.freq_res).into_raw_vec();
                v.push(last_freq); // `Array1::range` is an exclusive range.
                Vec1::try_from_vec(v).expect("v is never empty")
            }
        }
    } else {
        spw.get_all_freqs()
    };
    let missing_chanblocks = {
        let mut missing = HashSet::new();
        let incoming_chanblock_freqs = spw
            .chanblocks
            .iter()
            .map(|c| c.freq as u64)
            .collect::<HashSet<_>>();
        for (i_chanblock, chanblock_freq) in (0..).zip(chanblock_freqs.iter()) {
            let chanblock_freq = *chanblock_freq as u64;
            if !incoming_chanblock_freqs.contains(&chanblock_freq) {
                missing.insert(i_chanblock);
            }
        }
        missing
    };

    let start_timestamp = timeblocks.first().median;
    let num_baselines = unflagged_baseline_tile_pairs.len();
    let vis_ctx = VisContext {
        num_sel_timesteps: timeblocks.len() * time_average_factor.get(),
        start_timestamp,
        int_time: time_res,
        num_sel_chans: chanblock_freqs.len(),
        start_freq_hz: *chanblock_freqs.first(),
        freq_resolution_hz: spw.freq_res,
        sel_baselines: unflagged_baseline_tile_pairs.to_vec(),
        avg_time: time_average_factor.get(),
        avg_freq: freq_average_factor.get(),
        num_vis_pols: 4,
    };

    let sched_start_timestamp = match obsid {
        Some(gpst) => Epoch::from_gpst_seconds(f64::from(gpst)),
        None => start_timestamp,
    };
    let sched_duration = timeblocks.last().median + time_res - sched_start_timestamp;
    let (s_lat, c_lat) = array_pos.latitude_rad.sin_cos();
    let marlu_obs_ctx = MarluObsContext {
        sched_start_timestamp,
        sched_duration,
        name: obsid.map(|o| format!("{o}")),
        phase_centre,
        pointing_centre,
        array_pos,
        ant_positions_enh: tile_positions
            .iter()
            .map(|xyz| xyz.to_enh_inner(s_lat, c_lat))
            .collect(),
        ant_names: tile_names.to_vec(),
        // TODO(dev): is there any value in adding this metadata via hyperdrive obs context?
        field_name: None,
        project_id: None,
        observer: None,
    };

    // Prepare history for the output vis files. It's possible that the
    // command-line call has invalid UTF-8. So use args_os and attempt to
    // convert to UTF-8 strings. If there are problems on the way, don't bother
    // trying to write the CMDLINE key.
    let cmd_line = std::env::args_os()
        .map(|a| a.into_string())
        .collect::<Result<Vec<String>, _>>()
        .map(|v| v.join(" "))
        .ok();
    let history = History {
        application: Some("mwa_hyperdrive"),
        cmd_line: cmd_line.as_deref(),
        message: None,
    };
    let mut writers = vec![];
    for (output, vis_type) in outputs {
        debug!("Setting up {} ({vis_type})", output.display());
        let vis_writer: Box<dyn VisWrite> = match vis_type {
            VisOutputType::Uvfits => {
                let uvfits = UvfitsWriter::from_marlu(
                    output,
                    &vis_ctx,
                    array_pos,
                    phase_centre,
                    dut1,
                    marlu_obs_ctx.name.as_deref(),
                    tile_names.to_vec(),
                    tile_positions.to_vec(),
                    false,
                    Some(&history),
                )?;
                Box::new(uvfits)
            }

            VisOutputType::MeasurementSet => {
                let ms = MeasurementSetWriter::new(
                    output,
                    phase_centre,
                    array_pos,
                    tile_positions.to_vec(),
                    dut1,
                    false,
                );
                if let Some(marlu_mwa_obs_context) = marlu_mwa_obs_context {
                    ms.initialize_mwa(
                        &vis_ctx,
                        &marlu_obs_ctx,
                        marlu_mwa_obs_context,
                        Some(&history),
                        &(0..marlu_mwa_obs_context.coarse_chan_recs.len()),
                    )?;
                } else {
                    ms.initialize(&vis_ctx, &marlu_obs_ctx, Some(&history))?;
                }
                Box::new(ms)
            }
        };
        writers.push(vis_writer);
    }

    // These arrays will contain the post-averaged values and are written out by
    // the writer when all relevant timesteps have been added.
    // [time][freq][baseline]
    let out_shape = (
        timeblocks.len() * time_average_factor.get(),
        chanblock_freqs.len(),
        num_baselines,
    );
    let mut out_data_tfb = Array3::zeros((time_average_factor.get(), out_shape.1, out_shape.2));
    let mut out_weights_tfb =
        Array3::from_elem((time_average_factor.get(), out_shape.1, out_shape.2), -0.0);

    // Track a reference to the timeblock we're writing.
    let mut this_timeblock = timeblocks.first();
    // Also track the first timestamp of the tracked timeblock.
    let mut this_average_timestamp = None;
    let mut i_timeblock = 0;
    // And the timestep into the timeblock.
    let mut this_timestep = 0;

    // Receive visibilities from another thread.
    for (
        i_timestep,
        VisTimestep {
            cross_data_fb,
            cross_weights_fb,
            autos,
            timestamp,
        },
    ) in rx.iter().enumerate()
    {
        debug!(
            "Received timestep {i_timestep} (GPS {})",
            timestamp.to_gpst_seconds()
        );
        if this_average_timestamp.is_none() {
            this_average_timestamp = Some(
                timeblocks
                    .iter()
                    .find(|tb| tb.timestamps.iter().any(|e| *e == timestamp))
                    .unwrap()
                    .median,
            );
        }

        if let Some(autos) = autos.as_ref() {
            // Get the number of tiles from the lengths of the cross and auto
            // arrays.
            let num_cross_baselines = cross_data_fb.len_of(Axis(1));
            let num_auto_baselines = autos.0.len_of(Axis(1));
            assert_eq!(num_cross_baselines + num_auto_baselines, num_baselines);
            let num_tiles = num_tiles_from_num_baselines(num_cross_baselines + num_auto_baselines);
            assert_eq!(
                (num_tiles * (num_tiles + 1)) / 2,
                num_cross_baselines + num_auto_baselines,
            );

            // baseline
            assert_eq!(num_cross_baselines + num_auto_baselines, out_shape.2);
            // freq
            assert_eq!(cross_data_fb.len_of(Axis(0)), autos.0.len_of(Axis(0)));
        } else {
            // baseline
            assert_eq!(cross_data_fb.len_of(Axis(1)), out_shape.2);
        }
        // freq
        assert_eq!(cross_data_fb.len_of(Axis(0)), spw.chanblocks.len());

        // Pack `out_data` and `out_weights`. Start with cross-correlation data,
        // skipping any auto-correlation indices; we'll fill them soon.
        out_data_tfb
            .slice_mut(s![this_timestep, .., ..])
            .outer_iter_mut()
            .zip_eq(
                out_weights_tfb
                    .slice_mut(s![this_timestep, .., ..])
                    .outer_iter_mut(),
            )
            .enumerate()
            .filter(|(i_chan, _)| !missing_chanblocks.contains(&(*i_chan as u16)))
            // Discard the channel index
            .map(|(_, d)| d)
            .zip_eq(cross_data_fb.outer_iter())
            .zip_eq(cross_weights_fb.outer_iter())
            .for_each(
                |(((mut out_data_b, mut out_weights_b), in_data_b), in_weights_b)| {
                    out_data_b
                        .iter_mut()
                        .zip_eq(out_weights_b.iter_mut())
                        .zip_eq(unflagged_baseline_tile_pairs.iter())
                        .filter(|(_, baseline)| baseline.0 != baseline.1)
                        .zip_eq(in_data_b.iter())
                        .zip_eq(in_weights_b.iter())
                        .for_each(|((((out_jones, out_weight), _), in_jones), in_weight)| {
                            *out_jones = *in_jones;
                            *out_weight = *in_weight;
                        });
                },
            );
        // Autos.
        if let Some((auto_data_fb, auto_weights_fb)) = autos {
            (0..)
                .zip(
                    out_data_tfb
                        .slice_mut(s![this_timestep, .., ..])
                        .outer_iter_mut(),
                )
                .zip(
                    out_weights_tfb
                        .slice_mut(s![this_timestep, .., ..])
                        .outer_iter_mut(),
                )
                .filter(|((i_chan, _), _)| !missing_chanblocks.contains(i_chan))
                // Discard the channel index
                .map(|((_, d), w)| (d, w))
                .zip_eq(auto_data_fb.outer_iter())
                .zip_eq(auto_weights_fb.outer_iter())
                .for_each(
                    |(((mut out_data_b, mut out_weights_b), in_data_b), in_weights_b)| {
                        out_data_b
                            .iter_mut()
                            .zip_eq(out_weights_b.iter_mut())
                            .zip_eq(unflagged_baseline_tile_pairs.iter())
                            .filter(|(_, baseline)| baseline.0 == baseline.1)
                            .zip_eq(in_data_b.iter())
                            .zip_eq(in_weights_b.iter())
                            .for_each(|((((out_jones, out_weight), _), in_jones), in_weight)| {
                                *out_jones = *in_jones;
                                *out_weight = *in_weight;
                            });
                    },
                );
        }

        // Should we continue?
        if error.load() {
            return Ok(String::new());
        }

        // If the next timestep doesn't belong to our tracked timeblock, write
        // out this timeblock and track the next one.
        if !this_timeblock.range.contains(&(i_timestep + 1))
            || this_timestep + 1 >= time_average_factor.get()
        {
            debug!("Writing timeblock {i_timeblock}");
            let chunk_vis_ctx = VisContext {
                // TODO: Marlu expects "leading edge" timestamps, not centroids.
                // Fix this in Marlu.
                start_timestamp: this_average_timestamp.unwrap()
                    - time_res / 2 * time_average_factor.get() as f64,
                num_sel_timesteps: this_timeblock.range.len(),
                ..vis_ctx.clone()
            };

            trace!("this_timeblock.range: {:?}", this_timeblock.range);
            for vis_writer in writers.iter_mut() {
                vis_writer.write_vis(
                    out_data_tfb.slice(s![0..this_timeblock.range.len(), .., ..]),
                    out_weights_tfb.slice(s![0..this_timeblock.range.len(), .., ..]),
                    &chunk_vis_ctx,
                )?;
                // Should we continue?
                if error.load() {
                    return Ok(String::new());
                }
            }

            if let Some(progress_bar) = progress_bar.as_ref() {
                progress_bar.inc(1);
            }

            // Clear the output buffers.
            out_data_tfb.fill(Jones::default());
            out_weights_tfb.fill(-0.0);

            i_timeblock += 1;
            this_timeblock = match timeblocks.get(i_timeblock) {
                Some(t) => t,
                None => break,
            };
            this_average_timestamp = None;
            this_timestep = 0;
        } else {
            this_timestep += 1;
        }
    }

    if let Some(progress_bar) = progress_bar.as_ref() {
        progress_bar.abandon_with_message("Finished writing visibilities");
    }

    for vis_writer in writers.iter_mut() {
        vis_writer.finalise()?;
    }
    debug!("Finished writing");

    let output_vis_str = if outputs.len() == 1 {
        format!("Visibilities written to {}", outputs.first().0.display())
    } else {
        format!(
            "Visibilities written to: {}",
            outputs.iter().map(|(o, _)| o.display()).join(", ")
        )
    };
    Ok(output_vis_str)
}

/// Check if we are able to write to a file path. If we aren't able to write to
/// the file, it's either because the directory containing the file doesn't
/// exist, or there's another issue (probably bad permissions). In the former
/// case, create the parent directories, otherwise return an error.
/// Additionally, if the file exists, emit a warning that it will be
/// overwritten.
///
/// With this approach, we potentially avoid doing a whole run of calibration
/// only to be unable to write to a file at the end. This code _doesn't_ alter
/// the file if it exists.
pub(crate) fn can_write_to_file(file: &Path) -> Result<(), FileWriteError> {
    trace!("Testing whether we can write to {}", file.display());

    if file.is_dir() {
        let exists = can_write_to_dir(file)?;
        if exists {
            format!("Will overwrite the existing directory '{}'", file.display()).warn();
        }
    } else {
        let exists = can_write_to_file_inner(file)?;
        if exists {
            format!("Will overwrite the existing file '{}'", file.display()).warn();
        }
    }

    Ok(())
}

/// Iterate over all of the files and subdirectories of a directory and test
/// whether we can write to them. Note that testing whether directories are
/// writable is very weak; in my testing, changing a subdirectories owner to
/// root and running this function suggested that the file was writable, but it
/// was not. This appears to be a limitation of operating systems, and there's
/// not even a reliable way of checking if *your* user is able to write to a
/// directory. Files are much more rigorously tested.
fn can_write_to_dir(dir: &Path) -> Result<bool, FileWriteError> {
    let exists = dir.exists();

    let metadata = std::fs::metadata(dir)?;
    let permissions = metadata.permissions();
    if permissions.readonly() {
        return Err(FileWriteError::FileNotWritable {
            file: dir.display().to_string(),
        });
    }

    // Test whether every single entry in `dir` is writable.
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?.path();
        if entry.is_file() {
            can_write_to_file_inner(&entry)?;
        } else if entry.is_dir() {
            can_write_to_dir(&entry)?;
        }
    }

    Ok(exists)
}

fn can_write_to_file_inner(file: &Path) -> Result<bool, FileWriteError> {
    let file_exists = file.exists();

    match std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(file)
        .map_err(|e| e.kind())
    {
        // File is writable.
        Ok(_) => {
            // If the file in question didn't already exist, `OpenOptions::new`
            // creates it as part of its work. We don't want to keep the 0-sized
            // file; remove it if it didn't exist before.
            if !file_exists {
                std::fs::remove_file(file).map_err(FileWriteError::IO)?;
            }
        }

        // File doesn't exist. Attempt to make the directories leading up to the
        // file; if this fails, then we can't write the file anyway.
        Err(std::io::ErrorKind::NotFound) => {
            if let Some(p) = file.parent() {
                match std::fs::DirBuilder::new()
                    .recursive(true)
                    .create(p)
                    .map_err(|e| e.kind())
                {
                    Ok(()) => (),
                    Err(std::io::ErrorKind::PermissionDenied) => {
                        return Err(FileWriteError::NewDirectory(p.to_path_buf()))
                    }
                    Err(e) => return Err(FileWriteError::IO(e.into())),
                }
            }
        }

        Err(std::io::ErrorKind::PermissionDenied) => {
            return Err(FileWriteError::FileNotWritable {
                file: file.display().to_string(),
            })
        }

        Err(e) => {
            return Err(FileWriteError::IO(e.into()));
        }
    }

    Ok(file_exists)
}
