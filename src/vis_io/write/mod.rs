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
    ops::Range,
    path::{Path, PathBuf},
};

use crossbeam_channel::Receiver;
use crossbeam_utils::atomic::AtomicCell;
use hifitime::{Duration, Epoch};
use indicatif::ProgressBar;
use itertools::Itertools;
use log::{debug, trace, warn};
use marlu::{
    math::num_tiles_from_num_baselines, History, Jones, LatLngHeight, MeasurementSetWriter,
    MwaObsContext as MarluMwaObsContext, ObsContext as MarluObsContext, RADec, UvfitsWriter,
    VisContext, VisWrite, XyzGeodetic,
};
use ndarray::{prelude::*, ArcArray2};
use strum::IntoEnumIterator;
use strum_macros::{Display, EnumIter, EnumString};
use vec1::Vec1;

use crate::averaging::Timeblock;

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
    /// Cross-correlation visibilities ([baseline][channel]).
    pub(crate) cross_data: ArcArray<Jones<f32>, Ix2>,

    /// Cross-correlation weights (1:1 with the visibilities).
    pub(crate) cross_weights: ArcArray<f32, Ix2>,

    /// Visibilities followed by weights ([tile][channel]).
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
/// * `unflagged_baseline_tile_pairs` - the tile indices corresponding to
///   unflagged baselines. This includes auto-correlation "baselines" if they
///   are unflagged.
/// * `array_pos` - the position of the array that produced these visibilities.
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
/// * `timestamps` - all possible timestamps that could be written out. These
///   represent the centre of the integration bin, i.e. "centroid" and not
///   "leading edge". Must be ascendingly sorted and be regularly spaced in
///   terms of `time_res`, but gaps are allowed.
/// * `timesteps` - the timesteps to be written out. These are indices into
///   `timestamps`.
/// * `time_res` - the time resolution of the incoming visibilities.
/// * `fine_chan_freqs` - all of the fine channel frequencies \[Hz\] (flagged
///   and unflagged).
/// * `freq_res` - the frequency resolution of the incoming visibilities \[Hz\].
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
pub(crate) fn write_vis<'a>(
    outputs: &'a Vec1<(PathBuf, VisOutputType)>,
    array_pos: LatLngHeight,
    phase_centre: RADec,
    pointing_centre: Option<RADec>,
    tile_positions: &'a [XyzGeodetic],
    tile_names: &'a [String],
    obsid: Option<u32>,
    timestamps: &'a Vec1<Epoch>,
    timesteps: &'a Vec1<usize>,
    timeblocks: &'a Vec1<Timeblock>,
    time_res: Duration,
    dut1: Duration,
    freq_res: f64,
    fine_chan_freqs: &'a Vec1<f64>,
    unflagged_baseline_tile_pairs: &'a [(usize, usize)],
    flagged_fine_chans: &HashSet<usize>,
    time_average_factor: usize,
    freq_average_factor: usize,
    marlu_mwa_obs_context: Option<(&MarluMwaObsContext, &Range<usize>)>,
    rx: Receiver<VisTimestep>,
    error: &'a AtomicCell<bool>,
    progress_bar: Option<ProgressBar>,
) -> Result<String, VisWriteError> {
    // Ensure our timestamps are sensible.
    for &t in timestamps {
        let diff = (t - *timestamps.first()).total_nanoseconds();
        if diff % time_res.total_nanoseconds() > 0 {
            return Err(VisWriteError::IrregularTimestamps {
                first: timestamps.first().as_gpst_seconds(),
                bad: t.as_gpst_seconds(),
                time_res: time_res.in_seconds(),
            });
        }
    }

    let start_timestamp = timestamps[*timesteps.first()];
    let vis_ctx = VisContext {
        num_sel_timesteps: timeblocks.len() * time_average_factor,
        start_timestamp,
        int_time: time_res,
        num_sel_chans: fine_chan_freqs.len(),
        start_freq_hz: *fine_chan_freqs.first() as f64,
        freq_resolution_hz: freq_res,
        sel_baselines: unflagged_baseline_tile_pairs.to_vec(),
        avg_time: time_average_factor,
        avg_freq: freq_average_factor,
        num_vis_pols: 4,
    };

    let obs_name = obsid.map(|o| format!("{o}"));
    let sched_start_timestamp = match obsid {
        Some(gpst) => Epoch::from_gpst_seconds(f64::from(gpst)),
        None => start_timestamp,
    };
    let sched_duration = timestamps[*timesteps.last()] + time_res - sched_start_timestamp;
    let (s_lat, c_lat) = array_pos.latitude_rad.sin_cos();
    let marlu_obs_ctx = MarluObsContext {
        sched_start_timestamp,
        sched_duration,
        name: obs_name,
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
                );
                if let Some((marlu_mwa_obs_context, coarse_chan_range)) =
                    marlu_mwa_obs_context.as_ref()
                {
                    ms.initialize_mwa(
                        &vis_ctx,
                        &marlu_obs_ctx,
                        marlu_mwa_obs_context,
                        Some(&history),
                        coarse_chan_range,
                    )?;
                } else {
                    ms.initialize(&vis_ctx, &marlu_obs_ctx, None)?;
                }
                Box::new(ms)
            }
        };
        writers.push(vis_writer);
    }

    // These arrays will contain the post-averaged values and are written out by
    // the writer when all relevant timesteps have been added.
    // [time][freq][baseline]
    let out_shape = vis_ctx.sel_dims();
    let mut out_data = Array3::zeros((time_average_factor, out_shape.1, out_shape.2));
    let mut out_weights = Array3::from_elem((time_average_factor, out_shape.1, out_shape.2), -0.0);

    // Track a reference to the timeblock we're writing.
    let mut this_timeblock = timeblocks.first();
    // Also track the first timestamp of the tracked timeblock.
    // let mut this_start_timestamp = None;
    let mut this_average_timestamp = None;
    let mut i_timeblock = 0;
    // And the timestep into the timeblock.
    let mut this_timestep = 0;

    // Receive visibilities from another thread.
    for (
        i_timestep,
        VisTimestep {
            cross_data,
            cross_weights,
            autos,
            timestamp,
        },
    ) in rx.iter().enumerate()
    {
        debug!(
            "Received timestep {i_timestep} (GPS {})",
            timestamp.as_gpst_seconds()
        );
        if this_average_timestamp.is_none() {
            this_average_timestamp = Some(
                timeblocks
                    .iter()
                    .find(|tb| tb.timestamps.contains(&timestamp))
                    .unwrap()
                    .median,
            );
        }

        if let Some(autos) = autos.as_ref() {
            // Get the number of tiles from the lengths of the cross and auto
            // arrays.
            let num_cross_baselines = cross_data.len_of(Axis(0));
            let num_auto_baselines = autos.0.len_of(Axis(0));
            let num_tiles = num_tiles_from_num_baselines(num_cross_baselines + num_auto_baselines);
            assert_eq!(
                (num_tiles * (num_tiles + 1)) / 2,
                num_cross_baselines + num_auto_baselines,
            );

            // baseline
            assert_eq!(num_cross_baselines + num_auto_baselines, out_shape.2);
            // freq
            assert_eq!(
                cross_data.len_of(Axis(1)) + flagged_fine_chans.len(),
                out_shape.1
            );
            assert_eq!(cross_data.len_of(Axis(1)), autos.0.len_of(Axis(1)));
        } else {
            // baseline
            assert_eq!(cross_data.len_of(Axis(0)), out_shape.2);
            // freq
            assert_eq!(
                cross_data.len_of(Axis(1)) + flagged_fine_chans.len(),
                out_shape.1
            );
        }

        // Pack `out_data` and `out_weights`; a transpose is needed. Start with
        // cross-correlation data, skipping any auto-correlation indices; we'll
        // fill them soon.
        out_data
            .slice_mut(s![this_timestep, .., ..])
            .outer_iter_mut()
            .zip_eq(
                out_weights
                    .slice_mut(s![this_timestep, .., ..])
                    .outer_iter_mut(),
            )
            .enumerate()
            .filter(|(i_chan, _)| !flagged_fine_chans.contains(i_chan))
            // Discard the channel index
            .map(|(_, t)| t)
            .zip_eq(cross_data.axis_iter(Axis(1)))
            .zip_eq(cross_weights.axis_iter(Axis(1)))
            .for_each(|(((mut out_data, mut out_weights), in_data), in_weights)| {
                out_data
                    .iter_mut()
                    .zip(out_weights.iter_mut())
                    .zip(unflagged_baseline_tile_pairs.iter())
                    .filter(|(_, baseline)| baseline.0 != baseline.1)
                    .zip(in_data.iter())
                    .zip(in_weights.iter())
                    .for_each(|((((out_jones, out_weight), _), in_jones), in_weight)| {
                        *out_jones = *in_jones;
                        *out_weight = *in_weight;
                    });
            });
        // Autos.
        if let Some(autos) = autos {
            out_data
                .slice_mut(s![this_timestep, .., ..])
                .axis_iter_mut(Axis(0))
                .zip_eq(
                    out_weights
                        .slice_mut(s![this_timestep, .., ..])
                        .axis_iter_mut(Axis(0)),
                )
                .enumerate()
                .filter(|(i_chan, _)| !flagged_fine_chans.contains(i_chan))
                // Discard the channel index
                .map(|(_, t)| t)
                .zip_eq(autos.0.axis_iter(Axis(1)))
                .zip_eq(autos.1.axis_iter(Axis(1)))
                .for_each(|(((mut out_data, mut out_weights), in_data), in_weights)| {
                    out_data
                        .iter_mut()
                        .zip(out_weights.iter_mut())
                        .zip(unflagged_baseline_tile_pairs.iter())
                        .filter(|(_, baseline)| baseline.0 == baseline.1)
                        .zip(in_data.iter())
                        .zip(in_weights.iter())
                        .for_each(|((((out_jones, out_weight), _), in_jones), in_weight)| {
                            *out_jones = *in_jones;
                            *out_weight = *in_weight;
                        });
                });
        }

        // Should we continue?
        if error.load() {
            return Ok(String::new());
        }

        // If the next timestep doesn't belong to our tracked timeblock, write
        // out this timeblock and track the next one.
        if !this_timeblock.range.contains(&(i_timestep + 1))
            || this_timestep + 1 >= time_average_factor
        {
            debug!("Writing timeblock {i_timeblock}");
            let chunk_vis_ctx = VisContext {
                // TODO: Marlu expects "leading edge" timestamps, not centroids.
                // Fix this in Marlu.
                start_timestamp: this_average_timestamp.unwrap()
                    - time_res / 2 * time_average_factor as f64,
                num_sel_timesteps: this_timeblock.range.len(),
                ..vis_ctx.clone()
            };

            for vis_writer in writers.iter_mut() {
                vis_writer.write_vis(
                    out_data.slice(s![0..this_timeblock.range.len(), .., ..]),
                    out_weights.slice(s![0..this_timeblock.range.len(), .., ..]),
                    &chunk_vis_ctx,
                    false,
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
            out_data.fill(Jones::default());
            out_weights.fill(-0.0);

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
            warn!("Will overwrite the existing directory '{}'", file.display());
        }
    } else {
        let exists = can_write_to_file_inner(file)?;
        if exists {
            warn!("Will overwrite the existing file '{}'", file.display());
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
