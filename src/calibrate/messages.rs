// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Messages to report to the user.
//!
//! When unpacking input data, beam parameters etc. some things are useful to
//! report to the user. However, the order of these messages can appear
//! unrelated or random because of the way the code is ordered. This module
//! attempts to tidy this issue by categorising message types.

use std::path::{Path, PathBuf};

use hifitime::Epoch;
use itertools::Itertools;
use log::{info, trace, warn};
use marlu::{LatLngHeight, RADec};

use crate::{pfb_gains::PfbFlavour, unit_parsing::WavelengthUnit};
use mwa_hyperdrive_common::{hifitime, itertools, log, marlu};
use mwa_hyperdrive_srclist::SourceList;

#[must_use = "This struct must be consumed with its print() method"]
pub(super) enum InputFileDetails {
    Raw {
        obsid: u32,
        gpubox_count: usize,
        metafits_file_name: String,
        mwaf: Option<usize>,
        // Raw-data corrections.
        pfb: PfbFlavour,
        digital_gains: bool,
        cable_length: bool,
        geometric: bool,
    },
    MeasurementSet {
        obsid: Option<u32>,
        file_name: String,
        metafits_file_name: Option<String>,
    },
    UvfitsFile {
        obsid: Option<u32>,
        file_name: String,
        metafits_file_name: Option<String>,
    },
}

impl InputFileDetails {
    pub(super) fn print(self) {
        match self {
            InputFileDetails::Raw {
                obsid,
                gpubox_count,
                metafits_file_name,
                mwaf,
                pfb,
                digital_gains,
                cable_length,
                geometric,
            } => {
                info!("DI calibrating obsid {obsid}");
                info!("  from {gpubox_count} gpubox files");
                info!("  with metafits {metafits_file_name}");
                match mwaf {
                    Some(c) => info!("  with {c} mwaf files"),
                    None => warn!("No mwaf files supplied"),
                }

                let s = "Correcting PFB gains";
                match pfb {
                    PfbFlavour::None => info!("Not doing any PFB correction"),
                    PfbFlavour::Jake => info!("{s} with 'Jake Jones' gains"),
                    PfbFlavour::Cotter2014 => info!("{s} with 'Cotter 2014' gains"),
                    PfbFlavour::Empirical => info!("{s} with 'RTS empirical' gains"),
                    PfbFlavour::Levine => info!("{s} with 'Alan Levine' gains"),
                }
                if digital_gains {
                    info!("Correcting digital gains");
                } else {
                    info!("Not correcting digital gains");
                }
                if cable_length {
                    info!("Correcting cable lengths");
                } else {
                    info!("Not correcting cable lengths");
                }
                if geometric {
                    info!("Correcting geometric delays (if necessary)");
                } else {
                    info!("Not correcting geometric delays");
                }
            }

            InputFileDetails::MeasurementSet {
                obsid,
                file_name,
                metafits_file_name,
            } => {
                if let Some(o) = obsid {
                    info!("DI calibrating obsid {o}");
                    info!("  from measurement set {file_name}");
                } else {
                    info!("DI calibrating measurement set {file_name}");
                }
                if let Some(f) = metafits_file_name {
                    info!("  with metafits {f}");
                }
            }

            InputFileDetails::UvfitsFile {
                obsid,
                file_name,
                metafits_file_name,
            } => {
                if let Some(o) = obsid {
                    info!("DI calibrating obsid {o}");
                    info!("  from uvfits {file_name}");
                } else {
                    info!("DI calibrating uvfits {file_name}");
                }
                if let Some(f) = metafits_file_name {
                    info!("  with metafits {f}");
                }
            }
        }
    }
}

#[must_use = "This struct must be consumed with its print() method"]
pub(super) struct ArrayDetails {
    pub(super) array_position: LatLngHeight,
    /// \[radians\]
    pub(super) array_latitude_j2000: Option<f64>,
    pub(super) total_num_tiles: usize,
    pub(super) num_unflagged_tiles: usize,
    pub(super) flagged_tiles: Vec<(String, usize)>,
}

impl ArrayDetails {
    pub(super) fn print(self) {
        info!(
            "Array latitude:         {:>8.4}°",
            self.array_position.latitude_rad.to_degrees()
        );
        if let Some(rad) = self.array_latitude_j2000 {
            info!("Array latitude (J2000): {:>8.4}°", rad.to_degrees());
        }
        info!(
            "Array longitude:       {:>9.4}°",
            self.array_position.longitude_rad.to_degrees()
        );
        info!(
            "Array height:          {:>9.4}m",
            self.array_position.height_metres
        );

        info!("Total number of tiles:     {:>3}", self.total_num_tiles);
        info!("Number of unflagged tiles: {:>3}", self.num_unflagged_tiles);
        info!("Flagged tiles:             {:?}", self.flagged_tiles);
    }
}

#[must_use = "This struct must be consumed with its print() method"]
pub(super) struct ObservationDetails<'a> {
    pub(super) dipole_delays: [u32; 16],
    pub(super) beam_file: Option<&'a Path>,
    /// If this is `None`, then report that we're assuming all dipoles are
    /// "alive".
    pub(super) num_tiles_with_dead_dipoles: Option<usize>,

    pub(super) phase_centre: RADec,
    pub(super) pointing_centre: Option<RADec>,
    /// The local mean sidereal time of the first timestep \[radians\]
    pub(super) lmst: f64,
    /// The local mean sidereal time of the first timestep, precessed to the
    /// J2000 epoch \[radians\]
    pub(super) lmst_j2000: f64,

    pub(super) available_timesteps: &'a [usize],
    pub(super) unflagged_timesteps: &'a [usize],
    pub(super) using_timesteps: &'a [usize],
    pub(super) first_timestep: Option<Epoch>,
    pub(super) last_timestep: Option<Epoch>,
    pub(super) time_res_seconds: Option<f64>,

    pub(super) total_num_channels: usize,
    pub(super) num_unflagged_channels: usize,
    pub(super) flagged_chans_per_coarse_chan: &'a [usize],
    pub(super) first_freq_hz: Option<f64>,
    pub(super) last_freq_hz: Option<f64>,
    pub(super) freq_res_hz: Option<f64>,
}

impl ObservationDetails<'_> {
    pub(super) fn print(self) {
        let d = self.dipole_delays;
        info!(
            "Ideal dipole delays: [{:>2} {:>2} {:>2} {:>2}",
            d[0], d[1], d[2], d[3]
        );
        info!(
            "                      {:>2} {:>2} {:>2} {:>2}",
            d[4], d[5], d[6], d[7]
        );
        info!(
            "                      {:>2} {:>2} {:>2} {:>2}",
            d[8], d[9], d[10], d[11]
        );
        info!(
            "                      {:>2} {:>2} {:>2} {:>2}]",
            d[12], d[13], d[14], d[15]
        );
        if let Some(beam_file) = self.beam_file {
            info!("Using beam file {}", beam_file.display());
        }
        if let Some(num_tiles_with_dead_dipoles) = self.num_tiles_with_dead_dipoles {
            info!("Using dead dipole information ({num_tiles_with_dead_dipoles} tiles affected)");
        } else {
            info!("Assuming all dipoles are \"alive\"");
        }

        info!(
            "Phase centre (J2000): {:>9.4}°, {:>8.4}°",
            self.phase_centre.ra.to_degrees(),
            self.phase_centre.dec.to_degrees(),
        );
        if let Some(pc) = self.pointing_centre {
            info!(
                "Pointing centre:      {:>9.4}°, {:>8.4}°",
                pc.ra.to_degrees(),
                pc.dec.to_degrees()
            );
        }
        info!(
            "LMST of first timestep:         {:>9.4}°",
            self.lmst.to_degrees()
        );
        info!(
            "LMST of first timestep (J2000): {:>9.4}°",
            self.lmst_j2000.to_degrees()
        );

        info!(
            "{}",
            range_or_comma_separated(self.available_timesteps, Some("Available timesteps:"))
        );
        info!(
            "{}",
            range_or_comma_separated(self.unflagged_timesteps, Some("Unflagged timesteps:"))
        );
        // We don't require the timesteps to be used in calibration to be
        // sequential. But if they are, it looks a bit neater to print them out
        // as a range rather than individual indices.
        info!(
            "{}",
            range_or_comma_separated(self.using_timesteps, Some("Using timesteps:    "))
        );
        match (
            self.first_timestep,
            self.last_timestep,
            self.first_timestep.or(self.last_timestep),
        ) {
            (Some(f), Some(l), _) => {
                info!("First timestep (GPS): {:.2}", f.as_gpst_seconds());
                info!("Last timestep  (GPS): {:.2}", l.as_gpst_seconds());
            }
            (_, _, Some(f)) => info!("Only timestep (GPS): {:.2}", f.as_gpst_seconds()),
            _ => (),
        }
        match self.time_res_seconds {
            Some(r) => info!("Input data time resolution: {r:.2} seconds"),
            None => info!("Input data time resolution unknown"),
        }

        info!(
            "Total number of fine channels:     {}",
            self.total_num_channels
        );
        info!(
            "Number of unflagged fine channels: {}",
            self.num_unflagged_channels
        );
        info!(
            "Input data's fine-channel flags per coarse channel: {:?}",
            self.flagged_chans_per_coarse_chan
        );
        match (
            self.first_freq_hz,
            self.last_freq_hz,
            self.first_freq_hz.or(self.last_freq_hz),
        ) {
            (Some(f), Some(l), _) => {
                info!("First unflagged fine-channel frequency: {:.2} MHz", f / 1e6);
                info!("Last unflagged fine-channel frequency:  {:.2} MHz", l / 1e6);
            }
            (_, _, Some(f)) => info!("Only unflagged fine-channel frequency: {f:.2} MHz"),
            _ => (),
        }
        match self.freq_res_hz {
            Some(r) => info!("Input data freq. resolution: {:.2} kHz", r / 1e3),
            None => info!("Input data freq. resolution unknown"),
        }
    }
}

#[must_use = "This struct must be consumed with its print() method"]
pub(super) struct CalibrationDetails {
    pub(super) timesteps_per_timeblock: usize,
    pub(super) channels_per_chanblock: usize,
    pub(super) num_timeblocks: usize,
    pub(super) num_chanblocks: usize,
    pub(super) uvw_min: (f64, WavelengthUnit),
    pub(super) uvw_max: (f64, WavelengthUnit),
    /// The number of baselines to use in calibration.
    pub(super) num_calibration_baselines: usize,
    /// The number of total number of baselines.
    pub(super) total_num_baselines: usize,
    /// If the user specified UVW cutoffs in terms of wavelength, we need to
    /// come up with our own lambda to convert the cutoffs to metres (we use the
    /// centroid frequency of the observation). \[metres\]
    pub(super) lambda: f64,
    /// \[Hz\]
    pub(super) freq_centroid: f64,
    pub(super) min_threshold: f64,
    pub(super) stop_threshold: f64,
    pub(super) max_iterations: usize,
}

impl CalibrationDetails {
    pub(super) fn print(self) {
        // I'm quite bored right now.
        let timeblock_plural = if self.num_timeblocks > 1 {
            "timeblocks"
        } else {
            "timeblock"
        };
        let chanblock_plural = if self.num_chanblocks > 1 {
            "chanblocks"
        } else {
            "chanblock"
        };

        info!(
            "{} calibration {timeblock_plural}, {} calibration {chanblock_plural}",
            self.num_timeblocks, self.num_chanblocks
        );
        info!("  {} timesteps per timeblock", self.timesteps_per_timeblock);
        info!("  {} channels per chanblock", self.channels_per_chanblock);

        // Report extra info if we need to use our own lambda (the user
        // specified wavelengths).
        if matches!(self.uvw_min.1, WavelengthUnit::L)
            || matches!(self.uvw_max.1, WavelengthUnit::L)
        {
            info!(
                "Using observation centroid frequency {} MHz to convert lambdas to metres",
                self.freq_centroid / 1e6
            );
        }

        info!(
            "Calibrating with {} of {} baselines",
            self.num_calibration_baselines, self.total_num_baselines
        );
        match (self.uvw_min, self.uvw_min.0.is_infinite()) {
            // Again, bored.
            (_, true) => info!("  Minimum UVW cutoff: ∞"),
            ((quantity, WavelengthUnit::M), _) => info!("  Minimum UVW cutoff: {quantity}m"),
            ((quantity, WavelengthUnit::L), _) => info!(
                "  Minimum UVW cutoff: {quantity}λ ({:.3}m)",
                quantity * self.lambda
            ),
        }
        match (self.uvw_max, self.uvw_max.0.is_infinite()) {
            (_, true) => info!("  Maximum UVW cutoff: ∞"),
            ((quantity, WavelengthUnit::M), _) => info!("  Maximum UVW cutoff: {quantity}m"),
            ((quantity, WavelengthUnit::L), _) => info!(
                "  Maximum UVW cutoff: {quantity}λ ({:.3}m)",
                quantity * self.lambda
            ),
        }

        info!("Chanblocks will stop iterating");
        info!(
            "  when the error is less than {:e} (stop threshold)",
            self.stop_threshold
        );
        info!("  or after {} iterations.", self.max_iterations);
        info!(
            "Chanblocks with an error less than {:e} are considered converged (min. threshold)",
            self.min_threshold
        )
    }
}

#[must_use = "This struct must be consumed with its print() method"]
pub(super) struct SkyModelDetails<'a> {
    pub(super) source_list: &'a SourceList,
}

impl SkyModelDetails<'_> {
    pub(super) fn print(self) {
        let mwa_hyperdrive_srclist::ComponentCounts {
            num_points,
            num_gaussians,
            num_shapelets,
            ..
        } = self.source_list.get_counts();
        let num_components = num_points + num_gaussians + num_shapelets;
        info!(
            "Using {} sources with a total of {} components",
            self.source_list.len(),
            num_components
        );
        info!("  {num_points} points, {num_gaussians} Gaussians, {num_shapelets} shapelet");
        if num_components > 10000 {
            warn!("Using more than 10,000 components!");
        }
        trace!("Using sources: {:?}", self.source_list.keys());
    }
}

#[must_use = "This struct must be consumed with its print() method"]
pub(super) struct OutputFileDetails<'a> {
    pub(super) output_solutions: &'a [PathBuf],
    pub(super) output_vis: &'a [PathBuf],
    /// \[seconds\]
    pub(super) input_vis_time_res: Option<f64>,
    /// \[Hz\]
    pub(super) input_vis_freq_res: Option<f64>,
    pub(super) output_vis_time_average_factor: usize,
    pub(super) output_vis_freq_average_factor: usize,
}

impl OutputFileDetails<'_> {
    pub(super) fn print(self) {
        if !self.output_solutions.is_empty() {
            info!(
                "Writing calibration solutions to: {}",
                self.output_solutions
                    .iter()
                    .map(|pb| pb.display())
                    .join(", ")
            );
        }
        if !self.output_vis.is_empty() {
            info!(
                "Writing calibrated visibilities to: {}",
                self.output_vis.iter().map(|pb| pb.display()).join(", ")
            );

            info!("Averaging output calibrated visibilities");
            if let Some(tr) = self.input_vis_time_res {
                info!(
                    "  {}x in time  ({}s)",
                    self.output_vis_time_average_factor,
                    tr * self.output_vis_time_average_factor as f64
                );
            } else {
                info!(
                    "  {}x (only one timestep)",
                    self.output_vis_time_average_factor
                );
            }

            if let Some(fr) = self.input_vis_freq_res {
                info!(
                    "  {}x in freq. ({}kHz)",
                    self.output_vis_freq_average_factor,
                    fr * self.output_vis_freq_average_factor as f64 / 1000.0
                );
            } else {
                info!(
                    "  {}x (only one fine channel)",
                    self.output_vis_freq_average_factor
                );
            }
        }
    }
}

// It looks a bit neater to print out a collection of numbers as a range rather
// than individual indicies if they're sequential. This function inspects a
// collection and returns a string to be printed.
fn range_or_comma_separated(collection: &[usize], prefix: Option<&str>) -> String {
    if collection.is_empty() {
        return "".to_string();
    }

    let mut iter = collection.iter();
    let mut prev = *iter.next().unwrap();
    // Innocent until proven guilty.
    let mut is_sequential = true;
    for next in iter {
        if *next == prev + 1 {
            prev = *next;
        } else {
            is_sequential = false;
            break;
        }
    }

    if is_sequential {
        let suffix = if collection.len() == 1 {
            format!("[{}]", collection[0])
        } else {
            format!(
                "[{:?})",
                (*collection.first().unwrap()..*collection.last().unwrap() + 1)
            )
        };
        if let Some(p) = prefix {
            format!("{} {}", p, suffix)
        } else {
            suffix
        }
    } else {
        let suffix = collection
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        if let Some(p) = prefix {
            format!("{} [{}]", p, suffix)
        } else {
            suffix
        }
    }
}
