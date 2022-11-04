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

use hifitime::{Duration, Epoch};
use itertools::Itertools;
use log::{info, trace, warn};
use marlu::{LatLngHeight, RADec};
use ndarray::prelude::*;
use vec1::Vec1;

use crate::{
    flagging::MwafFlags,
    model::ModellerInfo,
    pfb_gains::PfbFlavour,
    solutions::CalibrationSolutions,
    srclist::{ComponentCounts, SourceList},
    unit_parsing::WavelengthUnit,
    vis_io::{read::RawDataCorrections, write::VisOutputType},
};

#[must_use = "This struct must be consumed with its print() method"]
pub(super) enum InputFileDetails<'a> {
    Raw {
        obsid: u32,
        gpubox_count: usize,
        metafits_file_name: String,
        mwaf: Option<&'a MwafFlags>,
        raw_data_corrections: RawDataCorrections,
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

impl InputFileDetails<'_> {
    pub(super) fn print(self, operation: &str) {
        match self {
            InputFileDetails::Raw {
                obsid,
                gpubox_count,
                metafits_file_name,
                mwaf,
                raw_data_corrections,
            } => {
                info!("{operation} obsid {obsid}");
                info!("  from {gpubox_count} gpubox files");
                info!("  with metafits {metafits_file_name}");
                match mwaf {
                    Some(flags) => {
                        let software_string = match flags.software_version.as_ref() {
                            Some(v) => format!("{} {}", flags.software, v),
                            None => flags.software.to_string(),
                        };
                        info!(
                            "  with {} mwaf files ({})",
                            flags.gpubox_nums.len(),
                            software_string,
                        );
                        if let Some(s) = flags.aoflagger_version.as_deref() {
                            info!("    AOFlagger version: {s}");
                        }
                        if let Some(s) = flags.aoflagger_strategy.as_deref() {
                            info!("    AOFlagger strategy: {s}");
                        }
                    }
                    None => warn!("No mwaf files supplied"),
                }

                let s = "Correcting PFB gains";
                match raw_data_corrections.pfb_flavour {
                    PfbFlavour::None => info!("Not doing any PFB correction"),
                    PfbFlavour::Jake => info!("{s} with 'Jake Jones' gains"),
                    PfbFlavour::Cotter2014 => info!("{s} with 'Cotter 2014' gains"),
                    PfbFlavour::Empirical => info!("{s} with 'RTS empirical' gains"),
                    PfbFlavour::Levine => info!("{s} with 'Alan Levine' gains"),
                }
                if raw_data_corrections.digital_gains {
                    info!("Correcting digital gains");
                } else {
                    info!("Not correcting digital gains");
                }
                if raw_data_corrections.cable_length {
                    info!("Correcting cable lengths");
                } else {
                    info!("Not correcting cable lengths");
                }
                if raw_data_corrections.geometric {
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
                    info!("{operation} obsid {o}");
                    info!("  from measurement set {file_name}");
                } else {
                    info!("{operation} measurement set {file_name}");
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
                    info!("{operation} obsid {o}");
                    info!("  from uvfits {file_name}");
                } else {
                    info!("{operation} uvfits {file_name}");
                }
                if let Some(f) = metafits_file_name {
                    info!("  with metafits {f}");
                }
            }
        }
    }
}

#[must_use = "This struct must be consumed with its print() method"]
pub(super) struct ArrayDetails<'a> {
    pub(super) array_position: Option<LatLngHeight>,
    /// \[radians\]
    pub(super) array_latitude_j2000: Option<f64>,
    pub(super) total_num_tiles: usize,
    pub(super) num_unflagged_tiles: usize,
    pub(super) flagged_tiles: &'a [(&'a str, usize)],
}

impl ArrayDetails<'_> {
    pub(super) fn print(self) {
        if let Some(pos) = self.array_position {
            info!(
                "Array latitude:         {:>8.4}°",
                pos.latitude_rad.to_degrees()
            );
        }
        if let Some(rad) = self.array_latitude_j2000 {
            info!("Array latitude (J2000): {:>8.4}°", rad.to_degrees());
        }
        if let Some(pos) = self.array_position {
            info!(
                "Array longitude:       {:>9.4}°",
                pos.longitude_rad.to_degrees()
            );
            info!("Array height:          {:>9.4}m", pos.height_metres);
        }

        info!("Total number of tiles:     {:>3}", self.total_num_tiles);
        info!("Number of unflagged tiles: {:>3}", self.num_unflagged_tiles);
        info!("Flagged tiles:             {:?}", self.flagged_tiles);
    }
}

#[must_use = "This struct must be consumed with its print() method"]
pub(super) struct ObservationDetails<'a> {
    /// If this is `None`, no dipole delay or alive/dead status reporting is
    /// done.
    pub(super) dipole_delays: Option<[u32; 16]>,
    pub(super) beam_file: Option<&'a Path>,
    /// If this is `None`, then report that we're assuming all dipoles are
    /// "alive".
    pub(super) num_tiles_with_dead_dipoles: Option<usize>,

    pub(super) phase_centre: RADec,
    pub(super) pointing_centre: Option<RADec>,
    /// Only printed if it's populated.
    pub(super) dut1: Option<Duration>,
    /// The local mean sidereal time of the first timestep \[radians\]
    pub(super) lmst: Option<f64>,
    /// The local mean sidereal time of the first timestep, precessed to the
    /// J2000 epoch \[radians\]
    pub(super) lmst_j2000: Option<f64>,

    pub(super) available_timesteps: Option<&'a [usize]>,
    pub(super) unflagged_timesteps: Option<&'a [usize]>,
    pub(super) using_timesteps: Option<&'a [usize]>,
    pub(super) first_timestamp: Option<Epoch>,
    pub(super) last_timestamp: Option<Epoch>,
    pub(super) time_res: Option<Duration>,

    pub(super) total_num_channels: usize,
    pub(super) num_unflagged_channels: Option<usize>,
    pub(super) flagged_chans_per_coarse_chan: Option<&'a [usize]>,
    pub(super) first_freq_hz: Option<f64>,
    pub(super) last_freq_hz: Option<f64>,
    pub(super) first_unflagged_freq_hz: Option<f64>,
    pub(super) last_unflagged_freq_hz: Option<f64>,
    pub(super) freq_res_hz: Option<f64>,
}

impl ObservationDetails<'_> {
    pub(super) fn print(self) {
        if let Some(d) = self.dipole_delays {
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
            // No need to report additional beam information if there are no
            // dipole delays; this implies that beam code isn't being used.
            if let Some(beam_file) = self.beam_file {
                info!("Using beam file {}", beam_file.display());
            }
            if let Some(num_tiles_with_dead_dipoles) = self.num_tiles_with_dead_dipoles {
                info!(
                    "Using dead dipole information ({num_tiles_with_dead_dipoles} tiles affected)"
                );
            } else {
                info!("Assuming all dipoles are \"alive\"");
            }
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

        if let Some(dut1) = self.dut1 {
            info!("DUT1: {} seconds", dut1.in_seconds());
        }
        match (self.lmst, self.lmst_j2000) {
            (Some(l), Some(l2)) => {
                info!("LMST of first timestep:         {:>9.6}°", l.to_degrees());
                info!("LMST of first timestep (J2000): {:>9.6}°", l2.to_degrees());
            }
            (Some(l), None) => info!("LMST of first timestep: {:>9.6}°", l.to_degrees()),
            (None, Some(l2)) => info!("LMST of first timestep (J2000): {:>9.6}°", l2.to_degrees()),
            (None, None) => (),
        }

        if let Some(available_timesteps) = self.available_timesteps {
            info!(
                "{}",
                range_or_comma_separated(available_timesteps, Some("Available timesteps:"))
            );
        }
        if let Some(unflagged_timesteps) = self.unflagged_timesteps {
            info!(
                "{}",
                range_or_comma_separated(unflagged_timesteps, Some("Unflagged timesteps:"))
            );
        }
        // We don't require the timesteps to be used in calibration to be
        // sequential. But if they are, it looks a bit neater to print them out
        // as a range rather than individual indices.
        if let Some(using_timesteps) = self.using_timesteps {
            info!(
                "{}",
                range_or_comma_separated(using_timesteps, Some("Using timesteps:    "))
            );
        }
        match (
            self.first_timestamp,
            self.last_timestamp,
            self.first_timestamp.or(self.last_timestamp),
        ) {
            (Some(f), Some(l), _) => {
                info!("First timestamp (GPS): {:.2}", f.as_gpst_seconds());
                info!("Last timestamp  (GPS): {:.2}", l.as_gpst_seconds());
            }
            (_, _, Some(f)) => info!("Only timestamp (GPS): {:.2}", f.as_gpst_seconds()),
            _ => (),
        }
        match self.time_res {
            Some(r) => info!("Input data time resolution: {:.2} seconds", r.in_seconds()),
            None => info!("Input data time resolution unknown"),
        }

        match self.num_unflagged_channels {
            Some(num_unflagged_channels) => {
                info!(
                    "Total number of fine channels:     {}",
                    self.total_num_channels
                );
                info!(
                    "Number of unflagged fine channels: {}",
                    num_unflagged_channels
                );
            }
            None => {
                info!("Total number of fine channels: {}", self.total_num_channels);
            }
        }
        if let Some(flagged_chans_per_coarse_chan) = self.flagged_chans_per_coarse_chan {
            info!(
                "Input data's fine-channel flags per coarse channel: {:?}",
                flagged_chans_per_coarse_chan
            );
        }
        match (
            self.first_freq_hz,
            self.last_freq_hz,
            self.first_unflagged_freq_hz,
            self.last_unflagged_freq_hz,
        ) {
            (Some(f), Some(l), Some(fu), Some(lu)) => {
                info!("First fine-channel frequency:           {:.3} MHz", f / 1e6);
                info!(
                    "First unflagged fine-channel frequency: {:.3} MHz",
                    fu / 1e6
                );
                info!("Last fine-channel frequency:            {:.3} MHz", l / 1e6);
                info!(
                    "Last unflagged fine-channel frequency:  {:.3} MHz",
                    lu / 1e6
                );
            }
            (Some(f), Some(l), None, None) => {
                info!("First fine-channel frequency: {:.3} MHz", f / 1e6);
                info!("Last fine-channel frequency:  {:.3} MHz", l / 1e6);
            }
            (None, None, Some(f), Some(l)) => {
                info!("First unflagged fine-channel frequency: {:.3} MHz", f / 1e6);
                info!("Last unflagged fine-channel frequency:  {:.3} MHz", l / 1e6);
            }
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
    pub(super) max_iterations: u32,
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
        let ComponentCounts {
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
        info!("  {num_points} points, {num_gaussians} Gaussians, {num_shapelets} shapelets");
        if num_components > 10000 {
            warn!("Using more than 10,000 components!");
        }
        if log::log_enabled!(log::Level::Trace) {
            trace!("Using sources:");
            let mut v = Vec::with_capacity(5);
            for source in self.source_list.keys() {
                if v.len() == 5 {
                    trace!("  {v:?}");
                    v.clear();
                }
                v.push(source);
            }
            if !v.is_empty() {
                trace!("  {v:?}");
            }
        }
    }
}

#[must_use = "This struct must be consumed with its print() method"]
pub(super) struct OutputFileDetails<'a> {
    pub(super) output_solutions: &'a [PathBuf],
    pub(super) vis_type: &'a str,
    pub(super) output_vis: Option<&'a Vec1<(PathBuf, VisOutputType)>>,
    pub(super) input_vis_time_res: Option<Duration>,
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
        if let Some(output_vis) = self.output_vis {
            info!(
                "Writing {} visibilities to: {}",
                self.vis_type,
                output_vis.iter().map(|pb| pb.0.display()).join(", ")
            );

            if self.output_vis_time_average_factor != 1 || self.output_vis_freq_average_factor != 1
            {
                info!("Averaging output visibilities");
                if let Some(tr) = self.input_vis_time_res {
                    info!(
                        "  {}x in time  ({}s)",
                        self.output_vis_time_average_factor,
                        tr.in_seconds() * self.output_vis_time_average_factor as f64
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
}

#[must_use = "This struct must be consumed with its print() method"]
pub(super) struct CalSolDetails<'a> {
    pub(super) filename: &'a Path,
    pub(super) sols: &'a CalibrationSolutions,
}

impl CalSolDetails<'_> {
    pub(super) fn print(self) {
        let s = self.sols;
        let num_timeblocks = s.di_jones.len_of(Axis(0));
        info!(
            "Using calibration solutions from {}",
            self.filename.display()
        );
        info!(
            "  {num_timeblocks} timeblocks, {} tiles, {} chanblocks",
            s.di_jones.len_of(Axis(1)),
            s.di_jones.len_of(Axis(2))
        );

        if let Some(c) = s.raw_data_corrections {
            info!("  Raw data corrections:");
            info!("    PFB flavour: {}", c.pfb_flavour);
            info!(
                "    digital gains: {}",
                match c.digital_gains {
                    true => "yes",
                    false => "no",
                }
            );
            info!(
                "    cable lengths: {}",
                match c.cable_length {
                    true => "yes",
                    false => "no",
                }
            );
            info!(
                "    geometric delays: {}",
                match c.geometric {
                    true => "yes",
                    false => "no",
                }
            );
        } else {
            info!("  No raw data correction information");
        }

        // If there's more than one timeblock, we can report dodgy-looking
        // solutions based on the available metadata.
        if num_timeblocks > 1
            && match (
                &s.start_timestamps,
                &s.end_timestamps,
                &s.average_timestamps,
            ) {
                // Are all types of timestamps available?
                (Some(s), Some(e), Some(a)) => {
                    // Are all the lengths the same?
                    num_timeblocks != s.len()
                        || num_timeblocks != e.len()
                        || num_timeblocks != a.len()
                }
                _ => true,
            }
        {
            warn!("  Time information is inconsistent; solution timeblocks");
            warn!("  may not be applied properly. hyperdrive-formatted");
            warn!("  solutions should be used to prevent this issue.");
        }
    }
}

pub(super) fn print_modeller_info(modeller_info: &ModellerInfo) {
    #[cfg(feature = "cuda")]
    let using_cuda = matches!(modeller_info, crate::model::ModellerInfo::Cuda { .. });
    #[cfg(not(feature = "cuda"))]
    let using_cuda = false;

    if using_cuda {
        cfg_if::cfg_if! {
            if #[cfg(feature = "cuda-single")] {
                info!("Generating sky model visibilities on the GPU (single precision)");
            } else {
                info!("Generating sky model visibilities on the GPU (double precision)");
            }
        }
    } else {
        info!("Generating sky model visibilities on the CPU (double precision)");
    }

    match modeller_info {
        crate::model::ModellerInfo::Cpu => (),

        #[cfg(feature = "cuda")]
        crate::model::ModellerInfo::Cuda {
            device_info,
            driver_info,
        } => {
            info!(
                "  CUDA device: {} (capability {}, {} MiB)",
                device_info.name, device_info.capability, device_info.total_global_mem
            );
            info!(
                "  CUDA driver: {}, runtime: {}",
                driver_info.driver_version, driver_info.runtime_version
            );
        }
    }
}

// It looks a bit neater to print out a collection of numbers as a range rather
// than individual indices if they're sequential. This function inspects a
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
            format!("{p} {suffix}")
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
            format!("{p} [{suffix}]")
        } else {
            suffix
        }
    }
}
