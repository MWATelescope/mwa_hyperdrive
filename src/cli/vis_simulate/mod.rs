// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Generate sky-model visibilities from a sky-model source list.

// TODO: Utilise tile flags.
// TODO: Allow the user to specify the mwa_version for the metafits file.

#[cfg(test)]
mod tests;

use std::{
    collections::HashSet,
    path::{Path, PathBuf},
};

use clap::Parser;
use console::style;
use hifitime::{Duration, Epoch};
use log::{debug, info, trace};
use marlu::{precession::precess_time, LatLngHeight, RADec, XyzGeodetic};
use mwalib::MetafitsContext;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use vec1::Vec1;

use super::common::{
    display_warnings, BeamArgs, ModellingArgs, OutputVisArgs, SkyModelWithVetoArgs, ARG_FILE_HELP,
    ARRAY_POSITION_HELP,
};
use crate::{
    beam::Delays,
    cli::common::InfoPrinter,
    io::write::VIS_OUTPUT_EXTENSIONS,
    math::TileBaselineFlags,
    metafits::{get_dipole_delays, get_dipole_gains},
    params::VisSimulateParams,
    srclist::ComponentCounts,
    HyperdriveError,
};

const DEFAULT_OUTPUT_VIS_FILENAME: &str = "hyp_model.uvfits";
const DEFAULT_NUM_FINE_CHANNELS: usize = 384;
const DEFAULT_FREQ_RES_KHZ: f64 = 80.0;
const DEFAULT_NUM_TIMESTEPS: usize = 14;
const DEFAULT_TIME_RES_SECONDS: f64 = 8.0;

lazy_static::lazy_static! {
    static ref NUM_FINE_CHANNELS_HELP: String =
        format!("The total number of fine channels in the observation. Default: {DEFAULT_NUM_FINE_CHANNELS}");

    static ref FREQ_RES_HELP: String =
        format!("The fine-channel resolution [kHz]. Default: {DEFAULT_FREQ_RES_KHZ}");

    static ref NUM_TIMESTEPS_HELP: String =
        format!("The number of time steps used from the metafits epoch. Default: {DEFAULT_NUM_TIMESTEPS}");

    static ref TIME_RES_HELP: String =
        format!("The time resolution [seconds]. Default: {DEFAULT_TIME_RES_SECONDS}");

    static ref OUTPUTS_HELP: String =
        format!("Paths to the output visibility files. Supported formats: {}. Default: {}", *VIS_OUTPUT_EXTENSIONS, DEFAULT_OUTPUT_VIS_FILENAME);
}

#[derive(Parser, Debug, Clone, Default, Serialize, Deserialize)]
pub(super) struct VisSimulateCliArgs {
    /// Path to the metafits file.
    #[clap(short, long, parse(from_str), help_heading = "INPUT FILES")]
    pub(super) metafits: Option<PathBuf>,

    /// Use this value as the DUT1 [seconds].
    #[clap(long, help_heading = "INPUT DATA")]
    #[serde(default)]
    pub(super) dut1: Option<f64>,

    /// Use a DUT1 value of 0 seconds rather than what is in the metafits file.
    #[clap(long, conflicts_with("dut1"), help_heading = "INPUT FILES")]
    pub(super) ignore_dut1: bool,

    /// The phase centre right ascension [degrees]. If this is not specified,
    /// then the metafits phase/pointing centre is used.
    #[clap(short, long, help_heading = "OBSERVATION PARAMETERS")]
    pub(super) ra: Option<f64>,

    /// The phase centre declination [degrees]. If this is not specified, then
    /// the metafits phase/pointing centre is used.
    #[clap(short, long, help_heading = "OBSERVATION PARAMETERS")]
    pub(super) dec: Option<f64>,

    #[clap(
        short = 'c',
        long,
        help = NUM_FINE_CHANNELS_HELP.as_str(),
        help_heading = "OBSERVATION PARAMETERS"
    )]
    pub(super) num_fine_channels: Option<usize>,

    #[clap(
        short,
        long,
        help = FREQ_RES_HELP.as_str(),
        help_heading = "OBSERVATION PARAMETERS"
    )]
    pub(super) freq_res: Option<f64>,

    /// The centroid frequency of the simulation [MHz]. If this is not
    /// specified, then the FREQCENT specified in the metafits is used.
    #[clap(long, help_heading = "OBSERVATION PARAMETERS")]
    pub(super) middle_freq: Option<f64>,

    #[clap(
        short = 't',
        long,
        help = NUM_TIMESTEPS_HELP.as_str(),
        help_heading = "OBSERVATION PARAMETERS"
    )]
    pub(super) num_timesteps: Option<usize>,

    #[clap(long, help = TIME_RES_HELP.as_str(), help_heading = "OBSERVATION PARAMETERS")]
    pub(super) time_res: Option<f64>,

    /// The time offset from the start [seconds]. The default start time is the
    /// is the obsid.
    #[clap(long, help_heading = "OBSERVATION PARAMETERS")]
    pub(super) time_offset: Option<f64>,

    #[clap(
        long, help = ARRAY_POSITION_HELP.as_str(), help_heading = "OBSERVATION PARAMETERS",
        number_of_values = 3,
        allow_hyphen_values = true,
        value_names = &["LONG_DEG", "LAT_DEG", "HEIGHT_M"]
    )]
    pub(super) array_position: Option<Vec<f64>>,

    #[clap(
        short = 'o',
        long,
        multiple_values(true),
        help = OUTPUTS_HELP.as_str(),
        help_heading = "OUTPUT FILES"
    )]
    pub(super) output_model_files: Option<Vec<PathBuf>>,

    /// When writing out model visibilities, average this many timesteps
    /// together. Also supports a target time resolution (e.g. 8s). The value
    /// must be a multiple of the input data's time resolution. The default is
    /// no averaging, i.e. a value of 1. Examples: If the input data is in 0.5s
    /// resolution and this variable is 4, then we average 2s worth of data
    /// together before writing the data out. If the variable is instead 4s,
    /// then 8 timesteps are averaged together before writing the data out.
    #[clap(long, help_heading = "OUTPUT FILES")]
    pub(super) output_model_time_average: Option<String>,

    /// When writing out model visibilities, average this many fine freq.
    /// channels together. Also supports a target freq. resolution (e.g. 80kHz).
    /// The value must be a multiple of the input data's freq. resolution. The
    /// default is no averaging, i.e. a value of 1. Examples: If the input data
    /// is in 40kHz resolution and this variable is 4, then we average 160kHz
    /// worth of data together before writing the data out. If the variable is
    /// instead 80kHz, then 2 fine freq. channels are averaged together before
    /// writing the data out.
    #[clap(long, help_heading = "OUTPUT FILES")]
    pub(super) output_model_freq_average: Option<String>,

    /// Include auto-correlations in the output.
    /// Default: don't include autos
    #[clap(long, help_heading = "OUTPUT FILES")]
    pub(super) output_autos: bool,

    /// Remove any "point" components from the input sky model.
    #[clap(long, help_heading = "SKY MODEL")]
    pub(super) filter_points: bool,

    /// Remove any "gaussian" components from the input sky model.
    #[clap(long, help_heading = "SKY MODEL")]
    pub(super) filter_gaussians: bool,

    /// Remove any "shapelet" components from the input sky model.
    #[clap(long, help_heading = "SKY MODEL")]
    pub(super) filter_shapelets: bool,
}

#[derive(Parser, Debug, Clone, Default, Serialize, Deserialize)]
pub(super) struct VisSimulateArgs {
    #[clap(name = "ARGUMENTS_FILE", help = ARG_FILE_HELP.as_str(), parse(from_os_str))]
    pub(super) args_file: Option<PathBuf>,

    #[clap(flatten)]
    #[serde(rename = "beam")]
    #[serde(default)]
    pub(super) beam_args: BeamArgs,

    #[clap(flatten)]
    #[serde(rename = "model")]
    #[serde(default)]
    pub(super) modelling_args: ModellingArgs,

    #[clap(flatten)]
    #[serde(rename = "sky-model")]
    #[serde(default)]
    pub(super) srclist_args: SkyModelWithVetoArgs,

    #[clap(flatten)]
    #[serde(rename = "vis-simulate")]
    #[serde(default)]
    pub(super) simulate_args: VisSimulateCliArgs,
}

impl VisSimulateArgs {
    /// Both command-line and file arguments overlap in terms of what is
    /// available; this function consolidates everything that was specified into
    /// a single struct. Where applicable, it will prefer CLI parameters over
    /// those in the file.
    ///
    /// The argument to this function is the path to the arguments file.
    ///
    /// This function should only ever merge arguments, and not try to make
    /// sense of them.
    pub(super) fn merge(self) -> Result<VisSimulateArgs, HyperdriveError> {
        debug!("Merging command-line arguments with the argument file");

        let cli_args = self;

        if let Some(arg_file) = cli_args.args_file {
            // Read in the file arguments. Ensure all of the file args are
            // accounted for by pattern matching.
            let VisSimulateArgs {
                args_file: _,
                beam_args,
                modelling_args,
                srclist_args,
                simulate_args,
            } = unpack_arg_file!(arg_file);

            // Merge all the arguments, preferring the CLI args when available.
            Ok(VisSimulateArgs {
                args_file: None,
                beam_args: cli_args.beam_args.merge(beam_args),
                modelling_args: cli_args.modelling_args.merge(modelling_args),
                srclist_args: cli_args.srclist_args.merge(srclist_args),
                simulate_args: cli_args.simulate_args.merge(simulate_args),
            })
        } else {
            Ok(cli_args)
        }
    }

    fn parse(self) -> Result<VisSimulateParams, HyperdriveError> {
        debug!("{:#?}", self);

        // Expose all the struct fields to ensure they're all used.
        let VisSimulateArgs {
            args_file: _,
            beam_args,
            modelling_args,
            srclist_args,
            simulate_args:
                VisSimulateCliArgs {
                    metafits,
                    dut1,
                    ignore_dut1,
                    ra,
                    dec,
                    num_fine_channels,
                    freq_res,
                    middle_freq,
                    num_timesteps,
                    time_res,
                    time_offset,
                    array_position,
                    output_model_files,
                    output_model_time_average,
                    output_model_freq_average,
                    output_autos,
                    filter_points,
                    filter_gaussians,
                    filter_shapelets,
                },
        } = self;

        // Read the metafits file with mwalib.
        let metafits = if let Some(metafits) = metafits {
            if !metafits.exists() {
                return Err(
                    VisSimulateArgsError::MetafitsDoesntExist(metafits.into_boxed_path()).into(),
                );
            }
            MetafitsContext::new(metafits, None)?
        } else {
            return Err(VisSimulateArgsError::NoMetafits.into());
        };

        let mut metadata_printer = InfoPrinter::new(
            format!("Simulating visibilities for obsid {}", metafits.obs_id).into(),
        );
        metadata_printer.push_line(format!("with {}", metafits.metafits_filename).into());
        metadata_printer.display();

        let mut coord_printer = InfoPrinter::new("Coordinates".into());
        // Get the phase centre.
        let phase_centre = match (ra, dec, &metafits) {
            (Some(ra), Some(dec), _) => {
                // Verify that the input coordinates are sensible.
                if !(0.0..=360.0).contains(&ra) {
                    return Err(VisSimulateArgsError::RaInvalid.into());
                }
                if !(-90.0..=90.0).contains(&dec) {
                    return Err(VisSimulateArgsError::DecInvalid.into());
                }
                RADec::from_degrees(ra, dec)
            }
            (Some(_), None, _) => return Err(VisSimulateArgsError::OnlyOneRAOrDec.into()),
            (None, Some(_), _) => return Err(VisSimulateArgsError::OnlyOneRAOrDec.into()),
            (None, None, m) => {
                // The phase centre in a metafits file may not be present. If not,
                // we have to use the pointing centre.
                match (m.ra_phase_center_degrees, m.dec_phase_center_degrees) {
                    (Some(ra), Some(dec)) => RADec::from_degrees(ra, dec),
                    (None, None) => {
                        RADec::from_degrees(m.ra_tile_pointing_degrees, m.dec_tile_pointing_degrees)
                    }
                    _ => unreachable!(),
                }
            }
        };
        let mut block = vec![];
        block.push(
            style("                   RA        Dec")
                .bold()
                .to_string()
                .into(),
        );
        if let Some((ra, dec)) = metafits
            .ra_phase_center_degrees
            .zip(metafits.dec_phase_center_degrees)
        {
            block.push(format!("Phase centre:      {:>8.4}° {:>8.4}° (J2000)", ra, dec).into());
        }
        block.push(
            format!(
                "Pointing centre:   {:>8.4}° {:>8.4}°",
                metafits.ra_tile_pointing_degrees, metafits.dec_tile_pointing_degrees
            )
            .into(),
        );
        coord_printer.push_block(block);

        // If the user supplied the array position, unpack it here.
        let array_position = match array_position {
            Some(v) => {
                if v.len() != 3 {
                    return Err(VisSimulateArgsError::BadArrayPosition { pos: v }.into());
                }
                LatLngHeight {
                    longitude_rad: v[0].to_radians(),
                    latitude_rad: v[1].to_radians(),
                    height_metres: v[2],
                }
            }
            None => LatLngHeight::mwa(),
        };
        coord_printer.push_line(
            format!(
                "Array position:    {:>8.4}° {:>8.4}° {:.4}m",
                array_position.longitude_rad.to_degrees(),
                array_position.latitude_rad.to_degrees(),
                array_position.height_metres
            )
            .into(),
        );
        coord_printer.display();

        // Get the geodetic XYZ coordinates of each of the MWA tiles.
        let tile_xyzs = XyzGeodetic::get_tiles(&metafits, array_position.latitude_rad);
        let tile_names: Vec<String> = metafits
            .antennas
            .iter()
            .map(|a| a.tile_name.clone())
            .collect();

        // Prepare a map between baselines and their constituent tiles.
        let flagged_tiles = HashSet::new();
        let tile_baseline_flags = TileBaselineFlags::new(metafits.num_ants, flagged_tiles);

        let mut tile_printer = InfoPrinter::new("Tile info".into());
        tile_printer.push_line(format!("{} tiles", tile_xyzs.len()).into());
        tile_printer.display();

        let time_res = Duration::from_seconds(time_res.unwrap_or(DEFAULT_TIME_RES_SECONDS));
        let timestamps = {
            let num_timesteps = num_timesteps.unwrap_or(DEFAULT_NUM_TIMESTEPS);
            let mut timestamps = Vec::with_capacity(num_timesteps);
            let start_ns = metafits
                .sched_start_gps_time_ms
                .checked_mul(1_000_000)
                .expect("does not overflow u64");
            let start = Epoch::from_gpst_nanoseconds(start_ns)
                + time_res / 2
                + Duration::from_seconds(time_offset.unwrap_or_default());
            for i in 0..num_timesteps {
                timestamps.push(start + time_res * i as i64);
            }
            Vec1::try_from_vec(timestamps).map_err(|_| VisSimulateArgsError::ZeroTimeSteps)?
        };
        let dut1 = match (ignore_dut1, dut1) {
            (true, _) => {
                debug!("Ignoring metafits and user DUT1");
                Duration::default()
            }
            (false, Some(dut1)) => {
                debug!("Using user DUT1");
                Duration::from_seconds(dut1)
            }
            (false, None) => {
                debug!("Using metafits DUT1");
                metafits
                    .dut1
                    .map(Duration::from_seconds)
                    .unwrap_or_default()
            }
        };
        let precession_info = precess_time(
            array_position.longitude_rad,
            array_position.latitude_rad,
            phase_centre,
            *timestamps.first(),
            dut1,
        );
        let (lst_rad, latitude_rad) = if !modelling_args.no_precession {
            (
                precession_info.lmst_j2000,
                precession_info.array_latitude_j2000,
            )
        } else {
            (precession_info.lmst, array_position.latitude_rad)
        };

        let mut time_printer = InfoPrinter::new("Time info".into());
        time_printer.push_line(format!("Simulating at resolution: {time_res}").into());
        time_printer.push_block(vec![
            format!("First timestamp: {}", timestamps.first()).into(),
            format!(
                "First timestamp (GPS): {}",
                timestamps.first().to_gpst_seconds()
            )
            .into(),
            format!(
                "Last timestamp (GPS):  {}",
                timestamps.last().to_gpst_seconds()
            )
            .into(),
            format!(
                "First LMST: {:.6}° (J2000)",
                precession_info.lmst_j2000.to_degrees()
            )
            .into(),
        ]);
        time_printer.push_line(format!("DUT1: {:.10} s", dut1.to_seconds()).into());
        time_printer.display();

        // Get the fine channel frequencies.
        let freq_res = freq_res.unwrap_or(DEFAULT_FREQ_RES_KHZ);
        let num_fine_channels = num_fine_channels.unwrap_or(DEFAULT_NUM_FINE_CHANNELS);
        if freq_res < f64::EPSILON {
            return Err(VisSimulateArgsError::FineChansWidthTooSmall.into());
        }
        let middle_freq = middle_freq
            .map(|f| f * 1e6) // MHz -> Hz
            .unwrap_or(metafits.centre_freq_hz as _);
        let freq_res = freq_res * 1e3; // kHz -> Hz
        let fine_chan_freqs = {
            let half_num_fine_chans = num_fine_channels as f64 / 2.0;
            let mut fine_chan_freqs = Vec::with_capacity(num_fine_channels);
            for i in 0..num_fine_channels {
                fine_chan_freqs
                    .push(middle_freq - half_num_fine_chans * freq_res + freq_res * i as f64);
            }
            Vec1::try_from_vec(fine_chan_freqs).map_err(|_| VisSimulateArgsError::FineChansZero)?
        };
        let coarse_chan_freqs = {
            let (mut coarse_chan_freqs, mut coarse_chan_nums): (Vec<f64>, Vec<u32>) =
                fine_chan_freqs
                    .iter()
                    .map(|&f| {
                        // MWA coarse channel numbers are a multiple of 1.28 MHz.
                        // This might change with MWAX, but ignore that until it
                        // becomes an issue; vis-simulate is mostly useful for
                        // testing.
                        let cc_num = (f / 1.28e6).round();
                        (cc_num * 1.28e6, cc_num as u32)
                    })
                    .unzip();
            // Deduplicate. As `fine_chan_freqs` is always sorted, we don't need
            // to sort here.
            coarse_chan_freqs.dedup();
            coarse_chan_nums.dedup();
            debug!("MWA coarse channel numbers: {coarse_chan_nums:?}");
            // Convert the coarse channel numbers to a range starting from 1.
            coarse_chan_freqs
        };
        debug!(
            "Coarse channel centre frequencies [Hz]: {:?}",
            coarse_chan_freqs
        );

        let mut chan_printer = InfoPrinter::new("Channel info".into());
        chan_printer
            .push_line(format!("Simulating at resolution: {:.2} kHz", freq_res / 1e3).into());
        chan_printer.push_block(vec![
            format!("Number of fine channels: {num_fine_channels}").into(),
            format!(
                "First fine-channel: {:.3} MHz",
                *fine_chan_freqs.first() / 1e6
            )
            .into(),
            format!(
                "Last fine-channel:  {:.3} MHz",
                *fine_chan_freqs.last() / 1e6
            )
            .into(),
        ]);
        chan_printer.display();

        let beam = beam_args.parse(
            metafits.num_ants,
            Some(Delays::Full(get_dipole_delays(&metafits))),
            Some(get_dipole_gains(&metafits)),
            None,
            Some(&tile_names),
        )?;
        let modelling_params = modelling_args.parse();

        let source_list = srclist_args.parse(
            phase_centre,
            lst_rad,
            latitude_rad,
            &coarse_chan_freqs,
            &*beam,
        )?;

        // Apply any filters.
        let source_list = if filter_points || filter_gaussians || filter_shapelets {
            let sl = source_list.filter(filter_points, filter_gaussians, filter_shapelets);
            let ComponentCounts {
                num_points,
                num_gaussians,
                num_shapelets,
                ..
            } = sl.get_counts();
            info!(
                "After filtering, there are {num_points} points, {num_gaussians} gaussians, {num_shapelets} shapelets"
            );
            sl
        } else {
            source_list
        };

        // Parse the output model vis args like normal output vis args, to
        // re-use existing code (we only make the args distinct to make it clear
        // that these visibilities are not calibrated, just the model vis).
        let output_vis_params = OutputVisArgs {
            outputs: output_model_files,
            output_vis_time_average: output_model_time_average,
            output_vis_freq_average: output_model_freq_average,
            output_autos,
        }
        .parse(
            time_res,
            freq_res,
            &timestamps,
            false,
            DEFAULT_OUTPUT_VIS_FILENAME,
            Some("simulated"),
        )?;

        display_warnings();

        Ok(VisSimulateParams {
            source_list,
            metafits,
            output_vis_params,
            phase_centre,
            fine_chan_freqs,
            freq_res_hz: freq_res,
            tile_xyzs,
            tile_names,
            tile_baseline_flags,
            timestamps,
            time_res,
            beam,
            array_position,
            dut1,
            modelling_params,
        })
    }

    pub(super) fn run(self, dry_run: bool) -> Result<(), HyperdriveError> {
        debug!("Converting arguments into parameters");
        trace!("{:#?}", self);
        let params = self.parse()?;

        if dry_run {
            info!("Dry run -- exiting now.");
            return Ok(());
        }

        params.run()?;
        Ok(())
    }
}

#[derive(Error, Debug)]
pub(super) enum VisSimulateArgsError {
    #[error("No metafits file was supplied")]
    NoMetafits,

    #[error("Metafits file '{0}' doesn't exist")]
    MetafitsDoesntExist(Box<Path>),

    #[error("Right Ascension was not within 0 to 360!")]
    RaInvalid,

    #[error("Declination was not within -90 to 90!")]
    DecInvalid,

    #[error("One of RA and Dec was specified, but none or both are required!")]
    OnlyOneRAOrDec,

    #[error("Number of fine channels cannot be 0!")]
    FineChansZero,

    #[error("The fine channel resolution cannot be 0 or negative!")]
    FineChansWidthTooSmall,

    #[error("Number of timesteps cannot be 0!")]
    ZeroTimeSteps,

    #[error("Array position specified as {pos:?}, not [<Longitude>, <Latitude>, <Height>]")]
    BadArrayPosition { pos: Vec<f64> },
}

impl VisSimulateCliArgs {
    fn merge(self, other: Self) -> Self {
        Self {
            metafits: self.metafits.or(other.metafits),
            dut1: self.dut1.or(other.dut1),
            ignore_dut1: self.ignore_dut1 || other.ignore_dut1,
            ra: self.ra.or(other.ra),
            dec: self.dec.or(other.dec),
            num_fine_channels: self.num_fine_channels.or(other.num_fine_channels),
            freq_res: self.freq_res.or(other.freq_res),
            middle_freq: self.middle_freq.or(other.middle_freq),
            num_timesteps: self.num_timesteps.or(other.num_timesteps),
            time_res: self.time_res.or(other.time_res),
            time_offset: self.time_offset.or(other.time_offset),
            array_position: self.array_position.or(other.array_position),
            output_model_files: self.output_model_files.or(other.output_model_files),
            output_model_time_average: self
                .output_model_time_average
                .or(other.output_model_time_average),
            output_model_freq_average: self
                .output_model_freq_average
                .or(other.output_model_freq_average),
            output_autos: self.output_autos || other.output_autos,
            filter_points: self.filter_points || other.filter_points,
            filter_gaussians: self.filter_gaussians || other.filter_gaussians,
            filter_shapelets: self.filter_shapelets || other.filter_shapelets,
        }
    }
}
