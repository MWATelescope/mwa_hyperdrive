// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

mod filenames;
#[cfg(test)]
mod tests;

use filenames::{InputDataTypes, GPUBOX_REGEX, MWAX_REGEX};

use std::{collections::HashSet, num::NonZeroUsize, path::PathBuf};

use clap::Parser;
use console::style;
use hifitime::Duration;
use itertools::Itertools;
use log::{
    debug, info, log_enabled,
    Level::{Debug, Info},
};
use marlu::{
    constants::{FREQ_WEIGHT_FACTOR, TIME_WEIGHT_FACTOR},
    precession::precess_time,
    LatLngHeight,
};
use ndarray::Axis;
use serde::{Deserialize, Serialize};
use vec1::Vec1;

use super::{InfoPrinter, ARRAY_POSITION_HELP};
use crate::{
    averaging::{
        channels_to_chanblocks, parse_freq_average_factor, parse_time_average_factor,
        timesteps_to_timeblocks, AverageFactorError,
    },
    cli::Warn,
    constants::DEFAULT_MS_DATA_COL_NAME,
    io::read::{
        pfb_gains::{PfbFlavour, DEFAULT_PFB_FLAVOUR, PFB_FLAVOURS},
        MsReader, RawDataCorrections, RawDataReader, UvfitsReader, VisInputType, VisRead,
    },
    math::TileBaselineFlags,
    params::InputVisParams,
    CalibrationSolutions,
};

lazy_static::lazy_static! {
    pub(super) static ref PFB_FLAVOUR_HELP: String =
        format!("The 'flavour' of poly-phase filter bank corrections applied to raw MWA data. The default is '{}'. Valid flavours are: {}", DEFAULT_PFB_FLAVOUR, *PFB_FLAVOURS);

    pub(super) static ref MS_DATA_COL_NAME_HELP: String =
        format!("If reading from a measurement set, this specifies the column to use in the main table containing visibilities. Default: {DEFAULT_MS_DATA_COL_NAME}");

    static ref SUPPORTED_INPUT_FILE_TYPES: String = format!(r#"
    metafits:         .metafits, _metafits.fits
    measurement sets: .ms
    uvfits files:     .uvfits
    gpubox files (regex): {GPUBOX_REGEX}
    MWAX files (regex):   {MWAX_REGEX}"#);
}

#[derive(Parser, Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct InputVisArgs {
    /// Paths to input data files. These can include a metafits file, a
    /// calibration solutions file, gpubox files, mwaf files, a measurement set,
    /// and/or a uvfits file.
    #[clap(
        short = 'd',
        long = "data",
        multiple_values(true),
        help_heading = "INPUT DATA"
    )]
    pub(crate) files: Option<Vec<String>>,

    /// The timesteps to use from the input data. Any input will be ascendingly
    /// sorted. No duplicates are allowed. The default is to use all unflagged
    /// timesteps. e.g. The following skips the first two timesteps and use the
    /// following three: --timesteps 2 3 4, --timesteps {2..4} (bash shell
    /// syntax)
    #[clap(long, multiple_values(true), help_heading = "INPUT DATA")]
    pub(crate) timesteps: Option<Vec<usize>>,

    /// Use all timesteps in the data, including flagged ones. The default is to
    /// use all unflagged timesteps.
    #[clap(long, conflicts_with("timesteps"), help_heading = "INPUT DATA")]
    #[serde(default)]
    pub(crate) use_all_timesteps: bool,

    #[clap(
        long, help = ARRAY_POSITION_HELP.as_str(), help_heading = "INPUT DATA",
        number_of_values = 3,
        allow_hyphen_values = true,
        value_names = &["LONG_DEG", "LAT_DEG", "HEIGHT_M"]
    )]
    pub(crate) array_position: Option<Vec<f64>>,

    /// Don't read autocorrelations from the input data.
    #[clap(long, help_heading = "INPUT DATA")]
    #[serde(default)]
    pub(crate) no_autos: bool,

    /// Use this value as the DUT1 [seconds].
    #[clap(long, help_heading = "INPUT DATA")]
    #[serde(default)]
    pub(crate) dut1: Option<f64>,

    /// Ignore the weights accompanying the visibilities. Internally, this will
    /// set all weights to 1, meaning all visibilities are equal, including
    /// those that would be otherwise flagged.
    #[clap(long, help_heading = "INPUT DATA")]
    #[serde(default)]
    pub(crate) ignore_weights: bool,

    /// Use a DUT1 value of 0 seconds rather than what is in the input data.
    #[clap(long, conflicts_with("dut1"), help_heading = "INPUT DATA")]
    #[serde(default)]
    pub(crate) ignore_dut1: bool,

    #[clap(long, help = MS_DATA_COL_NAME_HELP.as_str(), help_heading = "INPUT DATA (MS)")]
    pub(crate) ms_data_column_name: Option<String>,

    #[clap(long, help = PFB_FLAVOUR_HELP.as_str(), help_heading = "INPUT DATA (RAW)")]
    pub(crate) pfb_flavour: Option<String>,

    /// When reading in raw MWA data, don't apply digital gains.
    #[clap(long, help_heading = "INPUT DATA (RAW)")]
    #[serde(default)]
    pub(crate) no_digital_gains: bool,

    /// When reading in raw MWA data, don't apply cable length corrections. Note
    /// that some data may have already had the correction applied before it was
    /// written.
    #[clap(long, help_heading = "INPUT DATA (RAW)")]
    #[serde(default)]
    pub(crate) no_cable_length_correction: bool,

    /// When reading in raw MWA data, don't apply geometric corrections. Note
    /// that some data may have already had the correction applied before it was
    /// written.
    #[clap(long, help_heading = "INPUT DATA (RAW)")]
    #[serde(default)]
    pub(crate) no_geometric_correction: bool,

    /// Additional tiles to be flagged. These values correspond to either the
    /// values in the "Antenna" column of HDU 2 in the metafits file (e.g. 0 3
    /// 127), or the "TileName" (e.g. Tile011).
    #[clap(long, multiple_values(true), help_heading = "INPUT DATA (FLAGGING)")]
    pub(crate) tile_flags: Option<Vec<String>>,

    /// If specified, pretend that all tiles are unflagged in the input data.
    #[clap(long, help_heading = "INPUT DATA (FLAGGING)")]
    #[serde(default)]
    pub(crate) ignore_input_data_tile_flags: bool,

    /// If specified, pretend all fine channels in the input data are unflagged.
    /// Note that this does not unset any negative weights; visibilities
    /// associated with negative weights are still considered flagged even if
    /// we're ignoring input data fine channel flags.
    #[clap(long, help_heading = "INPUT DATA (FLAGGING)")]
    #[serde(default)]
    pub(crate) ignore_input_data_fine_channel_flags: bool,

    /// The fine channels to be flagged in each coarse channel. e.g. 0 1 16 30
    /// 31 are typical for 40 kHz data. If this is not specified, it defaults
    /// to flagging 80 kHz for raw data (or as close to this as possible) at the
    /// edges, as well as the centre channel for non-MWAX data. Other visibility
    /// file formats do not use this by default.
    #[clap(long, multiple_values(true), help_heading = "INPUT DATA (FLAGGING)")]
    pub(crate) fine_chan_flags_per_coarse_chan: Option<Vec<u16>>,

    /// The fine channels to be flagged across the whole observation band. e.g.
    /// 0 767 are the first and last fine channels for 40 kHz data. These flags
    /// are applied *before* any averaging is performed.
    #[clap(long, multiple_values(true), help_heading = "INPUT DATA (FLAGGING)")]
    pub(crate) fine_chan_flags: Option<Vec<u16>>,

    /// The number of timesteps to average together while reading in data. The
    /// value must be a multiple of the input data's time resolution, except if
    /// this is 0, in which case all timesteps are averaged together. A target
    /// resolution (e.g. 8s) may be used instead, in which case the specified
    /// resolution must be a multiple of the input data's resolution. The
    /// default is no averaging, i.e. a value of 1. Examples: If the input data
    /// is in 0.5s resolution and this variable is 4, then we average 2s worth
    /// of data together before performing work on it. If the variable is
    /// instead 4s, then 8 timesteps are averaged together.
    #[clap(long, help_heading = "INPUT DATA (AVERAGING)")]
    pub(crate) time_average: Option<String>,

    /// The number of fine-frequency channels to average together while reading
    /// in data. The value must be a multiple of the input data's freq.
    /// resolution, except if this is 0, in which case all channels are averaged
    /// together. A target resolution (e.g. 80kHz) may be used instead, in which
    /// case the specified resolution must be a multiple of the input data's
    /// resolution. The default is no averaging, i.e. a value of 1. Examples: If
    /// the input data is in 20kHz resolution and this variable was 2, then we
    /// average 40kHz worth of data together before performing work with it. If
    /// the variable is instead 80kHz, then 4 channels are averaged together.
    #[clap(short, long, help_heading = "INPUT DATA (AVERAGING)")]
    pub(crate) freq_average: Option<String>,
}

impl InputVisArgs {
    pub(crate) fn merge(self, other: Self) -> Self {
        InputVisArgs {
            files: self.files.or(other.files),
            timesteps: self.timesteps.or(other.timesteps),
            use_all_timesteps: self.use_all_timesteps || other.use_all_timesteps,
            array_position: self.array_position.or(other.array_position),
            no_autos: self.no_autos || other.no_autos,
            dut1: self.dut1.or(other.dut1),
            ignore_weights: self.ignore_weights || other.ignore_weights,
            ignore_dut1: self.ignore_dut1 || other.ignore_dut1,
            ms_data_column_name: self.ms_data_column_name.or(other.ms_data_column_name),
            pfb_flavour: self.pfb_flavour.or(other.pfb_flavour),
            no_digital_gains: self.no_digital_gains || other.no_digital_gains,
            no_cable_length_correction: self.no_cable_length_correction
                || other.no_cable_length_correction,
            no_geometric_correction: self.no_geometric_correction || other.no_geometric_correction,
            tile_flags: self.tile_flags.or(other.tile_flags),
            ignore_input_data_tile_flags: self.ignore_input_data_tile_flags
                || other.ignore_input_data_tile_flags,
            ignore_input_data_fine_channel_flags: self.ignore_input_data_fine_channel_flags
                || other.ignore_input_data_fine_channel_flags,
            fine_chan_flags_per_coarse_chan: self
                .fine_chan_flags_per_coarse_chan
                .or(other.fine_chan_flags_per_coarse_chan),
            fine_chan_flags: self.fine_chan_flags.or(other.fine_chan_flags),
            time_average: self.time_average.or(other.time_average),
            freq_average: self.freq_average.or(other.freq_average),
        }
    }

    pub(crate) fn parse(self, operation_verb: &str) -> Result<InputVisParams, InputVisArgsError> {
        let InputVisArgs {
            files,
            timesteps,
            use_all_timesteps,
            array_position,
            no_autos,
            dut1,
            ignore_weights,
            ignore_dut1,
            ms_data_column_name,
            pfb_flavour,
            no_digital_gains,
            no_cable_length_correction,
            no_geometric_correction,
            tile_flags,
            ignore_input_data_tile_flags,
            ignore_input_data_fine_channel_flags,
            fine_chan_flags_per_coarse_chan,
            fine_chan_flags,
            time_average,
            freq_average,
        } = self;

        // If the user supplied the array position, unpack it here.
        let array_position = match array_position {
            Some(v) => {
                if v.len() != 3 {
                    return Err(InputVisArgsError::BadArrayPosition { pos: v });
                }
                Some(LatLngHeight {
                    longitude_rad: v[0].to_radians(),
                    latitude_rad: v[1].to_radians(),
                    height_metres: v[2],
                })
            }
            None => None,
        };

        // Handle input data. We expect one of three possibilities:
        // - gpubox files, a metafits file (and maybe mwaf files),
        // - a measurement set (and maybe a metafits file), or
        // - uvfits files.
        // If none or multiple of these possibilities are met, then we must fail.
        let InputDataTypes {
            metafits,
            gpuboxes,
            mwafs,
            ms,
            uvfits,
            solutions,
        } = match files {
            Some(strings) => InputDataTypes::parse(&strings)?,
            None => return Err(InputVisArgsError::NoInputData),
        };
        let mut data_printer = InfoPrinter::new(format!("{operation_verb} data").into());
        let mut input_files_block = vec![];
        let mut raw_data_corrections_block = vec![];
        let mut vis_reader: Box<dyn VisRead> = match (metafits, gpuboxes, mwafs, ms, uvfits) {
            // Valid input for reading raw data.
            (Some(meta), Some(gpuboxes), mwafs, None, None) => {
                // Ensure that there's only one metafits.
                let meta = if meta.len() > 1 {
                    return Err(InputVisArgsError::MultipleMetafits(meta));
                } else {
                    meta.to_vec().swap_remove(0)
                };

                debug!("gpubox files: {:?}", &gpuboxes);
                debug!("mwaf files: {:?}", &mwafs);

                let corrections = RawDataCorrections::new(
                    pfb_flavour.as_deref(),
                    !no_digital_gains,
                    !no_cable_length_correction,
                    !no_geometric_correction,
                )?;
                let raw_reader = RawDataReader::new(
                    &meta,
                    &gpuboxes,
                    mwafs.as_deref(),
                    corrections,
                    array_position,
                )?;
                let obs_context = raw_reader.get_obs_context();
                let obsid = obs_context
                    .obsid
                    .expect("Raw data inputs always have the obsid specified");

                data_printer.overwrite_title(format!("{operation_verb} obsid {obsid}").into());
                input_files_block.push(format!("from {} gpubox files", gpuboxes.len()).into());
                input_files_block.push(format!("with metafits {}", meta.display()).into());

                match raw_reader.get_flags() {
                    Some(flags) => {
                        let software_string = match flags.software_version.as_ref() {
                            Some(v) => format!("{} {}", flags.software, v),
                            None => flags.software.to_string(),
                        };
                        input_files_block.push(
                            format!(
                                "with {} mwaf files ({})",
                                flags.gpubox_nums.len(),
                                software_string,
                            )
                            .into(),
                        );
                        if let Some(s) = flags.aoflagger_version.as_deref() {
                            info!("  AOFlagger version: {s}");
                        }
                        if let Some(s) = flags.aoflagger_strategy.as_deref() {
                            info!("  AOFlagger strategy: {s}");
                        }
                    }
                    None => "No mwaf files supplied".warn(),
                }

                let raw_data_corrections = raw_reader
                    .get_raw_data_corrections()
                    .expect("raw reader always has data corrections");
                match raw_data_corrections.pfb_flavour {
                    PfbFlavour::None => {
                        raw_data_corrections_block.push("Not doing any PFB correction".into())
                    }
                    //PfbFlavour::Jake => raw_data_corrections_block
                    //    .push("Correcting PFB gains with 'Jake Jones' gains".into()),
                    //PfbFlavour::Cotter2014 => raw_data_corrections_block
                    //    .push("Correcting PFB gains with 'Cotter 2014' gains".into()),
                    PfbFlavour::Empirical => raw_data_corrections_block
                        .push("Correcting PFB gains with 'RTS empirical' gains".into()),
                    PfbFlavour::Levine => raw_data_corrections_block
                        .push("Correcting PFB gains with 'Alan Levine' gains".into()),
                }
                if raw_data_corrections.digital_gains {
                    raw_data_corrections_block.push("Correcting digital gains".into());
                } else {
                    raw_data_corrections_block.push("Not correcting digital gains".into());
                }
                if raw_data_corrections.cable_length {
                    raw_data_corrections_block.push("Correcting cable lengths".into());
                } else {
                    raw_data_corrections_block.push("Not correcting cable lengths".into());
                }
                if raw_data_corrections.geometric {
                    raw_data_corrections_block
                        .push("Correcting geometric delays (if necessary)".into());
                } else {
                    raw_data_corrections_block.push("Not correcting geometric delays".into());
                }

                Box::new(raw_reader)
            }

            // Valid input for reading a measurement set.
            (meta, None, None, Some(ms), None) => {
                // Only one MS is supported at the moment.
                let ms: PathBuf = if ms.len() > 1 {
                    return Err(InputVisArgsError::MultipleMeasurementSets(ms));
                } else {
                    ms.into_vec().swap_remove(0)
                };

                // Ensure that there's only one metafits.
                let meta: Option<PathBuf> = match meta {
                    None => None,
                    Some(meta) => {
                        if meta.len() > 1 {
                            return Err(InputVisArgsError::MultipleMetafits(meta));
                        } else {
                            Some(meta.into_vec().swap_remove(0))
                        }
                    }
                };

                let ms_string = ms.display().to_string();
                let ms_reader =
                    MsReader::new(ms, ms_data_column_name, meta.as_deref(), array_position)?;
                let obs_context = ms_reader.get_obs_context();

                if let Some(o) = obs_context.obsid {
                    data_printer.overwrite_title(format!("{operation_verb} obsid {o}").into());
                    input_files_block.push(format!("from measurement set {}", ms_string).into());
                } else {
                    data_printer.overwrite_title(
                        format!("{operation_verb} measurement set {}", ms_string).into(),
                    );
                };
                if let Some(meta) = meta.as_ref() {
                    input_files_block.push(format!("with metafits {}", meta.display()).into());
                }

                Box::new(ms_reader)
            }

            // Valid input for reading uvfits files.
            (meta, None, None, None, Some(uvfits)) => {
                // Only one uvfits is supported at the moment.
                let uvfits: PathBuf = if uvfits.len() > 1 {
                    return Err(InputVisArgsError::MultipleUvfits(uvfits));
                } else {
                    uvfits.into_vec().swap_remove(0)
                };

                // Ensure that there's only one metafits.
                let meta: Option<PathBuf> = match meta {
                    None => None,
                    Some(meta) => {
                        if meta.len() > 1 {
                            return Err(InputVisArgsError::MultipleMetafits(meta));
                        } else {
                            Some(meta.into_vec().swap_remove(0))
                        }
                    }
                };

                let uvfits_string = uvfits.display().to_string();
                let uvfits_reader = UvfitsReader::new(uvfits, meta.as_deref(), array_position)?;
                let obs_context = uvfits_reader.get_obs_context();

                if let Some(o) = obs_context.obsid {
                    data_printer.overwrite_title(format!("{operation_verb} obsid {o}").into());
                    input_files_block.push(format!("from uvfits {}", uvfits_string).into());
                } else {
                    data_printer.overwrite_title(
                        format!("{operation_verb} uvfits {}", uvfits_string).into(),
                    );
                };
                if let Some(meta) = meta {
                    input_files_block.push(format!("with metafits {}", meta.display()).into());
                }

                Box::new(uvfits_reader)
            }

            // The following matches are for invalid combinations of input
            // files. Make an error message for the user.
            (Some(_), _, None, None, None) => {
                let msg = "Received only a metafits file; a uvfits file, a measurement set or gpubox files are required.";
                return Err(InputVisArgsError::InvalidDataInput(msg));
            }
            (Some(_), _, Some(_), None, None) => {
                let msg =
                    "Received only a metafits file and mwaf files; gpubox files are required.";
                return Err(InputVisArgsError::InvalidDataInput(msg));
            }
            (None, Some(_), _, None, None) => {
                let msg = "Received gpuboxes without a metafits file; this is not supported.";
                return Err(InputVisArgsError::InvalidDataInput(msg));
            }
            (None, None, Some(_), None, None) => {
                let msg = "Received mwaf files without gpuboxes and a metafits file; this is not supported.";
                return Err(InputVisArgsError::InvalidDataInput(msg));
            }
            (_, Some(_), _, Some(_), None) => {
                let msg = "Received gpuboxes and measurement set files; this is not supported.";
                return Err(InputVisArgsError::InvalidDataInput(msg));
            }
            (_, Some(_), _, None, Some(_)) => {
                let msg = "Received gpuboxes and uvfits files; this is not supported.";
                return Err(InputVisArgsError::InvalidDataInput(msg));
            }
            (_, _, _, Some(_), Some(_)) => {
                let msg = "Received uvfits and measurement set files; this is not supported.";
                return Err(InputVisArgsError::InvalidDataInput(msg));
            }
            (_, _, Some(_), Some(_), _) => {
                let msg = "Received mwafs and measurement set files; this is not supported.";
                return Err(InputVisArgsError::InvalidDataInput(msg));
            }
            (_, _, Some(_), _, Some(_)) => {
                let msg = "Received mwafs and uvfits files; this is not supported.";
                return Err(InputVisArgsError::InvalidDataInput(msg));
            }
            (None, None, None, None, None) => return Err(InputVisArgsError::NoInputData),
        };

        let total_num_tiles = vis_reader.get_obs_context().get_total_num_tiles();

        // Read the calibration solutions, if they were supplied.
        let mut solutions_block = vec![];
        let solutions = match solutions {
            Some(s) => {
                let s = if s.len() > 1 {
                    return Err(InputVisArgsError::MultipleSolutions(s));
                } else {
                    s.into_vec().remove(0)
                };
                // The optional metafits file is only used for reading RTS
                // solutions, which we won't support here.
                let sols = CalibrationSolutions::read_solutions_from_ext_inner(&s, None)?;
                solutions_block
                    .push(format!("On-the-fly-calibrating with solutions {}", s.display()).into());

                debug!(
                    "Raw data corrections in the solutions: {:?}",
                    sols.raw_data_corrections
                );

                // We can't do anything if the number of tiles in the data is
                // different to that of the solutions.

                // TODO: Check that all unflagged input tiles are in the
                // solutions; it's OK if the tile counts mismatch.
                if total_num_tiles != sols.di_jones.len_of(Axis(1)) {
                    return Err(InputVisArgsError::TileCountMismatch {
                        data: total_num_tiles,
                        solutions: sols.di_jones.len_of(Axis(1)),
                    });
                }

                // Replace raw data corrections in the data args with what's in
                // the solutions.
                match sols.raw_data_corrections {
                    Some(c) => {
                        vis_reader.set_raw_data_corrections(c);
                    }

                    None => {
                        // Warn the user if we're applying solutions to raw data
                        // without knowing what was applied during calibration.
                        if matches!(vis_reader.get_input_data_type(), VisInputType::Raw) {
                            [
                                "The calibration solutions do not list raw data corrections."
                                    .into(),
                                "Defaults and any user inputs are being used.".into(),
                            ]
                            .warn();
                        }
                    }
                };

                Some(sols)
            }
            None => None,
        };
        data_printer.push_block(input_files_block);
        data_printer.push_block(solutions_block);
        data_printer.push_block(raw_data_corrections_block);
        data_printer.display();

        let obs_context = vis_reader.get_obs_context();

        let mut coord_printer = InfoPrinter::new("Coordinates".into());
        let mut block = vec![
            style("                   RA        Dec")
                .bold()
                .to_string()
                .into(),
            format!(
                "Phase centre:      {:>8.4}° {:>8.4}° (J2000)",
                obs_context.phase_centre.ra.to_degrees(),
                obs_context.phase_centre.dec.to_degrees()
            )
            .into(),
        ];
        if let Some(pointing_centre) = obs_context.pointing_centre {
            block.push(
                format!(
                    "Pointing centre:   {:>8.4}° {:>8.4}°",
                    pointing_centre.ra.to_degrees(),
                    pointing_centre.dec.to_degrees()
                )
                .into(),
            );
        }
        coord_printer.push_block(block);
        let mut block = vec![format!(
            "Array position:    {:>8.4}° {:>8.4}° {:.4}m",
            obs_context.array_position.longitude_rad.to_degrees(),
            obs_context.array_position.latitude_rad.to_degrees(),
            obs_context.array_position.height_metres
        )
        .into()];
        let supplied = obs_context.supplied_array_position;
        let used = obs_context.array_position;
        if (used.longitude_rad - supplied.longitude_rad).abs() > f64::EPSILON
            || (used.latitude_rad - supplied.latitude_rad).abs() > f64::EPSILON
            || (used.height_metres - supplied.height_metres).abs() > f64::EPSILON
        {
            block.push(
                format!(
                    "Supplied position: {:>8.4}° {:>8.4}° {:.4}m",
                    supplied.longitude_rad.to_degrees(),
                    supplied.latitude_rad.to_degrees(),
                    supplied.height_metres
                )
                .into(),
            );
        }
        block.push(
            style("                   Longitude Latitude  Height")
                .bold()
                .to_string()
                .into(),
        );
        coord_printer.push_block(block);
        coord_printer.display();

        // Assign the tile flags. The flags depend on what's available in the
        // data, whether the user wants to use input data tile flags, and any
        // additional flags the user wants.
        let flagged_tiles = {
            let mut flagged_tiles = HashSet::new();

            if !ignore_input_data_tile_flags {
                // Add tiles that have already been flagged by the input data.
                flagged_tiles.extend(obs_context.flagged_tiles.iter());
            }
            // Unavailable tiles must be regarded as flagged.
            flagged_tiles.extend(obs_context.unavailable_tiles.iter());

            if let Some(flag_strings) = tile_flags {
                // We need to convert the strings into antenna indices. The strings
                // are either indices themselves or antenna names.
                for flag_string in flag_strings {
                    // Try to parse a naked number.
                    let result =
                        match flag_string.trim().parse().ok() {
                            Some(i) => {
                                if i >= total_num_tiles {
                                    Err(InputVisArgsError::BadTileIndexForFlagging {
                                        got: i,
                                        max: total_num_tiles - 1,
                                    })
                                } else {
                                    flagged_tiles.insert(i);
                                    Ok(())
                                }
                            }
                            None => {
                                // Check if this is an antenna name.
                                match obs_context.tile_names.iter().enumerate().find(|(_, name)| {
                                    name.to_lowercase() == flag_string.to_lowercase()
                                }) {
                                    // If there are no matches, complain that the user input
                                    // is no good.
                                    None => Err(InputVisArgsError::BadTileNameForFlagging(
                                        flag_string.to_string(),
                                    )),
                                    Some((i, _)) => {
                                        flagged_tiles.insert(i);
                                        Ok(())
                                    }
                                }
                            }
                        };
                    if result.is_err() {
                        // If there's a problem, show all the tile names and their
                        // indices to help out the user.
                        obs_context.print_tile_statuses(Info);
                        // Propagate the error.
                        result?;
                    }
                }
            }

            flagged_tiles
        };
        let num_unflagged_tiles = total_num_tiles - flagged_tiles.len();
        if num_unflagged_tiles == 0 {
            obs_context.print_tile_statuses(Debug);
            return Err(InputVisArgsError::NoTiles);
        }
        let flagged_tile_names_and_indices = flagged_tiles
            .iter()
            .cloned()
            .sorted()
            .map(|i| (obs_context.tile_names[i].as_str(), i))
            .collect::<Vec<_>>();
        let tile_baseline_flags = TileBaselineFlags::new(total_num_tiles, flagged_tiles);

        let mut tiles_printer = InfoPrinter::new("Tile info".into());
        tiles_printer.push_block(vec![
            format!("{total_num_tiles} total").into(),
            format!("{num_unflagged_tiles} unflagged").into(),
        ]);
        if !flagged_tile_names_and_indices.is_empty() {
            let mut block = vec!["Flagged tiles:".into()];
            for f in flagged_tile_names_and_indices.chunks(5) {
                block.push(format!("{f:?}").into());
            }
            tiles_printer.push_block(block);
        }
        tiles_printer.display();

        if log_enabled!(Debug) {
            obs_context.print_tile_statuses(Debug);
        }

        let timesteps_to_use = {
            match (use_all_timesteps, timesteps) {
                (true, _) => obs_context.all_timesteps.clone(),
                (false, None) => Vec1::try_from_vec(obs_context.unflagged_timesteps.clone())
                    .map_err(|_| InputVisArgsError::NoTimesteps)?,
                (false, Some(mut ts)) => {
                    // Make sure there are no duplicates.
                    let timesteps_hashset: HashSet<&usize> = ts.iter().collect();
                    if timesteps_hashset.len() != ts.len() {
                        return Err(InputVisArgsError::DuplicateTimesteps);
                    }

                    // Ensure that all specified timesteps are actually available.
                    for t in &ts {
                        if !(0..obs_context.timestamps.len()).contains(t) {
                            return Err(InputVisArgsError::UnavailableTimestep {
                                got: *t,
                                last: obs_context.timestamps.len() - 1,
                            });
                        }
                    }

                    ts.sort_unstable();
                    Vec1::try_from_vec(ts).map_err(|_| InputVisArgsError::NoTimesteps)?
                }
            }
        };

        let timestep_span = NonZeroUsize::new(
            timesteps_to_use
                .last()
                .checked_sub(*timesteps_to_use.first())
                .expect("last timestep index is bigger than first")
                + 1,
        )
        .expect("is not 0");
        let time_average_factor = match parse_time_average_factor(
            obs_context.time_res,
            time_average.as_deref(),
            NonZeroUsize::new(1).unwrap(),
        ) {
            Ok(f) => {
                // Check that the factor is not too big.
                if f > timestep_span {
                    format!(
                        "Cannot average {} timesteps; only {} are being used. Capping.",
                        f, timestep_span
                    )
                    .warn();
                    timestep_span
                } else {
                    f
                }
            }
            // The factor was 0, average everything together.
            Err(AverageFactorError::Zero) => timestep_span,
            Err(AverageFactorError::NotInteger) => {
                return Err(InputVisArgsError::TimeFactorNotInteger)
            }
            Err(AverageFactorError::NotIntegerMultiple { out, inp }) => {
                return Err(InputVisArgsError::TimeResNotMultiple { out, inp })
            }
            Err(AverageFactorError::Parse(e)) => {
                return Err(InputVisArgsError::ParseTimeAverageFactor(e))
            }
        };

        let dut1 = match (ignore_dut1, dut1) {
            (true, _) => {
                debug!("Ignoring input data and user DUT1");
                Duration::default()
            }
            (false, Some(dut1)) => {
                debug!("Using user DUT1");
                Duration::from_seconds(dut1)
            }
            (false, None) => {
                if let Some(dut1) = obs_context.dut1 {
                    debug!("Using input data DUT1");
                    dut1
                } else {
                    debug!("Input data has no DUT1");
                    Duration::default()
                }
            }
        };

        let mut time_printer = InfoPrinter::new("Time info".into());
        let time_res = match (obs_context.time_res, time_average_factor.get()) {
            (_, 0) => unreachable!("cannot be 0"),
            (None, _) => {
                time_printer.push_line(
                    format!("Resolution is unknown, assuming {TIME_WEIGHT_FACTOR}").into(),
                );
                obs_context
                    .time_res
                    .unwrap_or(Duration::from_seconds(TIME_WEIGHT_FACTOR))
            }
            (Some(r), 1) => {
                time_printer.push_line(format!("Resolution: {r}").into());
                r
            }
            (Some(r), f) => {
                time_printer.push_block(vec![
                    format!("Resolution: {r}").into(),
                    format!("Averaging {f}x ({})", r * f as i64).into(),
                ]);
                r
            }
        };
        time_printer
            .push_line(format!("First obs timestamp: {}", obs_context.timestamps.first()).into());
        time_printer.push_block(vec![
            format!(
                "Available timesteps: {}",
                range_or_comma_separated(&obs_context.all_timesteps)
            )
            .into(),
            format!(
                "Unflagged timesteps: {}",
                range_or_comma_separated(&obs_context.unflagged_timesteps)
            )
            .into(),
        ]);
        let mut block = vec![format!(
            "Using timesteps:     {}",
            range_or_comma_separated(&timesteps_to_use)
        )
        .into()];
        match timesteps_to_use.as_slice() {
            [t] => block.push(
                format!(
                    "Only timestamp (GPS): {:.2}",
                    obs_context.timestamps[*t].to_gpst_seconds()
                )
                .into(),
            ),

            [f, .., l] => {
                block.push(
                    format!(
                        "First timestamp (GPS): {:.2}",
                        obs_context.timestamps[*f].to_gpst_seconds()
                    )
                    .into(),
                );
                block.push(
                    format!(
                        "Last timestamp  (GPS): {:.2}",
                        obs_context.timestamps[*l].to_gpst_seconds()
                    )
                    .into(),
                );
            }

            [] => unreachable!("cannot be empty"),
        }
        {
            let p = precess_time(
                obs_context.array_position.longitude_rad,
                obs_context.array_position.latitude_rad,
                obs_context.phase_centre,
                obs_context.timestamps[*timesteps_to_use.first()],
                dut1,
            );
            block.push(format!("First LMST: {:.6}° (J2000)", p.lmst_j2000.to_degrees()).into());
        }
        time_printer.push_block(block);
        time_printer.push_line(format!("DUT1: {:.10} s", dut1.to_seconds()).into());
        time_printer.display();

        let timeblocks = timesteps_to_timeblocks(
            &obs_context.timestamps,
            time_res,
            time_average_factor,
            Some(&timesteps_to_use),
        );

        // Set up frequency information. Determine all of the fine-channel flags.
        let mut flagged_fine_chans: HashSet<u16> = match fine_chan_flags {
            Some(flags) => {
                // Check that all channel flags are within the allowed range.
                for &f in &flags {
                    if usize::from(f) > obs_context.fine_chan_freqs.len() {
                        return Err(InputVisArgsError::FineChanFlagTooBig {
                            got: f,
                            max: obs_context.fine_chan_freqs.len() - 1,
                        });
                    }
                }
                flags.into_iter().collect()
            }
            None => HashSet::new(),
        };
        if !ignore_input_data_fine_channel_flags {
            flagged_fine_chans.extend(obs_context.flagged_fine_chans.iter());
        }
        // Assign the per-coarse-channel fine-channel flags.
        let fine_chan_flags_per_coarse_chan = {
            let mut out_flags = HashSet::new();
            // Handle user flags.
            if let Some(fine_chan_flags_per_coarse_chan) = fine_chan_flags_per_coarse_chan {
                out_flags.extend(fine_chan_flags_per_coarse_chan);
            }
            // Handle input data flags.
            if let (false, Some(flags)) = (
                ignore_input_data_fine_channel_flags,
                obs_context.flagged_fine_chans_per_coarse_chan.as_ref(),
            ) {
                out_flags.extend(flags.iter());
            }
            out_flags
        };
        // Take the per-coarse-channel flags and put them in the fine channel
        // flags.
        match (
            obs_context.mwa_coarse_chan_nums.as_ref(),
            obs_context.num_fine_chans_per_coarse_chan.map(|n| n.get()),
        ) {
            (Some(mwa_coarse_chan_nums), Some(num_fine_chans_per_coarse_chan)) => {
                for (i_cc, _) in (0..).zip(mwa_coarse_chan_nums.iter()) {
                    for &f in &fine_chan_flags_per_coarse_chan {
                        if f > num_fine_chans_per_coarse_chan {
                            return Err(InputVisArgsError::FineChanFlagPerCoarseChanTooBig {
                                got: f,
                                max: num_fine_chans_per_coarse_chan - 1,
                            });
                        }

                        flagged_fine_chans.insert(f + num_fine_chans_per_coarse_chan * i_cc);
                    }
                }
            }

            // We can't do anything without the number of fine channels per
            // coarse channel.
            (_, None) => {
                "Flags per coarse channel were specified, but no information on how many fine channels per coarse channel is available; flags are being ignored.".warn();
            }

            // If we don't have MWA coarse channel numbers but we do have
            // per-coarse-channel flags, warn the user.
            (None, _) => {
                if !fine_chan_flags_per_coarse_chan.is_empty() {
                    "Flags per coarse channel were specified, but no MWA coarse channel information is available; flags are being ignored.".warn();
                }
            }
        }
        let mut unflagged_fine_chan_freqs = vec![];
        for (i_chan, &freq) in (0..).zip(obs_context.fine_chan_freqs.iter()) {
            if !flagged_fine_chans.contains(&i_chan) {
                unflagged_fine_chan_freqs.push(freq as f64);
            }
        }

        let num_unflagged_fine_chan_freqs = if unflagged_fine_chan_freqs.is_empty() {
            return Err(InputVisArgsError::NoChannels);
        } else {
            NonZeroUsize::new(unflagged_fine_chan_freqs.len()).expect("cannot be empty here")
        };
        let freq_average_factor = match parse_freq_average_factor(
            obs_context.freq_res,
            freq_average.as_deref(),
            NonZeroUsize::new(1).unwrap(),
        ) {
            Ok(f) => {
                // Check that the factor is not too big.
                if f > num_unflagged_fine_chan_freqs {
                    format!(
                        "Cannot average {} channels; only {} are being used. Capping.",
                        f,
                        unflagged_fine_chan_freqs.len()
                    )
                    .warn();
                    num_unflagged_fine_chan_freqs
                } else {
                    f
                }
            }
            // The factor was 0, average everything together.
            Err(AverageFactorError::Zero) => num_unflagged_fine_chan_freqs,
            Err(AverageFactorError::NotInteger) => {
                return Err(InputVisArgsError::FreqFactorNotInteger)
            }
            Err(AverageFactorError::NotIntegerMultiple { out, inp }) => {
                return Err(InputVisArgsError::FreqResNotMultiple { out, inp })
            }
            Err(AverageFactorError::Parse(e)) => {
                return Err(InputVisArgsError::ParseFreqAverageFactor(e))
            }
        };

        let mut chan_printer = InfoPrinter::new("Channel info".into());
        let freq_res = match (obs_context.freq_res, freq_average_factor.get()) {
            (_, 0) => unreachable!("cannot be 0"),
            (None, _) => {
                chan_printer.push_line(
                    format!("Resolution is unknown, assuming {FREQ_WEIGHT_FACTOR}").into(),
                );
                FREQ_WEIGHT_FACTOR
            }
            (Some(r), 1) => {
                chan_printer.push_line(format!("Resolution: {:.2} kHz", r / 1e3).into());
                r
            }
            (Some(r), f) => {
                chan_printer.push_block(vec![
                    format!("Resolution: {:.2} kHz", r / 1e3).into(),
                    format!("Averaging {f}x ({:.2} kHz)", r / 1e3 * f as f64).into(),
                ]);
                r
            }
        };

        // Set up the chanblocks.
        let mut spws = channels_to_chanblocks(
            &obs_context.fine_chan_freqs,
            freq_res.round() as u64,
            freq_average_factor,
            &flagged_fine_chans,
        );
        // There must be at least one chanblock to do anything.
        let spw = match spws.as_slice() {
            // No spectral windows is the same as no chanblocks.
            [] => return Err(InputVisArgsError::NoChannels),
            [spw] => {
                // Check that the chanblocks aren't all flagged.
                if spw.chanblocks.is_empty() {
                    return Err(InputVisArgsError::NoChannels);
                }
                spws.swap_remove(0)
            }
            [..] => {
                // TODO: Allow picket fence.
                eprintln!("\"Picket fence\" data detected. hyperdrive does not support this right now -- exiting.");
                eprintln!("See for more info: https://MWATelescope.github.io/mwa_hyperdrive/defs/mwa/picket_fence.html");
                std::process::exit(1);
            }
        };

        chan_printer.push_block(vec![
            format!(
                "Total number of fine channels:     {}",
                obs_context.fine_chan_freqs.len()
            )
            .into(),
            format!(
                "Number of unflagged fine channels: {}",
                unflagged_fine_chan_freqs.len()
            )
            .into(),
        ]);
        let mut block = vec![];
        if let Some(n) = obs_context.num_fine_chans_per_coarse_chan {
            block.push(format!("Number of fine chans per coarse channel: {}", n.get()).into());
        }
        if !fine_chan_flags_per_coarse_chan.is_empty() {
            let mut sorted = fine_chan_flags_per_coarse_chan
                .into_iter()
                .collect::<Vec<_>>();
            sorted.sort_unstable();
            block.push(format!("Flags per coarse channel: {sorted:?}").into());
        }
        chan_printer.push_block(block);
        match obs_context.fine_chan_freqs.as_slice() {
            [f] => chan_printer
                .push_line(format!("Only fine-channel: {:.3} MHz", *f as f64 / 1e6).into()),

            [f, .., l] => chan_printer.push_block(vec![
                format!("First fine-channel:           {:.3} MHz", *f as f64 / 1e6).into(),
                format!("Last fine-channel:            {:.3} MHz", *l as f64 / 1e6).into(),
            ]),

            [] => unreachable!("cannot be empty"),
        };
        match unflagged_fine_chan_freqs.as_slice() {
            [f] => chan_printer
                .push_line(format!("Only unflagged fine-channel: {:.3} MHz", *f / 1e6).into()),

            [f, .., l] => chan_printer.push_block(vec![
                format!("First unflagged fine-channel: {:.3} MHz", *f / 1e6).into(),
                format!("Last unflagged fine-channel:  {:.3} MHz", *l / 1e6).into(),
            ]),

            [] => unreachable!("cannot be empty"),
        };
        chan_printer.display();

        Ok(InputVisParams {
            vis_reader,
            solutions,
            timeblocks,
            time_res: time_res * time_average_factor.get() as i64,
            spw,
            tile_baseline_flags,
            using_autos: !no_autos,
            ignore_weights,
            dut1,
        })
    }
}

#[derive(thiserror::Error, Debug)]
pub(crate) enum InputVisArgsError {
    #[error("Specified file does not exist: {0}")]
    DoesNotExist(String),

    #[error("Could not read specified file: {0}")]
    CouldNotRead(String),

    #[error("The specified file '{0}' is a \"PPDs metafits\" and is not supported. Please use a newer metafits file.")]
    PpdMetafitsUnsupported(String),

    #[error("The specified file '{0}' was not a recognised file type.\n\nSupported file formats:{}", *SUPPORTED_INPUT_FILE_TYPES)]
    NotRecognised(String),

    #[error("No input data was given!")]
    NoInputData,

    #[error("Multiple metafits files were specified: {0:?}\nThis is unsupported.")]
    MultipleMetafits(Vec1<PathBuf>),

    #[error("Multiple measurement sets were specified: {0:?}\nThis is currently unsupported.")]
    MultipleMeasurementSets(Vec1<PathBuf>),

    #[error("Multiple uvfits files were specified: {0:?}\nThis is currently unsupported.")]
    MultipleUvfits(Vec1<PathBuf>),

    #[error("Multiple calibration solutions files were specified: {0:?}\nThis is unsupported.")]
    MultipleSolutions(Vec1<PathBuf>),

    #[error("{0}\n\nSupported file formats:{}", *SUPPORTED_INPUT_FILE_TYPES)]
    InvalidDataInput(&'static str),

    #[error("Array position specified as {pos:?}, not [<Longitude>, <Latitude>, <Height>]")]
    BadArrayPosition { pos: Vec<f64> },

    #[error("The data either contains no timesteps or no timesteps are being used")]
    NoTimesteps,

    #[error("Duplicate timesteps were specified; this is invalid")]
    DuplicateTimesteps,

    #[error("Timestep {got} was specified but it isn't available; the last timestep is {last}")]
    UnavailableTimestep { got: usize, last: usize },

    #[error("The data either contains no tiles or all tiles are flagged")]
    NoTiles,

    #[error("Got a tile flag {got}, but the biggest possible antenna index is {max}")]
    BadTileIndexForFlagging { got: usize, max: usize },

    #[error("Bad tile flag value: '{0}' is neither an integer or an available antenna name. Run with extra verbosity to see all tile statuses.")]
    BadTileNameForFlagging(String),

    #[error("The data either contains no frequency channels or all channels are flagged")]
    NoChannels,

    #[error("Got a fine-channel flag {got}, but the biggest possible index is {max}")]
    FineChanFlagTooBig { got: u16, max: usize },

    #[error(
        "Got a fine-channel-per-coarse-channel flag {got}, but the biggest possible index is {max}"
    )]
    FineChanFlagPerCoarseChanTooBig { got: u16, max: u16 },

    #[error("The input data and the solutions have different numbers of tiles (data: {data}, solutions: {solutions}); cannot continue")]
    TileCountMismatch { data: usize, solutions: usize },

    #[error("Error when parsing input data time average factor: {0}")]
    ParseTimeAverageFactor(crate::unit_parsing::UnitParseError),

    #[error("Input data time average factor isn't an integer")]
    TimeFactorNotInteger,

    #[error("Input data time resolution isn't a multiple of input data's: {out} seconds vs {inp} seconds")]
    TimeResNotMultiple { out: f64, inp: f64 },

    #[error("Error when parsing input data freq. average factor: {0}")]
    ParseFreqAverageFactor(crate::unit_parsing::UnitParseError),

    #[error("Input data freq. average factor isn't an integer")]
    FreqFactorNotInteger,

    #[error("Input data freq. resolution isn't a multiple of input data's: {out} Hz vs {inp} Hz")]
    FreqResNotMultiple { out: f64, inp: f64 },

    #[error(transparent)]
    PfbParse(#[from] crate::io::read::pfb_gains::PfbParseError),

    #[error(transparent)]
    Raw(#[from] crate::io::read::RawReadError),

    #[error(transparent)]
    Ms(#[from] crate::io::read::MsReadError),

    #[error(transparent)]
    Uvfits(#[from] crate::io::read::UvfitsReadError),

    #[error(transparent)]
    Solutions(#[from] crate::solutions::SolutionsReadError),

    #[error(transparent)]
    Glob(#[from] crate::io::GlobError),

    #[error("IO error when attempting to read file '{0}': {1}")]
    IO(String, std::io::Error),
}

// It looks a bit neater to print out a collection of numbers as a range rather
// than individual indices if they're sequential. This function inspects a
// collection and returns a string to be printed.
fn range_or_comma_separated(collection: &[usize]) -> String {
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
        if collection.len() == 1 {
            format!("[{}]", collection[0])
        } else {
            format!(
                "[{:?})",
                (*collection.first().unwrap()..*collection.last().unwrap() + 1)
            )
        }
    } else {
        collection
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    }
}
