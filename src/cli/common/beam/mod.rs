// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#[cfg(test)]
mod tests;

use std::{path::PathBuf, str::FromStr};

use clap::Parser;
use log::{debug, trace};
use ndarray::prelude::*;
use serde::{Deserialize, Serialize};

use super::{InfoPrinter, Warn};
use crate::{
    beam::{
        AnalyticBeam, Beam, BeamError, BeamType, Delays, FEEBeam, NoBeam,
        BEAM_TYPES_COMMA_SEPARATED,
    },
    io::read::VisInputType,
};

lazy_static::lazy_static! {
    static ref BEAM_TYPE_HELP: String =
        format!("The beam model to use. Supported models: {}. Default: {}", *BEAM_TYPES_COMMA_SEPARATED, BeamType::default());

    static ref NO_BEAM_HELP: String =
        format!("Don't apply a beam response when generating a sky model. The default is to use the {} beam.", BeamType::default());

    static ref BEAM_FILE_HELP: String =
        format!("The path to the HDF5 MWA FEE beam file. Only useful if the beam type is 'fee'. If not specified, this must be provided by the MWA_BEAM_FILE environment variable.");
}

#[derive(Parser, Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct BeamArgs {
    #[clap(short, long, help_heading = "BEAM", help = BEAM_TYPE_HELP.as_str())]
    pub(crate) beam_type: Option<String>,

    #[clap(long, conflicts_with("beam-type"), help_heading = "BEAM", help = NO_BEAM_HELP.as_str())]
    #[serde(default)]
    pub(crate) no_beam: bool,

    /// If specified, use these dipole delays for the MWA pointing. e.g. 0 1 2 3
    /// 0 1 2 3 0 1 2 3 0 1 2 3
    #[clap(long, multiple_values(true), help_heading = "BEAM")]
    pub(crate) delays: Option<Vec<u32>>,

    /// Pretend that all MWA dipoles are alive and well, ignoring whatever is in
    /// the metafits file.
    #[clap(long, help_heading = "BEAM")]
    #[serde(default)]
    pub(crate) unity_dipole_gains: bool,

    /// The path to the HDF5 MWA FEE beam file. If not specified, this must be
    /// provided by the MWA_BEAM_FILE environment variable.
    #[clap(long, help_heading = "BEAM", help = BEAM_FILE_HELP.as_str())]
    pub(crate) beam_file: Option<PathBuf>,

    /// Treat this tile as the CRAM tile. A tile index or tile name may be used.
    /// If this isn't specified, the default is to use the tile named "CRAM".
    #[clap(long, help_heading = "BEAM (CRAM)")]
    pub(crate) cram_tile: Option<String>,

    /// Use these values as the dipole gains for the CRAM tile (e.g. 1 for a
    /// normal dipole, 0 for a dead dipole). 64 values must be given. The
    /// default is to assume all dipoles are alive and have gains of 1, and
    /// there is currently no other way to supply CRAM dead dipole information.
    #[clap(long, multiple_values(true), help_heading = "BEAM (CRAM)")]
    pub(crate) cram_dipole_gains: Option<Vec<f64>>,

    /// If the CRAM tile is present, elect to ignore it.
    #[clap(long, conflicts_with("cram-tile"), help_heading = "BEAM (CRAM)")]
    #[serde(default)]
    pub(crate) ignore_cram: bool,
}

impl BeamArgs {
    pub(crate) fn merge(self, other: Self) -> Self {
        Self {
            beam_type: self.beam_type.or(other.beam_type),
            no_beam: self.no_beam || other.no_beam,
            delays: self.delays.or(other.delays),
            unity_dipole_gains: self.unity_dipole_gains || other.unity_dipole_gains,
            beam_file: self.beam_file.or(other.beam_file),
            cram_tile: self.cram_tile.or(other.cram_tile),
            cram_dipole_gains: self.cram_dipole_gains.or(other.cram_dipole_gains),
            ignore_cram: self.ignore_cram || other.ignore_cram,
        }
    }

    pub(crate) fn parse(
        self,
        total_num_tiles: usize,
        data_dipole_delays: Option<Delays>,
        dipole_gains: Option<Array2<f64>>,
        input_data_type: Option<VisInputType>,
        tile_names: Option<&[String]>,
    ) -> Result<Box<dyn Beam>, BeamError> {
        let Self {
            beam_type,
            no_beam,
            delays: user_dipole_delays,
            unity_dipole_gains,
            beam_file,
            cram_tile,
            cram_dipole_gains,
            ignore_cram,
        } = self;

        let mut printer = InfoPrinter::new("Beam info".into());
        debug!("Beam file: {beam_file:?}");

        let beam_type = match (
            no_beam,
            beam_type.as_deref(),
            beam_type
                .as_deref()
                .and_then(|b| BeamType::from_str(b).ok()),
        ) {
            (true, _, _) => BeamType::None,
            (false, None, _) => BeamType::default(),
            (false, Some(_), Some(b)) => b,
            (false, Some(s), None) => return Err(BeamError::Unrecognised(s.to_string())),
        };

        let i_cram_tile = match (tile_names, cram_tile) {
            // Attempt to automatically detect the CRAM tile (tile name "CRAM").
            // Only do this logic if the user didn't supply a CRAM tile name.
            (Some(tile_names), None) => {
                let mut i_cram_tile = None;
                for (i_tile, tile_name) in tile_names.iter().enumerate() {
                    if tile_name.as_str() == "CRAM" {
                        i_cram_tile = Some(i_tile);
                        break;
                    }
                }
                // Handle the ignore option.
                if ignore_cram {
                    None
                } else {
                    i_cram_tile
                }
            }

            (_, Some(tile_string)) => {
                // We need to work out if this is a number or a tile name.
                match (tile_string.trim().parse().ok(), tile_names) {
                    (Some(i), _) => {
                        if i >= total_num_tiles {
                            return Err(BeamError::BadCramTileIndex {
                                got: i,
                                max: total_num_tiles - 1,
                            });
                        }
                        Some(i)
                    }

                    (None, Some(tile_names)) => {
                        // Now we need to match the given tile name against all
                        // of them.
                        match tile_names
                            .iter()
                            .enumerate()
                            .find(|(_, name)| name.to_lowercase() == tile_string.to_lowercase())
                        {
                            // If there are no matches, complain that the user input
                            // is no good.
                            None => return Err(BeamError::BadTileNameForCram(tile_string.clone())),
                            Some((i, _)) => Some(i),
                        }
                    }

                    (None, None) => {
                        // This situation only arises if a user specified a tile
                        // name as the CRAM tile, but we have no tile names.
                        // There's no way to continue.
                        return Err(BeamError::NoTileNamesForCram(tile_string.clone()));
                    }
                }
            }

            // We have no tile names, and the user hasn't specified a CRAM tile,
            // so we have no idea if the CRAM tile is there. Proceed naively.
            (None, None) => {
                "No tile names are available; unsure if a CRAM tile is present".warn();
                None
            }
        };
        // Ensure there are 64 CRAM dipole gains.
        let cram_dipole_gains: Option<Box<[f64; 64]>> = cram_dipole_gains
            .map(|g| g.try_into().map_err(|_| BeamError::Not64CramDipoleGains))
            .transpose()?;
        let cram_tile = i_cram_tile.map(|i| (i, cram_dipole_gains.unwrap_or([1.0; 64].into())));

        let beam: Box<dyn Beam> = match beam_type {
            BeamType::None => {
                debug!("Setting up a \"NoBeam\" object");
                printer.push_line("Not using any beam responses".into());
                Box::new(NoBeam {
                    num_tiles: total_num_tiles,
                })
            }

            BeamType::FEE => {
                debug!("Setting up a FEE beam object");
                printer.push_line("Type: FEE".into());

                let mut dipole_delays = match user_dipole_delays {
                    Some(d) => Some(Delays::parse(d)?),
                    None => data_dipole_delays,
                }
                .ok_or(BeamError::NoDelays("FEE"))?;
                trace!("Attempting to use delays:");
                match &dipole_delays {
                    Delays::Full(d) => {
                        let mut last_row = None;
                        for (i, row) in d.outer_iter().enumerate() {
                            if let Some(last_row) = last_row {
                                if row == last_row {
                                    continue;
                                }
                            }
                            trace!("{i:03} {row}");
                            last_row = Some(row);
                        }
                    }
                    Delays::Partial(d) => trace!("{d:?}"),
                }

                // Check that the delays are sensible.
                match &dipole_delays {
                    Delays::Partial(v) => {
                        if v.len() != 16 || v.iter().any(|&v| v > 32) {
                            return Err(BeamError::BadDelays);
                        }
                    }

                    Delays::Full(a) => {
                        if a.len_of(Axis(1)) != 16 || a.iter().any(|&v| v > 32) {
                            return Err(BeamError::BadDelays);
                        }
                        if a.len_of(Axis(0)) != total_num_tiles {
                            return Err(BeamError::InconsistentDelays {
                                num_rows: a.len_of(Axis(0)),
                                num_tiles: total_num_tiles,
                            });
                        }
                    }
                }

                let dipole_gains = if unity_dipole_gains {
                    printer.push_line("Assuming all dipoles are \"alive\"".into());
                    None
                } else {
                    // If we don't have dipole gains from the input data, then
                    // we issue a warning that we must assume no dead dipoles.
                    if dipole_gains.is_none() {
                        match input_data_type {
                            Some(VisInputType::MeasurementSet) => [
                                "Measurement sets cannot supply dead dipole information.".into(),
                                "Without a metafits file, we must assume all dipoles are alive.".into(),
                                "This will make beam Jones matrices inaccurate in sky-model generation."
                                    .into(),
                            ]
                            .warn(),
                            Some(VisInputType::Uvfits) => [
                                "uvfits files cannot supply dead dipole information.".into(),
                                "Without a metafits file, we must assume all dipoles are alive.".into(),
                                "This will make beam Jones matrices inaccurate in sky-model generation."
                                    .into(),
                            ]
                            .warn(),
                            Some(VisInputType::Raw) => {
                                unreachable!("Raw data inputs always specify dipole gains")
                            }
                            None => (),
                        }
                    }
                    dipole_gains
                };
                if let Some(dipole_gains) = dipole_gains.as_ref() {
                    trace!("Attempting to use dipole gains:");
                    let mut last_row = None;
                    for (i, row) in dipole_gains.outer_iter().enumerate() {
                        if let Some(last_row) = last_row {
                            if row == last_row {
                                continue;
                            }
                        }
                        trace!("{i:03} {row}");
                        last_row = Some(row);
                    }

                    // Currently, the only way to have dipole gains other than
                    // zero or one is by using Aman's "DipAmps" metafits column.
                    if dipole_gains.iter().any(|&g| g != 0.0 && g != 1.0) {
                        printer.push_line(
                            "Using Aman's 'DipAmps' dipole gains from the metafits".into(),
                        );
                    } else {
                        let num_tiles_with_dead_dipoles = dipole_gains
                            .outer_iter()
                            .filter(|tile_dipole_gains| {
                                tile_dipole_gains.iter().any(|g| g.abs() < f64::EPSILON)
                            })
                            .count();
                        printer.push_line(
                            format!(
                                "Using dead dipole information ({num_tiles_with_dead_dipoles} tiles affected)"
                            )
                            .into(),
                        );
                    }
                } else {
                    // If we don't have dipole gains, we must assume all dipoles
                    // are "alive". But, if any dipole delays are 32, then the
                    // beam code will still ignore those dipoles. So use ideal
                    // dipole delays for all tiles.
                    dipole_delays.set_to_ideal_delays();
                    let ideal_delays = dipole_delays.get_ideal_delays();

                    // Warn the user if they wanted unity dipole gains but the
                    // ideal dipole delays contain 32.
                    if unity_dipole_gains && ideal_delays.contains(&32) {
                        "Some ideal dipole delays are 32; these dipoles will not have unity gains"
                            .warn()
                    }
                }

                let beam = if let Some(bf) = beam_file {
                    // Set up the FEE beam struct from the specified beam file.
                    FEEBeam::new(&bf, total_num_tiles, dipole_delays, dipole_gains, cram_tile)?
                } else {
                    // Set up the FEE beam struct from the MWA_BEAM_FILE environment
                    // variable.
                    FEEBeam::new_from_env(total_num_tiles, dipole_delays, dipole_gains, cram_tile)?
                };
                Box::new(beam)
            }

            BeamType::AnalyticMwaPb | BeamType::AnalyticRts => {
                match beam_type {
                    BeamType::AnalyticMwaPb => {
                        debug!("Setting up an mwa_pb-flavoured analytic beam object");
                        printer.push_line("Type: Analytic (mwa_pb)".into());
                    }
                    BeamType::AnalyticRts => {
                        debug!("Setting up an RTS-flavoured analytic beam object");
                        printer.push_line("Type: Analytic (RTS)".into());
                    }
                    BeamType::FEE => unreachable!(),
                    BeamType::None => unreachable!(),
                }

                let mut dipole_delays = match user_dipole_delays {
                    Some(d) => Some(Delays::parse(d)?),
                    None => data_dipole_delays,
                }
                .ok_or(BeamError::NoDelays("Analytic"))?;
                trace!("Attempting to use delays:");
                match &dipole_delays {
                    Delays::Full(d) => {
                        for row in d.outer_iter() {
                            trace!("{row}");
                        }
                    }
                    Delays::Partial(d) => trace!("{d:?}"),
                }
                let dipole_gains = if unity_dipole_gains {
                    printer.push_line("Assuming all dipoles are \"alive\"".into());
                    None
                } else {
                    // If we don't have dipole gains from the input data, then
                    // we issue a warning that we must assume no dead dipoles.
                    if dipole_gains.is_none() {
                        match input_data_type {
                            Some(VisInputType::MeasurementSet) => [
                                "Measurement sets cannot supply dead dipole information.".into(),
                                "Without a metafits file, we must assume all dipoles are alive.".into(),
                                "This will make beam Jones matrices inaccurate in sky-model generation."
                                    .into(),
                            ]
                            .warn(),
                            Some(VisInputType::Uvfits) => [
                                "uvfits files cannot supply dead dipole information.".into(),
                                "Without a metafits file, we must assume all dipoles are alive.".into(),
                                "This will make beam Jones matrices inaccurate in sky-model generation."
                                    .into(),
                            ]
                            .warn(),
                            Some(VisInputType::Raw) => {
                                unreachable!("Raw data inputs always specify dipole gains")
                            }
                            None => (),
                        }
                    }
                    dipole_gains
                };
                if let Some(dipole_gains) = dipole_gains.as_ref() {
                    trace!("Attempting to use dipole gains:");
                    for row in dipole_gains.outer_iter() {
                        trace!("{row}");
                    }

                    // Currently, the only way to have dipole gains other than
                    // zero or one is by using Aman's "DipAmps" metafits column.
                    if dipole_gains.iter().any(|&g| g != 0.0 && g != 1.0) {
                        printer.push_line(
                            "Using Aman's 'DipAmps' dipole gains from the metafits".into(),
                        );
                    } else {
                        let num_tiles_with_dead_dipoles = dipole_gains
                            .outer_iter()
                            .filter(|tile_dipole_gains| {
                                tile_dipole_gains.iter().any(|g| g.abs() < f64::EPSILON)
                            })
                            .count();
                        printer.push_line(
                            format!(
                                "Using dead dipole information ({num_tiles_with_dead_dipoles} tiles affected)"
                            )
                            .into(),
                        );
                    }
                } else {
                    // If we don't have dipole gains, we must assume all dipoles
                    // are "alive". But, if any dipole delays are 32, then the
                    // beam code will still ignore those dipoles. So use ideal
                    // dipole delays for all tiles.
                    dipole_delays.set_to_ideal_delays();
                    let ideal_delays = dipole_delays.get_ideal_delays();

                    // Warn the user if they wanted unity dipole gains but the
                    // ideal dipole delays contain 32.
                    if unity_dipole_gains && ideal_delays.iter().any(|&v| v == 32) {
                        "Some ideal dipole delays are 32; these dipoles will not have unity gains"
                            .warn()
                    }
                }

                let beam = match beam_type {
                    BeamType::AnalyticMwaPb => AnalyticBeam::new_mwa_pb(
                        total_num_tiles,
                        dipole_delays,
                        dipole_gains,
                        cram_tile,
                    )?,
                    BeamType::AnalyticRts => AnalyticBeam::new_rts(
                        total_num_tiles,
                        dipole_delays,
                        dipole_gains,
                        cram_tile,
                    )?,
                    _ => unreachable!("only analytic beams should be here"),
                };
                Box::new(beam)
            }
        };

        if let Some(d) = beam.get_ideal_dipole_delays() {
            printer.push_block(vec![
                format!(
                    "Ideal dipole delays: [{:>2} {:>2} {:>2} {:>2}",
                    d[0], d[1], d[2], d[3]
                )
                .into(),
                format!(
                    "                      {:>2} {:>2} {:>2} {:>2}",
                    d[4], d[5], d[6], d[7]
                )
                .into(),
                format!(
                    "                      {:>2} {:>2} {:>2} {:>2}",
                    d[8], d[9], d[10], d[11]
                )
                .into(),
                format!(
                    "                      {:>2} {:>2} {:>2} {:>2}]",
                    d[12], d[13], d[14], d[15]
                )
                .into(),
            ]);
        }

        if let Some(f) = beam.get_beam_file() {
            printer.push_line(format!("File: {}", f.display()).into());
        }

        match (i_cram_tile, beam.get_beam_type()) {
            (Some(_), BeamType::None) => {
                "Not simulating the CRAM tile as we're not using any beam".warn()
            }
            (Some(i), _) => {
                if let Some(tile_name) = tile_names {
                    let cram_tile_name = tile_name[i].as_str();
                    printer.push_line(
                        format!("Using '{cram_tile_name}' (index {i}) as the CRAM tile").into(),
                    );
                } else {
                    printer.push_line(format!("Using tile index {i} as the CRAM tile").into());
                }
            }
            (None, _) => (),
        }

        printer.display();
        Ok(beam)
    }
}
