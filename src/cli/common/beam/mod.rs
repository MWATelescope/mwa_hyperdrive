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
    beam::{Beam, BeamError, BeamType, Delays, FEEBeam, NoBeam, BEAM_TYPES_COMMA_SEPARATED},
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
}

impl BeamArgs {
    pub(crate) fn merge(self, other: Self) -> Self {
        Self {
            beam_type: self.beam_type.or(other.beam_type),
            no_beam: self.no_beam || other.no_beam,
            delays: self.delays.or(other.delays),
            unity_dipole_gains: self.unity_dipole_gains || other.unity_dipole_gains,
            beam_file: self.beam_file.or(other.beam_file),
        }
    }

    pub(crate) fn parse(
        self,
        total_num_tiles: usize,
        data_dipole_delays: Option<Delays>,
        dipole_gains: Option<Array2<f64>>,
        input_data_type: Option<VisInputType>,
    ) -> Result<Box<dyn Beam>, BeamError> {
        let Self {
            beam_type,
            no_beam,
            delays: user_dipole_delays,
            unity_dipole_gains,
            beam_file,
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
                        for row in d.outer_iter() {
                            trace!("{row}");
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

                let beam = if let Some(bf) = beam_file {
                    // Set up the FEE beam struct from the specified beam file.
                    FEEBeam::new(&bf, total_num_tiles, dipole_delays, dipole_gains)?
                } else {
                    // Set up the FEE beam struct from the MWA_BEAM_FILE environment
                    // variable.
                    FEEBeam::new_from_env(total_num_tiles, dipole_delays, dipole_gains)?
                };
                Box::new(beam)
            }

            BeamType::SkaGaussian => {
                printer.push_line("Type: SKA Gaussian".into());
                Box::new(crate::beam::SkaGaussianBeam)
            }

            BeamType::SkaAiry => {
                printer.push_line("Type: SKA Airy".into());
                Box::new(crate::beam::SkaAiryBeam)
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

        printer.display();
        Ok(beam)
    }
}
