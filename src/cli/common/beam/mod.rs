// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#[cfg(test)]
mod tests;

use std::path::PathBuf;

use clap::Parser;
use log::debug;
use ndarray::Array2;
use serde::{Deserialize, Serialize};

use super::{InfoPrinter, Warn};
use crate::{
    beam::{create_fee_beam_object, create_no_beam_object, Beam, BeamError, Delays},
    io::read::VisInputType,
};

#[derive(Parser, Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct BeamArgs {
    /// The path to the HDF5 MWA FEE beam file. If not specified, this must be
    /// provided by the MWA_BEAM_FILE environment variable.
    #[clap(long, help_heading = "BEAM")]
    pub(crate) beam_file: Option<PathBuf>,

    /// Pretend that all MWA dipoles are alive and well, ignoring whatever is in
    /// the metafits file.
    #[clap(long, help_heading = "BEAM")]
    #[serde(default)]
    pub(crate) unity_dipole_gains: bool,

    /// If specified, use these dipole delays for the MWA pointing. e.g. 0 1 2 3
    /// 0 1 2 3 0 1 2 3 0 1 2 3
    #[clap(long, multiple_values(true), help_heading = "BEAM")]
    pub(crate) delays: Option<Vec<u32>>,

    /// Don't apply a beam response when generating a sky model. The default is
    /// to use the FEE beam.
    #[clap(long, help_heading = "BEAM")]
    #[serde(default)]
    pub(crate) no_beam: bool,
}

impl BeamArgs {
    pub(crate) fn merge(self, other: Self) -> Self {
        Self {
            beam_file: self.beam_file.or(other.beam_file),
            unity_dipole_gains: self.unity_dipole_gains || other.unity_dipole_gains,
            delays: self.delays.or(other.delays),
            no_beam: self.no_beam || other.no_beam,
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
            beam_file,
            unity_dipole_gains,
            delays: user_dipole_delays,
            no_beam,
        } = self;

        let mut printer = InfoPrinter::new("Beam info".into());
        debug!("Beam file: {beam_file:?}");

        let user_dipole_delays = match user_dipole_delays {
            // We have user-provided delays; check that they're are sensible,
            // regardless of whether we actually need them.
            Some(d) => Some(Delays::parse(d)?),
            None => None,
        };

        let mut num_tiles_with_dead_dipoles = None;
        let beam: Box<dyn Beam> = if no_beam {
            printer.push_line("Not using any beam responses".into());
            create_no_beam_object(total_num_tiles)
        } else {
            printer.push_line("Type: FEE".into());
            let mut dipole_delays = user_dipole_delays
                .or(data_dipole_delays)
                .ok_or(BeamError::NoDelays)?;
            let dipole_gains = if unity_dipole_gains {
                None
            } else {
                // If we don't have dipole gains from the input data, then
                // we issue a warning that we must assume no dead dipoles.
                if dipole_gains.is_none() {
                    match input_data_type {
                        Some(VisInputType::MeasurementSet) => {
                            [
                                "Measurement sets cannot supply dead dipole information.".into(),
                                "Without a metafits file, we must assume all dipoles are alive.".into(),
                                "This will make beam Jones matrices inaccurate in sky-model generation.".into()
                            ].warn()
                        }
                        Some(VisInputType::Uvfits) => {
                            [
                                "uvfits files cannot supply dead dipole information.".into(),
                                "Without a metafits file, we must assume all dipoles are alive.".into(),
                                "This will make beam Jones matrices inaccurate in sky-model generation.".into()
                            ].warn()
                        }
                        Some(VisInputType::Raw) => {
                            unreachable!("Raw data inputs always specify dipole gains")
                        }
                        None => (),
                    }
                }
                dipole_gains
            };
            let ideal_delays = dipole_delays.get_ideal_delays();
            if dipole_gains.is_none() {
                // If we don't have dipole gains, we must assume all dipoles are
                // "alive". But, if any dipole delays are 32, then the beam code
                // will still ignore those dipoles. So use ideal dipole delays
                // for all tiles.

                // Warn the user if they wanted unity dipole gains but the ideal
                // dipole delays contain 32.
                if unity_dipole_gains && ideal_delays.iter().any(|&v| v == 32) {
                    "Some ideal dipole delays are 32; these dipoles will not have unity gains"
                        .warn()
                }
                dipole_delays.set_to_ideal_delays();
            }

            {
                let d = ideal_delays;
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

            if let Some(dipole_gains) = dipole_gains.as_ref() {
                num_tiles_with_dead_dipoles = Some(
                    dipole_gains
                        .outer_iter()
                        .filter(|tile_dipole_gains| {
                            tile_dipole_gains.iter().any(|g| g.abs() < f64::EPSILON)
                        })
                        .count(),
                );
            }

            create_fee_beam_object(beam_file, total_num_tiles, dipole_delays, dipole_gains)?
        };
        if let Some(f) = beam.get_beam_file() {
            printer.push_line(format!("File: {}", f.display()).into());
        }
        if let Some(num_tiles_with_dead_dipoles) = num_tiles_with_dead_dipoles {
            printer.push_line(
                format!(
                    "Using dead dipole information ({num_tiles_with_dead_dipoles} tiles affected)"
                )
                .into(),
            );
        } else {
            printer.push_line("Assuming all dipoles are \"alive\"".into());
        }

        printer.display();
        Ok(beam)
    }
}
