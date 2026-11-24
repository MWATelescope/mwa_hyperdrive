// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#[cfg(test)]
mod tests;

use std::{
    path::{Path, PathBuf},
    str::FromStr,
};

use clap::Parser;
use log::{debug, trace};
use mwa_hyperbeam::analytic::AnalyticType;
use ndarray::prelude::*;
use serde::{Deserialize, Serialize};

use super::{InfoPrinter, Warn};
#[cfg(feature = "everybeam")]
use crate::beam::{EveryBeam, EveryBeamOptions};
use crate::{
    beam::{
        validate_delays, AnalyticBeam, Beam, BeamError, BeamType, Delays, FEEBeam, NoBeam,
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
        "The path to the HDF5 MWA FEE beam file. Only useful if the beam type is 'fee'. If not specified, this must be provided by the MWA_BEAM_FILE environment variable.".to_string();
}

/// Options for the EveryBeam beam. These are only available if hyperdrive was
/// compiled with the "everybeam" feature.
#[cfg(feature = "everybeam")]
#[derive(Parser, Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct EveryBeamArgs {
    /// The EveryBeam element response model, e.g. default, hamaker, lobes,
    /// oskar_dipole, oskar_dipole_cos, skala40_wave, skalow_feko. The default
    /// depends on the telescope; for SKA-Low/OSKAR, it is oskar_dipole_cos.
    #[clap(long, help_heading = "BEAM (EVERYBEAM)")]
    pub(crate) everybeam_element_model: Option<String>,

    /// The EveryBeam beam mode: full, array_factor or element. Default: full
    #[clap(long, help_heading = "BEAM (EVERYBEAM)")]
    pub(crate) everybeam_mode: Option<String>,

    /// The EveryBeam beam normalisation mode: none, amplitude, full,
    /// preapplied or preapplied_or_full. 'none' uses the unnormalised station
    /// response. 'amplitude' scales each station's response (at each time and
    /// frequency) to unit amplitude in the measurement set's FIELD
    /// REFERENCE_DIR, keeping each station's feed basis. 'full' multiplies the
    /// response by the inverse of the response in REFERENCE_DIR, which is only
    /// appropriate for data that have had a beam correction applied. Default:
    /// none
    #[clap(long, help_heading = "BEAM (EVERYBEAM)")]
    pub(crate) everybeam_normalisation: Option<String>,

    /// The path to element-response coefficients for EveryBeam (telescope
    /// dependent; e.g. the MWA FEE HDF5 file, or a directory of LOBES
    /// coefficients). If not specified, EveryBeam uses its data directory
    /// (which can be set with the EVERYBEAM_DATADIR environment variable).
    #[clap(long, help_heading = "BEAM (EVERYBEAM)")]
    pub(crate) everybeam_coeff_path: Option<PathBuf>,

    /// The measurement set FIELD to use for the EveryBeam beam pointing.
    /// Default: 0
    #[clap(long, help_heading = "BEAM (EVERYBEAM)")]
    pub(crate) everybeam_field_id: Option<usize>,

    /// The data column EveryBeam uses to check for a beam that has already been
    /// applied to the data (LOFAR only). Default: DATA
    #[clap(long, help_heading = "BEAM (EVERYBEAM)")]
    pub(crate) everybeam_data_column: Option<String>,

    /// Use the subband (reference) frequency for the station beamformer,
    /// rather than each channel's frequency (LOFAR only).
    #[clap(long, help_heading = "BEAM (EVERYBEAM)")]
    #[serde(default)]
    pub(crate) everybeam_subband_frequency: bool,

    /// Interpolate EveryBeam beam responses over frequency (MWA only).
    #[clap(long, help_heading = "BEAM (EVERYBEAM)")]
    #[serde(default)]
    pub(crate) everybeam_frequency_interpolation: bool,

    /// The spacing of the frequencies at which EveryBeam is evaluated when
    /// modelling, e.g. 1.28MHz or 500kHz (a number without a unit is in Hz).
    /// Responses are splined in frequency to the fine channels. 0 evaluates
    /// EveryBeam at every fine channel, which is exact but much slower.
    /// Default: 1.28MHz
    #[clap(long, help_heading = "BEAM (EVERYBEAM)")]
    pub(crate) everybeam_knot_spacing: Option<String>,
}

#[cfg(feature = "everybeam")]
impl EveryBeamArgs {
    fn merge(self, other: Self) -> Self {
        Self {
            everybeam_element_model: self
                .everybeam_element_model
                .or(other.everybeam_element_model),
            everybeam_mode: self.everybeam_mode.or(other.everybeam_mode),
            everybeam_normalisation: self
                .everybeam_normalisation
                .or(other.everybeam_normalisation),
            everybeam_coeff_path: self.everybeam_coeff_path.or(other.everybeam_coeff_path),
            everybeam_field_id: self.everybeam_field_id.or(other.everybeam_field_id),
            everybeam_data_column: self.everybeam_data_column.or(other.everybeam_data_column),
            everybeam_subband_frequency: self.everybeam_subband_frequency
                || other.everybeam_subband_frequency,
            everybeam_frequency_interpolation: self.everybeam_frequency_interpolation
                || other.everybeam_frequency_interpolation,
            everybeam_knot_spacing: self.everybeam_knot_spacing.or(other.everybeam_knot_spacing),
        }
    }

    fn into_options(self) -> Result<EveryBeamOptions, BeamError> {
        let Self {
            everybeam_element_model,
            everybeam_mode,
            everybeam_normalisation,
            everybeam_coeff_path,
            everybeam_field_id,
            everybeam_data_column,
            everybeam_subband_frequency,
            everybeam_frequency_interpolation,
            everybeam_knot_spacing,
        } = self;
        let knot_spacing_hz = everybeam_knot_spacing
            .map(|s| -> Result<f64, BeamError> {
                let (value, unit) = crate::unit_parsing::parse_freq(&s).map_err(|e| {
                    BeamError::EveryBeam(format!("Couldn't parse --everybeam-knot-spacing: {e}"))
                })?;
                let hz = match unit {
                    None | Some(crate::unit_parsing::FreqFormat::Hz) => value,
                    Some(crate::unit_parsing::FreqFormat::kHz) => value * 1e3,
                    Some(crate::unit_parsing::FreqFormat::MHz) => value * 1e6,
                };
                if hz < 0.0 || !hz.is_finite() {
                    return Err(BeamError::EveryBeam(format!(
                        "--everybeam-knot-spacing must not be negative (got {s})"
                    )));
                }
                Ok(hz)
            })
            .transpose()?;
        Ok(EveryBeamOptions {
            element_response_model: everybeam_element_model,
            beam_mode: everybeam_mode,
            beam_normalisation_mode: Some(
                everybeam_normalisation.unwrap_or_else(|| "none".to_string()),
            ),
            coeff_path: everybeam_coeff_path,
            data_column_name: everybeam_data_column,
            field_id: everybeam_field_id.unwrap_or(0),
            use_subband_frequency: everybeam_subband_frequency,
            frequency_interpolation: everybeam_frequency_interpolation,
            knot_spacing_hz,
        })
    }
}

#[derive(Parser, Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct BeamArgs {
    #[arg(short, long, help_heading = "BEAM", help = BEAM_TYPE_HELP.as_str())]
    pub(crate) beam_type: Option<String>,

    #[arg(long, conflicts_with("beam_type"), help_heading = "BEAM", help = NO_BEAM_HELP.as_str())]
    #[serde(default)]
    pub(crate) no_beam: bool,

    /// If specified, use these dipole delays for the MWA pointing. e.g. 0 1 2 3
    /// 0 1 2 3 0 1 2 3 0 1 2 3
    #[arg(long, num_args(1..), help_heading = "BEAM")]
    pub(crate) delays: Option<Vec<u32>>,

    /// Pretend that all MWA dipoles are alive and well, ignoring whatever is in
    /// the metafits file.
    #[arg(long, help_heading = "BEAM")]
    #[serde(default)]
    pub(crate) unity_dipole_gains: bool,

    /// The path to the HDF5 MWA FEE beam file. If not specified, this must be
    /// provided by the MWA_BEAM_FILE environment variable.
    #[arg(long, help_heading = "BEAM", help = BEAM_FILE_HELP.as_str())]
    pub(crate) beam_file: Option<PathBuf>,

    /// The measurement set describing the telescope for the EveryBeam beam.
    /// If not specified, the input data is used (which must be a measurement
    /// set).
    #[cfg(feature = "everybeam")]
    #[arg(long, help_heading = "BEAM")]
    pub(crate) beam_ms: Option<PathBuf>,

    #[cfg(feature = "everybeam")]
    #[clap(flatten)]
    #[serde(flatten)]
    pub(crate) everybeam: EveryBeamArgs,
}

impl BeamArgs {
    pub(crate) fn merge(self, other: Self) -> Self {
        Self {
            beam_type: self.beam_type.or(other.beam_type),
            no_beam: self.no_beam || other.no_beam,
            delays: self.delays.or(other.delays),
            unity_dipole_gains: self.unity_dipole_gains || other.unity_dipole_gains,
            beam_file: self.beam_file.or(other.beam_file),
            #[cfg(feature = "everybeam")]
            beam_ms: self.beam_ms.or(other.beam_ms),
            #[cfg(feature = "everybeam")]
            everybeam: self.everybeam.merge(other.everybeam),
        }
    }

    pub(crate) fn parse(
        self,
        total_num_tiles: usize,
        data_dipole_delays: Option<Delays>,
        dipole_gains: Option<Array2<f64>>,
        input_data_type: Option<VisInputType>,
        input_ms: Option<&Path>,
    ) -> Result<Box<dyn Beam>, BeamError> {
        let Self {
            beam_type,
            no_beam,
            delays: user_dipole_delays,
            unity_dipole_gains,
            beam_file,
            #[cfg(feature = "everybeam")]
            beam_ms,
            #[cfg(feature = "everybeam")]
            everybeam,
        } = self;
        // The measurement set describing the telescope (for the beams that
        // need one).
        #[cfg(feature = "everybeam")]
        let telescope_ms = beam_ms.as_deref().or(input_ms);
        #[cfg(not(feature = "everybeam"))]
        let _ = input_ms;

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

                let (dipole_delays, dipole_gains) = parse_delays_and_gains(
                    "FEE",
                    total_num_tiles,
                    user_dipole_delays,
                    data_dipole_delays,
                    dipole_gains,
                    unity_dipole_gains,
                    input_data_type,
                    &mut printer,
                )?;

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

            #[cfg(feature = "everybeam")]
            BeamType::EveryBeam => {
                debug!("Setting up an EveryBeam object");
                printer.push_line("Type: EveryBeam".into());
                let options = everybeam.into_options()?;
                let ms = telescope_ms.ok_or(BeamError::NeedsBeamMs)?;
                printer.push_line(format!("Telescope measurement set: {}", ms.display()).into());
                let describe = |name: &str, v: Option<&str>| -> String {
                    format!("{name}: {}", v.unwrap_or("default"))
                };
                printer.push_block(vec![
                    describe("Element model", options.element_response_model.as_deref()).into(),
                    describe("Beam mode", options.beam_mode.as_deref()).into(),
                    describe("Normalisation", options.beam_normalisation_mode.as_deref()).into(),
                ]);
                if let Some(p) = options.coeff_path.as_ref() {
                    printer.push_line(format!("Coefficients: {}", p.display()).into());
                }
                printer.push_line(
                    match options
                        .knot_spacing_hz
                        .unwrap_or(crate::beam::BEAM_KNOT_SPACING_HZ)
                    {
                        s if s > 0.0 => format!(
                            "Evaluated every {} kHz and splined to the fine channels",
                            s / 1e3
                        ),
                        _ => "Evaluated at every fine channel".to_string(),
                    }
                    .into(),
                );
                let beam = EveryBeam::new(ms, Some(total_num_tiles), options)?;
                printer.push_line(format!("Stations: {}", beam.get_num_tiles()).into());
                Box::new(beam)
            }

            BeamType::AnalyticMwaPb | BeamType::AnalyticRts => {
                let at = beam_type
                    .analytic_type()
                    .expect("only the analytic beam types reach this arm");
                let flavour = match at {
                    AnalyticType::MwaPb => "mwa_pb",
                    AnalyticType::Rts => "RTS",
                };
                debug!("Setting up an {flavour}-flavoured analytic beam object");
                printer.push_line(format!("Type: Analytic ({flavour})").into());

                // Only the FEE beam reads a beam file.
                if beam_file.is_some() {
                    "Ignoring the supplied beam file; it is only used by the FEE beam".warn()
                }

                let (dipole_delays, dipole_gains) = parse_delays_and_gains(
                    "Analytic",
                    total_num_tiles,
                    user_dipole_delays,
                    data_dipole_delays,
                    dipole_gains,
                    unity_dipole_gains,
                    input_data_type,
                    &mut printer,
                )?;

                Box::new(AnalyticBeam::new(
                    at,
                    total_num_tiles,
                    dipole_delays,
                    dipole_gains,
                )?)
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

        if matches!(beam.get_beam_type(), BeamType::FEE) {
            if let Some(f) = beam.get_beam_file() {
                printer.push_line(format!("File: {}", f.display()).into());
            }
        }

        printer.display();
        Ok(beam)
    }
}

/// Work out the dipole delays and dipole gains that a beam object should be
/// created with. All of the beam types that model the MWA tile beam need this;
/// only [`BeamType::None`] doesn't.
///
/// `beam_name` is used to describe the beam if no delays are available.
#[allow(clippy::too_many_arguments)]
fn parse_delays_and_gains(
    beam_name: &'static str,
    total_num_tiles: usize,
    user_dipole_delays: Option<Vec<u32>>,
    data_dipole_delays: Option<Delays>,
    dipole_gains: Option<Array2<f64>>,
    unity_dipole_gains: bool,
    input_data_type: Option<VisInputType>,
    printer: &mut InfoPrinter,
) -> Result<(Delays, Option<Array2<f64>>), BeamError> {
    let mut dipole_delays = match user_dipole_delays {
        Some(d) => Some(Delays::parse(d)?),
        None => data_dipole_delays,
    }
    .ok_or(BeamError::NoDelays(beam_name))?;
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

    // Check that the delays are sensible. The beam objects do this too, but
    // doing it here means we don't print anything else before complaining.
    validate_delays(&dipole_delays, total_num_tiles)?;

    let dipole_gains = if unity_dipole_gains {
        printer.push_line("Assuming all dipoles are \"alive\"".into());
        None
    } else {
        // If we don't have dipole gains from the input data, then we issue a
        // warning that we must assume no dead dipoles.
        if dipole_gains.is_none() {
            match input_data_type {
                Some(VisInputType::MeasurementSet) => [
                    "Measurement sets cannot supply dead dipole information.".into(),
                    "Without a metafits file, we must assume all dipoles are alive.".into(),
                    "This will make beam Jones matrices inaccurate in sky-model generation.".into(),
                ]
                .warn(),
                Some(VisInputType::Uvfits) => [
                    "uvfits files cannot supply dead dipole information.".into(),
                    "Without a metafits file, we must assume all dipoles are alive.".into(),
                    "This will make beam Jones matrices inaccurate in sky-model generation.".into(),
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

        // Currently, the only way to have dipole gains other than zero or one
        // is by using Aman's "DipAmps" metafits column.
        if dipole_gains.iter().any(|&g| g != 0.0 && g != 1.0) {
            printer.push_line("Using Aman's 'DipAmps' dipole gains from the metafits".into());
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
        // If we don't have dipole gains, we must assume all dipoles are
        // "alive". But, if any dipole delays are 32, then the beam code will
        // still ignore those dipoles. So use ideal dipole delays for all tiles.
        dipole_delays.set_to_ideal_delays();
        let ideal_delays = dipole_delays.get_ideal_delays();

        // Warn the user if they wanted unity dipole gains but the ideal dipole
        // delays contain 32.
        if unity_dipole_gains && ideal_delays.contains(&32) {
            "Some ideal dipole delays are 32; these dipoles will not have unity gains".warn()
        }
    }

    Ok((dipole_delays, dipole_gains))
}
