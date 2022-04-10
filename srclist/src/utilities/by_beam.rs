// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to reduce a sky-model source list to the top N sources.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use clap::Parser;
use itertools::Itertools;
use log::{debug, info, trace, warn};
use marlu::RADec;
use ndarray::{Array1, Array2};

use crate::{
    ao, hyperdrive, rts, veto_sources, woden, HyperdriveFileType, SourceList, SourceListType,
    SrclistError, WriteSourceListError, DEFAULT_CUTOFF_DISTANCE, DEFAULT_VETO_THRESHOLD,
    SOURCE_DIST_CUTOFF_HELP, SOURCE_LIST_INPUT_TYPE_HELP, SOURCE_LIST_OUTPUT_TYPE_HELP,
    VETO_THRESHOLD_HELP,
};
use mwa_hyperdrive_beam::{create_fee_beam_object, Delays};
use mwa_hyperdrive_common::{clap, itertools, log, marlu, mwalib, ndarray};

/// Reduce a sky-model source list to the top N brightest sources, given
/// pointing information.
///
/// See for more info:
/// https://github.com/MWATelescope/mwa_hyperdrive/wiki/Source-lists
#[derive(Parser, Debug)]
pub struct ByBeamArgs {
    /// Path to the source list to be converted.
    #[clap(
        name = "INPUT_SOURCE_LIST",
        parse(from_os_str),
        help_heading = "INPUT/OUTPUT FILES"
    )]
    pub input_source_list: PathBuf,

    /// Path to the output source list. If not specified, then then "_N" is
    /// appended to the filename.
    #[clap(
        name = "OUTPUT_SOURCE_LIST",
        parse(from_os_str),
        help_heading = "INPUT/OUTPUT FILES"
    )]
    pub output_source_list: Option<PathBuf>,

    #[clap(short = 'i', long, parse(from_str), help = SOURCE_LIST_INPUT_TYPE_HELP.as_str(), help_heading = "INPUT/OUTPUT FILES")]
    pub input_type: Option<String>,

    #[clap(short = 'o', long, parse(from_str), help = SOURCE_LIST_OUTPUT_TYPE_HELP.as_str(), help_heading = "INPUT/OUTPUT FILES")]
    pub output_type: Option<String>,

    /// Path to the metafits file.
    #[clap(
        short = 'm',
        long,
        parse(from_str),
        help_heading = "INPUT/OUTPUT FILES"
    )]
    pub metafits: PathBuf,

    /// Path to the MWA FEE beam file. If this is not specified, then the
    /// MWA_BEAM_FILE environment variable should contain the path.
    #[clap(short, long, help_heading = "INPUT/OUTPUT FILES")]
    pub beam_file: Option<PathBuf>,

    /// Reduce the input source list to the brightest N sources and write them
    /// to the output source list. If the input source list has less than N
    /// sources, then all sources are used.
    #[clap(short = 'n', long, help_heading = "SOURCE FILTERING")]
    pub number: usize,

    #[clap(long, help = SOURCE_DIST_CUTOFF_HELP.as_str(), help_heading = "SOURCE FILTERING")]
    pub source_dist_cutoff: Option<f64>,

    #[clap(long, help = VETO_THRESHOLD_HELP.as_str(), help_heading = "SOURCE FILTERING")]
    pub veto_threshold: Option<f64>,

    /// Don't include point components from the input sky model.
    #[clap(long, help_heading = "SOURCE FILTERING")]
    filter_points: bool,

    /// Don't include Gaussian components from the input sky model.
    #[clap(long, help_heading = "SOURCE FILTERING")]
    filter_gaussians: bool,

    /// Don't include shapelet components from the input sky model.
    #[clap(long, help_heading = "SOURCE FILTERING")]
    filter_shapelets: bool,

    /// The verbosity of the program. The default is to print high-level
    /// information.
    #[clap(short, long, parse(from_occurrences))]
    pub verbosity: u8,

    /// Collapse all of the sky-model components into a single source; the
    /// apparently brightest source is used as the base source (unless overriden
    /// below). This is suitable for an "RTS patch source list" in DI
    /// calibration.
    #[clap(long, help_heading = "RTS-ONLY ARGUMENTS")]
    pub collapse_into_single_source: bool,

    /// If collapsing the source list into a single source, use this source as
    /// the base source; this is very important for RTS DI calibration.
    #[clap(long, help_heading = "RTS-ONLY ARGUMENTS")]
    pub rts_base_source: Option<String>,
}

impl ByBeamArgs {
    /// Run [by_beam] with these arguments.
    pub fn run(&self) -> Result<(), SrclistError> {
        by_beam(
            &self.input_source_list,
            self.output_source_list.as_ref(),
            self.input_type.as_ref(),
            self.output_type.as_ref(),
            self.number,
            &self.metafits,
            self.source_dist_cutoff,
            self.veto_threshold,
            self.beam_file.as_ref(),
            self.filter_points,
            self.filter_gaussians,
            self.filter_shapelets,
            self.collapse_into_single_source,
            self.rts_base_source.as_ref(),
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub fn by_beam<P: AsRef<Path>, S: AsRef<str>>(
    input_path: P,
    output_path: Option<P>,
    input_type: Option<S>,
    output_type: Option<S>,
    num_sources: usize,
    metafits: P,
    source_dist_cutoff: Option<f64>,
    veto_threshold: Option<f64>,
    beam_file: Option<P>,
    filter_points: bool,
    filter_gaussians: bool,
    filter_shapelets: bool,
    collapse_into_single_source: bool,
    rts_base_source: Option<S>,
) -> Result<(), SrclistError> {
    fn inner(
        input_path: &Path,
        output_path: Option<&Path>,
        input_type: Option<&str>,
        output_type: Option<&str>,
        num_sources: usize,
        metafits: &Path,
        source_dist_cutoff: Option<f64>,
        veto_threshold: Option<f64>,
        beam_file: Option<&Path>,
        filter_points: bool,
        filter_gaussians: bool,
        filter_shapelets: bool,
        collapse_into_single_source: bool,
        rts_base_source: Option<&str>,
    ) -> Result<(), SrclistError> {
        // Read the input source list.
        let input_type = input_type.and_then(|t| SourceListType::from_str(t).ok());
        let (sl, sl_type) = crate::read::read_source_list_file(&input_path, input_type)?;
        if input_type.is_none() {
            info!(
                "Successfully read {} as a {}-style source list",
                input_path.display(),
                sl_type
            );
        }
        let counts = sl.get_counts();
        info!(
            "{} points, {} gaussians, {} shapelets",
            counts.num_points, counts.num_gaussians, counts.num_shapelets
        );

        // Handle the output path and type.
        let output_path = match output_path {
            Some(p) => p.to_path_buf(),
            None => {
                let input_path_base = input_path
                    .file_stem()
                    .and_then(|os_str| os_str.to_str())
                    .map(|str| str.to_string())
                    .expect("Input file didn't have a filename stem");
                let input_path_ext = input_path
                    .extension()
                    .and_then(|os_str| os_str.to_str())
                    .expect("Input file didn't have an extension");
                let mut output_path = input_path_base;
                output_path.push_str(&format!("_{}", num_sources));
                output_path.push_str(&format!(".{}", input_path_ext));
                let output_pb = PathBuf::from(output_path);
                debug!("Writing reduced source list to {}", output_pb.display());
                output_pb
            }
        };
        let output_ext = output_path.extension().and_then(|e| e.to_str());
        let hyp_file_type = output_ext.and_then(|e| HyperdriveFileType::from_str(e).ok());
        let output_type = match (output_type, &hyp_file_type) {
            (Some(t), _) => {
                // Try to parse the specified output type.
                match SourceListType::from_str(t) {
                    Ok(t) => t,
                    Err(_) => return Err(WriteSourceListError::InvalidFormat.into()),
                }
            }

            (None, Some(_)) => SourceListType::Hyperdrive,

            // Use the input source list type as the output type.
            (None, None) => sl_type,
        };

        // Open the metafits.
        trace!("Attempting to open the metafits file");
        let meta = mwalib::MetafitsContext::new(&metafits, None)?;
        let ra_phase_centre = meta
            .ra_phase_center_degrees
            .unwrap_or(meta.ra_tile_pointing_degrees);
        let dec_phase_centre = meta
            .dec_phase_center_degrees
            .unwrap_or(meta.dec_tile_pointing_degrees);
        let phase_centre = RADec::new_degrees(ra_phase_centre, dec_phase_centre);
        debug!("Using {} as the phase centre", phase_centre);
        let lst = meta.lst_rad;
        debug!("Using {}° as the LST", lst.to_degrees());
        let coarse_chan_freqs: Vec<f64> = meta
            .metafits_coarse_chans
            .iter()
            .map(|cc| cc.chan_centre_hz as _)
            .collect();
        debug!(
            "Using coarse channel frequencies [MHz]: {}",
            coarse_chan_freqs
                .iter()
                .map(|cc_freq_hz| format!("{:.2}", *cc_freq_hz as f64 / 1e6))
                .join(", ")
        );

        // Set up the beam.
        // TODO: Use ideal dipole delays.
        let beam =
            create_fee_beam_object(beam_file, 1, Delays::Full(get_dipole_delays(&meta)), None)?;

        // Apply any filters.
        let mut sl = if filter_points || filter_gaussians || filter_shapelets {
            let sl = sl.filter(filter_points, filter_gaussians, filter_shapelets);
            let counts = sl.get_counts();
            debug!(
                "After filtering, there are {} points, {} gaussians, {} shapelets",
                counts.num_points, counts.num_gaussians, counts.num_shapelets
            );
            sl
        } else {
            sl
        };

        // Veto sources.
        veto_sources(
            &mut sl,
            phase_centre,
            lst,
            marlu::constants::MWA_LAT_RAD,
            &coarse_chan_freqs,
            beam.deref(),
            None,
            source_dist_cutoff.unwrap_or(DEFAULT_CUTOFF_DISTANCE),
            veto_threshold.unwrap_or(DEFAULT_VETO_THRESHOLD),
        )?;
        // Were any sources left after vetoing?
        if sl.is_empty() {
            return Err(SrclistError::NoSourcesAfterVeto);
        };

        // If requested, collapse the source list.
        sl = if collapse_into_single_source {
            let base = rts_base_source
                .unwrap_or(sl.get_index(0).unwrap().0)
                .to_owned();
            let mut collapsed = SourceList::new();
            let base = sl.remove_entry(&base).unwrap();
            collapsed.insert(base.0, base.1);
            let base_src = collapsed.get_index_mut(0).unwrap().1;
            sl.into_iter()
                .flat_map(|(_, src)| src.components)
                .for_each(|comp| base_src.components.push(comp));
            collapsed
        } else {
            if rts_base_source.is_some() {
                warn!("RTS base source was supplied, but we're not collapsing the source list into a single source.");
            }
            sl
        };

        // Write the output source list.
        // TODO: De-duplicate this code.
        trace!("Attempting to write output source list");
        let mut f = BufWriter::new(File::create(&output_path)?);

        match (output_type, hyp_file_type) {
            (SourceListType::Hyperdrive, None) => {
                return Err(WriteSourceListError::InvalidHyperdriveFormat(
                    output_ext.unwrap_or("<no extension>").to_string(),
                )
                .into())
            }
            (SourceListType::Rts, _) => {
                rts::write_source_list(&mut f, &sl, Some(num_sources))?;
                info!("Wrote rts-style source list to {}", output_path.display());
            }
            (SourceListType::AO, _) => {
                ao::write_source_list(&mut f, &sl, Some(num_sources))?;
                info!("Wrote ao-style source list to {}", output_path.display());
            }
            (SourceListType::Woden, _) => {
                woden::write_source_list(&mut f, &sl, Some(num_sources))?;
                info!("Wrote woden-style source list to {}", output_path.display());
            }
            (_, Some(HyperdriveFileType::Yaml)) => {
                hyperdrive::source_list_to_yaml(&mut f, &sl, Some(num_sources))?;
                info!(
                    "Wrote hyperdrive-style source list to {}",
                    output_path.display()
                );
            }
            (_, Some(HyperdriveFileType::Json)) => {
                hyperdrive::source_list_to_json(&mut f, &sl, Some(num_sources))?;
                info!(
                    "Wrote hyperdrive-style source list to {}",
                    output_path.display()
                );
            }
        }
        f.flush()?;

        Ok(())
    }
    inner(
        input_path.as_ref(),
        output_path.as_ref().map(|f| f.as_ref()),
        input_type.as_ref().map(|f| f.as_ref()),
        output_type.as_ref().map(|f| f.as_ref()),
        num_sources,
        metafits.as_ref(),
        source_dist_cutoff,
        veto_threshold,
        beam_file.as_ref().map(|f| f.as_ref()),
        filter_points,
        filter_gaussians,
        filter_shapelets,
        collapse_into_single_source,
        rts_base_source.as_ref().map(|f| f.as_ref()),
    )
}

/// Get the delays for each tile's dipoles.
pub(crate) fn get_dipole_delays(context: &mwalib::MetafitsContext) -> Array2<u32> {
    let mut delays = Array2::zeros((context.num_ants, 16));
    delays
        .outer_iter_mut()
        .zip(
            context
                .rf_inputs
                .iter()
                .filter(|rf| rf.pol == mwalib::Pol::X),
        )
        .for_each(|(mut tile_delays, rf)| {
            tile_delays.assign(&Array1::from(rf.dipole_delays.clone()));
        });
    delays
}
