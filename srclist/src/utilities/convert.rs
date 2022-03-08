// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to convert between sky-model source list files.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;

use clap::Parser;
use itertools::Itertools;
use log::{debug, info, trace, warn};
use marlu::RADec;
use rayon::prelude::*;

use crate::{
    ao, hyperdrive, rts, woden, HyperdriveFileType, SourceList, SourceListType, SrclistError,
    WriteSourceListError, SOURCE_LIST_INPUT_TYPE_HELP, SOURCE_LIST_OUTPUT_TYPE_HELP,
};
use mwa_hyperdrive_common::{clap, itertools, log, marlu, mwalib, rayon};

/// Convert a sky-model source list from one format to another.
///
/// See for more info:
/// https://github.com/MWATelescope/mwa_hyperdrive/wiki/Source-lists
#[derive(Parser, Debug)]
pub struct ConvertArgs {
    #[clap(short = 'i', long, parse(from_str), help = SOURCE_LIST_INPUT_TYPE_HELP.as_str())]
    pub input_type: Option<String>,

    /// Path to the source list to be converted.
    #[clap(name = "INPUT_SOURCE_LIST", parse(from_os_str))]
    pub input_source_list: PathBuf,

    #[clap(short = 'o', long, parse(from_str), help = SOURCE_LIST_OUTPUT_TYPE_HELP.as_str())]
    pub output_type: Option<String>,

    /// Path to the output source list. If the file extension is .json or .yaml,
    /// then it will written in the hyperdrive source list format. If it is
    /// .txt, then the --output-type flag should be used to specify the type of
    /// source list to be written.
    #[clap(name = "OUTPUT_SOURCE_LIST", parse(from_os_str))]
    pub output_source_list: PathBuf,

    /// Collapse all of the sky-model components into a single source; the
    /// apparently brightest source is used as the base source. This is suitable
    /// for an "RTS patch source list".
    #[clap(long)]
    pub collapse_into_single_source: bool,

    /// Path to the metafits file. Only needed if collapse-into-single-source is
    /// used.
    #[clap(short = 'm', long, parse(from_str))]
    pub metafits: Option<PathBuf>,

    /// Don't include point components from the input sky model.
    #[clap(long)]
    filter_points: bool,

    /// Don't include Gaussian components from the input sky model.
    #[clap(long)]
    filter_gaussians: bool,

    /// Don't include shapelet components from the input sky model.
    #[clap(long)]
    filter_shapelets: bool,

    /// The verbosity of the program. The default is to print high-level
    /// information.
    #[clap(short, long, parse(from_occurrences))]
    pub verbosity: u8,
}

impl ConvertArgs {
    /// Run [convert] with these arguments.
    pub fn run(&self) -> Result<(), SrclistError> {
        convert(
            &self.input_source_list,
            &self.output_source_list,
            self.input_type.as_ref(),
            self.output_type.as_ref(),
            self.collapse_into_single_source,
            self.metafits.as_ref(),
            self.filter_points,
            self.filter_gaussians,
            self.filter_shapelets,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub fn convert<P: AsRef<Path>, S: AsRef<str>>(
    input_path: P,
    output_path: P,
    input_type: Option<S>,
    output_type: Option<S>,
    collapse_into_single_source: bool,
    metafits: Option<P>,
    filter_points: bool,
    filter_gaussians: bool,
    filter_shapelets: bool,
) -> Result<(), SrclistError> {
    fn inner(
        input_path: &Path,
        output_path: &Path,
        input_type: Option<&str>,
        output_type: Option<&str>,
        collapse_into_single_source: bool,
        metafits: Option<&Path>,
        filter_points: bool,
        filter_gaussians: bool,
        filter_shapelets: bool,
    ) -> Result<(), SrclistError> {
        let input_type = input_type.and_then(|t| SourceListType::from_str(t).ok());
        let output_type = output_type.and_then(|t| SourceListType::from_str(t).ok());

        // When writing out hyperdrive-style source lists, we must know the output
        // file format. This can either be explicit from the user input, or the file
        // extension.
        let output_ext = output_path.extension().and_then(|e| e.to_str());
        let output_file_type = output_ext.and_then(|e| HyperdriveFileType::from_str(e).ok());
        if output_type.is_none() && output_file_type.is_some() {
            warn!("Assuming that the output file type is 'hyperdrive'");
        }

        // Read the input source list.
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
            counts.0, counts.1, counts.2
        );

        // Apply any filters.
        let mut sl = if filter_points || filter_gaussians || filter_shapelets {
            let sl = sl.filter(filter_points, filter_gaussians, filter_shapelets);
            let counts = sl.get_counts();
            debug!(
                "After filtering, there are {} points, {} gaussians, {} shapelets",
                counts.0, counts.1, counts.2
            );
            sl
        } else {
            sl
        };

        // If requested, collapse the source list.
        sl = if collapse_into_single_source {
            // Open the metafits.
            let metafits = match &metafits {
                Some(m) => m,
                None => return Err(SrclistError::MissingMetafits),
            };
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

            let mut collapsed = SourceList::new();
            // Use the apparently brightest source as the base. Not sure this is
            // necessary or important, but hey, it's the RTS we're talking about.
            let brightest = sl
                .par_iter()
                .map(|(name, src)| {
                    let stokes_i = src
                        .get_flux_estimates(150e6)
                        .iter()
                        .fold(0.0, |acc, fd| acc + fd.i);
                    (name, stokes_i)
                })
                .max_by(|x, y| x.1.partial_cmp(&y.1).unwrap())
                .unwrap();
            let brightest_name = brightest.0.clone();
            let brightest = sl.remove_entry(&brightest_name).unwrap();
            collapsed.insert(brightest_name, brightest.1);
            let base_src = collapsed.get_mut(&brightest.0).unwrap();
            sl.into_iter()
                .flat_map(|(_, src)| src.components)
                .for_each(|comp| base_src.components.push(comp));
            collapsed
        } else {
            sl
        };

        // Write the output source list.
        trace!("Attempting to write output source list");
        let mut f = BufWriter::new(File::create(&output_path)?);

        match (output_type, output_file_type) {
            (_, Some(HyperdriveFileType::Yaml)) => {
                hyperdrive::source_list_to_yaml(&mut f, &sl)?;
                info!(
                    "Wrote hyperdrive-style source list to {}",
                    output_path.display()
                );
            }
            (_, Some(HyperdriveFileType::Json)) => {
                hyperdrive::source_list_to_json(&mut f, &sl)?;
                info!(
                    "Wrote hyperdrive-style source list to {}",
                    output_path.display()
                );
            }
            (Some(SourceListType::Hyperdrive), None) => {
                return Err(WriteSourceListError::InvalidHyperdriveFormat(
                    output_ext.unwrap_or("<no extension>").to_string(),
                )
                .into())
            }
            (Some(SourceListType::Rts), _) => {
                rts::write_source_list(&mut f, &sl)?;
                info!("Wrote rts-style source list to {}", output_path.display());
            }
            (Some(SourceListType::AO), _) => {
                ao::write_source_list(&mut f, &sl)?;
                info!("Wrote ao-style source list to {}", output_path.display());
            }
            (Some(SourceListType::Woden), _) => {
                woden::write_source_list(&mut f, &sl)?;
                info!("Wrote woden-style source list to {}", output_path.display());
            }
            (None, None) => return Err(WriteSourceListError::NotEnoughInfo.into()),
        }

        f.flush()?;

        Ok(())
    }
    inner(
        input_path.as_ref(),
        output_path.as_ref(),
        input_type.as_ref().map(|f| f.as_ref()),
        output_type.as_ref().map(|f| f.as_ref()),
        collapse_into_single_source,
        metafits.as_ref().map(|f| f.as_ref()),
        filter_points,
        filter_gaussians,
        filter_shapelets,
    )
}
