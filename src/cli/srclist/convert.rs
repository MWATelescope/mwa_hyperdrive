// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to convert between sky-model source list files.

use std::{
    path::{Path, PathBuf},
    str::FromStr,
};

use clap::Parser;
use itertools::Itertools;
use log::{debug, info, trace};
use marlu::RADec;
use rayon::prelude::*;

use crate::{
    cli::common::{
        display_warnings, Warn, SOURCE_LIST_INPUT_TYPE_HELP, SOURCE_LIST_OUTPUT_TYPE_HELP,
    },
    srclist::{
        read::read_source_list_file, write_source_list, HyperdriveFileType, SourceList,
        SourceListType, SrclistError,
    },
    HyperdriveError,
};

/// Convert a sky-model source list from one format to another.
///
/// See for more info:
/// <https://mwatelescope.github.io/mwa_hyperdrive/defs/source_lists.html>
#[derive(Parser, Debug)]
pub struct SrclistConvertArgs {
    #[clap(short = 'i', long, parse(from_str), help = SOURCE_LIST_INPUT_TYPE_HELP.as_str())]
    input_type: Option<String>,

    /// Path to the source list to be converted.
    #[clap(name = "INPUT_SOURCE_LIST", parse(from_os_str))]
    input_source_list: PathBuf,

    #[clap(short = 'o', long, parse(from_str), help = SOURCE_LIST_OUTPUT_TYPE_HELP.as_str())]
    output_type: Option<String>,

    /// Path to the output source list. If the file extension is .json or .yaml,
    /// then it will written in the hyperdrive source list format. If it is
    /// .txt, then the --output-type flag should be used to specify the type of
    /// source list to be written.
    #[clap(name = "OUTPUT_SOURCE_LIST", parse(from_os_str))]
    output_source_list: PathBuf,

    /// Collapse all of the sky-model components into a single source; the
    /// apparently brightest source is used as the base source. This is suitable
    /// for an "RTS patch source list".
    #[clap(long)]
    collapse_into_single_source: bool,

    /// Path to the metafits file. Only needed if collapse-into-single-source is
    /// used.
    #[clap(short = 'm', long, parse(from_str))]
    metafits: Option<PathBuf>,

    /// Don't include point components from the input sky model.
    #[clap(long)]
    filter_points: bool,

    /// Don't include Gaussian components from the input sky model.
    #[clap(long)]
    filter_gaussians: bool,

    /// Don't include shapelet components from the input sky model.
    #[clap(long)]
    filter_shapelets: bool,
}

impl SrclistConvertArgs {
    /// Run [convert] with these arguments.
    pub fn run(&self) -> Result<(), HyperdriveError> {
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
        )?;
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn convert<P: AsRef<Path>, S: AsRef<str>>(
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
            "Assuming that the output file type is 'hyperdrive'".warn();
        }

        // Read the input source list.
        let (sl, sl_type) = crate::misc::expensive_op(
            || read_source_list_file(input_path, input_type),
            "Still reading source list file",
        )?;
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

        // If requested, collapse the source list.
        sl = if collapse_into_single_source {
            // Open the metafits.
            let metafits = match &metafits {
                Some(m) => m,
                None => return Err(SrclistError::MissingMetafits),
            };
            trace!("Attempting to open the metafits file");
            let meta = mwalib::MetafitsContext::new(metafits, None)?;
            let ra_phase_centre = meta
                .ra_phase_center_degrees
                .unwrap_or(meta.ra_tile_pointing_degrees);
            let dec_phase_centre = meta
                .dec_phase_center_degrees
                .unwrap_or(meta.dec_tile_pointing_degrees);
            let phase_centre = RADec::from_degrees(ra_phase_centre, dec_phase_centre);
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
                    .map(|cc_freq_hz| format!("{:.2}", *cc_freq_hz / 1e6))
                    .join(", ")
            );

            let mut collapsed = SourceList::new();
            // Use the apparently brightest source as the base.
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
            let mut base_comps = vec![].into_boxed_slice();
            std::mem::swap(&mut base_src.components, &mut base_comps);
            let mut base_comps = base_comps.into_vec();
            sl.into_iter()
                .flat_map(|(_, src)| src.components.to_vec())
                .for_each(|comp| base_comps.push(comp));
            std::mem::swap(&mut base_src.components, &mut base_comps.into_boxed_slice());
            collapsed
        } else {
            sl
        };

        // Write the output source list.
        trace!("Attempting to write output source list");
        write_source_list(&sl, output_path, sl_type, output_type, None)?;

        display_warnings();

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
