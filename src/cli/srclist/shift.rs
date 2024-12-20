// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    str::FromStr,
};

use clap::Parser;
use indexmap::IndexMap;
use itertools::Itertools;
use log::{debug, info, trace};
use marlu::RADec;
use rayon::prelude::*;
use serde::Deserialize;

use crate::{
    cli::common::{
        display_warnings, Warn, SOURCE_LIST_INPUT_TYPE_HELP, SOURCE_LIST_OUTPUT_TYPE_HELP,
    },
    srclist::{
        read::read_source_list_file, rts, write_source_list, Source, SourceList, SourceListType,
        SrclistError, WriteSourceListError,
    },
    HyperdriveError,
};

/// Shift the sources in a source list. Useful to correct for the ionosphere.
/// The shifts must be detailed in a .json file, with source names as keys
/// associated with an "ra" and "dec" in degrees. Only the sources specified in
/// the .json are written to the output source list.
#[derive(Parser, Debug)]
pub struct SrclistShiftArgs {
    /// Path to the source list to be shifted.
    #[clap(name = "SOURCE_LIST", parse(from_os_str))]
    source_list: PathBuf,

    /// Path to the .json shifts file.
    #[clap(name = "SOURCE_SHIFTS", parse(from_os_str))]
    source_shifts: PathBuf,

    /// Path to the output source list. If not specified, then then "_shifted"
    /// is appended to the filename.
    #[clap(name = "OUTPUT_SOURCE_LIST", parse(from_os_str))]
    output_source_list: Option<PathBuf>,

    #[clap(short = 'i', long, parse(from_str), help = SOURCE_LIST_INPUT_TYPE_HELP.as_str())]
    input_type: Option<String>,

    #[clap(short = 'o', long, parse(from_str), help = SOURCE_LIST_OUTPUT_TYPE_HELP.as_str())]
    output_type: Option<String>,

    /// Collapse all of the sky-model components into a single source; the
    /// apparently brightest source is used as the base source. This is suitable
    /// for an "RTS patch source list".
    #[clap(long)]
    collapse_into_single_source: bool,

    /// Don't throw away sources that have no shifts specified in the JSON file.
    #[clap(long)]
    include_unshifted_sources: bool,

    /// Path to the metafits file. Only needed if collapse-into-single-source is
    /// used.
    #[clap(short, long, parse(from_str))]
    metafits: Option<PathBuf>,
}

impl SrclistShiftArgs {
    /// Use the arguments to shift a source list.
    pub fn run(&self) -> Result<(), HyperdriveError> {
        shift(
            &self.source_list,
            &self.source_shifts,
            self.output_source_list.as_deref(),
            self.input_type.as_deref(),
            self.output_type.as_deref(),
            self.collapse_into_single_source,
            self.include_unshifted_sources,
            self.metafits.as_deref(),
        )?;
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn shift(
    source_list_file: &Path,
    source_shifts_file: &Path,
    output_source_list_file: Option<&Path>,
    source_list_input_type: Option<&str>,
    source_list_output_type: Option<&str>,
    collapse_into_single_source: bool,
    include_unshifted_sources: bool,
    metafits_file: Option<&Path>,
) -> Result<(), SrclistError> {
    let output_path: PathBuf = match output_source_list_file {
        Some(p) => p.to_path_buf(),
        None => {
            let input_path_base = source_list_file
                .file_stem()
                .and_then(|os_str| os_str.to_str())
                .expect("Input file didn't have a filename stem");
            let input_path_ext = source_list_file
                .extension()
                .and_then(|os_str| os_str.to_str())
                .expect("Input file didn't have an extension");
            let output_pb = PathBuf::from(format!("{input_path_base}_shifted.{input_path_ext}"));
            trace!("Writing shifted source list to {}", output_pb.display());
            output_pb
        }
    };
    let input_type = source_list_input_type.and_then(|t| SourceListType::from_str(t).ok());

    let f = BufReader::new(File::open(source_shifts_file)?);
    let source_shifts: BTreeMap<String, RaDec> =
        serde_json::from_reader(f).map_err(WriteSourceListError::from)?;
    let (sl, sl_type) = crate::misc::expensive_op(
        || read_source_list_file(source_list_file, input_type),
        "Still reading source list file",
    )?;
    info!(
        "Successfully read {} as a {}-style source list",
        source_list_file.display(),
        sl_type
    );
    let counts = sl.get_counts();
    info!(
        "Input has {} points, {} gaussians, {} shapelets",
        counts.num_points, counts.num_gaussians, counts.num_shapelets
    );

    let metafits: Option<mwalib::MetafitsContext> =
        match (collapse_into_single_source, metafits_file) {
            (false, _) => None,
            (true, None) => return Err(SrclistError::MissingMetafits),
            (true, Some(m)) => {
                trace!("Attempting to open the metafits file");
                let m = mwalib::MetafitsContext::new(m, None)?;
                Some(m)
            }
        };

    // If this an RTS source list, then the order of the sources in the source
    // list is important and must be preserved. When hyperdrive reads these
    // source lists, the ordering is thrown away because it was not designed
    // with this in mind (and, it should never consider it, as it's a detail
    // that we never want to care about). Here, we know that we've read an RTS
    // source list so we can (in a dirty fashion) get the order of the sources.
    let source_name_order: Option<Vec<String>> = match sl_type {
        SourceListType::Rts => {
            "Preserving the order of the RTS sources".warn();
            let mut names = vec![];
            let f = BufReader::new(File::open(source_list_file)?);
            for line in f.lines() {
                let line = line?;
                if line.starts_with("SOURCE") {
                    // unwrap is safe because we successfully read the RTS
                    // source list earlier.
                    let source_name = line.split_whitespace().nth(1).unwrap();
                    if include_unshifted_sources || source_shifts.contains_key(source_name) {
                        names.push(source_name.to_string());
                    }
                }
            }
            Some(names)
        }
        _ => None,
    };

    // Filter any sources that aren't in the shifts file, and shift the
    // sources. All components of a source get shifted the same amount.
    let mut sl = {
        let no_shift = RaDec { ra: 0.0, dec: 0.0 };
        let tmp_sl: IndexMap<String, Source> = sl
            .into_iter()
            .filter(|(name, _)| include_unshifted_sources || source_shifts.contains_key(name))
            .map(|(name, mut src)| {
                let shift = if source_shifts.contains_key(&name) {
                    &source_shifts[&name]
                } else {
                    &no_shift
                };
                src.components.iter_mut().for_each(|comp| {
                    comp.radec.ra += shift.ra.to_radians();
                    comp.radec.dec += shift.dec.to_radians();
                });
                (name, src)
            })
            .collect();
        SourceList::from(tmp_sl)
    };

    // If requested, collapse the source list.
    sl = if let Some(meta) = metafits {
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
        // If we're preserving the order of the RTS sources, then use the first
        // source as the base.
        if let Some(ordered) = &source_name_order {
            let base_name = ordered.first().unwrap().clone();
            let base = sl.remove_entry(&base_name).unwrap();
            collapsed.insert(base_name, base.1);
            let base_src = collapsed.get_mut(&base.0).unwrap();
            let mut base_comps = vec![].into_boxed_slice();
            std::mem::swap(&mut base_src.components, &mut base_comps);
            let mut base_comps = base_comps.to_vec();

            for name in &ordered[1..] {
                for comp in sl[name].components.iter() {
                    base_comps.push(comp.clone());
                }
            }
            std::mem::swap(&mut base_src.components, &mut base_comps.into_boxed_slice());
        } else {
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
            let base_name = brightest.0.clone();

            let base = sl.remove_entry(&base_name).unwrap();
            collapsed.insert(base_name, base.1);
            let base_src = collapsed.get_mut(&base.0).unwrap();
            let mut base_comps = vec![].into_boxed_slice();
            std::mem::swap(&mut base_src.components, &mut base_comps);
            let mut base_comps = base_comps.to_vec();
            sl.into_iter()
                .flat_map(|(_, src)| src.components.to_vec())
                .for_each(|comp| base_comps.push(comp));
            std::mem::swap(&mut base_src.components, &mut base_comps.into_boxed_slice());
        }

        collapsed
    } else {
        sl
    };
    let counts = sl.get_counts();
    info!(
        "Shifted {} points, {} gaussians, {} shapelets",
        counts.num_points, counts.num_gaussians, counts.num_shapelets
    );

    match (sl_type, source_name_order) {
        (SourceListType::Rts, Some(source_name_order)) => {
            trace!("Attempting to write ordered RTS source list");
            let mut f = std::io::BufWriter::new(std::fs::File::create(&output_path)?);
            rts::write_source_list_with_order(&mut f, &sl, source_name_order)?;
        }

        _ => write_source_list(
            &sl,
            &output_path,
            sl_type,
            source_list_output_type.and_then(|s| SourceListType::from_str(s).ok()),
            None,
        )?,
    }

    display_warnings();

    Ok(())
}

#[derive(Deserialize)]
struct RaDec {
    ra: f64,
    dec: f64,
}
