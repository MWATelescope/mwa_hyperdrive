// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to reduce a sky-model source list to the top N sources.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use itertools::Itertools;
use log::{debug, info, trace, warn};
use structopt::StructOpt;

use crate::{
    ao, constants::*, hyperdrive, rts, veto_sources, woden, HyperdriveFileType, SourceList,
    SourceListType, SrclistError, WriteSourceListError, CONVERT_INPUT_TYPE_HELP,
    CONVERT_OUTPUT_TYPE_HELP, SOURCE_DIST_CUTOFF_HELP, VETO_THRESHOLD_HELP,
};
use mwa_hyperdrive_beam::{create_fee_beam_object, Delays};
use mwa_rust_core::{mwalib, RADec};

/// Reduce a sky-model source list to the top N brightest sources, given
/// pointing information.
///
/// See for more info:
/// https://github.com/MWATelescope/mwa_hyperdrive/wiki/Source-lists
#[derive(StructOpt, Debug)]
pub struct ByBeamArgs {
    /// Path to the source list to be converted.
    #[structopt(name = "INPUT_SOURCE_LIST", parse(from_os_str))]
    pub input_source_list: PathBuf,

    /// Path to the output source list. If not specified, then then "_N" is
    /// appended to the filename.
    #[structopt(name = "OUTPUT_SOURCE_LIST", parse(from_os_str))]
    pub output_source_list: Option<PathBuf>,

    #[structopt(short = "i", long, parse(from_str), help = CONVERT_INPUT_TYPE_HELP.as_str())]
    pub input_type: Option<String>,

    #[structopt(short = "o", long, parse(from_str), help = CONVERT_OUTPUT_TYPE_HELP.as_str())]
    pub output_type: Option<String>,

    /// Reduce the input source list to the brightest N sources and write them
    /// to the output source list. If the input source list has less than N
    /// sources, then all sources are used.
    #[structopt(short = "n", long)]
    pub number: usize,

    /// Path to the metafits file.
    #[structopt(short = "m", long, parse(from_str))]
    pub metafits: PathBuf,

    #[structopt(long, help = SOURCE_DIST_CUTOFF_HELP.as_str())]
    pub source_dist_cutoff: Option<f64>,

    #[structopt(long, help = VETO_THRESHOLD_HELP.as_str())]
    pub veto_threshold: Option<f64>,

    /// Path to the MWA FEE beam file. If this is not specified, then the
    /// MWA_BEAM_FILE environment variable should contain the path.
    #[structopt(short, long)]
    pub beam_file: Option<PathBuf>,

    /// Attempt to convert flux density lists to power laws. See the online help
    /// for more information.
    #[structopt(short, long)]
    pub convert_lists: bool,

    /// Collapse all of the sky-model components into a single source; the
    /// apparently brightest source is used as the base source. This is suitable
    /// for an "RTS patch source list".
    #[structopt(long)]
    pub collapse_into_single_source: bool,

    /// Don't include point components from the input sky model.
    #[structopt(long)]
    filter_points: bool,

    /// Don't include Gaussian components from the input sky model.
    #[structopt(long)]
    filter_gaussians: bool,

    /// Don't include shapelet components from the input sky model.
    #[structopt(long)]
    filter_shapelets: bool,

    /// The verbosity of the program. The default is to print high-level
    /// information.
    #[structopt(short, long, parse(from_occurrences))]
    pub verbosity: u8,
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
            self.convert_lists,
            self.collapse_into_single_source,
            self.filter_points,
            self.filter_gaussians,
            self.filter_shapelets,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub fn by_beam<T: AsRef<Path>, S: AsRef<str>>(
    input_path: T,
    output_path: Option<T>,
    input_type: Option<S>,
    output_type: Option<S>,
    num_sources: usize,
    metafits: T,
    source_dist_cutoff: Option<f64>,
    veto_threshold: Option<f64>,
    beam_file: Option<T>,
    convert_lists: bool,
    collapse_into_single_source: bool,
    filter_points: bool,
    filter_gaussians: bool,
    filter_shapelets: bool,
) -> Result<(), SrclistError> {
    // Read the input source list.
    let input_path = input_path.as_ref().to_path_buf();
    let input_type = input_type.and_then(|t| SourceListType::from_str(t.as_ref()).ok());
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

    // Handle the output path and type.
    let output_path = match output_path {
        Some(p) => p.as_ref().to_path_buf(),
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
    let output_type = match output_type {
        Some(t) => {
            // Try to parse the specified output type.
            match SourceListType::from_str(t.as_ref()) {
                Ok(t) => t,
                Err(_) => return Err(WriteSourceListError::NotEnoughInfo.into()),
            }
        }
        // Use the input source list type as the output type.
        None => sl_type,
    };
    let output_ext = output_path.extension().and_then(|e| e.to_str());
    let output_file_type = output_ext.and_then(|e| HyperdriveFileType::from_str(e).ok());

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
    let delays = get_true_delays(&meta);
    let beam = create_fee_beam_object(Delays::Available(delays), beam_file)?;

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
    // If we were told to, attempt to convert lists of flux densities to power
    // laws.
    if convert_lists {
        sl.values_mut()
            .flat_map(|src| &mut src.components)
            .for_each(|comp| comp.flux_type.convert_list_to_power_law());
    }

    // Veto sources.
    let brightest_source_name = veto_sources(
        &mut sl,
        phase_centre,
        lst,
        mwa_rust_core::constants::MWA_LAT_RAD,
        &coarse_chan_freqs,
        beam.deref(),
        Some(num_sources),
        source_dist_cutoff.unwrap_or(DEFAULT_CUTOFF_DISTANCE),
        veto_threshold.unwrap_or(DEFAULT_VETO_THRESHOLD),
    )?;
    // Were any sources left after vetoing?
    let brightest_source_name = match brightest_source_name {
        Some(n) => n,
        None => return Err(SrclistError::NoSourcesAfterVeto),
    };

    // If requested, collapse the source list.
    sl = if collapse_into_single_source {
        let mut collapsed = SourceList::new();
        let brightest = sl.remove_entry(&brightest_source_name).unwrap();
        collapsed.insert(brightest.0, brightest.1);
        let base_src = collapsed.get_mut(&brightest_source_name).unwrap();
        sl.into_iter()
            .flat_map(|(_, src)| src.components)
            .for_each(|comp| base_src.components.push(comp));
        collapsed
    } else {
        sl
    };

    // Write the output source list.
    // TODO: De-duplicate this code.
    trace!("Attempting to write output source list");
    let mut f = BufWriter::new(File::create(&output_path)?);

    match (output_type, output_file_type) {
        (SourceListType::Hyperdrive, None) => {
            return Err(WriteSourceListError::InvalidHyperdriveFormat(
                output_ext.unwrap_or("<no extension>").to_string(),
            )
            .into())
        }
        (SourceListType::Rts, _) => {
            rts::write_source_list(&mut f, &sl)?;
            info!("Wrote rts-style source list to {}", output_path.display());
        }
        (SourceListType::AO, _) => {
            ao::write_source_list(&mut f, &sl)?;
            info!("Wrote ao-style source list to {}", output_path.display());
        }
        (SourceListType::Woden, _) => {
            woden::write_source_list(&mut f, &sl)?;
            info!("Wrote woden-style source list to {}", output_path.display());
        }
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
    }

    f.flush()?;

    Ok(())
}

/// MWA metafits files may have delays listed as all 32. This is code for "bad
/// observation, don't use". But, this has been a headache for researchers in
/// the past. When this situation is encountered, issue a warning, but then get
/// the actual observation's delays by iterating over each MWA tile's delays.
// TODO: Get this in mwalib.
fn get_true_delays(context: &mwalib::MetafitsContext) -> Vec<u32> {
    if !context.delays.iter().any(|&d| d == 32) {
        return context.delays.clone();
    }
    warn!("Metafits dipole delays contained 32s.");
    warn!("This may indicate that the observation's data shouldn't be used.");
    warn!("Proceeding to work out the true delays anyway...");

    // Individual dipoles in MWA tiles might be dead (i.e. delay of 32). To get
    // the true delays, iterate over all tiles until all values are non-32.
    let mut delays = [32; 16];
    for rf in &context.rf_inputs {
        for (mwalib_delay, true_delay) in rf.dipole_delays.iter().zip(delays.iter_mut()) {
            if *mwalib_delay != 32 {
                *true_delay = *mwalib_delay;
            }
        }

        // Are all delays non-32?
        if delays.iter().all(|&d| d != 32) {
            break;
        }
    }
    delays.to_vec()
}
