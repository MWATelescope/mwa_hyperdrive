// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to reduce a sky-model source list to the top N sources.

#[cfg(test)]
mod tests;

use std::{
    borrow::Cow,
    path::{Path, PathBuf},
    str::FromStr,
};

use clap::Parser;
use itertools::Itertools;
use log::{debug, info, trace};
use marlu::{LatLngHeight, RADec};

use crate::{
    beam::Delays,
    cli::common::{
        display_warnings, BeamArgs, Warn, ARRAY_POSITION_HELP, SOURCE_DIST_CUTOFF_HELP,
        SOURCE_LIST_INPUT_TYPE_HELP, SOURCE_LIST_OUTPUT_TYPE_HELP, VETO_THRESHOLD_HELP,
    },
    constants::{DEFAULT_CUTOFF_DISTANCE, DEFAULT_VETO_THRESHOLD},
    metafits::get_dipole_delays,
    srclist::{
        read::read_source_list_file, veto_sources, write_source_list, ReadSourceListError,
        SourceList, SourceListType, WriteSourceListError,
    },
    HyperdriveError,
};

/// Reduce a sky-model source list to the top N brightest sources, given
/// pointing information.
///
/// See for more info:
/// <https://mwatelescope.github.io/mwa_hyperdrive/defs/source_lists.html>
#[derive(Parser, Debug)]
pub struct SrclistByBeamArgs {
    /// Path to the source list to be converted.
    #[clap(
        name = "INPUT_SOURCE_LIST",
        parse(from_os_str),
        help_heading = "INPUT FILES"
    )]
    input_source_list: PathBuf,

    /// Path to the output source list. If not specified, then then "_N" is
    /// appended to the filename.
    #[clap(
        name = "OUTPUT_SOURCE_LIST",
        parse(from_os_str),
        help_heading = "OUTPUT FILES"
    )]
    output_source_list: Option<PathBuf>,

    #[clap(short = 'i', long, parse(from_str), help = SOURCE_LIST_INPUT_TYPE_HELP.as_str(), help_heading = "INPUT FILES")]
    input_type: Option<String>,

    #[clap(short = 'o', long, parse(from_str), help = SOURCE_LIST_OUTPUT_TYPE_HELP.as_str(), help_heading = "OUTPUT FILES")]
    output_type: Option<String>,

    /// Path to the metafits file, which contains the metadata needed to veto
    /// sources.
    #[clap(short = 'm', long, parse(from_str), help_heading = "METADATA")]
    metafits: Option<PathBuf>,

    #[clap(
        long, help = ARRAY_POSITION_HELP.as_str(), help_heading = "METADATA",
        number_of_values = 3,
        allow_hyphen_values = true,
        value_names = &["LONG_DEG", "LAT_DEG", "HEIGHT_M"]
    )]
    array_position: Option<Vec<f64>>,

    /// The LST in radians. Overrides the value in the metafits.
    #[clap(
        long = "lst",
        help_heading = "METADATA",
        allow_hyphen_values = true,
        required_unless_present = "metafits"
    )]
    lst_rad: Option<f64>,

    /// The RA and Dec. phase centre of the observation in degrees. Overrides
    /// the value in metafits.
    #[clap(
        long,
        help_heading = "METADATA",
        number_of_values = 2,
        allow_hyphen_values = true,
        value_names = &["RA", "DEC"],
        required_unless_present = "metafits"
    )]
    phase_centre: Option<Vec<f64>>,

    /// A representative sample of frequencies in the observation [Hz]; it's
    /// typical to use the centre frequencies of each MWA coarse channel.
    /// Overrides the coarse channels in the metafits.
    #[clap(
        long = "freqs",
        help_heading = "METADATA",
        multiple_values(true),
        required_unless_present = "metafits"
    )]
    freqs_hz: Option<Vec<f64>>,

    /// Reduce the input source list to the brightest N sources and write them
    /// to the output source list. If the input source list has less than N
    /// sources, then all sources are used.
    #[clap(short = 'n', long, help_heading = "SOURCE FILTERING")]
    number: usize,

    #[clap(long, help = SOURCE_DIST_CUTOFF_HELP.as_str(), help_heading = "SOURCE FILTERING")]
    source_dist_cutoff: Option<f64>,

    #[clap(long, help = VETO_THRESHOLD_HELP.as_str(), help_heading = "SOURCE FILTERING")]
    veto_threshold: Option<f64>,

    /// Don't include point components from the input sky model.
    #[clap(long, help_heading = "SOURCE FILTERING")]
    filter_points: bool,

    /// Don't include Gaussian components from the input sky model.
    #[clap(long, help_heading = "SOURCE FILTERING")]
    filter_gaussians: bool,

    /// Don't include shapelet components from the input sky model.
    #[clap(long, help_heading = "SOURCE FILTERING")]
    filter_shapelets: bool,

    /// Collapse all of the sky-model components into a single source; the
    /// apparently brightest source is used as the base source (unless overriden
    /// below). This is suitable for an "RTS patch source list" in DI
    /// calibration.
    #[clap(long, help_heading = "RTS-ONLY ARGUMENTS")]
    collapse_into_single_source: bool,

    /// If collapsing the source list into a single source, use this source as
    /// the base source; this is very important for RTS DI calibration.
    #[clap(long, help_heading = "RTS-ONLY ARGUMENTS")]
    rts_base_source: Option<String>,

    #[clap(flatten)]
    beam_args: BeamArgs,
}

impl SrclistByBeamArgs {
    /// Run [`by_beam`] with these arguments.
    pub fn run(self) -> Result<(), HyperdriveError> {
        by_beam(
            &self.input_source_list,
            self.output_source_list.as_deref(),
            self.input_type.as_deref(),
            self.output_type.as_deref(),
            self.number,
            self.metafits.as_deref(),
            self.array_position.as_ref().map(|a| LatLngHeight {
                longitude_rad: a[0].to_radians(),
                latitude_rad: a[1].to_radians(),
                height_metres: a[2],
            }),
            self.lst_rad,
            self.phase_centre
                .as_ref()
                .map(|p| RADec::from_degrees(p[0], p[1])),
            self.freqs_hz.as_deref(),
            self.source_dist_cutoff,
            self.veto_threshold,
            self.filter_points,
            self.filter_gaussians,
            self.filter_shapelets,
            self.collapse_into_single_source,
            self.rts_base_source.as_deref(),
            self.beam_args,
        )?;
        Ok(())
    }
}

struct Metadata<'a> {
    phase_centre: RADec,
    array_position: LatLngHeight,
    lst_rad: f64,
    freqs_hz: Cow<'a, [f64]>,
    dipole_delays: Option<Delays>,
}

#[allow(clippy::too_many_arguments)]
fn by_beam(
    input_path: &Path,
    output_path: Option<&Path>,
    input_type: Option<&str>,
    output_type: Option<&str>,
    mut num_sources: usize,
    metafits: Option<&Path>,
    array_position: Option<LatLngHeight>,
    lst_rad: Option<f64>,
    phase_centre: Option<RADec>,
    freqs_hz: Option<&[f64]>,
    source_dist_cutoff: Option<f64>,
    veto_threshold: Option<f64>,
    filter_points: bool,
    filter_gaussians: bool,
    filter_shapelets: bool,
    collapse_into_single_source: bool,
    rts_base_source: Option<&str>,
    beam_args: BeamArgs,
) -> Result<(), SrclistByBeamError> {
    // Read the input source list.
    let input_type = input_type.and_then(|t| SourceListType::from_str(t).ok());
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

    // Handle the output path and type.
    let output_path = match output_path {
        Some(p) => p.to_path_buf(),
        None => {
            let input_path_base = input_path
                .file_stem()
                .and_then(|os_str| os_str.to_str())
                .expect("Input file didn't have a filename stem");
            let input_path_ext = input_path
                .extension()
                .and_then(|os_str| os_str.to_str())
                .expect("Input file didn't have an extension");
            let output_pb =
                PathBuf::from(format!("{input_path_base}_{num_sources}.{input_path_ext}"));
            debug!("Writing reduced source list to {}", output_pb.display());
            output_pb
        }
    };

    let metadata = if let Some(metafits) = metafits {
        // Open the metafits.
        trace!("Attempting to open the metafits file");
        let metafits = mwalib::MetafitsContext::new(metafits, None)?;

        let mut dipole_delays = Delays::Full(get_dipole_delays(&metafits));
        dipole_delays.set_to_ideal_delays();

        let mut metadata = Metadata {
            phase_centre: RADec::from_degrees(
                metafits
                    .ra_phase_center_degrees
                    .unwrap_or(metafits.ra_tile_pointing_degrees),
                metafits
                    .dec_phase_center_degrees
                    .unwrap_or(metafits.dec_tile_pointing_degrees),
            ),
            array_position: LatLngHeight::mwa(),
            lst_rad: metafits.lst_rad,
            freqs_hz: metafits
                .metafits_coarse_chans
                .iter()
                .map(|cc| cc.chan_centre_hz as _)
                .collect(),
            dipole_delays: Some(dipole_delays),
        };

        // Override metafits values with anything that was manually specified.
        if let Some(phase_centre) = phase_centre {
            metadata.phase_centre = phase_centre;
        }
        if let Some(array_position) = array_position {
            metadata.array_position = array_position;
        }
        if let Some(lst_rad) = lst_rad {
            metadata.lst_rad = lst_rad;
        }
        if let Some(freqs_hz) = freqs_hz {
            metadata.freqs_hz = freqs_hz.into();
        }

        metadata
    } else {
        Metadata {
            phase_centre: match phase_centre {
                Some(p) => p,
                None => return Err(SrclistByBeamError::NoPhaseCentre),
            },
            array_position: match array_position {
                Some(a) => a,
                None => LatLngHeight::mwa(),
            },
            lst_rad: match lst_rad {
                Some(l) => l,
                None => return Err(SrclistByBeamError::NoLst),
            },
            freqs_hz: match freqs_hz {
                Some(f) => f.into(),
                None => return Err(SrclistByBeamError::NoFreqs),
            },
            // If the user didn't specify delays on the command line, and delays
            // are needed to set up the beam object, an error will be generated
            // below.
            dipole_delays: None,
        }
    };

    debug!("Using {} as the phase centre", metadata.phase_centre);
    debug!("Using {}° as the LST", metadata.lst_rad.to_degrees());
    debug!(
        "Using coarse channel frequencies [MHz]: {}",
        metadata
            .freqs_hz
            .iter()
            .map(|freq_hz| format!("{:.2}", *freq_hz / 1e6))
            .join(", ")
    );

    // Set up the beam. We use the ideal delays for all tiles because we
    // don't want to use any dead dipoles.
    info!("");
    let beam = beam_args.parse(1, metadata.dipole_delays, None, None)?;

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
        metadata.phase_centre,
        metadata.lst_rad,
        metadata.array_position.latitude_rad,
        &metadata.freqs_hz,
        &*beam,
        None,
        source_dist_cutoff.unwrap_or(DEFAULT_CUTOFF_DISTANCE),
        veto_threshold.unwrap_or(DEFAULT_VETO_THRESHOLD),
    )?;
    // Were any sources left after vetoing?
    if sl.is_empty() {
        return Err(ReadSourceListError::NoSourcesAfterVeto.into());
    };

    // If requested, collapse the source list.
    sl = if collapse_into_single_source {
        let base = rts_base_source
            .unwrap_or(sl.get_index(0).unwrap().0)
            .to_owned();
        let mut collapsed = SourceList::new();
        let base = sl.remove_entry(&base).unwrap();
        let mut num_collapsed_components = base.1.components.len() - 1;
        collapsed.insert(base.0, base.1);
        let base_src = collapsed.get_index_mut(0).unwrap().1;
        let mut base_comps = vec![].into_boxed_slice();
        std::mem::swap(&mut base_src.components, &mut base_comps);
        let mut base_comps = base_comps.to_vec();
        sl.into_iter()
            .take(num_sources)
            .flat_map(|(_, src)| src.components.to_vec())
            .for_each(|comp| {
                num_collapsed_components += 1;
                base_comps.push(comp);
            });
        std::mem::swap(&mut base_src.components, &mut base_comps.into_boxed_slice());
        info!(
            "Collapsed {num_sources} into 1 base source with {num_collapsed_components} components"
        );
        num_sources = 1;
        collapsed
    } else {
        if rts_base_source.is_some() {
            "RTS base source was supplied, but we're not collapsing the source list into a single source.".warn();
        }
        sl
    };

    write_source_list(
        &sl,
        &output_path,
        sl_type,
        output_type.and_then(|s| SourceListType::from_str(s).ok()),
        Some(num_sources),
    )?;

    display_warnings();

    Ok(())
}

#[derive(thiserror::Error, Debug)]
pub(crate) enum SrclistByBeamError {
    #[error("No metafits was supplied and no phase centre was specified; cannot continue")]
    NoPhaseCentre,

    #[error("No metafits was supplied and no LST was specified; cannot continue")]
    NoLst,

    #[error("No metafits was supplied and no frequencies were specified; cannot continue")]
    NoFreqs,

    #[error(transparent)]
    ReadSourceList(#[from] ReadSourceListError),

    #[error(transparent)]
    WriteSourceList(#[from] WriteSourceListError),

    #[error(transparent)]
    Beam(#[from] crate::beam::BeamError),

    #[error(transparent)]
    Mwalib(#[from] mwalib::MwalibError),

    #[error(transparent)]
    IO(#[from] std::io::Error),
}
