// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Generate sky-model visibilities from a sky-model source list.

mod error;

pub(crate) use error::VisSimulateError;

use std::{collections::HashSet, path::PathBuf, str::FromStr, thread};

use clap::Parser;
use crossbeam_channel::{bounded, Sender};
use crossbeam_utils::atomic::AtomicCell;
use hifitime::{Duration, Epoch};
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use itertools::Itertools;
use log::{debug, info, warn};
use marlu::{
    constants::{FREQ_WEIGHT_FACTOR, TIME_WEIGHT_FACTOR},
    precession::precess_time,
    Jones, LatLngHeight, MwaObsContext, RADec, XyzGeodetic,
};
use mwalib::MetafitsContext;
use ndarray::ArcArray2;
use scopeguard::defer_on_unwind;
use vec1::Vec1;

use crate::{
    averaging::{parse_freq_average_factor, parse_time_average_factor, timesteps_to_timeblocks},
    beam::{create_fee_beam_object, create_no_beam_object, Beam, Delays},
    constants::{DEFAULT_CUTOFF_DISTANCE, DEFAULT_VETO_THRESHOLD},
    context::Polarisations,
    help_texts::{
        ARRAY_POSITION_HELP, DIPOLE_DELAYS_HELP, SOURCE_DIST_CUTOFF_HELP, VETO_THRESHOLD_HELP,
    },
    io::{
        get_single_match_from_glob,
        write::{can_write_to_file, write_vis, VisOutputType, VisTimestep, VIS_OUTPUT_EXTENSIONS},
    },
    math::TileBaselineFlags,
    messages,
    metafits::{get_dipole_delays, get_dipole_gains},
    model::{self, ModellerInfo, SkyModeller},
    srclist::{read::read_source_list_file, veto_sources, ComponentCounts, SourceList},
    HyperdriveError,
};

const DEFAULT_OUTPUT_VIS_FILENAME: &str = "hyp_model.uvfits";

lazy_static::lazy_static! {
    static ref OUTPUTS_HELP: String =
        format!("Paths to the output visibility files. Supported formats: {}. Default: {}", *VIS_OUTPUT_EXTENSIONS, DEFAULT_OUTPUT_VIS_FILENAME);
}

#[derive(Parser, Debug, Default)]
pub struct VisSimulateArgs {
    /// Path to the metafits file.
    #[clap(short, long, parse(from_str), help_heading = "INPUT AND OUTPUT")]
    metafits: PathBuf,

    #[clap(
        short = 'o',
        long,
        multiple_values(true),
        help = OUTPUTS_HELP.as_str(),
        help_heading = "INPUT AND OUTPUT"
    )]
    output_model_files: Vec<PathBuf>,

    /// Path to the sky-model source list used for simulation.
    #[clap(short, long, help_heading = "INPUT AND OUTPUT")]
    source_list: String,

    /// When writing out model visibilities, average this many timesteps
    /// together. Also supports a target time resolution (e.g. 8s). The value
    /// must be a multiple of the input data's time resolution. The default is
    /// to preserve the input data's time resolution. e.g. If the input data is
    /// in 0.5s resolution and this variable is 4, then we average 2s worth of
    /// model data together before writing the data out. If the variable is
    /// instead 4s, then 8 model timesteps are averaged together before writing
    /// the data out.
    #[clap(long, help_heading = "OUTPUT FILES")]
    output_model_time_average: Option<String>,

    /// When writing out model visibilities, average this many fine freq.
    /// channels together. Also supports a target freq. resolution (e.g. 80kHz).
    /// The value must be a multiple of the input data's freq. resolution. The
    /// default is to preserve the input data's freq. resolution multiplied by
    /// the frequency average factor. e.g. If the input data is in 40kHz
    /// resolution, the frequency average factor is 2 and this variable is 4,
    /// then we average 320kHz worth of model data together before writing the
    /// data out. If the variable is instead 80kHz, then 4 model fine freq.
    /// channels are averaged together before writing the data out.
    #[clap(long, help_heading = "OUTPUT FILES")]
    output_model_freq_average: Option<String>,

    /// The number of sources to use in the source list. The default is to use
    /// them all. Example: If 1000 sources are specified here, then the top 1000
    /// sources are used (based on their flux densities after the beam
    /// attenuation) within the specified source distance cutoff.
    #[clap(short, long, help_heading = "SKY-MODEL SOURCES")]
    num_sources: Option<usize>,

    #[clap(long, help = SOURCE_DIST_CUTOFF_HELP.as_str(), help_heading = "SKY-MODEL SOURCES")]
    source_dist_cutoff: Option<f64>,

    #[clap(long, help = VETO_THRESHOLD_HELP.as_str(), help_heading = "SKY-MODEL SOURCES")]
    veto_threshold: Option<f64>,

    /// Don't include sources containing point components in the input sky
    /// model.
    #[clap(long, help_heading = "SKY-MODEL SOURCES")]
    filter_points: bool,

    /// Don't include sources containing Gaussian components in the input sky
    /// model.
    #[clap(long, help_heading = "SKY-MODEL SOURCES")]
    filter_gaussians: bool,

    /// Don't include sources containing shapelet components in the input sky
    /// model.
    #[clap(long, help_heading = "SKY-MODEL SOURCES")]
    filter_shapelets: bool,

    /// The phase centre right ascension [degrees]. If this is not specified,
    /// then the metafits phase/pointing centre is used.
    #[clap(short, long, help_heading = "OBSERVATION PARAMETERS")]
    ra: Option<f64>,

    /// The phase centre declination [degrees]. If this is not specified, then
    /// the metafits phase/pointing centre is used.
    #[clap(short, long, help_heading = "OBSERVATION PARAMETERS")]
    dec: Option<f64>,

    /// The total number of fine channels in the observation.
    #[clap(
        short = 'c',
        long,
        default_value = "384",
        help_heading = "OBSERVATION PARAMETERS"
    )]
    num_fine_channels: usize,

    /// The fine-channel resolution [kHz].
    #[clap(
        short,
        long,
        default_value = "80",
        help_heading = "OBSERVATION PARAMETERS"
    )]
    freq_res: f64,

    /// The centroid frequency of the simulation [MHz]. If this is not
    /// specified, then the FREQCENT specified in the metafits is used.
    #[clap(long, help_heading = "OBSERVATION PARAMETERS")]
    middle_freq: Option<f64>,

    /// The number of time steps used from the metafits epoch.
    #[clap(
        short = 't',
        long,
        default_value = "14",
        help_heading = "OBSERVATION PARAMETERS"
    )]
    num_timesteps: usize,

    /// The time resolution [seconds].
    #[clap(long, default_value = "8", help_heading = "OBSERVATION PARAMETERS")]
    time_res: f64,

    /// The time offset from the start [seconds]. The default start time is the
    /// is the obsid as GPS timestamp.
    #[clap(long, default_value = "0", help_heading = "OBSERVATION PARAMETERS")]
    time_offset: f64,

    /// Should we use a beam? Default is to use the FEE beam.
    #[clap(long, help_heading = "MODEL PARAMETERS")]
    no_beam: bool,

    /// The path to the HDF5 MWA FEE beam file. If not specified, this must be
    /// provided by the MWA_BEAM_FILE environment variable.
    #[clap(long, help_heading = "MODEL PARAMETERS")]
    beam_file: Option<PathBuf>,

    /// Pretend that all MWA dipoles are alive and well, ignoring whatever is in
    /// the metafits file.
    #[clap(long, help_heading = "MODEL PARAMETERS")]
    unity_dipole_gains: bool,

    #[clap(long, multiple_values(true), help = DIPOLE_DELAYS_HELP.as_str(), help_heading = "MODEL PARAMETERS")]
    delays: Option<Vec<u32>>,

    #[clap(
        long, help = ARRAY_POSITION_HELP.as_str(), help_heading = "MODEL PARAMETERS",
        number_of_values = 3,
        allow_hyphen_values = true,
        value_names = &["LONG_DEG", "LAT_DEG", "HEIGHT_M"]
    )]
    array_position: Option<Vec<f64>>,

    /// Use a DUT1 value of 0 seconds rather than what is in the metafits file.
    #[clap(long, help_heading = "INPUT FILES")]
    ignore_dut1: bool,

    /// If specified, don't precess the array to J2000. We assume that sky-model
    /// sources are specified in the J2000 epoch.
    #[clap(long, help_heading = "MODEL PARAMETERS")]
    no_precession: bool,

    /// Use the CPU for visibility generation. This is deliberately made
    /// non-default because using a GPU is much faster.
    #[cfg(feature = "cuda")]
    #[clap(long, help_heading = "MODEL PARAMETERS")]
    cpu: bool,

    /// When generating sky-model visibilities, don't draw progress bars.
    #[clap(long, help_heading = "USER INTERFACE")]
    no_progress_bars: bool,
}

impl VisSimulateArgs {
    pub fn run(&self, dry_run: bool) -> Result<(), HyperdriveError> {
        vis_simulate(self, dry_run)?;
        Ok(())
    }
}

/// Parameters needed to do sky-model visibility simulation.
struct VisSimParams {
    /// Sky-model source list.
    source_list: SourceList,

    /// mwalib metafits context
    metafits: MetafitsContext,

    /// The output visibility files.
    outputs: Vec1<(PathBuf, VisOutputType)>,

    /// The number of model time samples to average together before writing out
    /// model visibilities.
    output_time_average_factor: usize,

    /// The number of model frequencies samples to average together before
    /// writing out model visibilities.
    output_freq_average_factor: usize,

    /// The phase centre.
    phase_centre: RADec,

    /// The fine channel frequencies \[Hz\].
    fine_chan_freqs: Vec1<f64>,

    /// The frequency resolution of the fine channels.
    freq_res_hz: f64,

    /// The [XyzGeodetic] positions of the tiles.
    tile_xyzs: Vec<XyzGeodetic>,

    /// The names of the tiles.
    tile_names: Vec<String>,

    /// Information on flagged tiles, baselines and mapping between indices.
    tile_baseline_flags: TileBaselineFlags,

    /// Timestamps to be simulated.
    timestamps: Vec1<Epoch>,

    time_res: Duration,

    /// Interface to beam code.
    beam: Box<dyn Beam>,

    /// The Earth position of the interferometer.
    array_position: LatLngHeight,

    /// UT1 - UTC.
    dut1: Duration,

    /// Should we be precessing?
    apply_precession: bool,
}

impl VisSimParams {
    /// Convert arguments into parameters.
    fn new(args: &VisSimulateArgs) -> Result<VisSimParams, VisSimulateError> {
        debug!("{:#?}", &args);

        // Expose all the struct fields to ensure they're all used.
        let VisSimulateArgs {
            metafits,
            output_model_files,
            source_list,
            output_model_time_average,
            output_model_freq_average,
            num_sources,
            source_dist_cutoff,
            veto_threshold,
            filter_points,
            filter_gaussians,
            filter_shapelets,
            ra,
            dec,
            num_fine_channels,
            freq_res,
            middle_freq,
            num_timesteps,
            time_res,
            time_offset,
            no_beam,
            beam_file,
            unity_dipole_gains,
            delays,
            array_position,
            ignore_dut1,
            no_precession,
            no_progress_bars: _,
            #[cfg(feature = "cuda")]
            cpu,
        } = args;

        // If we're going to use a GPU for modelling, get the device info so we
        // can ensure a CUDA-capable device is available, and so we can report
        // it to the user later.
        #[cfg(feature = "cuda")]
        let modeller_info = if *cpu {
            ModellerInfo::Cpu
        } else {
            let (device_info, driver_info) = crate::cuda::get_device_info()?;
            ModellerInfo::Cuda {
                device_info,
                driver_info,
            }
        };
        #[cfg(not(feature = "cuda"))]
        let modeller_info = ModellerInfo::Cpu;

        let output_model_files = {
            let mut valid_model_files = Vec::with_capacity(output_model_files.len().max(1));
            for file in output_model_files {
                // Is the output file type supported?
                let ext = file.extension().and_then(|os_str| os_str.to_str());
                match ext.and_then(|s| VisOutputType::from_str(s).ok()) {
                    Some(t) => {
                        can_write_to_file(file)?;
                        valid_model_files.push((file.to_owned(), t));
                    }
                    None => return Err(VisSimulateError::InvalidOutputFormat(file.clone())),
                }
            }
            if valid_model_files.is_empty() {
                valid_model_files.push((
                    PathBuf::from(DEFAULT_OUTPUT_VIS_FILENAME),
                    VisOutputType::Uvfits,
                ));
            }
            Vec1::try_from_vec(valid_model_files).unwrap()
        };

        // Read the metafits file with mwalib.
        // TODO: Allow the user to specify the mwa_version.
        let metafits = mwalib::MetafitsContext::new(metafits, None)?;

        // Get the phase centre.
        let phase_centre = match (ra, dec, &metafits) {
            (Some(ra), Some(dec), _) => {
                // Verify that the input coordinates are sensible.
                if !(0.0..=360.0).contains(ra) {
                    return Err(VisSimulateError::RaInvalid);
                }
                if !(-90.0..=90.0).contains(dec) {
                    return Err(VisSimulateError::DecInvalid);
                }
                RADec::from_degrees(*ra, *dec)
            }
            (Some(_), None, _) => return Err(VisSimulateError::OnlyOneRAOrDec),
            (None, Some(_), _) => return Err(VisSimulateError::OnlyOneRAOrDec),
            (None, None, m) => {
                // The phase centre in a metafits file may not be present. If not,
                // we have to use the pointing centre.
                match (m.ra_phase_center_degrees, m.dec_phase_center_degrees) {
                    (Some(ra), Some(dec)) => RADec::from_degrees(ra, dec),
                    (None, None) => {
                        RADec::from_degrees(m.ra_tile_pointing_degrees, m.dec_tile_pointing_degrees)
                    }
                    _ => unreachable!(),
                }
            }
        };

        // Get the fine channel frequencies.
        if *freq_res < f64::EPSILON {
            return Err(VisSimulateError::FineChansWidthTooSmall);
        }
        let middle_freq = middle_freq
            .map(|f| f * 1e6) // MHz -> Hz
            .unwrap_or(metafits.centre_freq_hz as _);
        let freq_res = freq_res * 1e3; // kHz -> Hz
        let fine_chan_freqs = {
            let half_num_fine_chans = *num_fine_channels as f64 / 2.0;
            let mut fine_chan_freqs = Vec::with_capacity(*num_fine_channels);
            for i in 0..*num_fine_channels {
                fine_chan_freqs
                    .push(middle_freq - half_num_fine_chans * freq_res + freq_res * i as f64);
            }
            Vec1::try_from_vec(fine_chan_freqs).map_err(|_| VisSimulateError::FineChansZero)?
        };

        // Populate the timestamps.
        let time_res = Duration::from_seconds(*time_res);
        let timestamps = {
            let mut timestamps = Vec::with_capacity(*num_timesteps);
            let start_ns = metafits
                .sched_start_gps_time_ms
                .checked_mul(1_000_000)
                .expect("does not overflow u64");
            let start = Epoch::from_gpst_nanoseconds(start_ns)
                + time_res / 2
                + Duration::from_seconds(*time_offset);
            for i in 0..*num_timesteps {
                timestamps.push(start + time_res * i as i64);
            }
            Vec1::try_from_vec(timestamps).map_err(|_| VisSimulateError::ZeroTimeSteps)?
        };

        let array_position = match array_position {
            None => LatLngHeight::mwa(),
            Some(pos) => {
                if pos.len() != 3 {
                    return Err(VisSimulateError::BadArrayPosition {
                        pos: pos.to_owned(),
                    });
                }
                LatLngHeight {
                    longitude_rad: pos[0].to_radians(),
                    latitude_rad: pos[1].to_radians(),
                    height_metres: pos[2],
                }
            }
        };

        // Get the geodetic XYZ coordinates of each of the MWA tiles.
        let tile_xyzs = XyzGeodetic::get_tiles(&metafits, array_position.latitude_rad);
        let tile_names: Vec<String> = metafits
            .antennas
            .iter()
            .map(|a| a.tile_name.clone())
            .collect();

        // Prepare a map between baselines and their constituent tiles.
        // TODO: Utilise tile flags.
        let flagged_tiles = HashSet::new();
        let tile_baseline_flags = TileBaselineFlags::new(metafits.num_ants, flagged_tiles);

        // Treat the specified source list as file path. Does it exist? Then use it.
        // Otherwise, treat the specified source list as a glob and attempt to find
        // a single file with it.
        let sl_pb = PathBuf::from(&source_list);
        let sl_pb = if sl_pb.exists() {
            sl_pb
        } else {
            get_single_match_from_glob(source_list)?
        };
        // Read the source list.
        // TODO: Allow the user to specify a source list type.
        let source_list = match read_source_list_file(sl_pb, None) {
            Ok((sl, sl_type)) => {
                debug!("Successfully parsed {}-style source list", sl_type);
                sl
            }
            Err(e) => return Err(VisSimulateError::from(e)),
        };
        let ComponentCounts {
            num_points,
            num_gaussians,
            num_shapelets,
            ..
        } = source_list.get_counts();
        debug!("Found {num_points} points, {num_gaussians} gaussians, {num_shapelets} shapelets");

        // Apply any filters.
        let mut source_list = if *filter_points || *filter_gaussians || *filter_shapelets {
            let sl = source_list.filter(*filter_points, *filter_gaussians, *filter_shapelets);
            let ComponentCounts {
                num_points,
                num_gaussians,
                num_shapelets,
                ..
            } = sl.get_counts();
            debug!(
                "After filtering, there are {num_points} points, {num_gaussians} gaussians, {num_shapelets} shapelets"
            );
            sl
        } else {
            source_list
        };

        let mut delays = match delays {
            Some(d) => {
                if d.len() != 16 || d.iter().any(|&v| v > 32) {
                    return Err(VisSimulateError::BadDelays);
                }
                Delays::Partial(d.to_owned())
            }
            None => Delays::Full(get_dipole_delays(&metafits)),
        };
        let ideal_delays = delays.get_ideal_delays();
        let beam = if *no_beam {
            create_no_beam_object(tile_xyzs.len())
        } else {
            let dipole_gains = if *unity_dipole_gains {
                // We are treating all dipoles as "alive". But, if any dipole
                // delays are 32, then the beam code will still ignore those
                // dipoles. So use ideal dipole delays for all tiles.

                // Warn the user if they wanted unity dipole gains but the ideal
                // dipole delays contain 32.
                if *unity_dipole_gains && ideal_delays.iter().any(|&v| v == 32) {
                    warn!(
                        "Some ideal dipole delays are 32; these dipoles will not have unity gains"
                    );
                }
                delays.set_to_ideal_delays();
                None
            } else {
                Some(get_dipole_gains(&metafits))
            };

            create_fee_beam_object(beam_file.as_ref(), metafits.num_ants, delays, dipole_gains)?
        };
        let beam_file = beam.get_beam_file();
        debug!("Beam file: {beam_file:?}");

        let dut1 = if *ignore_dut1 {
            None
        } else {
            metafits.dut1.map(Duration::from_seconds)
        };
        let precession_info = precess_time(
            array_position.longitude_rad,
            array_position.latitude_rad,
            phase_centre,
            *timestamps.first(),
            dut1.unwrap_or_else(|| Duration::from_seconds(0.0)),
        );
        let (lmst, latitude) = if *no_precession {
            (precession_info.lmst, array_position.latitude_rad)
        } else {
            (
                precession_info.lmst_j2000,
                precession_info.array_latitude_j2000,
            )
        };

        messages::ObservationDetails {
            dipole_delays: Some(ideal_delays),
            beam_file,
            num_tiles_with_dead_dipoles: if *unity_dipole_gains {
                None
            } else {
                Some(
                    get_dipole_gains(&metafits)
                        .outer_iter()
                        .filter(|tile_dipole_gains| {
                            tile_dipole_gains.iter().any(|g| g.abs() < f64::EPSILON)
                        })
                        .count(),
                )
            },
            phase_centre,
            pointing_centre: None,
            dut1,
            lmst: Some(precession_info.lmst),
            lmst_j2000: if *no_precession {
                None
            } else {
                Some(precession_info.lmst_j2000)
            },
            available_timesteps: None,
            unflagged_timesteps: None,
            using_timesteps: None,
            first_timestamp: Some(*timestamps.first()),
            last_timestamp: Some(*timestamps.last()),
            time_res: Some(time_res),
            total_num_channels: *num_fine_channels,
            num_unflagged_channels: None,
            flagged_chans_per_coarse_chan: None,
            first_freq_hz: Some(*fine_chan_freqs.first()),
            last_freq_hz: Some(*fine_chan_freqs.last()),
            first_unflagged_freq_hz: None,
            last_unflagged_freq_hz: None,
            freq_res_hz: Some(freq_res),
        }
        .print();

        let coarse_chan_freqs = {
            let (mut coarse_chan_freqs, mut coarse_chan_nums): (Vec<f64>, Vec<u32>) =
                fine_chan_freqs
                    .iter()
                    .map(|&f| {
                        // MWA coarse channel numbers are a multiple of 1.28 MHz.
                        // This might change with MWAX, but ignore that until it
                        // becomes an issue; vis-simulate is mostly useful for
                        // testing.
                        let cc_num = (f / 1.28e6).round();
                        (cc_num * 1.28e6, cc_num as u32)
                    })
                    .unzip();
            // Deduplicate. As `fine_chan_freqs` is always sorted, we don't need
            // to sort here.
            coarse_chan_freqs.dedup();
            coarse_chan_nums.dedup();
            debug!("MWA coarse channel numbers: {coarse_chan_nums:?}");
            // Convert the coarse channel numbers to a range starting from 1.
            coarse_chan_freqs
        };
        debug!(
            "Coarse channel centre frequencies [Hz]: {:?}",
            coarse_chan_freqs
        );

        // Parse and verify user input (specified resolutions must evenly divide
        // the input data's resolutions).
        let time_factor =
            parse_time_average_factor(Some(time_res), output_model_time_average.as_deref(), 1)?;
        let freq_factor =
            parse_freq_average_factor(Some(freq_res), output_model_freq_average.as_deref(), 1)?;

        messages::OutputFileDetails {
            output_solutions: &[],
            vis_type: "simulated",
            output_vis: Some(&output_model_files),
            input_vis_time_res: Some(time_res),
            input_vis_freq_res: Some(freq_res),
            output_vis_time_average_factor: time_factor,
            output_vis_freq_average_factor: freq_factor,
        }
        .print();

        veto_sources(
            &mut source_list,
            phase_centre,
            lmst,
            latitude,
            &coarse_chan_freqs,
            &*beam,
            *num_sources,
            source_dist_cutoff.unwrap_or(DEFAULT_CUTOFF_DISTANCE),
            veto_threshold.unwrap_or(DEFAULT_VETO_THRESHOLD),
        )?;
        if source_list.is_empty() {
            return Err(VisSimulateError::NoSourcesAfterVeto);
        }

        messages::SkyModelDetails {
            source_list: &source_list,
        }
        .print();

        messages::print_modeller_info(&modeller_info);

        Ok(VisSimParams {
            source_list,
            metafits,
            outputs: output_model_files,
            output_time_average_factor: time_factor,
            output_freq_average_factor: freq_factor,
            phase_centre,
            fine_chan_freqs,
            freq_res_hz: freq_res,
            tile_xyzs,
            tile_names,
            tile_baseline_flags,
            timestamps,
            time_res,
            beam,
            array_position,
            dut1: dut1.unwrap_or_else(|| Duration::from_seconds(0.0)),
            apply_precession: !no_precession,
        })
    }
}

/// Simulate sky-model visibilities from a sky-model source list.
fn vis_simulate(args: &VisSimulateArgs, dry_run: bool) -> Result<(), VisSimulateError> {
    let VisSimParams {
        source_list,
        metafits,
        outputs,
        output_time_average_factor,
        output_freq_average_factor,
        phase_centre,
        fine_chan_freqs,
        freq_res_hz,
        tile_xyzs,
        tile_names,
        tile_baseline_flags,
        timestamps,
        time_res,
        beam,
        array_position,
        dut1,
        apply_precession,
    } = VisSimParams::new(args)?;

    let timesteps = {
        let timesteps = (0..timestamps.len()).collect::<Vec<_>>();
        // unwrap is safe because `timestamps` is never empty.
        Vec1::try_from_vec(timesteps).unwrap()
    };
    let timeblocks = timesteps_to_timeblocks(&timestamps, output_time_average_factor, &timesteps);

    if dry_run {
        info!("Dry run -- exiting now.");
        return Ok(());
    }

    // Channel for writing simulated visibilities.
    let (tx_model, rx_model) = bounded(5);

    // Progress bar.
    let multi_progress = MultiProgress::with_draw_target(if args.no_progress_bars {
        ProgressDrawTarget::hidden()
    } else {
        ProgressDrawTarget::stdout()
    });
    let model_progress = multi_progress.add(
        ProgressBar::new(timestamps.len() as u64)
            .with_style(
                ProgressStyle::default_bar()
                    .template("{msg:17}: [{wide_bar:.blue}] {pos:2}/{len:2} timesteps ({elapsed_precise}<{eta_precise})").unwrap()
                    .progress_chars("=> "),
            )
            .with_position(0)
            .with_message("Sky modelling"),
    );
    let write_progress = multi_progress.add(
        ProgressBar::new(timeblocks.len() as _)
            .with_style(
                ProgressStyle::default_bar()
                    .template("{msg:17}: [{wide_bar:.blue}] {pos:2}/{len:2} timeblocks ({elapsed_precise}<{eta_precise})").unwrap()
                    .progress_chars("=> "),
            )
            .with_position(0)
            .with_message("Model writing"),
    );

    // Generate the visibilities and write them out asynchronously.
    let error = AtomicCell::new(false);
    let scoped_threads_result = thread::scope(|s| {
        // Modelling thread.
        let model_handle = s.spawn(|| {
            defer_on_unwind! { error.store(true); }
            model_progress.tick();

            // Create a "modeller" object.
            let mut modeller = model::new_sky_modeller(
                #[cfg(feature = "cuda")]
                args.cpu,
                &*beam,
                &source_list,
                Polarisations::XX_XY_YX_YY,
                &tile_xyzs,
                &fine_chan_freqs,
                &tile_baseline_flags.flagged_tiles,
                phase_centre,
                array_position.longitude_rad,
                array_position.latitude_rad,
                dut1,
                apply_precession,
            )?;

            let cross_vis_shape = (
                fine_chan_freqs.len(),
                tile_baseline_flags
                    .unflagged_cross_baseline_to_tile_map
                    .len(),
            );
            let weight_factor =
                (freq_res_hz / FREQ_WEIGHT_FACTOR) * (time_res.to_seconds() / TIME_WEIGHT_FACTOR);
            let result = model_thread(
                &mut *modeller,
                &timestamps,
                cross_vis_shape,
                weight_factor,
                tx_model,
                &error,
                model_progress,
            );
            if result.is_err() {
                error.store(true);
            }
            result
        });

        // Writing thread.
        let write_handle = s.spawn(|| {
            defer_on_unwind! { error.store(true); }
            write_progress.tick();

            // Form (sorted) unflagged baselines from our cross- and
            // auto-correlation baselines.
            let unflagged_baseline_tile_pairs = tile_baseline_flags
                .unflagged_cross_baseline_to_tile_map
                .values()
                .copied()
                .sorted()
                .collect::<Vec<_>>();

            let result = write_vis(
                &outputs,
                array_position,
                phase_centre,
                None,
                &tile_xyzs,
                &tile_names,
                Some(metafits.obs_id),
                &timestamps,
                &timesteps,
                &timeblocks,
                time_res,
                dut1,
                freq_res_hz,
                &fine_chan_freqs,
                &unflagged_baseline_tile_pairs,
                &HashSet::new(),
                output_time_average_factor,
                output_freq_average_factor,
                Some(&MwaObsContext::from_mwalib(&metafits)),
                rx_model,
                &error,
                Some(write_progress),
            );
            if result.is_err() {
                error.store(true);
            }
            result
        });

        // Join all thread handles. This propagates any errors and lets us know
        // if any threads panicked, if panics aren't aborting as per the
        // Cargo.toml. (It would be nice to capture the panic information, if
        // it's possible, but I don't know how, so panics are currently
        // aborting.)
        let result = model_handle.join().unwrap();
        result.and_then(|_| write_handle.join().unwrap().map_err(VisSimulateError::from))
    });

    // Propagate errors and print out the write message.
    let s = scoped_threads_result?;
    info!("{s}");

    Ok(())
}

fn model_thread(
    modeller: &dyn SkyModeller,
    timestamps: &[Epoch],
    vis_shape: (usize, usize),
    weight_factor: f64,
    tx: Sender<VisTimestep>,
    error: &AtomicCell<bool>,
    progress_bar: ProgressBar,
) -> Result<(), VisSimulateError> {
    for &timestamp in timestamps {
        let mut cross_data_fb: ArcArray2<Jones<f32>> = ArcArray2::zeros(vis_shape);

        modeller.model_timestep_with(timestamp, cross_data_fb.view_mut())?;

        // Should we continue?
        if error.load() {
            return Ok(());
        }

        match tx.send(VisTimestep {
            cross_data_fb,
            cross_weights_fb: ArcArray2::from_elem(vis_shape, weight_factor as f32),
            autos: None,
            timestamp,
        }) {
            Ok(()) => (),
            // If we can't send the message, it's because the channel
            // has been closed on the other side. That should only
            // happen because the writer has exited due to error; in
            // that case, just exit this thread.
            Err(_) => return Ok(()),
        }

        progress_bar.inc(1);
    }

    progress_bar.abandon_with_message("Finished generating sky model");
    Ok(())
}
