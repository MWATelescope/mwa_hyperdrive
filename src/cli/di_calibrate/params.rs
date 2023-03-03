// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Parameters required for DI calibration.

use std::{collections::HashSet, ops::Deref, path::PathBuf, str::FromStr};

use hifitime::Duration;
use indexmap::IndexMap;
use itertools::Itertools;
use log::{debug, log_enabled, trace, warn, Level::Debug};
use marlu::{
    pos::{precession::precess_time, xyz::xyzs_to_cross_uvws},
    Jones, LatLngHeight, XyzGeodetic,
};
use ndarray::prelude::*;
use rayon::prelude::*;
use vec1::Vec1;

use super::{DiCalArgs, DiCalArgsError};
use crate::{
    averaging::{
        channels_to_chanblocks, parse_freq_average_factor, parse_time_average_factor,
        timesteps_to_timeblocks, AverageFactorError, Fence, Timeblock,
    },
    beam::{create_fee_beam_object, create_no_beam_object, Beam, Delays},
    cli::peel::SourceIonoConsts,
    constants::{DEFAULT_CUTOFF_DISTANCE, DEFAULT_VETO_THRESHOLD},
    context::ObsContext,
    di_calibrate::{calibrate_timeblocks, get_cal_vis, CalVis},
    filenames::InputDataTypes,
    io::{
        get_single_match_from_glob,
        read::{
            MsReader, RawDataCorrections, RawDataReader, UvfitsReader, VisInputType, VisRead,
            VisReadError,
        },
        write::{can_write_to_file, VisOutputType},
    },
    math::TileBaselineFlags,
    messages,
    model::ModellerInfo,
    solutions::{CalSolutionType, CalibrationSolutions},
    srclist::{read::read_source_list_file, veto_sources, SourceList, SourceListType},
    unit_parsing::{parse_wavelength, WavelengthUnit},
};

/// Parameters needed to perform calibration.
pub(crate) struct DiCalParams {
    /// Interface to the MWA data, and metadata on the input data.
    pub(crate) input_data: Box<dyn VisRead>,

    /// If the input data is raw MWA data, these are the corrections being
    /// applied as the visibilities are read.
    // TODO: Populate these if reading from a MS or uvfits - this can inform us
    // what corrections were used when forming those visibilities.
    pub(crate) raw_data_corrections: Option<RawDataCorrections>,

    /// Beam object.
    pub(crate) beam: Box<dyn Beam>,

    /// The sky-model source list.
    pub(crate) source_list: SourceList,

    /// Ionospheric constants.
    pub(crate) iono_consts: IndexMap<String, SourceIonoConsts>,

    /// The minimum UVW cutoff used in calibration \[metres\].
    pub(crate) uvw_min: f64,

    /// The maximum UVW cutoff used in calibration \[metres\].
    pub(crate) uvw_max: f64,

    /// The centroid frequency of the observation used to convert UVW cutoffs
    /// specified in lambdas to metres \[Hz\].
    pub(crate) freq_centroid: f64,

    /// Multiplicative factors to apply to unflagged baselines. These are mostly
    /// all 1.0, but flagged baselines (perhaps due to a UVW cutoff) have values
    /// of 0.0.
    pub(crate) baseline_weights: Vec1<f64>,

    /// Blocks of timesteps used for calibration. Each timeblock contains
    /// indices of the input data to average together during calibration. Each
    /// timeblock may have a different number of timesteps; the number of blocks
    /// and their lengths depends on which input data timesteps are being used
    /// as well as the `time_average_factor` (i.e. the number of timesteps to
    /// average during calibration; by default we average all timesteps).
    ///
    /// Simple examples: If we are averaging all data over time to form
    /// calibration solutions, there will only be one timeblock, and that block
    /// will contain all input data timestep indices. On the other hand, if
    /// `time_average_factor` is 1, then there are as many timeblocks as there
    /// are timesteps, and each block contains 1 timestep index.
    ///
    /// A more complicated example: If we are using input data timesteps 10, 11,
    /// 12 and 15 with a `time_average_factor` of 4, then there will be 2
    /// timeblocks, even though there are only 4 timesteps. This is because
    /// timestep 10 and 15 can't occupy the same timeblock with the "length" is
    /// 4. So the first timeblock contains 10, 11 and 12, whereas the second
    /// contains only 15.
    pub(crate) timeblocks: Vec1<Timeblock>,

    /// The timestep indices into the input data to be used for calibration.
    pub(crate) timesteps: Vec1<usize>,

    /// The number of frequency samples to average together during calibration.
    ///
    /// e.g. If the input data is in 40kHz resolution and this variable was 2,
    /// then we average 80kHz worth of data together during calibration.
    pub(crate) freq_average_factor: usize,

    /// Spectral windows, or, groups of contiguous-bands of channels to be
    /// calibrated. Multiple [Fence]s can represent a "picket fence"
    /// observation. Each [Fence] is composed of chanblocks, and the unflagged
    /// chanblocks are calibrated. Each chanblock may represent of multiple
    /// channels, depending on `freq_average_factor`; when visibilities are read
    /// from `input_data`, the channels are averaged according to
    /// `freq_average_factor`. If no frequency channels are flagged, then these
    /// chanblocks will represent all frequency channels. However, it's likely
    /// at least some channels are flagged, so the `flagged_chanblock_indices`
    /// in every [Fence] may be needed.
    pub(crate) fences: Vec1<Fence>,

    /// The frequencies of each of the observation's unflagged fine channels
    /// \[Hz\].
    pub(crate) unflagged_fine_chan_freqs: Vec<f64>,

    /// The fine channels to be flagged across the entire observation. e.g. For
    /// a 40 kHz observation, there are 768 fine channels, and this could
    /// contain 0 and 767.
    pub(crate) flagged_fine_chans: HashSet<usize>,

    /// Information on flagged tiles, baselines and mapping between indices.
    pub(crate) tile_baseline_flags: TileBaselineFlags,

    /// The unflagged [XyzGeodetic] coordinates of each tile \[metres\]. This
    /// does not change over time; it is determined only by the telescope's tile
    /// layout.
    pub(crate) unflagged_tile_xyzs: Vec<XyzGeodetic>,

    /// The Earth position of the array. This is populated by user input or the input data.
    pub(crate) array_position: LatLngHeight,

    /// The UT1 - UTC offset. If this is 0, effectively UT1 == UTC, which is a
    /// wrong assumption by up to 0.9s. We assume the this value does not change
    /// over the timestamps used in this `DiCalParams`.
    ///
    /// Note that this need not be the same DUT1 in the input data's
    /// [`ObsContext`]; the user may choose to suppress that DUT1 or supply
    /// their own.
    pub(crate) dut1: Duration,

    /// Should the array be precessed back to J2000?
    pub(crate) apply_precession: bool,

    /// The maximum number of times to iterate when performing "MitchCal".
    pub(crate) max_iterations: u32,

    /// The threshold at which we stop convergence when performing "MitchCal".
    /// This is smaller than `min_threshold`.
    pub(crate) stop_threshold: f64,

    /// The minimum threshold to satisfy convergence when performing "MitchCal".
    /// Reaching this threshold counts as "converged", but it's not as good as
    /// the stop threshold. This is bigger than `stop_threshold`.
    pub(crate) min_threshold: f64,

    /// The paths to the files where the calibration solutions are written. The
    /// same solutions are written to each file here, but the format may be
    /// different (indicated by the file extension). Supported formats are
    /// detailed by [super::solutions::CalSolutionType].
    pub(crate) output_solutions_filenames: Vec<(CalSolutionType, PathBuf)>,

    /// The optional sky-model visibilities files. If specified, model
    /// visibilities will be written out before calibration.
    pub(crate) model_files: Option<Vec1<(PathBuf, VisOutputType)>>,

    /// The number of calibrated time samples to average together before writing
    /// out calibrated visibilities.
    pub(crate) output_model_time_average_factor: usize,

    /// The number of calibrated frequencies samples to average together before
    /// writing out calibrated visibilities.
    pub(crate) output_model_freq_average_factor: usize,

    /// When reading in visibilities and generating sky-model visibilities,
    /// don't draw progress bars.
    pub(crate) no_progress_bars: bool,

    /// Information on the sky-modelling device (CPU or CUDA-capable device).
    pub(crate) modeller_info: ModellerInfo,
}

impl DiCalParams {
    /// Create a new params struct from arguments.
    ///
    /// If the time or frequency resolution aren't specified, they default to
    /// the observation's native resolution.
    ///
    /// Source list vetoing is performed in this function, using the specified
    /// number of sources and/or the veto threshold.
    pub(crate) fn new(
        DiCalArgs {
            args_file: _,
            data,
            source_list,
            source_list_type,
            ms_data_column_name,
            iono_consts_file,
            ignore_dut1,
            outputs,
            model_filenames,
            output_model_time_average,
            output_model_freq_average,
            num_sources,
            source_dist_cutoff,
            veto_threshold,
            beam_file,
            unity_dipole_gains,
            delays,
            no_beam,
            timesteps_per_timeblock,
            freq_average_factor,
            timesteps,
            use_all_timesteps,
            uvw_min,
            uvw_max,
            max_iterations,
            stop_thresh,
            min_thresh,
            array_position,
            no_precession,
            #[cfg(feature = "cuda")]
            cpu,
            tile_flags,
            ignore_input_data_tile_flags,
            ignore_input_data_fine_channel_flags,
            fine_chan_flags_per_coarse_chan,
            fine_chan_flags,
            pfb_flavour,
            no_digital_gains,
            no_cable_length_correction,
            no_geometric_correction,
            no_progress_bars,
        }: DiCalArgs,
    ) -> Result<DiCalParams, DiCalArgsError> {
        // If we're going to use a GPU for modelling, get the device info so we
        // can ensure a CUDA-capable device is available, and so we can report
        // it to the user later.
        #[cfg(feature = "cuda")]
        let modeller_info = if cpu {
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

        // If the user supplied the array position, unpack it here.
        let array_position = match array_position {
            Some(pos) => {
                if pos.len() != 3 {
                    return Err(DiCalArgsError::BadArrayPosition { pos });
                }
                Some(LatLngHeight {
                    longitude_rad: pos[0].to_radians(),
                    latitude_rad: pos[1].to_radians(),
                    height_metres: pos[2],
                })
            }
            None => None,
        };

        // Handle input data. We expect one of three possibilities:
        // - gpubox files, a metafits file (and maybe mwaf files),
        // - a measurement set (and maybe a metafits file), or
        // - uvfits files.
        // If none or multiple of these possibilities are met, then we must fail.
        let input_data_types = match data {
            Some(strings) => InputDataTypes::new(&strings)?,
            None => return Err(DiCalArgsError::NoInputData),
        };
        let (input_data, raw_data_corrections): (Box<dyn VisRead>, Option<RawDataCorrections>) =
            match (
                input_data_types.metafits,
                input_data_types.gpuboxes,
                input_data_types.mwafs,
                input_data_types.ms,
                input_data_types.uvfits,
            ) {
                // Valid input for reading raw data.
                (Some(meta), Some(gpuboxes), mwafs, None, None) => {
                    // Ensure that there's only one metafits.
                    let meta = if meta.len() > 1 {
                        return Err(DiCalArgsError::MultipleMetafits(meta));
                    } else {
                        meta.first()
                    };

                    debug!("gpubox files: {:?}", &gpuboxes);
                    debug!("mwaf files: {:?}", &mwafs);

                    let corrections = RawDataCorrections::new(
                        pfb_flavour.as_deref(),
                        !no_digital_gains,
                        !no_cable_length_correction,
                        !no_geometric_correction,
                    )?;
                    let input_data =
                        RawDataReader::new(meta, &gpuboxes, mwafs.as_deref(), corrections)?;

                    messages::InputFileDetails::Raw {
                        obsid: input_data.mwalib_context.metafits_context.obs_id,
                        gpubox_count: gpuboxes.len(),
                        metafits_file_name: meta.display().to_string(),
                        mwaf: input_data.get_flags(),
                        raw_data_corrections: corrections,
                    }
                    .print("DI calibrating");

                    (Box::new(input_data), Some(corrections))
                }

                // Valid input for reading a measurement set.
                (meta, None, None, Some(ms), None) => {
                    // Only one MS is supported at the moment.
                    let ms: PathBuf = if ms.len() > 1 {
                        return Err(DiCalArgsError::MultipleMeasurementSets(ms));
                    } else {
                        ms.first().clone()
                    };

                    // Ensure that there's only one metafits.
                    let meta: Option<&PathBuf> = match meta.as_ref() {
                        None => None,
                        Some(m) => {
                            if m.len() > 1 {
                                return Err(DiCalArgsError::MultipleMetafits(m.clone()));
                            } else {
                                Some(m.first())
                            }
                        }
                    };

                    let input_data = MsReader::new(&ms, ms_data_column_name, meta, array_position)?;

                    messages::InputFileDetails::MeasurementSet {
                        obsid: input_data.get_obs_context().obsid,
                        file_name: ms.display().to_string(),
                        metafits_file_name: meta.map(|m| m.display().to_string()),
                    }
                    .print("DI calibrating");

                    (Box::new(input_data), None)
                }

                // Valid input for reading uvfits files.
                (meta, None, None, None, Some(uvfits)) => {
                    // Only one uvfits is supported at the moment.
                    let uvfits: PathBuf = if uvfits.len() > 1 {
                        return Err(DiCalArgsError::MultipleUvfits(uvfits));
                    } else {
                        uvfits.first().clone()
                    };

                    // Ensure that there's only one metafits.
                    let meta: Option<&PathBuf> = match meta.as_ref() {
                        None => None,
                        Some(m) => {
                            if m.len() > 1 {
                                return Err(DiCalArgsError::MultipleMetafits(m.clone()));
                            } else {
                                Some(m.first())
                            }
                        }
                    };

                    let input_data = UvfitsReader::new(&uvfits, meta, array_position)?;

                    messages::InputFileDetails::UvfitsFile {
                        obsid: input_data.get_obs_context().obsid,
                        file_name: uvfits.display().to_string(),
                        metafits_file_name: meta.map(|m| m.display().to_string()),
                    }
                    .print("DI calibrating");

                    (Box::new(input_data), None)
                }

                // The following matches are for invalid combinations of input
                // files. Make an error message for the user.
                (Some(_), _, None, None, None) => {
                    let msg = "Received only a metafits file; a uvfits file, a measurement set or gpubox files are required.";
                    return Err(DiCalArgsError::InvalidDataInput(msg));
                }
                (Some(_), _, Some(_), None, None) => {
                    let msg =
                        "Received only a metafits file and mwaf files; gpubox files are required.";
                    return Err(DiCalArgsError::InvalidDataInput(msg));
                }
                (None, Some(_), _, None, None) => {
                    let msg = "Received gpuboxes without a metafits file; this is not supported.";
                    return Err(DiCalArgsError::InvalidDataInput(msg));
                }
                (None, None, Some(_), None, None) => {
                    let msg = "Received mwaf files without gpuboxes and a metafits file; this is not supported.";
                    return Err(DiCalArgsError::InvalidDataInput(msg));
                }
                (_, Some(_), _, Some(_), None) => {
                    let msg = "Received gpuboxes and measurement set files; this is not supported.";
                    return Err(DiCalArgsError::InvalidDataInput(msg));
                }
                (_, Some(_), _, None, Some(_)) => {
                    let msg = "Received gpuboxes and uvfits files; this is not supported.";
                    return Err(DiCalArgsError::InvalidDataInput(msg));
                }
                (_, _, _, Some(_), Some(_)) => {
                    let msg = "Received uvfits and measurement set files; this is not supported.";
                    return Err(DiCalArgsError::InvalidDataInput(msg));
                }
                (_, _, Some(_), Some(_), _) => {
                    let msg = "Received mwafs and measurement set files; this is not supported.";
                    return Err(DiCalArgsError::InvalidDataInput(msg));
                }
                (_, _, Some(_), _, Some(_)) => {
                    let msg = "Received mwafs and uvfits files; this is not supported.";
                    return Err(DiCalArgsError::InvalidDataInput(msg));
                }
                (None, None, None, None, None) => return Err(DiCalArgsError::NoInputData),
            };

        let obs_context = input_data.get_obs_context();

        // If the array position wasn't user defined, try the input data.
        // Otherwise warn that we're assuming MWA.
        let array_position = array_position
            .or(obs_context.array_position)
            .unwrap_or_else(|| {
                trace!("The array position was not specified in the input data; assuming MWA");
                LatLngHeight::mwa()
            });
        let dut1 = if ignore_dut1 { None } else { obs_context.dut1 };

        let timesteps_to_use = {
            match (use_all_timesteps, timesteps) {
                (true, _) => obs_context.all_timesteps.clone(),
                (false, None) => Vec1::try_from_vec(obs_context.unflagged_timesteps.clone())
                    .map_err(|_| DiCalArgsError::NoTimesteps)?,
                (false, Some(mut ts)) => {
                    // Make sure there are no duplicates.
                    let timesteps_hashset: HashSet<&usize> = ts.iter().collect();
                    if timesteps_hashset.len() != ts.len() {
                        return Err(DiCalArgsError::DuplicateTimesteps);
                    }

                    // Ensure that all specified timesteps are actually available.
                    for t in &ts {
                        if !(0..obs_context.timestamps.len()).contains(t) {
                            return Err(DiCalArgsError::UnavailableTimestep {
                                got: *t,
                                last: obs_context.timestamps.len() - 1,
                            });
                        }
                    }

                    ts.sort_unstable();
                    Vec1::try_from_vec(ts).map_err(|_| DiCalArgsError::NoTimesteps)?
                }
            }
        };

        let precession_info = precess_time(
            array_position.longitude_rad,
            array_position.latitude_rad,
            obs_context.phase_centre,
            obs_context.timestamps[*timesteps_to_use.first()],
            dut1.unwrap_or_else(|| Duration::from_seconds(0.0)),
        );
        let (lmst, latitude) = if no_precession {
            (precession_info.lmst, array_position.latitude_rad)
        } else {
            (
                precession_info.lmst_j2000,
                precession_info.array_latitude_j2000,
            )
        };

        // The length of the tile XYZ collection is the total number of tiles in
        // the array, even if some tiles are flagged.
        let total_num_tiles = obs_context.get_total_num_tiles();

        // Assign the tile flags.
        let flagged_tiles =
            obs_context.get_tile_flags(ignore_input_data_tile_flags, tile_flags.as_deref())?;
        let num_unflagged_tiles = total_num_tiles - flagged_tiles.len();
        if log_enabled!(Debug) {
            obs_context.print_debug_tile_statuses();
        }
        if num_unflagged_tiles == 0 {
            return Err(DiCalArgsError::NoTiles);
        }
        messages::ArrayDetails {
            array_position: Some(array_position),
            array_latitude_j2000: if no_precession {
                None
            } else {
                Some(precession_info.array_latitude_j2000)
            },
            total_num_tiles,
            num_unflagged_tiles,
            flagged_tiles: &flagged_tiles
                .iter()
                .sorted()
                .map(|&i| (obs_context.tile_names[i].as_str(), i))
                .collect::<Vec<_>>(),
        }
        .print();

        let dipole_delays = match delays {
            // We have user-provided delays; check that they're are sensible,
            // regardless of whether we actually need them.
            Some(d) => {
                if d.len() != 16 || d.iter().any(|&v| v > 32) {
                    return Err(DiCalArgsError::BadDelays);
                }
                Some(Delays::Partial(d))
            }

            // No delays were provided; use whatever was in the input data.
            None => obs_context.dipole_delays.as_ref().cloned(),
        };

        let beam: Box<dyn Beam> = if no_beam {
            create_no_beam_object(total_num_tiles)
        } else {
            let mut dipole_delays = dipole_delays.ok_or(DiCalArgsError::NoDelays)?;
            let dipole_gains = if unity_dipole_gains {
                None
            } else {
                // If we don't have dipole gains from the input data, then
                // we issue a warning that we must assume no dead dipoles.
                if obs_context.dipole_gains.is_none() {
                    match input_data.get_input_data_type() {
                        VisInputType::MeasurementSet => {
                            warn!("Measurement sets cannot supply dead dipole information.");
                            warn!("Without a metafits file, we must assume all dipoles are alive.");
                            warn!("This will make beam Jones matrices inaccurate in sky-model generation.");
                        }
                        VisInputType::Uvfits => {
                            warn!("uvfits files cannot supply dead dipole information.");
                            warn!("Without a metafits file, we must assume all dipoles are alive.");
                            warn!("This will make beam Jones matrices inaccurate in sky-model generation.");
                        }
                        VisInputType::Raw => unreachable!(),
                    }
                }
                obs_context.dipole_gains.clone()
            };
            if dipole_gains.is_none() {
                // If we don't have dipole gains, we must assume all dipoles are
                // "alive". But, if any dipole delays are 32, then the beam code
                // will still ignore those dipoles. So use ideal dipole delays
                // for all tiles.
                let ideal_delays = dipole_delays.get_ideal_delays();

                // Warn the user if they wanted unity dipole gains but the ideal
                // dipole delays contain 32.
                if unity_dipole_gains && ideal_delays.iter().any(|&v| v == 32) {
                    warn!(
                        "Some ideal dipole delays are 32; these dipoles will not have unity gains"
                    );
                }
                dipole_delays.set_to_ideal_delays();
            }

            create_fee_beam_object(beam_file, total_num_tiles, dipole_delays, dipole_gains)?
        };
        let beam_file = beam.get_beam_file();
        debug!("Beam file: {beam_file:?}");

        // Set up frequency information. Determine all of the fine-channel flags.
        let mut flagged_fine_chans: HashSet<usize> = match fine_chan_flags {
            Some(flags) => flags.into_iter().collect(),
            None => HashSet::new(),
        };
        if !ignore_input_data_fine_channel_flags {
            for &f in &obs_context.flagged_fine_chans {
                flagged_fine_chans.insert(f);
            }
        }
        // Assign the per-coarse-channel fine-channel flags.
        let fine_chan_flags_per_coarse_chan: HashSet<usize> = {
            let mut out_flags = match fine_chan_flags_per_coarse_chan {
                Some(flags) => flags.into_iter().collect(),
                None => HashSet::new(),
            };
            if !ignore_input_data_fine_channel_flags {
                for &obs_fine_chan_flag in &obs_context.flagged_fine_chans_per_coarse_chan {
                    out_flags.insert(obs_fine_chan_flag);
                }
            }
            out_flags
        };
        if !ignore_input_data_fine_channel_flags {
            for (i, _) in obs_context.coarse_chan_nums.iter().enumerate() {
                for f in &fine_chan_flags_per_coarse_chan {
                    flagged_fine_chans.insert(f + obs_context.num_fine_chans_per_coarse_chan * i);
                }
            }
        }
        let mut unflagged_fine_chan_freqs = vec![];
        for (i_chan, &freq) in obs_context.fine_chan_freqs.iter().enumerate() {
            if !flagged_fine_chans.contains(&i_chan) {
                unflagged_fine_chan_freqs.push(freq as f64);
            }
        }
        if log_enabled!(Debug) {
            let unflagged_fine_chans: Vec<_> = (0..obs_context.fine_chan_freqs.len())
                .filter(|i_chan| !flagged_fine_chans.contains(i_chan))
                .collect();
            match unflagged_fine_chans.as_slice() {
                [] => (),
                [f] => debug!("Only unflagged fine-channel: {}", f),
                [f_0, .., f_n] => {
                    debug!("First unflagged fine-channel: {}", f_0);
                    debug!("Last unflagged fine-channel:  {}", f_n);
                }
            }

            let fine_chan_flags_vec = flagged_fine_chans.iter().sorted().collect::<Vec<_>>();
            debug!("Flagged fine-channels: {:?}", fine_chan_flags_vec);
        }
        if unflagged_fine_chan_freqs.is_empty() {
            return Err(DiCalArgsError::NoChannels);
        }

        messages::ObservationDetails {
            dipole_delays: beam.get_ideal_dipole_delays(),
            beam_file,
            num_tiles_with_dead_dipoles: if unity_dipole_gains {
                None
            } else {
                obs_context.dipole_gains.as_ref().map(|array| {
                    array
                        .outer_iter()
                        .filter(|tile_dipole_gains| {
                            tile_dipole_gains.iter().any(|g| g.abs() < f64::EPSILON)
                        })
                        .count()
                })
            },
            phase_centre: obs_context.phase_centre,
            pointing_centre: obs_context.pointing_centre,
            dut1,
            lmst: Some(precession_info.lmst),
            lmst_j2000: if no_precession {
                Some(precession_info.lmst_j2000)
            } else {
                None
            },
            available_timesteps: Some(&obs_context.all_timesteps),
            unflagged_timesteps: Some(&obs_context.unflagged_timesteps),
            using_timesteps: Some(&timesteps_to_use),
            first_timestamp: Some(obs_context.timestamps[*timesteps_to_use.first()]),
            last_timestamp: if timesteps_to_use.len() > 1 {
                Some(obs_context.timestamps[*timesteps_to_use.last()])
            } else {
                None
            },
            time_res: obs_context.time_res,
            total_num_channels: obs_context.fine_chan_freqs.len(),
            num_unflagged_channels: Some(unflagged_fine_chan_freqs.len()),
            flagged_chans_per_coarse_chan: Some(&obs_context.flagged_fine_chans_per_coarse_chan),
            first_freq_hz: Some(*obs_context.fine_chan_freqs.first() as f64),
            last_freq_hz: Some(*obs_context.fine_chan_freqs.last() as f64),
            first_unflagged_freq_hz: unflagged_fine_chan_freqs.first().copied(),
            last_unflagged_freq_hz: unflagged_fine_chan_freqs.last().copied(),
            freq_res_hz: obs_context.freq_res,
        }
        .print();

        // Validate calibration solution outputs.
        let output_solutions_filenames = {
            match outputs {
                // Defaults.
                None => {
                    let pb =
                        PathBuf::from(crate::cli::di_calibrate::DEFAULT_OUTPUT_SOLUTIONS_FILENAME);
                    let sol_type = pb
                        .extension()
                        .and_then(|os_str| os_str.to_str())
                        .and_then(|s| CalSolutionType::from_str(s).ok())
                        // Tests should pick up a bad default filename.
                        .expect("DEFAULT_OUTPUT_SOLUTIONS_FILENAME has an unhandled extension!");
                    vec![(sol_type, pb)]
                }
                Some(outputs) => {
                    let mut cal_sols = vec![];
                    for file in outputs {
                        // Is the output file type supported?
                        let ext = file.extension().and_then(|os_str| os_str.to_str());
                        match ext.and_then(|s| CalSolutionType::from_str(s).ok()) {
                            Some(sol_type) => {
                                trace!("{} is a solution output", file.display());
                                can_write_to_file(&file)?;
                                cal_sols.push((sol_type, file));
                            }
                            None => {
                                return Err(DiCalArgsError::CalibrationOutputFile {
                                    ext: ext.unwrap_or("<no extension>").to_string(),
                                })
                            }
                        }
                    }
                    cal_sols
                }
            }
        };
        if output_solutions_filenames.is_empty() {
            return Err(DiCalArgsError::NoOutput);
        }

        // Handle the output model files, if specified.
        let model_files = if let Some(model_files) = model_filenames {
            let mut valid_model_files = Vec::with_capacity(model_files.len());
            for file in model_files {
                // Is the output file type supported?
                let ext = file.extension().and_then(|os_str| os_str.to_str());
                match ext.and_then(|s| VisOutputType::from_str(s).ok()) {
                    Some(t) => {
                        can_write_to_file(&file)?;
                        valid_model_files.push((file, t));
                    }
                    None => {
                        return Err(DiCalArgsError::VisFileType {
                            ext: ext.unwrap_or("<no extension>").to_string(),
                        })
                    }
                }
            }
            Vec1::try_from_vec(valid_model_files).ok()
        } else {
            None
        };

        // Set up the timeblocks.
        let time_average_factor = parse_time_average_factor(
            obs_context.time_res,
            timesteps_per_timeblock.as_deref(),
            *timesteps_to_use.last() - *timesteps_to_use.first() + 1,
        )
        .map_err(|e| match e {
            AverageFactorError::Zero => DiCalArgsError::CalTimeFactorZero,
            AverageFactorError::NotInteger => DiCalArgsError::CalTimeFactorNotInteger,
            AverageFactorError::NotIntegerMultiple { out, inp } => {
                DiCalArgsError::CalTimeResNotMultiple { out, inp }
            }
            AverageFactorError::Parse(e) => DiCalArgsError::ParseCalTimeAverageFactor(e),
        })?;
        // Check that the factor is not too big.
        let time_average_factor = if time_average_factor > timesteps_to_use.len() {
            warn!(
                "Cannot average {} timesteps during calibration; only {} are being used. Capping.",
                time_average_factor,
                timesteps_to_use.len()
            );
            timesteps_to_use.len()
        } else {
            time_average_factor
        };

        let timeblocks = timesteps_to_timeblocks(
            &obs_context.timestamps,
            time_average_factor,
            &timesteps_to_use,
        );

        // Set up the chanblocks.
        let freq_average_factor =
            parse_freq_average_factor(obs_context.freq_res, freq_average_factor.as_deref(), 1)
                .map_err(|e| match e {
                    AverageFactorError::Zero => DiCalArgsError::CalFreqFactorZero,
                    AverageFactorError::NotInteger => DiCalArgsError::CalFreqFactorNotInteger,
                    AverageFactorError::NotIntegerMultiple { out, inp } => {
                        DiCalArgsError::CalFreqResNotMultiple { out, inp }
                    }
                    AverageFactorError::Parse(e) => DiCalArgsError::ParseCalFreqAverageFactor(e),
                })?;
        // Check that the factor is not too big.
        let freq_average_factor = if freq_average_factor > unflagged_fine_chan_freqs.len() {
            warn!(
                "Cannot average {} channels; only {} are being used. Capping.",
                freq_average_factor,
                unflagged_fine_chan_freqs.len()
            );
            unflagged_fine_chan_freqs.len()
        } else {
            freq_average_factor
        };

        let fences = channels_to_chanblocks(
            &obs_context.fine_chan_freqs,
            obs_context.freq_res,
            freq_average_factor,
            &flagged_fine_chans,
        );
        // There must be at least one chanblock for calibration.
        match fences.as_slice() {
            // No fences is the same as no chanblocks.
            [] => return Err(DiCalArgsError::NoChannels),
            [f] => {
                // Check that the chanblocks aren't all flagged.
                if f.chanblocks.is_empty() {
                    return Err(DiCalArgsError::NoChannels);
                }
            }
            [f, ..] => {
                // Check that the chanblocks aren't all flagged.
                if f.chanblocks.is_empty() {
                    return Err(DiCalArgsError::NoChannels);
                }
                // TODO: Allow picket fence.
                eprintln!("\"Picket fence\" data detected. hyperdrive does not support this right now -- exiting.");
                eprintln!("See for more info: https://MWATelescope.github.io/mwa_hyperdrive/defs/mwa/picket_fence.html");
                std::process::exit(1);
            }
        }
        let fences = Vec1::try_from_vec(fences).map_err(|_| DiCalArgsError::NoChannels)?;

        let tile_index_maps = TileBaselineFlags::new(total_num_tiles, flagged_tiles);
        let flagged_tiles = &tile_index_maps.flagged_tiles;

        let unflagged_tile_xyzs: Vec<XyzGeodetic> = obs_context
            .tile_xyzs
            .par_iter()
            .enumerate()
            .filter(|(tile_index, _)| !flagged_tiles.contains(tile_index))
            .map(|(_, xyz)| *xyz)
            .collect();

        // Set baseline weights from UVW cuts. Use a lambda from the centroid
        // frequency if UVW cutoffs are specified as wavelengths.
        let freq_centroid = obs_context
            .fine_chan_freqs
            .iter()
            .map(|&u| u as f64)
            .sum::<f64>()
            / obs_context.fine_chan_freqs.len() as f64;
        let lambda = marlu::constants::VEL_C / freq_centroid;
        let (uvw_min, uvw_min_metres) = {
            let (quantity, unit) = parse_wavelength(
                uvw_min
                    .as_deref()
                    .unwrap_or(crate::cli::di_calibrate::DEFAULT_UVW_MIN),
            )
            .map_err(DiCalArgsError::ParseUvwMin)?;
            match unit {
                WavelengthUnit::M => ((quantity, unit), quantity),
                WavelengthUnit::L => ((quantity, unit), quantity * lambda),
            }
        };
        let (uvw_max, uvw_max_metres) = match uvw_max {
            None => ((f64::INFINITY, WavelengthUnit::M), f64::INFINITY),
            Some(s) => {
                let (quantity, unit) = parse_wavelength(&s).map_err(DiCalArgsError::ParseUvwMax)?;
                match unit {
                    WavelengthUnit::M => ((quantity, unit), quantity),
                    WavelengthUnit::L => ((quantity, unit), quantity * lambda),
                }
            }
        };

        let (baseline_weights, num_flagged_baselines) = {
            let mut baseline_weights = Vec1::try_from_vec(vec![
                1.0;
                tile_index_maps
                    .unflagged_cross_baseline_to_tile_map
                    .len()
            ])
            .map_err(|_| DiCalArgsError::NoTiles)?;
            let uvws = xyzs_to_cross_uvws(
                &unflagged_tile_xyzs,
                obs_context.phase_centre.to_hadec(lmst),
            );
            assert_eq!(baseline_weights.len(), uvws.len());
            let uvw_min = uvw_min_metres.powi(2);
            let uvw_max = uvw_max_metres.powi(2);
            let mut num_flagged_baselines = 0;
            for (uvw, baseline_weight) in uvws.into_iter().zip(baseline_weights.iter_mut()) {
                let uvw_length = uvw.u.powi(2) + uvw.v.powi(2) + uvw.w.powi(2);
                if uvw_length < uvw_min || uvw_length > uvw_max {
                    *baseline_weight = 0.0;
                    num_flagged_baselines += 1;
                }
            }
            (baseline_weights, num_flagged_baselines)
        };
        if num_flagged_baselines == baseline_weights.len() {
            return Err(DiCalArgsError::AllBaselinesFlaggedFromUvwCutoffs);
        }

        // Make sure the calibration thresholds are sensible.
        let mut stop_threshold =
            stop_thresh.unwrap_or(crate::cli::di_calibrate::DEFAULT_STOP_THRESHOLD);
        let min_threshold = min_thresh.unwrap_or(crate::cli::di_calibrate::DEFAULT_MIN_THRESHOLD);
        if stop_threshold > min_threshold {
            warn!("Specified stop threshold ({}) is bigger than the min. threshold ({}); capping stop threshold.", stop_threshold, min_threshold);
            stop_threshold = min_threshold;
        }
        let max_iterations =
            max_iterations.unwrap_or(crate::cli::di_calibrate::DEFAULT_MAX_ITERATIONS);

        messages::CalibrationDetails {
            timesteps_per_timeblock: time_average_factor,
            channels_per_chanblock: freq_average_factor,
            num_timeblocks: timeblocks.len(),
            num_chanblocks: fences.first().chanblocks.len(),
            uvw_min,
            uvw_max,
            num_calibration_baselines: baseline_weights.len() - num_flagged_baselines,
            total_num_baselines: baseline_weights.len(),
            lambda,
            freq_centroid,
            min_threshold,
            stop_threshold,
            max_iterations,
        }
        .print();

        let mut source_list: SourceList = {
            // Handle the source list argument.
            let sl_pb: PathBuf = match source_list {
                None => return Err(DiCalArgsError::NoSourceList),
                Some(sl) => {
                    // If the specified source list file can't be found, treat
                    // it as a glob and expand it to find a match.
                    let pb = PathBuf::from(&sl);
                    if pb.exists() {
                        pb
                    } else {
                        get_single_match_from_glob(&sl)?
                    }
                }
            };

            // Read the source list file. If the type was manually specified,
            // use that, otherwise the reading code will try all available
            // kinds.
            let sl_type_specified = source_list_type.is_none();
            let sl_type = source_list_type.and_then(|t| SourceListType::from_str(t.as_ref()).ok());
            let (sl, sl_type) = match read_source_list_file(sl_pb, sl_type) {
                Ok((sl, sl_type)) => (sl, sl_type),
                Err(e) => return Err(DiCalArgsError::from(e)),
            };

            // If the user didn't specify the source list type, then print out
            // what we found.
            if sl_type_specified {
                trace!("Successfully parsed {}-style source list", sl_type);
            }

            sl
        };
        trace!("Found {} sources in the source list", source_list.len());
        // Veto any sources that may be troublesome, and/or cap the total number
        // of sources. If the user doesn't specify how many source-list sources
        // to use, then all sources are used.
        if num_sources == Some(0) || source_list.is_empty() {
            return Err(DiCalArgsError::NoSources);
        }
        veto_sources(
            &mut source_list,
            obs_context.phase_centre,
            lmst,
            latitude,
            &obs_context.coarse_chan_freqs,
            beam.deref(),
            num_sources,
            source_dist_cutoff.unwrap_or(DEFAULT_CUTOFF_DISTANCE),
            veto_threshold.unwrap_or(DEFAULT_VETO_THRESHOLD),
        )?;
        if source_list.is_empty() {
            return Err(DiCalArgsError::NoSourcesAfterVeto);
        }

        messages::SkyModelDetails {
            source_list: &source_list,
        }
        .print();

        messages::print_modeller_info(&modeller_info);

        // Handle output visibility arguments.
        let (output_model_time_average_factor, output_model_freq_average_factor) = match model_files
            .as_ref()
        {
            None => {
                // If we're not writing out model visibilities but arguments
                // are set for them, issue warnings.
                match (output_model_time_average, output_model_freq_average) {
                    (None, None) => (),
                    (time, freq) => {
                        warn!("Not writing out model visibilities, but");
                        if time.is_some() {
                            warn!("  output_model_time_average is set");
                        }
                        if freq.is_some() {
                            warn!("  output_model_freq_average is set");
                        }
                    }
                }
                // We're not writing a file; it doesn't matter what these values
                // are.
                (1, 1)
            }
            Some(ms) => {
                // Parse and verify user input (specified resolutions must
                // evenly divide the input data's resolutions).
                let time_factor = parse_time_average_factor(
                    obs_context.time_res,
                    output_model_time_average.as_deref(),
                    1,
                )
                .map_err(|e| match e {
                    AverageFactorError::Zero => DiCalArgsError::OutputVisTimeAverageFactorZero,
                    AverageFactorError::NotInteger => DiCalArgsError::OutputVisTimeFactorNotInteger,
                    AverageFactorError::NotIntegerMultiple { out, inp } => {
                        DiCalArgsError::OutputVisTimeResNotMultiple { out, inp }
                    }
                    AverageFactorError::Parse(e) => {
                        DiCalArgsError::ParseOutputVisTimeAverageFactor(e)
                    }
                })?;
                let freq_factor = parse_freq_average_factor(
                    obs_context.freq_res.map(|f| f * freq_average_factor as f64),
                    output_model_freq_average.as_deref(),
                    1,
                )
                .map_err(|e| match e {
                    AverageFactorError::Zero => DiCalArgsError::OutputVisFreqAverageFactorZero,
                    AverageFactorError::NotInteger => DiCalArgsError::OutputVisFreqFactorNotInteger,
                    AverageFactorError::NotIntegerMultiple { out, inp } => {
                        DiCalArgsError::OutputVisFreqResNotMultiple { out, inp }
                    }
                    AverageFactorError::Parse(e) => {
                        DiCalArgsError::ParseOutputVisFreqAverageFactor(e)
                    }
                })?;

                // Test that we can write to the output files.
                for m in ms {
                    can_write_to_file(&m.0)?;
                }

                (time_factor, freq_factor)
            }
        };
        {
            messages::OutputFileDetails {
                output_solutions: &output_solutions_filenames
                    .iter()
                    .map(|p| p.1.clone())
                    .collect::<Vec<_>>(),
                vis_type: "model",
                output_vis: model_files.as_ref(),
                input_vis_time_res: obs_context.time_res,
                input_vis_freq_res: obs_context.freq_res,
                output_vis_time_average_factor: output_model_time_average_factor,
                output_vis_freq_average_factor: output_model_freq_average_factor,
            }
            .print();
        }

        Ok(DiCalParams {
            input_data,
            raw_data_corrections,
            beam,
            source_list,
            iono_consts: {
                match iono_consts_file {
                    Some(f) => {
                        let mut f = std::io::BufReader::new(std::fs::File::open(f)?);
                        serde_json::from_reader(&mut f).unwrap()
                    }
                    None => IndexMap::new(),
                }
            },
            uvw_min: uvw_min_metres,
            uvw_max: uvw_max_metres,
            freq_centroid,
            baseline_weights,
            timeblocks,
            timesteps: timesteps_to_use,
            freq_average_factor,
            fences,
            unflagged_fine_chan_freqs,
            flagged_fine_chans,
            tile_baseline_flags: tile_index_maps,
            unflagged_tile_xyzs,
            array_position,
            dut1: dut1.unwrap_or_else(|| Duration::from_seconds(0.0)),
            apply_precession: !no_precession,
            max_iterations,
            stop_threshold,
            min_threshold,
            output_solutions_filenames,
            model_files,
            output_model_time_average_factor,
            output_model_freq_average_factor,
            no_progress_bars,
            modeller_info,
        })
    }

    /// Get read-only access to the [ObsContext]. This reflects the state of the
    /// observation in the data.
    pub(crate) fn get_obs_context(&self) -> &ObsContext {
        self.input_data.get_obs_context()
    }

    /// Get the total number of tiles in the observation, i.e. flagged and
    /// unflagged.
    pub(crate) fn get_total_num_tiles(&self) -> usize {
        self.get_obs_context().get_total_num_tiles()
    }

    /// Get the number of unflagged tiles to be used (may not match what is in
    /// the observation data).
    pub(crate) fn get_num_unflagged_tiles(&self) -> usize {
        self.get_total_num_tiles() - self.tile_baseline_flags.flagged_tiles.len()
    }

    /// The number of calibration timesteps.
    pub(crate) fn get_num_timesteps(&self) -> usize {
        self.timeblocks
            .iter()
            .fold(0, |acc, tb| acc + tb.range.len())
    }

    /// The number of unflagged cross-correlation baselines.
    pub(crate) fn get_num_unflagged_cross_baselines(&self) -> usize {
        let n = self.unflagged_tile_xyzs.len();
        (n * (n - 1)) / 2
    }

    pub(crate) fn read_crosses(
        &self,
        vis: ArrayViewMut2<Jones<f32>>,
        weights: ArrayViewMut2<f32>,
        timestep: usize,
    ) -> Result<(), VisReadError> {
        self.input_data.read_crosses(
            vis,
            weights,
            timestep,
            &self.tile_baseline_flags,
            &self.flagged_fine_chans,
        )
    }

    /// Use the [`DiCalParams`] to perform calibration and obtain solutions.
    pub(crate) fn calibrate(&self) -> Result<CalibrationSolutions, super::DiCalibrateError> {
        // TODO: Fix.
        if self.freq_average_factor > 1 {
            panic!("Frequency averaging isn't working right now. Sorry!");
        }

        let mut cal_vis = get_cal_vis(self, !self.no_progress_bars)?;
        cal_vis.scale_by_weights(Some(&self.baseline_weights));
        let CalVis {
            vis_data,
            vis_weights,
            vis_model,
        } = cal_vis;
        assert_eq!(vis_weights.len_of(Axis(2)), self.baseline_weights.len());

        // The shape of the array containing output Jones matrices.
        let num_timeblocks = self.timeblocks.len();
        let num_chanblocks = self.fences.first().chanblocks.len();
        let num_unflagged_tiles = self.get_num_unflagged_tiles();

        if log_enabled!(Debug) {
            let shape = (num_timeblocks, num_unflagged_tiles, num_chanblocks);
            debug!(
            "Shape of DI Jones matrices array: ({} timeblocks, {} tiles, {} chanblocks; {} MiB)",
            shape.0,
            shape.1,
            shape.2,
            shape.0 * shape.1 * shape.2 * std::mem::size_of::<Jones<f64>>()
            // 1024 * 1024 == 1 MiB.
            / 1024 / 1024
        );
        }

        let (sols, results) = calibrate_timeblocks(
            vis_data.view(),
            vis_model.view(),
            &self.timeblocks,
            // TODO: Picket fences.
            &self.fences.first().chanblocks,
            self.max_iterations,
            self.stop_threshold,
            self.min_threshold,
            !self.no_progress_bars,
            true,
        );

        // "Complete" the solutions.
        let sols = sols.into_cal_sols(self, Some(results.map(|r| r.max_precision)));

        Ok(sols)
    }
}
