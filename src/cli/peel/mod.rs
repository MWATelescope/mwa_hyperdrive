// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::{
    collections::HashSet,
    f64::consts::TAU,
    ops::{Deref, DerefMut, Sub},
    path::PathBuf,
    str::FromStr,
};

use clap::Parser;
use crossbeam_channel::bounded;
use crossbeam_utils::atomic::AtomicCell;
use hifitime::{Duration, Epoch};
use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;
use log::info;
use marlu::{
    constants::{FREQ_WEIGHT_FACTOR, TIME_WEIGHT_FACTOR, VEL_C},
    math::num_tiles_from_num_cross_correlation_baselines,
    precession::precess_time,
    HADec, Jones, MwaObsContext, RADec, XyzGeodetic,
};
use ndarray::{prelude::*, Zip};
use num_complex::Complex;
use rayon::prelude::*;
use scopeguard::defer_on_unwind;
use thiserror::Error;
use vec1::Vec1;

use crate::{
    averaging::{channels_to_chanblocks, timesteps_to_timeblocks},
    di_calibrate::{get_cal_vis, CalVis},
    help_texts::*,
    math::average_epoch,
    model::{new_sky_modeller, ModellerInfo, SkyModeller},
    srclist::{IonoSource, IonoSourceList, SourceList},
    vis_io::write::{write_vis, VisOutputType, VisTimestep, VIS_OUTPUT_EXTENSIONS},
    HyperdriveError,
};

lazy_static::lazy_static! {
    static ref DUT1: Duration = Duration::from_seconds(0.0);

    static ref VIS_OUTPUTS_HELP: String = format!("The paths to the files where the peeled visibilities are written. Supported formats: {}", *VIS_OUTPUT_EXTENSIONS);
}

// Arguments that are exposed to users. All arguments except bools should be
// optional.
#[derive(Parser, Debug, Clone, Default)]
pub struct PeelArgs {
    /// Paths to input data files to be calibrated. These can include a metafits
    /// file, gpubox files, mwaf files, a measurement set and/or uvfits files.
    #[clap(short, long, multiple_values(true), help_heading = "INPUT FILES")]
    pub data: Option<Vec<String>>,

    /// Path to the sky-model source list file.
    #[clap(short, long, help_heading = "INPUT FILES")]
    pub source_list: Option<String>,

    #[clap(long, help = SOURCE_LIST_TYPE_HELP.as_str(), help_heading = "INPUT FILES")]
    pub source_list_type: Option<String>,

    /// Use a DUT1 value of 0 seconds rather than what is in the input data.
    #[clap(long, help_heading = "INPUT FILES")]
    pub ignore_dut1: bool,

    #[clap(short, long, multiple_values(true), help = VIS_OUTPUTS_HELP.as_str(), help_heading = "OUTPUT FILES")]
    pub outputs: Option<Vec<PathBuf>>,

    // #[clap(short, long, multiple_values(true), help = MODEL_FILENAME_HELP.as_str(), help_heading = "OUTPUT FILES")]
    // pub model_filenames: Option<Vec<PathBuf>>,
    /// When writing out model visibilities, average this many timesteps
    /// together. Also supports a target time resolution (e.g. 8s). The value
    /// must be a multiple of the input data's time resolution. The default is
    /// to preserve the input data's time resolution. e.g. If the input data is
    /// in 0.5s resolution and this variable is 4, then we average 2s worth of
    /// model data together before writing the data out. If the variable is
    /// instead 4s, then 8 model timesteps are averaged together before writing
    /// the data out.
    #[clap(long, help_heading = "OUTPUT FILES")]
    pub output_model_time_average: Option<String>,

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
    pub output_model_freq_average: Option<String>,

    /// The number of sources to use in the source list. The default is to use
    /// them all. Example: If 1000 sources are specified here, then the top 1000
    /// sources are used (based on their flux densities after the beam
    /// attenuation) within the specified source distance cutoff.
    #[clap(short, long, help_heading = "SKY-MODEL SOURCES", default_value = "100")]
    // pub num_sources: Option<usize>,
    pub num_sources: usize,

    #[clap(long, help = SOURCE_DIST_CUTOFF_HELP.as_str(), help_heading = "SKY-MODEL SOURCES")]
    pub source_dist_cutoff: Option<f64>,

    #[clap(long, help = VETO_THRESHOLD_HELP.as_str(), help_heading = "SKY-MODEL SOURCES")]
    pub veto_threshold: Option<f64>,

    /// The path to the HDF5 MWA FEE beam file. If not specified, this must be
    /// provided by the MWA_BEAM_FILE environment variable.
    #[clap(long, help_heading = "BEAM")]
    pub beam_file: Option<PathBuf>,

    /// Pretend that all MWA dipoles are alive and well, ignoring whatever is in
    /// the metafits file.
    #[clap(long, help_heading = "BEAM")]
    pub unity_dipole_gains: bool,

    #[clap(long, multiple_values(true), help = DIPOLE_DELAYS_HELP.as_str(), help_heading = "BEAM")]
    pub delays: Option<Vec<u32>>,

    /// Don't apply a beam response when generating a sky model. The default is
    /// to use the FEE beam.
    #[clap(long, help_heading = "BEAM")]
    pub no_beam: bool,

    /// The number of timesteps to average together during calibration. Also
    /// supports a target time resolution (e.g. 8s). If this is 0, then all data
    /// are averaged together. Default: 0. e.g. If this variable is 4, then we
    /// produce calibration solutions in timeblocks with up to 4 timesteps each.
    /// If the variable is instead 4s, then each timeblock contains up to 4s
    /// worth of data.
    #[clap(short, long, help_heading = "CALIBRATION")]
    pub timesteps_per_timeblock: Option<String>,

    /// The number of fine-frequency channels to average together before
    /// calibration. If this is 0, then all data is averaged together. Default:
    /// 1. e.g. If the input data is in 20kHz resolution and this variable was
    /// 2, then we average 40kHz worth of data into a chanblock before
    /// calibration. If the variable is instead 40kHz, then each chanblock
    /// contains up to 40kHz worth of data.
    #[clap(short, long, help_heading = "CALIBRATION")]
    pub freq_average_factor: Option<String>,

    /// The timesteps to use from the input data. The timesteps will be
    /// ascendingly sorted for calibration. No duplicates are allowed. The
    /// default is to use all unflagged timesteps.
    #[clap(long, multiple_values(true), help_heading = "CALIBRATION")]
    pub timesteps: Option<Vec<usize>>,

    /// Use all timesteps in the data, including flagged ones. The default is to
    /// use all unflagged timesteps.
    #[clap(long, conflicts_with("timesteps"), help_heading = "CALIBRATION")]
    pub use_all_timesteps: bool,

    // #[clap(long, help = UVW_MIN_HELP.as_str(), help_heading = "CALIBRATION")]
    // pub uvw_min: Option<String>,

    // #[clap(long, help = UVW_MAX_HELP.as_str(), help_heading = "CALIBRATION")]
    // pub uvw_max: Option<String>,

    // #[clap(long, help = MAX_ITERATIONS_HELP.as_str(), help_heading = "CALIBRATION")]
    // pub max_iterations: Option<u32>,

    // #[clap(long, help = STOP_THRESHOLD_HELP.as_str(), help_heading = "CALIBRATION")]
    // pub stop_thresh: Option<f64>,

    // #[clap(long, help = MIN_THRESHOLD_HELP.as_str(), help_heading = "CALIBRATION")]
    // pub min_thresh: Option<f64>,
    #[clap(
        long, help = ARRAY_POSITION_HELP.as_str(), help_heading = "CALIBRATION",
        number_of_values = 3,
        allow_hyphen_values = true,
        value_names = &["LONG_RAD", "LAT_RAD", "HEIGHT_M"]
    )]
    pub array_position: Option<Vec<f64>>,

    /// If specified, don't precess the array to J2000. We assume that sky-model
    /// sources are specified in the J2000 epoch.
    #[clap(long, help_heading = "CALIBRATION")]
    pub no_precession: bool,

    #[cfg(feature = "cuda")]
    /// Use the CPU for visibility generation. This is deliberately made
    /// non-default because using a GPU is much faster.
    #[clap(long, help_heading = "CALIBRATION")]
    pub cpu: bool,

    /// Additional tiles to be flagged. These values correspond to either the
    /// values in the "Antenna" column of HDU 2 in the metafits file (e.g. 0 3
    /// 127), or the "TileName" (e.g. Tile011).
    #[clap(long, multiple_values(true), help_heading = "FLAGGING")]
    pub tile_flags: Option<Vec<String>>,

    /// If specified, pretend that all tiles are unflagged in the input data.
    #[clap(long, help_heading = "FLAGGING")]
    pub ignore_input_data_tile_flags: bool,

    /// If specified, pretend all fine channels in the input data are unflagged.
    #[clap(long, help_heading = "FLAGGING")]
    pub ignore_input_data_fine_channel_flags: bool,

    /// The fine channels to be flagged in each coarse channel. e.g. 0 1 16 30
    /// 31 are typical for 40 kHz data. If this is not specified, it defaults to
    /// flagging 80 kHz (or as close to this as possible) at the edges, as well
    /// as the centre channel for non-MWAX data.
    #[clap(long, multiple_values(true), help_heading = "FLAGGING")]
    pub fine_chan_flags_per_coarse_chan: Option<Vec<usize>>,

    /// The fine channels to be flagged across the whole observation band. e.g.
    /// 0 767 are the first and last fine channels for 40 kHz data.
    #[clap(long, multiple_values(true), help_heading = "FLAGGING")]
    pub fine_chan_flags: Option<Vec<usize>>,

    #[clap(long, help = PFB_FLAVOUR_HELP.as_str(), help_heading = "RAW MWA DATA")]
    pub pfb_flavour: Option<String>,

    /// When reading in raw MWA data, don't apply digital gains.
    #[clap(long, help_heading = "RAW MWA DATA")]
    pub no_digital_gains: bool,

    /// When reading in raw MWA data, don't apply cable length corrections. Note
    /// that some data may have already had the correction applied before it was
    /// written.
    #[clap(long, help_heading = "RAW MWA DATA")]
    pub no_cable_length_correction: bool,

    /// When reading in raw MWA data, don't apply geometric corrections. Note
    /// that some data may have already had the correction applied before it was
    /// written.
    #[clap(long, help_heading = "RAW MWA DATA")]
    pub no_geometric_correction: bool,

    /// When reading in visibilities and generating sky-model visibilities,
    /// don't draw progress bars.
    #[clap(long, help_heading = "USER INTERFACE")]
    pub no_progress_bars: bool,
}

impl PeelArgs {
    pub fn run(self, dry_run: bool) -> Result<(), HyperdriveError> {
        self.run_inner(dry_run)?;
        Ok(())
    }

    fn run_inner(self, dry_run: bool) -> Result<(), PeelError> {
        // TODO: Dirty hack
        let mut di_cal_args = crate::DiCalArgs {
            args_file: None,
            data: self.data.clone(),
            source_list: self.source_list.clone(),
            source_list_type: self.source_list_type.clone(),
            ignore_dut1: self.ignore_dut1,
            outputs: Some(vec![PathBuf::from("/tmp/hyp_peel_di_sols.fits")]),
            model_filenames: None,
            output_model_time_average: self.output_model_time_average.clone(),
            output_model_freq_average: self.output_model_freq_average.clone(),
            num_sources: None,
            source_dist_cutoff: self.source_dist_cutoff,
            veto_threshold: self.veto_threshold,
            beam_file: self.beam_file.clone(),
            unity_dipole_gains: self.unity_dipole_gains,
            delays: self.delays.clone(),
            no_beam: self.no_beam,
            timesteps_per_timeblock: self.timesteps_per_timeblock.clone(),
            freq_average_factor: self.freq_average_factor.clone(),
            timesteps: self.timesteps.clone(),
            use_all_timesteps: self.use_all_timesteps,
            uvw_min: Some("0L".to_string()),
            uvw_max: None,
            max_iterations: None,
            stop_thresh: None,
            min_thresh: None,
            array_position: self.array_position.clone(),
            no_precession: self.no_precession,
            cpu: self.cpu,
            tile_flags: self.tile_flags.clone(),
            ignore_input_data_tile_flags: self.ignore_input_data_tile_flags,
            ignore_input_data_fine_channel_flags: self.ignore_input_data_fine_channel_flags,
            fine_chan_flags_per_coarse_chan: self.fine_chan_flags_per_coarse_chan.clone(),
            fine_chan_flags: self.fine_chan_flags.clone(),
            pfb_flavour: self.pfb_flavour.clone(),
            no_digital_gains: self.no_digital_gains,
            no_cable_length_correction: self.no_cable_length_correction,
            no_geometric_correction: self.no_geometric_correction,
            no_progress_bars: self.no_progress_bars,
        };

        let PeelArgs {
            data: _,
            source_list: _,
            source_list_type: _,
            ignore_dut1: _,
            outputs,
            output_model_time_average: _,
            output_model_freq_average: _,
            num_sources: _,
            source_dist_cutoff: _,
            veto_threshold: _,
            beam_file: _,
            unity_dipole_gains: _,
            delays: _,
            no_beam: _,
            timesteps_per_timeblock: _,
            freq_average_factor: _,
            timesteps: _,
            use_all_timesteps: _,
            array_position: _,
            no_precession,
            cpu: _,
            tile_flags: _,
            ignore_input_data_tile_flags: _,
            ignore_input_data_fine_channel_flags: _,
            fine_chan_flags_per_coarse_chan: _,
            fine_chan_flags: _,
            pfb_flavour: _,
            no_digital_gains: _,
            no_cable_length_correction: _,
            no_geometric_correction: _,
            no_progress_bars: _,
        } = self;

        let outputs = outputs.unwrap_or_else(|| vec![PathBuf::from("/tmp/hyp_peel.ms")]);
        let outputs = Vec1::try_from_vec(
            outputs
                .iter()
                .cloned()
                .map(|o| {
                    let ext = o.extension().and_then(|os_str| os_str.to_str()).unwrap();
                    let t = VisOutputType::from_str(ext).unwrap();
                    (o, t)
                })
                .collect(),
        )
        .unwrap();

        // TODO: Hack: set up DI cal params to get access to an obs context,
        // then use only the first 8s.
        let time_average_factor = {
            let di_cal_params = di_cal_args.clone().into_params().unwrap();
            let obs_context = di_cal_params.input_data.get_obs_context();
            let num_timesteps = (8.0 / obs_context.time_res.unwrap().to_seconds()).round() as usize;
            di_cal_args.timesteps = Some(
                di_cal_params
                    .timesteps
                    .into_iter()
                    .take(num_timesteps)
                    .collect(),
            );
            num_timesteps
        };
        let di_cal_params = di_cal_args.into_params().unwrap();
        let input_data = &di_cal_params.input_data;
        let obs_context = input_data.get_obs_context();
        let num_tiles = di_cal_params.unflagged_tile_xyzs.len();
        let num_cross_baselines = (num_tiles * (num_tiles - 1)) / 2;
        dbg!(num_tiles, num_cross_baselines);
        let fences = channels_to_chanblocks(
            &obs_context.fine_chan_freqs,
            obs_context.freq_res,
            32,
            &HashSet::new(),
        );
        assert_eq!(
            fences.len(),
            1,
            "Picket fence observations are not supported!"
        );
        let low_res_freqs_hz = fences
            .into_iter()
            .flat_map(|f| f.chanblocks.into_iter().map(|c| c._freq))
            .collect::<Vec<_>>();

        // Create an "iono source list" from our existing source list. This
        // involves finding the Stokes-I-weighted `RADec` of each source.

        // TODO: Sort the sources such that the brightest come first, dimmest last.
        let mut iono_srclist = {
            let mut iono_srclist = IonoSourceList::new();
            let mut component_radecs = vec![];
            let mut component_stokes_is = vec![];
            for (name, source) in di_cal_params.source_list.iter() {
                component_radecs.clear();
                component_stokes_is.clear();
                for comp in &source.components {
                    component_radecs.push(comp.radec);
                    // TODO: Do this properly.
                    component_stokes_is.push(1.0);
                }

                iono_srclist.insert(
                    name.clone(),
                    IonoSource {
                        source: source.clone(),
                        iono_consts: (0.0, 0.0),
                        weighted_radec: RADec::weighted_average(
                            &component_radecs,
                            &component_stokes_is,
                        )
                        .expect("component RAs aren't too far apart from one another"),
                    },
                );
            }
            iono_srclist
        };

        let timestamps = di_cal_params
            .timesteps
            .mapped_ref(|i| obs_context.timestamps[*i]);
        let precession_infos = timestamps
            .iter()
            .copied()
            .map(|time| {
                precess_time(
                    di_cal_params.array_position.longitude_rad,
                    di_cal_params.array_position.latitude_rad,
                    obs_context.phase_centre,
                    time,
                    di_cal_params.dut1,
                )
            })
            .collect::<Vec<_>>();
        let lmsts = precession_infos
            .iter()
            .map(|p| if no_precession { p.lmst } else { p.lmst_j2000 })
            .collect::<Vec<_>>();
        let mut precessed_tile_xyzs = Array2::from_elem(
            (timestamps.len(), di_cal_params.unflagged_tile_xyzs.len()),
            XyzGeodetic::default(),
        );
        precessed_tile_xyzs
            .outer_iter_mut()
            .zip(precession_infos.into_iter())
            .for_each(|(mut precessed_tile_xyzs, precession_info)| {
                let xyzs = precession_info.precess_xyz(&di_cal_params.unflagged_tile_xyzs);
                precessed_tile_xyzs.assign(&Array1::from_vec(xyzs));
            });

        // use the baseline taper from the RTS, 1-exp(-(u*u+v*v)/(2*sig^2));
        let short_baseline_sigma = 20.;
        // // let weight_synth = get_weights_rts(&vis_ctx, &obs_ctx, short_baseline_sigma);
        let weight_synth = get_weights_rts(
            &di_cal_params,
            // TODO: Do we care about weights changing over time?
            precessed_tile_xyzs
                .slice(s![0, ..])
                .as_slice()
                .expect("is contiguous"),
            lmsts[0],
            short_baseline_sigma,
        );

        // ////////////////// //
        // subtract model vis //
        // ////////////////// //

        // CHJ: Rotate to each source and remove its model from the data. Is
        // this better than just removing the full model without rotation?
        let CalVis {
            vis_data: mut vis_residual,
            mut vis_model,
            vis_weights,
        } = get_cal_vis(&di_cal_params, true).unwrap();

        // // Remove the model from the data, and set any visibilities with weights
        // // <= 0 equal to 0. Also scale weights by our peeling taper.
        // vis_residual
        //     .outer_iter_mut()
        //     .into_par_iter()
        //     .zip(vis_model.outer_iter_mut())
        //     .zip(vis_weights.outer_iter())
        //     .zip(weight_synth.outer_iter())
        //     .for_each(
        //         |(((mut vis_data, mut vis_model), vis_weights), weight_synth)| {
        //             vis_data
        //                 .outer_iter_mut()
        //                 .zip(vis_model.outer_iter_mut())
        //                 .zip(vis_weights.outer_iter())
        //                 .zip(weight_synth.outer_iter())
        //                 .zip(di_cal_params.baseline_weights.iter())
        //                 .for_each(
        //                     |(
        //                         (((mut vis_data, mut vis_model), vis_weights), weight_synth),
        //                         &baseline_weight,
        //                     )| {
        //                         vis_data
        //                             .iter_mut()
        //                             .zip(vis_model.iter_mut())
        //                             .zip(vis_weights.iter())
        //                             .zip(weight_synth.iter())
        //                             .for_each(
        //                                 |(((vis_data, vis_model), &vis_weight), weight_synth)| {
        //                                     let weight = f64::from(vis_weight)
        //                                         * baseline_weight
        //                                         * weight_synth;
        //                                     if weight <= 0.0 {
        //                                         *vis_data = Jones::default();
        //                                         *vis_model = Jones::default();
        //                                     } else {
        //                                         *vis_data = Jones::<f32>::from(
        //                                             Jones::<f64>::from(*vis_data) * weight,
        //                                         );
        //                                         *vis_model = Jones::<f32>::from(
        //                                             Jones::<f64>::from(*vis_model) * weight,
        //                                         );
        //                                     }
        //                                 },
        //                             );
        //                     },
        //                 );
        //         },
        //     );

        // vis_residual -= &vis_model;

        // for each source in the sourcelist:
        // - rotate the accumulated visibilities to the model phase centre
        // - simulate the visibilities and do not apply an ionospheric offset
        let mut modeller = new_sky_modeller(
            // matches!(di_cal_params.modeller_info, ModellerInfo::Cpu),
            false,
            di_cal_params.beam.deref(),
            &SourceList::new(),
            &di_cal_params.unflagged_tile_xyzs,
            &di_cal_params.unflagged_fine_chan_freqs,
            &di_cal_params.tile_baseline_flags.flagged_tiles,
            RADec::default(),
            di_cal_params.array_position.longitude_rad,
            di_cal_params.array_position.latitude_rad,
            di_cal_params.dut1,
            !no_precession,
        )
        .unwrap();

        let mut tile_ws_from =
            Array2::default((timestamps.len(), di_cal_params.unflagged_tile_xyzs.len()));
        let mut tile_ws_to = tile_ws_from.clone();
        // Set up the first set of Ws at the phase centre.
        tile_ws_from
            .outer_iter_mut()
            .into_par_iter()
            .zip(precessed_tile_xyzs.outer_iter())
            .zip(lmsts.par_iter())
            .for_each(|((mut tile_ws_from, precessed_tile_xyzs), lmst)| {
                let phase_centre = obs_context.phase_centre.to_hadec(*lmst);
                setup_ws(
                    tile_ws_from.view_mut(),
                    precessed_tile_xyzs.view(),
                    phase_centre,
                );
            });

        // Temporary visibility array, re-used for each timestep
        let mut vis_residual_tmp = vis_residual.clone();

        let mut source_list = SourceList::new();
        for (source_name, iono_source) in iono_srclist.iter_mut() {
            info!("Rotating to {source_name} and subtracting its model");
            source_list.clear();
            source_list.insert(String::new(), iono_source.source.clone());
            modeller
                .update_source_list(&source_list, iono_source.weighted_radec)
                .unwrap();

            vis_rotate(
                vis_residual.view_mut(),
                iono_source.weighted_radec,
                precessed_tile_xyzs.view(),
                &mut tile_ws_from,
                &mut tile_ws_to,
                &lmsts,
                &di_cal_params.unflagged_fine_chan_freqs,
                true,
            );

            vis_residual
                .outer_iter_mut()
                .zip(timestamps.iter())
                .for_each(|(mut vis_residual, epoch)| {
                    // model into vis_slice: (bl, ch), a 2d slice of vis_residual_tmp: (1, bl, ch)
                    // vis slice for modelling needs to be 2d, so we take a slice of vis_residual_tmp
                    let mut vis_slice = vis_residual_tmp.slice_mut(s![0, .., ..]);
                    eprintln!(
                        "modelling epoch {:?} {:?} with consts {:?}",
                        epoch.to_gregorian_utc_str(),
                        vis_slice.dim(),
                        iono_source.iono_consts
                    );
                    modeller
                        .model_timestep(vis_slice.view_mut(), *epoch)
                        .unwrap();

                    // TODO: This is currently useful as simulate_accumulate_iono is only in
                    // one place. But it might be useful in Shintaro feedback when the
                    // source already has constants?
                    // // apply iono to temp model
                    // if iono_source.iono_consts.0.abs() > 1e-9 || iono_source.iono_consts.1.abs() > 1e-9 {
                    //     let part_uvws = calc_part_uvws(
                    //         tile_xyzs.len(),
                    //         timestamps,
                    //         source_pos,
                    //         array_pos,
                    //         tile_xyzs,
                    //     );
                    //     apply_iono(
                    //         vis_residual_tmp.view_mut(),
                    //         part_uvws,
                    //         iono_source.iono_consts,
                    //         fine_chan_freqs,
                    //     );
                    // }
                    // after apply iono, copy to residual array
                    vis_residual -= &vis_residual_tmp.slice(s![0, .., ..]);
                });
        }
        // rotate back to original phase centre
        vis_rotate(
            vis_residual.view_mut(),
            obs_context.phase_centre,
            precessed_tile_xyzs.view(),
            &mut tile_ws_from,
            &mut tile_ws_to,
            &lmsts,
            &di_cal_params.unflagged_fine_chan_freqs,
            true,
        );

        // TODO: Do we allow multiple timesteps in the low-res data?

        let average_timestamp = average_epoch(&timestamps);
        let average_precession_info = precess_time(
            di_cal_params.array_position.longitude_rad,
            di_cal_params.array_position.latitude_rad,
            obs_context.phase_centre,
            average_timestamp,
            di_cal_params.dut1,
        );
        let average_lmst = if no_precession {
            average_precession_info.lmst
        } else {
            average_precession_info.lmst_j2000
        };
        let average_precessed_tile_xyzs = Array2::from_shape_vec(
            (1, num_tiles),
            average_precession_info.precess_xyz(&di_cal_params.unflagged_tile_xyzs),
        )
        .expect("correct shape");

        // temporary arrays for accumulation
        // TODO: Do a stocktake of arrays that are lying around!
        // These are time, bl, channel
        let mut vis_residual_low_res: Array3<Jones<f32>> =
            Array3::zeros((1, num_cross_baselines, low_res_freqs_hz.len()));
        let mut weight_residual_low_res: Array3<f32> = Array3::zeros(vis_residual_low_res.dim());
        let mut vis_model_low_res = vis_residual_low_res.clone();
        let mut vis_model_low_res_tmp: Array3<Jones<f32>> = vis_model_low_res.clone();
        let mut tile_uvs_high_res = Array2::default((timestamps.len(), num_tiles));
        let mut tile_uvs_low_res = Array2::default((1, num_tiles));

        // Pre-compute tile UVs.
        tile_uvs_high_res
            .outer_iter_mut()
            .zip(precessed_tile_xyzs.outer_iter())
            .zip(lmsts.iter())
            .for_each(|((mut tile_uvs, tile_xyzs), lmst)| {
                let phase_centre = obs_context.phase_centre.to_hadec(*lmst);
                setup_uvs(
                    tile_uvs.as_slice_mut().unwrap(),
                    tile_xyzs.as_slice().unwrap(),
                    phase_centre,
                );
            });

        // Set up the low-resolution modeller object.
        let mut low_res_modeller = new_sky_modeller(
            matches!(di_cal_params.modeller_info, ModellerInfo::Cpu),
            di_cal_params.beam.deref(),
            &SourceList::new(),
            &di_cal_params.unflagged_tile_xyzs,
            &low_res_freqs_hz,
            &di_cal_params.tile_baseline_flags.flagged_tiles,
            RADec::default(),
            di_cal_params.array_position.longitude_rad,
            di_cal_params.array_position.latitude_rad,
            di_cal_params.dut1,
            // TODO: Allow the user to turn off precession.
            true,
        )
        .unwrap();

        // /////////// //
        // UNPEEL LOOP //
        // /////////// //

        for (source_name, iono_source) in iono_srclist.iter_mut() {
            let start = std::time::Instant::now();

            let source_phase_centre = iono_source.weighted_radec;
            eprintln!(
                "unpeel loop: {source_name} at {source_phase_centre} (has iono {:?})",
                iono_source.iono_consts
            );

            let mut source_list = SourceList::new();
            source_list.insert(String::new(), iono_source.source.clone());
            modeller
                .update_source_list(&source_list, obs_context.phase_centre)
                .unwrap();
            low_res_modeller
                .update_source_list(&source_list, source_phase_centre)
                .unwrap();

            // ////////////// //
            // GENERATE MODEL //
            // ////////////// //

            // generate model again at higher resolution, obs phase centre
            dbg!(std::time::Instant::now() - start, "get new model vis");
            simulate_write(modeller.deref_mut(), &timestamps, vis_model.view_mut());

            // /////////////////// //
            // ROTATE, AVERAGE VIS //
            // /////////////////// //

            // Rotate the residual visibilities to the source phase centre and
            // average into vis_residual_low_res.
            dbg!(std::time::Instant::now() - start, "vis_rotate2");
            vis_rotate2(
                vis_residual.view(),
                vis_residual_tmp.view_mut(),
                source_phase_centre,
                precessed_tile_xyzs.view(),
                &mut tile_ws_from,
                &mut tile_ws_to,
                &lmsts,
                &di_cal_params.unflagged_fine_chan_freqs,
                false,
            );
            dbg!(std::time::Instant::now() - start, "vis_average");
            vis_average(
                vis_residual_tmp.view(),
                vis_residual_low_res.view_mut(),
                weight_synth.view(),
                weight_residual_low_res.view_mut(),
            );
            dbg!(std::time::Instant::now() - start, "get low-res model");

            // ////////////// //
            // GENERATE MODEL //
            // ////////////// //

            // generate model again at lower resolution, phased to source
            simulate_write(
                low_res_modeller.deref_mut(),
                &timestamps,
                vis_model_low_res.view_mut(),
            );
            dbg!(
                std::time::Instant::now() - start,
                "add low-res model source"
            );

            // ///////////// //
            // UNPEEL SOURCE //
            // ///////////// //
            // at lower resolution

            Zip::from(&mut vis_residual_low_res)
                .and(&vis_model_low_res)
                .for_each(|r, m| *r += *m);
            dbg!(std::time::Instant::now() - start, "alpha/beta loop");

            // ///////////////// //
            // CALCULATE OFFSETS //
            // ///////////////// //
            // iterate towards a convergent solution for ɑ, β

            let mut iteration = 0;
            while iteration != 10 {
                iteration += 1;

                vis_model_low_res_tmp.assign(&vis_model_low_res);
                // iono rotate model using existing iono consts (if they're
                // non-zero)
                if iono_source.iono_consts.0.abs() > f64::EPSILON
                    || iono_source.iono_consts.1.abs() > f64::EPSILON
                {
                    // Pre-compute tile UVs.
                    tile_uvs_low_res
                        .outer_iter_mut()
                        .zip(average_precessed_tile_xyzs.outer_iter())
                        .for_each(|(mut tile_uvs, tile_xyzs)| {
                            let phase_centre = source_phase_centre.to_hadec(average_lmst);
                            setup_uvs(
                                tile_uvs.as_slice_mut().unwrap(),
                                tile_xyzs.as_slice().unwrap(),
                                phase_centre,
                            );
                        });

                    apply_iono(
                        vis_model_low_res_tmp.view_mut(),
                        tile_uvs_low_res.view(),
                        iono_source.iono_consts,
                        &low_res_freqs_hz,
                    );
                }

                let offsets_rts = get_offsets_rts(
                    vis_residual_low_res.view(),
                    weight_residual_low_res.view(),
                    vis_model_low_res_tmp.view(),
                    average_precessed_tile_xyzs.as_slice().unwrap(),
                    &low_res_freqs_hz,
                    tile_uvs_low_res.as_slice_mut().unwrap(),
                    source_phase_centre.to_hadec(average_precession_info.lmst_j2000),
                );

                // if the offset is small, we've converged.
                iono_source.iono_consts.0 += offsets_rts[0];
                iono_source.iono_consts.1 += offsets_rts[1];
                if offsets_rts[0].abs() < 1e-12 && offsets_rts[1].abs() < 1e-12 {
                    eprintln!(
                        "iter {iteration}, consts: {:?}, finished",
                        iono_source.iono_consts
                    );
                    break;
                } else {
                    eprintln!("iter {iteration}, consts: {:?}", iono_source.iono_consts);
                }
            }
            dbg!(
                std::time::Instant::now() - start,
                "add high-res model source"
            );

            // /////////////// //
            // UPDATE RESIDUAL //
            // /////////////// //
            // at higher resolution, unpeel old model, then re-peel correctly rotated model.
            vis_residual
                .outer_iter_mut()
                // Doing this in parallel is only slightly faster, but it is
                // faster.
                .into_par_iter()
                .zip(vis_model.outer_iter())
                .for_each(|(mut vis_residual, vis_model)| {
                    vis_residual += &vis_model;
                });
            dbg!(std::time::Instant::now() - start, "apply_iono");
            apply_iono(
                vis_model.view_mut(),
                tile_uvs_high_res.view(),
                iono_source.iono_consts,
                &di_cal_params.unflagged_fine_chan_freqs,
            );
            dbg!(
                std::time::Instant::now() - start,
                "remove high-res model source"
            );
            vis_residual
                .outer_iter_mut()
                .into_par_iter()
                .zip(vis_model.outer_iter())
                .for_each(|(mut vis_residual, vis_model)| {
                    vis_residual -= &vis_model;
                });
            dbg!(std::time::Instant::now() - start, "end source loop");

            // ////// //
            // DI CAL //
            // ////// //

            // di_cal(vis_residual.view_mut(), vis_model.view(), &di_cal_params);

            // Zip::from(&mut vis_residual)
            //     .and(&vis_model)
            //     .for_each(|r, s| *r -= *s);

            // let vis_soln_path = OUT_DIR.join(format!("soln_{}.fits", source_name.clone()));
            // let metafits: Option<&str> = None;
            // sols.write_solutions_from_ext(vis_soln_path, metafits)
            //     .unwrap();
        }

        dbg!(iono_srclist
            .into_iter()
            .map(|(name, s)| (name, s.iono_consts.0, s.iono_consts.1))
            .collect::<Vec<_>>());

        // write final residual
        let error = AtomicCell::new(false);

        let timeblocks = timesteps_to_timeblocks(&timestamps, 1, &di_cal_params.timesteps);

        use std::thread;
        let (tx_write, rx_write) = bounded(1);

        thread::scope(|scope| {
            let consume_handle = scope.spawn(|| {
                defer_on_unwind! { error.store(true); }

                for ((cross_data, cross_weights), timestamp) in vis_residual
                    .outer_iter()
                    .zip(vis_weights.outer_iter())
                    .zip(timestamps.iter().copied())
                {
                    tx_write
                        .send(VisTimestep {
                            cross_data: cross_data.to_shared(),
                            cross_weights: cross_weights.to_shared(),
                            autos: None,
                            timestamp,
                        })
                        .unwrap();
                }
                drop(tx_write);
            });

            let write_handle = scope.spawn(|| {
                defer_on_unwind! { error.store(true); }

                let marlu_mwa_obs_context = input_data.get_metafits_context().map(|c| {
                    (
                        MwaObsContext::from_mwalib(c),
                        0..obs_context.coarse_chan_freqs.len(),
                    )
                });
                let pb = ProgressBar::new(timeblocks.len() as _)
                .with_style(
                    ProgressStyle::default_bar()
                        .template("{msg:17}: [{wide_bar:.blue}] {pos:2}/{len:2} timeblocks ({elapsed_precise}<{eta_precise})").unwrap()
                        .progress_chars("=> "),
                )
                .with_position(0)
                .with_message("Writing peeled visibilities");
                pb.tick();

                let result = write_vis(
                    &outputs,
                    di_cal_params.array_position,
                    obs_context.phase_centre,
                    obs_context.pointing_centre,
                    &obs_context.tile_xyzs,
                    &obs_context.tile_names,
                    obs_context.obsid,
                    &timestamps,
                    &di_cal_params.timesteps,
                    &timeblocks,
                    obs_context.guess_time_res(),
                    di_cal_params.dut1,
                    obs_context.guess_freq_res(),
                    &obs_context.fine_chan_freqs.mapped_ref(|&f| f as f64),
                    &di_cal_params
                        .tile_baseline_flags
                        .unflagged_cross_baseline_to_tile_map
                        .values()
                        .copied()
                        .sorted()
                        .collect::<Vec<_>>(),
                    &HashSet::new(),
                    1,
                    1,
                    marlu_mwa_obs_context.as_ref().map(|(c, r)| (c, r)),
                    rx_write,
                    &error,
                    Some(pb),
                );
                match result {
                    Ok(m) => {
                        info!("{m}");
                        Ok(())
                    },
                    Err(e) => {
                        error.store(true);
                        Err(e)
                    }
                }
            });

            consume_handle.join().unwrap();
            write_handle.join().unwrap().unwrap();
        });

        Ok(())
    }
}

// fn get_weights_rts(
//     vis_ctx: &VisContext,
//     obs_ctx: &MarluObsContext,
//     short_sigma: f64,
// ) -> Array3<f32> {
//     let sel_shape = vis_ctx.sel_dims();
//     let tile_xyzs: Vec<XyzGeodetic> = obs_ctx.ant_positions_geodetic().collect();
//     let centroid_timestamps: Vec<Epoch> = vis_ctx.timeseries(false, true).collect();
//     let phase_centre = obs_ctx.phase_centre;
//     let array_pos = obs_ctx.array_pos;
//     let ant_pairs = vis_ctx.sel_baselines.clone();
//     let part_uvws = calc_part_uvws(
//         &ant_pairs,
//         &centroid_timestamps,
//         phase_centre,
//         array_pos,
//         &tile_xyzs,
//     );
//     let weight_factor = vis_ctx.weight_factor();
//     let freqs_hz = vis_ctx.frequencies_hz();

//     Array3::from_shape_fn((sel_shape.0, sel_shape.2, sel_shape.1), |(ts, bl, ch)| {
//         let (ant1, ant2) = ant_pairs[bl];
//         let uvw = part_uvws[[ts, ant1]] - part_uvws[[ts, ant2]];
//         // to convert to RTS uvw (wavelengths) from PAL uvw (meters), dividy by λ.
//         let lambda = VEL_C / freqs_hz[ch];
//         let (u, v) = (uvw.u / lambda, uvw.v / lambda);
//         let uv_sq = u * u + v * v;
//         weight_factor as f32 * (1.0 - (-uv_sq / (2.0 * short_sigma * short_sigma).exp())) as f32
//     })
// }

fn get_weights_rts(
    params: &super::di_calibrate::DiCalParams,
    precessed_tile_xyzs: &[XyzGeodetic],
    lmst: f64,
    short_sigma: f64,
) -> Array3<f32> {
    let sel_shape = (
        params.timesteps.len(),
        params.unflagged_fine_chan_freqs.len(),
        params.get_num_unflagged_cross_baselines(),
    );
    let obs_context = params.get_obs_context();
    let uvws = marlu::pos::xyz::xyzs_to_cross_uvws(
        precessed_tile_xyzs,
        obs_context.phase_centre.to_hadec(lmst),
    );
    let weight_factor = (obs_context.guess_freq_res() / FREQ_WEIGHT_FACTOR)
        * (obs_context.guess_time_res().to_seconds() / TIME_WEIGHT_FACTOR);
    let freqs_hz = &params.unflagged_fine_chan_freqs;

    Array3::from_shape_fn((sel_shape.0, sel_shape.2, sel_shape.1), |(_ts, bl, ch)| {
        let uvw = uvws[bl];
        // to convert to RTS uvw (wavelengths) from PAL uvw (meters), divide by λ.
        let lambda = VEL_C / freqs_hz[ch];
        let (u, v) = (uvw.u / lambda, uvw.v / lambda);
        let uv_sq = u * u + v * v;
        (weight_factor * (1.0 - (-uv_sq / (2.0 * short_sigma * short_sigma).exp()))) as f32
    })
}

// // rotate visibilities and average them (along with weights) in time and frequency given by vis_ctx
// fn rotate_accumulate(
//     jones_from: ArrayView3<Jones<f32>>,
//     mut jones_to: ArrayViewMut3<Jones<f32>>,
//     weight_from: ArrayView3<f32>,
//     mut weight_to: ArrayViewMut3<f32>,
//     tile_xyzs: &[XyzGeodetic],
//     vis_ctx: &VisContext,
//     obs_ctx: &MarluObsContext,
//     phase_to: RADec,
// ) {
//     let freqs_hz = vis_ctx.frequencies_hz();
//     let centroid_timestamps: Vec<Epoch> = vis_ctx.timeseries(false, true).collect();
//     let phase_from = obs_ctx.phase_centre;
//     let array_pos = obs_ctx.array_pos;
//     let from_dims = jones_from.dim();
//     let avg_time = vis_ctx.avg_time;
//     let avg_freq = vis_ctx.avg_freq;

//     // assert_eq!(
//     //     from_dims,
//     //     (centroid_timestamps.len(), ant_pairs.len(), freqs_hz.len()),
//     //     "jones_from.dim()!=vis_ctx.{{ts,bl,ch}}"
//     // );
//     assert_eq!(from_dims, weight_from.dim());

//     let to_dims = jones_to.dim();
//     assert_eq!(
//         to_dims,
//         (
//             (from_dims.0 as f64 / avg_time as f64).floor() as usize,
//             from_dims.1,
//             (from_dims.2 as f64 / avg_freq as f64).floor() as usize,
//         ),
//         "jones_to.dim()!=vis_ctx.av{{ts,bl,ch}}"
//     );
//     assert_eq!(to_dims, weight_to.dim());

//     let num_tiles = tile_xyzs.len();

//     // pre-compute uvws:
//     let part_uvws_from = calc_part_uvws(
//         num_tiles,
//         &centroid_timestamps,
//         phase_from,
//         array_pos,
//         tile_xyzs,
//     );
//     let part_uvws_to = calc_part_uvws(
//         num_tiles,
//         &centroid_timestamps,
//         phase_to,
//         array_pos,
//         tile_xyzs,
//     );

//     // iterate along time axis in chunks of avg_time
//     jones_from
//         .axis_chunks_iter(Axis(0), avg_time)
//         .zip(weight_from.axis_chunks_iter(Axis(0), avg_time))
//         .zip(jones_to.outer_iter_mut())
//         .zip(weight_to.outer_iter_mut())
//         .zip(part_uvws_from.axis_chunks_iter(Axis(0), avg_time))
//         .zip(part_uvws_to.axis_chunks_iter(Axis(0), avg_time))
//         .for_each(
//             |(
//                 ((((jones_chunk, weight_chunk), mut jones_to), mut weight_to), part_uvws_from),
//                 part_uvws_to,
//             )| {
//                 // iterate along baseline axis
//                 let mut i_tile1 = 0;
//                 let mut i_tile2 = 0;
//                 jones_chunk
//                     .axis_iter(Axis(1))
//                     .zip(weight_chunk.axis_iter(Axis(1)))
//                     .zip(jones_to.outer_iter_mut())
//                     .zip(weight_to.outer_iter_mut())
//                     .for_each(
//                         |(((jones_chunk, weight_chunk), mut jones_to), mut weight_to)| {
//                             i_tile2 += 1;
//                             if i_tile2 == num_tiles {
//                                 i_tile1 += 1;
//                                 i_tile2 = i_tile1 + 1;
//                             }

//                             jones_chunk
//                                 .axis_chunks_iter(Axis(1), avg_freq)
//                                 .zip(weight_chunk.axis_chunks_iter(Axis(1), avg_freq))
//                                 .zip(jones_to.iter_mut())
//                                 .zip(weight_to.iter_mut())
//                                 .zip(freqs_hz.chunks(avg_freq))
//                                 .for_each(
//                                     |(
//                                         (((jones_chunk, weight_chunk), jones_to), weight_to),
//                                         freq_chunk,
//                                     )| {
//                                         let mut weight_sum_f64 = 0_f64;
//                                         let mut jones_sum = Jones::default();
//                                         let mut jones_weighted_sum = Jones::default();
//                                         let mut avg_flag = true;

//                                         // iterate through time chunks
//                                         jones_chunk
//                                             .outer_iter()
//                                             .zip(weight_chunk.outer_iter())
//                                             .zip(part_uvws_from.outer_iter())
//                                             .zip(part_uvws_to.outer_iter())
//                                             .for_each(
//                                                 |(
//                                                     ((jones_chunk, weights_chunk), part_uvws_from),
//                                                     part_uvws_to,
//                                                 )| {
//                                                     let arg_w_diff = (part_uvws_to[i_tile1].w
//                                                         - part_uvws_to[i_tile2].w)
//                                                         - (part_uvws_from[[i_tile1]].w
//                                                             - part_uvws_from[[i_tile2]].w);
//                                                     jones_chunk
//                                                         .iter()
//                                                         .zip(weights_chunk.iter())
//                                                         .zip(freq_chunk.iter())
//                                                         .for_each(|((jones, weight), &freq_hz)| {
//                                                             let rotation = Complex::cis(
//                                                                 -TAU * arg_w_diff * freq_hz / VEL_C,
//                                                             );
//                                                             let jones_rotated =
//                                                                 Jones::<f64>::from(*jones)
//                                                                     * rotation;
//                                                             jones_sum += jones_rotated;
//                                                             if weight.abs() > 0. {
//                                                                 avg_flag = false;
//                                                                 let weight_abs_f64 =
//                                                                     (*weight as f64).abs();
//                                                                 weight_sum_f64 += weight_abs_f64;
//                                                                 jones_weighted_sum +=
//                                                                     jones_rotated * weight_abs_f64;
//                                                             }
//                                                         });
//                                                 },
//                                             );

//                                         *jones_to = if !avg_flag {
//                                             Jones::from(jones_weighted_sum / weight_sum_f64)
//                                         } else {
//                                             let chunk_size = jones_chunk.len() as f64;
//                                             Jones::from(jones_sum / chunk_size)
//                                         };

//                                         *weight_to = weight_sum_f64 as f32;
//                                     },
//                                 );
//                         },
//                     );
//             },
//         );
// }

/// Average "high-res" data to "low-res" data.
/// TODO: Do we only ever want 1 timestep in the low-res data?
fn vis_average(
    jones_from: ArrayView3<Jones<f32>>,
    mut jones_to: ArrayViewMut3<Jones<f32>>,
    weight_from: ArrayView3<f32>,
    mut weight_to: ArrayViewMut3<f32>,
) {
    let from_dims = jones_from.dim();
    let time_axis = Axis(0);
    let freq_axis = Axis(2);
    let avg_time = jones_from.len_of(time_axis) / jones_to.len_of(time_axis);
    let avg_freq = jones_from.len_of(freq_axis) / jones_to.len_of(freq_axis);

    // assert_eq!(
    //     from_dims,
    //     (centroid_timestamps.len(), ant_pairs.len(), freqs_hz.len()),
    //     "jones_from.dim()!=vis_ctx.{{ts,bl,ch}}"
    // );
    assert_eq!(from_dims, weight_from.dim());

    let to_dims = jones_to.dim();
    assert_eq!(
        to_dims,
        (
            (from_dims.0 as f64 / avg_time as f64).floor() as usize,
            from_dims.1,
            (from_dims.2 as f64 / avg_freq as f64).floor() as usize,
        ),
        "jones_to.dim()!=vis_ctx.av{{ts,bl,ch}}"
    );
    assert_eq!(to_dims, weight_to.dim());

    let num_tiles = num_tiles_from_num_cross_correlation_baselines(jones_from.len_of(Axis(1)));

    // iterate along time axis in chunks of avg_time
    jones_from
        .axis_chunks_iter(Axis(0), avg_time)
        .zip(weight_from.axis_chunks_iter(Axis(0), avg_time))
        .zip(jones_to.outer_iter_mut())
        .zip(weight_to.outer_iter_mut())
        .for_each(
            |(((jones_chunk, weight_chunk), mut jones_to), mut weight_to)| {
                // iterate along baseline axis
                let mut i_tile1 = 0;
                let mut i_tile2 = 0;
                jones_chunk
                    .axis_iter(Axis(1))
                    .zip(weight_chunk.axis_iter(Axis(1)))
                    .zip(jones_to.outer_iter_mut())
                    .zip(weight_to.outer_iter_mut())
                    .for_each(
                        |(((jones_chunk, weight_chunk), mut jones_to), mut weight_to)| {
                            i_tile2 += 1;
                            if i_tile2 == num_tiles {
                                i_tile1 += 1;
                                i_tile2 = i_tile1 + 1;
                            }

                            jones_chunk
                                .axis_chunks_iter(Axis(1), avg_freq)
                                .zip(weight_chunk.axis_chunks_iter(Axis(1), avg_freq))
                                .zip(jones_to.iter_mut())
                                .zip(weight_to.iter_mut())
                                .for_each(
                                    |(((jones_chunk, weight_chunk), jones_to), weight_to)| {
                                        let mut jones_weighted_sum = Jones::default();
                                        let mut weight_sum = 0.0;

                                        // iterate through time chunks
                                        jones_chunk
                                            .outer_iter()
                                            .zip(weight_chunk.outer_iter())
                                            .for_each(|(jones_chunk, weights_chunk)| {
                                                jones_chunk
                                                    .iter()
                                                    .zip(weights_chunk.iter())
                                                    .for_each(|(jones, weight)| {
                                                        // Ignore any flagged
                                                        // vis.
                                                        if *weight > 0.0 {
                                                            let jones = Jones::<f64>::from(*jones);
                                                            let weight = *weight as f64;
                                                            jones_weighted_sum += jones * weight;
                                                            weight_sum += weight;
                                                        }
                                                    });
                                            });

                                        *jones_to = Jones::from(jones_weighted_sum / weight_sum);

                                        *weight_to = weight_sum as f32;
                                    },
                                );
                        },
                    );
            },
        );
}

/// Rotate the provided visibilities to the given phase centre. This function
/// expects:
///
/// 1) `tile_xyzs` to have already been precessed,
/// 2) `tile_ws_from` to already be populated with the correct [`W`]s for where
///    the data is currently phased,
/// 3) An equal number of timesteps in `jones_array`, `tile_xyzs`,
///    `tile_ws_from`, `tile_ws_to` and `lmsts`.
///
/// After the visibilities have been "rotated", the memory of `tile_ws_from` and
/// `tile_ws_to` is swapped. This allows this function to be called again with
/// the same arrays and a new phase centre without new allocations.
#[allow(clippy::too_many_arguments)]
fn vis_rotate(
    mut jones_array: ArrayViewMut3<Jones<f32>>,
    phase_to: RADec,
    tile_xyzs: ArrayView2<XyzGeodetic>,
    tile_ws_from: &mut Array2<W>,
    tile_ws_to: &mut Array2<W>,
    lmsts: &[f64],
    fine_chan_freqs: &[f64],
    swap: bool,
) {
    let num_tiles = tile_xyzs.len_of(Axis(1));
    assert_eq!(tile_ws_from.len_of(Axis(1)), num_tiles);
    assert_eq!(tile_ws_to.len_of(Axis(1)), num_tiles);

    // iterate along time axis in chunks of avg_time
    jones_array
        .outer_iter_mut()
        .into_par_iter()
        .zip(tile_ws_from.outer_iter())
        .zip(tile_ws_to.outer_iter_mut())
        .zip(tile_xyzs.outer_iter())
        .zip(lmsts.par_iter())
        .for_each(
            |((((mut jones_array, tile_ws_from), mut tile_ws_to), tile_xyzs), lmst)| {
                assert_eq!(tile_ws_from.len(), num_tiles);
                // Generate the "to" Ws.
                let phase_to = phase_to.to_hadec(*lmst);
                setup_ws(tile_ws_to.view_mut(), tile_xyzs.view(), phase_to);

                // iterate along baseline axis
                let mut i_tile1 = 0;
                let mut i_tile2 = 0;
                let mut tile1_w_from = tile_ws_from[i_tile1];
                let mut tile2_w_from = tile_ws_from[i_tile2];
                let mut tile1_w_to = tile_ws_to[i_tile1];
                let mut tile2_w_to = tile_ws_to[i_tile2];
                jones_array.outer_iter_mut().for_each(|mut jones_array| {
                    i_tile2 += 1;
                    if i_tile2 == num_tiles {
                        i_tile1 += 1;
                        i_tile2 = i_tile1 + 1;
                        tile1_w_from = tile_ws_from[i_tile1];
                        tile1_w_to = tile_ws_to[i_tile1];
                    }
                    tile2_w_from = tile_ws_from[i_tile2];
                    tile2_w_to = tile_ws_to[i_tile2];

                    let w_diff = (tile1_w_to - tile2_w_to) - (tile1_w_from - tile2_w_from);
                    let arg = -TAU * w_diff / VEL_C;
                    // iterate along frequency axis
                    jones_array.iter_mut().zip(fine_chan_freqs.iter()).for_each(
                        |(jones, &freq_hz)| {
                            let rotation = Complex::cis(arg * freq_hz);
                            *jones = Jones::<f32>::from(Jones::<f64>::from(*jones) * rotation);
                        },
                    );
                });
            },
        );

    if swap {
        // Swap the arrays, so that for the next source, the "from" Ws are our "to"
        // Ws.
        std::mem::swap(tile_ws_from, tile_ws_to);
    }
}

#[allow(clippy::too_many_arguments)]
fn vis_rotate2(
    jones_array: ArrayView3<Jones<f32>>,
    mut jones_array2: ArrayViewMut3<Jones<f32>>,
    phase_to: RADec,
    tile_xyzs: ArrayView2<XyzGeodetic>,
    tile_ws_from: &mut Array2<W>,
    tile_ws_to: &mut Array2<W>,
    lmsts: &[f64],
    fine_chan_freqs: &[f64],
    swap: bool,
) {
    let num_tiles = tile_xyzs.len_of(Axis(1));
    assert_eq!(tile_ws_from.len_of(Axis(1)), num_tiles);
    assert_eq!(tile_ws_to.len_of(Axis(1)), num_tiles);

    // iterate along time axis in chunks of avg_time
    jones_array
        .outer_iter()
        .into_par_iter()
        .zip(jones_array2.outer_iter_mut())
        .zip(tile_ws_from.outer_iter())
        .zip(tile_ws_to.outer_iter_mut())
        .zip(tile_xyzs.outer_iter())
        .zip(lmsts.par_iter())
        .for_each(
            |(
                ((((jones_array, mut jones_array2), tile_ws_from), mut tile_ws_to), tile_xyzs),
                lmst,
            )| {
                assert_eq!(tile_ws_from.len(), num_tiles);
                // Generate the "to" Ws.
                let phase_to = phase_to.to_hadec(*lmst);
                setup_ws(tile_ws_to.view_mut(), tile_xyzs.view(), phase_to);

                // iterate along baseline axis
                let mut i_tile1 = 0;
                let mut i_tile2 = 0;
                let mut tile1_w_from = tile_ws_from[i_tile1];
                let mut tile2_w_from = tile_ws_from[i_tile2];
                let mut tile1_w_to = tile_ws_to[i_tile1];
                let mut tile2_w_to = tile_ws_to[i_tile2];
                jones_array
                    .outer_iter()
                    .zip(jones_array2.outer_iter_mut())
                    .for_each(|(jones_array, mut jones_array2)| {
                        i_tile2 += 1;
                        if i_tile2 == num_tiles {
                            i_tile1 += 1;
                            i_tile2 = i_tile1 + 1;
                            tile1_w_from = tile_ws_from[i_tile1];
                            tile1_w_to = tile_ws_to[i_tile1];
                        }
                        tile2_w_from = tile_ws_from[i_tile2];
                        tile2_w_to = tile_ws_to[i_tile2];

                        let w_diff = (tile1_w_to - tile2_w_to) - (tile1_w_from - tile2_w_from);
                        let arg = -TAU * w_diff / VEL_C;
                        // iterate along frequency axis
                        jones_array
                            .iter()
                            .zip(jones_array2.iter_mut())
                            .zip(fine_chan_freqs.iter())
                            .for_each(|((jones, jones2), &freq_hz)| {
                                let rotation = Complex::cis(arg * freq_hz);
                                *jones2 = Jones::<f32>::from(Jones::<f64>::from(*jones) * rotation);
                            });
                    });
            },
        );

    if swap {
        // Swap the arrays, so that for the next source, the "from" Ws are our "to"
        // Ws.
        std::mem::swap(tile_ws_from, tile_ws_to);
    }
}

/// Rotate the supplied visibilities according to the `λ²` constants of
/// proportionality with `exp(-2πi(αu+βv)λ²)`.
fn apply_iono(
    mut jones: ArrayViewMut3<Jones<f32>>,
    tile_uvs: ArrayView2<UV>,
    const_lm: (f64, f64),
    freqs_hz: &[f64],
) {
    let num_tiles = tile_uvs.len_of(Axis(1));

    // iterate along time axis
    jones
        .outer_iter_mut()
        .into_par_iter()
        .zip(tile_uvs.outer_iter())
        .for_each(|(mut jones, tile_uvs)| {
            assert_eq!(tile_uvs.len(), num_tiles);

            // iterate along baseline axis
            let mut i_tile1 = 0;
            let mut i_tile2 = 0;
            for mut jones in jones.outer_iter_mut() {
                i_tile2 += 1;
                if i_tile2 == num_tiles {
                    i_tile1 += 1;
                    i_tile2 = i_tile1 + 1;
                }

                let uv = tile_uvs[i_tile1] - tile_uvs[i_tile2];
                let arg = -TAU * (uv.u * const_lm.0 + uv.v * const_lm.1) * VEL_C;
                // iterate along frequency axis
                jones.iter_mut().zip(freqs_hz).for_each(|(jones, freq_hz)| {
                    // The baseline UV is in units of metres, so we need to
                    // divide by λ to use it in an exponential. But we're also
                    // multiplying by λ², so just multiply by λ (divide by
                    // frequency).
                    let rotation = Complex::cis(arg / *freq_hz);
                    let rotation = Complex::new(rotation.re as f32, rotation.im as f32);
                    *jones *= rotation;
                });
            }
        });
}

// TODO: CHJ: Ask Dev if this is useful
// // apply ionospheric rotation approximation by `order` taylor expansion terms
// fn apply_iono_approx<F>(
//     mut jones: ArrayViewMut3<Jones<F>>,
//     vis_ctx: &VisContext,
//     obs_ctx: &MarluObsContext,
//     // constants of proportionality for ionospheric offset in l,m
//     const_lm: (f64, f64),
//     order: usize,
// ) where
//     F: Float + Num + NumAssign + Default,
// {
//     let jones_dims = jones.dim();
//     let freqs_hz = vis_ctx.frequencies_hz();
//     let tile_xyzs: Vec<XyzGeodetic> = obs_ctx.ant_positions_geodetic().collect();
//     let centroid_timestamps: Vec<Epoch> = vis_ctx.timeseries(false, true).collect();
//     let phase_centre = obs_ctx.phase_centre;
//     let array_pos = obs_ctx.array_pos;
//     // let ant_pairs = vis_ctx.sel_baselines.clone();
//     let num_tiles = num_tiles_from_num_cross_correlation_baselines(jones.dim().1);

//     assert_eq!(jones_dims.0, centroid_timestamps.len());
//     // assert_eq!(jones_dims.1, ant_pairs.len());
//     assert_eq!(jones_dims.2, freqs_hz.len());

//     // pre-compute partial uvws:
//     let part_uvws = calc_part_uvws(
//         num_tiles,
//         &centroid_timestamps,
//         phase_centre,
//         array_pos,
//         &tile_xyzs,
//     );

//     let lambdas = freqs_hz
//         .iter()
//         .map(|freq_hz| VEL_C / freq_hz)
//         .collect::<Vec<_>>();

//     // iterate along time axis
//     for (mut jones, part_uvws) in jones.outer_iter_mut().zip(part_uvws.outer_iter()) {
//         let mut i_tile1 = 0;
//         let mut i_tile2 = 1;

//         // iterate along baseline axis
//         for mut jones in jones.outer_iter_mut() {
//             i_tile2 += 1;
//             if i_tile2 == num_tiles {
//                 i_tile1 += 1;
//                 i_tile2 = i_tile1 + 1;
//             }

//             let uvw = part_uvws[[i_tile1]] - part_uvws[[i_tile2]];
//             let uv_lm = uvw.u * const_lm.0 + uvw.v * const_lm.1;
//             // iterate along frequency axis
//             for (jones, &lambda) in jones.iter_mut().zip(&lambdas) {
//                 // in RTS, uvw is in units of λ but pal uvw is in meters, so divide by wavelength,
//                 // but we're also multiplying by λ², so just multiply by λ

//                 // first order taylor expansion, data D from rotation of model M
//                 // D = M * exp(-i * phi * lambda^2 )
//                 //   = M * (1 - i * phi * lambda^2 + ... )
//                 let exponent = -Complex::i() * F::from(TAU * uv_lm * lambda).unwrap();
//                 let rotation: Complex<F> = (0..=order)
//                     .map(|n| exponent.powi(n as i32) / F::from(factorial(&n)).unwrap())
//                     .sum();
//                 *jones *= rotation;
//             }
//         }
//     }
// }

// the offsets as defined by the RTS code
// TODO: Assume there's only 1 timestep, because this is low res data?
fn get_offsets_rts(
    unpeeled: ArrayView3<Jones<f32>>,
    weights: ArrayView3<f32>,
    model: ArrayView3<Jones<f32>>,
    tile_xyzs: &[XyzGeodetic],
    freqs_hz: &[f64],
    tile_uvs_low_res: &mut [UV],
    phase_centre_hadec: HADec,
) -> [f64; 2] {
    let num_tiles = tile_xyzs.len();

    setup_uvs(tile_uvs_low_res, tile_xyzs, phase_centre_hadec);
    assert_eq!(tile_uvs_low_res.len(), num_tiles);

    // a-terms used in least-squares estimator
    let (mut a_uu, mut a_uv, mut a_vv) = (0., 0., 0.);
    // A-terms used in least-squares estimator
    let (mut aa_u, mut aa_v) = (0., 0.);

    // iterate over time
    unpeeled
        .outer_iter()
        .zip(weights.outer_iter())
        .zip(model.outer_iter())
        .for_each(|((unpeeled, weights), model)| {
            // iterate over frequency
            unpeeled
                .axis_iter(Axis(1))
                .zip(weights.axis_iter(Axis(1)))
                .zip(model.axis_iter(Axis(1)))
                .zip(freqs_hz.iter())
                .for_each(|(((unpeeled, weights), model), freq_hz)| {
                    let lambda = VEL_C / freq_hz;
                    // lambda^2
                    let lambda_2 = lambda * lambda;
                    // lambda^4
                    let lambda_4 = lambda_2 * lambda_2;

                    let mut i_tile1 = 0;
                    let mut i_tile2 = 0;
                    let mut uvw_tile1 = tile_uvs_low_res[i_tile1];
                    let mut uvw_tile2 = tile_uvs_low_res[i_tile2];

                    // iterate over baseline
                    unpeeled
                        .iter()
                        .zip(weights.iter())
                        .zip(model.iter())
                        .for_each(|((unpeeled, weight), model)| {
                            i_tile2 += 1;
                            if i_tile2 == num_tiles {
                                i_tile1 += 1;
                                i_tile2 = i_tile1 + 1;
                                uvw_tile1 = tile_uvs_low_res[i_tile1];
                            }

                            if *weight > 0. {
                                uvw_tile2 = tile_uvs_low_res[i_tile2];
                                let uvw = uvw_tile1 - uvw_tile2;

                                // stokes I power of the unpeeled visibilities (Data)
                                let unpeeled_i = 0.5 * (unpeeled[0] + unpeeled[3]);
                                // stokes I power of the model visibilities (Model)
                                let model_i = 0.5 * (model[0] + model[3]);

                                let mr = (model_i.re as f64) * (unpeeled_i - model_i).im as f64;
                                let mm = (model_i.re as f64) * model_i.re as f64;

                                let weight_f64 = *weight as f64;

                                // to convert to RTS uvw (wavelengths) from PAL uvw (meters), divide by λ.
                                let (u, v) = (uvw.u / lambda, uvw.v / lambda);
                                a_uu += weight_f64 * mm * u * u * lambda_4;
                                a_uv += weight_f64 * mm * u * v * lambda_4;
                                a_vv += weight_f64 * mm * v * v * lambda_4;
                                aa_u += weight_f64 * mr * u * -lambda_2;
                                aa_v += weight_f64 * mr * v * -lambda_2;
                            }
                        });
                });
        });
    let delta = TAU * (a_uu * a_vv - a_uv * a_uv);

    let offsets = [
        (aa_u * a_vv - aa_v * a_uv) / delta,
        (aa_v * a_uu - aa_u * a_uv) / delta,
    ];

    eprintln!("offsets: {offsets:?}");
    offsets
}

fn setup_ws(
    mut tile_ws: ArrayViewMut1<W>,
    tile_xyzs: ArrayView1<XyzGeodetic>,
    phase_centre: HADec,
) {
    let (s_ha, c_ha) = phase_centre.ha.sin_cos();
    let (s_dec, c_dec) = phase_centre.dec.sin_cos();
    tile_ws
        .iter_mut()
        .zip(tile_xyzs.iter().copied())
        .for_each(|(tile_w, tile_xyz)| {
            *tile_w = W::from_xyz(tile_xyz, s_ha, c_ha, s_dec, c_dec);
        });
}

fn setup_uvs(tile_uvs: &mut [UV], tile_xyzs: &[XyzGeodetic], phase_centre: HADec) {
    let (s_ha, c_ha) = phase_centre.ha.sin_cos();
    let (s_dec, c_dec) = phase_centre.dec.sin_cos();
    tile_uvs
        .iter_mut()
        .zip(tile_xyzs.iter().copied())
        .for_each(|(tile_uv, tile_xyz)| {
            *tile_uv = UV::from_xyz(tile_xyz, s_ha, c_ha, s_dec, c_dec);
        });
}

fn simulate_write(
    modeller: &mut dyn SkyModeller,
    timestamps: &[Epoch],
    mut vis_result: ArrayViewMut3<Jones<f32>>,
) {
    vis_result
        .outer_iter_mut()
        .zip(timestamps.iter())
        .for_each(|(mut vis_result, epoch)| {
            modeller
                .model_timestep(vis_result.view_mut(), *epoch)
                .unwrap();
        });
}

// fn di_cal(
//     mut vis_unpeeled: ArrayViewMut3<Jones<f32>>,
//     vis_model: ArrayView3<Jones<f32>>,
//     params: &DiCalParams,
// ) {
//     let num_tiles = params.unflagged_tile_xyzs.len();
//     let shape = (1, num_tiles, params.fences.first().chanblocks.len());
//     let mut di_jones = Array3::from_elem(shape, Jones::identity());
//     let pb = ProgressBar::new(params.fences.first().chanblocks.len() as _);
//     calibrate_timeblock(
//         vis_unpeeled.view(),
//         vis_model.view(),
//         di_jones.view_mut(),
//         params.timeblocks.first(),
//         &params.fences.first().chanblocks,
//         DEFAULT_MAX_ITERATIONS,
//         DEFAULT_STOP_THRESHOLD,
//         DEFAULT_MIN_THRESHOLD,
//         pb,
//         false,
//     );

//     vis_unpeeled.outer_iter_mut().for_each(|mut vis_unpeeled| {
//         let mut i_tile1 = 0;
//         let mut i_tile2 = 0;
//         let mut tile1_sol = di_jones.slice(s![0, i_tile1, ..]);
//         let mut tile2_sol = di_jones.slice(s![0, i_tile2, ..]);
//         vis_unpeeled.outer_iter_mut().for_each(|mut vis_unpeeled| {
//             i_tile2 += 1;
//             if i_tile2 == num_tiles {
//                 i_tile1 += 1;
//                 i_tile2 = i_tile1 + 1;
//                 tile1_sol = di_jones.slice(s![0, i_tile1, ..]);
//             }
//             tile2_sol = di_jones.slice(s![0, i_tile2, ..]);

//             vis_unpeeled
//                 .iter_mut()
//                 .zip(tile1_sol.iter())
//                 .zip(tile2_sol.iter())
//                 .for_each(|((vis_unpeeled, sol_tile1), sol_tile2)| {
//                     *vis_unpeeled = Jones::<f32>::from(
//                         sol_tile1.inv() * Jones::<f64>::from(*vis_unpeeled) * sol_tile2.inv().h(),
//                     );
//                 });
//         });
//     });
// }

/// Just the W terms of [`UVW`] coordinates.
#[derive(Clone, Copy, Default)]
struct W(f64);

impl W {
    fn from_xyz(xyz: XyzGeodetic, s_ha: f64, c_ha: f64, s_dec: f64, c_dec: f64) -> W {
        W(c_dec * c_ha * xyz.x - c_dec * s_ha * xyz.y + s_dec * xyz.z)
    }
}

impl Sub for W {
    type Output = f64;

    fn sub(self, rhs: Self) -> Self::Output {
        self.0 - rhs.0
    }
}

/// Just the U and V terms of [`UVW`] coordinates.
#[derive(Clone, Copy, Default)]
struct UV {
    u: f64,
    v: f64,
}

impl UV {
    fn from_xyz(xyz: XyzGeodetic, s_ha: f64, c_ha: f64, s_dec: f64, c_dec: f64) -> UV {
        UV {
            u: s_ha * xyz.x + c_ha * xyz.y,
            v: -s_dec * c_ha * xyz.x + s_dec * s_ha * xyz.y + c_dec * xyz.z,
        }
    }
}

impl Sub for UV {
    type Output = UV;

    fn sub(self, rhs: Self) -> Self::Output {
        UV {
            u: self.u - rhs.u,
            v: self.v - rhs.v,
        }
    }
}

#[derive(Error, Debug)]
pub(crate) enum PeelError {
    // #[error("No input data was given!")]
    // NoInputData,

    // #[error("{0}\n\nSupported combinations of file formats:\n{SUPPORTED_CALIBRATED_INPUT_FILE_COMBINATIONS}")]
    // InvalidDataInput(&'static str),

    // #[error("Multiple metafits files were specified: {0:?}\nThis is unsupported.")]
    // MultipleMetafits(Vec1<PathBuf>),

    // #[error("Multiple measurement sets were specified: {0:?}\nThis is currently unsupported.")]
    // MultipleMeasurementSets(Vec1<PathBuf>),

    // #[error("Multiple uvfits files were specified: {0:?}\nThis is currently unsupported.")]
    // MultipleUvfits(Vec1<PathBuf>),

    // #[error("No calibration output was specified. There must be at least one calibration solution file.")]
    // NoOutput,

    // #[error("No sky-model source list file supplied")]
    // NoSourceList,

    // #[error("Tried to create a beam object, but MWA dipole delay information isn't available!")]
    // NoDelays,

    // #[error(
    //     "The specified MWA dipole delays aren't valid; there should be 16 values between 0 and 32"
    // )]
    // BadDelays,

    // #[error("The data either contains no tiles or all tiles are flagged")]
    // NoTiles,

    // #[error("The data either contains no frequency channels or all channels are flagged")]
    // NoChannels,

    // #[error("The data either contains no timesteps or no timesteps are being used")]
    // NoTimesteps,

    // #[error("The number of specified sources was 0, or the size of the source list was 0")]
    // NoSources,

    // #[error("After vetoing sources, none were left. Decrease the veto threshold, or supply more sources")]
    // NoSourcesAfterVeto,

    // #[error("Duplicate timesteps were specified; this is invalid")]
    // DuplicateTimesteps,

    // #[error("Timestep {got} was specified but it isn't available; the last timestep is {last}")]
    // UnavailableTimestep { got: usize, last: usize },

    // #[error(
    //     "Cannot write visibilities to a file type '{ext}'. Supported formats are: {}", *crate::vis_io::write::VIS_OUTPUT_EXTENSIONS
    // )]
    // VisFileType { ext: String },

    // #[error(transparent)]
    // TileFlag(#[from] crate::context::InvalidTileFlag),

    // #[error("Cannot write calibration solutions to a file type '{ext}'.\nSupported formats are: {}", *crate::solutions::CAL_SOLUTION_EXTENSIONS)]
    // CalibrationOutputFile { ext: String },

    // #[error(transparent)]
    // ParsePfbFlavour(#[from] crate::pfb_gains::PfbParseError),

    // #[error("Error when parsing time average factor: {0}")]
    // ParseCalTimeAverageFactor(crate::unit_parsing::UnitParseError),

    // #[error("Error when parsing freq. average factor: {0}")]
    // ParseCalFreqAverageFactor(crate::unit_parsing::UnitParseError),

    // #[error("Calibration time average factor isn't an integer")]
    // CalTimeFactorNotInteger,

    // #[error("Calibration freq. average factor isn't an integer")]
    // CalFreqFactorNotInteger,

    // #[error("Calibration time resolution isn't a multiple of input data's: {out} seconds vs {inp} seconds")]
    // CalTimeResNotMultiple { out: f64, inp: f64 },

    // #[error("Calibration freq. resolution isn't a multiple of input data's: {out} Hz vs {inp} Hz")]
    // CalFreqResNotMultiple { out: f64, inp: f64 },

    // #[error("Calibration time average factor cannot be 0")]
    // CalTimeFactorZero,

    // #[error("Calibration freq. average factor cannot be 0")]
    // CalFreqFactorZero,

    // #[error("Error when parsing output vis. time average factor: {0}")]
    // ParseOutputVisTimeAverageFactor(crate::unit_parsing::UnitParseError),

    // #[error("Error when parsing output vis. freq. average factor: {0}")]
    // ParseOutputVisFreqAverageFactor(crate::unit_parsing::UnitParseError),

    // #[error("Output vis. time average factor isn't an integer")]
    // OutputVisTimeFactorNotInteger,

    // #[error("Output vis. freq. average factor isn't an integer")]
    // OutputVisFreqFactorNotInteger,

    // #[error("Output vis. time average factor cannot be 0")]
    // OutputVisTimeAverageFactorZero,

    // #[error("Output vis. freq. average factor cannot be 0")]
    // OutputVisFreqAverageFactorZero,

    // #[error("Output vis. time resolution isn't a multiple of input data's: {out} seconds vs {inp} seconds")]
    // OutputVisTimeResNotMultiple { out: f64, inp: f64 },

    // #[error("Output vis. freq. resolution isn't a multiple of input data's: {out} Hz vs {inp} Hz")]
    // OutputVisFreqResNotMultiple { out: f64, inp: f64 },

    // #[error("Error when parsing minimum UVW cutoff: {0}")]
    // ParseUvwMin(crate::unit_parsing::UnitParseError),

    // #[error("Error when parsing maximum UVW cutoff: {0}")]
    // ParseUvwMax(crate::unit_parsing::UnitParseError),

    // #[error("Array position specified as {pos:?}, not [<Longitude>, <Latitude>, <Height>]")]
    // BadArrayPosition { pos: Vec<f64> },

    // #[cfg(feature = "cuda")]
    // #[error("CUDA error: {0}")]
    // CudaError(String),

    // #[error(transparent)]
    // Glob(#[from] crate::glob::GlobError),

    // #[error(transparent)]
    // VisRead(#[from] crate::vis_io::read::VisReadError),

    // #[error(transparent)]
    // FileWrite(#[from] crate::vis_io::write::FileWriteError),

    // #[error(transparent)]
    // Veto(#[from] crate::srclist::VetoError),

    // #[error("Error when trying to read source list: {0}")]
    // SourceList(#[from] crate::srclist::ReadSourceListError),

    // #[error(transparent)]
    // Beam(#[from] crate::beam::BeamError),

    // #[error(transparent)]
    // IO(#[from] std::io::Error),
}
