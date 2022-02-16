// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Generate sky-model visibilities from a sky-model source list.

mod error;
pub use error::SimulateVisError;

use std::collections::{HashMap, HashSet};
use std::ops::Deref;
use std::path::PathBuf;

use clap::Parser;
use hifitime::Epoch;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use log::{debug, info};
use marlu::{
    constants::{MWA_LAT_RAD, MWA_LONG_RAD},
    Jones, RADec, XyzGeodetic,
};
use mwalib::MetafitsContext;
use ndarray::prelude::*;
use serde::Deserialize;

use crate::{
    data_formats::{get_dipole_delays, get_dipole_gains, UvfitsWriter},
    glob::get_single_match_from_glob,
    model,
};
use mwa_hyperdrive_beam::{create_fee_beam_object, create_no_beam_object, Beam, Delays};
use mwa_hyperdrive_common::{clap, hifitime, indicatif, log, marlu, mwalib, ndarray};
use mwa_hyperdrive_srclist::{read::read_source_list_file, SourceList};

#[derive(Parser, Debug, Default, Deserialize)]
pub struct SimulateVisArgs {
    /// Path to the sky-model source list used for simulation.
    #[clap(short, long)]
    source_list: String,

    /// Path to the metafits file.
    #[clap(short, long, parse(from_str))]
    metafits: PathBuf,

    /// Path to the output visibilities file.
    #[clap(short = 'o', long, default_value = "model.uvfits")]
    output_model_file: PathBuf,

    /// The phase centre right ascension [degrees]. If this is not specified,
    /// then the metafits phase/pointing centre is used.
    #[clap(short, long)]
    ra: Option<f64>,

    /// The phase centre declination [degrees]. If this is not specified, then
    /// the metafits phase/pointing centre is used.
    #[clap(short, long)]
    dec: Option<f64>,

    /// The total number of fine channels in the observation.
    #[clap(short = 'c', long, default_value = "384")]
    num_fine_channels: usize,

    /// The fine-channel resolution [kHz].
    #[clap(short, long, default_value = "80")]
    freq_res: f64,

    /// The middle frequency of the simulation [MHz]. If this is not specified,
    /// then the middle frequency specified in the metafits is used.
    #[clap(long)]
    middle_freq: Option<f64>,

    /// The number of time steps used from the metafits epoch.
    #[clap(short = 't', long, default_value = "14")]
    num_timesteps: usize,

    /// The time resolution [seconds].
    #[clap(long, default_value = "8")]
    time_res: f64,

    /// Should we use a beam? Default is to use the FEE beam.
    #[clap(long)]
    no_beam: bool,

    /// The path to the HDF5 MWA FEE beam file. If not specified, this must be
    /// provided by the MWA_BEAM_FILE environment variable.
    #[clap(long)]
    beam_file: Option<PathBuf>,

    /// Pretend that all MWA dipoles are alive and well, ignoring whatever is in
    /// the metafits file.
    #[clap(long)]
    unity_dipole_gains: bool,

    /// Specify the MWA dipoles delays, ignoring whatever is in the metafits
    /// file.
    #[clap(long, multiple_values(true))]
    dipole_delays: Option<Vec<u32>>,

    /// Don't attempt to convert "list" flux densities to power law flux
    /// densities. See for more info:
    /// https://github.com/MWATelescope/mwa_hyperdrive/wiki/Source-lists
    #[clap(long)]
    dont_convert_lists: bool,

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

/// Parameters needed to do sky-model visibility simulation.
struct SimVisParams {
    /// Sky-model source list.
    source_list: SourceList,

    /// mwalib metafits context
    metafits: MetafitsContext,

    /// The output visibilities file.
    output_model_file: PathBuf,

    /// The phase centre.
    phase_centre: RADec,

    /// The fine frequency channel frequencies \[Hz\].
    fine_chan_freqs: Vec<f64>,

    /// The [XyzGeodetic] positions of the tiles.
    xyzs: Vec<XyzGeodetic>,

    /// A map from baseline index to the baseline's constituent tiles.
    baseline_to_tile_map: HashMap<usize, (usize, usize)>,

    /// Flagged tiles.
    flagged_tiles: Vec<usize>,

    /// Timestamps to be simulated.
    timestamps: Vec<Epoch>,

    /// Interface to beam code.    
    beam: Box<dyn Beam>,

    /// The Earth latitude location of the interferometer \[radians\].
    array_latitude: f64,

    /// The Earth longitude location of the interferometer \[radians\].
    array_longitude: f64,
}

impl SimVisParams {
    /// Convert arguments into parameters.
    fn new(args: SimulateVisArgs) -> Result<SimVisParams, SimulateVisError> {
        debug!("{:#?}", &args);

        // Expose all the struct fields to ensure they're all used.
        let SimulateVisArgs {
            source_list,
            metafits,
            output_model_file,
            ra,
            dec,
            num_fine_channels,
            freq_res,
            middle_freq,
            num_timesteps,
            time_res,
            no_beam,
            beam_file,
            dipole_delays,
            unity_dipole_gains,
            dont_convert_lists,
            filter_points,
            filter_gaussians,
            filter_shapelets,
        } = args;

        // Read the metafits file with mwalib.
        // TODO: Allow the user to specify the mwa_version.
        let metafits = mwalib::MetafitsContext::new(&metafits, None)?;

        // Get the phase centre.
        let phase_centre = match (ra, dec, &metafits) {
            (Some(ra), Some(dec), _) => {
                // Verify that the input coordinates are sensible.
                if !(0.0..=360.0).contains(&ra) {
                    return Err(SimulateVisError::RaInvalid);
                }
                if !(-90.0..=90.0).contains(&dec) {
                    return Err(SimulateVisError::DecInvalid);
                }
                RADec::new_degrees(ra, dec)
            }
            (Some(_), None, _) => return Err(SimulateVisError::OnlyOneRAOrDec),
            (None, Some(_), _) => return Err(SimulateVisError::OnlyOneRAOrDec),
            (None, None, m) => {
                // The phase centre in a metafits file may not be present. If not,
                // we have to use the pointing centre.
                match (m.ra_phase_center_degrees, m.dec_phase_center_degrees) {
                    (Some(ra), Some(dec)) => RADec::new_degrees(ra, dec),
                    (None, None) => {
                        RADec::new_degrees(m.ra_tile_pointing_degrees, m.dec_tile_pointing_degrees)
                    }
                    _ => unreachable!(),
                }
            }
        };
        info!("Using phase centre {}", phase_centre);

        // Get the fine channel frequencies.
        if num_fine_channels == 0 {
            return Err(SimulateVisError::FineChansZero);
        }
        if freq_res < f64::EPSILON {
            return Err(SimulateVisError::FineChansWidthTooSmall);
        }
        info!("Number of fine channels: {}", num_fine_channels);
        info!("Fine-channel width:      {} kHz", freq_res);
        let middle_freq = middle_freq.unwrap_or(metafits.centre_freq_hz as _);
        let fine_chan_freqs = {
            let half_num_fine_chans = num_fine_channels as f64 / 2.0;
            let freq_res = freq_res * 1000.0; // kHz -> Hz
            let mut fine_chan_freqs = Vec::with_capacity(num_fine_channels);
            for i in 0..num_fine_channels {
                fine_chan_freqs
                    .push(middle_freq - half_num_fine_chans * freq_res + freq_res * i as f64);
            }
            fine_chan_freqs
        };
        match fine_chan_freqs.as_slice() {
            [] => unreachable!(), // Handled above.
            [f] => info!("Only fine-channel freq: {} MHz", f / 1e6),
            [f_0, .., f_n] => {
                info!("First fine-channel freq: {} MHz", f_0 / 1e6);
                info!("Last fine-channel freq:  {} MHz", f_n / 1e6);
            }
        }

        // Populate the timestamps.
        let timestamps = {
            let mut timestamps = Vec::with_capacity(num_timesteps);
            let start = Epoch::from_gpst_seconds(metafits.sched_start_gps_time_ms as f64 / 1e3);
            for i in 0..num_timesteps {
                timestamps.push(
                    start
                        + hifitime::Duration::from_f64(
                            time_res * i as f64,
                            hifitime::TimeUnit::Second,
                        ),
                );
            }
            timestamps
        };
        match timestamps.as_slice() {
            [] => return Err(SimulateVisError::ZeroTimeSteps),
            [t] => info!("Only timestep (GPS): {:.2}", t.as_gpst_seconds()),
            [t0, .., tn] => {
                info!("First timestep (GPS): {:.2}", t0.as_gpst_seconds());
                info!("Last timestep  (GPS): {:.2}", tn.as_gpst_seconds());
            }
        }

        // Get the geodetic XYZ coordinates of each of the MWA tiles.
        let xyzs = XyzGeodetic::get_tiles_mwa(&metafits);

        // Prepare a map between baselines and their constituent tiles.
        // TODO: Utilise tile flags.
        let flagged_tiles: Vec<usize> = vec![];
        let baseline_to_tile_map = {
            let mut baseline_to_tile_map = HashMap::new();
            let mut bl = 0;
            for tile1 in 0..metafits.num_ants {
                if flagged_tiles.contains(&tile1) {
                    continue;
                }
                for tile2 in tile1 + 1..metafits.num_ants {
                    if flagged_tiles.contains(&tile2) {
                        continue;
                    }
                    baseline_to_tile_map.insert(bl, (tile1, tile2));
                    bl += 1;
                }
            }
            baseline_to_tile_map
        };

        // Treat the specified source list as file path. Does it exist? Then use it.
        // Otherwise, treat the specified source list as a glob and attempt to find
        // a single file with it.
        let sl_pb = PathBuf::from(&source_list);
        let sl_pb = if sl_pb.exists() {
            sl_pb
        } else {
            get_single_match_from_glob(&source_list)?
        };
        // Read the source list.
        // TODO: Allow the user to specify a source list type.
        let source_list = match read_source_list_file(&sl_pb, None) {
            Ok((sl, sl_type)) => {
                debug!("Successfully parsed {}-style source list", sl_type);
                sl
            }
            Err(e) => return Err(SimulateVisError::from(e)),
        };
        let counts = source_list.get_counts();
        debug!(
            "Found {} points, {} gaussians, {} shapelets",
            counts.0, counts.1, counts.2
        );

        // Apply any filters.
        let mut source_list = if filter_points || filter_gaussians || filter_shapelets {
            let sl = source_list.filter(filter_points, filter_gaussians, filter_shapelets);
            let counts = sl.get_counts();
            debug!(
                "After filtering, there are {} points, {} gaussians, {} shapelets",
                counts.0, counts.1, counts.2
            );
            sl
        } else {
            source_list
        };
        if !dont_convert_lists {
            // Convert flux density lists into power laws.
            source_list
                .values_mut()
                .flat_map(|src| &mut src.components)
                .for_each(|comp| {
                    comp.flux_type.convert_list_to_power_law();
                });
        }
        let beam = if no_beam {
            create_no_beam_object(xyzs.len())
        } else {
            create_fee_beam_object(
                beam_file,
                metafits.num_ants,
                match dipole_delays {
                    Some(d) => {
                        if d.len() != 16 || d.iter().any(|&v| v > 32) {
                            return Err(SimulateVisError::BadDelays);
                        }
                        Delays::Partial(d)
                    }
                    None => Delays::Full(get_dipole_delays(&metafits)),
                },
                match unity_dipole_gains {
                    true => None,
                    false => Some(get_dipole_gains(&metafits)),
                },
            )?
        };

        info!("Writing the sky model to {}", output_model_file.display());

        Ok(SimVisParams {
            source_list,
            metafits,
            output_model_file,
            phase_centre,
            fine_chan_freqs,
            xyzs,
            baseline_to_tile_map,
            flagged_tiles,
            timestamps,
            beam,
            array_latitude: MWA_LAT_RAD,
            array_longitude: MWA_LONG_RAD,
        })
    }
}

/// Simulate sky-model visibilities from a sky-model source list.
pub fn simulate_vis(
    args: SimulateVisArgs,
    #[cfg(feature = "cuda")] use_cpu_for_modelling: bool,
    dry_run: bool,
) -> Result<(), SimulateVisError> {
    // Witchcraft to allow this code to be used with or without CUDA support
    // compiled.
    #[cfg(not(feature = "cuda"))]
    let use_cpu_for_modelling = true;

    if use_cpu_for_modelling {
        info!("Generating sky model visibilities on the CPU");
    } else {
        // TODO: Display GPU info.
        #[cfg(not(feature = "cuda-single"))]
        info!("Generating sky model visibilities on the GPU (double precision)");
        #[cfg(feature = "cuda-single")]
        info!("Generating sky model visibilities on the GPU (single precision)");
    }

    let params = SimVisParams::new(args)?;

    if dry_run {
        info!("Dry run -- exiting now.");
        return Ok(());
    }

    // Construct our visibilities array. This will be re-used for each timestep
    // before it written to disk.
    let vis_shape = (
        params.baseline_to_tile_map.len(),
        params.fine_chan_freqs.len(),
    );
    let mut vis_model: Array2<Jones<f32>> = Array2::from_elem(vis_shape, Jones::default());
    debug!(
        "Shape of model array: ({} baselines, {} channels; {} MiB)",
        vis_shape.0,
        vis_shape.1,
        vis_shape.0 * vis_shape.1 * std::mem::size_of_val(&vis_model[[0, 0]])
        // 1024 * 1024 == 1 MiB.
        / 1024 / 1024
    );

    // Prepare the output visibilities file.
    let fine_chan_flags = HashSet::new();
    let mut output_writer = UvfitsWriter::new(
        &params.output_model_file,
        params.timestamps.len(),
        params.baseline_to_tile_map.len(),
        params.fine_chan_freqs.len(),
        false,
        *params.timestamps.first().unwrap(),
        if params.fine_chan_freqs.len() == 1 {
            None
        } else {
            Some(params.fine_chan_freqs[1] - params.fine_chan_freqs[0])
        },
        params.fine_chan_freqs[params.fine_chan_freqs.len() / 2],
        params.phase_centre,
        Some(&format!(
            "Simulated visibilities for obsid {}",
            params.metafits.obs_id
        )),
        &params.baseline_to_tile_map,
        &fine_chan_flags,
    )?;

    // Create a "modeller" object.
    let modeller = unsafe {
        model::new_sky_modeller(
            use_cpu_for_modelling,
            params.beam.deref(),
            &params.source_list,
            &params.xyzs,
            &params.fine_chan_freqs,
            &params.flagged_tiles,
            params.phase_centre,
            params.array_longitude,
            params.array_latitude,
            // TODO: Allow the user to turn off precession.
            true,
        )
    }?;

    // Progress bar.
    let model_progress = ProgressBar::new(params.timestamps.len() as _)
        .with_style(
            ProgressStyle::default_bar()
                .template(
                    "{msg}: [{wide_bar:.blue}] {pos:2}/{len:2} timesteps ({elapsed_precise}<{eta_precise})",
                )
                .progress_chars("=> "),
        )
        .with_position(0)
        .with_message("Sky modelling");
    model_progress.set_draw_target(ProgressDrawTarget::stdout());
    model_progress.tick();

    // Generate the visibilities.
    for &timestep in params.timestamps.iter() {
        let uvws = modeller.model_timestep(vis_model.view_mut(), timestep)?;
        // Write the visibilities out.
        output_writer.write_cross_timestep_vis(
            vis_model.view(),
            Array2::ones(vis_model.dim()).view(),
            &uvws,
            timestep,
        )?;

        // Clear the visibilities before re-using the buffer.
        vis_model.fill(Jones::default());

        model_progress.inc(1);
    }
    model_progress.finish_with_message("Finished generating sky model");

    // Finalise writing the model.
    let names: Vec<&str> = params
        .metafits
        .rf_inputs
        .iter()
        .filter(|rf| rf.pol == mwalib::Pol::X)
        .map(|rf| rf.tile_name.as_str())
        .collect();
    output_writer.write_uvfits_antenna_table(&names, &params.xyzs)?;
    info!(
        "Finished writing sky model to {}",
        &params.output_model_file.display()
    );

    Ok(())
}
