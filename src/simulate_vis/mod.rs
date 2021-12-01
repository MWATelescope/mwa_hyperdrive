// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Generate sky-model visibilities from a sky-model source list.

mod error;
pub use error::SimulateVisError;

use std::collections::{HashMap, HashSet};
use std::ops::Deref;
use std::path::PathBuf;

use hifitime::Epoch;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use itertools::Either;
use log::{debug, info};
use mwa_rust_core::{
    constants::{MWA_LAT_RAD, MWA_LONG_RAD},
    mwalib,
    time::epoch_as_gps_seconds,
    Jones, RADec, XyzGeodetic,
};
use mwalib::MetafitsContext;
use ndarray::prelude::*;
use serde::Deserialize;
use structopt::StructOpt;

use crate::{
    data_formats::{metafits, uvfits::UvfitsWriter},
    glob::get_single_match_from_glob,
    model,
    precession::precess_time,
    time::mjd_to_epoch,
};
use mwa_hyperdrive_beam::{create_fee_beam_object, create_no_beam_object, Beam, Delays};
use mwa_hyperdrive_srclist::{read::read_source_list_file, ComponentList, SourceList};

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        use mwa_hyperdrive_cuda as cuda;
        use cuda::modeller::SkyModellerCuda;
    }
}

#[derive(StructOpt, Debug, Default, Deserialize)]
pub struct SimulateVisArgs {
    /// Path to the sky-model source list used for simulation.
    #[structopt(short, long)]
    source_list: String,

    /// Path to the metafits file.
    #[structopt(short, long, parse(from_str))]
    metafits: PathBuf,

    /// Path to the output visibilities file.
    #[structopt(short = "o", long, default_value = "model.uvfits")]
    output_model_file: PathBuf,

    /// The phase centre right ascension [degrees]. If this is not specified,
    /// then the metafits phase/pointing centre is used.
    #[structopt(short, long)]
    ra: Option<f64>,

    /// The phase centre declination [degrees]. If this is not specified, then
    /// the metafits phase/pointing centre is used.
    #[structopt(short, long)]
    dec: Option<f64>,

    /// The total number of fine channels in the observation.
    #[structopt(short = "c", long, default_value = "384")]
    num_fine_channels: usize,

    /// The fine-channel resolution [kHz].
    #[structopt(short, long, default_value = "80")]
    freq_res: f64,

    /// The middle frequency of the simulation [MHz]. If this is not specified,
    /// then the middle frequency specified in the metafits is used.
    #[structopt(long)]
    middle_freq: Option<f64>,

    /// The number of time steps used from the metafits epoch.
    #[structopt(short = "t", long, default_value = "14")]
    num_timesteps: usize,

    /// The time resolution [seconds].
    #[structopt(long, default_value = "8")]
    time_res: f64,

    /// Should we use a beam? Default is to use the FEE beam.
    #[structopt(long)]
    no_beam: bool,

    /// The path to the HDF5 MWA FEE beam file. If not specified, this must be
    /// provided by the MWA_BEAM_FILE environment variable.
    #[structopt(long)]
    beam_file: Option<PathBuf>,

    /// Pretend that all MWA dipoles are alive and well, ignoring whatever is in
    /// the metafits file.
    #[structopt(long)]
    unity_dipole_gains: bool,

    /// Specify the MWA dipoles delays, ignoring whatever is in the metafits
    /// file.
    #[structopt(long)]
    dipole_delays: Option<Vec<u32>>,

    /// Don't attempt to convert "list" flux densities to power law flux
    /// densities. See for more info:
    /// https://github.com/MWATelescope/mwa_hyperdrive/wiki/Source-lists
    #[structopt(long)]
    dont_convert_lists: bool,

    /// Don't include point components from the input sky model.
    #[structopt(long)]
    filter_points: bool,

    /// Don't include Gaussian components from the input sky model.
    #[structopt(long)]
    filter_gaussians: bool,

    /// Don't include shapelet components from the input sky model.
    #[structopt(long)]
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
    tile_flags: HashSet<usize>,

    /// Timesteps to be simulated.
    timesteps: Vec<Epoch>,

    /// Interface to beam code.    
    beam: Box<dyn Beam>,

    /// The Earth latitude location of the interferometer \[radians\].
    array_latitude_rad: f64,

    /// The Earth longitude location of the interferometer \[radians\].
    array_longitude_rad: f64,
}

impl SimVisParams {
    /// Convert arguments into parameters.
    fn new(args: SimulateVisArgs) -> Result<Self, SimulateVisError> {
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
        // TODO: Allow the user to specify the MWAVersion.
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

        // Populate the timesteps.
        let timesteps = {
            let mut timesteps = Vec::with_capacity(num_timesteps);
            let start = mjd_to_epoch(metafits.sched_start_mjd);
            for i in 0..num_timesteps {
                timesteps.push(
                    start
                        + hifitime::Duration::from_f64(
                            time_res * i as f64,
                            hifitime::TimeUnit::Second,
                        ),
                );
            }
            timesteps
        };
        match timesteps.as_slice() {
            [] => return Err(SimulateVisError::ZeroTimeSteps),
            [t] => info!("Only timestep (GPS): {:.2}", epoch_as_gps_seconds(*t)),
            [t0, .., tn] => {
                info!("First timestep (GPS): {:.2}", epoch_as_gps_seconds(*t0));
                info!("Last timestep  (GPS): {:.2}", epoch_as_gps_seconds(*tn));
            }
        }

        // Get the geodetic XYZ coordinates of each of the MWA tiles.
        let xyzs = XyzGeodetic::get_tiles_mwa(&metafits);

        // Prepare a map between baselines and their constituent tiles.
        // TODO: Utilise tile flags.
        let tile_flags: HashSet<usize> = HashSet::new();
        let baseline_to_tile_map = {
            let mut baseline_to_tile_map = HashMap::new();
            let mut bl = 0;
            for tile1 in 0..metafits.num_ants {
                if tile_flags.contains(&tile1) {
                    continue;
                }
                for tile2 in tile1 + 1..metafits.num_ants {
                    if tile_flags.contains(&tile2) {
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
            Err(e) => {
                eprintln!("Error when trying to read source list:");
                return Err(SimulateVisError::from(e));
            }
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
                    Some(d) => Delays::Partial(d),
                    None => Delays::Full(metafits::get_dipole_delays(&metafits)),
                },
                match unity_dipole_gains {
                    true => None,
                    false => Some(metafits::get_dipole_gains(&metafits)),
                },
            )?
        };

        info!("Writing the sky model to {}", output_model_file.display());

        Ok(Self {
            source_list,
            metafits,
            output_model_file,
            phase_centre,
            fine_chan_freqs,
            xyzs,
            baseline_to_tile_map,
            tile_flags,
            timesteps,
            beam,
            array_latitude_rad: MWA_LAT_RAD,
            array_longitude_rad: MWA_LONG_RAD,
        })
    }
}

/// Simulate sky-model visibilities from a sky-model source list.
pub fn simulate_vis(
    args: SimulateVisArgs,
    #[cfg(feature = "cuda")] cpu: bool,
    dry_run: bool,
) -> Result<(), SimulateVisError> {
    // Witchcraft to allow this code to be used with or without CUDA support
    // compiled.
    #[cfg(not(feature = "cuda"))]
    let cpu = true;

    if cpu {
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
        params.timesteps.len(),
        params.baseline_to_tile_map.len(),
        params.fine_chan_freqs.len(),
        false,
        *params.timesteps.first().unwrap(),
        if params.fine_chan_freqs.len() == 1 {
            None
        } else {
            Some(params.fine_chan_freqs[1] - params.fine_chan_freqs[0])
        },
        params.fine_chan_freqs[params.fine_chan_freqs.len() / 2],
        params.fine_chan_freqs.len() / 2,
        params.phase_centre,
        Some(&format!(
            "Simulated visibilities for obsid {}",
            params.metafits.obs_id
        )),
        &params.baseline_to_tile_map,
        &fine_chan_flags,
    )?;

    // Create a "modeller" object. Use an Either to hold one type or the other
    // (the types differ between CPU and GPU code).
    let modeller = if cpu {
        Either::Left(ComponentList::new(
            &params.source_list,
            &params.fine_chan_freqs,
            params.phase_centre,
        ))
    } else {
        #[cfg(feature = "cuda")]
        unsafe {
            Either::Right(SkyModellerCuda::new(
                params.beam.deref(),
                &params.source_list,
                &params.fine_chan_freqs,
                &params.xyzs,
                &params.tile_flags,
                params.phase_centre,
                params.array_latitude_rad,
                &crate::shapelets::SHAPELET_BASIS_VALUES,
                crate::shapelets::SBF_L,
                crate::shapelets::SBF_N,
                crate::shapelets::SBF_C,
                crate::shapelets::SBF_DX,
            )?)
        }

        // It doesn't matter what goes in Right when we're not using CUDA.
        #[cfg(not(feature = "cuda"))]
        Either::Right(0)
    };

    // Progress bar.
    let model_progress = ProgressBar::new(params.timesteps.len() as _)
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
    for &timestep in params.timesteps.iter() {
        let precession_info = precess_time(
            params.phase_centre,
            timestep,
            params.array_longitude_rad,
            params.array_latitude_rad,
        );
        // Apply precession to the tile XYZ positions.
        let precessed_tile_xyzs = precession_info.precess_xyz_parallel(&params.xyzs);
        let uvws = xyzs_to_cross_uvws_parallel(
            &precessed_tile_xyzs,
            params.phase_centre.to_hadec(precession_info.lmst_j2000),
        );

        if cpu {
            model::model_timestep(
                vis_model.view_mut(),
                modeller.as_ref().unwrap_left(),
                params.beam.deref(),
                precession_info.lmst_j2000,
                &params.xyzs,
                &uvws,
                &params.fine_chan_freqs,
                &params.baseline_to_tile_map,
            )?;
        } else {
            #[cfg(feature = "cuda")]
            unsafe {
                modeller.as_ref().unwrap_right().model_timestep(
                    vis_model.view_mut(),
                    precession_info.lmst_j2000,
                    &uvws,
                )?;
            }
        }
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
        .map(|rf| rf.tile_name.as_str())
        .collect();
    output_writer.write_uvfits_antenna_table(&names, &params.xyzs)?;
    info!(
        "Finished writing sky model to {}",
        &params.output_model_file.display()
    );

    Ok(())
}

// TODO: Have in mwa_rust_core
/// Convert [XyzGeodetic] tile coordinates to [UVW] baseline coordinates without
/// having to form [XyzGeodetic] baselines first. This function performs
/// calculations in parallel. Cross-correlation baselines only.
fn xyzs_to_cross_uvws_parallel(
    xyzs: &[mwa_rust_core::XyzGeodetic],
    phase_centre: mwa_rust_core::HADec,
) -> Vec<mwa_rust_core::UVW> {
    use rayon::prelude::*;

    let (s_ha, c_ha) = phase_centre.ha.sin_cos();
    let (s_dec, c_dec) = phase_centre.dec.sin_cos();
    // Get a UVW for each tile.
    let tile_uvws: Vec<mwa_rust_core::UVW> = xyzs
        .par_iter()
        .map(|&xyz| mwa_rust_core::UVW::from_xyz_inner(xyz, s_ha, c_ha, s_dec, c_dec))
        .collect();
    // Take the difference of every pair of UVWs.
    let num_tiles = xyzs.len();
    let num_baselines = (num_tiles * (num_tiles - 1)) / 2;
    (0..num_baselines)
        .into_par_iter()
        .map(|i_bl| {
            let (i, j) = mwa_rust_core::math::cross_correlation_baseline_to_tiles(num_tiles, i_bl);
            tile_uvws[i] - tile_uvws[j]
        })
        .collect()
}
