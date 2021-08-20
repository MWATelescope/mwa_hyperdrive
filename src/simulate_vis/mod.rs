// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Generate sky-model visibilities from a sky-model source list.

mod error;
pub use error::SimulateVisError;

use std::collections::HashSet;
use std::ops::Deref;
use std::path::PathBuf;

use hifitime::Epoch;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use itertools::Either;
use log::{debug, info};
use ndarray::prelude::*;
use serde::Deserialize;
use structopt::StructOpt;

use crate::{
    data_formats::{metafits, uvfits::UvfitsWriter},
    glob::get_single_match_from_glob,
    model,
    time::mjd_to_epoch,
};
use mwa_hyperdrive_beam::{create_beam_object, Beam, Delays};
use mwa_hyperdrive_srclist::{read::read_source_list_file, ComponentList, SourceList};
use mwa_rust_core::{
    constants::{DS2R, SOLAR2SIDEREAL},
    mwalib,
    pos::xyz,
    time::epoch_as_gps_seconds,
    Jones, RADec, XyzGeodetic,
};
use mwalib::MetafitsContext;

#[cfg(feature = "cuda")]
use mwa_hyperdrive_cuda as cuda;
#[cfg(feature = "cuda")]
use mwa_hyperdrive_srclist::ComponentListFFI;

#[derive(StructOpt, Debug, Default, Deserialize)]
pub struct SimulateVisArgs {
    /// Path to the sky-model source list used for simulation.
    #[structopt(short, long)]
    source_list: String,

    /// Path to the metafits file.
    #[structopt(short, long, parse(from_str))]
    metafits: PathBuf,

    /// Path to the output visibilities file.
    #[structopt(long, default_value = "model.uvfits")]
    model_file: PathBuf,

    /// The phase centre right ascension \[degrees\]. If this is not specified,
    /// then the metafits phase/pointing centre is used.
    #[structopt(short, long)]
    ra: Option<f64>,

    /// The phase centre declination \[degrees\]. If this is not specified, then
    /// the metafits phase/pointing centre is used.
    #[structopt(short, long)]
    dec: Option<f64>,

    /// The total number of fine channels in the observation.
    #[structopt(long, default_value = "384")]
    num_fine_channels: usize,

    /// The fine-channel resolution \[kHz\].
    #[structopt(short, long, default_value = "80")]
    freq_res: f64,

    /// The middle frequency of the simulation \[MHz\]. If this is not
    /// specified, then the middle frequency specified in the metafits is used.
    #[structopt(long)]
    middle_freq: Option<f64>,

    /// The number of time steps used from the metafits epoch.
    #[structopt(short, long, default_value = "14")]
    time_steps: usize,

    /// The time resolution \[seconds\].
    #[structopt(long, default_value = "8.0")]
    time_res: f64,

    /// Should we use a beam? Default is to use the FEE beam.
    #[structopt(long)]
    no_beam: bool,

    /// The path to the HDF5 MWA FEE beam file. If not specified, this must be
    /// provided by the MWA_BEAM_FILE environment variable.
    #[structopt(long)]
    beam_file: Option<PathBuf>,

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
    model_file: PathBuf,

    /// The phase centre.
    phase_centre: RADec,

    /// The fine frequency channel frequencies \[Hz\].
    fine_chan_freqs: Vec<f64>,

    /// Timesteps to be simulated.
    timesteps: Vec<Epoch>,

    /// Interface to beam code.    
    beam: Box<dyn Beam>,
}

impl SimVisParams {
    /// Convert arguments into parameters.
    fn new(args: SimulateVisArgs) -> Result<Self, SimulateVisError> {
        debug!("{:#?}", &args);

        // Expose all the struct fields to ensure they're all used.
        let SimulateVisArgs {
            source_list,
            metafits,
            model_file,
            ra,
            dec,
            num_fine_channels,
            freq_res,
            middle_freq,
            time_steps,
            time_res,
            no_beam,
            beam_file,
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
        debug!("Using phase centre {}", phase_centre);

        // Get the fine channel frequencies.
        if num_fine_channels == 0 {
            return Err(SimulateVisError::FineChansZero);
        }
        if freq_res < f64::EPSILON {
            return Err(SimulateVisError::FineChansWidthTooSmall);
        }
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
        if fine_chan_freqs.len() == 1 {
            debug!(
                "Only fine channel freq: {} MHz",
                *fine_chan_freqs.first().unwrap() / 1e6
            );
        } else {
            debug!(
                "First fine channel freq: {} MHz",
                *fine_chan_freqs.first().unwrap() / 1e6
            );
            debug!(
                "Last fine channel freq:  {} MHz",
                *fine_chan_freqs.last().unwrap() / 1e6
            );
        }

        // Populate the timesteps.
        let timesteps = {
            let mut timesteps = Vec::with_capacity(time_steps);
            let start = mjd_to_epoch(metafits.sched_start_mjd);
            for i in 0..time_steps {
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
            [t] => debug!("Only timestep (GPS): {:.2}", epoch_as_gps_seconds(*t)),
            [t0, .., tn] => {
                debug!("First timestep (GPS): {:.2}", epoch_as_gps_seconds(*t0));
                debug!("Last timestep  (GPS): {:.2}", epoch_as_gps_seconds(*tn));
            }
        }

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
        let delays = metafits::get_true_delays(&metafits);
        let beam = create_beam_object(no_beam, Delays::Available(delays), beam_file)?;

        info!("Writing the sky model to {}", model_file.display());

        Ok(Self {
            source_list,
            metafits,
            model_file,
            phase_centre,
            fine_chan_freqs,
            timesteps,
            beam,
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
        info!("Generating sky model visibilities on the GPU");
    }

    let params = SimVisParams::new(args)?;

    if dry_run {
        info!("Dry run -- exiting now.");
        return Ok(());
    }

    // Get the geodetic XYZ coordinates of each of the MWA tiles.
    let xyzs = XyzGeodetic::get_tiles_mwa(&params.metafits);
    let num_cross_correlation_baselines = params.metafits.num_baselines - params.metafits.num_ants;

    // Construct our visibilities array. This will be re-used for each timestep
    // before it written to disk.
    let vis_shape = (
        num_cross_correlation_baselines,
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
    let mut output_writer = UvfitsWriter::new(
        &params.model_file,
        params.timesteps.len(),
        num_cross_correlation_baselines,
        params.fine_chan_freqs.len(),
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
    )?;

    // Split the sky-model components (for efficiency). Use an Either to hold
    // one type or the other (the types differ between CPU and GPU code).
    let sky_model_comps = if cpu {
        Either::Left(ComponentList::new(
            &params.source_list,
            &params.fine_chan_freqs,
            params.phase_centre,
        ))
    } else {
        #[cfg(feature = "cuda")]
        {
            Either::Right(ComponentListFFI::new(
                params.source_list,
                &params.fine_chan_freqs,
                params.phase_centre,
            ))
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
        let lst_diff = (timestep - params.timesteps[0]).in_seconds();
        let lst = params.metafits.lst_rad + lst_diff * SOLAR2SIDEREAL * DS2R;
        let hadec_phase_centre = params.phase_centre.to_hadec(lst);
        let uvws = xyz::xyzs_to_uvws(&xyzs, hadec_phase_centre);

        if cpu {
            model::model_timestep(
                vis_model.view_mut(),
                sky_model_comps.as_ref().unwrap_left(),
                params.beam.deref(),
                lst,
                &xyzs,
                &uvws,
                &params.fine_chan_freqs,
            )?;
        } else {
            #[cfg(feature = "cuda")]
            {
                let sky_model_comps = sky_model_comps.as_ref().unwrap_right();

                let shapelet_power_law_uvs = if sky_model_comps.shapelet_power_law_radecs.is_empty()
                {
                    None
                } else {
                    Some(ComponentListFFI::get_shapelet_uvs(
                        &sky_model_comps.shapelet_power_law_radecs,
                        lst,
                        &xyzs,
                    ))
                };
                let shapelet_list_uvs = if sky_model_comps.shapelet_list_radecs.is_empty() {
                    None
                } else {
                    Some(ComponentListFFI::get_shapelet_uvs(
                        &sky_model_comps.shapelet_list_radecs,
                        lst,
                        &xyzs,
                    ))
                };
                unsafe {
                    let (points, gaussians, shapelets) = sky_model_comps.to_c_types(
                        shapelet_power_law_uvs.as_ref().map(|o| o.as_ptr()),
                        shapelet_list_uvs.as_ref().map(|o| o.as_ptr()),
                    );
                    let cuda_result = cuda::model_timestep(
                        num_cross_correlation_baselines,
                        params.fine_chan_freqs.len(),
                        uvws.as_ptr() as _,
                        params.fine_chan_freqs.as_ptr(),
                        &points,
                        &gaussians,
                        &shapelets,
                        crate::shapelets::SHAPELET_BASIS_VALUES.as_ptr(),
                        crate::shapelets::SBF_L,
                        crate::shapelets::SBF_N,
                        crate::shapelets::SBF_C,
                        crate::shapelets::SBF_DX,
                        vis_model.as_mut_ptr() as _,
                    );
                    // TODO: Handle cuda_result.
                    if cuda_result != 0 {
                        panic!(
                            "cuda::model_timestep exited with CUDA error code {}",
                            cuda_result
                        );
                    }
                }
            }
        }
        // Write the visibilities out.
        output_writer.write_from_vis(
            vis_model.view(),
            Array2::ones(vis_model.dim()).view(),
            &uvws,
            timestep,
            params.fine_chan_freqs.len(),
            &HashSet::new(),
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
    output_writer.write_uvfits_antenna_table(&names, &xyzs)?;
    info!(
        "Finished writing sky model to {}",
        &params.model_file.display()
    );

    Ok(())
}
