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
use log::{debug, info};
use ndarray::prelude::*;
use serde::Deserialize;
use structopt::StructOpt;

use crate::{
    constants::*,
    data_formats::{metafits, uvfits::UvfitsWriter},
    glob::get_single_match_from_glob,
    model, shapelets,
    time::mjd_to_epoch,
};
use mwa_hyperdrive_core::{beam::Delays, *};
#[cfg(feature = "cuda")]
use mwa_hyperdrive_cuda as cuda;
use mwa_hyperdrive_srclist::{read::read_source_list_file, ComponentList};
use mwalib::MetafitsContext;

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
}

/// Parameters needed to do sky-model visibility simulation.
struct SimVisParams {
    /// Sky-model source components.
    sky_model_comps: ComponentList,

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

        // Read the metafits file with mwalib.
        // TODO: Allow the user to specify the MWAVersion.
        let metafits =
            mwalib::MetafitsContext::new(&args.metafits, mwalib::MWAVersion::CorrLegacy)?;

        // Get the phase centre.
        let phase_centre = match (args.ra, args.dec, &metafits) {
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
        if args.num_fine_channels == 0 {
            return Err(SimulateVisError::FineChansZero);
        }
        if args.freq_res < f64::EPSILON {
            return Err(SimulateVisError::FineChansWidthTooSmall);
        }
        let middle_freq = args.middle_freq.unwrap_or(metafits.centre_freq_hz as _);
        let fine_chan_freqs = {
            let half_num_fine_chans = (args.num_fine_channels / 2) as f64;
            let freq_res = args.freq_res * 1000.0; // kHz -> Hz
            let mut fine_chan_freqs = Vec::with_capacity(args.num_fine_channels);
            for i in 0..args.num_fine_channels {
                fine_chan_freqs
                    .push(middle_freq - half_num_fine_chans * freq_res + freq_res * i as f64);
            }
            fine_chan_freqs
        };
        debug!(
            "First fine channel freq: {} MHz",
            *fine_chan_freqs.first().unwrap() / 1e6
        );
        debug!(
            "Last fine channel freq:  {} MHz",
            *fine_chan_freqs.last().unwrap() / 1e6
        );

        // Populate the timesteps.
        if args.time_steps == 0 {
            return Err(SimulateVisError::TimeStepsInvalid);
        }
        let timesteps = {
            let mut timesteps = Vec::with_capacity(args.time_steps);
            let start = mjd_to_epoch(metafits.sched_start_mjd);
            for i in 0..args.time_steps {
                timesteps.push(
                    start
                        + hifitime::Duration::from_f64(
                            args.time_res * i as f64,
                            hifitime::TimeUnit::Second,
                        ),
                );
            }
            timesteps
        };
        debug!(
            "First timestep (GPS): {:.2}",
            timesteps.first().unwrap().as_gpst_seconds() - HIFITIME_GPS_FACTOR
        );
        debug!(
            "Last timestep  (GPS): {:.2}",
            timesteps.first().unwrap().as_gpst_seconds() - HIFITIME_GPS_FACTOR
        );

        // Treat the specified source list as file path. Does it exist? Then use it.
        // Otherwise, treat the specified source list as a glob and attempt to find
        // a single file with it.
        let sl_pb = PathBuf::from(&args.source_list);
        let sl_pb = if sl_pb.exists() {
            sl_pb
        } else {
            get_single_match_from_glob(&args.source_list)?
        };
        // Read the source list.
        // TODO: Allow the user to specify a source list type.
        let mut source_list = match read_source_list_file(&sl_pb, None) {
            Ok((sl, sl_type)) => {
                debug!("Successfully parsed {}-style source list", sl_type);
                sl
            }
            Err(e) => {
                eprintln!("Error when trying to read source list:");
                return Err(SimulateVisError::from(e));
            }
        };
        // Convert flux density lists into power laws.
        source_list
            .values_mut()
            .flat_map(|src| &mut src.components)
            .for_each(|comp| {
                comp.flux_type.convert_list_to_power_law();
            });
        // Split the sky-model components (for efficiency).
        let sky_model_comps = ComponentList::new(source_list, &fine_chan_freqs, &phase_centre);

        let delays = metafits::get_true_delays(&metafits);
        let beam = crate::beam::create_beam_object(
            args.no_beam,
            Delays::Available(delays),
            args.beam_file,
        )?;

        info!("Writing the sky model to {}", args.model_file.display());

        Ok(Self {
            sky_model_comps,
            metafits,
            model_file: args.model_file,
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
    let params = SimVisParams::new(args)?;

    if dry_run {
        info!("Dry run -- exiting now.");
        return Ok(());
    }

    // Get the geodetic XYZ coordinates of each of the MWA tiles.
    let xyzs = XyzGeodetic::get_tiles_mwalib(&params.metafits);
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
        params.fine_chan_freqs[1] - params.fine_chan_freqs[0],
        params.fine_chan_freqs[params.fine_chan_freqs.len() / 2],
        params.fine_chan_freqs.len() / 2,
        &params.phase_centre,
        Some(&format!(
            "Simulated visibilities for obsid {}",
            params.metafits.obs_id
        )),
    )?;

    // Witchcraft to allow this code to be used with or without CUDA support
    // compiled.
    #[cfg(not(feature = "cuda"))]
    let cpu = true;

    // Generate the visibilities.
    for &timestep in params.timesteps.iter() {
        let lst_diff = (timestep - params.timesteps[0]).in_seconds();
        let lst = params.metafits.lst_rad + lst_diff * SOLAR2SIDEREAL * DS2R;
        let hadec_phase_centre = params.phase_centre.to_hadec(lst);
        let uvws = xyz::xyzs_to_uvws(&xyzs, &hadec_phase_centre);

        if cpu {
            model::model_timestep(
                vis_model.view_mut(),
                &params.sky_model_comps,
                params.beam.deref(),
                lst,
                &xyzs,
                &uvws,
                &params.fine_chan_freqs,
            )?;
        } else {
            // CUDA.
            #[cfg(feature = "cuda")]
            {
                let shapelet_uvs = params
                    .sky_model_comps
                    .shapelets
                    .get_shapelet_uvs_gpu(lst, &xyzs);
                let (shapelet_coeffs, num_shapelet_coeffs) =
                    params.sky_model_comps.shapelets.get_flattened_coeffs();
                unsafe {
                    // Shortcuts.
                    let p = &params.sky_model_comps.points;
                    let g = &params.sky_model_comps.gaussians;
                    let s = &params.sky_model_comps.shapelets;
                    dbg!(p.radecs.len());
                    dbg!(g.radecs.len());
                    dbg!(s.radecs.len());
                    let start = std::time::Instant::now();
                    let cuda_code = cuda::model_timestep(
                        num_cross_correlation_baselines,
                        params.fine_chan_freqs.len(),
                        // 0,
                        p.radecs.len(),
                        0,
                        // g.radecs.len(),
                        0,
                        // s.radecs.len(),
                        uvws.as_ptr() as _,
                        params.fine_chan_freqs.as_ptr(),
                        p.lmns.as_ptr() as _,
                        p.instrumental_flux_densities.as_ptr() as _,
                        g.lmns.as_ptr() as _,
                        g.instrumental_flux_densities.as_ptr() as _,
                        g.gaussian_params.as_ptr() as _,
                        s.lmns.as_ptr() as _,
                        s.instrumental_flux_densities.as_ptr() as _,
                        s.gaussian_params.as_ptr() as _,
                        shapelet_uvs.as_ptr(),
                        shapelet_coeffs.as_ptr(),
                        num_shapelet_coeffs.as_ptr(),
                        shapelets::SHAPELET_BASIS_VALUES.as_ptr(),
                        shapelets::SBF_L,
                        shapelets::SBF_N,
                        shapelets::SBF_C,
                        shapelets::SBF_DX,
                        vis_model.as_mut_ptr() as _,
                    );
                    dbg!(std::time::Instant::now() - start);
                    // TODO: Handle cuda_code.
                    if cuda_code != 0 {
                        panic!(
                            "cuda::model_timestep exited with CUDA error code {}",
                            cuda_code
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
        dbg!(&vis_model[[0, 0]]);

        // Clear the visibilities before re-using the buffer.
        vis_model.fill(Jones::default());
    }

    // Finalise writing the model.
    let names: Vec<&str> = params
        .metafits
        .rf_inputs
        .iter()
        .map(|rf| rf.tile_name.as_str())
        .collect();
    output_writer.write_uvfits_antenna_table(&names, &xyzs)?;

    Ok(())
}
