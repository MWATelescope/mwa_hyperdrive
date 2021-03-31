// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Code to handle calibration.
 */

pub mod args;
pub(crate) mod error;
pub(crate) mod params;
pub(crate) mod predict;
pub(crate) mod veto;

pub use error::CalibrateError;
use params::CalibrateParams;

use std::collections::{hash_map::DefaultHasher, HashMap};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use crossbeam_channel::bounded;
use crossbeam_utils::thread::scope;
use log::{debug, info};
use ndarray::prelude::*;
use rayon::prelude::*;

use crate::*;

pub fn calibrate(params: &CalibrateParams) -> Result<(), CalibrateError> {
    di_cal(params)?;

    // How much time is available?
    //
    // Assume we're doing a DI step. How much data gets averaged together? Does
    // this depend on baseline length?

    // Work with a single "scan" for now.
    // Assume we start at "time 0".

    // Rotate all the sources.

    // So all of the sources have their (RA, Dec) coordinates read in.
    //     params.source_list.par_iter().map(|(src_name, src)| {
    //         let rotated_comps: Vec<_> = src.components.iter_mut().map(|comp| {
    //             let hd = comp.radec.to_hadec(params.get_lst());
    //             (hd, comp.comp_type, comp.flux_type)
    //         }).collect();
    //         Source
    // rotated_comps
    //     }).collect()
    // Line 1735 of the RTS

    // If we're not starting at "time 0", the RTS "resets the initial Jones
    // matrices"; for each tile, get the beam-response Jones matrix toward each
    // primary calibrator source (there are usually 5 used by the RTS) at the
    // centre frequency of the entire observation.

    // TODO: RTS SetSourceSpectra. Needed?

    // TODO: "start processing at"

    // TODO: RTS's rts_options.do_MWA_rx_corrections. PFB gains.

    // TODO: Load existing calibration solutions. This should be higher up; fail fast.

    // TODO: Set "AlignmentFluxDensity" and "NewDIMatrices" (line 1988
    // mwa_rts.c). Is this just the estimated Stokes I FD of all components? Do
    // I need to track all estimated FDs?

    // TODO: PrecessZenithtoJ2000
    // Jesus Christ. Do I need this? Surely I can just multiply the visibilities
    // by e^{2pi i w}?

    // CalcMinIntegTime
    // Might be useful. I think it actually calculates the max integration time,
    // not the min - classic.

    // TODO: Load data! Birli's job.
    // import_uvfits_single is where the RTS reads gpubox files.
    // _importuvfits_set_uvdata_visgroup is where the actual fits data gets read.
    // VI_FillVisibilityData accumulates the visibilities (?)

    // The XYZ coordinates of all of the baselines does not change with time for
    // the observation.
    // let xyz = XYZ::get_baselines_mwalib(&params.context.metafits_context);

    // for t in params.get_timesteps() {
    //     let lst = params.get_lst();
    //     let uvw = UVW::get_baselines(&xyz, params.get_pointing());
    // }

    Ok(())
}

/// Direction-independent calibration.
///
/// This code "borrows" heavily from Torrance Hodgson's excellent Julia code at
/// https://github.com/torrance/MWAjl
pub fn di_cal(params: &CalibrateParams) -> Result<(), CalibrateError> {
    let obs_context = params.input_data.get_obs_context();
    let obs_freq_context = params.input_data.get_freq_context();

    let total_num_tiles = obs_context.tile_xyz.len();
    let num_fine_freq_chans = obs_freq_context.fine_chan_freqs.len();

    // The shape of the array containing output Jones matrices.
    let shape = (
        // TODO: Let the user determine this -- using all timesteps at once for
        // now.
        1,
        total_num_tiles,
        num_fine_freq_chans,
        NUM_POLARISATIONS,
    );
    debug!(
        "Shape of DI Jones matrices array: {:?} ({} MiB)",
        shape,
        [
            1,
            total_num_tiles,
            num_fine_freq_chans,
            NUM_POLARISATIONS
        ]
        .iter()
        .product::<usize>()
        // 64 bytes per Jones matrix, 1024 * 1024 == 1 MiB.
        * 64 / 1024
            / 1024
    );
    // The output Jones matrices. We wrap them in an Arc<Mutex<_>> to allow
    // multiple threads to mutate it in parallel.
    let out = Arc::new(Mutex::new(Array4::from_elem(shape, Jones::identity())));

    // As most of the tiles likely have the same configuration (all the same
    // delays and amps), we can be much more efficient with computation by
    // computing over only unique tile configurations (that is, unique
    // combinations of amplitudes/delays).
    let mut tile_configs: HashMap<u64, TileConfig> = HashMap::new();
    // for tile in params
    //     .get_obs_context()
    //     .
    //     .rf_inputs
    //     .iter()
    //     .filter(|&rf| !params.tile_flags.contains(&(rf.ant as _)))
    //     .filter(|&rf| rf.pol == mwa_hyperdrive_core::mwalib::Pol::Y)
    // {
    //     let h = TileConfig::hash(&tile.dipole_delays, &tile.dipole_gains);
    //     match tile_configs.get_mut(&h) {
    //         None => {
    //             tile_configs.insert(
    //                 h,
    //                 TileConfig::new(tile.ant, &tile.dipole_delays, &tile.dipole_gains),
    //             );
    //         }
    //         Some(c) => {
    //             c.antennas.push(tile.ant as _);
    //         }
    //     };
    // }

    // TODO: Let the user decide how many threads to use.
    let num_threads = rayon::current_num_threads();

    // Set up our producer (sender) thread and worker (receiver) threads.
    // TODO: Adjust the buffer size.
    let (sx, rx) = bounded(10);
    scope(|scope| {
        scope.spawn(move |_| {
            // Producer.
            for t in obs_context.timestep_indices.to_owned() {
                // let vis = params.input_data.read(1..2)?;
                let vis = params.input_data.read(t..t + 1).unwrap();
                let msg = t - obs_context.timestep_indices.start;
                sx.send(Box::new(msg)).unwrap();
                dbg!(t);
            }

            // By dropping the send channel, we signal to the receivers that
            // there is no more incoming data, and they can stop waiting.
            drop(sx);
        });

        // Workers.
        for i in 0..num_threads {
            let thread_id = i;
            // Get a thread-local receive channel and reference to the output
            // Jones matrices.
            let rx = rx.clone();
            let out = out.clone();
            scope.spawn(move |_| {
                // Iterate on the receive channel forever. This terminates when
                // there is no data in the channel, and the sender has
                // been dropped.
                for msg_box in rx.iter() {
                    let msg = *msg_box;
                    println!("Worker thread {}: Got {:?}", thread_id, msg);

                    // sleep(Duration::from_secs(3));
                    let mut arr = out.lock().unwrap();
                    arr[[msg, 0, 0, 0]] = Jones::identity() * msg as f64;
                }
            });
        }
    })
    .unwrap();

    dbg!(&params.tile_flags);

    Ok(())
}

#[derive(Debug)]
pub(crate) struct TileConfig<'a> {
    /// The tile antenna numbers that this configuration applies to.
    pub(crate) antennas: Vec<usize>,

    /// The delays of this configuration.
    pub(crate) delays: &'a [u32],

    /// The amps of this configuration.
    pub(crate) amps: &'a [f64],
}

impl<'a> TileConfig<'a> {
    /// Make a new `TileConfig`.
    pub(crate) fn new(antenna: u32, delays: &'a [u32], amps: &'a [f64]) -> Self {
        Self {
            antennas: vec![antenna as _],
            delays,
            amps,
        }
    }

    /// From tile delays and amplitudes, generate a hash. Useful to identify if
    /// this `TileConfig` matches another.
    pub(crate) fn hash(delays: &[u32], amps: &[f64]) -> u64 {
        let mut hasher = DefaultHasher::new();
        delays.hash(&mut hasher);
        // We can't hash f64 values, so convert them to ints. Multiply by a big
        // number to get away from integer rounding.
        let to_int = |x: f64| (x * 1e8) as u32;
        for &a in amps {
            to_int(a).hash(&mut hasher);
        }
        hasher.finish()
    }
}
