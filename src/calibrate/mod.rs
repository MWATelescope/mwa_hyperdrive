// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Code to handle calibration.
 */

pub mod args;
pub mod params;
pub mod veto;

use std::collections::{hash_map::DefaultHasher, HashMap};
use std::hash::{Hash, Hasher};

use ndarray::prelude::*;
use rayon::prelude::*;

use crate::*;
use mwa_hyperdrive_core::jones::Jones;
use mwa_hyperdrive_core::mwa_hyperbeam::fee::FEEBeamError;
use params::CalibrateParams;

pub fn calibrate(
    cli_args: args::CalibrateUserArgs,
    args_file: Option<PathBuf>,
    dry_run: bool,
) -> Result<(), anyhow::Error> {
    debug!("Merging command-line arguments with the argument file");
    let args = cli_args.merge(args_file)?;
    debug!("{:#?}", &args);
    debug!("Converting arguments into calibration parameters");
    let params = args.into_params()?;

    if dry_run {
        return Ok(());
    }

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
    let xyz = XYZ::get_baselines_mwalib(&params.context.metafits_context);

    // for t in params.get_timesteps() {
    //     let lst = params.get_lst();
    //     let uvw = UVW::get_baselines(&xyz, params.get_pointing());
    // }

    todo!();
}

#[derive(Debug)]
struct TileConfig<'a> {
    /// The tile antenna numbers that this configuration applies to.
    antennas: Vec<usize>,

    /// The delays of this configuration.
    delays: &'a [u32],

    /// The amps of this configuration.
    amps: &'a [f64],
}

impl<'a> TileConfig<'a> {
    /// Make a new `TileConfig`.
    fn new(antenna: u32, delays: &'a [u32], amps: &'a [f64]) -> Self {
        Self {
            antennas: vec![antenna as _],
            delays,
            amps,
        }
    }

    /// From tile delays and amplitudes, generate a hash. Useful to identify if
    /// this `TileConfig` matches another.
    fn hash(delays: &[u32], amps: &[f64]) -> u64 {
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
