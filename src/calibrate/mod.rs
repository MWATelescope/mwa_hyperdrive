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

    // mwa_rts.c, line 1719
    let tile_gains = init_calibrator_gains(&params)?;
    dbg!(&tile_gains.gains.shape());

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

    for t in params.get_timesteps() {
        let lst = params.get_lst();
        let uvw = UVW::get_baselines(&xyz, params.get_pointing());
    }

    todo!();
}

pub(crate) struct TileGains {
    /// The RTS calls this TileGainMatrices, which is a field of cal_context_t.
    gains: Array3<Jones>,

    /// The RTS calls this JinvJ0, which is a field of each
    /// source_info_t.src_info0 element.
    ratios: Array3<Jones>,

    /// The RTS calls this dJinvJ0dt, which is a field of each
    /// source_info_t.src_info0 element.
    derivatives: Array3<Jones>,
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

/// TODO: Why are these arrays needed? Update docs with info.
/// This function assumes that the latitude is the MWA site latitude.
///
/// The RTS calls this function "SetCalibratorMatrices".
pub(crate) fn init_calibrator_gains(params: &CalibrateParams) -> Result<TileGains, FEEBeamError> {
    debug!("Running init_calibrator_gains");

    // Use `dt` to determine the forward-difference derivative.
    let dt = params.time_res * DS2R * SOLAR2SIDEREAL;
    // Just in case multiplying is faster than dividing.
    let inv_dt = 1.0 / dt;
    let lst = params.get_lst();

    // Get azimuth and zenith angle calibrator positions at the current LST, as well as LST + `dt`.
    let (az, za) = {
        let (mut az, mut za) = params.source_list.get_azza_mwa(lst);
        // Get the "forward" coords by altering the LST.
        let (mut az_forward, mut za_forward) = params.source_list.get_azza_mwa(lst + dt);
        // Combine.
        az.append(&mut az_forward);
        za.append(&mut za_forward);
        (az, za)
    };

    // As most of the tiles likely have the same configuration (all the same
    // delays and amps), we can be much more efficient with computation here by
    // only iterating over tile configurations (that is, unique combinations of
    // amplitudes/delays), rather than just all tiles. There are two mwalib
    // rf_inputs for each tile (Pols X and Y), but we only need one per tile;
    // filter the other one.
    let mut tile_configs: HashMap<u64, TileConfig> = HashMap::new();
    for tile in params
        .context
        .metafits_context
        .rf_inputs
        .iter()
        .filter(|&rf| !params.tile_flags.contains(&(rf.ant as _)))
        .filter(|&rf| rf.pol == mwa_hyperdrive_core::mwalib::Pol::Y)
    {
        let h = TileConfig::hash(&tile.dipole_delays, &tile.dipole_gains);
        match tile_configs.get_mut(&h) {
            None => {
                tile_configs.insert(
                    h,
                    TileConfig::new(tile.ant, &tile.dipole_delays, &tile.dipole_gains),
                );
            }
            Some(c) => {
                c.antennas.push(tile.ant as _);
            }
        };
    }
    dbg!(&tile_configs.len());

    // Preallocate output arrays.
    let mut tile_gain_matrices = Array3::zeros((
        params.num_unflagged_tiles,
        params.context.num_coarse_chans,
        az.len() / 2,
    ));
    let mut ratios = Array3::zeros((
        params.num_unflagged_tiles,
        params.freq.num_unflagged_fine_chans,
        az.len() / 2,
    ));
    let mut jones_derivatives = Array3::zeros((
        params.num_unflagged_tiles,
        params.freq.num_unflagged_fine_chans,
        az.len() / 2,
    ));
    // Inverse Jones matrices are only done once per coarse band. As we've got
    // coordinates at time=now and time=now+dt in `az`, we only take half of the
    // length of `az`.
    let mut jones_inverse = Array2::zeros((params.context.coarse_chans.len(), az.len() / 2));

    // Iterate over all tiles.
    for tile_config in tile_configs.values() {
        // For this tile, get inverse Jones matrices for each of the coarse-band
        // channel centre frequencies.
        for (cc_index, (mut inv_row, cc)) in jones_inverse
            .outer_iter_mut()
            .zip(params.context.coarse_chans.iter())
            .enumerate()
        {
            let mut band_jones_matrices_results: Vec<Result<Jones, FEEBeamError>> =
                Vec::with_capacity(az.len() / 2);
            az.par_iter()
                .zip(za.par_iter())
                // Only use the coordinates at time=now.
                .take(az.len() / 2)
                .map(|(&a, &z)| {
                    params.jones_cache.get_jones(
                        &params.beam,
                        a,
                        z,
                        cc.chan_centre_hz,
                        &tile_config.delays,
                        &tile_config.amps,
                        true,
                    )
                })
                .collect_into_vec(&mut band_jones_matrices_results);
            // Validate all of the Jones matrices.
            let band_jones_matrices: Vec<Jones> = band_jones_matrices_results
                .into_iter()
                .collect::<Result<Vec<_>, _>>()?;
            let mut arr = Array1::from(band_jones_matrices);
            // Keep the current Jones matrices in the tile gain array.
            for &a in &tile_config.antennas {
                let i = params.get_ant_index(a);
                tile_gain_matrices
                    .slice_mut(s![i, cc_index, ..])
                    .assign(&arr);
            }

            // Invert, and put the results into the big array outside this loop.
            arr.map_inplace(|j| *j = j.inv());
            inv_row.assign(&arr);
        }

        // Iterate over all fine-channel frequencies, except those that have
        // been flagged.
        for (freq_index, &freq) in params.freq.unflagged_fine_chan_freqs.iter().enumerate() {
            let freq_int = freq as _;
            // Finally, iterate over all of the calibrators and put their Jones
            // matrices into `freq_results`.
            let mut jones_matrices_results: Vec<Result<Jones, FEEBeamError>> =
                Vec::with_capacity(az.len());
            az.par_iter()
                .zip(za.par_iter())
                .map(|(&a, &z)| {
                    params.jones_cache.get_jones(
                        &params.beam,
                        a,
                        z,
                        freq_int,
                        &tile_config.delays,
                        &tile_config.amps,
                        true,
                    )
                })
                .collect_into_vec(&mut jones_matrices_results);
            let mut jones_matrices: Vec<Jones> = jones_matrices_results
                .into_iter()
                .collect::<Result<Vec<_>, _>>()?;
            // Because `az` and `za` have coordinates for time=now and
            // time=now+dt, the results in `jones_matrices` need to be split.
            //
            // I haven't tested, but I suppose that having the beam code run in
            // parallel over more coordinates at once is more efficient than
            // running the beam code twice over two different sets of azimuth
            // and zenith angle.
            let second_half = jones_matrices.split_off(az.len() / 2);

            // Take the Jones matrices for time=now.
            let j = Array1::from(jones_matrices);

            // Multiply these Jones matrices by the inverses calculated above,
            // and store them in ratios. There's only one inverse per coarse
            // channel, so divide our fine-channel freq. index by the number of
            // fine channels per coarse band.
            let inv = &jones_inverse.slice(s![
                freq_index / params.freq.num_unflagged_fine_chans_per_coarse_band,
                ..
            ]);
            let j_ji = &j * inv;
            for &a in &tile_config.antennas {
                let i = params.get_ant_index(a);
                ratios.slice_mut(s![i, freq_index, ..]).assign(&j_ji);
            }

            // Get the "derivatives" of the Jones matrices. Because we're doing
            // a forward-difference derivative, the derivatives are most
            // accurate at (`lst` + `dt`) / 2, which is what we want.
            let mut jf = Array1::from(second_half);
            jf -= &j;
            jf.map_inplace(|j| *j *= inv_dt);

            // Multiply all of the derivative Jones matrices by the inverses we
            // made before, and finally write the derivatives out.
            jf *= inv;

            for &a in &tile_config.antennas {
                let i = params.get_ant_index(a);
                jones_derivatives
                    .slice_mut(s![i, freq_index, ..])
                    .assign(&jf);
            }
        }
    }

    Ok(TileGains {
        gains: tile_gain_matrices,
        ratios,
        derivatives: jones_derivatives,
    })
}

#[cfg(test)]
mod tests {
    use super::args::CalibrateUserArgs;
    use super::*;

    use approx::*;
    // Need to use serial tests because HDF5 is not necessarily reentrant.
    use serial_test::serial;

    use mwa_hyperdrive_tests::full_obsids::get_1065880128;

    #[test]
    #[serial]
    #[ignore]
    fn test_init_calibrator_gains() {
        let data = get_1065880128();
        let args = CalibrateUserArgs {
            metafits: Some(data.metafits),
            gpuboxes: Some(data.gpuboxes),
            mwafs: Some(data.mwafs),
            source_list: data.source_list,
            num_sources: Some(2),
            ..Default::default()
        };
        let params_result = args.into_params();
        let p = match params_result {
            Ok(p) => p,
            Err(e) => panic!("{}", e),
        };
        let gains_result = init_calibrator_gains(&p);
        let tile_gains = match gains_result {
            Ok(g) => g,
            Err(e) => panic!("{}", e),
        };

        // I have verified the following expected values by hand.
        let gains_first = &tile_gains.gains[[0, 0, 0]];
        let expected = Jones::from([
            c64::new(0.10625779960743599, -0.17678576491610976),
            c64::new(-0.11160131182106525, 0.18690780521243516),
            c64::new(-0.08298426576454786, 0.18751089642798635),
            c64::new(-0.0870924046375585, 0.20068612239381686),
        ]);
        assert_abs_diff_eq!(gains_first, &expected, epsilon = 1e-6);

        let deriv_first = &tile_gains.derivatives[[0, 0, 0]];
        let expected = Jones::from([
            c64::new(2.495081778732375, 0.07011887175255788),
            c64::new(0.8121040317440963, 0.10307629690448622),
            c64::new(-1.024387891504313, 0.13896843126971453),
            c64::new(2.3336948299029334, 0.2281966033903234),
        ]);
        assert_abs_diff_eq!(deriv_first, &expected, epsilon = 1e-6);

        // Last element.
        let gains_last = &tile_gains.gains[[
            p.num_unflagged_tiles - 1,
            p.context.num_coarse_chans - 1,
            p.num_components - 1,
        ]];
        let expected = Jones::from([
            c64::new(-0.005084727989539419, -0.0704341368156501),
            c64::new(0.0039433832003488035, 0.23718716527090428),
            c64::new(0.026142832997526944, 0.20136231764776708),
            c64::new(0.004834506081806902, 0.055337977604798756),
        ]);
        assert_abs_diff_eq!(gains_last, &expected, epsilon = 1e-6);

        let deriv_last = &tile_gains.derivatives[[
            p.num_unflagged_tiles - 1,
            p.freq.num_unflagged_fine_chans - 1,
            p.num_components - 1,
        ]];
        let expected = Jones::from([
            c64::new(-3.405833189967276, 0.06699365583072117),
            c64::new(3.041232553385687, 0.3000911159366356),
            c64::new(-2.3227179898202306, 0.22650136803513143),
            c64::new(-3.135968295274456, 0.6280481197895815),
        ]);
        assert_abs_diff_eq!(deriv_last, &expected, epsilon = 1e-6);

        // The ratios are all identity matrices. This would not be the case if
        // there was more frequency information per coarse channel in the FEE
        // beam code.
        let expected = Jones::identity();
        for axis0 in tile_gains.ratios.outer_iter() {
            for axis1 in axis0.outer_iter() {
                for j in &axis1 {
                    assert_abs_diff_eq!(j, &expected, epsilon = 1e-6);
                }
            }
        }
    }

    #[test]
    #[serial]
    #[ignore]
    fn test_init_calibrator_gains2() {
        let data = get_1065880128();
        let args = CalibrateUserArgs {
            metafits: Some(data.metafits),
            gpuboxes: Some(data.gpuboxes),
            mwafs: Some(data.mwafs),
            source_list: data.source_list,
            num_sources: Some(2),
            // Flag all but the first and last tile; the results should be the same
            // as that of test_init_calibrator_gains.
            tile_flags: Some((1..=126).collect()),
            ..Default::default()
        };
        let params_result = args.into_params();
        let p = match params_result {
            Ok(p) => p,
            Err(e) => panic!("{}", e),
        };
        let gains_result = init_calibrator_gains(&p);
        let tile_gains = match gains_result {
            Ok(g) => g,
            Err(e) => panic!("{}", e),
        };

        // I have verified the following expected values by hand.
        let gains_first = &tile_gains.gains[[0, 0, 0]];
        let expected = Jones::from([
            c64::new(0.10625779960743599, -0.17678576491610976),
            c64::new(-0.11160131182106525, 0.18690780521243516),
            c64::new(-0.08298426576454786, 0.18751089642798635),
            c64::new(-0.0870924046375585, 0.20068612239381686),
        ]);
        assert_abs_diff_eq!(gains_first, &expected, epsilon = 1e-6);

        let deriv_first = &tile_gains.derivatives[[0, 0, 0]];
        let expected = Jones::from([
            c64::new(2.495081778732375, 0.07011887175255788),
            c64::new(0.8121040317440963, 0.10307629690448622),
            c64::new(-1.024387891504313, 0.13896843126971453),
            c64::new(2.3336948299029334, 0.2281966033903234),
        ]);
        assert_abs_diff_eq!(deriv_first, &expected, epsilon = 1e-6);

        // Last element.
        let gains_last = &tile_gains.gains[[
            p.num_unflagged_tiles - 1,
            p.context.num_coarse_chans - 1,
            p.num_components - 1,
        ]];
        let expected = Jones::from([
            c64::new(-0.005084727989539419, -0.0704341368156501),
            c64::new(0.0039433832003488035, 0.23718716527090428),
            c64::new(0.026142832997526944, 0.20136231764776708),
            c64::new(0.004834506081806902, 0.055337977604798756),
        ]);
        assert_abs_diff_eq!(gains_last, &expected, epsilon = 1e-6);

        let deriv_last = &tile_gains.derivatives[[
            p.num_unflagged_tiles - 1,
            p.freq.num_unflagged_fine_chans - 1,
            p.num_components - 1,
        ]];
        let expected = Jones::from([
            c64::new(-3.405833189967276, 0.06699365583072117),
            c64::new(3.041232553385687, 0.3000911159366356),
            c64::new(-2.3227179898202306, 0.22650136803513143),
            c64::new(-3.135968295274456, 0.6280481197895815),
        ]);
        assert_abs_diff_eq!(deriv_last, &expected, epsilon = 1e-6);

        // The ratios are all identity matrices. This would not be the case if
        // there was more frequency information per coarse channel in the FEE
        // beam code.
        let expected = Jones::identity();
        for axis0 in tile_gains.ratios.outer_iter() {
            for axis1 in axis0.outer_iter() {
                for j in &axis1 {
                    assert_abs_diff_eq!(j, &expected, epsilon = 1e-6);
                }
            }
        }
    }
}
