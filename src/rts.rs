// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Abandoned RTS code.

// pub fn calibrate(
//     cli_args: args::CalibrateUserArgs,
//     args_file: Option<PathBuf>,
//     dry_run: bool,
// ) -> Result<(), anyhow::Error> {
//     debug!("Merging command-line arguments with the argument file");
//     let args = cli_args.merge(args_file)?;
//     debug!("{:#?}", &args);
//     debug!("Converting arguments into calibration parameters");
//     let params = args.into_params()?;

//     if dry_run {
//         return Ok(());
//     }
//     // mwa_rts.c, line 1719
//     let mut tile_gains = init_calibrator_gains(&params)?;
//     dbg!(&tile_gains.gains.shape());
//     do_calibration(&params, &mut tile_gains);
//     todo!();
// }

// /// RTS's DoCalibration. This should be broken into several components. Only
// /// doing DI for now.
// ///
// /// Assumes the latitude is the MWA latitude.
// pub(crate) fn do_calibration(params: &CalibrateParams, tile_gains: &mut TileGains) {
//     let lst = params.get_lst();

//     // For each source, get a inverted rotation matrix.
//     let src_rot_matrices: Vec<[f64; 4]> = params
//         .ranked_sources
//         .iter()
//         .map(|src| {
//             // These coordinate transforms are probably already coded up elsewhere.
//             // RTS comment: DAM: x and y are backwards. see fittilematrices.c. Not
//             // bothering now because being phased out
//             let sl_src = &params.source_list[&src.name];
//             // Iterate over all components.
//             let hadec = src.weighted_pos.to_hadec(lst);
//             let (s_ha, c_ha) = hadec.ha.sin_cos();
//             let (s_dec, c_dec) = hadec.dec.sin_cos();

//             let proj_px = -MWA_LAT_SIN * s_ha;
//             let proj_py = MWA_LAT_SIN * c_ha * s_dec + MWA_LAT_COS * c_dec;
//             let proj_qx = c_ha;
//             let proj_qy = s_ha * s_dec;

//             let psi_p = atan2(proj_px, proj_py);
//             let psi_q = atan2(proj_qx, proj_qy);
//             let (s_pp, c_pp) = psi_p.sin_cos();
//             let (s_pq, c_pq) = psi_q.sin_cos();

//             // Rotation matrix does not include foreshortening.
//             invert_2x2(&[c_pp, s_pp, c_pq, s_pq])
//         })
//         .collect();

//     // Fill the gain models.
//     for (mut curr_gain_tile, gain_tile) in tile_gains
//         .current_gain_models
//         .outer_iter_mut()
//         .zip(tile_gains.gains.outer_iter())
//     {
//         for (curr_gain_freq, gain_tile) in
//             curr_gain_tile.outer_iter_mut().zip(gain_tile.outer_iter())
//         {
//             // RTS DI calibration always involves one source. This source has
//             // all sky model sources as components.
//             // let fd = src.coarse_chan_fds;
//         }
//     }
//     info!("fill gain models done");
// }

// pub(crate) struct TileGains {
//     /// Order: [tile][coarse_band][src_component]
//     ///
//     /// Is this really used aside from current_gain_models?
//     ///
//     /// The RTS calls this cal_context_t->TileGainMatrices.
//     gains: Array3<Jones>,

//     /// Order: [tile][fine_freq_chan][src_component]
//     ///
//     /// The RTS calls this JinvJ0, which is a field of each
//     /// source_info_t.src_info0 element.
//     ratios: Array3<Jones>,

//     /// Order: [tile][fine_freq_chan][src_component]
//     ///
//     /// The RTS calls this dJinvJ0dt, which is a field of each
//     /// source_info_t.src_info0 element.
//     derivatives: Array3<Jones>,

//     /// Order: [tile][coarse_band][XX or YY]
//     ///
//     /// The RTS calls this cal_context_t->current_gain_models.
//     current_gain_models: Array3<c64>,
// }

// /// TODO: Why are these arrays needed? Update docs with info.
// /// This function assumes that the latitude is the MWA site latitude.
// ///
// /// The RTS calls this function "SetCalibratorMatrices".
// pub(crate) fn init_calibrator_gains(params: &CalibrateParams) -> Result<TileGains, FEEBeamError> {
//     debug!("Running init_calibrator_gains");

//     // Use `dt` to determine the forward-difference derivative.
//     let dt = params.time_res * DS2R * SOLAR2SIDEREAL;
//     // Just in case multiplying is faster than dividing.
//     let inv_dt = 1.0 / dt;
//     let lst = params.get_lst();

//     // Get azimuth and zenith angle calibrator positions at the current LST, as well as LST + `dt`.
//     let (az, za) = {
//         let (mut az, mut za) = params.source_list.get_azza_mwa(lst);
//         // Get the "forward" coords by altering the LST.
//         let (mut az_forward, mut za_forward) = params.source_list.get_azza_mwa(lst + dt);
//         // Combine.
//         az.append(&mut az_forward);
//         za.append(&mut za_forward);
//         (az, za)
//     };

//     // As most of the tiles likely have the same configuration (all the same
//     // delays and amps), we can be much more efficient with computation here by
//     // only iterating over tile configurations (that is, unique combinations of
//     // amplitudes/delays), rather than just all tiles. There are two mwalib
//     // rf_inputs for each tile (Pols X and Y), but we only need one per tile;
//     // filter the other one.
//     let mut tile_configs: HashMap<u64, TileConfig> = HashMap::new();
//     for tile in params
//         .context
//         .metafits_context
//         .rf_inputs
//         .iter()
//         .filter(|&rf| !params.tile_flags.contains(&(rf.ant as _)))
//         .filter(|&rf| rf.pol == mwa_hyperdrive_core::mwalib::Pol::Y)
//     {
//         let h = TileConfig::hash(&tile.dipole_delays, &tile.dipole_gains);
//         match tile_configs.get_mut(&h) {
//             None => {
//                 tile_configs.insert(
//                     h,
//                     TileConfig::new(tile.ant, &tile.dipole_delays, &tile.dipole_gains),
//                 );
//             }
//             Some(c) => {
//                 c.antennas.push(tile.ant as _);
//             }
//         };
//     }

//     // Preallocate output arrays.
//     let mut tile_gain_matrices = Array3::zeros((
//         params.num_unflagged_tiles,
//         params.context.num_coarse_chans,
//         az.len() / 2,
//     ));
//     let mut ratios = Array3::zeros((
//         params.num_unflagged_tiles,
//         params.freq.num_unflagged_fine_chans,
//         az.len() / 2,
//     ));
//     let mut jones_derivatives = Array3::zeros((
//         params.num_unflagged_tiles,
//         params.freq.num_unflagged_fine_chans,
//         az.len() / 2,
//     ));
//     // Inverse Jones matrices are only done once per coarse band. As we've got
//     // coordinates at time=now and time=now+dt in `az`, we only take half of the
//     // length of `az`.
//     let mut jones_inverse = Array2::zeros((params.context.coarse_chans.len(), az.len() / 2));

//     // Iterate over all tiles.
//     for tile_config in tile_configs.values() {
//         // For this tile, get inverse Jones matrices for each of the coarse-band
//         // channel centre frequencies.
//         for (cc_index, (mut inv_row, cc)) in jones_inverse
//             .outer_iter_mut()
//             .zip(params.context.coarse_chans.iter())
//             .enumerate()
//         {
//             let mut band_jones_matrices_results: Vec<Result<Jones, FEEBeamError>> =
//                 Vec::with_capacity(az.len() / 2);
//             az.par_iter()
//                 .zip(za.par_iter())
//                 // Only use the coordinates at time=now.
//                 .take(az.len() / 2)
//                 .map(|(&a, &z)| {
//                     params.jones_cache.get_jones(
//                         &params.beam,
//                         a,
//                         z,
//                         cc.chan_centre_hz,
//                         &tile_config.delays,
//                         &tile_config.amps,
//                         true,
//                     )
//                 })
//                 .collect_into_vec(&mut band_jones_matrices_results);
//             // Validate all of the Jones matrices.
//             let band_jones_matrices: Vec<Jones> = band_jones_matrices_results
//                 .into_iter()
//                 .collect::<Result<Vec<_>, _>>()?;
//             let mut arr = Array1::from(band_jones_matrices);
//             // Keep the current Jones matrices in the tile gain array.
//             for &a in &tile_config.antennas {
//                 let i = params.get_ant_index(a);
//                 tile_gain_matrices
//                     .slice_mut(s![i, cc_index, ..])
//                     .assign(&arr);
//             }

//             // Invert, and put the results into the big array outside this loop.
//             arr.map_inplace(|j| *j = j.inv());
//             inv_row.assign(&arr);
//         }

//         // Iterate over all fine-channel frequencies, except those that have
//         // been flagged.
//         for (freq_index, &freq) in params.freq.unflagged_fine_chan_freqs.iter().enumerate() {
//             let freq_int = freq as _;
//             // Finally, iterate over all of the calibrators and put their Jones
//             // matrices into `freq_results`.
//             let mut jones_matrices_results: Vec<Result<Jones, FEEBeamError>> =
//                 Vec::with_capacity(az.len());
//             az.par_iter()
//                 .zip(za.par_iter())
//                 .map(|(&a, &z)| {
//                     params.jones_cache.get_jones(
//                         &params.beam,
//                         a,
//                         z,
//                         freq_int,
//                         &tile_config.delays,
//                         &tile_config.amps,
//                         true,
//                     )
//                 })
//                 .collect_into_vec(&mut jones_matrices_results);
//             let mut jones_matrices: Vec<Jones> = jones_matrices_results
//                 .into_iter()
//                 .collect::<Result<Vec<_>, _>>()?;
//             // Because `az` and `za` have coordinates for time=now and
//             // time=now+dt, the results in `jones_matrices` need to be split.
//             //
//             // I haven't tested, but I suppose that having the beam code run in
//             // parallel over more coordinates at once is more efficient than
//             // running the beam code twice over two different sets of azimuth
//             // and zenith angle.
//             let second_half = jones_matrices.split_off(az.len() / 2);

//             // Take the Jones matrices for time=now.
//             let j = Array1::from(jones_matrices);

//             // Multiply these Jones matrices by the inverses calculated above,
//             // and store them in ratios. There's only one inverse per coarse
//             // channel, so divide our fine-channel freq. index by the number of
//             // fine channels per coarse band.
//             let inv = &jones_inverse.slice(s![
//                 freq_index / params.freq.num_unflagged_fine_chans_per_coarse_band,
//                 ..
//             ]);
//             let j_ji = &j * inv;
//             for &a in &tile_config.antennas {
//                 let i = params.get_ant_index(a);
//                 ratios.slice_mut(s![i, freq_index, ..]).assign(&j_ji);
//             }

//             // Get the "derivatives" of the Jones matrices. Because we're doing
//             // a forward-difference derivative, the derivatives are most
//             // accurate at (`lst` + `dt`) / 2, which is what we want.
//             let mut jf = Array1::from(second_half);
//             jf -= &j;
//             jf.map_inplace(|j| *j *= inv_dt);

//             // Multiply all of the derivative Jones matrices by the inverses we
//             // made before, and finally write the derivatives out.
//             jf *= inv;

//             for &a in &tile_config.antennas {
//                 let i = params.get_ant_index(a);
//                 jones_derivatives
//                     .slice_mut(s![i, freq_index, ..])
//                     .assign(&jf);
//             }
//         }
//     }

//     Ok(TileGains {
//         gains: tile_gain_matrices,
//         ratios,
//         derivatives: jones_derivatives,
//         current_gain_models: Array3::zeros((
//             params.num_unflagged_tiles,
//             params.context.num_coarse_chans,
//             2,
//         )),
//     })
// }

// #[cfg(test)]
// mod tests {
//     use super::args::CalibrateUserArgs;
//     use super::*;

//     use approx::*;
//     // Need to use serial tests because HDF5 is not necessarily reentrant.
//     use serial_test::serial;

//     use mwa_hyperdrive_tests::full_obsids::get_1065880128;

//     #[test]
//     #[serial]
//     #[ignore]
//     fn test_init_calibrator_gains() {
//         let data = get_1065880128();
//         let args = CalibrateUserArgs {
//             metafits: Some(data.metafits),
//             gpuboxes: Some(data.gpuboxes),
//             mwafs: Some(data.mwafs),
//             source_list: data.source_list,
//             num_sources: Some(2),
//             ..Default::default()
//         };
//         let params_result = args.into_params();
//         let p = match params_result {
//             Ok(p) => p,
//             Err(e) => panic!("{}", e),
//         };
//         let gains_result = init_calibrator_gains(&p);
//         let tile_gains = match gains_result {
//             Ok(g) => g,
//             Err(e) => panic!("{}", e),
//         };

//         // I have verified the following expected values by hand.
//         let gains_first = &tile_gains.gains[[0, 0, 0]];
//         let expected = Jones::from([
//             c64::new(0.10625779960743599, -0.17678576491610976),
//             c64::new(-0.11160131182106525, 0.18690780521243516),
//             c64::new(-0.08298426576454786, 0.18751089642798635),
//             c64::new(-0.0870924046375585, 0.20068612239381686),
//         ]);
//         assert_abs_diff_eq!(gains_first, &expected, epsilon = 1e-6);

//         let deriv_first = &tile_gains.derivatives[[0, 0, 0]];
//         let expected = Jones::from([
//             c64::new(2.495081778732375, 0.07011887175255788),
//             c64::new(0.8121040317440963, 0.10307629690448622),
//             c64::new(-1.024387891504313, 0.13896843126971453),
//             c64::new(2.3336948299029334, 0.2281966033903234),
//         ]);
//         assert_abs_diff_eq!(deriv_first, &expected, epsilon = 1e-6);

//         // Last element.
//         let gains_last = &tile_gains.gains[[
//             p.num_unflagged_tiles - 1,
//             p.context.num_coarse_chans - 1,
//             p.num_components - 1,
//         ]];
//         let expected = Jones::from([
//             c64::new(-0.005084727989539419, -0.0704341368156501),
//             c64::new(0.0039433832003488035, 0.23718716527090428),
//             c64::new(0.026142832997526944, 0.20136231764776708),
//             c64::new(0.004834506081806902, 0.055337977604798756),
//         ]);
//         assert_abs_diff_eq!(gains_last, &expected, epsilon = 1e-6);

//         let deriv_last = &tile_gains.derivatives[[
//             p.num_unflagged_tiles - 1,
//             p.freq.num_unflagged_fine_chans - 1,
//             p.num_components - 1,
//         ]];
//         let expected = Jones::from([
//             c64::new(-3.405833189967276, 0.06699365583072117),
//             c64::new(3.041232553385687, 0.3000911159366356),
//             c64::new(-2.3227179898202306, 0.22650136803513143),
//             c64::new(-3.135968295274456, 0.6280481197895815),
//         ]);
//         assert_abs_diff_eq!(deriv_last, &expected, epsilon = 1e-6);

//         // The ratios are all identity matrices. This would not be the case if
//         // there was more frequency information per coarse channel in the FEE
//         // beam code.
//         let expected = Jones::identity();
//         for axis0 in tile_gains.ratios.outer_iter() {
//             for axis1 in axis0.outer_iter() {
//                 for j in &axis1 {
//                     assert_abs_diff_eq!(j, &expected, epsilon = 1e-6);
//                 }
//             }
//         }
//     }

//     #[test]
//     #[serial]
//     #[ignore]
//     fn test_init_calibrator_gains2() {
//         let data = get_1065880128();
//         let args = CalibrateUserArgs {
//             metafits: Some(data.metafits),
//             gpuboxes: Some(data.gpuboxes),
//             mwafs: Some(data.mwafs),
//             source_list: data.source_list,
//             num_sources: Some(2),
//             // Flag all but the first and last tile; the results should be the same
//             // as that of test_init_calibrator_gains.
//             tile_flags: Some((1..=126).collect()),
//             ..Default::default()
//         };
//         let params_result = args.into_params();
//         let p = match params_result {
//             Ok(p) => p,
//             Err(e) => panic!("{}", e),
//         };
//         let gains_result = init_calibrator_gains(&p);
//         let tile_gains = match gains_result {
//             Ok(g) => g,
//             Err(e) => panic!("{}", e),
//         };

//         // I have verified the following expected values by hand.
//         let gains_first = &tile_gains.gains[[0, 0, 0]];
//         let expected = Jones::from([
//             c64::new(0.10625779960743599, -0.17678576491610976),
//             c64::new(-0.11160131182106525, 0.18690780521243516),
//             c64::new(-0.08298426576454786, 0.18751089642798635),
//             c64::new(-0.0870924046375585, 0.20068612239381686),
//         ]);
//         assert_abs_diff_eq!(gains_first, &expected, epsilon = 1e-6);

//         let deriv_first = &tile_gains.derivatives[[0, 0, 0]];
//         let expected = Jones::from([
//             c64::new(2.495081778732375, 0.07011887175255788),
//             c64::new(0.8121040317440963, 0.10307629690448622),
//             c64::new(-1.024387891504313, 0.13896843126971453),
//             c64::new(2.3336948299029334, 0.2281966033903234),
//         ]);
//         assert_abs_diff_eq!(deriv_first, &expected, epsilon = 1e-6);

//         // Last element.
//         let gains_last = &tile_gains.gains[[
//             p.num_unflagged_tiles - 1,
//             p.context.num_coarse_chans - 1,
//             p.num_components - 1,
//         ]];
//         let expected = Jones::from([
//             c64::new(-0.005084727989539419, -0.0704341368156501),
//             c64::new(0.0039433832003488035, 0.23718716527090428),
//             c64::new(0.026142832997526944, 0.20136231764776708),
//             c64::new(0.004834506081806902, 0.055337977604798756),
//         ]);
//         assert_abs_diff_eq!(gains_last, &expected, epsilon = 1e-6);

//         let deriv_last = &tile_gains.derivatives[[
//             p.num_unflagged_tiles - 1,
//             p.freq.num_unflagged_fine_chans - 1,
//             p.num_components - 1,
//         ]];
//         let expected = Jones::from([
//             c64::new(-3.405833189967276, 0.06699365583072117),
//             c64::new(3.041232553385687, 0.3000911159366356),
//             c64::new(-2.3227179898202306, 0.22650136803513143),
//             c64::new(-3.135968295274456, 0.6280481197895815),
//         ]);
//         assert_abs_diff_eq!(deriv_last, &expected, epsilon = 1e-6);

//         // The ratios are all identity matrices. This would not be the case if
//         // there was more frequency information per coarse channel in the FEE
//         // beam code.
//         let expected = Jones::identity();
//         for axis0 in tile_gains.ratios.outer_iter() {
//             for axis1 in axis0.outer_iter() {
//                 for j in &axis1 {
//                     assert_abs_diff_eq!(j, &expected, epsilon = 1e-6);
//                 }
//             }
//         }
//     }
// }
