// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Code to handle calibration.
 */

pub mod args;
pub mod params;
pub mod veto;

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
    debug!("Merging command-line arguments with the parameter file");
    let args = cli_args.merge(args_file)?;
    debug!("{:#?}", &args);
    let params = args.to_params()?;

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
    let (gains, derivatives) = init_calibrator_gains(&params)?;

    // TODO: "start processing at"

    // TODO: RTS's rts_options.do_MWA_rx_corrections. PFB gains.

    // TODO: Load existing calibration solutions. This should be higher up; fail fast.

    todo!();
}

/// Get derivatives of the gain Jones matrices for each tile, each frequency,
/// each calibrator; the resulting 3-dimensional array is indexed by tile,
/// unflagged fine-channel and calibrator number. This function assumes that the
/// latitude is the MWA site latitude.
///
/// The RTS calls this function "SetCalibratorMatrices".
pub(crate) fn init_calibrator_gains(
    params: &CalibrateParams,
) -> Result<(Array3<Jones>, Array3<Jones>), FEEBeamError> {
    debug!("Running init_calibrator_gains");

    // Use `dt` to determine the forward-difference derivative.
    let dt = params.time_res * DS2R * SOLAR2SIDEREAL;
    // Just in case multiplying is faster than dividing.
    let inv_dt = 1.0 / dt;
    let lst = params.get_lst();

    // Get azimuth and zenith angle calibrator positions at the current LST, as well as LST + `dt`.
    let (az, za) = {
        let (mut az, mut za) = params.source_list.get_azza(lst);
        // Get the "forward" coords by altering the LST.
        let (mut az_forward, mut za_forward) = params.source_list.get_azza(lst + dt);
        // Combine.
        az.append(&mut az_forward);
        za.append(&mut za_forward);
        (az, za)
    };

    // Preallocate output arrays. There are two mwalib rf_inputs for each tile
    // (Pols X and Y), but we only need one per tile; filter the other one.
    // Ignore tile flags.
    let mut tile_gain_matrices = Array3::zeros((
        params.context.num_rf_inputs / 2,
        params.context.num_coarse_channels,
        az.len() / 2,
    ));
    let mut jones_derivative = Array3::zeros((
        params.context.num_rf_inputs / 2,
        params.freq.num_unflagged_fine_chans,
        az.len() / 2,
    ));

    // Iterate over all tiles.
    for (mut gain_axis0, (mut deriv_axis0, tile)) in tile_gain_matrices.outer_iter_mut().zip(
        jones_derivative.outer_iter_mut().zip(
            params
                .context
                .rf_inputs
                .iter()
                .filter(|&rf| rf.pol == mwa_hyperdrive_core::mwalib::Pol::Y),
        ),
    ) {
        // For this tile, get inverse Jones matrices for each of the coarse-band
        // channel centre frequencies.
        let mut jones_inverse = Array2::zeros((params.context.coarse_channels.len(), az.len() / 2));
        for (cc_index, (mut inv_row, cc)) in jones_inverse
            .outer_iter_mut()
            .zip(params.context.coarse_channels.iter())
            .enumerate()
        {
            let mut band_jones_matrices_results: Vec<Result<Jones, FEEBeamError>> =
                Vec::with_capacity(az.len());
            az.par_iter()
                .zip(za.par_iter())
                // Only use the coordinates at time=now.
                .take(az.len() / 2)
                .map(|(&a, &z)| {
                    params.jones_cache.get_jones(
                        &params.beam,
                        a,
                        z,
                        cc.channel_centre_hz,
                        &tile.dipole_delays,
                        &tile.dipole_gains,
                        true,
                    )
                })
                .collect_into_vec(&mut band_jones_matrices_results);
            // Validate all of the Jones matrices.
            let band_jones_matrices: Vec<Jones> = band_jones_matrices_results
                .into_iter()
                .collect::<Result<Vec<_>, _>>()?;
            let arr = Array1::from(band_jones_matrices);
            // Keep the current Jones matrices in the tile gain array.
            gain_axis0.slice_mut(s![cc_index, ..]).assign(&arr);

            // Invert, and put the results into the big array outside this loop.
            let inv_arr = arr.mapv_into(|j| j.inv());
            inv_row.assign(&inv_arr);
        }

        // Iterate over all fine-channel frequencies, except those that have
        // been flagged.
        for (freq_index, (mut deriv_axis1, &freq)) in deriv_axis0
            .outer_iter_mut()
            .zip(params.freq.unflagged_fine_chan_freqs.iter())
            .enumerate()
        {
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
                        &tile.dipole_delays,
                        &tile.dipole_gains,
                        true,
                    )
                })
                .collect_into_vec(&mut jones_matrices_results);
            let mut jones_matrices: Vec<Jones> = jones_matrices_results
                .into_iter()
                .collect::<Result<Vec<_>, _>>()?;
            // Because `az` and `za` have coordinates for time=now and
            // time=now+dt, the results need to be split.
            //
            // I haven't tested, but I suppose that having the beam code run in
            // parallel over more coordinates at once is more efficient than
            // running the beam code twice over two different sets of azimuth
            // and zenith angle.
            let second_half = jones_matrices.split_off(az.len() / 2);

            // Get the "derivatives" of the Jones matrices. Because we're doing
            // a forward-difference derivative, the derivatives are most
            // accurate at (`lst` + `dt`) / 2, which is what we want.
            let j = Array1::from(jones_matrices);
            let jf = Array1::from(second_half);
            let mut forward_diff = (jf - j) * inv_dt;

            // Multiply all of the derivative Jones matrices by the inverses we
            // made before. There's only one inverse per coarse channel, so
            // divide our fine-channel freq. index by the number of fine
            // channels per coarse band.
            forward_diff *= &jones_inverse.slice(s![
                freq_index / params.freq.num_unflagged_fine_chans_per_coarse_band,
                ..
            ]);

            deriv_axis1.assign(&forward_diff);
        }
    }

    Ok((tile_gain_matrices, jones_derivative))
}

#[cfg(test)]
mod tests {
    use super::args::CalibrateUserArgs;
    use super::*;

    use approx::*;
    // Need to use serial tests because HDF5 is not necessarily reentrant.
    use serial_test::serial;

    use mwa_hyperdrive_tests::real_data::get_1065880128;

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
        let params_result = args.to_params();
        let p = match params_result {
            Ok(p) => p,
            Err(e) => panic!("{}", e),
        };
        let gains_result = init_calibrator_gains(&p);
        assert!(gains_result.is_ok(), "{}", gains_result.unwrap_err());
        let (gains, derivatives) = gains_result.unwrap();

        // I have verified the following expected values by hand.
        let gains_first = gains[[0, 0, 0]].clone().to_array();
        let expected = Jones([
            c64::new(0.10625779960743599, -0.17678576491610976),
            c64::new(-0.11160131182106525, 0.18690780521243516),
            c64::new(-0.08298426576454786, 0.18751089642798635),
            c64::new(-0.0870924046375585, 0.20068612239381686),
        ])
        .to_array();
        assert_abs_diff_eq!(gains_first, expected, epsilon = 1e-6);

        let deriv_first = derivatives[[0, 0, 0]].clone().to_array();
        let expected = Jones([
            c64::new(2.495081778732375, 0.07011887175255788),
            c64::new(0.8121040317440963, 0.10307629690448622),
            c64::new(-1.024387891504313, 0.13896843126971453),
            c64::new(2.3336948299029334, 0.2281966033903234),
        ])
        .to_array();
        assert_abs_diff_eq!(deriv_first, expected, epsilon = 1e-6);

        let gains_last = gains[[
            p.context.num_rf_inputs / 2 - 1,
            p.context.num_coarse_channels - 1,
            p.num_components - 1,
        ]]
        .clone()
        .to_array();
        let expected = Jones([
            c64::new(-0.005084727989539419, -0.0704341368156501),
            c64::new(0.0039433832003488035, 0.23718716527090428),
            c64::new(0.026142832997526944, 0.20136231764776708),
            c64::new(0.004834506081806902, 0.055337977604798756),
        ])
        .to_array();
        assert_abs_diff_eq!(gains_last, expected, epsilon = 1e-6);

        let deriv_last = derivatives[[
            p.context.num_rf_inputs / 2 - 1,
            p.freq.num_unflagged_fine_chans - 1,
            p.num_components - 1,
        ]]
        .clone()
        .to_array();
        let expected = Jones([
            c64::new(-3.405833189967276, 0.06699365583072117),
            c64::new(3.041232553385687, 0.3000911159366356),
            c64::new(-2.3227179898202306, 0.22650136803513143),
            c64::new(-3.135968295274456, 0.6280481197895815),
        ])
        .to_array();
        assert_abs_diff_eq!(deriv_last, expected, epsilon = 1e-6);
    }
}
