// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Code to predict visibilities, given a sky model and array. These functions
currently work only on a single timestep.
 */

use std::f64::consts::{FRAC_PI_2, PI, TAU};

use ndarray::{parallel::prelude::*, prelude::*};

use super::CalibrateError;
use crate::data_formats::Vis;
use crate::{constants::SQRT_8_LOG_2, math::cexp, MWA_LAT_RAD, VEL_C};
use mwa_hyperdrive_core::{
    erfa_sys, mwa_hyperbeam::fee::FEEBeam, AzEl, ComponentType, HADec, InstrumentalStokes, Jones,
    SourceList, LMN, UVW,
};

/// Beam-correct the expected flux densities of the sky-model source components.
///
/// `inst_flux_densities`: An ndarray slice of the instrumental Stokes flux
/// densities of all sky-model source components. The first axis is unflagged
/// fine channel, the second is sky-model component.
pub(crate) fn beam_correct_flux_densities(
    // flux_densities: ArrayView2<[f64; 4]>,
    flux_densities: ArrayView2<InstrumentalStokes>,
    azels: &[AzEl],
    hadecs: &[HADec],
    beam: &FEEBeam,
    dipole_delays: &[u32],
    dipole_gains: &[f64],
    freqs: &[f64],
) -> Result<Array2<InstrumentalStokes>, CalibrateError> {
    let mut beam_corrected_fds =
        Array2::from_elem(flux_densities.dim(), InstrumentalStokes::default());

    let results = beam_corrected_fds
        .outer_iter_mut()
        .into_par_iter()
        .zip(flux_densities.outer_iter())
        .zip(freqs.par_iter())
        .try_for_each(|((mut beam_corrected_fds, fds), freq)| {
            beam_corrected_fds
                .iter_mut()
                .zip(fds.iter())
                .zip(azels.iter())
                .zip(hadecs.iter())
                .try_for_each(|(((beam_corrected_fd, fd), azel), hadec)| {
                    // `jones_1` is a beam-response Jones matrix. hyperbeam only
                    // returns a 4-element array; wrap this in our special
                    // `Jones` struct.
                    let mut jones_1 = Jones::from(beam.calc_jones(
                        azel.az,
                        azel.za(),
                        *freq as u32,
                        dipole_delays,
                        dipole_gains,
                        false,
                    )?);

                    // Jack's alterations.
                    jones_1 = Jones::from([-jones_1[3], jones_1[2], -jones_1[1], jones_1[0]]);
                    // Parallactic angle correction.
                    let para_angle =
                        unsafe { erfa_sys::eraHd2pa(hadec.ha, hadec.dec, MWA_LAT_RAD) };
                    let (s_rot, c_rot) = (para_angle + FRAC_PI_2).sin_cos();
                    jones_1 = Jones::from([
                        jones_1[0] * c_rot - jones_1[1] * s_rot,
                        jones_1[0] * s_rot + jones_1[1] * c_rot,
                        jones_1[2] * c_rot - jones_1[3] * s_rot,
                        jones_1[2] * s_rot + jones_1[3] * c_rot,
                    ]);
                    // jones_1 = Jones::identity();

                    // `jones_2` is the beam response from the second tile in
                    // this baseline.
                    // TODO: Use a Jones matrix from another tile!
                    let jones_2 = &jones_1;

                    // *beam_corrected_fd = Jones::outer_mul(&jones_1, fd, jones_2).into();

                    // *beam_corrected_fd =
                    //     Jones::outer_mul(Jones::identity(), fd, Jones::identity()).into();

                    let z = Jones::axbh(&Jones::from(fd.to_array()), jones_2);
                    let fd_jones = Jones::axb(&jones_1, &z);
                    *beam_corrected_fd = InstrumentalStokes {
                        xx: fd_jones[0],
                        xy: fd_jones[1],
                        yx: fd_jones[2],
                        yy: fd_jones[3],
                    };
                    Ok(())
                })
        });

    // Handle any errors that happened in the closure.
    match results {
        Ok(()) => Ok(beam_corrected_fds),
        Err(e) => Err(e),
    }
}

/// For a single timestep, over a range of frequencies and baselines, predict
/// visibility values for each sky-model source component. Write the predicted
/// "model" into the supplied array.
///
/// `model_array`: A mutable ndarray slice of the model of all visibilities. The
/// first axis is unflagged baseline, the second unflagged fine channel.
///
/// `flux_densities`: An ndarray slice of the instrumental Stokes flux densities
/// of all sky-model source components. The first axis is unflagged fine
/// channel, the second is sky-model component.
///
/// `calc_jones`: A function that takes the horizon coordinates (i.e. [AzEl]) of
/// a source component and a frequency \[Hz\] returning a beam Jones matrix.
///
/// `lsts`: The local sidereal times that we have to predict over. This is a
/// proxy for time. Each LST is used with the pointing to make correct [AzEl]
/// coordinates for each source component.
///
/// `uvw`: The [UVW] coordinates of each baseline \[metres\]. Each row
/// corresponds to a unique time. This should be the same length as
/// `model_array`'s first axis.
///
/// `lmn`: The LMN coordinates of all sky-model source components.
pub(crate) fn predict_model(
    mut model_array: ArrayViewMut2<Vis<f32>>,
    weights: ArrayView2<f32>,
    flux_densities: ArrayView2<InstrumentalStokes>,
    source_list: &SourceList,
    lmns: &[LMN],
    uvws: &[UVW],
    freqs: &[f64],
) {
    // Unflagged baseline axis.
    model_array
        .outer_iter_mut()
        .into_par_iter()
        .zip(uvws.par_iter())
        .zip(weights.outer_iter().into_par_iter())
        // TODO: Not happy with the variable names here.
        .for_each(|((mut model_bl_axis, uvw), weights_bl_axis)| {
            // Unflagged fine-channel axis.
            model_bl_axis
                .iter_mut()
                .zip(flux_densities.outer_iter())
                .zip(weights_bl_axis.iter())
                .zip(freqs)
                .for_each(|(((model_vis, comp_fds), weight), freq)| {
                    // Divide by lambda to make UVW dimensionless.
                    let uvw = *uvw / (VEL_C / freq);

                    // Now that we have the UVW coordinates, we can determine
                    // each source component's envelope.
                    let envelopes = source_list
                        .iter()
                        .map(|(_, src)| &src.components)
                        .flatten()
                        .map(|comp| match comp.comp_type {
                            ComponentType::Point => 1.0,
                            ComponentType::Gaussian { maj, min, pa } => {
                                let (s_pa, c_pa) = pa.sin_cos();
                                // Get major and minor axes in sigmas.
                                let maj = maj / SQRT_8_LOG_2;
                                let min = min / SQRT_8_LOG_2;
                                (-2.0
                                    * PI.powi(2)
                                    * (maj.powi(2) * (uvw.u * s_pa + uvw.v * c_pa).powi(2)
                                        + (min.powi(2) * (uvw.u * c_pa - uvw.v * s_pa).powi(2))))
                                .exp()
                            }
                            _ => todo!(),
                        });

                    comp_fds.iter().zip(lmns.iter()).zip(envelopes).for_each(
                        |((comp_fd, lmn), envelope)| {
                            let arg = TAU * (uvw.u * lmn.l + uvw.v * lmn.m + uvw.w * (lmn.n - 1.0));
                            let phase = cexp(arg) * envelope;

                            *model_vis += Vis::from_fd_and_phase(*comp_fd, phase) * *weight;
                        },
                    )
                });
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use mwa_hyperdrive_core::c64;

    use approx::assert_abs_diff_eq;
    // Need to use serial tests because HDF5 is not necessarily reentrant.
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_beam_correct_flux_densities_170_mhz() {
        let beam = FEEBeam::new_from_env().unwrap();
        let mut source_list = mwa_hyperdrive_srclist::read::read_source_list_file(
            "tests/1090008640/srclist_pumav3_EoR0aegean_EoR1pietro+ForA_1090008640_100.yaml",
            mwa_hyperdrive_srclist::SourceListType::Hyperdrive,
        )
        .unwrap();

        // Prune all but two sources from the source list.
        let sources_to_keep = ["J000042-342358", "J000045-272248"];
        let mut sources_to_be_removed = vec![];
        for (name, _) in source_list.iter() {
            if !sources_to_keep.contains(&name.as_str()) {
                sources_to_be_removed.push(name.to_owned());
            }
        }
        for name in sources_to_be_removed {
            source_list.remove(&name);
        }
        let num_components = source_list
            .values()
            .fold(0, |a, src| a + src.components.len());

        // Use only frequency.
        let freqs = [170e6];
        let inst_flux_densities = {
            let mut fds =
                Array2::from_elem((num_components, freqs.len()), InstrumentalStokes::default());
            for (mut comp_axis, comp) in fds
                .outer_iter_mut()
                .zip(source_list.iter().map(|(_, src)| &src.components).flatten())
            {
                for (comp_fd, freq) in comp_axis.iter_mut().zip(freqs.iter()) {
                    *comp_fd = comp.estimate_at_freq(*freq).unwrap().into();
                }
            }
            // Flip the array axes; this makes things simpler later.
            fds.t().to_owned()
        };

        let result = match beam_correct_flux_densities(
            inst_flux_densities.view(),
            &source_list.get_azel_mwa(6.261977848),
            &source_list
                .iter()
                .map(|(_, src)| &src.components)
                .flatten()
                .map(|comp| comp.radec.to_hadec(lst))
                .collect::<Vec<_>>(),
            &beam,
            &[0; 16],
            &[1.0; 16],
            &freqs,
        ) {
            Ok(fds) => fds,
            Err(e) => panic!("{}", e),
        };
        assert_eq!(result.dim(), (freqs.len(), num_components));
        assert_eq!(result.dim(), (1, 2));

        // Hand-verified values from Jones::outer_mul
        let expected_jones_1 = Jones::from([
            c64::new(0.058438801501144624, 0.019127623488825452),
            c64::new(-0.3929914018344019, -0.12851599351917362),
            c64::new(-0.3899498110659575, -0.1221766621255064),
            c64::new(-0.058562589895788, -0.01833100712970549),
        ]);
        let expected_comp_fd_1 = [
            (expected_jones_1[0] * expected_jones_1[0].conj()
                + expected_jones_1[1] * expected_jones_1[1].conj())
                * inst_flux_densities[[0, 0]].xx
                + (expected_jones_1[0] * expected_jones_1[1].conj()
                    - expected_jones_1[1] * expected_jones_1[0].conj())
                    * inst_flux_densities[[0, 0]].yy
                    * c64::new(0.0, 1.0),
            (expected_jones_1[0] * expected_jones_1[2].conj()
                + expected_jones_1[1] * expected_jones_1[3].conj())
                * inst_flux_densities[[0, 0]].xx
                + (expected_jones_1[0] * expected_jones_1[3].conj()
                    - expected_jones_1[1] * expected_jones_1[2].conj())
                    * inst_flux_densities[[0, 0]].yy
                    * c64::new(0.0, 1.0),
            (expected_jones_1[2] * expected_jones_1[0].conj()
                + expected_jones_1[3] * expected_jones_1[1].conj())
                * inst_flux_densities[[0, 0]].xx
                + (expected_jones_1[2] * expected_jones_1[1].conj()
                    - expected_jones_1[3] * expected_jones_1[0].conj())
                    * inst_flux_densities[[0, 0]].yy
                    * c64::new(0.0, 1.0),
            (expected_jones_1[2] * expected_jones_1[2].conj()
                + expected_jones_1[3] * expected_jones_1[3].conj())
                * inst_flux_densities[[0, 0]].xx
                + (expected_jones_1[2] * expected_jones_1[3].conj()
                    - expected_jones_1[3] * expected_jones_1[2].conj())
                    * inst_flux_densities[[0, 0]].yy
                    * c64::new(0.0, 1.0),
        ];
        // Predend these things are `Jones` matrices, so we can use
        // assert_abs_diff_eq.
        assert_abs_diff_eq!(
            Jones::from(result[[0, 0]].to_array()),
            Jones::from(expected_comp_fd_1),
            epsilon = 1e-10
        );

        let expected_jones_2 = Jones::from([
            c64::new(0.4212549019831863, 0.13637399862252644),
            c64::new(-0.23101484929174965, -0.07483625313252451),
            c64::new(-0.2314816978394705, -0.0746126963253394),
            c64::new(-0.42174829224904786, -0.13602154224104593),
        ]);
        let expected_comp_fd_2 = [
            (expected_jones_2[0] * expected_jones_2[0].conj()
                + expected_jones_2[1] * expected_jones_2[1].conj())
                * inst_flux_densities[[0, 1]].xx
                + (expected_jones_2[0] * expected_jones_2[1].conj()
                    - expected_jones_2[1] * expected_jones_2[0].conj())
                    * inst_flux_densities[[0, 1]].yy
                    * c64::new(0.0, 1.0),
            (expected_jones_2[0] * expected_jones_2[2].conj()
                + expected_jones_2[1] * expected_jones_2[3].conj())
                * inst_flux_densities[[0, 1]].xx
                + (expected_jones_2[0] * expected_jones_2[3].conj()
                    - expected_jones_2[1] * expected_jones_2[2].conj())
                    * inst_flux_densities[[0, 1]].yy
                    * c64::new(0.0, 1.0),
            (expected_jones_2[2] * expected_jones_2[0].conj()
                + expected_jones_2[3] * expected_jones_2[1].conj())
                * inst_flux_densities[[0, 1]].xx
                + (expected_jones_2[2] * expected_jones_2[1].conj()
                    - expected_jones_2[3] * expected_jones_2[0].conj())
                    * inst_flux_densities[[0, 1]].yy
                    * c64::new(0.0, 1.0),
            (expected_jones_2[2] * expected_jones_2[2].conj()
                + expected_jones_2[3] * expected_jones_2[3].conj())
                * inst_flux_densities[[0, 1]].xx
                + (expected_jones_2[2] * expected_jones_2[3].conj()
                    - expected_jones_2[3] * expected_jones_2[2].conj())
                    * inst_flux_densities[[0, 1]].yy
                    * c64::new(0.0, 1.0),
        ];
        assert_abs_diff_eq!(
            Jones::from(result[[0, 1]].to_array()),
            Jones::from(expected_comp_fd_2),
            epsilon = 1e-10
        );
    }

    /* #[test]
    #[serial]
    // Same as above, but with a different frequency.
    fn test_predict_flux_densities_190_mhz() {
        let beam = FEEBeam::new_from_env().unwrap();
        let dipole_delays = [0; 16];
        let dipole_gains = [1.0; 16];
        let srclist =
            "tests/1090008640/srclist_pumav3_EoR0aegean_EoR1pietro+ForA_1090008640_100.yaml";
        let mut source_list = mwa_hyperdrive_srclist::read::read_source_list_file(
            srclist,
            mwa_hyperdrive_srclist::SourceListType::Hyperdrive,
        )
        .unwrap();

        let sources_to_keep = ["J000042-342358", "J000045-272248"];
        let mut sources_to_be_removed = vec![];
        for (name, _) in source_list.iter() {
            if !sources_to_keep.contains(&name.as_str()) {
                sources_to_be_removed.push(name.to_owned());
            }
        }
        for name in sources_to_be_removed {
            source_list.remove(&name);
        }
        let num_components = source_list
            .values()
            .fold(0, |a, src| a + src.components.len());

        let lsts = vec![6.261977848];
        let freqs = vec![190e6];

        let fds = match predict_flux_densities(
            &beam,
            &dipole_delays,
            &dipole_gains,
            &source_list,
            num_components,
            &lsts,
            &freqs,
        ) {
            Ok(fds) => fds,
            Err(e) => panic!("{}", e),
        };
        assert_eq!(fds.dim(), (lsts.len(), freqs.len(), num_components));
        assert_eq!(fds.dim(), (1, 1, 2));

        let expected_stokes_fds = vec![
            FluxDensity {
                freq: freqs[0],
                i: 2.474742389074019,
                q: 0.0,
                u: 0.0,
                v: 0.0,
            },
            FluxDensity {
                freq: 170000000.0,
                i: 1.539725139872509,
                q: 0.0,
                u: 0.0,
                v: 0.0,
            },
        ];
        let expected_instrumental_stokes_fds: Vec<InstrumentalStokes> =
            expected_stokes_fds.iter().map(|&fd| fd.into()).collect();
        assert_abs_diff_eq!(
            expected_instrumental_stokes_fds[0].xx,
            c64::new(2.474742389074019, 0.0),
            epsilon = 1e-10
        );
        assert_abs_diff_eq!(
            expected_instrumental_stokes_fds[0].xy,
            c64::new(0.0, 0.0),
            epsilon = 1e-10
        );
        assert_abs_diff_eq!(
            expected_instrumental_stokes_fds[0].yx,
            c64::new(0.0, 0.0),
            epsilon = 1e-10
        );
        assert_abs_diff_eq!(
            expected_instrumental_stokes_fds[0].yy,
            c64::new(2.474742389074019, 0.0),
            epsilon = 1e-10
        );
        assert_abs_diff_eq!(
            expected_instrumental_stokes_fds[1].xx,
            c64::new(1.539725139872509, 0.0),
            epsilon = 1e-10
        );
        assert_abs_diff_eq!(
            expected_instrumental_stokes_fds[1].xy,
            c64::new(0.0, 0.0),
            epsilon = 1e-10
        );
        assert_abs_diff_eq!(
            expected_instrumental_stokes_fds[1].yx,
            c64::new(0.0, 0.0),
            epsilon = 1e-10
        );
        assert_abs_diff_eq!(
            expected_instrumental_stokes_fds[1].yy,
            c64::new(1.539725139872509, 0.0),
            epsilon = 1e-10
        );

        let expected_jones_1 = Jones::from([
            c64::new(0.09832826855155767, 0.011401296268472689),
            c64::new(-0.6607054938233954, -0.07743419223298495),
            c64::new(-0.645337513429619, -0.06352428778291135),
            c64::new(-0.09680780207919074, -0.009647590836475156),
        ]);
        let expected_comp_fd_1 = [
            (expected_jones_1[0] * expected_jones_1[0].conj()
                + expected_jones_1[1] * expected_jones_1[1].conj())
                * expected_instrumental_stokes_fds[0].xx
                + (expected_jones_1[0] * expected_jones_1[1].conj()
                    - expected_jones_1[1] * expected_jones_1[0].conj())
                    * expected_instrumental_stokes_fds[0].yy
                    * c64::new(0.0, 1.0),
            (expected_jones_1[0] * expected_jones_1[2].conj()
                + expected_jones_1[1] * expected_jones_1[3].conj())
                * expected_instrumental_stokes_fds[0].xx
                + (expected_jones_1[0] * expected_jones_1[3].conj()
                    - expected_jones_1[1] * expected_jones_1[2].conj())
                    * expected_instrumental_stokes_fds[0].yy
                    * c64::new(0.0, 1.0),
            (expected_jones_1[2] * expected_jones_1[0].conj()
                + expected_jones_1[3] * expected_jones_1[1].conj())
                * expected_instrumental_stokes_fds[0].xx
                + (expected_jones_1[2] * expected_jones_1[1].conj()
                    - expected_jones_1[3] * expected_jones_1[0].conj())
                    * expected_instrumental_stokes_fds[0].yy
                    * c64::new(0.0, 1.0),
            (expected_jones_1[2] * expected_jones_1[2].conj()
                + expected_jones_1[3] * expected_jones_1[3].conj())
                * expected_instrumental_stokes_fds[0].xx
                + (expected_jones_1[2] * expected_jones_1[3].conj()
                    - expected_jones_1[3] * expected_jones_1[2].conj())
                    * expected_instrumental_stokes_fds[0].yy
                    * c64::new(0.0, 1.0),
        ];
        // Predend these things are `Jones` matrices, so we can use
        // assert_abs_diff_eq.
        assert_abs_diff_eq!(
            Jones::from(fds[[0, 0, 0]]),
            Jones::from(expected_comp_fd_1),
            epsilon = 1e-10
        );

        let expected_jones_2 = Jones::from([
            c64::new(0.7326847837488758, 0.07711948036342876),
            c64::new(-0.4018772475391574, -0.04242784695569143),
            c64::new(-0.4025206329105016, -0.041593110832569884),
            c64::new(-0.733506362374082, -0.07603897124302528),
        ]);
        let expected_comp_fd_2 = [
            (expected_jones_2[0] * expected_jones_2[0].conj()
                + expected_jones_2[1] * expected_jones_2[1].conj())
                * expected_instrumental_stokes_fds[1].xx
                + (expected_jones_2[0] * expected_jones_2[1].conj()
                    - expected_jones_2[1] * expected_jones_2[0].conj())
                    * expected_instrumental_stokes_fds[1].yy
                    * c64::new(0.0, 1.0),
            (expected_jones_2[0] * expected_jones_2[2].conj()
                + expected_jones_2[1] * expected_jones_2[3].conj())
                * expected_instrumental_stokes_fds[1].xx
                + (expected_jones_2[0] * expected_jones_2[3].conj()
                    - expected_jones_2[1] * expected_jones_2[2].conj())
                    * expected_instrumental_stokes_fds[1].yy
                    * c64::new(0.0, 1.0),
            (expected_jones_2[2] * expected_jones_2[0].conj()
                + expected_jones_2[3] * expected_jones_2[1].conj())
                * expected_instrumental_stokes_fds[1].xx
                + (expected_jones_2[2] * expected_jones_2[1].conj()
                    - expected_jones_2[3] * expected_jones_2[0].conj())
                    * expected_instrumental_stokes_fds[1].yy
                    * c64::new(0.0, 1.0),
            (expected_jones_2[2] * expected_jones_2[2].conj()
                + expected_jones_2[3] * expected_jones_2[3].conj())
                * expected_instrumental_stokes_fds[1].xx
                + (expected_jones_2[2] * expected_jones_2[3].conj()
                    - expected_jones_2[3] * expected_jones_2[2].conj())
                    * expected_instrumental_stokes_fds[1].yy
                    * c64::new(0.0, 1.0),
        ];
        assert_abs_diff_eq!(
            Jones::from(fds[[0, 0, 1]]),
            Jones::from(expected_comp_fd_2),
            epsilon = 1e-10
        );
    } */
}
