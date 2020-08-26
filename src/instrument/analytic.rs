// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Code for the analytic MWA beam.
 */

use super::*;

/// Helper function to `tile_response`. Adds the contributions of MWA dipoles to
/// a voltage gain model Jones matrix.
fn tile_response_from_dipoles(
    beam: &PrimaryBeam,
    azel: &AzEl,
    lambda: f64,
    delays: &[f32; 16],
) -> Jones {
    let mut out: Jones = [num::zero(), num::zero(), num::zero(), num::zero()];

    let za = *PIBY2 - azel.el;
    let (s_za, c_za) = za.sin_cos();

    // RTS comment: set elements of the look-dir vector
    let (s_az, c_az) = azel.az.sin_cos();
    let proj_e = s_za * s_az;
    let proj_n = s_za * c_az;
    let proj_z = c_za;

    /* RTS comment:
     * MAPS sets the delays precisely for each tile, with individual boresights due to Earth's curvature.
     * This is not correct for the az0 & za0 delay of real MWA data.
     */
    let beam_azel = beam.hadec.to_azel_mwa();
    let (s_beam_az, c_beam_az) = beam_azel.az.sin_cos();
    let beam_za = *PIBY2 - beam_azel.el;
    let (s_beam_za, c_beam_za) = beam_za.sin_cos();

    // To save myself going cross-eyed, I'm using the RTS' nomenclature here.
    let s_phi = s_beam_az;
    let c_phi = c_beam_az;
    let s_theta = -s_beam_za;
    let c_theta = c_beam_za;
    let s_psi = -s_beam_az;
    let c_psi = c_beam_az;
    let multiplier = Complex::new(0.0, *R2C_SIGN * *PI2 / lambda);

    // Loop over each dipole. There are always 16 delays per tile.
    for (k, delay) in delays.iter().enumerate() {
        let col = k / 4;
        let row = k % 4;
        // Some helper variables.
        let e = (col as f64 - 1.5) * MWA_DPL_SEP;
        let n = (row as f64 - 1.5) * MWA_DPL_SEP;
        let z = 0.0;

        // TODO: Understand this madness. Why does the RTS sometimes do one
        // and not the other?
        let (dip_e, dip_n, dip_z) = match beam.beam_type {
            // Rotate dipole positions.
            BeamType::MwaCrossedDipolesOnGroundPlane => {
                /* RTS comment:
                MAPS sets the delays precisely for each tile, with
                individual boresights due to Earth's curvature. Rotate
                the coordinate frame for the tile? This is not correct
                for real MWA data. */
                let dip_e = e * (c_psi * c_phi - c_theta * s_phi * s_psi)
                    + n * (c_psi * s_phi + c_theta * c_phi * s_psi)
                    + z * (s_psi * s_theta);
                let dip_n = e * (-s_psi * c_phi - c_theta * s_phi * c_psi)
                    + n * (-s_psi * s_phi + c_theta * c_phi * c_psi)
                    + z * (c_psi * s_theta);
                let dip_z = e * (s_theta * s_phi) + n * (-s_theta * c_phi) + z * (c_theta);
                (dip_e, dip_n, dip_z)
            }
            BeamType::Mwa32T => (e, n, z),
        };
        let phase_shift = Complex::exp(
            multiplier
                * (dip_e * proj_e + dip_n * proj_n + dip_z * proj_z - *delay as f64 * *VEL_C),
        );

        // Sum for p receptors.
        let response = (beam.p_gains[k] as f64) * cexp(beam.p_phases[k] as _) * phase_shift;
        out[0] += response;
        out[1] += response;

        // Sum for q receptors.
        let response = (beam.q_gains[k] as f64) * cexp(beam.q_phases[k] as _) * phase_shift;
        out[2] += response;
        out[3] += response;
    }
    out
}

/// Given a set of parameters and a look direction, this function returns the
/// Jones matrix tile gain model for the MWA analytic beam. The 4 elements of
/// the Jones matrix are:
///     [[dec->NS, ra->NS],
///      [dec->EW, ra->EW]]
///
/// # Arguments
///
/// * `beam` - Information on the instrument's primary beam
/// * `azel` - An instance of `AzEl`; the direction in which to sample
/// * `freq` - The observing frequency [Hz]. This is intended to be the centre
///          of the frequency band.
/// * `scaling` - An instance of `BeamScaling`; how should the output be scaled?
/// * `delays` - 16 dipole delay values. In the context of an MWA tile, these
///            should increment by column then row.
///
/// # Returns
///
/// * A Jones matrix containing the voltage gain.
// The RTS calls this function "FillMatrices".
pub(super) fn tile_response(
    beam: &PrimaryBeam,
    azel: &AzEl,
    freq: f64,
    scaling: &BeamScaling,
    delays: &[f32; 16],
) -> Result<Jones, InstrumentError> {
    let lambda = *VEL_C / freq;
    let mut dpl_resp = tile_response_from_dipoles(beam, azel, lambda, delays);
    let hadec = azel.to_hadec_mwa();

    // RTS comment: when dipole-dependent offsets and rotations are added, this
    // will all have to go into the loop assuming that the beam centre is normal
    // to the ground plane, the separation should be used instead of the za.
    let ground_plane = {
        let sep = hadec.separation(&beam.hadec);
        let gp = 2.0 * sin(*PI2 * MWA_DPL_HGT / lambda * cos(sep));
        match scaling {
            BeamScaling::None => gp,
            BeamScaling::UnityTowardZenith => gp / (2.0 * sin(*PI2 * MWA_DPL_HGT / lambda)),
            BeamScaling::UnityInLookDir => todo!(),
        }
    } / MWA_NUM_DIPOLES as f64;

    let response = {
        let (s_beam_dec, c_beam_dec) = beam.hadec.dec.sin_cos();
        let d_ha = hadec.ha - beam.hadec.ha;
        let (s_d_ha, c_d_ha) = d_ha.sin_cos();
        let (s_dec, c_dec) = hadec.dec.sin_cos();
        [
            c_beam_dec * c_dec + s_beam_dec * s_dec * c_d_ha,
            -s_beam_dec * s_d_ha,
            s_dec * s_d_ha,
            c_d_ha,
        ]
    };

    // Multiply the gains together element-wise.
    dpl_resp[0] = dpl_resp[0].scale(response[0] * ground_plane);
    dpl_resp[1] = dpl_resp[1].scale(response[1] * ground_plane);
    dpl_resp[2] = dpl_resp[2].scale(response[2] * ground_plane);
    dpl_resp[3] = dpl_resp[3].scale(response[3] * ground_plane);

    Ok(dpl_resp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;

    fn get_random_beam(beam_type: BeamType) -> PrimaryBeam {
        let mut beam = PrimaryBeam::default_without_context(beam_type, -101.529976, -585.674988);
        beam.hadec = HADec::new(0.0, -0.467397);
        beam.p_gains = [
            1.02546584, 1.03509277, 1.06490629, 1.19587243, 0.94777407, 1.02974419, 1.11602277,
            1.00246689, 0.87201539, 0.94125254, 0.77790027, 0.97723919, 0.9899277, 1.09522974,
            0.88180593, 0.85103283,
        ];
        beam.q_gains = [
            0.9688582, 0.97506558, 0.89600939, 0.79893265, 0.93259913, 0.80808801, 0.81711964,
            1.03962398, 1.00012551, 0.9909525, 0.94967498, 1.05833122, 0.92666765, 0.98024564,
            1.15943595, 0.96849327,
        ];
        beam
    }

    #[test]
    /// Match the RTS' behaviour with verified values.
    fn tile_response_32t() {
        let beam = get_random_beam(BeamType::Mwa32T);
        let sample_dir = AzEl::new(0.61086524, *PIBY2 - 1.4835299);

        let result = tile_response(&beam, &sample_dir, 180e6, &BeamScaling::None, &[0.0; 16]);
        assert!(result.is_ok());
        let jones = result.unwrap();
        let expected = [
            Complex::new(-0.001391, -0.000955),
            Complex::new(0.001086, 0.000746),
            Complex::new(0.002299, -0.001870),
            Complex::new(-0.002593, 0.002109),
        ];

        for (j, e) in jones.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(j.re, e.re, epsilon = 1e-6);
            assert_abs_diff_eq!(j.im, e.im, epsilon = 1e-6);
        }
    }

    #[test]
    /// Match the RTS' behaviour with verified values.
    fn tile_response_full() {
        let beam = get_random_beam(BeamType::MwaCrossedDipolesOnGroundPlane);
        let sample_dir = AzEl::new(0.61086524, *PIBY2 - 1.4835299);

        let result = tile_response(&beam, &sample_dir, 180e6, &BeamScaling::None, &[0.0; 16]);
        assert!(result.is_ok());
        let jones = result.unwrap();
        let expected = [
            Complex::new(-0.001386, -0.000954),
            Complex::new(0.001082, 0.000745),
            Complex::new(0.002294, -0.001870),
            Complex::new(-0.002588, 0.002109),
        ];

        for (j, e) in jones.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(j.re, e.re, epsilon = 1e-6);
            assert_abs_diff_eq!(j.im, e.im, epsilon = 1e-6);
        }
    }
}
