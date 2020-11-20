// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Instrument stuff.
 */

pub mod analytic;
pub mod error;

pub use error::InstrumentError;

use crate::*;
use mwa_hyperdrive_core::coord::jones::Jones;

/// The number of parameters per dipole.
pub static MWA_NUM_PARAMS_PER_DIPOLE: u32 = 2;
/// The number of dipoles per tile.
pub static MWA_NUM_DIPOLES: u32 = 16;
/// The number of polarised receptors on an antenna.
pub static MWA_NUM_POL_RECEPTORS: u32 = 2;
/// Ideal spacing between dipoles (centre to centre) in tiles [metres].
pub static MWA_DPL_SEP: f64 = 1.1;
/// Ideal height of dipole above the ground plane [metres].
pub static MWA_DPL_HGT: f64 = 0.3;

/// How should the beam response be attenuated?
#[derive(Debug, Clone, Copy)]
pub enum BeamScaling {
    /// No scaling.
    None,
    /// Attenuation at zenith is 1 (i.e. no attenuation).
    UnityTowardZenith,
    /// Attenuation at the pointing centre is 1 (i.e. no attenuation).
    UnityInLookDir,
}

// pub enum DipoleType {
//     MWA,
// }

/// The type of beam response for a specific instrument configuration.
#[derive(Debug, Clone, Copy)]
pub enum BeamType {
    /// The standard MWA 16-dipole tile.
    MwaCrossedDipolesOnGroundPlane,
    /// This 32T variant is still being heavily used in the RTS, and it is
    /// distinct from the non-32T variant. I will kill it when I understand why
    /// it's still being used.
    Mwa32T,
}

/// The beam parameters for a single tile. All elements are in MWA M&C order.
#[derive(Debug, Clone, Copy)]
pub struct PrimaryBeam {
    /// The type of beam response.
    pub beam_type: BeamType,
    /// The pointing direction of the beam.
    pub hadec: HADec,
    /// p-receptor gains (???).
    pub p_gains: [f32; 16],
    /// q-receptor gains (???).
    pub q_gains: [f32; 16],
    /// p-receptor phases (???).
    pub p_phases: [f32; 16],
    /// q-receptor phases (???).
    pub q_phases: [f32; 16],
    /// Which dipoles are dead? `true` for dead, `false` for alive.
    pub dead_dipoles: [bool; 16],
    // /// Taken from the RTS: if used to generate reference J matrices, some applications need a frequency.
    // pub ref_freq: f64,

    // /// Taken from the RTS: Beam modes used for Spherical Harmonic model
    // // double _Complex **Q1, **Q2;
    // double _Complex **p_T, **p_P;    //!< Some pre-computed values used in tile response
    // double **M,**N;
    // int nmax;
    // int nMN;
    // float _Complex norm_fac[MAX_POLS];

    // // BP 2019: All the Spherical Harmonic Beam data are double
    // // so we will use them on the GPUs as well or there will be all kinds
    // // of issues with copying

    // float _Complex *d_Q1, *d_Q2;
    // float _Complex *d_p_T, *d_p_P; // Precomputed values used in CPU optimization
    // float *d_M, *d_N;
    // int d_nmax;
    // int d_nMN;
    // float _Complex d_norm_fac[MAX_POLS];
}

impl PrimaryBeam {
    /// Create a default (i.e. uninitialised) MWA `PrimaryBeam` struct with the
    /// help of an mwalib context.
    ///
    /// # Arguments
    ///
    /// `beam_type` - One of the recognised hyperdrive primary beam models.
    /// `antenna_num` - The MWA antenna number. For 128T, this should be between 0 and 127.
    /// `pol` - The X or Y RF input.
    /// `context` - A reference to a `mwalibContext` struct.
    pub fn default(
        beam_type: BeamType,
        antenna_num: usize,
        pol: Pol,
        context: &mwalibContext,
    ) -> Self {
        // mwalib order its RF inputs with antenna 0 being at indices 0 and 1,
        // where index 0 is X and 1 is Y. Antenna 1 is at indices 2 and 3, with
        // 2 as X and 3 as Y, etc.
        let index = antenna_num * 2
            + match pol {
                Pol::X => 0,
                Pol::Y => 1,
            };

        // If a dipole's delay is 32, then it is considered dead.
        let mut dead_dipoles = [false; 16];
        for (i, delay) in context.rf_inputs[index].delays.iter().enumerate() {
            if *delay == 32 {
                dead_dipoles[i] = true;
            }
        }

        let mut beam = Self::default_without_context(
            beam_type,
            context.rf_inputs[index].north_m,
            context.rf_inputs[index].east_m,
        );
        beam.dead_dipoles = dead_dipoles;
        beam
    }

    /// Create a default (i.e. uninitialised) MWA `PrimaryBeam` struct without
    /// an `mwalibContext`. This requires the position of the MWA tile that this
    /// beam belongs to.
    ///
    /// # Arguments
    ///
    /// `beam_type` - One of the recognised hyperdrive primary beam models.
    /// `north` - The tile position north from the array centre [metres]
    /// `east` - The tile position east from the array centre [metres]
    pub fn default_without_context(beam_type: BeamType, north: f64, east: f64) -> Self {
        // RTS comment: rotation parameters of the boresight of the tile
        // (un-phased beam centre).
        let dec = MWA_LAT_RAD
            + match beam_type {
                BeamType::MwaCrossedDipolesOnGroundPlane => north / EARTH_RADIUS,
                BeamType::Mwa32T => 0.0,
            };
        let ha = match beam_type {
            BeamType::MwaCrossedDipolesOnGroundPlane => -east / EARTH_RADIUS / cos(dec),
            BeamType::Mwa32T => 0.0,
        };

        // All dipole gains are just 1 to start with.
        let p_gains = [1.0; 16];
        let q_gains = p_gains;
        // All phases are just 0 to start with.
        let p_phases = [0.0; 16];
        let q_phases = p_phases;
        Self {
            beam_type,
            hadec: HADec::new(ha, dec),
            p_gains,
            q_gains,
            p_phases,
            q_phases,
            // We assume all dipoles are alive.
            dead_dipoles: [false; 16],
        }
    }
}

pub fn tile_response(
    beam: &PrimaryBeam,
    azel: &AzEl,
    freq: f64,
    scaling: &BeamScaling,
    delays: &[f32; 16],
) -> Result<Jones, InstrumentError> {
    match beam.beam_type {
        BeamType::MwaCrossedDipolesOnGroundPlane | BeamType::Mwa32T => {
            analytic::tile_response(&beam, &azel, freq, &scaling, &delays)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_random_beam(
        beam_type: BeamType,
        antenna_num: usize,
        pol: Pol,
        context: &mwalibContext,
    ) -> PrimaryBeam {
        let mut beam = PrimaryBeam::default(beam_type, antenna_num, pol, context);
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
    /// Test that dead dipoles are detected.
    fn dead_dipole_found() {
        // This metafits file has a deliberately-changed dipole delay (set to
        // 32) to make it look like a dipole is dead.
        let metafits = "tests/1065880128_broken.metafits";
        let context = mwalibContext::new(&metafits, &[]).unwrap();

        // Pick an RF input we haven't corrupted.
        let beam = get_random_beam(BeamType::Mwa32T, 100, Pol::Y, &context);
        assert!(!beam.dead_dipoles.iter().any(|x| *x == true));

        // Now find the corrupted RF input.
        let beam = get_random_beam(BeamType::Mwa32T, 75, Pol::Y, &context);
        for (i, dead_dipole) in beam.dead_dipoles.iter().enumerate() {
            if i == 1 {
                assert!(dead_dipole)
            } else {
                assert!(!dead_dipole)
            }
        }
    }
}
