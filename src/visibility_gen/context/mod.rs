// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::path::Path;

use mwa_hyperdrive_core::{mwalib, XyzBaseline, XYZ};
use mwalib::*;

#[cfg(feature = "cuda")]
use mwa_hyperdrive_cuda::Context_c;

/// An observation context used throughout `hyperdrive`.
///
/// Frequencies are stored as integers to avoid floating-point issues.
pub struct Context {
    /// The context derived from mwalib.
    pub mwalib: MetafitsContext,
    /// The base frequency of the observation [Hz]. For reasons unknown, its
    /// calculation is insane.
    pub base_freq: u32,
    /// The LST at the start of the observation [radians]
    pub base_lst: f64,
    /// The `XyzBaselines` of the observations [metres].
    pub xyz: Vec<XyzBaseline>,
}

impl Context {
    /// Create a new `hyperdrive` observation `Context` from a metafits file.
    pub fn new<T: AsRef<Path>>(metafits: &T) -> Result<Self, MwalibError> {
        let mwalib = MetafitsContext::new(metafits)?;

        // Get some things that mwalib doesn't expose (yet?).
        let mut metafits = fits_open!(&metafits)?;
        let hdu = fits_open_hdu!(&mut metafits, 0)?;
        let freq_centre_hz: u32 = {
            let f: f64 = get_required_fits_key!(&mut metafits, &hdu, "FREQCENT")?;
            (f * 1e6) as _
        };
        let metafits_observation_bandwidth_hz: u32 = {
            let bw: f64 = get_required_fits_key!(&mut metafits, &hdu, "BANDWDTH")?;
            (bw * 1e6).round() as _
        };
        let base_freq = freq_centre_hz
            - (metafits_observation_bandwidth_hz + mwalib.corr_fine_chan_width_hz) / 2;

        let base_lst = mwalib.lst_rad;

        // Because mwalib provides the baselines in the order of antenna number,
        // the results won't match WODEN. Sort xyz by input number.
        let mut xyz = Vec::with_capacity(mwalib.num_rf_inputs / 2);
        let mut rf_inputs = mwalib.rf_inputs.clone();
        rf_inputs.sort_unstable_by(|a, b| a.input.cmp(&b.input));
        for rf in rf_inputs {
            // There is an RF input for both tile polarisations. The ENH
            // coordinates are the same for both polarisations of a tile; ignore
            // the RF input if it's associated with Y.
            if rf.pol == mwalib::Pol::Y {
                continue;
            }

            let enh = mwa_hyperdrive_core::coord::enh::ENH {
                e: rf.east_m,
                n: rf.north_m,
                h: rf.height_m,
            };
            xyz.push(enh.to_xyz_mwa());
        }

        Ok(Context {
            mwalib,
            base_freq,
            base_lst,
            xyz: XYZ::get_baselines(&xyz),
        })
    }
}

impl std::fmt::Display for Context {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            r#"hyperdrive observation context:
Obs. base freq.: {bf} MHz
Base LST:        {lst} rad
mwalib:

{mwalib}"#,
            bf = self.base_freq as f64 / 1e6,
            lst = self.base_lst,
            mwalib = self.mwalib,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_new_context_from_1090008640_metafits() -> Result<(), MwalibError> {
        let metafits = PathBuf::from("tests/1090008640/1090008640.metafits");
        let c = Context::new(&metafits)?;
        assert_eq!(c.base_freq, 167035000);
        assert_eq!(c.base_lst, 6.261977848001506);
        assert_eq!(c.mwalib.coarse_chan_width_hz, 1280000);
        assert_eq!(c.mwalib.corr_fine_chan_width_hz, 40000);
        assert_eq!(c.xyz.len(), 8128);

        Ok(())
    }
}
