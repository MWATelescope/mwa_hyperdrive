// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use mwalib::CorrelatorContext;

use crate::*;
use mwa_hyperdrive_cuda::Context_s;

/// An observation context used throughout `hyperdrive`.
///
/// Frequencies are stored as integers to avoid floating-point issues.
pub struct Context {
    /// The context derived from mwalib.
    pub mwalib: CorrelatorContext,
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
    pub fn new<T: AsRef<Path>>(metafits: &T, gpuboxes: &[T]) -> Result<Self, MwalibError> {
        let mwalib = CorrelatorContext::new(metafits, gpuboxes)?;

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
            - (metafits_observation_bandwidth_hz + mwalib.metafits_context.corr_fine_chan_width_hz)
                / 2;

        let base_lst = mwalib.metafits_context.lst_rad;

        // Because mwalib provides the baselines in the order of antenna number,
        // the results won't match WODEN. Sort xyz by input number.
        let mut xyz = Vec::with_capacity(mwalib.metafits_context.num_rf_inputs / 2);
        let mut rf_inputs = mwalib.metafits_context.rf_inputs.clone();
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
            xyz: XYZ::get_xyz_baselines(&xyz),
        })
    }

    /// Convert to a C-compatible struct.
    pub fn convert(&self) -> *const Context_s {
        // Return a pointer to a C Context_s struct.
        Box::into_raw(Box::new(Context_s {
            fine_channel_width: self.mwalib.metafits_context.corr_fine_chan_width_hz as f64,
            base_freq: self.base_freq as f64,
            base_lst: self.base_lst as f64,
        }))
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
        let gpuboxes: Vec<PathBuf> = [
            "tests/1090008640/1090008640_20140721201027_gpubox01_00.fits",
            "tests/1090008640/1090008640_20140721201027_gpubox02_00.fits",
            "tests/1090008640/1090008640_20140721201027_gpubox03_00.fits",
        ]
        .iter()
        .map(PathBuf::from)
        .collect();
        let c = Context::new(&metafits, &gpuboxes)?;
        assert_eq!(c.base_freq, 167035000);
        assert_eq!(c.base_lst, 6.261977848001506);
        assert_eq!(c.mwalib.metafits_context.coarse_chan_width_hz, 1280000);
        assert_eq!(c.mwalib.metafits_context.corr_fine_chan_width_hz, 40000);
        assert_eq!(c.xyz.len(), 8128);

        Ok(())
    }
}
