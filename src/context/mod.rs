// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use mwalib::mwalibContext;

use crate::*;

/// An observation context used throughout `hyperdrive`.
///
/// Frequencies are stored as integers to avoid floating-point issues.
pub struct Context {
    /// The context derived from mwalib.
    pub mwalib: mwalibContext,
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
        let mwalib = mwalibContext::new(metafits, gpuboxes)?;

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
        let base_freq =
            freq_centre_hz - (metafits_observation_bandwidth_hz + mwalib.fine_channel_width_hz) / 2;

        let base_lst = mwalib.lst_degrees.to_radians();

        let xyz = XYZ::get_baselines_mwalib(&mwalib);

        Ok(Context {
            mwalib,
            base_freq,
            base_lst,
            xyz,
        })
    }

    /// Convert to a C-compatible struct.
    pub fn convert(&self) -> *const Context_s {
        // Return a pointer to a C Context_s struct.
        Box::into_raw(Box::new(Context_s {
            fine_channel_width: self.mwalib.fine_channel_width_hz as f64,
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

    #[test]
    fn test_new_context_from_1065880128_metafits() -> Result<(), MwalibError> {
        let metafits = std::path::PathBuf::from("tests/1065880128.metafits");
        let c = Context::new(&metafits, &[])?;
        assert_eq!(c.base_freq, 167035000);
        assert_eq!(c.base_lst, 6.074823226561063);
        assert_eq!(c.mwalib.coarse_channel_width_hz, 1280000);
        assert_eq!(c.mwalib.fine_channel_width_hz, 40000);
        assert_eq!(c.xyz.len(), 8128);

        Ok(())
    }
}
