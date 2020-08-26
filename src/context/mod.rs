// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use crate::*;

/// An observation context used throughout `hyperdrive`.
///
/// Frequencies are stored as integers to avoid floating-point issues.
#[derive(Debug)]
pub struct Context {
    /// The base frequency of the observation [Hz]. For reasons unknown, its
    /// calculation is insane.
    pub base_freq: u64,
    /// The total bandwidth of the observation [Hz]
    pub bandwidth: u64,
    /// The coarse channels used. These are typically 0 to 23.
    pub coarse_channels: Vec<u8>,
    /// The frequency width of a single coarse channel [Hz]
    pub coarse_channel_width: u64,
    /// The observation's frequency resolution [Hz]
    pub fine_channel_width: u64,
    /// The LST at the start of the observation [radians]
    pub base_lst: f64,
    /// The number of cross-correlation baselines
    pub n_baselines: u64,
    /// The `XyzBaselines` of the observations [metres]
    pub xyz: Vec<XyzBaseline>,
}

impl Context {
    /// Create a new `hyperdrive` observation `Context` from a metafits file.
    pub fn new(metafits: &mut FitsFile) -> Result<Self, FitsError> {
        let hdu = fits_open_hdu!(metafits, 0)?;
        let freq_centre = {
            let f: f64 = get_required_fits_key!(metafits, &hdu, "FREQCENT")?;
            (f * 1e6) as u64
        };
        let fine_channel_width = {
            let f: f64 = get_required_fits_key!(metafits, &hdu, "FINECHAN")?;
            (f * 1e3) as u64
        };
        let bandwidth = {
            let f: f64 = get_required_fits_key!(metafits, &hdu, "BANDWDTH")?;
            (f * 1e6) as u64
        };
        let base_freq = freq_centre - (bandwidth + fine_channel_width) / 2;

        let chansel: String = get_required_fits_key!(metafits, &hdu, "CHANSEL")?;
        let coarse_channels: Vec<u8> = chansel
            .split(',')
            .map(|s| {
                s.parse()
                    .expect("Failed to parse one of the channels in the metafits' CHANSEL")
            })
            .collect();
        let coarse_channel_width = bandwidth / coarse_channels.len() as u64;

        let lst: f64 = get_required_fits_key!(metafits, &hdu, "LST")?;
        let base_lst = lst.to_radians();
        let n_tiles = {
            let n: u64 = get_required_fits_key!(metafits, &hdu, "NINPUTS")?;
            n / 2
        };
        let n_baselines = n_tiles / 2 * (n_tiles - 1);

        let xyz = XYZ::get_baselines_metafits(metafits)?;

        Ok(Context {
            base_freq,
            bandwidth,
            coarse_channels,
            coarse_channel_width,
            fine_channel_width,
            base_lst,
            n_baselines,
            xyz,
        })
    }

    /// Convert to a C-compatible struct.
    pub fn convert(&self) -> *const Context_s {
        // Return a pointer to a C Context_s struct.
        Box::into_raw(Box::new(Context_s {
            fine_channel_width: self.fine_channel_width as f64,
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
Obs. base freq.:      {bf} MHz
Total obs. bandwidth: {bw} MHz
Coarse channels:      {cc:?}
Coarse channel width: {ccw} MHz
Fine channel width:   {fcw} kHz
Num. baselines:       {nbl}
Base LST:             {lst} rad"#,
            bf = self.base_freq as f64 / 1e6,
            bw = self.bandwidth as f64 / 1e6,
            cc = self.coarse_channels,
            ccw = self.coarse_channel_width as f64 / 1e6,
            fcw = self.fine_channel_width as f64 / 1e3,
            lst = self.base_lst,
            nbl = self.n_baselines,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_context_from_1065880128_metafits() -> Result<(), FitsError> {
        let metafits = std::path::PathBuf::from("tests/1065880128.metafits");
        let mut fptr = fits_open!(&metafits)?;
        let c = Context::new(&mut fptr)?;
        assert_eq!(c.base_freq, 167035000);
        assert_eq!(c.bandwidth, 30720000);
        assert_eq!(c.coarse_channel_width, 1280000);
        assert_eq!(c.fine_channel_width, 40000);
        assert_eq!(c.base_lst, 6.074823226561063);
        assert_eq!(c.n_baselines, 8128);
        assert_eq!(c.xyz.len(), 8128);

        Ok(())
    }
}
