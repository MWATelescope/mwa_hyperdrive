// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use crate::*;

use fitsio::FitsFile;

/// An observation context used throughout `hyperdrive.`
///
/// Frequencies are stored as integers to avoid floating-point issues.
#[derive(Debug)]
pub struct Context {
    /// The base frequency of the observation [Hz]. For reasons unknown, its
    /// calculation is insane.
    pub base_freq: usize,
    /// The total bandwidth of the observation [Hz]
    pub bandwidth: usize,
    /// The coarse channels used. These are typically 0 to 23.
    pub coarse_channels: Vec<u8>,
    /// The frequency width of a single coarse channel [Hz]
    pub coarse_channel_width: usize,
    /// The observation's frequency resolution [Hz]
    pub fine_channel_width: usize,
    /// The LST at the start of the observation [radians]
    pub base_lst: f64,
    /// The number of cross-correlation baselines
    pub n_baselines: usize,
    /// The XYZ baselines of the observations [metres]
    pub xyz: Vec<XyzBaseline>,
}

impl Context {
    /// Create a new `hyperdrive` observation `Context` from a metafits file.
    pub fn new(metafits: &mut FitsFile) -> Result<Self, fitsio::errors::Error> {
        let hdu = metafits.hdu(0)?;
        let freq_centre = (hdu
            .read_key::<String>(metafits, "FREQCENT")?
            .parse::<f64>()
            .expect("Couldn't parse FREQCENT from the metafits as f64")
            * 1e6) as usize;
        let fine_channel_width = (hdu
            .read_key::<String>(metafits, "FINECHAN")?
            .parse::<f64>()
            .expect("Couldn't parse FINECHAN from the metafits as f64")
            * 1e3) as usize;
        let bandwidth = (hdu
            .read_key::<String>(metafits, "BANDWDTH")?
            .parse::<f64>()
            .expect("Couldn't parse BANDWDTH from the metafits as f64")
            * 1e6) as usize;
        let base_freq = freq_centre - (bandwidth + fine_channel_width) / 2;

        let coarse_channels: Vec<u8> = hdu
            .read_key::<String>(metafits, "CHANSEL")?
            .split(',')
            .map(|s| {
                s.parse()
                    .expect("Failed to parse one of the channels in the metafits' CHANSEL")
            })
            .collect();
        let coarse_channel_width = bandwidth / coarse_channels.len() as usize;

        let base_lst = hdu
            .read_key::<String>(metafits, "LST")?
            .parse::<f64>()
            .expect("Couldn't parse LST from the metafits as f64")
            .to_radians();

        let n_tiles = hdu
            .read_key::<String>(metafits, "NINPUTS")?
            .parse::<usize>()
            .expect("Couldn't parse NINPUTS from the metafits as u32")
            / 2;
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
