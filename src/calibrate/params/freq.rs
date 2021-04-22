// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Channel- and frequency-related parameters required for calibration and
associated functions.
 */

use std::collections::HashSet;
/// Channel- and frequency-related parameters required for calibration.
pub struct FrequencyParams {
    /// The target fine-channel frequency resolution \[Hz\].
    ///
    /// e.g. If the input data is in 40 kHz resolution and this variable was
    /// 80e3 Hz, then we average 2 scans worth of frequency data for
    /// calibration.
    ///
    /// In a perfect world, this variable would be an integer, but it's
    /// primarily used in floating-point calculations, so it's more convenient
    /// to store it as a float.
    pub res: f64,

    /// The number of unflagged fine-frequency channels per coarse band. For 40
    /// kHz data, this is probably 27 (5 channels flagged for each coarse band).
    pub(crate) num_unflagged_fine_chans_per_coarse_band: usize,

    /// The total number of unflagged fine-frequency channels. For 40 kHz data,
    /// this is probably 648 (5 channels flagged for each coarse band).
    pub(crate) num_unflagged_fine_chans: usize,

    /// The fine channels that have not been flagged.
    pub(crate) unflagged_fine_chans: Vec<usize>,

    /// The frequencies of each of the observation's unflagged fine channels
    /// \[Hz\].
    pub(crate) unflagged_fine_chan_freqs: Vec<f64>,

    /// The fine channels to be flagged across the entire observation. e.g. For
    /// a 40 kHz observation, there are 768 fine channels, and this could
    /// contain 0 and 767.
    pub(crate) fine_chan_flags: HashSet<usize>,
}
