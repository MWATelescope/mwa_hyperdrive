// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Code to handle reading from and writing to various data container formats.
 */

pub(crate) mod error;
pub(crate) mod ms;
pub(crate) mod raw;

pub use error::ReadInputDataError;
pub(crate) use ms::MS;
pub(crate) use raw::RawData;

use std::ops::Range;

use ndarray::prelude::*;

use crate::context::{FreqContext, ObsContext};

pub(crate) trait InputData: Sync + Send {
    fn get_obs_context(&self) -> &ObsContext;

    fn get_freq_context(&self) -> &FreqContext;

    /// TODO: What does this do? Read all baselines? Read all freqs?
    fn read(&self, time_range: Range<usize>) -> Result<Vec<Visibilities>, ReadInputDataError>;
}

/// Three floats corresponding to the real and imag parts of a visibility, as
/// well as the weight of the visibility.
pub(crate) struct Vis {
    /// Real component.
    re: f32,

    /// Imaginary component.
    im: f32,

    /// Weight.
    w: f32,
}

pub(crate) enum VisType {
    /// Visibilities are ordered [time][freq][baseline][pol].
    TimeFreqBlPol,

    /// Visibilities are ordered [freq][time][baseline][pol].
    FreqTimeBlPol,
}
/// A struct containing visibilities, as well as information about them.
///
/// It is assumed that all available baselines are included, as well as all
/// polarisations (XX, XY, YX, YY). Any baselines that should be flagged need to
/// be handled after this struct is created.
pub(crate) struct Visibilities {
    /// The ndarray of visibilities. The other struct fields detail the axes.
    pub(crate) vis: Array4<Vis>,

    /// How are the visibilities arranged?
    pub(crate) vis_type: VisType,

    /// The frequency range for these visibilities (Hz, exclusive).
    pub(crate) freq_range: Range<f64>,
}
