// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Code to handle reading from and writing to various data container formats.
 */

pub mod error;
pub(crate) mod ms;
pub(crate) mod raw;

pub(crate) use error::ReadInputDataError;
pub(crate) use ms::MS;
pub(crate) use raw::RawData;

use std::ops::Range;

use log::debug;
use ndarray::prelude::*;

use crate::context::FreqContext;
use mwa_hyperdrive_core::{c32, RADec, XyzBaseline, XYZ};

pub(crate) trait InputData: Sync + Send {
    /// TODO: What does this do? Read all baselines? Read all freqs?
    fn read(&self, time_range: Range<usize>) -> Result<Vec<Visibilities>, ReadInputDataError>;

    fn get_obsid(&self) -> u32;

    fn get_timesteps(&self) -> &[hifitime::Epoch];

    fn get_timestep_indices(&self) -> &Range<usize>;

    fn get_native_time_res(&self) -> f64;

    fn get_pointing(&self) -> &RADec;

    /// Get the centre frequencies of each of the coarse channels in this
    /// observation. If there are 24 coarse channels, then the returned slice
    /// has 24 elements.
    fn get_freq_context(&self) -> &FreqContext;

    fn get_tile_xyz(&self) -> &[XYZ];

    fn get_baseline_xyz(&self) -> &[XyzBaseline];

    /// Get the ideal electronic delays applied to MWA dipoles. Here, ideal
    /// means without any values of 32.
    fn get_ideal_delays(&self) -> &[u32];

    fn get_tile_flags(&self) -> &[usize];

    fn get_fine_chan_flags(&self) -> &[usize];

    // /// Set additional tile flags.
    // fn set_tile_flags(&mut self, tile_flags: Vec<usize>);

    // /// Set the fine channel flags *for each corase channel*.
    // fn set_fine_chan_flags(&mut self, fine_chan_flags: Vec<usize>)
    //     -> Result<(), NewInputDataError>;
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
