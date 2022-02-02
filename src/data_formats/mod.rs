// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to handle reading from and writing to various data container formats.

mod error;
mod metafits;
mod ms;
mod raw;
mod uvfits;

pub(crate) use error::ReadInputDataError;
pub use metafits::*;
pub(crate) use ms::{MsReadError, MS};
pub(crate) use raw::{RawDataReader, RawReadError};
pub(crate) use uvfits::{UvfitsReadError, UvfitsReader, UvfitsWriteError, UvfitsWriter};

use std::collections::{HashMap, HashSet};

use marlu::Jones;
use ndarray::prelude::*;
use strum_macros::{Display, EnumIter, EnumString};
use vec1::Vec1;

use crate::context::{FreqContext, ObsContext};
use mwa_hyperdrive_common::{marlu, ndarray};

#[derive(Debug)]
pub(crate) enum VisInputType {
    Raw,
    MeasurementSet,
    Uvfits,
}

#[derive(Debug, Display, EnumIter, EnumString)]
pub(crate) enum VisOutputType {
    #[strum(serialize = "uvfits")]
    Uvfits,
}

pub(crate) trait InputData: Sync + Send {
    fn get_obs_context(&self) -> &ObsContext;

    fn get_freq_context(&self) -> &FreqContext;

    fn get_input_data_type(&self) -> VisInputType;

    /// Read cross- and auto-correlation visibilities for all frequencies and
    /// baselines in a single timestep into corresponding arrays.
    #[allow(clippy::too_many_arguments)]
    fn read_crosses_and_autos(
        &self,
        cross_data_array: ArrayViewMut2<Jones<f32>>,
        cross_weights_array: ArrayViewMut2<f32>,
        auto_data_array: ArrayViewMut2<Jones<f32>>,
        auto_weights_array: ArrayViewMut2<f32>,
        timestep: usize,
        tile_to_unflagged_baseline_map: &HashMap<(usize, usize), usize>,
        flagged_tiles: &[usize],
        flagged_fine_chans: &HashSet<usize>,
    ) -> Result<(), ReadInputDataError>;

    /// Read cross-correlation visibilities for all frequencies and baselines in
    /// a single timestep into the `data_array` and similar for the weights.
    fn read_crosses(
        &self,
        data_array: ArrayViewMut2<Jones<f32>>,
        weights_array: ArrayViewMut2<f32>,
        timestep: usize,
        tile_to_unflagged_baseline_map: &HashMap<(usize, usize), usize>,
        flagged_fine_chans: &HashSet<usize>,
    ) -> Result<(), ReadInputDataError>;

    /// Read auto-correlation visibilities for all frequencies and tiles in a
    /// single timestep into the `data_array` and similar for the weights.
    fn read_autos(
        &self,
        data_array: ArrayViewMut2<Jones<f32>>,
        weights_array: ArrayViewMut2<f32>,
        timestep: usize,
        flagged_tiles: &[usize],
        flagged_fine_chans: &HashSet<usize>,
    ) -> Result<(), ReadInputDataError>;
}
