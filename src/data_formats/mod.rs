// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to handle reading from and writing to various data container formats.

mod error;
pub mod metafits;
pub(crate) mod ms;
pub(crate) mod raw;
pub(crate) mod uvfits;

pub(crate) use error::ReadInputDataError;
pub(crate) use ms::MS;
pub(crate) use raw::RawData;
pub(crate) use uvfits::Uvfits;

use std::collections::{HashMap, HashSet};

use ndarray::prelude::*;
use strum_macros::{Display, EnumIter, EnumString};

use crate::context::{FreqContext, ObsContext};
use mwa_rust_core::Jones;

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
    fn read_crosses_and_autos(
        &self,
        cross_data_array: ArrayViewMut2<Jones<f32>>,
        cross_weights_array: ArrayViewMut2<f32>,
        auto_data_array: ArrayViewMut2<Jones<f32>>,
        auto_weights_array: ArrayViewMut2<f32>,
        timestep: usize,
        tile_to_unflagged_baseline_map: &HashMap<(usize, usize), usize>,
        flagged_tiles: &HashSet<usize>,
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
        flagged_tiles: &HashSet<usize>,
        flagged_fine_chans: &HashSet<usize>,
    ) -> Result<(), ReadInputDataError>;
}
