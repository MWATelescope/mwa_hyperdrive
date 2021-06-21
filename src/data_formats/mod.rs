// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to handle reading from and writing to various data container formats.

mod error;
pub(crate) mod metafits;
pub(crate) mod ms;
pub(crate) mod raw;
pub(crate) mod uvfits;

pub(crate) use error::ReadInputDataError;
pub(crate) use ms::MS;
pub(crate) use raw::RawData;
pub(crate) use uvfits::Uvfits;

use std::collections::{HashMap, HashSet};

use ndarray::prelude::*;

use crate::context::{FreqContext, ObsContext};
use mwa_hyperdrive_core::Jones;

pub(crate) trait InputData: Sync + Send {
    fn get_obs_context(&self) -> &ObsContext;

    fn get_freq_context(&self) -> &FreqContext;

    /// Read all frequencies and baselines for a single timestep into the
    /// `data_array`, returning the associated weights.
    fn read(
        &self,
        data_array: ArrayViewMut2<Jones<f32>>,
        timestep: usize,
        tile_to_unflagged_baseline_map: &HashMap<(usize, usize), usize>,
        flagged_fine_chans: &HashSet<usize>,
    ) -> Result<Array2<f32>, ReadInputDataError>;
}
