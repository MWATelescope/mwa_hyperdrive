// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to handle reading from and writing to various data container formats.

mod error;
mod ms;
mod raw;
mod uvfits;

pub(crate) use error::VisReadError;
pub(crate) use ms::MsReader;
pub(crate) use raw::{RawDataCorrections, RawDataReader};
pub(crate) use uvfits::UvfitsReader;

use std::collections::HashSet;

use marlu::Jones;
use mwalib::MetafitsContext;
use ndarray::prelude::*;
use vec1::Vec1;

use crate::{context::ObsContext, flagging::MwafFlags, math::TileBaselineFlags};

#[derive(Debug)]
pub(crate) enum VisInputType {
    Raw,
    MeasurementSet,
    Uvfits,
}

pub(crate) trait VisRead: Sync + Send {
    fn get_obs_context(&self) -> &ObsContext;

    fn get_input_data_type(&self) -> VisInputType;

    /// If it's available, get a reference to the [`mwalib::MetafitsContext`]
    /// associated with this trait object.
    fn get_metafits_context(&self) -> Option<&MetafitsContext>;

    /// If it's available, get a reference to the [`MwafFlags`] associated with
    /// this trait object.
    fn get_flags(&self) -> Option<&MwafFlags>;

    /// Read cross- and auto-correlation visibilities for all frequencies and
    /// baselines in a single timestep into corresponding arrays.
    #[allow(clippy::too_many_arguments)]
    fn read_crosses_and_autos(
        &self,
        cross_vis_fb: ArrayViewMut2<Jones<f32>>,
        cross_weights_fb: ArrayViewMut2<f32>,
        auto_vis_fb: ArrayViewMut2<Jones<f32>>,
        auto_weights_fb: ArrayViewMut2<f32>,
        timestep: usize,
        tile_baseline_flags: &TileBaselineFlags,
        flagged_fine_chans: &HashSet<usize>,
    ) -> Result<(), VisReadError>;

    /// Read cross-correlation visibilities for all frequencies and baselines in
    /// a single timestep into the `data_array` and similar for the weights.
    fn read_crosses(
        &self,
        vis_fb: ArrayViewMut2<Jones<f32>>,
        weights_fb: ArrayViewMut2<f32>,
        timestep: usize,
        tile_baseline_flags: &TileBaselineFlags,
        flagged_fine_chans: &HashSet<usize>,
    ) -> Result<(), VisReadError>;

    /// Read auto-correlation visibilities for all frequencies and tiles in a
    /// single timestep into the `data_array` and similar for the weights.
    fn read_autos(
        &self,
        vis_fb: ArrayViewMut2<Jones<f32>>,
        weights_fb: ArrayViewMut2<f32>,
        timestep: usize,
        tile_baseline_flags: &TileBaselineFlags,
        flagged_fine_chans: &HashSet<usize>,
    ) -> Result<(), VisReadError>;
}

/// A private container for cross-correlation data. It only exists to give
/// meaning to the types.
struct CrossData<'a, 'b, 'c> {
    vis_fb: ArrayViewMut2<'a, Jones<f32>>,
    weights_fb: ArrayViewMut2<'b, f32>,
    tile_baseline_flags: &'c TileBaselineFlags,
}

/// A private container for auto-correlation data. It only exists to give
/// meaning to the types.
struct AutoData<'a, 'b, 'c> {
    vis_fb: ArrayViewMut2<'a, Jones<f32>>,
    weights_fb: ArrayViewMut2<'b, f32>,
    tile_baseline_flags: &'c TileBaselineFlags,
}
