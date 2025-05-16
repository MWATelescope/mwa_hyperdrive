// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to handle reading from and writing to various data container formats.

mod error;
pub(crate) mod fits;
mod ms;
mod raw;
mod uvfits;

pub(crate) use error::VisReadError;
pub(crate) use ms::MsReadError;
pub use ms::MsReader;
pub(crate) use raw::{pfb_gains, RawReadError};
pub use raw::{RawDataCorrections, RawDataReader};
pub(crate) use uvfits::UvfitsReadError;
pub use uvfits::UvfitsReader;

use std::collections::HashSet;

use hifitime::{Duration, Epoch};
use marlu::{
    precession::precess_time, Jones, LatLngHeight, MwaObsContext as MarluMwaObsContext, RADec,
    XyzGeodetic, UVW,
};
use mwalib::MetafitsContext;
use ndarray::prelude::*;
use vec1::Vec1;

use crate::{context::ObsContext, flagging::MwafFlags, math::TileBaselineFlags};

#[derive(Debug, Clone, Copy)]
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

    /// Get the raw data corrections that will be applied to the visibilities as
    /// they're read in. These may be distinct from what the user specified.
    fn get_raw_data_corrections(&self) -> Option<RawDataCorrections>;

    /// Set the raw data corrections that will be applied to the visibilities as
    /// they're read in. These are only applied to raw data.
    fn set_raw_data_corrections(&mut self, corrections: RawDataCorrections);

    fn read_inner_dispatch(
        &self,
        cross_data: Option<CrossData>,
        auto_data: Option<AutoData>,
        timestep: usize,
        flagged_fine_chans: &HashSet<u16>,
    ) -> Result<(), VisReadError>;

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
        flagged_fine_chans: &HashSet<u16>,
    ) -> Result<(), VisReadError> {
        self.read_inner_dispatch(
            Some(CrossData {
                vis_fb: cross_vis_fb,
                weights_fb: cross_weights_fb,
                tile_baseline_flags,
            }),
            Some(AutoData {
                vis_fb: auto_vis_fb,
                weights_fb: auto_weights_fb,
                tile_baseline_flags,
            }),
            timestep,
            flagged_fine_chans,
        )
    }

    /// Read cross-correlation visibilities for all frequencies and baselines in
    /// a single timestep into the `data_array` and similar for the weights.
    fn read_crosses(
        &self,
        vis_fb: ArrayViewMut2<Jones<f32>>,
        weights_fb: ArrayViewMut2<f32>,
        timestep: usize,
        tile_baseline_flags: &TileBaselineFlags,
        flagged_fine_chans: &HashSet<u16>,
    ) -> Result<(), VisReadError> {
        self.read_inner_dispatch(
            Some(CrossData {
                vis_fb,
                weights_fb,
                tile_baseline_flags,
            }),
            None,
            timestep,
            flagged_fine_chans,
        )
    }

    /// Read auto-correlation visibilities for all frequencies and tiles in a
    /// single timestep into the `data_array` and similar for the weights.
    #[allow(dead_code)] // this is used in tests
    fn read_autos(
        &self,
        vis_fb: ArrayViewMut2<Jones<f32>>,
        weights_fb: ArrayViewMut2<f32>,
        timestep: usize,
        tile_baseline_flags: &TileBaselineFlags,
        flagged_fine_chans: &HashSet<u16>,
    ) -> Result<(), VisReadError> {
        self.read_inner_dispatch(
            None,
            Some(AutoData {
                vis_fb,
                weights_fb,
                tile_baseline_flags,
            }),
            timestep,
            flagged_fine_chans,
        )
    }

    /// Get optional MWA information to give to `Marlu` when writing out
    /// visibilities.
    // The existence of this code is nothing but horrible. This optional info
    // is, to my knowledge, *only* useful because `wsclean` uses it to detect
    // MWA data (specifically via the MWA_TILE_POINTING table) and apply the MWA
    // FEE beam. The `Marlu` API should instead take MWA dipole delays.
    fn get_marlu_mwa_info(&self) -> Option<MarluMwaObsContext>;
}

/// A private container for cross-correlation data. It only exists to give
/// meaning to the types.
pub struct CrossData<'a, 'b, 'c> {
    pub vis_fb: ArrayViewMut2<'a, Jones<f32>>,
    pub weights_fb: ArrayViewMut2<'b, f32>,
    pub tile_baseline_flags: &'c TileBaselineFlags,
}

/// A private container for auto-correlation data. It only exists to give
/// meaning to the types.
pub struct AutoData<'a, 'b, 'c> {
    pub vis_fb: ArrayViewMut2<'a, Jones<f32>>,
    pub weights_fb: ArrayViewMut2<'b, f32>,
    pub tile_baseline_flags: &'c TileBaselineFlags,
}

/// With a dataset's UVW and the XYZs that correspond to it, compare with a UVW
/// that we form from the XYZs. This allows us to determine the "baseline order"
/// that the software that wrote this dataset used. `hyperdrive` and friends use
/// ant1-ant2, but others may use ant2-ant1. If we detect ant2-ant1, we know
/// that we have to conjugate this dataset's visibilities if want to continue
/// using ant1-ant2.
///
/// It is not anticipated that precession has an impact here.
fn baseline_convention_is_different(
    data_uvw: UVW,
    tile1_xyz: XyzGeodetic,
    tile2_xyz: XyzGeodetic,
    array_position: LatLngHeight,
    phase_centre: RADec,
    first_timestamp: Epoch,
    dut1: Option<Duration>,
) -> bool {
    let precession_info = precess_time(
        array_position.longitude_rad,
        array_position.latitude_rad,
        phase_centre,
        first_timestamp,
        dut1.unwrap_or_default(),
    );
    let xyzs = precession_info.precess_xyz(&[tile1_xyz, tile2_xyz]);
    let UVW {
        u: u_p1,
        v: v_p1,
        w: w_p1,
    } = UVW::from_xyz(
        xyzs[0] - xyzs[1],
        phase_centre.to_hadec(precession_info.lmst_j2000),
    );
    let UVW {
        u: u_p2,
        v: v_p2,
        w: w_p2,
    } = UVW::from_xyz(
        xyzs[1] - xyzs[0],
        phase_centre.to_hadec(precession_info.lmst_j2000),
    );

    // Which UVW is closer to the data?
    let UVW { u, v, w } = data_uvw;
    let diff1 = (u - u_p1).abs() + (v - v_p1).abs() + (w - w_p1).abs();
    let diff2 = (u - u_p2).abs() + (v - v_p2).abs() + (w - w_p2).abs();

    // If `diff2` is smaller than `diff1`, then the standard baseline order is
    // good, no need to do anything else. Otherwise, the other baseline order is
    // used; the tile XYZs need to be negated and the visibility data need to be
    // complex conjugated.
    diff2 < diff1
}
