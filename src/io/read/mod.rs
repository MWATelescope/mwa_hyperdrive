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
use log::{info, warn};
use marlu::{
    precession::precess_time, Jones, LatLngHeight, MwaObsContext as MarluMwaObsContext,
    PolConvention, RADec, StokesConvention, UvwFrame, XOrientation, XyzGeodetic, UVW,
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

/// The conventions a visibility file follows: what it records, else what its
/// provenance implies.
#[derive(Debug, Clone)]
pub(crate) struct FileConventions {
    pub(crate) pol_convention: PolConvention,
    pub(crate) uvw_frame: UvwFrame,
    pub(crate) feed_angles: Option<Vec<Vec<f64>>>,
}

impl FileConventions {
    /// Combine what a file records with what its provenance implies.
    ///
    /// A recorded Stokes convention (`pyuvdata_polconv` / `POLCONV`) and UVW
    /// frame (`marlu_uvw_frame`) win. The recorded feed angles decide the X
    /// orientation (X within 45° of east is east-west, as pyuvdata derives
    /// its `x_orientation` from `feed_angle`), except that an MWA file
    /// (`is_mwa`) whose feed angles are not trusted (`feed_angles_trusted`:
    /// the file was written by something that records conventions, i.e.
    /// has a Stokes convention or UVW frame keyword, or pyuvdata's
    /// `pyuvdata_has_feed`) is taken as X east-west, as cotter, Birli and
    /// hyperdrive have always written while recording the IAU angles. The
    /// frame follows the UVW column's baseline sign when it was checked
    /// (`uvw_sign_differs`, relative to antenna1 - antenna2), else the
    /// provenance, else `fallback_frame`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn infer(
        what: &str,
        stokes: Option<StokesConvention>,
        uvw_frame: Option<UvwFrame>,
        feed_angles: Option<Vec<Vec<f64>>>,
        feed_angles_trusted: bool,
        is_mwa: bool,
        is_oskar: bool,
        uvw_sign_differs: Option<bool>,
        fallback_frame: UvwFrame,
    ) -> FileConventions {
        let x_angle = feed_angles
            .as_ref()
            .and_then(|f| f.first())
            .and_then(|f| f.first())
            .copied();
        let (x, x_src) = match (is_mwa, feed_angles_trusted, x_angle) {
            (true, false, _) | (true, true, None) => {
                (XOrientation::East, "the file being MWA data")
            }
            (_, _, Some(x_angle)) => (
                XOrientation::from_feed_angle_rad(x_angle),
                "the recorded feed angles",
            ),
            (false, _, None) => (XOrientation::North, "the default for non-MWA data"),
        };
        let stokes = stokes.unwrap_or_default();
        // The UVW column's baseline sign, when it could be checked against
        // the antenna positions, decides between the frames' two signs.
        let (frame, frame_src) = match (uvw_frame, uvw_sign_differs, is_mwa, is_oskar) {
            (Some(f), Some(differs), _, _) => {
                if (f.baseline_sign() < 0.0) != differs {
                    warn!(
                        "The marlu_uvw_frame keyword says {f}, but the UVW column has the {} baseline sign; using the keyword",
                        if differs { "opposite" } else { "same" }
                    );
                }
                (f, "the marlu_uvw_frame keyword")
            }
            (Some(f), None, _, _) => (f, "the marlu_uvw_frame keyword"),
            (None, Some(true), _, _) => (
                UvwFrame::Casacore,
                "the UVW column's antenna2 - antenna1 baseline sign",
            ),
            (None, Some(false), true, _) => (UvwFrame::Hyperdrive, "the file being MWA data"),
            (None, Some(false), false, true) => {
                (UvwFrame::Oskar, "the PHASED_ARRAY table OSKAR writes")
            }
            (None, Some(false), false, false) => (
                UvwFrame::Hyperdrive,
                "the UVW column's antenna1 - antenna2 baseline sign",
            ),
            (None, None, true, _) => (UvwFrame::Hyperdrive, "the file being MWA data"),
            (None, None, false, true) => (UvwFrame::Oskar, "the PHASED_ARRAY table OSKAR writes"),
            (None, None, false, false) => (fallback_frame, "the default for this file type"),
        };
        let pol_convention = PolConvention {
            x_orientation: x,
            stokes,
        };
        info!("{what} polarisation convention: {pol_convention} (from {x_src}); UVW frame: {frame} (from {frame_src})");
        FileConventions {
            pol_convention,
            uvw_frame: frame,
            feed_angles,
        }
    }
}

#[cfg(test)]
mod convention_tests {
    use super::*;
    use std::f64::consts::FRAC_PI_2;

    fn infer(
        stokes: Option<StokesConvention>,
        frame: Option<UvwFrame>,
        feed_angles: Option<Vec<Vec<f64>>>,
        is_mwa: bool,
        is_oskar: bool,
        sign_differs: Option<bool>,
    ) -> FileConventions {
        // Feed angles are trusted when the file records conventions.
        let trusted = stokes.is_some() || frame.is_some();
        FileConventions::infer(
            "test",
            stokes,
            frame,
            feed_angles,
            trusted,
            is_mwa,
            is_oskar,
            sign_differs,
            UvwFrame::Hyperdrive,
        )
    }

    #[test]
    fn keywords_win() {
        let c = infer(
            Some(StokesConvention::Sum),
            Some(UvwFrame::Oskar),
            Some(vec![vec![0.0, FRAC_PI_2]]),
            false,
            false,
            Some(true),
        );
        // The feed angles say north, the keywords say sum and oskar.
        assert_eq!(c.pol_convention.x_orientation, XOrientation::North);
        assert_eq!(c.pol_convention.stokes, StokesConvention::Sum);
        assert_eq!(c.uvw_frame, UvwFrame::Oskar);
        assert_eq!(c.feed_angles, Some(vec![vec![0.0, FRAC_PI_2]]));
    }

    #[test]
    fn mwa_data_are_east_west_in_the_hyperdrive_frame() {
        // cotter records IAU feed angles for MWA data; they must be ignored.
        let c = infer(
            None,
            None,
            Some(vec![vec![0.0, FRAC_PI_2]]),
            true,
            false,
            Some(false),
        );
        assert_eq!(c.pol_convention, PolConvention::MWA);
        assert_eq!(c.uvw_frame, UvwFrame::Hyperdrive);
    }

    #[test]
    fn mwa_files_from_convention_aware_writers_trust_their_feed_angles() {
        // An MWA file written by Marlu (with conventions) or pyuvdata records
        // real feed angles; IAU angles there mean IAU.
        let c = infer(
            Some(StokesConvention::Avg),
            None,
            Some(vec![vec![0.0, FRAC_PI_2]]),
            true,
            false,
            Some(false),
        );
        assert_eq!(c.pol_convention, PolConvention::IAU);
        // ... and without feed angles, MWA data are still east-west.
        let c = infer(Some(StokesConvention::Avg), None, None, true, false, None);
        assert_eq!(c.pol_convention, PolConvention::MWA);
    }

    #[test]
    fn feed_angles_decide_the_x_orientation() {
        let c = infer(
            None,
            None,
            Some(vec![vec![0.0, FRAC_PI_2]]),
            false,
            false,
            None,
        );
        assert_eq!(c.pol_convention, PolConvention::IAU);
        let c = infer(
            None,
            None,
            Some(vec![vec![FRAC_PI_2, 0.0]]),
            false,
            false,
            None,
        );
        assert_eq!(c.pol_convention, PolConvention::MWA);
        // Just under 45 degrees from east still counts as east.
        let c = infer(
            None,
            None,
            Some(vec![vec![FRAC_PI_2 - 0.7, 0.0]]),
            false,
            false,
            None,
        );
        assert_eq!(c.pol_convention.x_orientation, XOrientation::East);
    }

    #[test]
    fn non_mwa_data_without_feed_angles_default_to_iau() {
        let c = infer(None, None, None, false, false, None);
        assert_eq!(c.pol_convention, PolConvention::IAU);
        assert_eq!(c.uvw_frame, UvwFrame::Hyperdrive, "fallback frame");
    }

    #[test]
    fn uvw_sign_decides_the_frame() {
        let c = infer(None, None, None, false, false, Some(true));
        assert_eq!(c.uvw_frame, UvwFrame::Casacore);
        let c = infer(None, None, None, false, false, Some(false));
        assert_eq!(c.uvw_frame, UvwFrame::Hyperdrive);
        // OSKAR uses the ant1 - ant2 sign but no precession.
        let c = infer(None, None, None, false, true, Some(false));
        assert_eq!(c.uvw_frame, UvwFrame::Oskar);
        let c = infer(None, None, None, false, true, None);
        assert_eq!(c.uvw_frame, UvwFrame::Oskar);
        // A recorded frame beats a contradicting UVW sign (with a warning).
        let c = infer(
            None,
            Some(UvwFrame::Casacore),
            None,
            false,
            false,
            Some(false),
        );
        assert_eq!(c.uvw_frame, UvwFrame::Casacore);
    }
}
