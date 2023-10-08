// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Parameters that are kept modular to be used in multiple aspects of
//! `hyperdrive`.
//!
//! The code here is kind of "mirroring" the code within the `cli` module; the
//! idea is that `cli` is unparsed, user-facing code, whereas parameters have
//! been parsed and are ready to be used directly. The code here should be
//! public to the entire `hyperdrive` crate.

mod di_calibration;
mod input_vis;
mod solutions_apply;
mod vis_convert;
mod vis_simulate;
mod vis_subtract;

#[cfg(test)]
pub(crate) use di_calibration::CalVis;
pub(crate) use di_calibration::{DiCalParams, DiCalibrateError};
pub(crate) use input_vis::InputVisParams;
pub(crate) use solutions_apply::SolutionsApplyParams;
pub(crate) use vis_convert::{VisConvertError, VisConvertParams};
pub(crate) use vis_simulate::{VisSimulateError, VisSimulateParams};
pub(crate) use vis_subtract::{VisSubtractError, VisSubtractParams};

use std::{num::NonZeroUsize, path::PathBuf};

use vec1::Vec1;

use crate::{averaging::Timeblock, io::write::VisOutputType};

pub(crate) struct OutputVisParams {
    pub(crate) output_files: Vec1<(PathBuf, VisOutputType)>,
    pub(crate) output_time_average_factor: NonZeroUsize,
    pub(crate) output_freq_average_factor: NonZeroUsize,
    pub(crate) output_autos: bool,
    pub(crate) output_timeblocks: Vec1<Timeblock>,

    /// Rather than writing out the entire input bandwidth, write out only the
    /// smallest contiguous band. e.g. Typical 40 kHz MWA data has 768 channels,
    /// but the first 2 and last 2 channels are usually flagged. Turning this
    /// option on means that 764 channels would be written out instead of 768.
    /// Note that other flagged channels in the band are unaffected, because the
    /// data written out must be contiguous.
    pub(crate) write_smallest_contiguous_band: bool,
}

#[derive(Clone, Copy)]
pub(crate) struct ModellingParams {
    pub(crate) apply_precession: bool,
}
