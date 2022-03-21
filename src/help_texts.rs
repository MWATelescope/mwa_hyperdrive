// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Help texts for command-line interfaces.

use itertools::Itertools;
use strum::IntoEnumIterator;

use crate::{
    constants::*,
    pfb_gains::{DEFAULT_PFB_FLAVOUR, PFB_FLAVOURS},
    solutions::CalSolutionType,
    vis_io::write::VisOutputType,
};
use mwa_hyperdrive_common::{itertools, lazy_static};

lazy_static::lazy_static! {
    pub(crate) static ref SOURCE_LIST_TYPE_HELP: String =
        format!("The type of sky-model source list. Valid types are: {}. If not specified, all types are attempted", *mwa_hyperdrive_srclist::SOURCE_LIST_TYPES_COMMA_SEPARATED);

    pub(crate) static ref VIS_OUTPUT_EXTENSIONS: String = VisOutputType::iter().join(", ");

    pub(crate) static ref CAL_SOL_EXTENSIONS: String = CalSolutionType::iter().join(", ");

    pub(crate) static ref PFB_FLAVOUR_HELP: String =
        format!("The 'flavour' of poly-phase filter bank corrections applied to raw MWA data. The default is '{}'. Valid flavours are: {}", DEFAULT_PFB_FLAVOUR, *PFB_FLAVOURS);

    pub(crate) static ref ARRAY_POSITION_HELP: String =
        format!("The Earth longitude, latitude, and height of the instrumental array [degrees, degrees, meters]. Default (MWA): ({}°, {}°, {}m)", MWA_LONG_DEG, MWA_LAT_DEG, MWA_HEIGHT_M);

    pub(crate) static ref DIPOLE_DELAYS_HELP: String =
        format!("If specified, use these dipole delays for the MWA pointing. e.g. 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3");
}
