// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Help texts for command-line interfaces.

use crate::{
    constants::{
        DEFAULT_CUTOFF_DISTANCE, DEFAULT_VETO_THRESHOLD, MWA_HEIGHT_M, MWA_LAT_DEG, MWA_LONG_DEG,
    },
    pfb_gains::{DEFAULT_PFB_FLAVOUR, PFB_FLAVOURS},
    srclist::SOURCE_LIST_TYPES_COMMA_SEPARATED,
};

lazy_static::lazy_static! {
    pub(crate) static ref PFB_FLAVOUR_HELP: String =
        format!("The 'flavour' of poly-phase filter bank corrections applied to raw MWA data. The default is '{}'. Valid flavours are: {}", DEFAULT_PFB_FLAVOUR, *PFB_FLAVOURS);

    pub(crate) static ref ARRAY_POSITION_HELP: String =
        format!("The Earth longitude, latitude, and height of the instrumental array [degrees, degrees, meters]. Default (MWA): ({MWA_LONG_DEG}°, {MWA_LAT_DEG}°, {MWA_HEIGHT_M}m)");

    pub(crate) static ref DIPOLE_DELAYS_HELP: String =
        format!("If specified, use these dipole delays for the MWA pointing. e.g. 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3");

    pub(crate) static ref SOURCE_LIST_TYPE_HELP: String =
        format!("The type of sky-model source list. Valid types are: {}. If not specified, all types are attempted", *SOURCE_LIST_TYPES_COMMA_SEPARATED);

    pub(crate) static ref SOURCE_DIST_CUTOFF_HELP: String =
        format!("Specifies the maximum distance from the phase centre a source can be [degrees]. Default: {DEFAULT_CUTOFF_DISTANCE}");

    pub(crate) static ref VETO_THRESHOLD_HELP: String =
        format!("Specifies the minimum Stokes XX+YY a source must have before it gets vetoed [Jy]. Default: {DEFAULT_VETO_THRESHOLD}");

    pub(crate) static ref SOURCE_LIST_INPUT_TYPE_HELP: String =
        format!("Specifies the type of the input source list. Currently supported types: {}",
                    *SOURCE_LIST_TYPES_COMMA_SEPARATED);

    pub(crate) static ref SOURCE_LIST_OUTPUT_TYPE_HELP: String =
        format!("Specifies the type of the output source list. May be required depending on the output filename. Currently supported types: {}",
                *SOURCE_LIST_TYPES_COMMA_SEPARATED);

    pub(crate) static ref SRCLIST_BY_BEAM_OUTPUT_TYPE_HELP: String =
        format!("Specifies the type of the output source list. If not specified, the input source list type is used. Currently supported types: {}",
                *SOURCE_LIST_TYPES_COMMA_SEPARATED);
}

pub(crate) const MS_DATA_COL_NAME_HELP: &str = "If reading from a measurement set, this specifies the column to use in the main table containing visibilities. Default: DATA";
