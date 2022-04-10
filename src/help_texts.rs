// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Help texts for command-line interfaces.

use itertools::Itertools;
use strum::IntoEnumIterator;

use crate::data_formats::VisOutputType;
use mwa_hyperdrive_common::{itertools, lazy_static};

lazy_static::lazy_static! {
    pub(crate) static ref SOURCE_LIST_TYPE_HELP: String =
        format!("The type of sky-model source list. Valid types are: {}. If not specified, all types are attempted", *mwa_hyperdrive_srclist::SOURCE_LIST_TYPES_COMMA_SEPARATED);

    pub(crate) static ref VIS_OUTPUT_EXTENSIONS: String = VisOutputType::iter().join(", ");
}
