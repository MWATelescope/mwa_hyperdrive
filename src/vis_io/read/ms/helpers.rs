// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to help interface with CASA measurement sets.

use std::path::Path;

use marlu::{rubbl_casatables, LatLngHeight, XyzGeocentric, XyzGeodetic};
use rayon::prelude::*;
use rubbl_casatables::{Table, TableOpenMode};

use super::error::*;

/// Open a measurement set table read only. If `table` is `None`, then open the
/// base table.
pub(super) fn read_table(ms: &Path, table: Option<&str>) -> Result<Table, MsReadError> {
    match Table::open(
        &format!("{}/{}", ms.display(), table.unwrap_or("")),
        TableOpenMode::Read,
    ) {
        Ok(t) => Ok(t),
        Err(e) => Err(MsReadError::RubblError(e.to_string())),
    }
}

/// casacore's antenna positions are [XyzGeocentric] coordinates, but we use
/// [XyzGeodetic] coordinates in hyperdrive. This function converts the casacore
/// positions.
pub(super) fn casacore_positions_to_local_xyz(
    pos: &[XyzGeocentric],
    array_pos: LatLngHeight,
) -> Result<Vec<XyzGeodetic>, MsReadError> {
    let vec = XyzGeocentric::get_geocentric_vector(array_pos)
        .map_err(|_| MsReadError::Geodetic2Geocentric)?;
    let (s_long, c_long) = array_pos.longitude_rad.sin_cos();
    Ok(pos
        .par_iter()
        .map(|xyz| xyz.to_geodetic_inner(vec, s_long, c_long))
        .collect())
}
