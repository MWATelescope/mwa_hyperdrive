// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to read in hyperdrive source lists.

use std::f64::consts::{FRAC_PI_2, TAU};

use marlu::RADec;

use crate::srclist::{error::ReadSourceListError, SourceList};

/// Convert a yaml file to a [`SourceList`].
pub(crate) fn source_list_from_yaml<T: std::io::BufRead>(
    buf: &mut T,
) -> Result<SourceList, ReadSourceListError> {
    let sl: SourceList = serde_yaml::from_reader(buf)?;

    // Complain if we spot something wrong.
    for comp in sl.values().flat_map(|s| s.components.iter()) {
        let RADec { ra, dec } = comp.radec;
        if !(0.0..TAU).contains(&ra) {
            return Err(ReadSourceListError::InvalidRa(ra.to_degrees()));
        }
        if !(-FRAC_PI_2..=FRAC_PI_2).contains(&dec) {
            return Err(ReadSourceListError::InvalidDec(dec.to_degrees()));
        }
    }

    Ok(sl)
}

/// Convert a json file to a [`SourceList`].
pub(crate) fn source_list_from_json<T: std::io::BufRead>(
    buf: &mut T,
) -> Result<SourceList, ReadSourceListError> {
    let sl: SourceList = serde_json::from_reader(buf)?;

    // Complain if we spot something wrong.
    for comp in sl.values().flat_map(|s| s.components.iter()) {
        let RADec { ra, dec } = comp.radec;
        if !(0.0..TAU).contains(&ra) {
            return Err(ReadSourceListError::InvalidRa(ra.to_degrees()));
        }
        if !(-FRAC_PI_2..=FRAC_PI_2).contains(&dec) {
            return Err(ReadSourceListError::InvalidDec(dec.to_degrees()));
        }
    }

    Ok(sl)
}
