// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Tests against command-line interfaces that aren't big enough to go in their
//! own modules.

use marlu::{constants::MWA_LAT_RAD, RADec};

use crate::srclist::ReadSourceListError;

use super::{BeamArgs, SkyModelWithVetoArgs};

#[test]
fn all_sources_vetoed_causes_error() {
    let beam = BeamArgs {
        no_beam: true,
        ..Default::default()
    }
    .parse(128, None, None, None, None)
    .expect("no problems setting up a NoBeam");

    let source_list = Some(
        "test_files/1090008640/srclist_pumav3_EoR0aegean_EoR1pietro+ForA_1090008640_peel100.txt"
            .to_string(),
    );

    // First, verify that vetoing all sources < 100 Jy leaves nothing behind.
    let result = SkyModelWithVetoArgs {
        source_list: source_list.clone(),
        veto_threshold: Some(100.0),
        ..Default::default()
    }
    .parse(
        RADec::from_degrees(0.0, -30.0),
        0.0,
        MWA_LAT_RAD,
        &[150e6],
        &*beam,
    );
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        ReadSourceListError::NoSourcesAfterVeto
    ));

    // Deliberately set the number of sources to 0.
    let result = SkyModelWithVetoArgs {
        source_list: source_list.clone(),
        num_sources: Some(0),
        ..Default::default()
    }
    .parse(
        RADec::from_degrees(0.0, -30.0),
        0.0,
        MWA_LAT_RAD,
        &[150e6],
        &*beam,
    );
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        ReadSourceListError::NoSources
    ));

    // Set the source dist cutoff to something not useful.
    let result = SkyModelWithVetoArgs {
        source_list: source_list.clone(),
        source_dist_cutoff: Some(0.01),
        ..Default::default()
    }
    .parse(
        RADec::from_degrees(0.0, -30.0),
        0.0,
        MWA_LAT_RAD,
        &[150e6],
        &*beam,
    );
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        ReadSourceListError::NoSourcesAfterVeto
    ));
}
