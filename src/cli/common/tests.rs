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

/// check that num-sources correctly limits the number of sources in skymodel parse
#[test]
fn skymodel_veto_parse_num() {
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

    // First, verify that filtering out a nonexistent source causes the correct error.
    let result = SkyModelWithVetoArgs {
        source_list: source_list.clone(),
        num_sources: Some(3),
        ..Default::default()
    }
    .parse(
        RADec::from_degrees(0.0, -30.0),
        0.0,
        MWA_LAT_RAD,
        &[150e6],
        &*beam,
    );
    assert!(result.is_ok());
    let sl = result.unwrap();
    assert!(sl.len() == 3);
}

/// check that named source filterng results only in the named sources
#[test]
fn skymodel_veto_parse_named() {
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

    let named_sources = vec!["J002549-260211".to_string(), "J002430-292847".to_string()];

    let result = SkyModelWithVetoArgs {
        source_list: source_list.clone(),
        named_sources: Some(named_sources.clone()),
        invert: Some(false),
        ..Default::default()
    }
    .parse(
        RADec::from_degrees(0.0, -30.0),
        0.0,
        MWA_LAT_RAD,
        &[150e6],
        &*beam,
    );
    assert!(result.is_ok());
    let sl = result.unwrap();
    assert!(sl.len() == 2);
    assert!(sl.contains_key("J002549-260211"));
    assert!(sl.contains_key("J002430-292847"));
}

/// check that named source filterng results in everything but the named sources
#[test]
fn skymodel_veto_parse_named_invert() {
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

    let named_sources = vec!["J002549-260211".to_string(), "J002430-292847".to_string()];

    let result = SkyModelWithVetoArgs {
        source_list: source_list.clone(),
        named_sources: Some(named_sources.clone()),
        invert: Some(true),
        ..Default::default()
    }
    .parse(
        RADec::from_degrees(0.0, -30.0),
        0.0,
        MWA_LAT_RAD,
        &[150e6],
        &*beam,
    );
    assert!(result.is_ok());
    let sl = result.unwrap();
    assert!(sl.len() == 98);
    assert!(!sl.contains_key("J002549-260211"));
    assert!(!sl.contains_key("J002430-292847"));
}

/// check that MissingSource results from nonexistent named_sources in parse_named_invert
#[test]
fn skymodel_veto_parse_named_invert_missing_source() {
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

    // First, verify that filtering out a nonexistent source causes the correct error.
    let result = SkyModelWithVetoArgs {
        source_list: source_list.clone(),
        named_sources: Some(vec!["nonexistent_source".to_string()]),
        invert: Some(false),
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
        ReadSourceListError::MissingNamedSource { name } if name == "nonexistent_source"
    ));
}
/// check that NamedSourcesAndNumSources results from --num-sources and --sources-to-subtract mismatch
#[test]
fn skymodel_veto_parse_named_and_num_sources() {
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

    // First, verify that filtering out a list of sources that with a different number of sources
    // causes the correct error.
    let result = SkyModelWithVetoArgs {
        source_list: source_list.clone(),
        num_sources: Some(3),
        named_sources: Some(vec!["J002549-260211".to_string()]),
        invert: Some(false),
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
        ReadSourceListError::NamedSourcesAndNumSources { num_sources, named_sources } if num_sources == 3 && named_sources == 1
    ));
}

/// check when --invert, --num-sources and --sources-to-subtract are all provided
#[test]
fn skymodel_veto_parse_named_and_num_sources_invert() {
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

    // First, verify that filtering out a list of sources that with a different number of sources
    // causes the correct error.
    let result = SkyModelWithVetoArgs {
        source_list: source_list.clone(),
        num_sources: Some(3),
        named_sources: Some(vec!["J002549-260211".to_string()]),
        invert: Some(true),
        ..Default::default()
    }
    .parse(
        RADec::from_degrees(0.0, -30.0),
        0.0,
        MWA_LAT_RAD,
        &[150e6],
        &*beam,
    );
    assert!(result.is_ok());
    let sl = result.unwrap();
    assert!(sl.len() == 3);
    assert!(!sl.contains_key("J002549-260211"));
}

// check that AllSourcesFiltered results from --invert and comprehensive --sources-to-subtract in parse_named_invert
#[test]
fn skymodel_veto_parse_named_and_num_sources_invert_all_sources() {
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

    // First, get the full list of source names
    let result = SkyModelWithVetoArgs {
        source_list: source_list.clone(),
        invert: Some(true),
        ..Default::default()
    }
    .parse(
        RADec::from_degrees(0.0, -30.0),
        0.0,
        MWA_LAT_RAD,
        &[150e6],
        &*beam,
    );
    assert!(result.is_ok());
    let sl = result.unwrap();
    assert!(sl.len() == 100);
    let named_sources: Vec<String> = sl.keys().cloned().collect();

    // now invert the full list of sources
    let result = SkyModelWithVetoArgs {
        source_list: source_list.clone(),
        named_sources: Some(named_sources.clone()),
        invert: Some(true),
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
        ReadSourceListError::AllSourcesFiltered { invert } if invert == true
    ));
}
