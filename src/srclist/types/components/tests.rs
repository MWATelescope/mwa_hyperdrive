// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::f64::consts::TAU;

use approx::assert_abs_diff_eq;
use marlu::{Jones, RADec};
use vec1::vec1;

use crate::srclist::{
    ComponentList, ComponentType, FluxDensity, FluxDensityType, SourceList, SourceListType,
};

fn get_srclist() -> SourceList {
    let (mut source_list, _) = crate::srclist::read::read_source_list_file(
        "test_files/1090008640/srclist_pumav3_EoR0aegean_EoR1pietro+ForA_1090008640_100.yaml",
        Some(SourceListType::Hyperdrive),
    )
    .unwrap();

    // Prune all but four sources from the source list.
    let sources_to_keep = [
        "J000042-342358",
        "J000045-272248",
        "J000816-193957",
        "J001612-312330A",
    ];
    let mut sources_to_be_removed = vec![];
    for (name, _) in source_list.iter() {
        if !sources_to_keep.contains(&name.as_str()) {
            sources_to_be_removed.push(name.to_owned());
        }
    }
    for name in sources_to_be_removed {
        source_list.remove(&name);
    }
    source_list
}

#[test]
fn test_split_components() {
    let freqs = [180e6];
    let phase_centre = RADec::from_degrees(0.0, -27.0);
    let srclist = get_srclist();

    let num_point_components = srclist.values().fold(0, |a, src| {
        a + src
            .components
            .iter()
            .filter(|comp| matches!(comp.comp_type, ComponentType::Point))
            .count()
    });
    let num_gauss_components = srclist.values().fold(0, |a, src| {
        a + src
            .components
            .iter()
            .filter(|comp| matches!(comp.comp_type, ComponentType::Gaussian { .. }))
            .count()
    });

    let split_components = ComponentList::new(&srclist, &freqs, phase_centre);
    let points = split_components.points;
    let gaussians = split_components.gaussians;
    let shapelets = split_components.shapelets;

    assert_eq!(points.radecs.len(), num_point_components);
    assert_eq!(points.radecs.len(), 2);
    assert_eq!(gaussians.radecs.len(), num_gauss_components);
    assert_eq!(gaussians.radecs.len(), 4);
    assert!(shapelets.radecs.is_empty());

    assert_eq!(points.lmns.len(), num_point_components);
    assert_eq!(gaussians.lmns.len(), num_gauss_components);
    assert!(shapelets.lmns.is_empty());
    assert_abs_diff_eq!(points.lmns[1].l / TAU, 0.0025326811687516274);
    assert_abs_diff_eq!(points.lmns[1].m / TAU, -0.12880688061967666);
    assert_abs_diff_eq!(points.lmns[1].n / TAU + 1.0, 0.9916664625927036);

    assert_eq!(points.flux_densities.dim(), (1, num_point_components));
    assert_eq!(gaussians.flux_densities.dim(), (1, num_gauss_components));
    assert_eq!(shapelets.flux_densities.dim(), (1, 0));

    // Test one of the component's instrumental flux densities.
    let fd = FluxDensityType::List(vec1![
        FluxDensity {
            freq: 80e6,
            i: 2.13017,
            ..Default::default()
        },
        FluxDensity {
            freq: 240e6,
            i: 0.33037,
            ..Default::default()
        },
    ])
    .estimate_at_freq(freqs[0]);
    let inst_fd: Jones<f64> = fd.to_inst_stokes();

    assert_abs_diff_eq!(gaussians.flux_densities[[0, 1]], inst_fd);
}
