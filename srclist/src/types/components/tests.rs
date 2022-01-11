// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::f64::consts::TAU;

use approx::assert_abs_diff_eq;
use marlu::{Jones, RADec};

use crate::{
    jones_test::TestJones,
    marlu,
    types::{ComponentList, ComponentType, FluxDensity, FluxDensityType, SourceList},
    SourceListType,
};

fn get_small_source_list() -> SourceList {
    let (mut source_list, _) = crate::read::read_source_list_file(
        "test_files/srclist_pumav3_EoR0aegean_EoR1pietro+ForA_1090008640_100.yaml",
        Some(SourceListType::Hyperdrive),
    )
    .unwrap();

    // Prune all but two sources from the source list.
    let sources_to_keep = ["J000042-342358", "J000045-272248"];
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

fn get_big_source_list() -> SourceList {
    let (mut source_list, _) = crate::read::read_source_list_file(
        "test_files/srclist_pumav3_EoR0aegean_EoR1pietro+ForA_1090008640_100.yaml",
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

// fn get_instrumental_flux_densities_for_srclist(
//     srclist: &SourceList,
//     freqs: &[f64],
// ) -> Array2<Jones<f64>> {
//     let mut comp_fds: Vec<FluxDensityType> = vec![];
//     for comp in srclist.iter().flat_map(|(_, src)| &src.components) {
//         match comp.comp_type {
//             ComponentType::Point => {
//                 comp_fds.push(comp.flux_type.clone());
//             }
//             ComponentType::Gaussian { .. } => {
//                 comp_fds.push(comp.flux_type.clone());
//             }
//             ComponentType::Shapelet { .. } => {
//                 comp_fds.push(comp.flux_type.clone());
//             }
//         }
//     }

//     get_instrumental_flux_densities(&comp_fds, freqs)
// }

// #[test]
// fn test_beam_correct_flux_densities_no_beam() {
//     let freqs = [170e6];
//     let lst = 6.261977848;
//     let dipole_gains = [1.0; 16];

//     let beam: Box<dyn Beam> = Box::new(NoBeam);
//     let srclist = get_small_source_list();
//     let inst_flux_densities = get_instrumental_flux_densities_for_srclist(&srclist, &freqs);
//     let result = match beam_correct_flux_densities_inner(
//         inst_flux_densities.view(),
//         beam.deref(),
//         &srclist.get_azel_mwa(lst),
//         &dipole_gains,
//         &freqs,
//     ) {
//         Ok(fds) => fds,
//         Err(e) => panic!("{}", e),
//     };
//     let num_components = srclist.values().fold(0, |a, src| a + src.components.len());
//     assert_eq!(result.dim(), (freqs.len(), num_components));

//     // Hand-verified results.
//     let expected_comp_fd_1 = Jones::from([
//         Complex::new(2.7473072919275476, 0.0),
//         Complex::new(0.0, 0.0),
//         Complex::new(0.0, 0.0),
//         Complex::new(2.7473072919275476, 0.0),
//     ]);
//     let expected_comp_fd_2 = Jones::from([
//         Complex::new(1.7047163998893684, 0.0),
//         Complex::new(0.0, 0.0),
//         Complex::new(0.0, 0.0),
//         Complex::new(1.7047163998893684, 0.0),
//     ]);
//     assert_abs_diff_eq!(result[[0, 0]], expected_comp_fd_1, epsilon = 1e-10);
//     assert_abs_diff_eq!(result[[0, 1]], expected_comp_fd_2, epsilon = 1e-10);
// }

// #[test]
// #[serial]
// fn test_beam_correct_flux_densities_170_mhz() {
//     let freqs = [170e6];
//     let lst = 6.261977848;
//     let dipole_delays = vec![0; 16];

//     let beam: Box<dyn Beam> =
//         Box::new(FEEBeam::new_from_env(1, Delays::Partial(dipole_delays), None).unwrap());
//     let srclist = get_small_source_list();
//     let inst_flux_densities = get_instrumental_flux_densities_for_srclist(&srclist, &freqs);

//     let result = match beam_correct_flux_densities_inner(
//         inst_flux_densities.view(),
//         beam.deref(),
//         &srclist.get_azel_mwa(lst),
//         &dipole_gains,
//         &freqs,
//     ) {
//         Ok(fds) => fds,
//         Err(e) => panic!("{}", e),
//     };
//     let num_components = srclist.values().fold(0, |a, src| a + src.components.len());
//     assert_eq!(result.dim(), (freqs.len(), num_components));

//     // Hand-verified results.
//     let expected_comp_fd_1 = Jones::from([
//         Complex::new(2.7473072919275476, 0.0),
//         Complex::new(0.0, 0.0),
//         Complex::new(0.0, 0.0),
//         Complex::new(2.7473072919275476, 0.0),
//     ]);
//     let expected_jones_1 = Jones::from([
//         Complex::new(0.7750324863535399, 0.24282289190335862),
//         Complex::new(-0.009009420577898178, -0.002856655664463373),
//         Complex::new(0.01021394523909512, 0.0033072019611734838),
//         Complex::new(0.7814897063974989, 0.25556799755364396),
//     ]);
//     assert_abs_diff_eq!(
//         result[[0, 0]],
//         expected_jones_1 * expected_comp_fd_1 * expected_jones_1.h(),
//         epsilon = 1e-10
//     );

//     let expected_comp_fd_2 = Jones::from([
//         Complex::new(1.7047163998893684, 0.0),
//         Complex::new(0.0, 0.0),
//         Complex::new(0.0, 0.0),
//         Complex::new(1.7047163998893684, 0.0),
//     ]);
//     let expected_jones_2 = Jones::from([
//         Complex::new(0.9455907247090378, 0.3049292024132071),
//         Complex::new(-0.010712295162757346, -0.0033779555969525588),
//         Complex::new(0.010367761993275826, 0.003441723575945327),
//         Complex::new(0.9450219468106582, 0.30598012238683214),
//     ]);
//     assert_abs_diff_eq!(
//         result[[0, 1]],
//         expected_jones_2 * expected_comp_fd_2 * expected_jones_2.h(),
//         epsilon = 1e-10
//     );
// }

// #[test]
// #[serial]
// // Same as above, but with a different frequency.
// fn test_beam_correct_flux_densities_180_mhz() {
//     let freqs = [180e6];
//     let lst = 6.261977848;
//     let dipole_delays = vec![0; 16];
//     let dipole_gains = [1.0; 16];

//     let beam: Box<dyn Beam> =
//         create_fee_beam_object(None, 1, Delays::Partial(dipole_delays), None).unwrap();
//     let srclist = get_small_source_list();
//     let inst_flux_densities = get_instrumental_flux_densities_for_srclist(&srclist, &freqs);
//     let result = match beam_correct_flux_densities_inner(
//         inst_flux_densities.view(),
//         beam.deref(),
//         &srclist.get_azel_mwa(lst),
//         &dipole_gains,
//         &freqs,
//     ) {
//         Ok(fds) => fds,
//         Err(e) => panic!("{}", e),
//     };
//     let num_components = srclist.values().fold(0, |a, src| a + src.components.len());
//     assert_eq!(result.dim(), (freqs.len(), num_components));

//     // Hand-verified results.
//     let expected_comp_fd_1 = Jones::from([
//         Complex::new(2.60247, 0.0),
//         Complex::new(0.0, 0.0),
//         Complex::new(0.0, 0.0),
//         Complex::new(2.60247, 0.0),
//     ]);
//     let expected_jones_1 = Jones::from([
//         Complex::new(0.7731976406423393, 0.17034253171231564),
//         Complex::new(-0.009017301710718753, -0.001961964125441071),
//         Complex::new(0.010223521132619665, 0.002456914956330356),
//         Complex::new(0.7838681411558177, 0.186582048535625),
//     ]);
//     assert_abs_diff_eq!(
//         result[[0, 0]],
//         expected_jones_1 * expected_comp_fd_1 * expected_jones_1.h(),
//         epsilon = 1e-10
//     );

//     let expected_comp_fd_2 = Jones::from([
//         Complex::new(1.61824, 0.0),
//         Complex::new(0.0, 0.0),
//         Complex::new(0.0, 0.0),
//         Complex::new(1.61824, 0.0),
//     ]);
//     let expected_jones_2 = Jones::from([
//         Complex::new(0.9682339089232415, 0.2198904292735457),
//         Complex::new(-0.01090619422142064, -0.0023800302690927533),
//         Complex::new(0.010687354909991509, 0.002535994729487373),
//         Complex::new(0.9676157155647803, 0.22121720658375732),
//     ]);
//     assert_abs_diff_eq!(
//         result[[0, 1]],
//         expected_jones_2 * expected_comp_fd_2 * expected_jones_2.h(),
//         epsilon = 1e-10
//     );
// }

#[test]
fn test_split_components() {
    let freqs = [180e6];
    let phase_centre = RADec::new_degrees(0.0, -27.0);
    let srclist = get_big_source_list();

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
    assert_abs_diff_eq!(points.lmns[0].l, 0.0025326811687516274 * TAU);
    assert_abs_diff_eq!(points.lmns[0].m, -0.12880688061967666 * TAU);
    assert_abs_diff_eq!(points.lmns[0].n, (0.9916664625927036 - 1.0) * TAU);

    assert_eq!(points.flux_densities.dim(), (1, num_point_components));
    assert_eq!(gaussians.flux_densities.dim(), (1, num_gauss_components));
    assert_eq!(shapelets.flux_densities.dim(), (1, 0));

    // Test one of the component's instrumental flux densities.
    let fd = FluxDensityType::List {
        fds: vec![
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
        ],
    }
    .estimate_at_freq(freqs[0]);
    let inst_fd: Jones<f64> = fd.to_inst_stokes();

    let gaussian_flux_densities = gaussians.flux_densities.mapv(TestJones::from);
    let inst_fd = TestJones::from(inst_fd);
    assert_abs_diff_eq!(gaussian_flux_densities[[0, 2]], inst_fd);
}
