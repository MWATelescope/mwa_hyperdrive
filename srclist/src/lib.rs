// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code and utilities for sky-model source lists. See for more info:
//! https://github.com/MWATelescope/mwa_hyperdrive/wiki/Source-lists

pub mod ao;
pub mod constants;
pub mod flux_density;
pub mod hyperdrive;
pub mod read;
pub mod rts;
pub mod source_lists;
pub mod utilities;
pub mod woden;

mod error;
mod veto;

pub use error::*;
pub use flux_density::*;
pub use source_lists::*;
pub use utilities::*;
pub use veto::*;

use itertools::Itertools;
use strum::IntoEnumIterator;
use strum_macros::{Display, EnumIter, EnumString};

use constants::*;
use mwa_rust_core::RADec;

/// All of the possible sky-model sources list types.
#[derive(Debug, Clone, Copy, PartialEq, Display, EnumIter, EnumString)]
pub enum SourceListType {
    #[strum(serialize = "hyperdrive")]
    Hyperdrive,

    #[strum(serialize = "rts")]
    Rts,

    #[strum(serialize = "woden")]
    Woden,

    #[strum(serialize = "ao")]
    AO,
}

/// All of the possible file extensions that a hyperdrive-style sky-model source
/// list can have.
#[derive(Debug, Display, EnumIter, EnumString)]
pub enum HyperdriveFileType {
    #[strum(serialize = "yaml")]
    Yaml,

    #[strum(serialize = "json")]
    Json,
}

lazy_static::lazy_static! {
    pub static ref SOURCE_LIST_TYPES_COMMA_SEPARATED: String = SourceListType::iter().join(", ");

    pub static ref HYPERDRIVE_SOURCE_LIST_FILE_TYPES_COMMA_SEPARATED: String = HyperdriveFileType::iter().join(", ");

    pub static ref SRCLIST_BY_BEAM_OUTPUT_TYPE_HELP: String =
    format!("Specifies the type of the output source list. If not specified, the input source list type is used. Currently supported types: {}",
            *SOURCE_LIST_TYPES_COMMA_SEPARATED);

    pub static ref SOURCE_DIST_CUTOFF_HELP: String =
    format!("Specifies the maximum distance from the phase centre a source can be [degrees]. Default: {}",
            DEFAULT_CUTOFF_DISTANCE);

    pub static ref VETO_THRESHOLD_HELP: String =
    format!("Specifies the minimum Stokes XX+YY a source must have before it gets vetoed [Jy]. Default: {}",
            DEFAULT_VETO_THRESHOLD);

    pub static ref CONVERT_INPUT_TYPE_HELP: String =
    format!("Specifies the type of the input source list. Currently supported types: {}",
                *SOURCE_LIST_TYPES_COMMA_SEPARATED);

    pub static ref CONVERT_OUTPUT_TYPE_HELP: String =
    format!("Specifies the type of the output source list. May be required depending on the output filename. Currently supported types: {}",
            *SOURCE_LIST_TYPES_COMMA_SEPARATED);
}

// External re-exports.
pub use mwa_rust_core;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::io::Cursor;

    fn test_two_sources_lists_are_the_same(sl1: &SourceList, sl2: &SourceList) {
        assert_eq!(sl1.len(), sl2.len());
        for ((sl1_name, s1), (sl2_name, s2)) in sl1.iter().zip(sl2.iter()) {
            assert_eq!(sl1_name, sl2_name);
            assert_eq!(s1.components.len(), s2.components.len());
            for (s1_comp, s2_comp) in s1.components.iter().zip(s2.components.iter()) {
                // Tolerances are a little looser here because AO source lists
                // use sexagesimal, so the floating-point errors compound
                // significantly when converting.
                assert_abs_diff_eq!(s1_comp.radec.ra, s2_comp.radec.ra, epsilon = 1e-8);
                assert_abs_diff_eq!(s1_comp.radec.dec, s2_comp.radec.dec, epsilon = 1e-8);
                match &s1_comp.comp_type {
                    ComponentType::Point => {
                        assert!(matches!(s2_comp.comp_type, ComponentType::Point))
                    }

                    ComponentType::Gaussian { maj, min, pa } => {
                        assert!(matches!(s2_comp.comp_type, ComponentType::Gaussian { .. }));
                        let s1_maj = *maj;
                        let s1_min = *min;
                        let s1_pa = *pa;
                        match s2_comp.comp_type {
                            ComponentType::Gaussian { maj, min, pa } => {
                                assert_abs_diff_eq!(s1_maj, maj, epsilon = 1e-10);
                                assert_abs_diff_eq!(s1_min, min, epsilon = 1e-10);
                                assert_abs_diff_eq!(s1_pa, pa, epsilon = 1e-10);
                            }
                            _ => unreachable!(),
                        }
                    }

                    ComponentType::Shapelet {
                        maj,
                        min,
                        pa,
                        coeffs,
                    } => {
                        assert!(matches!(s2_comp.comp_type, ComponentType::Shapelet { .. }));
                        let s1_maj = *maj;
                        let s1_min = *min;
                        let s1_pa = *pa;
                        let s1_coeffs = coeffs;
                        match &s2_comp.comp_type {
                            ComponentType::Shapelet {
                                maj,
                                min,
                                pa,
                                coeffs,
                            } => {
                                assert_abs_diff_eq!(s1_maj, maj, epsilon = 1e-10);
                                assert_abs_diff_eq!(s1_min, min, epsilon = 1e-10);
                                assert_abs_diff_eq!(s1_pa, pa, epsilon = 1e-10);
                                for (s1_coeff, s2_coeff) in s1_coeffs.iter().zip(coeffs.iter()) {
                                    assert_eq!(s1_coeff.n1, s2_coeff.n1);
                                    assert_eq!(s1_coeff.n2, s2_coeff.n2);
                                    assert_abs_diff_eq!(
                                        s1_coeff.value,
                                        s2_coeff.value,
                                        epsilon = 1e-10
                                    );
                                }
                            }
                            _ => unreachable!(),
                        }
                    }
                }

                match &s1_comp.flux_type {
                    FluxDensityType::List { fds } => {
                        assert!(
                            matches!(s2_comp.flux_type, FluxDensityType::List { .. }),
                            "{:?} {:?}",
                            s1_comp,
                            s2_comp
                        );
                        let s1_fds = fds;
                        match &s2_comp.flux_type {
                            FluxDensityType::List { fds } => {
                                assert_eq!(s1_fds.len(), fds.len());
                                for (s1_fd, s2_fd) in s1_fds.iter().zip(fds.iter()) {
                                    assert_abs_diff_eq!(s1_fd.freq, s2_fd.freq, epsilon = 1e-10);
                                    assert_abs_diff_eq!(s1_fd.i, s2_fd.i, epsilon = 1e-10);
                                    assert_abs_diff_eq!(s1_fd.q, s2_fd.q, epsilon = 1e-10);
                                    assert_abs_diff_eq!(s1_fd.u, s2_fd.u, epsilon = 1e-10);
                                    assert_abs_diff_eq!(s1_fd.v, s2_fd.v, epsilon = 1e-10);
                                }
                            }
                            _ => unreachable!(),
                        }
                    }

                    FluxDensityType::PowerLaw { .. } => {
                        assert!(matches!(
                            s2_comp.flux_type,
                            FluxDensityType::PowerLaw { .. }
                        ));
                        match s2_comp.flux_type {
                            FluxDensityType::PowerLaw { .. } => {
                                // The parameters of the power law may not
                                // match, but the estimated flux densities
                                // should.
                                let s1_fd = s1_comp.flux_type.estimate_at_freq(150e6);
                                let s2_fd = s2_comp.flux_type.estimate_at_freq(150e6);
                                assert_abs_diff_eq!(s1_fd.freq, s2_fd.freq, epsilon = 1e-10);
                                assert_abs_diff_eq!(s1_fd.i, s2_fd.i, epsilon = 1e-10);
                                assert_abs_diff_eq!(s1_fd.q, s2_fd.q, epsilon = 1e-10);
                                assert_abs_diff_eq!(s1_fd.u, s2_fd.u, epsilon = 1e-10);
                                assert_abs_diff_eq!(s1_fd.v, s2_fd.v, epsilon = 1e-10);
                                let s1_fd = s1_comp.flux_type.estimate_at_freq(250e6);
                                let s2_fd = s2_comp.flux_type.estimate_at_freq(250e6);
                                assert_abs_diff_eq!(s1_fd.freq, s2_fd.freq, epsilon = 1e-10);
                                assert_abs_diff_eq!(s1_fd.i, s2_fd.i, epsilon = 1e-10);
                                assert_abs_diff_eq!(s1_fd.q, s2_fd.q, epsilon = 1e-10);
                                assert_abs_diff_eq!(s1_fd.u, s2_fd.u, epsilon = 1e-10);
                                assert_abs_diff_eq!(s1_fd.v, s2_fd.v, epsilon = 1e-10);
                            }
                            _ => unreachable!(),
                        }
                    }

                    FluxDensityType::CurvedPowerLaw { .. } => {
                        assert!(matches!(
                            s2_comp.flux_type,
                            FluxDensityType::PowerLaw { .. }
                        ));
                        match s2_comp.flux_type {
                            FluxDensityType::CurvedPowerLaw { .. } => {
                                // The parameters of the curved power law may
                                // not match, but the estimated flux densities
                                // should.
                                let s1_fd = s1_comp.flux_type.estimate_at_freq(150e6);
                                let s2_fd = s2_comp.flux_type.estimate_at_freq(150e6);
                                assert_abs_diff_eq!(s1_fd.freq, s2_fd.freq, epsilon = 1e-10);
                                assert_abs_diff_eq!(s1_fd.i, s2_fd.i, epsilon = 1e-10);
                                assert_abs_diff_eq!(s1_fd.q, s2_fd.q, epsilon = 1e-10);
                                assert_abs_diff_eq!(s1_fd.u, s2_fd.u, epsilon = 1e-10);
                                assert_abs_diff_eq!(s1_fd.v, s2_fd.v, epsilon = 1e-10);
                                let s1_fd = s1_comp.flux_type.estimate_at_freq(250e6);
                                let s2_fd = s2_comp.flux_type.estimate_at_freq(250e6);
                                assert_abs_diff_eq!(s1_fd.freq, s2_fd.freq, epsilon = 1e-10);
                                assert_abs_diff_eq!(s1_fd.i, s2_fd.i, epsilon = 1e-10);
                                assert_abs_diff_eq!(s1_fd.q, s2_fd.q, epsilon = 1e-10);
                                assert_abs_diff_eq!(s1_fd.u, s2_fd.u, epsilon = 1e-10);
                                assert_abs_diff_eq!(s1_fd.v, s2_fd.v, epsilon = 1e-10);
                            }
                            _ => unreachable!(),
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn hyperdrive_conversion_works() {
        let mut sl = SourceList::new();
        sl.insert(
            "point".to_string(),
            Source {
                components: vec![SourceComponent {
                    radec: RADec::new_degrees(60.0, -27.0),
                    comp_type: ComponentType::Point,
                    flux_type: FluxDensityType::PowerLaw {
                        si: DEFAULT_SPEC_INDEX,
                        fd: FluxDensity {
                            freq: 180e6,
                            i: 10.0,
                            q: 1.0,
                            u: 2.0,
                            v: 3.0,
                        },
                    },
                }],
            },
        );
        sl.insert(
            "gaussian".to_string(),
            Source {
                components: vec![SourceComponent {
                    radec: RADec::new_degrees(61.0, -28.0),
                    comp_type: ComponentType::Gaussian {
                        maj: 1.0,
                        min: 0.5,
                        pa: 90.0,
                    },
                    flux_type: FluxDensityType::PowerLaw {
                        si: DEFAULT_SPEC_INDEX,
                        fd: FluxDensity {
                            freq: 190e6,
                            i: 11.0,
                            q: 1.0,
                            u: 2.0,
                            v: 3.0,
                        },
                    },
                }],
            },
        );
        sl.insert(
            "shapelet".to_string(),
            Source {
                components: vec![SourceComponent {
                    radec: RADec::new_degrees(59.0, -26.0),
                    comp_type: ComponentType::Shapelet {
                        maj: 1.0,
                        min: 0.5,
                        pa: 90.0,
                        coeffs: vec![ShapeletCoeff {
                            n1: 0,
                            n2: 0,
                            value: 1.0,
                        }],
                    },
                    flux_type: FluxDensityType::PowerLaw {
                        si: DEFAULT_SPEC_INDEX,
                        fd: FluxDensity {
                            freq: 170e6,
                            i: 9.0,
                            q: 1.0,
                            u: 2.0,
                            v: 3.0,
                        },
                    },
                }],
            },
        );

        // Write this source list as hyperdrive style into a buffer.
        let mut buf = Cursor::new(vec![]);
        hyperdrive::source_list_to_yaml(&mut buf, &sl).unwrap();
        // Read it back in, and test that things are sensible.
        buf.set_position(0);
        let new_sl = hyperdrive::source_list_from_yaml(&mut buf).unwrap();
        test_two_sources_lists_are_the_same(&sl, &new_sl);
    }

    #[test]
    fn rts_conversion_works() {
        let (hyperdrive_sl, _) = read::read_source_list_file(
            "tests/srclist_1099334672_100.yaml",
            Some(SourceListType::Hyperdrive),
        )
        .unwrap();
        let mut buf = Cursor::new(vec![]);
        rts::write_source_list(&mut buf, &hyperdrive_sl).unwrap();
        buf.set_position(0);
        let rts_sl = rts::parse_source_list(&mut buf).unwrap();
        test_two_sources_lists_are_the_same(&hyperdrive_sl, &rts_sl);
    }

    #[test]
    fn ao_conversion_works() {
        let (hyperdrive_sl, _) = read::read_source_list_file(
            "tests/srclist_1099334672_100.yaml",
            Some(SourceListType::Hyperdrive),
        )
        .unwrap();
        let mut buf = Cursor::new(vec![]);

        // AO source lists don't handle shapelets. Prune those from the
        // hyperdrive source list before continuing.
        let mut new_hyperdrive_sl = SourceList::new();
        for (name, src) in hyperdrive_sl.into_iter() {
            let comps: Vec<_> = src
                .components
                .into_iter()
                .filter(|comp| !matches!(comp.comp_type, ComponentType::Shapelet { .. }))
                .collect();
            if !comps.is_empty() {
                new_hyperdrive_sl.insert(name, Source { components: comps });
            }
        }

        ao::write_source_list(&mut buf, &new_hyperdrive_sl).unwrap();
        buf.set_position(0);
        let ao_sl = ao::parse_source_list(&mut buf).unwrap();
        test_two_sources_lists_are_the_same(&new_hyperdrive_sl, &ao_sl);
    }

    #[test]
    fn woden_conversion_works() {
        let (mut hyperdrive_sl, _) = read::read_source_list_file(
            "tests/srclist_1099334672_100.yaml",
            Some(SourceListType::Hyperdrive),
        )
        .unwrap();

        let mut buf = Cursor::new(vec![]);
        woden::write_source_list(&mut buf, &hyperdrive_sl).unwrap();
        buf.set_position(0);
        let woden_sl = woden::parse_source_list(&mut buf).unwrap();

        // WODEN only allows one flux density per component. Verify that things
        // are different.
        let mut flux_density_types_are_different = false;
        for (hyp_comp, woden_comp) in hyperdrive_sl
            .iter()
            .flat_map(|(_, s)| &s.components)
            .zip(woden_sl.iter().flat_map(|(_, s)| &s.components))
        {
            if std::mem::discriminant(&hyp_comp.flux_type)
                != std::mem::discriminant(&woden_comp.flux_type)
            {
                flux_density_types_are_different = true;
                break;
            }
        }
        assert!(flux_density_types_are_different);

        // Now alter the hyperdrive source list to match WODEN.
        for comp in hyperdrive_sl
            .iter_mut()
            .flat_map(|(_, s)| &mut s.components)
        {
            comp.flux_type = match &comp.flux_type {
                FluxDensityType::List {fds} => FluxDensityType::PowerLaw {fd: fds[0].clone(), si: DEFAULT_SPEC_INDEX},
                FluxDensityType::PowerLaw {fd, si} => FluxDensityType::PowerLaw {si: *si, fd: fd.clone()},
                FluxDensityType::CurvedPowerLaw {..} => panic!("Source list has a curved power law, but it shouldn't, and WODEN can't handle it."),
            };
        }
        test_two_sources_lists_are_the_same(&hyperdrive_sl, &woden_sl);
    }
}
