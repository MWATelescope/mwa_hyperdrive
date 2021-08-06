// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Ensure that example source list files and be read, and that the code's written
output also matches.
 */

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::{BufReader, Read};

    use tempfile::NamedTempFile;

    use mwa_hyperdrive_core::*;
    use mwa_hyperdrive_srclist::{
        hyperdrive, ComponentType, FluxDensity, FluxDensityType, ShapeletCoeff, Source,
        SourceComponent, SourceList,
    };

    /// The return value of this function is the sourcelist that should match
    /// what is in the examples.
    fn get_example_sl() -> SourceList {
        let source1 = Source {
            components: vec![SourceComponent {
                radec: RADec::new_degrees(10.0, -27.0),
                comp_type: ComponentType::Point,
                flux_type: FluxDensityType::List {
                    fds: vec![
                        FluxDensity {
                            freq: 180e6,
                            i: 10.0,
                            q: 0.0,
                            u: 0.0,
                            v: 0.0,
                        },
                        FluxDensity {
                            freq: 170e6,
                            i: 5.0,
                            q: 1.0,
                            u: 2.0,
                            v: 3.0,
                        },
                    ],
                },
            }],
        };
        let source2 = Source {
            components: vec![
                SourceComponent {
                    radec: RADec::new_degrees(0.0, -35.0),
                    comp_type: ComponentType::Gaussian {
                        maj: 20.0_f64.to_radians() / 3600.0,
                        min: 10.0_f64.to_radians() / 3600.0,
                        pa: 75.0_f64.to_radians(),
                    },
                    flux_type: FluxDensityType::PowerLaw {
                        si: -0.8,
                        fd: FluxDensity {
                            freq: 170e6,
                            i: 5.0,
                            q: 1.0,
                            u: 2.0,
                            v: 3.0,
                        },
                    },
                },
                SourceComponent {
                    radec: RADec::new_degrees(155.0, -10.0),
                    comp_type: ComponentType::Shapelet {
                        maj: 20.0_f64.to_radians() / 3600.0,
                        min: 10.0_f64.to_radians() / 3600.0,
                        pa: 75.0_f64.to_radians(),
                        coeffs: vec![ShapeletCoeff {
                            n1: 0,
                            n2: 1,
                            value: 0.5,
                        }],
                    },
                    flux_type: FluxDensityType::CurvedPowerLaw {
                        si: -0.6,
                        fd: FluxDensity {
                            freq: 150e6,
                            i: 50.0,
                            q: 0.5,
                            u: 0.1,
                            v: 0.0,
                        },
                        q: 0.2,
                    },
                },
            ],
        };

        let mut sl = SourceList::new();
        sl.insert("super_sweet_source1".to_string(), source1);
        sl.insert("super_sweet_source2".to_string(), source2);
        sl
    }

    #[test]
    fn read_yaml_file() {
        let f = File::open("tests/hyperdrive_srclist.yaml");
        assert!(f.is_ok(), "{}", f.unwrap_err());
        let mut f = BufReader::new(f.unwrap());

        let result = hyperdrive::source_list_from_yaml(&mut f);
        assert!(result.is_ok(), "{}", result.unwrap_err());
    }

    #[test]
    fn read_json_file() {
        let f = File::open("tests/hyperdrive_srclist.json");
        assert!(f.is_ok(), "{}", f.unwrap_err());
        let mut f = BufReader::new(f.unwrap());

        let result = hyperdrive::source_list_from_json(&mut f);
        assert!(result.is_ok(), "{}", result.unwrap_err());
    }

    #[test]
    fn write_yaml_file() {
        let mut temp = NamedTempFile::new().unwrap();
        let sl = get_example_sl();
        let result = hyperdrive::source_list_to_yaml(&mut temp, &sl);
        assert!(result.is_ok(), "{}", result.unwrap_err());

        // Compare file contents. Do they match?
        let mut example = String::new();
        let mut just_written = String::new();

        let f = File::open("tests/hyperdrive_srclist.yaml");
        assert!(f.is_ok(), "{}", f.unwrap_err());
        f.unwrap().read_to_string(&mut example).unwrap();

        let f = File::open(&temp.path());
        assert!(f.is_ok(), "{}", f.unwrap_err());
        f.unwrap().read_to_string(&mut just_written).unwrap();

        // Use trim to ignore any leading or trailing whitespace; we don't care
        // about that when comparing contents.
        assert_eq!(example.trim(), just_written.trim());
    }

    #[test]
    fn write_json_file() {
        let mut temp = NamedTempFile::new().unwrap();
        let sl = get_example_sl();
        let result = hyperdrive::source_list_to_json(&mut temp, &sl);
        assert!(result.is_ok(), "{}", result.unwrap_err());

        // Compare file contents. Do they match?
        let mut example = String::new();
        let mut just_written = String::new();
        {
            let f = File::open("tests/hyperdrive_srclist.json");
            assert!(f.is_ok(), "{}", f.unwrap_err());
            f.unwrap().read_to_string(&mut example).unwrap();

            let f = File::open(&temp.path());
            assert!(f.is_ok(), "{}", f.unwrap_err());
            f.unwrap().read_to_string(&mut just_written).unwrap();
            // Apparently there's a new line missing.
            just_written.push('\n');
        }
        assert_eq!(example, just_written);
    }
}
