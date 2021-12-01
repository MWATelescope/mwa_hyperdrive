// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;

use log::{debug, info, trace, warn};
use mwa_rust_core::{constants::DH2R, mwalib};
use rayon::prelude::*;
use serde::Deserialize;
use structopt::StructOpt;

use crate::*;
use mwa_hyperdrive_common::log;

/// Shift the sources in a source list. Useful to correct for the ionosphere.
/// The shifts must be detailed in a .json file, with source names as keys
/// associated with an "ra" and "dec" in degrees. Only the sources specified in
/// the .json are written to the output source list.
#[derive(StructOpt, Debug)]
pub struct ShiftArgs {
    /// Path to the source list to be shifted.
    #[structopt(name = "SOURCE_LIST", parse(from_os_str))]
    pub source_list: PathBuf,

    /// Path to the .json shifts file.
    #[structopt(name = "SOURCE_SHIFTS", parse(from_os_str))]
    pub source_shifts: PathBuf,

    /// Path to the output source list. If not specified, then then "_shifted"
    /// is appended to the filename.
    #[structopt(name = "OUTPUT_SOURCE_LIST", parse(from_os_str))]
    pub output_source_list: Option<PathBuf>,

    /// Attempt to convert flux density lists to power laws. See the online help
    /// for more information.
    #[structopt(short, long)]
    pub convert_lists: bool,

    /// Collapse all of the sky-model components into a single source; the
    /// apparently brightest source is used as the base source. This is suitable
    /// for an "RTS patch source list".
    #[structopt(long)]
    pub collapse_into_single_source: bool,

    /// Path to the metafits file. Only needed if collapse-into-single-source is
    /// used.
    #[structopt(short, long, parse(from_str))]
    pub metafits: Option<PathBuf>,

    /// Keep any SHAPELET components (SHAPELET2 components are never ignored).
    /// Applies only to rts-style source lists.
    #[structopt(short, long)]
    pub keep_shapelets: bool,

    /// The verbosity of the program. The default is to print high-level
    /// information.
    #[structopt(short, long, parse(from_occurrences))]
    pub verbosity: u8,
}

impl ShiftArgs {
    /// Use the arguments to shift a source list.
    pub fn run(&self) -> Result<(), SrclistError> {
        shift(
            &self.source_list,
            &self.source_shifts,
            self.output_source_list.as_ref(),
            self.convert_lists,
            self.collapse_into_single_source,
            self.metafits.as_ref(),
            self.keep_shapelets,
        )
    }
}

pub fn shift<T: AsRef<Path>>(
    source_list_file: T,
    source_shifts_file: T,
    output_source_list_file: Option<T>,
    convert_lists: bool,
    collapse_into_single_source: bool,
    metafits_file: Option<T>,
    keep_shapelets: bool,
) -> Result<(), SrclistError> {
    let f = BufReader::new(File::open(source_shifts_file)?);
    let source_shifts: BTreeMap<String, RaDec> =
        serde_json::from_reader(f).map_err(WriteSourceListError::from)?;
    let (mut sl, sl_type) = crate::read::read_source_list_file(&source_list_file, None)?;
    info!(
        "Successfully read {} as a {}-style source list",
        source_list_file.as_ref().display(),
        sl_type
    );
    let counts = sl.get_counts();
    info!(
        "Input has {} points, {} gaussians, {} shapelets",
        counts.0, counts.1, counts.2
    );

    let mut shapelets = vec![];
    if keep_shapelets {
        match sl_type {
            SourceListType::Rts => {
                // Pull out the SHAPELET sources. We can make a lot of shortcuts
                // here because the source list was successfully parsed above.
                // We expect that any SHAPELET components are actually RTS base
                // sources, not components.
                let mut buf = BufReader::new(File::open(&source_list_file)?);
                let mut line = String::new();
                let mut src = TempSource {
                    name: String::new(),
                    ra: 0.0,
                    dec: 0.0,
                    fds: vec![],
                    comp: ComponentType::Shapelet {
                        maj: 0.0,
                        min: 0.0,
                        pa: 0.0,
                        coeffs: vec![],
                    },
                    is_shapelet: false,
                };
                while buf.read_line(&mut line)? > 0 {
                    let mut items = line.split_whitespace();
                    match items.next() {
                        Some("SOURCE") => {
                            src.name = items.next().unwrap().to_string();
                            src.ra = items.next().unwrap().parse::<f64>().unwrap() * DH2R;
                            src.dec = items.next().unwrap().parse::<f64>().unwrap().to_radians();
                        },
                        Some("COMPONENT") => {
                            src.clear();
                        },
                        Some("FREQ") => {
                            let freq = items.next().unwrap().parse().unwrap();
                            let stokes_i = items.next().unwrap().parse().unwrap();
                            let stokes_q = items.next().unwrap().parse().unwrap();
                            let stokes_u = items.next().unwrap().parse().unwrap();
                            let stokes_v = items.next().unwrap().parse().unwrap();
                            src.fds.push(FluxDensity {
                                freq,
                                i: stokes_i,
                                q: stokes_q,
                                u: stokes_u,
                                v: stokes_v,
                            });
                        },
                        Some("SHAPELET2") => (),
                        Some("SHAPELET") => {
                            let mut pa = items.next().unwrap().parse::<f64>().unwrap();
                            let maj_arcmin = items.next().unwrap().parse::<f64>().unwrap();
                            let min_arcmin = items.next().unwrap().parse::<f64>().unwrap();

                            // Ensure the position angle is positive.
                            if pa < 0.0 {
                                pa += 360.0;
                            }

                            src.comp = ComponentType::Shapelet {
                                maj: maj_arcmin.to_radians() / 60.0,
                                min: min_arcmin.to_radians() / 60.0,
                                pa: pa.to_radians(),
                                coeffs: vec![],
                            };
                            src.is_shapelet = true;
                        },
                        Some("COEFF") => {
                            let n1 = items.next().unwrap().parse::<f64>().unwrap();
                            let n2 = items.next().unwrap().parse::<f64>().unwrap();
                            let value = items.next().unwrap().parse().unwrap();
                            match &mut src.comp {
                                ComponentType::Shapelet { coeffs, .. } => coeffs.push(ShapeletCoeff {
                                    n1: n1 as _, n2: n2 as _, value,
                                }),
                                _ => unreachable!(),
                            }
                        },
                        Some("ENDSOURCE") => {
                            if src.is_shapelet {
                               shapelets.push(src.clone());
                            }
                            src.clear();
                        }
                        Some("ENDCOMPONENT") => src.clear(),
                        _ => (),
                    }

                    line.clear(); // clear to reuse the buffer line.
                }
            },

            _ => warn!("keep_shapelets was specified, but this is meaningless if the input source list is not rts-style"),
        }
    }
    // If we found any SHAPELETs, add them in.
    if !shapelets.is_empty() {
        for shapelet in &shapelets {
            let (maj, min, pa, coeffs) = match &shapelet.comp {
                ComponentType::Shapelet {
                    maj,
                    min,
                    pa,
                    coeffs,
                } => (maj, min, pa, coeffs),
                _ => unreachable!(),
            };
            sl.insert(
                shapelet.name.clone(),
                Source {
                    components: vec![SourceComponent {
                        radec: RADec::new(shapelet.ra, shapelet.dec),
                        comp_type: ComponentType::Shapelet {
                            maj: *maj,
                            min: *min,
                            pa: *pa,
                            coeffs: coeffs.clone(),
                        },
                        flux_type: FluxDensityType::List {
                            fds: shapelet.fds.clone(),
                        },
                    }],
                },
            );
        }
    }

    // Filter any sources that aren't in the shifts file, and shift the
    // sources. All components of a source get shifted the same amount.
    let mut sl = {
        let tmp_sl: BTreeMap<String, Source> = sl
            .into_iter()
            .filter(|(name, _)| source_shifts.contains_key(name))
            .map(|(name, mut src)| {
                let shifts = &source_shifts[&name];
                src.components.iter_mut().for_each(|comp| {
                    comp.radec.ra += shifts.ra.to_radians();
                    comp.radec.dec += shifts.dec.to_radians();
                });
                (name, src)
            })
            .collect();
        SourceList::from(tmp_sl)
    };

    let output_path: PathBuf = match output_source_list_file {
        Some(p) => p.as_ref().to_path_buf(),
        None => {
            let input_path_base = source_list_file
                .as_ref()
                .file_stem()
                .and_then(|os_str| os_str.to_str())
                .map(|str| str.to_string())
                .expect("Input file didn't have a filename stem");
            let input_path_ext = source_list_file
                .as_ref()
                .extension()
                .and_then(|os_str| os_str.to_str())
                .expect("Input file didn't have an extension");
            let mut output_path = input_path_base;
            output_path.push_str("_shifted");
            output_path.push_str(&format!(".{}", input_path_ext));
            let output_pb = PathBuf::from(output_path);
            trace!("Writing shifted source list to {}", output_pb.display());
            output_pb
        }
    };

    // If we were told to, attempt to convert lists of flux densities to power
    // laws.
    if convert_lists {
        sl.values_mut()
            .flat_map(|src| &mut src.components)
            .for_each(|comp| comp.flux_type.convert_list_to_power_law());
    }

    // If requested, collapse the source list.
    sl = if collapse_into_single_source {
        // Open the metafits.
        let metafits = match &metafits_file {
            Some(m) => m,
            None => return Err(SrclistError::MissingMetafits),
        };
        trace!("Attempting to open the metafits file");
        let meta = mwalib::MetafitsContext::new(&metafits, None)?;
        let ra_phase_centre = meta
            .ra_phase_center_degrees
            .unwrap_or(meta.ra_tile_pointing_degrees);
        let dec_phase_centre = meta
            .dec_phase_center_degrees
            .unwrap_or(meta.dec_tile_pointing_degrees);
        let phase_centre = RADec::new_degrees(ra_phase_centre, dec_phase_centre);
        debug!("Using {} as the phase centre", phase_centre);
        let lst = meta.lst_rad;
        debug!("Using {}° as the LST", lst.to_degrees());
        let coarse_chan_freqs: Vec<f64> = meta
            .metafits_coarse_chans
            .iter()
            .map(|cc| cc.chan_centre_hz as _)
            .collect();
        debug!(
            "Using coarse channel frequencies [MHz]: {}",
            coarse_chan_freqs
                .iter()
                .map(|cc_freq_hz| format!("{:.2}", *cc_freq_hz as f64 / 1e6))
                .join(", ")
        );

        let mut collapsed = SourceList::new();
        // Use the apparently brightest source as the base. Not sure this is
        // necessary or important, but hey, it's the RTS we're talking about.
        let brightest = sl
            .par_iter()
            .map(|(name, src)| {
                let stokes_i = src
                    .get_flux_estimates(150e6)
                    .iter()
                    .fold(0.0, |acc, fd| acc + fd.i);
                (name, stokes_i)
            })
            .max_by(|x, y| x.1.partial_cmp(&y.1).unwrap())
            .unwrap();
        let brightest_name = brightest.0.clone();
        let brightest = sl.remove_entry(&brightest_name).unwrap();
        collapsed.insert(brightest_name, brightest.1);
        let base_src = collapsed.get_mut(&brightest.0).unwrap();
        sl.into_iter()
            .flat_map(|(_, src)| src.components)
            .for_each(|comp| base_src.components.push(comp));
        collapsed
    } else {
        sl
    };
    let counts = sl.get_counts();
    info!(
        "Shifted {} points, {} gaussians, {} shapelets",
        counts.0, counts.1, counts.2
    );

    // Write the output source list.
    trace!("Attempting to write output source list");
    let mut f = std::io::BufWriter::new(File::create(&output_path)?);

    match sl_type {
        SourceListType::Hyperdrive => {
            let sl_file_ext = output_path.extension().and_then(|e| e.to_str());
            match sl_file_ext.and_then(|e| HyperdriveFileType::from_str(e).ok()) {
                Some(HyperdriveFileType::Yaml) => {
                    hyperdrive::source_list_to_yaml(&mut f, &sl)?;
                }
                Some(HyperdriveFileType::Json) => {
                    hyperdrive::source_list_to_json(&mut f, &sl)?;
                }
                None => {
                    return Err(WriteSourceListError::InvalidHyperdriveFormat(
                        sl_file_ext.unwrap_or("<no extension>").to_string(),
                    )
                    .into())
                }
            }

            info!(
                "Wrote hyperdrive-style source list to {}",
                output_path.display()
            );
        }
        SourceListType::Rts => {
            rts::write_source_list(&mut f, &sl)?;
            info!("Wrote rts-style source list to {}", output_path.display());
        }
        SourceListType::AO => {
            ao::write_source_list(&mut f, &sl)?;
            info!("Wrote ao-style source list to {}", output_path.display());
        }
        SourceListType::Woden => {
            woden::write_source_list(&mut f, &sl)?;
            info!("Wrote woden-style source list to {}", output_path.display());
        }
    }

    // Doctor any SHAPELET sources. They will get written as SHAPELET2 but we
    // want SHAPELET.
    if !shapelets.is_empty() {
        let mut buf = BufReader::new(File::open(&output_path)?);
        let mut out = String::new();
        let mut line = String::new();
        while buf.read_line(&mut line)? > 0 {
            let mut items = line.split_whitespace();
            if let Some("SHAPELET2") = items.next() {
                // Make sure this isn't actually a SHAPELET2 component.
                let line_pa = items.next().unwrap().parse::<f64>().unwrap().to_radians();
                let line_maj = items.next().unwrap().parse::<f64>().unwrap().to_radians() / 60.0;
                let line_min = items.next().unwrap().parse::<f64>().unwrap().to_radians() / 60.0;
                if shapelets.iter().any(|s| match s.comp {
                    ComponentType::Shapelet { maj, min, pa, .. } => {
                        (line_maj - maj).abs() < 1e-3
                            && (line_min - min).abs() < 1e-3
                            && (line_pa - pa).abs() < 1e-3
                    }
                    _ => unreachable!(),
                }) {
                    line.replace_range(0..9, "SHAPELET");
                }
            }
            out.push_str(&line);

            line.clear(); // clear to reuse the buffer line.
        }
        File::create(&output_path)?.write_all(out.as_bytes())?;
    }

    Ok(())
}

#[derive(Deserialize)]
struct RaDec {
    ra: f64,
    dec: f64,
}

#[derive(Debug, Clone)]
struct TempSource {
    name: String,
    ra: f64,
    dec: f64,
    fds: Vec<FluxDensity>,
    comp: ComponentType,
    is_shapelet: bool,
}

impl TempSource {
    fn clear(&mut self) {
        self.fds.clear();
        self.is_shapelet = false;
    }
}
