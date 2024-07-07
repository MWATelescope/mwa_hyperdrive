// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::{collections::HashMap, path::Path};

use fitsio::{hdu::FitsHdu, FitsFile};
use indexmap::IndexMap;
use itertools::izip;
use log::debug;
use marlu::RADec;
use vec1::Vec1;

// The reference frequency of the power laws and curved power laws.
use super::REF_FREQ_HZ;
use crate::{
    io::read::fits::{fits_open, fits_open_hdu, FitsError},
    srclist::{
        ComponentType, FluxDensity, FluxDensityType, ShapeletCoeff, Source, SourceComponent,
        SourceList,
    },
};

/// A supported "flux type", but without any data.
#[derive(Clone, Copy, Debug)]
enum FluxType {
    PowerLaw,
    CurvedPowerLaw,
    List,
}

/// Wrap a FITS call with our error.
macro_rules! fe {
    ($file:expr, $result:expr) => {{
        $result.map_err(|e| FitsError::Fitsio {
            fits_error: Box::new(e),
            fits_filename: $file.to_path_buf().into_boxed_path(),
            hdu_description: "1".to_string().into_boxed_str(),
            source_file: file!(),
            source_line: line!(),
            source_column: column!(),
        })?
    }};
}

pub(crate) fn parse_source_list(file: &Path) -> Result<SourceList, FitsError> {
    let mut fptr = fits_open(file)?;
    // We assume everything is on HDU 2.
    let hdu = fits_open_hdu(&mut fptr, 1)?;

    // Get the fits column names.
    let col_names: Vec<String> = match &hdu.info {
        fitsio::hdu::HduInfo::TableInfo {
            column_descriptions,
            num_rows: _,
        } => column_descriptions
            .iter()
            .map(|cd| cd.name.clone())
            .collect(),
        fitsio::hdu::HduInfo::ImageInfo { .. } => todo!(),
        fitsio::hdu::HduInfo::AnyInfo => todo!(),
    };

    // Try to determine the format of the FITS file.
    if col_names.iter().any(|name| *name == "UNQ_SOURCE_ID") {
        debug!("Attempting to read the first NAME");
        let first_src_id = fe!(file, hdu.read_cell_value::<String>(&mut fptr, "NAME", 0));
        if first_src_id.contains("_GID") {
            debug!("I reckon this is a 'LoBES' source list");
            parse_lobes_source_list(file, fptr, hdu, col_names)
        } else {
            debug!("I reckon this is a 'Jack FITS' source list");
            parse_jack_source_list(file, fptr, hdu, col_names)
        }
    } else {
        debug!("I reckon this is a 'GLEAM FITS' source list");
        parse_gleam_x_source_list(file, fptr, hdu, col_names)
    }
}

/// FITS file source lists are pretty similar. This struct
/// contains the common columns.
struct CommonCols {
    unq_source_id: Vec<String>,
    names: Vec<String>,
    ra_degrees: Vec<f64>,
    dec_degrees: Vec<f64>,
    majors: Vec<f64>,
    minors: Vec<f64>,
    pas: Vec<f64>,
    shapelet_sources: Option<HashMap<String, Vec<ShapeletCoeff>>>,
    comp_types: Vec<char>,
    flux_types: Vec<FluxType>,
    power_law_stokes_is: Vec<f64>,
    power_law_alphas: Vec<f64>,
    curved_power_law_stokes_is: Vec<f64>,
    curved_power_law_alphas: Vec<f64>,
    curved_power_law_qs: Vec<f64>,
    list_flux_densities: Vec<Vec<f64>>,
    list_flux_density_freqs: Vec<f64>,
}

impl CommonCols {
    fn new(
        file: &Path,
        fptr: &mut FitsFile,
        hdu: &FitsHdu,
        col_names: &[String],
    ) -> Result<Self, FitsError> {
        macro_rules! read_optional_col {
            ($possible_col_names: expr) => {{
                let mut maybe_col = None;
                for possible_col_name in $possible_col_names {
                    if col_names
                        .iter()
                        .any(|col_name| col_name == possible_col_name)
                    {
                        maybe_col = Some(fe!(file, hdu.read_col(fptr, possible_col_name)));
                    }
                }
                if !maybe_col.is_some() {
                    debug!("None of {:?} were available columns!", $possible_col_names)
                }
                maybe_col
            }};
        }
        macro_rules! read_mandatory_col {
            ($possible_col_names: expr) => {{
                read_optional_col!($possible_col_names).unwrap_or_else(|| {
                    panic!("None of {:?} were available columns!", $possible_col_names)
                })
            }};
        }

        let unq_source_id = if col_names.iter().any(|col_name| col_name == "UNQ_SOURCE_ID") {
            fe!(file, hdu.read_col(fptr, "UNQ_SOURCE_ID"))
        } else {
            vec![]
        };

        let names = read_mandatory_col!(["NAME", "Name"]);
        let ra_degrees = read_mandatory_col!(["RA", "RAJ2000"]);
        let dec_degrees = read_mandatory_col!(["DEC", "DEJ2000"]);
        let majors = read_mandatory_col!(["MAJOR_DC", "a"]);
        let minors = read_mandatory_col!(["MINOR_DC", "b"]);
        let pas = read_mandatory_col!(["PA_DC", "pa"]);

        // Get any shapelet info ready. We assume that the info lives in HDU 3
        // (index 2 in sane languages), and if there's an error, we assume it's
        // because the HDU isn't present.
        let hdu_shapelets = fits_open_hdu(fptr, 2).ok();
        let shapelet_sources = match hdu_shapelets {
            Some(hdu) => {
                let mut map = HashMap::new();
                let shapelet_sources: Vec<String> = fe!(file, hdu.read_col(fptr, "NAME"));
                let shapelet_n1s: Vec<i64> = fe!(file, hdu.read_col(fptr, "N1"));
                let shapelet_n2s: Vec<i64> = fe!(file, hdu.read_col(fptr, "N2"));
                let shapelet_coeff_values: Vec<f64> = fe!(file, hdu.read_col(fptr, "COEFF"));

                // Pre-allocate vectors for the shapelet coeffs.
                let mut count = 0;
                let mut last_source = None;
                for this_source in &shapelet_sources {
                    match last_source.as_mut() {
                        None => {
                            last_source = Some(this_source);
                            count += 1;
                            continue;
                        }

                        Some(last_source) => {
                            if *last_source == this_source {
                                count += 1;
                                continue;
                            } else {
                                map.insert(last_source.clone(), Vec::with_capacity(count));
                                count = 0;
                                *last_source = this_source;
                            }
                        }
                    }
                }
                // Don't forget the last source.
                if let Some(last_source) = last_source {
                    map.insert(last_source.clone(), Vec::with_capacity(count));
                }

                // Now populate the shapelet coeffs associated with the sources.
                for (source, n1, n2, value) in izip!(
                    shapelet_sources,
                    shapelet_n1s,
                    shapelet_n2s,
                    shapelet_coeff_values
                ) {
                    let coeffs = map.get_mut(&source).expect("was populated previously");
                    coeffs.push(ShapeletCoeff {
                        n1: n1.try_into().expect("n1 is not larger than u8::MAX"),
                        n2: n2.try_into().expect("n2 is not larger than u8::MAX"),
                        value,
                    })
                }

                Some(map)
            }

            None => None,
        };

        let comp_types = if col_names.iter().any(|col_name| col_name == "COMP_TYPE") {
            // This is pretty gross, but, so is handling bespoke formats.
            let comp_types_as_strings: Vec<String> = fe!(file, hdu.read_col(fptr, "COMP_TYPE"));
            comp_types_as_strings
                .into_iter()
                .map(|mut s| s.pop().expect("COMP_TYPE strings aren't empty"))
                .collect()
        } else {
            // We need to determine the component types here.
            let mut comp_types = Vec::with_capacity(names.len());
            for (i_row, name) in names.iter().enumerate() {
                if let Some(shapelet_sources) = shapelet_sources.as_ref() {
                    if shapelet_sources.contains_key(name) {
                        comp_types.push('S');
                        continue;
                    }
                } else {
                    let gauss_maj: f64 = majors[i_row];
                    let gauss_min: f64 = minors[i_row];
                    let gauss_pa: f64 = pas[i_row];
                    if (gauss_maj.is_nan() && gauss_min.is_nan() && gauss_pa.is_nan())
                        || (gauss_maj.abs() < f64::EPSILON
                            && gauss_min.abs() < f64::EPSILON
                            && gauss_pa.abs() < f64::EPSILON)
                    {
                        comp_types.push('P');
                    } else {
                        comp_types.push('G');
                    }
                }
            }
            comp_types
        };

        let power_law_stokes_is: Vec<f64> = read_mandatory_col!(["NORM_COMP_PL", "S_200"]);
        let power_law_alphas = read_mandatory_col!(["ALPHA_PL", "alpha"]);

        let (curved_power_law_stokes_is, curved_power_law_alphas, curved_power_law_qs): (
            Vec<f64>,
            Vec<f64>,
            Vec<f64>,
        ) = if col_names.iter().any(|col_name| col_name == "NORM_COMP_CPL") {
            (
                fe!(file, hdu.read_col(fptr, "NORM_COMP_CPL")),
                fe!(file, hdu.read_col(fptr, "ALPHA_CPL")),
                fe!(file, hdu.read_col(fptr, "CURVE_CPL")),
            )
        } else if col_names.iter().any(|col_name| col_name == "beta") {
            (vec![], vec![], fe!(file, hdu.read_col(fptr, "beta")))
        } else {
            (vec![], vec![], vec![])
        };

        // For "list" flux density types, first, find all the relevant
        // columns. Then, pull out all their values.
        let (list_flux_densities, list_flux_density_freqs): (Vec<Vec<f64>>, Vec<f64>) = if col_names
            .iter()
            .any(|col_name| col_name.starts_with("INT_FLX"))
        {
            {
                let mut flux_densities = Vec::with_capacity(32);
                let mut flux_density_freqs = Vec::with_capacity(32);
                #[allow(non_snake_case)]
                for (col_name, int_flx_freq_MHz) in col_names.iter().filter_map(|col_name| {
                    col_name
                        .strip_prefix("INT_FLX")
                        .map(|suffix| (col_name.as_str(), suffix))
                }) {
                    flux_densities.push(fe!(file, hdu.read_col(fptr, col_name)));
                    flux_density_freqs
                        .push(int_flx_freq_MHz.parse::<f64>().expect("is a number") * 1e6);
                }
                (flux_densities, flux_density_freqs)
            }
        } else {
            (vec![], vec![])
        };

        let flux_types = if col_names.iter().any(|col_name| col_name == "MOD_TYPE") {
            let mod_types: Vec<String> = fe!(file, hdu.read_col(fptr, "MOD_TYPE"));
            mod_types.into_iter().enumerate().map(|(i_row, mod_type)| match mod_type.as_str() {
                "nan" => FluxType::List,
                "pl" => FluxType::PowerLaw,
                "cpl" => FluxType::CurvedPowerLaw,
                t => panic!("Got '{t}' in row {i_row} of the 'MOD_TYPE' column, which is none of 'nan', 'pl' or 'cpl'"),
            }).collect()
        } else {
            // MOD_TYPE isn't here. We assume that there are always power laws,
            // but check for curved power laws and lists.
            if curved_power_law_stokes_is.is_empty() && list_flux_densities.is_empty() {
                vec![FluxType::PowerLaw; power_law_stokes_is.len()]
            } else {
                // We have to iterate over everything...
                let mut results = Vec::with_capacity(power_law_stokes_is.len());
                let mut indices_to_check = Vec::with_capacity(power_law_stokes_is.len());
                // We assume that if a CPL q is present, this should be a CPL.
                if !curved_power_law_qs.is_empty() {
                    for (i, q) in curved_power_law_qs.iter().copied().enumerate() {
                        if !q.is_nan() && q != 0.0 {
                            results.push(FluxType::CurvedPowerLaw);
                        } else {
                            indices_to_check.push(i);
                        }
                    }
                } else {
                    for i in 0..power_law_stokes_is.len() {
                        indices_to_check.push(i);
                    }
                }
                // Do the same thing for PL Stokes Is.
                for stokes_i in indices_to_check.into_iter().map(|i| power_law_stokes_is[i]) {
                    if !stokes_i.is_nan() && stokes_i != 0.0 {
                        results.push(FluxType::PowerLaw);
                    } else {
                        results.push(FluxType::List);
                    }
                }

                results
            }
        };

        Ok(Self {
            unq_source_id,
            names,
            ra_degrees,
            dec_degrees,
            majors,
            minors,
            pas,
            shapelet_sources,
            comp_types,
            flux_types,
            power_law_stokes_is,
            power_law_alphas,
            curved_power_law_stokes_is,
            curved_power_law_alphas,
            curved_power_law_qs,
            list_flux_densities,
            list_flux_density_freqs,
        })
    }
}

fn parse_lobes_source_list(
    file: &Path,
    mut fptr: FitsFile,
    hdu: FitsHdu,
    col_names: Vec<String>,
) -> Result<SourceList, FitsError> {
    let CommonCols {
        unq_source_id: _,
        names,
        ra_degrees,
        dec_degrees,
        majors,
        minors,
        pas,
        flux_types,
        power_law_stokes_is,
        power_law_alphas,
        curved_power_law_stokes_is,
        curved_power_law_alphas,
        curved_power_law_qs,
        list_flux_densities,
        list_flux_density_freqs,
        ..
    } = CommonCols::new(file, &mut fptr, &hdu, &col_names)?;

    // UNQ_SOURCE_ID comes in as strings of ints. Parse them.

    let mut map = IndexMap::with_capacity(names.len());
    // Get all of the source names.
    for (i_row, name) in names.into_iter().enumerate() {
        let prefix_len = name.rsplit_once("_GID").expect("contains '_GID'").0.len();
        // Here, we assume pure ASCII.
        let name = String::from_utf8({
            let mut bytes = name.into_bytes();
            bytes.resize(prefix_len, 0);
            bytes
        })
        .expect("is valid");

        map.entry(name)
            .and_modify(|v: &mut Vec<usize>| v.push(i_row))
            .or_insert(vec![i_row]);
    }

    // Find all of the components that belong with this source and populate it.
    let mut source_list = IndexMap::with_capacity(map.len());
    for (name, rows) in map {
        let src = source_list.entry(name).or_insert(vec![]);

        for i_row in rows {
            let radec = RADec::from_degrees(ra_degrees[i_row], dec_degrees[i_row]);
            let maj = majors[i_row];
            let min = minors[i_row];
            let pa = pas[i_row];
            let comp_type = if (maj.is_nan() && min.is_nan() && pa.is_nan())
                || (maj.abs() < f64::EPSILON && min.abs() < f64::EPSILON && pa.abs() < f64::EPSILON)
            {
                ComponentType::Point
            } else {
                ComponentType::Gaussian {
                    maj: maj.to_radians(),
                    min: min.to_radians(),
                    pa: pa.to_radians(),
                }
            };

            let flux_type = match flux_types[i_row] {
                FluxType::List => {
                    let mut list_fds = Vec::with_capacity(list_flux_densities.len());
                    for (fds, &freq) in list_flux_densities
                        .iter()
                        .zip(list_flux_density_freqs.iter())
                    {
                        let fd = fds[i_row];
                        if !fd.is_nan() {
                            list_fds.push(FluxDensity {
                                freq,
                                i: fd,
                                q: 0.0,
                                u: 0.0,
                                v: 0.0,
                            });
                        }
                    }
                    FluxDensityType::List(Vec1::try_from_vec(list_fds).unwrap_or_else(|_| {
                        panic!("No valid flux densities were provided on row {i_row}")
                    }))
                }

                FluxType::PowerLaw => {
                    let si = power_law_alphas[i_row];
                    let i = power_law_stokes_is[i_row];
                    FluxDensityType::PowerLaw {
                        si,
                        fd: FluxDensity {
                            freq: REF_FREQ_HZ,
                            i,
                            q: 0.0,
                            u: 0.0,
                            v: 0.0,
                        },
                    }
                }

                FluxType::CurvedPowerLaw => {
                    let si = curved_power_law_alphas[i_row];
                    let i = curved_power_law_stokes_is[i_row];
                    let q = curved_power_law_qs[i_row];
                    FluxDensityType::CurvedPowerLaw {
                        si,
                        fd: FluxDensity {
                            freq: REF_FREQ_HZ,
                            i,
                            q: 0.0,
                            u: 0.0,
                            v: 0.0,
                        },
                        q,
                    }
                }
            };

            src.push(SourceComponent {
                radec,
                comp_type,
                flux_type,
            })
        }
    }

    // Box all of the vectors.
    let source_list = source_list
        .into_iter()
        .map(|(name, comps)| {
            (
                name,
                Source {
                    components: comps.into_boxed_slice(),
                },
            )
        })
        .collect::<SourceList>();

    Ok(source_list)
}

fn parse_jack_source_list(
    file: &Path,
    mut fptr: FitsFile,
    hdu: FitsHdu,
    col_names: Vec<String>,
) -> Result<SourceList, FitsError> {
    let CommonCols {
        unq_source_id: src_names,
        names: comp_names,
        ra_degrees,
        dec_degrees,
        majors,
        minors,
        pas,
        mut shapelet_sources,
        comp_types,
        flux_types,
        power_law_stokes_is,
        power_law_alphas,
        curved_power_law_stokes_is,
        curved_power_law_alphas,
        curved_power_law_qs,
        list_flux_densities,
        list_flux_density_freqs,
    } = CommonCols::new(file, &mut fptr, &hdu, &col_names)?;
    let mut map = IndexMap::with_capacity(src_names.len());
    // Get all of the source names.
    for name in src_names {
        map.entry(name).or_insert(vec![]);
    }

    // Find all of the components that belong with this source and populate
    // it.
    for (
        i_row,
        (ra_degrees, dec_degrees, flux_type, comp_name, comp_type, gauss_maj, gauss_min, gauss_pa),
    ) in izip!(
        ra_degrees.into_iter(),
        dec_degrees.into_iter(),
        flux_types.into_iter(),
        comp_names.into_iter(),
        comp_types.into_iter(),
        majors.into_iter(),
        minors.into_iter(),
        pas.into_iter(),
    )
    .enumerate()
    {
        let prefix = comp_name
            .rsplit_once("_C")
            .unwrap_or_else(|| panic!("{comp_name:?} does not contain '_C'"))
            .0;
        let src_comps = map.get_mut(prefix).unwrap_or_else(|| {
            panic!("Component '{comp_name}' couldn't be matched against any of the UNQ_SOURCE_ID")
        });

        let radec = RADec::from_degrees(ra_degrees, dec_degrees);
        let gaussian_params_arent_available =
            (gauss_maj.is_nan() && gauss_min.is_nan() && gauss_pa.is_nan())
                || (gauss_maj.abs() < f64::EPSILON
                    && gauss_min.abs() < f64::EPSILON
                    && gauss_pa.abs() < f64::EPSILON);
        let comp_type = match comp_type {
            // The COMP_TYPE column indicates a point source
            'P' => {
                // Check if Gaussian parameters were provided; if so, complain.
                if gaussian_params_arent_available {
                    ComponentType::Point
                } else {
                    panic!("Gaussian parameters were provided for COMP_TYPE 'P' on row {i_row}");
                }
            }

            // The COMP_TYPE column indicates a Gaussian source
            'G' => {
                if gaussian_params_arent_available {
                    panic!("Gaussian parameters weren't provided for COMP_TYPE 'G' on row {i_row}");
                } else {
                    ComponentType::Gaussian {
                        maj: gauss_maj.to_radians(),
                        min: gauss_min.to_radians(),
                        pa: gauss_pa.to_radians(),
                    }
                }
            }

            // The COMP_TYPE column indicates a shapelet source
            'S' => {
                if gaussian_params_arent_available {
                    panic!("Gaussian parameters weren't provided for COMP_TYPE 'S' on row {i_row}");
                } else {
                    // We need to extract the shapelet coeffs associated with
                    // this source.
                    let coeffs = shapelet_sources
                        .as_mut()
                        .expect("shapelet coeffs available if a COMP_TYPE 'S' is present")
                        .remove(&comp_name)
                        .expect("shapelet coeffs available for COMP_TYPE 'S' source");

                    ComponentType::Shapelet {
                        maj: gauss_maj.to_radians(),
                        min: gauss_min.to_radians(),
                        pa: gauss_pa.to_radians(),
                        coeffs: coeffs.into_boxed_slice(),
                    }
                }
            }

            t => panic!("Got an unexpected COMP_TYPE '{t}' on row {i_row}"),
        };

        let flux_type = match flux_type {
            FluxType::List => {
                let mut list_fds = Vec::with_capacity(list_flux_densities.len());
                for (fds, &freq) in list_flux_densities
                    .iter()
                    .zip(list_flux_density_freqs.iter())
                {
                    let fd = fds[i_row];
                    if !fd.is_nan() {
                        list_fds.push(FluxDensity {
                            freq,
                            i: fd,
                            q: 0.0,
                            u: 0.0,
                            v: 0.0,
                        });
                    }
                }
                FluxDensityType::List(Vec1::try_from_vec(list_fds).unwrap_or_else(|_| {
                    panic!("No valid flux densities were provided on row {i_row}")
                }))
            }

            FluxType::PowerLaw => {
                let si = power_law_alphas[i_row];
                let i = power_law_stokes_is[i_row];
                FluxDensityType::PowerLaw {
                    si,
                    fd: FluxDensity {
                        freq: REF_FREQ_HZ,
                        i,
                        q: 0.0,
                        u: 0.0,
                        v: 0.0,
                    },
                }
            }

            FluxType::CurvedPowerLaw => {
                let si = curved_power_law_alphas[i_row];
                let i = curved_power_law_stokes_is[i_row];
                let q = curved_power_law_qs[i_row];
                FluxDensityType::CurvedPowerLaw {
                    si,
                    fd: FluxDensity {
                        freq: REF_FREQ_HZ,
                        i,
                        q: 0.0,
                        u: 0.0,
                        v: 0.0,
                    },
                    q,
                }
            }
        };

        src_comps.push(SourceComponent {
            radec,
            comp_type,
            flux_type,
        });
    }

    // Box all of the vectors.
    let source_list = map
        .into_iter()
        .map(|(name, comps)| {
            if comps.is_empty() {
                panic!("No components are against source {name}; this is a programmer error");
            }

            (
                name,
                Source {
                    components: comps.into_boxed_slice(),
                },
            )
        })
        .collect::<SourceList>();

    Ok(source_list)
}

fn parse_gleam_x_source_list(
    file: &Path,
    mut fptr: FitsFile,
    hdu: FitsHdu,
    col_names: Vec<String>,
) -> Result<SourceList, FitsError> {
    let CommonCols {
        unq_source_id: _,
        names: src_names,
        ra_degrees,
        dec_degrees,
        majors,
        minors,
        pas,
        power_law_stokes_is,
        power_law_alphas,
        curved_power_law_qs,
        ..
    } = CommonCols::new(file, &mut fptr, &hdu, &col_names)?;

    let mut source_list = IndexMap::with_capacity(src_names.len());

    // It appears that each source has one component and everything is a power law.
    for (src_name, ra_degrees, dec_degrees, maj, min, pa, stokes_i, si, q) in izip!(
        src_names,
        ra_degrees,
        dec_degrees,
        majors,
        minors,
        pas,
        power_law_stokes_is,
        power_law_alphas,
        curved_power_law_qs
    ) {
        let radec = RADec::from_degrees(ra_degrees, dec_degrees);
        let comp_type = if (maj.is_nan() && min.is_nan() && pa.is_nan())
            || (maj.abs() < f64::EPSILON && min.abs() < f64::EPSILON && pa.abs() < f64::EPSILON)
        {
            ComponentType::Point
        } else {
            ComponentType::Gaussian {
                maj: (maj / 3600.0).to_radians(),
                min: (min / 3600.0).to_radians(),
                pa: pa.to_radians(),
            }
        };
        let flux_type = if q.is_nan() || q.abs() < f64::EPSILON {
            FluxDensityType::PowerLaw {
                si,
                fd: FluxDensity {
                    freq: REF_FREQ_HZ,
                    i: stokes_i,
                    q: 0.0,
                    u: 0.0,
                    v: 0.0,
                },
            }
        } else {
            FluxDensityType::CurvedPowerLaw {
                si,
                fd: FluxDensity {
                    freq: REF_FREQ_HZ,
                    i: stokes_i,
                    q: 0.0,
                    u: 0.0,
                    v: 0.0,
                },
                q,
            }
        };

        source_list.insert(
            src_name,
            Source {
                components: vec![SourceComponent {
                    radec,
                    comp_type,
                    flux_type,
                }]
                .into_boxed_slice(),
            },
        );
    }

    Ok(SourceList::from(source_list))
}
