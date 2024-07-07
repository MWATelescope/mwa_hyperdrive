use std::{ffi::CString, path::Path};

use super::REF_FREQ_HZ;
use crate::{
    io::read::fits::fits_open_hdu,
    srclist::{error::WriteSourceListError, ComponentType, FluxDensityType, SourceList},
};
use fitsio::{
    errors::check_status as fits_check_status,
    tables::{ColumnDataType, ColumnDescription},
    FitsFile,
};

pub(crate) fn write_source_list_jack(
    file: &Path,
    sl: &SourceList,
    num_sources: Option<usize>,
) -> Result<(), WriteSourceListError> {
    if file.exists() {
        std::fs::remove_file(file)?;
    }
    let mut fptr = FitsFile::create(file).open()?;
    let hdu = fits_open_hdu(&mut fptr, 0)?;
    let mut status = 0;

    // Signal that we're using long strings.
    unsafe {
        // ffplsw = fits_write_key_longwarn
        fitsio_sys::ffplsw(
            fptr.as_raw(), /* I - FITS file pointer  */
            &mut status,   /* IO - error status       */
        );
        fits_check_status(status)?;
    }

    // Write the documentation URL as a comment.
    unsafe {
        let comm = CString::new("The contents of this file are documented at:").unwrap();
        // ffpcom = fits_write_comment
        fitsio_sys::ffpcom(fptr.as_raw(), comm.as_ptr(), &mut status);
        fits_check_status(status)?;
        let comm = CString::new(
            "https://mwatelescope.github.io/mwa_hyperdrive/defs/source_list_fits_jack.html",
        )
        .unwrap();
        fitsio_sys::ffpcom(fptr.as_raw(), comm.as_ptr(), &mut status);
        fits_check_status(status)?;
    }

    hdu.write_key(
        &mut fptr,
        "SOFTWARE",
        format!(
            "Created by {} v{}",
            env!("CARGO_PKG_NAME"),
            env!("CARGO_PKG_VERSION")
        ),
    )?;

    // Write out the current command-line call ("CMDLINE").
    unsafe {
        // It's possible that the command-line call has invalid UTF-8. So use
        // args_os and attempt to convert to UTF-8 strings. If there are
        // problems on the way, don't bother trying to write the CMDLINE key.
        match std::env::args_os()
            .map(|a| a.into_string())
            .collect::<Result<Vec<String>, _>>()
            .and_then(|v| CString::new(v.join(" ")).map_err(|_| std::ffi::OsString::from("")))
        {
            // This represents failure to convert an argument to UTF-8.
            Err(_) => (),
            Ok(value) => {
                let key_name = CString::new("CMDLINE").unwrap();
                let comment = CString::new("Command-line call").unwrap();
                let mut status = 0;
                // ffpkls = fits_write_key_longstr
                fitsio_sys::ffpkls(
                    fptr.as_raw(),     /* I - FITS file pointer        */
                    key_name.as_ptr(), /* I - name of keyword to write */
                    value.as_ptr(),    /* I - keyword value            */
                    comment.as_ptr(),  /* I - keyword comment          */
                    &mut status,       /* IO - error status            */
                );
                fits_check_status(status)?;
            }
        }
    }

    // build the components and shapelets tables
    let mut unq_source_id = Vec::new();
    let mut names = Vec::new();
    let mut ra_degrees = Vec::new();
    let mut dec_degrees = Vec::new();
    let mut majors = Vec::new();
    let mut minors = Vec::new();
    let mut pas = Vec::new();
    let mut comp_types = Vec::new();
    let mut mod_types = Vec::new();
    let mut norm_comp_pls = Vec::new();
    let mut alpha_pls = Vec::new();
    let mut norm_comp_cpls = Vec::new();
    let mut alpha_cpls = Vec::new();
    let mut curve_cpls = Vec::new();

    let mut shapelet_sources = Vec::new();
    let mut shapelet_n1s: Vec<i32> = Vec::new();
    let mut shapelet_n2s: Vec<i32> = Vec::new();
    let mut shapelet_coeff_values = Vec::new();

    // get unique flux list frequencies (to nearest MHz)
    let mut flux_freqs = vec![];
    for comp in sl.iter().flat_map(|(_, src)| src.components.iter()) {
        match &comp.flux_type {
            FluxDensityType::List(l) => {
                for f in l.iter() {
                    let freq_mhz = (f.freq / 1e6).round() as i32;
                    if !flux_freqs.contains(&freq_mhz) {
                        flux_freqs.push(freq_mhz);
                    }
                }
            }
            _ => continue,
        }
    }
    flux_freqs.sort();
    let mut flux_lists: Vec<Vec<f64>> = Vec::new();
    for _ in flux_freqs.iter() {
        flux_lists.push(Vec::new());
    }

    let mut max_src_name = 0;

    for (sidx, (src_name, src)) in sl.iter().enumerate() {
        if src_name.len() > max_src_name {
            max_src_name = src_name.len();
        }
        if let Some(num_sources) = num_sources {
            if sidx >= num_sources {
                break;
            }
        }
        for (cidx, comp) in src.components.iter().enumerate() {
            unq_source_id.push(src_name.to_string());
            let comp_name = format!("{src_name}_C{cidx:02}");
            names.push(comp_name.clone());
            ra_degrees.push(comp.radec.ra.to_degrees());
            dec_degrees.push(comp.radec.dec.to_degrees());
            match &comp.comp_type {
                ComponentType::Point => {
                    majors.push(f64::NAN);
                    minors.push(f64::NAN);
                    pas.push(f64::NAN);
                    comp_types.push("P".to_string());
                }

                ComponentType::Gaussian { maj, min, pa } => {
                    majors.push(maj.to_degrees());
                    minors.push(min.to_degrees());
                    pas.push(pa.to_degrees());
                    comp_types.push("G".to_string());
                }

                ComponentType::Shapelet {
                    maj,
                    min,
                    pa,
                    coeffs,
                } => {
                    majors.push(maj.to_degrees());
                    minors.push(min.to_degrees());
                    pas.push(pa.to_degrees());
                    comp_types.push("S".to_string());

                    for coeff in coeffs.iter() {
                        shapelet_sources.push(comp_name.clone());
                        shapelet_n1s.push(coeff.n1 as i32);
                        shapelet_n2s.push(coeff.n2 as i32);
                        shapelet_coeff_values.push(coeff.value);
                    }
                }
            }
            match &comp.flux_type {
                FluxDensityType::List(l) => {
                    mod_types.push("nan".to_string());
                    for (idx, freq) in flux_freqs.iter().enumerate() {
                        let flux = l.iter().find(|f| (f.freq / 1e6).round() as i32 == *freq);
                        match flux {
                            Some(f) => flux_lists[idx].push(f.i),
                            None => flux_lists[idx].push(f64::NAN),
                        }
                    }
                    norm_comp_pls.push(f64::NAN);
                    alpha_pls.push(f64::NAN);
                    norm_comp_cpls.push(f64::NAN);
                    alpha_cpls.push(f64::NAN);
                    curve_cpls.push(f64::NAN);
                }
                FluxDensityType::CurvedPowerLaw { si, fd, q } => {
                    mod_types.push("cpl".to_string());
                    for (idx, _) in flux_freqs.iter().enumerate() {
                        flux_lists[idx].push(f64::NAN);
                    }
                    if fd.freq != REF_FREQ_HZ {
                        panic!("Curve Power-law flux densities must be at the reference frequency");
                    }
                    norm_comp_pls.push(f64::NAN);
                    alpha_pls.push(f64::NAN);
                    norm_comp_cpls.push(fd.i);
                    alpha_cpls.push(*si);
                    curve_cpls.push(*q);
                }
                FluxDensityType::PowerLaw { si, fd } => {
                    mod_types.push("pl".to_string());
                    for (idx, _) in flux_freqs.iter().enumerate() {
                        flux_lists[idx].push(f64::NAN);
                    }
                    if fd.freq != REF_FREQ_HZ {
                        panic!("Power-law flux densities must be at the reference frequency");
                    }
                    norm_comp_pls.push(fd.i);
                    alpha_pls.push(*si);
                    norm_comp_cpls.push(f64::NAN);
                    alpha_cpls.push(f64::NAN);
                    curve_cpls.push(f64::NAN);
                }
            }
        }
    }

    // write the components HDU
    let mut table_description = vec![
        ColumnDescription::new("UNQ_SOURCE_ID")
            .with_type(ColumnDataType::String)
            .that_repeats(max_src_name)
            .create()?,
        ColumnDescription::new("NAME")
            .with_type(ColumnDataType::String)
            .that_repeats(max_src_name + 4)
            .create()?,
        ColumnDescription::new("RA")
            .with_type(ColumnDataType::Double)
            .create()?,
        ColumnDescription::new("DEC")
            .with_type(ColumnDataType::Double)
            .create()?,
    ];
    for freq in flux_freqs.iter() {
        table_description.push(
            ColumnDescription::new(&format!("INT_FLX{freq}"))
                .with_type(ColumnDataType::Double)
                .create()?,
        );
    }
    let mut extra = vec![
        ColumnDescription::new("MAJOR_DC")
            .with_type(ColumnDataType::Double)
            .create()?,
        ColumnDescription::new("MINOR_DC")
            .with_type(ColumnDataType::Double)
            .create()?,
        ColumnDescription::new("PA_DC")
            .with_type(ColumnDataType::Double)
            .create()?,
        ColumnDescription::new("MOD_TYPE")
            .with_type(ColumnDataType::String)
            .that_repeats(3)
            .create()?,
        ColumnDescription::new("COMP_TYPE")
            .with_type(ColumnDataType::String)
            .create()?,
        ColumnDescription::new("NORM_COMP_PL")
            .with_type(ColumnDataType::Double)
            .create()?,
        ColumnDescription::new("ALPHA_PL")
            .with_type(ColumnDataType::Double)
            .create()?,
        ColumnDescription::new("NORM_COMP_CPL")
            .with_type(ColumnDataType::Double)
            .create()?,
        ColumnDescription::new("ALPHA_CPL")
            .with_type(ColumnDataType::Double)
            .create()?,
        ColumnDescription::new("CURVE_CPL")
            .with_type(ColumnDataType::Double)
            .create()?,
    ];
    table_description.append(&mut extra);
    let hdu = fptr.create_table("COMPONENTS", &table_description)?;
    hdu.write_col(&mut fptr, "UNQ_SOURCE_ID", &unq_source_id)?;
    hdu.write_col(&mut fptr, "NAME", &names)?;
    hdu.write_col(&mut fptr, "RA", &ra_degrees)?;
    hdu.write_col(&mut fptr, "DEC", &dec_degrees)?;
    for (idx, freq) in flux_freqs.iter().enumerate() {
        hdu.write_col(&mut fptr, &format!("INT_FLX{freq}"), &flux_lists[idx])?;
    }
    hdu.write_col(&mut fptr, "MAJOR_DC", &majors)?;
    hdu.write_col(&mut fptr, "MINOR_DC", &minors)?;
    hdu.write_col(&mut fptr, "PA_DC", &pas)?;
    hdu.write_col(&mut fptr, "MOD_TYPE", &mod_types)?;
    hdu.write_col(&mut fptr, "COMP_TYPE", &comp_types)?;
    hdu.write_col(&mut fptr, "NORM_COMP_PL", &norm_comp_pls)?;
    hdu.write_col(&mut fptr, "ALPHA_PL", &alpha_pls)?;
    hdu.write_col(&mut fptr, "NORM_COMP_CPL", &norm_comp_cpls)?;
    hdu.write_col(&mut fptr, "ALPHA_CPL", &alpha_cpls)?;
    hdu.write_col(&mut fptr, "CURVE_CPL", &curve_cpls)?;

    let table_description = vec![
        ColumnDescription::new("NAME")
            .with_type(ColumnDataType::String)
            .that_repeats(max_src_name + 4)
            .create()?,
        ColumnDescription::new("N1")
            .with_type(ColumnDataType::Int)
            .create()?,
        ColumnDescription::new("N2")
            .with_type(ColumnDataType::Int)
            .create()?,
        ColumnDescription::new("COEFF")
            .with_type(ColumnDataType::Double)
            .create()?,
    ];

    let hdu = fptr.create_table("SHAPELETS", &table_description)?;
    hdu.write_col(&mut fptr, "NAME", &shapelet_sources)?;
    hdu.write_col(&mut fptr, "N1", &shapelet_n1s)?;
    hdu.write_col(&mut fptr, "N2", &shapelet_n2s)?;
    hdu.write_col(&mut fptr, "COEFF", &shapelet_coeff_values)?;

    Ok(())
}
