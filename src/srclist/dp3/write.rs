// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Writing DP3/BBS-style ("makesourcedb") text sky models.
//!
//! The inverse of [`super::read`]:
//!
//! - Each hyperdrive [`Source`](crate::srclist::Source) becomes a patch named
//!   after the source, placed at the Stokes-I-weighted centroid of its
//!   components; its components are named `<patch>_<index>`. Characters that
//!   DP3's parser treats specially (commas, whitespace, brackets and quotes)
//!   are replaced by underscores in patch names, and repeated names get a
//!   numeric suffix.
//! - Positions are written in degrees with an explicit `deg` unit: DP3 reads a
//!   bare number as radians, and a `:` in a Dec as hours.
//! - Gaussian axes are written in arcsec, orientations in degrees.
//! - Power laws are `SpectralIndex=[si]`, curved power laws `[si, q ln(10)]`
//!   (both with LogarithmicSI=true), which DP3 evaluates exactly as hyperdrive
//!   does. DP3 only applies spectral terms to Stokes I, so non-zero Q, U or V
//!   become flat spectra (a warning is printed).
//! - DP3 has no list-type flux densities. A one-entry list is written as the
//!   power law hyperdrive extrapolates it with (spectral index
//!   [`DEFAULT_SPEC_INDEX`]), a two-entry list as the power law through both
//!   entries (both exact), and longer lists as the least-squares curved power
//!   law in log10(S) vs log10(ν) (a warning reports the largest relative error
//!   at the listed frequencies). Lists with a non-positive Stokes I can't be
//!   fitted in log space; sources containing them are not written.
//! - Shapelet components are not supported by DP3; sources containing them
//!   are not written.

use std::{collections::HashSet, f64::consts::LN_10};

use log::debug;

use crate::{
    cli::Warn,
    constants::DEFAULT_SPEC_INDEX,
    srclist::{
        error::WriteSourceListError, ComponentType, FluxDensity, FluxDensityType, Source,
        SourceList,
    },
};

const FORMAT_LINE: &str = "FORMAT = Name, Type, Patch, Ra, Dec, I, Q, U, V, SpectralIndex, LogarithmicSI, ReferenceFrequency, MajorAxis, MinorAxis, Orientation";

/// A Stokes I reference flux density and the DP3 logarithmic spectral terms
/// that go with it.
struct Dp3Spectrum {
    fd: FluxDensity,
    terms: Vec<f64>,
}

/// Why a flux density can't be written; used to skip a whole source.
enum Unwritable {
    NonPositiveList,
}

/// The largest relative Stokes I error, at the listed frequencies, of the
/// curved power laws fitted to lists with three or more entries.
struct ListFit {
    num: usize,
    max_rel_err: f64,
}

fn spectrum(
    flux_type: &FluxDensityType,
    list_fit: &mut ListFit,
) -> Result<Dp3Spectrum, Unwritable> {
    Ok(match flux_type {
        FluxDensityType::PowerLaw { si, fd } => Dp3Spectrum {
            fd: *fd,
            terms: vec![*si],
        },

        FluxDensityType::CurvedPowerLaw { si, fd, q } => Dp3Spectrum {
            fd: *fd,
            terms: vec![*si, q * LN_10],
        },

        FluxDensityType::List(fds) => {
            if fds.iter().any(|fd| fd.i <= 0.0) {
                return Err(Unwritable::NonPositiveList);
            }
            match fds.as_slice() {
                // hyperdrive extrapolates a single entry with the default
                // spectral index.
                [fd] => Dp3Spectrum {
                    fd: *fd,
                    terms: vec![DEFAULT_SPEC_INDEX],
                },

                // hyperdrive interpolates (and extrapolates) between two entries
                // with the power law through both.
                [fd1, fd2] => Dp3Spectrum {
                    fd: *fd1,
                    terms: vec![(fd2.i / fd1.i).ln() / (fd2.freq / fd1.freq).ln()],
                },

                _ => {
                    let (spectrum, rel_err) = fit_curved_power_law(fds);
                    list_fit.num += 1;
                    list_fit.max_rel_err = list_fit.max_rel_err.max(rel_err);
                    spectrum
                }
            }
        }
    })
}

/// Least-squares fit of log10(I) = log10(I0) + c0 x + c1 x², x = log10(ν/ν0),
/// with ν0 the geometric mean of the listed frequencies. Q, U and V are taken
/// from the entry nearest ν0. Returns the spectrum and the largest relative
/// Stokes I error at the listed frequencies.
fn fit_curved_power_law(fds: &[FluxDensity]) -> (Dp3Spectrum, f64) {
    let n = fds.len() as f64;
    let freq0 = (fds.iter().map(|fd| fd.freq.ln()).sum::<f64>() / n).exp();
    let xs: Vec<f64> = fds.iter().map(|fd| (fd.freq / freq0).log10()).collect();
    let ys: Vec<f64> = fds.iter().map(|fd| fd.i.log10()).collect();

    // Normal equations for y = b0 + b1 x + b2 x².
    let mut a = [[0.0; 3]; 3];
    let mut rhs = [0.0; 3];
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        let powers = [1.0, x, x * x];
        for (r, &pr) in powers.iter().enumerate() {
            rhs[r] += pr * y;
            for (c, &pc) in powers.iter().enumerate() {
                a[r][c] += pr * pc;
            }
        }
    }
    let b = solve3(a, rhs);

    let nearest = fds
        .iter()
        .min_by(|f1, f2| {
            (f1.freq / freq0)
                .ln()
                .abs()
                .total_cmp(&(f2.freq / freq0).ln().abs())
        })
        .expect("lists are non-empty");
    let fd = FluxDensity {
        freq: freq0,
        i: 10_f64.powf(b[0]),
        q: nearest.q,
        u: nearest.u,
        v: nearest.v,
    };
    let rel_err = xs
        .iter()
        .zip(fds.iter())
        .map(|(&x, fd)| (10_f64.powf(b[0] + b[1] * x + b[2] * x * x) / fd.i - 1.0).abs())
        .fold(0.0, f64::max);
    (
        Dp3Spectrum {
            fd,
            terms: vec![b[1], b[2]],
        },
        rel_err,
    )
}

/// Solve a 3x3 linear system with Gaussian elimination and partial pivoting.
/// A singular system (e.g. three entries at only two distinct frequencies)
/// gets a zero for the undetermined coefficient.
fn solve3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> [f64; 3] {
    for col in 0..3 {
        let pivot = (col..3)
            .max_by(|&r1, &r2| a[r1][col].abs().total_cmp(&a[r2][col].abs()))
            .unwrap();
        a.swap(col, pivot);
        b.swap(col, pivot);
        if a[col][col].abs() < 1e-300 {
            continue;
        }
        let pivot_row = a[col];
        for row in col + 1..3 {
            let f = a[row][col] / pivot_row[col];
            for (a_rc, p_c) in a[row][col..].iter_mut().zip(&pivot_row[col..]) {
                *a_rc -= f * p_c;
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = [0.0; 3];
    for row in (0..3).rev() {
        if a[row][row].abs() < 1e-300 {
            continue;
        }
        let s: f64 = (row + 1..3).map(|c| a[row][c] * x[c]).sum();
        x[row] = (b[row] - s) / a[row][row];
    }
    x
}

/// A patch name DP3 can parse, unique within the file.
fn patch_name(name: &str, used: &mut HashSet<String>) -> String {
    let mut base: String = name
        .chars()
        .map(|c| match c {
            ',' | '[' | ']' | '\'' | '"' => '_',
            c if c.is_whitespace() => '_',
            c => c,
        })
        .collect();
    if base.is_empty() {
        base = "source".to_string();
    }
    let mut candidate = base.clone();
    let mut n = 2;
    while used.contains(&candidate) {
        candidate = format!("{base}_{n}");
        n += 1;
    }
    used.insert(candidate.clone());
    candidate
}

/// The Stokes-I-weighted centroid of a source's components (unweighted if
/// the weights cancel), as (RA, Dec) in degrees.
fn patch_position(source: &Source) -> (f64, f64) {
    let centroid = |weighted: bool| {
        let mut v = [0.0; 3];
        for c in source.components.iter() {
            let w = if weighted {
                c.flux_type.estimate_at_freq(150e6).i.abs()
            } else {
                1.0
            };
            let (ra, dec) = (c.radec.ra, c.radec.dec);
            v[0] += w * dec.cos() * ra.cos();
            v[1] += w * dec.cos() * ra.sin();
            v[2] += w * dec.sin();
        }
        v
    };
    let mut v = centroid(true);
    if v.iter().map(|x| x * x).sum::<f64>() < 1e-300 {
        v = centroid(false);
    }
    let ra = v[1].atan2(v[0]).to_degrees().rem_euclid(360.0);
    let dec = v[2].atan2(v[0].hypot(v[1])).to_degrees();
    (ra, dec)
}

pub(crate) fn write_source_list<T: std::io::Write>(
    buf: &mut T,
    sl: &SourceList,
    num_sources: Option<usize>,
) -> Result<(), WriteSourceListError> {
    let mut skipped_shapelets = 0;
    let mut skipped_non_positive_lists = 0;
    let mut pol_with_spectral_terms = 0;
    let mut list_fit = ListFit {
        num: 0,
        max_rel_err: 0.0,
    };
    let mut used_names = HashSet::new();

    writeln!(buf, "{FORMAT_LINE}")?;

    let mut num_written_sources = 0;
    // Note that, if sorted, each source in the source list is dimmer than the
    // last!
    for (name, source) in sl.iter() {
        // If `num_sources` is supplied, then check that we're not writing out
        // too many sources.
        if let Some(num_sources) = num_sources {
            if num_written_sources == num_sources {
                break;
            }
        }

        if source
            .components
            .iter()
            .any(|c| matches!(c.comp_type, ComponentType::Shapelet { .. }))
        {
            debug!("Ignoring source {name} as it contains a shapelet component");
            skipped_shapelets += 1;
            continue;
        }
        let spectra: Result<Vec<Dp3Spectrum>, Unwritable> = source
            .components
            .iter()
            .map(|c| spectrum(&c.flux_type, &mut list_fit))
            .collect();
        let spectra = match spectra {
            Ok(s) => s,
            Err(Unwritable::NonPositiveList) => {
                debug!("Ignoring source {name} as it has a list flux density with a non-positive Stokes I");
                skipped_non_positive_lists += 1;
                continue;
            }
        };

        let patch = patch_name(name, &mut used_names);
        let (patch_ra, patch_dec) = patch_position(source);
        writeln!(buf, " , , {patch}, {patch_ra}deg, {patch_dec}deg")?;

        for (i_comp, (c, spec)) in source.components.iter().zip(spectra).enumerate() {
            let (comp_type, shape) = match c.comp_type {
                ComponentType::Point => ("POINT", String::from(", , ")),
                ComponentType::Gaussian { maj, min, pa } => (
                    "GAUSSIAN",
                    format!(
                        "{}, {}, {}",
                        maj.to_degrees() * 3600.0,
                        min.to_degrees() * 3600.0,
                        pa.to_degrees()
                    ),
                ),
                ComponentType::Shapelet { .. } => {
                    unreachable!("sources with shapelets are skipped")
                }
            };
            let Dp3Spectrum { fd, terms } = spec;
            if terms.iter().any(|&t| t != 0.0) && (fd.q != 0.0 || fd.u != 0.0 || fd.v != 0.0) {
                pol_with_spectral_terms += 1;
            }
            let terms = terms
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            writeln!(
                buf,
                "{patch}_{i_comp}, {comp_type}, {patch}, {}deg, {}deg, {}, {}, {}, {}, [{terms}], true, {}, {shape}",
                c.radec.ra.to_degrees(),
                c.radec.dec.to_degrees(),
                fd.i,
                fd.q,
                fd.u,
                fd.v,
                fd.freq,
            )?;
        }
        num_written_sources += 1;
    }
    buf.flush()?;

    if skipped_shapelets > 0 {
        format!("DP3 sky models don't support shapelet components; {skipped_shapelets} sources containing them weren't written").warn();
    }
    if skipped_non_positive_lists > 0 {
        format!("{skipped_non_positive_lists} sources have list flux densities with a non-positive Stokes I, which DP3's logarithmic spectra can't represent; they weren't written").warn();
    }
    if list_fit.num > 0 {
        format!(
            "{} list flux densities with 3 or more entries were written as fitted curved power laws (largest Stokes I error at a listed frequency: {:.2}%)",
            list_fit.num,
            list_fit.max_rel_err * 100.0
        )
        .warn();
    }
    if pol_with_spectral_terms > 0 {
        format!("{pol_with_spectral_terms} components have non-zero Q, U or V and spectral terms; DP3 keeps Q, U and V constant with frequency, hyperdrive scales them with Stokes I").warn();
    }
    if let Some(num_sources) = num_sources {
        if num_sources > num_written_sources {
            format!("Couldn't write the requested number of sources ({num_sources}): wrote {num_written_sources}").warn()
        }
    }

    Ok(())
}
