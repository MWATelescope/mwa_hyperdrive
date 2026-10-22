// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Reading per-baseline crosstalk scores and summarising them over a night.
//!
//! The input is one CSV per observation, with a row per searchable baseline
//! per instrumental polarisation, as written by the `crosstalk-qa` tool. The
//! columns used here are `pol`, `ant1`, `ant2` (tile names), `length_m`,
//! `tau_peak_ns`, `t`, `z` and `flag`; any others are ignored. A missing value
//! is written as `nan`, and booleans as `True`/`False`.

use std::collections::HashMap;
use std::path::Path;

use log::warn;

use super::{tiles::Tiles, CrosstalkPlotError};

/// The instrumental polarisations that are plotted.
pub(crate) const POLS: [&str; 2] = ["XX", "YY"];

/// One baseline's crosstalk score in one polarisation of one observation.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Score {
    /// Index into [`POLS`].
    pub(crate) pol: usize,
    /// Tile indices, as ordered in the [`Tiles`].
    pub(crate) tile1: usize,
    pub(crate) tile2: usize,
    pub(crate) length_m: f64,
    pub(crate) tau_peak_ns: f64,
    pub(crate) t: f64,
    pub(crate) z: f64,
    pub(crate) flag: bool,
}

/// The scores of one observation.
#[derive(Debug, Clone)]
pub(crate) struct ObsScores {
    /// The obsid if the filename has one, otherwise the file stem.
    pub(crate) label: String,
    pub(crate) scores: Vec<Score>,
}

/// The obsid from a 10-digit run in the file stem, or the stem itself.
pub(crate) fn obs_label(path: &Path) -> String {
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    let bytes = stem.as_bytes();
    let mut run_start = None;
    for (i, b) in bytes.iter().enumerate() {
        if b.is_ascii_digit() {
            let start = *run_start.get_or_insert(i);
            if i + 1 - start == 10 && bytes.get(i + 1).is_none_or(|n| !n.is_ascii_digit()) {
                return stem[start..=i].to_string();
            }
        } else {
            run_start = None;
        }
    }
    stem
}

fn parse_bool(s: &str) -> Option<bool> {
    match s.trim() {
        "True" | "true" | "TRUE" | "1" => Some(true),
        "False" | "false" | "FALSE" | "0" | "" => Some(false),
        _ => None,
    }
}

fn parse_f64(s: &str) -> Option<f64> {
    let s = s.trim();
    if s.is_empty() || s.eq_ignore_ascii_case("nan") {
        Some(f64::NAN)
    } else {
        s.parse().ok()
    }
}

/// Parse one observation's scores. Rows for polarisations other than XX and
/// YY, and rows naming tiles that aren't in the metafits, are skipped; the
/// number of unknown tiles is returned so that the caller can warn once.
pub(crate) fn parse_scores(
    text: &str,
    tiles: &Tiles,
    file: &Path,
) -> Result<(Vec<Score>, usize), CrosstalkPlotError> {
    let mut lines = text.lines().enumerate();
    let header = match lines.next() {
        Some((_, h)) => h,
        None => return Ok((vec![], 0)),
    };
    let columns: Vec<&str> = header.trim_end_matches('\r').split(',').collect();
    let col = |name: &str| {
        columns
            .iter()
            .position(|c| c.trim() == name)
            .ok_or_else(|| CrosstalkPlotError::MissingColumn {
                file: file.to_path_buf(),
                column: name.to_string(),
            })
    };
    let (c_pol, c_ant1, c_ant2) = (col("pol")?, col("ant1")?, col("ant2")?);
    let (c_len, c_tau, c_t, c_z, c_flag) = (
        col("length_m")?,
        col("tau_peak_ns")?,
        col("t")?,
        col("z")?,
        col("flag")?,
    );

    let mut scores = vec![];
    let mut num_unknown = 0;
    for (i_line, line) in lines {
        let line = line.trim_end_matches('\r');
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        let bad_row = || CrosstalkPlotError::BadRow {
            file: file.to_path_buf(),
            line: i_line + 1,
        };
        let get = |c: usize| fields.get(c).copied().ok_or_else(bad_row);
        let pol_name = get(c_pol)?.trim();
        let Some(pol) = POLS.iter().position(|p| *p == pol_name) else {
            continue;
        };
        let (Some(tile1), Some(tile2)) = (
            tiles.index_of(get(c_ant1)?.trim()),
            tiles.index_of(get(c_ant2)?.trim()),
        ) else {
            num_unknown += 1;
            continue;
        };
        let num = |c: usize| get(c).and_then(|s| parse_f64(s).ok_or_else(bad_row));
        scores.push(Score {
            pol,
            tile1,
            tile2,
            length_m: num(c_len)?,
            tau_peak_ns: num(c_tau)?,
            t: num(c_t)?,
            z: num(c_z)?,
            flag: parse_bool(get(c_flag)?).ok_or_else(bad_row)?,
        });
    }
    Ok((scores, num_unknown))
}

/// Read every observation's scores, sorted by label.
pub(crate) fn read_all(
    files: &[impl AsRef<Path>],
    tiles: &Tiles,
) -> Result<Vec<ObsScores>, CrosstalkPlotError> {
    let mut night = vec![];
    for file in files {
        let file = file.as_ref();
        let text = std::fs::read_to_string(file)?;
        let (scores, num_unknown) = parse_scores(&text, tiles, file)?;
        if num_unknown > 0 {
            warn!(
                "{}: skipped {num_unknown} rows whose tiles aren't in the metafits; is the metafits for this array?",
                file.display()
            );
        }
        night.push(ObsScores {
            label: obs_label(file),
            scores,
        });
    }
    night.sort_by(|a, b| a.label.cmp(&b.label));
    Ok(night)
}

/// The flag thresholds the scores were made with, from the JSON sidecar that
/// `crosstalk-qa` writes next to each CSV, if there is one.
pub(crate) fn read_thresholds(csv: &Path) -> Option<(f64, f64)> {
    let text = std::fs::read_to_string(csv.with_extension("json")).ok()?;
    let json: serde_json::Value = serde_json::from_str(&text).ok()?;
    let params = json.get("params")?;
    Some((
        params.get("t_min")?.as_f64()?,
        params.get("z_min")?.as_f64()?,
    ))
}

/// The number of flagged baselines touching each tile in each observation, for
/// one polarisation, as `[tile][obs]`. A tile with no scored baselines in an
/// observation is NaN, to tell "nothing to search" apart from "nothing found".
pub(crate) fn tile_flag_counts(night: &[ObsScores], num_tiles: usize, pol: usize) -> Vec<Vec<f64>> {
    let mut counts = vec![vec![f64::NAN; night.len()]; num_tiles];
    for (i_obs, obs) in night.iter().enumerate() {
        for s in obs.scores.iter().filter(|s| s.pol == pol) {
            for tile in [s.tile1, s.tile2] {
                let c = &mut counts[tile][i_obs];
                if c.is_nan() {
                    *c = 0.0;
                }
                if s.flag {
                    *c += 1.0;
                }
            }
        }
    }
    counts
}

/// A baseline's scores in one polarisation, gathered over the night.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BaselineSummary {
    /// Tile indices, with `tile1 < tile2`.
    pub(crate) tile1: usize,
    pub(crate) tile2: usize,
    pub(crate) length_m: f64,
    pub(crate) median_z: f64,
    pub(crate) num_obs: usize,
    pub(crate) num_flagged: usize,
}

impl BaselineSummary {
    /// The fraction of observations in which this baseline was flagged.
    pub(crate) fn persistence(&self) -> f64 {
        self.num_flagged as f64 / self.num_obs.max(1) as f64
    }
}

/// The median of the finite values, or NaN if there are none.
pub(crate) fn median(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut v: Vec<f64> = values.into_iter().filter(|v| v.is_finite()).collect();
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// Per-baseline summaries for one polarisation, sorted by tile pair.
pub(crate) fn baseline_summaries(night: &[ObsScores], pol: usize) -> Vec<BaselineSummary> {
    let mut gathered: HashMap<(usize, usize), (f64, Vec<f64>, usize)> = HashMap::new();
    for s in night
        .iter()
        .flat_map(|o| o.scores.iter())
        .filter(|s| s.pol == pol)
    {
        let key = (s.tile1.min(s.tile2), s.tile1.max(s.tile2));
        let entry = gathered.entry(key).or_insert((s.length_m, vec![], 0));
        entry.1.push(s.z);
        if s.flag {
            entry.2 += 1;
        }
    }
    let mut summaries: Vec<BaselineSummary> = gathered
        .into_iter()
        .map(
            |((tile1, tile2), (length_m, zs, num_flagged))| BaselineSummary {
                tile1,
                tile2,
                length_m,
                num_obs: zs.len(),
                median_z: median(zs),
                num_flagged,
            },
        )
        .collect();
    summaries.sort_by_key(|s| (s.tile1, s.tile2));
    summaries
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::cli::crosstalk_plot::tiles::TileInfo;

    fn tiles() -> Tiles {
        Tiles::new(
            ["Tile011", "Tile012", "Tile014"]
                .iter()
                .enumerate()
                .map(|(i, n)| TileInfo {
                    name: n.to_string(),
                    ant: i as u32,
                    rx: 19,
                    flavour: "RG6_90".to_string(),
                })
                .collect(),
        )
    }

    const CSV: &str = "pol,ant1,ant2,length_m,horizon_ns,tau_peak_ns,t,z,same_rx,flag\r\n\
XX,Tile011,Tile012,54.6,182.1,-2669.3,4.27,-0.47,True,False\r\n\
XX,Tile011,Tile014,71.47,238.4,-455.7,85.06,6.60,True,True\r\n\
XX,Tile012,Tile099,20.4,68.0,-455.7,535.79,7.21,True,True\r\n\
YY,Tile012,Tile014,20.4,68.0,nan,0.0,-7.32,False,False\r\n\
XY,Tile011,Tile012,54.6,182.1,1.0,1.0,1.0,True,True\r\n";

    #[test]
    fn parses_rows_and_skips_unknown_tiles_and_cross_pols() {
        let (scores, unknown) = parse_scores(CSV, &tiles(), &PathBuf::from("x.csv")).unwrap();
        assert_eq!(unknown, 1);
        assert_eq!(scores.len(), 3);
        assert_eq!(scores[1].tile2, 2);
        assert!(scores[1].flag);
        assert_eq!(scores[2].pol, 1);
        assert!(scores[2].tau_peak_ns.is_nan());
        assert_eq!(scores[2].z, -7.32);
    }

    #[test]
    fn missing_column_is_an_error() {
        let csv = "pol,ant1,ant2,length_m,t,z,flag\nXX,Tile011,Tile012,1,1,1,True\n";
        match parse_scores(csv, &tiles(), &PathBuf::from("x.csv")) {
            Err(CrosstalkPlotError::MissingColumn { column, .. }) => {
                assert_eq!(column, "tau_peak_ns")
            }
            other => panic!("unexpected result: {other:?}"),
        }
    }

    #[test]
    fn short_row_is_an_error() {
        let csv = "pol,ant1,ant2,length_m,tau_peak_ns,t,z,flag\nXX,Tile011,Tile012,1\n";
        match parse_scores(csv, &tiles(), &PathBuf::from("x.csv")) {
            Err(CrosstalkPlotError::BadRow { line: 2, .. }) => (),
            other => panic!("unexpected result: {other:?}"),
        }
    }

    #[test]
    fn obs_labels_prefer_the_obsid() {
        assert_eq!(obs_label(Path::new("xt-out/1472670968.csv")), "1472670968");
        assert_eq!(obs_label(Path::new("xt_1472670968_v2.csv")), "1472670968");
        // An 11-digit run isn't an obsid.
        assert_eq!(obs_label(Path::new("14726709681.csv")), "14726709681");
        assert_eq!(obs_label(Path::new("night.csv")), "night");
    }

    #[test]
    fn tile_counts_tell_unsearched_from_unflagged() {
        let t = tiles();
        let (scores, _) = parse_scores(CSV, &t, &PathBuf::from("x.csv")).unwrap();
        let night = vec![ObsScores {
            label: "a".to_string(),
            scores,
        }];
        let xx = tile_flag_counts(&night, t.len(), 0);
        assert_eq!(xx[0][0], 1.0);
        assert_eq!(xx[1][0], 0.0);
        assert_eq!(xx[2][0], 1.0);
        let yy = tile_flag_counts(&night, t.len(), 1);
        assert!(yy[0][0].is_nan());
        assert_eq!(yy[1][0], 0.0);
    }

    #[test]
    fn baselines_are_summarised_over_the_night() {
        let t = tiles();
        let (scores, _) = parse_scores(CSV, &t, &PathBuf::from("x.csv")).unwrap();
        let mut flipped = scores.clone();
        for s in &mut flipped {
            std::mem::swap(&mut s.tile1, &mut s.tile2);
            s.z += 2.0;
            s.flag = false;
        }
        let night = vec![
            ObsScores {
                label: "a".to_string(),
                scores,
            },
            ObsScores {
                label: "b".to_string(),
                scores: flipped,
            },
        ];
        let xx = baseline_summaries(&night, 0);
        assert_eq!(xx.len(), 2);
        assert_eq!((xx[1].tile1, xx[1].tile2), (0, 2));
        assert_eq!(xx[1].num_obs, 2);
        assert_eq!(xx[1].num_flagged, 1);
        assert!((xx[1].median_z - 7.6).abs() < 1e-9);
        assert_eq!(xx[1].persistence(), 0.5);
    }

    #[test]
    fn median_ignores_nan() {
        assert_eq!(median([3.0, f64::NAN, 1.0, 2.0]), 2.0);
        assert_eq!(median([4.0, 1.0]), 2.5);
        assert!(median([f64::NAN]).is_nan());
    }
}
