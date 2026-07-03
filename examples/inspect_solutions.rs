use std::{
    cmp::Ordering,
    env, fs,
    io::Write,
    path::{Path, PathBuf},
};

use mwa_hyperdrive::CalibrationSolutions;
use serde::Serialize;

#[derive(Debug, Serialize)]
struct OverallSummary {
    num_timeblocks: usize,
    num_tiles_total: usize,
    num_chanblocks_total: usize,
    num_flagged_tiles: usize,
    num_flagged_chanblocks: usize,
    num_unflagged_tiles: usize,
    num_unflagged_chanblocks: usize,
    total_unflagged_tile_chanblock_cells: usize,
    nan_tile_chanblock_cells: usize,
    nan_tile_chanblock_fraction: f64,
    fully_failed_timeblock_chanblocks: usize,
    fully_failed_timeblock_chanblock_fraction: f64,
    partially_failed_timeblock_chanblocks: usize,
    partially_failed_timeblock_chanblock_fraction: f64,
    valid_solution_cells: usize,
    example_valid_gain_amp_mean: Option<f64>,
}

#[derive(Debug, Serialize)]
struct PrecisionSummary {
    finite_count: usize,
    min: f64,
    p50: f64,
    p90: f64,
    p99: f64,
    max: f64,
    above_min_threshold_count: usize,
    above_min_threshold_fraction: f64,
}

#[derive(Debug, Serialize, Clone)]
struct TimeblockSummary {
    timeblock: usize,
    average_timestamp_unix_s: Option<f64>,
    failed_chanblocks: usize,
    failed_chanblock_fraction: f64,
    partial_chanblocks: usize,
    partial_chanblock_fraction: f64,
    nan_tile_cells: usize,
    nan_tile_cell_fraction: f64,
}

#[derive(Debug, Serialize, Clone)]
struct ChanblockSummary {
    chanblock: usize,
    freq_mhz: Option<f64>,
    failed_timeblocks: usize,
    failed_timeblock_fraction: f64,
    partial_timeblocks: usize,
    partial_timeblock_fraction: f64,
    nan_tile_cells: usize,
    nan_tile_cell_fraction: f64,
    gain_amp_mean: Option<f64>,
    gain_amp_std: Option<f64>,
    precision_p50: Option<f64>,
    precision_p90: Option<f64>,
}

#[derive(Debug, Serialize, Clone)]
struct TileSummary {
    tile: usize,
    nan_cells: usize,
    nan_cell_fraction: f64,
}

#[derive(Debug, Serialize)]
struct InspectSummary {
    file: String,
    beam_file: Option<String>,
    modeller: Option<String>,
    uvw_min_m: Option<f64>,
    uvw_max_m: Option<f64>,
    freq_centroid_hz: Option<f64>,
    stop_threshold: Option<f64>,
    min_threshold: Option<f64>,
    overall: OverallSummary,
    precision: Option<PrecisionSummary>,
    worst_timeblocks: Vec<TimeblockSummary>,
    worst_chanblocks: Vec<ChanblockSummary>,
    worst_tiles: Vec<TileSummary>,
    example_non_nan_solution: Option<[f64; 8]>,
}

#[derive(Clone, Copy, Default)]
struct RunningStats {
    count: usize,
    mean: f64,
    m2: f64,
}

impl RunningStats {
    fn push(&mut self, x: f64) {
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    fn mean(self) -> Option<f64> {
        (self.count > 0).then_some(self.mean)
    }

    fn stddev(self) -> Option<f64> {
        (self.count > 1).then_some((self.m2 / self.count as f64).sqrt())
    }
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let q = q.clamp(0.0, 1.0);
    let idx = q * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let w = idx - lo as f64;
        sorted[lo] * (1.0 - w) + sorted[hi] * w
    }
}

fn cmp_desc_f64(a: f64, b: f64) -> Ordering {
    b.partial_cmp(&a).unwrap_or(Ordering::Equal)
}

fn scalar_gain_amp(j: marlu::Jones<f64>) -> Option<f64> {
    if j.any_nan() {
        return None;
    }
    let amps = [j[0].norm_sqr().sqrt(), j[3].norm_sqr().sqrt()]
        .into_iter()
        .filter(|a| a.is_finite() && *a > 0.0)
        .collect::<Vec<_>>();
    if amps.is_empty() {
        None
    } else {
        Some(amps.iter().sum::<f64>() / amps.len() as f64)
    }
}

fn csv_path(base_json: &Path, suffix: &str) -> PathBuf {
    let parent = base_json.parent().unwrap_or_else(|| Path::new("."));
    let stem = base_json
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("inspect_summary");
    parent.join(format!("{stem}_{suffix}.csv"))
}

fn write_timeblock_csv(
    path: &Path,
    rows: &[TimeblockSummary],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut f = fs::File::create(path)?;
    writeln!(
        f,
        "timeblock,average_timestamp_unix_s,failed_chanblocks,failed_chanblock_fraction,partial_chanblocks,partial_chanblock_fraction,nan_tile_cells,nan_tile_cell_fraction"
    )?;
    for r in rows {
        let ts = r
            .average_timestamp_unix_s
            .map(|v| v.to_string())
            .unwrap_or_default();
        writeln!(
            f,
            "{},{},{},{:.12},{},{:.12},{},{:.12}",
            r.timeblock,
            ts,
            r.failed_chanblocks,
            r.failed_chanblock_fraction,
            r.partial_chanblocks,
            r.partial_chanblock_fraction,
            r.nan_tile_cells,
            r.nan_tile_cell_fraction
        )?;
    }
    Ok(())
}

fn write_chanblock_csv(
    path: &Path,
    rows: &[ChanblockSummary],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut f = fs::File::create(path)?;
    writeln!(
        f,
        "chanblock,freq_mhz,failed_timeblocks,failed_timeblock_fraction,partial_timeblocks,partial_timeblock_fraction,nan_tile_cells,nan_tile_cell_fraction,gain_amp_mean,gain_amp_std,precision_p50,precision_p90"
    )?;
    for r in rows {
        let freq = r.freq_mhz.map(|v| v.to_string()).unwrap_or_default();
        let gain_amp_mean = r.gain_amp_mean.map(|v| v.to_string()).unwrap_or_default();
        let gain_amp_std = r.gain_amp_std.map(|v| v.to_string()).unwrap_or_default();
        let precision_p50 = r.precision_p50.map(|v| v.to_string()).unwrap_or_default();
        let precision_p90 = r.precision_p90.map(|v| v.to_string()).unwrap_or_default();
        writeln!(
            f,
            "{},{},{},{:.12},{},{:.12},{},{:.12},{},{},{},{}",
            r.chanblock,
            freq,
            r.failed_timeblocks,
            r.failed_timeblock_fraction,
            r.partial_timeblocks,
            r.partial_timeblock_fraction,
            r.nan_tile_cells,
            r.nan_tile_cell_fraction,
            gain_amp_mean,
            gain_amp_std,
            precision_p50,
            precision_p90
        )?;
    }
    Ok(())
}

fn write_tile_csv(path: &Path, rows: &[TileSummary]) -> Result<(), Box<dyn std::error::Error>> {
    let mut f = fs::File::create(path)?;
    writeln!(f, "tile,nan_cells,nan_cell_fraction")?;
    for r in rows {
        writeln!(f, "{},{},{:.12}", r.tile, r.nan_cells, r.nan_cell_fraction)?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args_os().skip(1);
    let input = args
        .next()
        .map(PathBuf::from)
        .ok_or("usage: inspect_solutions <solutions.fits> [output.json]")?;
    let output = args.next().map(PathBuf::from);

    let sols = CalibrationSolutions::read_solutions_from_ext(&input, None::<&Path>)?;
    let (num_timeblocks, num_tiles_total, num_chanblocks_total) = sols.di_jones.dim();

    let mut tile_is_flagged = vec![false; num_tiles_total];
    for &i in &sols.flagged_tiles {
        if i < num_tiles_total {
            tile_is_flagged[i] = true;
        }
    }
    let mut chanblock_is_flagged = vec![false; num_chanblocks_total];
    for &i in &sols.flagged_chanblocks {
        let i = i as usize;
        if i < num_chanblocks_total {
            chanblock_is_flagged[i] = true;
        }
    }

    let num_unflagged_tiles = tile_is_flagged.iter().filter(|&&f| !f).count();
    let num_unflagged_chanblocks = chanblock_is_flagged.iter().filter(|&&f| !f).count();

    let mut tile_nan_counts = vec![0usize; num_tiles_total];
    let mut timeblock_failed_counts = vec![0usize; num_timeblocks];
    let mut timeblock_partial_counts = vec![0usize; num_timeblocks];
    let mut timeblock_nan_cells = vec![0usize; num_timeblocks];
    let mut chanblock_failed_counts = vec![0usize; num_chanblocks_total];
    let mut chanblock_partial_counts = vec![0usize; num_chanblocks_total];
    let mut chanblock_nan_cells = vec![0usize; num_chanblocks_total];
    let mut chanblock_amp_stats = vec![RunningStats::default(); num_chanblocks_total];
    let mut overall_amp_stats = RunningStats::default();
    let mut example_non_nan_solution = None;

    let mut total_unflagged_cells = 0usize;
    let mut nan_cells = 0usize;
    let mut fully_failed_timeblock_chanblocks = 0usize;
    let mut partially_failed_timeblock_chanblocks = 0usize;
    let mut valid_solution_cells = 0usize;

    for t in 0..num_timeblocks {
        for c in 0..num_chanblocks_total {
            if chanblock_is_flagged[c] {
                continue;
            }

            let mut nan_in_this_chanblock = 0usize;
            for tile in 0..num_tiles_total {
                if tile_is_flagged[tile] {
                    continue;
                }
                total_unflagged_cells += 1;
                let j = sols.di_jones[(t, tile, c)];
                if j.any_nan() {
                    nan_cells += 1;
                    nan_in_this_chanblock += 1;
                    tile_nan_counts[tile] += 1;
                } else {
                    valid_solution_cells += 1;
                    if example_non_nan_solution.is_none() {
                        example_non_nan_solution = Some(j.to_float_array());
                    }
                    if let Some(amp) = scalar_gain_amp(j) {
                        chanblock_amp_stats[c].push(amp);
                        overall_amp_stats.push(amp);
                    }
                }
            }

            chanblock_nan_cells[c] += nan_in_this_chanblock;
            timeblock_nan_cells[t] += nan_in_this_chanblock;

            if nan_in_this_chanblock == num_unflagged_tiles {
                fully_failed_timeblock_chanblocks += 1;
                timeblock_failed_counts[t] += 1;
                chanblock_failed_counts[c] += 1;
            } else if nan_in_this_chanblock > 0 {
                partially_failed_timeblock_chanblocks += 1;
                timeblock_partial_counts[t] += 1;
                chanblock_partial_counts[c] += 1;
            }
        }
    }

    let overall = OverallSummary {
        num_timeblocks,
        num_tiles_total,
        num_chanblocks_total,
        num_flagged_tiles: sols.flagged_tiles.len(),
        num_flagged_chanblocks: sols.flagged_chanblocks.len(),
        num_unflagged_tiles,
        num_unflagged_chanblocks,
        total_unflagged_tile_chanblock_cells: total_unflagged_cells,
        nan_tile_chanblock_cells: nan_cells,
        nan_tile_chanblock_fraction: nan_cells as f64 / total_unflagged_cells.max(1) as f64,
        fully_failed_timeblock_chanblocks,
        fully_failed_timeblock_chanblock_fraction: fully_failed_timeblock_chanblocks as f64
            / (num_timeblocks * num_unflagged_chanblocks).max(1) as f64,
        partially_failed_timeblock_chanblocks,
        partially_failed_timeblock_chanblock_fraction: partially_failed_timeblock_chanblocks as f64
            / (num_timeblocks * num_unflagged_chanblocks).max(1) as f64,
        valid_solution_cells,
        example_valid_gain_amp_mean: overall_amp_stats.mean(),
    };

    let precision = sols.calibration_results.as_ref().map(|results| {
        let mut finite = results
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .collect::<Vec<_>>();
        finite.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let min_threshold = sols.min_threshold.unwrap_or(1e-4);
        let above_min_threshold_count = finite.iter().filter(|&&v| v > min_threshold).count();
        PrecisionSummary {
            finite_count: finite.len(),
            min: *finite.first().unwrap_or(&f64::NAN),
            p50: percentile(&finite, 0.50),
            p90: percentile(&finite, 0.90),
            p99: percentile(&finite, 0.99),
            max: *finite.last().unwrap_or(&f64::NAN),
            above_min_threshold_count,
            above_min_threshold_fraction: above_min_threshold_count as f64
                / finite.len().max(1) as f64,
        }
    });

    let all_timeblocks = (0..num_timeblocks)
        .map(|t| TimeblockSummary {
            timeblock: t,
            average_timestamp_unix_s: sols
                .average_timestamps
                .as_ref()
                .and_then(|v| v.get(t))
                .map(|e| e.to_unix_seconds()),
            failed_chanblocks: timeblock_failed_counts[t],
            failed_chanblock_fraction: timeblock_failed_counts[t] as f64
                / num_unflagged_chanblocks.max(1) as f64,
            partial_chanblocks: timeblock_partial_counts[t],
            partial_chanblock_fraction: timeblock_partial_counts[t] as f64
                / num_unflagged_chanblocks.max(1) as f64,
            nan_tile_cells: timeblock_nan_cells[t],
            nan_tile_cell_fraction: timeblock_nan_cells[t] as f64
                / (num_unflagged_tiles * num_unflagged_chanblocks).max(1) as f64,
        })
        .collect::<Vec<_>>();
    let worst_timeblocks = {
        let mut rows = all_timeblocks.clone();
        rows.sort_by(|a, b| {
            cmp_desc_f64(a.failed_chanblock_fraction, b.failed_chanblock_fraction)
                .then_with(|| cmp_desc_f64(a.nan_tile_cell_fraction, b.nan_tile_cell_fraction))
        });
        rows.truncate(12);
        rows
    };

    let all_chanblocks = (0..num_chanblocks_total)
        .filter(|&c| !chanblock_is_flagged[c])
        .map(|c| {
            let precision_pair = sols.calibration_results.as_ref().map(|results| {
                let mut vals = results
                    .column(c)
                    .iter()
                    .copied()
                    .filter(|v| v.is_finite())
                    .collect::<Vec<_>>();
                vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
                (percentile(&vals, 0.50), percentile(&vals, 0.90))
            });
            ChanblockSummary {
                chanblock: c,
                freq_mhz: sols
                    .chanblock_freqs
                    .as_ref()
                    .and_then(|v| v.get(c))
                    .map(|f| *f / 1e6),
                failed_timeblocks: chanblock_failed_counts[c],
                failed_timeblock_fraction: chanblock_failed_counts[c] as f64
                    / num_timeblocks.max(1) as f64,
                partial_timeblocks: chanblock_partial_counts[c],
                partial_timeblock_fraction: chanblock_partial_counts[c] as f64
                    / num_timeblocks.max(1) as f64,
                nan_tile_cells: chanblock_nan_cells[c],
                nan_tile_cell_fraction: chanblock_nan_cells[c] as f64
                    / (num_timeblocks * num_unflagged_tiles).max(1) as f64,
                gain_amp_mean: chanblock_amp_stats[c].mean(),
                gain_amp_std: chanblock_amp_stats[c].stddev(),
                precision_p50: precision_pair.map(|p| p.0),
                precision_p90: precision_pair.map(|p| p.1),
            }
        })
        .collect::<Vec<_>>();
    let worst_chanblocks = {
        let mut rows = all_chanblocks.clone();
        rows.sort_by(|a, b| {
            cmp_desc_f64(a.failed_timeblock_fraction, b.failed_timeblock_fraction)
                .then_with(|| cmp_desc_f64(a.nan_tile_cell_fraction, b.nan_tile_cell_fraction))
                .then_with(|| {
                    cmp_desc_f64(a.partial_timeblock_fraction, b.partial_timeblock_fraction)
                })
        });
        rows.truncate(20);
        rows
    };

    let all_tiles = {
        let denom = num_timeblocks * num_unflagged_chanblocks;
        (0..num_tiles_total)
            .filter(|&tile| !tile_is_flagged[tile])
            .map(|tile| TileSummary {
                tile,
                nan_cells: tile_nan_counts[tile],
                nan_cell_fraction: tile_nan_counts[tile] as f64 / denom.max(1) as f64,
            })
            .collect::<Vec<_>>()
    };
    let worst_tiles = {
        let mut rows = all_tiles.clone();
        rows.sort_by(|a, b| cmp_desc_f64(a.nan_cell_fraction, b.nan_cell_fraction));
        rows.truncate(20);
        rows
    };

    let summary = InspectSummary {
        file: input.display().to_string(),
        beam_file: sols.beam_file.as_ref().map(|p| p.display().to_string()),
        modeller: sols.modeller.clone(),
        uvw_min_m: sols.uvw_min,
        uvw_max_m: sols.uvw_max,
        freq_centroid_hz: sols.freq_centroid,
        stop_threshold: sols.stop_threshold,
        min_threshold: sols.min_threshold,
        overall,
        precision,
        worst_timeblocks,
        worst_chanblocks,
        worst_tiles,
        example_non_nan_solution,
    };

    let json = serde_json::to_string_pretty(&summary)?;
    println!("{json}");
    if let Some(output) = output {
        fs::write(&output, json)?;
        write_timeblock_csv(&csv_path(&output, "timeblocks"), &all_timeblocks)?;
        write_chanblock_csv(&csv_path(&output, "chanblocks"), &all_chanblocks)?;
        write_tile_csv(&csv_path(&output, "tiles"), &all_tiles)?;
    }

    Ok(())
}
