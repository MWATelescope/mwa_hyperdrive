// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#[cfg(test)]
mod tests;

mod ins;

use std::{
    collections::HashSet,
    path::{Path, PathBuf},
};

use birli::{
    corrections::ScrunchType,
    metrics::{AntennaMetadata, AutoMetrics, CrossMetrics, MetricsContext, EAVILS, SSINS},
    passband_gains::{OSPFB_JAKE_2025_200HZ, PFB_COTTER_2014_10KHZ, PFB_JAKE_2022_200HZ},
    FlagContext, VisSelection,
};
use clap::Parser;
use log::{debug, info, warn};
use marlu::{
    fitsio::FitsFile,
    hifitime::{Duration, Epoch},
    mwalib::{CorrelatorContext, MWAVersion},
    ndarray::{s, Array2, Array3, ArrayView1, ArrayView3, ArrayViewMut3, Axis},
    Jones, LatLngHeight, XyzGeodetic,
};
use serde::{Deserialize, Serialize};

use self::ins::{IncoherentNoiseSpectrum, InsTelescope};
use crate::cli::common::ARG_FILE_HELP;
use crate::cli::HyperdriveError;
use crate::io::read::{MsReader, UvfitsReader, VisRead};
use crate::math::TileBaselineFlags;

#[derive(Parser, Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct BmetricsArgs {
    #[clap(value_name = "ARGUMENTS_FILE", help = ARG_FILE_HELP.as_str())]
    args_file: Option<PathBuf>,

    /// Input data files:
    /// - Raw MWA data: provide metafits file followed by gpubox files.
    /// - UVFITS: provide a single UVFITS file.
    /// - Measurement Set: provide a single `.ms` directory.
    #[clap(short, long = "data", required = true, num_args(1..))]
    data: Vec<PathBuf>,

    /// Optional metafits file (required when using UVFITS input)
    #[clap(short = 'm', long = "metafits")]
    metafits: Option<PathBuf>,

    /// Output metrics FITS file path
    #[clap(short, long, required_unless_present = "ins")]
    output: Option<PathBuf>,

    /// Also write an SSINS incoherent noise spectrum of the cross-correlations
    /// to this HDF5 file, readable with `SSINS.INS(path)` (name it
    /// `<obsid>_SSINS_data.h5` to match SSINS's `INS.write`).
    #[clap(long)]
    ins: Option<PathBuf>,

    /// Before making the INS from raw data, remove the digital gains and the
    /// coarse-channel (PFB) passband, as pyuvdata does by default when SSINS
    /// reads raw files. This scales each channel's INS; z-scores are unchanged.
    #[clap(long)]
    ins_corrections: bool,

    /// Baseline length cutoff for cross metrics in metres
    #[clap(long, default_value = "100.0")]
    cross_metrics_cutoff: f32,
}

impl BmetricsArgs {
    pub(crate) fn merge(self) -> Result<BmetricsArgs, HyperdriveError> {
        let cli_args = self;

        if let Some(arg_file) = cli_args.args_file {
            let BmetricsArgs {
                args_file: _,
                data,
                metafits,
                output,
                ins,
                ins_corrections,
                cross_metrics_cutoff: _,
            } = unpack_arg_file!(arg_file);

            Ok(BmetricsArgs {
                args_file: None,
                data: if cli_args.data.is_empty() {
                    data
                } else {
                    cli_args.data
                },
                metafits: cli_args.metafits.or(metafits),
                output: cli_args.output.or(output),
                ins: cli_args.ins.or(ins),
                ins_corrections: cli_args.ins_corrections || ins_corrections,
                cross_metrics_cutoff: cli_args.cross_metrics_cutoff,
            })
        } else {
            Ok(cli_args)
        }
    }

    pub(crate) fn run(self, dry_run: bool) -> Result<(), HyperdriveError> {
        if dry_run {
            info!(
                "Dry run: would calculate metrics and write to {:?} (INS: {:?})",
                self.output, self.ins
            );
            return Ok(());
        }

        if self.output.is_none() && self.ins.is_none() {
            return Err(HyperdriveError::Generic(
                "Nothing to do: specify --output and/or --ins".to_string(),
            ));
        }
        if self.data.is_empty() {
            return Err(HyperdriveError::Generic(
                "No input data files specified".to_string(),
            ));
        }

        // Detect input type from the first path.
        let first_path = &self.data[0];
        let first_file_str = first_path.to_str().unwrap_or("");

        // UVFITS (single file)
        if first_file_str.ends_with(".uvfits") || first_file_str.contains(".uvfits.") {
            if self.data.len() > 1 {
                return Err(HyperdriveError::Generic(
                    "UVFITS input should be a single file".to_string(),
                ));
            }
            info!("Reading UVFITS file: {:?}", first_path);
            warn!("Receiver metadata (type, slot, cable) is not available from UVFITS");
            let reader = UvfitsReader::new(first_path.clone(), self.metafits.as_deref(), None)
                .map_err(|e| HyperdriveError::VisRead(e.to_string()))?;
            return self.run_from_reader(&reader);
        }

        // Measurement Set (directory ending in .ms)
        if first_file_str.ends_with(".ms") {
            if self.data.len() > 1 {
                return Err(HyperdriveError::Generic(
                    "Measurement set input should be a single .ms directory".to_string(),
                ));
            }
            info!("Reading measurement set: {:?}", first_path);
            warn!("Receiver metadata (type, slot, cable) may not be available from MS");
            let reader = MsReader::new(first_path.clone(), None, self.metafits.as_deref(), None)
                .map_err(|e| HyperdriveError::VisRead(e.to_string()))?;
            return self.run_from_reader(&reader);
        }

        if self.data.len() < 2 {
            return Err(HyperdriveError::Generic(
                "Input requires at least metafits and one gpubox file".to_string(),
            ));
        }

        let metafits = &self.data[0];
        let gpuboxes: Vec<PathBuf> = self.data[1..].to_vec();

        info!(
            "Reading MWA data: {:?} + {} gpubox files",
            metafits,
            gpuboxes.len()
        );

        let corr_ctx = CorrelatorContext::new(metafits, &gpuboxes).map_err(|e| {
            HyperdriveError::Generic(format!("Failed to create CorrelatorContext: {}", e))
        })?;

        let vis_sel = VisSelection::from_mwalib(&corr_ctx).map_err(|e| {
            HyperdriveError::Generic(format!("Failed to create VisSelection: {}", e))
        })?;

        let flag_ctx = FlagContext::from_mwalib(&corr_ctx);

        let fine_chans_per_coarse = corr_ctx.metafits_context.num_corr_fine_chans_per_coarse;
        let mut flag_array = vis_sel
            .allocate_flags(fine_chans_per_coarse)
            .map_err(|e| HyperdriveError::Generic(format!("Failed to allocate flags: {}", e)))?;
        let mut jones_array = vis_sel.allocate_jones(fine_chans_per_coarse).map_err(|e| {
            HyperdriveError::Generic(format!("Failed to allocate jones array: {}", e))
        })?;

        // These flags are only used by the INS, which, like SSINS, leaves out
        // flagged tiles and missing data but not the DC channel. The Birli
        // metrics take their channel flags from `flag_ctx`.
        FlagContext {
            flag_dc: false,
            ..flag_ctx.clone()
        }
        .set_flags(
            flag_array.view_mut(),
            &vis_sel.timestep_range,
            &vis_sel.coarse_chan_range,
            &vis_sel.get_ant_pairs(&corr_ctx.metafits_context),
        )
        .map_err(|e| HyperdriveError::Generic(format!("Failed to set flags: {}", e)))?;

        info!("Reading visibilities...");
        birli::io::read_mwalib(
            &vis_sel,
            &corr_ctx,
            jones_array.view_mut(),
            flag_array.view_mut(),
            false,
        )
        .map_err(|e| HyperdriveError::Generic(format!("Failed to read visibilities: {}", e)))?;

        if self.output.is_some() {
            self.calculate_and_save_metrics(
                jones_array.view(),
                flag_array.view(),
                &corr_ctx,
                &vis_sel,
                &flag_ctx,
            )?;
        }

        if let Some(ins_path) = &self.ins {
            let metafits = &corr_ctx.metafits_context;
            let ant_pairs = vis_sel.get_ant_pairs(metafits);
            if self.ins_corrections {
                // After the Birli metrics, which use the uncorrected data.
                correct_like_pyuvdata(&corr_ctx, &vis_sel, &ant_pairs, jones_array.view_mut())?;
            }
            let half_int = metafits.corr_int_time_ms / 2;
            let timestamps: Vec<Epoch> = corr_ctx.timesteps[vis_sel.timestep_range.clone()]
                .iter()
                .map(|ts| Epoch::from_gpst_seconds((ts.gps_time_ms + half_int) as f64 / 1e3))
                .collect();
            let freqs_hz = corr_ctx.get_fine_chan_freqs_hz_array(
                &vis_sel.coarse_chan_range.clone().collect::<Vec<_>>(),
            );
            let array_position = LatLngHeight::mwa();
            let telescope = InsTelescope {
                name: "MWA".to_string(),
                array_position,
                dut1: Duration::from_seconds(metafits.dut1.unwrap_or(0.0)),
                antenna_names: metafits
                    .antennas
                    .iter()
                    .map(|a| a.tile_name.clone())
                    .collect(),
                antenna_numbers: metafits.antennas.iter().map(|a| a.tile_id as i32).collect(),
                antenna_xyzs: XyzGeodetic::get_tiles(metafits, array_position.latitude_rad),
            };
            write_ins(
                ins_path,
                jones_array.view(),
                flag_array.view(),
                &ant_pairs,
                &timestamps,
                &freqs_hz,
                f64::from(metafits.corr_fine_chan_width_hz),
                &telescope,
            )?;
        }

        Ok(())
    }

    /// Calculate metrics from a uvfits file or measurement set.
    fn run_from_reader(&self, reader: &dyn VisRead) -> Result<(), HyperdriveError> {
        let obs_context = reader.get_obs_context();
        let num_timesteps = obs_context.all_timesteps.len();
        let num_freqs = obs_context.fine_chan_freqs.len();
        let num_tiles = obs_context.tile_names.len();
        let num_cross_baselines = num_tiles * (num_tiles - 1) / 2;
        let num_baselines = num_tiles * (num_tiles + 1) / 2;

        info!(
            "Input contains {} antennas, {} timesteps, {} channels",
            num_tiles, num_timesteps, num_freqs
        );

        // Storage in (time, freq, baseline) with baseline axis in upper-triangular
        // (including autos) order: (0,0),(0,1)...(0,n-1),(1,1),(1,2)...
        let mut jones_array =
            Array3::<Jones<f32>>::zeros((num_timesteps, num_freqs, num_baselines));
        let mut flag_array = Array3::from_elem(jones_array.dim(), false);
        let mut antenna_pairs = Vec::with_capacity(num_baselines);
        for i in 0..num_tiles {
            for j in i..num_tiles {
                antenna_pairs.push((i, j));
            }
        }

        // Read every tile; flagged tiles are flagged below.
        let tile_baseline_flags = TileBaselineFlags::new(num_tiles, HashSet::new());

        let mut cross_vis_fb = Array2::<Jones<f32>>::zeros((num_freqs, num_cross_baselines));
        let mut cross_weights_fb = Array2::<f32>::zeros((num_freqs, num_cross_baselines));
        let mut auto_vis_fb = Array2::<Jones<f32>>::zeros((num_freqs, num_tiles));
        let mut auto_weights_fb = Array2::<f32>::zeros((num_freqs, num_tiles));

        info!("Reading visibilities...");
        for (t_idx, &timestep) in obs_context.all_timesteps.iter().enumerate() {
            // Clear buffers (reader adds into existing arrays in some paths).
            cross_vis_fb.fill(Jones::default());
            cross_weights_fb.fill(0.0);
            auto_vis_fb.fill(Jones::default());
            auto_weights_fb.fill(0.0);

            if obs_context.autocorrelations_present {
                reader.read_crosses_and_autos(
                    cross_vis_fb.view_mut(),
                    cross_weights_fb.view_mut(),
                    auto_vis_fb.view_mut(),
                    auto_weights_fb.view_mut(),
                    timestep,
                    &tile_baseline_flags,
                    &HashSet::new(),
                )?;
            } else {
                reader.read_crosses(
                    cross_vis_fb.view_mut(),
                    cross_weights_fb.view_mut(),
                    timestep,
                    &tile_baseline_flags,
                    &HashSet::new(),
                )?;
            }

            let mut put = |b: usize, vis: ArrayView1<Jones<f32>>, weights: ArrayView1<f32>| {
                jones_array.slice_mut(s![t_idx, .., b]).assign(&vis);
                flag_array
                    .slice_mut(s![t_idx, .., b])
                    .zip_mut_with(&weights, |f, &w| *f = w <= 0.0);
            };
            for ant in 0..num_tiles {
                put(
                    bl_index(ant, ant, num_tiles),
                    auto_vis_fb.column(ant),
                    auto_weights_fb.column(ant),
                );
            }
            for bl in 0..num_cross_baselines {
                let (i, j) = marlu::math::cross_correlation_baseline_to_tiles(num_tiles, bl);
                put(
                    bl_index(i, j, num_tiles),
                    cross_vis_fb.column(bl),
                    cross_weights_fb.column(bl),
                );
            }
        }
        // Flagged tiles don't contribute, like pyuvdata's default
        // remove_flagged_ants.
        for (b, &(i, j)) in antenna_pairs.iter().enumerate() {
            if obs_context.flagged_tiles.contains(&i) || obs_context.flagged_tiles.contains(&j) {
                flag_array.slice_mut(s![.., .., b]).fill(true);
            }
        }

        let fine_chan_freqs_hz: Vec<f64> = obs_context
            .fine_chan_freqs
            .iter()
            .map(|&f| f as f64)
            .collect();
        let timestamps: Vec<Epoch> = obs_context
            .all_timesteps
            .iter()
            .map(|&t| obs_context.timestamps[t])
            .collect();

        if let Some(ins_path) = &self.ins {
            let channel_width_hz = obs_context.freq_res.unwrap_or_else(|| {
                fine_chan_freqs_hz
                    .get(1)
                    .map(|f1| f1 - fine_chan_freqs_hz[0])
                    .unwrap_or(1.0)
            });
            let telescope = InsTelescope {
                name: "MWA".to_string(),
                array_position: obs_context.array_position,
                dut1: obs_context.dut1.unwrap_or_default(),
                antenna_names: obs_context.tile_names.to_vec(),
                antenna_numbers: (0..num_tiles as i32).collect(),
                antenna_xyzs: obs_context.tile_xyzs.to_vec(),
            };
            write_ins(
                ins_path,
                jones_array.view(),
                flag_array.view(),
                &antenna_pairs,
                &timestamps,
                &fine_chan_freqs_hz,
                channel_width_hz,
                &telescope,
            )?;
        }

        if self.output.is_none() {
            return Ok(());
        }

        let latitude_rad = obs_context.array_position.latitude_rad;
        let antennas: Vec<AntennaMetadata> = obs_context
            .tile_xyzs
            .iter()
            .zip(obs_context.tile_names.iter())
            .enumerate()
            .map(|(idx, (xyz, name))| {
                let enh = xyz.to_enh(latitude_rad);
                AntennaMetadata {
                    tile_name: name.clone(),
                    tile_id: idx as u32 + 1,
                    ant_id: idx as u32,
                    east_m: enh.e,
                    north_m: enh.n,
                    height_m: enh.h,
                    // Receiver info generally isn't stored in a way we can rely on.
                    rec_number: 0,
                    rec_slot_number: 0,
                    rec_type: "Unknown".to_string(),
                    cable_flavour: "Unknown".to_string(),
                    has_whitening_filter: false,
                }
            })
            .collect();

        let metrics_ctx = MetricsContext {
            antennas,
            fine_chan_freqs_hz,
            timestamps_s: timestamps.iter().map(|e| e.to_gpst_seconds()).collect(),
            antenna_pairs,
        };

        // For now, treat everything as unflagged (birli metrics uses these for bookkeeping).
        let timestep_flags = vec![false; num_timesteps];
        let chan_flags = vec![false; num_freqs];

        info!("Calculating metrics...");
        self.calculate_and_save_metrics_from_context(
            jones_array.view(),
            flag_array.view(),
            &metrics_ctx,
            &timestep_flags,
            &chan_flags,
        )
    }

    fn calculate_and_save_metrics(
        &self,
        jones_array: marlu::ndarray::ArrayView3<marlu::Jones<f32>>,
        flag_array: marlu::ndarray::ArrayView3<bool>,
        corr_ctx: &CorrelatorContext,
        vis_sel: &VisSelection,
        flag_ctx: &FlagContext,
    ) -> Result<(), HyperdriveError> {
        let mut fptr = self.create_output()?;

        info!("Calculating AutoMetrics...");
        let auto_metrics = AutoMetrics::new(jones_array, corr_ctx, vis_sel, flag_ctx);
        auto_metrics
            .save_to_fits(&mut fptr)
            .map_err(|e| HyperdriveError::Generic(format!("Failed to save AutoMetrics: {}", e)))?;

        // SSINS requires at least 2 timesteps
        let num_timesteps = jones_array.len_of(Axis(0));
        if num_timesteps >= 2 {
            info!("Calculating SSINS...");
            let ssins = SSINS::new(jones_array, flag_array, corr_ctx, vis_sel, flag_ctx);
            ssins
                .save_to_fits(&mut fptr)
                .map_err(|e| HyperdriveError::Generic(format!("Failed to save SSINS: {}", e)))?;
        } else {
            warn!("Skipping SSINS (requires at least 2 timesteps, have {num_timesteps})");
        }

        info!("Calculating EAVILS...");
        let eavils = EAVILS::new(jones_array, corr_ctx, vis_sel, flag_ctx);
        eavils
            .save_to_fits(&mut fptr)
            .map_err(|e| HyperdriveError::Generic(format!("Failed to save EAVILS: {}", e)))?;

        info!(
            "Calculating CrossMetrics (baseline cutoff: {} m)...",
            self.cross_metrics_cutoff
        );
        let cross_metrics = CrossMetrics::new(
            jones_array,
            corr_ctx,
            vis_sel,
            flag_ctx,
            self.cross_metrics_cutoff,
        );
        cross_metrics
            .save_to_fits(&mut fptr)
            .map_err(|e| HyperdriveError::Generic(format!("Failed to save CrossMetrics: {}", e)))?;

        info!("Metrics written to {:?}", self.output);
        Ok(())
    }

    fn create_output(&self) -> Result<FitsFile, HyperdriveError> {
        let output = self
            .output
            .as_ref()
            .expect("only called when there's an output");
        if output.exists() {
            std::fs::remove_file(output).map_err(|e| {
                HyperdriveError::Generic(format!("Failed to remove existing output file: {}", e))
            })?;
        }
        FitsFile::create(output)
            .open()
            .map_err(|e| HyperdriveError::Generic(format!("Failed to create FITS file: {}", e)))
    }

    fn calculate_and_save_metrics_from_context(
        &self,
        jones_array: ArrayView3<Jones<f32>>,
        flag_array: ArrayView3<bool>,
        metrics_ctx: &MetricsContext,
        timestep_flags: &[bool],
        chan_flags: &[bool],
    ) -> Result<(), HyperdriveError> {
        let mut fptr = self.create_output()?;

        info!("Calculating AutoMetrics...");
        let auto_metrics =
            AutoMetrics::new_from_metadata(jones_array, metrics_ctx, timestep_flags, chan_flags);
        auto_metrics
            .save_to_fits(&mut fptr)
            .map_err(|e| HyperdriveError::Generic(format!("Failed to save AutoMetrics: {}", e)))?;

        // SSINS requires at least 2 timesteps
        if timestep_flags.len() >= 2 {
            info!("Calculating SSINS...");
            let ssins = SSINS::new_from_metadata(
                jones_array,
                flag_array,
                metrics_ctx,
                timestep_flags,
                chan_flags,
            );
            ssins
                .save_to_fits(&mut fptr)
                .map_err(|e| HyperdriveError::Generic(format!("Failed to save SSINS: {}", e)))?;
        } else {
            warn!(
                "Skipping SSINS (requires at least 2 timesteps, have {})",
                timestep_flags.len()
            );
        }

        info!("Calculating EAVILS...");
        let eavils =
            EAVILS::new_from_metadata(jones_array, metrics_ctx, timestep_flags, chan_flags);
        eavils
            .save_to_fits(&mut fptr)
            .map_err(|e| HyperdriveError::Generic(format!("Failed to save EAVILS: {}", e)))?;

        info!(
            "Calculating CrossMetrics (baseline cutoff: {} m)...",
            self.cross_metrics_cutoff
        );
        let cross_metrics = CrossMetrics::new_from_metadata(
            jones_array,
            metrics_ctx,
            timestep_flags,
            chan_flags,
            self.cross_metrics_cutoff,
        );
        cross_metrics
            .save_to_fits(&mut fptr)
            .map_err(|e| HyperdriveError::Generic(format!("Failed to save CrossMetrics: {}", e)))?;

        info!("Metrics written to {:?}", self.output);
        Ok(())
    }
}

/// Index of baseline (i, j), i <= j, in the packed upper triangle (with autos).
fn bl_index(i: usize, j: usize, n: usize) -> usize {
    debug_assert!(i <= j);
    let tri = if i == 0 { 0 } else { (i * (i - 1)) / 2 };
    i * n - tri + (j - i)
}

#[allow(clippy::too_many_arguments)]
fn write_ins(
    path: &Path,
    vis_tfb: ArrayView3<Jones<f32>>,
    flags_tfb: ArrayView3<bool>,
    ant_pairs: &[(usize, usize)],
    timestamps: &[Epoch],
    freqs_hz: &[f64],
    channel_width_hz: f64,
    telescope: &InsTelescope,
) -> Result<(), HyperdriveError> {
    if timestamps.len() < 2 {
        return Err(HyperdriveError::Generic(format!(
            "An INS needs at least 2 timesteps, have {}",
            timestamps.len()
        )));
    }
    info!("Calculating the incoherent noise spectrum...");
    let ins = IncoherentNoiseSpectrum::new(vis_tfb, flags_tfb, ant_pairs, timestamps);
    if path.exists() {
        std::fs::remove_file(path).map_err(|e| {
            HyperdriveError::Generic(format!("Failed to remove existing INS file: {e}"))
        })?;
    }
    ins.write_uvflag_h5(path, freqs_hz, channel_width_hz, telescope)
        .map_err(|e| HyperdriveError::Generic(format!("Failed to write INS to {path:?}: {e}")))?;
    info!("INS written to {path:?}");
    Ok(())
}

/// Remove the digital gains and the PFB passband from raw visibilities, like
/// pyuvdata's `remove_dig_gains` and `remove_coarse_band` (with Birli's
/// passband shapes).
fn correct_like_pyuvdata(
    corr_ctx: &CorrelatorContext,
    vis_sel: &VisSelection,
    ant_pairs: &[(usize, usize)],
    mut jones_array: ArrayViewMut3<Jones<f32>>,
) -> Result<(), HyperdriveError> {
    let metafits = &corr_ctx.metafits_context;
    birli::corrections::correct_digital_gains(
        corr_ctx,
        jones_array.view_mut(),
        &vis_sel.coarse_chan_range,
        ant_pairs,
    )
    .map_err(|e| HyperdriveError::Generic(format!("Failed to correct digital gains: {e}")))?;

    let gains = match (corr_ctx.mwa_version, metafits.oversampled) {
        _ if metafits.deripple_applied => return Ok(()),
        (MWAVersion::CorrMWAXv2, false) => PFB_JAKE_2022_200HZ,
        (MWAVersion::CorrMWAXv2, true) => OSPFB_JAKE_2025_200HZ,
        _ => PFB_COTTER_2014_10KHZ,
    };
    let scrunch = ScrunchType::from_mwa_version(corr_ctx.mwa_version)
        .map_err(|e| HyperdriveError::Generic(e.to_string()))?;
    // The weights aren't used.
    let mut weights = Array3::<f32>::ones(jones_array.dim());
    birli::corrections::correct_coarse_passband_gains(
        jones_array,
        weights.view_mut(),
        gains,
        metafits.num_corr_fine_chans_per_coarse,
        &scrunch,
    )
    .map_err(|e| HyperdriveError::Generic(format!("Failed to correct the passband: {e}")))
}
