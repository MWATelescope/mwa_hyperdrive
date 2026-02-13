// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::{collections::HashSet, path::PathBuf};

use birli::{
    metrics::{AntennaMetadata, AutoMetrics, CrossMetrics, MetricsContext, EAVILS, SSINS},
    FlagContext, VisSelection,
};
use clap::Parser;
use log::{debug, info, warn};
use marlu::{
    fitsio::FitsFile,
    mwalib::CorrelatorContext,
    ndarray::{s, Array2, Array3, ArrayView3},
    Jones,
};
use serde::{Deserialize, Serialize};

use crate::cli::common::ARG_FILE_HELP;
use crate::cli::HyperdriveError;
use crate::io::read::{CrossData, MsReader, UvfitsReader, VisRead};
use crate::math::TileBaselineFlags;

#[derive(Parser, Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct BmetricsArgs {
    #[clap(name = "ARGUMENTS_FILE", help = ARG_FILE_HELP.as_str(), parse(from_os_str))]
    args_file: Option<PathBuf>,

    /// Input data files:
    /// - Raw MWA data: provide metafits file followed by gpubox files.
    /// - UVFITS: provide a single UVFITS file.
    /// - Measurement Set: provide a single `.ms` directory.
    #[clap(short, long = "data", required = true, multiple_values = true)]
    data: Vec<PathBuf>,

    /// Optional metafits file (required when using UVFITS input)
    #[clap(short = 'm', long = "metafits")]
    metafits: Option<PathBuf>,

    /// Output metrics FITS file path
    #[clap(short, long, required = true)]
    output: PathBuf,

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
                output: _,
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
                output: cli_args.output,
                cross_metrics_cutoff: cli_args.cross_metrics_cutoff,
            })
        } else {
            Ok(cli_args)
        }
    }

    pub(crate) fn run(self, dry_run: bool) -> Result<(), HyperdriveError> {
        if dry_run {
            info!(
                "Dry run: would calculate metrics and write to {:?}",
                self.output
            );
            return Ok(());
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
            return self.run_from_uvfits(first_path);
        }

        // Measurement Set (directory ending in .ms)
        if first_file_str.ends_with(".ms") {
            if self.data.len() > 1 {
                return Err(HyperdriveError::Generic(
                    "Measurement set input should be a single .ms directory".to_string(),
                ));
            }
            info!("Reading measurement set: {:?}", first_path);
            return self.run_from_ms(first_path);
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

        flag_ctx
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

        self.calculate_and_save_metrics(
            jones_array.view(),
            flag_array.view(),
            &corr_ctx,
            &vis_sel,
            &flag_ctx,
        )
    }

    fn run_from_ms(&self, ms_path: &PathBuf) -> Result<(), HyperdriveError> {
        warn!("========================================================================");
        warn!("Reading measurement set for metrics calculation");
        warn!("Note: Receiver metadata (type, slot, cable) may not be available from MS");
        warn!("========================================================================");

        let reader = MsReader::new(ms_path.clone(), None, self.metafits.as_deref(), None)
            .map_err(|e| HyperdriveError::VisRead(e.to_string()))?;

        let obs_context = reader.get_obs_context();
        let num_timesteps = obs_context.all_timesteps.len();
        let num_freqs = obs_context.fine_chan_freqs.len();
        let num_tiles = obs_context.tile_names.len();
        let num_cross_baselines = num_tiles * (num_tiles - 1) / 2;
        let num_baselines = num_tiles * (num_tiles + 1) / 2;

        info!(
            "MS contains {} antennas, {} timesteps, {} channels",
            num_tiles, num_timesteps, num_freqs
        );

        // Storage in (time, freq, baseline) with baseline axis in upper-triangular
        // (including autos) order: (0,0),(0,1)...(0,n-1),(1,1),(1,2)...
        let mut jones_array =
            Array3::<Jones<f32>>::zeros((num_timesteps, num_freqs, num_baselines));

        // Build a TileBaselineFlags with no tiles flagged. (The MS reader will still
        // apply any per-row/per-channel flags when populating weights/flags.)
        let tile_baseline_flags = TileBaselineFlags::new(num_tiles, HashSet::new());

        // Temporary buffers for each timestep.
        let mut cross_vis_fb = Array2::<Jones<f32>>::zeros((num_freqs, num_cross_baselines));
        let mut cross_weights_fb = Array2::<f32>::zeros((num_freqs, num_cross_baselines));
        let mut auto_vis_fb = Array2::<Jones<f32>>::zeros((num_freqs, num_tiles));
        let mut auto_weights_fb = Array2::<f32>::zeros((num_freqs, num_tiles));

        // Helper: baseline index for (i,j) with i<=j in the packed upper triangle.
        #[inline]
        fn bl_index(i: usize, j: usize, n: usize) -> usize {
            debug_assert!(i <= j);
            let tri = if i == 0 { 0 } else { (i * (i - 1)) / 2 };
            i * n - tri + (j - i)
        }

        info!("Reading visibility data from measurement set...");
        for t_idx in 0..num_timesteps {
            // Clear buffers (reader adds into existing arrays in some paths).
            cross_vis_fb.fill(Jones::default());
            cross_weights_fb.fill(0.0);
            auto_vis_fb.fill(Jones::default());
            auto_weights_fb.fill(0.0);

            reader.read_crosses_and_autos(
                cross_vis_fb.view_mut(),
                cross_weights_fb.view_mut(),
                auto_vis_fb.view_mut(),
                auto_weights_fb.view_mut(),
                t_idx,
                &tile_baseline_flags,
                &HashSet::new(),
            )?;

            // Autos
            for ant in 0..num_tiles {
                let b = bl_index(ant, ant, num_tiles);
                jones_array
                    .slice_mut(s![t_idx, .., b])
                    .assign(&auto_vis_fb.slice(s![.., ant]));
            }

            // Crosses
            for bl in 0..num_cross_baselines {
                let (i, j) = marlu::math::cross_correlation_baseline_to_tiles(num_tiles, bl);
                let b = bl_index(i, j, num_tiles);
                jones_array
                    .slice_mut(s![t_idx, .., b])
                    .assign(&cross_vis_fb.slice(s![.., bl]));
            }
        }

        // Build MetricsContext from MS metadata.
        let antennas: Vec<AntennaMetadata> = obs_context
            .tile_xyzs
            .iter()
            .zip(obs_context.tile_names.iter())
            .enumerate()
            .map(|(idx, (xyz, name))| AntennaMetadata {
                tile_name: name.clone(),
                tile_id: idx as u32 + 1,
                ant_id: idx as u32,
                east_m: xyz.x as f64,
                north_m: xyz.y as f64,
                height_m: xyz.z as f64,
                // Receiver info generally isn't stored in MS in a way we can rely on.
                rec_number: 0,
                rec_slot_number: 0,
                rec_type: "Unknown".to_string(),
                cable_flavour: "Unknown".to_string(),
                has_whitening_filter: false,
            })
            .collect();

        let fine_chan_freqs_hz: Vec<f64> = obs_context
            .fine_chan_freqs
            .iter()
            .map(|&f| f as f64)
            .collect();

        let timestamps_s: Vec<f64> = obs_context
            .timestamps
            .iter()
            .map(|epoch| epoch.to_gpst_seconds())
            .collect();

        let mut antenna_pairs = Vec::with_capacity(num_baselines);
        for i in 0..num_tiles {
            for j in i..num_tiles {
                antenna_pairs.push((i, j));
            }
        }

        let metrics_ctx = MetricsContext {
            antennas,
            fine_chan_freqs_hz,
            timestamps_s,
            antenna_pairs,
        };

        // For now, treat everything as unflagged (birli metrics uses these for bookkeeping).
        let timestep_flags = vec![false; num_timesteps];
        let chan_flags = vec![false; num_freqs];

        info!("Calculating metrics from measurement set data...");
        self.calculate_and_save_metrics_from_context(
            jones_array.view(),
            &metrics_ctx,
            &timestep_flags,
            &chan_flags,
        )
    }

    fn run_from_uvfits(&self, uvfits_path: &PathBuf) -> Result<(), HyperdriveError> {
        warn!("========================================================================");
        warn!("Reading UVFITS file for metrics calculation");
        warn!("Note: Receiver metadata (type, slot, cable) not available from UVFITS");
        warn!("For complete metadata, use raw MWA data (metafits + gpubox files)");
        warn!("========================================================================");

        // Read UVFITS using hyperdrive's reader
        let reader = UvfitsReader::new(uvfits_path.clone(), None, None)
            .map_err(|e| HyperdriveError::VisRead(e.to_string()))?;
        let obs_context = reader.get_obs_context();

        info!(
            "UVFITS contains {} antennas, {} timesteps, {} channels",
            obs_context.tile_names.len(),
            obs_context.all_timesteps.len(),
            obs_context.fine_chan_freqs.len()
        );

        // Read all visibility data
        let num_timesteps = obs_context.all_timesteps.len();
        let num_freqs = obs_context.fine_chan_freqs.len();
        let num_tiles = obs_context.tile_names.len();
        let num_baselines = num_tiles * (num_tiles + 1) / 2;

        let mut jones_array =
            Array3::<Jones<f32>>::zeros((num_timesteps, num_freqs, num_baselines));
        let mut weights_array = Array2::<f32>::zeros((num_freqs, num_baselines));

        info!("Reading visibility data from UVFITS...");

        // Create TileBaselineFlags (all unflagged for now)
        let tile_baseline_flags = TileBaselineFlags::new(num_tiles, HashSet::new());

        // Read data for each timestep
        for (t_idx, _timeblock) in obs_context.all_timesteps.iter().enumerate() {
            let cross_data = CrossData {
                vis_fb: jones_array.slice_mut(s![t_idx, .., ..]),
                weights_fb: weights_array.view_mut(),
                tile_baseline_flags: &tile_baseline_flags,
            };

            reader.read_inner_dispatch(Some(cross_data), None, t_idx, &HashSet::new())?;
        }

        // Create MetricsContext from UVFITS metadata
        let antennas: Vec<AntennaMetadata> = obs_context
            .tile_xyzs
            .iter()
            .zip(obs_context.tile_names.iter())
            .enumerate()
            .map(|(idx, (_xyz, name))| AntennaMetadata {
                tile_name: name.clone(),
                tile_id: idx as u32 + 1,
                ant_id: idx as u32,
                // XyzGeodetic has x, y, z fields (east, north, up in meters)
                east_m: 0.0,
                north_m: 0.0,
                height_m: 0.0,
                // Receiver info unavailable from UVFITS - use defaults
                rec_number: 0,
                rec_slot_number: 0,
                rec_type: "Unknown".to_string(),
                cable_flavour: "Unknown".to_string(),
                has_whitening_filter: false,
            })
            .collect();

        let fine_chan_freqs_hz: Vec<f64> = obs_context
            .fine_chan_freqs
            .iter()
            .map(|&f| f as f64)
            .collect();

        let timestamps_s: Vec<f64> = obs_context
            .timestamps
            .iter()
            .map(|epoch| epoch.to_gpst_seconds())
            .collect();

        // Create antenna pairs (all baselines including autos)
        let mut antenna_pairs = Vec::new();
        for i in 0..num_tiles {
            for j in i..num_tiles {
                antenna_pairs.push((i, j));
            }
        }

        let metrics_ctx = MetricsContext {
            antennas,
            fine_chan_freqs_hz,
            timestamps_s,
            antenna_pairs,
        };

        // Create flag arrays (all unflagged for now)
        let timestep_flags = vec![false; num_timesteps];
        let chan_flags = vec![false; num_freqs];

        info!("Calculating metrics from UVFITS data...");

        // Calculate and save metrics using the new MetricsContext
        self.calculate_and_save_metrics_from_context(
            jones_array.view(),
            &metrics_ctx,
            &timestep_flags,
            &chan_flags,
        )
    }

    fn calculate_and_save_metrics(
        &self,
        jones_array: marlu::ndarray::ArrayView3<marlu::Jones<f32>>,
        _flag_array: marlu::ndarray::ArrayView3<bool>,
        corr_ctx: &CorrelatorContext,
        vis_sel: &VisSelection,
        flag_ctx: &FlagContext,
    ) -> Result<(), HyperdriveError> {
        if self.output.exists() {
            std::fs::remove_file(&self.output).map_err(|e| {
                HyperdriveError::Generic(format!("Failed to remove existing output file: {}", e))
            })?;
        }

        let mut fptr = FitsFile::create(&self.output)
            .open()
            .map_err(|e| HyperdriveError::Generic(format!("Failed to create FITS file: {}", e)))?;

        info!("Calculating AutoMetrics...");
        let auto_metrics = AutoMetrics::new(jones_array, corr_ctx, vis_sel, flag_ctx);
        auto_metrics
            .save_to_fits(&mut fptr)
            .map_err(|e| HyperdriveError::Generic(format!("Failed to save AutoMetrics: {}", e)))?;

        info!("Calculating SSINS...");
        let ssins = SSINS::new(jones_array, corr_ctx, vis_sel, flag_ctx);
        ssins
            .save_to_fits(&mut fptr)
            .map_err(|e| HyperdriveError::Generic(format!("Failed to save SSINS: {}", e)))?;

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

    #[allow(dead_code)]
    fn calculate_and_save_metrics_from_context(
        &self,
        jones_array: ArrayView3<Jones<f32>>,
        metrics_ctx: &MetricsContext,
        timestep_flags: &[bool],
        chan_flags: &[bool],
    ) -> Result<(), HyperdriveError> {
        if self.output.exists() {
            std::fs::remove_file(&self.output).map_err(|e| {
                HyperdriveError::Generic(format!("Failed to remove existing output file: {}", e))
            })?;
        }

        let mut fptr = FitsFile::create(&self.output)
            .open()
            .map_err(|e| HyperdriveError::Generic(format!("Failed to create FITS file: {}", e)))?;

        info!("Calculating AutoMetrics...");
        let auto_metrics =
            AutoMetrics::new_from_metadata(jones_array, metrics_ctx, timestep_flags, chan_flags);
        auto_metrics
            .save_to_fits(&mut fptr)
            .map_err(|e| HyperdriveError::Generic(format!("Failed to save AutoMetrics: {}", e)))?;

        // SSINS requires at least 2 timesteps
        if timestep_flags.len() >= 2 {
            info!("Calculating SSINS...");
            let ssins =
                SSINS::new_from_metadata(jones_array, metrics_ctx, timestep_flags, chan_flags);
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
