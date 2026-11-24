// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to write visibilities into a column (e.g. `CORRECTED_DATA`) of a
//! measurement set that is being read.
//!
//! The rows of the measurement set are not changed; visibilities are put in
//! the rows they were read from. This means that there can be no averaging
//! between what is read and what is written.

use std::collections::HashSet;

use marlu::{c32, rubbl_casatables::GlueDataType, Jones};
use ndarray::prelude::*;

use super::{MsReadError, MsReader, ReadState, StagedRows, Table, TableOpenMode};
use crate::{context::Polarisations, math::TileBaselineFlags};

/// The number of polarisations a column written by hyperdrive has.
const NUM_POLS: usize = 4;

/// The visibilities of one timestep to be written to a measurement set
/// column. The arrays are those filled by reading the same timestep with the
/// same tile and channel flags: `[unflagged channel][unflagged baseline or
/// tile]`.
pub(crate) struct ColumnTimestep<'a> {
    pub(crate) cross_vis_fb: ArrayView2<'a, Jones<f32>>,
    pub(crate) cross_weights_fb: ArrayView2<'a, f32>,
    /// Auto-correlation visibilities and weights. If this is `None`, the
    /// auto-correlation rows are treated as having no visibilities.
    pub(crate) autos_fb: Option<(ArrayView2<'a, Jones<f32>>, ArrayView2<'a, f32>)>,
}

impl MsReader {
    /// Get ready to write visibilities to a column of the main table. The
    /// column is created if it doesn't exist, with the shape of the data
    /// column that's being read. Until [`MsReader::finish_writing`] is called,
    /// the measurement set is open for writing.
    pub(crate) fn prepare_column(&self, col_name: &str) -> Result<(), MsReadError> {
        if self.obs_context.polarisations != Polarisations::XX_XY_YX_YY {
            return Err(MsReadError::ColumnWriteNeedsFourPols {
                col_name: col_name.to_string(),
                pols: self.obs_context.polarisations.to_string(),
            });
        }

        let mut read_state = self.read_state.lock().expect("lock isn't poisoned");
        // Drop any table that's open only for reading before opening it for
        // writing.
        *read_state = None;
        let mut main_table = Table::open(&self.ms, TableOpenMode::ReadWrite)?;

        if !main_table.column_names()?.iter().any(|c| c == col_name) {
            let data: Vec<c32> = main_table.get_cell_as_vec(&self.data_col_name, 0)?;
            let num_chans = (data.len() / NUM_POLS) as u64;
            // Aim for tiles of about 4 MiB.
            let rows_per_tile =
                ((4 * 1024 * 1024) / (data.len().max(1) * std::mem::size_of::<c32>())).max(1);
            main_table.add_tiled_array_column(
                GlueDataType::TpComplex,
                col_name,
                Some("Written by mwa_hyperdrive"),
                &[num_chans, NUM_POLS as u64],
                &[rows_per_tile as u64, num_chans, NUM_POLS as u64],
                &format!("Tiled{col_name}"),
                true,
            )?;
        }

        *read_state = Some(ReadState {
            main_table,
            staged: StagedRows::default(),
        });
        Ok(())
    }

    /// Close the measurement set after writing to it, so that everything
    /// written is on disk. Reading still works afterwards.
    pub(crate) fn finish_writing(&self) {
        *self.read_state.lock().expect("lock isn't poisoned") = None;
    }

    /// Write a timestep's visibilities to a column of the main table.
    /// [`MsReader::prepare_column`] must have been called for this column.
    ///
    /// Every cell of the timestep's rows is written. Cells that have no
    /// visibility supplied (those of flagged tiles and flagged channels, or
    /// all of them if `vis` is `None`) get zeros.
    ///
    /// If `add_flags` is true, then the `FLAG` column is set for every cell
    /// that has no visibility or whose weight isn't positive. Existing flags
    /// are never removed.
    pub(crate) fn write_column_timestep(
        &self,
        col_name: &str,
        timestep: usize,
        vis: Option<ColumnTimestep>,
        tile_baseline_flags: &TileBaselineFlags,
        flagged_fine_chans: &HashSet<u16>,
        add_flags: bool,
    ) -> Result<(), MsReadError> {
        let mut read_state = self.read_state.lock().expect("lock isn't poisoned");
        let main_table = match read_state.as_mut() {
            Some(s) => &mut s.main_table,
            None => {
                // Not prepared; the table would only be open for reading.
                drop(read_state);
                self.prepare_column(col_name)?;
                return self.write_column_timestep(
                    col_name,
                    timestep,
                    vis,
                    tile_baseline_flags,
                    flagged_fine_chans,
                    add_flags,
                );
            }
        };

        // For each channel of the measurement set, its index in the supplied
        // arrays.
        let chan_to_unflagged_index: Vec<Option<usize>> = {
            let mut i_unflagged = 0;
            (0..self.obs_context.fine_chan_freqs.len())
                .map(|i_chan| {
                    if flagged_fine_chans.contains(&(i_chan as u16)) {
                        None
                    } else {
                        i_unflagged += 1;
                        Some(i_unflagged - 1)
                    }
                })
                .collect()
        };

        let row_range = (timestep * self.step) as u64..((timestep + 1) * self.step) as u64;
        for row in row_range {
            let ant1: i32 = main_table.get_cell("ANTENNA1", row)?;
            let ant2: i32 = main_table.get_cell("ANTENNA2", row)?;
            let ant1 = self.tile_map[&ant1];
            let ant2 = self.tile_map[&ant2];
            // As when reading, a row stored as ANTENNA1 > ANTENNA2 has the
            // conjugate transpose of the (ANTENNA2, ANTENNA1) baseline's
            // visibilities.
            let swapped = ant1 > ant2;
            let (ant1, ant2) = if swapped { (ant2, ant1) } else { (ant1, ant2) };

            // Find this row's visibilities and weights, if there are any.
            let row_vis = vis.as_ref().and_then(|vis| {
                if ant1 == ant2 {
                    let i_tile = *tile_baseline_flags
                        .tile_to_unflagged_auto_index_map
                        .get(&ant1)?;
                    let (auto_vis_fb, auto_weights_fb) = vis.autos_fb.as_ref()?;
                    Some((auto_vis_fb.column(i_tile), auto_weights_fb.column(i_tile)))
                } else {
                    let i_bl = *tile_baseline_flags
                        .tile_to_unflagged_cross_baseline_map
                        .get(&(ant1, ant2))?;
                    Some((
                        vis.cross_vis_fb.column(i_bl),
                        vis.cross_weights_fb.column(i_bl),
                    ))
                }
            });

            // The shape of the row's cells comes from its flags, which have
            // to be read anyway if they're to be added to.
            let flags: Vec<bool> = main_table.get_cell_as_vec("FLAG", row)?;
            let num_row_chans = flags.len() / NUM_POLS;
            let mut flags = Array2::from_shape_vec((num_row_chans, NUM_POLS), flags)
                .unwrap_or_else(|_| panic!("MS row {row} doesn't have 4 polarisations"));
            let mut data: Array2<c32> = Array2::default((num_row_chans, NUM_POLS));
            let mut flags_changed = false;
            for (i_chan, (mut data, mut flags)) in data
                .outer_iter_mut()
                .zip(flags.outer_iter_mut())
                .enumerate()
            {
                let cell = row_vis.as_ref().and_then(|(vis_f, weights_f)| {
                    let i = (*chan_to_unflagged_index.get(i_chan)?)?;
                    Some((vis_f[i], weights_f[i]))
                });
                let good = match cell {
                    Some((vis, weight)) => {
                        let vis = if swapped { vis.h() } else { vis };
                        data.iter_mut().zip(vis.iter()).for_each(|(d, v)| *d = *v);
                        weight > 0.0
                    }
                    None => false,
                };
                if add_flags && !good && flags.iter().any(|f| !*f) {
                    flags.fill(true);
                    flags_changed = true;
                }
            }

            main_table.put_cell(col_name, row, &data)?;
            if flags_changed {
                main_table.put_cell("FLAG", row, &flags)?;
            }
        }

        Ok(())
    }

    /// The path to the measurement set.
    pub(crate) fn path(&self) -> &std::path::Path {
        &self.ms
    }
}
