// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::collections::HashSet;
use std::ops::Deref;

use criterion::*;
use marlu::{AzEl, Jones, LMN, UVW};
use ndarray::Array2;

use mwa_hyperdrive::math::TileBaselineMaps;
use mwa_hyperdrive_beam::{create_fee_beam_object, Delays};
use mwa_hyperdrive_common::{marlu, ndarray};

fn model(c: &mut Criterion) {
    let num_tiles = 30;
    let num_bls = (num_tiles * (num_tiles - 1)) / 2;
    let num_freqs = 500;
    let num_points = 1000;
    let mut model_array = Array2::from_elem((num_bls, num_freqs), Jones::default());
    let azels = vec![AzEl::new_degrees(0.0, 90.0); num_points];
    let fds = Array2::from_elem((num_freqs, num_points), Jones::identity());
    let uvws = vec![UVW::default(); num_bls];
    let lmns = vec![LMN::default(); num_points];
    let freqs = vec![0.0; num_freqs];
    let maps = TileBaselineMaps::new(num_tiles, &HashSet::new());
    let beam_file: Option<&str> = None; // Assume the env. variable is set.
    let beam =
        create_fee_beam_object(beam_file, num_tiles, Delays::Partial(vec![0; 16]), None).unwrap();
    c.bench_function("model_points", |b| {
        b.iter(|| {
            mwa_hyperdrive::model::model_points(
                model_array.view_mut(),
                beam.deref(),
                &azels,
                &lmns,
                fds.view(),
                &uvws,
                &freqs,
                &maps.unflagged_cross_baseline_to_tile_map,
            );
        })
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets =
        model,
);
criterion_main!(benches);
