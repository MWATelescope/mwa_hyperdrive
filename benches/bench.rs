// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use criterion::*;

use mwa_rust_core::{c64, Jones};

fn misc(c: &mut Criterion) {
    // c.bench_function("hermitian multiply", |b| b.iter(|| j.mul_hermitian(&j2)));
}

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets =
        misc,
);
criterion_main!(benches);
