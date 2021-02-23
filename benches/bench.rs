// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use criterion::*;

use mwa_hyperdrive_core::{c64, Jones};

fn jones_operations(c: &mut Criterion) {
    let j = Jones::from([
        c64::new(1.0, -2.0),
        c64::new(5.0, -6.0),
        c64::new(3.0, -4.0),
        c64::new(7.0, -8.0),
    ]);
    let j2 = j.clone() * 2.0;

    c.bench_function("hermitian multiply", |b| b.iter(|| j.mul_hermitian(&j2)));
}

criterion_group!(benches, jones_operations);
criterion_main!(benches);
