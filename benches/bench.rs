// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// use criterion::*;

// use mwa_hyperdrive::instrument::*;
// use mwa_hyperdrive::*;

// fn mwa_tile_response(c: &mut Criterion) {
//     let metafits = "tests/1065880128.metafits";
//     let context = mwalib::mwalibContext::new(&metafits, &[]).unwrap();
//     let beam = PrimaryBeam::default(BeamType::Mwa32T, 0, Pol::X, &context);
//     let scaling = mwa_hyperdrive::instrument::BeamScaling::None;
//     let azel = AzEl::new_degrees(135.0, -45.0);
//     let delays = [0.0; 16];

//     c.bench_function("analytic beam tile response", |b| {
//         b.iter(|| {
//             mwa_hyperdrive::instrument::tile_response(&beam, &azel, 180e6, &scaling, &delays)
//                 .unwrap();
//         })
//     });

//     c.bench_function("analytic beam tile response - 128 times", |b| {
//         b.iter(|| {
//             for _ in 0..128 {
//                 mwa_hyperdrive::instrument::tile_response(&beam, &azel, 180e6, &scaling, &delays)
//                     .unwrap();
//             }
//         })
//     });
// }

// criterion_group!(benches, mwa_tile_response);
// criterion_main!(benches);
