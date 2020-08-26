// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
CUDA code to be used by hyperdrive, contained within its own crate.
 */

// Allow non snake case names. This prevents a warning automatically generated
// by bindgen.
#[allow(non_snake_case)]
pub mod foreign;
pub use foreign::*;
