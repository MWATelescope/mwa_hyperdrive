// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! CUDA code to be used by hyperdrive, contained within its own crate.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
// This is needed only because tests inside bindgen-produced files (model.rs and
// memory.rs) trigger the warning.
#![allow(deref_nullptr)]

// Link hyperdrive_cu produced from build.rs
#[link(name = "hyperdrive_cu", kind = "static")]

include!("model.rs");
include!("memory.rs");
