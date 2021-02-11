// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
This module handles interfacing with C, C++ and/or CUDA.

Foreign functions and interfaces from C, C++ and/or CUDA are automatically
provided here by bindgen.
 */

#![allow(non_snake_case)]

// Link hyperdrive_cu produced from build.rs
#[link(name = "hyperdrive_cu", kind = "static")]

include!("bindings.rs");
