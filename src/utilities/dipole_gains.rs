// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Print neatly-formatted information on the dipole gains (i.e. which are dead)
//! in a metafits file.

use std::path::PathBuf;

use clap::Parser;
use log::info;
use mwalib::{MetafitsContext, MwalibError};

use mwa_hyperdrive_common::{clap, log, mwalib};

/// Print information on the dipole gains listed by a metafits file.
#[derive(Parser, Debug)]
pub struct DipoleGains {
    #[clap(name = "METAFITS_FILE", parse(from_os_str))]
    metafits: PathBuf,
}

impl DipoleGains {
    pub fn run(&self) -> Result<(), MwalibError> {
        let meta = MetafitsContext::new(&self.metafits, None).unwrap();
        let gains = crate::metafits::get_dipole_gains(&meta);
        let mut all_unity = Vec::with_capacity(128);
        let mut non_unity = Vec::with_capacity(128);
        for (i, tile_gains) in gains.outer_iter().enumerate() {
            if tile_gains
                .iter()
                .all(|&g| g.is_finite() && (g - 1.0).abs() < f64::EPSILON)
            {
                all_unity.push(i);
            } else {
                non_unity.push((i, tile_gains));
            }
        }

        if all_unity.len() == meta.num_ants {
            info!("All dipoles on all tiles have a gain of 1.0!");
        } else {
            info!(
                "Tiles with all dipoles alive ({}): {:?}",
                all_unity.len(),
                all_unity
            );
            info!("Other tiles:");
            let mut bad_x = Vec::with_capacity(16);
            let mut bad_y = Vec::with_capacity(16);
            let mut bad_string = String::new();
            for (tile_num, tile_gains) in non_unity {
                let tile_gains = tile_gains.as_slice().unwrap();
                tile_gains[..16].iter().enumerate().for_each(|(i, &g)| {
                    if (g - 1.0).abs() > f64::EPSILON {
                        bad_x.push(i);
                    }
                });
                tile_gains[16..].iter().enumerate().for_each(|(i, &g)| {
                    if (g - 1.0).abs() > f64::EPSILON {
                        bad_y.push(i);
                    }
                });
                bad_string.push_str(&format!("    Tile {:>3}: ", tile_num));
                if !bad_x.is_empty() {
                    bad_string.push_str(&format!("X {:?}", &bad_x));
                }
                if !bad_x.is_empty() && !bad_y.is_empty() {
                    bad_string.push_str(", ");
                }
                if !bad_y.is_empty() {
                    bad_string.push_str(&format!("Y {:?}", &bad_y));
                }
                info!("{}", bad_string);
                bad_x.clear();
                bad_y.clear();
                bad_string.clear();
            }
        }

        Ok(())
    }
}
