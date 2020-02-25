// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

use anyhow::bail;
use structopt::StructOpt;

use mwa_hyperdrive::sourcelist::read::parse_source_list;
use mwa_hyperdrive::sourcelist::source::*;
use mwa_hyperdrive::visibility_gen::{vis_gen, TimeFreqParams};
use mwa_hyperdrive::*;

#[derive(StructOpt, Debug)]
#[structopt(name = "mwa_hyperdrive", author)]
enum Opt {
    /// Run a WODEN-like simulation
    Woden {
        /// Path to the source list used for sky modelling.
        #[structopt(short, long)]
        source_list: PathBuf,

        /// Path to the metafits file.
        #[structopt(short, long)]
        metafits: PathBuf,

        /// The pointing-centre right ascension [degrees].
        #[structopt(short, long, default_value = "50.67")]
        ra: f64,

        /// The pointing-centre declination [degrees].
        #[structopt(short, long, default_value = "-37.2")]
        dec: f64,

        /// The fine-channel resolution [kHz].
        #[structopt(short, long, default_value = "80")]
        fine_channel_width: f64,

        /// The number of coarse bands used, starting from 0 (e.g. specifying 3
        /// will use coarse bands 0, 1 and 2).
        #[structopt(short, long)]
        num_bands: Option<u8>,

        /// The coarse bands used (e.g. -b 0 1 2).
        #[structopt(short, long)]
        bands: Option<Vec<u8>>,

        /// The number of time steps used from the metafits epoch.
        #[structopt(long, default_value = "14")]
        steps: u8,

        /// The time resolution [seconds].
        #[structopt(short, long, default_value = "8.0")]
        time_res: f64,

        /// Use the CPU for visibility generation.
        #[structopt(short, long)]
        cpu: bool,
    },
    /// Verify that a source list can be read by the hyperdrive.
    VerifySrclist {
        /// Path to the source list(s) to be verified.
        #[structopt(name = "SOURCE_LISTS", parse(from_os_str))]
        source_lists: Vec<PathBuf>,
    },
}

fn woden(
    context: Context,
    params: TimeFreqParams,
    source_list: PathBuf,
    ra: f64,
    dec: f64,
    cuda: bool,
) -> Result<(), anyhow::Error> {
    // Read in the source list.
    let sources: Vec<Source> = {
        let mut f = File::open(source_list)?;
        let mut contents = String::new();
        f.read_to_string(&mut contents)?;
        parse_source_list(&contents)?
    };

    // Create a pointing centre struct using the specified RA and Dec.
    let pc = PC::new_from_ra(context.base_lst, ra, dec);
    // Generate the visibilities.
    vis_gen(&context, &sources[0], &params, pc, cuda)?;

    Ok(())
}

fn main() -> Result<(), anyhow::Error> {
    match Opt::from_args() {
        Opt::Woden {
            source_list,
            metafits,
            ra,
            dec,
            fine_channel_width,
            num_bands,
            bands,
            steps,
            time_res,
            cpu,
        } => {
            let bands = if num_bands.is_none() && bands.is_none() {
                bail!("Neither --num_bands nor --bands were supplied!");
            } else if num_bands.is_none() {
                bands.unwrap()
            } else {
                // Assume we start from 0.
                (0..num_bands.unwrap()).collect()
            };

            // Use the metafits file.
            let mut metafits_fptr = fitsio::FitsFile::open(metafits)?;
            let context = Context::new(&mut metafits_fptr)?;
            let fine_channel_width = fine_channel_width * 1e3;
            let params = TimeFreqParams {
                n_time_steps: steps as usize,
                time_resolution: time_res,
                freq_bands: bands,
                n_fine_channels: context.coarse_channel_width / fine_channel_width as usize,
                fine_channel_width,
            };

            woden(
                context,
                params,
                source_list,
                ra.to_radians(),
                dec.to_radians(),
                !cpu,
            )?;
        }
        Opt::VerifySrclist { source_lists } => {
            if source_lists.is_empty() {
                bail!("No source lists were supplied!");
            }

            for source_list in source_lists {
                let sources: Vec<Source> = {
                    let mut f = File::open(&source_list)?;
                    let mut contents = String::new();
                    f.read_to_string(&mut contents)?;
                    parse_source_list(&contents)?
                };

                println!("{}:", source_list.to_string_lossy());
                println!(
                    "{} sources, {} components\n",
                    sources.len(),
                    sources.iter().map(|s| s.components.len()).sum::<usize>()
                );
            }
        }
    }

    Ok(())
}
