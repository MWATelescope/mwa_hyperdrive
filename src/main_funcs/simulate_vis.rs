// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use serde::Deserialize;
use structopt::StructOpt;

use super::*;
use mwa_hyperdrive::sourcelist::read::parse_source_list;
use mwa_hyperdrive::sourcelist::source::Source;
use mwa_hyperdrive::visibility_gen::{vis_gen, TimeFreqParams};
use mwa_hyperdrive::{Context, PC};

/// Contains all the arguments needed to do visibility simulation.
#[derive(Debug)]
pub struct SimulateVisConcrete {
    /// Path to the source list used for sky modelling.
    pub source_list: PathBuf,

    /// Path to the metafits file.
    pub metafits: PathBuf,

    /// The pointing-centre right ascension [degrees].
    pub ra: f64,

    /// The pointing-centre declination [degrees].
    pub dec: f64,

    /// The fine-channel resolution [Hz].
    pub fine_channel_width: f64,

    /// The MWA coarse bands used. Cannot contain 0.
    pub bands: Vec<u8>,

    /// The number of time steps used from the metafits epoch.
    pub steps: u8,

    /// The time resolution [seconds].
    pub time_res: f64,
}

// These values should match the defaults reported in the struct below.
pub static DEFAULT_FINE_CHANNEL_WIDTH: f64 = 80.0;
pub static DEFAULT_NUM_TIME_STEPS: u8 = 14;
pub static DEFAULT_TIME_RES: f64 = 8.0;

#[derive(StructOpt, Debug, Default, Deserialize)]
pub struct SimulateVisArgs {
    /// Path to the source list used for sky modelling.
    #[structopt(short, long)]
    source_list: Option<PathBuf>,

    /// Path to the metafits file.
    #[structopt(short, long)]
    metafits: Option<PathBuf>,

    /// The pointing-centre right ascension [degrees].
    #[structopt(short, long)]
    ra: Option<f64>,

    /// The pointing-centre declination [degrees].
    #[structopt(short, long)]
    dec: Option<f64>,

    /// The number of fine channels per coarse band [default: 16].
    #[structopt(long)]
    num_fine_channels: Option<u16>,

    /// The fine-channel resolution [kHz] [default: 80].
    #[structopt(short, long)]
    fine_channel_width: Option<f64>,

    /// The number of MWA coarse bands used, starting from 1 (e.g. specifying 3
    /// will use coarse bands 1, 2 and 3).
    #[structopt(short, long)]
    num_bands: Option<u8>,

    /// The MWA coarse bands used (e.g. -b 1 2 3). Cannot contain 0.
    #[structopt(short, long)]
    bands: Option<Vec<u8>>,

    /// The number of time steps used from the metafits epoch [default: 14].
    #[structopt(long)]
    steps: Option<u8>,

    /// The time resolution [seconds] [default: 8.0].
    #[structopt(short, long)]
    time_res: Option<f64>,
}

/// Both command-line and parameter-file arguments overlap in terms of what is
/// available; this function consolidates everything that was specified into a
/// single struct. Where applicable, it will prefer CLI parameters over those in
/// the file.
fn merge_cli_and_file_params(
    cli_args: SimulateVisArgs,
    param_file: Option<PathBuf>,
    debug: bool,
) -> Result<(SimulateVisConcrete, Context), anyhow::Error> {
    // If available, read in the parameter file. Otherwise, all fields are
    // `None`.
    let file_params: SimulateVisArgs = if let Some(fp) = param_file {
        if debug {
            eprintln!("Found a parameter file; attempting to parse...");
        }
        let mut fh = File::open(&fp)?;
        let mut contents = String::new();
        fh.read_to_string(&mut contents)?;
        match fp.extension() {
                    None => bail!("Specified parameter file doesn't have a file extension!"),
                    Some(s) => match s.to_str() {
                        Some("toml") => {
                            if debug {
                                eprintln!("Parsing toml file...");
                            }
                            toml::from_str(&contents)?
                        }
                        Some("json") => {
                            if debug {
                                eprintln!("Parsing json file...");
                            }
                            serde_json::from_str(&contents)?
                        }
                        _ => bail!("Specified parameter file doesn't have a recognised file extension!\nValid extensions are .toml and .json"),
                    },
                }
    } else {
        std::default::Default::default()
    };

    // From either `bands` or `num_bands`, build up the vector of
    // freq. bands.
    if debug {
        eprintln!("Attempting to determine the freq. bands...");
    }
    // Handle various combinations of --bands and --num_bands being specified.
    // To help us dear humans, use the compiler's pattern matching to tell us
    // about any combinations that we're not handling.
    let bands = match (
        cli_args.bands,
        cli_args.num_bands,
        file_params.bands,
        file_params.num_bands,
    ) {
        (None, None, None, None) => {
            bail!("Neither --num-bands nor --bands were supplied!");
        }

        // Neglect the file parameters.
        (Some(c_bands), c_num_bands, _, _) => {
            if c_num_bands.is_some() {
                eprintln!(
                    "WARNING: Both --num-bands and --bands were specified; ignoring --num-bands"
                );
            }
            c_bands
        }

        (None, None, Some(f_bands), f_num_bands) => {
            if f_num_bands.is_some() {
                eprintln!(
                "WARNING: Both \"num_bands\" and \"bands\" were specified; ignoring \"num_bands\""
            );
            }
            f_bands
        }

        // Override file-specified parameters.
        (None, Some(c_num_bands), _, _) => {
            // The number of bands cannot be 0.
            if c_num_bands == 0 {
                bail!("--num-bands cannot be 0!");
            }
            // Assume we start from 1.
            (1..=c_num_bands).collect::<Vec<_>>()
        }

        (None, None, None, Some(f_num_bands)) => {
            // The number of bands cannot be 0.
            if f_num_bands == 0 {
                bail!("\"num_bands\" cannot be 0!");
            }
            // Assume we start from 1.
            (1..=f_num_bands).collect::<Vec<_>>()
        }
    };

    if debug {
        eprintln!("Attempting to open the metafits...");
    }
    let metafits = match cli_args.metafits.or(file_params.metafits) {
        Some(m) => m,
        None => bail!("No metafits was supplied!"),
    };
    let mut metafits_fptr = fitsio::FitsFile::open(&metafits)?;

    if debug {
        eprintln!("Attempting to create a Context from the metafits...");
    }
    let context = Context::new(&mut metafits_fptr)?;

    // Assign `fine_channel_width` from a specified `fine_channel_width` or
    // `num_fine_channels`. The specified units are in kHz, but the output here
    // should be in Hz.
    if debug {
        eprintln!("Attempting to determine the fine channel width...");
    }
    // Handle various combinations of --fine_channel_width and
    // --num_fine_channels, similar to --bands and --num_bands above.
    let fine_channel_width = match (
        cli_args.fine_channel_width.map(|f| f * 1e3),
        cli_args.num_fine_channels,
        file_params.fine_channel_width.map(|f| f * 1e3),
        file_params.num_fine_channels,
    ) {
        // If nothing is specified, use the default.
        (None, None, None, None) => DEFAULT_FINE_CHANNEL_WIDTH * 1e3,

        // Neglect the file parameters.
        (Some(c_fcw), c_nfc, _, _) => {
            if c_nfc.is_some() {
                eprintln!("WARNING: Both --fine-channel-width and --num-fine-channels were specified; ignoring --num-fine-channels");
            }
            c_fcw
        }

        (None, None, Some(f_fcw), f_nfc) => {
            if f_nfc.is_some() {
                eprintln!("WARNING: Both \"fine_channel_width\" and \"num_fine_channels\" were specified; ignoring \"num_fine_channels\"");
            }
            f_fcw
        }

        // Override file-specified parameters.
        (None, Some(c_nfc), _, _) => {
            if c_nfc == 0 {
                bail!("--num-fine-channels cannot be 0!");
            }
            (context.coarse_channel_width / c_nfc as usize) as f64
        }

        (None, None, None, Some(f_nfc)) => {
            if f_nfc == 0 {
                bail!("\"num_fine_channels\" cannot be 0!");
            }
            (context.coarse_channel_width / f_nfc as usize) as f64
        }
    };

    // Construct the complete argument struct. We must report any
    // arguments that are missing.
    let concrete_args = SimulateVisConcrete {
        source_list: match cli_args.source_list.or(file_params.source_list) {
            Some(s) => s,
            None => bail!("No source_list was supplied!"),
        },
        metafits,
        ra: match cli_args.ra.or(file_params.ra) {
            Some(r) => r,
            None => bail!("No ra was supplied!"),
        },
        dec: match cli_args.dec.or(file_params.dec) {
            Some(d) => d,
            None => bail!("No dec was supplied!"),
        },
        fine_channel_width,
        bands,
        steps: cli_args
            .steps
            .or(file_params.steps)
            .unwrap_or(DEFAULT_NUM_TIME_STEPS),
        time_res: cli_args
            .time_res
            .or(file_params.time_res)
            .unwrap_or(DEFAULT_TIME_RES),
    };
    if debug {
        eprintln!(
            "Proceeding with the following arguments:\n{:?}",
            concrete_args
        );
    }
    Ok((concrete_args, context))
}

pub(crate) fn simulate_vis(
    cli_args: SimulateVisArgs,
    param_file: Option<PathBuf>,
    cpu: bool,
    text: bool,
    debug: bool,
) -> Result<(), anyhow::Error> {
    let (args, context) = merge_cli_and_file_params(cli_args, param_file, debug)?;

    // Sanity checks.
    if args.ra < 0.0 || args.ra > 360.0 {
        bail!("--ra is not within 0 to 360!");
    }
    if args.dec > 90.0 || args.dec < -90.0 {
        bail!("--dec is not within -90 to 90!");
    }
    // The band list cannot be empty.
    if args.bands.is_empty() {
        bail!("No bands were specified!");
    }
    // 0 cannot be in the band list.
    if args.bands.contains(&0) {
        bail!("0 was specified in --bands; this isn't allowed!");
    }
    // The number must be greater than 0.
    if args.fine_channel_width.abs() < 1e-6 {
        bail!("The fine channel width cannot be 0!");
    }
    // Check that the fine-channel width is not too large.
    if args.fine_channel_width > context.coarse_channel_width as f64 {
        bail!("The fine channel width ({} kHz) is larger than this observation's coarse channel width ({} kHz)!", args.fine_channel_width / 1000.0, context.coarse_channel_width / 1000);
    }

    let params = TimeFreqParams {
        n_time_steps: args.steps as usize,
        time_resolution: args.time_res,
        freq_bands: args.bands,
        n_fine_channels: context.coarse_channel_width / args.fine_channel_width as usize,
        fine_channel_width: args.fine_channel_width,
    };
    if debug {
        eprintln!("Parameters for vis_gen:\n{:?}", params);
    }

    // Read in the source list.
    if debug {
        eprintln!("Reading in the source list...");
    }
    let sources: Vec<Source> = {
        let mut f = File::open(args.source_list)?;
        let mut contents = String::new();
        f.read_to_string(&mut contents)?;
        parse_source_list(&contents)?
    };

    // Create a pointing centre struct using the specified RA and Dec.
    let pc = PC::new_from_ra(
        context.base_lst,
        args.ra.to_radians(),
        args.dec.to_radians(),
    );
    // Generate the visibilities.
    vis_gen(&context, &sources[0], &params, pc, !cpu, text)?;

    Ok(())
}
