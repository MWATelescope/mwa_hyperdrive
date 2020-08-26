// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::io::BufReader;

use serde::Deserialize;
use structopt::StructOpt;
use thiserror::Error;

use mwa_hyperdrive::visibility_gen::{vis_gen, TimeFreqParams};
use mwa_hyperdrive::*;
use mwa_hyperdrive_srclist::woden::parse_source_list;

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
    #[structopt(short, long, parse(from_os_str))]
    source_list: Option<PathBuf>,

    /// Path to the metafits file.
    #[structopt(short, long, parse(from_str))]
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
    #[structopt(long)]
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

impl std::fmt::Display for SimulateVisConcrete {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            r#"simulate-vis args:
RA:                 {ra} deg
Dec:                {dec} deg
Fine channel width: {fcw} kHz
Time resolution:    {t} s
Coarse band chans:  {bands:?}
Num. time steps:    {steps}
Source list:        {sl}
Metafits file:      {m}"#,
            sl = self.source_list.display(),
            m = self.metafits.display(),
            ra = self.ra,
            dec = self.dec,
            fcw = self.fine_channel_width / 1e3,
            bands = self.bands,
            steps = self.steps,
            t = self.time_res,
        )
    }
}

#[derive(Error, Debug)]
pub(crate) enum ParamError {
    #[error("Neither --num-bands nor --bands were supplied!")]
    BandsMissing,

    #[error("Number of bands cannot be 0 or contain 0!")]
    BandsZero,

    #[error("Number of fine channels cannot be 0!")]
    FineChansZero,

    #[error("The fine channel width cannot be 0 or negative!")]
    FineChansWidthTooSmall,

    #[error("The fine channel width ({fcw} kHz) is larger than this observation's coarse channel width ({ccw} kHz)!")]
    FineChanWidthTooBig { fcw: f64, ccw: u64 },

    #[error("No metafits was supplied!")]
    MetafitsMissing,

    #[error("No source list was supplied!")]
    SrclistMissing,

    #[error("No Right Ascension was supplied!")]
    RaMissing,

    #[error("No Declination was supplied!")]
    DecMissing,

    #[error("Right Ascension was not within 0 to 360!")]
    RaInvalid,

    #[error("Declination was not within -90 to 90!")]
    DecInvalid,

    #[error("Number of time steps cannot be 0!")]
    TimeStepsInvalid,

    #[error("{0}")]
    Mwalib(#[from] MwalibError),

    #[error("Generic IO error: {0}")]
    IO(#[from] std::io::Error),

    #[error(
        "Parameter file ({file}) doesn't have a recognised file extension!
Valid extensions are .toml and .json"
    )]
    InvalidFileExtension { file: PathBuf },

    #[error("Couldn't decode toml structure from {file}")]
    TomlConvert { file: PathBuf },

    #[error("Couldn't decode json structure from {file}")]
    JsonConvert { file: PathBuf },
}

/// Both command-line and parameter-file arguments overlap in terms of what is
/// available; this function consolidates everything that was specified into a
/// single struct. Where applicable, it will prefer CLI parameters over those in
/// the file.
pub(crate) fn merge_cli_and_file_params(
    cli_args: SimulateVisArgs,
    param_file: Option<PathBuf>,
) -> Result<(SimulateVisConcrete, Context), ParamError> {
    // If available, read in the parameter file. Otherwise, all fields are
    // `None`.
    let file_params: SimulateVisArgs = if let Some(fp) = param_file {
        debug!("Found a parameter file; attempting to parse...");
        let mut contents = String::new();
        match fp.extension().and_then(|e| e.to_str()) {
            Some("toml") => {
                debug!("Parsing toml file...");
                let mut fh = File::open(&fp)?;
                fh.read_to_string(&mut contents)?;
                match toml::from_str(&contents) {
                    Ok(p) => p,
                    Err(_) => return Err(ParamError::TomlConvert { file: fp }),
                }
            }
            Some("json") => {
                debug!("Parsing json file...");
                let mut fh = File::open(&fp)?;
                fh.read_to_string(&mut contents)?;
                match serde_json::from_str(&contents) {
                    Ok(p) => p,
                    Err(_) => return Err(ParamError::JsonConvert { file: fp }),
                }
            }
            _ => return Err(ParamError::InvalidFileExtension { file: fp }),
        }
    } else {
        std::default::Default::default()
    };

    // From either `bands` or `num_bands`, build up the vector of
    // freq. bands.
    debug!("Attempting to determine the freq. bands...");
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
            return Err(ParamError::BandsMissing);
        }

        // Neglect the file parameters.
        (Some(c_bands), c_num_bands, _, _) => {
            if c_num_bands.is_some() {
                warn!("Both --num-bands and --bands were specified; ignoring --num-bands");
            }
            c_bands
        }

        (None, None, Some(f_bands), f_num_bands) => {
            if f_num_bands.is_some() {
                warn!("Both \"num_bands\" and \"bands\" were specified; ignoring \"num_bands\"");
            }
            f_bands
        }

        // Override file-specified parameters.
        (None, Some(c_num_bands), _, _) => {
            // The number of bands cannot be 0.
            if c_num_bands == 0 {
                return Err(ParamError::BandsZero);
            }
            // Assume we start from 1.
            (1..=c_num_bands).collect::<Vec<_>>()
        }

        (None, None, None, Some(f_num_bands)) => {
            // The number of bands cannot be 0.
            if f_num_bands == 0 {
                return Err(ParamError::BandsZero);
            }
            // Assume we start from 1.
            (1..=f_num_bands).collect::<Vec<_>>()
        }
    };
    if bands.is_empty() {
        return Err(ParamError::BandsMissing);
    }
    if bands.contains(&0) {
        return Err(ParamError::BandsZero);
    }

    debug!("Attempting to open the metafits...");
    let metafits = match cli_args.metafits.or(file_params.metafits) {
        Some(m) => m,
        None => return Err(ParamError::MetafitsMissing),
    };

    debug!("Attempting to create a Context from the metafits...");
    let context = Context::new(&metafits, &[])?;

    // Assign `fine_channel_width` from a specified `fine_channel_width` or
    // `num_fine_channels`. The specified units are in kHz, but the output here
    // should be in Hz.
    debug!("Attempting to determine the fine channel width...");
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
                warn!("Both --fine-channel-width and --num-fine-channels were specified; ignoring --num-fine-channels");
            }
            c_fcw
        }

        (None, None, Some(f_fcw), f_nfc) => {
            if f_nfc.is_some() {
                warn!("Both \"fine_channel_width\" and \"num_fine_channels\" were specified; ignoring \"num_fine_channels\"");
            }
            f_fcw
        }

        // Override file-specified parameters.
        (None, Some(c_nfc), _, _) => {
            if c_nfc == 0 {
                return Err(ParamError::FineChansZero);
            }
            (context.mwalib.coarse_channel_width_hz / c_nfc as u32) as f64
        }

        (None, None, None, Some(f_nfc)) => {
            if f_nfc == 0 {
                return Err(ParamError::FineChansZero);
            }
            (context.mwalib.coarse_channel_width_hz / f_nfc as u32) as f64
        }
    };
    if fine_channel_width < 0.0 || fine_channel_width.abs() < 1e-6 {
        return Err(ParamError::FineChansWidthTooSmall);
    }
    if fine_channel_width > context.mwalib.coarse_channel_width_hz as f64 {
        return Err(ParamError::FineChanWidthTooBig {
            fcw: fine_channel_width / 1000.0,
            ccw: context.mwalib.coarse_channel_width_hz as u64 / 1000,
        });
    }

    let ra = match cli_args.ra.or(file_params.ra) {
        None => return Err(ParamError::RaMissing),
        Some(ra) => {
            if ra < 0.0 || ra > 360.0 {
                return Err(ParamError::RaInvalid);
            }
            ra
        }
    };
    let dec = match cli_args.dec.or(file_params.dec) {
        None => return Err(ParamError::DecMissing),
        Some(dec) => {
            if dec > 90.0 || dec < -90.0 {
                return Err(ParamError::DecInvalid);
            }
            dec
        }
    };

    let steps = cli_args
        .steps
        .or(file_params.steps)
        .unwrap_or(DEFAULT_NUM_TIME_STEPS);
    if steps == 0 {
        return Err(ParamError::TimeStepsInvalid);
    }

    // Construct the complete argument struct. We must report any
    // arguments that are missing.
    let concrete_args = SimulateVisConcrete {
        source_list: match cli_args.source_list.or(file_params.source_list) {
            Some(s) => s,
            None => return Err(ParamError::SrclistMissing),
        },
        metafits,
        ra,
        dec,
        fine_channel_width,
        bands,
        steps,
        time_res: cli_args
            .time_res
            .or(file_params.time_res)
            .unwrap_or(DEFAULT_TIME_RES),
    };
    debug!(
        "Proceeding with the following arguments:\n{:?}",
        concrete_args
    );
    Ok((concrete_args, context))
}

/// Simulate visibilities. This is similar to what WODEN does.
///
/// CHJ created this early in hyperdrive's life to get a Rust proof of concept
/// off the ground. Unless this code gets a bit more love, WODEN should be used
/// instead.
///
/// This function assumes that the input source list is WODEN style.
pub(crate) fn simulate_vis(
    cli_args: SimulateVisArgs,
    param_file: Option<PathBuf>,
    cpu: bool,
    text: bool,
    dry_run: bool,
) -> Result<(), anyhow::Error> {
    let (args, context) = merge_cli_and_file_params(cli_args, param_file)?;

    if dry_run {
        println!("{}\n", args);
        println!("{}", context);
        return Ok(());
    }

    let params = TimeFreqParams {
        // `steps` was already a u8, so we aren't doing anything crazy by
        // putting it into a bigger type. Make it usize for convenience.
        n_time_steps: args.steps as usize,
        time_resolution: args.time_res,
        freq_bands: args.bands,
        n_fine_channels: context.mwalib.coarse_channel_width_hz as u64
            / args.fine_channel_width as u64,
        fine_channel_width: args.fine_channel_width,
    };
    debug!("Parameters for vis_gen:\n{:?}", params);

    // Read in the source list.
    debug!("Reading in the source list...");
    let sl: SourceList = {
        let mut f = BufReader::new(File::open(args.source_list)?);
        parse_source_list(&mut f)?
    };

    // Create a pointing centre struct using the specified RA and Dec.
    let pc = PointingCentre::new_from_ra(
        context.base_lst,
        args.ra.to_radians(),
        args.dec.to_radians(),
    );
    // Generate the visibilities.
    vis_gen(
        &context,
        sl.iter().next().unwrap().1,
        &params,
        pc,
        !cpu,
        text,
    )?;

    Ok(())
}
