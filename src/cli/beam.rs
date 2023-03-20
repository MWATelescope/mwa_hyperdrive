// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::{
    f64::consts::{FRAC_PI_2, TAU},
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
};

use clap::Parser;
use marlu::AzEl;
use num_traits::{Float, FromPrimitive};

use crate::{
    beam::{create_beam_object, Delays, BEAM_TYPES_COMMA_SEPARATED},
    HyperdriveError,
};

lazy_static::lazy_static! {
    static ref BEAM_TYPE_HELP: String = format!("The type of beam to use. Supported types: {}", *BEAM_TYPES_COMMA_SEPARATED);
}

/// Generate beam response values.
#[derive(Parser, Debug)]
pub struct BeamArgs {
    #[clap(help = BEAM_TYPE_HELP.as_str())]
    beam_type: String,

    /// The frequency to use for the beam model [MHz].
    #[clap(short, long, default_value = "150")]
    freq_mhz: f64,

    /// If specified, use these dipole delays for the MWA pointing. e.g. 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3
    #[clap(short, long, multiple_values(true))]
    delays: Option<Vec<u32>>,

    /// The array latitude to use. This only affects the parallactic-angle
    /// correction.
    #[clap(short, long, allow_hyphen_values = true, default_value = "-27.0")]
    latitude_deg: f64,

    /// Get beam responses from zenith down to this zenith angle [degrees]. e.g.
    /// If this is 20 degrees, then beam responses from zenith angles 0 to 20
    /// degrees are generated (corresponds to elevations 70 to 90 degrees).
    #[clap(long, default_value = "90.0")]
    max_za: f64,

    /// The step in azimuth and elevation to use [degrees]. Larger steps produce
    /// fewer values, but will give coarser resolution.
    #[clap(short, long, default_value = "1.0")]
    step: f64,

    /// The file to write the results to. The files are formatted as
    /// tab-separated, with each line (1) the azimuth in radians, (2) the
    /// elevation in radians, and (3) the "proxy Stokes I" value of the beam
    /// response (i.e. if the beam response is a Jones matrix j, then |j[0]| +
    /// |j[3]| is returned).
    #[clap(short, long, default_value = "beam_responses.tsv")]
    output: PathBuf,

    /// Use CUDA to generate the beam responses.
    #[cfg(feature = "cuda")]
    #[clap(short, long)]
    cuda: bool,
}

impl BeamArgs {
    pub(super) fn run(&self) -> Result<(), HyperdriveError> {
        cfg_if::cfg_if! {
            if #[cfg(feature = "cuda")] {
                if self.cuda {
                    calc_cuda(self)
                } else {
                    calc_cpu(self)
                }
            } else {
                calc_cpu(self)
            }
        }
    }
}

/// This makes all of the zenith angles based off the cutoff, then maps each ZA
/// with all azimuths.
fn gen_azzas<F: Float + FromPrimitive>(
    max_za_rad: f64,
    step_radians: f64,
) -> impl Iterator<Item = (F, F)> {
    (0..)
        .map(move |i| step_radians * i as f64)
        .take_while(move |zenith_angle| *zenith_angle < max_za_rad)
        .flat_map(move |zenith_angle| {
            let zenith_angle = F::from_f64(zenith_angle).unwrap();
            (0..)
                .map(move |i| step_radians * i as f64)
                .take_while(|angle| *angle < TAU)
                .map(move |azimuth| (F::from_f64(azimuth).unwrap(), zenith_angle))
        })
}

fn calc_cpu(args: &BeamArgs) -> Result<(), HyperdriveError> {
    let BeamArgs {
        beam_type,
        delays,
        freq_mhz,
        latitude_deg,
        max_za,
        step,
        output,
        #[cfg(feature = "cuda")]
            cuda: _,
    } = args;

    let beam = create_beam_object(
        Some(beam_type.as_str()),
        1,
        Delays::Partial(delays.clone().unwrap_or(vec![0; 16])),
    )?;
    let mut out = BufWriter::new(File::create(output)?);

    let azels: Vec<_> = gen_azzas(max_za.to_radians(), step.to_radians())
        .map(|(az, za)| AzEl::from_radians(az, FRAC_PI_2 - za))
        .collect();
    let jones = beam.calc_jones_array(&azels, freq_mhz * 1e6, None, latitude_deg.to_radians())?;
    for (j, azel) in jones.into_iter().zip(azels) {
        writeln!(
            &mut out,
            "{}\t{}\t{:e}",
            azel.az,
            azel.za(),
            j[0].norm() + j[3].norm()
        )?;
    }

    Ok(())
}

#[cfg(feature = "cuda")]
fn calc_cuda(args: &BeamArgs) -> Result<(), HyperdriveError> {
    use itertools::izip;
    use num_complex::Complex;

    use crate::cuda::{CudaFloat, CudaJones, DevicePointer};

    let BeamArgs {
        beam_type,
        delays,
        freq_mhz,
        latitude_deg,
        max_za,
        step,
        output,
        cuda: _,
    } = args;

    let beam = create_beam_object(
        Some(beam_type.as_str()),
        1,
        Delays::Partial(delays.clone().unwrap_or(vec![0; 16])),
    )?;
    let cuda_beam = beam.prepare_cuda_beam(&[(freq_mhz * 1e6) as u32])?;
    let mut out = BufWriter::new(File::create(output)?);

    let (azs, zas): (Vec<_>, Vec<_>) =
        gen_azzas::<CudaFloat>(max_za.to_radians(), step.to_radians()).unzip();
    let mut d_jones: DevicePointer<CudaJones> = DevicePointer::malloc(
        cuda_beam.get_num_unique_tiles() as usize
            * cuda_beam.get_num_unique_freqs() as usize
            * azs.len()
            * std::mem::size_of::<CudaJones>(),
    )?;

    unsafe {
        cuda_beam.calc_jones_pair(
            &azs,
            &zas,
            latitude_deg.to_radians(),
            d_jones.get_mut().cast(),
        )?;
    }

    let jones = d_jones.copy_from_device_new()?;
    for (j, az, za) in izip!(jones, azs, zas) {
        let j0 = Complex::new(j.j00_re, j.j00_im);
        let j3 = Complex::new(j.j11_re, j.j11_im);
        writeln!(&mut out, "{}\t{}\t{:e}", az, za, j0.norm() + j3.norm())?;
    }

    Ok(())
}
