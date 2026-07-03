use std::{
    env, fs,
    io::Write,
    path::{Path, PathBuf},
};

use mwa_hyperdrive::CalibrationSolutions;

fn scalar_amp_xx(j: marlu::Jones<f64>) -> Option<f64> {
    if j.any_nan() {
        None
    } else {
        let amp = j[0].norm_sqr().sqrt();
        (amp.is_finite() && amp > 0.0).then_some(amp)
    }
}

fn phase_xx_deg(j: marlu::Jones<f64>) -> Option<f64> {
    if j.any_nan() {
        None
    } else {
        let phase = j[0].arg().to_degrees();
        phase.is_finite().then_some(phase)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args_os().skip(1);
    let input = args
        .next()
        .map(PathBuf::from)
        .ok_or("usage: export_solution_timeblocks <solutions.fits> <output_dir> <timeblock> [timeblock ...]")?;
    let output_dir = args
        .next()
        .map(PathBuf::from)
        .ok_or("usage: export_solution_timeblocks <solutions.fits> <output_dir> <timeblock> [timeblock ...]")?;
    let timeblocks = args
        .map(|a| {
            a.to_str()
                .ok_or("timeblock argument is not valid UTF-8")?
                .parse::<usize>()
                .map_err(|_| "failed to parse timeblock as usize")
        })
        .collect::<Result<Vec<_>, _>>()?;
    if timeblocks.is_empty() {
        return Err("need at least one timeblock".into());
    }

    fs::create_dir_all(&output_dir)?;

    let sols = CalibrationSolutions::read_solutions_from_ext(&input, None::<&Path>)?;
    let (num_timeblocks, num_tiles, num_chanblocks) = sols.di_jones.dim();

    for &tb in &timeblocks {
        if tb >= num_timeblocks {
            return Err(format!(
                "timeblock {tb} out of range; solutions have {num_timeblocks} timeblocks"
            )
            .into());
        }
        let path = output_dir.join(format!("timeblock_{tb:04}_solutions.csv"));
        let mut f = fs::File::create(path)?;
        writeln!(
            f,
            "timeblock,tile,chanblock,freq_mhz,amp_xx,phase_xx_deg,is_nan"
        )?;
        for tile in 0..num_tiles {
            for chanblock in 0..num_chanblocks {
                let j = sols.di_jones[(tb, tile, chanblock)];
                let freq_mhz = sols
                    .chanblock_freqs
                    .as_ref()
                    .and_then(|v| v.get(chanblock))
                    .map(|f| *f / 1e6)
                    .unwrap_or(f64::NAN);
                let amp = scalar_amp_xx(j)
                    .map(|v| v.to_string())
                    .unwrap_or_else(String::new);
                let phase = phase_xx_deg(j)
                    .map(|v| v.to_string())
                    .unwrap_or_else(String::new);
                let is_nan = if j.any_nan() { 1 } else { 0 };
                writeln!(
                    f,
                    "{tb},{tile},{chanblock},{freq_mhz},{amp},{phase},{is_nan}"
                )?;
            }
        }
    }

    Ok(())
}
