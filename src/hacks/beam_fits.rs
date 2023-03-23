use std::path::PathBuf;

use crate::beam::{create_beam_object, Delays};
use crate::io::read::fits::*;
use crate::io::write::fits::*;
use crate::HyperdriveError;
use clap::Parser;
use hifitime::{Epoch, Duration};
use marlu::RADec;
use marlu::constants::DAYSEC;
use marlu::precession::get_lmst;

/// Print information on the dipole gains listed by a metafits file.
#[derive(Parser, Debug)]
pub struct BeamFitsArgs {
    #[clap(long)]
    beam_type: String,

    #[clap()]
    fits_from: PathBuf,

    #[clap()]
    fits_to: PathBuf,

    #[clap(long, allow_hyphen_values = true, default_value="-26.825")]
    latitude_deg: f64,

    #[clap(long, allow_hyphen_values = true, default_value="116.764")]
    longitude_deg: f64,
}

impl BeamFitsArgs {
    pub fn run(&self) -> Result<(), HyperdriveError> {
        self.run_inner().unwrap();
        Ok(())
    }

    fn run_inner(&self) -> Result<(), FitsError> {
        let Self {
            beam_type,
            fits_from,
            fits_to,
            latitude_deg,
            longitude_deg,
        } = self;
        let latitude_rad = latitude_deg.to_radians();
        let longitude_rad = longitude_deg.to_radians();

        let beam =
            create_beam_object(Some(beam_type.as_str()), 1, Delays::Partial(vec![0; 16])).unwrap();

        std::fs::copy(fits_from, fits_to).unwrap();

        let mut fptr = fits_edit(fits_to)?;
        let hdu = fits_open_hdu(&mut fptr, 0)?;

        let types = _fits_get_axis_types(&mut fptr, &hdu)?;
        dbg!(&types);
        // let ra_axis = types.iter()
        if types[0] != "RA---SIN" || types[1] != "DEC--SIN" || types[2] != "FREQ" {
            panic!("axis types are not RA--SIN, DEC--SIN, FREQ")
        }

        let obsra: f64 = fits_get_required_key(&mut fptr, &hdu, "OBSRA")?;
        let obsdec: f64 = fits_get_required_key(&mut fptr, &hdu, "OBSDEC")?;
        let phase_centre = RADec::from_degrees(obsra, obsdec);

        let ras = _fits_get_axis_array(&mut fptr, &hdu, 1)?;
        let decs = _fits_get_axis_array(&mut fptr, &hdu, 2)?;
        let freqs = _fits_get_axis_array(&mut fptr, &hdu, 3)?;
        dbg!(&phase_centre, &ras[..10], &decs[..10], &freqs[..10]);

        let utc: f64 = fits_get_required_key(&mut fptr, &hdu, "UTC")?;
        let mjd_obs: f64 = fits_get_required_key(&mut fptr, &hdu, "MJD-OBS")?;
        // DATE is just when it was simulated, ignore this.
        // let date: String = fits_get_required_key(&mut fptr, &hdu, "DATE")?;
        let obs_epoch = Epoch::from_mjd_utc(mjd_obs);
        let epoch = Epoch::from_mjd_utc(utc / DAYSEC);
        let dut1 = Duration::from_seconds(0.);
        let mjd_utc_days = epoch.to_mjd_utc_days();
        let lst_rad = get_lmst(longitude_rad, epoch, dut1);
        dbg!(&utc, &mjd_obs, &obs_epoch, &epoch, &mjd_utc_days, &lst_rad);

        let new_data =
            ndarray::Array3::from_shape_fn((freqs.len(), decs.len(), ras.len()), |(k, j, i)| {
                let radec = RADec::from_degrees(ras[i] as f64, decs[j] as f64);
                let azel = radec.to_hadec(lst_rad).to_azel(latitude_rad);
                let jones = beam
                    .calc_jones(azel, freqs[k] as f64, None, lst_rad as f64)
                    .unwrap();
                jones[0].norm() as f32
            });

        dbg!("writing");

        fits_write_image(&mut fptr, &hdu, &new_data.into_raw_vec())?;

        Ok(())
    }
}
