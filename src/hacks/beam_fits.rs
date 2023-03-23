use std::path::PathBuf;

use crate::beam::{create_beam_object, Delays};
use crate::io::read::fits::*;
use crate::io::write::fits::*;
use crate::HyperdriveError;
use clap::Parser;
use marlu::RADec;

/// Print information on the dipole gains listed by a metafits file.
#[derive(Parser, Debug)]
pub struct BeamFitsArgs {
    #[clap()]
    beam_type: String,

    #[clap()]
    fits_from: PathBuf,

    #[clap()]
    fits_to: PathBuf,

    #[clap()]
    lst_rad: f64,

    #[clap()]
    latitude_deg: f64,
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
            lst_rad,
        } = self;
        let latitude_rad = latitude_deg.to_radians();

        let beam =
            create_beam_object(Some(beam_type.as_str()), 1, Delays::Partial(vec![0; 16])).unwrap();

        std::fs::copy(fits_from, fits_to).unwrap();

        let mut fptr = fits_edit(fits_to)?;
        let hdu = fits_open_hdu(&mut fptr, 0)?;

        let types = _fits_get_axis_types(&mut fptr, &hdu)?;
        dbg!(&types);
        if types[0] != "RA---SIN" || types[1] != "DEC--SIN" || types[2] != "FREQ" {
            panic!("axis types are not RA--SIN, DEC--SIN, FREQ")
        }

        let obsra: f64 = fits_get_required_key(&mut fptr, &hdu, "OBSRA")?;
        let obsdec: f64 = fits_get_required_key(&mut fptr, &hdu, "OBSDEC")?;
        let phase_centre = RADec::from_degrees(obsra, obsdec);
        let utc: f64 = fits_get_required_key(&mut fptr, &hdu, "UTC")?;

        let ras = _fits_get_axis_array(&mut fptr, &hdu, 1)?;
        let decs = _fits_get_axis_array(&mut fptr, &hdu, 2)?;
        let freqs = _fits_get_axis_array(&mut fptr, &hdu, 3)?;
        dbg!(&phase_centre, &utc, &ras[..10], &decs[..10], &freqs[..10]);

        // let epoch = Epoch::from_jde_utc(utc);

        let new_data =
            ndarray::Array3::from_shape_fn((freqs.len(), decs.len(), ras.len()), |(k, j, i)| {
                let radec = RADec::from_degrees(ras[i] as f64, decs[j] as f64);
                let azel = radec.to_hadec(*lst_rad).to_azel(latitude_rad);
                let jones = beam
                    .calc_jones(azel, freqs[k] as f64, None, latitude_rad as f64)
                    .unwrap();
                jones[0].norm() as f32
            });

        fits_write_image(&mut fptr, &hdu, &new_data.into_raw_vec())?;

        Ok(())
    }
}
