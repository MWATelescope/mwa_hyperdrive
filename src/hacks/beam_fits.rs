use std::{path::PathBuf, sync::Arc};

use crate::{
    beam::{create_beam_object, Delays},
    io::{read::fits::*, write::fits::*},
    HyperdriveError,
};
use clap::Parser;
use hifitime::{Duration, Epoch};
use mapproj::{
    img2celestial::Img2Celestial,
    img2proj::{ImgXY2ProjXY, WcsImgXY2ProjXY},
    zenithal::sin::Sin,
    CenteredProjection, ImgXY, LonLat,
};
use marlu::{constants::DAYSEC, precession::get_lmst, RADec};

/// Print information on the dipole gains listed by a metafits file.
#[derive(Parser, Debug)]
pub struct BeamFitsArgs {
    #[clap(long)]
    beam_type: String,

    #[clap()]
    fits_from: PathBuf,

    #[clap()]
    fits_to: PathBuf,

    #[clap(long, allow_hyphen_values = true, default_value = "-26.825")]
    latitude_deg: f64,

    #[clap(long, allow_hyphen_values = true, default_value = "116.764")]
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
        dbg!(&phase_centre);

        let naxis1: usize = fits_get_required_key(&mut fptr, &hdu, "NAXIS1")?;
        let crpix1: f64 = fits_get_required_key(&mut fptr, &hdu, "CRPIX1")?;
        let crval1: f64 = fits_get_required_key(&mut fptr, &hdu, "CRVAL1")?;
        let cdelt1: f64 = fits_get_required_key(&mut fptr, &hdu, "CDELT1")?;
        println!(
            "axis1: n={} rp={} rv={} d={}",
            naxis1, crpix1, crval1, cdelt1
        );
        let ra_pixs: Vec<usize> = (1..=naxis1).into_iter().collect();
        println!(
            "ra_pixs: ({naxis1}) {},{}..{}..{},{}",
            ra_pixs[0],
            ra_pixs[1],
            ra_pixs[naxis1 / 2],
            ra_pixs[naxis1 - 2],
            ra_pixs[naxis1 - 1]
        );
        let ra_sins = _fits_get_axis_array(&mut fptr, &hdu, 1)?;
        println!(
            "ra_sins: ({naxis1}) {},{}..{}..{},{}",
            ra_sins[0],
            ra_sins[1],
            ra_sins[naxis1 / 2],
            ra_sins[naxis1 - 2],
            ra_sins[naxis1 - 1]
        );

        let naxis2: usize = fits_get_required_key(&mut fptr, &hdu, "NAXIS2")?;
        let crpix2: f64 = fits_get_required_key(&mut fptr, &hdu, "CRPIX2")?;
        let crval2: f64 = fits_get_required_key(&mut fptr, &hdu, "CRVAL2")?;
        let cdelt2: f64 = fits_get_required_key(&mut fptr, &hdu, "CDELT2")?;
        let crota2: f64 = fits_get_required_key(&mut fptr, &hdu, "CROTA2")?;
        println!(
            "axis2: n={} rp={} rv={} d={} rot={}",
            naxis2, crpix2, crval2, cdelt2, crota2
        );
        let dec_pixs: Vec<usize> = (1..=naxis2).into_iter().collect();
        println!(
            "dec_pixs: ({naxis2}) {},{}..{}..{},{}",
            dec_pixs[0],
            dec_pixs[1],
            dec_pixs[naxis2 / 2],
            dec_pixs[naxis2 - 2],
            dec_pixs[naxis2 - 1]
        );
        let dec_sins = _fits_get_axis_array(&mut fptr, &hdu, 2)?;
        println!(
            "dec_sins: ({naxis2}) {},{}..{}..{},{}",
            dec_sins[0],
            dec_sins[1],
            dec_sins[naxis2 / 2],
            dec_sins[naxis2 - 2],
            dec_sins[naxis2 - 1]
        );

        let naxis3: usize = fits_get_required_key(&mut fptr, &hdu, "NAXIS3")?;
        let crpix3: f64 = fits_get_required_key(&mut fptr, &hdu, "CRPIX3")?;
        let crval3: f64 = fits_get_required_key(&mut fptr, &hdu, "CRVAL3")?;
        let cdelt3: f64 = fits_get_required_key(&mut fptr, &hdu, "CDELT3")?;
        println!(
            "axis3: n={} rp={} rv={} d={}",
            naxis3, crpix3, crval3, cdelt3
        );
        let freqs = _fits_get_axis_array(&mut fptr, &hdu, 3)?;
        println!(
            "freqs: ({naxis3}) {},{}..{}..{},{}",
            freqs[0],
            freqs[1],
            freqs[naxis3 / 2],
            freqs[naxis3 - 2],
            freqs[naxis3 - 1]
        );

        let mut proj = CenteredProjection::new(Sin::default());
        let proj_center = LonLat::new(crval1.to_radians(), crval2.to_radians());
        proj.set_proj_center_from_lonlat(&proj_center);
        let img2proj = WcsImgXY2ProjXY::from_cd(crpix1, crpix2, cdelt1, 0., 0., cdelt2);
        // let img2proj = WcsImgXY2ProjXY::from_cr(crpix1, crpix2, crota2, cdelt1, cdelt2);
        let img2lonlat = Img2Celestial::new(img2proj.clone(), proj);

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

        // let projs = ndarray::Array2::from_shape_fn((naxis2, naxis1), |(j, i)| {
        //     let pix =ImgXY::new(ra_pixs[i] as f64, dec_pixs[j] as f64);
        //     let proj = img2proj.img2proj(&pix);
        //     (proj.x(), proj.y())
        // });
        // let (proj_0_0, proj_0_n, proj_n_0, proj_n_n) = (projs[[0,0]], projs[[0,naxis1-1]], projs[[naxis2-1,0]], projs[[naxis2-1,naxis1-1]]);
        // println!("projs[  0,  0]=(x={:5.3}, y={:5.3}) .. projs[ -1,  0]=(x={:5.3},y={:5.3})", proj_0_0.0, proj_0_0.1, proj_n_0.0, proj_n_0.1);
        // println!("projs[  0, -1]=(x={:5.3}, y={:5.3}) .. projs[ -1, -1]=(x={:5.3},y={:5.3})", proj_0_n.0, proj_0_n.1, proj_n_n.0, proj_n_n.1);

        // let lonlats = ndarray::Array2::from_shape_fn((naxis2, naxis1), |(j, i)| {
        //     let pix =ImgXY::new(ra_pixs[i] as f64, dec_pixs[j] as f64);
        //     let lonlat = img2lonlat.img2lonlat(&pix).unwrap();
        //     (lonlat.lon(), lonlat.lat())
        // });
        // let (lonlat_0_0, lonlat_0_n, lonlat_n_0, lonlat_n_n) = (lonlats[[0,0]], lonlats[[0,naxis1-1]], lonlats[[naxis2-1,0]], lonlats[[naxis2-1,naxis1-1]]);
        // println!("lonlats[  0,  0]=(lon={:5.3}, lat={:5.3}) .. lonlats[ -1,  0]=(lon={:5.3},lat={:5.3})", lonlat_0_0.0, lonlat_0_0.1, lonlat_n_0.0, lonlat_n_0.1);
        // println!("lonlats[  0, -1]=(lon={:5.3}, lat={:5.3}) .. lonlats[ -1, -1]=(lon={:5.3},lat={:5.3})", lonlat_0_n.0, lonlat_0_n.1, lonlat_n_n.0, lonlat_n_n.1);

        // let radecs = ndarray::Array2::from_shape_fn((naxis2, naxis1), |(j, i)| {
        //     let pix =ImgXY::new(ra_pixs[i] as f64, dec_pixs[j] as f64);
        //     let lonlat = img2lonlat.img2lonlat(&pix).unwrap();
        //     RADec::from_radians(lonlat.lon(), lonlat.lat())
        // });
        // let (radec_0_0, radec_0_n, radec_n_0, radec_n_n) = (radecs[[0,0]], radecs[[0,naxis1-1]], radecs[[naxis2-1,0]], radecs[[naxis2-1,naxis1-1]]);
        // println!("radecs[  0,  0]=(ra={:5.3}, dec={:5.3})=>(ra={:5.3}°, dec={:5.3}°) .. radecs[ -1,  0]=(ra={:5.3}, dec={:5.3})=>(ra={:5.3}°, dec={:5.3}°)", radec_0_0.ra, radec_0_0.dec, radec_0_0.ra.to_degrees(), radec_0_0.dec.to_degrees(), radec_n_0.ra, radec_n_0.dec, radec_n_0.ra.to_degrees(), radec_n_0.dec.to_degrees());
        // println!("radecs[  0, -1]=(ra={:5.3}, dec={:5.3})=>(ra={:5.3}°, dec={:5.3}°) .. radecs[ -1, -1]=(ra={:5.3}, dec={:5.3})=>(ra={:5.3}°, dec={:5.3}°)", radec_0_n.ra, radec_0_n.dec, radec_0_n.ra.to_degrees(), radec_0_n.dec.to_degrees(), radec_n_n.ra, radec_n_n.dec, radec_n_n.ra.to_degrees(), radec_n_n.dec.to_degrees());

        let azels = ndarray::Array2::from_shape_fn((naxis2, naxis1), |(j, i)| {
            let pix = ImgXY::new(ra_pixs[i] as f64, dec_pixs[j] as f64);
            let lonlat = img2lonlat.img2lonlat(&pix).unwrap();
            let radec = RADec::from_radians(lonlat.lon(), lonlat.lat());
            radec.to_hadec(lst_rad).to_azel(latitude_rad)
        });
        let (azel_0_0, azel_0_n, azel_n_0, azel_n_n) = (
            azels[[0, 0]],
            azels[[0, naxis1 - 1]],
            azels[[naxis2 - 1, 0]],
            azels[[naxis2 - 1, naxis1 - 1]],
        );
        println!(
            "azels[  0,  0]=(az={:5.3}, el={:5.3}) .. azels[ -1,  0]=(az={:5.3},el={:5.3})",
            azel_0_0.az, azel_0_0.el, azel_n_0.az, azel_n_0.el
        );
        println!(
            "azels[  0, -1]=(az={:5.3}, el={:5.3}) .. azels[ -1, -1]=(az={:5.3},el={:5.3})",
            azel_0_n.az, azel_0_n.el, azel_n_n.az, azel_n_n.el
        );

        let new_data = ndarray::Array3::from_shape_fn((naxis3, naxis2, naxis1), |(k, j, i)| {
            let azel = azels[(j, i)];
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
