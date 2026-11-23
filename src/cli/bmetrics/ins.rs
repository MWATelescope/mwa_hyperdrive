// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! An SSINS-compatible incoherent noise spectrum (INS).
//!
//! This follows `SSINS.INS` (<https://github.com/mwilensky768/SSINS>) built
//! from an `SS` read with `diff=True` and `spectrum_type="cross"`: each
//! cross-correlation is differenced in time, and the amplitudes of the
//! differences are averaged over the unflagged baselines. The result is written
//! as a pyuvdata `UVFlag` waterfall in metric mode, which is what
//! `INS.write(prefix, output_type="data")` writes to `<prefix>_SSINS_data.h5`,
//! so `SSINS.INS(path)` and `pyuvdata.UVFlag(path)` can read it.

use std::path::Path;

use hdf5::types::VarLenAscii;
use marlu::{
    erfa,
    hifitime::{Duration, Epoch},
    ndarray::prelude::*,
    rayon::prelude::*,
    Jones, LatLngHeight, XyzGeocentric, XyzGeodetic,
};

/// pyuvdata polarisation numbers in the order of the INS polarisation axis
/// (XX, YY, XY, YX).
pub(crate) const INS_POLS: [i32; 4] = [-5, -6, -7, -8];
/// For each INS polarisation, the index into a [`Jones`] (XX, XY, YX, YY).
const JONES_INDEX: [usize; 4] = [0, 3, 1, 2];

/// The sky-subtracted, baseline-averaged visibility amplitudes of the cross
/// correlations.
pub(crate) struct IncoherentNoiseSpectrum {
    /// Mean amplitude of the time-differenced visibilities over the unflagged
    /// cross-correlation baselines `[time pair, freq, pol]`, in the units of the
    /// input visibilities. Entries without any contributing baseline are
    /// infinite, as in SSINS.
    pub(crate) metric: Array3<f64>,
    /// The number of baselines that contributed to each entry of `metric`.
    pub(crate) weights: Array3<f64>,
    /// The centre of each differenced pair of timesteps.
    pub(crate) times: Vec<Epoch>,
}

impl IncoherentNoiseSpectrum {
    /// Make the spectrum of visibilities `vis_tfb` (time, freq, baseline),
    /// whose baselines are the tile pairs `ant_pairs`. A visibility differenced
    /// with a flagged one (`flags_tfb`) doesn't contribute. `timestamps` are the
    /// centroids of the timesteps.
    ///
    /// # Panics
    ///
    /// If the array dimensions don't agree or there are fewer than two
    /// timesteps.
    pub(crate) fn new(
        vis_tfb: ArrayView3<Jones<f32>>,
        flags_tfb: ArrayView3<bool>,
        ant_pairs: &[(usize, usize)],
        timestamps: &[Epoch],
    ) -> IncoherentNoiseSpectrum {
        let (num_times, num_freqs, num_baselines) = vis_tfb.dim();
        assert_eq!(flags_tfb.dim(), vis_tfb.dim());
        assert_eq!(ant_pairs.len(), num_baselines);
        assert_eq!(timestamps.len(), num_times);
        assert!(num_times >= 2, "an INS needs at least two timesteps");

        let crosses: Vec<usize> = ant_pairs
            .iter()
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .map(|(i, _)| i)
            .collect();

        let mut metric = Array3::zeros((num_times - 1, num_freqs, INS_POLS.len()));
        let mut weights = Array3::zeros(metric.dim());
        metric
            .outer_iter_mut()
            .into_par_iter()
            .zip(weights.outer_iter_mut())
            .enumerate()
            .for_each(|(t, (mut metric_fp, mut weights_fp))| {
                for f in 0..num_freqs {
                    let mut sums = [0.0; 4];
                    let mut count = 0.0;
                    for &b in &crosses {
                        if flags_tfb[(t, f, b)] || flags_tfb[(t + 1, f, b)] {
                            continue;
                        }
                        let diff = vis_tfb[(t + 1, f, b)] - vis_tfb[(t, f, b)];
                        for (sum, &j) in sums.iter_mut().zip(JONES_INDEX.iter()) {
                            *sum += f64::from(diff[j].norm());
                        }
                        count += 1.0;
                    }
                    for (p, sum) in sums.into_iter().enumerate() {
                        metric_fp[(f, p)] = if count > 0.0 {
                            sum / count
                        } else {
                            f64::INFINITY
                        };
                        weights_fp[(f, p)] = count;
                    }
                }
            });

        let times = timestamps
            .windows(2)
            .map(|w| w[0] + (w[1] - w[0]) / 2)
            .collect();
        IncoherentNoiseSpectrum {
            metric,
            weights,
            times,
        }
    }

    /// SSINS's mean-subtracted spectrum (`INS.metric_ms` with `order=0`):
    /// `(metric / time-mean(metric) - 1) * sqrt(N_bl / (4/pi - 1))`, the
    /// z-scores of a clean observation. Entries without baselines are NaN.
    #[cfg(test)]
    pub(crate) fn mean_subtracted(&self) -> Array3<f64> {
        let c = 4.0 / std::f64::consts::PI - 1.0;
        let mut ms = Array3::from_elem(self.metric.dim(), f64::NAN);
        for ((metric_t, weights_t), mut ms_t) in self
            .metric
            .axis_iter(Axis(1))
            .zip(self.weights.axis_iter(Axis(1)))
            .zip(ms.axis_iter_mut(Axis(1)))
        {
            // Each (freq, pol) separately, over time.
            for p in 0..metric_t.len_of(Axis(1)) {
                let (sum, weight_sum) = metric_t
                    .column(p)
                    .iter()
                    .zip(weights_t.column(p))
                    .filter(|(_, &w)| w > 0.0)
                    .fold((0.0, 0.0), |(s, ws), (&m, &w)| (s + m * w, ws + w));
                if weight_sum == 0.0 {
                    continue;
                }
                let mean = sum / weight_sum;
                marlu::ndarray::Zip::from(ms_t.column_mut(p))
                    .and(metric_t.column(p))
                    .and(weights_t.column(p))
                    .for_each(|ms, &m, &w| {
                        if w > 0.0 {
                            // weights / sqrt(C * weights_square), and every
                            // baseline has a weight of 1.
                            *ms = (m / mean - 1.0) * (w / c).sqrt();
                        }
                    });
            }
        }
        ms
    }

    /// Write the spectrum as a pyuvdata `UVFlag` HDF5 file (waterfall type,
    /// metric mode), like `INS.write(prefix, output_type="data")`.
    pub(crate) fn write_uvflag_h5(
        &self,
        path: &Path,
        freqs_hz: &[f64],
        channel_width_hz: f64,
        telescope: &InsTelescope,
    ) -> Result<(), hdf5::Error> {
        let (num_times, num_freqs, num_pols) = self.metric.dim();
        assert_eq!(freqs_hz.len(), num_freqs);

        let times_jd: Vec<f64> = self.times.iter().map(|e| e.to_jde_utc_days()).collect();
        let lsts_rad: Vec<f64> = self
            .times
            .iter()
            .map(|&e| apparent_lst(e, telescope.dut1, telescope.array_position.longitude_rad))
            .collect();

        let file = hdf5::File::create(path)?;
        let header = file.create_group("Header")?;
        let data = file.create_group("Data")?;

        write_str(&header, "type", "waterfall")?;
        write_str(&header, "mode", "metric")?;
        write_scalar(&header, "Ntimes", num_times as i64)?;
        write_scalar(&header, "Nfreqs", num_freqs as i64)?;
        write_scalar(&header, "Npols", num_pols as i64)?;
        write_scalar(&header, "Nspws", 1_i64)?;
        write_1d(&header, "time_array", &times_jd)?;
        write_1d(&header, "lst_array", &lsts_rad)?;
        write_1d(&header, "freq_array", freqs_hz)?;
        write_1d(&header, "channel_width", &vec![channel_width_hz; num_freqs])?;
        write_1d(&header, "spw_array", &[0_i64])?;
        write_1d(&header, "flex_spw_id_array", &vec![0_i64; num_freqs])?;
        write_1d(&header, "polarization_array", &INS_POLS[..num_pols])?;
        write_str(&header, "label", "")?;
        // SSINS checks the history for the spectrum type it was made from.
        write_str(
            &header,
            "history",
            &format!(
                "Written by mwa_hyperdrive bmetrics {}. Initialized spectrum_type:cross from visibility data. ",
                env!("CARGO_PKG_VERSION")
            ),
        )?;
        telescope.write(&header)?;

        let shape = [num_times, num_freqs, num_pols];
        let metric: Vec<f64> = self.metric.iter().copied().collect();
        write_nd(&data, "metric_array", &shape, &metric)?;
        let weights: Vec<f64> = self.weights.iter().copied().collect();
        write_nd(&data, "weights_array", &shape, &weights)?;
        // Every baseline has a weight of 1, so the sum of the squared weights is
        // the number of baselines.
        write_nd(&data, "weights_square_array", &shape, &weights)?;

        Ok(())
    }
}

/// The telescope metadata pyuvdata needs to check a `UVFlag`.
pub(crate) struct InsTelescope {
    pub(crate) name: String,
    pub(crate) array_position: LatLngHeight,
    pub(crate) dut1: Duration,
    pub(crate) antenna_names: Vec<String>,
    pub(crate) antenna_numbers: Vec<i32>,
    pub(crate) antenna_xyzs: Vec<XyzGeodetic>,
}

impl InsTelescope {
    fn write(&self, header: &hdf5::Group) -> Result<(), hdf5::Error> {
        let num_ants = self.antenna_names.len();
        assert_eq!(self.antenna_numbers.len(), num_ants);
        assert_eq!(self.antenna_xyzs.len(), num_ants);

        write_str(header, "telescope_name", &self.name)?;
        write_str(header, "instrument", &self.name)?;
        write_str(header, "telescope_frame", "itrs")?;
        write_scalar(
            header,
            "latitude",
            self.array_position.latitude_rad.to_degrees(),
        )?;
        write_scalar(
            header,
            "longitude",
            self.array_position.longitude_rad.to_degrees(),
        )?;
        write_scalar(header, "altitude", self.array_position.height_metres)?;
        write_scalar(header, "Nants_telescope", num_ants as i64)?;
        write_1d(header, "antenna_numbers", &self.antenna_numbers)?;
        write_strs(header, "antenna_names", &[num_ants], &self.antenna_names)?;

        // pyuvdata wants ITRF positions relative to the array position, i.e.
        // the geodetic XYZs rotated by the array longitude.
        let centre = XyzGeocentric::get_geocentric_vector(self.array_position);
        let positions: Vec<f64> = self
            .antenna_xyzs
            .iter()
            .flat_map(|xyz| {
                let g = xyz.to_geocentric(self.array_position);
                [g.x - centre.x, g.y - centre.y, g.z - centre.z]
            })
            .collect();
        write_nd(header, "antenna_positions", &[num_ants, 3], &positions)?;

        // MWA dipoles: X is east-west, Y north-south.
        write_scalar(header, "Nfeeds", 2_i64)?;
        let feeds: Vec<String> = (0..num_ants)
            .flat_map(|_| ["x".to_string(), "y".to_string()])
            .collect();
        write_strs(header, "feed_array", &[num_ants, 2], &feeds)?;
        let angles: Vec<f64> = (0..num_ants)
            .flat_map(|_| [std::f64::consts::FRAC_PI_2, 0.0])
            .collect();
        write_nd(header, "feed_angle", &[num_ants, 2], &angles)?;
        let mounts = vec!["phased".to_string(); num_ants];
        write_strs(header, "mount_type", &[num_ants], &mounts)?;
        Ok(())
    }
}

/// Local apparent sidereal time (IAU 2006/2000A, as astropy and pyuvdata use).
pub(crate) fn apparent_lst(time: Epoch, dut1: Duration, longitude_rad: f64) -> f64 {
    let ut1 = (time + dut1).to_jde_utc_days();
    let tt = time.to_jde_tt_days();
    let gast = erfa::time::gst06a(ut1, 0.0, tt, 0.0);
    (gast + longitude_rad).rem_euclid(std::f64::consts::TAU)
}

fn write_scalar<T: hdf5::H5Type>(group: &hdf5::Group, name: &str, value: T) -> hdf5::Result<()> {
    group.new_dataset::<T>().create(name)?.write_scalar(&value)
}

fn write_str(group: &hdf5::Group, name: &str, value: &str) -> hdf5::Result<()> {
    write_scalar(group, name, ascii(value)?)
}

fn write_1d<T: hdf5::H5Type>(group: &hdf5::Group, name: &str, values: &[T]) -> hdf5::Result<()> {
    group
        .new_dataset::<T>()
        .shape([values.len()])
        .create(name)?
        .write_raw(values)
}

fn write_nd<T: hdf5::H5Type>(
    group: &hdf5::Group,
    name: &str,
    shape: &[usize],
    values: &[T],
) -> hdf5::Result<()> {
    group
        .new_dataset::<T>()
        .shape(shape)
        .create(name)?
        .write_raw(values)
}

fn write_strs(
    group: &hdf5::Group,
    name: &str,
    shape: &[usize],
    values: &[String],
) -> hdf5::Result<()> {
    let values = values
        .iter()
        .map(|s| ascii(s))
        .collect::<hdf5::Result<Vec<_>>>()?;
    write_nd(group, name, shape, &values)
}

fn ascii(s: &str) -> hdf5::Result<VarLenAscii> {
    VarLenAscii::from_ascii(s).map_err(|e| hdf5::Error::from(e.to_string().as_str()))
}
