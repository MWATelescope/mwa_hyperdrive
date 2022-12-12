// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to generate sky-model visibilities with CUDA.

use std::{collections::HashSet, ffi::CStr};

use hifitime::{Duration, Epoch};
use log::debug;
use marlu::{
    pos::xyz::xyzs_to_cross_uvws,
    precession::{get_lmst, precess_time},
    Jones, LmnRime, RADec, XyzGeodetic, UVW,
};
use ndarray::prelude::*;
use rayon::prelude::*;

use super::ModelError;
use crate::{
    beam::{Beam, BeamCUDA},
    cuda::{self, CudaError, CudaFloat, CudaJones, DevicePointer},
    shapelets,
    srclist::{
        get_instrumental_flux_densities, ComponentType, FluxDensityType, ShapeletCoeff, Source,
        SourceList,
    },
};

use super::SkyModeller;

/// The first axis of `*_list_fds` is unflagged fine channel frequency, the
/// second is the source component. The length of `hadecs`, `lmns`,
/// `*_list_fds`'s second axis are the same.
pub(crate) struct SkyModellerCuda<'a> {
    cuda_beam: Box<dyn BeamCUDA>,
    /// The buffer for beam response Jones matrices.
    d_beam_jones: DevicePointer<CudaJones>,

    /// The phase centre used for all modelling.
    phase_centre: RADec,
    /// The longitude of the array we're using \[radians\].
    array_longitude: f64,
    /// The *unprecessed* latitude of the array we're using \[radians\]. If we
    /// are precessing, this latitude isn't used when calculating [`AzEl`]s.
    array_latitude: f64,
    /// The UT1 - UTC offset. If this is 0, effectively UT1 == UTC, which is a
    /// wrong assumption by up to 0.9s. We assume the this value does not change
    /// over the timestamps given to this `SkyModellerCuda`.
    dut1: Duration,
    /// Shift baselines, LSTs and array latitudes back to J2000.
    precess: bool,

    /// The buffer for UVW coordinates.
    pub(crate) d_uvws: DevicePointer<cuda::UVW>,
    /// The [`XyzGeodetic`] positions of each of the unflagged tiles.
    unflagged_tile_xyzs: &'a [XyzGeodetic],
    num_baselines: i32,
    num_freqs: i32,

    /// A simple map from an absolute tile index into an unflagged tile index.
    /// This is important because CUDA will use tile indices from 0 to the
    /// length of `unflagged_tile_xyzs`, but the beam code has dipole delays and
    /// dipole gains available for *all* tiles. So if tile 32 is flagged, any
    /// CUDA thread with a tile index of 32 would naively get the flagged beam
    /// info. This map would make tile index go to the next unflagged tile,
    /// perhaps 33.
    tile_index_to_unflagged_tile_index_map: DevicePointer<i32>,

    sbf_l: i32,
    sbf_n: i32,
    sbf_c: CudaFloat,
    sbf_dx: CudaFloat,

    pub(crate) d_vis: DevicePointer<Jones<f32>>,
    freqs: &'a [f64],
    d_freqs: DevicePointer<CudaFloat>,
    d_shapelet_basis_values: DevicePointer<CudaFloat>,

    point_power_law_radecs: Vec<RADec>,
    point_power_law_lmns: DevicePointer<cuda::LmnRime>,
    /// Instrumental flux densities calculated at 150 MHz.
    point_power_law_fds: DevicePointer<CudaJones>,
    /// Spectral indices.
    point_power_law_sis: DevicePointer<CudaFloat>,

    point_curved_power_law_radecs: Vec<RADec>,
    point_curved_power_law_lmns: DevicePointer<cuda::LmnRime>,
    point_curved_power_law_fds: DevicePointer<CudaJones>,
    point_curved_power_law_sis: DevicePointer<CudaFloat>,
    point_curved_power_law_qs: DevicePointer<CudaFloat>,

    point_list_radecs: Vec<RADec>,
    point_list_lmns: DevicePointer<cuda::LmnRime>,
    /// Instrumental (i.e. XX, XY, YX, XX).
    point_list_fds: DevicePointer<CudaJones>,

    gaussian_power_law_radecs: Vec<RADec>,
    gaussian_power_law_lmns: DevicePointer<cuda::LmnRime>,
    /// Instrumental flux densities calculated at 150 MHz.
    gaussian_power_law_fds: DevicePointer<CudaJones>,
    /// Spectral indices.
    gaussian_power_law_sis: DevicePointer<CudaFloat>,
    gaussian_power_law_gps: DevicePointer<cuda::GaussianParams>,

    gaussian_curved_power_law_radecs: Vec<RADec>,
    gaussian_curved_power_law_lmns: DevicePointer<cuda::LmnRime>,
    gaussian_curved_power_law_fds: DevicePointer<CudaJones>,
    gaussian_curved_power_law_sis: DevicePointer<CudaFloat>,
    gaussian_curved_power_law_qs: DevicePointer<CudaFloat>,
    gaussian_curved_power_law_gps: DevicePointer<cuda::GaussianParams>,

    gaussian_list_radecs: Vec<RADec>,
    gaussian_list_lmns: DevicePointer<cuda::LmnRime>,
    /// Instrumental (i.e. XX, XY, YX, XX).
    gaussian_list_fds: DevicePointer<CudaJones>,
    gaussian_list_gps: DevicePointer<cuda::GaussianParams>,

    shapelet_power_law_radecs: Vec<RADec>,
    shapelet_power_law_lmns: DevicePointer<cuda::LmnRime>,
    /// Instrumental flux densities calculated at 150 MHz.
    shapelet_power_law_fds: DevicePointer<CudaJones>,
    /// Spectral indices.
    shapelet_power_law_sis: DevicePointer<CudaFloat>,
    shapelet_power_law_gps: DevicePointer<cuda::GaussianParams>,
    shapelet_power_law_coeffs: DevicePointer<cuda::ShapeletCoeff>,
    shapelet_power_law_coeff_lens: DevicePointer<i32>,

    shapelet_curved_power_law_radecs: Vec<RADec>,
    shapelet_curved_power_law_lmns: DevicePointer<cuda::LmnRime>,
    shapelet_curved_power_law_fds: DevicePointer<CudaJones>,
    shapelet_curved_power_law_sis: DevicePointer<CudaFloat>,
    shapelet_curved_power_law_qs: DevicePointer<CudaFloat>,
    shapelet_curved_power_law_gps: DevicePointer<cuda::GaussianParams>,
    shapelet_curved_power_law_coeffs: DevicePointer<cuda::ShapeletCoeff>,
    shapelet_curved_power_law_coeff_lens: DevicePointer<i32>,

    shapelet_list_radecs: Vec<RADec>,
    shapelet_list_lmns: DevicePointer<cuda::LmnRime>,
    /// Instrumental (i.e. XX, XY, YX, XX).
    shapelet_list_fds: DevicePointer<CudaJones>,
    shapelet_list_gps: DevicePointer<cuda::GaussianParams>,
    shapelet_list_coeffs: DevicePointer<cuda::ShapeletCoeff>,
    shapelet_list_coeff_lens: DevicePointer<i32>,
}

// You're not re-using pointers after they've been sent to another thread,
// right?
unsafe impl<'a> Send for SkyModellerCuda<'a> {}

impl<'a> SkyModellerCuda<'a> {
    /// Given a source list, split the components into each component type (e.g.
    /// points, shapelets) and by each flux density type (e.g. list, power law),
    /// then copy them to a GPU ready for modelling. Where possible, list flux
    /// density types should be converted to power laws before calling this
    /// function, because using power laws is more efficient and probably more
    /// accurate.
    ///
    /// # Safety
    ///
    /// This function interfaces directly with the CUDA API. Rust errors attempt
    /// to catch problems but there are no guarantees.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn new(
        beam: &dyn Beam,
        source_list: &SourceList,
        unflagged_tile_xyzs: &'a [XyzGeodetic],
        unflagged_fine_chan_freqs: &'a [f64],
        flagged_tiles: &HashSet<usize>,
        phase_centre: RADec,
        array_longitude_rad: f64,
        array_latitude_rad: f64,
        dut1: Duration,
        apply_precession: bool,
    ) -> Result<SkyModellerCuda<'a>, ModelError> {
        // Variables for CUDA. They're made flexible in their types for
        // whichever precision is being used in the CUDA code.
        let (unflagged_fine_chan_freqs_ints, unflagged_fine_chan_freqs_floats): (Vec<_>, Vec<_>) =
            unflagged_fine_chan_freqs
                .iter()
                .map(|&f| (f as u32, f as CudaFloat))
                .unzip();
        let shapelet_basis_values: Vec<CudaFloat> = shapelets::SHAPELET_BASIS_VALUES
            .iter()
            .map(|&f| f as CudaFloat)
            .collect();

        let num_baselines = (unflagged_tile_xyzs.len() * (unflagged_tile_xyzs.len() - 1)) / 2;
        let num_freqs = unflagged_fine_chan_freqs.len();

        let mut d_vis =
            DevicePointer::malloc(num_baselines * num_freqs * std::mem::size_of::<Jones<f32>>())?;
        // Ensure the visibilities are zero'd.
        d_vis.clear();
        let d_freqs = DevicePointer::copy_to_device(&unflagged_fine_chan_freqs_floats)?;

        let mut tile_index_to_unflagged_tile_index_map: Vec<i32> =
            Vec::with_capacity(unflagged_tile_xyzs.len());
        let mut i_unflagged_tile = 0;
        for i_tile in 0..unflagged_tile_xyzs.len() + flagged_tiles.len() {
            if flagged_tiles.contains(&i_tile) {
                i_unflagged_tile += 1;
                continue;
            }
            tile_index_to_unflagged_tile_index_map.push(i_unflagged_tile);
            i_unflagged_tile += 1;
        }
        let d_tile_index_to_unflagged_tile_index_map =
            DevicePointer::copy_to_device(&tile_index_to_unflagged_tile_index_map)?;

        let mut modeller = SkyModellerCuda {
            cuda_beam: beam.prepare_cuda_beam(&unflagged_fine_chan_freqs_ints)?,
            d_beam_jones: DevicePointer::default(),

            phase_centre,
            array_longitude: array_longitude_rad,
            array_latitude: array_latitude_rad,
            dut1,
            precess: apply_precession,

            d_uvws: DevicePointer::default(),
            unflagged_tile_xyzs,
            num_baselines: num_baselines.try_into().expect("not too large"),
            num_freqs: num_freqs.try_into().expect("not too large"),

            tile_index_to_unflagged_tile_index_map: d_tile_index_to_unflagged_tile_index_map,

            sbf_l: shapelets::SBF_L
                .try_into()
                .expect("is positive and not too big"),
            sbf_n: shapelets::SBF_N
                .try_into()
                .expect("is positive and not too big"),
            sbf_c: shapelets::SBF_C as CudaFloat,
            sbf_dx: shapelets::SBF_DX as CudaFloat,

            d_vis,
            freqs: unflagged_fine_chan_freqs,
            d_freqs,
            d_shapelet_basis_values: DevicePointer::copy_to_device(&shapelet_basis_values)?,

            point_power_law_radecs: vec![],
            point_power_law_lmns: DevicePointer::default(),
            point_power_law_fds: DevicePointer::default(),
            point_power_law_sis: DevicePointer::default(),

            point_curved_power_law_radecs: vec![],
            point_curved_power_law_lmns: DevicePointer::default(),
            point_curved_power_law_fds: DevicePointer::default(),
            point_curved_power_law_sis: DevicePointer::default(),
            point_curved_power_law_qs: DevicePointer::default(),

            point_list_radecs: vec![],
            point_list_lmns: DevicePointer::default(),
            point_list_fds: DevicePointer::default(),

            gaussian_power_law_radecs: vec![],
            gaussian_power_law_lmns: DevicePointer::default(),
            gaussian_power_law_fds: DevicePointer::default(),
            gaussian_power_law_sis: DevicePointer::default(),
            gaussian_power_law_gps: DevicePointer::default(),

            gaussian_curved_power_law_radecs: vec![],
            gaussian_curved_power_law_lmns: DevicePointer::default(),
            gaussian_curved_power_law_fds: DevicePointer::default(),
            gaussian_curved_power_law_sis: DevicePointer::default(),
            gaussian_curved_power_law_qs: DevicePointer::default(),
            gaussian_curved_power_law_gps: DevicePointer::default(),

            gaussian_list_radecs: vec![],
            gaussian_list_lmns: DevicePointer::default(),
            gaussian_list_fds: DevicePointer::default(),
            gaussian_list_gps: DevicePointer::default(),

            shapelet_power_law_radecs: vec![],
            shapelet_power_law_lmns: DevicePointer::default(),
            shapelet_power_law_fds: DevicePointer::default(),
            shapelet_power_law_sis: DevicePointer::default(),
            shapelet_power_law_gps: DevicePointer::default(),
            shapelet_power_law_coeffs: DevicePointer::default(),
            shapelet_power_law_coeff_lens: DevicePointer::default(),

            shapelet_curved_power_law_radecs: vec![],
            shapelet_curved_power_law_lmns: DevicePointer::default(),
            shapelet_curved_power_law_fds: DevicePointer::default(),
            shapelet_curved_power_law_sis: DevicePointer::default(),
            shapelet_curved_power_law_qs: DevicePointer::default(),
            shapelet_curved_power_law_gps: DevicePointer::default(),
            shapelet_curved_power_law_coeffs: DevicePointer::default(),
            shapelet_curved_power_law_coeff_lens: DevicePointer::default(),

            shapelet_list_radecs: vec![],
            shapelet_list_lmns: DevicePointer::default(),
            shapelet_list_fds: DevicePointer::default(),
            shapelet_list_gps: DevicePointer::default(),
            shapelet_list_coeffs: DevicePointer::default(),
            shapelet_list_coeff_lens: DevicePointer::default(),
        };
        modeller.update_source_list(source_list, phase_centre)?;
        Ok(modeller)
    }

    /// This function is mostly used for testing. For a single timestep, over
    /// the already-provided baselines and frequencies, generate visibilities
    /// for each specified sky-model point-source component. The
    /// `SkyModellerCuda` object *must* already have its UVW coordinates set;
    /// see [`SkyModellerCuda::set_uvws`].
    ///
    /// `lst_rad`: The local sidereal time in \[radians\].
    ///
    /// `array_latitude_rad`: The latitude of the array/telescope/interferometer
    /// in \[radians\].
    pub(super) unsafe fn model_points(
        &mut self,
        lst_rad: f64,
        array_latitude_rad: f64,
    ) -> Result<(), ModelError> {
        if self.point_power_law_radecs.is_empty()
            && self.point_curved_power_law_radecs.is_empty()
            && self.point_list_radecs.is_empty()
        {
            return Ok(());
        }

        {
            let (azs, zas): (Vec<CudaFloat>, Vec<CudaFloat>) = self
                .point_power_law_radecs
                .iter()
                .chain(self.point_curved_power_law_radecs.iter())
                .chain(self.point_list_radecs.iter())
                .map(|radec| {
                    let azel = radec.to_hadec(lst_rad).to_azel(array_latitude_rad);
                    (azel.az as CudaFloat, azel.za() as CudaFloat)
                })
                .unzip();
            self.d_beam_jones.realloc(
                self.cuda_beam.get_num_unique_tiles() as usize
                    * self.cuda_beam.get_num_unique_freqs() as usize
                    * azs.len()
                    * std::mem::size_of::<CudaJones>(),
            )?;
            self.cuda_beam.calc_jones_pair(
                &azs,
                &zas,
                array_latitude_rad,
                self.d_beam_jones.get_mut().cast(),
            )?;
        }

        let error_message_ptr = cuda::model_points(
            &cuda::Points {
                num_power_laws: self
                    .point_power_law_radecs
                    .len()
                    .try_into()
                    .expect("number not too big to fit into i32"),
                power_law_lmns: self.point_power_law_lmns.get(),
                power_law_fds: self.point_power_law_fds.get(),
                power_law_sis: self.point_power_law_sis.get(),
                num_curved_power_laws: self
                    .point_curved_power_law_radecs
                    .len()
                    .try_into()
                    .expect("number not too big to fit into i32"),
                curved_power_law_lmns: self.point_curved_power_law_lmns.get(),
                curved_power_law_fds: self.point_curved_power_law_fds.get(),
                curved_power_law_sis: self.point_curved_power_law_sis.get(),
                curved_power_law_qs: self.point_curved_power_law_qs.get(),
                num_lists: self
                    .point_list_radecs
                    .len()
                    .try_into()
                    .expect("number not too big to fit into i32"),
                list_lmns: self.point_list_lmns.get(),
                list_fds: self.point_list_fds.get(),
            },
            &self.get_addresses(),
            self.d_uvws.get(),
            self.d_beam_jones.get(),
        );
        if error_message_ptr.is_null() {
            Ok(())
        } else {
            // Get the CUDA error message associated with the enum variant.
            let error_message = CStr::from_ptr(error_message_ptr)
                .to_str()
                .unwrap_or("<cannot read CUDA error string>");
            let our_error_str = format!("{}:{}: model_points: {error_message}", file!(), line!());
            Err(CudaError::Kernel(our_error_str).into())
        }
    }

    /// This function is mostly used for testing. For a single timestep, over
    /// the already-provided baselines and frequencies, generate visibilities
    /// for each specified sky-model Gaussian-source component. The
    /// `SkyModellerCuda` object *must* already have its UVW coordinates set;
    /// see [`SkyModellerCuda::set_uvws`].
    ///
    /// `lst_rad`: The local sidereal time in \[radians\].
    ///
    /// `array_latitude_rad`: The latitude of the array/telescope/interferometer
    /// in \[radians\].
    pub(super) unsafe fn model_gaussians(
        &mut self,
        lst_rad: f64,
        array_latitude_rad: f64,
    ) -> Result<(), ModelError> {
        if self.gaussian_power_law_radecs.is_empty()
            && self.gaussian_curved_power_law_radecs.is_empty()
            && self.gaussian_list_radecs.is_empty()
        {
            return Ok(());
        }

        {
            let (azs, zas): (Vec<CudaFloat>, Vec<CudaFloat>) = self
                .gaussian_power_law_radecs
                .iter()
                .chain(self.gaussian_curved_power_law_radecs.iter())
                .chain(self.gaussian_list_radecs.iter())
                .map(|radec| {
                    let azel = radec.to_hadec(lst_rad).to_azel(array_latitude_rad);
                    (azel.az as CudaFloat, azel.za() as CudaFloat)
                })
                .unzip();
            self.d_beam_jones.realloc(
                self.cuda_beam.get_num_unique_tiles() as usize
                    * self.cuda_beam.get_num_unique_freqs() as usize
                    * azs.len()
                    * std::mem::size_of::<CudaJones>(),
            )?;
            self.cuda_beam.calc_jones_pair(
                &azs,
                &zas,
                array_latitude_rad,
                self.d_beam_jones.get_mut().cast(),
            )?;
        }

        let error_message_ptr = cuda::model_gaussians(
            &cuda::Gaussians {
                num_power_laws: self
                    .gaussian_power_law_radecs
                    .len()
                    .try_into()
                    .expect("number not too big to fit into i32"),
                power_law_lmns: self.gaussian_power_law_lmns.get(),
                power_law_fds: self.gaussian_power_law_fds.get(),
                power_law_sis: self.gaussian_power_law_sis.get(),
                power_law_gps: self.gaussian_power_law_gps.get(),
                num_curved_power_laws: self
                    .gaussian_curved_power_law_radecs
                    .len()
                    .try_into()
                    .expect("number not too big to fit into i32"),
                curved_power_law_lmns: self.gaussian_curved_power_law_lmns.get(),
                curved_power_law_fds: self.gaussian_curved_power_law_fds.get(),
                curved_power_law_sis: self.gaussian_curved_power_law_sis.get(),
                curved_power_law_qs: self.gaussian_curved_power_law_qs.get(),
                curved_power_law_gps: self.gaussian_curved_power_law_gps.get(),
                num_lists: self
                    .gaussian_list_radecs
                    .len()
                    .try_into()
                    .expect("number not too big to fit into i32"),
                list_lmns: self.gaussian_list_lmns.get(),
                list_fds: self.gaussian_list_fds.get(),
                list_gps: self.gaussian_list_gps.get(),
            },
            &self.get_addresses(),
            self.d_uvws.get(),
            self.d_beam_jones.get(),
        );
        if error_message_ptr.is_null() {
            Ok(())
        } else {
            // Get the CUDA error message associated with the enum variant.
            let error_message = CStr::from_ptr(error_message_ptr)
                .to_str()
                .unwrap_or("<cannot read CUDA error string>");
            let our_error_str =
                format!("{}:{}: model_gaussians: {error_message}", file!(), line!());
            Err(CudaError::Kernel(our_error_str).into())
        }
    }

    /// This function is mostly used for testing. For a single timestep, over
    /// the already-provided baselines and frequencies, generate visibilities
    /// for each specified sky-model Gaussian-source component. The
    /// `SkyModellerCuda` object *must* already have its UVW coordinates set;
    /// see [`SkyModellerCuda::set_uvws`].
    ///
    /// `lst_rad`: The local sidereal time in \[radians\].
    ///
    /// `array_latitude_rad`: The latitude of the array/telescope/interferometer
    /// in \[radians\].
    pub(super) unsafe fn model_shapelets(
        &mut self,
        lst_rad: f64,
        array_latitude_rad: f64,
    ) -> Result<(), ModelError> {
        if self.shapelet_power_law_radecs.is_empty()
            && self.shapelet_curved_power_law_radecs.is_empty()
            && self.shapelet_list_radecs.is_empty()
        {
            return Ok(());
        }

        {
            let (azs, zas): (Vec<CudaFloat>, Vec<CudaFloat>) = self
                .shapelet_power_law_radecs
                .iter()
                .chain(self.shapelet_curved_power_law_radecs.iter())
                .chain(self.shapelet_list_radecs.iter())
                .map(|radec| {
                    let azel = radec.to_hadec(lst_rad).to_azel(array_latitude_rad);
                    (azel.az as CudaFloat, azel.za() as CudaFloat)
                })
                .unzip();
            self.d_beam_jones.realloc(
                self.cuda_beam.get_num_unique_tiles() as usize
                    * self.cuda_beam.get_num_unique_freqs() as usize
                    * azs.len()
                    * std::mem::size_of::<CudaJones>(),
            )?;
            self.cuda_beam.calc_jones_pair(
                &azs,
                &zas,
                array_latitude_rad,
                self.d_beam_jones.get_mut().cast(),
            )?
        };
        cuda::peek_and_sync(cuda::CudaCall::CopyFromDevice)?;
        let uvs = self.get_shapelet_uvs(lst_rad);
        let power_law_uvs =
            DevicePointer::copy_to_device(uvs.power_law.as_slice().expect("is contiguous"))?;
        let curved_power_law_uvs =
            DevicePointer::copy_to_device(uvs.curved_power_law.as_slice().expect("is contiguous"))?;
        let list_uvs = DevicePointer::copy_to_device(uvs.list.as_slice().expect("is contiguous"))?;

        let error_message_ptr = cuda::model_shapelets(
            &cuda::Shapelets {
                num_power_laws: self
                    .shapelet_power_law_radecs
                    .len()
                    .try_into()
                    .expect("number not too big to fit into i32"),
                power_law_lmns: self.shapelet_power_law_lmns.get(),
                power_law_fds: self.shapelet_power_law_fds.get(),
                power_law_sis: self.shapelet_power_law_sis.get(),
                power_law_gps: self.shapelet_power_law_gps.get(),
                power_law_shapelet_uvs: power_law_uvs.get(),
                power_law_shapelet_coeffs: self.shapelet_power_law_coeffs.get(),
                power_law_num_shapelet_coeffs: self.shapelet_power_law_coeff_lens.get(),
                num_curved_power_laws: self
                    .shapelet_curved_power_law_radecs
                    .len()
                    .try_into()
                    .expect("number not too big to fit into i32"),
                curved_power_law_lmns: self.shapelet_curved_power_law_lmns.get(),
                curved_power_law_fds: self.shapelet_curved_power_law_fds.get(),
                curved_power_law_sis: self.shapelet_curved_power_law_sis.get(),
                curved_power_law_qs: self.shapelet_curved_power_law_qs.get(),
                curved_power_law_gps: self.shapelet_curved_power_law_gps.get(),
                curved_power_law_shapelet_uvs: curved_power_law_uvs.get(),
                curved_power_law_shapelet_coeffs: self.shapelet_curved_power_law_coeffs.get(),
                curved_power_law_num_shapelet_coeffs: self
                    .shapelet_curved_power_law_coeff_lens
                    .get(),
                num_lists: self
                    .shapelet_list_radecs
                    .len()
                    .try_into()
                    .expect("number not too big to fit into i32"),
                list_lmns: self.shapelet_list_lmns.get(),
                list_fds: self.shapelet_list_fds.get(),
                list_gps: self.shapelet_list_gps.get(),
                list_shapelet_uvs: list_uvs.get(),
                list_shapelet_coeffs: self.shapelet_list_coeffs.get(),
                list_num_shapelet_coeffs: self.shapelet_list_coeff_lens.get(),
            },
            &self.get_addresses(),
            self.d_uvws.get(),
            self.d_beam_jones.get(),
        );
        if error_message_ptr.is_null() {
            Ok(())
        } else {
            // Get the CUDA error message associated with the enum variant.
            let error_message = CStr::from_ptr(error_message_ptr)
                .to_str()
                .unwrap_or("<cannot read CUDA error string>");
            let our_error_str =
                format!("{}:{}: model_shapelets: {error_message}", file!(), line!());
            Err(CudaError::Kernel(our_error_str).into())
        }
    }

    /// Get a populated [`cuda::Addresses`]. This should never outlive `self`.
    fn get_addresses(&mut self) -> cuda::Addresses {
        cuda::Addresses {
            num_freqs: self.num_freqs,
            num_vis: self.num_baselines * self.num_freqs,
            num_baselines: self.num_baselines,
            sbf_l: self.sbf_l,
            sbf_n: self.sbf_n,
            sbf_c: self.sbf_c,
            sbf_dx: self.sbf_dx,
            d_freqs: self.d_freqs.get(),
            d_shapelet_basis_values: self.d_shapelet_basis_values.get(),
            num_unique_beam_freqs: self.cuda_beam.get_num_unique_freqs(),
            d_tile_map: self.cuda_beam.get_tile_map(),
            d_freq_map: self.cuda_beam.get_freq_map(),
            d_tile_index_to_unflagged_tile_index_map: self
                .tile_index_to_unflagged_tile_index_map
                .get(),
            d_vis: self.d_vis.get_mut().cast(),
        }
    }

    /// Shapelets need their own special kind of UVW coordinates. Each shapelet
    /// component's position is treated as the phase centre. This function uses
    /// the FFI type [`cuda::ShapeletUV`]; the W isn't actually used in
    /// computation, and omitting it is hopefully a little more efficient.
    ///
    /// The returned arrays have baseline as the first axis and component as the
    /// second.
    pub(super) fn get_shapelet_uvs(&self, lst_rad: f64) -> ShapeletUVs {
        ShapeletUVs {
            power_law: get_shapelet_uvs_inner(
                &self.shapelet_power_law_radecs,
                lst_rad,
                self.unflagged_tile_xyzs,
            ),
            curved_power_law: get_shapelet_uvs_inner(
                &self.shapelet_curved_power_law_radecs,
                lst_rad,
                self.unflagged_tile_xyzs,
            ),
            list: get_shapelet_uvs_inner(
                &self.shapelet_list_radecs,
                lst_rad,
                self.unflagged_tile_xyzs,
            ),
        }
    }

    /// Copy the visibilities from the device and clear them.
    #[cfg(test)]
    pub(super) fn get_vis(&mut self, mut vis_model_slice: ArrayViewMut2<Jones<f32>>) {
        self.d_vis
            .copy_from_device(vis_model_slice.as_slice_mut().expect("is contiguous"))
            .unwrap();
        self.d_vis.clear();
    }

    #[cfg(test)]
    pub(super) fn set_uvws(&mut self, uvws: &[cuda::UVW]) -> Result<(), CudaError> {
        self.d_uvws.overwrite(uvws)
    }

    pub(crate) fn model_with_uvws(
        &mut self,
        mut vis_model_slice: ArrayViewMut2<Jones<f32>>,
        d_uvws: &DevicePointer<cuda::UVW>,
        lst: f64,
        latitude: f64,
    ) -> Result<(), ModelError> {
        self.d_uvws.ptr = d_uvws.ptr;
        unsafe {
            self.model_points(lst, latitude)?;
            self.model_gaussians(lst, latitude)?;
            self.model_shapelets(lst, latitude)?;

            self.d_vis
                .copy_from_device(vis_model_slice.as_slice_mut().expect("is contiguous"))?;
        }
        Ok(())
    }

    pub(crate) fn model_with_uvws2(
        &mut self,
        d_uvws: &DevicePointer<cuda::UVW>,
        lst: f64,
        latitude: f64,
    ) -> Result<(), ModelError> {
        let tmp = self.d_uvws.ptr;
        self.d_uvws.ptr = d_uvws.ptr;
        unsafe {
            self.model_points(lst, latitude)?;
            self.model_gaussians(lst, latitude)?;
            self.model_shapelets(lst, latitude)?;
        }
        self.d_uvws.ptr = tmp;
        Ok(())
    }

    pub(crate) fn model_with_uvws3(
        &mut self,
        d_vis: *mut Jones<f32>,
        d_uvws: &DevicePointer<cuda::UVW>,
        lst: f64,
        latitude: f64,
    ) -> Result<(), ModelError> {
        let tmp = self.d_vis.ptr;
        let tmp2 = self.d_uvws.ptr;
        self.d_vis.ptr = d_vis;
        self.d_uvws.ptr = d_uvws.ptr;
        unsafe {
            self.model_points(lst, latitude)?;
            self.model_gaussians(lst, latitude)?;
            self.model_shapelets(lst, latitude)?;
        }
        self.d_vis.ptr = tmp;
        self.d_uvws.ptr = tmp2;
        Ok(())
    }

    /// Clear the device visibilities.
    pub(crate) fn clear_vis(&mut self) {
        self.d_vis.clear();
    }
}

impl<'a> super::SkyModeller<'a> for SkyModellerCuda<'a> {
    fn update_source_list(
        &mut self,
        source_list: &SourceList,
        phase_centre: RADec,
    ) -> Result<(), ModelError> {
        self.phase_centre = phase_centre;

        self.point_power_law_radecs.clear();
        let mut point_power_law_lmns: Vec<cuda::LmnRime> = vec![];
        let mut point_power_law_fds: Vec<_> = vec![];
        let mut point_power_law_sis: Vec<_> = vec![];

        self.point_curved_power_law_radecs.clear();
        let mut point_curved_power_law_lmns: Vec<cuda::LmnRime> = vec![];
        let mut point_curved_power_law_fds: Vec<_> = vec![];
        let mut point_curved_power_law_sis: Vec<_> = vec![];
        let mut point_curved_power_law_qs: Vec<_> = vec![];

        self.point_list_radecs.clear();
        let mut point_list_lmns: Vec<cuda::LmnRime> = vec![];
        let mut point_list_fds: Vec<&FluxDensityType> = vec![];

        self.gaussian_power_law_radecs.clear();
        let mut gaussian_power_law_lmns: Vec<cuda::LmnRime> = vec![];
        let mut gaussian_power_law_fds: Vec<_> = vec![];
        let mut gaussian_power_law_sis: Vec<_> = vec![];
        let mut gaussian_power_law_gps: Vec<cuda::GaussianParams> = vec![];

        self.gaussian_curved_power_law_radecs.clear();
        let mut gaussian_curved_power_law_lmns: Vec<cuda::LmnRime> = vec![];
        let mut gaussian_curved_power_law_fds: Vec<_> = vec![];
        let mut gaussian_curved_power_law_sis: Vec<_> = vec![];
        let mut gaussian_curved_power_law_qs: Vec<_> = vec![];
        let mut gaussian_curved_power_law_gps: Vec<cuda::GaussianParams> = vec![];

        self.gaussian_list_radecs.clear();
        let mut gaussian_list_lmns: Vec<cuda::LmnRime> = vec![];
        let mut gaussian_list_fds: Vec<&FluxDensityType> = vec![];
        let mut gaussian_list_gps: Vec<cuda::GaussianParams> = vec![];

        self.shapelet_power_law_radecs.clear();
        let mut shapelet_power_law_lmns: Vec<cuda::LmnRime> = vec![];
        let mut shapelet_power_law_fds: Vec<_> = vec![];
        let mut shapelet_power_law_sis: Vec<_> = vec![];
        let mut shapelet_power_law_gps: Vec<cuda::GaussianParams> = vec![];
        let mut shapelet_power_law_coeffs: Vec<&[ShapeletCoeff]> = vec![];

        self.shapelet_curved_power_law_radecs.clear();
        let mut shapelet_curved_power_law_lmns: Vec<cuda::LmnRime> = vec![];
        let mut shapelet_curved_power_law_fds: Vec<_> = vec![];
        let mut shapelet_curved_power_law_sis: Vec<_> = vec![];
        let mut shapelet_curved_power_law_qs: Vec<_> = vec![];
        let mut shapelet_curved_power_law_gps: Vec<cuda::GaussianParams> = vec![];
        let mut shapelet_curved_power_law_coeffs: Vec<&[ShapeletCoeff]> = vec![];

        self.shapelet_list_radecs.clear();
        let mut shapelet_list_lmns: Vec<cuda::LmnRime> = vec![];
        let mut shapelet_list_fds: Vec<&FluxDensityType> = vec![];
        let mut shapelet_list_gps: Vec<cuda::GaussianParams> = vec![];
        let mut shapelet_list_coeffs: Vec<&[ShapeletCoeff]> = vec![];

        let jones_to_cuda_jones = |j: Jones<f64>| -> CudaJones {
            CudaJones {
                j00_re: j[0].re as CudaFloat,
                j00_im: j[0].im as CudaFloat,
                j01_re: j[1].re as CudaFloat,
                j01_im: j[1].im as CudaFloat,
                j10_re: j[2].re as CudaFloat,
                j10_im: j[2].im as CudaFloat,
                j11_re: j[3].re as CudaFloat,
                j11_im: j[3].im as CudaFloat,
            }
        };

        // Reverse the source list; if the source list has been sorted
        // (brightest sources first), reversing makes the dimmest sources get
        // used first. This is good because floating-point precision errors are
        // smaller when similar values are accumulated. Accumulating into a
        // float starting from the brightest component means that the
        // floating-point precision errors are greater as we work through the
        // source list.
        for comp in source_list
            .iter()
            .rev()
            .flat_map(|(_, src)| &src.components)
        {
            let radec = comp.radec;
            let LmnRime { l, m, n } = radec.to_lmn(phase_centre).prepare_for_rime();
            let lmn = cuda::LmnRime {
                l: l as CudaFloat,
                m: m as CudaFloat,
                n: n as CudaFloat,
            };
            match &comp.comp_type {
                ComponentType::Point => match comp.flux_type {
                    FluxDensityType::PowerLaw { si, .. } => {
                        self.point_power_law_radecs.push(radec);
                        point_power_law_lmns.push(lmn);
                        let fd_at_150mhz = comp.estimate_at_freq(cuda::POWER_LAW_FD_REF_FREQ as _);
                        let inst_fd: Jones<f64> = fd_at_150mhz.to_inst_stokes();
                        let cuda_inst_fd = jones_to_cuda_jones(inst_fd);
                        point_power_law_fds.push(cuda_inst_fd);
                        point_power_law_sis.push(si as CudaFloat);
                    }

                    FluxDensityType::CurvedPowerLaw { si, q, .. } => {
                        self.point_curved_power_law_radecs.push(radec);
                        point_curved_power_law_lmns.push(lmn);
                        let fd_at_150mhz = comp.estimate_at_freq(cuda::POWER_LAW_FD_REF_FREQ as _);
                        let inst_fd: Jones<f64> = fd_at_150mhz.to_inst_stokes();
                        let cuda_inst_fd = jones_to_cuda_jones(inst_fd);
                        point_curved_power_law_fds.push(cuda_inst_fd);
                        point_curved_power_law_qs.push(q as CudaFloat);
                        point_curved_power_law_sis.push(si as CudaFloat);
                    }

                    FluxDensityType::List { .. } => {
                        self.point_list_radecs.push(radec);
                        point_list_lmns.push(lmn);
                        point_list_fds.push(&comp.flux_type);
                    }
                },

                ComponentType::Gaussian { maj, min, pa } => {
                    let gp = cuda::GaussianParams {
                        maj: *maj as CudaFloat,
                        min: *min as CudaFloat,
                        pa: *pa as CudaFloat,
                    };
                    match comp.flux_type {
                        FluxDensityType::PowerLaw { si, .. } => {
                            self.gaussian_power_law_radecs.push(radec);
                            gaussian_power_law_lmns.push(lmn);
                            let fd_at_150mhz =
                                comp.estimate_at_freq(cuda::POWER_LAW_FD_REF_FREQ as _);
                            let inst_fd: Jones<f64> = fd_at_150mhz.to_inst_stokes();
                            let cuda_inst_fd = jones_to_cuda_jones(inst_fd);
                            gaussian_power_law_fds.push(cuda_inst_fd);
                            gaussian_power_law_sis.push(si as CudaFloat);
                            gaussian_power_law_gps.push(gp);
                        }

                        FluxDensityType::CurvedPowerLaw { si, q, .. } => {
                            self.gaussian_curved_power_law_radecs.push(radec);
                            gaussian_curved_power_law_lmns.push(lmn);
                            let fd_at_150mhz =
                                comp.estimate_at_freq(cuda::POWER_LAW_FD_REF_FREQ as _);
                            let inst_fd: Jones<f64> = fd_at_150mhz.to_inst_stokes();
                            let cuda_inst_fd = jones_to_cuda_jones(inst_fd);
                            gaussian_curved_power_law_fds.push(cuda_inst_fd);
                            gaussian_curved_power_law_qs.push(q as CudaFloat);
                            gaussian_curved_power_law_sis.push(si as CudaFloat);
                            gaussian_curved_power_law_gps.push(gp);
                        }

                        FluxDensityType::List { .. } => {
                            self.gaussian_list_radecs.push(radec);
                            gaussian_list_lmns.push(lmn);
                            gaussian_list_fds.push(&comp.flux_type);
                            gaussian_list_gps.push(gp);
                        }
                    };
                }

                ComponentType::Shapelet {
                    maj,
                    min,
                    pa,
                    coeffs,
                } => {
                    let gp = cuda::GaussianParams {
                        maj: *maj as CudaFloat,
                        min: *min as CudaFloat,
                        pa: *pa as CudaFloat,
                    };
                    match comp.flux_type {
                        FluxDensityType::PowerLaw { si, .. } => {
                            self.shapelet_power_law_radecs.push(radec);
                            shapelet_power_law_lmns.push(lmn);
                            let fd_at_150mhz = comp
                                .flux_type
                                .estimate_at_freq(cuda::POWER_LAW_FD_REF_FREQ as _);
                            let inst_fd: Jones<f64> = fd_at_150mhz.to_inst_stokes();
                            let cuda_inst_fd = jones_to_cuda_jones(inst_fd);
                            shapelet_power_law_fds.push(cuda_inst_fd);
                            shapelet_power_law_sis.push(si as CudaFloat);
                            shapelet_power_law_gps.push(gp);
                            shapelet_power_law_coeffs.push(coeffs);
                        }

                        FluxDensityType::CurvedPowerLaw { si, q, .. } => {
                            self.shapelet_curved_power_law_radecs.push(radec);
                            shapelet_curved_power_law_lmns.push(lmn);
                            let fd_at_150mhz =
                                comp.estimate_at_freq(cuda::POWER_LAW_FD_REF_FREQ as _);
                            let inst_fd: Jones<f64> = fd_at_150mhz.to_inst_stokes();
                            let cuda_inst_fd = jones_to_cuda_jones(inst_fd);
                            shapelet_curved_power_law_fds.push(cuda_inst_fd);
                            shapelet_curved_power_law_qs.push(q as CudaFloat);
                            shapelet_curved_power_law_sis.push(si as CudaFloat);
                            shapelet_curved_power_law_gps.push(gp);
                            shapelet_curved_power_law_coeffs.push(coeffs);
                        }

                        FluxDensityType::List { .. } => {
                            self.shapelet_list_radecs.push(radec);
                            shapelet_list_lmns.push(lmn);
                            shapelet_list_fds.push(&comp.flux_type);
                            shapelet_list_gps.push(gp);
                            shapelet_list_coeffs.push(coeffs);
                        }
                    }
                }
            }
        }

        let point_list_fds =
            get_instrumental_flux_densities(&point_list_fds, &self.freqs).mapv(jones_to_cuda_jones);
        let gaussian_list_fds = get_instrumental_flux_densities(&gaussian_list_fds, &self.freqs)
            .mapv(jones_to_cuda_jones);
        let shapelet_list_fds = get_instrumental_flux_densities(&shapelet_list_fds, &self.freqs)
            .mapv(jones_to_cuda_jones);

        let (shapelet_power_law_coeffs, shapelet_power_law_coeff_lens) =
            get_flattened_coeffs(shapelet_power_law_coeffs);
        let (shapelet_curved_power_law_coeffs, shapelet_curved_power_law_coeff_lens) =
            get_flattened_coeffs(shapelet_curved_power_law_coeffs);
        let (shapelet_list_coeffs, shapelet_list_coeff_lens) =
            get_flattened_coeffs(shapelet_list_coeffs);

        self.point_power_law_lmns.overwrite(&point_power_law_lmns)?;
        self.point_power_law_fds.overwrite(&point_power_law_fds)?;
        self.point_power_law_sis.overwrite(&point_power_law_sis)?;

        self.point_curved_power_law_lmns
            .overwrite(&point_curved_power_law_lmns)?;
        self.point_curved_power_law_fds
            .overwrite(&point_curved_power_law_fds)?;
        self.point_curved_power_law_sis
            .overwrite(&point_curved_power_law_sis)?;
        self.point_curved_power_law_qs
            .overwrite(&point_curved_power_law_qs)?;

        self.point_list_lmns.overwrite(&point_list_lmns)?;
        self.point_list_fds
            .overwrite(point_list_fds.as_slice().unwrap())?;

        self.gaussian_power_law_lmns
            .overwrite(&gaussian_power_law_lmns)?;
        self.gaussian_power_law_fds
            .overwrite(&gaussian_power_law_fds)?;
        self.gaussian_power_law_sis
            .overwrite(&gaussian_power_law_sis)?;
        self.gaussian_power_law_gps
            .overwrite(&gaussian_power_law_gps)?;

        self.gaussian_curved_power_law_lmns
            .overwrite(&gaussian_curved_power_law_lmns)?;
        self.gaussian_curved_power_law_fds
            .overwrite(&gaussian_curved_power_law_fds)?;
        self.gaussian_curved_power_law_sis
            .overwrite(&gaussian_curved_power_law_sis)?;
        self.gaussian_curved_power_law_qs
            .overwrite(&gaussian_curved_power_law_qs)?;
        self.gaussian_curved_power_law_gps
            .overwrite(&gaussian_curved_power_law_gps)?;

        self.gaussian_list_lmns.overwrite(&gaussian_list_lmns)?;
        self.gaussian_list_fds
            .overwrite(gaussian_list_fds.as_slice().unwrap())?;
        self.gaussian_list_gps.overwrite(&gaussian_list_gps)?;

        self.shapelet_power_law_lmns
            .overwrite(&shapelet_power_law_lmns)?;
        self.shapelet_power_law_fds
            .overwrite(&shapelet_power_law_fds)?;
        self.shapelet_power_law_sis
            .overwrite(&shapelet_power_law_sis)?;
        self.shapelet_power_law_gps
            .overwrite(&shapelet_power_law_gps)?;
        self.shapelet_power_law_coeffs
            .overwrite(&shapelet_power_law_coeffs)?;
        self.shapelet_power_law_coeff_lens
            .overwrite(&shapelet_power_law_coeff_lens)?;

        self.shapelet_curved_power_law_lmns
            .overwrite(&shapelet_curved_power_law_lmns)?;
        self.shapelet_curved_power_law_fds
            .overwrite(&shapelet_curved_power_law_fds)?;
        self.shapelet_curved_power_law_sis
            .overwrite(&shapelet_curved_power_law_sis)?;
        self.shapelet_curved_power_law_qs
            .overwrite(&shapelet_curved_power_law_qs)?;
        self.shapelet_curved_power_law_gps
            .overwrite(&shapelet_curved_power_law_gps)?;
        self.shapelet_curved_power_law_coeffs
            .overwrite(&shapelet_curved_power_law_coeffs)?;
        self.shapelet_curved_power_law_coeff_lens
            .overwrite(&shapelet_curved_power_law_coeff_lens)?;

        self.shapelet_list_lmns.overwrite(&shapelet_list_lmns)?;
        self.shapelet_list_fds
            .overwrite(shapelet_list_fds.as_slice().unwrap())?;
        self.shapelet_list_gps.overwrite(&shapelet_list_gps)?;
        self.shapelet_list_coeffs.overwrite(&shapelet_list_coeffs)?;
        self.shapelet_list_coeff_lens
            .overwrite(&shapelet_list_coeff_lens)?;

        Ok(())
    }

    // Ugh, is there a way to merge this with the above?
    fn update_with_a_source(
        &mut self,
        source: &Source,
        phase_centre: RADec,
    ) -> Result<(), ModelError> {
        self.phase_centre = phase_centre;

        self.point_power_law_radecs.clear();
        let mut point_power_law_lmns: Vec<cuda::LmnRime> = vec![];
        let mut point_power_law_fds: Vec<_> = vec![];
        let mut point_power_law_sis: Vec<_> = vec![];

        self.point_curved_power_law_radecs.clear();
        let mut point_curved_power_law_lmns: Vec<cuda::LmnRime> = vec![];
        let mut point_curved_power_law_fds: Vec<_> = vec![];
        let mut point_curved_power_law_sis: Vec<_> = vec![];
        let mut point_curved_power_law_qs: Vec<_> = vec![];

        self.point_list_radecs.clear();
        let mut point_list_lmns: Vec<cuda::LmnRime> = vec![];
        let mut point_list_fds: Vec<&FluxDensityType> = vec![];

        self.gaussian_power_law_radecs.clear();
        let mut gaussian_power_law_lmns: Vec<cuda::LmnRime> = vec![];
        let mut gaussian_power_law_fds: Vec<_> = vec![];
        let mut gaussian_power_law_sis: Vec<_> = vec![];
        let mut gaussian_power_law_gps: Vec<cuda::GaussianParams> = vec![];

        self.gaussian_curved_power_law_radecs.clear();
        let mut gaussian_curved_power_law_lmns: Vec<cuda::LmnRime> = vec![];
        let mut gaussian_curved_power_law_fds: Vec<_> = vec![];
        let mut gaussian_curved_power_law_sis: Vec<_> = vec![];
        let mut gaussian_curved_power_law_qs: Vec<_> = vec![];
        let mut gaussian_curved_power_law_gps: Vec<cuda::GaussianParams> = vec![];

        self.gaussian_list_radecs.clear();
        let mut gaussian_list_lmns: Vec<cuda::LmnRime> = vec![];
        let mut gaussian_list_fds: Vec<&FluxDensityType> = vec![];
        let mut gaussian_list_gps: Vec<cuda::GaussianParams> = vec![];

        self.shapelet_power_law_radecs.clear();
        let mut shapelet_power_law_lmns: Vec<cuda::LmnRime> = vec![];
        let mut shapelet_power_law_fds: Vec<_> = vec![];
        let mut shapelet_power_law_sis: Vec<_> = vec![];
        let mut shapelet_power_law_gps: Vec<cuda::GaussianParams> = vec![];
        let mut shapelet_power_law_coeffs: Vec<&[ShapeletCoeff]> = vec![];

        self.shapelet_curved_power_law_radecs.clear();
        let mut shapelet_curved_power_law_lmns: Vec<cuda::LmnRime> = vec![];
        let mut shapelet_curved_power_law_fds: Vec<_> = vec![];
        let mut shapelet_curved_power_law_sis: Vec<_> = vec![];
        let mut shapelet_curved_power_law_qs: Vec<_> = vec![];
        let mut shapelet_curved_power_law_gps: Vec<cuda::GaussianParams> = vec![];
        let mut shapelet_curved_power_law_coeffs: Vec<&[ShapeletCoeff]> = vec![];

        self.shapelet_list_radecs.clear();
        let mut shapelet_list_lmns: Vec<cuda::LmnRime> = vec![];
        let mut shapelet_list_fds: Vec<&FluxDensityType> = vec![];
        let mut shapelet_list_gps: Vec<cuda::GaussianParams> = vec![];
        let mut shapelet_list_coeffs: Vec<&[ShapeletCoeff]> = vec![];

        let jones_to_cuda_jones = |j: Jones<f64>| -> CudaJones {
            CudaJones {
                j00_re: j[0].re as CudaFloat,
                j00_im: j[0].im as CudaFloat,
                j01_re: j[1].re as CudaFloat,
                j01_im: j[1].im as CudaFloat,
                j10_re: j[2].re as CudaFloat,
                j10_im: j[2].im as CudaFloat,
                j11_re: j[3].re as CudaFloat,
                j11_im: j[3].im as CudaFloat,
            }
        };

        for comp in &source.components {
            let radec = comp.radec;
            let LmnRime { l, m, n } = radec.to_lmn(phase_centre).prepare_for_rime();
            let lmn = cuda::LmnRime {
                l: l as CudaFloat,
                m: m as CudaFloat,
                n: n as CudaFloat,
            };
            match &comp.comp_type {
                ComponentType::Point => match comp.flux_type {
                    FluxDensityType::PowerLaw { si, .. } => {
                        self.point_power_law_radecs.push(radec);
                        point_power_law_lmns.push(lmn);
                        let fd_at_150mhz = comp.estimate_at_freq(cuda::POWER_LAW_FD_REF_FREQ as _);
                        let inst_fd: Jones<f64> = fd_at_150mhz.to_inst_stokes();
                        let cuda_inst_fd = jones_to_cuda_jones(inst_fd);
                        point_power_law_fds.push(cuda_inst_fd);
                        point_power_law_sis.push(si as CudaFloat);
                    }

                    FluxDensityType::CurvedPowerLaw { si, q, .. } => {
                        self.point_curved_power_law_radecs.push(radec);
                        point_curved_power_law_lmns.push(lmn);
                        let fd_at_150mhz = comp.estimate_at_freq(cuda::POWER_LAW_FD_REF_FREQ as _);
                        let inst_fd: Jones<f64> = fd_at_150mhz.to_inst_stokes();
                        let cuda_inst_fd = jones_to_cuda_jones(inst_fd);
                        point_curved_power_law_fds.push(cuda_inst_fd);
                        point_curved_power_law_qs.push(q as CudaFloat);
                        point_curved_power_law_sis.push(si as CudaFloat);
                    }

                    FluxDensityType::List { .. } => {
                        self.point_list_radecs.push(radec);
                        point_list_lmns.push(lmn);
                        point_list_fds.push(&comp.flux_type);
                    }
                },

                ComponentType::Gaussian { maj, min, pa } => {
                    let gp = cuda::GaussianParams {
                        maj: *maj as CudaFloat,
                        min: *min as CudaFloat,
                        pa: *pa as CudaFloat,
                    };
                    match comp.flux_type {
                        FluxDensityType::PowerLaw { si, .. } => {
                            self.gaussian_power_law_radecs.push(radec);
                            gaussian_power_law_lmns.push(lmn);
                            let fd_at_150mhz =
                                comp.estimate_at_freq(cuda::POWER_LAW_FD_REF_FREQ as _);
                            let inst_fd: Jones<f64> = fd_at_150mhz.to_inst_stokes();
                            let cuda_inst_fd = jones_to_cuda_jones(inst_fd);
                            gaussian_power_law_fds.push(cuda_inst_fd);
                            gaussian_power_law_sis.push(si as CudaFloat);
                            gaussian_power_law_gps.push(gp);
                        }

                        FluxDensityType::CurvedPowerLaw { si, q, .. } => {
                            self.gaussian_curved_power_law_radecs.push(radec);
                            gaussian_curved_power_law_lmns.push(lmn);
                            let fd_at_150mhz =
                                comp.estimate_at_freq(cuda::POWER_LAW_FD_REF_FREQ as _);
                            let inst_fd: Jones<f64> = fd_at_150mhz.to_inst_stokes();
                            let cuda_inst_fd = jones_to_cuda_jones(inst_fd);
                            gaussian_curved_power_law_fds.push(cuda_inst_fd);
                            gaussian_curved_power_law_qs.push(q as CudaFloat);
                            gaussian_curved_power_law_sis.push(si as CudaFloat);
                            gaussian_curved_power_law_gps.push(gp);
                        }

                        FluxDensityType::List { .. } => {
                            self.gaussian_list_radecs.push(radec);
                            gaussian_list_lmns.push(lmn);
                            gaussian_list_fds.push(&comp.flux_type);
                            gaussian_list_gps.push(gp);
                        }
                    };
                }

                ComponentType::Shapelet {
                    maj,
                    min,
                    pa,
                    coeffs,
                } => {
                    let gp = cuda::GaussianParams {
                        maj: *maj as CudaFloat,
                        min: *min as CudaFloat,
                        pa: *pa as CudaFloat,
                    };
                    match comp.flux_type {
                        FluxDensityType::PowerLaw { si, .. } => {
                            self.shapelet_power_law_radecs.push(radec);
                            shapelet_power_law_lmns.push(lmn);
                            let fd_at_150mhz = comp
                                .flux_type
                                .estimate_at_freq(cuda::POWER_LAW_FD_REF_FREQ as _);
                            let inst_fd: Jones<f64> = fd_at_150mhz.to_inst_stokes();
                            let cuda_inst_fd = jones_to_cuda_jones(inst_fd);
                            shapelet_power_law_fds.push(cuda_inst_fd);
                            shapelet_power_law_sis.push(si as CudaFloat);
                            shapelet_power_law_gps.push(gp);
                            shapelet_power_law_coeffs.push(coeffs);
                        }

                        FluxDensityType::CurvedPowerLaw { si, q, .. } => {
                            self.shapelet_curved_power_law_radecs.push(radec);
                            shapelet_curved_power_law_lmns.push(lmn);
                            let fd_at_150mhz =
                                comp.estimate_at_freq(cuda::POWER_LAW_FD_REF_FREQ as _);
                            let inst_fd: Jones<f64> = fd_at_150mhz.to_inst_stokes();
                            let cuda_inst_fd = jones_to_cuda_jones(inst_fd);
                            shapelet_curved_power_law_fds.push(cuda_inst_fd);
                            shapelet_curved_power_law_qs.push(q as CudaFloat);
                            shapelet_curved_power_law_sis.push(si as CudaFloat);
                            shapelet_curved_power_law_gps.push(gp);
                            shapelet_curved_power_law_coeffs.push(coeffs);
                        }

                        FluxDensityType::List { .. } => {
                            self.shapelet_list_radecs.push(radec);
                            shapelet_list_lmns.push(lmn);
                            shapelet_list_fds.push(&comp.flux_type);
                            shapelet_list_gps.push(gp);
                            shapelet_list_coeffs.push(coeffs);
                        }
                    }
                }
            }
        }

        let point_list_fds =
            get_instrumental_flux_densities(&point_list_fds, &self.freqs).mapv(jones_to_cuda_jones);
        let gaussian_list_fds = get_instrumental_flux_densities(&gaussian_list_fds, &self.freqs)
            .mapv(jones_to_cuda_jones);
        let shapelet_list_fds = get_instrumental_flux_densities(&shapelet_list_fds, &self.freqs)
            .mapv(jones_to_cuda_jones);

        let (shapelet_power_law_coeffs, shapelet_power_law_coeff_lens) =
            get_flattened_coeffs(shapelet_power_law_coeffs);
        let (shapelet_curved_power_law_coeffs, shapelet_curved_power_law_coeff_lens) =
            get_flattened_coeffs(shapelet_curved_power_law_coeffs);
        let (shapelet_list_coeffs, shapelet_list_coeff_lens) =
            get_flattened_coeffs(shapelet_list_coeffs);

        self.point_power_law_lmns.overwrite(&point_power_law_lmns)?;
        self.point_power_law_fds.overwrite(&point_power_law_fds)?;
        self.point_power_law_sis.overwrite(&point_power_law_sis)?;

        self.point_curved_power_law_lmns
            .overwrite(&point_curved_power_law_lmns)?;
        self.point_curved_power_law_fds
            .overwrite(&point_curved_power_law_fds)?;
        self.point_curved_power_law_sis
            .overwrite(&point_curved_power_law_sis)?;
        self.point_curved_power_law_qs
            .overwrite(&point_curved_power_law_qs)?;

        self.point_list_lmns.overwrite(&point_list_lmns)?;
        self.point_list_fds
            .overwrite(point_list_fds.as_slice().unwrap())?;

        self.gaussian_power_law_lmns
            .overwrite(&gaussian_power_law_lmns)?;
        self.gaussian_power_law_fds
            .overwrite(&gaussian_power_law_fds)?;
        self.gaussian_power_law_sis
            .overwrite(&gaussian_power_law_sis)?;
        self.gaussian_power_law_gps
            .overwrite(&gaussian_power_law_gps)?;

        self.gaussian_curved_power_law_lmns
            .overwrite(&gaussian_curved_power_law_lmns)?;
        self.gaussian_curved_power_law_fds
            .overwrite(&gaussian_curved_power_law_fds)?;
        self.gaussian_curved_power_law_sis
            .overwrite(&gaussian_curved_power_law_sis)?;
        self.gaussian_curved_power_law_qs
            .overwrite(&gaussian_curved_power_law_qs)?;
        self.gaussian_curved_power_law_gps
            .overwrite(&gaussian_curved_power_law_gps)?;

        self.gaussian_list_lmns.overwrite(&gaussian_list_lmns)?;
        self.gaussian_list_fds
            .overwrite(gaussian_list_fds.as_slice().unwrap())?;
        self.gaussian_list_gps.overwrite(&gaussian_list_gps)?;

        self.shapelet_power_law_lmns
            .overwrite(&shapelet_power_law_lmns)?;
        self.shapelet_power_law_fds
            .overwrite(&shapelet_power_law_fds)?;
        self.shapelet_power_law_sis
            .overwrite(&shapelet_power_law_sis)?;
        self.shapelet_power_law_gps
            .overwrite(&shapelet_power_law_gps)?;
        self.shapelet_power_law_coeffs
            .overwrite(&shapelet_power_law_coeffs)?;
        self.shapelet_power_law_coeff_lens
            .overwrite(&shapelet_power_law_coeff_lens)?;

        self.shapelet_curved_power_law_lmns
            .overwrite(&shapelet_curved_power_law_lmns)?;
        self.shapelet_curved_power_law_fds
            .overwrite(&shapelet_curved_power_law_fds)?;
        self.shapelet_curved_power_law_sis
            .overwrite(&shapelet_curved_power_law_sis)?;
        self.shapelet_curved_power_law_qs
            .overwrite(&shapelet_curved_power_law_qs)?;
        self.shapelet_curved_power_law_gps
            .overwrite(&shapelet_curved_power_law_gps)?;
        self.shapelet_curved_power_law_coeffs
            .overwrite(&shapelet_curved_power_law_coeffs)?;
        self.shapelet_curved_power_law_coeff_lens
            .overwrite(&shapelet_curved_power_law_coeff_lens)?;

        self.shapelet_list_lmns.overwrite(&shapelet_list_lmns)?;
        self.shapelet_list_fds
            .overwrite(shapelet_list_fds.as_slice().unwrap())?;
        self.shapelet_list_gps.overwrite(&shapelet_list_gps)?;
        self.shapelet_list_coeffs.overwrite(&shapelet_list_coeffs)?;
        self.shapelet_list_coeff_lens
            .overwrite(&shapelet_list_coeff_lens)?;

        Ok(())
    }

    fn model_timestep(
        &mut self,
        mut vis_model_slice: ArrayViewMut2<Jones<f32>>,
        timestamp: Epoch,
    ) -> Result<Vec<UVW>, ModelError> {
        // Ensure the visibilities are zero'd.
        self.d_vis.clear();

        let (uvws, lst, latitude) = if self.precess {
            let precession_info = precess_time(
                self.array_longitude,
                self.array_latitude,
                self.phase_centre,
                timestamp,
                self.dut1,
            );
            // Apply precession to the tile XYZ positions.
            let precessed_tile_xyzs = precession_info.precess_xyz(self.unflagged_tile_xyzs);
            let uvws = xyzs_to_cross_uvws(
                &precessed_tile_xyzs,
                self.phase_centre.to_hadec(precession_info.lmst_j2000),
            );
            debug!(
                "Modelling GPS timestamp {}, LMST {}°, J2000 LMST {}°",
                timestamp.to_gpst_seconds(),
                precession_info.lmst.to_degrees(),
                precession_info.lmst_j2000.to_degrees()
            );
            (
                uvws,
                precession_info.lmst_j2000,
                precession_info.array_latitude_j2000,
            )
        } else {
            let lst = get_lmst(self.array_longitude, timestamp, self.dut1);
            let uvws =
                xyzs_to_cross_uvws(self.unflagged_tile_xyzs, self.phase_centre.to_hadec(lst));
            debug!(
                "Modelling GPS timestamp {}, LMST {}°",
                timestamp.to_gpst_seconds(),
                lst.to_degrees()
            );
            (uvws, lst, self.array_latitude)
        };

        let cuda_uvws: Vec<cuda::UVW> = uvws
            .iter()
            .map(|&uvw| cuda::UVW {
                u: uvw.u as CudaFloat,
                v: uvw.v as CudaFloat,
                w: uvw.w as CudaFloat,
            })
            .collect();

        unsafe {
            self.d_uvws.overwrite(&cuda_uvws)?;

            self.model_points(lst, latitude)?;
            self.model_gaussians(lst, latitude)?;
            self.model_shapelets(lst, latitude)?;

            self.d_vis
                .copy_from_device(vis_model_slice.as_slice_mut().expect("is contiguous"))?;
        }

        Ok(uvws)
    }
}

impl std::fmt::Debug for SkyModellerCuda<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SkyModellerCuda").finish()
    }
}

/// The return type of [SkyModellerCuda::get_shapelet_uvs]. These arrays have
/// baseline as the first axis and component as the second.
pub(super) struct ShapeletUVs {
    power_law: Array2<cuda::ShapeletUV>,
    curved_power_law: Array2<cuda::ShapeletUV>,
    pub(super) list: Array2<cuda::ShapeletUV>,
}

fn get_shapelet_uvs_inner(
    radecs: &[RADec],
    lst_rad: f64,
    tile_xyzs: &[XyzGeodetic],
) -> Array2<cuda::ShapeletUV> {
    let n = tile_xyzs.len();
    let num_baselines = (n * (n - 1)) / 2;

    let mut shapelet_uvs: Array2<cuda::ShapeletUV> = Array2::from_elem(
        (num_baselines, radecs.len()),
        cuda::ShapeletUV { u: 0.0, v: 0.0 },
    );
    shapelet_uvs
        .axis_iter_mut(Axis(1))
        .into_par_iter()
        .zip(radecs.par_iter())
        .for_each(|(mut baseline_uv, radec)| {
            let hadec = radec.to_hadec(lst_rad);
            let shapelet_uvs: Vec<cuda::ShapeletUV> = xyzs_to_cross_uvws(tile_xyzs, hadec)
                .into_iter()
                .map(|uvw| cuda::ShapeletUV {
                    u: uvw.u as CudaFloat,
                    v: uvw.v as CudaFloat,
                })
                .collect();
            baseline_uv.assign(&Array1::from(shapelet_uvs));
        });
    shapelet_uvs
}

/// There are a variable number of shapelet coefficients for each shapelet
/// component. To avoid excessive dereferencing on GPUs (expensive), this
/// method flattens the coefficients into a single array (lengths of the
/// array-of-arrays).
fn get_flattened_coeffs(
    shapelet_coeffs: Vec<&[ShapeletCoeff]>,
) -> (Vec<cuda::ShapeletCoeff>, Vec<i32>) {
    let mut coeffs: Vec<cuda::ShapeletCoeff> = vec![];
    let mut coeff_lengths = Vec::with_capacity(coeffs.len());

    for coeffs_for_comp in shapelet_coeffs {
        coeff_lengths.push(
            coeffs_for_comp
                .len()
                .try_into()
                .expect("number not too big to fit into i32"),
        );
        for coeff in coeffs_for_comp {
            coeffs.push(cuda::ShapeletCoeff {
                n1: coeff
                    .n1
                    .try_into()
                    .expect("number not too big to fit into i32"),
                n2: coeff
                    .n2
                    .try_into()
                    .expect("number not too big to fit into i32"),
                value: coeff.value as CudaFloat,
            })
        }
    }

    coeffs.shrink_to_fit();
    (coeffs, coeff_lengths)
}
