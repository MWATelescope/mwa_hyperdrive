use crate::{
    averaging::Timeblock,
    context::ObsContext,
    di_calibrate::calibrate_timeblock,
    gpu::{self, gpu_kernel_call, DevicePointer, GpuError, GpuFloat},
    model::{SkyModeller, SkyModellerGpu},
    srclist::SourceList,
    Chanblock, TileBaselineFlags,
};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use itertools::{izip, Itertools};
use log::{debug, info, trace, warn};
use marlu::{
    constants::VEL_C,
    pos::xyz::xyzs_to_cross_uvws,
    precession::{get_lmst, precess_time},
    Jones, RADec, XyzGeodetic, UVW,
};
use ndarray::prelude::*;
use rayon::prelude::*;

use super::{weights_average, IonoConsts, PeelError, PeelLoopParams, UV, W};

#[allow(clippy::too_many_arguments)]
pub(crate) fn peel_gpu(
    mut vis_residual_tfb: ArrayViewMut3<Jones<f32>>,
    vis_weights_tfb: ArrayView3<f32>,
    timeblock: &Timeblock,
    source_list: &SourceList,
    iono_consts: &mut [IonoConsts],
    source_weighted_positions: &[RADec],
    num_sources_to_peel: usize,
    di_max_iterations: u32,
    di_stop_threshold: f64,
    di_min_threshold: f64,
    peel_loop_params: &PeelLoopParams,
    chanblocks: &[Chanblock],
    low_res_lambdas_m: &[f64],
    obs_context: &ObsContext,
    tile_baseline_flags: &TileBaselineFlags,
    high_res_modeller: &mut SkyModellerGpu,
    no_precession: bool,
    multi_progress_bar: &MultiProgress,
) -> Result<(), PeelError> {
    let (num_loops, num_passes, convergence) = peel_loop_params.get();

    let array_position = obs_context.array_position;
    let dut1 = obs_context.dut1.unwrap_or_default();

    let all_fine_chan_lambdas_m = chanblocks
        .iter()
        .map(|c| VEL_C / c.freq)
        .collect::<Vec<_>>();

    let flagged_tiles = &tile_baseline_flags.flagged_tiles;
    let unflagged_tile_xyzs: Vec<XyzGeodetic> = obs_context
        .tile_xyzs
        .par_iter()
        .enumerate()
        .filter(|(tile_index, _)| !flagged_tiles.contains(tile_index))
        .map(|(_, xyz)| *xyz)
        .collect();

    let timestamps = &timeblock.timestamps;

    let num_timesteps = vis_residual_tfb.len_of(Axis(0));
    let num_tiles = unflagged_tile_xyzs.len();
    let num_cross_baselines = (num_tiles * (num_tiles - 1)) / 2;
    let num_high_res_chans = all_fine_chan_lambdas_m.len();
    let num_high_res_chans_spw = chanblocks.len();
    assert_eq!(
        num_high_res_chans, num_high_res_chans_spw,
        "chans from fine_chan_lambdas {} != chans in high_res_chanblocks (flagged+unflagged) {}",
        num_high_res_chans, num_high_res_chans_spw
    );

    let num_low_res_chans = low_res_lambdas_m.len();
    assert!(
        num_high_res_chans % num_low_res_chans == 0,
        "TODO: averaging can't deal with non-integer ratios. channels high {} low {}",
        num_high_res_chans,
        num_low_res_chans
    );

    let num_timesteps_i32: i32 = num_timesteps.try_into().expect("smaller than i32::MAX");
    let num_tiles_i32: i32 = num_tiles.try_into().expect("smaller than i32::MAX");
    let num_cross_baselines_i32: i32 = num_cross_baselines
        .try_into()
        .expect("smaller than i32::MAX");
    let num_high_res_chans_i32 = num_high_res_chans
        .try_into()
        .expect("smaller than i32::MAX");
    let num_low_res_chans_i32 = num_low_res_chans.try_into().expect("smaller than i32::MAX");

    let num_sources_to_iono_subtract = iono_consts.len();

    let (time_axis, freq_axis, baseline_axis) = (Axis(0), Axis(1), Axis(2));

    assert_eq!(vis_residual_tfb.len_of(time_axis), timestamps.len());
    assert_eq!(vis_weights_tfb.len_of(time_axis), timestamps.len());

    assert_eq!(vis_residual_tfb.len_of(baseline_axis), num_cross_baselines);
    assert_eq!(vis_weights_tfb.len_of(baseline_axis), num_cross_baselines);

    assert_eq!(vis_residual_tfb.len_of(freq_axis), num_high_res_chans);
    assert_eq!(vis_weights_tfb.len_of(freq_axis), num_high_res_chans);

    assert!(num_sources_to_iono_subtract <= source_list.len());

    if num_sources_to_iono_subtract == 0 {
        return Ok(());
    }

    let timestamps = &timeblock.timestamps;
    let peel_progress = multi_progress_bar.add(
        ProgressBar::new(num_sources_to_iono_subtract as _)
            .with_style(
                ProgressStyle::default_bar()
                    .template("{msg:17}: [{wide_bar:.blue}] {pos:2}/{len:2} sources ({elapsed_precise}<{eta_precise})").unwrap()
                    .progress_chars("=> "),
            )
            .with_position(0)
    );
    peel_progress.tick();

    macro_rules! pb_warn { ($($arg:tt)+) => (multi_progress_bar.suspend(|| warn!($($arg)+))) }
    macro_rules! pb_info { ($($arg:tt)+) => (multi_progress_bar.suspend(|| info!($($arg)+))) }
    macro_rules! pb_debug { ($($arg:tt)+) => (multi_progress_bar.suspend(|| debug!($($arg)+))) }
    macro_rules! pb_trace { ($($arg:tt)+) => (multi_progress_bar.suspend(|| trace!($($arg)+))) }

    let mut lmsts = vec![0.; timestamps.len()];
    let mut latitudes = vec![0.; timestamps.len()];
    let mut tile_xyzs_high_res = Array2::<XyzGeodetic>::default((timestamps.len(), num_tiles));
    let mut high_res_uvws = Array2::default((timestamps.len(), num_cross_baselines));
    let mut tile_uvs_high_res = Array2::<UV>::default((timestamps.len(), num_tiles));
    let mut tile_ws_high_res = Array2::<W>::default((timestamps.len(), num_tiles));

    // Pre-compute high-res tile UVs and Ws at observation phase centre.
    for (
        &time,
        lmst,
        latitude,
        mut tile_xyzs_high_res,
        mut high_res_uvws,
        mut tile_uvs_high_res,
        mut tile_ws_high_res,
    ) in izip!(
        timestamps.iter(),
        lmsts.iter_mut(),
        latitudes.iter_mut(),
        tile_xyzs_high_res.outer_iter_mut(),
        high_res_uvws.outer_iter_mut(),
        tile_uvs_high_res.outer_iter_mut(),
        tile_ws_high_res.outer_iter_mut(),
    ) {
        if !no_precession {
            let precession_info = precess_time(
                array_position.longitude_rad,
                array_position.latitude_rad,
                obs_context.phase_centre,
                time,
                dut1,
            );
            tile_xyzs_high_res
                .iter_mut()
                .zip_eq(&precession_info.precess_xyz(&unflagged_tile_xyzs))
                .for_each(|(a, b)| *a = *b);
            *lmst = precession_info.lmst_j2000;
            *latitude = precession_info.array_latitude_j2000;
        } else {
            tile_xyzs_high_res
                .iter_mut()
                .zip_eq(&unflagged_tile_xyzs)
                .for_each(|(a, b)| *a = *b);
            *lmst = get_lmst(array_position.longitude_rad, time, dut1);
            *latitude = array_position.latitude_rad;
        };
        let hadec_phase = obs_context.phase_centre.to_hadec(*lmst);
        let (s_ha, c_ha) = hadec_phase.ha.sin_cos();
        let (s_dec, c_dec) = hadec_phase.dec.sin_cos();
        let mut tile_uvws_high_res = vec![UVW::default(); num_tiles];
        for (tile_uvw, tile_uv, tile_w, &tile_xyz) in izip!(
            tile_uvws_high_res.iter_mut(),
            tile_uvs_high_res.iter_mut(),
            tile_ws_high_res.iter_mut(),
            tile_xyzs_high_res.iter(),
        ) {
            let uvw = UVW::from_xyz_inner(tile_xyz, s_ha, c_ha, s_dec, c_dec);
            *tile_uvw = uvw;
            *tile_uv = UV { u: uvw.u, v: uvw.v };
            *tile_w = W(uvw.w);
        }

        // The UVWs for every timestep will be the same (because the phase
        // centres are always the same). Make these ahead of time for
        // efficiency.
        let mut count = 0;
        for (i, t1) in tile_uvws_high_res.iter().enumerate() {
            for t2 in tile_uvws_high_res.iter().skip(i + 1) {
                high_res_uvws[count] = *t1 - *t2;
                count += 1;
            }
        }
    }

    // ////////////////// //
    // LOW RES PRECESSION //
    // ////////////////// //

    let average_timestamp = timeblock.median;
    let (average_lmst, _average_latitude, average_tile_xyzs) = if no_precession {
        let average_tile_xyzs =
            ArrayView2::from_shape((1, num_tiles), &unflagged_tile_xyzs).expect("correct shape");
        (
            get_lmst(array_position.longitude_rad, average_timestamp, dut1),
            array_position.latitude_rad,
            CowArray::from(average_tile_xyzs),
        )
    } else {
        let average_precession_info = precess_time(
            array_position.longitude_rad,
            array_position.latitude_rad,
            obs_context.phase_centre,
            average_timestamp,
            dut1,
        );
        let average_precessed_tile_xyzs = Array2::from_shape_vec(
            (1, num_tiles),
            average_precession_info.precess_xyz(&unflagged_tile_xyzs),
        )
        .expect("correct shape");

        (
            average_precession_info.lmst_j2000,
            average_precession_info.array_latitude_j2000,
            CowArray::from(average_precessed_tile_xyzs),
        )
    };

    let gpu_xyzs_high_res: Vec<_> = tile_xyzs_high_res
        .iter()
        .copied()
        .map(|XyzGeodetic { x, y, z }| gpu::XYZ {
            x: x as GpuFloat,
            y: y as GpuFloat,
            z: z as GpuFloat,
        })
        .collect();
    let d_xyzs = DevicePointer::copy_to_device(&gpu_xyzs_high_res)?;

    // temporary arrays for accumulation
    let vis_residual_low_res_fb: Array3<Jones<f32>> =
        Array3::zeros((1, num_low_res_chans, num_cross_baselines));
    let mut vis_weights_low_res_fb: Array3<f32> = Array3::zeros(vis_residual_low_res_fb.raw_dim());

    // The low-res weights only need to be populated once.
    weights_average(vis_weights_tfb.view(), vis_weights_low_res_fb.view_mut());

    let freq_average_factor: i32 = (all_fine_chan_lambdas_m.len() / num_low_res_chans)
        .try_into()
        .expect("smaller than i32::MAX");

    unsafe {
        let mut gpu_uvws: ArrayBase<ndarray::OwnedRepr<gpu::UVW>, Dim<[usize; 2]>> =
            Array2::default((num_timesteps, num_cross_baselines));
        gpu_uvws
            .outer_iter_mut()
            .zip(tile_xyzs_high_res.outer_iter())
            .zip(lmsts.iter())
            .for_each(|((mut gpu_uvws, xyzs), lmst)| {
                let phase_centre = obs_context.phase_centre.to_hadec(*lmst);
                let v = xyzs_to_cross_uvws(xyzs.as_slice().unwrap(), phase_centre)
                    .into_iter()
                    .map(|uvw| gpu::UVW {
                        u: uvw.u as GpuFloat,
                        v: uvw.v as GpuFloat,
                        w: uvw.w as GpuFloat,
                    })
                    .collect::<Vec<_>>();
                gpu_uvws.assign(&ArrayView1::from(&v));
            });
        let mut d_uvws_from = DevicePointer::copy_to_device(gpu_uvws.as_slice().unwrap())?;
        let mut d_uvws_to =
            DevicePointer::malloc(gpu_uvws.len() * std::mem::size_of::<gpu::UVW>())?;

        let gpu_lmsts: Vec<GpuFloat> = lmsts.iter().map(|l| *l as GpuFloat).collect();
        let d_lmsts = DevicePointer::copy_to_device(&gpu_lmsts)?;

        let gpu_lambdas: Vec<GpuFloat> = all_fine_chan_lambdas_m
            .iter()
            .map(|l| *l as GpuFloat)
            .collect();
        let d_lambdas = DevicePointer::copy_to_device(&gpu_lambdas)?;

        let gpu_xyzs_low_res: Vec<_> = average_tile_xyzs
            .iter()
            .copied()
            .map(|XyzGeodetic { x, y, z }| gpu::XYZ {
                x: x as GpuFloat,
                y: y as GpuFloat,
                z: z as GpuFloat,
            })
            .collect();
        let d_xyzs_low_res = DevicePointer::copy_to_device(&gpu_xyzs_low_res)?;

        let gpu_low_res_lambdas: Vec<GpuFloat> =
            low_res_lambdas_m.iter().map(|l| *l as GpuFloat).collect();
        let d_low_res_lambdas = DevicePointer::copy_to_device(&gpu_low_res_lambdas)?;

        let d_average_lmsts = DevicePointer::copy_to_device(&[average_lmst as GpuFloat])?;
        let mut d_low_res_uvws: DevicePointer<gpu::UVW> =
            DevicePointer::malloc(gpu_uvws.len() * std::mem::size_of::<gpu::UVW>())?;
        // Make the amount of elements in `d_iono_fits` a power of 2, for
        // efficiency.
        let mut d_iono_fits = {
            let min_size =
                num_cross_baselines * num_low_res_chans * std::mem::size_of::<Jones<f64>>();
            let n = (min_size as f64).log2().ceil() as u32;
            let size = 2_usize.pow(n);
            let mut d: DevicePointer<Jones<f64>> = DevicePointer::malloc(size).unwrap();
            d.clear();
            d
        };

        let mut d_high_res_vis_tfb =
            DevicePointer::copy_to_device(vis_residual_tfb.as_slice().unwrap())?;
        let mut d_high_res_resid_tfb =
            DevicePointer::copy_to_device(vis_residual_tfb.as_slice().unwrap())?;
        let d_high_res_weights_tfb =
            DevicePointer::copy_to_device(vis_weights_tfb.as_slice().unwrap())?;

        let mut d_low_res_resid_fb =
            DevicePointer::copy_to_device(vis_residual_low_res_fb.as_slice().unwrap())?;
        let d_low_res_weights_fb =
            DevicePointer::copy_to_device(vis_weights_low_res_fb.as_slice().unwrap())?;

        let mut d_high_res_model_tfb: DevicePointer<Jones<f32>> = DevicePointer::malloc(
            timestamps.len()
                * num_cross_baselines
                * all_fine_chan_lambdas_m.len()
                * std::mem::size_of::<Jones<f32>>(),
        )?;
        let mut d_low_res_model_fb =
            DevicePointer::copy_to_device(vis_residual_low_res_fb.as_slice().unwrap())?;
        let mut d_low_res_model_rotated =
            DevicePointer::copy_to_device(vis_residual_low_res_fb.as_slice().unwrap())?;

        // One pointer per timestep.
        let mut d_uvws = Vec::with_capacity(high_res_uvws.len_of(Axis(0)));
        // Temp vector to store results.
        let mut gpu_uvws = vec![gpu::UVW::default(); high_res_uvws.len_of(Axis(1))];
        for uvws in high_res_uvws.outer_iter() {
            // Convert the type and push the results to the device,
            // saving the resulting pointer.
            uvws.iter()
                .zip_eq(gpu_uvws.iter_mut())
                .for_each(|(&UVW { u, v, w }, gpu_uvw)| {
                    *gpu_uvw = gpu::UVW {
                        u: u as GpuFloat,
                        v: v as GpuFloat,
                        w: w as GpuFloat,
                    }
                });
            d_uvws.push(DevicePointer::copy_to_device(&gpu_uvws)?);
        }
        let mut d_beam_jones = DevicePointer::default();

        for pass in 0..num_passes {
            peel_progress.reset();
            peel_progress.set_message(format!(
                "Peeling timeblock {}, pass {}",
                timeblock.index + 1,
                pass + 1
            ));

            let mut pass_issues = 0;
            let mut pass_alpha_mag = 0.;
            let mut pass_beta_mag = 0.;
            let mut pass_gain_mag = 0.;

            // this needs to be inside the pass loop, because d_uvws_from gets swapped with d_uvws_to
            gpu_kernel_call!(
                gpu::xyzs_to_uvws,
                d_xyzs.get(),
                d_lmsts.get(),
                d_uvws_from.get_mut(),
                gpu::RADec {
                    ra: obs_context.phase_centre.ra as GpuFloat,
                    dec: obs_context.phase_centre.dec as GpuFloat,
                },
                num_tiles_i32,
                num_cross_baselines_i32,
                num_timesteps_i32
            )?;

            for (i_source, (((source_name, source), iono_consts), source_pos)) in source_list
                .iter()
                .take(num_sources_to_iono_subtract)
                .zip_eq(iono_consts.iter_mut())
                .zip_eq(source_weighted_positions.iter().copied())
                .enumerate()
            {
                let start = std::time::Instant::now();

                let old_iono_consts = IonoConsts {
                    alpha: iono_consts.alpha,
                    beta: iono_consts.beta,
                    gain: iono_consts.gain,
                };
                let gpu_old_iono_consts = gpu::IonoConsts {
                    alpha: old_iono_consts.alpha,
                    beta: old_iono_consts.beta,
                    gain: old_iono_consts.gain,
                };

                gpu_kernel_call!(
                    gpu::xyzs_to_uvws,
                    d_xyzs.get(),
                    d_lmsts.get(),
                    d_uvws_to.get_mut(),
                    gpu::RADec {
                        ra: source_pos.ra as GpuFloat,
                        dec: source_pos.dec as GpuFloat,
                    },
                    num_tiles_i32,
                    num_cross_baselines_i32,
                    num_timesteps_i32,
                )?;
                pb_trace!("{:?}: xyzs_to_uvws", start.elapsed());

                // rotate d_high_res_vis_tfb in place
                gpu_kernel_call!(
                    gpu::rotate,
                    d_high_res_vis_tfb.get_mut().cast(),
                    num_timesteps_i32,
                    num_cross_baselines_i32,
                    num_high_res_chans_i32,
                    d_uvws_from.get(),
                    d_uvws_to.get(),
                    d_lambdas.get()
                )?;
                pb_trace!("{:?}: rotate", start.elapsed());

                // there's a bug in the modeller where it ignores --no-precession
                // high_res_modeller.update_with_a_source(source, obs_context.phase_centre)?;
                high_res_modeller.update_with_a_source(source, source_pos)?;
                // Clear the old memory before reusing the buffer.
                d_high_res_model_tfb.clear();
                for (i_time, (lmst, latitude)) in lmsts.iter().zip(latitudes.iter()).enumerate() {
                    let original_model_ptr = d_high_res_model_tfb.ptr;
                    d_high_res_model_tfb.ptr = d_high_res_model_tfb
                        .ptr
                        .add(i_time * num_cross_baselines * all_fine_chan_lambdas_m.len());
                    let original_uvw_ptr = d_uvws_to.ptr;
                    d_uvws_to.ptr = d_uvws_to.ptr.add(i_time * num_cross_baselines);
                    high_res_modeller.model_timestep_with(
                        *lmst,
                        *latitude,
                        &d_uvws_to,
                        &mut d_beam_jones,
                        &mut d_high_res_model_tfb,
                    )?;
                    d_high_res_model_tfb.ptr = original_model_ptr;
                    d_uvws_to.ptr = original_uvw_ptr;
                }
                pb_trace!("{:?}: high res model", start.elapsed());

                // d_high_res_resid_tfb = residuals@src.
                d_high_res_vis_tfb.copy_to(&mut d_high_res_resid_tfb)?;
                // add iono@src to residuals@src
                gpu_kernel_call!(
                    gpu::add_model,
                    d_high_res_resid_tfb.get_mut().cast(),
                    d_high_res_model_tfb.get().cast(),
                    gpu_old_iono_consts,
                    d_lambdas.get(),
                    d_uvws_to.get(),
                    num_timesteps_i32,
                    num_high_res_chans_i32,
                    num_cross_baselines_i32,
                )?;
                pb_trace!("{:?}: add_model", start.elapsed());

                // *** UGLY HACK ***
                // d_high_res_model_rotated = iono@src
                let mut d_high_res_model_rotated: DevicePointer<Jones<f32>> =
                    DevicePointer::malloc(d_high_res_model_tfb.get_size())?;
                d_high_res_model_rotated.clear();
                gpu_kernel_call!(
                    gpu::add_model,
                    d_high_res_model_rotated.get_mut().cast(),
                    d_high_res_model_tfb.get().cast(),
                    gpu_old_iono_consts,
                    d_lambdas.get(),
                    d_uvws_to.get(),
                    num_timesteps_i32,
                    num_high_res_chans_i32,
                    num_cross_baselines_i32,
                )?;
                pb_trace!("{:?}: add_model", start.elapsed());

                gpu_kernel_call!(
                    gpu::average,
                    d_high_res_resid_tfb.get().cast(),
                    d_high_res_weights_tfb.get(),
                    d_low_res_resid_fb.get_mut().cast(),
                    num_timesteps_i32,
                    num_cross_baselines_i32,
                    num_high_res_chans_i32,
                    freq_average_factor
                )?;
                pb_trace!("{:?}: average high res vis", start.elapsed());

                gpu_kernel_call!(
                    gpu::average,
                    d_high_res_model_rotated.get().cast(),
                    d_high_res_weights_tfb.get(),
                    d_low_res_model_fb.get_mut().cast(),
                    num_timesteps_i32,
                    num_cross_baselines_i32,
                    num_high_res_chans_i32,
                    freq_average_factor
                )?;
                pb_trace!("{:?}: average high res model", start.elapsed());

                gpu_kernel_call!(
                    gpu::xyzs_to_uvws,
                    d_xyzs_low_res.get(),
                    d_average_lmsts.get(),
                    d_low_res_uvws.get_mut(),
                    gpu::RADec {
                        ra: source_pos.ra as GpuFloat,
                        dec: source_pos.dec as GpuFloat,
                    },
                    num_tiles_i32,
                    num_cross_baselines_i32,
                    1,
                )?;
                pb_trace!("{:?}: low res xyzs_to_uvws", start.elapsed());

                let mut gpu_iono_consts = gpu::IonoConsts {
                    alpha: 0.0,
                    beta: 0.0,
                    gain: 1.0,
                };
                // get size of device ptr
                let lrblch = (num_cross_baselines_i32 * num_low_res_chans_i32) as f64;
                pb_trace!("before iono_loop nt{:?} nxbl{:?} nlrch{:?} = lrblch{:?}; lrvfb{:?} lrwfb{:?} lrmfb{:?} lrmrfb{:?}",
                    num_tiles_i32,
                    num_cross_baselines_i32,
                    num_low_res_chans_i32,
                    lrblch,
                    d_low_res_resid_fb.get_size() as f64 / lrblch,
                    d_low_res_weights_fb.get_size() as f64 / lrblch,
                    d_low_res_model_fb.get_size() as f64 / lrblch,
                    d_low_res_model_rotated.get_size() as f64 / lrblch,
                );
                gpu_kernel_call!(
                    gpu::iono_loop,
                    d_low_res_resid_fb.get().cast(),
                    d_low_res_weights_fb.get(),
                    d_low_res_model_fb.get().cast(),
                    d_low_res_model_rotated.get_mut().cast(),
                    d_iono_fits.get_mut().cast(),
                    &mut gpu_iono_consts,
                    num_cross_baselines_i32,
                    num_low_res_chans_i32,
                    num_loops as i32,
                    d_low_res_uvws.get(),
                    d_low_res_lambdas.get(),
                    convergence as GpuFloat,
                )?;
                pb_trace!("{:?}: iono_loop", start.elapsed());

                iono_consts.alpha = old_iono_consts.alpha + gpu_iono_consts.alpha;
                iono_consts.beta = old_iono_consts.beta + gpu_iono_consts.beta;
                iono_consts.gain = old_iono_consts.gain * gpu_iono_consts.gain;

                #[rustfmt::skip]
                let issues = format!(
                    "{}{}{}",
                    if iono_consts.alpha.abs() > 1e-3 {
                        if iono_consts.alpha > 0.0 { "A" } else { "a" }
                    } else {
                        ""
                    },
                    if iono_consts.beta.abs() > 1e-3 {
                        if iono_consts.beta > 0.0 { "B" } else { "b" }
                    } else {
                        ""
                    },
                    if iono_consts.gain < 0.0 {
                        "g"
                    } else if iono_consts.gain > 1.5 {
                        "G"
                    } else {
                        ""
                    },
                );
                let message = format!(
                    "t{:3} pass {:2} s{i_source:6}|{source_name:16} @ ra {:+7.2} d {:+7.2} | a {:+8.6} b {:+8.6} g {:+3.2} | da {:+8.6} db {:+8.6} dg {:+3.2} | {}",
                    timeblock.index,
                    pass+1,
                    (source_pos.ra.to_degrees() + 180.) % 360. - 180.,
                    source_pos.dec.to_degrees(),
                    iono_consts.alpha,
                    iono_consts.beta,
                    iono_consts.gain,
                    iono_consts.alpha - old_iono_consts.alpha,
                    iono_consts.beta - old_iono_consts.beta,
                    iono_consts.gain - old_iono_consts.gain,
                    issues,
                );
                if issues.is_empty() {
                    pb_debug!("[peel_gpu] {}", message);
                    pass_alpha_mag += (iono_consts.alpha - old_iono_consts.alpha).abs();
                    pass_beta_mag += (iono_consts.beta - old_iono_consts.beta).abs();
                    pass_gain_mag += iono_consts.gain - old_iono_consts.gain;
                } else {
                    pb_debug!(
                        "[peel_gpu] {} (reverting to a {:+8.6} b {:+8.6} g {:+3.2})",
                        message,
                        old_iono_consts.alpha,
                        old_iono_consts.beta,
                        old_iono_consts.gain,
                    );
                    iono_consts.alpha = old_iono_consts.alpha;
                    iono_consts.beta = old_iono_consts.beta;
                    iono_consts.gain = old_iono_consts.gain;
                    pass_issues += 1;
                }

                let gpu_iono_consts = gpu::IonoConsts {
                    alpha: iono_consts.alpha,
                    beta: iono_consts.beta,
                    gain: iono_consts.gain,
                };

                pb_trace!("subtracting {gpu_old_iono_consts:?} adding {gpu_iono_consts:?}");

                // vis += iono(model, old_consts) - iono(model, consts)
                gpu_kernel_call!(
                    gpu::subtract_iono,
                    d_high_res_vis_tfb.get_mut().cast(),
                    d_high_res_model_tfb.get().cast(),
                    gpu_iono_consts,
                    gpu_old_iono_consts,
                    d_uvws_to.get(),
                    d_lambdas.get(),
                    num_timesteps_i32,
                    num_cross_baselines_i32,
                    num_high_res_chans_i32,
                )?;
                pb_trace!("{:?}: subtract_iono", start.elapsed());

                // Peel?
                let num_sources_to_peel = 0;
                if pass == num_passes - 1 && i_source < num_sources_to_peel {
                    // We currently can only do DI calibration on the CPU. Copy the visibilities back to the host.
                    let vis = d_high_res_vis_tfb.copy_from_device_new()?;

                    // *** UGLY HACK ***
                    let mut d_high_res_model_rotated: DevicePointer<Jones<f32>> =
                        DevicePointer::malloc(d_high_res_model_tfb.get_size())?;
                    d_high_res_model_rotated.clear();
                    gpu_kernel_call!(
                        gpu::add_model,
                        d_high_res_model_rotated.get_mut().cast(),
                        d_high_res_model_tfb.get().cast(),
                        gpu_iono_consts,
                        d_lambdas.get(),
                        d_uvws_to.get(),
                        num_timesteps_i32,
                        num_high_res_chans_i32,
                        num_cross_baselines_i32,
                    )?;
                    let model = d_high_res_model_rotated.copy_from_device_new()?;

                    let mut di_jones =
                        Array3::from_elem((1, num_tiles, num_high_res_chans), Jones::identity());
                    let shape = (num_timesteps, num_high_res_chans, num_cross_baselines);
                    let pb = ProgressBar::hidden();
                    let di_cal_results = calibrate_timeblock(
                        ArrayView3::from_shape(shape, &vis).expect("correct shape"),
                        ArrayView3::from_shape(shape, &model).expect("correct shape"),
                        di_jones.view_mut(),
                        timeblock,
                        chanblocks,
                        50,
                        1e-8,
                        1e-4,
                        obs_context.polarisations,
                        pb,
                        true,
                    );
                    if di_cal_results.into_iter().all(|r| r.converged) {
                        // Apply.
                    }
                }

                // The new phase centre becomes the old one.
                std::mem::swap(&mut d_uvws_from, &mut d_uvws_to);

                peel_progress.inc(1);
            }

            let num_good_sources = (num_sources_to_iono_subtract - pass_issues) as f64;
            if num_good_sources > 0. {
                let msg = format!(
                    "t{:3} pass {:2} ma {:+7.2e} mb {:+7.2e} mg {:+7.2e}",
                    timeblock.index,
                    pass + 1,
                    pass_alpha_mag / num_good_sources,
                    pass_beta_mag / num_good_sources,
                    pass_gain_mag / num_good_sources
                );
                if pass_issues > 0 {
                    pb_warn!("[peel_gpu] {} ({} issues)", msg, pass_issues);
                } else {
                    pb_info!("[peel_gpu] {} (no issues)", msg);
                }
            } else {
                pb_warn!(
                    "[peel_gpu] t{:03} pass {:2} all sources had issues",
                    timeblock.index,
                    pass + 1
                );
            }

            // Rotate back to the phase centre.
            gpu_kernel_call!(
                gpu::xyzs_to_uvws,
                d_xyzs.get(),
                d_lmsts.get(),
                d_uvws_to.get_mut(),
                gpu::RADec {
                    ra: obs_context.phase_centre.ra as GpuFloat,
                    dec: obs_context.phase_centre.dec as GpuFloat,
                },
                num_tiles_i32,
                num_cross_baselines_i32,
                num_timesteps_i32
            )?;

            gpu_kernel_call!(
                gpu::rotate,
                d_high_res_vis_tfb.get_mut().cast(),
                num_timesteps_i32,
                num_cross_baselines_i32,
                num_high_res_chans_i32,
                d_uvws_from.get(),
                d_uvws_to.get(),
                d_lambdas.get()
            )?;
        }

        // copy results back to host
        d_high_res_vis_tfb.copy_from_device(vis_residual_tfb.as_slice_mut().unwrap())?;
    }
    Ok(())
}
