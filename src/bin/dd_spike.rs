use indexmap::{indexmap, IndexMap};
use mwa_hyperdrive::{
    calibrate::{
        channels_to_chanblocks, di::calibrate_timeblocks, timesteps_to_timeblocks, solutions::CalibrationSolutions
    },
    constants::{DEFAULT_MAX_ITERATIONS, DEFAULT_MIN_THRESHOLD, DEFAULT_STOP_THRESHOLD},
    model::new_sky_modeller,
    HyperdriveError,
    math::cexp,
};
use mwa_hyperdrive_beam::{create_fee_beam_object, Beam, Delays};
use mwa_hyperdrive_common::{
    hifitime::{Duration, Epoch, Unit},
    itertools::{izip, Itertools},
    mwalib,
    num_traits::{Float, Num, NumAssign, Zero},
    vec1::vec1,
};
use mwa_hyperdrive_srclist::{
    hyperdrive::source_list_to_yaml, ComponentType, FluxDensity, FluxDensityType,
    Source, SourceComponent, SourceList, IonoSourceList, IonoSource,
};
use ndarray::{s, ArrayView3, ArrayViewMut3, Array2};
use std::{
    collections::{HashSet},
    f64::consts::{PI, TAU},
    fs::File,
    io::{BufWriter, Write},
    ops::{Deref},
    path::{PathBuf},
};

use birli::{
    marlu::{
        constants::VEL_C,
        precession::{get_lmst, precess_time},
        sexagesimal::{sexagesimal_dms_to_degrees, sexagesimal_hms_to_float},
        LatLngHeight, MeasurementSetWriter, ObsContext as MarluObsContext, RADec,
        VisContext, VisWritable, XyzGeodetic, UVW, LMN,
    },
    Array3, Axis, Jones};

// apply ionospheric rotation
fn apply_iono<F>(
    mut jones: ArrayViewMut3<Jones<F>>,
    phase_centre: RADec,
    ant_pairs: &[(usize, usize)],
    tile_xyzs: &[XyzGeodetic],
    freqs_hz: &[f64],
    centroid_timestamps: &[Epoch],
    array_pos: LatLngHeight,
    // constants of proportionality for ionospheric offset in l,m
    const_lm: (f64, f64),
) where
    F: Float + Num + NumAssign + Default,
{
    let jones_dims = jones.dim();

    assert_eq!(jones_dims.0, centroid_timestamps.len());
    assert_eq!(jones_dims.1, ant_pairs.len());
    assert_eq!(jones_dims.2, freqs_hz.len());

    // pre-compute partial uvws:
    let max_ant = ant_pairs.iter().map(|&(a, b)| a.max(b)).max().unwrap();
    let mut part_uvws = Array2::from_elem((centroid_timestamps.len(), max_ant+1), UVW::default());
    for (t, &epoch) in centroid_timestamps.iter().enumerate() {
        let prec = precess_time(
            phase_centre,
            epoch,
            array_pos.longitude_rad,
            array_pos.latitude_rad,
        );
        let tiles_xyz_prec = prec.precess_xyz_parallel(tile_xyzs);
        for (a, &xyz) in tiles_xyz_prec.iter().enumerate() {
            let uvw = UVW::from_xyz(
                xyz,
                prec.hadec_j2000,
            );
            part_uvws[[t, a]] = uvw;
        }
    }

    // iterate along time axis
    for (
        mut jones,
        part_uvws
    ) in izip!(
        jones.outer_iter_mut(),
        part_uvws.outer_iter(),
    ) {
        // iterate along baseline axis
        for (
            mut jones,
            &(ant1, ant2)
        ) in izip!(
            jones.outer_iter_mut(),
            ant_pairs
        ) {
            let uvw= part_uvws[[ant1]] - part_uvws[[ant2]];
            let uv_lm = uvw.u * const_lm.0 + uvw.v * const_lm.1;
            eprintln!("uvw {:?} uv_lm: {}", uvw, uv_lm);
            // iterate along frequency axis
            for (
                jones,
                &freq_hz,
            ) in izip!(
                jones.iter_mut(),
                freqs_hz
            ) {
                let lambda_2 = freq_hz * freq_hz / VEL_C / VEL_C;
                // let lm = (lambda_2 * const_lm.0, lambda_2 * const_lm.1);
                // let lmn = LMN {
                //     l: lm.0,
                //     m: lm.1,
                //     n: (1. - lm.0 * lm.0 - lm.1 * lm.1).sqrt(),
                // };
                let rotation = cexp(F::from(-TAU * uv_lm * lambda_2).unwrap());
                *jones *= rotation;
            }
        }
    }
}

fn rotate_accumulate<F>(
    jones_from: ArrayView3<Jones<F>>,
    mut jones_to: ArrayViewMut3<Jones<F>>,
    weight_from: ArrayView3<f32>,
    mut weight_to: ArrayViewMut3<f32>,
    phase_from: RADec,
    phase_to: RADec,
    ant_pairs: &[(usize, usize)],
    tile_xyzs: Vec<XyzGeodetic>,
    freqs_hz: Vec<f64>,
    centroid_timestamps: Vec<Epoch>,
    array_pos: LatLngHeight,
    avg_time: usize,
    avg_freq: usize,
) where
    F: Float + Num + NumAssign + Default,
{

    let from_dims = jones_from.dim();

    // eprintln!("jones_from {:?}", from_dims);

    assert_eq!(from_dims.0, centroid_timestamps.len());
    assert_eq!(from_dims.1, ant_pairs.len());
    assert_eq!(from_dims.2, freqs_hz.len());
    assert_eq!(from_dims, weight_from.dim());

    let to_dims = jones_to.dim();
    // eprintln!("jones_to {:?}", to_dims);
    assert_eq!((from_dims.0 as f64/ avg_time as f64).floor() as usize, to_dims.0);
    assert_eq!(from_dims.1, to_dims.1);
    assert_eq!((from_dims.2 as f64/ avg_freq as f64).floor() as usize, to_dims.2);
    assert_eq!(to_dims, weight_to.dim());

    let lmn = phase_to.to_lmn(phase_from);
    eprintln!("lmn {:?}", lmn);

    // pre-compute uvws:
    let max_ant = ant_pairs.iter().map(|&(a, b)| a.max(b)).max().unwrap();
    let mut part_uvws = Array2::from_elem((centroid_timestamps.len(), max_ant+1), (UVW::default(), UVW::default()));
    for (t, &epoch) in centroid_timestamps.iter().enumerate() {
        let prec_from = precess_time(
            phase_from,
            epoch,
            array_pos.longitude_rad,
            array_pos.latitude_rad,
        );
        let tiles_xyz_prec_from = prec_from.precess_xyz_parallel(&tile_xyzs);
        let prec_to = precess_time(
            phase_to,
            epoch,
            array_pos.longitude_rad,
            array_pos.latitude_rad,
        );
        let tiles_xyz_prec_to = prec_to.precess_xyz_parallel(&tile_xyzs);
        for (a, (xyz_from, xyz_to)) in izip!(tiles_xyz_prec_from, tiles_xyz_prec_to).enumerate() {
            let uvw_from = UVW::from_xyz(
                xyz_from,
                prec_from.hadec_j2000,
            );
            let uvw_to = UVW::from_xyz(
                xyz_to,
                prec_to.hadec_j2000,
            );
            part_uvws[[t, a]] = (uvw_from, uvw_to);
        }
    }

    // iterate along time axis in chunks of avg_time
    for (
        jones_chunk,
        weight_chunk,
        mut jones_to,
        mut weight_to,
        part_uvws
    ) in izip!(
        jones_from.axis_chunks_iter(Axis(0), avg_time),
        weight_from.axis_chunks_iter(Axis(0), avg_time),
        jones_to.outer_iter_mut(),
        weight_to.outer_iter_mut(),
        part_uvws.axis_chunks_iter(Axis(0), avg_time),
    ) {

        // iterate along baseline axis
        for (
            jones_chunk,
            weight_chunk,
            mut jones_to,
            mut weight_to,
            &(ant1, ant2)
        ) in izip!(
            jones_chunk.axis_iter(Axis(1)),
            weight_chunk.axis_iter(Axis(1)),
            jones_to.outer_iter_mut(),
            weight_to.outer_iter_mut(),
            ant_pairs
        ) {
            for (
                jones_chunk,
                weight_chunk,
                jones_to,
                weight_to,
                freq_chunk,
            ) in izip!(
                jones_chunk.axis_chunks_iter(Axis(1), avg_freq),
                weight_chunk.axis_chunks_iter(Axis(1), avg_freq),
                jones_to.iter_mut(),
                weight_to.iter_mut(),
                freqs_hz.chunks(avg_freq)
            ) {
                let chunk_size = jones_chunk.len();

                let mut weight_sum_f64 = 0_f64;
                let mut jones_sum = Jones::<F>::default();
                let mut jones_weighted_sum = Jones::<F>::default();
                let mut avg_flag = true;

                // iterate through time chunks
                for (
                    jones_chunk,
                    weights_chunk,
                    part_uvws
                ) in izip!(
                    jones_chunk.axis_iter(Axis(0)),
                    weight_chunk.axis_iter(Axis(0)),
                    part_uvws.axis_iter(Axis(0))
                ) {
                    // TODO(Dev): this gets re-calculated every chunk
                    let (uvw_from1, uvw_to1) = part_uvws[[ant1]];
                    let (uvw_from2, uvw_to2) = part_uvws[[ant2]];
                    let arg_w_diff = (uvw_to1.w - uvw_to2.w) - (uvw_from1.w - uvw_from2.w);
                    for (
                        jones,
                        weight,
                        &freq_hz
                    ) in izip!(
                        jones_chunk.iter(),
                        weights_chunk.iter(),
                        freq_chunk.iter()
                    ) {
                        // XXX(Dev): not sure if sign is right here
                        let rotation = cexp(F::from(-TAU * arg_w_diff * (freq_hz as f64) / VEL_C).unwrap());
                        let jones_rotated = Jones::<F>::from(*jones * rotation);
                        jones_sum += jones_rotated;
                        if weight.abs() > 0. {
                            avg_flag = false;
                            let weight_abs_f64 = (*weight as f64).abs();
                            weight_sum_f64 += weight_abs_f64;
                            jones_weighted_sum += jones_rotated * F::from(weight_abs_f64).unwrap();
                        }
                    }
                }

                *jones_to = if !avg_flag {
                    jones_weighted_sum / F::from(weight_sum_f64).unwrap()
                } else {
                    jones_sum / F::from(chunk_size).unwrap()
                };

                *weight_to = weight_sum_f64 as f32;
            }
        }
    };
}

fn vis_rotate<F>(
    mut jones_array: ArrayViewMut3<Jones<F>>,
    phase_from: RADec,
    phase_to: RADec,
    ant_pairs: &[(usize, usize)],
    tile_xyzs: Vec<XyzGeodetic>,
    freqs_hz: Vec<f64>,
    centroid_timestamps: Vec<Epoch>,
    array_pos: LatLngHeight,
) where
    F: Float + Num + NumAssign + Default,
{

    let jones_dims = jones_array.dim();

    // eprintln!("jones_dims {:?}", jones_dims);

    assert_eq!(jones_dims.0, centroid_timestamps.len());
    assert_eq!(jones_dims.1, ant_pairs.len());
    assert_eq!(jones_dims.2, freqs_hz.len());

    // eprintln!("phase from {:?} to {:?}", phase_from, phase_to);

    // let lmn = phase_to.to_lmn(phase_from);

    // pre-compute partial uvws:
    let max_ant = ant_pairs.iter().map(|&(a, b)| a.max(b)).max().unwrap();
    let mut part_uvws = Array2::from_elem((centroid_timestamps.len(), max_ant+1), (UVW::default(), UVW::default()));
    for (t, &epoch) in centroid_timestamps.iter().enumerate() {
        let prec_from = precess_time(
            phase_from,
            epoch,
            array_pos.longitude_rad,
            array_pos.latitude_rad,
        );
        let tiles_xyz_prec_from = prec_from.precess_xyz_parallel(&tile_xyzs);
        let prec_to = precess_time(
            phase_to,
            epoch,
            array_pos.longitude_rad,
            array_pos.latitude_rad,
        );
        let tiles_xyz_prec_to = prec_to.precess_xyz_parallel(&tile_xyzs);
        for (a, (xyz_from, xyz_to)) in izip!(tiles_xyz_prec_from, tiles_xyz_prec_to).enumerate() {
            let uvw_from = UVW::from_xyz(
                xyz_from,
                prec_from.hadec_j2000,
            );
            let uvw_to = UVW::from_xyz(
                xyz_to,
                prec_to.hadec_j2000,
            );
            part_uvws[[t, a]] = (uvw_from, uvw_to);
        }
    }

    // iterate along time axis in chunks of avg_time
    for (
        mut jones_array,
        part_uvws
    ) in izip!(
        jones_array.outer_iter_mut(),
        part_uvws.outer_iter_mut()
    ) {
        // iterate along baseline axis
        for (
            mut jones_array,
            &(ant1, ant2)
        ) in izip!(
            jones_array.outer_iter_mut(),
            ant_pairs
        ) {
            let (uvw_from1, uvw_to1) = part_uvws[[ant1]];
            let (uvw_from2, uvw_to2) = part_uvws[[ant2]];
            let arg_w_diff = (uvw_to1.w - uvw_to2.w) - (uvw_from1.w - uvw_from2.w);
            // iterate along frequency axis
            for (
                jones,
                freq_hz,
            ) in izip!(
                jones_array.iter_mut(),
                &freqs_hz
            ) {
                // XXX(Dev): not sure if sign is right here
                let rotation = cexp(F::from(-TAU * (arg_w_diff) * (*freq_hz as f64) / VEL_C).unwrap());
                *jones *= rotation;
            }
        }
    };
}

fn vis_average<F>(
    jones: ArrayView3<Jones<F>>,
    mut jones_avg: ArrayViewMut3<Jones<F>>,
    weight: ArrayView3<f32>,
    mut weight_avg: ArrayViewMut3<F>,
    avg_time: usize,
    avg_freq: usize,
) where
    F: Float + Num + Default + NumAssign,
{
    let jones_dims = jones.dim();
    let weight_dims = weight.dim();
    assert_eq!(weight_dims, jones_dims);

    let avg_dims = (
        (jones_dims.0 as f64 / avg_time as f64).ceil() as usize,
        jones_dims.1,
        (jones_dims.2 as f64 / avg_freq as f64).ceil() as usize,
    );

    assert_eq!(avg_dims, jones_avg.dim());
    assert_eq!(avg_dims, weight_avg.dim());

    // let mut jones_avg = Array3::<Jones<F>>::zeros(avg_dims);
    // let mut weight_avg = Array3::<f32>::zeros(avg_dims);

    // iterate through the time dimension of the arrays in chunks of size `time_factor`.
    for (jones_chunk, weight_chunk, mut jones_avg, mut weight_avg) in izip!(
        jones.axis_chunks_iter(Axis(0), avg_time),
        weight.axis_chunks_iter(Axis(0), avg_time),
        jones_avg.outer_iter_mut(),
        weight_avg.outer_iter_mut(),
    ) {
        // iterate through the baseline dimension of the arrays.
        for (jones_chunk, weight_chunk, mut jones_avg, mut weight_avg) in izip!(
            jones_chunk.axis_iter(Axis(1)),
            weight_chunk.axis_iter(Axis(1)),
            jones_avg.outer_iter_mut(),
            weight_avg.outer_iter_mut(),
        ) {
            // iterate through the channel dimension of the arrays in chunks of size `frequency_factor`.
            for (jones_chunk, weight_chunk, mut jones_avg, mut weight_avg) in izip!(
                jones_chunk.axis_chunks_iter(Axis(1), avg_freq),
                weight_chunk.axis_chunks_iter(Axis(1), avg_freq),
                jones_avg.outer_iter_mut(),
                weight_avg.outer_iter_mut(),
            ) {
                let chunk_size = jones_chunk.len();

                let mut weight_sum_f64 = 0_f64;
                let mut jones_sum = Jones::<F>::default();
                let mut jones_weighted_sum = Jones::<F>::default();
                let mut avg_flag = true;

                for (jones_chunk, weights_chunk) in izip!(
                    jones_chunk.axis_iter(Axis(0)),
                    weight_chunk.axis_iter(Axis(0)),
                ) {
                    for (jones, weight) in izip!(jones_chunk.iter(), weights_chunk.iter()) {
                        jones_sum += *jones;
                        if weight.abs() > 0. {
                            avg_flag = false;
                            let weight_abs_f64 = (*weight as f64).abs();
                            weight_sum_f64 += weight_abs_f64;
                            jones_weighted_sum += *jones * F::from(weight_abs_f64).unwrap();
                        }
                    }
                }

                jones_avg[()] = if !avg_flag {
                    jones_weighted_sum / F::from(weight_sum_f64).unwrap()
                } else {
                    jones_sum / F::from(chunk_size).unwrap()
                };

                weight_avg[()] = F::from(weight_sum_f64).unwrap();
            }
        }
    }

    // (jones_avg, weight_avg)
}

// fn promote_jones(jones: Array3<Jones<f32>>) -> Array3<Jones<f64>> {
//     // Array3::<Jones<f64>>::from_shape_fn(jones.dim(), |idx| Jones::<f64>::from(jones[idx]))
//     jones.mapv(Jones::<f64>::from)
// }

// fn demote_jones(jones: Array3<Jones<f64>>) -> Array3<Jones<f32>> {
//     // Array3::<Jones<f64>>::from_shape_fn(jones.dim(), |idx| Jones::<f64>::from(jones[idx]))
//     jones.mapv(Jones::<f32>::from)
// }

fn main() {
    // Stolen from BurntSushi. We don't return Result from main because it
    // prints the debug representation of the error. The code below prints the
    // "display" or human readable representation of the error.

    if let Err(e) = try_main() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn try_main() -> Result<(), HyperdriveError> {
    env_logger::init_from_env(
        env_logger::Env::default().filter_or(env_logger::DEFAULT_FILTER_ENV, "info"),
    );

    let out_dir = PathBuf::from("/tmp/crux");
    if !out_dir.is_dir() {
        std::fs::create_dir_all(&out_dir).unwrap();
    }

    // //////////////////// //
    // Observation metadata //
    // //////////////////// //

    let (vis_ctx, marlu_obs_ctx) = get_obs_metadata();
    let sel_shape = vis_ctx.sel_dims();

    // let sources = vec![
    //     RADec::new_degrees(1., -27.),
    //     RADec::new_degrees(0., -26.),
    //     RADec::new_degrees(-1., -27.),
    //     RADec::new_degrees(0., -28.),
    // ];

    // //////////// //
    // Source lists //
    // //////////// //

    // the synthetic source list is the visibilities we're capibrating,
    // the model source list is the model we're calibrating to.
    // model has been perturbed from synth

    let iono_srclist = get_source_list(&vis_ctx);

    let srclist_path = out_dir.join("srclist_model.yaml");
    let mut srclist_buf = BufWriter::new(File::create(&srclist_path).unwrap());
    source_list_to_yaml(&mut srclist_buf, &SourceList::from(iono_srclist.clone()), None).unwrap();
    srclist_buf.flush().unwrap();

    // let srclist_synth = get_source_list(&vis_ctx);
    // let first_key = srclist_model.keys().next().unwrap().clone();
    // srclist_synth[first_key].components[0].radec.ra += sexagesimal_hms_to_float(0., 1., 0.).to_radians();
    // srclist_synth[first_key].components[0].radec.ra += sexagesimal_hms_to_float(0., 0., 20.).to_radians();
    // srclist_synth[first_key].components[0].flux_type = FluxDensityType::PowerLaw {
    //     si: 0.,
    //     fd: FluxDensity {
    //         freq: vis_ctx.start_freq_hz,
    //         i: 4.0,
    //         q: 0.0,
    //         u: 0.0,
    //         v: 0.0,
    //     },
    // };

    // let mut iono_offsets = IndexMap::<String, (f64, f64)>::new();
    // for key in srclist_synth.keys() {
    //     let offset = (0.3, 0.) if key == first_key else (0., 0.);
    //     iono_offsets.insert(key.clone(), );
    // }

    // let srclist_path = out_dir.join("srclist_synth.yaml");
    // let mut srclist_buf = BufWriter::new(File::create(&srclist_path).unwrap());
    // source_list_to_yaml(&mut srclist_buf, &SourceList::from(srclist_synth.clone()), None).unwrap();
    // srclist_buf.flush().unwrap();

    // ////////////////////// //
    // generate synthetic vis //
    // ////////////////////// //


    #[cfg(feature = "cuda")]
    let use_cpu_for_modelling = false;

    let beam = get_beam(&marlu_obs_ctx);

    let mut vis_synth = Array3::from_elem(
        (sel_shape.0, sel_shape.2, sel_shape.1), Jones::<f32>::zero()
    );
    // synthesize some weights
    let weight_synth = Array3::from_elem(
        (sel_shape.0, sel_shape.2, sel_shape.1),
        vis_ctx.weight_factor() as f32,
    );

    let mut current_phase = marlu_obs_ctx.phase_centre;
    for (source_name, source) in iono_srclist.clone().into_iter() {
        let source_phase_centre = source.source.components[0].radec;
        let marlu_rot_obs_ctx = MarluObsContext {
            phase_centre: source_phase_centre,
            ..marlu_obs_ctx.clone()
        };
        vis_rotate(
            vis_synth.view_mut(),
            current_phase,
            source_phase_centre,
            &vis_ctx.sel_baselines,
            marlu_obs_ctx.ant_positions_geodetic().collect(),
            vis_ctx.frequencies_hz(),
            vis_ctx.timeseries(false, true).collect(),
            marlu_obs_ctx.array_pos,
        );
        simulate_accumulate_iono(
            vis_synth.view_mut(),
            &vis_ctx,
            #[cfg(feature = "cuda")]
            use_cpu_for_modelling,
            &beam,
            source,
            source_name,
            &marlu_rot_obs_ctx,
        );
        current_phase = source_phase_centre;
    }
    // finally rotate synth back to original phase centre
    vis_rotate(
        vis_synth.view_mut(),
        current_phase,
        marlu_obs_ctx.phase_centre,
        &vis_ctx.sel_baselines,
        marlu_obs_ctx.ant_positions_geodetic().collect(),
        vis_ctx.frequencies_hz(),
        vis_ctx.timeseries(false, true).collect(),
        marlu_obs_ctx.array_pos,
    );

    let vis_synth_path = out_dir.join("vis_synth.ms");
    write_vis(
        vis_synth.view(),
        weight_synth.view(),
        &vis_ctx,
        &marlu_obs_ctx,
        vis_synth_path,
    );


    // ////////////////// //
    // peel model vis //
    // ////////////////// //

    // TODO: rotate and peel individually
    let vis_model_path = out_dir.join("vis_model.ms");
    let vis_model = simulate_write(
        &vis_ctx,
        #[cfg(feature = "cuda")]
        use_cpu_for_modelling,
        &beam,
        SourceList::from(iono_srclist.clone()),
        &marlu_obs_ctx,
        Some(vis_model_path),
    );

    // ////////////////// //
    // GENERATE Synth vis //
    // ////////////////// //

    // residual visibilities = synthetic - model

    let mut vis_residual = vis_synth;
    vis_residual -= &vis_model;

    // TODO: peeling model selection

    let vis_residual_path = out_dir.join("vis_residual.ms");
    write_vis(
        vis_residual.view(),
        weight_synth.view(),
        &vis_ctx,
        &marlu_obs_ctx,
        vis_residual_path
    );

    // vis context that takes into account averaging
    let avg_vis_ctx = VisContext {
        avg_time: 4,
        avg_freq: vis_ctx.num_sel_chans / 24,
        ..vis_ctx.clone()
    };
    let avg_shape = avg_vis_ctx.avg_dims();
    // temporary arrays for accumulation
    let mut vis_source_avg = Array3::from_elem(
        (avg_shape.0, avg_shape.2, avg_shape.1), Jones::<f32>::default()
    );
    let mut weight_source_avg = Array3::from_elem(
        (avg_shape.0, avg_shape.2, avg_shape.1), f32::default()
    );
    let mut vis_residual_avg = Array3::from_elem(
        (avg_shape.0, avg_shape.2, avg_shape.1), Jones::<f32>::default()
    );
    let mut weight_residual_avg = Array3::from_elem(
        (avg_shape.0, avg_shape.2, avg_shape.1), f32::default()
    );

    // /////////// //
    // UNPEEL LOOP //
    // /////////// //

    let mut current_phase = marlu_obs_ctx.phase_centre;

    for (source_name, source) in iono_srclist.into_iter() {
        // let source_vis_ctx = vis_ctx.clone();

        // XXX(dev): a source can have multiple components with different phase centres.
        let source_phase_centre = source.source.components[0].radec;

        let marlu_rot_obs_ctx = MarluObsContext {
            phase_centre: source_phase_centre,
            ..marlu_obs_ctx.clone()
        };

        let srclist_source = SourceList::from(indexmap! {
            source_name.clone() => source.source
        });

        // /////////////////// //
        // ROTATE, AVERAGE VIS //
        // /////////////////// //

        rotate_accumulate::<f32>(
            vis_residual.view(),
            vis_residual_avg.view_mut(),
            weight_synth.view(),
            weight_residual_avg.view_mut(),
            current_phase,
            source_phase_centre,
            &vis_ctx.sel_baselines,
            marlu_obs_ctx.ant_positions_geodetic().collect(),
            vis_ctx.frequencies_hz(),
            vis_ctx.timeseries(false, true).collect(),
            marlu_obs_ctx.array_pos,
            avg_vis_ctx.avg_time,
            avg_vis_ctx.avg_freq
        );

        let vis_residual_avg_path = out_dir.join(format!("vis_residual_avg_{}.ms", source_name.clone()));
        let low_vis_ctx = VisContext {
            avg_time: 1,
            avg_freq: 1,
            freq_resolution_hz: avg_vis_ctx.avg_freq_resolution_hz(),
            int_time: avg_vis_ctx.avg_int_time(),
            num_sel_timesteps: avg_shape.0,
            num_sel_chans: avg_shape.1,
            ..vis_ctx.clone()
        };

        write_vis(
            vis_residual_avg.view(),
            weight_residual_avg.view(),
            &low_vis_ctx,
            &marlu_rot_obs_ctx,
            vis_residual_avg_path
        );

        // /////////////// //
        // GENEREATE MODEL //
        // /////////////// //

        // model is phased to source

        let vis_source_path = out_dir.join(format!("vis_model_{}.ms", source_name.clone()));
        let vis_source_avg = simulate_write(
            &low_vis_ctx,
            #[cfg(feature = "cuda")]
            use_cpu_for_modelling,
            &beam,
            srclist_source,
            &marlu_rot_obs_ctx,
            Some(vis_source_path),
        );

        let model_view = vis_source_avg.view();

        // ///////////// //
        // UNPEEL SOURCE //
        // ///////////// //

        let vis_unpeeled = Array3::from_shape_fn(vis_residual_avg.dim(), |idx| {
            vis_residual_avg[idx] + vis_source_avg[idx]
        });

        let vis_unpeeled_path = out_dir.join(format!("vis_unpeeled_{}.ms", source_name.clone()));

        write_vis(
            vis_unpeeled.view(),
            weight_residual_avg.view(),
            &low_vis_ctx,
            &marlu_rot_obs_ctx,
            vis_unpeeled_path
        );

        let offsets = single_cal_measurements(
            vis_unpeeled.view(),
            weight_residual_avg.view(),
            model_view,
            &avg_vis_ctx.sel_baselines,
            marlu_obs_ctx.ant_positions_geodetic().collect(),
            avg_vis_ctx.avg_frequencies_hz(),
            marlu_rot_obs_ctx.phase_centre,
            avg_vis_ctx.timeseries(true, true).collect(),
            marlu_obs_ctx.array_pos,
            source_name.clone(),
        );

        // ////// //
        // DI CAL //
        // ////// //

        let sols = di_cal(&low_vis_ctx, &vis_unpeeled, &vis_source_avg, &marlu_obs_ctx);

        let vis_soln_path = out_dir.join(format!("soln_{}.fits", source_name.clone()));
        let metafits: Option<&str> = None;
        sols.write_solutions_from_ext(vis_soln_path, metafits)
            .unwrap();

        // analyse_di_jones(sols.di_jones, &frequencies);

        current_phase = source_phase_centre;
    }

    Ok(())
}

fn di_cal(
    vis_ctx: &VisContext,
    vis_unpeeled: &Array3<Jones<f32>>,
    vis_model: &ndarray::ArrayBase<ndarray::OwnedRepr<Jones<f32>>, ndarray::Dim<[usize; 3]>>,
    marlu_obs_ctx: &MarluObsContext,
) -> CalibrationSolutions {
    let timestamps: Vec<Epoch> = vis_ctx.timeseries(true, true).collect();
    let timesteps: Vec<usize> = (0..timestamps.len()).collect();
    let timeblocks = timesteps_to_timeblocks(&timestamps, 4, &timesteps);
    let frequencies: Vec<u64> = vis_ctx
        .avg_frequencies_hz()
        .iter()
        .map(|&f| f as u64)
        .collect();
    let fences = channels_to_chanblocks(
        &frequencies,
        Some(vis_ctx.avg_freq_resolution_hz()),
        1,
        &HashSet::<usize>::new(),
    );
    let baseline_weights = vec![1.0; vis_ctx.sel_baselines.len()];
    let (sols, _) = calibrate_timeblocks(
        vis_unpeeled.view(),
        vis_model.view(),
        &timeblocks,
        &fences[0].chanblocks,
        &baseline_weights,
        DEFAULT_MAX_ITERATIONS,
        DEFAULT_STOP_THRESHOLD,
        DEFAULT_MIN_THRESHOLD,
        false,
        false,
    );
    // "Complete" the solutions.
    let sols = sols.into_cal_sols(
        marlu_obs_ctx.num_ants(),
        &[],
        &fences[0].flagged_chanblock_indices,
        None,
    );
    sols
}

fn single_cal_measurements(
    unpeeled: ArrayView3<Jones<f32>>,
    weights: ArrayView3<f32>,
    model: ArrayView3<Jones<f32>>,
    ant_pairs: &[(usize, usize)],
    tile_xyzs: Vec<XyzGeodetic>,
    freqs_hz: Vec<f64>,
    phase: RADec,
    centroid_timestamps: Vec<Epoch>,
    array_pos: LatLngHeight,
    src_name: String,
) -> Array3<f64> {

    let jones_dims = unpeeled.dim();

    assert_eq!(jones_dims.0, centroid_timestamps.len());
    assert_eq!(jones_dims.1, ant_pairs.len());
    assert_eq!(jones_dims.2, freqs_hz.len());
    assert_eq!(jones_dims, weights.dim());
    assert_eq!(jones_dims, model.dim());

    let mut offsets = Array3::zeros((jones_dims.0, jones_dims.1, 2));

    // println!("epoch, ant1, ant2, freq, u, v, w, weight, model_I_r, model_I_th, unpeeled_I_r, unpeeled_I_th, ")
    println!("name, epoch, lambda_2, a_uu, a_uv, a_vv, A_u, A_v, c_l, c_m");

    // iterate over time
    for (
        unpeeled,
        weights,
        model,
        mut offsets,
        epoch
    ) in izip!(
        unpeeled.outer_iter(),
        weights.outer_iter(),
        model.outer_iter(),
        offsets.outer_iter_mut(),
        centroid_timestamps,
    ) {
        let prec = precess_time(
            phase,
            epoch,
            array_pos.longitude_rad,
            array_pos.latitude_rad,
        );
        let tiles_xyz_prec = prec.precess_xyz_parallel(&tile_xyzs);

        // iterate over frequency
        for (
            unpeeled,
            weights,
            model,
            mut offsets,
            freq_hz,
        ) in izip!(
            unpeeled.axis_iter(Axis(1)),
            weights.axis_iter(Axis(1)),
            model.axis_iter(Axis(1)),
            offsets.outer_iter_mut(),
            freqs_hz.iter(),
        ) {

            let mut weight_sum_f64 = 0.;
            // TODO: why are these not used?
            let (mut s_vm, mut s_mm) = (0., 0.);
            let (mut a_uu, mut a_uv, mut a_vv) = (0., 0., 0.);
            let (mut A_u, mut A_v) = (0., 0.);
            let lambda = VEL_C / freq_hz;
            let lambda_2 = lambda * lambda;
            let lambda_4 = lambda_2 * lambda_2;

            // iterate over baseline
            for (
                unpeeled,
                weight,
                model,
                &(ant1, ant2)
            ) in izip!(
                unpeeled.iter(),
                weights.iter(),
                model.iter(),
                ant_pairs,
            ) {
                // TODO: this is re-calculated
                let uvw = UVW::from_xyz(
                    tiles_xyz_prec[ant1] - tiles_xyz_prec[ant2],
                    prec.hadec_j2000,
                );

                // eprintln!(
                //     "> epoch {:16} bl ({:3} {:3}) freq {:8} weight {:8}",
                //     epoch.as_gregorian_utc_str(),
                //     ant1, ant2,
                //     freq_hz,
                //     *weight
                // );
                if *weight > 0. {
                    // unsure about this method of gettings stokes I from jones
                    let unpeeled_i = unpeeled[0] + unpeeled[3];
                    let model_i = model[0] + model[3];

                    // eprintln!(
                    //     "> epoch {:16} bl ({:3} {:3}) freq {:8} unpeeled_I ({:6.4}, {:6.4}) model_I ({:6.4}, {:6.4})",
                    //     epoch.as_gregorian_utc_str(),
                    //     ant1, ant2,
                    //     freq_hz,
                    //     unpeeled_i.re,
                    //     unpeeled_i.im,
                    //     model_i.re,
                    //     model_i.im,
                    // );

                    let mr = (model_i.re as f64) * (unpeeled_i - model_i).im as f64;
                    let mm = (model_i.re as f64) * model_i.re as f64;

                    let weight_f64 = *weight as f64;
                    weight_sum_f64 += weight_f64;

                    // should really only accumulate these if rts_options->update_cal_amplitudes is true
                    s_vm += weight_f64 * (model_i.re as f64) * (unpeeled_i.re as f64);
                    s_mm += weight_f64 * mm;

                    // The sign of the fft exponent and the lambda^2 weighting is taken care of after the loop
                    a_uu += weight_f64 * mm * uvw.u * uvw.u;
                    a_uv += weight_f64 * mm * uvw.u * uvw.v;
                    a_vv += weight_f64 * mm * uvw.v * uvw.v;
                    A_u  += weight_f64 * mr * uvw.u;
                    A_v  += weight_f64 * mr * uvw.v;
                }
            }

            a_uu *= lambda_4;
            a_uv *= lambda_4;
            a_vv *= lambda_4;
            A_u *= lambda_2;
            A_v *= lambda_2;
            s_vm /= weight_sum_f64;
            s_mm /= weight_sum_f64;

            let delta = TAU * ( a_uu*a_vv - a_uv*a_uv );

            offsets[[0]] = (A_u*a_vv - A_v*a_uv) / delta;
            offsets[[1]] = (A_v*a_uu - A_u*a_uv) / delta;

            println!(
                "{}, {}, {}, {}, {}, {}, {}, {}, {}, {}",
                src_name,
                epoch.as_gpst_seconds(),
                lambda_2,
                a_uu,
                a_uv,
                a_vv,
                A_u,
                A_v,
                offsets[[0]] * lambda_2,
                offsets[[1]] * lambda_2
            );


            // eprintln!("epoch {:16} freq {:8} offsets {} {}", epoch.as_gregorian_utc_str(), freq_hz, offsets[[0]] * lambda_2, offsets[[1]] * lambda_2);
        }
    }

    offsets
}

// fn analyse_di_jones(di_jones: Array3<Jones<f64>>, frequencies: &[u64]) {
//     let lambda_squares = frequencies
//         .iter()
//         .map(|&freq_hz| (VEL_C / (freq_hz as f64)).powi(2));
//     for di_jones in di_jones.outer_iter() {
//         for di_jones in di_jones.outer_iter() {}
//     }
// }

fn get_beam(marlu_obs_ctx: &MarluObsContext) -> Box<dyn Beam> {
    let beam_file = "/data/dev/calibration/mwa_full_embedded_element_pattern.h5".into();
    // https://github.com/MWATelescope/mwa_pb/blob/90d6fbfc11bf4fca35796e3d5bde3ab7c9833b66/mwa_pb/mwa_sweet_spots.py#L60
    let delays = vec![0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12];

    let beam = create_fee_beam_object(
        beam_file,
        marlu_obs_ctx.ant_positions_geodetic().count(),
        Delays::Partial(delays),
        None,
        // Array2::from_elem((tile_xyzs.len(), 32), 1.)
    )
    .unwrap();
    beam
}

fn get_obs_metadata() -> (VisContext, MarluObsContext) {
    let num_timesteps = 4; //* 4;
    let num_channels = 24; // * 16;
    let tile_limit = 128;
    // let tile_limit = 32;

    let meta_path = PathBuf::from("test_files/1090008640/1090008640.metafits");
    let meta_ctx = mwalib::MetafitsContext::new::<PathBuf>(&meta_path, None).unwrap();
    // when crux is at zenith
    let zenith_time_utc = Epoch::from_gregorian_str("2022-05-05T21:53:00.00").unwrap();
    let zenith_time = zenith_time_utc - Duration::from_f64(8., Unit::Hour);
    let obsid = zenith_time.as_gpst_seconds().round() as i32;
    let tile_names: Vec<String> = meta_ctx
        .antennas
        .iter()
        .map(|ant| ant.tile_name.clone())
        .collect();
    let tile_xyzs: Vec<XyzGeodetic> = XyzGeodetic::get_tiles_mwa(&meta_ctx)
        .into_iter()
        .take(tile_limit)
        .collect();
    let ant_pairs = meta_ctx
        .baselines
        .iter()
        .filter(|&bl| bl.ant1_index != bl.ant2_index)
        .filter(|&bl| bl.ant1_index < tile_xyzs.len() && bl.ant2_index < tile_xyzs.len())
        .map(|bl| (bl.ant1_index, bl.ant2_index))
        .collect();
    let vis_ctx = VisContext {
        num_sel_timesteps: num_timesteps,
        start_timestamp: Epoch::from_gpst_seconds(obsid as f64),
        int_time: Duration::from_f64(1., Unit::Second),
        num_sel_chans: num_channels,
        start_freq_hz: 140_000_000.,
        freq_resolution_hz: 10_000.,
        sel_baselines: ant_pairs,
        avg_time: 1,
        avg_freq: 1,
        num_vis_pols: 4,
    };
    let phase_centre = RADec::new_degrees(
        // NOTE: these aren't J2000, not that it matters
        sexagesimal_hms_to_float(12., 30., 59.),
        sexagesimal_dms_to_degrees(-59., 22., 0.1),
    );
    let array_pos = LatLngHeight::new_mwa();
    let lst_rad = get_lmst(zenith_time, array_pos.longitude_rad);
    dbg!(lst_rad);
    let hadec = phase_centre.to_hadec(lst_rad);
    dbg!(hadec);
    dbg!(hadec.to_azel_mwa());
    dbg!(phase_centre.ra.to_degrees(), phase_centre.dec.to_degrees());
    let sched_start_timestamp = vis_ctx.start_timestamp;
    let sched_duration = vis_ctx.int_time * (vis_ctx.num_sel_timesteps + 1) as f64;
    let obs_name = Some("Simulated Crux visibilities".into());
    let marlu_obs_ctx = MarluObsContext {
        sched_start_timestamp,
        sched_duration,
        name: obs_name,
        phase_centre,
        pointing_centre: None,
        array_pos,
        ant_positions_enh: tile_xyzs
            .iter()
            .map(|xyz| xyz.to_enh(array_pos.latitude_rad))
            .collect(),
        ant_names: tile_names,
        field_name: None,
        project_id: None,
        observer: None,
    };
    (vis_ctx, marlu_obs_ctx)
}

fn simulate_write(
    vis_ctx: &VisContext,
    #[cfg(feature = "cuda")] use_cpu_for_modelling: bool,
    beam: &Box<dyn Beam>,
    source_list: SourceList,
    marlu_obs_ctx: &MarluObsContext,
    out_path: Option<PathBuf>,
) -> Array3<Jones<f32>> {
    // let beam = create_no_beam_object(tile_xyzs.len());
    let freqs_hz = vis_ctx.frequencies_hz();
    let sel_shape = vis_ctx.sel_dims();
    // Construct our visibilities array. This will be re-used for each timestep
    // before it's written to disk. Simulated vis is [baseline][chan] but
    // vis output requires [timestep][chan][baseline], this is re-used.
    let mut vis_out = Array3::from_elem((1, sel_shape.1, sel_shape.2), Jones::<f32>::default());
    let weight_out = Array3::from_elem(
        (1, sel_shape.1, sel_shape.2),
        vis_ctx.weight_factor() as f32,
    );
    let mut vis_result = Array3::from_elem(
        (sel_shape.0, sel_shape.2, sel_shape.1),
        Jones::<f32>::default(),
    );

    let tile_xyzs: Vec<XyzGeodetic> = marlu_obs_ctx.ant_positions_geodetic().collect();
    let phase_centre = marlu_obs_ctx.phase_centre;
    let array_pos = marlu_obs_ctx.array_pos;
    // Create a "modeller" object.
    let modeller = new_sky_modeller(
        #[cfg(feature = "cuda")]
        use_cpu_for_modelling,
        beam.deref(),
        &source_list,
        &tile_xyzs,
        &freqs_hz,
        &[],
        phase_centre,
        array_pos.longitude_rad,
        array_pos.latitude_rad,
        // TODO: Allow the user to turn off precession.
        true,
    )
    .unwrap();
    let mut writer = out_path.map(|out_path| {
        let writer = MeasurementSetWriter::new(&out_path, phase_centre, Some(array_pos));
        writer.initialize(vis_ctx, marlu_obs_ctx).unwrap();
        writer
    });

    let timeseries = vis_ctx.timeseries(false, true).enumerate();
    for (i, epoch) in timeseries {
        let mut vis_result = vis_result.slice_mut(s![i, .., ..]);
        eprintln!("modelling epoch {} {:?}", i, epoch.as_gregorian_utc_str());

        vis_result.fill(Jones::default());
        modeller
            .model_timestep(vis_result.view_mut(), epoch)
            .unwrap();

        // transpose model vis to output ordering. first axis is baseline.
        for (vis_model, mut vis_out) in
            izip!(vis_result.outer_iter(), vis_out.axis_iter_mut(Axis(2)))
        {
            // second axis is channel
            for (model_jones, mut vis_out) in
                izip!(vis_model.iter(), vis_out.axis_iter_mut(Axis(1)))
            {
                vis_out[(0)] = *model_jones;
            }
        }

        let chunk_vis_ctx = VisContext {
            start_timestamp: epoch - vis_ctx.int_time / 2.0,
            num_sel_timesteps: 1,
            ..vis_ctx.clone()
        };

        // eprintln!("data out shape {:?}", vis_out.shape());

        // Write the visibilities out.
        if let Some(writer) = writer.as_mut() {
            writer
                .write_vis_marlu(
                    vis_out.view(),
                    weight_out.view(),
                    &chunk_vis_ctx,
                    &tile_xyzs,
                    false,
                )
                .unwrap()
        }
    }
    vis_result
}

fn simulate_accumulate_iono(
    mut jones: ArrayViewMut3<Jones<f32>>,
    vis_ctx: &VisContext,
    #[cfg(feature = "cuda")] use_cpu_for_modelling: bool,
    beam: &Box<dyn Beam>,
    source: IonoSource,
    source_name: String,
    marlu_obs_ctx: &MarluObsContext,
) {
    // let beam = create_no_beam_object(tile_xyzs.len());
    let freqs_hz = vis_ctx.frequencies_hz();
    let jones_shape = jones.dim();
    // Construct our visibilities array. This will be re-used for each timestep
    // before it's written jones
    let mut vis_tmp = Array3::from_elem((1, jones_shape.1, jones_shape.2), Jones::<f32>::default());

    let tile_xyzs: Vec<XyzGeodetic> = marlu_obs_ctx.ant_positions_geodetic().collect();
    let centroid_timestamps = vis_ctx.timeseries(false, true);
    let source_list = SourceList::from(indexmap! {
        source_name => source.source
    });
    // Create a "modeller" object.
    let modeller = new_sky_modeller(
        #[cfg(feature = "cuda")]
        use_cpu_for_modelling,
        beam.deref(),
        &source_list,
        &tile_xyzs,
        &freqs_hz,
        &[],
        marlu_obs_ctx.phase_centre,
        marlu_obs_ctx.array_pos.longitude_rad,
        marlu_obs_ctx.array_pos.latitude_rad,
        // TODO: Allow the user to turn off precession.
        true,
    )
    .unwrap();

    for (epoch, mut jones) in izip!(centroid_timestamps, jones.outer_iter_mut()) {
        let mut vis_slice = vis_tmp.slice_mut(s![0, .., ..]);
        vis_slice.fill(Jones::default());
        modeller
            .model_timestep(vis_slice.view_mut(), epoch)
            .unwrap();
        drop(vis_slice);
        if (source.iono_consts.0 - 0.).abs() > 1e-8 || (source.iono_consts.1 - 0.).abs() > 1e-8 {
            apply_iono(
                vis_tmp.view_mut(),
                marlu_obs_ctx.phase_centre,
                &vis_ctx.sel_baselines,
                &tile_xyzs,
                &freqs_hz,
                &[epoch],
                marlu_obs_ctx.array_pos,
                source.iono_consts,
            );
        }
        let vis_slice = vis_tmp.slice(s![0, .., ..]);
        for (vis_model, jones) in izip!(vis_slice.iter(), jones.iter_mut()) {
            *jones += *vis_model;
        }
    }
}

fn write_vis(
    vis_write: ArrayView3<Jones<f32>>,
    weight_write: ArrayView3<f32>,
    vis_ctx: &VisContext,
    marlu_obs_ctx: &MarluObsContext,
    out_path: PathBuf,
) {
    // timesteps, channels, baselines
    let sel_shape = vis_ctx.sel_dims();

    let write_dims = vis_write.dim();
    assert_eq!(sel_shape.0, write_dims.0);
    assert_eq!(sel_shape.1, write_dims.2);
    assert_eq!(sel_shape.2, write_dims.1);

    let tile_xyzs: Vec<XyzGeodetic> = marlu_obs_ctx.ant_positions_geodetic().collect();
    let phase_centre = marlu_obs_ctx.phase_centre;
    let array_pos = marlu_obs_ctx.array_pos;
    let mut writer = MeasurementSetWriter::new(&out_path, phase_centre, Some(array_pos));
    writer.initialize(vis_ctx, marlu_obs_ctx).unwrap();

    // temporary arrays to write each timestep
    let mut vis_out = Array3::from_elem(
        (1, sel_shape.1, sel_shape.2),
        Jones::<f32>::default(),
    );
    let mut weight_out = Array3::from_elem(
        (1, sel_shape.1, sel_shape.2),
        f32::default(),
    );

    let timeseries = vis_ctx.timeseries(false, true).enumerate();
    for (
        (i, epoch),
        vis_write,
        weight_write
    ) in izip!(
        timeseries,
        vis_write.outer_iter(),
        weight_write.outer_iter(),
    ) {
        eprintln!("writing to {:?} epoch {} {:?}", out_path, i, epoch.as_gregorian_utc_str());

        // transpose model vis to output ordering. first axis is baseline.
        for (
            vis_write,
            weight_write,
            mut vis_out,
            mut weight_out,
        ) in izip!(
            vis_write.outer_iter(),
            weight_write.outer_iter(),
            vis_out.axis_iter_mut(Axis(2)),
            weight_out.axis_iter_mut(Axis(2))
        ) {
            // second axis is channel
            for (
                jones_write,
                weight_write,
                mut vis_out,
                mut weight_out
            ) in izip!(
                vis_write.iter(),
                weight_write.iter(),
                vis_out.axis_iter_mut(Axis(1)),
                weight_out.axis_iter_mut(Axis(1))
            )
            {
                vis_out[(0)] = *jones_write;
                weight_out[(0)] = *weight_write;
            }
        }

        let chunk_vis_ctx = VisContext {
            start_timestamp: epoch - vis_ctx.int_time / 2.0,
            num_sel_timesteps: 1,
            ..vis_ctx.clone()
        };

        // eprintln!("data out shape {:?}", vis_out.shape());

        // Write the visibilities out.
        writer
            .write_vis_marlu(
                vis_out.view(),
                weight_out.view(),
                &chunk_vis_ctx,
                &tile_xyzs,
                false,
            )
            .unwrap()
    }
}

fn get_source_list(vis_ctx: &VisContext) -> IonoSourceList {
    let mut srclist = IonoSourceList::new();
    for ra in (180..195).step_by(5) {
        for neg_dec in (50..65).step_by(5) {
            let iono_consts = if ra==180 && neg_dec==50 {(0.0008, 0.)} else {(0., 0.)};
            srclist.insert(format!("{:.2}_{:.2}", ra, neg_dec), IonoSource {
                source: Source {components: vec1![
                    SourceComponent {
                        radec: RADec::new_degrees(ra as f64, -neg_dec as f64),
                        comp_type: ComponentType::Point,
                        flux_type: FluxDensityType::PowerLaw {
                            // NOTE: these aren't the real values, not that it matters
                            si: 0.,
                            fd: FluxDensity {
                                freq: vis_ctx.start_freq_hz,
                                i: 5.0,
                                q: 0.0,
                                u: 0.0,
                                v: 0.0,
                            },
                        },
                    }
                ]},
                iono_consts
            });
        }
    }
    IonoSourceList::from(srclist)
    // SourceList::from(indexmap! {
    // String::from("acrux") => Source {
    //     components: vec1![SourceComponent {
    //         radec: RADec::new_degrees(
    //             // NOTE: these aren't J2000, not that it matters
    //             sexagesimal_hms_to_float(12., 27., 51.9),
    //             sexagesimal_dms_to_degrees(-63., 13., 29.9)
    //         ),
    //         comp_type: ComponentType::Point,
    //         flux_type: FluxDensityType::PowerLaw {
    //             // NOTE: these aren't the real values, not that it matters
    //             si: 0.,
    //             fd: FluxDensity {
    //                 freq: vis_ctx.start_freq_hz,
    //                 i: 5.0,
    //                 q: 0.0,
    //                 u: 0.0,
    //                 v: 0.0,
    //             },
    //         },
    //     }],
    // },
    //     String::from("becrux") => Source {
    //         components: vec1![SourceComponent {
    //             radec: RADec::new_degrees(
    //                 sexagesimal_hms_to_float(12., 49., 3.0),
    //                 sexagesimal_dms_to_degrees(-59., 48., 45.1)
    //             ),
    //             comp_type: ComponentType::Point,
    //             flux_type: FluxDensityType::PowerLaw {
    //                 si: 0.,
    //                 fd: FluxDensity {
    //                     freq: vis_ctx.start_freq_hz,
    //                     i: 4.0,
    //                     q: 0.0,
    //                     u: 0.0,
    //                     v: 0.0,
    //                 },
    //             },
    //         }],
    //     },
    //     String::from("gacrux") => Source {
    //         components: vec1![SourceComponent {
    //             radec: RADec::new_degrees(
    //                 sexagesimal_hms_to_float(12., 32., 25.4),
    //                 sexagesimal_dms_to_degrees(-57., 14., 25.)
    //             ),
    //             comp_type: ComponentType::Point,
    //             flux_type: FluxDensityType::PowerLaw {
    //                 si: 0.,
    //                 fd: FluxDensity {
    //                     freq: vis_ctx.start_freq_hz,
    //                     i: 3.0,
    //                     q: 0.0,
    //                     u: 0.0,
    //                     v: 0.0,
    //                 },
    //             },
    //         }],
    //     },
    //     String::from("dcrux") => Source {
    //         components: vec1![SourceComponent {
    //             radec: RADec::new_degrees(
    //                 sexagesimal_hms_to_float(12., 16., 20.9),
    //                 sexagesimal_dms_to_degrees(-58., 52., 31.3)
    //             ),
    //             comp_type: ComponentType::Point,
    //             flux_type: FluxDensityType::PowerLaw {
    //                 si: 0.,
    //                 fd: FluxDensity {
    //                     freq: vis_ctx.start_freq_hz,
    //                     i: 2.0,
    //                     q: 0.0,
    //                     u: 0.0,
    //                     v: 0.0,
    //                 },
    //             },
    //         }],
    //     },
    //     String::from("ecrux") => Source {
    //         components: vec1![SourceComponent {
    //             radec: RADec::new_degrees(
    //                 sexagesimal_hms_to_float(12., 22., 35.2),
    //                 sexagesimal_dms_to_degrees(-60., 31., 35.8)
    //             ),
    //             comp_type: ComponentType::Point,
    //             flux_type: FluxDensityType::PowerLaw {
    //                 si: 0.,
    //                 fd: FluxDensity {
    //                     freq: vis_ctx.start_freq_hz,
    //                     i: 1.0,
    //                     q: 0.0,
    //                     u: 0.0,
    //                     v: 0.0,
    //                 },
    //             },
    //         }],
    //     },
    // })
}
