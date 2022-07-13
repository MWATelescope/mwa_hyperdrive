use birli::marlu::HADec;
use indexmap::indexmap;
use mwa_hyperdrive::{
    model::new_sky_modeller,
    HyperdriveError,
    math::{cexp},
};
use mwa_hyperdrive_beam::{create_fee_beam_object, Beam, Delays};
use mwa_hyperdrive_common::{
    hifitime::{Duration, Epoch, Unit},
    itertools::{izip, Itertools},
    mwalib,
    num_traits::{Float, Num, NumAssign, Zero, One},
    vec1::vec1,
    Jones, Complex,
    marlu::{
        constants::VEL_C,
        precession::{get_lmst, precess_time},
        LatLngHeight, MeasurementSetWriter, ObsContext as MarluObsContext, RADec,
        VisContext, VisWritable, XyzGeodetic, UVW,
    },
    ndarray::{Array3, Axis}, lazy_static
};
use mwa_hyperdrive_srclist::{
    hyperdrive::source_list_to_yaml, ComponentType, FluxDensity, FluxDensityType,
    Source, SourceComponent, SourceList, IonoSourceList, IonoSource,
};
use ndarray::{s, ArrayView3, ArrayViewMut3, Array2};
use std::{
    f64::consts::TAU,
    fs::File,
    io::{BufWriter, Write},
    ops::Deref,
    path::PathBuf,
};

fn main() {
    if let Err(e) = try_main() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

lazy_static::lazy_static! {
    pub static ref OUT_DIR: PathBuf = PathBuf::from("/tmp/crux");
}

fn try_main() -> Result<(), HyperdriveError> {
    env_logger::init_from_env(
        env_logger::Env::default().filter_or(env_logger::DEFAULT_FILTER_ENV, "info"),
    );

    if !OUT_DIR.is_dir() {
        std::fs::create_dir_all(OUT_DIR.as_path()).unwrap();
    }

    // //////////////////// //
    // Observation metadata //
    // //////////////////// //

    // vis_ctx contains the timesteps, frequencies and baselines in a visibility array.
    // obs_ctx contains the metadata for the observation, including phase centre, array position.
    let (vis_ctx, obs_ctx) = get_obs_metadata(LatLngHeight::new_mwa());

    // //////////// //
    // Source lists //
    // //////////// //

    // the synthetic source list is the visibilities we're calibrating,
    // the model source list is the model we're calibrating to.
    // synth is just model perturbed by ionosphere
    let iono_srclist = get_source_list(&vis_ctx, &obs_ctx);

    // write sourcelist to disk
    let srclist_path = OUT_DIR.join("srclist_model.yaml");
    let mut srclist_buf = BufWriter::new(File::create(&srclist_path).unwrap());
    source_list_to_yaml(&mut srclist_buf, &SourceList::from(iono_srclist.clone()), None).unwrap();
    srclist_buf.flush().unwrap();

    // ////////////////////// //
    // generate synthetic vis //
    // ////////////////////// //

    #[cfg(feature = "cuda")]
    let use_cpu_for_modelling = false;

    let beam = get_beam(obs_ctx.ant_positions_geodetic().count());

    // initialize an array to accumulate synthetic visibilities into
    let sel_shape = vis_ctx.sel_dims();
    let mut vis_synth = Array3::from_elem(
        (sel_shape.0, sel_shape.2, sel_shape.1), Jones::<f32>::zero()
    );

    // use the baseline taper from the RTS, 1-exp(-(u*u+v*v)/(2*sig^2));
    let short_baseline_sigma = 20.;
    let weight_synth = get_weights_rts(&vis_ctx, &obs_ctx, short_baseline_sigma);

    // for each source in the sourcelist:
    // - rotate the accumulated visibilities to the model phase centre
    // - simulate the visibilities and apply an ionospheric offset
    let mut rot_obs_ctx = obs_ctx.clone();
    for (source_name, source) in iono_srclist.clone().into_iter() {
        let source_phase_centre = source.source.components[0].radec;

        vis_rotate(
            vis_synth.view_mut(),
            &vis_ctx,
            &rot_obs_ctx,
            source_phase_centre,
        );
        rot_obs_ctx.phase_centre = source_phase_centre;
        simulate_accumulate_iono(
            vis_synth.view_mut(),
            &vis_ctx,
            #[cfg(feature = "cuda")]
            use_cpu_for_modelling,
            &beam,
            source,
            source_name,
            &rot_obs_ctx,
        );
    }
    // finally rotate synth back to original phase centre
    vis_rotate(
        vis_synth.view_mut(),
        &vis_ctx,
        &rot_obs_ctx,
        obs_ctx.phase_centre
    );

    // write out the synthetic visibilities
    let vis_synth_path = OUT_DIR.join("vis_synth.ms");
    write_vis(
        vis_synth.view(),
        weight_synth.view(),
        &vis_ctx,
        &obs_ctx,
        vis_synth_path,
    );

    // ////////////////// //
    // generate model vis //
    // ////////////////// //

    let vis_model_path = OUT_DIR.join("vis_model.ms");
    let vis_model = simulate_write(
        &vis_ctx,
        #[cfg(feature = "cuda")]
        use_cpu_for_modelling,
        &beam,
        SourceList::from(iono_srclist.clone()),
        &obs_ctx,
        Some(vis_model_path),
    );

    // ////////////// //
    // peel model vis //
    // ////////////// //

    // residual visibilities = synthetic - model
    // TODO: rotate and peel individually

    let mut vis_residual = vis_synth;
    vis_residual -= &vis_model;

    let vis_residual_path = OUT_DIR.join("vis_residual.ms");
    write_vis(
        vis_residual.view(),
        weight_synth.view(),
        &vis_ctx,
        &obs_ctx,
        vis_residual_path
    );

    // vis context at higher resolution that takes into account averaging
    let avg_vis_ctx = VisContext {
        avg_time: 4,
        avg_freq: vis_ctx.num_sel_chans / 24,
        ..vis_ctx.clone()
    };
    // shape of the averaged visibilities
    let avg_shape = avg_vis_ctx.avg_dims();
    // vis context at the averaged resolution
    let low_vis_ctx = VisContext {
        avg_time: 1,
        avg_freq: 1,
        freq_resolution_hz: avg_vis_ctx.avg_freq_resolution_hz(),
        int_time: avg_vis_ctx.avg_int_time(),
        num_sel_timesteps: avg_shape.0,
        num_sel_chans: avg_shape.1,
        ..vis_ctx
    };
    // temporary arrays for accumulation
    let mut vis_residual_avg = Array3::from_elem(
        (avg_shape.0, avg_shape.2, avg_shape.1), Jones::<f32>::default()
    );
    let mut weight_residual_avg = Array3::from_elem(
        (avg_shape.0, avg_shape.2, avg_shape.1), f32::default()
    );

    // /////////// //
    // UNPEEL LOOP //
    // /////////// //

    // print a header
    println!("{}", vec![
        "source",
        "expected ɑ",
        "expected β",
        "rts ɑ",
        "rts β",
        "ɑ expected / rts",
        "β expected / rts",
        "paper ɑ",
        "paper β",
    ].join("\t"));

    for (source_name, iono_source) in iono_srclist.into_iter() {
        // TODO(dev): a source can have multiple components with different phase centres.
        // this only looks at the first component's ra dec.
        let source_phase_centre = iono_source.source.components[0].radec;
        eprintln!("unpeel loop: {} at {} (has iono {:?})", source_name, source_phase_centre, iono_source.iono_consts);

        rot_obs_ctx.phase_centre = source_phase_centre;
        let srclist_source = SourceList::from(indexmap! {
            source_name.clone() => iono_source.source
        });

        // /////////////////// //
        // ROTATE, AVERAGE VIS //
        // /////////////////// //

        // rotate the residual visibilities to the model phase centre and average into vis_residual_avg
        rotate_accumulate::<f32>(
            vis_residual.view(),
            vis_residual_avg.view_mut(),
            weight_synth.view(),
            weight_residual_avg.view_mut(),
            &avg_vis_ctx,
            &obs_ctx,
            source_phase_centre,
        );

        // write the averaged, rotated residual visibilities to disk
        let vis_residual_avg_path = OUT_DIR.join(format!("vis_residual_avg_{}.ms", source_name.clone()));
        write_vis(
            vis_residual_avg.view(),
            weight_residual_avg.view(),
            &low_vis_ctx,
            &rot_obs_ctx,
            vis_residual_avg_path
        );

        // /////////////// //
        // GENEREATE MODEL //
        // /////////////// //

        // generate model again at lower resolution, phased to source
        let vis_source_path = OUT_DIR.join(format!("vis_model_{}.ms", source_name.clone()));
        let vis_source_avg = simulate_write(
            &low_vis_ctx,
            #[cfg(feature = "cuda")]
            use_cpu_for_modelling,
            &beam,
            srclist_source,
            &rot_obs_ctx,
            Some(vis_source_path),
        );

        // ///////////// //
        // UNPEEL SOURCE //
        // ///////////// //

        let vis_unpeeled = Array3::from_shape_fn(vis_residual_avg.dim(), |idx| {
            vis_residual_avg[idx] + vis_source_avg[idx]
        });

        let vis_unpeeled_path = OUT_DIR.join(format!("vis_unpeeled_{}.ms", source_name.clone()));

        write_vis(
            vis_unpeeled.view(),
            weight_residual_avg.view(),
            &low_vis_ctx,
            &rot_obs_ctx,
            vis_unpeeled_path
        );

        // ///////////////// //
        // CALCULATE OFFSETS //
        // ///////////////// //

        let offsets_rts = get_offsets_rts(
            vis_unpeeled.view(),
            weight_residual_avg.view(),
            vis_source_avg.view(),
            &low_vis_ctx,
            &rot_obs_ctx,
            // source_name.clone(),
        );

        let offsets_paper = get_offsets_paper(
            vis_unpeeled.view(),
            weight_residual_avg.view(),
            vis_source_avg.view(),
            &low_vis_ctx,
            &rot_obs_ctx,
            // source_name.clone(),
        );

        let cells: Vec<String> = vec![
            source_name.clone(),
            // expected
            iono_source.iono_consts.0.to_string(),
            iono_source.iono_consts.1.to_string(),
            // rts
            offsets_rts[0].to_string(),
            offsets_rts[1].to_string(),
            // expected / rts
            (iono_source.iono_consts.0 / offsets_rts[0]).to_string(),
            (iono_source.iono_consts.1 / offsets_rts[1]).to_string(),
            // paper
            offsets_paper[0].to_string(),
            offsets_paper[1].to_string(),

        ];
        println!("{}", cells.join("\t"));

        let chi_squared_path = OUT_DIR.join(format!("chi_squared_{}.tsv", source_name.clone()));
        plot_chi_squared(
            vis_unpeeled.view(),
            weight_residual_avg.view(),
            vis_source_avg.view(),
            &low_vis_ctx,
            &rot_obs_ctx,
            chi_squared_path,
        );

        // ////// //
        // DI CAL //
        // ////// //

        // let sols = di_cal(&low_vis_ctx, &vis_unpeeled, &vis_source_avg, &obs_ctx);

        // let vis_soln_path = OUT_DIR.join(format!("soln_{}.fits", source_name.clone()));
        // let metafits: Option<&str> = None;
        // sols.write_solutions_from_ext(vis_soln_path, metafits)
        //     .unwrap();
    }

    Ok(())
}

// create fake observation metadata using antennas from a real metafits,
// phase centre: hour angle = 0, declination = latitude
// obs time is nearest time when phase centre is at zenith
fn get_obs_metadata(array_pos: LatLngHeight) -> (VisContext, MarluObsContext) {
    let num_timesteps = 4; // * 4;
    let num_channels = 24; // * 16;
    let tile_limit = 128;
    // let tile_limit = 32;

    eprintln!("array position is {:?}", array_pos);
    // let meta_path = PathBuf::from("/data/dev/1336955216/1336955216.metafits");
    let meta_path = PathBuf::from("test_files/1090008640/1090008640.metafits");
    let meta_ctx = mwalib::MetafitsContext::new::<PathBuf>(&meta_path, None).unwrap();
    let obsid = meta_ctx.obs_id;
    let mut obs_time = Epoch::from_gpst_seconds(obsid as _);
    let obs_lst_rad = get_lmst(obs_time, array_pos.longitude_rad);
    // shift obs_time to the nearest time when the phase centre is at zenith
    if obs_lst_rad.abs() > 1e-6 {
        let sidereal2solar = 365.24/366.24;
        obs_time -= Duration::from_f64(sidereal2solar*obs_lst_rad/TAU, Unit::Day);
    }
    let zenith_lst_rad = get_lmst(obs_time, array_pos.longitude_rad);
    eprintln!("lst % 𝜏 should be 0: {:?}", zenith_lst_rad);
    let phase_centre = RADec::from_hadec(HADec::new(0., array_pos.latitude_rad), zenith_lst_rad);
    eprintln!("phase centre: {:?}", phase_centre);
    let hadec = phase_centre.to_hadec(zenith_lst_rad);
    eprintln!("ha % 𝜏 should be 0: {:?}", hadec);
    let azel = hadec.to_azel(array_pos.latitude_rad);
    eprintln!("(az, el) % 𝜏 should be 0, pi/2: {:?}", azel);
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
        start_timestamp: obs_time,
        int_time: Duration::from_f64(1., Unit::Second),
        num_sel_chans: num_channels,
        start_freq_hz: 140_000_000.,
        freq_resolution_hz: 10_000.,
        sel_baselines: ant_pairs,
        avg_time: 1,
        avg_freq: 1,
        num_vis_pols: 4,
    };
    let sched_start_timestamp = vis_ctx.start_timestamp;
    let sched_duration = vis_ctx.int_time * vis_ctx.num_sel_timesteps as f64;
    let obs_name = Some("Simulated Grid visibilities".into());
    let obs_ctx = MarluObsContext {
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
    (vis_ctx, obs_ctx)
}

// construct a pair of ionospheric offsets that would shift a source by
// `worst_angle_offsets` at maximum λ
fn get_iono_consts(
    max_lambda_sq: f64,
    source_pos: RADec,
    worst_angle_offsets: (f64, f64),
) -> (f64, f64) {
    // dbg!(&max_lambda_sq);
    // eprintln!(
    //     "src radec: {:?} => {}, {}",
    //     source_pos, source_pos.ra.to_degrees(), source_pos.dec.to_degrees()
    // );

    // the position of the worst ionospheric offset
    let worst_iono_radec = RADec {
        ra: source_pos.ra + worst_angle_offsets.0,
        dec: source_pos.dec + worst_angle_offsets.1,
    };
    // eprintln!("worst iono radec: {:?} => {}, {}", worst_iono_radec, worst_iono_radec.ra.to_degrees(), worst_iono_radec.dec.to_degrees());

    // get the lmn coordinates of the worst offset relative to source position
    let worst_iono_lmn = worst_iono_radec.to_lmn(source_pos);
    // eprintln!("worst iono lmn: {:?} => {}, {}", worst_iono_lmn, worst_iono_lmn.l.asin().to_degrees(), worst_iono_lmn.m.asin().to_degrees());

    // iono rotation: exp[-2πi(αu+βv)λ²]
    // normal rotation: exp[−2πi(ul+vm+w(√(1−l²−m²)−1))]
    // so α~l/λ², β~m/λ²
    (
        worst_iono_lmn.l / max_lambda_sq,
        worst_iono_lmn.m / max_lambda_sq,
    )
}

// Generate a model containing several flat spectrum point sources at various positions on a grid
// aligned to the nearest degree.
// Each source has a different ionospheric offset, which is calculated from a
// desired maximum sky coordinate offset which happens at the lowest frequency.
fn get_source_list(vis_ctx: &VisContext, obs_ctx: &MarluObsContext) -> IonoSourceList {
    let phase_centre = obs_ctx.phase_centre;
    let mut srclist = IonoSourceList::new();
    // grid size in degrees
    let quant = 1;
    let range = quant * 2;
    let p_ra = quant * (phase_centre.ra.to_degrees() / quant as f64).round() as i32;
    let p_dec = quant * (phase_centre.dec.to_degrees() / quant as f64).round() as i32;
    eprintln!("phase centre {:?} => {}, {}", phase_centre, p_ra, p_dec);

    let min_freq = vis_ctx.start_freq_hz;
    let max_lambda_sq = (VEL_C * VEL_C) / (min_freq * min_freq);

    // a small ionospheric deviation of 15 arcsec at lowest freq
    let small_iono_rad = (15_f64/60./60.).to_radians();
    let big_iono_rad = (59_f64/60./60.).to_radians();
    eprintln!("small iono {:?} deg, big iono {:?} deg", small_iono_rad.to_degrees(), big_iono_rad.to_degrees());

    let mut i = 0;
    for ra in (0..=range).step_by(quant as usize) {
        for dec in (0..=range).step_by(quant as usize) {
            i += 1;
            // uncomment this to only use the source at 0,-27
            // if i != 5 { continue; }
            // whether the source offset is big or small
            let iono_rad = if i==5 {big_iono_rad} else {small_iono_rad};
            // the source model position
            let src_ra = p_ra + ra - range / 2;
            let src_dec = p_dec + dec - range / 2;
            let src_radec = RADec::new_degrees(src_ra as f64, src_dec as f64);

            let worst_angle_offsets = (
                iono_rad * (TAU * (i%4) as f64 / 4.).sin(),
                iono_rad * (TAU * (i%4) as f64 / 4.).cos(),
            );
            let iono_consts = get_iono_consts(max_lambda_sq, src_radec, worst_angle_offsets);
            dbg!(&iono_consts);
            srclist.insert(format!("ra{:+.2}dec_{:+.2}", src_ra, src_dec), IonoSource {
                source: Source {components: vec1![
                    SourceComponent {
                        radec: src_radec,
                        comp_type: ComponentType::Point,
                        flux_type: FluxDensityType::PowerLaw {
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
    srclist
}

fn get_beam(num_tiles: usize) -> Box<dyn Beam> {
    let beam_file = "/data/dev/calibration/mwa_full_embedded_element_pattern.h5".into();
    let delays = vec![
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
    ];
    // https://github.com/MWATelescope/mwa_pb/blob/90d6fbfc11bf4fca35796e3d5bde3ab7c9833b66/mwa_pb/mwa_sweet_spots.py#L60
    // let delays = vec![0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12];

    let beam = create_fee_beam_object(
        beam_file,
        num_tiles,
        Delays::Partial(delays),
        None,
        // Array2::from_elem((tile_xyzs.len(), 32), 1.)
    )
    .unwrap();
    beam
}

fn get_weights_rts(
    vis_ctx: &VisContext,
    obs_ctx: &MarluObsContext,
    short_sigma: f64,
) -> Array3<f32> {
    let sel_shape = vis_ctx.sel_dims();
    let tile_xyzs: Vec<XyzGeodetic> = obs_ctx.ant_positions_geodetic().collect();
    let centroid_timestamps: Vec<Epoch> = vis_ctx.timeseries(false, true).collect();
    let phase_centre = obs_ctx.phase_centre;
    let array_pos = obs_ctx.array_pos;
    let ant_pairs = vis_ctx.sel_baselines.clone();
    let part_uvws = calc_part_uvws(&ant_pairs, &centroid_timestamps, phase_centre, array_pos, &tile_xyzs);
    let weight_factor = vis_ctx.weight_factor();
    let freqs_hz = vis_ctx.frequencies_hz();

    Array3::from_shape_fn(
        (sel_shape.0, sel_shape.2, sel_shape.1),
        |(ts, bl, ch)| {
            let (ant1, ant2) = ant_pairs[bl];
            let uvw= part_uvws[[ts, ant1]] - part_uvws[[ts, ant2]];
            // to convert to RTS uvw (wavelengths) from PAL uvw (meters), dividy by λ.
            let lambda = VEL_C / freqs_hz[ch];
            let (u, v) = (uvw.u/lambda, uvw.v/lambda);
            let uv_sq = u * u + v * v;
            weight_factor as f32 * (1.0 - (-uv_sq/(2.0*short_sigma*short_sigma).exp() )) as f32
        },
    )
}

// rotate visibilities and average them (along with weights) in time and frequency given by vis_ctx
fn rotate_accumulate<F>(
    jones_from: ArrayView3<Jones<F>>,
    mut jones_to: ArrayViewMut3<Jones<F>>,
    weight_from: ArrayView3<f32>,
    mut weight_to: ArrayViewMut3<f32>,
    vis_ctx: &VisContext,
    obs_ctx: &MarluObsContext,
    phase_to: RADec,
) where
    F: Float + Num + NumAssign + Default,
{
    let freqs_hz = vis_ctx.frequencies_hz();
    let tile_xyzs: Vec<XyzGeodetic> = obs_ctx.ant_positions_geodetic().collect();
    let centroid_timestamps: Vec<Epoch> = vis_ctx.timeseries(false, true).collect();
    let phase_from = obs_ctx.phase_centre;
    let array_pos = obs_ctx.array_pos;
    let ant_pairs = vis_ctx.sel_baselines.clone();
    let from_dims = jones_from.dim();
    let avg_time = vis_ctx.avg_time;
    let avg_freq = vis_ctx.avg_freq;

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
    let part_uvws_from = calc_part_uvws(&ant_pairs, &centroid_timestamps, phase_from, array_pos, &tile_xyzs);
    let part_uvws_to = calc_part_uvws(&ant_pairs, &centroid_timestamps, phase_to, array_pos, &tile_xyzs);

    // iterate along time axis in chunks of avg_time
    for (
        jones_chunk,
        weight_chunk,
        mut jones_to,
        mut weight_to,
        part_uvws_from,
        part_uvws_to
    ) in izip!(
        jones_from.axis_chunks_iter(Axis(0), avg_time),
        weight_from.axis_chunks_iter(Axis(0), avg_time),
        jones_to.outer_iter_mut(),
        weight_to.outer_iter_mut(),
        part_uvws_from.axis_chunks_iter(Axis(0), avg_time),
        part_uvws_to.axis_chunks_iter(Axis(0), avg_time),
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
            &ant_pairs
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
                    part_uvws_from,
                    part_uvws_to,
                ) in izip!(
                    jones_chunk.axis_iter(Axis(0)),
                    weight_chunk.axis_iter(Axis(0)),
                    part_uvws_from.axis_iter(Axis(0)),
                    part_uvws_to.axis_iter(Axis(0))
                ) {
                    let arg_w_diff = (part_uvws_to[[ant1]].w - part_uvws_to[[ant2]].w) - (part_uvws_from[[ant1]].w - part_uvws_from[[ant2]].w);
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
                        let jones_rotated = *jones * rotation;
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

// rotate the visibilities in place
fn vis_rotate<F>(
    mut jones_array: ArrayViewMut3<Jones<F>>,
    vis_ctx: &VisContext,
    obs_ctx: &MarluObsContext,
    phase_to: RADec,
) where
    F: Float + Num + NumAssign + Default,
{
    let freqs_hz = vis_ctx.frequencies_hz();
    let tile_xyzs: Vec<XyzGeodetic> = obs_ctx.ant_positions_geodetic().collect();
    let centroid_timestamps: Vec<Epoch> = vis_ctx.timeseries(false, true).collect();
    let phase_from = obs_ctx.phase_centre;
    let array_pos = obs_ctx.array_pos;
    let ant_pairs = vis_ctx.sel_baselines.clone();
    let jones_dims = jones_array.dim();

    // eprintln!("jones_dims {:?}", jones_dims);

    assert_eq!(jones_dims.0, centroid_timestamps.len());
    assert_eq!(jones_dims.1, ant_pairs.len());
    assert_eq!(jones_dims.2, freqs_hz.len());

    eprintln!("phase from {:?} to {:?}", phase_from, phase_to);

    // let lmn = phase_to.to_lmn(phase_from);

    // pre-compute partial uvws:
    let part_uvws_from = calc_part_uvws(&ant_pairs, &centroid_timestamps, phase_from, array_pos, &tile_xyzs);
    let part_uvws_to = calc_part_uvws(&ant_pairs, &centroid_timestamps, phase_to, array_pos, &tile_xyzs);

    // iterate along time axis in chunks of avg_time
    for (
        mut jones_array,
        part_uvws_from,
        part_uvws_to,
    ) in izip!(
        jones_array.outer_iter_mut(),
        part_uvws_from.outer_iter(),
        part_uvws_to.outer_iter(),
    ) {
        // iterate along baseline axis
        for (
            mut jones_array,
            &(ant1, ant2)
        ) in izip!(
            jones_array.outer_iter_mut(),
            &ant_pairs
        ) {
            let arg_w_diff = (part_uvws_to[[ant1]].w - part_uvws_to[[ant2]].w) - (part_uvws_from[[ant1]].w - part_uvws_from[[ant2]].w);
            // iterate along frequency axis
            for (
                jones,
                &freq_hz,
            ) in izip!(
                jones_array.iter_mut(),
                &freqs_hz
            ) {
                // in RTS, uvw is in units of λ but pal uvw is in meters, so divide by wavelength
                let rotation = cexp(F::from(-TAU * (arg_w_diff) * (freq_hz as f64) / VEL_C).unwrap());
                *jones *= rotation;
            }
        }
    };
}

fn simulate_accumulate_iono(
    mut jones: ArrayViewMut3<Jones<f32>>,
    vis_ctx: &VisContext,
    #[cfg(feature = "cuda")] use_cpu_for_modelling: bool,
    beam: &Box<dyn Beam>,
    source: IonoSource,
    source_name: String,
    obs_ctx: &MarluObsContext,
) {
    // let beam = create_no_beam_object(tile_xyzs.len());
    let freqs_hz = vis_ctx.frequencies_hz();
    let jones_shape = jones.dim();
    // Temporary visibility array, re-used for each timestep
    let mut vis_tmp = Array3::from_elem((1, jones_shape.1, jones_shape.2), Jones::<f32>::default());
    let mut vis_ctx_tmp = VisContext {
        num_sel_timesteps: 1,
        ..vis_ctx.clone()
    };

    let tile_xyzs: Vec<XyzGeodetic> = obs_ctx.ant_positions_geodetic().collect();
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
        obs_ctx.phase_centre,
        obs_ctx.array_pos.longitude_rad,
        obs_ctx.array_pos.latitude_rad,
        // TODO: Allow the user to turn off precession.
        true,
    )
    .unwrap();

    for (epoch, mut jones) in izip!(centroid_timestamps, jones.outer_iter_mut()) {
        let mut vis_slice = vis_tmp.slice_mut(s![0, .., ..]);
        eprintln!("modelling epoch {:?} {:?} with consts {:?}", epoch.as_gregorian_utc_str(), vis_slice.dim(), source.iono_consts);
        vis_slice.fill(Jones::default());
        modeller
            .model_timestep(vis_slice.view_mut(), epoch)
            .unwrap();
        drop(vis_slice);
        vis_ctx_tmp.start_timestamp = epoch - vis_ctx.int_time / 2.0;
        if (source.iono_consts.0 - 0.).abs() > 1e-9 || (source.iono_consts.1 - 0.).abs() > 1e-9 {
            apply_iono(
                vis_tmp.view_mut(),
                &vis_ctx_tmp,
                obs_ctx,
                source.iono_consts,
            );
        }
        // let vis_slice = vis_tmp.slice(s![0, .., ..]);
        for (vis_model, jones) in izip!(vis_tmp.iter(), jones.iter_mut()) {
            *jones += *vis_model;
        }
    }
}

// apply ionospheric rotation of exp(-2πi(αu+βv)λ²)
fn apply_iono<F>(
    jones: ArrayViewMut3<Jones<F>>,
    vis_ctx: &VisContext,
    obs_ctx: &MarluObsContext,
    // constants of proportionality for ionospheric offset in l,m
    const_lm: (f64, f64),
) where
    F: Float + Num + NumAssign + Default,
{
    let jones_dims = jones.dim();

    let freqs_hz = vis_ctx.frequencies_hz();
    let tile_xyzs: Vec<XyzGeodetic> = obs_ctx.ant_positions_geodetic().collect();
    let centroid_timestamps: Vec<Epoch> = vis_ctx.timeseries(false, true).collect();
    let phase_centre = obs_ctx.phase_centre;
    let array_pos = obs_ctx.array_pos;
    let ant_pairs = vis_ctx.sel_baselines.clone();

    assert_eq!(jones_dims.0, centroid_timestamps.len());
    assert_eq!(jones_dims.1, ant_pairs.len());
    assert_eq!(jones_dims.2, freqs_hz.len());

    // pre-compute partial uvws:
    let part_uvws = calc_part_uvws(&ant_pairs, &centroid_timestamps, phase_centre, array_pos, &tile_xyzs);

    _apply_iono(jones, part_uvws, &ant_pairs, const_lm, &freqs_hz);
}

fn _apply_iono<F>(
    mut jones: ArrayViewMut3<Jones<F>>,
    part_uvws: Array2<UVW>,
    ant_pairs: &[(usize, usize)],
    const_lm: (f64, f64),
    freqs_hz: &[f64]
) where
F: Float + Num + NumAssign + Default,
{
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
            // iterate along frequency axis
            for (
                jones,
                &freq_hz,
            ) in izip!(
                jones.iter_mut(),
                freqs_hz
            ) {
                // in RTS, uvw is in units of λ but pal uvw is in meters, so divide by wavelength,
                // but we're also multiplying by λ², so just multiply by λ
                let lambda = VEL_C / freq_hz;
                let rotation = cexp(F::from(-TAU * uv_lm * lambda).unwrap());
                *jones *= rotation;
            }
        }
    }
}

// Generic factorial function.
//
// Stolen from https://gist.github.com/derceg/9362312
// Accepts a reference to any type
// that implements the Int trait
// (e.g. int, uint, BigInt, BigUint,
// etc).
// It is up to the caller to ensure
// that the input is >= 0. */
pub fn factorial<T>(num: &T) -> T
where
    T: Num + One + Zero + Copy,
{
	if *num == Zero::zero() ||
		*num == One::one() {
		One::one()
	} else {
		*num * factorial(&(*num - One::one()))
	}
}

// apply ionospheric rotation approximation by `order` taylor expansion terms
fn apply_iono_approx<F>(
    mut jones: ArrayViewMut3<Jones<F>>,
    vis_ctx: &VisContext,
    obs_ctx: &MarluObsContext,
    // constants of proportionality for ionospheric offset in l,m
    const_lm: (f64, f64),
    order: usize,
) where
    F: Float + Num + NumAssign + Default,
{
    let jones_dims = jones.dim();
    let freqs_hz = vis_ctx.frequencies_hz();
    let tile_xyzs: Vec<XyzGeodetic> = obs_ctx.ant_positions_geodetic().collect();
    let centroid_timestamps: Vec<Epoch> = vis_ctx.timeseries(false, true).collect();
    let phase_centre = obs_ctx.phase_centre;
    let array_pos = obs_ctx.array_pos;
    let ant_pairs = vis_ctx.sel_baselines.clone();

    assert_eq!(jones_dims.0, centroid_timestamps.len());
    assert_eq!(jones_dims.1, ant_pairs.len());
    assert_eq!(jones_dims.2, freqs_hz.len());

    // pre-compute partial uvws:
    let part_uvws = calc_part_uvws(&ant_pairs, &centroid_timestamps, phase_centre, array_pos, &tile_xyzs);

    let lambdas = freqs_hz.iter().map(|freq_hz| VEL_C / freq_hz).collect::<Vec<_>>();

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
            &ant_pairs
        ) {
            let uvw= part_uvws[[ant1]] - part_uvws[[ant2]];
            let uv_lm = uvw.u * const_lm.0 + uvw.v * const_lm.1;
            // iterate along frequency axis
            for (
                jones,
                &lambda,
            ) in izip!(
                jones.iter_mut(),
                &lambdas
            ) {
                // in RTS, uvw is in units of λ but pal uvw is in meters, so divide by wavelength,
                // but we're also multiplying by λ², so just multiply by λ

                // first order taylor expansion, data D from rotation of model M
                // D = M * exp(-i * phi * lambda^2 )
                //   = M * (1 - i * phi * lambda^2 + ... )
                let exponent = - Complex::i() * F::from(TAU * uv_lm * lambda).unwrap();
                let rotation: Complex<F> = (0..=order).map(|n| {
                    exponent.powi(n as i32) / F::from(factorial(&n)).unwrap()
                }).sum();
                *jones *= rotation;
            }
        }
    }
}


// the offsets as defined by the RTS code
fn get_offsets_rts(
    unpeeled: ArrayView3<Jones<f32>>,
    weights: ArrayView3<f32>,
    model: ArrayView3<Jones<f32>>,
    vis_ctx: &VisContext,
    obs_ctx: &MarluObsContext,
    // src_name: String,
) -> Vec<f64> {
    let freqs_hz = vis_ctx.frequencies_hz();
    let tile_xyzs: Vec<XyzGeodetic> = obs_ctx.ant_positions_geodetic().collect();
    let centroid_timestamps: Vec<Epoch> = vis_ctx.timeseries(false, true).collect();
    let phase_centre = obs_ctx.phase_centre;
    let array_pos = obs_ctx.array_pos;
    let ant_pairs = vis_ctx.sel_baselines.clone();
    let jones_dims = unpeeled.dim();

    assert_eq!(jones_dims.0, centroid_timestamps.len());
    assert_eq!(jones_dims.1, ant_pairs.len());
    assert_eq!(jones_dims.2, freqs_hz.len());
    assert_eq!(jones_dims, weights.dim());
    assert_eq!(jones_dims, model.dim());

    let mut offsets = Array::zeros((jones_dims.0, 2));

    // pre-compute partial uvws:
    let part_uvws = calc_part_uvws(&ant_pairs, &centroid_timestamps, phase_centre, array_pos, &tile_xyzs);

    // iterate over time
    for (
        unpeeled,
        weights,
        model,
        mut offsets,
        part_uvws
    ) in izip!(
        unpeeled.outer_iter(),
        weights.outer_iter(),
        model.outer_iter(),
        offsets.outer_iter_mut(),
        part_uvws.outer_iter(),
    ) {
        // a-terms used in least-squares estimator
        let (mut a_uu, mut a_uv, mut a_vv) = (0., 0., 0.);
        // A-terms used in least-squares estimator
        let (mut aa_u, mut aa_v) = (0., 0.);

        // iterate over frequency
        for (
            unpeeled,
            weights,
            model,
            freq_hz,
        ) in izip!(
            unpeeled.axis_iter(Axis(1)),
            weights.axis_iter(Axis(1)),
            model.axis_iter(Axis(1)),
            freqs_hz.iter(),
        ) {
            let lambda = VEL_C / freq_hz;
            // lambda^2
            let lambda_2 = lambda * lambda;
            // lambda^4
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
                &ant_pairs,
            ) {
                let uvw= part_uvws[[ant1]] - part_uvws[[ant2]];

                if *weight > 0. {
                    // stokes I power of the unpeeled visibilities (Data)
                    let unpeeled_i = 0.5 * (unpeeled[0] + unpeeled[3]);
                    // stokes I power of the model visibilities (Model)
                    let model_i = 0.5 * (model[0] + model[3]);

                    let mr = (model_i.re as f64) * (unpeeled_i - model_i).im as f64;
                    let mm = (model_i.re as f64) * model_i.re as f64;

                    let weight_f64 = *weight as f64;

                    // to convert to RTS uvw (wavelengths) from PAL uvw (meters), dividy by λ.
                    let (u, v) = (uvw.u/lambda, uvw.v/lambda);
                    a_uu += weight_f64 * mm * u * u * lambda_4;
                    a_uv += weight_f64 * mm * u * v * lambda_4;
                    a_vv += weight_f64 * mm * v * v * lambda_4;
                    aa_u  += weight_f64 * mr * u * -lambda_2;
                    aa_v  += weight_f64 * mr * v * -lambda_2;
                }
            }
        }
        let delta = TAU * ( a_uu*a_vv - a_uv*a_uv );

        offsets[[0]] = (aa_u*a_vv - aa_v*a_uv) / delta;
        offsets[[1]] = (aa_v*a_uu - aa_u*a_uv) / delta;
    }
    eprintln!("{:?}", offsets);
    offsets.axis_iter(Axis(1)).map(|x| x.mean().unwrap()).collect_vec()
}

fn calculate_chi_squared(
    vis: ArrayView3<Jones<f32>>,
    weights: ArrayView3<f32>,
    model: ArrayView3<Jones<f32>>,
) -> (f32, f32) {

    let mut chi_squared_re = 0.0;
    let mut chi_squared_im = 0.0;

    // iterate along time axis
    for (
        vis,
        model,
        weights,
    ) in izip!(
        vis.outer_iter(),
        model.outer_iter(),
        weights.outer_iter(),
    ) {
        // iterate along baseline axis
        for (
            vis,
            model,
            weights,
        ) in izip!(
            vis.outer_iter(),
            model.outer_iter(),
            weights.outer_iter(),
        ) {
            for (
                vis,
                model,
                weight,
            ) in izip!(
                vis.iter(),
                model.iter(),
                weights.iter(),
            ) {
                let vis_i = vis[0] + vis[3];
                let model_i = model[0] + model[3];
                chi_squared_re += ((vis_i.re - model_i.re) / weight).powi(2);
                chi_squared_im += ((vis_i.im - model_i.im) / weight).powi(2);
            }
        }
    }
    (
        chi_squared_re,
        chi_squared_im
    )
}

// calculate the difference between the synthetic visibilities and the model rotated by a range of offsets
fn plot_chi_squared(
    unpeeled: ArrayView3<Jones<f32>>,
    weights: ArrayView3<f32>,
    model: ArrayView3<Jones<f32>>,
    vis_ctx: &VisContext,
    obs_ctx: &MarluObsContext,
    chi_squared_path: PathBuf,
) {
    eprintln!("Plotting chi squared to {:?}", &chi_squared_path);
    let mut srclist_buf = BufWriter::new(File::create(&chi_squared_path).unwrap());
    srclist_buf.write_all("alpha\tbeta\tchisq_re\tchisq_im\tchisq_o1_re\tchisq_o1_im\tchisq_o3_re\tchisq_o3_im\n".as_bytes()).unwrap();

    let freqs_hz = vis_ctx.frequencies_hz();

    let min_freq = freqs_hz[0];
    let max_lambda_sq = (VEL_C * VEL_C) / (min_freq * min_freq);
    // number of arc seconds to step phi by
    let iono_step_as = 5_f64;
    // the range of angle offsets to try
    let phis_as = (-20..20).map(|i| i as f64 * iono_step_as).collect::<Vec<_>>();
    for phi_as in phis_as {
        let worst_angle_offsets = (
            (phi_as/60./60.).to_radians(),
            0.
        );
        let const_lm = get_iono_consts(max_lambda_sq, obs_ctx.phase_centre, worst_angle_offsets);
        // copy unpeeled and apply ionospheric rotation
        let mut iono_rotated = Array3::from_shape_fn(model.dim(), |idx| {
            model[idx]
        });
        apply_iono(
            iono_rotated.view_mut(),
            vis_ctx,
            obs_ctx, const_lm
        );
        let (chisq_re, chisq_im) = calculate_chi_squared(
            unpeeled.view(),
            weights.view(),
            iono_rotated.view(),
        );
        let mut iono_rotated_order_1 = Array3::from_shape_fn(model.dim(), |idx| {
            model[idx]
        });
        apply_iono_approx(iono_rotated_order_1.view_mut(), vis_ctx, obs_ctx, const_lm, 1);
        let (chisq_order_1_re, chisq_order_1_im) = calculate_chi_squared(
            unpeeled.view(),
            weights.view(),
            iono_rotated_order_1.view(),
        );
        let mut iono_rotated_order_3 = Array3::from_shape_fn(model.dim(), |idx| {
            model[idx]
        });
        apply_iono_approx(iono_rotated_order_3.view_mut(), vis_ctx, obs_ctx, const_lm, 3);
        let (chisq_order_3_re, chisq_order_3_im) = calculate_chi_squared(
            unpeeled.view(),
            weights.view(),
            iono_rotated_order_3.view(),
        );
        srclist_buf.write_all(format!(
            "{:12.10}\t{:12.10}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            const_lm.0, const_lm.1,
            chisq_re, chisq_im,
            chisq_order_1_re, chisq_order_1_im,
            chisq_order_3_re, chisq_order_3_im
        ).as_bytes()).unwrap();
    }
}

// calculate partial uvw components for each antenna. uvw = part_uvw[ant1] - part_uvw[ant2]
fn calc_part_uvws(
    ant_pairs: &[(usize, usize)],
    centroid_timestamps: &[Epoch],
    phase_centre: RADec,
    array_pos: LatLngHeight,
    tile_xyzs: &[XyzGeodetic]
) -> Array2<UVW> {
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
    part_uvws
}

fn simulate_write(
    vis_ctx: &VisContext,
    #[cfg(feature = "cuda")] use_cpu_for_modelling: bool,
    beam: &Box<dyn Beam>,
    source_list: SourceList,
    obs_ctx: &MarluObsContext,
    out_path: Option<PathBuf>,
) -> Array3<Jones<f32>> {
    // let beam = create_no_beam_object(tile_xyzs.len());
    let freqs_hz = vis_ctx.frequencies_hz();
    let sel_shape = vis_ctx.sel_dims();
    // Construct our visibilities array. This will be re-used for each timestep
    // before it's written to disk. Simulated vis is [baseline][chan] but
    // vis output requires [timestep][chan][baseline], this is re-used.
    let mut vis_tmp = Array3::from_elem((1, sel_shape.1, sel_shape.2), Jones::<f32>::default());
    let weight_tmp = Array3::from_elem(
        (1, sel_shape.1, sel_shape.2),
        vis_ctx.weight_factor() as f32,
    );
    let mut vis_result = Array3::from_elem(
        (sel_shape.0, sel_shape.2, sel_shape.1),
        Jones::<f32>::default(),
    );


    let tile_xyzs: Vec<XyzGeodetic> = obs_ctx.ant_positions_geodetic().collect();
    let centroid_timestamps = vis_ctx.timeseries(false, true);
    let phase_centre = obs_ctx.phase_centre;
    let array_pos = obs_ctx.array_pos;
    eprintln!("simulating to {:?} at phase {}", out_path.as_ref().unwrap_or(&"".into()), &phase_centre);
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
        writer.initialize(vis_ctx, obs_ctx).unwrap();
        writer
    });

    for (epoch, mut vis_result) in izip!(centroid_timestamps, vis_result.outer_iter_mut()) {
        eprintln!("modelling epoch {:?} {:?}", epoch.as_gregorian_utc_str(), vis_result.dim());

        vis_result.fill(Jones::default());
        modeller
            .model_timestep(vis_result.view_mut(), epoch)
            .unwrap();

        // transpose model vis to output ordering. first axis is baseline.
        for (vis_model, mut vis_out) in
            izip!(vis_result.outer_iter(), vis_tmp.axis_iter_mut(Axis(2)))
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
                    vis_tmp.view(),
                    weight_tmp.view(),
                    &chunk_vis_ctx,
                    &tile_xyzs,
                    false,
                )
                .unwrap()
        }
    }
    vis_result
}

fn write_vis(
    vis_write: ArrayView3<Jones<f32>>,
    weight_write: ArrayView3<f32>,
    vis_ctx: &VisContext,
    obs_ctx: &MarluObsContext,
    out_path: PathBuf,
) {
    // timesteps, channels, baselines
    let sel_shape = vis_ctx.sel_dims();

    let write_dims = vis_write.dim();
    assert_eq!(sel_shape.0, write_dims.0);
    assert_eq!(sel_shape.1, write_dims.2);
    assert_eq!(sel_shape.2, write_dims.1);

    let tile_xyzs: Vec<XyzGeodetic> = obs_ctx.ant_positions_geodetic().collect();
    let phase_centre = obs_ctx.phase_centre;
    let array_pos = obs_ctx.array_pos;
    let mut writer = MeasurementSetWriter::new(&out_path, phase_centre, Some(array_pos));
    writer.initialize(vis_ctx, obs_ctx).unwrap();

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

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    // create visibilities with an exaggerated ionospheric rotation, to validate by eye that the offsets are correct.
    #[test]
    fn vaidate_iono_rotation() {
        // iono_srclist: &IonoSourceList

        if !OUT_DIR.is_dir() {
            std::fs::create_dir_all(OUT_DIR.as_path()).unwrap();
        }
        let mut array_pos = LatLngHeight::new_mwa();
        array_pos.latitude_rad = 0.;
        array_pos.longitude_rad = 0.;

        let (mut vis_ctx, obs_ctx) = get_obs_metadata(array_pos);
        vis_ctx.freq_resolution_hz=1_280_000.;
        vis_ctx.start_freq_hz=81_920_000.;
        vis_ctx.num_sel_chans=12;

        let beam = get_beam(obs_ctx.ant_positions_geodetic().count());

        let freqs_hz = vis_ctx.frequencies_hz();
        let max_lambda_sq = (VEL_C / freqs_hz.first().unwrap()).powi(2);
        let min_lambda_sq = (VEL_C / freqs_hz.last().unwrap()).powi(2);
        eprintln!("min / max lambda sq: {} / {}", min_lambda_sq, max_lambda_sq);
        // a big ionospheric rotation in RA
        let worst_offsets_rad = (
            (1.).to_radians(),
            0.,
        );
        // position the source on the nearest degree grid
        let source_pos = RADec::new_degrees(
            obs_ctx.phase_centre.ra.to_degrees().round(),
            obs_ctx.phase_centre.dec.to_degrees().round()
        );
        let iono_source = IonoSource {
            source: Source {components: vec1![
                SourceComponent {
                    radec: source_pos,
                    comp_type: ComponentType::Point,
                    flux_type: FluxDensityType::PowerLaw {
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
            iono_consts: get_iono_consts(
                max_lambda_sq, source_pos, worst_offsets_rad
            )
        };

        let sel_shape = vis_ctx.sel_dims();
        let mut vis_synth = Array3::from_elem(
            (sel_shape.0, sel_shape.2, sel_shape.1), Jones::<f32>::zero()
        );
        let weight_synth = Array3::from_elem(
            (sel_shape.0, sel_shape.2, sel_shape.1), vis_ctx.weight_factor() as f32
        );

        let iono_test_path = OUT_DIR.join("vis_iono_test.ms");

        simulate_accumulate_iono(
            vis_synth.view_mut(),
            &vis_ctx,
            #[cfg(feature = "cuda")]
            use_cpu_for_modelling,
            &beam,
            iono_source,
            "iono_test".into(),
            &obs_ctx,
        );
        write_vis(
            vis_synth.view(),
            weight_synth.view(),
            &vis_ctx,
            &obs_ctx,
            iono_test_path,
        );

    }

    // simplify the problem down to 1 dimension
    // lat/lng = 0
    // this only works if you override the hardcoded mwa location in hyperdrive/model.
    #[test]
    fn one_dimension() {
        let array_pos = LatLngHeight {
            longitude_rad: 0.,
            latitude_rad: 0.,
            height_metres: 100.,
        };
        let mut obs_time = Epoch::from_gpst_seconds(1090008640.);
        // shift zenith_time to the nearest time when the phase centre is at zenith
        let obs_lst_rad = get_lmst(obs_time, array_pos.longitude_rad);
        if obs_lst_rad.abs() > 1e-6 {
            let sidereal2solar = 365.24/366.24;
            obs_time -= Duration::from_f64(sidereal2solar*obs_lst_rad/TAU, Unit::Day);
        }
        let zenith_lst_rad = get_lmst(obs_time, array_pos.longitude_rad);
        let phase_centre = RADec::from_hadec(HADec::new(0., array_pos.latitude_rad), zenith_lst_rad);
        let timesteps = vec![obs_time];

        let tile_names: Vec<String> = "OXYZ".chars().map(|c| c.to_string()).collect();
        let tile_xyzs= vec![
            XyzGeodetic {x: 0., y: 0., z: 0.},
            XyzGeodetic {x: 1., y: 0., z: 0.},
            XyzGeodetic {x: 0., y: 1., z: 0.},
            XyzGeodetic {x: 0., y: 0., z: 1.},
        ];
        let ant_pairs = vec![
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
        ];

        let lambda = 1.;
        let freq_hz = VEL_C/lambda;
        dbg!(freq_hz);
        let freqs_hz = vec![freq_hz];

        let iono_consts = (
            (1./60.).to_radians(),
            0.,
        );
        dbg!(&iono_consts);

        let source_list = SourceList::from(indexmap! {
            "One".into() => Source { components: vec1![
                SourceComponent {
                    radec: phase_centre,
                    comp_type: ComponentType::Point,
                    flux_type: FluxDensityType::PowerLaw {
                        si: 0.,
                        fd: FluxDensity {
                            freq: freq_hz,
                            i: 1.0,
                            q: 0.0,
                            u: 0.0,
                            v: 0.0,
                        },
                    },
                }
            ]}
        });

        let beam = get_beam(tile_xyzs.len());

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

        // let sel_shape = vis_ctx.sel_dims();
        let mut vis_model = Array3::from_elem(
            (1, ant_pairs.len(), 1), Jones::<f32>::zero()
        );
        let model_slice = vis_model.slice_mut(s![0_usize, .., ..]);
        modeller
            .model_timestep(model_slice, obs_time)
            .unwrap();

        // copy model slice into vis slice and iono shift
        let mut vis_synth = Array3::from_shape_fn(vis_model.dim(), |idx| vis_model[idx]);
        let part_uvws = calc_part_uvws(&ant_pairs, &timesteps, phase_centre, array_pos, &tile_xyzs);
        _apply_iono(vis_synth.view_mut(), part_uvws, &ant_pairs, iono_consts, &freqs_hz);

        // generate weights of 1.
        let weights = Array3::from_elem(vis_model.dim(), 1.);

        let vis_ctx = VisContext {
            num_sel_timesteps: 1,
            start_timestamp: obs_time,
            int_time: Duration::from_f64(1., Unit::Second),
            num_sel_chans: 1,
            start_freq_hz: freq_hz,
            freq_resolution_hz: 10_000.,
            sel_baselines: ant_pairs,
            avg_time: 1,
            avg_freq: 1,
            num_vis_pols: 4,
        };

        let sched_start_timestamp = vis_ctx.start_timestamp;
        let sched_duration = vis_ctx.int_time * vis_ctx.num_sel_timesteps as f64;
        let obs_name = Some("Simulated 1D visibilities".into());
        let obs_ctx = MarluObsContext {
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

        let offsets_rts = get_offsets_rts(
            vis_synth.view(),
            weights.view(),
            vis_model.view(),
            &vis_ctx,
            &obs_ctx,
            // source_name.clone(),
        );

        // println!("retrieved offsets {:?}", offsets_rts);
        assert_abs_diff_eq!(iono_consts.0, offsets_rts[0], epsilon=1e-7);
        assert_abs_diff_eq!(iono_consts.1, offsets_rts[1], epsilon=1e-7);

    }
}
// use mwa_hyperdrive::{
//     calibrate::{
//         channels_to_chanblocks, di::calibrate_timeblocks, timesteps_to_timeblocks, solutions::CalibrationSolutions
//     },
//     constants::{DEFAULT_MAX_ITERATIONS, DEFAULT_MIN_THRESHOLD, DEFAULT_STOP_THRESHOLD},
// };
// use std::{
//     collections::HashSet,
// };

// fn di_cal(
//     vis_ctx: &VisContext,
//     vis_unpeeled: &Array3<Jones<f32>>,
//     vis_model: &ndarray::ArrayBase<ndarray::OwnedRepr<Jones<f32>>, ndarray::Dim<[usize; 3]>>,
//     obs_ctx: &MarluObsContext,
// ) -> CalibrationSolutions {
//     let timestamps: Vec<Epoch> = vis_ctx.timeseries(true, true).collect();
//     let timesteps: Vec<usize> = (0..timestamps.len()).collect();
//     let timeblocks = timesteps_to_timeblocks(&timestamps, 4, &timesteps);
//     let frequencies: Vec<u64> = vis_ctx
//         .avg_frequencies_hz()
//         .iter()
//         .map(|&f| f as u64)
//         .collect();
//     let fences = channels_to_chanblocks(
//         &frequencies,
//         Some(vis_ctx.avg_freq_resolution_hz()),
//         1,
//         &HashSet::<usize>::new(),
//     );
//     let baseline_weights = vec![1.0; vis_ctx.sel_baselines.len()];
//     let (sols, _) = calibrate_timeblocks(
//         vis_unpeeled.view(),
//         vis_model.view(),
//         &timeblocks,
//         &fences[0].chanblocks,
//         &baseline_weights,
//         DEFAULT_MAX_ITERATIONS,
//         DEFAULT_STOP_THRESHOLD,
//         DEFAULT_MIN_THRESHOLD,
//         false,
//         false,
//     );
//     // "Complete" the solutions.
//     sols.into_cal_sols(
//         obs_ctx.num_ants(),
//         &[],
//         &fences[0].flagged_chanblock_indices,
//         None,
//     )
// }