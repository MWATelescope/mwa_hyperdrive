// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <math.h>
#include <stdlib.h>

#include "common.cuh"
#include "model.h"
#include "types.h"

const int NUM_THREADS_PER_BLOCK_POINTS = 256;
const int NUM_THREADS_PER_BLOCK_GAUSSIANS = 256;
const int NUM_THREADS_PER_BLOCK_SHAPELETS = 32;
const int NUM_THREADS_PER_BLOCK_AUTOS = 512;

inline __device__ JONES extrap_power_law_fd(const FLOAT freq, const JONES ref_flux_density,
                                            const FLOAT spectral_index) {
    const FLOAT flux_ratio = POW(freq / POWER_LAW_FD_REF_FREQ, spectral_index);
    return ref_flux_density * flux_ratio;
}

inline __device__ JONES extrap_curved_power_law_fd(const FLOAT freq, const JONES ref_flux_density,
                                                   const FLOAT spectral_index, const FLOAT q) {
    const FLOAT flux_ratio = POW(freq / POWER_LAW_FD_REF_FREQ, spectral_index);
    const FLOAT log_term = LOG(freq / POWER_LAW_FD_REF_FREQ);
    const FLOAT curved_component = EXP(q * log_term * log_term);
    return ref_flux_density * flux_ratio * curved_component;
}

inline __device__ FLOAT get_gaussian_envelope(const GaussianParams g_params, const UVW uvw) {
    FLOAT s_pa, c_pa;
    SINCOS(g_params.pa, &s_pa, &c_pa);
    // Temporary variables for clarity.
    const FLOAT k_x = uvw.u * s_pa + uvw.v * c_pa;
    const FLOAT k_y = uvw.u * c_pa - uvw.v * s_pa;
    return EXP(EXP_CONST * ((g_params.maj * g_params.maj) * (k_x * k_x) + (g_params.min * g_params.min) * (k_y * k_y)));
}

inline __device__ COMPLEX get_shapelet_envelope(const GaussianParams g_params, const ShapeletUV s_uv,
                                                const int num_coeffs, const ShapeletCoeff *coeffs,
                                                const FLOAT *shapelet_basis_values) {
    const FLOAT I_POWERS_REAL[4] = {1.0, 0.0, -1.0, 0.0};
    const FLOAT I_POWERS_IMAG[4] = {0.0, 1.0, 0.0, -1.0};

    FLOAT s_pa, c_pa;
    SINCOS(g_params.pa, &s_pa, &c_pa);

    // Temporary variables for clarity.
    const FLOAT x = s_uv.u * s_pa + s_uv.v * c_pa;
    const FLOAT y = s_uv.u * c_pa - s_uv.v * s_pa;
    const FLOAT const_x = g_params.maj * SHAPELET_CONST;
    const FLOAT const_y = -g_params.min * SHAPELET_CONST;
    const FLOAT x_pos = x * const_x + SBF_C;
    const FLOAT y_pos = y * const_y + SBF_C;
    // If the positions (indices) are negative, our range of shapelet basis
    // functions/values/whatever is too small for the incoming shapelet UVs. CHJ
    // spotted this when (accidentally) trying to simulate Fornax A with SDC3
    // data; these indices only went negative for the long baselines. Instead of
    // increasing the range of allowed UVs, we're doing the lazy thing of
    // sweeping this issue under the carpet, because (1) MWA data is unlikely to
    // ever have baselines rivalling SKA-Low and (2) SDC3 data doesn't use
    // Fornax A or shapelets.
    int x_pos_int = x_pos < 0 ? 0 : (int)FLOOR(x_pos);
    int y_pos_int = y_pos < 0 ? 0 : (int)FLOOR(y_pos);

    COMPLEX envelope = MAKE_COMPLEX(0.0, 0.0);
    for (int i_coeff = 0; i_coeff < num_coeffs; i_coeff++) {
        const ShapeletCoeff coeff = coeffs[i_coeff];

        FLOAT x_low = shapelet_basis_values[SBF_L * (int)coeff.n1 + x_pos_int];
        FLOAT x_high = shapelet_basis_values[SBF_L * (int)coeff.n1 + x_pos_int + 1];
        FLOAT u_value = x_low + (x_high - x_low) * (x_pos - FLOOR(x_pos));

        FLOAT y_low = shapelet_basis_values[SBF_L * (int)coeff.n2 + y_pos_int];
        FLOAT y_high = shapelet_basis_values[SBF_L * (int)coeff.n2 + y_pos_int + 1];
        FLOAT v_value = y_low + (y_high - y_low) * (y_pos - FLOOR(y_pos));

        FLOAT rest = coeff.value * u_value * v_value;

        // I_POWER_TABLE stuff. The intention is just to find the
        // appropriate power of i, i.e.:
        // index = (n1 + n2) % 4    (so that index is between 0 and 3 inclusive)
        // i^index, e.g.
        // i^0 =  1.0 + 0.0i
        // i^1 =  0.0 + 1.0i
        // i^2 = -1.0 + 0.0i
        // i^3 =  0.0 - 1.0i
        //
        // The following is my attempt at doing this efficiently.
        int i_power_index = (int)((coeff.n1 + coeff.n2) % 4);
        COMPLEX i_power = MAKE_COMPLEX(I_POWERS_REAL[i_power_index], I_POWERS_IMAG[i_power_index]);

        envelope += i_power * rest;
    }

    return envelope;
}

// The sky-model kernels below factorise the measurement equation per tile, as
// DP3's predict (and hyperdrive's CPU modeller) do:
//
// - flux densities are evaluated once per (channel, component), not per
//   visibility;
// - the phase term is evaluated once per (tile, channel, component), and a
//   baseline's phasor is cis(a) * conj(cis(b)) of its two tiles' phasors;
// - the beam term B1 F B2^H is hoisted out of the baseline loop: with a single
//   unique tile (which includes "no beam") it is the same B F B^H for every
//   baseline, and otherwise F = L L^H (Cholesky) so B1 F B2^H = (B1 L)(B2 L)^H
//   needs one Jones product per baseline from per-tile half products. A flux
//   density that isn't positive semi-definite (e.g. a negative component) makes
//   the chunk fall back to the full two-product path.
//
// Components are processed in chunks so the device scratch space stays
// bounded. Tile UVWs are not passed in; they're recovered from the baselines to
// tile 0 (U_t - U_0 = -uvw(0, t)), and the common U_0 term cancels in the
// baseline phasor.

const int NUM_THREADS_PER_BLOCK_PREP = 256;
const int NUM_THREADS_PER_BLOCK_PHASORS = 256;

// How much device memory the per-chunk scratch arrays may use.
const size_t MODEL_CHUNK_BYTES = 256 * 1024 * 1024;

// Relative tolerance for the Cholesky factorisation's Hermitian and
// semi-definite checks.
#ifdef SINGLE
const FLOAT CHOL_TOL = 1e-6;
const FLOAT CHOL_TINY = 1e-30;
#else
const FLOAT CHOL_TOL = 1e-12;
const FLOAT CHOL_TINY = 1e-300;
#endif

enum BeamMode {
    // One unique tile: the beam-weighted flux density B F B^H is shared by all
    // baselines.
    BEAM_SHARED = 0,
    // Several unique tiles, all flux densities PSD: per-tile half products.
    BEAM_HALF = 1,
    // Several unique tiles, a flux density isn't PSD: B1 F B2^H per baseline.
    BEAM_FULL = 2,
};

/**
 * The flux density of component `i_comp` (indexed over power laws, then curved
 * power laws, then lists) at frequency index `i_freq`.
 */
template <typename T>
inline __device__ JONES comp_fd(const T &comps, const int i_comp, const int i_freq, const FLOAT freq) {
    if (i_comp < comps.num_power_laws) {
        return extrap_power_law_fd(freq, comps.power_law_fds[i_comp], comps.power_law_sis[i_comp]);
    }
    int i = i_comp - comps.num_power_laws;
    if (i < comps.num_curved_power_laws) {
        return extrap_curved_power_law_fd(freq, comps.curved_power_law_fds[i], comps.curved_power_law_sis[i],
                                          comps.curved_power_law_qs[i]);
    }
    i -= comps.num_curved_power_laws;
    return comps.list_fds[i_freq * comps.num_lists + i];
}

template <typename T> inline __device__ LmnRime comp_lmn(const T &comps, const int i_comp) {
    if (i_comp < comps.num_power_laws)
        return comps.power_law_lmns[i_comp];
    int i = i_comp - comps.num_power_laws;
    if (i < comps.num_curved_power_laws)
        return comps.curved_power_law_lmns[i];
    return comps.list_lmns[i - comps.num_curved_power_laws];
}

template <typename T> inline __device__ GaussianParams comp_gp(const T &comps, const int i_comp) {
    if (i_comp < comps.num_power_laws)
        return comps.power_law_gps[i_comp];
    int i = i_comp - comps.num_power_laws;
    if (i < comps.num_curved_power_laws)
        return comps.curved_power_law_gps[i];
    return comps.list_gps[i - comps.num_curved_power_laws];
}

template <typename T> inline __host__ __device__ int num_comps(const T &comps) {
    return comps.num_power_laws + comps.num_curved_power_laws + comps.num_lists;
}

/**
 * Per-component envelopes, applied to a baseline's phasor. Points have none.
 */
struct EnvelopeArgs {
    // The baseline's UVW in wavelengths.
    UVW uvw;
    FLOAT one_on_lambda;
    int i_bl;
    // Shapelets only: each component's offset into its type's coefficient
    // array, and the basis-function values.
    const int *coeff_offsets;
    const FLOAT *shapelet_basis_values;
};

inline __device__ COMPLEX envelope(const Points &, const int, const EnvelopeArgs &, const COMPLEX phasor) {
    return phasor;
}

inline __device__ COMPLEX envelope(const Gaussians &comps, const int i_comp, const EnvelopeArgs &e,
                                   const COMPLEX phasor) {
    return phasor * get_gaussian_envelope(comp_gp(comps, i_comp), e.uvw);
}

inline __device__ COMPLEX envelope(const Shapelets &comps, const int i_comp, const EnvelopeArgs &e,
                                   const COMPLEX phasor) {
    const ShapeletUV *uvs;
    const ShapeletCoeff *coeffs;
    const int *num_coeffs;
    int i = i_comp, n;
    if (i < comps.num_power_laws) {
        uvs = comps.power_law_shapelet_uvs;
        coeffs = comps.power_law_shapelet_coeffs;
        num_coeffs = comps.power_law_num_shapelet_coeffs;
        n = comps.num_power_laws;
    } else if ((i -= comps.num_power_laws) < comps.num_curved_power_laws) {
        uvs = comps.curved_power_law_shapelet_uvs;
        coeffs = comps.curved_power_law_shapelet_coeffs;
        num_coeffs = comps.curved_power_law_num_shapelet_coeffs;
        n = comps.num_curved_power_laws;
    } else {
        i -= comps.num_curved_power_laws;
        uvs = comps.list_shapelet_uvs;
        coeffs = comps.list_shapelet_coeffs;
        num_coeffs = comps.list_num_shapelet_coeffs;
        n = comps.num_lists;
    }
    const ShapeletUV s_uv = uvs[e.i_bl * n + i] * e.one_on_lambda;
    const COMPLEX env = get_shapelet_envelope(comp_gp(comps, i_comp), s_uv, num_coeffs[i],
                                              coeffs + e.coeff_offsets[i_comp], e.shapelet_basis_values);
    return phasor * env;
}

/**
 * a . b
 */
inline __device__ JONES jones_mul(const JONES &a, const JONES &b) {
    const JONES_C *ac = (const JONES_C *)&a;
    const JONES_C *bc = (const JONES_C *)&b;
    JONES out;
    JONES_C *oc = (JONES_C *)&out;
    oc->j00 = ac->j00 * bc->j00 + ac->j01 * bc->j10;
    oc->j01 = ac->j00 * bc->j01 + ac->j01 * bc->j11;
    oc->j10 = ac->j10 * bc->j00 + ac->j11 * bc->j10;
    oc->j11 = ac->j10 * bc->j01 + ac->j11 * bc->j11;
    return out;
}

/**
 * a . b^H
 */
inline __device__ JONES jones_mul_h(const JONES &a, const JONES &b) {
    const JONES_C *ac = (const JONES_C *)&a;
    const JONES_C *bc = (const JONES_C *)&b;
    JONES out;
    JONES_C *oc = (JONES_C *)&out;
    oc->j00 = ac->j00 * CUCONJ(bc->j00) + ac->j01 * CUCONJ(bc->j01);
    oc->j01 = ac->j00 * CUCONJ(bc->j10) + ac->j01 * CUCONJ(bc->j11);
    oc->j10 = ac->j10 * CUCONJ(bc->j00) + ac->j11 * CUCONJ(bc->j01);
    oc->j11 = ac->j10 * CUCONJ(bc->j10) + ac->j11 * CUCONJ(bc->j11);
    return out;
}

/**
 * The lower-triangular Cholesky factor L of a Hermitian positive semi-definite
 * Jones matrix F (F = L L^H). Returns false if F isn't one. A flux-density
 * Jones matrix [[I+Q, U+iV], [U-iV, I-Q]] is one whenever
 * I >= sqrt(Q^2 + U^2 + V^2), i.e. for every physical source. This mirrors
 * `jones_cholesky` in the CPU modeller.
 */
inline __device__ bool jones_cholesky(const JONES f, JONES *l) {
    const FLOAT tol = CHOL_TOL * fmax(hypot(f.j00_re, f.j00_im) + hypot(f.j11_re, f.j11_im), CHOL_TINY);
    if (fabs(f.j00_im) > tol || fabs(f.j11_im) > tol || hypot(f.j01_re - f.j10_re, f.j01_im + f.j10_im) > tol)
        return false;
    const FLOAT f00 = f.j00_re;
    const FLOAT f11 = f.j11_re;
    if (f00 < -tol || f11 < -tol)
        return false;
    FLOAT a, b_re, b_im;
    if (f00 <= tol) {
        if (hypot(f.j10_re, f.j10_im) > tol)
            return false;
        a = 0.0;
        b_re = 0.0;
        b_im = 0.0;
    } else {
        a = sqrt(f00);
        b_re = f.j10_re / a;
        b_im = f.j10_im / a;
    }
    const FLOAT c2 = f11 - (b_re * b_re + b_im * b_im);
    if (c2 < -tol)
        return false;
    *l = JONES{
        .j00_re = a,
        .j00_im = 0.0,
        .j01_re = 0.0,
        .j01_im = 0.0,
        .j10_re = b_re,
        .j10_im = b_im,
        .j11_re = sqrt(fmax(c2, (FLOAT)0.0)),
        .j11_im = 0.0,
    };
    return true;
}

/**
 * One thread per (channel, chunk component): the flux density, and the beam
 * weighting that doesn't depend on the baseline.
 *
 * - BEAM_SHARED: `fds` gets B F B^H of the single unique tile.
 * - otherwise `fds` gets F, and, if every F in the chunk is PSD, `half` gets
 *   B_u L per unique tile u (layout [unique tile][channel][component]). A
 *   non-PSD F sets `*not_psd`.
 */
template <typename T>
__global__ void model_prep_kernel(const T comps, const int comp_start, const int chunk_comps, const int num_freqs,
                                  const FLOAT *__restrict__ freqs, const JONES *__restrict__ beam_jones,
                                  const int *__restrict__ freq_map, const int num_fee_freqs,
                                  const int num_unique_tiles, JONES *__restrict__ fds, JONES *__restrict__ half,
                                  int *not_psd) {
    const int num_directions = num_comps(comps);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_freqs * chunk_comps; i += gridDim.x * blockDim.x) {
        const int i_c = i % chunk_comps;
        const int i_freq = i / chunk_comps;
        const int i_comp = comp_start + i_c;
        const int i_col = freq_map[i_freq];

        JONES fd = comp_fd(comps, i_comp, i_freq, freqs[i_freq]);
        if (num_unique_tiles == 1) {
            const JONES *b = beam_jones + num_directions * i_col + i_comp;
            apply_beam(b, &fd, b);
            fds[i] = fd;
            continue;
        }

        fds[i] = fd;
        JONES l;
        if (!jones_cholesky(fd, &l)) {
            *not_psd = 1;
            continue;
        }
        for (int i_unique = 0; i_unique < num_unique_tiles; i_unique++) {
            const JONES b = beam_jones[num_directions * (num_fee_freqs * i_unique + i_col) + i_comp];
            half[(i_unique * num_freqs + i_freq) * chunk_comps + i_c] = jones_mul(b, l);
        }
    }
}

/**
 * One thread per (unflagged tile, channel, chunk component): the tile phasor
 * cis(u_t l + v_t m + w_t n), laid out as [channel][component][tile] so that
 * a warp of adjacent baselines reads adjacent tiles.
 */
template <typename T>
__global__ void model_phasors_kernel(const T comps, const int comp_start, const int chunk_comps, const int num_freqs,
                                     const int num_tiles, const FLOAT *__restrict__ freqs,
                                     const UVW *__restrict__ uvws, COMPLEX *__restrict__ phasors) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_tiles * num_freqs * chunk_comps;
         i += gridDim.x * blockDim.x) {
        const int i_tile = i % num_tiles;
        const int i_fc = i / num_tiles;
        const int i_c = i_fc % chunk_comps;
        const int i_freq = i_fc / chunk_comps;

        // U_t - U_0 = -uvw(0, t); baseline (0, t) has index t - 1.
        UVW tile_uvw = UVW{.u = 0.0, .v = 0.0, .w = 0.0};
        if (i_tile > 0) {
            const UVW bl = uvws[i_tile - 1];
            tile_uvw = UVW{.u = -bl.u, .v = -bl.v, .w = -bl.w};
        }
        // The 2 PI is already multiplied on the LMN terms (as well as a -1 on
        // the n).
        const LmnRime lmn = comp_lmn(comps, comp_start + i_c);
        COMPLEX p;
#ifdef SINGLE
        // A tile's phase is relative to tile 0, so it can be much bigger than
        // a (short) baseline's, and a float argument would lose that
        // baseline's precision. This kernel only runs per tile, so form the
        // argument in double and reduce it to [-pi, pi] before the float
        // sincos.
        const double one_on_lambda = (double)freqs[i_freq] / (double)VEL_C;
        double phase = one_on_lambda * ((double)tile_uvw.u * (double)lmn.l + (double)tile_uvw.v * (double)lmn.m +
                                        (double)tile_uvw.w * (double)lmn.n);
        phase -= 6.283185307179586 * rint(phase / 6.283185307179586);
        SINCOS((float)phase, &p.y, &p.x);
#else
        const UVW uvw = tile_uvw * freqs[i_freq] / VEL_C;
        SINCOS(uvw.u * lmn.l + uvw.v * lmn.m + uvw.w * lmn.n, &p.y, &p.x);
#endif
        phasors[i] = p;
    }
}

/**
 * One thread per (cross-correlation baseline, channel), accumulating the
 * chunk's components.
 */
template <typename T>
__global__ void model_baselines_kernel(const T comps, const int comp_start, const int chunk_comps,
                                       const int num_freqs, const int num_baselines, const int num_tiles,
                                       const FLOAT *__restrict__ freqs, const UVW *__restrict__ uvws,
                                       const BeamMode mode, const JONES *__restrict__ fds,
                                       const JONES *__restrict__ half, const COMPLEX *__restrict__ phasors,
                                       const JONES *__restrict__ beam_jones, const int *__restrict__ tile_map,
                                       const int *__restrict__ freq_map, const int num_fee_freqs,
                                       const int *__restrict__ tile_index_to_unflagged_tile_index_map,
                                       const int *__restrict__ coeff_offsets,
                                       const FLOAT *__restrict__ shapelet_basis_values,
                                       JonesF32 *__restrict__ vis_fb) {
    // The 0-indexed number of tiles as a float (n-1).
    const float ntiles_sub1f = NTILES_SUB1F_FROM_BASELINES(num_baselines);
    const int num_directions = num_comps(comps);

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_baselines * num_freqs; i += gridDim.x * blockDim.x) {
        const int i_bl = i % num_baselines;
        const int i_freq = i / num_baselines;

        const FLOAT freq = freqs[i_freq];
        const FLOAT one_on_lambda = freq / VEL_C;
        const EnvelopeArgs e{
            .uvw = uvws[i_bl] * freq / VEL_C,
            .one_on_lambda = one_on_lambda,
            .i_bl = i_bl,
            .coeff_offsets = coeff_offsets,
            .shapelet_basis_values = shapelet_basis_values,
        };

        BASELINE_TO_TILES(i_bl, ntiles_sub1f, i_tile1, i_tile2);

        // `i_j1_row` and `i_j2_row` are indices into beam responses (unique
        // tiles).
        const int i_j1_row = tile_map[tile_index_to_unflagged_tile_index_map[i_tile1]];
        const int i_j2_row = tile_map[tile_index_to_unflagged_tile_index_map[i_tile2]];
        const int i_col = freq_map[i_freq];

        const COMPLEX *ph = phasors + (size_t)i_freq * chunk_comps * num_tiles;
        const JONES *f = fds + i_freq * chunk_comps;
        const JONES *h1 = half + (i_j1_row * num_freqs + i_freq) * chunk_comps;
        const JONES *h2 = half + (i_j2_row * num_freqs + i_freq) * chunk_comps;
        const JONES *b1 = beam_jones + num_directions * (num_fee_freqs * i_j1_row + i_col) + comp_start;
        const JONES *b2 = beam_jones + num_directions * (num_fee_freqs * i_j2_row + i_col) + comp_start;

#ifdef SINGLE
        // The tile phasors use tile UVWs recovered from the (float) baselines
        // to tile 0, which carry rounding errors proportional to the tiles'
        // distances from tile 0 rather than this baseline's length. Correct
        // the phasor by the (tiny) difference between this baseline's own UVW
        // and the tile UVWs' difference; it's exact in double.
        UVW d;
        {
            const UVW bl = uvws[i_bl];
            const UVW u0j = uvws[i_tile2 - 1];
            const UVW u0i = i_tile1 > 0 ? uvws[i_tile1 - 1] : UVW{.u = 0.0, .v = 0.0, .w = 0.0};
            d = UVW{
                .u = (float)((double)bl.u - ((double)u0j.u - (double)u0i.u)),
                .v = (float)((double)bl.v - ((double)u0j.v - (double)u0i.v)),
                .w = (float)((double)bl.w - ((double)u0j.w - (double)u0i.w)),
            };
            d = d * one_on_lambda;
        }
#endif

        JONES delta_vis = JONES{
            .j00_re = 0.0,
            .j00_im = 0.0,
            .j01_re = 0.0,
            .j01_im = 0.0,
            .j10_re = 0.0,
            .j10_im = 0.0,
            .j11_re = 0.0,
            .j11_im = 0.0,
        };

        for (int i_c = 0; i_c < chunk_comps; i_c++) {
            const COMPLEX p1 = ph[i_c * num_tiles + i_tile1];
            const COMPLEX p2 = ph[i_c * num_tiles + i_tile2];
#ifdef SINGLE
            // d is ~1e-4 wavelengths, so second order is plenty.
            const LmnRime lmn = comp_lmn(comps, comp_start + i_c);
            const float delta = d.u * lmn.l + d.v * lmn.m + d.w * lmn.n;
            const COMPLEX correction = MAKE_COMPLEX(1.0f - 0.5f * delta * delta, delta);
            const COMPLEX phasor = envelope(comps, comp_start + i_c, e, p1 * CUCONJ(p2) * correction);
#else
            const COMPLEX phasor = envelope(comps, comp_start + i_c, e, p1 * CUCONJ(p2));
#endif

            JONES v;
            if (mode == BEAM_SHARED) {
                v = f[i_c];
            } else if (mode == BEAM_HALF) {
                v = jones_mul_h(h1[i_c], h2[i_c]);
            } else {
                v = f[i_c];
                apply_beam(b1 + i_c, &v, b2 + i_c);
            }
            delta_vis += v * phasor;
        }

        // Visibilities are ordered over baselines and frequencies, with
        // baselines moving faster than frequencies.
        vis_fb[i_freq * num_baselines + i_bl] += delta_vis;
    }
}

__global__ void model_autos_kernel(
    int num_tiles, int num_freqs, const FLOAT *__restrict__ freqs,
    const Points points, const Gaussians gaussians, const Shapelets shapelets,
    const JONES *__restrict__ beam_jones, const int *__restrict__ tile_map,
    const int *__restrict__ freq_map, const int num_fee_freqs,
    const int *__restrict__ tile_index_to_unflagged_tile_index_map,
    JonesF32 *__restrict__ vis_fb
) {
    const int num_directions = points.num_power_laws + points.num_curved_power_laws + points.num_lists +
                              gaussians.num_power_laws + gaussians.num_curved_power_laws + gaussians.num_lists +
                              shapelets.num_power_laws + shapelets.num_curved_power_laws + shapelets.num_lists;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_tiles * num_freqs; i += gridDim.x * blockDim.x) {
        const int i_freq = i % num_freqs;
        const int i_tile = i / num_freqs;

        const FLOAT freq = freqs[i_freq];

        // Get beam response indices for this tile and frequency
        const int i_j_row = tile_map[tile_index_to_unflagged_tile_index_map[i_tile]];
        const int i_col = freq_map[i_freq];
        const JONES *beam_ptr = beam_jones + num_directions * (num_fee_freqs * i_j_row + i_col);

        JONES delta_vis = JONES{
            .j00_re = 0.0,
            .j00_im = 0.0,
            .j01_re = 0.0,
            .j01_im = 0.0,
            .j10_re = 0.0,
            .j10_im = 0.0,
            .j11_re = 0.0,
            .j11_im = 0.0,
        };

        // Process point sources - power law
        for (int i_comp = 0; i_comp < points.num_power_laws; i_comp++) {
            JONES fd = extrap_power_law_fd(freq, points.power_law_fds[i_comp], points.power_law_sis[i_comp]);
            JONES beam_response = beam_ptr[i_comp];
            delta_vis += apply_beam_auto(beam_response, fd);
        }
        beam_ptr += points.num_power_laws;

        // Process point sources - curved power law
        for (int i_comp = 0; i_comp < points.num_curved_power_laws; i_comp++) {
            JONES fd = extrap_curved_power_law_fd(freq, points.curved_power_law_fds[i_comp],
                                                  points.curved_power_law_sis[i_comp], points.curved_power_law_qs[i_comp]);
            JONES beam_response = beam_ptr[i_comp];
            delta_vis += apply_beam_auto(beam_response, fd);
        }
        beam_ptr += points.num_curved_power_laws;

        // Process point sources - list
        for (int i_comp = 0; i_comp < points.num_lists; i_comp++) {
            JONES fd = points.list_fds[i_freq * points.num_lists + i_comp];
            JONES beam_response = beam_ptr[i_comp];
            delta_vis += apply_beam_auto(beam_response, fd);
        }
        beam_ptr += points.num_lists;

        // For gaussians and shapelets, we just need the same computation as points
        // since the envelope is 1.0 for auto-correlations

        // Process gaussian sources - power law
        for (int i_comp = 0; i_comp < gaussians.num_power_laws; i_comp++) {
            JONES fd = extrap_power_law_fd(freq, gaussians.power_law_fds[i_comp], gaussians.power_law_sis[i_comp]);
            JONES beam_response = beam_ptr[i_comp];
            delta_vis += apply_beam_auto(beam_response, fd);
        }
        beam_ptr += gaussians.num_power_laws;

        // Process gaussian sources - curved power law
        for (int i_comp = 0; i_comp < gaussians.num_curved_power_laws; i_comp++) {
            JONES fd = extrap_curved_power_law_fd(freq, gaussians.curved_power_law_fds[i_comp],
                                                  gaussians.curved_power_law_sis[i_comp],
                                                  gaussians.curved_power_law_qs[i_comp]);
            JONES beam_response = beam_ptr[i_comp];
            delta_vis += apply_beam_auto(beam_response, fd);
        }
        beam_ptr += gaussians.num_curved_power_laws;

        // Process gaussian sources - list
        for (int i_comp = 0; i_comp < gaussians.num_lists; i_comp++) {
            JONES fd = gaussians.list_fds[i_freq * gaussians.num_lists + i_comp];
            JONES beam_response = beam_ptr[i_comp];
            delta_vis += apply_beam_auto(beam_response, fd);
        }
        beam_ptr += gaussians.num_lists;

        // Process shapelet sources - power law
        for (int i_comp = 0; i_comp < shapelets.num_power_laws; i_comp++) {
            JONES fd = extrap_power_law_fd(freq, shapelets.power_law_fds[i_comp], shapelets.power_law_sis[i_comp]);
            JONES beam_response = beam_ptr[i_comp];
            delta_vis += apply_beam_auto(beam_response, fd);
        }
        beam_ptr += shapelets.num_power_laws;

        // Process shapelet sources - curved power law
        for (int i_comp = 0; i_comp < shapelets.num_curved_power_laws; i_comp++) {
            JONES fd = extrap_curved_power_law_fd(freq, shapelets.curved_power_law_fds[i_comp],
                                                  shapelets.curved_power_law_sis[i_comp],
                                                  shapelets.curved_power_law_qs[i_comp]);
            JONES beam_response = beam_ptr[i_comp];
            delta_vis += apply_beam_auto(beam_response, fd);
        }
        beam_ptr += shapelets.num_curved_power_laws;

        // Process shapelet sources - list
        for (int i_comp = 0; i_comp < shapelets.num_lists; i_comp++) {
            JONES fd = shapelets.list_fds[i_freq * shapelets.num_lists + i_comp];
            JONES beam_response = beam_ptr[i_comp];
            delta_vis += apply_beam_auto(beam_response, fd);
        }
        beam_ptr += shapelets.num_lists;

        // Store result - auto-correlations are indexed as [freq, tile]
        vis_fb[i_freq * num_tiles + i_tile] += delta_vis;
    }
}

/**
 * Model all of `comps`' components, chunked, with the factorised kernels.
 * `coeff_offsets` is only used by shapelets.
 */
template <typename T>
static const char *model_factorised(const T &comps, const Addresses *a, const UVW *d_uvws, const JONES *d_beam_jones,
                                    JonesF32 *d_vis_fb, const int *d_coeff_offsets, const int threads_per_block) {
    const int total_comps = num_comps(comps);
    if (total_comps == 0 || a->num_baselines == 0 || a->num_freqs == 0)
        return NULL;
    const int num_freqs = a->num_freqs;
    const int num_tiles = (int)lround((1.0 + sqrt(1.0 + 8.0 * (double)a->num_baselines)) / 2.0);
    const int num_unique_tiles = a->num_unique_tiles;

    // Scratch space per component: flux densities, tile phasors and (with
    // more than one unique tile) half products.
    const size_t bytes_per_comp = (size_t)num_freqs * sizeof(JONES) + (size_t)num_tiles * num_freqs * sizeof(COMPLEX) +
                                  (num_unique_tiles > 1 ? (size_t)num_unique_tiles * num_freqs * sizeof(JONES) : 0);
    int comps_per_chunk = (int)(MODEL_CHUNK_BYTES / bytes_per_comp);
    if (comps_per_chunk < 1)
        comps_per_chunk = 1;
    if (comps_per_chunk > total_comps)
        comps_per_chunk = total_comps;

    JONES *d_fds = NULL, *d_half = NULL;
    COMPLEX *d_phasors = NULL;
    int *d_not_psd = NULL;
    CHECK_GPU_ERROR(gpuMalloc((void **)&d_fds, (size_t)comps_per_chunk * num_freqs * sizeof(JONES)));
    CHECK_GPU_ERROR(gpuMalloc((void **)&d_phasors, (size_t)comps_per_chunk * num_freqs * num_tiles * sizeof(COMPLEX)));
    if (num_unique_tiles > 1) {
        CHECK_GPU_ERROR(gpuMalloc((void **)&d_half, (size_t)comps_per_chunk * num_freqs * num_unique_tiles * sizeof(JONES)));
        CHECK_GPU_ERROR(gpuMalloc((void **)&d_not_psd, sizeof(int)));
    }

    for (int comp_start = 0; comp_start < total_comps; comp_start += comps_per_chunk) {
        const int chunk_comps = comp_start + comps_per_chunk > total_comps ? total_comps - comp_start : comps_per_chunk;

        BeamMode mode = BEAM_SHARED;
        if (num_unique_tiles > 1) {
            const int zero = 0;
            CHECK_GPU_ERROR(gpuMemcpy(d_not_psd, &zero, sizeof(int), gpuMemcpyHostToDevice));
        }
        {
            const int n = num_freqs * chunk_comps;
            model_prep_kernel<<<(n + NUM_THREADS_PER_BLOCK_PREP - 1) / NUM_THREADS_PER_BLOCK_PREP,
                                NUM_THREADS_PER_BLOCK_PREP>>>(comps, comp_start, chunk_comps, num_freqs, a->d_freqs,
                                                              d_beam_jones, a->d_freq_map, a->num_unique_beam_freqs,
                                                              num_unique_tiles, d_fds, d_half, d_not_psd);
            CHECK_GPU_ERROR(gpuGetLastError());
        }
        if (num_unique_tiles > 1) {
            int not_psd;
            CHECK_GPU_ERROR(gpuMemcpy(&not_psd, d_not_psd, sizeof(int), gpuMemcpyDeviceToHost));
            mode = not_psd ? BEAM_FULL : BEAM_HALF;
        }
        {
            const int n = num_tiles * num_freqs * chunk_comps;
            model_phasors_kernel<<<(n + NUM_THREADS_PER_BLOCK_PHASORS - 1) / NUM_THREADS_PER_BLOCK_PHASORS,
                                   NUM_THREADS_PER_BLOCK_PHASORS>>>(comps, comp_start, chunk_comps, num_freqs,
                                                                    num_tiles, a->d_freqs, d_uvws, d_phasors);
            CHECK_GPU_ERROR(gpuGetLastError());
        }
        {
            const int n = a->num_baselines * num_freqs;
            model_baselines_kernel<<<(n + threads_per_block - 1) / threads_per_block, threads_per_block>>>(
                comps, comp_start, chunk_comps, num_freqs, a->num_baselines, num_tiles, a->d_freqs, d_uvws, mode,
                d_fds, d_half, d_phasors, d_beam_jones, a->d_tile_map, a->d_freq_map, a->num_unique_beam_freqs,
                a->d_tile_index_to_unflagged_tile_index_map, d_coeff_offsets, a->d_shapelet_basis_values, d_vis_fb);
            CHECK_GPU_ERROR(gpuGetLastError());
        }
#ifdef DEBUG
        CHECK_GPU_ERROR(gpuDeviceSynchronize());
#endif
    }

    CHECK_GPU_ERROR(gpuFree(d_fds));
    CHECK_GPU_ERROR(gpuFree(d_phasors));
    if (d_half != NULL)
        CHECK_GPU_ERROR(gpuFree(d_half));
    if (d_not_psd != NULL)
        CHECK_GPU_ERROR(gpuFree(d_not_psd));

    return NULL;
}

extern "C" const char *model_points(const Points *comps, const Addresses *a, const UVW *d_uvws,
                                    const JONES *d_beam_jones, JonesF32 *d_vis_fb) {
    return model_factorised(*comps, a, d_uvws, d_beam_jones, d_vis_fb, NULL, NUM_THREADS_PER_BLOCK_POINTS);
}

extern "C" const char *model_gaussians(const Gaussians *comps, const Addresses *a, const UVW *d_uvws,
                                       const JONES *d_beam_jones, JonesF32 *d_vis_fb) {
    return model_factorised(*comps, a, d_uvws, d_beam_jones, d_vis_fb, NULL, NUM_THREADS_PER_BLOCK_GAUSSIANS);
}

extern "C" const char *model_shapelets(const Shapelets *comps, const Addresses *a, const UVW *d_uvws,
                                       const JONES *d_beam_jones, JonesF32 *d_vis_fb) {
    // Each component's offset into its type's coefficient array (the kernels
    // index components out of order, so a running pointer doesn't work).
    const int counts[3] = {comps->num_power_laws, comps->num_curved_power_laws, comps->num_lists};
    const int *d_lens[3] = {comps->power_law_num_shapelet_coeffs, comps->curved_power_law_num_shapelet_coeffs,
                            comps->list_num_shapelet_coeffs};
    const int total = counts[0] + counts[1] + counts[2];
    if (total == 0)
        return NULL;
    int *offsets = (int *)malloc(total * sizeof(int));
    int *lens = (int *)malloc(total * sizeof(int));
    int i_comp = 0;
    for (int t = 0; t < 3; t++) {
        if (counts[t] == 0)
            continue;
        CHECK_GPU_ERROR(gpuMemcpy(lens, d_lens[t], counts[t] * sizeof(int), gpuMemcpyDeviceToHost));
        int offset = 0;
        for (int i = 0; i < counts[t]; i++) {
            offsets[i_comp++] = offset;
            offset += lens[i];
        }
    }
    int *d_offsets = NULL;
    CHECK_GPU_ERROR(gpuMalloc((void **)&d_offsets, total * sizeof(int)));
    CHECK_GPU_ERROR(gpuMemcpy(d_offsets, offsets, total * sizeof(int), gpuMemcpyHostToDevice));
    free(offsets);
    free(lens);

    const char *error =
        model_factorised(*comps, a, d_uvws, d_beam_jones, d_vis_fb, d_offsets, NUM_THREADS_PER_BLOCK_SHAPELETS);
    CHECK_GPU_ERROR(gpuFree(d_offsets));
    return error;
}

extern "C" const char *model_autos(int num_tiles, int num_freqs, const FLOAT *freqs, const Points *points, const Gaussians *gaussians, const Shapelets *shapelets, const JONES *beam_jones, const int *tile_map, const int *freq_map, int num_fee_freqs, const int *tile_index_to_unflagged_tile_index_map, JonesF32 *vis_fb) {
    dim3 blockDim(NUM_THREADS_PER_BLOCK_AUTOS);
    dim3 gridDim((num_tiles * num_freqs + blockDim.x - 1) / blockDim.x);

    model_autos_kernel<<<gridDim, blockDim>>>(num_tiles, num_freqs, freqs, *points, *gaussians, *shapelets, beam_jones,
                                               tile_map, freq_map, num_fee_freqs, tile_index_to_unflagged_tile_index_map, vis_fb);

#ifdef DEBUG
    CHECK_GPU_ERROR(gpuDeviceSynchronize());
#endif
    CHECK_GPU_ERROR(gpuGetLastError());

    return NULL;
}
