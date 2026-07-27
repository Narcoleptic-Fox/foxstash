//! `GpuFlatIndex` — an optional, portable GPU brute-force index for foxstash.
//!
//! The graph stays on the CPU (`HNSWIndex`); this is the *flat* companion for the regime where a GPU
//! flat scan beats the CPU graph — small-to-medium corpora (Phase 0: ~30k f32 / ~60–70k SQ8 on an
//! RX 6600; higher on bigger GPUs). Built on [CubeCL], so it runs on CUDA, ROCm, Vulkan, Metal and
//! **WebGPU (browser)** from one codebase. Nothing here touches `foxstash-core`'s wasm/embeddable core.
//!
//! Phase 1a: SQ8 codes, scanned coalesced on the GPU (per-query bandwidth-bound), exact-ish ranking
//! (8-bit ⇒ ~0.97 recall with no rerank — a flat scan has no navigation loss, only quant error).
//! Codes are packed 4-per-`u32` because WGSL/WebGPU has no `u8` element type.
//!
//! [CubeCL]: https://github.com/tracel-ai/cubecl

use cubecl::prelude::*;
use foxstash_core::vector::rabitq::RaBitQuantizer;
use foxstash_core::vector::turborabit::{TurboRabitCode, TurboRabitQuantizer};
pub use foxstash_core::SearchResult;

// Backend: native ROCm/HIP when built with `--features hip`, else wgpu→WGSL→Vulkan (the portable default).
#[cfg(feature = "hip")]
type Rt = cubecl::hip::HipRuntime;
#[cfg(not(feature = "hip"))]
type Rt = cubecl::wgpu::WgpuRuntime;
type Client = ComputeClient<Rt>;

/// GEMM-tiled SQ8 scan: **one thread per candidate `i`** (grid = `n`, not `b·n`), looping over all `b`
/// queries so each corpus code word is read from DRAM **once** and reused across the whole query batch. A
/// naive `b·n`-thread scan re-reads the corpus `b` times — the profiled batch wall (92% of the pipeline, ~158
/// GB/s ≈ the RX 6600's bandwidth ceiling). Loop order `for d { c = read; for q { acc[q] += q·c } }` gives
/// the reuse; `b` is comptime so the `q` loops `#[unroll]` and the `b` accumulators stay in registers.
#[cube(launch_unchecked)]
fn scan_sq8_tiled(
    query_t: &Array<f32>,
    corpus_w: &Array<u32>,
    out: &mut Array<f32>,
    #[comptime] n: u32,
    #[comptime] b: u32,
    #[comptime] dim: u32,
) {
    let i = ABSOLUTE_POS; // one thread per candidate
    if i < n as usize {
        let (nn, bb, dd) = (n as usize, b as usize, dim as usize);
        let nw = nn / 4usize;
        let wi = i / 4usize;
        let sh = ((i % 4usize) as u32) * 8u32;
        let mut acc = Array::<f32>::new(bb);
        #[unroll]
        for q in 0..bb {
            acc[q] = 0.0f32;
        }
        for d in 0..dd {
            let c = ((corpus_w[d * nw + wi] >> sh) & 255u32) as f32;
            #[unroll]
            for q in 0..bb {
                acc[q] += query_t[d * bb + q] * c;
            }
        }
        #[unroll]
        for q in 0..bb {
            out[q * nn + i] = acc[q];
        }
    }
}

/// Tiled Sign1 scan — [`scan_sq8_tiled`]'s one-thread-per-candidate corpus reuse, applied to the sign-dot.
#[cube(launch_unchecked)]
fn scan_sign1_tiled(
    query_t: &Array<f32>,
    signs_w: &Array<u32>,
    out: &mut Array<f32>,
    #[comptime] n: u32,
    #[comptime] b: u32,
    #[comptime] dim: u32,
) {
    let i = ABSOLUTE_POS;
    if i < n as usize {
        let (nn, bb, dd) = (n as usize, b as usize, dim as usize);
        let nw = nn / 32usize;
        let wi = i / 32usize;
        let sh = (i % 32usize) as u32;
        let mut acc = Array::<f32>::new(bb);
        #[unroll]
        for q in 0..bb {
            acc[q] = 0.0f32;
        }
        for d in 0..dd {
            let sign = 2.0f32 * (((signs_w[d * nw + wi] >> sh) & 1u32) as f32) - 1.0f32;
            #[unroll]
            for q in 0..bb {
                acc[q] += query_t[d * bb + q] * sign;
            }
        }
        #[unroll]
        for q in 0..bb {
            out[q * nn + i] = acc[q];
        }
    }
}

/// Tiled RaBitQ scan — corpus reuse across the batch, then the per-candidate affine combine (scalars read
/// once): `score = 2·est_factor·s − dtc_sq`.
#[cube(launch_unchecked)]
fn scan_rabitq_tiled(
    rq_t: &Array<f32>,
    signs_w: &Array<u32>,
    dtc_sq: &Array<f32>,
    est_factor: &Array<f32>,
    out: &mut Array<f32>,
    #[comptime] n: u32,
    #[comptime] b: u32,
    #[comptime] dim: u32,
) {
    let i = ABSOLUTE_POS;
    if i < n as usize {
        let (nn, bb, dd) = (n as usize, b as usize, dim as usize);
        let nw = nn / 32usize;
        let wi = i / 32usize;
        let sh = (i % 32usize) as u32;
        let mut acc = Array::<f32>::new(bb);
        #[unroll]
        for q in 0..bb {
            acc[q] = 0.0f32;
        }
        for d in 0..dd {
            let sign = 2.0f32 * (((signs_w[d * nw + wi] >> sh) & 1u32) as f32) - 1.0f32;
            #[unroll]
            for q in 0..bb {
                acc[q] += rq_t[d * bb + q] * sign;
            }
        }
        let ef = est_factor[i];
        let dt = dtc_sq[i];
        #[unroll]
        for q in 0..bb {
            out[q * nn + i] = 2.0f32 * ef * acc[q] - dt;
        }
    }
}

/// Tiled TurboRabit scan — corpus reuse across the batch, then the per-candidate affine combine:
/// `score = −(dtc_sq + f_rescale·(ip + cb_sum))`.
#[cube(launch_unchecked)]
fn scan_turborabit_tiled(
    rq_t: &Array<f32>,
    codes_w: &Array<u32>,
    dtc_sq: &Array<f32>,
    f_rescale: &Array<f32>,
    cb_sum: &Array<f32>,
    out: &mut Array<f32>,
    #[comptime] n: u32,
    #[comptime] b: u32,
    #[comptime] dim: u32,
    #[comptime] cpw: u32,
    #[comptime] bpc: u32,
    #[comptime] mask: u32,
) {
    let i = ABSOLUTE_POS;
    if i < n as usize {
        let (nn, bb, dd, cw) = (n as usize, b as usize, dim as usize, cpw as usize);
        let nw = nn / cw;
        let wi = i / cw;
        let sh = ((i % cw) as u32) * bpc;
        let mut acc = Array::<f32>::new(bb);
        #[unroll]
        for q in 0..bb {
            acc[q] = 0.0f32;
        }
        for d in 0..dd {
            let c = ((codes_w[d * nw + wi] >> sh) & mask) as f32;
            #[unroll]
            for q in 0..bb {
                acc[q] += rq_t[d * bb + q] * c;
            }
        }
        let fr = f_rescale[i];
        let dt = dtc_sq[i];
        #[unroll]
        for q in 0..bb {
            out[q * nn + i] = -(dt + fr * (acc[q] + cb_sum[q]));
        }
    }
}

/// GPU query rotation: `rq_t[j][q] = Σ_d R[j][d]·centered_t[d][q]` — the `R·(q−c)` that TR4/Warren need for
/// rotated space, done on device instead of a per-query CPU `O(dim²)` matvec. One thread per `(j, q)` output
/// element; `centered` is **transposed `[dim][b]`** so at each `d` adjacent threads (adjacent `q`) read
/// adjacent values — coalesced. (Reading it `[b][dim]` strided ran this at ~10 GFLOP/s — the whole TR4/Warren
/// scan penalty.) `R`'s row `j` is broadcast across the warp (same `j`). Output is `[dim][b]` for the scan.
#[cube(launch_unchecked)]
fn rotate_queries(
    centered: &Array<f32>,
    r_mat: &Array<f32>,
    out: &mut Array<f32>,
    #[comptime] b: u32,
    #[comptime] dim: u32,
) {
    let pos = ABSOLUTE_POS; // 0..dim*b
    if pos < (dim * b) as usize {
        let (bb, dd) = (b as usize, dim as usize);
        let j = pos / bb;
        let q = pos % bb;
        let rrow = j * dd;
        let mut acc = 0.0f32;
        for d in 0..dd {
            acc += r_mat[rrow + d] * centered[d * bb + q];
        }
        out[j * bb + q] = acc;
    }
}

/// GPU exact f32 rerank: one thread per (query, candidate) computes the exact dot with the candidate's
/// full-precision vector (row-major `[n][dim]`), moving the `O(C·dim)` rerank off the CPU. Mode-agnostic —
/// the query is used directly, **no rotation** — so it fully uncaps SQ8 (which needs none). OOB/padded
/// candidates score −inf.
#[cube(launch_unchecked)]
fn rerank_f32_dot(
    query: &Array<f32>,
    cand_idx: &Array<u32>,
    corpus: &Array<f32>,
    out: &mut Array<f32>,
    #[comptime] b: u32,
    #[comptime] dim: u32,
    #[comptime] cc: u32,
    #[comptime] n_real: u32,
) {
    let pos = ABSOLUTE_POS;
    if pos < (b * cc) as usize {
        let (dd, ccu) = (dim as usize, cc as usize);
        let q = pos / ccu;
        let cand = cand_idx[pos] as usize;
        if cand < n_real as usize {
            let row = cand * dd;
            let qb = q * dd;
            let mut acc = 0.0f32;
            for d in 0..dd {
                acc += query[qb + d] * corpus[row + d];
            }
            out[pos] = acc;
        } else {
            out[pos] = -3.0e38f32;
        }
    }
}

/// IVF cell scan (SQ8, coalesced). Thread per (query, candidate-slot); `cand_idx[q][slot]` is a **reordered**
/// index. Because the corpus is reordered by cell and candidate lists are filled cell-by-cell, a cell's
/// entries are *contiguous* — so adjacent threads read adjacent codes (`codes_w[d·n/4 + cand/4]`), the same
/// coalescing the flat scan gets. SQ8 (4× less data than the f32 gather) + coalesced is what makes IVF pay.
/// Query pre-scaled `[dim][b]` (the `Σq·min` constant drops, as in the flat SQ8 scan). Sentinel ⇒ −inf.
#[cube(launch_unchecked)]
fn ivf_scan_sq8(
    query_t: &Array<f32>,
    codes_w: &Array<u32>,
    cand_idx: &Array<u32>,
    out: &mut Array<f32>,
    #[comptime] n: u32,
    #[comptime] b: u32,
    #[comptime] dim: u32,
    #[comptime] cc: u32,
    #[comptime] n_real: u32,
) {
    let pos = ABSOLUTE_POS;
    if pos < (b * cc) as usize {
        let (nn, bb, dd, ccu) = (n as usize, b as usize, dim as usize, cc as usize);
        let q = pos / ccu;
        let cand = cand_idx[pos] as usize;
        if cand < n_real as usize {
            let nw = nn / 4usize;
            let wi = cand / 4usize;
            let sh = ((cand % 4usize) as u32) * 8u32;
            let mut s = 0.0f32;
            for d in 0..dd {
                let c = (codes_w[d * nw + wi] >> sh) & 255u32;
                s += query_t[d * bb + q] * (c as f32);
            }
            out[pos] = s;
        } else {
            out[pos] = -3.0e38f32;
        }
    }
}

/// GPU Warren residual rerank: **one thread per (query, candidate)** recomputes the residual-refined dot on
/// device, moving the `O(C·dim)` rerank off the CPU (which capped the two-stage at ~1000 QPS). It gathers
/// the candidate's 4-bit scan code (nibble) + 8+8 residual (byte) and evaluates
/// `⟨q,x⟩ = qc + Σ_d rq_full[d]·(scale·u + bias + step1·r1 + step2·r2)`, where `bias = scale·c_B + lo1 + lo2`
/// folds the grid constant and both residual offsets — the same value the CPU path gets via
/// `reconstruct_rotated`. OOB/padded candidates (`cand ≥ n_real`) score −inf so the final top-k drops them.
#[cube(launch_unchecked)]
fn rerank_warren(
    rq_t: &Array<f32>,
    warren_rc: &Array<f32>,
    qc: &Array<f32>,
    cand_idx: &Array<u32>,
    codes_w: &Array<u32>,
    c1_w: &Array<u32>,
    c2_w: &Array<u32>,
    scale: &Array<f32>,
    bias: &Array<f32>,
    step1: &Array<f32>,
    step2: &Array<f32>,
    out: &mut Array<f32>,
    #[comptime] n: u32,
    #[comptime] b: u32,
    #[comptime] dim: u32,
    #[comptime] cc: u32,
    #[comptime] n_real: u32,
) {
    let pos = ABSOLUTE_POS; // 0..b*cc
    if pos < (b * cc) as usize {
        let (nn, bb, dd, ccu) = (n as usize, b as usize, dim as usize, cc as usize);
        let q = pos / ccu;
        let cand = cand_idx[pos] as usize;
        if cand < n_real as usize {
            let nw_nib = nn / 8usize;
            let nw_byte = nn / 4usize;
            let win = cand / 8usize;
            let shn = ((cand % 8usize) as u32) * 4u32;
            let wib = cand / 4usize;
            let shb = ((cand % 4usize) as u32) * 8u32;
            let sc = scale[cand];
            let bi = bias[cand];
            let s1 = step1[cand];
            let s2 = step2[cand];
            let mut acc = 0.0f32;
            for d in 0..dd {
                let u = ((codes_w[d * nw_nib + win] >> shn) & 15u32) as f32;
                let r1 = ((c1_w[d * nw_byte + wib] >> shb) & 255u32) as f32;
                let r2 = ((c2_w[d * nw_byte + wib] >> shb) & 255u32) as f32;
                let r_full = sc * u + bi + s1 * r1 + s2 * r2;
                // rq_full[d] = R·q = R·(q−c) + R·c = rq_t[d][q] + warren_rc[d]  (rotation done on the GPU).
                acc += (rq_t[d * bb + q] + warren_rc[d]) * r_full;
            }
            out[pos] = qc[q] + acc;
        } else {
            out[pos] = -3.0e38f32;
        }
    }
}

/// On-device top-k, parallel: `T` workers per query each keep a local top-k over a **disjoint** chunk
/// of the `n` distances (so no index overlap), writing their partial into their own slice — no shared
/// memory, no barriers. A multi-level merge tree ([`GpuFlatIndex::topk_pipeline`]) reduces the `T*k`
/// partials to the final `k`. Only `b*k` comes back, not the `b*n` matrix. `T` is chosen adaptively per
/// batch size (see `topk_pipeline`) so a single query still fills the GPU.
#[cube(launch_unchecked)]
fn topk_partial(
    dists: &Array<f32>,
    ps: &mut Array<f32>,
    pi: &mut Array<u32>,
    #[comptime] n: u32,
    #[comptime] b: u32,
    #[comptime] t: u32,
    #[comptime] k: u32,
) {
    let pos = ABSOLUTE_POS; // 0..b*t
    if pos < (b * t) as usize {
        let (nn, tt, kk) = (n as usize, t as usize, k as usize);
        let q = pos / tt;
        let tid = pos % tt;
        let steps = nn.div_ceil(tt);
        let base = pos * kk; // = (q*t + tid)*k
        for j in 0..kk {
            ps[base + j] = -3.0e38f32;
            pi[base + j] = 0u32;
        }
        // Threshold-guarded: `minv` = the running k-th-best (the current admission threshold), `minj` its
        // slot. Most elements fail `d > minv` and reject in O(1); only a *survivor* pays the O(k) re-scan to
        // find the new min. (The old code did the O(k) min-find on every element — the large-C wall.)
        // Worker `tid` walks the **strided** residue class `i = tid, tid+T, …` (a disjoint cover, so still
        // exact) — adjacent workers read adjacent distances = coalesced, where contiguous chunks were not.
        // Seed from slot 0 (= −3e38 after the init loop) so the type matches the array reads.
        let mut minv = ps[base];
        let mut minj = 0usize;
        for c in 0..steps {
            let i = tid + c * tt;
            if i < nn {
                let d = dists[q * nn + i];
                if d > minv {
                    ps[base + minj] = d;
                    pi[base + minj] = i as u32;
                    minv = ps[base];
                    minj = 0usize;
                    for j in 1..kk {
                        if ps[base + j] < minv {
                            minv = ps[base + j];
                            minj = j;
                        }
                    }
                }
            }
        }
    }
}

/// Generalized parallel merge: reduce `p` per-query top-k lists to `g` (each of the `b*g` workers merges a
/// disjoint span of `p/g` input lists — requires `p % g == 0`). Two calls form a reduction tree:
/// `T → G` (b·G-way parallel) then `G → 1`. This replaces the old single-pass merge, which ran on just `b`
/// threads serially over all `T·k` candidates — the narrow wall that made higher `T` unaffordable and
/// capped end-to-end QPS once the scan got cheap. The disjoint-span cover keeps the merge provably exact
/// (a global top-k item is best in its own partial, whose span some worker owns).
#[cube(launch_unchecked)]
fn topk_reduce(
    ps: &Array<f32>,
    pi: &Array<u32>,
    os: &mut Array<f32>,
    oi: &mut Array<u32>,
    #[comptime] b: u32,
    #[comptime] p: u32,
    #[comptime] g: u32,
    #[comptime] k: u32,
) {
    let pos = ABSOLUTE_POS; // 0..b*g
    if pos < (b * g) as usize {
        let (pp, gg, kk) = (p as usize, g as usize, k as usize);
        let grp = pos % gg;
        let per = pp / gg; // input lists this worker merges
        let obase = pos * kk;
        for j in 0..kk {
            os[obase + j] = -3.0e38f32;
            oi[obase + j] = 0u32;
        }
        let sbase = (pos / gg) * pp * kk + grp * per * kk; // query block + this worker's span
        let cnt = per * kk;
        // Same threshold-guard as topk_partial: O(1) reject, O(k) only on a survivor.
        let mut minv = os[obase];
        let mut minj = 0usize;
        for c in 0..cnt {
            let d = ps[sbase + c];
            if d > minv {
                os[obase + minj] = d;
                oi[obase + minj] = pi[sbase + c];
                minv = os[obase];
                minj = 0usize;
                for j in 1..kk {
                    if os[obase + j] < minv {
                        minv = os[obase + j];
                        minj = j;
                    }
                }
            }
        }
    }
}

/// Which quantizer's codes sit on the device.
enum Mode {
    /// 8-bit scalar, 4 codes/u32; `scale` pre-multiplies the query. ~0.97 recall, no rerank.
    Sq8 { scale: Vec<f32> },
    /// 1-bit sign of the mean-centred vector, 32 bits/u32; ~8× less code bandwidth. Recall is
    /// distribution-dependent (blind on nomic — needs the residual rerank, 1c).
    Sign1,
    /// Calibrated 1-bit RaBitQ: same 32 signs/u32 packing as Sign1, plus per-vector `dtc_sq`/`est_factor`
    /// device arrays and the fitted quantizer (for the per-query rotation). Unbiased estimator ⇒ usable
    /// recall where raw Sign1 is blind. `foxstash`'s `turborabit4` default is the multi-bit extension.
    RaBitQ {
        quant: Box<RaBitQuantizer>,
        dtc_sq: cubecl::server::Handle,
        est_factor: cubecl::server::Handle,
    },
    /// Multi-bit TurboRabit (`turborabit4` = 4-bit is the `foxstash` default). `B`-bit codes packed `cpw`
    /// per u32 (8 nibbles for `B ≤ 4`, 4 bytes for `B ∈ 5..=8`), plus per-vector `dtc_sq`/`f_rescale`
    /// device arrays and the fitted quantizer.
    TurboRabit {
        quant: Box<TurboRabitQuantizer>,
        dtc_sq: cubecl::server::Handle,
        f_rescale: cubecl::server::Handle,
        cpw: u32,
        /// The rotation `R` (row-major `[dim][dim]`) on the GPU, for the on-device query rotation.
        r_w: cubecl::server::Handle,
        /// Column sums of `R` (`colsum_r[d] = Σ_j R[j][d]`), so `cb_sum = c_B·Σ rq = c_B·Σ (q−c)·colsum_r`
        /// stays an `O(dim)` CPU dot instead of needing the rotated query summed on the GPU.
        colsum_r: Vec<f32>,
    },
}

/// A GPU-resident flat index. Build once (uploads packed codes); query with a coalesced scan.
pub struct GpuFlatIndex {
    client: Client,
    corpus_w: cubecl::server::Handle,
    mode: Mode,
    ids: Vec<String>,
    n: usize,
    dim: usize,
    /// Optional GPU-resident f32 corpus (row-major `[n][dim]`) for the on-device exact rerank
    /// ([`Self::search_batch_reranked_gpu`]); set by [`Self::with_gpu_rerank`]. `None` until requested.
    rerank_f32: Option<cubecl::server::Handle>,
}

impl GpuFlatIndex {
    /// SQ8 index: per-dimension `min`/`scale`, 4 codes/`u32` transposed to `[dim][n/4]`. `n` is padded
    /// to a multiple of 32 (covers both the 4- and 32-code packings) with zero rows that never win.
    pub fn build(vecs: &[Vec<f32>], ids: Vec<String>) -> Self {
        assert!(!vecs.is_empty(), "empty corpus");
        let dim = vecs[0].len();
        let n = vecs.len().div_ceil(32) * 32;

        let mut lo = vec![f32::INFINITY; dim];
        let mut hi = vec![f32::NEG_INFINITY; dim];
        for v in vecs {
            for d in 0..dim {
                lo[d] = lo[d].min(v[d]);
                hi[d] = hi[d].max(v[d]);
            }
        }
        let scale: Vec<f32> = (0..dim)
            .map(|d| ((hi[d] - lo[d]) / 255.0).max(1e-12))
            .collect();

        let nw = n / 4;
        let mut cw = vec![0u32; dim * nw];
        for (i, v) in vecs.iter().enumerate() {
            for d in 0..dim {
                let code = (((v[d] - lo[d]) / scale[d]).round().clamp(0.0, 255.0)) as u32;
                cw[d * nw + i / 4] |= code << (8 * (i % 4) as u32);
            }
        }

        let client = Rt::client(&Default::default());
        let corpus_w = client.create(cubecl::bytes::Bytes::from_elems(cw));
        Self {
            client,
            corpus_w,
            mode: Mode::Sq8 { scale },
            ids,
            n,
            dim,
            rerank_f32: None,
        }
    }

    /// 1-bit sign index (Phase 1b): bit = `v[d] >= mean[d]`, 32 bits/`u32` transposed to `[dim][n/32]`.
    /// ~8× less code memory than SQ8. Recall is distribution-dependent — high on friendly data, blind on
    /// nomic (needs the residual rerank, 1c). The bandwidth/crossover lever.
    pub fn build_sign1(vecs: &[Vec<f32>], ids: Vec<String>) -> Self {
        assert!(!vecs.is_empty(), "empty corpus");
        let dim = vecs[0].len();
        let n = vecs.len().div_ceil(32) * 32;

        let mut mean = vec![0f64; dim];
        for v in vecs {
            for d in 0..dim {
                mean[d] += v[d] as f64;
            }
        }
        let inv = 1.0 / vecs.len() as f64;
        let mean: Vec<f32> = mean.iter().map(|&m| (m * inv) as f32).collect();

        // pack: signs_w[d * (n/32) + i/32], bit (i%32) = (v[i][d] >= mean[d]). padded rows -> bit 0.
        let nw = n / 32;
        let mut sw = vec![0u32; dim * nw];
        for (i, v) in vecs.iter().enumerate() {
            for d in 0..dim {
                if v[d] >= mean[d] {
                    sw[d * nw + i / 32] |= 1u32 << (i % 32) as u32;
                }
            }
        }

        let client = Rt::client(&Default::default());
        let corpus_w = client.create(cubecl::bytes::Bytes::from_elems(sw));
        Self {
            client,
            corpus_w,
            mode: Mode::Sign1,
            ids,
            n,
            dim,
            rerank_f32: None,
        }
    }

    /// Calibrated 1-bit RaBitQ index (the real Phase 1b). Fits a `RaBitQuantizer` (centroid + orthonormal
    /// rotation), encodes each vector to sign bits packed `[dim][n/32]` like Sign1, and uploads the two
    /// per-vector estimator scalars (`dtc_sq`, `est_factor`). At query time the rotation is applied on the
    /// CPU (one O(D²) matvec/query — cheap next to scanning `n`), then the GPU scan is Sign1 + affine.
    pub fn build_rabitq(vecs: &[Vec<f32>], ids: Vec<String>) -> Self {
        assert!(!vecs.is_empty(), "empty corpus");
        let dim = vecs[0].len();
        let n = vecs.len().div_ceil(32) * 32;

        let quant = RaBitQuantizer::fit(vecs);
        let nw = n / 32;
        let mut sw = vec![0u32; dim * nw];
        // Padded rows (i >= vecs.len()) must NEVER win. Their score is `2·est·s − dtc_sq`; with the naive
        // zero-fill that is exactly 0, which outranks real vectors whenever real scores go negative (they
        // do on hostile distributions like nomic, where dtc_sq dominates) — flooding the top-k with rows
        // `assemble` then discards, cratering recall at any n not a multiple of 32. Sentinel dtc_sq = 3e38
        // (est stays 0) ⇒ padded score = −3e38, a guaranteed loser. (SQ8/Sign1 dodge this only because
        // their near-neighbour scores stay strongly positive; here 1-bit on nomic does not.)
        let mut dtc = vec![3.0e38f32; n];
        let mut est = vec![0f32; n];
        for (i, v) in vecs.iter().enumerate() {
            let code = quant.encode(v);
            // code.bits are indexed by *dimension* (sign of the rotated residual); transpose to [dim][n/32].
            for d in 0..dim {
                if (code.bits[d / 8] >> (d % 8)) & 1 == 1 {
                    sw[d * nw + i / 32] |= 1u32 << (i % 32) as u32;
                }
            }
            dtc[i] = code.dtc_sq;
            est[i] = code.est_factor;
        }

        let client = Rt::client(&Default::default());
        let corpus_w = client.create(cubecl::bytes::Bytes::from_elems(sw));
        let dtc_sq = client.create(cubecl::bytes::Bytes::from_elems(dtc));
        let est_factor = client.create(cubecl::bytes::Bytes::from_elems(est));
        Self {
            client,
            corpus_w,
            mode: Mode::RaBitQ {
                quant: Box::new(quant),
                dtc_sq,
                est_factor,
            },
            ids,
            n,
            dim,
            rerank_f32: None,
        }
    }

    /// Multi-bit TurboRabit index (`total_bits` ∈ 1..=8; 4 = the `turborabit4` default). Fits a
    /// `TurboRabitQuantizer` and packs the `B`-bit codes: `B ≤ 4` **nibble-packs** 8/`u32` (½ the bandwidth
    /// of SQ8 — what lets 4-bit beat SQ8 on the GPU), `B ∈ 5..=8` byte-packs 4/`u32`. Uploads the two
    /// per-vector estimator scalars.
    pub fn build_turborabit(vecs: &[Vec<f32>], ids: Vec<String>, total_bits: usize) -> Self {
        assert!(!vecs.is_empty(), "empty corpus");
        let dim = vecs[0].len();
        let n = vecs.len().div_ceil(32) * 32;

        let quant = TurboRabitQuantizer::fit(vecs, total_bits);
        // Nibble-pack when the codes fit 4 bits (8/u32); otherwise byte-pack (4/u32). n is a multiple of
        // 32, so both n/8 and n/4 are exact.
        let (cpw, bpc) = if total_bits <= 4 {
            (8usize, 4u32)
        } else {
            (4usize, 8u32)
        };
        let nw = n / cpw;
        let mut cw = vec![0u32; dim * nw];
        // Padded rows must lose: sentinel dtc_sq=3e38 + f_rescale=0 ⇒ score −3e38 (see build_rabitq).
        let mut dtc = vec![3.0e38f32; n];
        let mut fr = vec![0f32; n];
        for (i, v) in vecs.iter().enumerate() {
            let code = quant.encode(v);
            for d in 0..dim {
                cw[d * nw + i / cpw] |= (code.codes[d] as u32) << (bpc * (i % cpw) as u32);
            }
            dtc[i] = code.dtc_sq;
            fr[i] = code.f_rescale;
        }

        // Upload R for the on-device rotation, and precompute its column sums for the cheap CPU cb_sum.
        let r = quant.rotation();
        let mut colsum_r = vec![0f32; dim];
        for j in 0..dim {
            for d in 0..dim {
                colsum_r[d] += r[j * dim + d];
            }
        }

        let client = Rt::client(&Default::default());
        let corpus_w = client.create(cubecl::bytes::Bytes::from_elems(cw));
        let dtc_sq = client.create(cubecl::bytes::Bytes::from_elems(dtc));
        let f_rescale = client.create(cubecl::bytes::Bytes::from_elems(fr));
        let r_w = client.create(cubecl::bytes::Bytes::from_elems(r.to_vec()));
        Self {
            client,
            corpus_w,
            mode: Mode::TurboRabit {
                quant: Box::new(quant),
                dtc_sq,
                f_rescale,
                cpw: cpw as u32,
                r_w,
                colsum_r,
            },
            ids,
            n,
            dim,
            rerank_f32: None,
        }
    }

    /// Scan one query chunk on the GPU (mode-specific), returning the `b*n` distances handle.
    fn scan_chunk(&self, chunk: &[Vec<f32>], b: usize) -> cubecl::server::Handle {
        let (dim, n) = (self.dim, self.n);
        let oh = self.client.empty(b * n * core::mem::size_of::<f32>());
        let threads = 256u32;
        // All modes use the tiled scan (grid = n, one thread per candidate); tiled_count is set per arm.
        // transpose the query chunk to [dim][b] (pre-scaled for SQ8, raw for Sign1)
        let mut qt = vec![0f32; dim * b];
        match &self.mode {
            Mode::Sq8 { scale } => {
                for (qi, q) in chunk.iter().enumerate() {
                    for d in 0..dim {
                        qt[d * b + qi] = q[d] * scale[d];
                    }
                }
                let qh = self.client.create(cubecl::bytes::Bytes::from_elems(qt));
                // Tiled scan: one thread per candidate (grid = n), corpus read once, reused across b queries.
                let tiled_count = CubeCount::Static((n as u32).div_ceil(threads), 1, 1);
                unsafe {
                    scan_sq8_tiled::launch_unchecked(
                        &self.client,
                        tiled_count,
                        CubeDim::new_1d(threads),
                        ArrayArg::from_raw_parts(qh, b * dim),
                        ArrayArg::from_raw_parts(self.corpus_w.clone(), dim * (n / 4)),
                        ArrayArg::from_raw_parts(oh.clone(), b * n),
                        n as u32,
                        b as u32,
                        dim as u32,
                    );
                }
            }
            Mode::Sign1 => {
                for (qi, q) in chunk.iter().enumerate() {
                    for d in 0..dim {
                        qt[d * b + qi] = q[d];
                    }
                }
                let qh = self.client.create(cubecl::bytes::Bytes::from_elems(qt));
                let tiled_count = CubeCount::Static((n as u32).div_ceil(threads), 1, 1);
                unsafe {
                    scan_sign1_tiled::launch_unchecked(
                        &self.client,
                        tiled_count,
                        CubeDim::new_1d(threads),
                        ArrayArg::from_raw_parts(qh, b * dim),
                        ArrayArg::from_raw_parts(self.corpus_w.clone(), dim * (n / 32)),
                        ArrayArg::from_raw_parts(oh.clone(), b * n),
                        n as u32,
                        b as u32,
                        dim as u32,
                    );
                }
            }
            Mode::RaBitQ {
                quant,
                dtc_sq,
                est_factor,
            } => {
                // Rotate each query into RaBitQ space on the CPU (rq = R·(q−c)); transpose rq to [dim][b].
                for (qi, q) in chunk.iter().enumerate() {
                    let prep = quant.prepare_query(q);
                    for (d, &r) in prep.rq().iter().enumerate() {
                        qt[d * b + qi] = r;
                    }
                }
                let qh = self.client.create(cubecl::bytes::Bytes::from_elems(qt));
                let tiled_count = CubeCount::Static((n as u32).div_ceil(threads), 1, 1);
                unsafe {
                    scan_rabitq_tiled::launch_unchecked(
                        &self.client,
                        tiled_count,
                        CubeDim::new_1d(threads),
                        ArrayArg::from_raw_parts(qh, b * dim),
                        ArrayArg::from_raw_parts(self.corpus_w.clone(), dim * (n / 32)),
                        ArrayArg::from_raw_parts(dtc_sq.clone(), n),
                        ArrayArg::from_raw_parts(est_factor.clone(), n),
                        ArrayArg::from_raw_parts(oh.clone(), b * n),
                        n as u32,
                        b as u32,
                        dim as u32,
                    );
                }
            }
            Mode::TurboRabit {
                quant,
                dtc_sq,
                f_rescale,
                cpw,
                r_w,
                colsum_r,
            } => {
                // Centre each query on the CPU (O(dim), cheap) and fold cb_sum via colsum_r (also O(dim));
                // the expensive R·(q−c) rotation is done on the GPU (rotate_queries), not per-query on CPU.
                let cb = -(((1u32 << quant.total_bits()) - 1) as f32) / 2.0;
                let centroid = quant.centroid();
                let mut centered = vec![0f32; dim * b];
                let mut cbs = vec![0f32; b];
                for (qi, q) in chunk.iter().enumerate() {
                    let mut s = 0.0f32;
                    for d in 0..dim {
                        let cd = q[d] - centroid[d];
                        centered[d * b + qi] = cd; // transposed [dim][b] for the coalesced rotation
                        s += cd * colsum_r[d];
                    }
                    cbs[qi] = cb * s;
                }
                let cw = *cpw as usize;
                let bpc = 32u32 / cpw;
                let mask = (1u32 << bpc) - 1;
                let cenh = self
                    .client
                    .create(cubecl::bytes::Bytes::from_elems(centered));
                let cbh = self.client.create(cubecl::bytes::Bytes::from_elems(cbs));
                // GPU rotation: centered [b][dim] → rq_t [dim][b].
                let rqth = self.client.empty(dim * b * core::mem::size_of::<f32>());
                let rot_count = CubeCount::Static(((dim * b) as u32).div_ceil(threads), 1, 1);
                let tiled_count = CubeCount::Static((n as u32).div_ceil(threads), 1, 1);
                unsafe {
                    rotate_queries::launch_unchecked(
                        &self.client,
                        rot_count,
                        CubeDim::new_1d(threads),
                        ArrayArg::from_raw_parts(cenh, b * dim),
                        ArrayArg::from_raw_parts(r_w.clone(), dim * dim),
                        ArrayArg::from_raw_parts(rqth.clone(), dim * b),
                        b as u32,
                        dim as u32,
                    );
                    scan_turborabit_tiled::launch_unchecked(
                        &self.client,
                        tiled_count,
                        CubeDim::new_1d(threads),
                        ArrayArg::from_raw_parts(rqth, b * dim),
                        ArrayArg::from_raw_parts(self.corpus_w.clone(), dim * (n / cw)),
                        ArrayArg::from_raw_parts(dtc_sq.clone(), n),
                        ArrayArg::from_raw_parts(f_rescale.clone(), n),
                        ArrayArg::from_raw_parts(cbh, b),
                        ArrayArg::from_raw_parts(oh.clone(), b * n),
                        n as u32,
                        b as u32,
                        dim as u32,
                        *cpw,
                        bpc,
                        mask,
                    );
                }
            }
        }
        oh
    }

    /// Nearest `k` to a single query (ranked by the SQ8 approximate similarity).
    pub fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        self.search_batch(std::slice::from_ref(&query.to_vec()), k)
            .pop()
            .unwrap_or_default()
    }

    /// Instrumentation: time the pipeline stages (scan / +top-k / +readback) with GPU syncs between them,
    /// so we know which stage is the wall before optimising it. Returns `(scan_s, topk_s, readback_s)`
    /// summed over `reps` runs of the first `b`-query chunk. `#[doc(hidden)]` — a measuring tool, not API.
    #[doc(hidden)]
    pub fn profile_pipeline(&self, queries: &[Vec<f32>], k: usize, reps: usize) -> (f64, f64, f64) {
        use std::time::Instant;
        let b = queries.len().min(64);
        let chunk = &queries[..b];
        let sync = || cubecl::future::block_on(self.client.sync()).unwrap();
        let (mut t_scan, mut t_topk, mut t_read) = (0.0, 0.0, 0.0);
        for _ in 0..reps {
            // stage 1: scan only
            let t = Instant::now();
            let oh = self.scan_chunk(chunk, b);
            sync();
            t_scan += t.elapsed().as_secs_f64();
            // stage 2: adaptive top-k (partial + multi-level merge tree)
            let t = Instant::now();
            let (os, oi) = self.topk_pipeline(oh, b, k);
            sync();
            t_topk += t.elapsed().as_secs_f64();
            // stage 3: host readback
            let t = Instant::now();
            let _sb = self.client.read_one(os);
            let _ib = self.client.read_one(oi);
            t_read += t.elapsed().as_secs_f64();
        }
        (t_scan, t_topk, t_read)
    }

    /// Adaptive on-device top-k over the `b×n` distances in `oh`. Partial workers scale as ~`1/b`, so a
    /// single query still fills the GPU — the fixed `b·T` model gives only 256 workers at b=1, which the
    /// profile showed makes the top-k 32% of a single-query search. A multi-level merge tree (factor ~16
    /// per level) keeps every level parallel down to the tiny serial tail. Returns `(scores, idxs)` [b×k].
    fn topk_pipeline(
        &self,
        oh: cubecl::server::Handle,
        b: usize,
        k: usize,
    ) -> (cubecl::server::Handle, cubecl::server::Handle) {
        let n = self.n;
        let th = 256u32;
        // per-query partial workers: aim to fill ~16384 total across the batch, clamped and rounded down
        // to a power of two so `p /= 16` stays exact through the merge tree.
        let tw = {
            let t = (16384u32 / b as u32).clamp(64, 16384);
            1u32 << (31 - t.leading_zeros())
        };
        let ps = self
            .client
            .empty(b * tw as usize * k * core::mem::size_of::<f32>());
        let pi = self
            .client
            .empty(b * tw as usize * k * core::mem::size_of::<u32>());
        unsafe {
            topk_partial::launch_unchecked(
                &self.client,
                CubeCount::Static(((b as u32) * tw).div_ceil(th), 1, 1),
                CubeDim::new_1d(th),
                ArrayArg::from_raw_parts(oh, b * n),
                ArrayArg::from_raw_parts(ps.clone(), b * tw as usize * k),
                ArrayArg::from_raw_parts(pi.clone(), b * tw as usize * k),
                n as u32,
                b as u32,
                tw,
                k as u32,
            );
        }
        // merge tree: p -> g = max(p/16, 1) until a single list per query remains.
        let (mut cur_s, mut cur_i, mut p) = (ps, pi, tw);
        while p > 1 {
            let g = (p / 16).max(1);
            let os = self
                .client
                .empty(b * g as usize * k * core::mem::size_of::<f32>());
            let oi = self
                .client
                .empty(b * g as usize * k * core::mem::size_of::<u32>());
            unsafe {
                topk_reduce::launch_unchecked(
                    &self.client,
                    CubeCount::Static(((b as u32) * g).div_ceil(th).max(1), 1, 1),
                    CubeDim::new_1d(th),
                    ArrayArg::from_raw_parts(cur_s, b * p as usize * k),
                    ArrayArg::from_raw_parts(cur_i, b * p as usize * k),
                    ArrayArg::from_raw_parts(os.clone(), b * g as usize * k),
                    ArrayArg::from_raw_parts(oi.clone(), b * g as usize * k),
                    b as u32,
                    p,
                    g,
                    k as u32,
                );
            }
            cur_s = os;
            cur_i = oi;
            p = g;
        }
        (cur_s, cur_i)
    }

    /// Nearest `k` for many queries. Scans on the GPU; selects top-`k` per query.
    pub fn search_batch(&self, queries: &[Vec<f32>], k: usize) -> Vec<Vec<SearchResult>> {
        let mut out_all = Vec::with_capacity(queries.len());
        let batch = 64usize;
        for chunk in queries.chunks(batch) {
            let b = chunk.len();
            let oh = self.scan_chunk(chunk, b);
            let (os, oi) = self.topk_pipeline(oh, b, k);
            let sb = self.client.read_one(os).unwrap();
            let ib = self.client.read_one(oi).unwrap();
            let scores: &[f32] = bytemuck::cast_slice(&sb);
            let idxs: &[u32] = bytemuck::cast_slice(&ib);
            for qi in 0..b {
                out_all.push(
                    self.assemble(&scores[qi * k..(qi + 1) * k], &idxs[qi * k..(qi + 1) * k]),
                );
            }
        }
        out_all
    }

    /// The coarse stage of a two-stage search: GPU scan (the index's quantized mode) → top-`c` candidate
    /// **indices** per query (real indices only). The caller reranks this small pool with a precise
    /// representation (exact f32 in [`search_batch_reranked`](Self::search_batch_reranked), or Warren's
    /// 8-bit residual in [`WarrenRerankIndex`]).
    pub fn coarse_candidates(&self, queries: &[Vec<f32>], c: usize) -> Vec<Vec<usize>> {
        let n_real = self.ids.len();
        let mut out = Vec::with_capacity(queries.len());
        let batch = 64usize;
        for chunk in queries.chunks(batch) {
            let b = chunk.len();
            let oh = self.scan_chunk(chunk, b);
            let (_os, oi) = self.topk_pipeline(oh, b, c);
            let ib = self.client.read_one(oi).unwrap();
            let idxs: &[u32] = bytemuck::cast_slice(&ib);
            for qi in 0..b {
                out.push(
                    idxs[qi * c..(qi + 1) * c]
                        .iter()
                        .map(|&i| i as usize)
                        .filter(|&i| i < n_real)
                        .collect(),
                );
            }
        }
        out
    }

    /// Two-stage search (**1c**): coarse GPU scan → top-`rerank_c` candidates → exact f32 rerank → top-`k`.
    /// The coarse stage ranks by the quantized mode's score; the rerank rescores just the small candidate
    /// pool against the full-precision `vecs` (exact dot), recovering the recall the coarse quantizer lost.
    /// Because the rerank recomputes an exact dot, it is correct for *any* coarse mode (the mode only picks
    /// which candidates enter the pool). `vecs` must be the corpus the index was built from. Falls back to
    /// [`search_batch`](Self::search_batch) when `rerank_c <= k` (nothing to refine).
    ///
    /// This is the flat-index form of Warren's two-phase idea. Warren avoids retaining f32 by reranking
    /// against an 8-bit residual (⅓ less memory); here f32 is read only for the `rerank_c` candidates, and
    /// establishes the recall ceiling the residual variant would approximate.
    pub fn search_batch_reranked(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        rerank_c: usize,
        vecs: &[Vec<f32>],
    ) -> Vec<Vec<SearchResult>> {
        if rerank_c <= k {
            return self.search_batch(queries, k);
        }
        self.coarse_candidates(queries, rerank_c)
            .into_iter()
            .zip(queries)
            .map(|(cands, q)| {
                let mut scored: Vec<(f32, usize)> = cands
                    .into_iter()
                    .map(|i| (q.iter().zip(&vecs[i]).map(|(a, b)| a * b).sum::<f32>(), i))
                    .collect();
                scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
                scored.truncate(k);
                scored
                    .into_iter()
                    .map(|(score, i)| SearchResult {
                        id: self.ids[i].clone(),
                        content: String::new(),
                        score,
                        metadata: None,
                    })
                    .collect()
            })
            .collect()
    }

    /// Turn the GPU's unsorted top-k (scores + row indices) into sorted `SearchResult`s, dropping any
    /// padded row (index >= real count) that a `k > n` query could surface.
    fn assemble(&self, scores: &[f32], idxs: &[u32]) -> Vec<SearchResult> {
        let n_real = self.ids.len();
        let mut pairs: Vec<(f32, usize)> = scores
            .iter()
            .zip(idxs)
            .map(|(&s, &i)| (s, i as usize))
            .filter(|&(_, i)| i < n_real)
            .collect();
        pairs.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        pairs
            .into_iter()
            .map(|(score, i)| SearchResult {
                id: self.ids[i].clone(),
                content: String::new(),
                score,
                metadata: None,
            })
            .collect()
    }

    /// Retain the full-precision corpus on the GPU (row-major `[n][dim]`, padded rows zero) so
    /// [`Self::search_batch_reranked_gpu`] can rerank on-device. `vecs` must be the corpus the index was
    /// built from. Costs ~`n·dim·4` bytes of VRAM — the price of an exact rerank that isn't CPU-bound.
    pub fn with_gpu_rerank(mut self, vecs: &[Vec<f32>]) -> Self {
        let (dim, n) = (self.dim, self.n);
        let mut f = vec![0f32; n * dim];
        for (i, v) in vecs.iter().enumerate() {
            f[i * dim..i * dim + dim].copy_from_slice(v);
        }
        self.rerank_f32 = Some(self.client.create(cubecl::bytes::Bytes::from_elems(f)));
        self
    }

    /// Two-stage search with the exact f32 rerank **on the GPU**: coarse scan → top-`rerank_c` → GPU
    /// dot-rerank → cheap CPU top-`k`. Requires [`Self::with_gpu_rerank`]. Same result as
    /// [`Self::search_batch_reranked`] (CPU) but the `O(C·dim)` rerank runs in parallel — this is what
    /// uncaps the CPU-rerank ceiling, fully for SQ8 (no query rotation).
    pub fn search_batch_reranked_gpu(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        rerank_c: usize,
    ) -> Vec<Vec<SearchResult>> {
        let corpus = self
            .rerank_f32
            .as_ref()
            .expect("call with_gpu_rerank() first");
        let (dim, n) = (self.dim, self.n);
        let n_real = self.ids.len();
        let mut out_all = Vec::with_capacity(queries.len());
        let batch = 64usize;
        for chunk in queries.chunks(batch) {
            let b = chunk.len();
            let oh = self.scan_chunk(chunk, b);
            let (_os, oi) = self.topk_pipeline(oh, b, rerank_c);
            let mut qf = vec![0f32; b * dim];
            for (qi, q) in chunk.iter().enumerate() {
                qf[qi * dim..qi * dim + dim].copy_from_slice(q);
            }
            let qh = self.client.create(cubecl::bytes::Bytes::from_elems(qf));
            let outh = self
                .client
                .empty(b * rerank_c * core::mem::size_of::<f32>());
            let threads = 256u32;
            let total = (b * rerank_c) as u32;
            unsafe {
                rerank_f32_dot::launch_unchecked(
                    &self.client,
                    CubeCount::Static(total.div_ceil(threads), 1, 1),
                    CubeDim::new_1d(threads),
                    ArrayArg::from_raw_parts(qh, b * dim),
                    ArrayArg::from_raw_parts(oi.clone(), b * rerank_c),
                    ArrayArg::from_raw_parts(corpus.clone(), n * dim),
                    ArrayArg::from_raw_parts(outh.clone(), b * rerank_c),
                    b as u32,
                    dim as u32,
                    rerank_c as u32,
                    n_real as u32,
                );
            }
            let sb = self.client.read_one(outh).unwrap();
            let ib = self.client.read_one(oi).unwrap();
            let scores: &[f32] = bytemuck::cast_slice(&sb);
            let idxs: &[u32] = bytemuck::cast_slice(&ib);
            for qi in 0..b {
                let mut pairs: Vec<(f32, usize)> = (0..rerank_c)
                    .map(|ci| {
                        (
                            scores[qi * rerank_c + ci],
                            idxs[qi * rerank_c + ci] as usize,
                        )
                    })
                    .filter(|&(s, i)| i < n_real && s > -3.0e37)
                    .collect();
                pairs.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
                pairs.truncate(k);
                out_all.push(
                    pairs
                        .into_iter()
                        .map(|(score, i)| SearchResult {
                            id: self.ids[i].clone(),
                            content: String::new(),
                            score,
                            metadata: None,
                        })
                        .collect(),
                );
            }
        }
        out_all
    }

    /// Instrumentation: time the reranked pipeline's stages — coarse (scan + top-C) / rerank kernel /
    /// readback+CPU-topk — with GPU syncs between them. `#[doc(hidden)]`. Returns `(coarse, rerank, read)` s.
    #[doc(hidden)]
    pub fn profile_rerank(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        rerank_c: usize,
        reps: usize,
    ) -> (f64, f64, f64) {
        use std::time::Instant;
        let corpus = self
            .rerank_f32
            .as_ref()
            .expect("call with_gpu_rerank() first");
        let (dim, n) = (self.dim, self.n);
        let n_real = self.ids.len();
        let b = queries.len().min(64);
        let chunk = &queries[..b];
        let sync = || cubecl::future::block_on(self.client.sync()).unwrap();
        let (mut tc, mut tr, mut td) = (0.0, 0.0, 0.0);
        for _ in 0..reps {
            let t = Instant::now();
            let oh = self.scan_chunk(chunk, b);
            let (_os, oi) = self.topk_pipeline(oh, b, rerank_c);
            sync();
            tc += t.elapsed().as_secs_f64();

            let t = Instant::now();
            let mut qf = vec![0f32; b * dim];
            for (qi, q) in chunk.iter().enumerate() {
                qf[qi * dim..qi * dim + dim].copy_from_slice(q);
            }
            let qh = self.client.create(cubecl::bytes::Bytes::from_elems(qf));
            let outh = self.client.empty(b * rerank_c * 4);
            let th = 256u32;
            unsafe {
                rerank_f32_dot::launch_unchecked(
                    &self.client,
                    CubeCount::Static(((b * rerank_c) as u32).div_ceil(th), 1, 1),
                    CubeDim::new_1d(th),
                    ArrayArg::from_raw_parts(qh, b * dim),
                    ArrayArg::from_raw_parts(oi.clone(), b * rerank_c),
                    ArrayArg::from_raw_parts(corpus.clone(), n * dim),
                    ArrayArg::from_raw_parts(outh.clone(), b * rerank_c),
                    b as u32,
                    dim as u32,
                    rerank_c as u32,
                    n_real as u32,
                );
            }
            sync();
            tr += t.elapsed().as_secs_f64();

            let t = Instant::now();
            let sb = self.client.read_one(outh).unwrap();
            let ib = self.client.read_one(oi).unwrap();
            let scores: &[f32] = bytemuck::cast_slice(&sb);
            let idxs: &[u32] = bytemuck::cast_slice(&ib);
            for qi in 0..b {
                let mut pairs: Vec<(f32, usize)> = (0..rerank_c)
                    .map(|ci| {
                        (
                            scores[qi * rerank_c + ci],
                            idxs[qi * rerank_c + ci] as usize,
                        )
                    })
                    .filter(|&(s, i)| i < n_real && s > -3.0e37)
                    .collect();
                pairs.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
                pairs.truncate(k);
            }
            td += t.elapsed().as_secs_f64();
        }
        (tc, tr, td)
    }

    /// The per-vector ids, in index order.
    pub fn ids(&self) -> &[String] {
        &self.ids
    }

    /// Number of indexed vectors.
    pub fn len(&self) -> usize {
        self.ids.len()
    }
    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

/// Per-vector Warren residual: a **two-level 8+8-bit** quantization of `R·(x−c) − reconstruct_rotated`,
/// each level with its own affine (`lo`/`step`). Two levels reach the f32 rerank ceiling where one is
/// ~0.002 short (a single 8-bit range can't resolve both the coarse residual and what it leaves).
struct WarrenRes {
    lo1: f32,
    step1: f32,
    lo2: f32,
    step2: f32,
    c1: Vec<u8>,
    c2: Vec<u8>,
}

/// Quantize `v` to 8 bits with a shared affine — mirrors `foxstash-core`'s Warren `quant` closure exactly.
fn warren_quant(v: &[f32]) -> (f32, f32, Vec<u8>) {
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &x in v {
        lo = lo.min(x);
        hi = hi.max(x);
    }
    let step = if hi > lo { (hi - lo) / 255.0 } else { 0.0 };
    let codes = v
        .iter()
        .map(|&x| {
            if step > 0.0 {
                (((x - lo) / step) + 0.5).floor().clamp(0.0, 255.0) as u8
            } else {
                0
            }
        })
        .collect();
    (lo, step, codes)
}

/// Warren-style two-stage index (**1c, the memory path**): a TurboRabit 4-bit GPU scan for the coarse
/// stage plus a per-vector **8+8-bit residual** — and **no retained f32**. The rerank rescores the coarse
/// top-C in rotated space, `⟨q,x⟩ = ⟨q,c⟩ + ⟨R·q, r_recon + e⟩` (`R` orthonormal ⇒ this is the exact dot
/// as the residual → the true one), reaching ~0.99 recall at ~⅓ less memory than retaining f32. Port of
/// `foxstash-core`'s `Storage::Warren`; the residual encode and rerank mirror the CPU derivation exactly.
///
/// This is the memory-conscious backend for the two-stage framework: pick it over
/// [`GpuFlatIndex::search_batch_reranked`]'s exact f32 rerank when you chose 4-bit codes *for* their half
/// memory, since storing f32 would erase that win.
pub struct WarrenRerankIndex {
    inner: GpuFlatIndex,
    quant: TurboRabitQuantizer,
    /// `R·c` — the rotated centroid, added to `prepare_query`'s `rq` to get `R·q` once per query.
    warren_rc: Vec<f32>,
    /// Per-vector 4-bit codes; drive `reconstruct_rotated` (the coarse reconstruction the residual refines).
    codes: Vec<TurboRabitCode>,
    res: Vec<WarrenRes>,
    /// GPU-resident residual for [`Self::search_batch_gpu`]: byte-packed 8-bit levels `[dim][n/4]` plus the
    /// per-vector `scale`/`bias`/`step1`/`step2` (`bias = scale·c_B + lo1 + lo2`). No f32 — 16 bits/dim.
    c1_w: cubecl::server::Handle,
    c2_w: cubecl::server::Handle,
    g_scale: cubecl::server::Handle,
    g_bias: cubecl::server::Handle,
    g_step1: cubecl::server::Handle,
    g_step2: cubecl::server::Handle,
}

impl WarrenRerankIndex {
    /// Build a Warren index: a TurboRabit-`bits` GPU scan index plus the per-vector 8+8 residual. `bits`
    /// ∈ 1..=4 (Warren is the 4-bit-walk mode). The residual quantizer is refit deterministically, so it
    /// matches the codes the inner scan was built with.
    pub fn build(vecs: &[Vec<f32>], ids: Vec<String>, bits: usize) -> Self {
        assert!(
            (1..=4).contains(&bits),
            "Warren bits must be in 1..=4, got {bits}"
        );
        let inner = GpuFlatIndex::build_turborabit(vecs, ids, bits);
        let quant = TurboRabitQuantizer::fit(vecs, bits);
        let warren_rc = quant.rotate(quant.centroid());

        // GPU residual arrays: byte-packed 8-bit levels [dim][n/4] + per-vector scale/bias/step1/step2,
        // padded to the inner index's n (padded rows stay zero → they score −inf at rerank).
        let (dim, n) = (inner.dim, inner.n);
        let cb = -(((1u32 << bits) - 1) as f32) / 2.0;
        let nwb = n / 4;
        let mut c1_w = vec![0u32; dim * nwb];
        let mut c2_w = vec![0u32; dim * nwb];
        let (mut g_scale, mut g_bias) = (vec![0f32; n], vec![0f32; n]);
        let (mut g_step1, mut g_step2) = (vec![0f32; n], vec![0f32; n]);

        let mut codes = Vec::with_capacity(vecs.len());
        let mut res = Vec::with_capacity(vecs.len());
        for (i, v) in vecs.iter().enumerate() {
            let code = quant.encode(v);
            // residual e = R·(x−c) − reconstruct_rotated(code), then two 8-bit levels (each rescaled).
            let r_recon = quant.reconstruct_rotated(&code);
            let centred: Vec<f32> = v.iter().zip(quant.centroid()).map(|(a, b)| a - b).collect();
            let r_true = quant.rotate(&centred);
            let e: Vec<f32> = r_true.iter().zip(&r_recon).map(|(a, b)| a - b).collect();
            let (lo1, step1, c1) = warren_quant(&e);
            let e2: Vec<f32> = e
                .iter()
                .zip(&c1)
                .map(|(&x, &c)| x - (lo1 + c as f32 * step1))
                .collect();
            let (lo2, step2, c2) = warren_quant(&e2);

            // pack residual codes [dim][n/4] (byte per code) and the folded per-vector scalars for the GPU.
            for d in 0..dim {
                c1_w[d * nwb + i / 4] |= (c1[d] as u32) << (8 * (i % 4) as u32);
                c2_w[d * nwb + i / 4] |= (c2[d] as u32) << (8 * (i % 4) as u32);
            }
            let l = code.dtc_sq.sqrt();
            let gnorm = code
                .codes
                .iter()
                .map(|&u| {
                    let g = u as f32 + cb;
                    g * g
                })
                .sum::<f32>()
                .sqrt();
            let scale = if gnorm > 0.0 && l > f32::EPSILON {
                l / gnorm
            } else {
                0.0
            };
            g_scale[i] = scale;
            g_bias[i] = scale * cb + lo1 + lo2;
            g_step1[i] = step1;
            g_step2[i] = step2;

            codes.push(code);
            res.push(WarrenRes {
                lo1,
                step1,
                lo2,
                step2,
                c1,
                c2,
            });
        }

        let up = |v: Vec<u32>| inner.client.create(cubecl::bytes::Bytes::from_elems(v));
        let upf = |v: Vec<f32>| inner.client.create(cubecl::bytes::Bytes::from_elems(v));
        Self {
            c1_w: up(c1_w),
            c2_w: up(c2_w),
            g_scale: upf(g_scale),
            g_bias: upf(g_bias),
            g_step1: upf(g_step1),
            g_step2: upf(g_step2),
            inner,
            quant,
            warren_rc,
            codes,
            res,
        }
    }

    /// Two-stage search: GPU 4-bit scan → top-`rerank_c` → **8-bit residual rerank** → top-`k`.
    pub fn search_batch(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        rerank_c: usize,
    ) -> Vec<Vec<SearchResult>> {
        let ids = self.inner.ids();
        let cands = self.inner.coarse_candidates(queries, rerank_c);
        cands
            .into_iter()
            .zip(queries)
            .map(|(cand, q)| {
                // R·q = rq + R·c; qc = ⟨q,c⟩ (a per-query constant, kept so the score is the true dot).
                let prep = self.quant.prepare_query(q);
                let rq_full: Vec<f32> = prep
                    .rq()
                    .iter()
                    .zip(&self.warren_rc)
                    .map(|(a, b)| a + b)
                    .collect();
                let qc: f32 = q
                    .iter()
                    .zip(self.quant.centroid())
                    .map(|(a, b)| a * b)
                    .sum();
                let mut scored: Vec<(f32, usize)> = cand
                    .into_iter()
                    .map(|i| {
                        let r_recon = self.quant.reconstruct_rotated(&self.codes[i]);
                        let wr = &self.res[i];
                        // acc = ⟨q,c⟩ + ⟨R·q, r_recon + e1 + e2⟩ = the (residual-refined) exact dot.
                        let dot: f32 = rq_full
                            .iter()
                            .enumerate()
                            .map(|(d, &rq)| {
                                let r_full = r_recon[d]
                                    + (wr.lo1 + wr.c1[d] as f32 * wr.step1)
                                    + (wr.lo2 + wr.c2[d] as f32 * wr.step2);
                                rq * r_full
                            })
                            .sum();
                        (qc + dot, i)
                    })
                    .collect();
                scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
                scored.truncate(k);
                scored
                    .into_iter()
                    .map(|(score, i)| SearchResult {
                        id: ids[i].clone(),
                        content: String::new(),
                        score,
                        metadata: None,
                    })
                    .collect()
            })
            .collect()
    }

    /// Two-stage search with the rerank **on the GPU**: coarse 4-bit scan → top-`rerank_c` → the
    /// `rerank_warren` kernel recomputes the residual-refined dot on device → a cheap CPU top-`k` over the
    /// `rerank_c` refined scores. Same result as [`search_batch`](Self::search_batch) (which reranks on the
    /// CPU) but the `O(C·dim)` work runs in parallel — this is what uncaps the ~1000-QPS CPU-rerank ceiling.
    pub fn search_batch_gpu(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        rerank_c: usize,
    ) -> Vec<Vec<SearchResult>> {
        let ids = self.inner.ids();
        let (dim, n) = (self.inner.dim, self.inner.n);
        let n_real = ids.len();
        let client = &self.inner.client;
        let mut out_all = Vec::with_capacity(queries.len());
        let batch = 64usize;
        for chunk in queries.chunks(batch) {
            let b = chunk.len();
            // coarse stage on the GPU: scan → top-C candidate indices (kept on device as `oi`).
            let oh = self.inner.scan_chunk(chunk, b);
            let (_os, oi) = self.inner.topk_pipeline(oh, b, rerank_c);
            // R lives in the inner TR4 index; the rerank's R·q is done on the GPU too (rq_t + warren_rc).
            let r_w = match &self.inner.mode {
                Mode::TurboRabit { r_w, .. } => r_w,
                _ => unreachable!("WarrenRerankIndex inner is always TurboRabit"),
            };
            let centroid = self.quant.centroid();
            let mut centered = vec![0f32; b * dim];
            let mut qcv = vec![0f32; b];
            for (qi, q) in chunk.iter().enumerate() {
                let mut qc = 0.0f32;
                for d in 0..dim {
                    centered[d * b + qi] = q[d] - centroid[d]; // transposed [dim][b] for the coalesced rotation
                    qc += q[d] * centroid[d];
                }
                qcv[qi] = qc;
            }
            let threads = 256u32;
            // GPU rotation: centered [b][dim] → rq_t [dim][b].
            let cenh = client.create(cubecl::bytes::Bytes::from_elems(centered));
            let rqth = client.empty(dim * b * core::mem::size_of::<f32>());
            let rot_count = CubeCount::Static(((dim * b) as u32).div_ceil(threads), 1, 1);
            unsafe {
                rotate_queries::launch_unchecked(
                    client,
                    rot_count,
                    CubeDim::new_1d(threads),
                    ArrayArg::from_raw_parts(cenh, b * dim),
                    ArrayArg::from_raw_parts(r_w.clone(), dim * dim),
                    ArrayArg::from_raw_parts(rqth.clone(), dim * b),
                    b as u32,
                    dim as u32,
                );
            }
            let rch = client.create(cubecl::bytes::Bytes::from_elems(self.warren_rc.clone()));
            let qch = client.create(cubecl::bytes::Bytes::from_elems(qcv));
            let outh = client.empty(b * rerank_c * core::mem::size_of::<f32>());
            let total = (b * rerank_c) as u32;
            unsafe {
                rerank_warren::launch_unchecked(
                    client,
                    CubeCount::Static(total.div_ceil(threads), 1, 1),
                    CubeDim::new_1d(threads),
                    ArrayArg::from_raw_parts(rqth, dim * b),
                    ArrayArg::from_raw_parts(rch, dim),
                    ArrayArg::from_raw_parts(qch, b),
                    ArrayArg::from_raw_parts(oi.clone(), b * rerank_c),
                    ArrayArg::from_raw_parts(self.inner.corpus_w.clone(), dim * (n / 8)),
                    ArrayArg::from_raw_parts(self.c1_w.clone(), dim * (n / 4)),
                    ArrayArg::from_raw_parts(self.c2_w.clone(), dim * (n / 4)),
                    ArrayArg::from_raw_parts(self.g_scale.clone(), n),
                    ArrayArg::from_raw_parts(self.g_bias.clone(), n),
                    ArrayArg::from_raw_parts(self.g_step1.clone(), n),
                    ArrayArg::from_raw_parts(self.g_step2.clone(), n),
                    ArrayArg::from_raw_parts(outh.clone(), b * rerank_c),
                    n as u32,
                    b as u32,
                    dim as u32,
                    rerank_c as u32,
                    n_real as u32,
                );
            }
            let sb = client.read_one(outh).unwrap();
            let ib = client.read_one(oi).unwrap();
            let scores: &[f32] = bytemuck::cast_slice(&sb);
            let idxs: &[u32] = bytemuck::cast_slice(&ib);
            for qi in 0..b {
                let mut pairs: Vec<(f32, usize)> = (0..rerank_c)
                    .map(|ci| {
                        (
                            scores[qi * rerank_c + ci],
                            idxs[qi * rerank_c + ci] as usize,
                        )
                    })
                    .filter(|&(s, i)| i < n_real && s > -3.0e37)
                    .collect();
                pairs.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
                pairs.truncate(k);
                out_all.push(
                    pairs
                        .into_iter()
                        .map(|(score, i)| SearchResult {
                            id: ids[i].clone(),
                            content: String::new(),
                            score,
                            metadata: None,
                        })
                        .collect(),
                );
            }
        }
        out_all
    }

    /// Nearest `k` to a single query.
    pub fn search(&self, query: &[f32], k: usize, rerank_c: usize) -> Vec<SearchResult> {
        self.search_batch(std::slice::from_ref(&query.to_vec()), k, rerank_c)
            .pop()
            .unwrap_or_default()
    }

    /// Number of indexed vectors.
    pub fn len(&self) -> usize {
        self.inner.len()
    }
    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

/// Lloyd's k-means, deterministic (evenly-spaced init — the crate has no `rand` outside tests). Returns
/// `(centroids[nlist][dim], assignment[n])`. Build-time only; a few iterations suffice for IVF cells.
fn kmeans(vecs: &[Vec<f32>], nlist: usize, iters: usize) -> (Vec<Vec<f32>>, Vec<usize>) {
    let (n, dim) = (vecs.len(), vecs[0].len());
    let mut cent: Vec<Vec<f32>> = (0..nlist).map(|i| vecs[(i * n) / nlist].clone()).collect();
    let mut assign = vec![0usize; n];
    for _ in 0..iters {
        for (vi, v) in vecs.iter().enumerate() {
            let mut best = 0usize;
            let mut bestd = f32::INFINITY;
            for (ci, c) in cent.iter().enumerate() {
                let d: f32 = v.iter().zip(c).map(|(a, b)| (a - b) * (a - b)).sum();
                if d < bestd {
                    bestd = d;
                    best = ci;
                }
            }
            assign[vi] = best;
        }
        let mut sums = vec![vec![0f64; dim]; nlist];
        let mut cnts = vec![0usize; nlist];
        for (vi, v) in vecs.iter().enumerate() {
            let a = assign[vi];
            cnts[a] += 1;
            for d in 0..dim {
                sums[a][d] += v[d] as f64;
            }
        }
        for ci in 0..nlist {
            if cnts[ci] > 0 {
                for d in 0..dim {
                    cent[ci][d] = (sums[ci][d] / cnts[ci] as f64) as f32;
                }
            }
        }
    }
    (cent, assign)
}

/// GPU **IVF** index (Phase 2 — the scale lever). Cluster the corpus into `nlist` cells (k-means); at query
/// time scan only the `nprobe` nearest cells instead of all `n`, so the scan is `~n·nprobe/nlist` distances
/// — **sub-linear in n**, where the flat scan is O(n). The corpus is **reordered by cell** and **SQ8-coded
/// transposed `[dim][n/4]`**, so a cell's vectors are contiguous and the per-cell scan is *coalesced* (the
/// `ivf_scan_sq8` kernel) — 4× less data than an f32 gather and warp-coalesced, the two things that make
/// IVF pay. Centroid search is on the CPU (`nlist` small). Candidates are the ~0.97-recall SQ8 estimate
/// within the probed cells; recall vs exact rises with `nprobe` (cell coverage) toward SQ8's ceiling.
pub struct GpuIvfIndex {
    client: Client,
    /// SQ8 codes of the **reordered** corpus, transposed `[dim][n/4]` (coalesced cell scan).
    codes_w: cubecl::server::Handle,
    /// Per-dim SQ8 scale, to pre-scale the query (`Σq·min` constant drops, as in the flat scan).
    scale: Vec<f32>,
    /// `[nlist][dim]` cell centroids (CPU centroid search).
    centroids: Vec<Vec<f32>>,
    /// `cell_offsets[c]..cell_offsets[c+1]` = cell `c`'s contiguous span of **reordered** indices.
    cell_offsets: Vec<usize>,
    /// Ids in reordered order (a reordered index maps straight to its id).
    ids: Vec<String>,
    max_cell: usize,
    n: usize,      // padded to a multiple of 4 for the [dim][n/4] packing
    n_real: usize, // = ids.len()
    dim: usize,
}

impl GpuIvfIndex {
    /// Build: k-means into `nlist` cells (`iters` Lloyd iterations), reorder the corpus by cell, and upload
    /// the reordered SQ8 codes transposed for a coalesced cell scan.
    pub fn build(vecs: &[Vec<f32>], ids: Vec<String>, nlist: usize, iters: usize) -> Self {
        assert!(!vecs.is_empty(), "empty corpus");
        assert!(nlist >= 1 && nlist <= vecs.len(), "nlist must be in 1..=n");
        let (n_real, dim) = (vecs.len(), vecs[0].len());
        let (centroids, assign) = kmeans(vecs, nlist, iters);

        // Reorder by cell: order = concat of cells; cell_offsets delimit each cell's contiguous span.
        let mut cells = vec![Vec::new(); nlist];
        for (vi, &a) in assign.iter().enumerate() {
            cells[a].push(vi);
        }
        let max_cell = cells.iter().map(|c| c.len()).max().unwrap_or(0);
        let mut order = Vec::with_capacity(n_real);
        let mut cell_offsets = vec![0usize; nlist + 1];
        for (c, cell) in cells.iter().enumerate() {
            cell_offsets[c] = order.len();
            order.extend_from_slice(cell);
        }
        cell_offsets[nlist] = order.len();
        let ids: Vec<String> = order.iter().map(|&o| ids[o].clone()).collect();

        // SQ8 over the whole corpus; code the reordered vectors, transposed [dim][n/4].
        let n = n_real.div_ceil(4) * 4;
        let mut lo = vec![f32::INFINITY; dim];
        let mut hi = vec![f32::NEG_INFINITY; dim];
        for v in vecs {
            for d in 0..dim {
                lo[d] = lo[d].min(v[d]);
                hi[d] = hi[d].max(v[d]);
            }
        }
        let scale: Vec<f32> = (0..dim)
            .map(|d| ((hi[d] - lo[d]) / 255.0).max(1e-12))
            .collect();
        let nw = n / 4;
        let mut cw = vec![0u32; dim * nw];
        for (newi, &orig) in order.iter().enumerate() {
            let v = &vecs[orig];
            for d in 0..dim {
                let code = (((v[d] - lo[d]) / scale[d]).round().clamp(0.0, 255.0)) as u32;
                cw[d * nw + newi / 4] |= code << (8 * (newi % 4) as u32);
            }
        }

        let client = Rt::client(&Default::default());
        let codes_w = client.create(cubecl::bytes::Bytes::from_elems(cw));
        Self {
            client,
            codes_w,
            scale,
            centroids,
            cell_offsets,
            ids,
            max_cell,
            n,
            n_real,
            dim,
        }
    }

    /// Nearest `k` per query, probing the `nprobe` nearest cells. Higher `nprobe` → more cell coverage →
    /// recall rises toward SQ8's flat-scan ceiling (which `nprobe = nlist` reproduces exactly).
    pub fn search_batch(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        nprobe: usize,
    ) -> Vec<Vec<SearchResult>> {
        let (n, dim) = (self.n, self.dim);
        let nprobe = nprobe.min(self.centroids.len());
        let max_cand = (nprobe * self.max_cell.max(1)).max(1); // fixed candidate width (padded to n sentinel)
        let mut out_all = Vec::with_capacity(queries.len());
        let batch = 64usize;
        for chunk in queries.chunks(batch) {
            let b = chunk.len();
            let mut qt = vec![0f32; dim * b]; // [dim][b] pre-scaled
            let mut cand = vec![n as u32; b * max_cand]; // sentinel n ≥ n_real ⇒ −inf, dropped
            for (qi, q) in chunk.iter().enumerate() {
                for d in 0..dim {
                    qt[d * b + qi] = q[d] * self.scale[d];
                }
                // centroid search (CPU): the nprobe nearest cells.
                let mut cd: Vec<(f32, usize)> = self
                    .centroids
                    .iter()
                    .enumerate()
                    .map(|(ci, c)| {
                        (
                            q.iter().zip(c).map(|(a, b)| (a - b) * (a - b)).sum::<f32>(),
                            ci,
                        )
                    })
                    .collect();
                cd.sort_by(|a, b| a.0.total_cmp(&b.0));
                // fill candidates cell-by-cell with each cell's contiguous reordered-index span (coalesced).
                let mut w = qi * max_cand;
                for &(_, ci) in cd.iter().take(nprobe) {
                    for newi in self.cell_offsets[ci]..self.cell_offsets[ci + 1] {
                        cand[w] = newi as u32;
                        w += 1;
                    }
                }
            }
            let qh = self.client.create(cubecl::bytes::Bytes::from_elems(qt));
            let ch = self
                .client
                .create(cubecl::bytes::Bytes::from_elems(cand.clone()));
            let outh = self
                .client
                .empty(b * max_cand * core::mem::size_of::<f32>());
            let threads = 256u32;
            unsafe {
                ivf_scan_sq8::launch_unchecked(
                    &self.client,
                    CubeCount::Static(((b * max_cand) as u32).div_ceil(threads), 1, 1),
                    CubeDim::new_1d(threads),
                    ArrayArg::from_raw_parts(qh, b * dim),
                    ArrayArg::from_raw_parts(self.codes_w.clone(), dim * (n / 4)),
                    ArrayArg::from_raw_parts(ch, b * max_cand),
                    ArrayArg::from_raw_parts(outh.clone(), b * max_cand),
                    n as u32,
                    b as u32,
                    dim as u32,
                    max_cand as u32,
                    self.n_real as u32,
                );
            }
            let sb = self.client.read_one(outh).unwrap();
            let scores: &[f32] = bytemuck::cast_slice(&sb);
            for qi in 0..b {
                let base = qi * max_cand;
                let mut pairs: Vec<(f32, usize)> = (0..max_cand)
                    .map(|j| (scores[base + j], cand[base + j] as usize))
                    .filter(|&(s, i)| i < self.n_real && s > -3.0e37)
                    .collect();
                pairs.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
                pairs.truncate(k);
                out_all.push(
                    pairs
                        .into_iter()
                        .map(|(score, i)| SearchResult {
                            id: self.ids[i].clone(),
                            content: String::new(),
                            score,
                            metadata: None,
                        })
                        .collect(),
                );
            }
        }
        out_all
    }

    /// Number of indexed vectors.
    pub fn len(&self) -> usize {
        self.n_real
    }
    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.n_real == 0
    }
}

/// splitmix64 — a tiny deterministic PRNG for the sign projection (the crate has no `rand` outside tests).
fn splitmix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Achlioptas sign random projection `[dim][rdim]`, entries `±1/√rdim` — a Johnson–Lindenstrauss map that
/// preserves distances up to `~1/√rdim`, deterministic (no learned PCA yet; that's the accuracy follow-up).
fn sign_projection(dim: usize, rdim: usize) -> Vec<f32> {
    let s = 1.0 / (rdim as f32).sqrt();
    let mut p = vec![0f32; dim * rdim];
    for d in 0..dim {
        for r in 0..rdim {
            p[d * rdim + r] = if splitmix(((d as u64) << 20) ^ r as u64) & 1 == 0 {
                s
            } else {
                -s
            };
        }
    }
    p
}

fn project(v: &[f32], proj: &[f32], rdim: usize) -> Vec<f32> {
    let mut out = vec![0f32; rdim];
    for (d, &vd) in v.iter().enumerate() {
        let base = d * rdim;
        for r in 0..rdim {
            out[r] += vd * proj[base + r];
        }
    }
    out
}

/// **LeanVec-style** two-stage index: coarse-scan in a **reduced dimension** (a JL projection `dim→rdim`,
/// SQ8), then rerank the top-C on the **full-dim** f32. The reduced scan does `rdim/dim` of the MACs and
/// code bytes (~3× at 768→256), and the full-dim rerank recovers the accuracy the projection lost — so it
/// can be *both* faster than a full-dim SQ8 flat scan *and* higher recall. First cut: a **sign random
/// projection** (distance-preserving but not learned; a PCA/learned projection is the accuracy follow-up).
pub struct LeanVecIndex {
    inner: GpuFlatIndex, // reduced-dim SQ8 coarse scan
    full_f32: cubecl::server::Handle,
    proj: Vec<f32>,
    dim: usize,
    rdim: usize,
    n_real: usize,
}

impl LeanVecIndex {
    /// Build: project the corpus `dim→rdim`, build a reduced-dim SQ8 flat index for the coarse stage, and
    /// upload the full-precision corpus for the rerank.
    pub fn build(vecs: &[Vec<f32>], ids: Vec<String>, rdim: usize) -> Self {
        assert!(!vecs.is_empty(), "empty corpus");
        let (n_real, dim) = (vecs.len(), vecs[0].len());
        assert!(rdim >= 1 && rdim < dim, "rdim must be in 1..dim");
        let proj = sign_projection(dim, rdim);
        let reduced: Vec<Vec<f32>> = vecs.iter().map(|v| project(v, &proj, rdim)).collect();
        let inner = GpuFlatIndex::build(&reduced, ids);
        let mut f = vec![0f32; n_real * dim];
        for (i, v) in vecs.iter().enumerate() {
            f[i * dim..i * dim + dim].copy_from_slice(v);
        }
        let full_f32 = inner.client.create(cubecl::bytes::Bytes::from_elems(f));
        Self {
            inner,
            full_f32,
            proj,
            dim,
            rdim,
            n_real,
        }
    }

    /// Nearest `k`: reduced-dim coarse scan → top-`rerank_c` → full-dim f32 rerank → top-`k`.
    pub fn search_batch(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        rerank_c: usize,
    ) -> Vec<Vec<SearchResult>> {
        let (dim, n_real) = (self.dim, self.n_real);
        let rq: Vec<Vec<f32>> = queries
            .iter()
            .map(|q| project(q, &self.proj, self.rdim))
            .collect();
        let cands = self.inner.coarse_candidates(&rq, rerank_c); // reduced-dim top-C indices
        let ids = self.inner.ids();
        let client = &self.inner.client;
        let mut out_all = Vec::with_capacity(queries.len());
        let batch = 64usize;
        for (ci, chunk) in queries.chunks(batch).enumerate() {
            let b = chunk.len();
            let mut qf = vec![0f32; b * dim];
            let mut cand = vec![n_real as u32; b * rerank_c]; // sentinel ⇒ −inf
            for (qi, q) in chunk.iter().enumerate() {
                qf[qi * dim..qi * dim + dim].copy_from_slice(q);
                let cl = &cands[ci * batch + qi];
                for (j, &c) in cl.iter().take(rerank_c).enumerate() {
                    cand[qi * rerank_c + j] = c as u32;
                }
            }
            let qh = client.create(cubecl::bytes::Bytes::from_elems(qf));
            let ch = client.create(cubecl::bytes::Bytes::from_elems(cand.clone()));
            let outh = client.empty(b * rerank_c * core::mem::size_of::<f32>());
            let threads = 256u32;
            unsafe {
                rerank_f32_dot::launch_unchecked(
                    client,
                    CubeCount::Static(((b * rerank_c) as u32).div_ceil(threads), 1, 1),
                    CubeDim::new_1d(threads),
                    ArrayArg::from_raw_parts(qh, b * dim),
                    ArrayArg::from_raw_parts(ch, b * rerank_c),
                    ArrayArg::from_raw_parts(self.full_f32.clone(), n_real * dim),
                    ArrayArg::from_raw_parts(outh.clone(), b * rerank_c),
                    b as u32,
                    dim as u32,
                    rerank_c as u32,
                    n_real as u32,
                );
            }
            let sb = client.read_one(outh).unwrap();
            let scores: &[f32] = bytemuck::cast_slice(&sb);
            for qi in 0..b {
                let base = qi * rerank_c;
                let mut pairs: Vec<(f32, usize)> = (0..rerank_c)
                    .map(|j| (scores[base + j], cand[base + j] as usize))
                    .filter(|&(s, i)| i < n_real && s > -3.0e37)
                    .collect();
                pairs.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
                pairs.truncate(k);
                out_all.push(
                    pairs
                        .into_iter()
                        .map(|(score, i)| SearchResult {
                            id: ids[i].clone(),
                            content: String::new(),
                            score,
                            metadata: None,
                        })
                        .collect(),
                );
            }
        }
        out_all
    }

    /// Number of indexed vectors.
    pub fn len(&self) -> usize {
        self.n_real
    }
    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.n_real == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    /// GPU SQ8 flat scan must recover most of the exact (f32 brute-force) top-k on real-ish data.
    /// 8-bit quantization ⇒ high-but-not-perfect recall, and a flat scan has no navigation loss.
    #[test]
    fn sq8_scan_recovers_exact_topk() {
        let (n, dim, k) = (2000usize, 256usize, 10usize);
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        // clustered data so top-k is meaningful (not a random near-tie shell)
        let centers: Vec<Vec<f32>> = (0..20)
            .map(|_| (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect())
            .collect();
        let vecs: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let c = &centers[i % centers.len()];
                let mut v: Vec<f32> = c
                    .iter()
                    .map(|&x| x + (rng.random::<f32>() - 0.5) * 0.5)
                    .collect();
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                v.iter_mut().for_each(|x| *x /= norm + 1e-9);
                v
            })
            .collect();
        let ids: Vec<String> = (0..n).map(|i| i.to_string()).collect();
        let index = GpuFlatIndex::build(&vecs, ids);

        let mut hits = 0usize;
        let nq = 50usize;
        for qi in 0..nq {
            let q = &vecs[qi * 7 % n];
            // exact f32 top-k (dot on unit vectors == cosine)
            let mut exact: Vec<(f32, usize)> = (0..n)
                .map(|i| (q.iter().zip(&vecs[i]).map(|(a, b)| a * b).sum::<f32>(), i))
                .collect();
            exact.sort_by(|a, b| b.0.total_cmp(&a.0));
            let truth: std::collections::HashSet<usize> =
                exact.iter().take(k).map(|&(_, i)| i).collect();
            let got: std::collections::HashSet<usize> = index
                .search(q, k)
                .into_iter()
                .map(|r| r.id.parse::<usize>().unwrap())
                .collect();
            hits += truth.intersection(&got).count();
        }
        let recall = hits as f64 / (nq * k) as f64;
        assert!(
            recall > 0.88,
            "SQ8 GPU flat recall {recall:.3} too low (kernel bug -> ~0.005)"
        );
    }

    /// The 1-bit sign kernel is validated against a CPU sign-dot (data-independent — recall vs f32 is a
    /// separate, distribution-dependent property). GPU and CPU must rank the same corpus the same way.
    #[test]
    fn sign1_gpu_matches_cpu_signdot() {
        let (n, dim, k) = (2000usize, 256usize, 10usize);
        let mut rng = rand::rngs::StdRng::seed_from_u64(2);
        let centers: Vec<Vec<f32>> = (0..20)
            .map(|_| (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect())
            .collect();
        let vecs: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let c = &centers[i % centers.len()];
                c.iter()
                    .map(|&x| x + (rng.random::<f32>() - 0.5) * 0.5)
                    .collect()
            })
            .collect();
        let ids: Vec<String> = (0..n).map(|i| i.to_string()).collect();
        let index = GpuFlatIndex::build_sign1(&vecs, ids);

        let mut mean = vec![0f64; dim];
        for v in &vecs {
            for d in 0..dim {
                mean[d] += v[d] as f64;
            }
        }
        let mean: Vec<f32> = mean.iter().map(|&m| (m / n as f64) as f32).collect();

        let (nq, mut agree) = (30usize, 0usize);
        for qi in 0..nq {
            let q = &vecs[qi * 7 % n];
            let mut cpu: Vec<(f32, usize)> = (0..n)
                .map(|i| {
                    let s: f32 = (0..dim)
                        .map(|d| q[d] * if vecs[i][d] >= mean[d] { 1.0 } else { -1.0 })
                        .sum();
                    (s, i)
                })
                .collect();
            cpu.sort_by(|a, b| b.0.total_cmp(&a.0));
            let truth: std::collections::HashSet<usize> =
                cpu.iter().take(k).map(|&(_, i)| i).collect();
            let got: std::collections::HashSet<usize> = index
                .search(q, k)
                .into_iter()
                .map(|r| r.id.parse::<usize>().unwrap())
                .collect();
            agree += truth.intersection(&got).count();
        }
        let a = agree as f64 / (nq * k) as f64;
        assert!(a > 0.90, "GPU sign1 != CPU sign-dot ({a:.3}) — kernel bug");
    }

    /// The GPU RaBitQ scan must rank the corpus the same way the CPU `estimate_dist_sq` does — same
    /// quantizer (deterministic default seed), same estimator, so agreement is a pure kernel check
    /// (recall vs f32 is the separate, distribution-dependent property measured in the bench).
    #[test]
    fn rabitq_gpu_matches_cpu_estimate() {
        // n = 2000 pads to 2016 (16 padded rows), and vectors are **unit-norm** like real embeddings, so
        // real RaBitQ scores can go negative — the exact condition under which a naive score-0 padding
        // sentinel would flood the top-k. The `got.len() == k` assertion below is the padding-leak guard.
        let (n, dim, k) = (2000usize, 256usize, 10usize);
        let mut rng = rand::rngs::StdRng::seed_from_u64(4);
        let centers: Vec<Vec<f32>> = (0..20)
            .map(|_| (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect())
            .collect();
        let vecs: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let c = &centers[i % centers.len()];
                let v: Vec<f32> = c
                    .iter()
                    .map(|&x| x + (rng.random::<f32>() - 0.5) * 0.5)
                    .collect();
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                v.iter().map(|x| x / (norm + 1e-9)).collect()
            })
            .collect();
        let ids: Vec<String> = (0..n).map(|i| i.to_string()).collect();
        let index = GpuFlatIndex::build_rabitq(&vecs, ids);

        // CPU reference: the same quantizer (fit is deterministic), rank by estimated dsq (smallest first).
        let quant = RaBitQuantizer::fit(&vecs);
        let codes: Vec<_> = vecs.iter().map(|v| quant.encode(v)).collect();

        let (nq, mut agree) = (30usize, 0usize);
        for qi in 0..nq {
            let q = &vecs[qi * 7 % n];
            let prep = quant.prepare_query(q);
            let mut cpu: Vec<(f32, usize)> = codes
                .iter()
                .enumerate()
                .map(|(i, c)| (quant.estimate_dist_sq(&prep, c), i))
                .collect();
            cpu.sort_by(|a, b| a.0.total_cmp(&b.0));
            let truth: std::collections::HashSet<usize> =
                cpu.iter().take(k).map(|&(_, i)| i).collect();
            let res = index.search(q, k);
            assert_eq!(
                res.len(),
                k,
                "RaBitQ returned {} < k — padded rows leaked into top-k",
                res.len()
            );
            let got: std::collections::HashSet<usize> = res
                .into_iter()
                .map(|r| r.id.parse::<usize>().unwrap())
                .collect();
            agree += truth.intersection(&got).count();
        }
        let a = agree as f64 / (nq * k) as f64;
        assert!(
            a > 0.95,
            "GPU RaBitQ != CPU estimator ({a:.3}) — kernel bug"
        );
    }

    /// GPU TurboRabit must rank the corpus the same way the CPU `estimate_dist_sq` does — same quantizer,
    /// same estimator (rotation + per-vector f_rescale + per-query cb_sum). Runs **both packings**: 4-bit
    /// (nibble, 8/u32) and 6-bit (byte, 4/u32). Unit-norm vectors + non-multiple-of-32 n stress the padding
    /// sentinel; `res.len() == k` is the leak guard.
    #[test]
    fn turborabit_gpu_matches_cpu_estimate() {
        let (n, dim, k) = (2000usize, 256usize, 10usize);
        let mut rng = rand::rngs::StdRng::seed_from_u64(5);
        let centers: Vec<Vec<f32>> = (0..20)
            .map(|_| (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect())
            .collect();
        let vecs: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let c = &centers[i % centers.len()];
                let v: Vec<f32> = c
                    .iter()
                    .map(|&x| x + (rng.random::<f32>() - 0.5) * 0.5)
                    .collect();
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                v.iter().map(|x| x / (norm + 1e-9)).collect()
            })
            .collect();
        let ids: Vec<String> = (0..n).map(|i| i.to_string()).collect();

        for bits in [4usize, 6usize] {
            let index = GpuFlatIndex::build_turborabit(&vecs, ids.clone(), bits);
            let quant = TurboRabitQuantizer::fit(&vecs, bits);
            let codes: Vec<_> = vecs.iter().map(|v| quant.encode(v)).collect();

            let (nq, mut agree) = (30usize, 0usize);
            for qi in 0..nq {
                let q = &vecs[qi * 7 % n];
                let prep = quant.prepare_query(q);
                let mut cpu: Vec<(f32, usize)> = codes
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (quant.estimate_dist_sq(&prep, c), i))
                    .collect();
                cpu.sort_by(|a, b| a.0.total_cmp(&b.0));
                let truth: std::collections::HashSet<usize> =
                    cpu.iter().take(k).map(|&(_, i)| i).collect();
                let res = index.search(q, k);
                assert_eq!(
                    res.len(),
                    k,
                    "TurboRabit-{bits} returned {} < k — padded rows leaked",
                    res.len()
                );
                let got: std::collections::HashSet<usize> = res
                    .into_iter()
                    .map(|r| r.id.parse::<usize>().unwrap())
                    .collect();
                agree += truth.intersection(&got).count();
            }
            let a = agree as f64 / (nq * k) as f64;
            assert!(
                a > 0.95,
                "GPU TurboRabit-{bits} != CPU estimator ({a:.3}) — kernel bug"
            );
        }
    }

    /// Two-stage rerank (1c): exact f32 rerank of the coarse top-C must recover recall — reranked ≥ plain,
    /// and near-exact since the pool is rescored exactly. Uses TR4 (the coarse mode 1c targets).
    #[test]
    fn rerank_recovers_recall() {
        let (n, dim, k, c) = (3000usize, 128usize, 10usize, 100usize);
        let mut rng = rand::rngs::StdRng::seed_from_u64(6);
        let centers: Vec<Vec<f32>> = (0..30)
            .map(|_| (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect())
            .collect();
        let vecs: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let ce = &centers[i % centers.len()];
                let v: Vec<f32> = ce
                    .iter()
                    .map(|&x| x + (rng.random::<f32>() - 0.5) * 0.8)
                    .collect();
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                v.iter().map(|x| x / (norm + 1e-9)).collect()
            })
            .collect();
        let ids: Vec<String> = (0..n).map(|i| i.to_string()).collect();
        let index = GpuFlatIndex::build_turborabit(&vecs, ids, 4);
        let queries: Vec<Vec<f32>> = (0..40).map(|qi| vecs[qi * 13 % n].clone()).collect();

        let recall = |res: &[Vec<SearchResult>]| -> f64 {
            let mut hits = 0usize;
            for (qi, q) in queries.iter().enumerate() {
                let mut exact: Vec<(f32, usize)> = (0..n)
                    .map(|i| (q.iter().zip(&vecs[i]).map(|(a, b)| a * b).sum::<f32>(), i))
                    .collect();
                exact.sort_by(|a, b| b.0.total_cmp(&a.0));
                let truth: std::collections::HashSet<usize> =
                    exact.iter().take(k).map(|&(_, i)| i).collect();
                let got: std::collections::HashSet<usize> = res[qi]
                    .iter()
                    .map(|r| r.id.parse::<usize>().unwrap())
                    .collect();
                hits += truth.intersection(&got).count();
            }
            hits as f64 / (queries.len() * k) as f64
        };

        let plain = recall(&index.search_batch(&queries, k));
        let reranked = recall(&index.search_batch_reranked(&queries, k, c, &vecs));
        assert!(
            reranked >= plain && reranked > 0.95,
            "rerank should recover recall: plain {plain:.3} -> reranked {reranked:.3}"
        );

        // GPU f32 rerank must reproduce the CPU f32 rerank top-k per query.
        let ids2: Vec<String> = (0..n).map(|i| i.to_string()).collect();
        let gpu_idx = GpuFlatIndex::build_turborabit(&vecs, ids2, 4).with_gpu_rerank(&vecs);
        let gpu = gpu_idx.search_batch_reranked_gpu(&queries, k, c);
        let cpu = gpu_idx.search_batch_reranked(&queries, k, c, &vecs);
        for (g, cp) in gpu.iter().zip(&cpu) {
            let gs: std::collections::HashSet<&str> = g.iter().map(|r| r.id.as_str()).collect();
            let cs: std::collections::HashSet<&str> = cp.iter().map(|r| r.id.as_str()).collect();
            assert_eq!(gs, cs, "GPU f32 rerank != CPU f32 rerank — kernel bug");
        }
    }

    /// Warren's 8+8-bit residual rerank (no f32) must recover recall like the f32 rerank does — it tracks
    /// the exact dot because the reconstruction approaches R·(x−c) and R is orthonormal.
    #[test]
    fn warren_residual_reranks_without_f32() {
        let (n, dim, k, c) = (3000usize, 128usize, 10usize, 100usize);
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let centers: Vec<Vec<f32>> = (0..30)
            .map(|_| (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect())
            .collect();
        let vecs: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let ce = &centers[i % centers.len()];
                let v: Vec<f32> = ce
                    .iter()
                    .map(|&x| x + (rng.random::<f32>() - 0.5) * 0.8)
                    .collect();
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                v.iter().map(|x| x / (norm + 1e-9)).collect()
            })
            .collect();
        let ids: Vec<String> = (0..n).map(|i| i.to_string()).collect();
        let plain_idx = GpuFlatIndex::build_turborabit(&vecs, ids.clone(), 4);
        let warren = WarrenRerankIndex::build(&vecs, ids, 4);
        let queries: Vec<Vec<f32>> = (0..40).map(|qi| vecs[qi * 13 % n].clone()).collect();

        let recall = |res: &[Vec<SearchResult>]| -> f64 {
            let mut hits = 0usize;
            for (qi, q) in queries.iter().enumerate() {
                let mut exact: Vec<(f32, usize)> = (0..n)
                    .map(|i| (q.iter().zip(&vecs[i]).map(|(a, b)| a * b).sum::<f32>(), i))
                    .collect();
                exact.sort_by(|a, b| b.0.total_cmp(&a.0));
                let truth: std::collections::HashSet<usize> =
                    exact.iter().take(k).map(|&(_, i)| i).collect();
                let got: std::collections::HashSet<usize> = res[qi]
                    .iter()
                    .map(|r| r.id.parse::<usize>().unwrap())
                    .collect();
                hits += truth.intersection(&got).count();
            }
            hits as f64 / (queries.len() * k) as f64
        };

        let plain = recall(&plain_idx.search_batch(&queries, k));
        let f32rr = recall(&plain_idx.search_batch_reranked(&queries, k, c, &vecs));
        let warrenrr = recall(&warren.search_batch(&queries, k, c));
        // Warren recovers most of what the f32 rerank does, well above the un-reranked coarse recall.
        assert!(
            warrenrr > plain && warrenrr >= f32rr - 0.05 && warrenrr > 0.9,
            "warren rerank: plain {plain:.3}, f32 {f32rr:.3}, warren {warrenrr:.3}"
        );

        // IVF: probing more cells raises recall toward exact; enough probes ⇒ near-exact.
        let ivf = GpuIvfIndex::build(&vecs, (0..n).map(|i| i.to_string()).collect(), 32, 6);
        let recall_ivf = |nprobe: usize| -> f64 {
            let res = ivf.search_batch(&queries, k, nprobe);
            let mut hits = 0usize;
            for (qi, q) in queries.iter().enumerate() {
                let mut ex: Vec<(f32, usize)> = (0..n)
                    .map(|i| (q.iter().zip(&vecs[i]).map(|(a, b)| a * b).sum::<f32>(), i))
                    .collect();
                ex.sort_by(|a, b| b.0.total_cmp(&a.0));
                let truth: std::collections::HashSet<usize> =
                    ex.iter().take(k).map(|&(_, i)| i).collect();
                let got: std::collections::HashSet<usize> = res[qi]
                    .iter()
                    .map(|r| r.id.parse::<usize>().unwrap())
                    .collect();
                hits += truth.intersection(&got).count();
            }
            hits as f64 / (queries.len() * k) as f64
        };
        let lo = recall_ivf(1);
        let hi = recall_ivf(16);
        assert!(
            hi >= lo && hi > 0.9,
            "IVF recall should rise with nprobe: p1 {lo:.3} -> p16 {hi:.3}"
        );

        // LeanVec: reduced-dim coarse scan + full-dim rerank should recover high recall.
        let lv = LeanVecIndex::build(&vecs, (0..n).map(|i| i.to_string()).collect(), dim / 2);
        let lvr = lv.search_batch(&queries, k, 64);
        let mut hits = 0usize;
        for (qi, q) in queries.iter().enumerate() {
            let mut ex: Vec<(f32, usize)> = (0..n)
                .map(|i| (q.iter().zip(&vecs[i]).map(|(a, b)| a * b).sum::<f32>(), i))
                .collect();
            ex.sort_by(|a, b| b.0.total_cmp(&a.0));
            let truth: std::collections::HashSet<usize> =
                ex.iter().take(k).map(|&(_, i)| i).collect();
            let got: std::collections::HashSet<usize> = lvr[qi]
                .iter()
                .map(|r| r.id.parse::<usize>().unwrap())
                .collect();
            hits += truth.intersection(&got).count();
        }
        let lv_recall = hits as f64 / (queries.len() * k) as f64;
        // Sign random-projection to dim/2 is lossy (a learned/PCA projection would recover more); the
        // rerank still lifts it well above chance. Real-data behaviour is measured in examples/leanvec.rs.
        assert!(
            lv_recall > 0.75,
            "LeanVec rerank recall too low: {lv_recall:.3}"
        );

        // The GPU rerank kernel must reproduce the CPU rerank (the tested oracle) top-k per query.
        let gpu = warren.search_batch_gpu(&queries, k, c);
        let cpu = warren.search_batch(&queries, k, c);
        for (g, cp) in gpu.iter().zip(&cpu) {
            let gs: std::collections::HashSet<&str> = g.iter().map(|r| r.id.as_str()).collect();
            let cs: std::collections::HashSet<&str> = cp.iter().map(|r| r.id.as_str()).collect();
            assert_eq!(
                gs, cs,
                "GPU Warren rerank != CPU Warren rerank — kernel bug"
            );
        }
    }
}
