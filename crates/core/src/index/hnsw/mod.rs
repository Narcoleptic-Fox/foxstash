//! HNSW (Hierarchical Navigable Small World) index implementation
//!
//! Based on the paper "Efficient and robust approximate nearest neighbor search using
//! Hierarchical Navigable Small World graphs" by Malkov & Yashunin (2018).
//!
//! The HNSW algorithm creates a multi-layer graph structure where:
//! - Layer 0 contains all elements
//! - Higher layers contain exponentially fewer elements
//! - Each element has connections to its nearest neighbors at each layer
//! - Search starts at the top layer and zooms in to find nearest neighbors

use crate::{Document, Result, SearchResult};
use parking_lot::{Mutex, RwLock};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;
use std::cmp::{max, Reverse};
use std::collections::{BinaryHeap, HashSet};

/// Cross-platform cache line prefetch (read hint).
/// x86_64: _mm_prefetch T0, aarch64: PLDL1KEEP, other: no-op.
#[inline(always)]
unsafe fn prefetch_read(ptr: *const u8) {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(ptr as *const i8);
    }
    #[cfg(target_arch = "aarch64")]
    {
        // Use inline asm instead of std::arch::aarch64::_prefetch which is unstable
        std::arch::asm!("prfm pldl1keep, [{ptr}]", ptr = in(reg) ptr, options(nostack, preserves_flags));
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = ptr;
    }
}

/// Prefetch multiple cache lines starting at `ptr` (64 bytes per line).
#[inline(always)]
unsafe fn prefetch_embedding(ptr: *const u8, cache_lines: usize) {
    for i in 0..cache_lines {
        prefetch_read(ptr.add(i * 64));
    }
}

/// Wrapper for f32 that implements Ord for use in BinaryHeap
#[derive(Debug, Clone, Copy, PartialEq)]
struct OrderedFloat(f32);

impl Eq for OrderedFloat {}

impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// Packed bitset for visited node tracking.
///
/// 1 bit per node = 12.5 KB for 100K nodes (fits L1 cache).
/// Replaces the previous generation-based Vec<u64> (800 KB, spills to L2/L3).
struct BitsetVisited {
    bits: Vec<u64>,
}

impl BitsetVisited {
    fn new(n: usize) -> Self {
        Self {
            bits: vec![0u64; (n + 63) / 64],
        }
    }

    #[inline(always)]
    fn is_visited(&self, node: usize) -> bool {
        debug_assert!(
            (node >> 6) < self.bits.len(),
            "BitsetVisited::is_visited: node {} out of bounds (capacity {})",
            node,
            self.bits.len() * 64
        );
        let word = unsafe { *self.bits.get_unchecked(node >> 6) };
        word & (1u64 << (node & 63)) != 0
    }

    #[inline(always)]
    fn mark_visited(&mut self, node: usize) {
        debug_assert!(
            (node >> 6) < self.bits.len(),
            "BitsetVisited::mark_visited: node {} out of bounds (capacity {})",
            node,
            self.bits.len() * 64
        );
        unsafe {
            *self.bits.get_unchecked_mut(node >> 6) |= 1u64 << (node & 63);
        }
    }

    #[inline]
    fn clear(&mut self) {
        self.bits.fill(0);
    }
}

/// An immutable allow-list over internal node slots, for filtered search.
///
/// One bit per node — `contains(slot)` is a single word load, cheap enough to call on every
/// candidate in the layer-0 walk. Build it **once** with [`HNSWIndex::filter_mask`] or
/// [`HNSWIndex::filter_mask_ids`] and reuse it across many queries: the mask is what keeps
/// filtered search sub-linear. Rebuilding it per query (an O(n) scan of every document's
/// metadata) would erase the graph's advantage and make a flat scan the better choice — which
/// is precisely the flat-index niche this feature exists to *not* cede.
///
/// The mask is tied to the node numbering of the index that produced it. Slots shift on
/// [`HNSWIndex::clear`] or a rebuild, so a mask outlives its index only until the next
/// structural change; treat it as derived state, not a durable handle.
#[derive(Clone, Debug)]
pub struct FilterMask {
    bits: Vec<u64>,
    allowed: usize,
}

impl FilterMask {
    /// Every slot in `0..n` denied. Callers flip the ones they want with [`FilterMask::allow`].
    fn empty(n: usize) -> Self {
        Self {
            bits: vec![0u64; (n + 63) / 64],
            allowed: 0,
        }
    }

    #[inline]
    fn allow(&mut self, node: usize) {
        let w = node >> 6;
        let bit = 1u64 << (node & 63);
        if self.bits[w] & bit == 0 {
            self.bits[w] |= bit;
            self.allowed += 1;
        }
    }

    /// Is `node` in the allowed set? Out-of-range slots read as denied.
    #[inline(always)]
    pub fn contains(&self, node: usize) -> bool {
        match self.bits.get(node >> 6) {
            Some(word) => word & (1u64 << (node & 63)) != 0,
            None => false,
        }
    }

    /// How many nodes the mask admits. A search cannot return more than this many results,
    /// regardless of `k` — and a very small count means the walk may traverse most of the graph
    /// to collect them (see [`HNSWIndex::search_filtered`]).
    #[inline]
    pub fn allowed_count(&self) -> usize {
        self.allowed
    }
}

/// Per-query state computed once and threaded through [`HNSWIndex::search_layer`] and
/// [`HNSWIndex::distance_to_node`] for the whole search (or the whole `insert_node` call).
///
/// `norm` is for cosine's fused distance. `rabitq` is `Some` only under [`Storage::RaBitQ`]:
/// it is the query rotated into RaBitQ space, which costs an O(dim²) matrix-vector multiply
/// and must be paid exactly once per query — not once per node visited, which is what
/// recomputing it inside `distance_to_node` would do. Bundled into one struct (rather than two
/// more parameters on `search_layer`) because that function was already at clippy's
/// `too_many_arguments` ceiling; a struct scales if a future storage mode needs its own
/// per-query state without every call site growing another positional argument.
struct QueryPrep<'a> {
    norm: f32,
    rabitq: Option<&'a crate::vector::rabitq::PreparedQuery>,
    turboquant: Option<&'a crate::vector::turboquant::PreparedQuery>,
    turborabit: Option<&'a crate::vector::turborabit::PreparedQuery>,
    /// Filtered search's admit gate: `Some(f)` ⇒ a node enters the result heap only if `f(node_id)`.
    /// `Some` only from [`HNSWIndex::search_filtered`] (a prebuilt [`FilterMask`], `|id| mask.contains(id)`)
    /// and [`HNSWIndex::search_filtered_by`] (a caller predicate over the node's id/metadata). Unlike
    /// the other fields this is *not* distance-computation prep — it gates which nodes are *returned*,
    /// and `search_layer` applies it **only at layer 0** (upper-layer descent must navigate freely
    /// through excluded nodes or the walk disconnects). A single `dyn Fn(usize) -> bool` unifies the
    /// mask and predicate paths onto one gating mechanism. It rides in `QueryPrep` because
    /// `search_layer` is already at clippy's argument ceiling and this struct is the documented place
    /// to add per-query state without growing every call site.
    filter: Option<&'a dyn Fn(usize) -> bool>,
}

/// Per-query scratch space: the visited bitset and the two heaps.
///
/// Private. Reusing it is [`Searcher`]'s job, and it is *not* a speed feature — see
/// [`Searcher`] for the measurement.
struct SearchContext {
    /// Packed bitset for visited tracking (fits L1 cache)
    visited: BitsetVisited,
    /// Number of nodes this context supports
    capacity: usize,
    /// Reusable min-heap for candidates
    candidates: BinaryHeap<Reverse<(OrderedFloat, usize)>>,
    /// Reusable max-heap for best results
    best: BinaryHeap<(OrderedFloat, usize)>,
    /// Distance computations performed, cumulative until [`Self::reset_stats`].
    ///
    /// The unit of work an HNSW search is made of. Comparing *this* between two
    /// implementations at matched recall separates the two ways one can be slower: doing
    /// more work (a worse graph, or a search that stops too late) from doing the same work
    /// more slowly (a worse inner loop, or worse latency hiding). Without it you are
    /// guessing, and guessing is how this project shipped three false performance claims.
    ///
    /// faiss exposes the same counter as `hnsw_stats.ndis`; hnswlib's Python bindings do
    /// not expose theirs.
    distance_calls: u64,
}

impl SearchContext {
    fn new(n: usize) -> Self {
        Self {
            visited: BitsetVisited::new(n),
            capacity: n,
            candidates: BinaryHeap::with_capacity(256),
            best: BinaryHeap::with_capacity(256),
            distance_calls: 0,
        }
    }

    /// Reset for a new search — clears bitset (~12.5 KB memset for 100K nodes)
    #[inline]
    fn reset(&mut self) {
        self.visited.clear();
        self.candidates.clear();
        self.best.clear();
    }

    /// Check if node was visited in current search
    #[inline(always)]
    fn is_visited(&self, node: usize) -> bool {
        self.visited.is_visited(node)
    }

    /// Mark node as visited
    #[inline(always)]
    fn mark_visited(&mut self, node: usize) {
        self.visited.mark_visited(node);
    }
}

/// A cursor over an [`HNSWIndex`] that keeps its scratch space between queries and counts
/// the distance computations it performs.
///
/// # This is not a speed feature
///
/// It replaces a `search_with_context` / `create_search_context` pair whose docs claimed
/// "~2-3x faster than `search()`". That claim was never true. Measured on SIFT1M — the size
/// at which the O(n) visited bitset should hurt most:
///
/// | | QPS | speedup |
/// |---|---|---|
/// | `search()` | 4,118 | — |
/// | reused context | 4,121 | **1.00x** |
///
/// `search()` allocates a fresh bitset per call: 125 KB at 1M nodes. But it is the *same*
/// size every time, so it comes straight back off the allocator's free list, and zeroing it
/// costs a couple of microseconds against a query that spends ~234 µs stalled on DRAM for
/// its ~3,500 distance computations. The search is memory-latency bound, not allocation
/// bound — the same fact that makes [`Storage::SQ8`] a win. Reuse the context all you like;
/// there is nothing there to win. (`cargo run --release -p foxstash-benches --example
/// search_api_cost`.)
///
/// # What it is for
///
/// [`Searcher::distance_calls`] — the unit of work an HNSW search is made of. Comparing it
/// between implementations at matched recall separates *doing more work* (a worse graph, or
/// a search that stops too late) from *doing the same work more slowly* (a worse inner loop,
/// or worse latency hiding). Those have completely different fixes and QPS alone cannot tell
/// them apart. faiss exposes the same counter as `hnsw_stats.ndis`.
///
/// ```
/// # use foxstash_core::index::hnsw::{HNSWIndex, HNSWConfig};
/// # let index = HNSWIndex::build(vec![vec![1.0, 0.0], vec![0.0, 1.0]], HNSWConfig::default());
/// let mut searcher = index.searcher();
/// for query in [[1.0, 0.0], [0.0, 1.0]] {
///     searcher.search(&query, 1)?;
/// }
/// println!("{} distance computations", searcher.distance_calls());
/// # Ok::<(), foxstash_core::RagError>(())
/// ```
pub struct Searcher<'a> {
    index: &'a HNSWIndex,
    ctx: SearchContext,
}

impl Searcher<'_> {
    /// Search for the `k` nearest neighbours of `query`, reusing this searcher's scratch.
    ///
    /// # Errors
    /// [`RagError::DimensionMismatch`](crate::RagError::DimensionMismatch) if `query` is not
    /// the index's dimension.
    pub fn search(&mut self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        self.index.search_inner(query, k, &mut self.ctx, None)
    }

    /// Distance computations performed since the last [`Self::reset_stats`].
    pub fn distance_calls(&self) -> u64 {
        self.ctx.distance_calls
    }

    /// Zero the distance counter. Searching does not reset it — the count is meant to
    /// accumulate across a whole query set.
    pub fn reset_stats(&mut self) {
        self.ctx.distance_calls = 0;
    }
}

/// HNSW index for efficient similarity search.
///
/// # Memory layout
///
/// Everything touched while traversing the graph lives in one interleaved arena,
/// [`Self::nodes`], with a node's neighbours and its vector in the *same* contiguous
/// block. Visiting a node is therefore a single random memory read.
///
/// This used to be Struct-of-Arrays — the vector in `embeddings`, the neighbours in
/// `connections_l0`, the norm in `norms`, three separate allocations — on the theory that
/// SoA gives "better cache locality". It does, for a linear scan. Graph traversal never
/// scans linearly: it jumps to an arbitrary node, reads its neighbour list, then jumps to
/// each neighbour's vector. Under that access pattern SoA costs *three* independent random
/// DRAM reads per visit where an interleaved block costs one.
///
/// The difference is invisible while the index fits in L3 and decisive once it does not,
/// which is exactly what the benchmarks showed: foxstash beat hnswlib on SIFT10K (9 MB
/// index, cache-resident) and lost to it by ~20% on SIFT1M (940 MB). hnswlib and faiss have
/// always interleaved. See `benchmarks/RESULTS.md`.
pub struct HNSWIndex {
    /// Dimensionality of embeddings
    embedding_dim: usize,
    /// Configuration parameters
    config: HNSWConfig,

    // === HOT PATH ===
    /// Interleaved node arena. Node `i` occupies `nodes[i * stride .. (i + 1) * stride]`,
    /// where `stride = node_stride(m0, dim)`, laid out as:
    ///
    /// ```text
    ///   [0]                  layer-0 neighbour count
    ///   [1]                  L2 norm of the vector (f32 bits; cosine reads it per distance)
    ///   [2 ..= 2 + m0]       layer-0 neighbour ids (m0 + 1 slots, one spare for pruning)
    ///   [.. hdr]             padding to a 16-byte boundary
    ///   [hdr .. hdr + dim]   the vector (f32 bits)
    /// ```
    ///
    /// Stored as `u32` because the block mixes ids, a count and floats; the float regions
    /// are read back with `bytemuck` casts, which are free and require no `unsafe` here.
    nodes: Vec<u32>,

    /// `node_stride(m0, dim)`, cached. Derived from config, but `get_embedding` and
    /// `get_neighbors_l0` run millions of times per query and should not recompute it.
    stride: usize,
    /// `node_hdr_len(m0)`, cached. Same reason.
    hdr: usize,

    // === SQ8 storage (empty under Storage::F32) ===
    /// Per-dimension quantization offset: `value ~= min[d] + code * scale[d]`.
    q_min: Vec<f32>,
    /// Per-dimension quantization step.
    ///
    /// Per-dimension, not global. A single shared scale would let a near-constant dimension's
    /// full 0-255 code swing carry the same weight as a high-variance dimension's — the exact
    /// bug that cost `SQ8HNSWIndex` 28 points of recall (see commit 1df91b6).
    q_scale: Vec<f32>,
    /// Full-precision vectors, kept out of the hot arena.
    ///
    /// Read only when rescoring the final candidate pool (`O(rerank_candidates)` per query),
    /// never during the walk (`O(nodes visited)`). Keeping them here rather than in the node
    /// block is the entire point: the walk must not pay for bytes it does not use.
    full: Vec<f32>,

    // === RaBitQ storage (empty under other Storage variants) ===
    /// The fitted quantizer: corpus centroid + shared random rotation. `encode`s each vector
    /// at build time and `prepare_query`s each query once per search
    /// (not once per node visit — see [`Self::distance_to_node`]).
    rabitq: Option<crate::vector::rabitq::RaBitQuantizer>,

    // === TurboQuant storage (None unless Storage::TurboQuant) ===
    /// Data-oblivious multi-bit quantizer (rotation + Gaussian sketch + derived codebook).
    /// Codes are arena-packed per node as `[gamma][qjl bits][mse nibbles]`.
    turboquant: Option<crate::vector::turboquant::TurboQuantizer>,

    // === TurboRabit storage (None unless Storage::TurboRabit) ===
    /// Extended-RaBitQ B-bit quantizer (centroid + rotation, fitted like RaBitQ's).
    /// Codes are arena-packed per node as `[dtc_sq][f_rescale][B bit-planes]`.
    turborabit: Option<crate::vector::turborabit::TurboRabitQuantizer>,

    // === Warren storage (empty unless Storage::Warren) ===
    /// Per-node 8-bit residual against TurboRabit's reconstruction, in ROTATED space, flat-packed
    /// as `[min f32][step f32][dim bytes]`. Cold — read only during rerank, never in the walk.
    warren_res: Vec<u8>,
    /// `R·c`, precomputed once: rerank needs `R·q = rq + R·c` and `rq` is per-query.
    warren_rc: Vec<f32>,

    // === GRAPH STRUCTURE (layers >= 1 only) ===
    /// Connections above layer 0: `connections[node_id][layer]` → neighbours.
    ///
    /// Layer 0 is **not** here — it lives in [`Self::nodes`], which is its sole owner.
    /// Upper layers hold ~1/M of the links and are touched a handful of times per query,
    /// so the pointer chasing costs nothing measurable.
    connections: Vec<Vec<Vec<u32>>>,

    // === COLD PATH (only accessed when returning results) ===
    /// Document IDs
    ids: Vec<String>,
    /// Document content
    contents: Vec<String>,
    /// Document metadata
    metadata: Vec<Option<serde_json::Value>>,

    /// Entry point (node index with highest layer)
    entry_point: Option<usize>,
    /// Maximum layer in the index
    max_layer: usize,

    /// The level-assignment stream for incremental [`Self::add`].
    ///
    /// Seeded once, from `config.seed`, and then advanced. It exists because `random_level` used
    /// to call `rand::rng()` per insert -- fresh OS entropy, with `self.config.seed` sitting right
    /// there unread. Bulk builds seed their own RNG, so they were reproducible (under
    /// `Sequential`) while an index grown by `add()` never was, at any seed. The bug survived
    /// because its test, `random_level_never_panics`, asserted only that the call did not panic.
    level_rng: StdRng,
}

/// **The** neighbour-selection algorithm. HNSW paper, Algorithm 4 (SELECT-NEIGHBORS-HEURISTIC).
///
/// There used to be two of these — `select_neighbors` for the sequential builder and
/// `par_select_heuristic` for the parallel one — and that duplication *is* the bug class this
/// library spent its 1.0 audit paying for. Every option had to be plumbed into both. Every test
/// was written against one. `Parallel` then became the default, so the tested copy stopped being
/// the shipped copy, and `m`, `m0`, `use_heuristic` and `extend_candidates` were each silently
/// ignored by the builder that actually ran.
///
/// The second copy also got the distances wrong. Under `extend_candidates` it scored each newly
/// pulled-in candidate against *the candidate it came from* rather than against the query — the
/// query point was never passed in, so it reached for the nearest vector to hand. The sequential
/// copy had always been right. A bug can only hide in the difference between two implementations
/// of one idea, so now there is one.
///
/// The graph structure is injected, because that is the only thing the two builders genuinely
/// disagree about: the sequential builder reads `self`, and the parallel one reads a slice of
/// `RwLock<ZeroNode>`. The *algorithm* is not theirs to have opinions about.
///
/// `candidates` must be sorted nearest-first, and every distance in it — and every distance the
/// closures return — is a distance **to the query**.
#[allow(clippy::too_many_arguments)]
fn select_neighbors_core(
    candidates: &[(f32, usize)],
    m: usize,
    use_heuristic: bool,
    extend_candidates: bool,
    keep_pruned: bool,
    neighbours_of: impl Fn(usize) -> Vec<usize>,
    dist_to_query: impl Fn(usize) -> f32,
    dist_between: impl Fn(usize, usize) -> f32,
) -> Vec<(f32, usize)> {
    // No diversity filter: keep the m nearest. `select_neighbors_simple`, in the old world.
    if !use_heuristic {
        return candidates.iter().take(m).copied().collect();
    }

    // `extendCandidates`: widen the pool with the candidates' own neighbours before filtering.
    let mut pool: Vec<(f32, usize)> = candidates.to_vec();
    if extend_candidates {
        let mut seen: HashSet<usize> = candidates.iter().map(|&(_, id)| id).collect();
        let mut extra: Vec<(f32, usize)> = Vec::new();
        for &(_, cid) in candidates {
            for n in neighbours_of(cid) {
                if seen.insert(n) {
                    extra.push((dist_to_query(n), n));
                }
            }
        }
        pool.extend(extra);
        pool.sort_by(|a, b| a.0.total_cmp(&b.0));
    }

    // Accept a candidate unless it lies closer to an already-accepted neighbour than to the query
    // — i.e. it sits "behind" one, and adds reach we already have.
    let mut selected: Vec<(f32, usize)> = Vec::with_capacity(m);
    let mut pruned: Vec<(f32, usize)> = Vec::new();
    for &(dq, cid) in &pool {
        if selected.len() >= m {
            break;
        }
        if selected
            .iter()
            .all(|&(_, sid)| dist_between(cid, sid) >= dq)
        {
            selected.push((dq, cid));
        } else {
            pruned.push((dq, cid));
        }
    }

    // Backfill from the rejects, nearest-first, if the filter left us short.
    if keep_pruned && selected.len() < m {
        for c in pruned {
            if selected.len() >= m {
                break;
            }
            selected.push(c);
        }
    }
    selected
}

impl HNSWIndex {
    /// Creates a new HNSW index with custom configuration
    ///
    /// # Arguments
    /// * `embedding_dim` - Dimensionality of embedding vectors
    /// * `config` - HNSW configuration parameters
    pub fn new(embedding_dim: usize, config: HNSWConfig) -> Self {
        let level_rng = StdRng::seed_from_u64(config.seed.unwrap_or_else(rand::random));
        Self {
            level_rng,
            embedding_dim,
            stride: node_stride(
                config.m0,
                embedding_dim,
                config.storage,
                config.quant_bits(),
            ),
            hdr: node_hdr_len(config.m0),
            config,
            nodes: Vec::new(),
            connections: Vec::new(),
            q_min: Vec::new(),
            q_scale: Vec::new(),
            full: Vec::new(),
            rabitq: None,
            turboquant: None,
            turborabit: None,
            warren_res: Vec::new(),
            warren_rc: Vec::new(),
            ids: Vec::new(),
            contents: Vec::new(),
            metadata: Vec::new(),
            entry_point: None,
            max_layer: 0,
        }
    }

    /// Fit whatever codebook the configured storage needs, over the corpus. No-op under
    /// `Storage::F32`.
    ///
    /// `SQ8`: each dimension gets its own `min`/`scale` from its own observed range. A
    /// dimension with no spread gets `scale = 0`, which dequantizes to a constant — correct,
    /// and it contributes nothing to any distance, which is exactly right for a dimension
    /// carrying no information.
    ///
    /// `RaBitQ`: fits [`RaBitQuantizer`](crate::vector::rabitq::RaBitQuantizer) — a corpus
    /// centroid plus a shared random rotation — used by [`Self::push_node`] to encode each
    /// vector and by [`Self::distance_to_node`] to prepare each query.
    #[allow(deprecated)] // internal handling of Storage::TurboQuant until its 0.8 removal
    fn fit_codebook(&mut self, embeddings: &[Vec<f32>]) {
        if embeddings.is_empty() {
            return;
        }
        match self.config.storage {
            Storage::F32 => {}
            Storage::SQ8 => {
                let dim = self.embedding_dim;
                let mut lo = vec![f32::INFINITY; dim];
                let mut hi = vec![f32::NEG_INFINITY; dim];
                for v in embeddings {
                    for d in 0..dim {
                        lo[d] = lo[d].min(v[d]);
                        hi[d] = hi[d].max(v[d]);
                    }
                }
                self.q_scale = (0..dim).map(|d| (hi[d] - lo[d]) / 255.0).collect();
                self.q_min = lo;
            }
            Storage::RaBitQ => {
                if self.config.metric == DistanceMetric::Cosine {
                    // See `rabitq_cosine_input`: fit the quantizer on unit-normalized
                    // vectors so its L2 estimator becomes an exact affine function of
                    // cosine distance. A one-time O(n) copy at fit time, not a hot path.
                    let normalized: Vec<Vec<f32>> = embeddings
                        .iter()
                        .map(|v| {
                            let mut n = v.clone();
                            crate::vector::ops::normalize(&mut n);
                            n
                        })
                        .collect();
                    self.rabitq = Some(crate::vector::rabitq::RaBitQuantizer::fit(&normalized));
                } else {
                    self.rabitq = Some(crate::vector::rabitq::RaBitQuantizer::fit(embeddings));
                }
            }
            // Data-oblivious: needs only `dim` and the bit budget, never the data itself.
            Storage::TurboQuant => {
                // Hard bound, not a clamp: the arena layout (`vec_words`) reads the raw
                // config value, so silently coercing it here would make the quantizer and
                // the layout disagree about the block size. ≤ 4 total bits = ≤ 3 MSE bits,
                // the most one nibble + the 8-entry `vpermd` LUT kernel can dequantize.
                assert!(
                    (1..=4).contains(&self.config.turbo_bits),
                    "turbo_bits must be in 1..=4 (got {}): the packed MSE kernel dequantizes \
                     through an 8-entry LUT",
                    self.config.turbo_bits
                );
                self.turboquant = Some(crate::vector::turboquant::TurboQuantizer::new(
                    self.embedding_dim,
                    self.config.turbo_bits,
                ));
            }
            // Fitted exactly like RaBitQ (centroid + rotation) — including the cosine
            // unit-normalization trick, which works for the same reason: the estimator
            // computes squared L2, and on unit vectors that is 2·cosine_distance exactly.
            Storage::TurboRabit => {
                // Same hard-bound rationale as TurboQuant above. ≤ 4 because codes are
                // nibble-packed for the fused kernel — and because b=4 already reaches
                // F32 recall (coco 0.9998), so 5..8 would be a second layout for a range
                // with no measured use case (an untested public option = a shipped bug).
                assert!(
                    (1..=4).contains(&self.config.rabit_bits),
                    "rabit_bits must be in 1..=4, got {}: codes are nibble-packed, and b=4 \
                     already reaches F32 recall",
                    self.config.rabit_bits
                );
                let bits = self.config.rabit_bits;
                if self.config.metric == DistanceMetric::Cosine {
                    let normalized: Vec<Vec<f32>> = embeddings
                        .iter()
                        .map(|v| {
                            let mut n = v.clone();
                            crate::vector::ops::normalize(&mut n);
                            n
                        })
                        .collect();
                    self.turborabit = Some(crate::vector::turborabit::TurboRabitQuantizer::fit(
                        &normalized,
                        bits,
                    ));
                } else {
                    self.turborabit = Some(crate::vector::turborabit::TurboRabitQuantizer::fit(
                        embeddings, bits,
                    ));
                }
            }
            // Same fit as TurboRabit (the walk is TurboRabit's), plus the constant `R·c` the
            // rotated-space rerank needs.
            Storage::Warren => {
                assert!(
                    (1..=4).contains(&self.config.rabit_bits),
                    "rabit_bits must be in 1..=4, got {}",
                    self.config.rabit_bits
                );
                let bits = self.config.rabit_bits;
                let q = if self.config.metric == DistanceMetric::Cosine {
                    let normalized: Vec<Vec<f32>> = embeddings
                        .iter()
                        .map(|v| {
                            let mut n = v.clone();
                            crate::vector::ops::normalize(&mut n);
                            n
                        })
                        .collect();
                    crate::vector::turborabit::TurboRabitQuantizer::fit(&normalized, bits)
                } else {
                    crate::vector::turborabit::TurboRabitQuantizer::fit(embeddings, bits)
                };
                self.warren_rc = q.rotate(q.centroid());
                self.turborabit = Some(q);
            }
        }
    }

    /// Normalize `v` to unit length when this index is `Storage::RaBitQ` + `DistanceMetric::
    /// Cosine`; return it unchanged otherwise (borrowed — no allocation on the `L2` hot path).
    ///
    /// RaBitQ's estimator (`crates/core/src/vector/rabitq.rs`) computes squared L2. On unit
    /// vectors, squared L2 is an *exact* affine function of cosine distance:
    /// `‖â−b̂‖² = ‖â‖² + ‖b̂‖² − 2â·b̂ = 2 − 2cosθ = 2·(1 − cosθ) = 2·cosine_distance`.
    /// So fitting/encoding/querying on unit-normalized vectors repurposes the *same*
    /// estimator for cosine with no new estimation error — the approximation quality is
    /// identical to the already-measured L2 case, just applied to different (normalized)
    /// input. [`Self::distance_to_node`] divides the kernel's output by 2 on the way out to
    /// recover the exact cosine distance.
    ///
    /// This is why `Storage::RaBitQ` needed no new kernel for cosine, unlike `Storage::SQ8`
    /// (whose codes are metric-agnostic reconstructions of the original values, so cosine
    /// there is a new *dot-product* kernel over the existing codes, not a change to encoding).
    fn rabitq_cosine_input<'a>(&self, v: &'a [f32]) -> std::borrow::Cow<'a, [f32]> {
        if self.config.metric == DistanceMetric::Cosine {
            let mut owned = v.to_vec();
            crate::vector::ops::normalize(&mut owned);
            std::borrow::Cow::Owned(owned)
        } else {
            std::borrow::Cow::Borrowed(v)
        }
    }

    /// Whether this index's storage has what it needs to encode a vector.
    ///
    /// Always `true` under `Storage::F32` — an `f32` vector needs no fitted state to store.
    /// Under `SQ8`/`RaBitQ` this is `true` once [`Self::fit_codebook`] has actually run
    /// (via `build()`/`build_parallel()`, or [`Self::train`]) — checked by looking at the
    /// codebook state itself rather than a separate `bool` flag, so there is no second
    /// source of truth that could drift out of sync with it.
    #[allow(deprecated)] // internal handling of Storage::TurboQuant until its 0.8 removal
    fn is_trained(&self) -> bool {
        match self.config.storage {
            Storage::F32 => true,
            Storage::SQ8 => !self.q_scale.is_empty(),
            Storage::RaBitQ => self.rabitq.is_some(),
            Storage::TurboQuant => self.turboquant.is_some(),
            Storage::TurboRabit => self.turborabit.is_some(),
            Storage::Warren => self.turborabit.is_some(),
        }
    }

    /// Fit this index's codebook from a representative sample, ahead of incremental
    /// [`Self::add`]/[`Self::add_embedding`] calls.
    ///
    /// The quantized storages (`SQ8`, `RaBitQ`) cannot encode a single vector without first
    /// knowing the data distribution — `SQ8` needs each dimension's observed range, `RaBitQ`
    /// needs a corpus centroid and rotation. `train()` makes that a named, explicit step
    /// instead of an implicit one, the same shape as faiss's `index.train(xb)` before
    /// `index.add(xb)`.
    ///
    /// `build()`/`build_parallel()` call this internally from the full corpus and remain the
    /// right choice for bulk loads — reach for `train()` only when you are building the index
    /// incrementally via `add()`/`add_embedding()` and need to fit the codebook from a sample
    /// first (it need not be the whole corpus, just representative of it).
    ///
    /// A no-op that always succeeds under `Storage::F32`, which needs no codebook.
    ///
    /// # Errors
    /// - [`RagError::InvalidInput`](crate::RagError::InvalidInput) if `sample` is empty (for
    ///   a storage that needs one), or if the index already has documents — retraining a
    ///   non-empty index would desynchronize the vectors already encoded under the old
    ///   codebook from the new one, silently corrupting their distances.
    /// - [`RagError::DimensionMismatch`](crate::RagError::DimensionMismatch) if any sample
    ///   vector doesn't match this index's dimension.
    pub fn train(&mut self, sample: &[Vec<f32>]) -> Result<()> {
        if self.config.storage == Storage::F32 {
            return Ok(());
        }
        if !self.is_empty() {
            return Err(crate::RagError::InvalidInput(
                "train() must be called before any add()/add_embedding() — retraining a \
                 non-empty index would desynchronize already-encoded vectors from the new \
                 codebook"
                    .into(),
            ));
        }
        if sample.is_empty() {
            return Err(crate::RagError::InvalidInput(
                "train() requires a non-empty sample to fit a codebook".into(),
            ));
        }
        for v in sample {
            if v.len() != self.embedding_dim {
                return Err(crate::RagError::DimensionMismatch {
                    expected: self.embedding_dim,
                    actual: v.len(),
                });
            }
        }
        self.fit_codebook(sample);
        Ok(())
    }

    /// Creates a new HNSW index with default configuration
    ///
    /// # Arguments
    /// * `embedding_dim` - Dimensionality of embedding vectors
    pub fn with_defaults(embedding_dim: usize) -> Self {
        Self::new(embedding_dim, HNSWConfig::default())
    }

    /// Set the search-time `ef` on a built index.
    ///
    /// `ef_search` is the recall/speed dial: it bounds the candidate pool during the
    /// layer-0 scan and has no effect whatsoever on graph structure. Sweeping it does
    /// **not** require rebuilding — walking the recall/QPS curve is a search-time
    /// operation. (hnswlib exposes the same thing as `set_ef`.)
    ///
    /// # Example
    /// ```
    /// use foxstash_core::index::{HNSWIndex, HNSWConfig};
    ///
    /// let embeddings = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.7, 0.7]];
    /// let mut index = HNSWIndex::build(embeddings, HNSWConfig::default());
    ///
    /// index.set_ef_search(200);          // more thorough
    /// assert_eq!(index.ef_search(), 200);
    /// ```
    pub fn set_ef_search(&mut self, ef: usize) {
        self.config.ef_search = ef;
    }

    /// The search-time `ef` currently in effect. See [`Self::set_ef_search`].
    pub fn ef_search(&self) -> usize {
        self.config.ef_search
    }

    /// Resize the exact-rerank pool at search time.
    ///
    /// On a quantized index (`Storage::SQ8` / `Storage::RaBitQ`) the graph walk ranks by an
    /// *estimated* distance; the top `rerank_candidates` are then rescored against the exact
    /// f32 vectors. Like [`Self::set_ef_search`], this walks the recall/QPS curve and does
    /// **not** require a rebuild.
    ///
    /// # Errors
    ///
    /// Returns [`RagError::FullPrecisionDropped`] if the index was built with
    /// `rerank_candidates: 0`. That configuration *discards the f32 vectors entirely* — it is
    /// the smallest index foxstash can build — so there is nothing left to rerank against, and
    /// raising the pool afterwards cannot be honored. Rejecting this is the whole reason the
    /// method returns `Result`: silently accepting it would rerank against an empty array and
    /// quietly return the coarse ranking, which is precisely the class of no-op knob this
    /// codebase keeps shipping. Lowering to 0, or any value on an index that kept its vectors,
    /// is fine.
    pub fn set_rerank_candidates(&mut self, n: usize) -> Result<()> {
        if n > 0
            && self.config.storage.rerank_needs_full()
            && self.full.is_empty()
            && !self.is_empty()
        {
            return Err(crate::RagError::FullPrecisionDropped);
        }
        self.config.rerank_candidates = n;
        Ok(())
    }

    /// The exact-rerank pool size currently in effect. See [`Self::set_rerank_candidates`].
    pub fn rerank_candidates(&self) -> usize {
        self.config.rerank_candidates
    }

    /// Build an HNSW index from embeddings using the configured strategy
    ///
    /// This is the recommended way to create an index from bulk embeddings.
    /// The build strategy is controlled by `config.build_strategy`:
    /// - `Parallel` (default): 5.2x faster at 1M, for 0.02-0.31 recall points. Not reproducible
    ///   even at a fixed `seed` — threads race to write neighbour lists.
    /// - `Sequential`: slower, and bit-reproducible at a fixed `seed`.
    ///
    /// # Arguments
    /// * `embeddings` - Vector of embedding vectors (all must have same dimension)
    /// * `config` - HNSW configuration parameters
    ///
    /// # Returns
    /// A new HNSWIndex built from the embeddings
    ///
    /// # Example
    /// ```
    /// use foxstash_core::index::{HNSWIndex, HNSWConfig, BuildStrategy};
    ///
    /// // Create a small set of 4-dimensional embeddings
    /// let embeddings = vec![
    ///     vec![1.0, 0.0, 0.0, 0.0],
    ///     vec![0.0, 1.0, 0.0, 0.0],
    ///     vec![0.0, 0.0, 1.0, 0.0],
    ///     vec![0.0, 0.0, 0.0, 1.0],
    ///     vec![0.5, 0.5, 0.0, 0.0],
    /// ];
    ///
    /// let config = HNSWConfig::default()
    ///     .with_build_strategy(BuildStrategy::Parallel)
    ///     .with_seed(42);
    /// let index = HNSWIndex::build(embeddings, config);
    /// assert_eq!(index.len(), 5);
    /// ```
    pub fn build(embeddings: Vec<Vec<f32>>, config: HNSWConfig) -> Self {
        if embeddings.is_empty() {
            return Self::new(0, config);
        }

        let expected_dim = embeddings[0].len();
        for (i, embedding) in embeddings.iter().enumerate() {
            assert!(
                embedding.len() == expected_dim,
                "All embeddings must have the same dimension: expected {}, got {} at index {}",
                expected_dim,
                embedding.len(),
                i
            );
        }

        // No size-based dispatch. `BuildStrategy::Auto` used to send anything >= 50k vectors to
        // `Sequential` on the premise that `Parallel` lost recall at scale. Measured at 1M:
        // Parallel is 5.2x faster and gives up 0.02-0.31 recall points. The premise was a fixed
        // bug's warning that outlived it. See `BuildStrategy`.
        match config.build_strategy {
            BuildStrategy::Sequential => Self::build_sequential(embeddings, config),
            BuildStrategy::Parallel => Self::build_parallel(embeddings, config),
        }
    }

    /// Build using sequential insertion (high recall, slower)
    fn build_sequential(embeddings: Vec<Vec<f32>>, config: HNSWConfig) -> Self {
        let embedding_dim = embeddings[0].len();
        let n = embeddings.len();

        let seed = config.seed.unwrap_or_else(rand::random);
        let mut rng = StdRng::seed_from_u64(seed);
        let ml = config.ml;

        // Pre-generate all node levels (clamp to EPSILON to prevent ln(0) = -inf)
        let levels: Vec<usize> = (0..n)
            .map(|_| {
                let r: f32 = rng.random::<f32>().max(f32::EPSILON);
                (-r.ln() * ml).floor() as usize
            })
            .collect();

        let _max_level = *levels.iter().max().unwrap_or(&0);

        // Sort by level descending
        let mut sorted_indices: Vec<usize> = (0..n).collect();
        sorted_indices.sort_by(|&a, &b| levels[b].cmp(&levels[a]));

        // Create index
        let mut index = Self::new(embedding_dim, config);

        // `rerank_candidates: 0` means "don't ship the f32 vectors" — a property of the finished
        // INDEX, not of the build. Sequential insertion still needs exact vectors *while* it
        // runs: `insert_node` reads back each candidate's embedding to select neighbours, and on
        // quantized storage the only place that lives is `full`. Skipping it made
        // `HNSWIndex::build(.., Storage::RaBitQ + rerank_candidates: 0)` — the README's "smallest
        // index foxstash can build", on the #[default] build strategy — panic in release with
        // "range start index 24 out of range for slice of length 0".
        //
        // (`build_parallel` never hit this: it builds the graph from the caller's f32 slice and
        // quantizes at the end, so it never reads a vector back. Every existing caller happened
        // to use it, and the one test covering this config used `build_parallel` explicitly —
        // with a comment noting that `insert_node` "assumes `full` is populated". The landmine
        // was documented and stepped over rather than defused.)
        //
        // So: retain the vectors for the duration of the build, then drop them below. Same end
        // state, same memory profile for the shipped index.
        let drop_full_after_build =
            index.config.storage != Storage::F32 && index.config.rerank_candidates == 0;
        if drop_full_after_build {
            index.config.rerank_candidates = 1; // make `push_node` populate `full`
        }

        // Pre-allocate
        index.nodes.reserve(
            n * node_stride(
                index.config.m0,
                embedding_dim,
                index.config.storage,
                index.config.quant_bits(),
            ),
        );
        index.fit_codebook(&embeddings);
        index.connections.reserve(n);
        index.ids.reserve(n);
        index.contents.reserve(n);
        index.metadata.reserve(n);

        // Add nodes in sorted order
        for &i in &sorted_indices {
            let level = levels[i];
            let node_id = index.len();

            // Create connections (using Vec<u32> for cache-friendly traversal)
            let mut node_connections: Vec<Vec<u32>> = Vec::with_capacity(level + 1);
            for _ in 0..=level {
                node_connections.push(Vec::new());
            }

            // Add to storage. `push_node` appends the whole block — vector, norm and an
            // empty layer-0 neighbour list — so `insert_node` below has somewhere to write.
            index.push_node(&embeddings[i]);
            index.connections.push(node_connections);
            index.ids.push(i.to_string());
            index.contents.push(String::new());
            index.metadata.push(None);

            if index.entry_point.is_none() {
                index.entry_point = Some(node_id);
                index.max_layer = level;
                continue;
            }

            index.insert_node(node_id, level);

            if level > index.max_layer {
                index.max_layer = level;
                index.entry_point = Some(node_id);
            }
        }

        // NO `build_l0_cache()` here. That migrates nested layer-0 links into the flat
        // array, but this builder never put any there — `insert_node` wrote them directly
        // to the flat array. Running the migration would copy the empty nested layer 0 over
        // the real graph and silently erase every layer-0 link.

        // The graph is built; the exact vectors have done their job. Honor the caller's
        // `rerank_candidates: 0` by dropping them now — restoring both the config the caller
        // actually asked for and the memory profile it promises.
        if drop_full_after_build {
            index.config.rerank_candidates = 0;
            index.full = Vec::new();
        }

        index.shrink_to_fit();
        index.finalize_reorder()
    }

    /// Returns the number of nodes in the index
    #[inline]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Returns true if the index is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Full-precision vector for a node.
    ///
    /// Under `Storage::F32` this reads the arena (hot). Under the quantized storages it reads
    /// the cold `full` array — correct, but NOT what the traversal uses; see
    /// `distance_to_node`.
    ///
    /// Bounds-checked, deliberately. Replacing these with `get_unchecked` was measured at
    /// +1.3% on SIFT1M — inside run-to-run noise, and not worth `unsafe` in the hottest
    /// accessor in the library. The bounds check is not what separates us from hnswlib.
    #[inline(always)]
    #[allow(deprecated)] // internal handling of Storage::TurboQuant until its 0.8 removal
    fn get_embedding(&self, node_id: usize) -> &[f32] {
        match self.config.storage {
            Storage::F32 => {
                let start = node_id * self.stride + self.hdr;
                bytemuck::cast_slice(&self.nodes[start..start + self.embedding_dim])
            }
            Storage::SQ8 | Storage::RaBitQ | Storage::TurboQuant | Storage::TurboRabit => {
                debug_assert!(
                    !self.full.is_empty(),
                    "full-precision vectors were dropped (rerank_candidates = 0); \
                     no exact embedding exists to return"
                );
                let start = node_id * self.embedding_dim;
                &self.full[start..start + self.embedding_dim]
            }
            // Warren keeps no f32 and returns early in the walk/rerank, so this is off its path.
            // `get_all_documents`/`requantize` would reach it — reconstruct via the codebook there
            // rather than paying to retain f32. Experimental mode, not yet exposed to bindings.
            Storage::Warren => panic!(
                "Storage::Warren retains no exact f32 vectors; rerank uses the residual code"
            ),
        }
    }

    /// The 8-bit codes for a node, from the hot arena. `Storage::SQ8` only.
    #[inline(always)]
    fn get_codes(&self, node_id: usize) -> &[u8] {
        let start = node_id * self.stride + self.hdr;
        let words = vec_words(Storage::SQ8, self.embedding_dim, 0);
        let bytes: &[u8] = bytemuck::cast_slice(&self.nodes[start..start + words]);
        &bytes[..self.embedding_dim]
    }

    /// A node's RaBitQ code — `(dtc_sq, est_factor, packed sign bits)` — from the hot arena.
    /// `Storage::RaBitQ` only.
    #[inline(always)]
    fn get_rabitq_code(&self, node_id: usize) -> (f32, f32, &[u8]) {
        let base = node_id * self.stride + self.hdr;
        let dtc_sq = f32::from_bits(self.nodes[base]);
        let est_factor = f32::from_bits(self.nodes[base + 1]);
        let bit_words = rabitq_bit_words(self.embedding_dim);
        let bytes: &[u8] = bytemuck::cast_slice(&self.nodes[base + 2..base + 2 + bit_words]);
        let n_bytes = self.embedding_dim.div_ceil(8);
        (dtc_sq, est_factor, &bytes[..n_bytes])
    }

    /// Precomputed L2 norm of a node's vector. Lives in the node's own block, so cosine
    /// reads it from a cache line it has already pulled in for the vector.
    #[inline(always)]
    fn get_norm(&self, node_id: usize) -> f32 {
        f32::from_bits(self.nodes[node_id * self.stride + 1])
    }

    fn get_neighbors_l0(&self, node_id: usize) -> &[u32] {
        let base = node_id * self.stride;
        let count = self.nodes[base] as usize;
        &self.nodes[base + 2..base + 2 + count]
    }

    /// True if `node_id` already links to `neighbor` at layer 0.
    #[inline]
    fn l0_contains(&self, node_id: usize, neighbor: u32) -> bool {
        self.get_neighbors_l0(node_id).contains(&neighbor)
    }

    /// Append a layer-0 link. Capacity is `m0 + 1`; the caller prunes back to `m0`.
    #[inline]
    fn l0_push(&mut self, node_id: usize, neighbor: u32) {
        let base = node_id * self.stride;
        let count = self.nodes[base] as usize;
        debug_assert!(
            count < self.config.m0 + 1,
            "layer-0 overflow: prune before pushing again"
        );
        self.nodes[base + 2 + count] = neighbor;
        self.nodes[base] = (count + 1) as u32;
    }

    /// Replace a node's entire layer-0 neighbour list (used after pruning).
    #[inline]
    fn l0_replace(&mut self, node_id: usize, neighbors: &[u32]) {
        let base = node_id * self.stride;
        let count = neighbors.len().min(self.config.m0);
        self.nodes[base + 2..base + 2 + count].copy_from_slice(&neighbors[..count]);
        self.nodes[base] = count as u32;
    }

    /// Byte stride of one node's Warren residual block:
    /// `[min1 f32][step1 f32][scale f32][min2 f32][step2 f32][dim bytes L1][dim bytes L2]`.
    ///
    /// **Two 8-bit levels, not one.** A single 8-bit residual left Warren 0.002 recall short of
    /// exact f32 rerank — enough to miss the 0.99 threshold and, via the cliff, read as a 25% QPS
    /// loss. Measured on coco at pool=500 against an f32 ceiling of 0.9948: 8 bits → 0.9935,
    /// 10 → 0.9944, **12 → 0.9948 (exact match)**, 16 → no further gain.
    ///
    /// 12 bits flat would need bit-packing; 16 flat would need a `u16` SIMD kernel. Two 8-bit
    /// levels reuse `byte_uint_dot_simd` untouched and, because each level rescales to its own
    /// range, are typically *better* than flat 16 for the same bits — the same reason Warren's
    /// 4+8 beat a standalone 8.
    ///
    /// `scale = ℓ/‖v‖` is stored rather than recomputed because deriving it needs a pass over the
    /// nibbles, and a scalar per-dimension pass costs more than the two SIMD dots the rest of the
    /// rerank takes. Four bytes to delete a loop.
    #[inline]
    fn warren_res_stride(&self) -> usize {
        20 + 2 * self.embedding_dim
    }

    /// `(min1, step1, scale, min2, step2, level1, level2)` — cold path, rerank only.
    #[inline]
    #[allow(clippy::type_complexity)]
    fn warren_res_at(&self, node_id: usize) -> (f32, f32, f32, f32, f32, &[u8], &[u8]) {
        let st = self.warren_res_stride();
        let d = self.embedding_dim;
        let b = &self.warren_res[node_id * st..node_id * st + st];
        let f = |o: usize| f32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        (
            f(0),
            f(4),
            f(8),
            f(12),
            f(16),
            &b[20..20 + d],
            &b[20 + d..20 + 2 * d],
        )
    }

    /// Append a node block: zero neighbours, norm and vector filled in.
    ///
    /// Every construction path must go through this. The sequential builder previously
    /// pushed the vector and forgot to grow the layer-0 storage, which panicked on every
    /// input; a single append keeps the arena's invariant impossible to half-satisfy.
    #[allow(deprecated)] // internal handling of Storage::TurboQuant until its 0.8 removal
    fn push_node(&mut self, embedding: &[f32]) {
        debug_assert_eq!(embedding.len(), self.embedding_dim);
        let base = self.nodes.len();
        self.nodes.resize(base + self.stride, 0);
        self.nodes[base + 1] = crate::vector::simd::norm_simd(embedding).to_bits();
        let v = base + self.hdr;

        match self.config.storage {
            Storage::F32 => {
                self.nodes[v..v + self.embedding_dim]
                    .copy_from_slice(bytemuck::cast_slice(embedding));
            }
            Storage::SQ8 => {
                // Codes go in the hot block. The f32 vector goes to the cold side array only
                // if a rerank stage will actually read it — otherwise it is pure memory cost.
                let words = vec_words(Storage::SQ8, self.embedding_dim, 0);
                let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut self.nodes[v..v + words]);
                for (d, &x) in embedding.iter().enumerate() {
                    let s = self.q_scale[d];
                    bytes[d] = if s <= 0.0 {
                        0
                    } else {
                        (((x - self.q_min[d]) / s).round().clamp(0.0, 255.0)) as u8
                    };
                }
                if self.config.rerank_candidates > 0 {
                    self.full.extend_from_slice(embedding);
                }
            }
            Storage::RaBitQ => {
                let rq = self
                    .rabitq
                    .as_ref()
                    .expect("RaBitQ storage requires fit_codebook to run before push_node");
                // See `rabitq_cosine_input`: under cosine, encode a unit-normalized copy so
                // the estimator's squared-L2 output is an exact affine function of cosine
                // distance. `self.full` (below) always keeps the ORIGINAL embedding — cosine
                // is scale-invariant, so the exact rerank stage needs no normalization, and
                // the original is also what an `L2` index needs for exact rerank.
                let encode_input = self.rabitq_cosine_input(embedding);
                let code = rq.encode(&encode_input);
                self.nodes[v] = code.dtc_sq.to_bits();
                self.nodes[v + 1] = code.est_factor.to_bits();
                let bit_words = rabitq_bit_words(self.embedding_dim);
                let bytes: &mut [u8] =
                    bytemuck::cast_slice_mut(&mut self.nodes[v + 2..v + 2 + bit_words]);
                bytes[..code.bits.len()].copy_from_slice(&code.bits);
                if self.config.rerank_candidates > 0 {
                    self.full.extend_from_slice(embedding);
                }
            }
            Storage::TurboQuant => {
                // TurboQuant assumes unit-norm input (its codebook is derived for N(0,1/d) on the
                // sphere), so encode a normalized copy unconditionally. The cold `full` array
                // keeps the ORIGINAL for exact rerank. Block: `[gamma][qjl bits][mse nibbles]` —
                // qjl is already bit-packed by `encode`; the byte-per-coordinate `idx` is packed
                // to nibbles here (mse_bits ≤ 3 guaranteed by `fit_codebook`, so codes fit).
                let tq = self
                    .turboquant
                    .as_ref()
                    .expect("TurboQuant storage requires fit_codebook to run before push_node");
                let mut unit = embedding.to_vec();
                crate::vector::ops::normalize(&mut unit);
                let code = tq.encode(&unit);
                self.nodes[v] = code.gamma.to_bits();
                let bit_words = rabitq_bit_words(self.embedding_dim);
                let qjl_bytes: &mut [u8] =
                    bytemuck::cast_slice_mut(&mut self.nodes[v + 1..v + 1 + bit_words]);
                qjl_bytes[..code.qjl.len()].copy_from_slice(&code.qjl);
                if !code.idx.is_empty() {
                    let nw = nibble_words(tq.padded_dim());
                    let start = v + 1 + bit_words;
                    let nib: &mut [u8] =
                        bytemuck::cast_slice_mut(&mut self.nodes[start..start + nw]);
                    for (i, &c) in code.idx.iter().enumerate() {
                        nib[i / 2] |= c << (4 * (i % 2));
                    }
                }
                if self.config.rerank_candidates > 0 {
                    self.full.extend_from_slice(embedding);
                }
            }
            Storage::TurboRabit => {
                // Same cosine convention as RaBitQ: encode a unit-normalized copy under
                // cosine (see `rabitq_cosine_input`), the original under L2. `self.full`
                // always keeps the ORIGINAL embedding for exact rerank.
                // Block: `[dtc_sq][f_rescale][nibble-packed codes]` — one 4-bit slot per
                // coordinate (B ≤ 4, asserted at fit), read by the fused
                // `nibble_uint_dot_simd` kernel in `distance_to_node`.
                let tr = self
                    .turborabit
                    .as_ref()
                    .expect("TurboRabit storage requires fit_codebook to run before push_node");
                let encode_input = self.rabitq_cosine_input(embedding);
                let code = tr.encode(&encode_input);
                self.nodes[v] = code.dtc_sq.to_bits();
                self.nodes[v + 1] = code.f_rescale.to_bits();
                let nw = nibble_words(self.embedding_dim);
                let nib: &mut [u8] = bytemuck::cast_slice_mut(&mut self.nodes[v + 2..v + 2 + nw]);
                for (i, &c) in code.codes.iter().enumerate() {
                    nib[i / 2] |= c << (4 * (i % 2));
                }
                if self.config.rerank_candidates > 0 {
                    self.full.extend_from_slice(embedding);
                }
            }
            // Parallel-array encode: the arena block is header-only (`vec_words == 0`), and there
            // is NO retained f32 — the 8-bit residual code IS the rerank representation. `node_id`
            // grows in id order because `push_node` is called in id order by every build
            // path, keeping the array index-aligned with the arena.
            // Warren writes TurboRabit's arena block VERBATIM (so the walk is the same code) and,
            // instead of the f32 copy, an 8-bit residual against TurboRabit's own reconstruction —
            // taken in ROTATED space, which is where the rerank consumes it.
            Storage::Warren => {
                let tr = self
                    .turborabit
                    .as_ref()
                    .expect("Storage::Warren requires fit_codebook before push_node");
                let encode_input = self.rabitq_cosine_input(embedding);
                let code = tr.encode(&encode_input);
                self.nodes[v] = code.dtc_sq.to_bits();
                self.nodes[v + 1] = code.f_rescale.to_bits();
                let nw = nibble_words(self.embedding_dim);
                let nib: &mut [u8] = bytemuck::cast_slice_mut(&mut self.nodes[v + 2..v + 2 + nw]);
                for (i, &c) in code.codes.iter().enumerate() {
                    nib[i / 2] |= c << (4 * (i % 2));
                }
                // residual = R·(x − c) − r_recon, quantized per-vector to 8 bits
                let r_recon = tr.reconstruct_rotated(&code);
                let centred: Vec<f32> = encode_input
                    .iter()
                    .zip(tr.centroid())
                    .map(|(a, b)| a - b)
                    .collect();
                let r_true = tr.rotate(&centred);
                let e: Vec<f32> = r_true.iter().zip(&r_recon).map(|(a, b)| a - b).collect();
                // Level 1 over the residual, then level 2 over what level 1 leaves. Each level
                // rescales to its own range, which is why 8+8 reaches the f32 ceiling where a
                // single 8 does not.
                let quant = |v: &[f32]| -> (f32, f32, Vec<u8>) {
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
                };
                let (lo1, step1, c1) = quant(&e);
                let e2: Vec<f32> = e
                    .iter()
                    .zip(&c1)
                    .map(|(&x, &c)| x - (lo1 + c as f32 * step1))
                    .collect();
                let (lo2, step2, c2) = quant(&e2);
                // scale = ℓ/‖grid‖ — the factor `reconstruct_rotated` applies. Precomputed here so
                // rerank never has to walk the nibbles to find it.
                let l = code.dtc_sq.sqrt();
                let cb = -(((1u32 << self.config.rabit_bits) - 1) as f32) / 2.0;
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
                self.warren_res.extend_from_slice(&lo1.to_le_bytes());
                self.warren_res.extend_from_slice(&step1.to_le_bytes());
                self.warren_res.extend_from_slice(&scale.to_le_bytes());
                self.warren_res.extend_from_slice(&lo2.to_le_bytes());
                self.warren_res.extend_from_slice(&step2.to_le_bytes());
                self.warren_res.extend_from_slice(&c1);
                self.warren_res.extend_from_slice(&c2);
            }
        }
    }

    /// Move layer-0 links out of the nested `connections` and into the arena.
    ///
    /// Only the parallel builder needs this: it materialises the whole nested structure
    /// first, then hands layer 0 over to its real owner. The sequential builder writes
    /// layer-0 links straight into the arena via `insert_node` and must **not** call this —
    /// doing so would copy an empty nested layer 0 over the real graph and silently erase
    /// every layer-0 link.
    fn migrate_l0_into_arena(&mut self) {
        for i in 0..self.len() {
            if !self.connections[i].is_empty() {
                let neighbors = std::mem::take(&mut self.connections[i][0]);
                self.l0_replace(i, &neighbors);
            }
        }
    }

    /// Re-encode this index's vectors into a different storage mode, **reusing the graph**.
    ///
    /// The graph is storage-independent by construction — `build_parallel` selects edges with
    /// exact f32 distances and only quantizes the traversal storage afterwards — so building
    /// the same corpus once per storage mode repeats identical (and expensive) graph work to
    /// reach an identical graph. `requantize` extracts this index's vectors and links, fits
    /// the target storage's codebook, and re-encodes: minutes of encode instead of a rebuild.
    ///
    /// It is also the *cleaner experiment*: two direct builds never share a graph (the
    /// parallel builder is non-reproducible even at a fixed seed), so storage comparisons
    /// between them carry graph noise. Requantized siblings share the graph exactly — any
    /// recall difference is the quantizer's alone.
    ///
    /// # Contract
    /// - The source must be `Storage::F32` — the arena then holds exact vectors to re-encode.
    ///   (A quantized source would re-encode its own reconstruction error.)
    /// - `new_config` may change `storage`, the bit budgets, `rerank_candidates` and
    ///   `ef_search`. It must NOT change `m`, `m0`, `metric` or `ef_construction`: the graph
    ///   was built under those, and silently relabeling them would misdescribe the index
    ///   (the same footgun class as the wasm config round-trip bug).
    ///
    /// # Errors
    /// Returns [`RagError::InvalidInput`](crate::RagError::InvalidInput) on a non-F32 source
    /// or a graph-relevant config mismatch.
    pub fn requantize(&self, new_config: HNSWConfig) -> Result<HNSWIndex> {
        if self.config.storage != Storage::F32 {
            return Err(crate::RagError::InvalidInput(format!(
                "requantize requires a Storage::F32 source (got {:?}) — a quantized source \
                 would re-encode its own reconstruction error",
                self.config.storage
            )));
        }
        if new_config.m != self.config.m
            || new_config.m0 != self.config.m0
            || new_config.metric != self.config.metric
            || new_config.ef_construction != self.config.ef_construction
        {
            return Err(crate::RagError::InvalidInput(
                "requantize cannot change m, m0, metric or ef_construction — the graph was \
                 built under them; build a fresh index instead"
                    .into(),
            ));
        }

        let n = self.len();
        let points: Vec<Vec<f32>> = (0..n).map(|i| self.get_embedding(i).to_vec()).collect();

        // Reassemble the nested per-layer links: upper layers are still nested; layer 0
        // lives in the arena (post-`migrate_l0_into_arena`), so put it back into slot 0
        // for the new index's own migrate pass.
        let mut connections = self.connections.clone();
        for (i, c) in connections.iter_mut().enumerate() {
            let l0 = self.get_neighbors_l0(i).to_vec();
            if c.is_empty() {
                c.push(l0);
            } else {
                c[0] = l0;
            }
        }

        // Same assembly as `build_parallel`'s finale: fit the codebook from the exact
        // vectors, push every node in id order (id order is what aligns arena blocks with
        // `ids`/`contents`/`metadata`), then hand layer 0 to the arena.
        let mut index = Self {
            level_rng: StdRng::seed_from_u64(new_config.seed.unwrap_or_else(rand::random)),
            embedding_dim: self.embedding_dim,
            stride: node_stride(
                new_config.m0,
                self.embedding_dim,
                new_config.storage,
                new_config.quant_bits(),
            ),
            hdr: node_hdr_len(new_config.m0),
            config: new_config,
            nodes: Vec::new(),
            connections,
            q_min: Vec::new(),
            q_scale: Vec::new(),
            full: Vec::new(),
            rabitq: None,
            turboquant: None,
            turborabit: None,
            warren_res: Vec::new(),
            warren_rc: Vec::new(),
            ids: self.ids.clone(),
            contents: self.contents.clone(),
            metadata: self.metadata.clone(),
            entry_point: self.entry_point,
            max_layer: self.max_layer,
        };
        index.fit_codebook(&points);
        index.nodes.reserve(n * index.stride);
        if index.config.storage != Storage::F32 && index.config.rerank_candidates > 0 {
            index.full.reserve(n * self.embedding_dim);
        }
        for p in &points {
            index.push_node(p);
        }
        index.migrate_l0_into_arena();
        index.shrink_to_fit();
        Ok(index)
    }

    /// Apply the BFS locality relabel iff `config.reorder_for_locality` — the shared build
    /// finale for both builders, so a direct `build_parallel`/`build_sequential` call gets the
    /// same treatment as `build`. Consuming `self` keeps the peak at one index plus the reorder's
    /// transient copy, not three.
    fn finalize_reorder(self) -> Self {
        if self.config.reorder_for_locality {
            self.reorder_for_locality()
        } else {
            self
        }
    }

    /// Relabel nodes in **breadth-first order from the entry point** so that graph-adjacent
    /// nodes land at nearby arena offsets. The search visits a node then its neighbours, so
    /// when those neighbours sit in nearby cache lines/pages the walk suffers fewer misses —
    /// the measured bottleneck at high recall (see the `coco_m_scaling` diagnosis: per-hop
    /// visit cost, not graph degree). AVX-512 cut the *compute* half of that cost; this cuts
    /// the *memory* half.
    ///
    /// This is a **pure layout change**: the graph topology, the codes, and every returned
    /// document id are identical — the result is the same index, faster. `self` is untouched;
    /// a new index is returned. Storage-agnostic (it copies encoded arena blocks verbatim
    /// rather than re-encoding, so it needs no f32 source, unlike [`Self::requantize`]).
    pub fn reorder_for_locality(&self) -> HNSWIndex {
        let n = self.len();
        let stride = self.stride;

        // BFS over layer 0 from the entry point → visitation order `order[new] = old`.
        // A layer-0-disconnected node (rare) can't be reached; append any stragglers in id
        // order so the permutation stays a bijection.
        let start = self.entry_point.unwrap_or(0);
        let mut order: Vec<usize> = Vec::with_capacity(n);
        let mut seen = vec![false; n];
        if n > 0 {
            let mut queue = std::collections::VecDeque::with_capacity(n);
            queue.push_back(start);
            seen[start] = true;
            while let Some(u) = queue.pop_front() {
                order.push(u);
                for &nb in self.get_neighbors_l0(u) {
                    let nb = nb as usize;
                    if !seen[nb] {
                        seen[nb] = true;
                        queue.push_back(nb);
                    }
                }
            }
            for (i, &s) in seen.iter().enumerate() {
                if !s {
                    order.push(i);
                }
            }
        }
        // Inverse permutation: `pos[old] = new`.
        let mut pos = vec![0u32; n];
        for (new, &old) in order.iter().enumerate() {
            pos[old] = new as u32;
        }

        // Copy each arena block to its new slot, then remap the layer-0 neighbour ids it
        // carries (the vector/codes half of the block copies verbatim).
        let mut nodes = vec![0u32; n * stride];
        for (new, &old) in order.iter().enumerate() {
            let (dst, src) = (new * stride, old * stride);
            nodes[dst..dst + stride].copy_from_slice(&self.nodes[src..src + stride]);
            let count = nodes[dst] as usize;
            for k in 0..count {
                nodes[dst + 2 + k] = pos[nodes[dst + 2 + k] as usize];
            }
        }

        // Upper layers: permute the outer index and remap every neighbour id. Layer 0 slots
        // are empty post-`migrate_l0_into_arena`, so they remap to nothing.
        let connections: Vec<Vec<Vec<u32>>> = order
            .iter()
            .map(|&old| {
                self.connections[old]
                    .iter()
                    .map(|layer| layer.iter().map(|&nb| pos[nb as usize]).collect())
                    .collect()
            })
            .collect();

        // Per-node cold arrays follow the permutation. `full` (f32 rerank vectors) is a flat
        // n×dim blob; permute it row-wise if present.
        let permute = |src: &[String]| order.iter().map(|&o| src[o].clone()).collect::<Vec<_>>();
        let dim = self.embedding_dim;
        let full = if self.full.is_empty() {
            Vec::new()
        } else {
            let mut f = vec![0f32; n * dim];
            for (new, &old) in order.iter().enumerate() {
                f[new * dim..new * dim + dim]
                    .copy_from_slice(&self.full[old * dim..old * dim + dim]);
            }
            f
        };

        Self {
            level_rng: StdRng::seed_from_u64(self.config.seed.unwrap_or_else(rand::random)),
            embedding_dim: dim,
            stride,
            hdr: self.hdr,
            config: self.config.clone(),
            nodes,
            connections,
            q_min: self.q_min.clone(),
            q_scale: self.q_scale.clone(),
            full,
            rabitq: self.rabitq.clone(),
            turboquant: self.turboquant.clone(),
            turborabit: self.turborabit.clone(),
            warren_res: if self.warren_res.is_empty() {
                Vec::new()
            } else {
                let ws = self.warren_res_stride();
                let mut packed = vec![0u8; n * ws];
                for (new, &old) in order.iter().enumerate() {
                    packed[new * ws..new * ws + ws]
                        .copy_from_slice(&self.warren_res[old * ws..old * ws + ws]);
                }
                packed
            },
            warren_rc: self.warren_rc.clone(),
            // Flat code buffer follows the same node permutation as the arena blocks, copied one
            // fixed-stride block at a time (NOT byte-by-byte — this is a Vec<u8> of stride blocks).
            ids: permute(&self.ids),
            contents: permute(&self.contents),
            metadata: order.iter().map(|&o| self.metadata[o].clone()).collect(),
            entry_point: self.entry_point.map(|e| pos[e] as usize),
            max_layer: self.max_layer,
        }
    }

    /// Write a **verbatim binary snapshot**: arena words, links, codebooks, docs — the graph
    /// comes back *identical*, not rebuilt.
    ///
    /// This exists because the JSON save/load path (`storage/file.rs`) deliberately
    /// re-inserts documents through `add()` on load — which re-runs graph construction
    /// (O(build) load time) and, the parallel builder being non-reproducible, returns a
    /// *different graph* than was saved. Fine for portability; wrong for caching a build.
    ///
    /// **This is a same-version cache format, not an archival format**: it is raw bincode of
    /// the in-memory layout, guarded by a version stamp — a snapshot written by a different
    /// foxstash version refuses to load (with a clear error) instead of misreading. Use the
    /// JSON path for anything that must outlive a version bump.
    pub fn snapshot_to_file(&self, path: &std::path::Path) -> Result<()> {
        let snap = HNSWSnapshot {
            format_version: SNAPSHOT_FORMAT_VERSION,
            crate_version: env!("CARGO_PKG_VERSION").to_string(),
            embedding_dim: self.embedding_dim,
            config: self.config.clone(),
            nodes: self.nodes.clone(),
            connections: self.connections.clone(),
            q_min: self.q_min.clone(),
            q_scale: self.q_scale.clone(),
            full: self.full.clone(),
            rabitq: self.rabitq.clone(),
            turboquant: self.turboquant.clone(),
            turborabit: self.turborabit.clone(),
            warren_res: self.warren_res.clone(),
            warren_rc: self.warren_rc.clone(),
            ids: self.ids.clone(),
            contents: self.contents.clone(),
            metadata: self
                .metadata
                .iter()
                .map(|m| m.as_ref().map(|v| v.to_string()))
                .collect(),
            entry_point: self.entry_point,
            max_layer: self.max_layer,
        };
        let file = std::fs::File::create(path)?;
        bincode::serialize_into(std::io::BufWriter::new(file), &snap)?;
        Ok(())
    }

    /// Load a [`Self::snapshot_to_file`] snapshot. The graph is restored verbatim; the level
    /// RNG is reseeded from `config.seed` (a later incremental `add()` draws a fresh seeded
    /// stream rather than continuing the original one — same caveat as the JSON path).
    pub fn snapshot_from_file(path: &std::path::Path) -> Result<HNSWIndex> {
        use bincode::Options;
        let file = std::fs::File::open(path)?;
        // Limit = the file's own size. Without it, a corrupt/foreign file's garbage length
        // prefix makes bincode try to allocate whatever number it read — found as a hard
        // ABORT (allocation failure), not a catchable error, when fed a non-snapshot file.
        // `fixint + allow_trailing_bytes` is the exact config `bincode::serialize` writes;
        // the bare `options()` default is varint and would misread our own files.
        let limit = file.metadata()?.len();
        let snap: HNSWSnapshot = bincode::options()
            .with_fixint_encoding()
            .allow_trailing_bytes()
            .with_limit(limit)
            .deserialize_from(std::io::BufReader::new(file))?;
        if snap.format_version != SNAPSHOT_FORMAT_VERSION
            || snap.crate_version != env!("CARGO_PKG_VERSION")
        {
            return Err(crate::RagError::InvalidInput(format!(
                "snapshot was written by foxstash {} (format v{}), this is {} (format v{}) — \
                 snapshots are a same-version cache, rebuild the index",
                snap.crate_version,
                snap.format_version,
                env!("CARGO_PKG_VERSION"),
                SNAPSHOT_FORMAT_VERSION,
            )));
        }
        let stride = node_stride(
            snap.config.m0,
            snap.embedding_dim,
            snap.config.storage,
            snap.config.quant_bits(),
        );
        if !snap.nodes.len().is_multiple_of(stride) {
            return Err(crate::RagError::InvalidInput(format!(
                "snapshot arena length {} is not a multiple of the node stride {} its config \
                 implies — corrupt or mismatched snapshot",
                snap.nodes.len(),
                stride
            )));
        }
        let n = snap.nodes.len() / stride;
        if snap.ids.len() != n || snap.contents.len() != n || snap.metadata.len() != n {
            return Err(crate::RagError::InvalidInput(format!(
                "snapshot document arrays ({} ids) do not match its {} arena nodes",
                snap.ids.len(),
                n
            )));
        }
        let metadata: Vec<Option<serde_json::Value>> = snap
            .metadata
            .into_iter()
            .map(|m| {
                m.map(|s| {
                    serde_json::from_str(&s).map_err(|e| {
                        crate::RagError::InvalidInput(format!(
                            "snapshot metadata is not valid JSON: {e}"
                        ))
                    })
                })
                .transpose()
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            level_rng: StdRng::seed_from_u64(snap.config.seed.unwrap_or_else(rand::random)),
            embedding_dim: snap.embedding_dim,
            stride,
            hdr: node_hdr_len(snap.config.m0),
            config: snap.config,
            nodes: snap.nodes,
            connections: snap.connections,
            q_min: snap.q_min,
            q_scale: snap.q_scale,
            full: snap.full,
            rabitq: snap.rabitq,
            turboquant: snap.turboquant,
            turborabit: snap.turborabit,
            warren_res: snap.warren_res,
            warren_rc: snap.warren_rc,
            ids: snap.ids,
            contents: snap.contents,
            metadata,
            entry_point: snap.entry_point,
            max_layer: snap.max_layer,
        })
    }

    /// Adds a document to the index
    ///
    /// # Arguments
    /// * `document` - Document with embedding to add
    ///
    /// # Errors
    /// - Returns [`RagError::DimensionMismatch`](crate::RagError::DimensionMismatch) if the
    ///   embedding dimension doesn't match the index dimension.
    /// - Returns [`RagError::NotTrained`](crate::RagError::NotTrained) if this index's
    ///   storage is quantized (`SQ8`/`RaBitQ`) and [`Self::train`] hasn't been called yet —
    ///   those storages cannot encode a vector without a fitted codebook.
    pub fn add(&mut self, document: Document) -> Result<()> {
        let Document {
            id,
            content,
            embedding,
            metadata,
        } = document;
        self.add_parts(&embedding, id, content, metadata)
    }

    /// Add a document the caller wants to keep, cloning only what the index must own.
    ///
    /// [`Self::add`] takes the document by value and *moves* its id and content in —
    /// the right shape when the caller is handing ownership over. A caller that keeps
    /// its own copy (as `foxstash-db` does, for get-by-id and checkpointing) has to
    /// pass `doc.clone()` instead, and that clone allocates a `Vec<f32>` for the
    /// embedding which `add` immediately copies into the arena and drops.
    ///
    /// That allocate-copy-free per insert is pure churn, and churn is where the
    /// memory goes: 78% of a collection's resident footprint measured as allocator
    /// retention rather than live data.
    ///
    /// This clones the id, content and metadata — which the index genuinely must own —
    /// and reads the embedding straight through, so no vector allocation happens.
    pub fn add_borrowed(&mut self, document: &Document) -> Result<()> {
        self.add_parts(
            &document.embedding,
            document.id.clone(),
            document.content.clone(),
            document.metadata.clone(),
        )
    }

    /// Shared body of [`Self::add`] and [`Self::add_borrowed`].
    ///
    /// Takes the embedding by reference because the index copies it into the arena
    /// rather than retaining the `Vec`, and takes the owned fields the index really
    /// does keep. Having one implementation is the point: two copies of insertion
    /// logic is exactly how a validation check ends up on only one path.
    /// Validate an embedding against everything [`Self::add`] requires, without
    /// mutating anything.
    ///
    /// Exists so a caller that journals before mutating can reject a bad vector
    /// *before* it is durable. `foxstash-db` writes its WAL first for crash safety,
    /// and used to discover a non-finite embedding only once `add` ran — after the
    /// WAL entry was on disk. `serde_json` writes `NaN` as `null`, which can never
    /// be read back as `f32`, so a single rejected insert made the collection
    /// permanently unopenable.
    ///
    /// One implementation, called by both [`Self::add`] and any pre-flight check —
    /// a second copy of these rules is how one path ends up enforcing something the
    /// other does not.
    pub fn validate_embedding_for_add(&self, embedding: &[f32]) -> Result<()> {
        if embedding.len() != self.embedding_dim {
            return Err(crate::RagError::DimensionMismatch {
                expected: self.embedding_dim,
                actual: embedding.len(),
            });
        }
        if !self.is_trained() {
            return Err(crate::RagError::NotTrained(format!(
                "Storage::{:?} requires a fitted codebook before add() — call \
                 `index.train(&sample)` first, or build the index via `HNSWIndex::build`/ \
                 `build_parallel`, which trains internally from the full corpus",
                self.config.storage
            )));
        }
        if matches!(self.config.storage, Storage::Warren) {
            return Err(crate::RagError::InvalidInput(format!(
                "Storage::{:?} does not support incremental add() — it retains no f32 for exact \
                 graph construction. Build the whole corpus at once via HNSWIndex::build / \
                 build_parallel.",
                self.config.storage
            )));
        }
        if embedding.iter().any(|v| !v.is_finite()) {
            return Err(crate::RagError::InvalidInput(
                "embedding contains non-finite values (NaN or Inf)".into(),
            ));
        }
        Ok(())
    }

    fn add_parts(
        &mut self,
        embedding: &[f32],
        id: String,
        content: String,
        metadata: Option<serde_json::Value>,
    ) -> Result<()> {
        self.validate_embedding_for_add(embedding)?;

        let node_id = self.len();
        let node_level = self.random_level();

        // Create connections for each layer (Vec<u32> for cache-friendly traversal)
        let mut node_connections: Vec<Vec<u32>> = Vec::with_capacity(node_level + 1);
        for _ in 0..=node_level {
            node_connections.push(Vec::new());
        }

        self.push_node(embedding);
        self.connections.push(node_connections);
        self.ids.push(id);
        self.contents.push(content);
        self.metadata.push(metadata);

        // If this is the first node, make it the entry point
        if self.entry_point.is_none() {
            self.entry_point = Some(node_id);
            self.max_layer = node_level;
            return Ok(());
        }

        self.insert_node(node_id, node_level);

        // Update entry point if this node has more layers
        if node_level > self.max_layer {
            self.max_layer = node_level;
            self.entry_point = Some(node_id);
        }

        Ok(())
    }

    /// Add just an embedding vector (faster than add() for bulk operations)
    ///
    /// This method is optimized for benchmarks and bulk data loading where
    /// you only need to store embeddings without document content.
    ///
    /// # Arguments
    /// * `id` - Unique identifier for this vector
    /// * `embedding` - The embedding vector to add
    ///
    /// # Errors
    /// - Returns [`RagError::DimensionMismatch`](crate::RagError::DimensionMismatch) if the
    ///   embedding dimension doesn't match the index dimension.
    /// - Returns [`RagError::NotTrained`](crate::RagError::NotTrained) if this index's
    ///   storage is quantized (`SQ8`/`RaBitQ`) and [`Self::train`] hasn't been called yet.
    pub fn add_embedding(&mut self, id: String, embedding: Vec<f32>) -> Result<()> {
        if embedding.len() != self.embedding_dim {
            return Err(crate::RagError::DimensionMismatch {
                expected: self.embedding_dim,
                actual: embedding.len(),
            });
        }

        if !self.is_trained() {
            return Err(crate::RagError::NotTrained(format!(
                "Storage::{:?} requires a fitted codebook before add_embedding() — call \
                 `index.train(&sample)` first, or build the index via `HNSWIndex::build`/ \
                 `build_parallel`, which trains internally from the full corpus",
                self.config.storage
            )));
        }

        // Bulk-build-only — see the identical guard in `add()`.
        if matches!(self.config.storage, Storage::Warren) {
            return Err(crate::RagError::InvalidInput(format!(
                "Storage::{:?} does not support incremental add() — it retains no f32 for exact \
                 graph construction. Build the whole corpus at once via HNSWIndex::build / \
                 build_parallel.",
                self.config.storage
            )));
        }

        if embedding.iter().any(|v| !v.is_finite()) {
            return Err(crate::RagError::InvalidInput(
                "embedding contains non-finite values (NaN or Inf)".into(),
            ));
        }

        let node_id = self.len();
        let node_level = self.random_level();

        // Create connections for each layer (Vec<u32> for cache-friendly traversal)
        let mut node_connections: Vec<Vec<u32>> = Vec::with_capacity(node_level + 1);
        for _ in 0..=node_level {
            node_connections.push(Vec::new());
        }

        self.push_node(&embedding);
        self.connections.push(node_connections);
        self.ids.push(id);
        self.contents.push(String::new());
        self.metadata.push(None);

        // If this is the first node, make it the entry point
        if self.entry_point.is_none() {
            self.entry_point = Some(node_id);
            self.max_layer = node_level;
            return Ok(());
        }

        self.insert_node(node_id, node_level);

        // Update entry point if this node has more layers
        if node_level > self.max_layer {
            self.max_layer = node_level;
            self.entry_point = Some(node_id);
        }

        Ok(())
    }

    /// Search for the `k` nearest neighbours of `query`.
    ///
    /// # Errors
    /// [`RagError::DimensionMismatch`](crate::RagError::DimensionMismatch) if `query` is not
    /// this index's dimension.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        // Reuse a per-thread scratch context across calls. A plain `search()` otherwise
        // allocates and zero-inits a fresh visited-bitset + two candidate heaps on *every*
        // query — ~184 KB on a 1.5M-node index, a per-query cost that grows with the corpus and
        // is pure overhead. `search_inner` resizes the context if this index is larger than the
        // last one this thread searched (line ~2291), and `search_layer` resets the visited set,
        // so reuse is transparent — the search sees a clean context regardless. `search_batch`
        // and `Searcher` already reuse; this brings the same to the single-query path the
        // Python/wasm bindings call once per query.
        thread_local! {
            static CTX: std::cell::RefCell<SearchContext> =
                std::cell::RefCell::new(SearchContext::new(0));
        }
        CTX.with(|c| self.search_inner(query, k, &mut c.borrow_mut(), None))
    }

    /// Search, returning only results whose node is allowed by `filter` — up to `k` of them.
    ///
    /// The graph is walked in full — excluded nodes are still *traversed*, because they are
    /// load-bearing for connectivity — but only allowed nodes enter the result set. You get up to
    /// `k` allowed nearest neighbours with no over-fetch and no separate post-filter step. Build
    /// `filter` once with [`HNSWIndex::filter_mask`] / [`HNSWIndex::filter_mask_ids`] and reuse it
    /// across queries.
    ///
    /// **Cost scales with selectivity.** A permissive filter costs about the same as an unfiltered
    /// search. A very selective one (few allowed nodes, scattered through the graph) forces the walk
    /// to explore widely to collect `k` of them — in the limit, most of the graph. When the allowed
    /// set is a *tiny* fraction of the corpus a brute-force scan of just those nodes is cheaper;
    /// [`FilterMask::allowed_count`] lets the caller pick the strategy. Raising `ef_search` recovers
    /// recall lost to a selective filter, at proportional cost.
    ///
    /// # Errors
    /// [`RagError::DimensionMismatch`](crate::RagError::DimensionMismatch) if `query` is not this
    /// index's dimension.
    pub fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        filter: &FilterMask,
    ) -> Result<Vec<SearchResult>> {
        let admit = |id: usize| filter.contains(id);
        thread_local! {
            static CTX: std::cell::RefCell<SearchContext> =
                std::cell::RefCell::new(SearchContext::new(0));
        }
        CTX.with(|c| self.search_inner(query, k, &mut c.borrow_mut(), Some(&admit)))
    }

    /// Filtered search against a **predicate**, for one-off filters that aren't worth a
    /// [`FilterMask`] — the metadata queries a `Collection` runs, where each search carries a
    /// different filter. `allow` is evaluated **lazily, during the walk**, only on the nodes the
    /// traversal actually visits (it receives each candidate's external id and metadata), so unlike
    /// a mask there is no O(n) up-front pass, and unlike post-filtering an over-fetched result set
    /// there is no repeated widening: one graph walk collects up to `k` allowed neighbours directly.
    ///
    /// Prefer [`HNSWIndex::search_filtered`] when the *same* filter is reused across many queries (a
    /// prebuilt mask is a bit-test per candidate rather than a predicate call). Same layer-0 gating,
    /// same "cost scales with selectivity" behaviour, same `ef_search` recall lever — see
    /// [`HNSWIndex::search_filtered`].
    ///
    /// # Errors
    /// [`RagError::DimensionMismatch`](crate::RagError::DimensionMismatch) if `query` is not this
    /// index's dimension.
    pub fn search_filtered_by<F>(
        &self,
        query: &[f32],
        k: usize,
        allow: F,
    ) -> Result<Vec<SearchResult>>
    where
        F: Fn(&str, Option<&serde_json::Value>) -> bool,
    {
        let admit = |id: usize| allow(&self.ids[id], self.metadata[id].as_ref());
        thread_local! {
            static CTX: std::cell::RefCell<SearchContext> =
                std::cell::RefCell::new(SearchContext::new(0));
        }
        CTX.with(|c| self.search_inner(query, k, &mut c.borrow_mut(), Some(&admit)))
    }

    /// Build a reusable [`FilterMask`] by testing every document against `pred` once (O(n)).
    ///
    /// `pred` receives each document's `(id, content, metadata)` and returns `true` to allow it.
    /// This is the O(n) step; amortise it by caching the mask and reusing it across every query
    /// that shares the predicate — see [`FilterMask`] and [`HNSWIndex::search_filtered`].
    pub fn filter_mask<F>(&self, mut pred: F) -> FilterMask
    where
        F: FnMut(&str, &str, Option<&serde_json::Value>) -> bool,
    {
        let mut mask = FilterMask::empty(self.len());
        for i in 0..self.len() {
            if pred(&self.ids[i], &self.contents[i], self.metadata[i].as_ref()) {
                mask.allow(i);
            }
        }
        mask
    }

    /// Build a [`FilterMask`] allowing exactly the documents whose external id is in `allowed`.
    pub fn filter_mask_ids(&self, allowed: &std::collections::HashSet<String>) -> FilterMask {
        let mut mask = FilterMask::empty(self.len());
        for i in 0..self.len() {
            if allowed.contains(&self.ids[i]) {
                mask.allow(i);
            }
        }
        mask
    }

    /// Search many queries in parallel, across all rayon worker threads.
    ///
    /// # Errors
    /// [`RagError::DimensionMismatch`](crate::RagError::DimensionMismatch) if any query is
    /// not this index's dimension.
    pub fn search_batch(&self, queries: &[Vec<f32>], k: usize) -> Result<Vec<Vec<SearchResult>>> {
        use rayon::prelude::*;

        // One scratch context per worker thread, not per query — `map_init` hands each rayon
        // worker a context it reuses across every query it receives.
        queries
            .par_iter()
            .map_init(
                || SearchContext::new(self.len()),
                |ctx, query| self.search_inner(query, k, ctx, None),
            )
            .collect()
    }

    /// [`HNSWIndex::search_batch`] with the predicate gate of [`HNSWIndex::search_filtered_by`]:
    /// each query is a single filtered graph walk, fanned across rayon workers with a per-worker
    /// reusable context. `allow(id, metadata)` is evaluated lazily on visited nodes and must be
    /// `Sync` (it runs on every worker). This is what a `Collection`'s parallel filtered batch uses
    /// instead of re-running the whole batch at escalating over-fetch sizes.
    ///
    /// # Errors
    /// [`RagError::DimensionMismatch`](crate::RagError::DimensionMismatch) if any query is not this
    /// index's dimension.
    pub fn search_batch_filtered_by<F>(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        allow: F,
    ) -> Result<Vec<Vec<SearchResult>>>
    where
        F: Fn(&str, Option<&serde_json::Value>) -> bool + Sync,
    {
        use rayon::prelude::*;
        let admit = |id: usize| allow(&self.ids[id], self.metadata[id].as_ref());
        queries
            .par_iter()
            .map_init(
                || SearchContext::new(self.len()),
                |ctx, query| self.search_inner(query, k, ctx, Some(&admit)),
            )
            .collect()
    }

    /// A [`Searcher`]: a cursor that holds its scratch space across queries and counts the
    /// distance computations it performs.
    ///
    /// Reach for it to read [`Searcher::distance_calls`], not to go faster — see [`Searcher`].
    pub fn searcher(&self) -> Searcher<'_> {
        Searcher {
            index: self,
            ctx: SearchContext::new(self.len()),
        }
    }

    fn search_inner(
        &self,
        query: &[f32],
        k: usize,
        ctx: &mut SearchContext,
        filter: Option<&dyn Fn(usize) -> bool>,
    ) -> Result<Vec<SearchResult>> {
        if query.len() != self.embedding_dim {
            return Err(crate::RagError::DimensionMismatch {
                expected: self.embedding_dim,
                actual: query.len(),
            });
        }

        if self.is_empty() {
            return Ok(Vec::new());
        }

        // Ensure search context is large enough for current index size.
        if ctx.capacity < self.len() {
            *ctx = SearchContext::new(self.len());
        }

        // Precompute query norm (cosine) and, under RaBitQ, rotate the query into RaBitQ space
        // — once for the whole search. The rotation is an O(dim²) matvec; recomputing it per
        // node visit (as calling `distance_to_node` naively might invite) would turn every
        // O(dim) distance in the walk back into an O(dim²) one, the exact regression
        // `Storage::RaBitQ` exists to avoid.
        let query_norm = crate::vector::simd::norm_simd(query);
        let rq_prepared = self.prepare_rabitq_query(query);
        let tq_prepared = self.prepare_turboquant_query(query);
        let tr_prepared = self.prepare_turborabit_query(query);
        let qprep = QueryPrep {
            norm: query_norm,
            rabitq: rq_prepared.as_ref(),
            turboquant: tq_prepared.as_ref(),
            turborabit: tr_prepared.as_ref(),
            filter,
        };

        let entry_point = self.entry_point.unwrap();
        let mut current_nearest = vec![entry_point];

        // Search from top layer to layer 1 (ef=1 greedy descent — ids only)
        for layer in (1..=self.max_layer).rev() {
            current_nearest = self
                .search_layer(query, &current_nearest, 1, layer, ctx, &qprep)
                .into_iter()
                .map(|(_, id)| id)
                .collect();
        }

        // Search layer 0 with ef_search candidates
        let ef = self.config.ef_search.max(k);
        let mut found = self.search_layer(query, &current_nearest, ef, 0, ctx, &qprep);

        // Under the quantized storages the walk ranked by an approximate distance, so its
        // top-k is not the true top-k. Rescore a pool with exact distances and re-sort. The
        // pool is read from the cold `full` array — O(pool) reads, against O(nodes visited)
        // during the walk, which is why the walk gets to keep its small blocks.
        if self.config.storage != Storage::F32 && self.config.rerank_candidates > 0 {
            let pool = self.config.rerank_candidates.max(k).min(found.len());
            found.truncate(pool);
            // Software-pipeline the pool reads. Each `exact_distance` gathers a
            // `dim×4`-byte vector from the cold `full` array at an effectively random
            // offset; unprefetched, every iteration serializes a full DRAM round-trip
            // before the next can start. Prefetching a few entries ahead overlaps the
            // fetch with the current entry's scoring — the same trick the walk uses at
            // `search_layer` — and at rerank=500 this loop reads ~1.5 MB/query at 768-d,
            // which dominates the high-recall operating points. Only the head lines are
            // issued; `exact_distance` reads the vector sequentially, so the hardware
            // streamer covers the tail (same rationale as VECTOR_LINES above).
            if self.config.storage == Storage::Warren {
                // No f32: rescore against TurboRabit's reconstruction plus the 8-bit residual,
                // entirely in ROTATED space so nothing here is O(dim²):
                //     ⟨q,x⟩ = ⟨q,c⟩ + ⟨R·q, r_recon + e⟩,   R·q = rq + R·c
                let tr = self
                    .turborabit
                    .as_ref()
                    .expect("Storage::Warren rerank requires a fitted codebook");
                let prepared = qprep
                    .turborabit
                    .expect("Storage::Warren rerank requires prepare_turborabit_query");
                let dim = self.embedding_dim;
                let cos_in = self.rabitq_cosine_input(query);
                let rq_full: Vec<f32> = prepared
                    .rq()
                    .iter()
                    .zip(&self.warren_rc)
                    .map(|(a, b)| a + b)
                    .collect();
                let qc: f32 = cos_in.iter().zip(tr.centroid()).map(|(a, b)| a * b).sum();
                // Three SIMD dots per candidate, no scalar per-dimension loop. Using turborabit's
                // own identity `grid = u + c_B`:
                //   ⟨rq, recon⟩ = scale·(⟨rq,u⟩ + c_B·Σrq) + (lo1+lo2)·Σrq
                //                 + step1·⟨rq,L1⟩ + step2·⟨rq,L2⟩
                // A hand-rolled scalar unpack here cost 5.4 µs/candidate and made Warren 3x
                // slower than TurboRabit despite an identical walk.
                let cb = -(((1u32 << self.config.rabit_bits) - 1) as f32) / 2.0;
                let sum_rq: f32 = rq_full.iter().sum();
                let nw = nibble_words(dim);
                for entry in found.iter_mut() {
                    let v = entry.1 * self.stride + self.hdr;
                    let nib: &[u8] = bytemuck::cast_slice(&self.nodes[v + 2..v + 2 + nw]);
                    let (lo1, step1, scale, lo2, step2, l1, l2) = self.warren_res_at(entry.1);
                    let nibble_dot = crate::vector::simd::nibble_uint_dot_simd(&rq_full, nib);
                    let d1 = crate::vector::simd::byte_uint_dot_simd(&rq_full, l1);
                    let d2 = crate::vector::simd::byte_uint_dot_simd(&rq_full, l2);
                    let acc = qc
                        + scale * (nibble_dot + cb * sum_rq)
                        + (lo1 + lo2) * sum_rq
                        + step1 * d1
                        + step2 * d2;
                    entry.0 = match self.config.metric {
                        DistanceMetric::L2 => (2.0 - 2.0 * acc).max(0.0),
                        DistanceMetric::Cosine => (1.0 - acc).clamp(0.0, 2.0),
                    };
                }
            } else {
                const RERANK_PREFETCH_AHEAD: usize = 4;
                const RERANK_HEAD_LINES: usize = 8; // 512 B of a 3 KB vector at 768-d
                let dim = self.embedding_dim;
                for i in 0..found.len() {
                    let ahead = i + RERANK_PREFETCH_AHEAD;
                    if ahead < found.len() {
                        // SAFETY: prefetch is a hint — `wrapping_add` cannot leave the
                        // allocation for a valid node id, and a stale line costs nothing.
                        unsafe {
                            let p =
                                self.full.as_ptr().wrapping_add(found[ahead].1 * dim) as *const u8;
                            prefetch_embedding(p, RERANK_HEAD_LINES);
                        }
                    }
                    found[i].0 = self.exact_distance(query, found[i].1);
                }
            }
            found.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        }

        // `found` is already sorted nearest-first and carries its distances, so cut to k
        // *before* materialising anything. Building a SearchResult per candidate would clone
        // an id, a content string and a metadata blob for all `ef` of them (500 by default)
        // only to discard all but k — and would recompute every distance to do it.
        Ok(found
            .into_iter()
            .take(k)
            .map(|(dist, node_id)| SearchResult {
                id: self.ids[node_id].clone(),
                content: self.contents[node_id].clone(),
                score: self.score_from_distance(dist),
                metadata: self.metadata[node_id].clone(),
            })
            .collect())
    }

    /// Clears all documents from the index
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.full.clear();
        self.connections.clear();
        self.ids.clear();
        self.contents.clear();
        self.metadata.clear();
        self.entry_point = None;
        self.max_layer = 0;
    }

    /// Get all documents in the index
    ///
    /// Returns a vector containing clones of all documents in the index.
    /// Useful for serialization and persistence.
    ///
    /// # Returns
    /// * `Vec<Document>` - Vector of all documents
    pub fn get_all_documents(&self) -> Vec<Document> {
        (0..self.len())
            .map(|i| Document {
                id: self.ids[i].clone(),
                content: self.contents[i].clone(),
                embedding: self.get_embedding(i).to_vec(),
                metadata: self.metadata[i].clone(),
            })
            .collect()
    }

    /// Get the HNSW configuration
    ///
    /// # Returns
    /// * `&HNSWConfig` - Reference to the configuration
    pub fn config(&self) -> &HNSWConfig {
        &self.config
    }

    /// Get the entry point node index
    ///
    /// # Returns
    /// * `Option<usize>` - Entry point node index, or None if index is empty
    pub fn entry_point(&self) -> Option<usize> {
        self.entry_point
    }

    /// Get the maximum layer in the index
    ///
    /// # Returns
    /// * `usize` - Maximum layer
    pub fn max_layer(&self) -> usize {
        self.max_layer
    }

    /// Get the embedding dimension
    ///
    /// # Returns
    /// * `usize` - Embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Release surplus allocation capacity.
    ///
    /// `Vec` growth doubles, so after a build the node arena can hold ~2x the bytes it
    /// needs. The build paths call this for you; call it yourself after a run of `add()` if
    /// the index is now static.
    pub fn shrink_to_fit(&mut self) {
        self.nodes.shrink_to_fit();
        self.full.shrink_to_fit();
        self.ids.shrink_to_fit();
        self.contents.shrink_to_fit();
        self.metadata.shrink_to_fit();
        for layers in &mut self.connections {
            for l in layers.iter_mut() {
                l.shrink_to_fit();
            }
            layers.shrink_to_fit();
        }
        self.connections.shrink_to_fit();
    }

    /// Bytes actually retained by this index, broken down by component.
    ///
    /// Counts allocated *capacity*, not just length, and includes the per-`Vec` headers in
    /// the nested graph — those are easy to forget and, at one `Vec` per node per layer,
    /// they are not small.
    ///
    /// Prefer this over measuring RSS: RSS around a build also captures the builder's
    /// transient allocations and whatever the allocator declines to return to the OS.
    /// Mean number of layer-0 neighbours per node.
    ///
    /// The cost of one hop. A search performs roughly `nodes_expanded * avg_degree` distance
    /// computations, so this is half of the "how much work does a query do" equation — and it
    /// is a property of the *graph*, fixed at build time, not of the search.
    ///
    /// Worth checking against the competition: at matched recall on SIFT1M, foxstash performs
    /// ~32% more distance computations per query than faiss while being ~15% *faster* per
    /// computation. Work per query, not speed per unit of work, is where the remaining gap
    /// lives, and out-degree is the first place to look for it.
    pub fn avg_degree_l0(&self) -> f32 {
        if self.is_empty() {
            return 0.0;
        }
        let total: usize = (0..self.len())
            .map(|i| self.get_neighbors_l0(i).len())
            .sum();
        total as f32 / self.len() as f32
    }

    pub fn memory_breakdown(&self) -> MemoryBreakdown {
        let vec_header = std::mem::size_of::<Vec<u32>>();

        let nested: usize = self
            .connections
            .iter()
            .map(|layers| {
                vec_header
                    + layers
                        .iter()
                        .map(|l| vec_header + l.capacity() * std::mem::size_of::<u32>())
                        .sum::<usize>()
            })
            .sum();

        // Vectors, norms and layer-0 links share one arena, so the split below is logical
        // rather than physical. `layer0_links` carries the block headers (count + norm +
        // neighbour slots + padding) plus any surplus arena capacity.
        let n = self.len();
        let arena = self.nodes.capacity() * std::mem::size_of::<u32>();
        // Bytes the *traversal* reads for vectors: 4/dim under F32, 1/dim under SQ8.
        let hot_vectors =
            n * vec_words(
                self.config.storage,
                self.embedding_dim,
                self.config.quant_bits(),
            ) * 4;
        // Under SQ8 the f32 vectors still exist, in the cold rerank array.
        let cold_vectors = self.full.capacity() * std::mem::size_of::<f32>();

        MemoryBreakdown {
            embeddings: hot_vectors + cold_vectors,
            norms: n * std::mem::size_of::<f32>(),
            layer0_links: arena
                .saturating_sub(hot_vectors)
                .saturating_sub(n * std::mem::size_of::<f32>()),
            upper_layer_links: nested,
            payload: self
                .ids
                .iter()
                .map(|s| s.capacity() + std::mem::size_of::<String>())
                .sum::<usize>()
                + self
                    .contents
                    .iter()
                    .map(|s| s.capacity() + std::mem::size_of::<String>())
                    .sum::<usize>(),
        }
    }

    /// Generates a level for a new node using exponential decay, from the index's seeded stream.
    ///
    /// Clamps the uniform sample to `[EPSILON, 1)` to prevent `ln(0.0) = -inf`.
    ///
    /// Takes `&mut self` because it advances that stream. It previously took `&self` and called
    /// `rand::rng()`, which is what made `add()` ignore `seed`.
    fn random_level(&mut self) -> usize {
        let uniform: f32 = self.level_rng.random::<f32>().max(f32::EPSILON);
        (-uniform.ln() * self.config.ml).floor() as usize
    }

    /// Inserts a node into the graph structure.
    ///
    /// Returns the IDs of all nodes whose layer-0 connections were modified
    /// (the new node itself plus all layer-0 neighbors it linked to). The caller
    /// `build_l0_cache` rebuild.
    ///
    /// # Arguments
    /// * `node_id` - ID of the node to insert
    /// * `node_level` - Maximum layer of the node
    fn insert_node(&mut self, node_id: usize, node_level: usize) {
        let entry_point = self.entry_point.unwrap();
        let mut current_nearest = vec![entry_point];

        // Get embedding once (hot path optimization)
        let node_embedding = self.get_embedding(node_id).to_vec();
        let query_norm = crate::vector::simd::norm_simd(&node_embedding);
        // See `search_inner`: prepared once per inserted node, not once per distance call.
        let rq_prepared = self.prepare_rabitq_query(&node_embedding);
        let tq_prepared = self.prepare_turboquant_query(&node_embedding);
        let tr_prepared = self.prepare_turborabit_query(&node_embedding);
        let qprep = QueryPrep {
            norm: query_norm,
            rabitq: rq_prepared.as_ref(),
            turboquant: tq_prepared.as_ref(),
            turborabit: tr_prepared.as_ref(),
            // Insert path never runs under Warren (bulk-build-only; add() is rejected).
            filter: None, // builds are never filtered
        };
        let mut ctx = SearchContext::new(self.len());

        // Search for nearest neighbors from top to target layer + 1
        for layer in (node_level + 1..=self.max_layer).rev() {
            current_nearest = self
                .search_layer(
                    &node_embedding,
                    &current_nearest,
                    1,
                    layer,
                    &mut ctx,
                    &qprep,
                )
                .into_iter()
                .map(|(_, id)| id)
                .collect();
        }

        // Insert into layers from top to bottom
        for layer in (0..=node_level).rev() {
            current_nearest = self
                .search_layer(
                    &node_embedding,
                    &current_nearest,
                    self.config.ef_construction,
                    layer,
                    &mut ctx,
                    &qprep,
                )
                .into_iter()
                .map(|(_, id)| id)
                .collect();

            // Determine M for this layer
            let m = if layer == 0 {
                self.config.m0
            } else {
                self.config.m
            };

            // Select M nearest neighbors
            let neighbors = self.select_neighbors(&current_nearest, &node_embedding, m, layer);

            // Add bidirectional links. Layer 0 is owned by the flat array; layers >= 1 by
            // the nested structure. Keeping both would duplicate the hottest data in the index.
            for &neighbor_id in &neighbors {
                let neighbor_u32 = neighbor_id as u32;
                let node_u32 = node_id as u32;

                if layer == 0 {
                    if !self.l0_contains(node_id, neighbor_u32) {
                        self.l0_push(node_id, neighbor_u32);
                    }
                    if !self.l0_contains(neighbor_id, node_u32) {
                        self.l0_push(neighbor_id, node_u32);
                    }

                    let m0 = self.config.m0;
                    if self.get_neighbors_l0(neighbor_id).len() > m0 {
                        let neighbor_embedding = self.get_embedding(neighbor_id).to_vec();
                        let current: Vec<usize> = self
                            .get_neighbors_l0(neighbor_id)
                            .iter()
                            .map(|&x| x as usize)
                            .collect();
                        let pruned =
                            self.select_neighbors(&current, &neighbor_embedding, m0, layer);
                        let pruned: Vec<u32> = pruned.into_iter().map(|x| x as u32).collect();
                        self.l0_replace(neighbor_id, &pruned);
                    }
                    continue;
                }

                if !self.connections[node_id][layer].contains(&neighbor_u32) {
                    self.connections[node_id][layer].push(neighbor_u32);
                }

                // Only add bidirectional link if neighbor exists at this layer
                if layer < self.connections[neighbor_id].len() {
                    if !self.connections[neighbor_id][layer].contains(&node_u32) {
                        self.connections[neighbor_id][layer].push(node_u32);
                    }

                    let neighbor_m = self.config.m;
                    if self.connections[neighbor_id][layer].len() > neighbor_m {
                        let neighbor_embedding = self.get_embedding(neighbor_id).to_vec();
                        let neighbor_connections: Vec<usize> = self.connections[neighbor_id][layer]
                            .iter()
                            .map(|&x| x as usize)
                            .collect();
                        let pruned = self.select_neighbors(
                            &neighbor_connections,
                            &neighbor_embedding,
                            neighbor_m,
                            layer,
                        );

                        self.connections[neighbor_id][layer] =
                            pruned.into_iter().map(|x| x as u32).collect();
                    }
                }
            }
        }
    }

    /// Searches a specific layer for nearest neighbors.
    ///
    /// Uses reusable context, fused SIMD distance, prefetching, and batch heap updates.
    #[inline]
    fn search_layer(
        &self,
        query: &[f32],
        entry_points: &[usize],
        ef: usize,
        layer: usize,
        ctx: &mut SearchContext,
        qprep: &QueryPrep,
    ) -> Vec<(f32, usize)> {
        ctx.reset();

        // Filtered search gates only the *result* heap (`best`), and only at layer 0. The frontier
        // (`candidates`) still expands through excluded nodes — they are load-bearing for graph
        // connectivity, and dropping them from the walk would disconnect the allowed nodes behind
        // them and collapse recall. Upper-layer descent (layer > 0) is never filtered: it must
        // navigate freely to reach the right layer-0 neighbourhood. `None` here ⇒ zero-cost
        // (one perfectly-predicted branch per candidate) for every unfiltered search.
        let filter = if layer == 0 { qprep.filter } else { None };
        let admit = |id: usize| filter.is_none_or(|f| f(id));

        // Initialize with entry points
        for &ep in entry_points {
            let dist = self.distance_to_node(query, ep, qprep);
            ctx.distance_calls += 1;
            ctx.candidates.push(Reverse((OrderedFloat(dist), ep)));
            if admit(ep) {
                ctx.best.push((OrderedFloat(dist), ep));
            }
            ctx.mark_visited(ep);
        }

        while let Some(Reverse((current_dist, current_id))) = ctx.candidates.pop() {
            // If current is farther than the ef-th nearest, we're done
            if ctx.best.len() >= ef {
                if let Some(&(furthest_dist, _)) = ctx.best.peek() {
                    if current_dist > furthest_dist {
                        break;
                    }
                }
            }

            // Layer 0 lives in the node arena; upper layers in the nested structure.
            let neighbors_l0_slice;
            let neighbors: &[u32] = if layer == 0 && !self.nodes.is_empty() {
                neighbors_l0_slice = self.get_neighbors_l0(current_id);
                neighbors_l0_slice
            } else if layer < self.connections[current_id].len() {
                &self.connections[current_id][layer]
            } else {
                &[]
            };

            if !neighbors.is_empty() {
                let n_neighbors = neighbors.len();

                // 128d * 4 bytes = 512 bytes = 8 cache lines; prefetch first 3 (192 bytes)
                const PREFETCH_AHEAD: usize = 2;

                // A node's links, norm and vector are one contiguous block, so these
                // prefetches land in a single allocation — one DRAM row, one TLB entry —
                // where the old Struct-of-Arrays layout touched three separate arrays.
                //
                // Prefetch the lines we actually need, not the whole block: the header
                // (count + norm + the first links) and the head of the vector. Issuing a
                // prefetch for all 13 lines of a 784-byte block costs more in instructions
                // than it saves, and measurably slowed the cache-resident case.
                const VECTOR_LINES: usize = 3;
                let stride = self.stride;
                let vec_byte_offset = self.hdr * std::mem::size_of::<u32>();

                // Phase 1: Compute distances for all unvisited neighbors into stack buffer.
                // This separates compute from heap ops for better ILP and fewer branch
                // mispredictions.
                let mut batch_buf: [(f32, usize); 64] = [(0.0, 0); 64];
                let mut batch_count = 0usize;
                let mut overflow = Vec::new();

                for (i, &neighbor_u32) in neighbors.iter().enumerate() {
                    let neighbor_id = neighbor_u32 as usize;

                    unsafe {
                        let lookahead = i + PREFETCH_AHEAD;
                        if lookahead < n_neighbors {
                            let ahead_id = neighbors[lookahead] as usize;
                            let block =
                                self.nodes.as_ptr().wrapping_add(ahead_id * stride) as *const u8;
                            prefetch_read(block);
                            prefetch_embedding(block.wrapping_add(vec_byte_offset), VECTOR_LINES);

                            // Prefetch the visited bitset word for the lookahead neighbor
                            let bitset_ptr =
                                ctx.visited.bits.as_ptr().wrapping_add(ahead_id >> 6) as *const u8;
                            prefetch_read(bitset_ptr);
                        }
                    }

                    if !ctx.is_visited(neighbor_id) {
                        ctx.mark_visited(neighbor_id);
                        let dist = self.distance_to_node(query, neighbor_id, qprep);
                        ctx.distance_calls += 1;
                        if batch_count < batch_buf.len() {
                            batch_buf[batch_count] = (dist, neighbor_id);
                            batch_count += 1;
                        } else {
                            overflow.push((dist, neighbor_id));
                        }
                    }
                }

                // Phase 2: Batch heap updates from the computed distances.
                let mut consider = |dist: f32, neighbor_id: usize| {
                    let dist_ord = OrderedFloat(dist);
                    let allowed = admit(neighbor_id);

                    // The frontier bound is set by the *result* heap. Under a filter `best` holds
                    // only allowed nodes, so a selective filter keeps `best.len() < ef` and every
                    // neighbour is pushed to `candidates` — the walk widens until it has collected
                    // `ef` allowed nodes (or drained the frontier). That is the intended cost of a
                    // selective filter on a graph; the excluded nodes are still expanded, only never
                    // returned.
                    if ctx.best.len() < ef {
                        ctx.candidates.push(Reverse((dist_ord, neighbor_id)));
                        if allowed {
                            ctx.best.push((dist_ord, neighbor_id));
                        }
                    } else if let Some(&(furthest_dist, _)) = ctx.best.peek() {
                        if dist_ord < furthest_dist {
                            ctx.candidates.push(Reverse((dist_ord, neighbor_id)));
                            if allowed {
                                ctx.best.push((dist_ord, neighbor_id));
                                if ctx.best.len() > ef {
                                    ctx.best.pop();
                                }
                            }
                        }
                    }
                };

                for &(dist, neighbor_id) in &batch_buf[..batch_count] {
                    consider(dist, neighbor_id);
                }
                for &(dist, neighbor_id) in &overflow {
                    consider(dist, neighbor_id);
                }
            }
        }

        // Return (distance, id) sorted nearest-first. The distances are already computed;
        // handing back bare ids would force every caller to recompute all `ef` of them.
        let mut results: Vec<(f32, usize)> = ctx
            .best
            .drain()
            .map(|(OrderedFloat(dist), id)| (dist, id))
            .collect();

        results.sort_by(|a, b| a.0.total_cmp(&b.0));
        results
    }

    /// The sequential builder's adapter over [`select_neighbors_core`]. It supplies the graph
    /// access — layer 0 lives in the flat arena, layers >= 1 in `connections` — and nothing else.
    fn select_neighbors(
        &self,
        candidates: &[usize],
        query: &[f32],
        m: usize,
        layer: usize,
    ) -> Vec<usize> {
        let mut scored: Vec<(f32, usize)> = candidates
            .iter()
            .map(|&id| (self.distance(query, self.get_embedding(id)), id))
            .collect();
        scored.sort_by(|a, b| a.0.total_cmp(&b.0));

        select_neighbors_core(
            &scored,
            m,
            self.config.use_heuristic,
            self.config.extend_candidates,
            self.config.keep_pruned_connections,
            |id| {
                if layer == 0 {
                    self.get_neighbors_l0(id)
                        .iter()
                        .map(|&n| n as usize)
                        .collect()
                } else if layer < self.connections[id].len() {
                    self.connections[id][layer]
                        .iter()
                        .map(|&n| n as usize)
                        .collect()
                } else {
                    Vec::new()
                }
            },
            |id| self.distance(query, self.get_embedding(id)),
            |a, b| self.distance(self.get_embedding(a), self.get_embedding(b)),
        )
        .into_iter()
        .map(|(_, id)| id)
        .collect()
    }

    /// Computes distance between two vectors under the configured metric (SIMD accelerated).
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        Self::metric_distance(self.config.metric, a, b)
    }

    /// Distance under an explicit metric. Shared by the sequential and parallel paths.
    #[inline]
    fn metric_distance(metric: DistanceMetric, a: &[f32], b: &[f32]) -> f32 {
        match metric {
            DistanceMetric::Cosine => 1.0 - crate::vector::simd::cosine_similarity_simd(a, b),
            // Squared, not rooted: monotonic in L2, so ordering is identical and the
            // inner loop skips ~8,500 sqrts per query. `score_from_distance` roots it back.
            DistanceMetric::L2 => crate::vector::simd::l2_squared_distance_simd(a, b),
        }
    }

    /// Rotate `query` into RaBitQ space, once, for a whole search or insertion. `None` under
    /// any other storage.
    ///
    /// The result must be computed exactly once per top-level query and threaded through
    /// every [`Self::search_layer`]/[`Self::distance_to_node`] call it makes — calling this
    /// per node visit would replace an O(dim) distance with an O(dim²) matvec on every hop.
    #[inline]
    fn prepare_rabitq_query(&self, query: &[f32]) -> Option<crate::vector::rabitq::PreparedQuery> {
        if self.config.storage != Storage::RaBitQ {
            return None;
        }
        let rq = self
            .rabitq
            .as_ref()
            .expect("RaBitQ storage requires fit_codebook to run before any query");
        // See `rabitq_cosine_input`: under cosine, prepare against a unit-normalized query
        // to match how the codes were encoded — both sides of the estimator must agree on
        // which geometry (raw or normalized) they're operating in.
        let query = self.rabitq_cosine_input(query);
        Some(rq.prepare_query(&query))
    }

    /// Prepare a query for [`Storage::TurboQuant`] traversal: rotate + sketch a unit-normalized
    /// copy so both sides of the estimator share the same (unit-sphere) geometry the codes use.
    /// `None` under any other storage.
    #[allow(deprecated)] // internal handling of Storage::TurboQuant until its 0.8 removal
    fn prepare_turboquant_query(
        &self,
        query: &[f32],
    ) -> Option<crate::vector::turboquant::PreparedQuery> {
        if self.config.storage != Storage::TurboQuant {
            return None;
        }
        let tq = self
            .turboquant
            .as_ref()
            .expect("TurboQuant storage requires fit_codebook to run before any query");
        let mut unit = query.to_vec();
        crate::vector::ops::normalize(&mut unit);
        Some(tq.prepare_query(&unit))
    }

    /// Prepare a query for [`Storage::TurboRabit`] traversal: same cosine convention as
    /// [`Self::prepare_rabitq_query`] (unit-normalize under cosine, raw under L2), so the
    /// estimator's squared-L2 output stays an exact affine function of cosine distance.
    /// `None` under any other storage.
    fn prepare_turborabit_query(
        &self,
        query: &[f32],
    ) -> Option<crate::vector::turborabit::PreparedQuery> {
        // Warren's walk IS TurboRabit's, so it needs the same prepared query — and its rerank
        // additionally needs `rq` to build `R·q`.
        if !matches!(self.config.storage, Storage::TurboRabit | Storage::Warren) {
            return None;
        }
        let tr = self
            .turborabit
            .as_ref()
            .expect("TurboRabit storage requires fit_codebook to run before any query");
        let query = self.rabitq_cosine_input(query);
        Some(tr.prepare_query(&query))
    }

    /// Fused distance from query to a stored node.
    ///
    /// Cosine uses the precomputed norms for a single SIMD dispatch and a single pass.
    /// L2 needs no norms, so `qprep.norm` is ignored there. `qprep.rabitq` must be `Some`
    /// whenever storage is [`Storage::RaBitQ`] — see [`Self::prepare_rabitq_query`].
    ///
    /// Both quantized storages honor `self.config.metric` — they used to silently ignore it
    /// and always compute L2, which under the (default!) `DistanceMetric::Cosine` meant the
    /// entire walk ran under the wrong metric, and `score_from_distance` then treated the
    /// resulting squared-L2 value as a cosine distance, emitting unbounded/negative scores.
    /// No test ever constructed a quantized index with the default metric, so nothing caught
    /// it — classic could-not-fail.
    #[inline]
    #[allow(deprecated)] // internal handling of Storage::TurboQuant until its 0.8 removal
    fn distance_to_node(&self, query: &[f32], node_id: usize, qprep: &QueryPrep) -> f32 {
        // Under SQ8 the walk reads 8-bit codes straight out of the node's own block and
        // never touches `full`. This is the whole point of the storage mode: the block
        // shrinks from 784 bytes to 400, and the search is bound by exactly these reads.
        if self.config.storage == Storage::SQ8 {
            return match self.config.metric {
                DistanceMetric::L2 => crate::vector::simd::sq8_asymmetric_l2_simd(
                    query,
                    self.get_codes(node_id),
                    &self.q_min,
                    &self.q_scale,
                ),
                DistanceMetric::Cosine => {
                    let norm_b = self.get_norm(node_id);
                    if qprep.norm == 0.0 || norm_b == 0.0 {
                        return 1.0;
                    }
                    // The codes are a metric-agnostic reconstruction of the original values
                    // (unlike RaBitQ's estimator below), so cosine here is a genuine
                    // dot-product kernel over the same codes, composed with the norms
                    // already available for free: `qprep.norm` (once per query) and
                    // `get_norm` (in the node's own block, already read this visit).
                    let dot = crate::vector::simd::sq8_asymmetric_dot_simd(
                        query,
                        self.get_codes(node_id),
                        &self.q_min,
                        &self.q_scale,
                    );
                    let similarity = (dot / (qprep.norm * norm_b)).clamp(-1.0, 1.0);
                    1.0 - similarity
                }
            };
        }

        // Under RaBitQ the walk reads one packed sign-bit code plus two scalars, straight out
        // of the node's own block — the block shrinks to 296 bytes (vs SQ8's 400), and the
        // query's rotation was already paid for once, by the caller, in `qprep.rabitq`.
        if self.config.storage == Storage::RaBitQ {
            let prepared = qprep.rabitq.expect(
                "Storage::RaBitQ traversal requires a query prepared via prepare_rabitq_query",
            );
            let (dtc_sq, est_factor, bits) = self.get_rabitq_code(node_id);
            let raw = crate::vector::simd::rabitq_asymmetric_l2_simd(
                prepared.rq(),
                bits,
                dtc_sq,
                est_factor,
                prepared.qn_sq(),
            );
            return match self.config.metric {
                DistanceMetric::L2 => raw,
                // See `rabitq_cosine_input`: both sides were unit-normalized before
                // encoding, so `raw` is `‖â-b̂‖² = 2·cosine_distance` exactly (not an
                // additional approximation on top of the estimator's own error). Clamped
                // defensively — cosine distance is bounded to [0, 2] by definition, and the
                // estimator's own `.max(0.0)` only guards the lower bound.
                DistanceMetric::Cosine => (raw * 0.5).clamp(0.0, 2.0),
            };
        }

        // Under TurboQuant the walk reads `[gamma][qjl bits][mse nibbles]` from the node's own
        // arena block. The MSE term dequantizes the nibble codes through the Lloyd–Max LUT
        // (`nibble_lut_dot_simd`); the QJL term is the 1-bit signed-sum kernel over the
        // sketched query. Codes were encoded from unit-normalized vectors, so the estimate
        // is of cosine similarity. Must match `TurboQuantizer::estimate_ip` exactly — the
        // `packed_walk_matches_module_estimator` tests are the guard.
        if self.config.storage == Storage::TurboQuant {
            let prepared = qprep.turboquant.expect(
                "Storage::TurboQuant traversal requires a query prepared via prepare_turboquant_query",
            );
            let tq = self
                .turboquant
                .as_ref()
                .expect("TurboQuant codebook missing during traversal");
            let v = node_id * self.stride + self.hdr;
            let gamma = f32::from_bits(self.nodes[v]);
            let bit_words = rabitq_bit_words(self.embedding_dim);
            let qjl: &[u8] = bytemuck::cast_slice(&self.nodes[v + 1..v + 1 + bit_words]);
            let s = crate::vector::simd::rabitq_signed_sum(prepared.sq(), qjl);
            let mut ip = gamma * tq.qjl_scale() * s;
            if tq.mse_bits() > 0 {
                let start = v + 1 + bit_words;
                let nib: &[u8] =
                    bytemuck::cast_slice(&self.nodes[start..start + nibble_words(tq.padded_dim())]);
                ip += crate::vector::simd::nibble_lut_dot_simd(prepared.pq(), nib, tq.levels());
            }
            return match self.config.metric {
                // Cosine distance in [0, 2]; the estimate is unbiased but noisy, so clamp.
                DistanceMetric::Cosine => (1.0 - ip).clamp(0.0, 2.0),
                // TODO: norm-aware L2 encoding. For now a monotone-in-cosine proxy — TurboQuant is
                // exercised under cosine in the VIBE sweep, so this path is not on the hot road yet.
                DistanceMetric::L2 => (2.0 * (1.0 - ip)).clamp(0.0, 4.0),
            };
        }

        // Under TurboRabit the walk reads `[dtc_sq][f_rescale][nibble codes]` from the
        // node's own arena block and computes `⟨u, rq⟩` with the FUSED kernel — the
        // Extended-RaBitQ estimate is linear in the code value, so one cvt+FMA per
        // coordinate does what B bit-plane passes did (B×dim FMAs + B reductions → dim
        // FMAs + 1): `dsq = dtc² + qn² + f_rescale·(⟨u,rq⟩ + c_B·Σrq)`, with `c_B·Σrq`
        // precomputed once per query (`cb_sum`). Same metric dispatch as RaBitQ: raw
        // under L2, halved under cosine (both sides unit-normalized before encoding,
        // making `raw = 2·cosine_distance`). Must match
        // `TurboRabitQuantizer::estimate_dist_sq` exactly — guarded by the same tests.
        if matches!(self.config.storage, Storage::TurboRabit | Storage::Warren) {
            let prepared = qprep.turborabit.expect(
                "Storage::TurboRabit traversal requires a query prepared via prepare_turborabit_query",
            );
            let v = node_id * self.stride + self.hdr;
            let dtc_sq = f32::from_bits(self.nodes[v]);
            let f_rescale = f32::from_bits(self.nodes[v + 1]);
            let nib: &[u8] =
                bytemuck::cast_slice(&self.nodes[v + 2..v + 2 + nibble_words(self.embedding_dim)]);
            let dot = crate::vector::simd::nibble_uint_dot_simd(prepared.rq(), nib);
            let raw = (dtc_sq + prepared.qn_sq() + f_rescale * (dot + prepared.cb_sum())).max(0.0);
            return match self.config.metric {
                DistanceMetric::L2 => raw,
                DistanceMetric::Cosine => (raw * 0.5).clamp(0.0, 2.0),
            };
        }

        let embedding = self.get_embedding(node_id);
        match self.config.metric {
            DistanceMetric::Cosine => {
                let norm_b = self.get_norm(node_id);
                // We compute: 1 - dot(q, e) / (||q|| * ||e||)
                if qprep.norm == 0.0 || norm_b == 0.0 {
                    return 1.0;
                }
                crate::vector::simd::cosine_distance_prenorm(query, embedding, norm_b)
            }
            DistanceMetric::L2 => crate::vector::simd::l2_squared_distance_simd(query, embedding),
        }
    }

    /// Exact distance against the full-precision vector, for the rerank stage. Honors
    /// `self.config.metric` — a rerank stage that rescores under a different metric than the
    /// walk used would silently corrupt the final ranking regardless of how good the walk was.
    #[inline]
    fn exact_distance(&self, query: &[f32], node_id: usize) -> f32 {
        let embedding = self.get_embedding(node_id);
        match self.config.metric {
            DistanceMetric::L2 => crate::vector::simd::l2_squared_distance_simd(query, embedding),
            DistanceMetric::Cosine => {
                // `cosine_distance_prenorm` guards both zero-norm cases (query and stored)
                // internally and returns 1.0 — no need to duplicate that check here.
                crate::vector::simd::cosine_distance_prenorm(
                    query,
                    embedding,
                    self.get_norm(node_id),
                )
            }
        }
    }

    /// Map a distance to a similarity score (higher is better), per metric.
    ///
    /// Cosine distance is bounded in [0, 2], so `1 - d` recovers the cosine similarity.
    /// L2 is unbounded, where `1 - d` would emit large negative scores; `1/(1+d)` keeps
    /// the score in (0, 1] and is monotonically decreasing in `d`, so ranking is identical.
    #[inline]
    fn score_from_distance(&self, dist: f32) -> f32 {
        match self.config.metric {
            DistanceMetric::Cosine => 1.0 - dist,
            // `dist` is squared L2 here (see `metric_distance`); root it so the score
            // reflects true Euclidean distance rather than its square.
            DistanceMetric::L2 => 1.0 / (1.0 + dist.max(0.0).sqrt()),
        }
    }

    /// Build an HNSW index from embeddings using parallel construction
    ///
    /// Uses instant-distance's layer-copying approach for safe parallelization:
    /// - All connections stored in zero layer (M*2 neighbors per node)
    /// - Upper layers are read-only snapshots (M neighbors, copied after each batch)
    /// - Process batches top-to-bottom: top batch sequential, rest parallel
    ///
    /// # Arguments
    /// * `embeddings` - Vector of embedding vectors (all must have same dimension)
    /// * `config` - HNSW configuration parameters
    ///
    /// # Returns
    /// A new HNSWIndex built from the embeddings
    /// Build in parallel from full [`Document`]s, preserving each document's id,
    /// content and metadata.
    ///
    /// [`Self::build_parallel`] takes bare embeddings and assigns *synthetic* ids
    /// (`"0"`, `"1"`, …) — it is built for anonymous vector corpora. Callers with
    /// real documents need this.
    ///
    /// # Why the documents are not attached positionally
    ///
    /// The obvious implementation — build, then write `docs[i]` into slot `i` — is
    /// **wrong**, and wrong only sometimes, which is worse. `build_parallel`
    /// randomly shuffles insertion order, and then `finalize_reorder` applies
    /// [`Self::reorder_for_locality`] (on by default), which permutes every node
    /// into breadth-first order. Slot `i` has no relationship to input `i`.
    ///
    /// What makes this safe is that the builder already records provenance: it
    /// writes each node's **original input index** as its id, and the reorder
    /// permutes ids alongside their vectors. So this reads that mapping back
    /// rather than assuming one, and is correct whether or not reordering runs.
    ///
    /// Returns an empty index for an empty input (`build_parallel` panics there).
    pub fn build_parallel_from_documents(documents: Vec<Document>, config: HNSWConfig) -> Self {
        if documents.is_empty() {
            return Self::new(0, config);
        }
        let embedding_dim = documents[0].embedding.len();
        let embeddings: Vec<Vec<f32>> = documents.iter().map(|d| d.embedding.clone()).collect();
        let mut index = Self::build_parallel(embeddings, config);

        // `ids[slot]` is the original input index, written by
        // `convert_parallel_to_index` and carried through the reorder.
        let provenance: Vec<usize> = index
            .ids
            .iter()
            .map(|id| {
                id.parse::<usize>().expect(
                    "build_parallel writes the original input index as each id; \
                     if this fails the builder's id contract has changed",
                )
            })
            .collect();

        for (slot, original) in provenance.into_iter().enumerate() {
            let doc = &documents[original];
            index.ids[slot] = doc.id.clone();
            index.contents[slot] = doc.content.clone();
            index.metadata[slot] = doc.metadata.clone();
        }
        debug_assert_eq!(index.embedding_dim, embedding_dim);
        index
    }

    pub fn build_parallel(embeddings: Vec<Vec<f32>>, config: HNSWConfig) -> Self {
        assert!(!embeddings.is_empty(), "Cannot build from empty embeddings");
        let embedding_dim = embeddings[0].len();
        let n = embeddings.len();

        if n == 1 {
            return Self::build_single(embeddings, config);
        }

        // `M0_MAX`/`M_MAX` are the ZeroNode/UpperNode ARRAY CAPACITIES, not the degree.
        //
        // They used to be the degree too: this builder passed `M0_MAX` everywhere `config.m0`
        // belonged and never read `config.m`/`config.m0` at all, then truncated the finished
        // graph down to `config.m0` in `convert_parallel_to_index`. So the two most important
        // knobs in HNSW were SILENTLY IGNORED by the default builder. Asking for `m0: 24` did
        // not build a degree-24 graph — it built a degree-64 graph at full cost and then threw
        // away 40 edges per node, which is both slower AND a worse graph than a real degree-24
        // build (the heuristic never got to make its choices at 24). Measured: build cost was
        // FLAT at ~61,500 distance computations per insert across m0 = 24/32/48/64, because it
        // was doing the m0=64 build every time.
        let m0 = config.m0.min(M0_MAX);
        let m = config.m.min(M_MAX);
        assert!(
            config.m0 <= M0_MAX && config.m <= M_MAX,
            "parallel build supports m <= {M_MAX} and m0 <= {M0_MAX} (the node arrays are              fixed-size); got m={} m0={}. Use BuildStrategy::Sequential for a larger degree.",
            config.m,
            config.m0
        );

        let ml = config.ml;
        let ef_construction = config.ef_construction;
        let seed = config.seed.unwrap_or_else(rand::random);
        let mut rng = StdRng::seed_from_u64(seed);

        // Calculate batch sizes (how many nodes per layer batch)
        // This determines insertion order, not graph structure
        let mut sizes = Vec::new();
        let mut num = n;
        loop {
            let next = (num as f32 * ml) as usize;
            if next < M_MAX {
                break;
            }
            sizes.push((num - next, num));
            num = next;
        }
        sizes.push((num, num));
        sizes.reverse();
        let num_batches = sizes.len();
        let top = LayerId(num_batches - 1);

        // Shuffle points randomly for insertion order
        assert!(n < u32::MAX as usize);
        let mut shuffled: Vec<(u32, usize)> = (0..n).map(|i| (rng.random::<u32>(), i)).collect();
        shuffled.sort_unstable_by_key(|&(r, _)| r);

        // Reorder embeddings according to shuffle
        let points: Vec<Vec<f32>> = shuffled
            .iter()
            .map(|&(_, idx)| embeddings[idx].clone())
            .collect();

        // Build ranges for each batch
        let mut ranges = Vec::with_capacity(num_batches);
        for (i, (size, cumulative)) in sizes.into_iter().enumerate() {
            let start = cumulative - size;
            let batch_id = LayerId(num_batches - i - 1);
            // Skip first point (it's the entry point, inserted implicitly)
            ranges.push((batch_id, max(start, 1)..cumulative));
        }

        // Zero layer: all nodes, M*2 connections each (the LIVE data)
        let zero: Vec<RwLock<ZeroNode>> =
            (0..n).map(|_| RwLock::new(ZeroNode::default())).collect();

        // Upper layers: snapshots copied after each batch (READ-ONLY during search)
        let mut layers: Vec<Vec<UpperNode>> = vec![Vec::new(); top.0];

        // Search pool for thread-local state reuse
        // The parallel builder used to see NEITHER `use_heuristic` NOR `extend_candidates`:
        // `par_select_heuristic` took only (metric, sorted, points, m, keep_pruned), and the whole
        // parallel path mentioned those two options ZERO times. So setting `use_heuristic: false`
        // or `extend_candidates: true` was silently ignored on the DEFAULT builder — the same bug
        // as `m`/`m0`, on the same path, found the same way (asking "which options reach which
        // code path?" rather than "is this option read anywhere?").
        //
        // The tests for those two options did not catch it because they call `select_neighbors`
        // directly — the SEQUENTIAL path. An option tested against one implementation of a
        // strategy is not a tested option.
        let pool = SearchPool::new(n, config.metric, config.keep_pruned_connections);
        let use_heuristic = config.use_heuristic;
        let extend_candidates = config.extend_candidates;

        // Process batches from top to bottom
        for (batch, range) in ranges {
            let end = range.end;

            if batch.0 == top.0 {
                // Top batch: insert sequentially (forms the backbone)
                for i in range {
                    Self::par_insert(
                        PointId(i as u32),
                        batch,
                        &zero,
                        &layers,
                        &points,
                        &pool,
                        ef_construction,
                        top,
                        m,
                        m0,
                        use_heuristic,
                        extend_candidates,
                    );
                }
            } else {
                // Lower batches: insert in parallel (safe because upper layers are snapshots)
                range.into_par_iter().for_each(|i| {
                    Self::par_insert(
                        PointId(i as u32),
                        batch,
                        &zero,
                        &layers,
                        &points,
                        &pool,
                        ef_construction,
                        top,
                        m,
                        m0,
                        use_heuristic,
                        extend_candidates,
                    );
                });
            }

            // After each batch, snapshot zero layer to create upper layer
            // layers[batch-1] = snapshot of zero[0..end] truncated to M neighbors
            if !batch.is_zero() {
                zero[..end]
                    .par_iter()
                    .map(|z| UpperNode::from_zero(&z.read(), m))
                    .collect_into_vec(&mut layers[batch.0 - 1]);
            }
        }

        // Convert to final index format
        Self::convert_parallel_to_index(zero, layers, points, shuffled, embedding_dim, config, top)
            .finalize_reorder()
    }

    /// Insert a single node during parallel construction
    /// Always updates zero layer; searches use upper layer snapshots + zero layer
    #[allow(clippy::too_many_arguments)]
    fn par_insert(
        new: PointId,
        target_layer: LayerId, // The batch/layer this node belongs to
        zero: &[RwLock<ZeroNode>],
        layers: &[Vec<UpperNode>],
        points: &[Vec<f32>],
        pool: &SearchPool,
        ef_construction: usize,
        top: LayerId,
        m: usize,
        m0: usize,
        use_heuristic: bool,
        extend_candidates: bool,
    ) {
        let metric = pool.metric;
        let keep_pruned = pool.keep_pruned;
        let mut search = pool.pop();
        search.visited.reserve(points.len());

        let point = &points[new.as_usize()];
        search.reset();

        // Start search from entry point (always node 0)
        search.push(PointId(0), point, points);

        // Descend through layers from top to bottom
        for cur in top.descend() {
            // Use ef=1 for greedy descent ABOVE target layer
            // Use ef_construction at target layer and below
            search.ef = if cur.0 <= target_layer.0 {
                ef_construction
            } else {
                1
            };

            if cur.0 > target_layer.0 {
                // Above target layer: search upper layer snapshot, then cull
                if cur.0 <= layers.len() && !layers[cur.0 - 1].is_empty() {
                    search.search_upper(point, &layers[cur.0 - 1], points, m);
                    search.cull();
                }
                // If snapshot doesn't exist, just continue descent
            } else {
                // At or below target layer: search zero layer and BREAK.
                //
                // Keep the FULL `ef_construction`-wide beam here — do NOT truncate to `m0`. The
                // diversity heuristic below can only bridge clusters if it is handed the whole
                // candidate pool to choose from; feeding it the `m0` nearest leaves it nothing to
                // diversify (it picks `m0` from `m0`), so every node connects to its closest
                // same-cluster neighbours and the graph has no long-range edges. That was the
                // parallel builder's ~15-recall-point deficit vs the sequential path, which feeds
                // its heuristic all `ef_construction` candidates. It was also why more
                // `ef_construction` never helped: the pool was truncated to `m0` before selection
                // regardless of how wide the beam explored.
                search.search_zero(point, zero, points, ef_construction);
                break; // Key fix: don't keep searching!
            }
        }

        // Get best candidates from search, diversified via the Algorithm-4
        // heuristic. Using the raw nearest set here (select_simple) connects each
        // node only to its closest same-cluster neighbors, which leaves clusters
        // poorly bridged and tanks recall on structured data. The heuristic keeps
        // a diverse neighbor set so search can cross cluster boundaries — matching
        // the sequential build path.
        let found = Self::par_select_heuristic(
            metric,
            point,
            search.select_simple(),
            points,
            m0,
            keep_pruned,
            use_heuristic,
            extend_candidates,
            zero,
        );

        // Add connections: new node → neighbors (in zero layer)
        {
            let mut node = zero[new.as_usize()].write();
            for (i, candidate) in found.iter().take(m0).enumerate() {
                node.nearest[i] = candidate.pid;
            }
        }

        // Add reverse connections: neighbors → new node (bidirectional)
        for candidate in found.iter().take(m0) {
            Self::add_reverse_connection(
                metric,
                zero,
                points,
                new,
                candidate.pid,
                keep_pruned,
                m0,
                use_heuristic,
            );
        }

        pool.push(search);
    }

    /// Algorithm-4 diversity neighbor selection for the parallel build path.
    ///
    /// `sorted` must be ascending by distance to the new point (as produced by the search's
    /// `nearest` list). Keep a candidate only if it is closer to the new point than to any
    /// already-selected neighbour — that is, drop the ones sitting "behind" a node we have
    /// already taken.
    ///
    /// `keep_pruned` then optionally backfills to `m` from the rejected candidates. Note what
    /// that does: it puts back exactly the neighbours the heuristic just decided were
    /// redundant, which drives every node to the maximum degree `m` and undoes the pruning.
    /// This backfill used to be unconditional here — the config flag was named in a comment
    /// and never read — so the parallel builder (the default) saturated every node to `m0=64`
    /// while faiss, at the same `M`, averages 25. Every hop then scans 2.5x more neighbours,
    /// which is where foxstash's extra ~32% of distance computations per query came from.
    #[allow(clippy::too_many_arguments)]
    /// The parallel builder's adapter over [`select_neighbors_core`]. It supplies the graph
    /// access (a slice of `RwLock<ZeroNode>`) and the distance function, and nothing else.
    ///
    /// `query` is the point the candidates were scored against. It is a parameter because the
    /// previous copy of this algorithm did not have it, and under `extend_candidates` scored each
    /// newly pulled-in candidate against the candidate it came from instead — the wrong distance,
    /// in the field the diversity filter then compares against.
    #[allow(clippy::too_many_arguments)]
    fn par_select_heuristic(
        metric: DistanceMetric,
        query: &[f32],
        sorted: &[Candidate],
        points: &[Vec<f32>],
        m: usize,
        keep_pruned: bool,
        use_heuristic: bool,
        extend_candidates: bool,
        zero: &[RwLock<ZeroNode>],
    ) -> Vec<Candidate> {
        let scored: Vec<(f32, usize)> = sorted
            .iter()
            .map(|c| (c.distance, c.pid.as_usize()))
            .collect();

        select_neighbors_core(
            &scored,
            m,
            use_heuristic,
            extend_candidates,
            keep_pruned,
            |id| zero[id].read().iter().map(|pid| pid.as_usize()).collect(),
            |id| Self::parallel_distance(metric, query, &points[id]),
            |a, b| Self::parallel_distance(metric, &points[a], &points[b]),
        )
        .into_iter()
        .map(|(distance, id)| Candidate {
            distance,
            pid: PointId(id as u32),
        })
        .collect()
    }

    /// Add reverse connection from neighbor to new node, maintaining SORTED order by distance
    /// This is critical: UpperNode::from_zero takes the first M entries, so they must be the M closest
    #[allow(clippy::too_many_arguments)]
    fn add_reverse_connection(
        metric: DistanceMetric,
        zero: &[RwLock<ZeroNode>],
        points: &[Vec<f32>],
        new: PointId,
        neighbor: PointId,
        keep_pruned: bool,
        m0: usize,
        use_heuristic: bool,
    ) {
        let mut node = zero[neighbor.as_usize()].write();
        let neighbor_point = &points[neighbor.as_usize()];
        let count = node.count();

        // Skip if the edge already exists.
        if node.nearest[..count].contains(&new) {
            return;
        }

        if count < m0 {
            // Room available: insert maintaining ascending-distance order.
            let new_dist = Self::parallel_distance(metric, neighbor_point, &points[new.as_usize()]);
            let pos = {
                let mut left = 0;
                let mut right = count;
                while left < right {
                    let mid = (left + right) / 2;
                    let mid_dist = Self::parallel_distance(
                        metric,
                        neighbor_point,
                        &points[node.nearest[mid].as_usize()],
                    );
                    if mid_dist < new_dist {
                        left = mid + 1;
                    } else {
                        right = mid;
                    }
                }
                left
            };
            for i in (pos..count).rev() {
                node.nearest[i + 1] = node.nearest[i];
            }
            node.nearest[pos] = new;
            return;
        }

        // Full: re-diversify {existing neighbors ∪ new} with the Algorithm-4
        // heuristic instead of dropping the furthest. Naive "keep M closest"
        // leaves cluster boundaries unconnected — the core recall bug on
        // structured data. This mirrors the sequential build's reverse-prune.
        let mut cands: Vec<Candidate> = node
            .iter()
            .chain(std::iter::once(new))
            .map(|pid| Candidate {
                distance: Self::parallel_distance(metric, neighbor_point, &points[pid.as_usize()]),
                pid,
            })
            .collect();
        cands.sort_unstable();
        // `extend_candidates: false` here deliberately: this is the reverse-prune of an existing
        // neighbour's list, where the candidate set IS the neighbour's neighbourhood already.
        // Extending it again would re-walk the same graph. Algorithm 4 extends only at insertion.
        let selected = Self::par_select_heuristic(
            metric,
            neighbor_point,
            &cands,
            points,
            m0,
            keep_pruned,
            use_heuristic,
            false,
            zero,
        );

        // Without the backfill the heuristic may return fewer than M0_MAX, so the tail must
        // be padded: `ZeroNode::iter` stops at the first INVALID, and leaving stale ids there
        // would resurrect neighbours we just pruned.
        for (i, slot) in node.nearest.iter_mut().enumerate() {
            *slot = selected.get(i).map_or(PointId(INVALID), |c| c.pid);
        }
    }

    /// Convert parallel construction data to final HNSWIndex format
    fn convert_parallel_to_index(
        zero: Vec<RwLock<ZeroNode>>,
        layers: Vec<Vec<UpperNode>>,
        points: Vec<Vec<f32>>,
        shuffled: Vec<(u32, usize)>,
        embedding_dim: usize,
        config: HNSWConfig,
        top: LayerId,
    ) -> Self {
        let n = points.len();
        let num_layers = top.0 + 1; // Total layers including layer 0
        let zero_final: Vec<ZeroNode> = zero.into_iter().map(|n| n.into_inner()).collect();

        // Build connections from fixed-array format
        // Uses Vec<u32> for cache-friendly traversal (4-5x faster than HashSet)
        let mut connections: Vec<Vec<Vec<u32>>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut node_connections: Vec<Vec<u32>> = Vec::with_capacity(num_layers);

            // Layer 0: from zero layer (M0 connections)
            let layer0: Vec<u32> = zero_final[i].iter().map(|p| p.as_usize() as u32).collect();
            node_connections.push(layer0);

            // Upper layers: ONLY add connections where node actually existed in snapshot
            // Nodes should NOT have fake connections at layers they don't belong to
            // (This matches instant-distance's behavior where late nodes simply don't exist at upper layers)
            for layer in &layers {
                if i < layer.len() {
                    // Node exists in this snapshot - use its actual connections
                    let layer_conns: Vec<u32> =
                        layer[i].iter().map(|p| p.as_usize() as u32).collect();
                    node_connections.push(layer_conns);
                }
                // else: node doesn't exist at this layer - don't add fake connections!
            }

            connections.push(node_connections);
        }

        // Create ID mapping (shuffled index → original index)
        let ids: Vec<String> = shuffled.iter().map(|&(_, orig)| orig.to_string()).collect();

        // The GRAPH was built with exact f32 distances — `points` is right there, and an exact
        // graph is strictly better than one built on lossy codes. Only the *traversal* storage
        // is quantized, below. Layer-0 links still live in `connections`;
        // `migrate_l0_into_arena` hands them to their owner.
        let mut index = Self {
            // Seeded from the same `seed` this build used, so a later incremental `add()` on a
            // bulk-built index continues a reproducible stream rather than starting a random one.
            level_rng: StdRng::seed_from_u64(config.seed.unwrap_or_else(rand::random)),
            embedding_dim,
            stride: node_stride(
                config.m0,
                embedding_dim,
                config.storage,
                config.quant_bits(),
            ),
            hdr: node_hdr_len(config.m0),
            config,
            nodes: Vec::new(),
            connections,
            q_min: Vec::new(),
            q_scale: Vec::new(),
            full: Vec::new(),
            rabitq: None,
            turboquant: None,
            turborabit: None,
            warren_res: Vec::new(),
            warren_rc: Vec::new(),
            ids,
            contents: vec![String::new(); n],
            metadata: vec![None; n],
            entry_point: Some(0),
            max_layer: top.0,
        };
        index.fit_codebook(&points);
        index.nodes.reserve(n * index.stride);
        if index.config.storage != Storage::F32 && index.config.rerank_candidates > 0 {
            index.full.reserve(n * embedding_dim);
        }
        for p in &points {
            index.push_node(p);
        }
        index.migrate_l0_into_arena();
        index.shrink_to_fit();
        index
    }

    /// Build single-node index (trivial case)
    fn build_single(embeddings: Vec<Vec<f32>>, config: HNSWConfig) -> Self {
        let embedding_dim = embeddings[0].len();
        let mut index = Self::new(embedding_dim, config);
        // Same requirement as every other build path: a quantized storage needs a fitted
        // codebook before the first `push_node`. This path is easy to miss because it only
        // runs for a single-vector corpus (`build_parallel` special-cases n == 1) — exactly
        // the kind of default-only-tested gap that let `Storage::SQ8` + `add()` panic.
        index.fit_codebook(&embeddings);
        index.push_node(&embeddings[0]);
        index.connections.push(vec![Vec::new()]);
        index.ids.push("0".to_string());
        index.contents.push(String::new());
        index.metadata.push(None);
        index.entry_point = Some(0);
        index
    }

    /// Distance function for parallel construction (SIMD accelerated)
    #[inline]
    fn parallel_distance(metric: DistanceMetric, a: &[f32], b: &[f32]) -> f32 {
        Self::metric_distance(metric, a, b)
    }
}

impl crate::index::VectorIndex for HNSWIndex {
    fn add(&mut self, document: Document) -> Result<()> {
        self.add(document)
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        self.search(query, k)
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn clear(&mut self) {
        self.clear()
    }

    fn embedding_dim(&self) -> usize {
        self.embedding_dim()
    }
}

impl crate::index::VectorIndexSnapshot for HNSWIndex {
    fn get_all_documents(&self) -> Vec<Document> {
        self.get_all_documents()
    }
}

// ============================================================================
// INSTANT-DISTANCE STYLE PARALLEL CONSTRUCTION
// Uses fixed-size arrays and layer-copying for safe parallelization
// ============================================================================

mod config;
mod layout;
mod snapshot;
pub use config::*;
use layout::*;
use snapshot::*;
mod parallel;
use parallel::*;

#[cfg(test)]
mod tests;
