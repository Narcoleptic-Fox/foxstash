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
        self.index.search_inner(query, k, &mut self.ctx)
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

/// Strategy for building the HNSW index
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BuildStrategy {
    /// Sequential insertion - slower but guarantees high recall (97%+)
    /// Best for: production search, accuracy-critical applications
    #[default]
    Sequential,
    /// Parallel insertion using layer-copying approach
    /// Fast build (6-10x faster), good recall at small scale (<50k)
    /// May have lower recall at larger scales (needs more work)
    Parallel,
    /// Automatically choose based on dataset size
    /// Uses Parallel for <50k vectors (where it works well)
    /// Uses Sequential for larger datasets (reliability over speed)
    Auto,
}

/// Configuration for HNSW index
#[derive(Debug, Clone)]
pub struct HNSWConfig {
    /// Distance metric used for both construction and search.
    ///
    /// Defaults to [`DistanceMetric::Cosine`] for backward compatibility. Set
    /// [`DistanceMetric::L2`] to run against Euclidean benchmarks and datasets —
    /// SIFT, GIST and Deep1B are all L2, and a cosine index scores ~55% against
    /// their ground truth purely because it is answering a different question.
    pub metric: DistanceMetric,

    /// Number of bidirectional links created for each element (except layer 0)
    /// Typical value: 16-32. Higher values increase recall but use more memory.
    pub m: usize,

    /// Number of bidirectional links created for each element in layer 0
    /// Typical value: 2 * m (64 for m=32)
    pub m0: usize,

    /// Size of the dynamic candidate list during construction
    /// Typical value: 100-200. Higher values improve quality but slow down construction.
    pub ef_construction: usize,

    /// Size of the dynamic candidate list during search
    /// Typical value: 50-100. Higher values improve recall but slow down search.
    pub ef_search: usize,

    /// Normalization factor for level generation
    /// Typical value: 1.0 / ln(m) ≈ 0.29 for m=32
    pub ml: f32,

    /// Use the heuristic neighbor selection algorithm (Algorithm 4 from paper)
    /// When true, selects diverse neighbors that aren't "behind" already-selected ones.
    /// This improves graph connectivity and recall at slight construction cost.
    pub use_heuristic: bool,

    /// When using heuristic, also consider neighbors of candidates (extend_candidates)
    /// This can find better neighbors but increases construction time.
    pub extend_candidates: bool,

    /// When pruning, keep some pruned connections for better connectivity
    /// Only applies when use_heuristic is true.
    pub keep_pruned_connections: bool,

    /// Build strategy: Sequential (high recall), Parallel (faster), or Auto
    pub build_strategy: BuildStrategy,

    /// Random seed for reproducible builds (None = random)
    pub seed: Option<u64>,

    /// What the traversal reads for each node's vector. See [`Storage`].
    pub storage: Storage,

    /// Candidates rescored against full-precision vectors before returning `k`.
    ///
    /// Only meaningful under [`Storage::SQ8`]: the coarse walk ranks by an approximate
    /// distance, so its top-`k` is not necessarily the true top-`k`. Rescoring a pool of this
    /// size with exact distances recovers it. The pool reads come from a cold array and are
    /// `O(pool)` per query, against `O(nodes visited)` for the walk — which is why the walk
    /// gets to keep its small blocks.
    ///
    /// **Set to 0 to drop the full-precision vectors entirely.** That is the memory-optimal
    /// configuration: the index then stores only 8-bit codes and pays no `f32` array at all.
    /// It costs whatever recall the approximate ranking loses, which is a real trade and
    /// should be measured on your data, not assumed.
    pub rerank_candidates: usize,
}

/// Distance metric for [`HNSWIndex`].
///
/// The quantized indexes (`SQ8HNSWIndex`, `RaBitQHNSWIndex`) are L2-only. Before this
/// enum existed `HNSWIndex` was cosine-only, so swapping index type to save memory
/// silently changed the metric. Set this explicitly to keep them consistent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum DistanceMetric {
    /// `1 - cosine_similarity`. Magnitude-invariant; the historical default.
    #[default]
    Cosine,
    /// Euclidean. Ranked by *squared* L2 internally (monotonic, so ordering is
    /// identical) — what SIFT/GIST/Deep1B ground truth is computed with.
    L2,
}

/// What the graph traversal reads for each node's vector.
///
/// HNSW search is **memory-latency bound**: a distance computation costs 77-98 ns on SIFT1M,
/// which is essentially one DRAM round-trip. Foxstash already computes distances *faster* than
/// faiss (84 ns vs 98 ns) and still loses, because it issues more of them and each one waits on
/// memory. The only remaining lever is to move fewer bytes per node visit.
///
/// With `m0 = 64` and 128 dimensions, one node block is:
///
/// | storage | header + links | vector | block |
/// |---|---|---|---|
/// | `F32` | 272 B | 512 B | **784 B** |
/// | `SQ8` | 272 B | 128 B | **400 B** |
/// | `RaBitQ` | 272 B | 24 B | **296 B** |
///
/// Note the adjacency does not shrink, so this is ~2x less traffic under SQ8, not 4x —
/// quantization cannot buy more than the vector's share of the block. `RaBitQ`'s vector share
/// is 16 bytes of packed sign bits (1 bit/dim) plus 8 bytes for the two per-vector estimator
/// scalars (`dtc_sq`, `est_factor`) — see [`Storage::RaBitQ`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum Storage {
    /// Full-precision `f32` vectors in the arena. Exact distances, largest blocks.
    #[default]
    F32,
    /// 8-bit codes in the arena, with per-dimension `min`/`scale` fitted over the corpus.
    ///
    /// Traversal computes an *asymmetric* distance: the query stays `f32` and each database
    /// value is dequantized on the fly, so error enters on one side only. The full-precision
    /// vectors are retained in a cold side array and used to rescore the final candidates —
    /// they are never touched during the walk.
    SQ8,
    /// 1-bit-per-dimension RaBitQ codes in the arena (see [`crate::vector::rabitq`]).
    ///
    /// Each node stores a packed sign-bit vector plus two `f32` scalars (`dtc_sq`,
    /// `est_factor`) that the RaBitQ derivation needs to turn those bits into a real distance
    /// estimate — not a Hamming proxy. Traversal is asymmetric in the same sense as `SQ8`:
    /// the query is rotated into RaBitQ space once per query
    /// ([`RaBitQuantizer::prepare_query`](crate::vector::rabitq::RaBitQuantizer::prepare_query)),
    /// and every node visit thereafter is O(dim) with no further matrix work. As with `SQ8`,
    /// the full-precision vectors live in a cold side array and are read only during rerank.
    ///
    /// # Do not reach for this at low dimension
    ///
    /// On SIFT1M (**128-d**) it is **~12x slower than [`Storage::SQ8`] at matched recall**: it
    /// buys a 7% cheaper distance (66.6 → 61.9 ns) and pays **10x more distance computations**
    /// (13,085 vs ~1,250 for ~93% recall@10), because the coarse metric misleads the graph walk.
    /// Without rerank it collapses to **26% recall**. Use `SQ8`.
    ///
    /// The reason is dimensional, and it is why this variant still exists. A node block is
    /// `header(m0) + vector`, and the header is 272 B at m0=64 regardless of `dim`. At 128-d the
    /// SQ8 vector is already only 128 B, so 1-bit codes fight for 104 B out of a 400 B block and
    /// wreck the metric to get it. At 1536-d the SQ8 vector is 1,536 B of an 1,808 B block, and
    /// RaBitQ would cut that block to **472 B — 3.8x below SQ8** — with the adjacency as noise.
    ///
    /// | dim | SQ8 block | RaBitQ block |
    /// |---|---|---|
    /// | 128 | 400 B | 296 B |
    /// | 384 | 656 B | 320 B |
    /// | 1536 | 1,808 B | **472 B** |
    ///
    /// So the SIFT result is a *low-dimension* result and must not be generalized. It is
    /// **unverified** at the dimensionality real embeddings actually use (384–1536). Until
    /// someone runs GIST1M (960-d) or a real 768/1536-d corpus, treat this as: loses badly at
    /// low dim, untested where it should win. See `benchmarks/RESULTS.md`.
    RaBitQ,
}

impl Default for HNSWConfig {
    fn default() -> Self {
        let m = 32; // Match instant-distance for good recall
        Self {
            metric: DistanceMetric::default(),
            m,
            m0: m * 2,
            ef_construction: 100,
            ef_search: 100,
            ml: 1.0 / (m as f32).ln(),
            use_heuristic: true,
            extend_candidates: false,
            keep_pruned_connections: true,
            build_strategy: BuildStrategy::default(),
            seed: None,
            storage: Storage::default(),
            rerank_candidates: 100,
        }
    }
}

impl HNSWConfig {
    /// Use simple nearest-neighbor selection (faster construction, lower recall)
    pub fn with_simple_selection(mut self) -> Self {
        self.use_heuristic = false;
        self
    }

    /// Enable extended candidate search (better quality, slower construction)
    pub fn with_extended_candidates(mut self) -> Self {
        self.extend_candidates = true;
        self
    }

    /// Set ef_search parameter
    pub fn with_ef_search(mut self, ef: usize) -> Self {
        self.ef_search = ef;
        self
    }

    /// Set build strategy
    /// - Sequential: slower but guarantees high recall (97%+)
    /// - Parallel: faster using instant-distance's layer-copying approach
    /// - Auto: Sequential for <50k vectors, Parallel for larger
    pub fn with_build_strategy(mut self, strategy: BuildStrategy) -> Self {
        self.build_strategy = strategy;
        self
    }

    /// Set random seed for reproducible builds
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set ef_construction parameter
    pub fn with_ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = ef;
        self
    }

    /// Set M parameter (connections per node)
    pub fn with_m(mut self, m: usize) -> Self {
        // The old `m <= 127` cap existed only because the layer-0 neighbour count was
        // stored as a `u8`. It lives in the node arena as a `u32` now, so the cap is gone.
        self.m = m;
        self.m0 = m * 2;
        self.ml = 1.0 / (m as f32).ln();
        self
    }
}

/// Where an [`HNSWIndex`]'s memory actually goes. See [`HNSWIndex::memory_breakdown`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryBreakdown {
    /// Contiguous f32 vectors — the irreducible cost of the data itself.
    pub embeddings: usize,
    /// Precomputed L2 norms. Used by the cosine metric; dead weight under L2.
    pub norms: usize,
    /// Flat layer-0 adjacency plus its per-node counts. The hot path, and the bulk of the graph.
    pub layer0_links: usize,
    /// Nested adjacency for layers >= 1, including per-`Vec` headers.
    pub upper_layer_links: usize,
    /// Document ids and contents.
    pub payload: usize,
}

impl MemoryBreakdown {
    /// Total retained bytes.
    pub fn total(&self) -> usize {
        self.embeddings + self.norms + self.layer0_links + self.upper_layer_links + self.payload
    }
}

/// Header size, in 4-byte units, of one node block. See [`HNSWIndex::nodes`].
///
/// `count | norm | m0 + 1 neighbour slots`, rounded up to a 16-byte boundary so the
/// vector that follows stays SIMD-aligned. The spare neighbour slot lets insertion
/// push-then-prune in place rather than spilling to a `Vec`.
#[inline(always)]
const fn node_hdr_len(m0: usize) -> usize {
    (2 + m0 + 1).div_ceil(4) * 4
}

/// Words needed for `dim` packed RaBitQ sign bits (1 bit/dim, byte-packed then word-rounded).
///
/// Composed as `dim -> bytes -> words` (not a single `div_ceil(32)`) so it matches exactly
/// how [`RaBitCode::bits`](crate::vector::rabitq::RaBitCode::bits) packs — `bits[i/8]` bit
/// `i%8` — which is byte-granular, not word-granular.
#[inline(always)]
const fn rabitq_bit_words(dim: usize) -> usize {
    dim.div_ceil(8).div_ceil(4)
}

/// Size, in 4-byte units, of the vector region of a node block.
///
/// `F32` needs one word per dimension. `SQ8` packs one byte per dimension, rounded up to a
/// whole word. `RaBitQ` packs one *bit* per dimension plus two `f32` scalars (`dtc_sq`,
/// `est_factor`) that its distance estimator needs per vector — see [`Storage::RaBitQ`]. In
/// every case the codes sit in the same arena as the links, so a node visit still touches
/// exactly one contiguous block.
#[inline(always)]
const fn vec_words(storage: Storage, dim: usize) -> usize {
    match storage {
        Storage::F32 => dim,
        Storage::SQ8 => dim.div_ceil(4),
        Storage::RaBitQ => 2 + rabitq_bit_words(dim),
    }
}

/// Size, in 4-byte units, of one node block.
#[inline(always)]
const fn node_stride(m0: usize, dim: usize, storage: Storage) -> usize {
    node_hdr_len(m0) + vec_words(storage, dim)
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
}

impl HNSWIndex {
    /// Creates a new HNSW index with custom configuration
    ///
    /// # Arguments
    /// * `embedding_dim` - Dimensionality of embedding vectors
    /// * `config` - HNSW configuration parameters
    pub fn new(embedding_dim: usize, config: HNSWConfig) -> Self {
        Self {
            embedding_dim,
            stride: node_stride(config.m0, embedding_dim, config.storage),
            hdr: node_hdr_len(config.m0),
            config,
            nodes: Vec::new(),
            connections: Vec::new(),
            q_min: Vec::new(),
            q_scale: Vec::new(),
            full: Vec::new(),
            rabitq: None,
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
                self.rabitq = Some(crate::vector::rabitq::RaBitQuantizer::fit(embeddings));
            }
        }
    }

    /// Whether this index's storage has what it needs to encode a vector.
    ///
    /// Always `true` under `Storage::F32` — an `f32` vector needs no fitted state to store.
    /// Under `SQ8`/`RaBitQ` this is `true` once [`Self::fit_codebook`] has actually run
    /// (via `build()`/`build_parallel()`, or [`Self::train`]) — checked by looking at the
    /// codebook state itself rather than a separate `bool` flag, so there is no second
    /// source of truth that could drift out of sync with it.
    fn is_trained(&self) -> bool {
        match self.config.storage {
            Storage::F32 => true,
            Storage::SQ8 => !self.q_scale.is_empty(),
            Storage::RaBitQ => self.rabitq.is_some(),
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

    /// Build an HNSW index from embeddings using the configured strategy
    ///
    /// This is the recommended way to create an index from bulk embeddings.
    /// The build strategy is controlled by `config.build_strategy`:
    /// - `Sequential`: Slower but guarantees high recall (97%+)
    /// - `Parallel`: Faster using instant-distance's layer-copying approach
    /// - `Auto`: Sequential for <50k vectors, Parallel for larger
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

        let n = embeddings.len();
        let strategy = match config.build_strategy {
            BuildStrategy::Auto => {
                // Parallel works well for <50k, Sequential for larger
                if n < 50_000 {
                    BuildStrategy::Parallel
                } else {
                    BuildStrategy::Sequential
                }
            }
            other => other,
        };

        match strategy {
            BuildStrategy::Sequential => Self::build_sequential(embeddings, config),
            BuildStrategy::Parallel | BuildStrategy::Auto => {
                Self::build_parallel(embeddings, config)
            }
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

        // Pre-allocate
        index
            .nodes
            .reserve(n * node_stride(index.config.m0, embedding_dim, index.config.storage));
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
        index.shrink_to_fit();
        index
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
    fn get_embedding(&self, node_id: usize) -> &[f32] {
        match self.config.storage {
            Storage::F32 => {
                let start = node_id * self.stride + self.hdr;
                bytemuck::cast_slice(&self.nodes[start..start + self.embedding_dim])
            }
            Storage::SQ8 | Storage::RaBitQ => {
                debug_assert!(
                    !self.full.is_empty(),
                    "full-precision vectors were dropped (rerank_candidates = 0); \
                     no exact embedding exists to return"
                );
                let start = node_id * self.embedding_dim;
                &self.full[start..start + self.embedding_dim]
            }
        }
    }

    /// The 8-bit codes for a node, from the hot arena. `Storage::SQ8` only.
    #[inline(always)]
    fn get_codes(&self, node_id: usize) -> &[u8] {
        let start = node_id * self.stride + self.hdr;
        let words = vec_words(Storage::SQ8, self.embedding_dim);
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

    /// Get layer 0 neighbours. Same cache lines as the node's vector.
    #[inline(always)]
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

    /// Append a node block: zero neighbours, norm and vector filled in.
    ///
    /// Every construction path must go through this. The sequential builder previously
    /// pushed the vector and forgot to grow the layer-0 storage, which panicked on every
    /// input; a single append keeps the arena's invariant impossible to half-satisfy.
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
                let words = vec_words(Storage::SQ8, self.embedding_dim);
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
                let code = rq.encode(embedding);
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
        if document.embedding.len() != self.embedding_dim {
            return Err(crate::RagError::DimensionMismatch {
                expected: self.embedding_dim,
                actual: document.embedding.len(),
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

        if document.embedding.iter().any(|v| !v.is_finite()) {
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

        self.push_node(&document.embedding);
        self.connections.push(node_connections);
        self.ids.push(document.id);
        self.contents.push(document.content);
        self.metadata.push(document.metadata);

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
        let mut ctx = SearchContext::new(self.len());
        self.search_inner(query, k, &mut ctx)
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
                |ctx, query| self.search_inner(query, k, ctx),
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
        let qprep = QueryPrep {
            norm: query_norm,
            rabitq: rq_prepared.as_ref(),
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
            for entry in found.iter_mut() {
                entry.0 = self.exact_distance(query, entry.1);
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
        let hot_vectors = n * vec_words(self.config.storage, self.embedding_dim) * 4;
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

    /// Generates a random level for a new node using exponential decay.
    ///
    /// Clamps the uniform sample to `[EPSILON, 1)` to prevent `ln(0.0) = -inf`.
    fn random_level(&self) -> usize {
        let mut rng = rand::rng();
        let uniform: f32 = rng.random::<f32>().max(f32::EPSILON);
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
        let qprep = QueryPrep {
            norm: query_norm,
            rabitq: rq_prepared.as_ref(),
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

        // Initialize with entry points
        for &ep in entry_points {
            let dist = self.distance_to_node(query, ep, qprep);
            ctx.distance_calls += 1;
            ctx.candidates.push(Reverse((OrderedFloat(dist), ep)));
            ctx.best.push((OrderedFloat(dist), ep));
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

                    if ctx.best.len() < ef {
                        ctx.candidates.push(Reverse((dist_ord, neighbor_id)));
                        ctx.best.push((dist_ord, neighbor_id));
                    } else if let Some(&(furthest_dist, _)) = ctx.best.peek() {
                        if dist_ord < furthest_dist {
                            ctx.candidates.push(Reverse((dist_ord, neighbor_id)));
                            ctx.best.push((dist_ord, neighbor_id));

                            if ctx.best.len() > ef {
                                ctx.best.pop();
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

    /// Selects M best neighbors using a heuristic
    ///
    /// When `use_heuristic` is enabled (default), uses Algorithm 4 from the HNSW paper
    /// which ensures diversity by only selecting candidates that are closer to the query
    /// than to any already-selected neighbor. This prevents selecting neighbors that are
    /// "behind" other neighbors, improving graph connectivity.
    ///
    /// # Arguments
    /// * `candidates` - Candidate neighbor IDs
    /// * `query` - Query point
    /// * `m` - Number of neighbors to select
    /// * `layer` - Current layer
    fn select_neighbors(
        &self,
        candidates: &[usize],
        query: &[f32],
        m: usize,
        layer: usize,
    ) -> Vec<usize> {
        if !self.config.use_heuristic {
            // Simple heuristic: select M closest neighbors
            return self.select_neighbors_simple(candidates, query, m);
        }

        // Algorithm 4 from the HNSW paper: SELECT-NEIGHBORS-HEURISTIC
        // This ensures diversity by checking if each candidate is closer to query
        // than to any already-selected neighbor.

        // Optionally extend candidates with their neighbors
        let mut working_candidates: Vec<usize> = candidates.to_vec();
        if self.config.extend_candidates {
            let mut seen: HashSet<usize> = candidates.iter().copied().collect();
            for &candidate in candidates {
                // Layer 0 lives in the flat array; layers >= 1 in the nested structure.
                let neighbors: Vec<u32> = if layer == 0 {
                    self.get_neighbors_l0(candidate).to_vec()
                } else if layer < self.connections[candidate].len() {
                    self.connections[candidate][layer].clone()
                } else {
                    Vec::new()
                };
                for neighbor_u32 in neighbors {
                    let neighbor = neighbor_u32 as usize;
                    if seen.insert(neighbor) {
                        working_candidates.push(neighbor);
                    }
                }
            }
        }

        // Score and sort candidates by distance to query
        let mut scored: Vec<(f32, usize)> = working_candidates
            .iter()
            .map(|&id| {
                let dist = self.distance(query, self.get_embedding(id));
                (dist, id)
            })
            .collect();
        scored.sort_by(|a, b| a.0.total_cmp(&b.0));

        // Select neighbors using the heuristic
        let mut selected: Vec<usize> = Vec::with_capacity(m);
        let mut pruned: Vec<(f32, usize)> = Vec::new();

        for (dist_to_query, candidate_id) in scored {
            if selected.len() >= m {
                break;
            }

            // Check if this candidate is closer to query than to any selected neighbor
            let candidate_embedding = self.get_embedding(candidate_id);
            let mut is_good = true;

            for &selected_id in &selected {
                let selected_embedding = self.get_embedding(selected_id);
                let dist_to_selected = self.distance(candidate_embedding, selected_embedding);

                // If candidate is closer to a selected neighbor than to query,
                // it's "behind" that neighbor and we should skip it
                if dist_to_selected < dist_to_query {
                    is_good = false;
                    pruned.push((dist_to_query, candidate_id));
                    break;
                }
            }

            if is_good {
                selected.push(candidate_id);
            }
        }

        // Optionally add back some pruned connections if we didn't get enough
        if self.config.keep_pruned_connections && selected.len() < m {
            for (_, pruned_id) in pruned {
                if selected.len() >= m {
                    break;
                }
                if !selected.contains(&pruned_id) {
                    selected.push(pruned_id);
                }
            }
        }

        selected
    }

    /// Simple neighbor selection: just pick M closest
    #[inline]
    fn select_neighbors_simple(&self, candidates: &[usize], query: &[f32], m: usize) -> Vec<usize> {
        let mut scored: Vec<(f32, usize)> = candidates
            .iter()
            .map(|&id| {
                let dist = self.distance(query, self.get_embedding(id));
                (dist, id)
            })
            .collect();

        scored.sort_by(|a, b| a.0.total_cmp(&b.0));
        scored.truncate(m);
        scored.into_iter().map(|(_, id)| id).collect()
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
        Some(rq.prepare_query(query))
    }

    /// Fused distance from query to a stored node.
    ///
    /// Cosine uses the precomputed norms for a single SIMD dispatch and a single pass.
    /// L2 needs no norms, so `qprep.norm` is ignored there. `qprep.rabitq` must be `Some`
    /// whenever storage is [`Storage::RaBitQ`] — see [`Self::prepare_rabitq_query`].
    #[inline]
    fn distance_to_node(&self, query: &[f32], node_id: usize, qprep: &QueryPrep) -> f32 {
        // Under SQ8 the walk reads 8-bit codes straight out of the node's own block and
        // never touches `full`. This is the whole point of the storage mode: the block
        // shrinks from 784 bytes to 400, and the search is bound by exactly these reads.
        if self.config.storage == Storage::SQ8 {
            return crate::vector::simd::sq8_asymmetric_l2_simd(
                query,
                self.get_codes(node_id),
                &self.q_min,
                &self.q_scale,
            );
        }

        // Under RaBitQ the walk reads one packed sign-bit code plus two scalars, straight out
        // of the node's own block — the block shrinks to 296 bytes (vs SQ8's 400), and the
        // query's rotation was already paid for once, by the caller, in `qprep.rabitq`.
        if self.config.storage == Storage::RaBitQ {
            let prepared = qprep.rabitq.expect(
                "Storage::RaBitQ traversal requires a query prepared via prepare_rabitq_query",
            );
            let (dtc_sq, est_factor, bits) = self.get_rabitq_code(node_id);
            return crate::vector::simd::rabitq_asymmetric_l2_simd(
                prepared.rq(),
                bits,
                dtc_sq,
                est_factor,
                prepared.qn_sq(),
            );
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

    /// Exact squared-L2 against the full-precision vector, for the rerank stage.
    #[inline]
    fn exact_distance(&self, query: &[f32], node_id: usize) -> f32 {
        crate::vector::simd::l2_squared_distance_simd(query, self.get_embedding(node_id))
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
    pub fn build_parallel(embeddings: Vec<Vec<f32>>, config: HNSWConfig) -> Self {
        assert!(!embeddings.is_empty(), "Cannot build from empty embeddings");
        let embedding_dim = embeddings[0].len();
        let n = embeddings.len();

        if n == 1 {
            return Self::build_single(embeddings, config);
        }

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
        let pool = SearchPool::new(n, config.metric, config.keep_pruned_connections);

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
                    );
                });
            }

            // After each batch, snapshot zero layer to create upper layer
            // layers[batch-1] = snapshot of zero[0..end] truncated to M neighbors
            if !batch.is_zero() {
                zero[..end]
                    .par_iter()
                    .map(|z| UpperNode::from_zero(&z.read()))
                    .collect_into_vec(&mut layers[batch.0 - 1]);
            }
        }

        // Convert to final index format
        Self::convert_parallel_to_index(zero, layers, points, shuffled, embedding_dim, config, top)
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
                    search.search_upper(point, &layers[cur.0 - 1], points, M_MAX);
                    search.cull();
                }
                // If snapshot doesn't exist, just continue descent
            } else {
                // At or below target layer: search zero layer and BREAK
                search.search_zero(point, zero, points, M0_MAX);
                break; // Key fix: don't keep searching!
            }
        }

        // Get best candidates from search, diversified via the Algorithm-4
        // heuristic. Using the raw nearest set here (select_simple) connects each
        // node only to its closest same-cluster neighbors, which leaves clusters
        // poorly bridged and tanks recall on structured data. The heuristic keeps
        // a diverse neighbor set so search can cross cluster boundaries — matching
        // the sequential build path.
        let found =
            Self::par_select_heuristic(metric, search.select_simple(), points, M0_MAX, keep_pruned);

        // Add connections: new node → neighbors (in zero layer)
        {
            let mut node = zero[new.as_usize()].write();
            for (i, candidate) in found.iter().take(M0_MAX).enumerate() {
                node.nearest[i] = candidate.pid;
            }
        }

        // Add reverse connections: neighbors → new node (bidirectional)
        for candidate in found.iter().take(M0_MAX) {
            Self::add_reverse_connection(metric, zero, points, new, candidate.pid, keep_pruned);
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
    fn par_select_heuristic(
        metric: DistanceMetric,
        sorted: &[Candidate],
        points: &[Vec<f32>],
        m: usize,
        keep_pruned: bool,
    ) -> Vec<Candidate> {
        let mut selected: Vec<Candidate> = Vec::with_capacity(m);
        for &cand in sorted {
            if selected.len() >= m {
                break;
            }
            // Keep `cand` unless it is closer to an already-selected neighbor
            // than to the query point (i.e. it sits "behind" a selected node).
            let cand_point = &points[cand.pid.as_usize()];
            let diverse = selected.iter().all(|s| {
                Self::parallel_distance(metric, cand_point, &points[s.pid.as_usize()])
                    >= cand.distance
            });
            if diverse {
                selected.push(cand);
            }
        }

        if keep_pruned && selected.len() < m {
            for &cand in sorted {
                if selected.len() >= m {
                    break;
                }
                if !selected.iter().any(|s| s.pid == cand.pid) {
                    selected.push(cand);
                }
            }
        }
        selected
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
    ) {
        let mut node = zero[neighbor.as_usize()].write();
        let neighbor_point = &points[neighbor.as_usize()];
        let count = node.count();

        // Skip if the edge already exists.
        if node.nearest[..count].contains(&new) {
            return;
        }

        if count < M0_MAX {
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
        let selected = Self::par_select_heuristic(metric, &cands, points, M0_MAX, keep_pruned);

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
            embedding_dim,
            stride: node_stride(config.m0, embedding_dim, config.storage),
            hdr: node_hdr_len(config.m0),
            config,
            nodes: Vec::new(),
            connections,
            q_min: Vec::new(),
            q_scale: Vec::new(),
            full: Vec::new(),
            rabitq: None,
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

/// Maximum connections per node in layer 0 (M * 2)
const M0_MAX: usize = 64;
/// Maximum connections per node in upper layers (M)
const M_MAX: usize = 32;
/// Invalid point ID marker
const INVALID: u32 = u32::MAX;

/// Point ID wrapper (u32 for memory efficiency)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct PointId(u32);

impl PointId {
    fn as_usize(self) -> usize {
        self.0 as usize
    }

    fn is_valid(self) -> bool {
        self.0 != INVALID
    }
}

/// Layer 0 node with M*2 fixed connections
#[derive(Clone)]
struct ZeroNode {
    /// Fixed array of neighbor IDs (INVALID = empty slot)
    nearest: [PointId; M0_MAX],
}

impl Default for ZeroNode {
    fn default() -> Self {
        Self {
            nearest: [PointId(INVALID); M0_MAX],
        }
    }
}

impl ZeroNode {
    /// Count of valid neighbors
    fn count(&self) -> usize {
        self.nearest.iter().take_while(|p| p.is_valid()).count()
    }

    /// Iterate over valid neighbors
    fn iter(&self) -> impl Iterator<Item = PointId> + '_ {
        self.nearest.iter().copied().take_while(|p| p.is_valid())
    }
}

/// Upper layer node with M fixed connections
#[derive(Clone)]
struct UpperNode {
    nearest: [PointId; M_MAX],
}

impl Default for UpperNode {
    fn default() -> Self {
        Self {
            nearest: [PointId(INVALID); M_MAX],
        }
    }
}

impl UpperNode {
    /// Create from ZeroNode, truncating to M neighbors
    fn from_zero(zero: &ZeroNode) -> Self {
        let mut node = Self::default();
        for (i, &pid) in zero.nearest.iter().take(M_MAX).enumerate() {
            node.nearest[i] = pid;
        }
        node
    }

    fn iter(&self) -> impl Iterator<Item = PointId> + '_ {
        self.nearest.iter().copied().take_while(|p| p.is_valid())
    }
}

/// Visited bitmap with generation counter (O(1) clear)
struct Visited {
    store: Vec<u8>,
    generation: u8,
}

impl Visited {
    fn new(capacity: usize) -> Self {
        Self {
            store: vec![0; capacity],
            generation: 1,
        }
    }

    fn clear(&mut self) {
        if self.generation == 255 {
            self.store.fill(0);
            self.generation = 1;
        } else {
            self.generation += 1;
        }
    }

    fn insert(&mut self, pid: PointId) -> bool {
        let idx = pid.as_usize();
        if self.store[idx] == self.generation {
            false
        } else {
            self.store[idx] = self.generation;
            true
        }
    }

    fn reserve(&mut self, capacity: usize) {
        if self.store.len() < capacity {
            self.store.resize(capacity, 0);
        }
    }
}

/// Candidate for search (distance + point ID)
#[derive(Debug, Clone, Copy, PartialEq)]
struct Candidate {
    distance: f32,
    pid: PointId,
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Sort by distance, then by pid for stability
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.pid.cmp(&other.pid))
    }
}

/// Search state for parallel construction
struct Search {
    /// Metric used for every distance computed during this search.
    metric: DistanceMetric,
    /// Candidates to explore (min-heap by distance)
    candidates: BinaryHeap<Reverse<Candidate>>,
    /// Best results found (sorted by distance)
    nearest: Vec<Candidate>,
    /// Visited nodes
    visited: Visited,
    /// Current ef value
    ef: usize,
}

impl Search {
    fn new(capacity: usize, metric: DistanceMetric) -> Self {
        Self {
            metric,
            candidates: BinaryHeap::new(),
            nearest: Vec::new(),
            visited: Visited::new(capacity),
            ef: 1,
        }
    }

    fn reset(&mut self) {
        self.candidates.clear();
        self.nearest.clear();
        self.visited.clear();
    }

    fn push(&mut self, pid: PointId, point: &[f32], points: &[Vec<f32>]) {
        let distance = HNSWIndex::parallel_distance(self.metric, point, &points[pid.as_usize()]);
        let candidate = Candidate { distance, pid };
        self.candidates.push(Reverse(candidate));
        self.nearest.push(candidate);
        self.visited.insert(pid);
    }

    /// After searching a layer, prepare for the next layer down
    fn cull(&mut self) {
        self.candidates.clear();
        for &candidate in &self.nearest {
            self.candidates.push(Reverse(candidate));
        }
        self.visited.clear();
        for c in &self.nearest {
            self.visited.insert(c.pid);
        }
    }

    /// Search within a layer (generic over layer type)
    fn search_zero(
        &mut self,
        point: &[f32],
        layer: &[RwLock<ZeroNode>],
        points: &[Vec<f32>],
        num: usize,
    ) {
        while let Some(Reverse(candidate)) = self.candidates.pop() {
            if let Some(furthest) = self.nearest.last() {
                if candidate.distance > furthest.distance && self.nearest.len() >= self.ef {
                    break;
                }
            }

            // Explore neighbors
            let node = layer[candidate.pid.as_usize()].read();
            for neighbor_pid in node.iter() {
                if self.visited.insert(neighbor_pid) {
                    let distance = HNSWIndex::parallel_distance(
                        self.metric,
                        point,
                        &points[neighbor_pid.as_usize()],
                    );
                    let new_candidate = Candidate {
                        distance,
                        pid: neighbor_pid,
                    };

                    // Add to candidates if potentially useful
                    let dominated = self.nearest.len() >= self.ef
                        && self
                            .nearest
                            .last()
                            .map(|f| distance > f.distance)
                            .unwrap_or(false);

                    if !dominated {
                        self.candidates.push(Reverse(new_candidate));

                        // Insert into nearest (sorted)
                        let pos = self
                            .nearest
                            .binary_search(&new_candidate)
                            .unwrap_or_else(|i| i);
                        if pos < self.ef {
                            self.nearest.insert(pos, new_candidate);
                            if self.nearest.len() > self.ef {
                                self.nearest.pop();
                            }
                        }
                    }
                }
            }
        }
        self.nearest.truncate(num);
    }

    fn search_upper(
        &mut self,
        point: &[f32],
        layer: &[UpperNode],
        points: &[Vec<f32>],
        num: usize,
    ) {
        if layer.is_empty() {
            return;
        }

        while let Some(Reverse(candidate)) = self.candidates.pop() {
            if let Some(furthest) = self.nearest.last() {
                if candidate.distance > furthest.distance && self.nearest.len() >= self.ef {
                    break;
                }
            }

            // Safety: skip if candidate is beyond current layer snapshot
            if candidate.pid.as_usize() >= layer.len() {
                continue;
            }

            let node = &layer[candidate.pid.as_usize()];
            for neighbor_pid in node.iter() {
                if self.visited.insert(neighbor_pid) {
                    let distance = HNSWIndex::parallel_distance(
                        self.metric,
                        point,
                        &points[neighbor_pid.as_usize()],
                    );
                    let new_candidate = Candidate {
                        distance,
                        pid: neighbor_pid,
                    };

                    let dominated = self.nearest.len() >= self.ef
                        && self
                            .nearest
                            .last()
                            .map(|f| distance > f.distance)
                            .unwrap_or(false);

                    if !dominated {
                        self.candidates.push(Reverse(new_candidate));
                        let pos = self
                            .nearest
                            .binary_search(&new_candidate)
                            .unwrap_or_else(|i| i);
                        if pos < self.ef {
                            self.nearest.insert(pos, new_candidate);
                            if self.nearest.len() > self.ef {
                                self.nearest.pop();
                            }
                        }
                    }
                }
            }
        }
        self.nearest.truncate(num);
    }

    /// Get best candidates (sorted by distance)
    fn select_simple(&self) -> &[Candidate] {
        &self.nearest
    }
}

/// Pool of search states for thread-local reuse
struct SearchPool {
    pool: Mutex<Vec<Search>>,
    capacity: usize,
    metric: DistanceMetric,
    /// Whether to backfill a node's neighbour list to `m` with candidates the diversity
    /// heuristic rejected. See `par_select_heuristic` — this used to be ignored entirely.
    keep_pruned: bool,
}

impl SearchPool {
    fn new(capacity: usize, metric: DistanceMetric, keep_pruned: bool) -> Self {
        Self {
            pool: Mutex::new(Vec::new()),
            capacity,
            keep_pruned,
            metric,
        }
    }

    fn pop(&self) -> Search {
        self.pool
            .lock()
            .pop()
            .unwrap_or_else(|| Search::new(self.capacity, self.metric))
    }

    fn push(&self, mut search: Search) {
        search.reset();
        self.pool.lock().push(search);
    }
}

/// Layer ID wrapper
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct LayerId(usize);

impl LayerId {
    fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// Iterate from this layer down to 0
    fn descend(self) -> impl Iterator<Item = LayerId> {
        (0..=self.0).rev().map(LayerId)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_document(id: &str, embedding: Vec<f32>) -> Document {
        Document {
            id: id.to_string(),
            content: format!("Content for {}", id),
            embedding,
            metadata: None,
        }
    }

    fn generate_random_vector(dim: usize, seed: u64) -> Vec<f32> {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect()
    }

    #[test]
    fn test_hnsw_config_default() {
        let config = HNSWConfig::default();
        assert_eq!(config.m, 32); // Changed from 16 to match instant-distance
        assert_eq!(config.m0, 64); // m * 2
        assert_eq!(config.ef_construction, 100);
        assert_eq!(config.ef_search, 100);
        assert!((config.ml - (1.0 / 32_f32.ln())).abs() < 0.01);
        assert!(config.use_heuristic); // Heuristic enabled by default
        assert!(!config.extend_candidates);
        assert!(config.keep_pruned_connections);
    }

    #[test]
    fn test_hnsw_config_builders() {
        let config = HNSWConfig::default()
            .with_m(32)
            .with_ef_search(100)
            .with_ef_construction(400)
            .with_simple_selection()
            .with_extended_candidates();

        assert_eq!(config.m, 32);
        assert_eq!(config.m0, 64);
        assert_eq!(config.ef_search, 100);
        assert_eq!(config.ef_construction, 400);
        assert!(!config.use_heuristic);
        assert!(config.extend_candidates);
    }

    #[test]
    fn test_hnsw_new() {
        let index = HNSWIndex::with_defaults(128);
        assert_eq!(index.embedding_dim, 128);
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn search_layer_considers_neighbors_beyond_fixed_stack_batch() {
        let mut config = HNSWConfig::default().with_m(64);
        config.m0 = 128;

        let mut index = HNSWIndex::new(2, config);
        let total_nodes = 66usize; // node 0 + 65 neighbors
        index.connections = vec![vec![Vec::new()]; total_nodes];
        index.metadata = vec![None; total_nodes];
        index.entry_point = Some(0);
        index.max_layer = 0;

        for node in 0..total_nodes {
            // Node 65 is the best match for query [1, 0]; the rest are orthogonal to it.
            let v: [f32; 2] = if node == 65 { [1.0, 0.0] } else { [0.0, 1.0] };
            index.push_node(&v);
            index.ids.push(format!("doc-{node}"));
            index.contents.push(String::new());
        }

        // Give node 0 all 65 others as layer-0 neighbours, via the arena's owner.
        let neighbors: Vec<u32> = (1..=65).map(|n| n as u32).collect();
        index.l0_replace(0, &neighbors);

        let mut ctx = SearchContext::new(index.len());
        let qprep = QueryPrep {
            norm: 1.0,
            rabitq: None,
        };
        let candidates = index.search_layer(&[1.0, 0.0], &[0], 66, 0, &mut ctx, &qprep);
        assert!(
            candidates.iter().any(|&(_, id)| id == 65),
            "best neighbor from position >64 should be considered"
        );
    }

    #[test]
    fn test_add_single_document() {
        let mut index = HNSWIndex::with_defaults(3);
        let doc = create_test_document("doc1", vec![1.0, 0.0, 0.0]);

        assert!(index.add(doc).is_ok());
        assert_eq!(index.len(), 1);
        assert!(!index.is_empty());
    }

    #[test]
    fn test_add_dimension_mismatch() {
        let mut index = HNSWIndex::with_defaults(3);
        let doc = create_test_document("doc1", vec![1.0, 0.0]); // Wrong dimension

        assert!(index.add(doc).is_err());
    }

    // ========================================================================
    // train() / is_trained() — quantized storages must not panic on add()
    // ========================================================================

    /// `Storage::SQ8` + `new()` + `add()`, skipping `train()`, must return `Err`, never
    /// panic. This is the exact bug that used to crash on the first `add()`: `push_node`
    /// indexed into `q_scale`/`q_min`, which stay empty until a codebook is fit.
    #[test]
    fn sq8_add_without_train_errs_not_panics() {
        let mut index = HNSWIndex::new(
            4,
            HNSWConfig {
                storage: Storage::SQ8,
                rerank_candidates: 100,
                metric: DistanceMetric::L2,
                ..Default::default()
            },
        );
        let err = index
            .add(create_test_document("doc1", vec![0.5, -0.3, 0.8, 0.1]))
            .expect_err("add() before train() must error, not panic");
        assert!(
            matches!(err, crate::RagError::NotTrained(_)),
            "expected NotTrained, got {err:?}"
        );
        let msg = err.to_string();
        assert!(
            msg.contains("train("),
            "error message should name the method to call: {msg}"
        );
    }

    /// Same bug class, `Storage::RaBitQ`: `push_node` calls `self.rabitq.as_ref().expect(...)`,
    /// which would panic identically without this guard.
    #[test]
    fn rabitq_add_without_train_errs_not_panics() {
        let mut index = HNSWIndex::new(
            4,
            HNSWConfig {
                storage: Storage::RaBitQ,
                rerank_candidates: 100,
                metric: DistanceMetric::L2,
                ..Default::default()
            },
        );
        let err = index
            .add(create_test_document("doc1", vec![0.5, -0.3, 0.8, 0.1]))
            .expect_err("add() before train() must error, not panic");
        assert!(
            matches!(err, crate::RagError::NotTrained(_)),
            "expected NotTrained, got {err:?}"
        );
    }

    /// `Storage::F32` needs no codebook, so `add()` without `train()` must keep working
    /// exactly as before — `train()` is a no-op there and must never gate it.
    #[test]
    fn f32_add_works_without_training() {
        let mut index = HNSWIndex::new(4, HNSWConfig::default());
        assert!(index
            .add(create_test_document("doc1", vec![0.5, -0.3, 0.8, 0.1]))
            .is_ok());
        assert_eq!(index.len(), 1);
        assert!(index.search(&[0.5, -0.3, 0.8, 0.1], 1).unwrap()[0].id == "doc1");
    }

    /// End-to-end: `new()` -> `train(sample)` -> incremental `add_embedding()` -> search,
    /// for both quantized storages. Ground truth is brute-force exact L2 over the base set;
    /// queries are HELD OUT (never added), for the same reason as the build_parallel recall
    /// test — self-retrieval can't distinguish a working metric from a broken one.
    fn train_then_add_recall(storage: Storage) -> f32 {
        let mut rng = StdRng::seed_from_u64(303);
        let dim = 16;
        let n_clusters = 8;
        let per_cluster = 40;
        let centers: Vec<Vec<f32>> = (0..n_clusters)
            .map(|_| (0..dim).map(|_| rng.random::<f32>() * 10.0).collect())
            .collect();
        let base: Vec<Vec<f32>> = (0..n_clusters * per_cluster)
            .map(|i| {
                let c = &centers[i % n_clusters];
                c.iter().map(|x| x + rng.random::<f32>() * 0.4).collect()
            })
            .collect();
        let queries: Vec<Vec<f32>> = (0..40)
            .map(|i| {
                let c = &centers[i % n_clusters];
                c.iter().map(|x| x + rng.random::<f32>() * 0.4).collect()
            })
            .collect();

        let mut index = HNSWIndex::new(
            dim,
            HNSWConfig {
                metric: DistanceMetric::L2,
                m: 16,
                m0: 32,
                ef_construction: 150,
                ef_search: 150,
                storage,
                rerank_candidates: 50,
                ..Default::default()
            },
        );
        // Train from a sample (need not be the whole corpus — half of it here), then add
        // every vector incrementally, the way a caller without the full corpus up front
        // would use this API.
        index.train(&base[..base.len() / 2]).expect("train");
        for (i, v) in base.iter().enumerate() {
            index
                .add_embedding(i.to_string(), v.clone())
                .expect("add_embedding");
        }

        let k = 10;
        let mut total_recall = 0.0f32;
        for q in &queries {
            let mut exact: Vec<(f32, usize)> = base
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let d: f32 = v.iter().zip(q).map(|(a, b)| (a - b) * (a - b)).sum();
                    (d, i)
                })
                .collect();
            exact.sort_by(|a, b| a.0.total_cmp(&b.0));
            let truth: HashSet<usize> = exact.iter().take(k).map(|(_, i)| *i).collect();

            let got: HashSet<usize> = index
                .search(q, k)
                .expect("search")
                .into_iter()
                .filter_map(|r| r.id.parse::<usize>().ok())
                .collect();
            total_recall += truth.intersection(&got).count() as f32 / k as f32;
        }
        total_recall / queries.len() as f32
    }

    /// SQ8 via incremental `train()` + `add_embedding()` — not `build_parallel`, so
    /// construction itself uses the quantized metric, a strictly harder case than the
    /// build-time recall tests. Measured on this exact seed/config: 100%; floor set well
    /// below that for margin, not at the measurement (see `lesson_untested_public_options`).
    #[test]
    fn sq8_train_then_add_retrieves_correctly() {
        let recall = train_then_add_recall(Storage::SQ8);
        assert!(
            recall > 0.6,
            "Storage::SQ8 via train()+add_embedding(): recall@10 = {:.1}%, held-out queries, \
             brute-force ground truth",
            recall * 100.0
        );
    }

    /// Same as above for RaBitQ. Measured on this exact seed/config: 100%.
    #[test]
    fn rabitq_train_then_add_retrieves_correctly() {
        let recall = train_then_add_recall(Storage::RaBitQ);
        assert!(
            recall > 0.6,
            "Storage::RaBitQ via train()+add_embedding(): recall@10 = {:.1}%, held-out \
             queries, brute-force ground truth",
            recall * 100.0
        );
    }

    /// `train()` on a non-empty index must refuse — retraining would desynchronize vectors
    /// already encoded under the old codebook.
    #[test]
    fn train_on_nonempty_index_errs() {
        let mut index = HNSWIndex::new(
            3,
            HNSWConfig {
                storage: Storage::SQ8,
                ..Default::default()
            },
        );
        index.train(&[vec![1.0, 2.0, 3.0]]).expect("first train");
        index
            .add_embedding("0".into(), vec![1.0, 2.0, 3.0])
            .expect("add after train");
        assert!(
            index.train(&[vec![4.0, 5.0, 6.0]]).is_err(),
            "retraining a non-empty index must be rejected"
        );
    }

    #[test]
    fn rejects_inf_embedding() {
        let mut index = HNSWIndex::new(3, HNSWConfig::default());
        let doc = Document {
            id: "inf".to_string(),
            content: "test".to_string(),
            embedding: vec![f32::INFINITY, 0.0, 0.0],
            metadata: None,
        };
        assert!(index.add(doc).is_err());

        let doc_neg = Document {
            id: "neg_inf".to_string(),
            content: "test".to_string(),
            embedding: vec![0.0, f32::NEG_INFINITY, 0.0],
            metadata: None,
        };
        assert!(index.add(doc_neg).is_err());
    }

    #[test]
    fn an_undersized_scratch_context_is_regrown_before_use() {
        // `BitsetVisited` indexes with `get_unchecked`. A context sized for a smaller index
        // than the one it is used against is therefore not merely wrong, it is UB in release.
        // `search_inner` guards that with `if ctx.capacity < self.len()`.
        //
        // A *public* caller can no longer reach this state: `search`/`search_batch` size the
        // scratch at the moment of use, and `Searcher` borrows the index, so the index cannot
        // grow while a searcher is alive — the borrow checker retired the runtime hazard.
        // This test drives the guard directly, from inside the module, because the guard is
        // load-bearing for an `unsafe` block and must not be deleted as "unreachable".
        let mut index = HNSWIndex::new(3, HNSWConfig::default());
        for i in 0..64 {
            index
                .add(Document {
                    id: format!("doc-{i}"),
                    content: String::new(),
                    embedding: vec![(i as f32) * 0.01, 1.0 - (i as f32) * 0.01, 0.0],
                    metadata: None,
                })
                .unwrap();
        }

        // Deliberately far too small: one node's worth of bitset for a 64-node index.
        let mut stale = SearchContext::new(1);
        assert!(stale.capacity < index.len());

        let results = index.search_inner(&[1.0, 0.0, 0.0], 5, &mut stale).unwrap();

        assert_eq!(results.len(), 5);
        assert!(
            stale.capacity >= index.len(),
            "search_inner must regrow an undersized context, not index past the end of its bitset"
        );
    }

    #[test]
    fn test_search_empty_index() {
        let index = HNSWIndex::with_defaults(3);
        let query = vec![1.0, 0.0, 0.0];

        let results = index.search(&query, 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_single_document() {
        let mut index = HNSWIndex::with_defaults(3);
        let doc = create_test_document("doc1", vec![1.0, 0.0, 0.0]);
        index.add(doc).unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let results = index.search(&query, 1).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "doc1");
        assert!((results[0].score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_search_multiple_documents() {
        let mut index = HNSWIndex::with_defaults(3);

        // Add documents with different embeddings
        let docs = vec![
            create_test_document("doc1", vec![1.0, 0.0, 0.0]),
            create_test_document("doc2", vec![0.0, 1.0, 0.0]),
            create_test_document("doc3", vec![0.0, 0.0, 1.0]),
            create_test_document("doc4", vec![1.0, 1.0, 0.0]),
        ];

        for doc in docs {
            index.add(doc).unwrap();
        }

        // Query closest to doc1
        let query = vec![1.0, 0.0, 0.0];
        let results = index.search(&query, 2).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "doc1");
        assert!(results[0].score > 0.9);
    }

    #[test]
    fn test_search_exact_match() {
        let mut index = HNSWIndex::with_defaults(3);

        let embedding = vec![0.5, 0.5, 0.7072];
        let doc = create_test_document("doc1", embedding.clone());
        index.add(doc).unwrap();

        let results = index.search(&embedding, 1).unwrap();
        assert_eq!(results.len(), 1);
        assert!((results[0].score - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_clear() {
        let mut index = HNSWIndex::with_defaults(3);

        for i in 0..5 {
            let doc = create_test_document(&format!("doc{}", i), vec![i as f32, 0.0, 0.0]);
            index.add(doc).unwrap();
        }

        assert_eq!(index.len(), 5);

        index.clear();

        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn test_random_dataset_100_vectors() {
        let dim = 128;
        let mut index = HNSWIndex::with_defaults(dim);

        // Add 100 random vectors
        for i in 0..100 {
            let embedding = generate_random_vector(dim, i);
            let doc = create_test_document(&format!("doc{}", i), embedding);
            index.add(doc).unwrap();
        }

        assert_eq!(index.len(), 100);

        // Search with a random query
        let query = generate_random_vector(dim, 9999);
        let results = index.search(&query, 10).unwrap();

        assert_eq!(results.len(), 10);

        // Results should be sorted by score (descending)
        for i in 0..results.len() - 1 {
            assert!(results[i].score >= results[i + 1].score);
        }
    }

    #[test]
    fn test_random_dataset_1000_vectors() {
        let dim = 64;
        let mut index = HNSWIndex::with_defaults(dim);

        // Add 1000 random vectors
        for i in 0..1000 {
            let embedding = generate_random_vector(dim, i);
            let doc = create_test_document(&format!("doc{}", i), embedding);
            index.add(doc).unwrap();
        }

        assert_eq!(index.len(), 1000);

        // Perform multiple searches
        for seed in [111, 222, 333, 444, 555] {
            let query = generate_random_vector(dim, seed);
            let results = index.search(&query, 20).unwrap();

            assert_eq!(results.len(), 20);

            // Verify ordering
            for i in 0..results.len() - 1 {
                assert!(results[i].score >= results[i + 1].score);
            }

            // All scores should be between -1 and 1
            for result in &results {
                assert!(result.score >= -1.0 && result.score <= 1.0);
            }
        }
    }

    #[test]
    fn test_recall_with_known_neighbors() {
        let dim = 32;
        let mut index = HNSWIndex::with_defaults(dim);

        // Create a query vector
        let query = generate_random_vector(dim, 0);

        // Create 100 vectors with varying similarity to query
        for i in 0..100 {
            let mut embedding = generate_random_vector(dim, i + 1);

            // First 10 vectors are more similar to query
            if i < 10 {
                for j in 0..dim {
                    embedding[j] = query[j] * 0.9 + embedding[j] * 0.1;
                }
            }

            let doc = create_test_document(&format!("doc{}", i), embedding);
            index.add(doc).unwrap();
        }

        // Search for top 10
        let results = index.search(&query, 10).unwrap();

        // Count how many of the actual top 10 were found
        let mut recall_count = 0;
        for result in &results {
            let doc_num: usize = result.id.strip_prefix("doc").unwrap().parse().unwrap();
            if doc_num < 10 {
                recall_count += 1;
            }
        }

        // HNSW should find most of the true nearest neighbors
        // Expect at least 70% recall
        assert!(recall_count >= 7, "Recall too low: {}/10", recall_count);
    }

    #[test]
    fn test_search_dimension_mismatch() {
        let mut index = HNSWIndex::with_defaults(3);
        let doc = create_test_document("doc1", vec![1.0, 0.0, 0.0]);
        index.add(doc).unwrap();

        let query = vec![1.0, 0.0]; // Wrong dimension
        assert!(index.search(&query, 1).is_err());
    }

    #[test]
    fn test_metadata_preservation() {
        let mut index = HNSWIndex::with_defaults(3);

        let mut doc = create_test_document("doc1", vec![1.0, 0.0, 0.0]);
        doc.metadata = Some(serde_json::json!({"category": "test", "priority": 5}));

        index.add(doc).unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let results = index.search(&query, 1).unwrap();

        assert_eq!(results.len(), 1);
        assert!(results[0].metadata.is_some());

        let metadata = results[0].metadata.as_ref().unwrap();
        assert_eq!(metadata["category"], "test");
        assert_eq!(metadata["priority"], 5);
    }

    #[test]
    fn test_search_with_nan_query_does_not_panic() {
        let mut index = HNSWIndex::with_defaults(3);
        index
            .add(create_test_document("doc1", vec![1.0, 0.0, 0.0]))
            .unwrap();
        index
            .add(create_test_document("doc2", vec![0.0, 1.0, 0.0]))
            .unwrap();

        let query = vec![f32::NAN, 0.0, 0.0];
        let outcome = std::panic::catch_unwind(|| index.search(&query, 2));

        assert!(outcome.is_ok(), "search panicked when query contains NaN");
    }

    #[test]
    #[should_panic(expected = "All embeddings must have the same dimension")]
    fn test_build_rejects_mismatched_dimensions() {
        let _ = HNSWIndex::build(
            vec![vec![1.0, 0.0, 0.0], vec![1.0, 0.0]],
            HNSWConfig::default(),
        );
    }

    #[test]
    fn test_add_rejects_nan_embedding() {
        let mut index = HNSWIndex::with_defaults(3);
        let doc = create_test_document("nan_doc", vec![1.0, f32::NAN, 0.0]);
        let result = index.add(doc);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("NaN"),
            "Error should mention NaN, got: {}",
            err_msg
        );
    }

    #[test]
    fn test_add_embedding_rejects_nan() {
        let mut index = HNSWIndex::with_defaults(3);
        let result = index.add_embedding("nan_vec".into(), vec![f32::NAN, 0.0, 0.0]);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("NaN"),
            "Error should mention NaN, got: {}",
            err_msg
        );
    }

    #[test]
    fn test_add_rejects_all_nan_embedding() {
        let mut index = HNSWIndex::with_defaults(3);
        let doc = create_test_document("all_nan", vec![f32::NAN, f32::NAN, f32::NAN]);
        assert!(index.add(doc).is_err());
    }

    #[test]
    fn test_zero_vector_accepted_and_searchable() {
        let mut index = HNSWIndex::with_defaults(3);

        // Zero vectors are valid (they just have zero norm)
        let doc_zero = create_test_document("zero", vec![0.0, 0.0, 0.0]);
        assert!(index.add(doc_zero).is_ok());

        let doc_normal = create_test_document("normal", vec![1.0, 0.0, 0.0]);
        assert!(index.add(doc_normal).is_ok());

        // Search should not panic with zero vectors in the index
        let query = vec![1.0, 0.0, 0.0];
        let results = index.search(&query, 2).unwrap();
        assert_eq!(results.len(), 2);

        // The normal vector should rank higher than the zero vector
        assert_eq!(results[0].id, "normal");
    }

    #[test]
    fn test_zero_vector_query_does_not_panic() {
        let mut index = HNSWIndex::with_defaults(3);
        let doc = create_test_document("doc1", vec![1.0, 0.0, 0.0]);
        index.add(doc).unwrap();

        // Zero query should not panic (distance_to_node handles zero norm)
        let query = vec![0.0, 0.0, 0.0];
        let results = index.search(&query, 1).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_random_level_never_panics() {
        // Exercise random_level many times to verify ln(0) is impossible
        let index = HNSWIndex::with_defaults(3);
        for _ in 0..10_000 {
            let _level = index.random_level();
        }
    }

    // ========================================================================
    // Distance metric
    // ========================================================================

    /// Cosine and L2 must genuinely disagree, and each must pick its own winner.
    ///
    /// q = [10, 0]:
    ///   far_same_direction = [100, 0] — cosine distance 0 (identical direction),
    ///                                   but L2 distance 90 (way off in magnitude)
    ///   near_off_axis      = [9, 3]   — cosine distance ~0.051 (direction differs),
    ///                                   but L2 distance ~3.16 (much closer in space)
    ///
    /// A cosine index must return `far_same_direction`; an L2 index must return
    /// `near_off_axis`. This is exactly why scoring foxstash's cosine HNSW against
    /// SIFT's L2 ground truth read 55% for a graph that is actually 97.7% correct.
    #[test]
    fn cosine_and_l2_pick_different_neighbors() {
        let query = vec![10.0, 0.0];
        let docs = [
            ("far_same_direction", vec![100.0, 0.0]),
            ("near_off_axis", vec![9.0, 3.0]),
        ];

        let winner = |metric: DistanceMetric| {
            let mut index = HNSWIndex::new(
                2,
                HNSWConfig {
                    metric,
                    ..Default::default()
                },
            );
            for (id, v) in &docs {
                index
                    .add(Document {
                        id: (*id).to_string(),
                        content: String::new(),
                        embedding: v.clone(),
                        metadata: None,
                    })
                    .unwrap();
            }
            index.search(&query, 1).unwrap()[0].id.clone()
        };

        assert_eq!(winner(DistanceMetric::Cosine), "far_same_direction");
        assert_eq!(winner(DistanceMetric::L2), "near_off_axis");
    }

    /// L2 scores must stay in (0, 1] and decrease with distance. Cosine's `1 - d`
    /// convention would emit large negative scores for unbounded L2 distances.
    #[test]
    fn l2_scores_are_bounded_and_monotonic() {
        let mut index = HNSWIndex::new(
            2,
            HNSWConfig {
                metric: DistanceMetric::L2,
                ..Default::default()
            },
        );
        for (i, v) in [vec![0.0, 0.0], vec![50.0, 0.0], vec![500.0, 0.0]]
            .into_iter()
            .enumerate()
        {
            index
                .add(Document {
                    id: i.to_string(),
                    content: String::new(),
                    embedding: v,
                    metadata: None,
                })
                .unwrap();
        }

        let results = index.search(&[0.0, 0.0], 3).unwrap();
        assert_eq!(results[0].id, "0", "nearest must come first");
        for r in &results {
            assert!(
                r.score > 0.0 && r.score <= 1.0,
                "L2 score {} outside (0, 1]",
                r.score
            );
        }
        for w in results.windows(2) {
            assert!(w[0].score >= w[1].score, "scores must be descending");
        }
    }

    /// Cosine remains the default, so existing code and persisted indexes are unaffected.
    #[test]
    fn cosine_is_still_the_default() {
        assert_eq!(HNSWConfig::default().metric, DistanceMetric::Cosine);
        assert_eq!(DistanceMetric::default(), DistanceMetric::Cosine);
    }

    /// Every build strategy must produce a working graph.
    ///
    /// `BuildStrategy::Sequential` panicked outright for an entire release: the layer-0
    /// refactor made the flat array the sole owner of layer-0 links, and the sequential
    /// builder was never taught to grow it. Nothing caught it — every other test and every
    /// doctest either forces `Parallel` or takes the default, so `Sequential` had no
    /// coverage at all despite being a documented public option.
    ///
    /// This asserts *recall*, not merely absence of a panic: the same refactor left a
    /// `build_l0_cache()` call that would have copied an empty nested layer 0 over the real
    /// graph, erasing every layer-0 link and failing silently with a still-"working" index.
    #[test]
    /// `keep_pruned_connections` must actually do something in **both** builders.
    ///
    /// It did not. The parallel builder — the default path, and the one every benchmark uses —
    /// backfilled each node's neighbour list to `m0` unconditionally: the config flag was named
    /// in a comment above the backfill and never read. So the Algorithm-4 diversity heuristic
    /// ran, correctly pruned, and had its output immediately refilled with the exact candidates
    /// it had just rejected. Every node ended up saturated at `m0` (measured: degree 64.0/64,
    /// against faiss's 25.4 at the same M), and every hop paid for it.
    ///
    /// Nothing caught it because no test ever set the flag to `false` — the same DEFAULT-ONLY
    /// blind spot that let `BuildStrategy::Sequential` panic on every input for a release.
    /// A flag that is only ever exercised at its default is not tested, it is assumed.
    #[test]
    fn keep_pruned_connections_controls_graph_density_in_both_builders() {
        let mut rng = StdRng::seed_from_u64(11);
        let centers: Vec<Vec<f32>> = (0..12)
            .map(|_| (0..24).map(|_| rng.random::<f32>() * 10.0).collect())
            .collect();
        let embeddings: Vec<Vec<f32>> = (0..600)
            .map(|i| {
                let c = &centers[i % 12];
                c.iter().map(|x| x + rng.random::<f32>() * 0.5).collect()
            })
            .collect();

        for strategy in [BuildStrategy::Sequential, BuildStrategy::Parallel] {
            let build = |keep: bool| {
                HNSWIndex::build(
                    embeddings.clone(),
                    HNSWConfig {
                        m: 16,
                        m0: 32,
                        ef_construction: 100,
                        keep_pruned_connections: keep,
                        build_strategy: strategy,
                        seed: Some(3),
                        ..Default::default()
                    },
                )
            };

            let dense = build(true).avg_degree_l0();
            let sparse = build(false).avg_degree_l0();

            assert!(
                sparse < dense,
                "{strategy:?}: keep_pruned_connections has no effect \
                 (degree {sparse:.1} with it off vs {dense:.1} with it on) — \
                 the flag is being ignored and the diversity heuristic's pruning is discarded"
            );
        }
    }

    #[test]
    fn every_build_strategy_produces_a_searchable_graph() {
        // Clustered, not uniform-random: random vectors have no structure to recover, and
        // every ANN scores ~60% on them whether or not its graph is intact.
        let mut rng = StdRng::seed_from_u64(7);
        let centers: Vec<Vec<f32>> = (0..8)
            .map(|_| (0..16).map(|_| rng.random::<f32>() * 10.0).collect())
            .collect();
        let embeddings: Vec<Vec<f32>> = (0..400)
            .map(|i| {
                let c = &centers[i % 8];
                c.iter().map(|x| x + rng.random::<f32>() * 0.3).collect()
            })
            .collect();

        for strategy in [
            BuildStrategy::Sequential,
            BuildStrategy::Parallel,
            BuildStrategy::Auto,
        ] {
            let config = HNSWConfig::default()
                .with_build_strategy(strategy)
                .with_seed(42)
                .with_ef_search(100);
            let index = HNSWIndex::build(embeddings.clone(), config);
            assert_eq!(
                index.len(),
                embeddings.len(),
                "{strategy:?}: wrong node count"
            );

            // Self-retrieval: querying with an indexed vector must return that vector.
            // A graph with its layer-0 links erased fails this immediately.
            let hits = embeddings
                .iter()
                .enumerate()
                .filter(|(i, e)| {
                    index
                        .search(e, 1)
                        .expect("search")
                        .first()
                        .is_some_and(|r| r.id == i.to_string())
                })
                .count();
            let recall = hits as f32 / embeddings.len() as f32;
            assert!(
                recall > 0.95,
                "{strategy:?}: self-retrieval recall {:.1}%, graph is broken",
                recall * 100.0,
            );
        }
    }

    // ========================================================================
    // Storage::RaBitQ
    // ========================================================================

    /// `vec_words`/`node_stride` must match the documented arena layout: 2 scalar words
    /// (`dtc_sq`, `est_factor`) plus `dim` packed sign bits, byte-then-word-rounded — not a
    /// bare `dim.div_ceil(32)`, which would silently disagree with `RaBitCode::bits`' byte
    /// granularity whenever `dim` isn't a multiple of 32 but is a multiple of 8.
    #[test]
    fn rabitq_vec_words_matches_documented_layout() {
        // dim = 40: divisible by 8 (5 bytes) but not by 32, so a naive dim.div_ceil(32)
        // would round to the same 2 words as dim.div_ceil(8).div_ceil(4) here — pick a case
        // where they'd actually differ: dim = 100.
        // bytes = ceil(100/8) = 13, words = ceil(13/4) = 4.
        assert_eq!(rabitq_bit_words(100), 4);
        assert_eq!(vec_words(Storage::RaBitQ, 100), 2 + 4);

        // dim = 128 (SIFT-adjacent): bytes = 16, words = 4 -> vector region = 24 bytes,
        // matching the doc comment on `Storage`.
        assert_eq!(vec_words(Storage::RaBitQ, 128), 2 + 4);
        assert_eq!(vec_words(Storage::RaBitQ, 128) * 4, 24);
    }

    /// End-to-end recall gate for `Storage::RaBitQ`, built the same way the benchmarks do
    /// (`build_parallel`, which builds the graph in exact f32 space and only quantizes the
    /// traversal storage afterward — see `convert_parallel_to_index`).
    ///
    /// Ground truth is exact brute-force L2 over the base set. Queries are HELD OUT: distinct
    /// vectors near (not equal to) cluster centers, never inserted into the index. Querying
    /// with an indexed vector would re-derive that vector's own RaBitQ code and could score
    /// well even against a broken estimator — see `lesson_untestable_by_construction`. Data is
    /// clustered, not uniform-random, for the same reason every other recall test here is:
    /// uniform-random vectors have no structure to lose and every ANN scores ~60% on them
    /// regardless of whether the graph or the metric is correct.
    #[test]
    fn rabitq_recall_on_clustered_data_with_held_out_queries() {
        let mut rng = StdRng::seed_from_u64(2024);
        let dim = 32;
        let n_clusters = 16;
        let per_cluster = 50;
        let centers: Vec<Vec<f32>> = (0..n_clusters)
            .map(|_| (0..dim).map(|_| rng.random::<f32>() * 20.0).collect())
            .collect();

        let base: Vec<Vec<f32>> = (0..n_clusters * per_cluster)
            .map(|i| {
                let c = &centers[i % n_clusters];
                c.iter().map(|x| x + rng.random::<f32>() * 0.8).collect()
            })
            .collect();

        // Held-out queries: fresh noise around the same centers, drawn from the same RNG
        // stream *after* all base vectors, so none of them coincides with a base vector.
        let n_queries = 60;
        let queries: Vec<Vec<f32>> = (0..n_queries)
            .map(|i| {
                let c = &centers[i % n_clusters];
                c.iter().map(|x| x + rng.random::<f32>() * 0.8).collect()
            })
            .collect();

        let config = HNSWConfig {
            metric: DistanceMetric::L2,
            m: 16,
            m0: 32,
            ef_construction: 150,
            ef_search: 150,
            storage: Storage::RaBitQ,
            rerank_candidates: 50,
            seed: Some(7),
            ..Default::default()
        };
        let index = HNSWIndex::build_parallel(base.clone(), config);

        let k = 10;
        let mut total_recall = 0.0f32;
        for q in &queries {
            let mut exact: Vec<(f32, usize)> = base
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let d: f32 = v.iter().zip(q).map(|(a, b)| (a - b) * (a - b)).sum();
                    (d, i)
                })
                .collect();
            exact.sort_by(|a, b| a.0.total_cmp(&b.0));
            let truth: HashSet<usize> = exact.iter().take(k).map(|(_, i)| *i).collect();

            let got: HashSet<usize> = index
                .search(q, k)
                .expect("search")
                .into_iter()
                .filter_map(|r| r.id.parse::<usize>().ok())
                .collect();

            total_recall += truth.intersection(&got).count() as f32 / k as f32;
        }
        let recall = total_recall / n_queries as f32;

        // Measured on this exact seed/config at the time this test was written: 100%. (These
        // are well-separated Gaussian-ish blobs at ef_search=150 for n=800 — an easy corpus,
        // deliberately: the point of this test is to catch a broken *metric*, and a
        // discriminating-power check confirms it does. With the traversal kernel sabotaged to
        // return a constant (carrying zero information), recall on this exact test collapsed
        // to 7% — so the floor below is not a rubber stamp, and 0.75 leaves a wide margin
        // below the real 100% for run-to-run noise without coming anywhere near the ~7% a
        // broken kernel produces. See `lesson_untested_public_options`: a floor equal to the
        // measurement is not a regression test, it is a coin flip against float
        // non-determinism.
        assert!(
            recall > 0.75,
            "Storage::RaBitQ recall@{k} on clustered data = {:.1}% (held-out queries, \
             brute-force ground truth) — below floor, traversal metric likely broken",
            recall * 100.0,
        );
    }

    /// `rerank_candidates: 0` must work: codes-only, the cold `full` array dropped entirely,
    /// and the estimate itself used as the final ranking with no exact-distance correction.
    /// Must not panic even though `full` stays empty for the whole life of the index.
    #[test]
    fn rabitq_zero_rerank_drops_full_precision_vectors_and_does_not_panic() {
        let mut rng = StdRng::seed_from_u64(55);
        let dim = 24;
        let centers: Vec<Vec<f32>> = (0..8)
            .map(|_| (0..dim).map(|_| rng.random::<f32>() * 10.0).collect())
            .collect();
        let base: Vec<Vec<f32>> = (0..320)
            .map(|i| {
                let c = &centers[i % 8];
                c.iter().map(|x| x + rng.random::<f32>() * 0.5).collect()
            })
            .collect();

        let config = HNSWConfig {
            metric: DistanceMetric::L2,
            storage: Storage::RaBitQ,
            rerank_candidates: 0,
            seed: Some(3),
            ..Default::default()
        };
        // `build_parallel` explicitly: it quantizes via `push_node` without ever calling
        // `get_embedding` on quantized storage mid-build (unlike `insert_node`, which
        // `Sequential` uses and which — like `Storage::SQ8` — assumes `full` is populated
        // whenever it needs a candidate's embedding for neighbour selection).
        let index = HNSWIndex::build_parallel(base.clone(), config);

        assert!(
            index.full.is_empty(),
            "rerank_candidates = 0 must drop the full-precision side array"
        );

        for q in base.iter().step_by(37) {
            let results = index.search(q, 5).expect("search must not panic");
            assert_eq!(results.len(), 5);
        }
    }
}
