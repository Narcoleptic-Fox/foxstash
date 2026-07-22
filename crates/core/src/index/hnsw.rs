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
    /// Filtered search's allow-list. `Some` only from [`HNSWIndex::search_filtered`]. Unlike the
    /// other fields this is *not* distance-computation prep — it gates which nodes enter the result
    /// heap, and `search_layer` applies it **only at layer 0** (upper-layer descent must navigate
    /// freely through excluded nodes or the walk disconnects). It rides in `QueryPrep` because
    /// `search_layer` is already at clippy's argument ceiling and this struct is the documented
    /// place to add per-query state without growing every call site.
    filter: Option<&'a FilterMask>,
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

/// How to build the graph.
///
/// # Use `Parallel` (the default). `Sequential` costs 5.2x the build time to buy ~0.1 recall.
///
/// Measured on SIFT1M — 1,000,000 vectors, M=32/m0=64, ef_c=200, real ground truth
/// (`--example build_strategy_recall sift1m 1000000`, idle machine):
///
/// ```text
///   build:  Sequential 865.3s   Parallel 165.8s   (5.2x faster)
///
///       ef     Sequential       Parallel      delta
///       50         98.52%         98.21%     -0.31
///      100         99.61%         99.52%     -0.08
///      200         99.88%         99.86%     -0.02
/// ```
///
/// # The `Auto` variant was removed, and the story is a warning
///
/// This enum used to carry a third variant, `Auto`, which resolved to `Parallel` below 50k
/// vectors and `Sequential` at or above it — "reliability over speed" — because `Parallel` was
/// documented as *"may have lower recall at larger scales (needs more work)"*. `Sequential` was
/// the default for the same reason.
///
/// That caveat described a **real bug, which was fixed.** The parallel builder genuinely did
/// wreck recall once — the defect that hid behind uniform-random vectors for an entire release
/// (every ANN scores ~60% on random data, so nothing could see it). The bug died; the warning
/// did not. It went on quietly routing **every production-sized index** onto a builder that takes
/// five times longer, to avoid a problem that no longer existed.
///
/// Note what `Auto` actually was: its *only* distinct behaviour was choosing `Sequential` at
/// scale. That is precisely the wrong choice. It was not a dispatcher with a bad threshold — it
/// was the bug, wearing a dispatcher's clothes. So it is gone rather than inverted.
///
/// The 1M numbers above were produced *by this library's own flagship benchmark* (`storage_pareto`
/// builds with `Parallel` at 1M and reports 99.5% recall), which means the evidence refuting the
/// caveat was sitting in the README the whole time. A stale warning is not free: it keeps costing
/// you until someone measures it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum BuildStrategy {
    /// Rayon-parallel build. **The default.** 5.2x faster than `Sequential` at 1M vectors, for a
    /// recall cost of 0.02–0.31 points (see the table above).
    #[default]
    Parallel,
    /// One-at-a-time insertion. Slower by ~5x at 1M; buys ~0.1–0.3 recall points at low `ef`.
    ///
    /// Reach for this only if you have measured that the last fraction of a point matters on
    /// *your* data — not on the strength of the word "sequential". It is also the only strategy
    /// that must read vectors back out of storage mid-build, which is why quantized builds with
    /// `rerank_candidates: 0` retain their f32 vectors for the duration and drop them afterwards.
    Sequential,
}

/// Configuration for HNSW index
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
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

    /// Build strategy. Defaults to [`BuildStrategy::Parallel`] — 5.2x faster at 1M vectors for
    /// a recall cost of 0.02-0.31 points. See [`BuildStrategy`] for the measurements.
    pub build_strategy: BuildStrategy,

    /// Random seed for the level assignments (`None` = drawn from entropy).
    ///
    /// # A seed alone does not make a build reproducible
    ///
    /// It does under [`BuildStrategy::Sequential`], which is bit-identical across runs at a fixed
    /// seed. It does **not** under [`BuildStrategy::Parallel`] — the default. The seed fixes which
    /// level each vector lands on and the order they are inserted in, but not the order in which
    /// threads win the lock on a neighbour list. Two parallel builds at the same seed produce
    /// graphs that differ on a large fraction of nodes (measured: ~78% of nodes at m0=16, n=600).
    ///
    /// Recall is unaffected — every such graph is a valid HNSW, and they are equivalent to within
    /// run-to-run noise. What you lose is byte-identical indexes and exactly-reproducible search
    /// results. If you need either — golden-file tests, debugging a specific graph, regulated
    /// reproducibility — set `build_strategy: BuildStrategy::Sequential` as well as `seed`.
    ///
    /// Incremental [`HNSWIndex::add`] and [`HNSWIndex::add_embedding`] **are** reproducible at a
    /// fixed seed: they are sequential by nature, so they have no thread race to lose determinism
    /// to. (They
    /// were not, until 1.0. `random_level` called `rand::rng()` per insert with `config.seed`
    /// sitting unread on `&self`, so an index grown by `add()` was random at every seed.)
    ///
    /// Pinned by `seed_gives_reproducible_builds_only_on_the_sequential_builder` and
    /// `seed_reaches_the_incremental_add_path`.
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

    /// Total bits per dimension for [`Storage::TurboQuant`] (`b`): `b−1` MSE bits + 1 QJL bit.
    /// Ignored by every other storage mode. `b = 2` (≈2.5 bits/dim, between RaBitQ and SQ8) is a
    /// sensible default; sweep `{2, 3, 4}` to trade memory for recall. Must be ≥ 1.
    pub turbo_bits: usize,

    /// Total bits per dimension for [`Storage::TurboRabit`] (`B`): 1 sign bit + `B−1`
    /// magnitude bits. Ignored by every other storage mode. `B = 1` is exactly classic
    /// RaBitQ; sweep `{2, 3, 4}` against `turbo_bits` at matched budgets. Must be in
    /// `1..=4` (nibble-packed codes; b=4 already reaches F32 recall).
    pub rabit_bits: usize,

    /// Relabel nodes in BFS order after the build so graph-adjacent nodes sit at nearby arena
    /// offsets — a pure locality win worth **+10–15% query QPS** at zero recall cost (the walk's
    /// bottleneck is per-hop cache misses, not compute; see [`HNSWIndex::reorder_for_locality`]).
    /// It is transparent: returned document ids are unchanged, only faster.
    ///
    /// **Default `true`** — this is a free lunch for query-heavy indexes, which is most of them.
    /// Its one cost is at build time: a single reorder pass (sub-second on 300k nodes, a few
    /// seconds on 1.5M) that briefly holds a second copy of the arena, so peak build memory is
    /// ~2× the index size for the duration. Set `false` if your build is memory-constrained or
    /// you rebuild far more often than you query.
    pub reorder_for_locality: bool,
}

/// Distance metric for [`HNSWIndex`]. **Every** storage mode honours it.
///
/// That sentence is the whole point, and it was not always true. This library used to ship three
/// standalone quantized index types (`SQ8HNSWIndex`, `RaBitQHNSWIndex`, `PQHNSWIndex`), and not
/// one of them had a `metric` field: all three were hardcoded L2, while [`HNSWConfig`] defaults
/// to **cosine**. So swapping index type to save memory silently changed *the question being
/// asked*, and every caller escaped only by passing `L2` by hand. Worse, the same bug lived
/// inside [`HNSWIndex`] itself for a release — the quantized traversal ignored `config.metric`
/// and always ranked by L2, which is what made SQ8 read 71.4% recall when it was really capable
/// of 99.33%. A metric bug does not look like a crash. It looks like a mediocre quantizer.
///
/// All three standalone types are now deleted and quantization is a [`Storage`] mode on this one
/// index, so there is exactly one place the metric can be set and exactly one place it is read.
///
/// **Set it explicitly.** The default is [`DistanceMetric::Cosine`], which is right for
/// embeddings; SIFT/GIST-style benchmarks want [`DistanceMetric::L2`]. Getting it wrong does not
/// error — it silently answers a different question. And note the two are *not* interchangeable
/// for quantization: under cosine, `Storage::RaBitQ` encodes a unit-normalized copy, discarding
/// magnitude, which is exactly what cosine ignores and exactly what a 1-bit code cannot carry.
/// RaBitQ is therefore materially *better* under cosine than under L2 (see [`Storage`]).
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
    /// # Use this at high dimension. Use [`Storage::SQ8`] at low dimension.
    ///
    /// The two modes **swap places** across the dimension range, and the crossover is not
    /// subtle:
    ///
    /// | | SIFT1M (128-d) | GIST1M (960-d) |
    /// |---|---|---|
    /// | [`Storage::SQ8`] | **1.20x hnswlib** — the win | worthless (1.03x F32) |
    /// | `Storage::RaBitQ` | **~12x slower** than SQ8 | **1.4–1.6x faster** than SQ8 *or* F32 |
    ///
    /// At 960-d, matched recall, single-threaded: RaBitQ serves 1,136 QPS to F32's 789 at 93.8%
    /// recall@10, and 410 to F32's 252 at 99.0% — while costing **138 ns per distance against
    /// F32's 277 and SQ8's 269**.
    ///
    /// # Why they trade places
    ///
    /// Every quantized traversal trades ALU work for memory traffic, and these two sit on
    /// opposite sides of that trade:
    ///
    /// * **SQ8** must widen `u8` → `i32` → `f32` before it can compute — roughly **3x the ALU
    ///   uops per dimension of plain f32**. What it buys is skipped DRAM round-trips, which is
    ///   roughly *fixed* per node visit. Fixed benefit, cost linear in `dim`: it wins at low
    ///   `dim` and dies at high `dim`.
    /// * **RaBitQ** compares sign bits against a query rotated once per search — **cheaper per
    ///   dimension than f32**, with no widening at all. What it pays is a *coarser* estimate,
    ///   which makes the graph walk take more hops. That penalty **shrinks sharply as `dim`
    ///   rises**, because the code is 1 bit *per dimension*: a higher-dimensional vector gets a
    ///   proportionally longer code, so the estimate gets better exactly where the vector gets
    ///   more expensive to read. It loses at low `dim` and wins at high `dim`.
    ///
    /// Both blades close at once — it is a scissors, not a single crossing line. Measured on one
    /// corpus, prefix-truncated so `dim` is the only variable (`--example dim_crossover`, GIST,
    /// n = 200k, ef = 100):
    ///
    /// ```text
    ///   dim          64    128    192    256    384    512    768    960
    ///   RaBitQ recall  63.7%  78.0%  84.9%  88.2%  91.4%  94.2%  96.9%  96.7%
    ///   gap vs F32    -35.8  -20.6  -13.4  -10.0   -6.4   -3.8   -0.9   -0.9
    ///   RaBitQ speed   1.28x  1.55x  1.58x  1.74x  1.88x  2.01x  1.98x  1.89x   (F32 ns/dist ÷ RaBitQ)
    /// ```
    ///
    /// The accuracy penalty collapses by ~40x across that range while the speed advantage grows.
    /// (An earlier version of this doc claimed the penalty was "roughly independent of `dim`" and
    /// cited, as *supporting* evidence, that RaBitQ issues 10x more distance computations at
    /// 128-d but only 2% more at 960-d. That is the refutation, printed next to the claim.)
    ///
    /// # Rule of thumb
    ///
    /// Real embeddings are 384-d (MiniLM) to 1536-d (OpenAI); nobody runs RAG on 128-d vectors.
    ///
    /// **Use `SQ8` at 384-d and below. Use `RaBitQ` at 768-d and above.** In between, measure.
    ///
    /// 384-d is decided by measurement, not by reading the table above: at **matched recall** —
    /// the only comparison that bills RaBitQ for its extra hops — RaBitQ *loses* at 384-d to both
    /// SQ8 and plain F32 (`--example dim_pareto gist1m 384`, n = 200k):
    ///
    /// ```text
    ///   recall@10     F32     SQ8   RaBitQ
    ///      93.9%     3116    3371    ~2674
    ///      98.1%     1826    1935    ~1398
    ///      99.35%    1068    1137     ~750
    /// ```
    ///
    /// At a *fixed* `ef` RaBitQ looks like the winner at 384-d (3,257 QPS vs 1,898). That reading
    /// is an artifact: at fixed `ef` it is simply searching less hard, and 6.4 recall points
    /// behind. A mode that is fast because it stopped finding things is not fast. Compare at
    /// matched recall. Reproduce with `--example storage_pareto gist1m` and `--example dim_pareto`.
    ///
    /// One caveat carried from 128-d: with a coarse metric, a **too-small rerank pool** makes
    /// recall *fall* as `ef` rises, because distractors crowd the fixed pool and evict true
    /// neighbours before the exact rescore sees them. If recall goes down when you search
    /// harder, raise [`HNSWConfig::rerank_candidates`] before suspecting the quantizer.
    RaBitQ,
    /// Data-oblivious multi-bit TurboQuant codes (see [`crate::vector::turboquant`]).
    ///
    /// A separate, self-contained alternative to [`Storage::RaBitQ`]: `b−1` MSE bits per
    /// coordinate plus a 1-bit QJL residual, with an unbiased inner-product estimator and a
    /// codebook **derived** from the known post-rotation Gaussian (no k-means on the data). The
    /// bit budget `b` is set by [`HNSWConfig::turbo_bits`]. Correctness-first integration stores
    /// codes in a parallel array; arena-packing is a later QPS optimization.
    TurboQuant,
    /// Extended RaBitQ — B-bit codes with the RaBitQ unbiased estimator (see
    /// [`crate::vector::turborabit`]).
    ///
    /// A separate, self-contained multi-bit extension of [`Storage::RaBitQ`] (which stays
    /// frozen as the 1-bit baseline): the signed grid `{u − (2^B−1)/2}`, encoded by the
    /// optimal-rescale critical-value sweep, estimated by the same folded
    /// `dtc² + ‖q−c‖² − 2ℓ²⟨v,rq⟩/⟨r,v⟩` algebra. `B` is set by [`HNSWConfig::rabit_bits`].
    /// Correctness-first integration stores codes in a parallel array; arena-packing is the
    /// same later QPS optimization as TurboQuant's.
    TurboRabit,
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
            turbo_bits: 2,
            rabit_bits: 3,
            reorder_for_locality: true,
        }
    }
}

impl HNSWConfig {
    /// The bit budget the active storage mode actually reads: `turbo_bits` under
    /// [`Storage::TurboQuant`], `rabit_bits` under [`Storage::TurboRabit`], `0` otherwise.
    /// The arena layout ([`vec_words`]) is a function of this, so it must be resolved the
    /// same way everywhere — one accessor, not per-call-site matches that could drift.
    #[inline]
    pub(crate) fn quant_bits(&self) -> usize {
        match self.storage {
            Storage::TurboQuant => self.turbo_bits,
            Storage::TurboRabit => self.rabit_bits,
            _ => 0,
        }
    }

    /// Preset: **high-recall RAG at scale**. `TurboRabit` at 4 bits + reranking + the locality
    /// relabel — it owns the high-recall Pareto (r ≥ 0.99) at roughly half SQ8's hot-block bytes
    /// on the datasets measured, matching SQ8's throughput with a better ceiling.
    ///
    /// A good **starting point, not a universal optimum**: quantizer recall is distribution-
    /// dependent (a cone-shaped corpus is hostile to few-bit codes — see [`Self::with_auto_storage`]),
    /// so verify on your data. Graph knobs come from [`Self::default`] (M=32, cosine).
    pub fn rag_high_recall() -> Self {
        Self {
            storage: Storage::TurboRabit,
            rabit_bits: 4,
            rerank_candidates: 200,
            ..Self::default()
        }
    }

    /// Preset: **throughput RAG**. `SQ8` + reranking — the mid-recall throughput frontier
    /// (r ≈ 0.95): tracks F32 recall within ~0.3 points at higher QPS and 4× compression. Pick
    /// this when you do not need the last recall point. Starting point, not a universal optimum.
    pub fn rag_throughput() -> Self {
        Self {
            storage: Storage::SQ8,
            rerank_candidates: 100,
            ..Self::default()
        }
    }

    /// Resolve storage from a corpus **sample** via centroid dominance `‖μ‖ / E‖x − μ‖` — the
    /// cheap, build-free predictor of how hostile a distribution is to sign-based (RaBitQ-family)
    /// codes (see [`recommend_turborabit_bits`](crate::vector::turborabit::recommend_turborabit_bits)).
    /// This encodes the one durable lesson of the quantizer work: the right bit budget is
    /// **data-dependent**, not fixed — a nomic-style cone needs 4 bits where a well-spread corpus
    /// is fine at 2.
    ///
    /// Sets `storage = TurboRabit` with the recommended budget (2–4), and — because TurboRabit
    /// reaches its recall through reranking — ensures `rerank_candidates > 0` (a 0 is bumped to
    /// the default). Everything else is left as you set it. The returned config always holds a
    /// concrete storage the builder can encode directly.
    ///
    /// A few thousand sample vectors is plenty (dominance is a distribution statistic, not a
    /// per-vector one); the full corpus works but is unnecessary.
    pub fn with_auto_storage(mut self, sample: &[Vec<f32>]) -> Self {
        self.storage = Storage::TurboRabit;
        self.rabit_bits = crate::vector::turborabit::recommend_turborabit_bits(sample);
        if self.rerank_candidates == 0 {
            self.rerank_candidates = Self::default().rerank_candidates;
        }
        self
    }

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
    /// - [`BuildStrategy::Parallel`] (default): rayon-parallel; 5.2x faster at 1M.
    /// - [`BuildStrategy::Sequential`]: ~5x slower; buys 0.02-0.31 recall points at 1M.
    ///
    /// The `Auto` variant was removed — its only distinct behaviour was choosing `Sequential`
    /// above 50k vectors, which the measurements show is the wrong choice. See [`BuildStrategy`].
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
/// Words needed for `dim` nibble-packed codes (4 bits/dim, byte-packed then word-rounded) —
/// TurboQuant's MSE indices. Byte-granular like [`rabitq_bit_words`], for the same reason.
#[inline(always)]
const fn nibble_words(dim: usize) -> usize {
    dim.div_ceil(2).div_ceil(4)
}

/// `quant_bits` is the active multi-bit budget: `turbo_bits` under `TurboQuant`,
/// `rabit_bits` under `TurboRabit`, ignored (pass 0) otherwise — see
/// [`HNSWConfig::quant_bits`].
#[inline(always)]
const fn vec_words(storage: Storage, dim: usize, quant_bits: usize) -> usize {
    match storage {
        Storage::F32 => dim,
        Storage::SQ8 => dim.div_ceil(4),
        Storage::RaBitQ => 2 + rabitq_bit_words(dim),
        // `[gamma][qjl sign bits][mse nibbles]` — the nibble section exists only when
        // there are MSE bits (`total_bits > 1`); `total_bits = 1` is a pure QJL sketch.
        // Nibbles are sized by the FWHT-PADDED dim (next power of two): the structured
        // rotation quantizes in the padded space, so `TurboCode::idx` is padded-length.
        // The qjl section stays at the raw dim (the sketch runs over the unpadded residual).
        Storage::TurboQuant => {
            1 + rabitq_bit_words(dim)
                + if quant_bits > 1 {
                    nibble_words(crate::vector::turboquant::fht_padded_dim(dim))
                } else {
                    0
                }
        }
        // `[dtc_sq][f_rescale][nibble-packed codes]` — one 4-bit slot per coordinate
        // regardless of B (bounded 1..=4 at fit time), read by the FUSED kernel: the
        // Extended-RaBitQ estimate is linear in the code value, so one cvt+FMA per
        // coordinate replaces the earlier B bit-plane passes. `quant_bits` no longer
        // affects the block size, only the values inside the slots.
        Storage::TurboRabit => 2 + nibble_words(dim),
    }
}

/// Size, in 4-byte units, of one node block.
#[inline(always)]
const fn node_stride(m0: usize, dim: usize, storage: Storage, quant_bits: usize) -> usize {
    node_hdr_len(m0) + vec_words(storage, dim, quant_bits)
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

/// Bump when the meaning of any [`HNSWSnapshot`] field changes. Belt-and-braces on top of the
/// crate-version check: two builds of the *same* crate version can still disagree if a field is
/// reinterpreted on a dev branch.
///
/// v2: `HNSWConfig` gained `reorder_for_locality` (serialized in the snapshot's `config`) without a
/// crate-version bump, so a v1 snapshot's bincode layout no longer matches — it must be rejected
/// cleanly here rather than deserialized into garbage (bincode read a stray byte as a bool and
/// panicked with `InvalidBoolEncoding`).
const SNAPSHOT_FORMAT_VERSION: u32 = 2;

/// The verbatim on-disk image behind [`HNSWIndex::snapshot_to_file`]. Every field is a direct
/// clone of the corresponding `HNSWIndex` field except the two version stamps; `stride`, `hdr`
/// and `level_rng` are derived from `config` on load rather than stored.
///
/// Same-version cache format only — see `snapshot_to_file` for why this is not the portable path.
#[derive(serde::Serialize, serde::Deserialize)]
struct HNSWSnapshot {
    format_version: u32,
    crate_version: String,
    embedding_dim: usize,
    config: HNSWConfig,
    nodes: Vec<u32>,
    connections: Vec<Vec<Vec<u32>>>,
    q_min: Vec<f32>,
    q_scale: Vec<f32>,
    full: Vec<f32>,
    rabitq: Option<crate::vector::rabitq::RaBitQuantizer>,
    turboquant: Option<crate::vector::turboquant::TurboQuantizer>,
    turborabit: Option<crate::vector::turborabit::TurboRabitQuantizer>,
    ids: Vec<String>,
    contents: Vec<String>,
    /// `serde_json::Value` cannot ride bincode (its `Deserialize` is self-describing —
    /// `deserialize_any` — which a non-self-describing format rejects at runtime, not compile
    /// time). Stored as JSON text and re-parsed on load.
    metadata: Vec<Option<String>>,
    entry_point: Option<usize>,
    max_layer: usize,
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
    fn is_trained(&self) -> bool {
        match self.config.storage {
            Storage::F32 => true,
            Storage::SQ8 => !self.q_scale.is_empty(),
            Storage::RaBitQ => self.rabitq.is_some(),
            Storage::TurboQuant => self.turboquant.is_some(),
            Storage::TurboRabit => self.turborabit.is_some(),
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
        if n > 0 && self.config.storage != Storage::F32 && self.full.is_empty() && !self.is_empty()
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
        thread_local! {
            static CTX: std::cell::RefCell<SearchContext> =
                std::cell::RefCell::new(SearchContext::new(0));
        }
        CTX.with(|c| self.search_inner(query, k, &mut c.borrow_mut(), Some(filter)))
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
        filter: Option<&FilterMask>,
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
            const RERANK_PREFETCH_AHEAD: usize = 4;
            const RERANK_HEAD_LINES: usize = 8; // 512 B of a 3 KB vector at 768-d
            let dim = self.embedding_dim;
            for i in 0..found.len() {
                let ahead = i + RERANK_PREFETCH_AHEAD;
                if ahead < found.len() {
                    // SAFETY: prefetch is a hint — `wrapping_add` cannot leave the
                    // allocation for a valid node id, and a stale line costs nothing.
                    unsafe {
                        let p = self.full.as_ptr().wrapping_add(found[ahead].1 * dim) as *const u8;
                        prefetch_embedding(p, RERANK_HEAD_LINES);
                    }
                }
                found[i].0 = self.exact_distance(query, found[i].1);
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
        let admit = |id: usize| filter.is_none_or(|f| f.contains(id));

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
        if self.config.storage != Storage::TurboRabit {
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
        if self.config.storage == Storage::TurboRabit {
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
    fn from_zero(zero: &ZeroNode, m: usize) -> Self {
        let mut node = Self::default();
        for (i, &pid) in zero.nearest.iter().take(m.min(M_MAX)).enumerate() {
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

    /// Filtered search must return the true top-k *within the allowed set* — not the unfiltered
    /// top-k with excluded nodes dropped afterward (which would under-fill k), and never an excluded
    /// node. Asserts the user-visible OUTPUT against brute force, with `ef_search >= n` so the walk
    /// is exhaustive and the comparison is exact (no graph-miss slack to hide a logic bug).
    #[test]
    fn filtered_search_matches_bruteforce_over_allowed() {
        let n = 300usize;
        let dim = 16usize;
        let embeddings: Vec<Vec<f32>> = (0..n).map(|i| generate_random_vector(dim, i as u64)).collect();
        let config = HNSWConfig {
            metric: DistanceMetric::Cosine,
            ef_search: n, // exhaustive walk → exact
            ..HNSWConfig::default()
        };
        let index = HNSWIndex::build(embeddings.clone(), config);

        // Allow even-id nodes only. `build` assigns ids "0".."n-1".
        let allow = |i: usize| i % 2 == 0;
        let mask = index.filter_mask(|id, _content, _meta| allow(id.parse::<usize>().unwrap()));
        assert_eq!(mask.allowed_count(), n / 2);

        let k = 10usize;
        for qseed in [1000u64, 2000, 3000] {
            let q = generate_random_vector(dim, qseed);
            let got = index.search_filtered(&q, k, &mask).unwrap();

            // Never an excluded node; exactly k results (allowed_count >> k here).
            assert_eq!(got.len(), k);
            for r in &got {
                assert!(allow(r.id.parse::<usize>().unwrap()), "returned excluded id {}", r.id);
            }

            // Brute-force true top-k over the ALLOWED set (cosine == -distance ordering).
            let qn = crate::vector::simd::norm_simd(&q);
            let mut scored: Vec<(f32, usize)> = (0..n)
                .filter(|&i| allow(i))
                .map(|i| {
                    let e = &embeddings[i];
                    let en = crate::vector::simd::norm_simd(e);
                    let dot: f32 = q.iter().zip(e).map(|(a, b)| a * b).sum();
                    (dot / (qn * en), i)
                })
                .collect();
            scored.sort_by(|a, b| b.0.total_cmp(&a.0));
            let want: std::collections::HashSet<usize> =
                scored.iter().take(k).map(|&(_, i)| i).collect();
            let got_ids: std::collections::HashSet<usize> =
                got.iter().map(|r| r.id.parse::<usize>().unwrap()).collect();
            assert_eq!(got_ids, want, "filtered top-{k} != brute force over allowed (seed {qseed})");
        }
    }

    /// A filter more selective than `k` yields exactly the allowed nodes ("up to k"), all of them,
    /// and `filter_mask_ids` selects by external id.
    #[test]
    fn filtered_search_fewer_than_k_and_by_id() {
        let n = 100usize;
        let dim = 8usize;
        let embeddings: Vec<Vec<f32>> = (0..n).map(|i| generate_random_vector(dim, i as u64)).collect();
        let config = HNSWConfig { ef_search: n, ..HNSWConfig::default() };
        let index = HNSWIndex::build(embeddings, config);

        let allowed: std::collections::HashSet<String> =
            ["7", "42", "99"].iter().map(|s| s.to_string()).collect();
        let mask = index.filter_mask_ids(&allowed);
        assert_eq!(mask.allowed_count(), 3);

        let q = generate_random_vector(dim, 555);
        let got = index.search_filtered(&q, 10, &mask).unwrap();
        // Only 3 allowed → at most 3 back, and exactly the allowed ids.
        assert_eq!(got.len(), 3);
        let got_ids: std::collections::HashSet<String> = got.into_iter().map(|r| r.id).collect();
        assert_eq!(got_ids, allowed);
    }

    /// Unfiltered `search` must be byte-for-byte unchanged by the filter plumbing: an all-allowed
    /// mask returns the same ids as a plain search. Guards the "None ⇒ zero-cost" claim's correctness.
    #[test]
    fn all_allowed_mask_equals_unfiltered() {
        let n = 200usize;
        let dim = 12usize;
        let embeddings: Vec<Vec<f32>> = (0..n).map(|i| generate_random_vector(dim, i as u64 + 9)).collect();
        let config = HNSWConfig { metric: DistanceMetric::Cosine, ef_search: 120, ..HNSWConfig::default() };
        let index = HNSWIndex::build(embeddings, config);
        let mask = index.filter_mask(|_, _, _| true);
        assert_eq!(mask.allowed_count(), n);

        let q = generate_random_vector(dim, 77);
        let plain: Vec<String> = index.search(&q, 10).unwrap().into_iter().map(|r| r.id).collect();
        let filtered: Vec<String> =
            index.search_filtered(&q, 10, &mask).unwrap().into_iter().map(|r| r.id).collect();
        assert_eq!(plain, filtered);
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
            turboquant: None,
            turborabit: None,
            filter: None,
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

        let results = index.search_inner(&[1.0, 0.0, 0.0], 5, &mut stale, None).unwrap();

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

    /// Level assignment must never hit `ln(0) = -inf`, AND must come from the seeded stream.
    ///
    /// The old version of this test ran the loop and asserted nothing but the absence of a panic.
    /// It therefore could not tell a seeded draw from `rand::rng()` — and `random_level` was in
    /// fact calling `rand::rng()`, so `add()` ignored `seed` entirely. The panic property was
    /// real and is kept; what it lacked was any way to fail for the reason that mattered.
    #[test]
    fn random_level_never_panics_and_comes_from_the_seeded_stream() {
        let draws = |seed: Option<u64>| -> Vec<usize> {
            let mut index = HNSWIndex::new(
                3,
                HNSWConfig {
                    seed,
                    ..Default::default()
                },
            );
            (0..10_000).map(|_| index.random_level()).collect()
        };

        // ln(0) is impossible: a level is finite, so it is small. (`as usize` on -inf saturates
        // to 0 rather than panicking, so assert the shape of the distribution, not just liveness.)
        let a = draws(Some(7));
        assert!(
            a.iter().all(|&l| l < 64),
            "exponential decay must not produce absurd levels"
        );
        assert!(
            a.iter().any(|&l| l > 0),
            "every node landing on layer 0 means ml is not applied"
        );

        assert_eq!(
            a,
            draws(Some(7)),
            "a fixed seed must give a reproducible level sequence"
        );
        assert_ne!(
            a,
            draws(Some(8)),
            "a different seed must give a different level sequence"
        );
    }

    /// `seed` must reach the INCREMENTAL path too, not just the bulk builders.
    ///
    /// The bulk builders seed their own RNG, so they were reproducible while an index grown by
    /// `add_embedding()` was not, at any seed. Found by the generated config x code-path matrix
    /// (`xtask/config_matrix.py`): the `seed` row was empty in the `add` column, and an empty cell
    /// there means nothing on that path reads the option.
    #[test]
    fn seed_reaches_the_incremental_add_path() {
        let vecs: Vec<Vec<f32>> = (0..200)
            .map(|i| {
                (0..8)
                    .map(|d| ((i * 7 + d * 13) % 40) as f32 * 0.1)
                    .collect()
            })
            .collect();

        let grown = |seed: u64| -> Vec<Vec<u32>> {
            let mut ix = HNSWIndex::new(
                8,
                HNSWConfig {
                    seed: Some(seed),
                    m: 4,
                    m0: 8,
                    ..Default::default()
                },
            );
            for (i, v) in vecs.iter().enumerate() {
                ix.add_embedding(i.to_string(), v.clone()).unwrap();
            }
            (0..ix.len())
                .map(|i| {
                    let mut n = ix.get_neighbors_l0(i).to_vec();
                    n.sort_unstable();
                    n
                })
                .collect()
        };

        assert_eq!(
            grown(7),
            grown(7),
            "an index grown by add() must be reproducible at a fixed seed -- `add` is inherently \
             sequential, so unlike the parallel bulk builder it has no thread race to blame"
        );
        assert_ne!(
            grown(7),
            grown(9),
            "a different seed must give a different graph"
        );
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

        for strategy in [BuildStrategy::Sequential, BuildStrategy::Parallel] {
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
        assert_eq!(vec_words(Storage::RaBitQ, 100, 0), 2 + 4);

        // dim = 128 (SIFT-adjacent): bytes = 16, words = 4 -> vector region = 24 bytes,
        // matching the doc comment on `Storage`.
        assert_eq!(vec_words(Storage::RaBitQ, 128, 0), 2 + 4);
        assert_eq!(vec_words(Storage::RaBitQ, 128, 0) * 4, 24);
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
    /// End-to-end recall through the real index under [`Storage::TurboQuant`] + cosine — a public
    /// storage variant with no integration test is a shipped bug (`lesson_untested_public_options`).
    /// Held-out queries only (never self-retrieval), clustered data, and a discriminating-power
    /// floor: with the estimator sabotaged this recall collapses (verified separately). Also checks
    /// that more bits ⇒ at least as much recall, end to end.
    #[test]
    fn turboquant_recall_on_clustered_data_with_held_out_queries() {
        let mut rng = StdRng::seed_from_u64(2025);
        let dim = 96;
        let n_clusters = 16;
        let per_cluster = 40;
        let centers: Vec<Vec<f32>> = (0..n_clusters)
            .map(|_| (0..dim).map(|_| rng.random::<f32>() * 4.0 - 2.0).collect())
            .collect();
        let jitter = |rng: &mut StdRng, c: &[f32]| -> Vec<f32> {
            c.iter()
                .map(|x| x + rng.random::<f32>() * 0.5 - 0.25)
                .collect()
        };
        let base: Vec<Vec<f32>> = (0..n_clusters * per_cluster)
            .map(|i| jitter(&mut rng, &centers[i % n_clusters]))
            .collect();
        let queries: Vec<Vec<f32>> = (0..60)
            .map(|i| jitter(&mut rng, &centers[i % n_clusters]))
            .collect();

        let k = 10;
        // Cosine ground truth (TurboQuant estimates cosine similarity).
        let cos = |a: &[f32], b: &[f32]| -> f32 {
            let (mut d, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
            for (x, y) in a.iter().zip(b) {
                d += x * y;
                na += x * x;
                nb += y * y;
            }
            d / (na.sqrt() * nb.sqrt()).max(1e-9)
        };
        let truth: Vec<HashSet<usize>> = queries
            .iter()
            .map(|q| {
                let mut s: Vec<(f32, usize)> = base
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (cos(v, q), i))
                    .collect();
                s.sort_by(|a, b| b.0.total_cmp(&a.0));
                s.into_iter().take(k).map(|(_, i)| i).collect()
            })
            .collect();

        let recall_for = |bits: usize| -> f32 {
            let config = HNSWConfig {
                metric: DistanceMetric::Cosine,
                m: 16,
                m0: 32,
                ef_construction: 200,
                ef_search: 200,
                storage: Storage::TurboQuant,
                turbo_bits: bits,
                rerank_candidates: 50,
                seed: Some(9),
                ..Default::default()
            };
            let index = HNSWIndex::build_parallel(base.clone(), config);
            let mut total = 0.0f32;
            for (q, gt) in queries.iter().zip(&truth) {
                let got: HashSet<usize> = index
                    .search(q, k)
                    .expect("search")
                    .into_iter()
                    .filter_map(|r| r.id.parse::<usize>().ok())
                    .collect();
                total += gt.intersection(&got).count() as f32 / k as f32;
            }
            total / queries.len() as f32
        };

        let r2 = recall_for(2);
        let r4 = recall_for(4);
        // Non-vacuous floor (sabotaging the estimator collapses this), and more bits never hurt.
        assert!(r2 > 0.6, "TurboQuant b=2 recall too low end-to-end: {r2}");
        assert!(
            r4 >= r2 - 0.05,
            "b=4 ({r4}) unexpectedly far below b=2 ({r2})"
        );
    }

    /// End-to-end recall through the real index under [`Storage::TurboRabit`] — same contract
    /// as the TurboQuant test above (public variant + no integration test = shipped bug), and
    /// additionally under **both metrics**: honest L2 support is TurboRabit's differentiator
    /// over TurboQuant, so an untested L2 path here would be the untested half of the point.
    #[test]
    fn turborabit_recall_on_clustered_data_with_held_out_queries() {
        let mut rng = StdRng::seed_from_u64(2026);
        let dim = 96;
        let n_clusters = 16;
        let per_cluster = 40;
        let centers: Vec<Vec<f32>> = (0..n_clusters)
            .map(|_| (0..dim).map(|_| rng.random::<f32>() * 4.0 - 2.0).collect())
            .collect();
        let jitter = |rng: &mut StdRng, c: &[f32]| -> Vec<f32> {
            c.iter()
                .map(|x| x + rng.random::<f32>() * 0.5 - 0.25)
                .collect()
        };
        let base: Vec<Vec<f32>> = (0..n_clusters * per_cluster)
            .map(|i| jitter(&mut rng, &centers[i % n_clusters]))
            .collect();
        let queries: Vec<Vec<f32>> = (0..60)
            .map(|i| jitter(&mut rng, &centers[i % n_clusters]))
            .collect();

        let k = 10;
        let cos = |a: &[f32], b: &[f32]| -> f32 {
            let (mut d, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
            for (x, y) in a.iter().zip(b) {
                d += x * y;
                na += x * x;
                nb += y * y;
            }
            d / (na.sqrt() * nb.sqrt()).max(1e-9)
        };
        let l2 =
            |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum() };

        let truth_for = |better_first: &dyn Fn(&[f32], &[f32]) -> f32, descending: bool| {
            queries
                .iter()
                .map(|q| {
                    let mut s: Vec<(f32, usize)> = base
                        .iter()
                        .enumerate()
                        .map(|(i, v)| (better_first(v, q), i))
                        .collect();
                    if descending {
                        s.sort_by(|a, b| b.0.total_cmp(&a.0));
                    } else {
                        s.sort_by(|a, b| a.0.total_cmp(&b.0));
                    }
                    s.into_iter()
                        .take(k)
                        .map(|(_, i)| i)
                        .collect::<HashSet<usize>>()
                })
                .collect::<Vec<_>>()
        };

        let recall_for = |metric: DistanceMetric, bits: usize, truth: &[HashSet<usize>]| -> f32 {
            let config = HNSWConfig {
                metric,
                m: 16,
                m0: 32,
                ef_construction: 200,
                ef_search: 200,
                storage: Storage::TurboRabit,
                rabit_bits: bits,
                rerank_candidates: 50,
                seed: Some(9),
                ..Default::default()
            };
            let index = HNSWIndex::build_parallel(base.clone(), config);
            let mut total = 0.0f32;
            for (q, gt) in queries.iter().zip(truth) {
                let got: HashSet<usize> = index
                    .search(q, k)
                    .expect("search")
                    .into_iter()
                    .filter_map(|r| r.id.parse::<usize>().ok())
                    .collect();
                total += gt.intersection(&got).count() as f32 / k as f32;
            }
            total / queries.len() as f32
        };

        // Cosine: floor + more-bits-never-hurt, same contract as the TurboQuant test.
        let truth_cos = truth_for(&cos, true);
        let r2 = recall_for(DistanceMetric::Cosine, 2, &truth_cos);
        let r4 = recall_for(DistanceMetric::Cosine, 4, &truth_cos);
        assert!(
            r2 > 0.6,
            "TurboRabit b=2 cosine recall too low end-to-end: {r2}"
        );
        assert!(
            r4 >= r2 - 0.05,
            "b=4 ({r4}) unexpectedly far below b=2 ({r2})"
        );

        // L2: the estimator is native squared-L2, no proxy — hold it to the same floor.
        let truth_l2 = truth_for(&l2, false);
        let r3_l2 = recall_for(DistanceMetric::L2, 3, &truth_l2);
        assert!(
            r3_l2 > 0.6,
            "TurboRabit b=3 L2 recall too low end-to-end: {r3_l2}"
        );
    }

    /// The packed arena walk and the quantizer module are two implementations of one
    /// estimator — exactly the shape every 1.0-audit bug had. This pins them together:
    /// for every node, `distance_to_node` (arena bit-planes + shared SIMD kernel) must
    /// equal `TurboRabitQuantizer::estimate_dist_sq` (reference, allocating) on a fresh
    /// encode of the same input. Odd dim stresses the plane-packing tail; both metrics
    /// because their dispatch differs.
    #[test]
    fn turborabit_packed_walk_matches_module_estimator() {
        let mut rng = StdRng::seed_from_u64(77);
        let dim = 97; // not a multiple of 8: partial final byte in every bit-plane
        let base: Vec<Vec<f32>> = (0..80)
            .map(|_| (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect())
            .collect();
        let query: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();

        for metric in [DistanceMetric::Cosine, DistanceMetric::L2] {
            let config = HNSWConfig {
                metric,
                storage: Storage::TurboRabit,
                rabit_bits: 3,
                rerank_candidates: 10, // keep `full` so get_embedding works
                seed: Some(5),
                ..Default::default()
            };
            let index = HNSWIndex::build_parallel(base.clone(), config);
            let tr = index.turborabit.as_ref().expect("quantizer fitted");
            let prep = index.prepare_turborabit_query(&query).expect("prepared");
            let qprep = QueryPrep {
                norm: crate::vector::simd::norm_simd(&query),
                rabitq: None,
                turboquant: None,
                turborabit: Some(&prep),
                filter: None,
            };
            for node_id in 0..index.len() {
                let packed = index.distance_to_node(&query, node_id, &qprep);
                // Re-encode the same input push_node saw (get_embedding returns the
                // original; the cosine path encodes a unit-normalized copy of it).
                let stored = index.get_embedding(node_id).to_vec();
                let code = tr.encode(&index.rabitq_cosine_input(&stored));
                let raw = tr.estimate_dist_sq(&prep, &code);
                let expected = match metric {
                    DistanceMetric::L2 => raw,
                    DistanceMetric::Cosine => (raw * 0.5).clamp(0.0, 2.0),
                };
                let rel = (packed - expected).abs() / expected.abs().max(1e-4);
                // 1e-4, not 1e-3: both paths run the same f32 algebra, so the only honest
                // difference is summation order (~1e-5). A loose tolerance let a 1%-of-one-
                // term sabotage through; this one catches it.
                assert!(
                    rel < 1e-4,
                    "{metric:?} node {node_id}: packed walk {packed} != module {expected} (rel {rel:.2e})"
                );
            }
        }
    }

    /// Same pin for TurboQuant: arena `[gamma][qjl][nibbles]` + LUT/signed-sum kernels
    /// must equal `TurboQuantizer::estimate_ip` on a fresh encode. Odd dim stresses the
    /// half-used final nibble byte; b=4 exercises the full 8-entry LUT, b=1 the
    /// no-nibble-section layout.
    #[test]
    fn turboquant_packed_walk_matches_module_estimator() {
        let mut rng = StdRng::seed_from_u64(78);
        let dim = 97;
        let base: Vec<Vec<f32>> = (0..80)
            .map(|_| (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect())
            .collect();
        let query: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();

        for bits in [1usize, 2, 4] {
            let config = HNSWConfig {
                metric: DistanceMetric::Cosine,
                storage: Storage::TurboQuant,
                turbo_bits: bits,
                rerank_candidates: 10,
                seed: Some(5),
                ..Default::default()
            };
            let index = HNSWIndex::build_parallel(base.clone(), config);
            let tq = index.turboquant.as_ref().expect("quantizer fitted");
            let prep = index.prepare_turboquant_query(&query).expect("prepared");
            let qprep = QueryPrep {
                norm: crate::vector::simd::norm_simd(&query),
                rabitq: None,
                turboquant: Some(&prep),
                turborabit: None,
                filter: None,
            };
            for node_id in 0..index.len() {
                let packed = index.distance_to_node(&query, node_id, &qprep);
                let mut unit = index.get_embedding(node_id).to_vec();
                crate::vector::ops::normalize(&mut unit);
                let ip = tq.estimate_ip(&prep, &tq.encode(&unit));
                let expected = (1.0 - ip).clamp(0.0, 2.0);
                let rel = (packed - expected).abs() / expected.abs().max(1e-4);
                assert!(
                    rel < 1e-3,
                    "b={bits} node {node_id}: packed walk {packed} != module {expected} (rel {rel:.2e})"
                );
            }
        }
    }

    /// `reorder_for_locality: true` is the default and must be **transparent**: a build with it
    /// on returns the same search results as one with it off (only faster), while actually
    /// changing the internal layout. This pins both halves — the default is applied (arena
    /// differs) and it is safe (results identical).
    #[test]
    fn reorder_default_is_transparent_but_real() {
        let mut rng = StdRng::seed_from_u64(51);
        let base: Vec<Vec<f32>> = (0..400)
            .map(|_| (0..48).map(|_| rng.random::<f32>()).collect())
            .collect();
        let queries: Vec<Vec<f32>> = (0..30)
            .map(|_| (0..48).map(|_| rng.random::<f32>()).collect())
            .collect();
        let cfg = |reorder: bool| HNSWConfig {
            metric: DistanceMetric::Cosine,
            m: 16,
            m0: 32,
            ef_construction: 200,
            ef_search: 100,
            seed: Some(9),
            reorder_for_locality: reorder,
            ..Default::default()
        };
        let plain = HNSWIndex::build_parallel(base.clone(), cfg(false));
        let reordered = HNSWIndex::build_parallel(base.clone(), cfg(true));

        // Real: the default actually relabelled the arena (entry point almost surely moves to a
        // low id under BFS; the arenas are not byte-identical).
        assert_ne!(
            plain.nodes, reordered.nodes,
            "reorder_for_locality: true must change the layout, but the arenas are identical"
        );
        // Transparent: identical results, scores included.
        for q in &queries {
            let a: Vec<(String, u32)> = plain
                .search(q, 10)
                .unwrap()
                .into_iter()
                .map(|r| (r.id, r.score.to_bits()))
                .collect();
            let b: Vec<(String, u32)> = reordered
                .search(q, 10)
                .unwrap()
                .into_iter()
                .map(|r| (r.id, r.score.to_bits()))
                .collect();
            assert_eq!(a, b, "default reorder changed a query's results");
        }
    }

    /// The presets and `with_auto_storage` must produce configs that **build and search**, and
    /// the auto-picker must pick sensibly (max bits + rerank on a hostile cone corpus).
    #[test]
    fn presets_and_auto_storage_are_sane() {
        let mut rng = StdRng::seed_from_u64(61);
        let dim = 64;

        // Cone-shaped (hostile) corpus: big shared offset, tiny residuals → auto wants max bits.
        let offset: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() * 3.0).collect();
        let cone: Vec<Vec<f32>> = (0..500)
            .map(|_| {
                offset
                    .iter()
                    .map(|&o| o + (rng.random::<f32>() - 0.5) * 0.1)
                    .collect()
            })
            .collect();
        let auto = HNSWConfig {
            rerank_candidates: 0,
            ..Default::default()
        }
        .with_auto_storage(&cone);
        assert_eq!(auto.storage, Storage::TurboRabit);
        assert_eq!(auto.rabit_bits, 4, "cone corpus should auto-pick max bits");
        assert!(
            auto.rerank_candidates > 0,
            "auto must enable rerank for TurboRabit"
        );

        assert_eq!(HNSWConfig::rag_high_recall().storage, Storage::TurboRabit);
        assert_eq!(HNSWConfig::rag_high_recall().rabit_bits, 4);
        assert_eq!(HNSWConfig::rag_throughput().storage, Storage::SQ8);

        // Each config builds a working index. Clustered data + held-out queries, cosine.
        let centers: Vec<Vec<f32>> = (0..8)
            .map(|_| (0..dim).map(|_| rng.random::<f32>() * 4.0).collect())
            .collect();
        let mk = |n: usize, rng: &mut StdRng| -> Vec<Vec<f32>> {
            (0..n)
                .map(|i| {
                    centers[i % 8]
                        .iter()
                        .map(|x| x + rng.random::<f32>() * 0.3)
                        .collect()
                })
                .collect()
        };
        let base = mk(600, &mut rng);
        let queries = mk(40, &mut rng);
        let cos = |a: &[f32], b: &[f32]| {
            let (mut d, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
            for (x, y) in a.iter().zip(b) {
                d += x * y;
                na += x * x;
                nb += y * y;
            }
            d / (na.sqrt() * nb.sqrt()).max(1e-9)
        };
        let truth: Vec<HashSet<usize>> = queries
            .iter()
            .map(|q| {
                let mut s: Vec<(f32, usize)> = base
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (cos(v, q), i))
                    .collect();
                s.sort_by(|a, b| b.0.total_cmp(&a.0));
                s.into_iter().take(10).map(|(_, i)| i).collect()
            })
            .collect();

        for (label, mut cfg) in [
            ("high_recall", HNSWConfig::rag_high_recall()),
            ("throughput", HNSWConfig::rag_throughput()),
            ("auto", HNSWConfig::default().with_auto_storage(&base)),
        ] {
            cfg.ef_search = 200;
            cfg.seed = Some(3);
            let idx = HNSWIndex::build_parallel(base.clone(), cfg);
            let hits: usize = queries
                .iter()
                .zip(&truth)
                .map(|(q, gt)| {
                    let got: HashSet<usize> = idx
                        .search(q, 10)
                        .unwrap()
                        .into_iter()
                        .filter_map(|r| r.id.parse().ok())
                        .collect();
                    gt.intersection(&got).count()
                })
                .sum();
            let recall = hits as f32 / (queries.len() * 10) as f32;
            assert!(
                recall > 0.80,
                "{label} preset recall {recall:.2} too low to be working"
            );
        }
    }

    /// `reorder_for_locality` is a pure layout change: it must return **byte-identical search
    /// results** (same ids, same scores, same order) as the source for every query, in every
    /// storage. That is the whole contract — if a relabel is wrong it shows up here as a
    /// changed result, not a crash. Also checks the permutation is a bijection (every id
    /// still present exactly once).
    #[test]
    fn reorder_for_locality_preserves_search_results() {
        let mut rng = StdRng::seed_from_u64(41);
        let dim = 80;
        let centers: Vec<Vec<f32>> = (0..10)
            .map(|_| (0..dim).map(|_| rng.random::<f32>() * 4.0 - 2.0).collect())
            .collect();
        let base: Vec<Vec<f32>> = (0..500)
            .map(|i| {
                centers[i % 10]
                    .iter()
                    .map(|x| x + rng.random::<f32>() * 0.5 - 0.25)
                    .collect()
            })
            .collect();
        let queries: Vec<Vec<f32>> = (0..50)
            .map(|i| {
                centers[i % 10]
                    .iter()
                    .map(|x| x + rng.random::<f32>() * 0.5 - 0.25)
                    .collect()
            })
            .collect();

        for (storage, tb, rb) in [
            (Storage::F32, 2, 3),
            (Storage::SQ8, 2, 3),
            (Storage::TurboRabit, 2, 4),
            (Storage::TurboQuant, 3, 3),
        ] {
            let config = HNSWConfig {
                metric: DistanceMetric::Cosine,
                m: 16,
                m0: 32,
                ef_construction: 200,
                ef_search: 100,
                storage,
                turbo_bits: tb,
                rabit_bits: rb,
                rerank_candidates: if storage == Storage::F32 { 0 } else { 50 },
                seed: Some(5),
                ..Default::default()
            };
            let src = HNSWIndex::build_parallel(base.clone(), config);
            let re = src.reorder_for_locality();

            assert_eq!(re.len(), src.len(), "{storage:?}: node count changed");
            // Bijection: the multiset of document ids is unchanged.
            let mut a = src.ids.clone();
            let mut b = re.ids.clone();
            a.sort();
            b.sort();
            assert_eq!(a, b, "{storage:?}: reorder is not a bijection over ids");

            // Byte-identical results per query — ids, scores, order.
            for q in &queries {
                let rs = src.search(q, 10).expect("src search");
                let rr = re.search(q, 10).expect("reordered search");
                let sv: Vec<(String, u32)> =
                    rs.into_iter().map(|r| (r.id, r.score.to_bits())).collect();
                let rv: Vec<(String, u32)> =
                    rr.into_iter().map(|r| (r.id, r.score.to_bits())).collect();
                assert_eq!(sv, rv, "{storage:?}: reorder changed a query's results");
            }
        }
    }

    /// `requantize` must preserve the graph EXACTLY (that is its whole claim) and produce a
    /// working index in every target storage. Graph identity is asserted structurally —
    /// layer-0 neighbour lists, upper layers, entry point — not via a recall proxy
    /// (measure-the-output applies to the *quantizer*; the graph has an exact answer).
    #[test]
    fn requantize_preserves_graph_and_searches_in_every_storage() {
        let mut rng = StdRng::seed_from_u64(31);
        let dim = 96;
        let centers: Vec<Vec<f32>> = (0..12)
            .map(|_| (0..dim).map(|_| rng.random::<f32>() * 4.0 - 2.0).collect())
            .collect();
        let base: Vec<Vec<f32>> = (0..600)
            .map(|i| {
                centers[i % 12]
                    .iter()
                    .map(|x| x + rng.random::<f32>() * 0.5 - 0.25)
                    .collect()
            })
            .collect();
        let queries: Vec<Vec<f32>> = (0..40)
            .map(|i| {
                centers[i % 12]
                    .iter()
                    .map(|x| x + rng.random::<f32>() * 0.5 - 0.25)
                    .collect()
            })
            .collect();

        let src_config = HNSWConfig {
            metric: DistanceMetric::Cosine,
            m: 16,
            m0: 32,
            ef_construction: 200,
            ef_search: 100,
            storage: Storage::F32,
            rerank_candidates: 0,
            seed: Some(3),
            ..Default::default()
        };
        let src = HNSWIndex::build_parallel(base.clone(), src_config.clone());

        // Exact cosine ground truth for a recall floor per target.
        let k = 10;
        let cos = |a: &[f32], b: &[f32]| -> f32 {
            let (mut d, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
            for (x, y) in a.iter().zip(b) {
                d += x * y;
                na += x * x;
                nb += y * y;
            }
            d / (na.sqrt() * nb.sqrt()).max(1e-9)
        };
        let truth: Vec<HashSet<usize>> = queries
            .iter()
            .map(|q| {
                let mut s: Vec<(f32, usize)> = base
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (cos(v, q), i))
                    .collect();
                s.sort_by(|a, b| b.0.total_cmp(&a.0));
                s.into_iter().take(k).map(|(_, i)| i).collect()
            })
            .collect();

        for (storage, tb, rb, floor) in [
            (Storage::SQ8, 2, 3, 0.85),
            (Storage::RaBitQ, 2, 3, 0.35), // 1-bit is legitimately coarse; floor is non-vacuous
            (Storage::TurboQuant, 3, 3, 0.55),
            (Storage::TurboRabit, 2, 3, 0.80),
        ] {
            let new_config = HNSWConfig {
                storage,
                turbo_bits: tb,
                rabit_bits: rb,
                rerank_candidates: 50,
                ..src_config.clone()
            };
            let re = src.requantize(new_config).expect("requantize");

            // Graph identity — exact, node by node.
            assert_eq!(re.len(), src.len());
            assert_eq!(
                re.entry_point, src.entry_point,
                "{storage:?}: entry point moved"
            );
            assert_eq!(re.max_layer, src.max_layer, "{storage:?}: max layer moved");
            for i in 0..src.len() {
                assert_eq!(
                    re.get_neighbors_l0(i),
                    src.get_neighbors_l0(i),
                    "{storage:?}: node {i} layer-0 links differ"
                );
            }
            assert_eq!(
                re.connections, src.connections,
                "{storage:?}: upper layers differ"
            );

            // And it actually searches.
            let mut hits = 0usize;
            for (q, gt) in queries.iter().zip(&truth) {
                let got: HashSet<usize> = re
                    .search(q, k)
                    .expect("search")
                    .into_iter()
                    .filter_map(|r| r.id.parse::<usize>().ok())
                    .collect();
                hits += gt.intersection(&got).count();
            }
            let recall = hits as f32 / (queries.len() * k) as f32;
            assert!(
                recall > floor,
                "{storage:?}: requantized recall {recall} below floor {floor}"
            );
        }
    }

    /// The contract errors: quantized source, and graph-relevant config changes.
    #[test]
    fn requantize_rejects_bad_inputs() {
        let mut rng = StdRng::seed_from_u64(32);
        let base: Vec<Vec<f32>> = (0..200)
            .map(|_| (0..32).map(|_| rng.random::<f32>()).collect())
            .collect();
        let config = HNSWConfig {
            m: 8,
            m0: 16,
            seed: Some(1),
            ..Default::default()
        };
        let f32_idx = HNSWIndex::build_parallel(base.clone(), config.clone());

        // Non-F32 source.
        let sq8 = f32_idx
            .requantize(HNSWConfig {
                storage: Storage::SQ8,
                ..config.clone()
            })
            .expect("f32 -> sq8");
        assert!(
            sq8.requantize(HNSWConfig {
                storage: Storage::RaBitQ,
                ..config.clone()
            })
            .is_err(),
            "requantizing a quantized source must be rejected"
        );

        // Graph-relevant change.
        assert!(
            f32_idx
                .requantize(HNSWConfig {
                    m: 12,
                    storage: Storage::SQ8,
                    ..config.clone()
                })
                .is_err(),
            "changing m must be rejected"
        );
    }

    /// The snapshot's whole claim is *verbatim*: the loaded index is bit-identical where the
    /// JSON path is merely equivalent-ish (file.rs re-inserts through `add()`, so the parallel
    /// builder hands back a different graph). Every config field is set off its default —
    /// the save/load bug-class this guards against is a field silently dropped on one side
    /// (the wasm path shipped exactly that: `turbo_bits` was never serialized).
    #[test]
    fn snapshot_round_trip_is_verbatim_in_every_storage() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut rng = StdRng::seed_from_u64(77);
        let dim = 48;
        let base: Vec<Vec<f32>> = (0..400)
            .map(|_| (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect())
            .collect();
        let queries: Vec<Vec<f32>> = (0..20)
            .map(|_| (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect())
            .collect();

        for (label, storage) in [
            ("f32", Storage::F32),
            ("sq8", Storage::SQ8),
            ("rabitq", Storage::RaBitQ),
            ("turboquant", Storage::TurboQuant),
            ("turborabit", Storage::TurboRabit),
        ] {
            // Every field non-default, so a dropped field cannot hide behind its default.
            let config = HNSWConfig {
                metric: DistanceMetric::Cosine,
                m: 12,
                m0: 24,
                ef_construction: 150,
                ef_search: 80,
                storage,
                turbo_bits: 4,
                rabit_bits: 2,
                rerank_candidates: if storage == Storage::F32 { 0 } else { 40 },
                seed: Some(9),
                ..Default::default()
            };
            let mut src = HNSWIndex::build_parallel(base.clone(), config);
            // One incremental add with metadata, so `metadata` round-trips something real.
            src.add(crate::Document {
                id: "meta-doc".into(),
                content: "has metadata".into(),
                embedding: base[0].clone(),
                metadata: Some(serde_json::json!({"k": 1})),
            })
            .expect("add");

            let path = dir.path().join(format!("{label}.snap"));
            src.snapshot_to_file(&path).expect("snapshot");
            let re = HNSWIndex::snapshot_from_file(&path).expect("load");

            // Verbatim: the arena and every sibling structure, bit for bit.
            assert_eq!(re.nodes, src.nodes, "{label}: arena differs");
            assert_eq!(
                re.connections, src.connections,
                "{label}: upper layers differ"
            );
            assert_eq!(re.stride, src.stride, "{label}: derived stride differs");
            assert_eq!(re.hdr, src.hdr, "{label}: derived hdr differs");
            assert_eq!(
                re.entry_point, src.entry_point,
                "{label}: entry point differs"
            );
            assert_eq!(re.max_layer, src.max_layer, "{label}: max layer differs");
            assert_eq!(re.q_min, src.q_min, "{label}: q_min differs");
            assert_eq!(re.q_scale, src.q_scale, "{label}: q_scale differs");
            assert_eq!(re.full, src.full, "{label}: full vectors differ");
            assert_eq!(re.ids, src.ids, "{label}: ids differ");
            assert_eq!(re.contents, src.contents, "{label}: contents differ");
            assert_eq!(re.metadata, src.metadata, "{label}: metadata differs");
            assert_eq!(re.embedding_dim, src.embedding_dim);

            // And behaviourally: identical results, scores included (same arena, same
            // codebooks, same kernels — any difference is a load bug, not noise).
            for q in &queries {
                let a = src.search(q, 10).expect("src search");
                let b = re.search(q, 10).expect("re search");
                let a: Vec<(String, u32)> =
                    a.into_iter().map(|r| (r.id, r.score.to_bits())).collect();
                let b: Vec<(String, u32)> =
                    b.into_iter().map(|r| (r.id, r.score.to_bits())).collect();
                assert_eq!(a, b, "{label}: search results differ after load");
            }
        }
    }

    /// A snapshot is a same-version cache: a stamp from any other version (or a truncated
    /// arena) must refuse to load with a clear error, never misread.
    #[test]
    fn snapshot_rejects_version_mismatch_and_corruption() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut rng = StdRng::seed_from_u64(78);
        let base: Vec<Vec<f32>> = (0..100)
            .map(|_| (0..16).map(|_| rng.random::<f32>()).collect())
            .collect();
        let src = HNSWIndex::build_parallel(
            base,
            HNSWConfig {
                m: 8,
                m0: 16,
                seed: Some(1),
                ..Default::default()
            },
        );
        let path = dir.path().join("good.snap");
        src.snapshot_to_file(&path).expect("snapshot");

        let good = std::fs::read(&path).expect("read");
        let mut snap: HNSWSnapshot = bincode::deserialize(&good).expect("decode");

        // Wrong crate version.
        snap.crate_version = "0.0.0-other".into();
        let bad = dir.path().join("bad-version.snap");
        std::fs::write(&bad, bincode::serialize(&snap).unwrap()).unwrap();
        match HNSWIndex::snapshot_from_file(&bad) {
            Err(err) => assert!(
                err.to_string().contains("0.0.0-other"),
                "error should name the offending version, got: {err}"
            ),
            Ok(_) => panic!("wrong crate version must be rejected"),
        }

        // Wrong format version.
        snap.crate_version = env!("CARGO_PKG_VERSION").into();
        snap.format_version = SNAPSHOT_FORMAT_VERSION + 1;
        std::fs::write(&bad, bincode::serialize(&snap).unwrap()).unwrap();
        assert!(HNSWIndex::snapshot_from_file(&bad).is_err());

        // Truncated arena (valid bincode, wrong length for the config's stride).
        snap.format_version = SNAPSHOT_FORMAT_VERSION;
        snap.nodes.pop();
        std::fs::write(&bad, bincode::serialize(&snap).unwrap()).unwrap();
        assert!(
            HNSWIndex::snapshot_from_file(&bad).is_err(),
            "arena not a multiple of stride must be rejected"
        );

        // The untampered file still loads.
        assert!(HNSWIndex::snapshot_from_file(&path).is_ok());
    }

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
    // `HNSWIndex::build(.., Storage::RaBitQ + rerank_candidates: 0)` used to PANIC IN RELEASE:
    //   range start index 24 out of range for slice of length 0   (hnsw.rs, get_embedding)
    //
    // Both halves of that config are things the docs actively recommend: `rerank_candidates: 0`
    // is the README's "smallest index foxstash can build", and `BuildStrategy::Sequential` is
    // the #[default]. Nobody hit it only because every caller in the tree reached for
    // `build_parallel`, which builds the graph from the caller's f32 slice and never reads a
    // vector back. The one test covering this config hard-coded `build_parallel` and left a
    // comment explaining that `insert_node` "assumes `full` is populated" — the bug was
    // documented and walked around instead of fixed.
    //
    // This test goes through the PUBLIC `build`, on the DEFAULT strategy, for both quantized
    // modes, and checks the memory promise too: dropping the vectors is the entire point of
    // `rerank_candidates: 0`, so "it stopped panicking because we kept them" is not a fix.
    #[test]
    fn zero_rerank_quantized_builds_on_the_default_strategy_and_still_drops_its_vectors() {
        let mut rng = StdRng::seed_from_u64(11);
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

        // BOTH strategies, explicitly. `Sequential` is where the panic lived — it is the only
        // builder that reads vectors back out of storage mid-build. It used to be the #[default],
        // which is what made this a default-config crash; it no longer is (see `BuildStrategy`),
        // but it is still a supported public option, so it still must not blow up. Naming it
        // explicitly rather than leaning on the default also keeps this test from quietly
        // becoming a no-op the next time the default moves.
        for storage in [Storage::SQ8, Storage::RaBitQ] {
            for build_strategy in [BuildStrategy::Sequential, BuildStrategy::Parallel] {
                let config = HNSWConfig {
                    metric: DistanceMetric::L2,
                    storage,
                    rerank_candidates: 0,
                    seed: Some(4),
                    build_strategy,
                    ..Default::default()
                };

                let index = HNSWIndex::build(base.clone(), config);

                assert!(
                    index.full.is_empty(),
                    "{storage:?}/{build_strategy:?}: rerank_candidates = 0 must still DROP the f32 \
                     vectors — retaining them would 'fix' the panic by silently ignoring the \
                     caller's memory request"
                );
                assert_eq!(
                    index.rerank_candidates(),
                    0,
                    "{storage:?}/{build_strategy:?}: the caller's rerank_candidates must be \
                     restored after the build"
                );

                // And the graph must actually work — a build that produces a valid-but-empty
                // index would pass every assertion above.
                let mut hits = 0;
                for (i, q) in base.iter().enumerate().step_by(17) {
                    let got = index.search(q, 5).expect("search must not panic");
                    assert_eq!(got.len(), 5);
                    if got.iter().any(|r| r.id == i.to_string()) {
                        hits += 1;
                    }
                }
                assert!(
                    hits > 0,
                    "{storage:?}/{build_strategy:?}: index returns results but finds nothing — \
                     graph is broken"
                );
            }
        }
    }

    // `set_rerank_candidates` exists so the rerank pool can be swept at search time, the way
    // `set_ef_search` sweeps `ef` — the legacy `RaBitQHNSWIndex::search_and_rerank(q, pool, k)`
    // took the pool per call, and that was the one capability `Storage::RaBitQ` lacked.
    //
    // The interesting half is the REFUSAL. `rerank_candidates: 0` discards the f32 vectors, so
    // raising the pool afterwards has nothing to rescore against. Accepting it would silently
    // return the coarse ranking — a knob that reports success and does nothing, which is the
    // exact bug shape this codebase has now shipped ten times. So it must be an `Err`, and the
    // test asserts the error rather than just "doesn't panic".
    #[test]
    fn raising_the_rerank_pool_on_an_index_that_dropped_its_vectors_is_an_error() {
        let mut rng = StdRng::seed_from_u64(77);
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

        let cfg = |rerank| HNSWConfig {
            metric: DistanceMetric::L2,
            storage: Storage::RaBitQ,
            rerank_candidates: rerank,
            seed: Some(9),
            ..Default::default()
        };

        // Built WITHOUT the f32 vectors: raising the pool must be refused, not silently ignored.
        let mut dropped = HNSWIndex::build(base.clone(), cfg(0));
        assert!(
            matches!(
                dropped.set_rerank_candidates(64),
                Err(crate::RagError::FullPrecisionDropped)
            ),
            "raising the rerank pool on a vectors-dropped index must be an error"
        );
        assert_eq!(
            dropped.rerank_candidates(),
            0,
            "the refused set must not take effect"
        );
        // Lowering to 0 is always fine — nothing to rescore against is what it already wants.
        assert!(dropped.set_rerank_candidates(0).is_ok());

        // Built WITH the f32 vectors: the pool is a live search-time dial.
        let mut kept = HNSWIndex::build(base, cfg(100));
        assert!(kept.set_rerank_candidates(64).is_ok());
        assert_eq!(kept.rerank_candidates(), 64);
        assert!(kept.set_rerank_candidates(0).is_ok());
        assert_eq!(kept.rerank_candidates(), 0);
        // ...and back up again, because this index kept what it needs to honor that.
        assert!(kept.set_rerank_candidates(200).is_ok());
        assert_eq!(kept.rerank_candidates(), 200);
    }

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

    // ========================================================================
    // Quantized storage must honor `config.metric` (it silently ignored it and always
    // computed L2, which under the DEFAULT metric — Cosine — meant the whole walk ran under
    // the wrong metric and `score_from_distance` scored a squared-L2 value as if it were a
    // bounded cosine distance).
    // ========================================================================

    /// Directions with per-point norms scaled 0.5x-50x apart. Without varying norms, every
    /// point in a cluster has roughly the same magnitude and cosine/L2 rank near-identically
    /// — this fixture exists specifically so a metric mix-up *must* change the answer. See
    /// `cosine_and_l2_pick_different_neighbors` for the two-point version of the same idea;
    /// this is its recall-scale generalisation.
    fn nonuniform_norm_clusters(
        seed: u64,
        dim: usize,
        n_clusters: usize,
        per_cluster: usize,
        n_queries: usize,
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let directions: Vec<Vec<f32>> = (0..n_clusters)
            .map(|_| {
                let mut v: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
                crate::vector::ops::normalize(&mut v);
                v
            })
            .collect();

        let make = |n: usize, rng: &mut StdRng| -> Vec<Vec<f32>> {
            (0..n)
                .map(|i| {
                    let dir = &directions[i % n_clusters];
                    // Small angular jitter so points sharing a direction aren't identical.
                    let jittered: Vec<f32> =
                        dir.iter().map(|x| x + rng.random::<f32>() * 0.05).collect();
                    // Wildly different magnitude per point — the part that makes cosine and
                    // L2 disagree.
                    let scale = 0.5 + rng.random::<f32>() * 49.5;
                    jittered.into_iter().map(|x| x * scale).collect()
                })
                .collect()
        };

        let base = make(n_clusters * per_cluster, &mut rng);
        let queries = make(n_queries, &mut rng);
        (base, queries)
    }

    /// Prove the fixture above is discriminating *before* trusting any test built on it — an
    /// assertion that can't fail proves nothing (the self-retrieval trap, generalized).
    #[test]
    fn nonuniform_norm_fixture_discriminates_cosine_from_l2() {
        let (base, queries) = nonuniform_norm_clusters(11, 16, 10, 30, 20);

        let build = |metric: DistanceMetric| {
            HNSWIndex::build_parallel(
                base.clone(),
                HNSWConfig {
                    metric,
                    ef_construction: 150,
                    ef_search: 150,
                    seed: Some(1),
                    ..Default::default()
                },
            )
        };
        let cosine_idx = build(DistanceMetric::Cosine);
        let l2_idx = build(DistanceMetric::L2);

        let disagreements = queries
            .iter()
            .filter(|q| {
                let c = cosine_idx.search(q, 1).unwrap()[0].id.clone();
                let l = l2_idx.search(q, 1).unwrap()[0].id.clone();
                c != l
            })
            .count();

        assert!(
            disagreements * 2 >= queries.len(),
            "fixture is not discriminating: cosine and L2 only disagreed on {disagreements}/{} \
             queries — a metric mix-up test built on this fixture could pass by accident",
            queries.len()
        );
    }

    /// The regression test for the actual bug: `Storage::SQ8` with `..Default::default()` —
    /// metric deliberately NOT spelled out, so this exercises the default (Cosine) path the
    /// way a real caller who forgets to set `metric` would. Recall is measured against
    /// brute-force COSINE ground truth on held-out queries; the old, broken code would have
    /// silently run the walk under L2 instead, which — on this fixture, where cosine and L2
    /// disagree on most queries — would collapse recall against the cosine ground truth.
    #[test]
    fn sq8_default_metric_is_cosine_not_l2() {
        let (base, queries) = nonuniform_norm_clusters(23, 24, 12, 30, 40);

        let config = HNSWConfig {
            storage: Storage::SQ8,
            rerank_candidates: 50,
            ef_construction: 150,
            ef_search: 150,
            seed: Some(2),
            ..Default::default() // metric: Cosine, the default — not spelled out on purpose
        };
        assert_eq!(
            config.metric,
            DistanceMetric::Cosine,
            "test setup sanity check"
        );
        let index = HNSWIndex::build_parallel(base.clone(), config);

        let k = 10;
        let mut cosine_recall = 0.0f32;
        let mut l2_recall = 0.0f32;
        for q in &queries {
            let mut by_cosine: Vec<(f32, usize)> = base
                .iter()
                .enumerate()
                .map(|(i, v)| (1.0 - crate::vector::simd::cosine_similarity_simd(v, q), i))
                .collect();
            by_cosine.sort_by(|a, b| a.0.total_cmp(&b.0));
            let cosine_truth: HashSet<usize> = by_cosine.iter().take(k).map(|(_, i)| *i).collect();

            let mut by_l2: Vec<(f32, usize)> = base
                .iter()
                .enumerate()
                .map(|(i, v)| (crate::vector::simd::l2_squared_distance_simd(v, q), i))
                .collect();
            by_l2.sort_by(|a, b| a.0.total_cmp(&b.0));
            let l2_truth: HashSet<usize> = by_l2.iter().take(k).map(|(_, i)| *i).collect();

            let got: HashSet<usize> = index
                .search(q, k)
                .expect("search")
                .into_iter()
                .filter_map(|r| r.id.parse::<usize>().ok())
                .collect();

            cosine_recall += cosine_truth.intersection(&got).count() as f32 / k as f32;
            l2_recall += l2_truth.intersection(&got).count() as f32 / k as f32;
        }
        cosine_recall /= queries.len() as f32;
        l2_recall /= queries.len() as f32;

        assert!(
            cosine_recall > 0.6,
            "Storage::SQ8 with default metric: recall@{k} against COSINE ground truth = \
             {:.1}% — the default metric is meant to be cosine",
            cosine_recall * 100.0
        );
        assert!(
            cosine_recall > l2_recall + 0.2,
            "Storage::SQ8 with default metric answers cosine ({:.1}% recall) no better than \
             L2 ({:.1}% recall) — this is the exact shape of the metric-ignoring bug",
            cosine_recall * 100.0,
            l2_recall * 100.0
        );
    }

    /// Same regression test, `Storage::RaBitQ`.
    #[test]
    fn rabitq_default_metric_is_cosine_not_l2() {
        let (base, queries) = nonuniform_norm_clusters(29, 24, 12, 30, 40);

        let config = HNSWConfig {
            storage: Storage::RaBitQ,
            rerank_candidates: 50,
            ef_construction: 150,
            ef_search: 150,
            seed: Some(4),
            ..Default::default() // metric: Cosine, the default — not spelled out on purpose
        };
        assert_eq!(
            config.metric,
            DistanceMetric::Cosine,
            "test setup sanity check"
        );
        let index = HNSWIndex::build_parallel(base.clone(), config);

        let k = 10;
        let mut cosine_recall = 0.0f32;
        let mut l2_recall = 0.0f32;
        for q in &queries {
            let mut by_cosine: Vec<(f32, usize)> = base
                .iter()
                .enumerate()
                .map(|(i, v)| (1.0 - crate::vector::simd::cosine_similarity_simd(v, q), i))
                .collect();
            by_cosine.sort_by(|a, b| a.0.total_cmp(&b.0));
            let cosine_truth: HashSet<usize> = by_cosine.iter().take(k).map(|(_, i)| *i).collect();

            let mut by_l2: Vec<(f32, usize)> = base
                .iter()
                .enumerate()
                .map(|(i, v)| (crate::vector::simd::l2_squared_distance_simd(v, q), i))
                .collect();
            by_l2.sort_by(|a, b| a.0.total_cmp(&b.0));
            let l2_truth: HashSet<usize> = by_l2.iter().take(k).map(|(_, i)| *i).collect();

            let got: HashSet<usize> = index
                .search(q, k)
                .expect("search")
                .into_iter()
                .filter_map(|r| r.id.parse::<usize>().ok())
                .collect();

            cosine_recall += cosine_truth.intersection(&got).count() as f32 / k as f32;
            l2_recall += l2_truth.intersection(&got).count() as f32 / k as f32;
        }
        cosine_recall /= queries.len() as f32;
        l2_recall /= queries.len() as f32;

        assert!(
            cosine_recall > 0.5,
            "Storage::RaBitQ with default metric: recall@{k} against COSINE ground truth = \
             {:.1}%",
            cosine_recall * 100.0
        );
        assert!(
            cosine_recall > l2_recall + 0.2,
            "Storage::RaBitQ with default metric answers cosine ({:.1}% recall) no better \
             than L2 ({:.1}% recall) — this is the exact shape of the metric-ignoring bug",
            cosine_recall * 100.0,
            l2_recall * 100.0
        );
    }

    /// The old code fed a squared-L2 value (unbounded, frequently large) into
    /// `score_from_distance`'s `1.0 - dist` cosine formula, producing large negative scores.
    /// Under a correctly metric-aware SQ8, distances are true (rescored, exact) cosine
    /// distances in `[0, 2]`, so scores (`1 - dist`) must land in `[-1, 1]` and decrease
    /// monotonically as the true angle widens.
    #[test]
    fn sq8_cosine_scores_are_bounded_and_monotonic() {
        // Same axis, increasingly different directions; magnitudes deliberately unequal so
        // an L2-in-disguise bug (which cares about magnitude) would rank these differently
        // than a genuine cosine metric (which does not).
        let query = vec![10.0, 0.0, 0.0, 0.0];
        let vectors = [
            ("same_dir", vec![500.0, 0.0, 0.0, 0.0]), // identical direction, huge magnitude
            ("close_dir", vec![8.0, 2.0, 0.0, 0.0]),
            ("far_dir", vec![2.0, 8.0, 0.0, 0.0]),
            ("opposite_dir", vec![-30.0, 0.0, 0.0, 0.0]),
        ];

        let mut index = HNSWIndex::new(
            4,
            HNSWConfig {
                storage: Storage::SQ8,
                rerank_candidates: 100,
                ..Default::default() // metric: Cosine
            },
        );
        index
            .train(&vectors.iter().map(|(_, v)| v.clone()).collect::<Vec<_>>())
            .unwrap();
        for (id, v) in &vectors {
            index.add_embedding((*id).to_string(), v.clone()).unwrap();
        }

        let results = index.search(&query, vectors.len()).unwrap();
        assert_eq!(results.len(), vectors.len());
        for r in &results {
            assert!(
                (-1.0..=1.0).contains(&r.score),
                "SQ8 cosine score {} for {} outside [-1, 1] — the metric-ignoring bug fed a \
                 squared-L2 value into the cosine score formula",
                r.score,
                r.id
            );
        }
        for w in results.windows(2) {
            assert!(
                w[0].score >= w[1].score,
                "scores must be descending: {:?}",
                results
            );
        }
        assert_eq!(
            results[0].id, "same_dir",
            "identical direction must rank first under cosine"
        );
        assert_eq!(
            results.last().unwrap().id,
            "opposite_dir",
            "opposite direction must rank last under cosine"
        );
    }

    // ========================================================================
    // Discriminating tests for options flagged VACUOUS/UNCOVERED in the public-option audit:
    // each one had a test that set the field without any assertion able to tell whether it was
    // actually read. Every test below states, in its doc comment, the specific sabotage it
    // would catch ("if I hardcoded this to its default and deleted the config read, would this
    // fail?"), per the standard the rest of this module already holds itself to.
    //
    // NOT COMPILED. Written and reasoned through by hand while a benchmark held the CPU; the
    // team lead will compile and sabotage-verify these directly. Where a test's margin depends
    // on empirical behavior I could not run (rather than being guaranteed by construction), the
    // doc comment says so.
    // ========================================================================

    /// `ef_search` must bound how many candidates the layer-0 walk explores.
    ///
    /// Sabotage this catches: hardcode `ef` in `search_inner` to a fixed value (e.g.
    /// `k.max(100)`) instead of reading `self.config.ef_search`. `distance_calls()` would then
    /// stay flat no matter what a caller sets `ef_search` to, because the real code never
    /// explores more than the hardcoded constant.
    #[test]
    fn ef_search_controls_distance_calls() {
        let mut rng = StdRng::seed_from_u64(9001);
        let centers: Vec<Vec<f32>> = (0..20)
            .map(|_| (0..32).map(|_| rng.random::<f32>() * 10.0).collect())
            .collect();
        let embeddings: Vec<Vec<f32>> = (0..2000)
            .map(|i| {
                let c = &centers[i % 20];
                c.iter().map(|x| x + rng.random::<f32>() * 0.4).collect()
            })
            .collect();

        let mut index = HNSWIndex::build(embeddings.clone(), HNSWConfig::default().with_seed(7));
        let query = embeddings[0].clone();

        let calls_at = |index: &mut HNSWIndex, ef: usize| -> u64 {
            index.set_ef_search(ef);
            let mut searcher = index.searcher();
            searcher.search(&query, 10).unwrap();
            searcher.distance_calls()
        };

        let low = calls_at(&mut index, 10);
        let high = calls_at(&mut index, 800);

        assert!(
            high > low,
            "ef_search has no effect on work done: {high} distance calls at ef=800 vs {low} at \
             ef=10 — ef_search is being ignored"
        );
    }

    /// `ef_construction` must bound the candidate pool used while building each node's edges. A
    /// starved pool at build time produces a measurably worse GRAPH.
    ///
    /// Sabotage this catches: hardcode `ef_construction` in `insert_node` to a fixed value
    /// instead of reading `self.config.ef_construction`. Both configs below would then build the
    /// identical graph and score identical recall, regardless of which value a caller set.
    ///
    /// # The first version of this test was wrong, and wrong in an instructive way
    ///
    /// It held `ef_search: 300` at query time, with a comment claiming that "isolates the
    /// build-time effect". It does the **opposite**. A large search-time `ef` explores most of
    /// the corpus regardless of how the graph is wired, which *compensates for a bad graph* and
    /// masks the very thing under test. On 320 vectors with `ef_search: 300` the search is
    /// nearly exhaustive, so a graph built with `ef_construction: 1` still scored **98.7%** —
    /// the test failed, and it deserved to.
    ///
    /// To see graph quality you must make the search *depend* on it: a small `ef_search`, on a
    /// corpus too large to sweep by brute force. Then a badly-linked graph has nowhere to hide.
    /// The threshold below is unchanged from the original — the fixture was hardened rather than
    /// the assertion weakened, which is the rule whenever a test like this comes back red.
    #[test]
    fn ef_construction_controls_graph_quality() {
        let mut rng = StdRng::seed_from_u64(3113);
        let centers: Vec<Vec<f32>> = (0..16)
            .map(|_| (0..24).map(|_| rng.random::<f32>() * 10.0).collect())
            .collect();
        let base: Vec<Vec<f32>> = (0..800)
            .map(|i| {
                let c = &centers[i % 16];
                c.iter().map(|x| x + rng.random::<f32>() * 0.5).collect()
            })
            .collect();
        let queries: Vec<Vec<f32>> = (0..60)
            .map(|i| {
                let c = &centers[i % 16];
                c.iter().map(|x| x + rng.random::<f32>() * 0.5).collect()
            })
            .collect();

        let recall_for = |ef_construction: usize| -> f32 {
            let index = HNSWIndex::build(
                base.clone(),
                HNSWConfig {
                    m: 8,
                    m0: 16,
                    ef_construction,
                    // SMALL, deliberately. A generous `ef_search` papers over a badly-linked
                    // graph by exploring everything anyway — which is how the first version of
                    // this test scored 98.7% on a graph built with ef_construction = 1.
                    ef_search: 12,
                    seed: Some(11),
                    build_strategy: BuildStrategy::Sequential,
                    ..Default::default()
                },
            );
            let k = 10;
            let mut total = 0.0f32;
            for q in &queries {
                let mut exact: Vec<(f32, usize)> = base
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (1.0 - crate::vector::simd::cosine_similarity_simd(v, q), i))
                    .collect();
                exact.sort_by(|a, b| a.0.total_cmp(&b.0));
                let truth: HashSet<usize> = exact.iter().take(k).map(|(_, i)| *i).collect();

                let got: HashSet<usize> = index
                    .search(q, k)
                    .expect("search")
                    .into_iter()
                    .filter_map(|r| r.id.parse::<usize>().ok())
                    .collect();
                total += truth.intersection(&got).count() as f32 / k as f32;
            }
            total / queries.len() as f32
        };

        let starved = recall_for(1);
        let generous = recall_for(200);

        assert!(
            generous > starved + 0.1,
            "ef_construction has no measurable effect on graph quality: recall@10 = {:.3} at \
             ef_construction=1 vs {:.3} at ef_construction=200 — ef_construction is being \
             ignored at build time",
            starved,
            generous
        );
    }

    /// `use_heuristic` must select which neighbour-selection algorithm actually runs: Algorithm
    /// 4's diversity heuristic (default) vs plain nearest-M. Whitebox test of `select_neighbors`
    /// directly — the two algorithms are *proven* to disagree on this exact fixture by hand
    /// below, so there is no fixture-sensitivity risk the way an end-to-end recall test has.
    ///
    /// Fixture (2-D, `DistanceMetric::L2`, query at the origin):
    ///   A = (1.00, 0.0)  dist to query = 1.00
    ///   B = (1.05, 0.0)  dist to query = 1.05, dist to A  = 0.05
    ///   C = (0.00, 1.2)  dist to query = 1.20, dist to A  = 1.562
    ///
    /// Nearest-2 by raw distance: {A, B} — 1.00 and 1.05 both beat 1.20.
    /// Algorithm 4 with m=2: A is accepted first (always is). B is checked against A:
    /// dist(B,A)=0.05 < dist(B,query)=1.05, so B is "behind" A and pruned. C is checked against
    /// A: dist(C,A)=1.562 is NOT less than dist(C,query)=1.20, so C is accepted. Heuristic result:
    /// {A, C}.
    ///
    /// Sabotage this catches: hardcode `use_heuristic` to `true` (delete the `if
    /// !self.config.use_heuristic` early return in `select_neighbors`) — the `false` config
    /// below would then also return {A, C} instead of {A, B}.
    #[test]
    fn use_heuristic_selects_a_different_neighbor_set_than_simple() {
        let a = [1.0f32, 0.0];
        let b = [1.05f32, 0.0];
        let c = [0.0f32, 1.2];
        let query = [0.0f32, 0.0];

        let build_index = |use_heuristic: bool| -> HNSWIndex {
            let config = HNSWConfig {
                metric: DistanceMetric::L2,
                use_heuristic,
                extend_candidates: false,
                ..Default::default()
            };
            let mut index = HNSWIndex::new(2, config);
            index.push_node(&a); // id 0
            index.push_node(&b); // id 1
            index.push_node(&c); // id 2
            index
        };

        let heuristic_selected: HashSet<usize> = build_index(true)
            .select_neighbors(&[0, 1, 2], &query, 2, 0)
            .into_iter()
            .collect();
        let simple_selected: HashSet<usize> = build_index(false)
            .select_neighbors(&[0, 1, 2], &query, 2, 0)
            .into_iter()
            .collect();

        assert_eq!(
            heuristic_selected,
            HashSet::from([0, 2]),
            "Algorithm-4 heuristic should pick the diverse pair {{A, C}}, got \
             {heuristic_selected:?}"
        );
        assert_eq!(
            simple_selected,
            HashSet::from([0, 1]),
            "simple selection should pick the two nearest {{A, B}}, got {simple_selected:?}"
        );
        assert_ne!(
            heuristic_selected, simple_selected,
            "use_heuristic has no effect: both configs picked the same neighbours — \
             use_heuristic is being ignored"
        );
    }

    /// `extend_candidates` must broaden the pool `select_neighbors`'s heuristic prunes from —
    /// pulling in each direct candidate's own layer-0 neighbours before scoring. Whitebox again:
    /// D is the only member of `candidates`; a strictly-second point E is reachable *exclusively*
    /// through D's layer-0 neighbour list, never passed to `select_neighbors` directly. Without
    /// `extend_candidates`, `select_neighbors` cannot see E at all. With it, D's neighbour list
    /// is walked and E enters the working pool before pruning.
    ///
    /// `keep_pruned_connections: true` (the default) is held fixed in both configs so the size
    /// difference below is attributable only to `extend_candidates`, not to whether pruned
    /// candidates get backfilled.
    ///
    /// Sabotage this catches: hardcode `extend_candidates` to `false` (delete the `if
    /// self.config.extend_candidates` block in `select_neighbors`, or make it a no-op) — the
    /// `true` config below would then also return only `{D}`, size 1, instead of `{D, E}`, size 2.
    #[test]
    fn extend_candidates_pulls_in_neighbors_of_candidates() {
        let d = [1.0f32, 0.0];
        let e = [2.0f32, 0.0];
        let query = [0.0f32, 0.0];

        let build_index = |extend_candidates: bool| -> HNSWIndex {
            let config = HNSWConfig {
                metric: DistanceMetric::L2,
                use_heuristic: true,
                extend_candidates,
                keep_pruned_connections: true,
                m0: 4,
                ..Default::default()
            };
            let mut index = HNSWIndex::new(2, config);
            index.push_node(&d); // id 0
            index.push_node(&e); // id 1
                                 // D's only layer-0 neighbour is E. `candidates` passed to `select_neighbors` below
                                 // is `[0]` (D) only — E is reachable exclusively by walking this link, which only
                                 // happens when `extend_candidates` is set.
            index.l0_push(0, 1);
            index
        };

        let extended = build_index(true).select_neighbors(&[0], &query, 2, 0);
        let not_extended = build_index(false).select_neighbors(&[0], &query, 2, 0);

        assert_eq!(
            not_extended.len(),
            1,
            "without extend_candidates, only the directly-passed candidate D can be selected, \
             got {not_extended:?}"
        );
        assert_eq!(
            extended.len(),
            2,
            "extend_candidates has no effect: expected D's neighbour E to be pulled into the \
             pool and selected alongside D, got {extended:?} — extend_candidates is being \
             ignored"
        );
    }

    /// `HNSWConfig::m` must bound the number of edges kept per node at layers >= 1, independent
    /// of `m0` (which bounds layer 0 only — see `keep_pruned_connections_controls_graph_density_
    /// in_both_builders` for that one). `m0`, `ml`, `ef_construction` and `seed` are held
    /// IDENTICAL between the two configs below; only `m` differs, so any difference in average
    /// layer-1 degree is attributable to `m` alone. `ml` is fixed at an unusually high 0.8
    /// (rather than the typical `1/ln(m)`) purely to get enough nodes above layer 0 to average
    /// over; sharing both `seed` and `ml` means the two builds assign the *same* set of nodes to
    /// layer >= 1, so the population being averaged over is identical too.
    ///
    /// # This test was VACUOUS in its first form, and the way it failed is worth keeping
    ///
    /// It originally compared the *average* layer-1 degree at `m=4` vs `m=32` and asserted the
    /// wide one was 2x the narrow one. But `config.m` is read in TWO places during insertion:
    /// the cap on the new node's own edge count, and the pruning of each existing neighbour's
    /// edge list. Sabotaging *either one alone* left the other to spread the averages apart, so
    /// the test still passed. It only failed if you broke **both**. That is an OR, not an AND —
    /// and a real regression hardcodes ONE site. The test would have sailed straight past the
    /// bug it was written to catch. (Verified: hardcoding each site in turn → still green.)
    ///
    /// So don't measure a statistic; assert the **invariant**. With `m = 4`, no node above layer
    /// 0 may hold more than 4 neighbours — *ever*. Either read site breaking that produces an
    /// over-degree node, and there is nowhere for it to hide in an average.
    ///
    /// Sabotage this now catches: hardcode EITHER `self.config.m` site in `insert_node` (the
    /// select-neighbours cap, or `neighbor_m` in the prune step) to a constant.
    ///
    /// The control comes first, as always: assert the cap actually BINDS on this fixture (some
    /// node genuinely reaches `m` neighbours). If nothing ever reaches the cap, "no node exceeds
    /// the cap" is trivially true and proves nothing — that is exactly how the `ml: 0.8` fixture
    /// hid this, by promoting too few nodes for any degree limit to matter.
    /// `use_heuristic` and `extend_candidates` must be honoured by **both** builders.
    ///
    /// They were not. `par_select_heuristic` took only `(metric, sorted, points, m, keep_pruned)`
    /// and the entire parallel build path mentioned these two options **zero times** — so
    /// `use_heuristic: false` / `extend_candidates: true` were silently ignored on the DEFAULT
    /// builder. The pre-existing tests missed it because they call `select_neighbors` directly:
    /// the *sequential* path. **An option tested against one implementation of a strategy is not a
    /// tested option.**
    ///
    /// # This test needs a NOISE FLOOR, and the first version of it did not have one
    ///
    /// My first attempt compared "graph with the option on" against "graph with it off" and
    /// asserted they differed on >5% of nodes. It passed under sabotage. The reason is its own
    /// bug (see `seed`): **the parallel builder is not reproducible even at a fixed seed** —
    /// threads race to write neighbour lists, and two builds of the *identical* config differ on
    /// ~15% of nodes. The test was measuring the race, not the option.
    ///
    /// So the control comes first and it is not optional: build the SAME config twice, measure how
    /// much it varies from thread scheduling alone, and only then require the option's effect to
    /// exceed that floor by a clear margin. A difference smaller than the noise is not evidence.
    /// `build_parallel` shuffles its insertion order. The ids it hands back must still be the
    /// caller's ORIGINAL row indices.
    ///
    /// This is load-bearing far outside this crate. The Python binding maps a `SearchResult` back
    /// to a row of the `X` numpy handed us by parsing `r.id` as an integer — so if the shuffle
    /// leaked, every recall number the binding reported would be scored against the wrong
    /// ground-truth rows. It would not crash. It would not look wrong. It would be fiction, and we
    /// would publish it. The binding asserted this in a doc comment, and a doc comment cannot fail.
    ///
    /// It asserts the MAPPING, not the search. The first version of this test queried each row
    /// with its own vector and demanded itself back at k=1 — and failed, on a correct index, at
    /// the default `ef_search`, because an approximate index is allowed to miss. Which is to say
    /// it was testing recall while claiming to test identity. The direct check below cannot be
    /// confused by search quality: node `j` stores the vector of the row whose id it claims.
    #[test]
    fn build_parallel_returns_original_row_indices_despite_its_shuffle() {
        let n = 300;
        let base: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let mut v = vec![0.0f32; 32];
                v[i % 32] = 1.0 + i as f32;
                v[(i * 7 + 3) % 32] = 0.5 + (i % 13) as f32;
                v
            })
            .collect();

        let ix = HNSWIndex::build(
            base.clone(),
            HNSWConfig {
                metric: DistanceMetric::L2,
                seed: Some(4),
                build_strategy: BuildStrategy::Parallel,
                ..Default::default()
            },
        );
        assert_eq!(ix.len(), n, "the build dropped or duplicated rows");

        for j in 0..ix.len() {
            let claimed: usize = ix.ids[j]
                .parse()
                .expect("build_parallel labels every node with its original row index");
            assert_eq!(
                ix.get_embedding(j),
                base[claimed].as_slice(),
                "node {j} claims to be row {claimed}, but the vector it stores is not row \
                 {claimed}'s. build_parallel's insertion shuffle has leaked into the ids it \
                 returns, so every id this index reports is a permutation of the caller's rows."
            );
        }

        // End to end, through the public API -- but on the SEQUENTIAL builder.
        //
        // This half of the test was flaky and it was my own fault. It ran on the default parallel
        // builder and demanded, for each probe row, that the index return that exact row at k=1.
        // The parallel builder is not reproducible (see
        // `seed_gives_reproducible_builds_only_on_the_sequential_builder`), so whether any given
        // node lands well-connected varies run to run, and an approximate index is entitled to
        // miss one. It passed for a while and then failed on an unrelated commit, which is the
        // worst way for a test to spend its time.
        //
        // An exact assertion needs a deterministic build. The mapping claim itself is already
        // proven exhaustively above, for all n nodes, without going through search at all.
        let seq = HNSWIndex::build(
            base.clone(),
            HNSWConfig {
                metric: DistanceMetric::L2,
                seed: Some(4),
                ef_search: 300,
                build_strategy: BuildStrategy::Sequential,
                ..Default::default()
            },
        );
        for i in [0, 7, 42, 199, 292, n - 1] {
            assert_eq!(
                seq.search(&base[i], 1).unwrap()[0].id,
                i.to_string(),
                "row {i} queried with its own vector did not come back as itself"
            );
        }
    }

    /// The two builders must produce the SAME GRAPH, to within the parallel builder's thread noise.
    ///
    /// This is the guard on the bug class, rather than on any one bug. Both builders now call one
    /// [`select_neighbors_core`], so an option cannot be honoured by one and ignored by the other —
    /// but they still adapt it to different graph storage, and an adapter can lie. This test fails
    /// the moment those adapters disagree about anything: a dropped option, a wrong distance, a
    /// different pruning rule.
    ///
    /// It is the test that would have caught, in one shot, every builder bug in the 1.0 audit:
    ///
    /// | bug | the gap it opened |
    /// |---|---|
    /// | parallel ignored `m`/`m0` | degree pinned at the capacity constant |
    /// | parallel ignored `use_heuristic` | 16.00 vs 8.11 |
    /// | parallel ignored `extend_candidates` | 7.99 vs 9.06 |
    /// | parallel scored extended candidates against the wrong point | 10.58 vs 9.06 |
    ///
    /// The last of those was introduced *while fixing* the third, and shipped for three commits
    /// under a test that watched degree move and concluded the option worked. Degree moved. It
    /// moved to the wrong number.
    #[test]
    fn both_builders_produce_the_same_graph() {
        let mut rng = StdRng::seed_from_u64(31337);
        let centers: Vec<Vec<f32>> = (0..8)
            .map(|_| (0..16).map(|_| rng.random::<f32>() * 10.0).collect())
            .collect();
        let base: Vec<Vec<f32>> = (0..600)
            .map(|i| {
                let c: &Vec<f32> = &centers[i % 8];
                c.iter().map(|x| x + rng.random::<f32>() * 0.4).collect()
            })
            .collect();

        let avg_degree = |strategy: BuildStrategy, heuristic: bool, extend: bool| -> f64 {
            let reps = 3;
            (0..reps)
                .map(|r| {
                    let ix = HNSWIndex::build(
                        base.clone(),
                        HNSWConfig {
                            metric: DistanceMetric::L2,
                            m: 8,
                            m0: 16,
                            ef_construction: 16,
                            seed: Some(21 + r),
                            use_heuristic: heuristic,
                            extend_candidates: extend,
                            keep_pruned_connections: false,
                            build_strategy: strategy,
                            ..Default::default()
                        },
                    );
                    let total: usize = (0..ix.len()).map(|i| ix.get_neighbors_l0(i).len()).sum();
                    total as f64 / ix.len() as f64
                })
                .sum::<f64>()
                / reps as f64
        };

        for (heuristic, extend) in [(false, false), (true, false), (true, true)] {
            let par = avg_degree(BuildStrategy::Parallel, heuristic, extend);
            let seq = avg_degree(BuildStrategy::Sequential, heuristic, extend);
            assert!(
                (par - seq).abs() < 0.5,
                "use_heuristic={heuristic} extend_candidates={extend}: the parallel builder \
                 produced average degree {par:.2} and the sequential one {seq:.2}. They run the \
                 same algorithm on the same seed, so a gap this size means an adapter is dropping \
                 an option or computing a distance against the wrong point."
            );
        }
    }

    /// Degree equivalence (above) is a PROXY for graph quality; recall at fixed `ef_search` IS the
    /// quality. The two builders once matched on layer-0 degree (64 vs 64) while the parallel graph
    /// recalled ~15 points BELOW the sequential one — because `par_insert` truncated the candidate
    /// beam to `m0` before the diversity heuristic, so the heuristic picked `m0` from `m0` and could
    /// not bridge clusters. The degree test could never see it. This one measures the output.
    ///
    /// If the parallel builder ever again feeds its heuristic a truncated pool (or otherwise builds
    /// a worse graph), parallel recall drops below sequential and the parity assertion fails. The
    /// floor assertion keeps it from passing vacuously by comparing two equally-broken indexes.
    #[test]
    fn both_builders_reach_similar_recall() {
        // Fixture chosen to EXPOSE the failure mode, not merely to run: UNIT-NORMALISED vectors
        // with gaussian noise, so the clusters sit close on the sphere and a query's true top-10
        // spans cluster boundaries — which the diversity heuristic must bridge. Queried at a modest
        // ef_search where a bridge-poor graph visibly loses recall. Verified by sabotage: truncating
        // the candidate pool to m0 before the heuristic drops parallel recall ~15 points here and
        // fails the parity assertion below. Well-separated clusters (uniform centers, tiny noise)
        // give BOTH builders 100% and hide the gap — the classic wrong fixture (benchmarking-traps /
        // fixture-per-failure-mode lessons).
        let mut rng = StdRng::seed_from_u64(0xEF54);
        let dim = 128;
        let norm = |v: &mut Vec<f32>| {
            let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if n > 0.0 {
                v.iter_mut().for_each(|x| *x /= n);
            }
        };
        // Box-Muller standard normal from the StdRng uniform stream.
        let mut gauss = |rng: &mut StdRng| -> f32 {
            let u1 = (rng.random::<f32>()).max(1e-7);
            let u2 = rng.random::<f32>();
            (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
        };
        let unit = |rng: &mut StdRng| -> Vec<f32> {
            let mut v: Vec<f32> = (0..dim).map(|_| gauss(rng)).collect();
            norm(&mut v);
            v
        };
        let sample_from = |rng: &mut StdRng,
                           centers: &[Vec<f32>],
                           gauss: &mut dyn FnMut(&mut StdRng) -> f32|
         -> Vec<f32> {
            let c = &centers[rng.random::<u64>() as usize % centers.len()];
            let mut v: Vec<f32> = c.iter().map(|x| x + 0.05 * gauss(rng)).collect();
            norm(&mut v);
            v
        };
        // Base and queries are drawn from INDEPENDENT centre sets, so a query's true top-10 is not
        // trivially its own dense cluster — it requires real cross-cluster search, which is what
        // stresses the graph's long-range bridges. (Queries from the SAME centres as the base give
        // both builders ~100% and hide the gap.)
        let base_centers: Vec<Vec<f32>> = (0..100).map(|_| unit(&mut rng)).collect();
        let query_centers: Vec<Vec<f32>> = (0..100).map(|_| unit(&mut rng)).collect();
        let base: Vec<Vec<f32>> = (0..10_000)
            .map(|_| sample_from(&mut rng, &base_centers, &mut gauss))
            .collect();
        let queries: Vec<Vec<f32>> = (0..200)
            .map(|_| sample_from(&mut rng, &query_centers, &mut gauss))
            .collect();

        const K: usize = 10;
        let truth: Vec<Vec<usize>> = queries
            .iter()
            .map(|q| {
                let mut d: Vec<(f32, usize)> = base
                    .iter()
                    .enumerate()
                    .map(|(j, v)| {
                        (
                            q.iter().zip(v).map(|(a, b)| (a - b).powi(2)).sum::<f32>(),
                            j,
                        )
                    })
                    .collect();
                d.sort_by(|a, b| a.0.total_cmp(&b.0));
                d.iter().take(K).map(|(_, j)| *j).collect()
            })
            .collect();

        let recall_of = |strategy: BuildStrategy| -> f32 {
            let mut ix = HNSWIndex::build(
                base.clone(),
                HNSWConfig {
                    metric: DistanceMetric::L2,
                    m: 32,
                    m0: 64,
                    ef_construction: 200,
                    ef_search: 40,
                    seed: Some(7),
                    build_strategy: strategy,
                    ..Default::default()
                },
            );
            ix.set_ef_search(40);
            let mut hit = 0.0f32;
            for (qi, q) in queries.iter().enumerate() {
                let got: std::collections::HashSet<usize> = ix
                    .search(q, K)
                    .unwrap()
                    .iter()
                    .filter_map(|r| r.id.parse::<usize>().ok())
                    .collect();
                hit += truth[qi].iter().filter(|t| got.contains(t)).count() as f32 / K as f32;
            }
            hit / queries.len() as f32
        };

        let seq = recall_of(BuildStrategy::Sequential);
        let par = recall_of(BuildStrategy::Parallel);

        // Non-vacuous: both builders must actually work on this clustered data.
        assert!(
            seq > 0.65 && par > 0.55,
            "recall floor not met (seq={seq:.3}, par={par:.3}) — the test is comparing broken \
             indexes and would pass vacuously"
        );
        // The point of the test: the default (parallel) builder must not ship a materially worse
        // graph than the sequential one. 5 points covers the parallel builder's non-reproducibility.
        assert!(
            seq - par < 0.05,
            "parallel recall {par:.3} trails sequential {seq:.3} by {:.3} at equal ef_search — the \
             parallel builder is producing a worse graph (last time: it truncated the candidate \
             pool to m0 before the diversity heuristic).",
            seq - par
        );
    }

    #[test]
    fn use_heuristic_and_extend_candidates_are_honoured_by_both_builders() {
        let mut rng = StdRng::seed_from_u64(31337);
        let centers: Vec<Vec<f32>> = (0..8)
            .map(|_| (0..16).map(|_| rng.random::<f32>() * 10.0).collect())
            .collect();
        let base: Vec<Vec<f32>> = (0..600)
            .map(|i| {
                let c: &Vec<f32> = &centers[i % 8];
                c.iter().map(|x| x + rng.random::<f32>() * 0.4).collect()
            })
            .collect();

        const M0: usize = 16;
        // Averaged over builds, so that thread scheduling shows up as a small wobble in the mean
        // rather than as a signal. `keep_pruned_connections: false` is essential: the backfill
        // exists precisely to refill the slots the heuristic emptied, and would erase the effect.
        let avg_degree = |heuristic: bool, extend: bool, strategy: BuildStrategy| -> f64 {
            // 6, not 3: this averages away the parallel builder's thread-scheduling wobble, and a
            // CI runner with a different core count than the dev box wobbles differently. More
            // reps pull each mean closer to its true expectation, so the effect sizes below clear
            // their margins on every platform (this test was flaky on Windows/macOS at reps=3).
            let reps = 6;
            (0..reps)
                .map(|r| {
                    let ix = HNSWIndex::build(
                        base.clone(),
                        HNSWConfig {
                            metric: DistanceMetric::L2,
                            m: 8,
                            m0: M0,
                            ef_construction: 16,
                            seed: Some(21 + r),
                            use_heuristic: heuristic,
                            extend_candidates: extend,
                            keep_pruned_connections: false,
                            build_strategy: strategy,
                            ..Default::default()
                        },
                    );
                    let total: usize = (0..ix.len()).map(|i| ix.get_neighbors_l0(i).len()).sum();
                    total as f64 / ix.len() as f64
                })
                .sum::<f64>()
                / reps as f64
        };

        for strategy in [BuildStrategy::Parallel, BuildStrategy::Sequential] {
            let greedy = avg_degree(false, false, strategy);
            let heuristic = avg_degree(true, false, strategy);
            let extended = avg_degree(true, true, strategy);

            // The noise floor, measured on the same statistic the assertions use. The parallel
            // builder is not reproducible (see `seed_gives_reproducible_builds_only_on_the_...`),
            // so an effect must be shown to exceed the wobble, not merely to exist.
            //
            // A SINGLE resample diff is itself one draw from that wobble — on a differently-
            // scheduled runner it can land high and break a `noise * 10` guard calibrated on the
            // dev box (the Windows/macOS flake). Average several independent same-config pairs so
            // the floor is a stable statistic. Sequential is deterministic, so its floor is 0.
            let noise = {
                let pairs = 5;
                (0..pairs)
                    .map(|_| {
                        (avg_degree(true, false, strategy) - avg_degree(true, false, strategy))
                            .abs()
                    })
                    .sum::<f64>()
                    / pairs as f64
            };

            // Greedy selection takes the m0 nearest candidates and fills every slot.
            assert!(
                (greedy - M0 as f64).abs() < 0.01,
                "{strategy:?}: use_heuristic=false must keep the m0 nearest candidates, filling \
                 every slot, but average degree was {greedy:.2} of a possible {M0}"
            );
            // The heuristic rejects any candidate that lies closer to an already-accepted
            // neighbour than to the query, which empties slots that greedy would have filled.
            // Measured ~8/16; require a margin far above `noise` (~0.05).
            assert!(
                heuristic < 0.75 * M0 as f64 && (greedy - heuristic) > noise * 10.0,
                "{strategy:?}: use_heuristic=true must prune candidates that hide behind an \
                 accepted neighbour, dropping degree below m0={M0}, but degree was \
                 {heuristic:.2} vs greedy's {greedy:.2} (build-to-build noise {noise:.3}) — the \
                 flag is not reaching this builder"
            );
            // Extending the candidate set with the neighbours-of-neighbours gives the heuristic
            // more diverse candidates to accept, so degree climbs back. Measured +12% (sequential)
            // to +33% (parallel).
            assert!(
                extended > heuristic * 1.05 && (extended - heuristic) > noise * 10.0,
                "{strategy:?}: extend_candidates=true must widen the candidate pool and let the \
                 heuristic accept more of it, but degree was {extended:.2} vs {heuristic:.2} \
                 without it (build-to-build noise {noise:.3}) — the flag is not reaching this \
                 builder"
            );
        }
    }

    /// `seed` promises reproducible builds. **It does not deliver them on the default builder.**
    ///
    /// Measured: two `BuildStrategy::Parallel` builds at the same seed differ on ~15% of nodes;
    /// `Sequential` is bit-identical. Threads race to write neighbour lists through the `RwLock`s,
    /// and the seed fixes the level assignments and insertion order — not the interleaving.
    ///
    /// The pre-existing `seed_makes_builds_reproducible_and_distinguishable` test pinned
    /// `Sequential` — the path where the promise happens to hold — so it passed. Third instance of
    /// the same pattern today, after `m`/`m0` and `use_heuristic`/`extend_candidates`.
    ///
    /// This test pins the ACTUAL guarantee rather than the one the docs used to imply, so that the
    /// day someone makes the parallel build deterministic, it fails and tells them to update the
    /// contract. A test that encodes a lie is worse than no test; a test that encodes the truth,
    /// including an unwelcome truth, is a spec.
    #[test]
    fn seed_gives_reproducible_builds_only_on_the_sequential_builder() {
        let base: Vec<Vec<f32>> = (0..600)
            .map(|i| {
                (0..16)
                    .map(|d| (((i * 7 + d * 13) % 50) as f32) * 0.1 + (i % 8) as f32)
                    .collect()
            })
            .collect();

        let graph_of = |strategy: BuildStrategy| -> Vec<Vec<u32>> {
            let ix = HNSWIndex::build(
                base.clone(),
                HNSWConfig {
                    metric: DistanceMetric::L2,
                    m: 8,
                    m0: 16,
                    ef_construction: 100,
                    seed: Some(21),
                    build_strategy: strategy,
                    ..Default::default()
                },
            );
            (0..ix.len())
                .map(|i| {
                    let mut n: Vec<u32> = ix.get_neighbors_l0(i).to_vec();
                    n.sort_unstable();
                    n
                })
                .collect()
        };

        let seq_a = graph_of(BuildStrategy::Sequential);
        let seq_b = graph_of(BuildStrategy::Sequential);
        assert_eq!(
            seq_a, seq_b,
            "Sequential + a fixed seed must be bit-reproducible — that is the whole contract of \
             `seed`, and it is the only builder that honours it"
        );

        // And the unwelcome half of the truth, pinned so it cannot rot silently.
        let par_a = graph_of(BuildStrategy::Parallel);
        let par_b = graph_of(BuildStrategy::Parallel);
        let differing = par_a.iter().zip(&par_b).filter(|(x, y)| x != y).count();
        assert!(
            differing > 0,
            "the parallel builder has become reproducible under a fixed seed ({differing} nodes \
             differ). That is GOOD — but `HNSWConfig::seed`'s documentation says it is not, and \
             this test exists to catch that contract changing. Update the docs and this assertion."
        );
    }

    /// The sibling of `m_caps_upper_layer_degree_independent_of_m0`, and the test that should
    /// have existed first.
    ///
    /// That test pinned `BuildStrategy::Sequential`. The PARALLEL builder — now the default —
    /// ignored `config.m` and `config.m0` **entirely**: it passed the hardcoded `M0_MAX`/`M_MAX`
    /// constants everywhere the config values belonged, built a degree-64 graph no matter what
    /// you asked for, and let the conversion clean up afterwards. Build cost was FLAT at ~61,500
    /// distance computations per insert across m0 = 24/32/48/64 — it was doing the m0=64 build
    /// every single time.
    ///
    /// I wrote the Sequential test and then made the untested builder the default. **Testing an
    /// option against ONE implementation of a strategy is not testing the option.**
    ///
    /// # This test is on LAYER 1, and that is not an accident
    ///
    /// The obvious version — assert layer-0 degree respects `m0` — is VACUOUS, and I wrote it
    /// that way first and watched the sabotage pass. Layer 0 is stored in the flat node block,
    /// whose stride is `node_stride(config.m0, ..)`: the block physically has room for exactly
    /// `m0` neighbours, so the surplus is silently dropped at conversion no matter what the
    /// builder did. The storage layout *launders* the bug. Layer >= 1 lives in `UpperNode`
    /// (capacity `M_MAX = 32`) and is copied out untruncated — so that is the one place where
    /// "the builder ignored `config.m`" is actually observable.
    ///
    /// Sabotage this catches: revert `UpperNode::from_zero(z, m)` to `take(M_MAX)`, or
    /// `search_upper(.., m)` to `M_MAX`.
    #[test]
    fn m_caps_upper_layer_degree_in_the_parallel_builder_too() {
        let mut rng = StdRng::seed_from_u64(4242);
        let centers: Vec<Vec<f32>> = (0..10)
            .map(|_| (0..16).map(|_| rng.random::<f32>() * 10.0).collect())
            .collect();
        let embeddings: Vec<Vec<f32>> = (0..2000)
            .map(|i| {
                let c = &centers[i % 10];
                c.iter().map(|x| x + rng.random::<f32>() * 0.5).collect()
            })
            .collect();

        let build = |m: usize| {
            HNSWIndex::build(
                embeddings.clone(),
                HNSWConfig {
                    m,
                    m0: 64,
                    ef_construction: 100,
                    seed: Some(5),
                    keep_pruned_connections: true, // saturate, so the cap actually binds
                    build_strategy: BuildStrategy::Parallel, // THE DEFAULT. The untested path.
                    ..Default::default()
                },
            )
        };

        // Peak degree over every node present at layer 1.
        let peak_l1 = |ix: &HNSWIndex| -> (usize, usize) {
            let mut peak = 0;
            let mut count = 0;
            for id in 0..ix.len() {
                if ix.connections[id].len() > 1 {
                    peak = peak.max(ix.connections[id][1].len());
                    count += 1;
                }
            }
            (peak, count)
        };

        let (narrow_peak, narrow_count) = peak_l1(&build(4));
        let (wide_peak, _) = peak_l1(&build(32));

        // CONTROL 1: layer 1 is actually populated. A cap cannot bind on an empty layer.
        assert!(
            narrow_count > 10,
            "fixture put only {narrow_count} nodes on layer 1 — the assertions below would be \
             vacuous"
        );
        // CONTROL 2: the cap BINDS. Something must reach m=4, or "nothing exceeds 4" is free.
        assert_eq!(
            narrow_peak, 4,
            "the m=4 cap never bound (peak layer-1 degree {narrow_peak}) — nothing is pressing \
             against the limit, so the invariant below proves nothing"
        );

        // THE INVARIANT: m = 4 means no layer-1 node may hold more than 4 edges.
        assert!(
            narrow_peak <= 4,
            "m = 4 but a layer-1 node holds {narrow_peak} neighbours — the parallel builder is \
             ignoring config.m (it used to hardcode M_MAX = 32)"
        );
        // ...and m = 32 must genuinely produce a wider graph.
        assert!(
            wide_peak > 4,
            "m = 32 produced a peak layer-1 degree of only {wide_peak}, no better than m=4 — \
             config.m is not reaching the parallel builder"
        );
    }

    #[test]
    fn m_caps_upper_layer_degree_independent_of_m0() {
        let mut rng = StdRng::seed_from_u64(5150);
        let centers: Vec<Vec<f32>> = (0..12)
            .map(|_| (0..16).map(|_| rng.random::<f32>() * 10.0).collect())
            .collect();
        let embeddings: Vec<Vec<f32>> = (0..800)
            .map(|i| {
                let c = &centers[i % 12];
                c.iter().map(|x| x + rng.random::<f32>() * 0.5).collect()
            })
            .collect();

        // Max degree over every node present at `layer`, and how many nodes that was.
        let peak_degree_at = |index: &HNSWIndex, layer: usize| -> (usize, usize) {
            let mut peak = 0usize;
            let mut count = 0usize;
            for node_id in 0..index.len() {
                if layer < index.connections[node_id].len() {
                    peak = peak.max(index.connections[node_id][layer].len());
                    count += 1;
                }
            }
            (peak, count)
        };

        let build_with_m = |m: usize| -> HNSWIndex {
            HNSWIndex::build(
                embeddings.clone(),
                HNSWConfig {
                    m,
                    m0: 64,
                    // ml: 3.0, NOT the 0.8 this test used to carry. `ml` sets how aggressively
                    // nodes are promoted above layer 0; at 0.8 barely any were, so no node ever
                    // accumulated enough neighbours for a degree cap of 4 — let alone 32 — to
                    // bind. An option that is never placed under load cannot be measured.
                    ml: 3.0,
                    ef_construction: 150,
                    seed: Some(99),
                    build_strategy: BuildStrategy::Sequential,
                    ..Default::default()
                },
            )
        };

        let narrow = build_with_m(4);
        let wide = build_with_m(32);

        let (narrow_peak, narrow_count) = peak_degree_at(&narrow, 1);
        let (wide_peak, wide_count) = peak_degree_at(&wide, 1);

        // CONTROL 1: the fixture actually populates layer 1.
        assert!(
            narrow_count > 20 && wide_count > 20,
            "fixture put almost nothing on layer 1 ({narrow_count} / {wide_count} nodes) — a \
             degree cap cannot bind on an empty layer, so the assertions below would be vacuous"
        );

        // CONTROL 2: the cap actually BINDS. Some node must reach exactly m=4 neighbours,
        // otherwise "no node exceeds 4" is true for free and tests nothing.
        assert_eq!(
            narrow_peak, 4,
            "the m=4 cap never bound (peak layer-1 degree was {narrow_peak}) — with nothing
             pressing against the limit, the over-degree assertion below proves nothing"
        );

        // THE INVARIANT: m=4 means *no* node above layer 0 may exceed 4 edges. Breaking either
        // read site lets some node through, and a peak — unlike a mean — cannot absorb it.
        assert!(
            narrow_peak <= 4,
            "m = 4 but a layer-1 node holds {narrow_peak} neighbours — config.m is not capping \
             upper-layer degree"
        );

        // ...and m=32 must genuinely permit a wider graph, or `m` is being clamped somewhere.
        assert!(
            wide_peak > 4,
            "m = 32 produced a peak layer-1 degree of only {wide_peak}, no better than m=4 — \
             config.m is being ignored during insertion"
        );
    }

    /// `HNSWConfig::seed` must control the RNG used for level generation (and everything else
    /// stochastic in the builder) deterministically: the same seed on the same data must produce
    /// a bit-identical graph, and two different seeds must produce a different one. Both halves
    /// matter — see the two sabotage cases below, each caught by a different assertion.
    ///
    /// `BuildStrategy::Sequential` is used deliberately: `Parallel` uses rayon, and this test
    /// asks whether `config.seed` is read at all, not whether the parallel builder is internally
    /// deterministic under concurrent scheduling — a separate question, out of scope here.
    ///
    /// Sabotage this catches (two different bugs, one per assertion):
    ///  - "same seed -> same graph" fails if `config.seed` is ignored and the code always calls
    ///    `rand::random()` regardless of what the caller passed — two `Some(42)` builds would
    ///    then almost certainly diverge.
    ///  - "different seed -> different graph" fails if the code reads `config.seed` but maps it
    ///    through a broken or constant transform (e.g. hardcoding the RNG's internal seed to a
    ///    fixed value regardless of what `Some(N)` says) — `Some(1)` and `Some(2)` would then
    ///    collapse onto the same graph.
    #[test]
    fn seed_makes_builds_reproducible_and_distinguishable() {
        let mut rng = StdRng::seed_from_u64(2718);
        let centers: Vec<Vec<f32>> = (0..10)
            .map(|_| (0..12).map(|_| rng.random::<f32>() * 10.0).collect())
            .collect();
        let embeddings: Vec<Vec<f32>> = (0..400)
            .map(|i| {
                let c = &centers[i % 10];
                c.iter().map(|x| x + rng.random::<f32>() * 0.5).collect()
            })
            .collect();

        let build_with_seed = |seed: u64| -> HNSWIndex {
            HNSWIndex::build(
                embeddings.clone(),
                HNSWConfig {
                    seed: Some(seed),
                    build_strategy: BuildStrategy::Sequential,
                    ..Default::default()
                },
            )
        };

        #[allow(clippy::type_complexity)]
        let fingerprint =
            |index: &HNSWIndex| -> (Vec<u32>, Vec<Vec<Vec<u32>>>, Option<usize>, usize) {
                (
                    index.nodes.clone(),
                    index.connections.clone(),
                    index.entry_point,
                    index.max_layer,
                )
            };

        let a1 = build_with_seed(42);
        let a2 = build_with_seed(42);
        let b = build_with_seed(43);

        assert_eq!(
            fingerprint(&a1),
            fingerprint(&a2),
            "two builds with the same seed produced different graphs — seed is not being used \
             deterministically (or is being ignored in favor of a fresh random seed each time)"
        );
        assert_ne!(
            fingerprint(&a1),
            fingerprint(&b),
            "two builds with DIFFERENT seeds produced the identical graph — seed is being \
             ignored in favor of some fixed internal value"
        );
    }

    /// `rerank_candidates > 0` must measurably improve recall over `rerank_candidates: 0` on the
    /// *same* built index (same seed, same embeddings) — reranking is supposed to correct the
    /// coarse quantized ranking's mistakes, not merely "not panic". The existing
    /// `rabitq_zero_rerank_drops_full_precision_vectors_and_does_not_panic` only covers the zero
    /// case in isolation and cannot tell a working rerank from a disabled one.
    ///
    /// This is exactly the shape of the historical `PQHNSWConfig::rerank_candidates` bug: gated
    /// behind a second field that defaulted off, silently reranking nothing, with recall
    /// unchanged (0.840 vs 0.840) between "on" and "off". Nothing at the `HNSWIndex` level
    /// pairs the two the way this test does.
    ///
    /// `Storage::RaBitQ` is used because its 1-bit estimate is the coarsest ranking in the
    /// crate — and per this crate's own dimension-crossover findings, RaBitQ's coarseness cost
    /// is *worse*, not better, at low dimension, which is why `dim: 16` is used deliberately
    /// rather than a higher, easier dimension.
    ///
    /// Empirical margin, not a hand-proof: this relies on the coarse RaBitQ ranking genuinely
    /// misordering some held-out queries' top-10 on this fixture. It was reasoned through, not
    /// run. If it turns out both configs score at or near 100% recall on this exact fixture
    /// (i.e. the assertion fails because the values are equal or too close), that means the
    /// fixture is too easy, not that the test is wrong — harden it (denser clusters, e.g. more
    /// points per cluster) rather than weakening the assertion.
    ///
    /// Sabotage this catches: make the `rerank_candidates > 0` branch in `search_inner` a no-op
    /// (skip the rescore-and-resort) while still keeping the full-precision array allocated —
    /// recall for `rerank_candidates: 50` would then equal `rerank_candidates: 0`'s recall
    /// instead of exceeding it.
    #[test]
    fn rerank_candidates_nonzero_beats_zero_on_the_same_index() {
        let mut rng = StdRng::seed_from_u64(707);
        let dim = 16;
        let n_clusters = 10;
        let per_cluster = 60;
        let centers: Vec<Vec<f32>> = (0..n_clusters)
            .map(|_| (0..dim).map(|_| rng.random::<f32>() * 20.0).collect())
            .collect();
        let base: Vec<Vec<f32>> = (0..n_clusters * per_cluster)
            .map(|i| {
                let c = &centers[i % n_clusters];
                c.iter().map(|x| x + rng.random::<f32>() * 0.9).collect()
            })
            .collect();
        let n_queries = 50;
        let queries: Vec<Vec<f32>> = (0..n_queries)
            .map(|i| {
                let c = &centers[i % n_clusters];
                c.iter().map(|x| x + rng.random::<f32>() * 0.9).collect()
            })
            .collect();

        let recall_for = |rerank_candidates: usize| -> f32 {
            let config = HNSWConfig {
                metric: DistanceMetric::L2,
                m: 16,
                m0: 32,
                ef_construction: 150,
                ef_search: 150,
                storage: Storage::RaBitQ,
                rerank_candidates,
                seed: Some(21),
                ..Default::default()
            };
            let index = HNSWIndex::build_parallel(base.clone(), config);

            let k = 10;
            let mut total = 0.0f32;
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
                total += truth.intersection(&got).count() as f32 / k as f32;
            }
            total / queries.len() as f32
        };

        let coarse_only = recall_for(0);
        let reranked = recall_for(50);

        assert!(
            reranked > coarse_only,
            "rerank_candidates=50 must beat rerank_candidates=0 on the same index — got \
             {reranked:.3} vs {coarse_only:.3}. Equal recall means the rerank pool is being \
             silently ignored, exactly the shape of the historical PQHNSWConfig bug."
        );
    }
}
