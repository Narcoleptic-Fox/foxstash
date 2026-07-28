//! Index configuration: build strategy, metric, storage mode, and presets.
//!
//! Split out of `hnsw/mod.rs` as pure code motion.

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
    ///
    /// **Deprecated (0.7, removal in 0.8).** [`Storage::TurboRabit`] (Extended RaBitQ) dominates it
    /// at every matched bit budget and, unlike plain TurboQuant, holds on out-of-distribution data
    /// where TurboQuant collapses — recall 0.888 vs TurboRabit's 0.987 on the yandex-200 OOD set
    /// (see `docs/projects/foxstash/experiments.md` § Phase 4). Prefer `TurboRabit`, or `SQ8` for a
    /// robust default.
    #[deprecated(
        since = "0.7.0",
        note = "dominated by Storage::TurboRabit at every bit budget and collapses on OOD data; \
                scheduled for removal in 0.8 — use Storage::TurboRabit or Storage::SQ8"
    )]
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
    /// **Warren** — TurboRabit's 4-bit walk + an 8-bit residual rerank, and **no retained f32**.
    ///
    /// The combination the profile asked for. `dist/query` is dominated by the *walk* (~2,400
    /// visits/query against ~200 rerank candidates, roughly 6:1 in bytes), so the hot path must be
    /// the cheapest code we have — TurboRabit's 384 B at 768-d.
    ///
    /// Warren keeps TurboRabit's arena block **byte-identical**, so the walk is literally the same
    /// code, and replaces the retained f32 with an 8-bit residual taken against TurboRabit's own
    /// reconstruction. Measured on coco/nomic-768 against an f32 ceiling of 0.9930: residual rerank
    /// reaches **0.9905** at a third of TurboRabit's vector memory (1,152 B vs 3,456 B).
    ///
    /// Rerank stays in **rotated space** — `⟨q,x⟩ = ⟨q,c⟩ + ⟨R·q, r⟩` — because the inverse
    /// rotation is `O(dim²)` and unaffordable per candidate. `R·q` is `prepare_query`'s `rq` plus
    /// the constant `R·c`.
    ///
    /// Bulk-build only: incremental `add()` selects edges with exact f32 distances that this mode
    /// does not retain.
    Warren,
}

impl Storage {
    /// Whether reranking under this mode needs the retained full-precision vectors (`full`).
    ///
    /// True for every mode whose rerank rescores against f32 — `SQ8`, `RaBitQ`, `TurboQuant`,
    /// `TurboRabit`. False for:
    ///
    /// - [`Storage::F32`], which never reranks (the walk is already exact), and
    /// - [`Storage::Warren`], whose **8-bit residual code *is* the rerank representation** — it
    ///   retains no f32 at all and reranks perfectly well without it.
    ///
    /// This distinction exists because "`full` is empty" and "cannot rerank" were the same
    /// statement until the no-f32 modes, and code that conflated them rejected a legal configuration:
    /// `set_rerank_candidates` returned `FullPrecisionDropped` on every no-f32 (Warren) index, which killed
    /// the whole arm on its first query group in the 2026-07-19 sweep.
    #[inline]
    pub(super) fn rerank_needs_full(self) -> bool {
        !matches!(self, Storage::F32 | Storage::Warren)
    }
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
    #[allow(deprecated)] // internal handling of Storage::TurboQuant until its 0.8 removal
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
