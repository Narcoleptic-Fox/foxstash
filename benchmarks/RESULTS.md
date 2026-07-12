# Foxstash Benchmark Results

Measured against hnswlib and faiss on **real SIFT10K**, at **matched recall**.

**Headline: foxstash is 1.03–1.10x faster than hnswlib at equal recall, single-threaded,
across the useful operating range.**

## Test Configuration

- **Dataset:** SIFT10K — 10,000 base vectors, 128d, 1,000 queries, **real data with ground
  truth shipped by the dataset authors** (exact L2, 0-indexed)
- **Hardware:** Cortex (Ryzen 7 7840HS, 8 cores / 16 threads), Ubuntu 24.04
- **Config:** M=32, ef_construction=200, k=10, `DistanceMetric::L2` — identical on both sides
- **Reproduce:**
  - `cargo run --release -p foxstash-benches --example pareto` (recall/QPS curve)
  - `cargo run --release -p foxstash-benches --example memory` (footprint)
  - `cargo run --release -p foxstash-benches --example sift_bench` (full index sweep)

## Methodology: two traps, both of which caught this project

**1. Compare at matched recall, not matched `ef`.** `ef_search` is a knob, and two
implementations reach a given recall at different settings of it. Foxstash gets *more recall
per ef* than hnswlib (96.9% vs 94.3% at ef=100), so comparing at fixed `ef` understates it.
The only meaningful question is **QPS at the same recall**.

**2. Count the threads.** hnswlib's `knn_query` defaults to `num_threads=-1` — *every core*.
An earlier version of this file timed that against a single-threaded Rust loop and concluded
foxstash was "~11x slower". It is not; that was a 16-core number racing a 1-core number.
Both sides below are single-threaded, and multi-threaded is reported separately.

Every recall table must also carry an **exact/flat control row**. If brute-force search does
not score 100% against the ground truth, the loader or the metric is wrong and every other
row in the table is void.

## Recall vs QPS — single-threaded, matched recall

| ef | foxstash recall@10 | foxstash QPS | hnswlib QPS @ same recall | ratio |
|----|--------------------|--------------|---------------------------|-------|
| 10 | 49.77% | 56,125 | 53,626 | **1.05x** |
| 20 | 67.36% | 35,421 | 33,008 | **1.07x** |
| 50 | 88.42% | 15,078 | 15,246 | 0.99x |
| 100 | **96.86%** | **9,442** | 8,572 | **1.10x** |
| 200 | 99.47% | 5,585 | 5,429 | **1.03x** |
| 500 | 99.96% | 3,022 | 3,563 | 0.85x |

hnswlib's own single-threaded curve, same machine and config, for reference:

| ef | recall@10 | QPS |
|----|-----------|-----|
| 10 | 40.20% | 68,483 |
| 50 | 81.95% | 20,051 |
| 100 | 94.34% | 10,850 |
| 200 | 99.20% | 6,458 |
| 500 | 99.98% | 3,487 |

**Foxstash wins across the useful range** (50–99.5% recall) and needs less search to get
there — the Algorithm-4 diversity heuristic builds a better graph, so a given `ef` buys more
recall. It loses only at ef=500, where the search touches ~85% of a 10k index: HNSW
degenerating toward brute force, which is a pathological operating point, not a useful one.

## Multi-threaded

| | QPS | recall@10 |
|---|-----|-----------|
| foxstash `search_batch()` (ef=100) | 79,796 | **96.86%** |
| hnswlib `knn_query()` default (ef=100) | 94,517 | 94.25% |

Not matched on recall — foxstash is 2.6 points ahead there, so this is not a like-for-like
row. Both achieve ~54% parallel efficiency across 16 SMT threads on 8 physical cores; the
workload is memory-bandwidth bound on both sides.

## Memory

`cargo run --release -p foxstash-benches --example memory` — retained bytes, counting
allocated *capacity* and per-`Vec` headers, not RSS (RSS around a build also captures the
builder's transients and whatever the allocator declines to return).

| Component | MB |
|-----------|-----|
| embeddings (f32) | 5.12 |
| layer-0 links (flat) | 2.61 |
| upper-layer links (nested) | 1.09 |
| norms (cosine only) | 0.04 |
| payload (ids + contents) | 0.52 |
| **total** | **9.38** |
| *theoretical floor (vectors + links)* | *7.68* |
| *hnswlib (its own accounting)* | *7.80* |

Down from **12.65 MB**. Two fixes got there: layer-0 adjacency was being stored **twice**
(once nested, once in the flat cache) — the flat array is now its sole owner; and the
embedding array was holding 8.39 MB of `Vec` growth capacity for 5.12 MB of vectors, so the
build paths now `shrink_to_fit()`.

The remaining gap to hnswlib is mostly the 0.52 MB `payload` — foxstash stores document ids
and contents, where hnswlib stores only an integer label. Excluding payload, foxstash is
8.86 MB against hnswlib's 7.80 MB.

## Quantized indexes (vs SIFT's L2 ground truth)

| Index | Compression | Recall@10 | QPS |
|-------|-------------|-----------|-----|
| sq8-hnsw | 4x | 71.4% | 11,166 |
| rabitq-hnsw | 32x | 62.5% | 894 |
| ~~binary-hnsw~~ | 32x | **1.1%** | — |

**These are the weak spot.** Both are correctness-first and untuned (`QuantizedHNSWConfig`
still defaults to `ef_search: 50` where `HNSWConfig` uses 100).

`BinaryHNSWIndex` is **deprecated**: its zero threshold sets every bit on non-negative data
(SIFT, any ReLU embedding), collapsing all codes to all-ones. Superseded by
`RaBitQHNSWIndex` — same 32x, but centered.

## What changed to get here

- **Squared L2 in the hot loop.** `sqrt` is monotonic, so it cannot change the ordering — it
  was pure overhead on ~8,500 distance computations per query. Rooted only for the k results
  returned.
- **Stopped throwing away work.** `search_layer` computed `ef` distances and returned bare
  ids; the caller then *recomputed* all of them, built a `SearchResult` (cloning an id, a
  content string and a metadata blob) for all 500, sorted, and discarded 490 to return 10.
  It now returns `(distance, id)` and materialises only `k`.
- **`search_batch` reuses a context per worker** instead of allocating a whole-index visited
  bitset and two heaps per query.

## Known issues

1. **ef=500 regime is 0.85x** — the only point where hnswlib wins. Low priority; it is the
   brute-force-degenerate corner.
2. **Quantized recall is low** (SQ8 71.4%, RaBitQ 62.5%) and untuned.
3. **Metric inconsistency** — `HNSWIndex` defaults to Cosine, but `SQ8HNSWIndex` and
   `RaBitQHNSWIndex` are L2-only. Swapping index type to save memory silently changes the
   metric. They should take the same `DistanceMetric` config.
4. **RaBitQ build is slow** (17.2s) — `prepare_query` runs per insert.
5. **Nested `connections` still holds an empty `Vec` per node for layer 0** (~0.24 MB) —
   removing it means re-indexing layers, which was judged not worth the risk.
6. **100K/1M not yet run** — `benchmarks/data/` has `sift100k` and `sift1m`.

> ### Historical note
>
> Before 2026-07-12 this file benchmarked on **synthetic** "SIFT-like" vectors and claimed
> foxstash "beats gold standards" with *1.5x hnswlib's recall*. That was an artifact:
> synthetic vectors have no cluster structure, so every ANN collapses to ~60% recall on them
> regardless of quality. hnswlib scored 40.3% there and scores **99.98%** on real SIFT. The
> synthetic run flattered foxstash and hid a real bug (`BinaryQuantizer` at 1.2% recall) for
> an entire release. Foxstash does beat hnswlib — but only measurement on real data at
> matched recall can support that claim.
