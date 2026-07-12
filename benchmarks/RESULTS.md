# Foxstash Benchmark Results

Measured against **hnswlib** and **faiss** on real SIFT, at **matched recall**, at three scales.

**Headline: at realistic scale, foxstash is currently ~20% slower than hnswlib and ~13%
slower than faiss.** It wins only on SIFT10K — the one dataset small enough to fit in L3
cache. It does build 2.1x faster than hnswlib, and it reaches a given recall at a lower `ef`
than either competitor.

Reproduce the whole table: `benchmarks/run-scoreboard.sh`

## Test configuration

- **Data:** real SIFT, exact-L2 ground truth. Fetch with `benchmarks/fetch-data.sh`.
  | dataset | base | queries | difficulty `d(100th)/d(10th)` |
  |---|---|---|---|
  | sift10k | 10,000 | 1,000 | 1.047 |
  | sift100k | 100,000 | 10,000 | 1.145 |
  | sift1m | 1,000,000 | 10,000 | 1.123 |
- **Hardware:** Ryzen 7 7840HS, 8 cores / 16 threads, 16 MB L3, Ubuntu 24.04
- **Config:** M=32, ef_construction=200, k=10, L2 — identical on all three libraries
- **Search is single-threaded on every library.** Build uses all cores on every library.

## Four traps, every one of which has caught this project

**1. Compare at matched recall, not at matched `ef`.** `ef` is a knob and different
implementations reach a given recall at different settings of it. Foxstash gets *more recall
per ef* than hnswlib, so a fixed-`ef` table flatters it.

**2. Count the threads.** hnswlib's `knn_query` defaults to `num_threads=-1` — every core. An
earlier version of this file timed that against a single-threaded Rust loop and reported
foxstash as "~11x slower". That was a 16-core number racing a 1-core number.

**3. Never run two benchmarks at once.** Running the Python harness while a Rust 1M build held
all 16 cores *halved* hnswlib's apparent QPS (5,478 vs 10,850 at ef=100). `run-scoreboard.sh`
serializes every run behind an idle gate.

**4. Verify the corpus is the corpus.** `benchmarks/data/sift1m/` contained a **10,000**-vector
base. Benchmarking "SIFT1M" against it would have yielded an entirely plausible number off an
index 100x smaller than its label. `Dataset::load` now validates shape against a manifest and
the bench asserts an exact brute-force control row before printing anything.

**And do not compare recall across datasets.** On SIFT10K the 100th neighbour is only 4.7%
further from the query than the 10th, so the true top-10 hides inside a shell of ~90
near-equidistant vectors; on SIFT100K it is 13.5% further. Every index scores far better on
SIFT100K because the task is easier, not because the index improved. The `difficulty` column
above exists to stop that comparison being made by accident.

## QPS at matched recall

Competitor QPS is linearly interpolated along its own recall/QPS curve to foxstash's recall.
Ratio > 1.00x means foxstash serves more queries per second at the same recall.

### SIFT10K — 9.4 MB index, fits in 16 MB L3

| recall@10 | foxstash | hnswlib | vs hnswlib | faiss | vs faiss |
|---|---|---|---|---|---|
| 88.25% | 16,881 | 15,663 | **1.08x** | 19,585 | 0.86x |
| 96.88% | 9,366 | 8,745 | **1.07x** | 11,017 | 0.85x |
| 99.51% | 5,608 | 5,233 | **1.07x** | 7,018 | 0.80x |

### SIFT100K — 94 MB index, exceeds L3

| recall@10 | foxstash | hnswlib | vs hnswlib | faiss | vs faiss |
|---|---|---|---|---|---|
| 91.10% | 23,729 | 30,479 | 0.78x | 29,880 | 0.79x |
| 96.61% | 15,609 | 19,216 | 0.81x | 18,453 | 0.85x |
| 99.41% | 8,238 | 10,579 | 0.78x | 9,696 | 0.85x |
| 99.88% | 4,953 | 6,440 | 0.77x | 5,668 | 0.87x |

### SIFT1M — 940 MB index

| recall@10 | foxstash | hnswlib | vs hnswlib | faiss | vs faiss |
|---|---|---|---|---|---|
| 85.14% | 13,590 | 18,241 | 0.75x | 17,274 | 0.79x |
| 93.03% | 9,351 | 11,787 | 0.79x | 10,961 | 0.85x |
| 98.18% | 5,043 | 6,352 | 0.79x | 5,749 | 0.88x |
| 99.48% | 2,978 | 3,636 | 0.82x | 3,384 | 0.88x |

## Why foxstash wins at 10K and loses past it

The crossover sits exactly at the L3 boundary, and the cause is the memory layout.

`HNSWIndex` is Struct-of-Arrays: a node's vector lives in `embeddings`, its neighbours live in
`connections_l0`, and the two are separate allocations. Visiting a node therefore issues **two
independent random DRAM reads**. hnswlib and faiss interleave per node — `[link_count | links |
vector | label]` in one contiguous block — so a node visit is **one read**, and pulling a
neighbour's vector also warms the links you need if you expand it.

Below L3 that costs nothing: everything is already in cache and foxstash's better graph wins on
recall-per-ef. Above L3 the extra random read is the whole ballgame, and it is worth about the
20% we measure. The in-code comment claiming SoA gives "better cache locality" is true of linear
scans and false of graph traversal, which never scans linearly.

## Build time and memory

| | foxstash | hnswlib | faiss |
|---|---|---|---|
| build, SIFT1M (all cores) | **164 s** | 342 s | 78 s |
| index, SIFT1M | 940 MB | ~776 MB | ~776 MB |

Foxstash builds **2.1x faster than hnswlib**. Its index is ~21% larger: 512 MB vectors +
371 MB links, where the link array carries an `m0 + 1` stride and the nested upper-layer
`Vec`s cost a 24-byte header per node.

## What foxstash is genuinely better at

- **Recall per `ef`.** At every scale it needs a lower `ef` than hnswlib or faiss to reach a
  given recall — the Algorithm-4 diversity heuristic builds a measurably better graph. This is
  a real asset and the reason the matched-recall gap (0.78x) is *smaller* than the raw
  fixed-`ef` gap.
- **Build throughput**, 2.1x hnswlib.

## Known issues

1. **Slower than both competitors above L3** (0.78x hnswlib, 0.85x faiss). Root cause is the
   SoA layout above; the fix is node-interleaved storage plus prefetch.
2. **Quantized indexes are unfinished.** `SQ8HNSWIndex` has **no rerank path at all**, which is
   why it sits at 71.4% recall; `RaBitQHNSWIndex` has one but defaults to `ef_search: 50` where
   `HNSWConfig` uses 100. Compressed traversal with exact rerank is the standard way to beat a
   memory-bandwidth wall and it is half-built here.
3. **Metric inconsistency.** `HNSWIndex` defaults to cosine; `SQ8HNSWIndex` and
   `RaBitQHNSWIndex` are L2-only. Swapping index type to save memory silently changes the
   metric.
4. **Index memory is ~21% above hnswlib.**
5. **False claims in rustdoc.** `index/mod.rs` advertises SQ8 at "100.0%" recall (measured:
   71.4%) and full-precision HNSW at "100%" (measured: ~97%).

> ### Historical note
>
> Before 2026-07-12 this file reported foxstash as "1.03–1.10x faster than hnswlib" without
> qualification. That was measured only on SIFT10K, and it does not hold at 100K or 1M, where
> foxstash loses to both hnswlib and faiss. faiss was not benchmarked against at all.
>
> Before that, the file benchmarked on **synthetic** vectors and claimed foxstash "beats gold
> standards" with 1.5x hnswlib's recall — an artifact of synthetic data having no cluster
> structure, which collapses every ANN to ~60% and hid a real bug (`BinaryQuantizer` at 1.2%
> recall) for a full release.
>
> The pattern is consistent: every previous headline was produced by measuring one convenient
> configuration and not asking what would falsify it.
