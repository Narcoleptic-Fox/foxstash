# Foxstash Benchmark Results

Measured against **hnswlib** and **faiss** on real SIFT, at **matched recall**, at three scales.

**Headline: foxstash is at parity with faiss and ~10% behind hnswlib at 1M.** It builds 2.1x
faster than hnswlib and reaches any given recall at a lower `ef` than either competitor.

Before the node-arena interleave (commit 0617c6c) it was ~20% behind hnswlib and only won on
SIFT10K, the one dataset small enough to live in L3 cache. See *Why the layout mattered* below.

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

### SIFT10K — 9.5 MB index, fits in 16 MB L3

| recall@10 | foxstash | hnswlib | vs hnswlib | faiss | vs faiss |
|---|---|---|---|---|---|
| 88.80% | 16,501 | 15,900 | **1.04x** | 19,900 | 0.83x |
| 96.87% | 9,213 | 8,754 | **1.05x** | 11,031 | 0.84x |
| 99.54% | 5,546 | 5,340 | **1.04x** | 7,081 | 0.78x |

### SIFT100K — 95 MB index, exceeds L3

| recall@10 | foxstash | hnswlib | vs hnswlib | faiss | vs faiss |
|---|---|---|---|---|---|
| 91.04% | 24,151 | 30,530 | 0.79x | 29,930 | 0.81x |
| 96.59% | 16,257 | 19,266 | 0.84x | 18,502 | 0.88x |
| 99.42% | 8,751 | 10,513 | 0.83x | 9,637 | 0.91x |
| 99.90% | 5,321 | 6,019 | 0.88x | 5,269 | **1.01x** |

### SIFT1M — 947 MB index

| recall@10 | foxstash | hnswlib | vs hnswlib | faiss | vs faiss |
|---|---|---|---|---|---|
| 85.01% | 14,436 | 18,241 | 0.79x | 17,274 | 0.83x |
| 92.96% | 10,044 | 11,787 | 0.85x | 10,961 | 0.91x |
| 98.25% | 5,455 | 6,352 | 0.88x | 5,749 | 0.97x |
| 99.49% | 3,267 | 3,636 | 0.91x | 3,384 | 0.98x |

## Why the layout mattered

The old `HNSWIndex` was Struct-of-Arrays: a node's vector in `embeddings`, its neighbours in
`connections_l0`, its norm in `norms` — three separate allocations. Visiting a node therefore
issued **three independent random DRAM reads**. hnswlib and faiss interleave per node —
`[link_count | links | vector | label]` in one contiguous block — so a node visit is **one**.

The stated justification for SoA was "better cache locality". That is true of a linear scan.
Graph traversal never scans linearly: it jumps to an arbitrary node, reads its links, then jumps
to each neighbour's vector. Under *that* access pattern SoA is strictly worse.

Below L3 the difference is invisible — which is exactly why the old SIFT10K-only benchmark
showed foxstash winning. Above L3 it was worth ~20%. Commit 0617c6c interleaved the arena
(`[count | norm | m0+1 links | pad | vector]`, 784 bytes against hnswlib's ~780):

| | before | after | gain |
|---|---|---|---|
| SIFT1M @ 98.2% | 5,043 | 5,455 | **+8.2%** |
| SIFT1M @ 99.5% | 2,978 | 3,267 | **+9.7%** |
| SIFT100K @ 99.9% | 4,953 | 5,321 | **+7.4%** |
| SIFT10K @ 96.9% | 9,366 | 9,213 | unchanged (cache-resident) |

The gain grows with **both scale and recall** — the signature of a cost paid in random DRAM
reads, since a higher `ef` means more node visits. Recall is identical at every `ef`, which is
the safety oracle for a graph-structure change.

Two things measured and **not** kept, which matter as much as what was:
- `get_unchecked` on the hot accessors: **+1.3%**, inside run-to-run noise. Not worth `unsafe`
  in the hottest function in the library — and it disproves "bounds checks are the gap".
- Prefetching the whole 784-byte block (13 cache lines): *slower* than prefetching the header
  plus 3 lines of the vector. More prefetch instructions than the saved misses were worth.

So the remaining ~10% against hnswlib is **not** layout and **not** bounds checking. It is
not yet explained, and it will be profiled rather than guessed at.

## Build time and memory

| | foxstash | hnswlib | faiss |
|---|---|---|---|
| build, SIFT1M (all cores) | **167 s** | 342 s | 78 s |
| index, SIFT1M | 947 MB | ~776 MB | ~776 MB |

Foxstash builds **2.1x faster than hnswlib**. Its index is ~21% larger: 512 MB vectors +
378 MB of links and block headers, where the nested upper-layer `Vec`s cost a 24-byte header
per node and the arena block carries padding to keep the vector 16-byte aligned.

## What foxstash is genuinely better at

- **Recall per `ef`.** At every scale it needs a lower `ef` than hnswlib or faiss to reach a
  given recall — the Algorithm-4 diversity heuristic builds a measurably better graph. This is
  a real asset and the reason the matched-recall gap (0.78x) is *smaller* than the raw
  fixed-`ef` gap.
- **Build throughput**, 2.1x hnswlib.

## Known issues

1. **Still ~10% behind hnswlib at 1M**, cause not yet established. Layout and bounds checks are
   both ruled out (above). Next step is to measure distance computations per query at matched
   recall — that separates *doing more work* (a worse graph, or a search that stops too late)
   from *doing the same work more slowly* (a worse inner loop, or worse latency hiding). Those
   have completely different fixes and QPS alone cannot tell them apart.
2. **Quantized indexes are unfinished.** `SQ8HNSWIndex` has **no rerank path at all**, which is
   why it sits at 71.4% recall; `RaBitQHNSWIndex` has one but defaults to `ef_search: 50` where
   `HNSWConfig` uses 100. Compressed traversal with exact rerank is the standard way to beat a
   memory-bandwidth wall — it moves *fewer bytes* per node visit rather than moving the same
   bytes better — and it is half-built here.
3. **Metric inconsistency (a correctness footgun).** `HNSWIndex` defaults to cosine;
   `SQ8HNSWIndex` and `RaBitQHNSWIndex` are L2-only. Swapping index type to save memory
   silently changes your distance function, and nothing warns you.
4. **Index memory is ~21% above hnswlib.**
5. **`SQ8HNSWIndex::search_symmetric` has zero test coverage** — public, documented, a distinct
   code path. Its sibling `PQHNSWIndex::search_symmetric` is tested. This is the same shape as
   `BuildStrategy::Sequential`, which was public, documented, recommended, and panicked on every
   input for an entire release while 247 tests passed.
6. **PQ has never been measured on real data**, yet `hnsw_pq.rs` advertises "good search
   quality" at up to 192x compression.

> ### Historical note
>
> Before 2026-07-12 this file reported foxstash as "1.03–1.10x faster than hnswlib" without
> qualification. That was measured only on SIFT10K, and it did not hold at 100K or 1M, where
> foxstash lost to both hnswlib and faiss. faiss was not benchmarked against at all.
>
> Before that, the file — and the README — benchmarked on **synthetic** vectors and claimed
> foxstash "beats gold standards". The README table reported **hnswlib at 39.5% recall**; on
> real SIFT hnswlib scores ~99%. Synthetic vectors have no cluster structure, so every ANN
> collapses to ~60% on them regardless of quality. That table flattered foxstash and concealed
> a real bug (`BinaryQuantizer` at 1.2% recall) for an entire release. It also compared
> foxstash's all-cores batch search against single-threaded competitors, and then blamed their
> `ef=64` for the difference.
>
> The pattern is consistent, and worth stating plainly: every previous headline was produced by
> measuring one convenient configuration and never asking what would falsify it. The traps above
> are not trivia — they are the specific ways this project has already fooled itself.
