# Foxstash Benchmark Results

Measured against **hnswlib** and **faiss** on real SIFT (128-d) and real GIST (960-d), at
**matched recall**, single-threaded, on an idle machine. Competitors are shown at *their own*
Pareto frontier — the best of their M=16 and M=32 curves at each recall — not at a fixed M that
happens to flatter us.

## Headline: foxstash wins at both dimensions — but with a *different* storage mode at each

| | **SIFT1M (128-d)** | **GIST1M (960-d)** |
|---|---|---|
| use this mode | **`Storage::SQ8`** | **`Storage::RaBitQ`** |
| vs hnswlib | **1.20x** @ 99.5% recall | **1.79x** @ 98.3% recall |
| vs faiss | **1.30x** @ 99.5% recall | **1.43x** @ 98.3% recall |
| the *other* mode | RaBitQ: **~12x slower** | SQ8: **worthless** (1.03x F32) |

**GIST1M (960-d), QPS at matched recall:**

| recall@10 | hnswlib | faiss | fox F32 | fox SQ8 | **fox RaBitQ** | vs hnswlib | vs faiss |
|---|---|---|---|---|---|---|---|
| 90.61% | 986 | 1,197 | 1,054 | 1,012 | **1,487** | **1.51x** | **1.24x** |
| 93.79% | 719 | 877 | 789 | 758 | **1,136** | **1.58x** | **1.30x** |
| 96.42% | 477 | 591 | 555 | 544 | **842** | **1.76x** | **1.43x** |
| 98.26% | 299 | 376 | 359 | 367 | **536** | **1.79x** | **1.43x** |

### The honest costs

**faiss builds faster than us at high dimension — but hnswlib does not.** GIST1M, M=32:

| | build |
|---|---|
| hnswlib | 1,433 s |
| **foxstash** | **1,141 s** (1.26x faster than hnswlib) |
| faiss | **408 s** (2.8x faster than us) |

This file used to say our build time was "a real regression and it is not yet explained." That was
wrong, and wrong in a specific way worth naming: **I read only the faiss column of my own table.**
Against the reference HNSW implementation we are *faster*. faiss is the outlier.

The cause is now measured, and it is `keep_pruned_connections: true` (the default). It backfills
every node to the maximum degree `m0`, deliberately restoring the exact neighbours the diversity
heuristic just rejected:

| | build (200k, 960-d) | distance computations / insert | avg degree |
|---|---|---|---|
| `keep_pruned: true` | **197 s** | **61,559** | **64.0** — saturated |
| `keep_pruned: false` | 42.6 s | 3,167 | 10.5 |

faiss runs at degree **~25**. Our heuristic alone prunes to **10.5** (too sparse) and the backfill
saturates to **64** (too dense) — we sit at both extremes and neither is the sweet spot. Turning
the backfill off is not free: at matched recall the dense graph is *slightly faster to query*
(930 vs ~832 QPS at 97.9% recall). The dense graph buys recall-per-`ef`; it just pays far too much
for it at build time.

**Full precision is unremarkable.** `Storage::F32` at 960-d is 1.10–1.20x hnswlib but only
0.90–0.96x faiss — it sits *between* them. At 128-d it is 0.88x hnswlib. Every win foxstash has
comes from the quantized traversal, not from the graph or the kernel.

**Memory.** At 128-d, SQ8 + rerank costs 1,076 MB against their ~776 MB (`rerank_candidates: 0`
drops it to 564 MB / 0.73x, at a ~98.9% recall ceiling). At 960-d, SQ8 is the *largest* index of
the three (5,236 MB vs F32's 4,276) — the rerank pool still keeps the f32 vectors and at 960-d
those dominate.

## The graph degree was wrong, the knob was disconnected, and our own harness hid it

Three findings, one root.

**1. `config.m` and `config.m0` did nothing in the default builder.** The parallel builder — which
is the default — passed the hardcoded constants `M0_MAX = 64` / `M_MAX = 32` everywhere the config
values belonged. It built a degree-64 graph no matter what you asked for and truncated afterwards.
The tell was a *flat line where a curve was predicted*: build cost sat at ~61,500 distance
computations per insert at m0 = 24, 32, 48 **and** 64 alike. It was doing the m0=64 build every
time.

**2. The optimal degree depends on `n`, and a 200k measurement DOES NOT generalize to 1M.**

First I measured the degree sweep at 200k, found `m0=32` beat `m0=64` for `F32` (67 s vs 167 s to
build, and faster at matched recall), wrote "our default degree was wrong" into this file, and
committed it.

Then I ran it at **1M — the scale we actually ship at — and it reversed.** `M=32/m0=64` wins for
*every* storage mode:

| storage | M=16/m0=32 | **M=32/m0=64** |
|---|---|---|
| F32 @ ~97.7% recall | 356 QPS | **429 QPS** (1.20x) |
| SQ8 @ ~97.7% recall | 372 QPS | **439 QPS** (1.18x) |
| RaBitQ @ ~97% recall | 675 QPS | **862 QPS** (1.28x) |
| build (F32) | **452 s** | 1,138 s |

The sparse graph builds **2.5x faster** and gives up **~20% QPS** at matched recall. That is a real
trade — `M=16` is the "fast build" option — but it is **not** domination, and `M=32` is the right
default at 1M. A smaller corpus needs less connectivity to stay navigable; a 1M graph does not.

**This is the same error as reading a 128-d SIFT result and generalizing it to 960-d — committed in
the same session that documented that error.** Measure at the scale you ship at. A cheaper regime
is a different regime, not a preview of the real one.

(The degree/storage *interaction* at 200k was real — RaBitQ, whose distances cost 137 ns against
F32's 259 ns, tolerated the dense graph far better. But at 1M every mode wants the dense graph, so
the interaction does not change the recommendation. Sweep `M` per storage mode anyway: it is cheap
insurance and it is what we already demand of the competition.)

**3. Our benchmark harness was rigged — against ourselves.** `benchmarks/python/competitors.py`
carries this comment:

> *"Sweep M on EVERY library, not just ours. Tuning our own M while pinning theirs at a default
> would manufacture a win — which is precisely the class of error that produced this repo's earlier
> false headlines."*

The comment is correct. The code below it swept hnswlib's and faiss's `M ∈ {16, 32}` and **pinned
ours at 32.** We showed the competition at its best-of-M frontier and ourselves at a single degree
nobody chose.

The sweep now runs, and it **vindicates the published numbers**: `M=32` was in fact the best choice
at 1M for every storage mode. We were right by luck rather than by measurement — which is not the
same as being right, and is exactly why the sweep has to exist.

This is a different failure from the other twelve. They were all *checks that could not fail*. This
one was a **principle that was stated but not enforced** — written down, correct, sitting in the
file, with the code directly beneath it doing the opposite. A rule you do not test is a comment.

`storage_pareto` now sweeps our own `M`, per storage mode.

### What it means for the build-time gap

The dense graph is what makes our query numbers good, and it is what makes our build slow. At 1M
every storage mode wants `m0=64`, so **the build cost is real, not an artifact**: ~1,141 s against
faiss's 408 s, while we query 1.4–1.8x faster than faiss at matched recall.

**That is a trade, and it should be stated as one rather than filed as a mystery.** If you need a
fast build more than you need QPS, `M=16/m0=32` builds in 452 s (2.5x faster) for about 20% fewer
queries per second at matched recall. Both points are now on the published frontier instead of one
being hidden.

> ### ⚠️ Pick your storage mode by dimension. The two quantizers swap places.
>
> Every quantized traversal trades ALU work for memory traffic, and the two codes sit on opposite
> sides of that trade:
>
> * **SQ8** must widen `u8` → `i32` → `f32` before it can compute: ~**3x the ALU uops per
>   dimension** of plain f32. What it buys — skipped DRAM round-trips — is roughly **fixed** per
>   node visit. Fixed benefit, cost linear in `dim`: **wins small, dies big.**
> * **RaBitQ** compares sign bits against a once-per-query rotated vector: **cheaper per dimension
>   than f32**, no widening at all. What it pays is a *coarser* estimate, which makes the graph
>   walk take more hops — and that penalty **shrinks sharply with `dim`**, because the code is
>   1 bit *per dimension*: a higher-`dim` vector gets a proportionally **longer code**. RaBitQ
>   gets cheaper *and* more accurate as `dim` rises. **Loses small, wins big.**
>
> It is a **scissors, not a single crossing line** — both blades close at once. One corpus,
> prefix-truncated so `dim` is the only variable (`--example dim_crossover`, GIST, n=200k, ef=100):
>
> | dim | 64 | 128 | 192 | 256 | 384 | 512 | 768 | 960 |
> |---|---|---|---|---|---|---|---|---|
> | RaBitQ recall@10 | 63.7% | 78.0% | 84.9% | 88.2% | 91.4% | 94.2% | 96.9% | 96.7% |
> | gap vs F32 | −35.8 | −20.6 | −13.4 | −10.0 | −6.4 | −3.8 | **−0.9** | **−0.9** |
> | RaBitQ speed (F32 ns/dist ÷ RaBitQ) | 1.28x | 1.55x | 1.58x | 1.74x | 1.88x | 2.01x | 1.98x | 1.89x |
>
> **Rule of thumb: `SQ8` at 384-d and below, `RaBitQ` at 768-d and above, measure in between.**
> Real embeddings are 384-d (MiniLM) to 1536-d (OpenAI) — **nobody runs RAG on 128-d vectors**, so
> the SIFT numbers below are the *best* case for SQ8, not a typical one.
> Do not quote the SIFT numbers below at your dimension without checking. See
> **"The right quantizer depends on the dimension"** and **"384-d: where I got it wrong twice"**.

## SIFT1M — QPS at matched recall, single-threaded

| recall@10 | foxstash SQ8 | vs hnswlib | vs faiss | foxstash F32 | vs hnswlib |
|---|---|---|---|---|---|
| 92.99% | 13,107 | **1.11x** | **1.19x** | 10,045 | 0.84x |
| 98.24% | 7,183 | **1.15x** | **1.27x** | 5,478 | 0.87x |
| 99.51% | 4,254 | **1.20x** | **1.30x** | 3,290 | 0.88x |
| 99.85% | 2,549 | **1.33x** | **1.34x** | 1,925 | 0.90x |

| index size @ 1M | |
|---|---|
| foxstash SQ8 + rerank | 1,076 MB |
| foxstash F32 | 948 MB |
| hnswlib / faiss | ~776 MB |
| **foxstash SQ8, codes only** (no rerank; recall ceiling ~98.9%) | **564 MB** |

### Why it works, and why it nearly didn't

Search is **memory-latency bound**: a distance computation costs 77–98 ns, about one DRAM
round-trip. Foxstash already computed distances *faster* than faiss (84 ns vs 98 ns) and still
lost, because it issued more of them and each one waited on memory. Making the kernel faster
was never going to help. Moving fewer bytes was.

`Storage::SQ8` puts 8-bit codes in the hot node block — 400 bytes instead of 784 — and keeps
the f32 vectors in a **cold** array read only when rescoring the final candidate pool
(`O(rerank_candidates)` per query, against `O(nodes visited)` for the walk). The graph itself
is still built with **exact f32 distances**, so it is bit-identical to the F32 index:

| | F32 | SQ8 |
|---|---|---|
| recall@10 (ef=100) | 99.45% | 99.51% |
| distance computations / query | 3,492 | 3,491 |
| **ns per distance** | **87.1** | **67.3** |

Same graph, same work, fewer bytes. That is the whole result.

**The near-miss:** the first version of the SQ8 kernel was a scalar loop, and it made SQ8
*slower* — 132 ns per distance against F32's 87 — because the `u8 -> f32` widening cost more
than the bandwidth saved. Measured naively, that reads as "compressed traversal does not work".
It does; the widening just has to be one instruction. AVX2's `cvtepu8_epi32` is the difference
between 132 ns and 67 ns. A portable `pulp` kernel cannot express it, which is why this is the
one hand-written `std::arch` path in the library — and why it carries a test asserting the AVX2
and scalar paths agree.

## History: before the SQ8 storage mode

Before the node-arena interleave (commit 0617c6c) foxstash was ~20% behind hnswlib and only won
on SIFT10K, the one dataset small enough to live in L3 cache. See *Why the layout mattered*
below — that fix is what made the SQ8 win reachable.

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


## Graph degree at 1M: `M=32/m0=64` is right, and the crossover is RECALL

Swept on **both** corpora now, because the last time this was measured at 200k it produced a
conclusion that was published and then retracted. SIFT1M, 128-d, read at **matched recall**:

| recall@10 | F32 `M=16` | F32 `M=32` | SQ8 `M=16` | SQ8 `M=32` |
|---|---|---|---|---|
| ~95% | **8,866** | ~8,384 | — | — |
| ~98.3% | 5,263 | **5,479** | 6,687 | **7,128** |
| ~99.5% | 3,048 | **3,279** | 3,968 | **4,229** |
| ~99.9% | 1,418 | **1,920** (+35%) | 1,900 | **2,520** (+33%) |

Build: `M=16` 81 s / `M=32` 165–170 s. Index: `M=16` 794 MB / `M=32` 948 MB (F32).

**The crossover is recall, not merely corpus size.** Below ~96% recall the sparser graph wins —
it visits fewer nodes and that is all that matters when you are stopping early anyway. Above ~98%
the denser graph wins, and **the gap widens as recall rises**, reaching +35% at 99.9%. A sparse
graph has to explore much harder to find the last percent, because the neighbours that would have
taken it there were pruned at build time.

RAG runs at high recall. `M=32/m0=64` stays the default, now confirmed on a second dataset at a
second dimensionality.

> The retracted claim ("our default degree was wrong, m0=32 dominates") was a **200k artifact**,
> and it is now falsified at 1M on GIST *and* SIFT. Two datasets, same answer. That is what the
> original claim needed and did not have.

### SQ8 wins at 128-d; RaBitQ cannot even finish the race

At matched recall SQ8 delivers **1.27x** F32's QPS (6,687 vs 5,263 at 98.4%), off 71.9 vs 91.4
ns/dist — the latency saving is fixed while the dequant cost is small at 128-d, exactly as the
dimensional rule predicts.

RaBitQ, on the same corpus, **ceilings at 93.17% recall — at ef=500, with reranking enabled.** It
cannot reach the operating points the other two run at. This is not slowness, it is blindness, and
it is the identical failure that killed `PQHNSWIndex`: when the graph is traversed on a metric too
coarse to rank, the true neighbours never enter the candidate pool, and **reranking cannot rescue
what the walk never retrieved.** At 128-d, 1 bit per dimension is that metric.

The same code gets 97.98% on 960-d GIST. Nothing about it changed. The dimension did.


## Build time and memory

| | foxstash | hnswlib | faiss |
|---|---|---|---|
| build, SIFT1M (all cores) | **167 s** | 342 s | 78 s |
| index, SIFT1M | 947 MB | ~776 MB | ~776 MB |

Foxstash builds **2.1x faster than hnswlib**. Its index is ~21% larger: 512 MB vectors +
378 MB of links and block headers, where the nested upper-layer `Vec`s cost a 24-byte header
per node and the arena block carries padding to keep the vector 16-byte aligned.

## Why `SQ8HNSWIndex` was deleted

Two independent SQ8 implementations existed: `Storage::SQ8` on the main `HNSWIndex` (codes
inline in the interleaved node arena, graph built with exact f32 distances, parallel build)
and a standalone `SQ8HNSWIndex` (its own config, fat `Vec<HashSet<usize>>` adjacency,
sequential build). SIFT100K, M=32, ef_c=200, rerank pool 100, serialized on an idle machine:

| ef | index | recall@10 | QPS | build |
|----|-------|-----------|-----|-------|
| 50 | `HNSWIndex` + `Storage::SQ8` | **99.43%** | **12,128** | **7.2 s** |
| 50 | `SQ8HNSWIndex` | 93.74% | 3,919 | 68.8 s |
| 100 | `HNSWIndex` + `Storage::SQ8` | **99.88%** | **7,238** | **7.2 s** |
| 100 | `SQ8HNSWIndex` | 95.80% | 2,453 | 67.1 s |
| 200 | `HNSWIndex` + `Storage::SQ8` | **99.98%** | **4,440** | **7.2 s** |
| 200 | `SQ8HNSWIndex` | 97.75% | 1,373 | 69.1 s |

Dominated on every axis, and not narrowly: `SQ8HNSWIndex` at ef=200 (97.75%) never reaches
what the arena does at ef=50 (99.43%) with 2.7x the throughput. It built 9.6x slower because
it built sequentially.

It was also a **metric footgun**: `QuantizedHNSWConfig` has no `metric` field and is hardcoded
L2, while `HNSWConfig::metric` defaults to Cosine. Swapping index types to save memory
silently changed the question being asked — recall would collapse and nothing would error.

The bar for deletion was that the survivor win on recall *and* throughput *and* build time.
It did.

## Quantized indexes — the metric was broken

`ScalarQuantizer::distance_quantized` computed L2 directly on the raw 0-255 codes. But
`fit()` gives **every dimension its own scale** — that is the entire point of fitting — so a
near-constant dimension's full code swing carried the same weight as a high-variance
dimension's. The comment said "(scaled)". Nothing was scaled.

This was not confined to `search_symmetric`'s output. `search_layer` calls
`distance_quantized` during **insertion**, so the SQ8 graph was *built* under a distorted
metric. On a fixture with heterogeneous per-dimension variance, `search_symmetric` scored
**7% recall — indistinguishable from chance** — against `search()`'s 73% on the identical
graph and candidate pool. After the fix, SQ8 on real SIFT100K reaches **99.33% recall@10**
(rerank pool 50), against the **71.4%** this file previously reported.

That 71.4% was never a quantization ceiling, and never a missing rerank. It was a bug, and it
cost 28 points of recall while the rustdoc advertised "100.0%".

**Any SQ8 index previously built via `fit()` is invalid, not stale** — its edges were selected
under the wrong metric. It must be rebuilt from source vectors; re-scoring cannot repair it.

Nothing caught this because `search_symmetric` had **zero tests** — and, the subtle part,
**self-retrieval cannot detect it.** Querying with an indexed vector re-quantizes to that
vector's own code, so a quantized-vs-quantized metric returns distance 0 against itself
whether or not it is scale-correct. A "symmetric" quantized index that passes a self-retrieval
test has proven nothing. Only held-out queries scored against real brute-force ground truth
expose it.

## Compressed traversal: thesis proven — but the first attempt hid it

This section is kept because the wrong conclusion was very nearly drawn, twice.

The reason to quantize is bandwidth. Search is memory-latency bound — 77–98 ns per distance
computation, about one DRAM round-trip — so moving a quarter of the bytes per node visit ought
to buy throughput. The first measurement said it did the opposite. SIFT100K, matched ~99.4%
recall:

| | QPS | build | memory |
|---|---|---|---|
| full-precision HNSW | **8,867** | 7 s | 95 MB |
| `SQ8HNSWIndex` + exact rerank | 1,541 | 135 s | 74 MB |

**5.8x slower while touching a quarter of the vector bytes.** Read naively, that kills the
idea. It was not a bandwidth result; it was a data-structure result. `SQ8HNSWIndex` was a
wholly separate implementation that never touched the interleaved node arena:

```rust
struct SQ8Node {
    id: String,                       // heap allocation, per node
    content: String,                  // heap allocation, per node
    quantized: ScalarQuantizedVector, // heap allocation, per node
    full_precision: Option<Vec<f32>>, // heap allocation, per node
    connections: Vec<HashSet<usize>>, // a HashSet. Per layer. Per node.
}
```

Every node visit pointer-chased through a fat struct into a heap-allocated code vector and
iterated neighbours through a **hash set** of 8-byte ids. The compression win was real and
entirely swamped by the container carrying it. **The vehicle could not hold the experiment.**

Rebuilding SQ8 as a *storage mode* of the main index — sharing the arena, the packed u32
adjacency and the parallel builder — is what actually tested the thesis. It holds: `Storage::SQ8`
is **1.20x hnswlib at 99.5% recall on SIFT1M**, and the mechanism is exactly the predicted one
(node block 784 → 400 bytes, ns/dist 87.1 → 67.3, distance count unchanged at ~3,491).

**Then it nearly died a second time.** The first arena kernel was a scalar u8→f32 widening
loop: **132 ns/dist, slower than the 87 ns f32 baseline it replaced.** The widening cost more
than the bandwidth saved. That reads, again, as "compressed traversal doesn't work". It was the
kernel. An AVX2 path (`_mm256_cvtepu8_epi32` → `cvtepi32_ps` → `fmadd`) took it to 67 ns.

Two separate implementation defects, each of which produced a *plausible, coherent* negative
result about the underlying idea. Neither was about the idea.

## What foxstash is genuinely better at

- **Throughput at matched recall.** 1.11–1.33x hnswlib and 1.19–1.34x faiss on SIFT1M with
  `Storage::SQ8`, widening as recall rises (see the top of this file).
- **Recall per `ef`.** At every scale it needs a lower `ef` than hnswlib or faiss to reach a
  given recall — the Algorithm-4 diversity heuristic builds a measurably better graph.
- **Build throughput**, 2.1x hnswlib.

## Known issues

1. **Memory: 1,076 MB against hnswlib's 776 MB** at the fastest setting. Rerank needs the
   full-precision vectors, so the fastest configuration is also the largest. Setting
   `rerank_candidates: 0` drops the f32 array entirely — 564 MB, **0.73x hnswlib** — but recall
   then ceilings around 98.9%. Speed crown or memory crown; not both, yet.
2. **Metric inconsistency (a correctness footgun), partially closed.** `HNSWIndex` defaults to
   cosine, but `RaBitQHNSWIndex` and `PQHNSWIndex` have **no `metric` field at all** — they are
   hardcoded L2. Swapping index type to save memory silently changes your distance function and
   nothing warns you. `SQ8HNSWIndex` was the worst offender and is now deleted (above); the fix
   for the remaining two is the same — make them storage modes of the one index, which honours
   `HNSWConfig::metric`, rather than parallel index types.
3. **PQ has never been measured on real data**, and `PQHNSWIndex` carries the same fat-node
   pathology (`Vec<HashSet<usize>>` adjacency, sequential build) that made `SQ8HNSWIndex` 5.8x
   slower than the arena. Its distance math is sound — verified, no unit-mismatch analog of the
   SQ8 scale bug — but the vehicle is the one already proven not to hold the experiment.

## SQ8 at 960-d: the thesis has a dimension limit

The whole case for `Storage::SQ8` is that HNSW search is **memory-latency bound**, so moving
fewer bytes per node visit buys throughput. On SIFT (128-d) that is true and worth 1.20x against
hnswlib. **On GIST1M (960-d) it is worth nothing.** Same machine, same builder, matched recall,
single-threaded:

| | node block | ns/dist @ ef=500 | QPS @ ~99.4% recall | index size |
|---|---|---|---|---|
| `Storage::F32` | 4,112 B | 277.1 | 209 | 4,276 MB |
| `Storage::SQ8` + rerank | **1,232 B** | **268.8** | **215** | **5,236 MB** |

The block shrank **3.3x** and the cost per distance fell **3%**. SQ8 is also *larger* here — the
rerank pool still needs the f32 vectors, and at 960-d those dominate.

`dist/query` is the control that proves this is a cost-per-distance story and not a graph story:
F32 issues 17,245 distance computations and SQ8 issues 17,266 — identical, as they must be, since
**both graphs are built with exact f32 distances**. Same graph, same work. Only the price of each
distance changed, and it didn't.

### Why it flips — the two sides of SQ8 scale differently

| | benefit of SQ8 | cost of SQ8 |
|---|---|---|
| what it is | DRAM round-trips skipped per node visit | ~3x extra ALU uops to widen `u8` → `i32` → `f32` |
| scaling in `dim` | roughly **fixed** — one block fetch per visit | **linear in `dim`** |

At 128-d a distance is ~16 AVX2 ops — nothing — so the entire cost is one DRAM round-trip, and
skipping ~6 cache lines takes 87 → 67 ns/dist. At 960-d the vector is a **65-cache-line
sequential stream**, which the hardware prefetcher hides almost completely, while the dequant tax
has grown 7.5x (≈120 → ≈480 uops). A fixed benefit against a linearly growing cost: they cross.

**This is not "quantization doesn't work at high dim."** It is "*this* quantizer's dequant cost
outgrows the latency it saves." A code whose kernel is *cheaper* per dimension than f32 — rather
than 3x more expensive — would not have this problem. That is exactly what RaBitQ is, and at
960-d it wins by 1.4–1.6x. See the next section.

### What this costs us in honesty

The headline "1.20x hnswlib" is a **128-dimensional** claim. Nobody runs RAG on 128-d vectors:
MiniLM is 384-d, OpenAI's embeddings are 1536-d. We have **not** benchmarked hnswlib or faiss on
GIST, so we cannot yet say where foxstash stands against them at 960-d. **That measurement is the
top open item.**

I predicted the opposite. Before running this I wrote in the README that SQ8 "should widen its
lead at higher dim, since there are more bytes per hop to save" — reasoning from the block
arithmetic alone and forgetting that the dequant cost scales with those same bytes. A clean,
plausible, wrong story that would have shipped as fact. The experiment cost 40 minutes.

## The right quantizer depends on the dimension

This is the headline finding of the GIST run, and it inverts the SIFT conclusion.

**GIST1M (960-d), QPS at matched recall, single-threaded:**

| recall@10 | F32 | SQ8 | **RaBitQ** | RaBitQ vs F32 | RaBitQ vs SQ8 |
|---|---|---|---|---|---|
| 93.79% | 789 | 758 | **1,136** | **1.44x** | **1.50x** |
| 96.42% | 555 | 544 | **842** | **1.52x** | **1.55x** |
| 97.65% | 446 | 445 | **638** | **1.43x** | **1.43x** |
| 99.02% | 252 | 270 | **410** | **1.63x** | **1.52x** |

| | ns/dist | index size |
|---|---|---|
| F32 | 258–277 | 4,276 MB |
| SQ8 | 250–269 (block 3.3x smaller — bought **nothing**) | 5,236 MB |
| **RaBitQ** | **138–157** (**1.9x cheaper** than either) | 4,404 MB |

**The two storage modes trade places at opposite ends of the dimension range:**

| | SIFT1M (128-d) | GIST1M (960-d) |
|---|---|---|
| `Storage::SQ8` | **1.20x hnswlib** — the win | worthless (1.03x F32) |
| `Storage::RaBitQ` | **~12x slower** than SQ8 | **1.4–1.6x faster** than SQ8 or F32 |

### Why, mechanically

Every quantized traversal trades **ALU work** for **memory traffic**. The two codes sit on
opposite sides of that trade:

- **SQ8** must widen `u8` → `i32` → `f32` before it can do arithmetic: roughly **3x more ALU
  uops per dimension than plain f32**. Its saving is skipped DRAM round-trips, which is roughly
  *fixed* per node visit. Fixed benefit, cost linear in `dim` — so it wins at low `dim` and dies
  at high `dim`.
- **RaBitQ** compares sign bits against a pre-rotated query: **cheaper per dimension than f32**,
  no widening at all. Its cost is a *coarser* distance estimate, which makes the graph walk take
  more hops. That penalty **shrinks sharply as `dim` rises** — the code is 1 bit *per dimension*,
  so a higher-`dim` vector gets a proportionally **longer code** and a better estimate, exactly
  where its f32 counterpart is getting more expensive to read. Cheaper *and* more accurate with
  `dim`: it loses at low `dim` and wins at high `dim`.

So it is a **scissors**: as `dim` grows, RaBitQ's cost advantage widens *and* its accuracy gap
closes. Both blades. Measured on one corpus, prefix-truncated so `dim` is the only variable
(`--example dim_crossover`, GIST, n=200k, ef=100 — same base vectors, same queries, same graph
parameters, exact ground truth recomputed at every `dim`):

| dim | 64 | 128 | 192 | 256 | 384 | 512 | 768 | 960 |
|---|---|---|---|---|---|---|---|---|
| RaBitQ recall@10 | 63.7% | 78.0% | 84.9% | 88.2% | 91.4% | 94.2% | 96.9% | 96.7% |
| F32 recall@10 | 99.5% | 98.6% | 98.3% | 98.2% | 97.8% | 98.0% | 97.8% | 97.6% |
| **gap** | **−35.8** | −20.6 | −13.4 | −10.0 | −6.4 | −3.8 | **−0.9** | **−0.9** |
| ns/dist F32 | 53.0 | 72.4 | 85.7 | 103.7 | 137.4 | 166.3 | 221.3 | 259.4 |
| ns/dist RaBitQ | 41.3 | 46.8 | 54.4 | 59.7 | 73.0 | 82.8 | 111.8 | 137.6 |
| **RaBitQ speed** | 1.28x | 1.55x | 1.58x | 1.74x | 1.88x | 2.01x | 1.98x | 1.89x |

The accuracy penalty collapses by a factor of ~40 across the range while the speed advantage
roughly doubles.

### The claim this replaces, and how it survived

This section used to say RaBitQ's hop penalty was "**roughly independent of `dim`**" — and then
cited, as *supporting* evidence, the very numbers that refute it: **10x more distance computations
at 128-d, 2% more at 960-d.** A ratio that moves from 10x to 1.02x across the dimension range is
the *opposite* of dim-independent. The disproof was printed three lines below the claim, formatted
as a citation for it.

The conclusion (use RaBitQ at high `dim`) happened to be right. The mechanism under it was half
wrong, and a right answer resting on a wrong mechanism is a coin that has not landed yet.

**Rule: read the evidence you are about to cite as if you were trying to refute the sentence above
it.** Ask of every number: *if this claim were false, would this number look different?* If the
figure you are citing varies along the axis you just called invariant, the claim is already dead.
This is the prose form of the same failure as "the test that cannot fail" — a check with no power
to discriminate.

## 384-d: where I got it wrong twice

384-d is the dimension that matters — MiniLM is the most common RAG embedding in existence. So it
is worth recording that I got the 384-d answer **wrong twice, in opposite directions**, before
measuring it properly.

**Wrong once (SIFT-only):** on 128-d SIFT, RaBitQ loses ~12x, and every number said delete it. The
senior engineer who implemented it recommended deleting it. It survived only because someone
noticed SIFT is 128-d and *nobody runs RAG on 128-d vectors*.

**Wrong twice (fixed-`ef`):** the `dim_crossover` sweep runs at a fixed `ef=100`, where RaBitQ at
384-d posts **3,257 QPS against SQ8's 1,898** — a 1.7x win. I wrote down the caveat that fixed-`ef`
flatters a coarse quantizer, and then reasoned as if I hadn't, and concluded 384-d was "approximately
the crossover — MiniLM can use RaBitQ."

Then I measured it at **matched recall**, which is the only comparison that bills RaBitQ for the
extra hops its coarse estimate costs (`--example dim_pareto gist1m 384`, n=200k, exact GT recomputed
at 384-d):

| recall@10 | F32 | SQ8 | RaBitQ |
|---|---|---|---|
| 93.9% | 3,116 | **3,371** | ~2,674 |
| 98.1% | 1,826 | **1,935** | ~1,398 |
| 99.35% | 1,068 | **1,137** | ~750 |

**Under L2, RaBitQ loses at 384-d — to SQ8, and to plain F32.** Its kernel really is ~1.9x cheaper
(73 ns/dist vs 137), but it needs **2.6x more distance computations** to reach the same recall, and
the discount does not cover the bill. Its 3,257 QPS at fixed `ef` was the throughput of *giving up
early* — 91.4% recall against F32's 97.8%.

### Wrong a third time: the whole analysis assumed L2, and RAG is cosine

Everything above — every crossover number this project has ever published — was measured with
`DistanceMetric::L2`. `dim_crossover.rs`, `dim_pareto.rs` and `storage_pareto.rs` all hardcoded it.
But **MiniLM, OpenAI embeddings, GloVe, and five of the six standard ANN benchmark datasets are
compared by cosine.** The storage-mode advice was extrapolated across that gap and nobody noticed —
the same species of error as reading a 128-d SIFT result and generalizing it to 960-d. Right
experiment, wrong regime.

Measured (`--example dim_pareto gist1m <dim> 200000 cosine`, n=200k, GT recomputed under the metric
being tested):

| recall@10 | metric | F32 | SQ8 | RaBitQ |
|---|---|---|---|---|
| ~97.7% | `L2` | 1,826 | **1,935** | ~1,398 — *loses 0.72x* |
| ~97.7% | `Cosine` | 1,636 | 1,897 | **~1,903** — *dead even* |

**Cosine moves the crossover down.** At 960-d cosine RaBitQ beats SQ8 by **1.2–1.3x** (≈1,400 vs
1,077 QPS at 97.7%), consistent with L2; but at 384-d it goes from *losing* under L2 to *tying*
under cosine.

**The mechanism was already in our own code.** Under cosine, `push_node` encodes a **unit-normalized**
copy of the vector for RaBitQ (`rabitq_cosine_input`). Normalization discards magnitude — exactly the
information a 1-bit sign code cannot carry, and exactly the information cosine does not want. Under
L2, magnitude *is* signal, so throwing it away costs real recall. **RaBitQ is structurally suited to
cosine**, and we had been grading it on the metric it is worst at.

At 384-d cosine RaBitQ is also the *smaller* index (405 MB vs SQ8's 471 MB — `SQ8 + rerank` is the
largest of the three, keeping f32 vectors *and* codes). On a tie, take RaBitQ.

**The rule of thumb needs two axes:**

| | `L2` | `Cosine` (what RAG uses) |
|---|---|---|
| ≤ 256-d | `SQ8` | `SQ8` |
| 384-d (MiniLM) | `SQ8` | tie — take `RaBitQ` for the smaller index |
| ≥ 768-d | `RaBitQ` | `RaBitQ` |

> A mode that is fast because it stopped finding things is not fast. **Fixed-`ef` comparisons cannot
> see this.** They are useful for isolating a *mechanism* (`ns/dist`, `dist/query` — what the code
> does per unit work) and worthless for choosing a *configuration* (which mode to ship). This repo
> has now published a wrong conclusion from a fixed-`ef` reading once. That is once more than the
> caveat being written down was worth.

## `PQHNSWIndex` was deleted: dominated, and judged on an invalid number

PQ was the last standalone index type and the only 192x-compression option in the crate. It is
gone. The measurement is worth keeping because the figure that had been defending it was produced
with the feature that makes it usable **switched off**.

The docs said PQ got **~55% recall@10**. That was measured with `PQHNSWConfig::rerank_candidates`
at its default of **0** — the exact-rescoring stage disabled. Measured properly on GIST (960-d,
100k, L2 — PQ's *best* case, since PQ was L2-only and RaBitQ is structurally better under cosine):

| | MB | recall@10 | QPS |
|---|---|---|---|
| `PQHNSWIndex`, rerank 0 | **18** | **23.07%** | 1,293 |
| `PQHNSWIndex`, rerank 50 | 402 | 61.80% | 775 |
| `PQHNSWIndex`, rerank 100 | 402 | **62.27%** ← ceiling | 790 |
| `PQHNSWIndex`, rerank 400 | 402 | 60.97% ← **worse** | 446 |
| `Storage::RaBitQ` + rerank | 440 | **97.97%** | **1,970** |
| `Storage::SQ8`, rerank 0 | **139** | **98.40%** | 760 |

**The ~62% is a ceiling, not a knob.** The graph is *traversed* on PQ codes, so the candidate pool
handed to the rescoring stage does not **contain** the true neighbours — and you cannot rerank your
way to items you never retrieved. Widening the pool from 100 to 400 made recall *fall*.

**The compression evaporates exactly when it becomes useful.** Reaching even 62% requires
`rerank_candidates > 0`, which retains the f32 vectors: 402 MB. At that size `Storage::RaBitQ`
costs 440 MB and delivers 98%. PQ's only unique point on the frontier was 18 MB at 23% recall,
which is not a retrieval index — it is a random number generator with a graph attached.

And the true no-rerank number is **23%**, not 55%. The 55% came from a small synthetic clustered
fixture; it matched neither of the real operating points.

### The pattern, for the fourth time

*A bad number produced by a disabled or broken feature makes the feature look inherently bad — and
then nobody re-measures it.* The bad number becomes the reason not to look.

| the number | what produced it | the truth |
|---|---|---|
| SQ8 at **71.4%** recall | the metric bug (quantized traversal ignored `config.metric`) | **99.33%** |
| PQ rerank changes nothing (0.840 vs 0.840) | `rerank_candidates` gated behind `store_original`, which defaulted false — a silent no-op | rerank works; PQ still ceilings at 62% |
| RaBitQ is **~12x slower**, delete it | measured on 128-d SIFT, in a library whose users run 384–1536-d | **1.79x hnswlib at 960-d** |
| PQ gets **~55%** recall | measured with `rerank_candidates: 0` — accuracy stage off | 23% off / 62% ceiling on |

Three of those four nearly removed a good feature or preserved a bad one. **Check what a number was
measured *with* before you let it decide anything.**

The [`ProductQuantizer`](../crates/core/src/vector/product_quantize.rs) primitive is kept. It is a
perfectly good quantizer. It is just not a viable way to traverse a graph.

### The near-miss

On SIFT, `Storage::RaBitQ` loses ~12x and every measurement said delete it. The senior engineer
who implemented it recommended deleting it. The only thing that saved it was noticing that
**SIFT is 128-d and nobody runs RAG on 128-d vectors** — and that the node-block arithmetic makes
every quantization conclusion a function of the dimension.

Deleting it would have been defensible, well-measured, and **wrong**. Reproduce with:

```
benchmarks/fetch-data.sh --with-gist
cargo run --release -p foxstash-benches --example storage_pareto gist1m
```

## `Storage::RaBitQ` at 128-d: the negative result, and where the bottleneck went

1-bit-per-dimension traversal was the obvious next step after SQ8's win. It does not pay off,
and *why* it doesn't is the most useful thing in this file.

SIFT1M, M=32/m0=64, single-threaded, matched recall:

| | recall@10 | QPS | dist/query | ns/dist |
|---|---|---|---|---|
| `Storage::SQ8` | 92.99% | **13,107** | ~1,250 | 66.6 |
| `Storage::RaBitQ` (rerank 400) | 92.67% | **1,100** | 13,085 | 61.9 |

RaBitQ buys a **7% cheaper distance** and pays **10x more distances** to reach the same recall.
~12x slower end to end. The coarser metric misleads the graph walk, and the walk's extra hops
cost far more than the narrower codes save.

*(A first pass with `rerank_candidates: 100` looked even worse — recall ceilinged at 74% and then
**declined** with more ef, 74.09% → 73.01% → 72.03%. Non-monotonic recall is not a quality
ceiling, it is a rerank-pool artifact: a coarse metric lets distractors crowd into a fixed top-100
pool and evict true neighbours before the exact rescore ever sees them. Deepening the pool to 400
restored monotonicity. Worth knowing: **if recall falls as ef rises, suspect the rerank pool, not
the quantizer.**)*

### The bottleneck moved from the vector to the adjacency list

The node block is `header(m0) + vector`. At m0=64:

| | header (2 + 64 ids + 1, padded) | vector | block |
|---|---|---|---|
| F32 | 272 B | 512 B | **784 B** |
| SQ8 | 272 B | 128 B | **400 B** |
| RaBitQ | 272 B | 24 B | **296 B** |

SQ8 already harvested the large saving (512 → 128 B). RaBitQ shaves a further 104 B off a 400 B
block — 26% — and pays for it with a drastically coarser metric. **The vector is now only ~32% of
the block; the other 272 bytes are neighbour ids.** Compressing the vector further has hit
diminishing returns by construction.

Testing that directly — SQ8 at m0=32, a **272 B** block, *smaller than RaBitQ's* and with a far
better metric — at matched recall on SIFT1M:

| recall@10 | SQ8 m0=64 | SQ8 m0=32 | |
|---|---|---|---|
| ~98.85% | ~5,700 | **6,123** | +7% |
| ~99.67% | ~3,300 | **3,670** | +11% |
| ~99.90% | ~1,600 | **1,781** | +11% |

Memory also falls, 1,076 → 948 MB. **At 128 dimensions, adjacency is the next lever, not the
vector.**

**Two limits on that claim, both load-bearing:**

**1. It is not yet a competitive claim.** Trap 1 below cuts both ways: lowering M helps hnswlib and
faiss too, roughly equally. So m0=32 is a better operating point for foxstash's own Pareto curve,
but the multiple against the competition cannot be restated until they are re-run at matched M.
The defaults have deliberately **not** been changed on the strength of this.

**2. It is a statement about SIFT's 128 dimensions, and it very likely inverts at the
dimensionality foxstash is actually used at.** The block is `header(m0) + vector`, so which half
dominates is a pure function of `dim`. At m0=64:

| dim | F32 block | SQ8 block | RaBitQ block | vector's share of the SQ8 block |
|---|---|---|---|---|
| 128 (SIFT) | 784 B | 400 B | 296 B | 32% |
| 384 (MiniLM) | 1,808 B | 656 B | 320 B | 59% |
| 768 | 3,344 B | 1,040 B | 368 B | 74% |
| 1536 (OpenAI) | 6,416 B | 1,808 B | **472 B** | 85% |

At 1536-d, RaBitQ shrinks the node block **3.8x below SQ8**, and the 272 B of adjacency is noise.
The whole reason RaBitQ fails on SIFT — that it fights for 104 B out of 400 — simply does not
apply there. Nobody runs a RAG pipeline on 128-d vectors; MiniLM is 384-d and OpenAI's embeddings
are 1536-d.

**So `Storage::RaBitQ` is NOT deleted, and the negative result above must not be quoted as
general.** It is a 128-d result. The deciding experiment is GIST1M (960-d) or a real 768/1536-d
embedding corpus, and until that is run the honest position is: *RaBitQ loses badly at low
dimension, is untested where it should win, and SQ8 is the default.* Deleting it on SIFT evidence
alone would be the same dataset-generalization error catalogued below — the one that has already
caught this project four times.

## The bugs that survived because nothing could fail

Every one of these was public, documented, and green across the whole test suite. They are
listed together because they share a single cause: the tests exercised the code, but none of
them were *able* to fail.

| bug | what it did | why the tests missed it |
|---|---|---|
| `BuildStrategy::Sequential` | **panicked on every input**, an entire release | no test or doctest ever selected it |
| `keep_pruned_connections` | silently ignored in the parallel builder — degree saturated at 64.0/64 vs faiss's 25.4 | the config flag was named in a comment and never read; no test asserted graph density |
| `ScalarQuantizer::distance_quantized` | ignored per-dimension scale → **7% recall, indistinguishable from chance**; the graph was *built* under the distorted metric | `search_symmetric` had zero tests — and self-retrieval **cannot** catch it (see below) |
| `Storage::SQ8` + `add()` | **panicked on the first document** — codebook never fitted outside `build*()`. Reachable from `foxstash-db`, so every `insert` on an SQ8 collection crashed | every SQ8 test went through `build_parallel` |
| quantized save/load | an SQ8 index could be **saved but never loaded** — reload rebuilds via `new()` + `add()`, i.e. untrained | the roundtrip test used the default F32 storage, and asserted only `len()` and "returns 2 results" — which a reload that reinterpreted 8-bit codes as f32 would also satisfy |
| `HNSWConfigWrapper` | **`seed` silently dropped on deserialize** → a reloaded index assigned nodes to *different layers*, so save/load quietly destroyed reproducibility for anyone who had asked for it | nothing round-tripped a seeded config and compared the graphs |
| `distance_to_node` | **quantized traversal ignored `config.metric`** — hardcoded L2 under SQ8/RaBitQ, while `score_from_distance` still branched on the metric and **Cosine is the default**. So `HNSWConfig { storage: SQ8, ..Default::default() }` searched under L2 and scored it `1.0 - squared_L2` (unbounded, negative) | **no test anywhere built a quantized index with the default metric** — every caller passed `metric: L2` by hand, avoiding the bug by coincidence |
| `PQHNSWConfig::rerank_candidates` | **a silent no-op alone** — `search()` gated reranking on `store_original && rerank_candidates > 0`, and `store_original` defaulted to `false`. `PQHNSWConfig { rerank_candidates: 100, ..Default::default() }` reranked **nothing** | the un-reranked path returns k results too, just worse ones — so "no panic" and "returns k" both pass. Only recall-vs-exact catches it (measured: 0.840 with and without the pool) |
| `benchmarks/data/sift1m/` | held a **10,000**-vector base | nothing validated the corpus against its label |
| `BinaryHNSWIndex` | 1.2% recall while the rustdoc advertised 90% | benchmarked only on synthetic vectors, where every ANN scores ~60% |

**The test that could not fail.** The SQ8 metric bug's first regression test was self-retrieval,
and it scored **100% on the completely broken metric**. Querying with an indexed vector
re-quantizes to that vector's own code, so a quantized-vs-quantized distance returns 0 against
itself no matter how wrong the scale factors are. A quantized index that passes a self-retrieval
test has proven *nothing*. Only held-out queries scored against real brute-force ground truth
expose it.

The pattern to take from this: a test that exercises a code path is not the same as a test that
can fail on it. Every bug above had code that *ran* under test. What the tests lacked was the
ability to **discriminate** — the fixture didn't have the property that would make a wrong
answer look wrong:

- **self-retrieval** cannot catch a broken *quantized* metric (the query re-quantizes to its own code)
- **uniform-norm vectors** cannot catch a **cosine-vs-L2 swap** — on equal norms the two rank *identically*
- **an F32 fixture** cannot catch anything about quantized storage
- **counting results** cannot catch wrong results — "returns 10 items" is satisfied by 10 wrong items
- **uniform-random vectors** cannot catch a broken ANN — everything scores ~60%

So, two rules, both cheap:

1. **Prove the fixture discriminates before trusting the assertion.** For a metric test, first
   assert that `Cosine` and `L2` actually *disagree* on your fixture. If they agree, the fixture
   is inert and your real assertion is vacuous no matter how it reads.
2. **Sabotage the code and watch the test fail.** This is definitive and takes a minute. It
   caught a would-be-vacuous RaBitQ recall test (sabotaged kernel → recall fell 100% → 7%),
   confirmed the SQ8 save/load test (fix disabled → `NotTrained`), and confirmed the PQ rerank
   test (bug reinstated → recall identical, 0.840 vs 0.840, exactly the silent no-op).
   **If you cannot make a test fail on purpose, it will never fail by accident either.**

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
