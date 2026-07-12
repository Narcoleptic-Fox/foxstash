# Foxstash Benchmark Results

Comparative benchmarks against industry-standard ANN libraries, on **real SIFT10K**.

## Test Configuration

- **Dataset:** SIFT10K — 10,000 base vectors, 128d, 1,000 queries, **real data with
  ground truth shipped by the dataset authors** (exact L2, 0-indexed)
- **Hardware:** Cortex dev server (Ryzen 7 7840HS, Ubuntu 24.04)
- **Date:** 2026-07-12
- **Reproduce:**
  - Foxstash: `cargo run --release -p foxstash-benches --example sift_bench`
  - Competitors: `cd benchmarks/python && ./venv/bin/python run_benchmarks.py --dataset sift10k`

> ### This file previously reported numbers that were not real
>
> The prior version of this document benchmarked on **"Synthetic SIFT-like"** vectors and
> concluded Foxstash "beats gold standards", with *1.5x hnswlib's recall*. That was an
> artifact. Synthetic vectors have no cluster structure, so **every** ANN collapses to
> ~60% recall on them regardless of quality: hnswlib scored 40.3% there, and **99.98%**
> here on real data. The synthetic run flattered Foxstash and concealed a real bug
> (`BinaryQuantizer` degenerating to 1.2% recall on non-negative data) for an entire
> release. Every number below is measured on real data against shipped ground truth.

## Metric matters — read this before comparing rows

SIFT's ground truth is exact **L2**. `HNSWIndex` now supports both metrics via
`HNSWConfig::metric` (`DistanceMetric::L2` / `::Cosine`, defaulting to Cosine for backward
compatibility). **Only the L2 rows are comparable to faiss/hnswlib.**

This is worth stating plainly because it caused a scare: scored against SIFT's L2 key, the
*cosine* index reads 55%. That number measures the metric gap, not the graph — the same
index scores 97.7% against a cosine ground truth. SIFT magnitudes vary 1.4x, so cosine and
L2 genuinely disagree about who the nearest neighbours are. Before `DistanceMetric` existed
there was no L2 index at all, which is why Foxstash appears nowhere in the historical
`report_sift10k.md`.

## Directly comparable — L2 metric, vs SIFT's L2 ground truth

| Library | Algorithm | Recall@10 | QPS | Build |
|---------|-----------|-----------|-----|-------|
| faiss | ivf | 100.00% | **78,749** | 0.03s |
| faiss | flat | 100.00% | 38,004 | 0.00s |
| faiss | hnsw | 99.99% | 34,005 | 0.18s |
| hnswlib | hnsw (ef=500) | 99.98% | 32,715 | 0.19s |
| annoy | annoy | 100.00% | 599 | 0.51s |
| **foxstash** | **hnsw-l2 (ef=500)** | **99.90%** | 2,864 | 1.11s |
| **foxstash** | **hnsw-l2 (ef=200)** | **99.60%** | 5,464 | 1.10s |
| **foxstash** | **hnsw-l2 (ef=100)** | **96.90%** | 8,732 | 1.11s |
| **foxstash** | **flat** (control) | **100.00%** | 1,310 | 1.53s |
| **foxstash** | **sq8-hnsw** (4x) | 71.40% | 11,166 | 2.67s |
| **foxstash** | **rabitq-hnsw** (32x) | 62.50% | 894 | 17.23s |

Parallel build is nondeterministic, so the HNSW rows move ~0.1–0.5 points between runs
(ef=500 has been observed at both 99.9% and 100.0%). Treat these as ±0.5, not exact.

The `flat` control reads 100%, which is what validates the loader and the metric. **Any
recall table without a passing control row is void** — if exact search doesn't score 100%
against the ground truth, nothing else in the table means anything.

### What this says

**Recall is competitive. Throughput is not.** At `ef=500` — hnswlib's own configuration —
Foxstash's L2 HNSW reaches **99.9%** recall@10 against hnswlib's 99.98%. The recall gap is
gone; it was tuning, not quality.

The real gap is speed. At matched recall (~100%), Foxstash serves **2,864 QPS vs hnswlib's
32,715** — roughly **11x slower**. Build is 1.11s vs 0.19s (~6x slower). That is the honest
headline, and it is the thing to optimise.

**The quantized indexes are the weak spot.** SQ8 gives up 28 points of recall at 4x
compression; RaBitQ 37 points at 32x. Both are correctness-first implementations that have
not been tuned (`QuantizedHNSWConfig` still defaults to `ef_search: 50`).

## Foxstash HNSW (cosine) — scored against cosine ground truth

Kept for reference. Not comparable to the L2 table above — a different question.

| Build path | Recall@10 | Recall@100 | Build | QPS |
|------------|-----------|------------|-------|-----|
| sequential | 97.7% | 93.3% | 14.00s | 8,992 |
| incremental (`add()` loop) | 97.7% | 93.3% | 14.56s | 8,406 |
| parallel | 97.6% | 92.8% | **1.88s** | 8,605 |

**Parallel build is recall-safe as of `fix(hnsw): apply diversity heuristic in parallel
build path`.** Before it, the parallel builder skipped the Algorithm-4 diversity heuristic
— it "built" in 0.18s because it was doing less work, and paid 1.7 points of recall for it
(95.4%). With the heuristic it reaches 97.6%, level with sequential, and still builds
**7.4x faster**. Use parallel.

## Where Foxstash actually stands

Being blunt, because the last version of this file wasn't:

- **Recall:** 99.9% @ ef=500 on L2 — matches hnswlib. Genuinely good.
- **Search throughput:** ~11x slower than hnswlib at matched recall. The real gap.
- **Build time:** ~6x slower than hnswlib (parallel).
- **Quantized recall:** well behind (SQ8 71.4%, RaBitQ 62.6%), and untuned.

Foxstash's advantages are architectural: pure Rust, no C++ toolchain, WASM and on-device,
local-first persistence. It is not currently competitive with faiss/hnswlib on throughput,
and pretending otherwise cost this project a release with a silent quantizer bug in it.

## Known issues

1. **Search throughput is ~11x off hnswlib** at matched recall. Now the top gap.
2. **Metric inconsistency** — `HNSWIndex` defaults to Cosine while `SQ8HNSWIndex` and
   `RaBitQHNSWIndex` are L2-only. Swapping index type to save memory silently changes the
   metric. `DistanceMetric` makes this *fixable*; the quantized indexes should take the
   same config.
3. **SQ8 recall is low** (71.4%) for a 4x quantizer. `QuantizedHNSWConfig` defaults to
   `ef_search: 50` where `HNSWConfig` uses 100; likely under-searched rather than broken.
4. **RaBitQ build is slow** (17.2s) — `prepare_query` runs per insert.
5. **`BinaryHNSWIndex` is deprecated** — 1.1% recall@10 here. Its zero threshold sets every
   bit on non-negative data (SIFT, any ReLU embedding), collapsing all codes to all-ones.
   Superseded by `RaBitQHNSWIndex` (same 32x, centered).
6. **100K/1M not yet run** — `benchmarks/data/` has `sift100k` and `sift1m`; only `sift10k`
   is reported here.
