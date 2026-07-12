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

## ⚠️ Read this before comparing the tables

**Foxstash's `HNSWIndex` ranks by cosine distance. Every competitor here ranks by L2, and
SIFT's ground truth is L2.** SIFT vector magnitudes vary by 1.4x, so the two metrics
genuinely disagree about who the nearest neighbours are.

This means **Foxstash's flagship HNSW cannot be placed in the same column as faiss or
hnswlib on this benchmark.** Scored against SIFT's L2 key it reads 55.1% — but that number
measures the metric gap, not the index. Against a cosine ground truth (computed brute-force
over the same data) the very same index scores **97.7%**. The graph is healthy; it is
answering a different question than the one SIFT asks.

Note also that Foxstash is **internally inconsistent**: `HNSWIndex` is cosine, while
`SQ8HNSWIndex` and `RaBitQHNSWIndex` are L2. Swapping index type to save memory silently
changes your distance metric. See "Known issues".

## Directly comparable — L2 metric, vs SIFT's L2 ground truth

These rows all answer the same question, so they can be ranked against each other.

| Library | Algorithm | Recall@10 | QPS | Build |
|---------|-----------|-----------|-----|-------|
| faiss | flat | 100.00% | 38,004 | 0.00s |
| faiss | ivf | 100.00% | **78,749** | 0.03s |
| annoy | annoy | 100.00% | 599 | 0.51s |
| faiss | hnsw | 99.99% | 34,005 | 0.18s |
| hnswlib | hnsw | 99.98% | 32,715 | 0.19s |
| **foxstash** | **flat** (control) | **100.00%** | 1,277 | 1.57s |
| **foxstash** | **sq8-hnsw** (4x) | **71.59%** | 11,050 | 2.66s |
| **foxstash** | **rabitq-hnsw** (32x) | **62.60%** | 885 | 17.20s |

The `flat` control reads 100%, which is what validates the loader and the metric. **Any
recall table without a passing control row is void** — if exact search doesn't score 100%
against the ground truth, nothing else in the table means anything.

**Foxstash's quantized indexes are well behind on equal terms.** SQ8 gives up 28 points of
recall to hnswlib while being 3x slower to query. This is the honest headline and it is not
what the old file claimed.

## Foxstash HNSW — scored against cosine ground truth

The fair question for a cosine index: does the graph find the true *cosine* neighbours?

| Build path | Recall@10 | Recall@100 | Build | QPS | (vs L2 key) |
|------------|-----------|------------|-------|-----|-------------|
| sequential | **97.72%** | 93.3% | 13.96s | 9,283 | 55.1% |
| incremental (`add()` loop) | 97.63% | 93.3% | 14.58s | 9,057 | 55.0% |
| parallel | **97.11%** | 92.7% | **1.89s** | 9,263 | 54.8% |

Yes: the graph is good. 97.7% recall@10 is a respectable HNSW.

**Parallel build is now recall-safe.** Before `fix(hnsw): apply diversity heuristic in
parallel build path`, the parallel builder skipped the Algorithm-4 diversity heuristic —
it "built" in 0.18s because it was doing less work, and paid 1.7 points of recall for it
(95.4%). With the heuristic it reaches 97.11%, statistically level with sequential's
97.72%, and still builds **7.4x faster** (1.89s vs 13.96s). Use parallel.

## Where Foxstash actually stands

Being blunt, because the last version of this file wasn't:

- **Search throughput:** ~9,300 QPS vs hnswlib's 32,715. Foxstash is **~3.5x slower**,
  not "2x faster" as previously claimed.
- **Build time:** 13.96s sequential / 1.89s parallel vs hnswlib's 0.19s. **10-73x slower.**
- **Recall (cosine):** 97.7% — genuinely good, and the one number that holds up.
- **Recall (L2):** not offered. There is no L2 HNSW index.

Foxstash's real advantages are architectural, not numeric: pure Rust, no C++ toolchain,
runs in WASM and on-device, local-first persistence. It is not currently competitive with
faiss/hnswlib on raw speed, and pretending otherwise cost this project a release with a
silent quantizer bug in it.

## Known issues

1. **No L2 metric on `HNSWIndex`** — blocks comparison on every standard ANN benchmark
   (SIFT, GIST, Deep1B are all L2). This is the single highest-value gap.
2. **Metric inconsistency** — `HNSWIndex` is cosine; `SQ8HNSWIndex` and `RaBitQHNSWIndex`
   are L2. Silent, and a real correctness footgun for callers.
3. **SQ8 recall is low** (71.6%) for a 4x quantizer. `QuantizedHNSWConfig` defaults to
   `ef_search: 50` where `HNSWConfig` uses 100; likely under-searched rather than broken.
4. **RaBitQ build is slow** (17.2s) — `prepare_query` runs per insert.
5. **`BinaryHNSWIndex` is deprecated** — 1.1% recall@10 here. Its zero threshold sets every
   bit on non-negative data (SIFT, any ReLU embedding), collapsing all codes to all-ones.
   Superseded by `RaBitQHNSWIndex` (same 32x, centered).
6. **100K/1M not yet run** — `benchmarks/data/` has `sift100k` and `sift1m`; only `sift10k`
   is reported here.
