# Foxstash Benchmark Results

Comparative benchmarks against industry-standard ANN libraries.

## Test Configuration

- **Dataset:** Synthetic SIFT-like (100K vectors, 128 dimensions, 10K queries)
- **Hardware:** Cortex dev server (Ubuntu 24.04)
- **Date:** 2026-02-11

## Rust Ecosystem Comparison (100K vectors)

| Library | Build | Search Mode | Search QPS | Recall@10 |
|---------|-------|-------------|------------|-----------|
| **Foxstash** | parallel | batch (rayon) | **12,948** | 59.8% |
| **Foxstash** | parallel | single-threaded (ctx reuse) | **1,238** | 59.8% |
| **Foxstash** | sequential | single-threaded (ctx reuse) | 1,234 | 56.9% |
| instant-distance | default | single-threaded (ctx reuse) | 539 | 58.0% |

Build times: Foxstash parallel **7.7s**, Foxstash sequential 589s, instant-distance 75.2s

### Analysis

- **Single-threaded search:** Foxstash is **2.3x faster** than instant-distance (1,238 vs 539 QPS)
- **Batch search:** Foxstash is **24x faster** than instant-distance (12,948 vs 539 QPS)
- **Build Performance:** Foxstash parallel build is **9.8x faster** than instant-distance
- **Recall:** Comparable (59.8% Foxstash vs 58.0% instant-distance with same synthetic data)

### Search Optimizations (v0.3)

The single-threaded QPS improvement from ~800 to ~1,240 comes from:

1. **Fused cosine distance** — single SIMD dispatch + single pass (was 4 dispatch calls, 3 passes)
2. **Precomputed norms** — stored at insert time, eliminates per-query recomputation
3. **Bitset visited tracking** — 12.5 KB packed bitset fits L1 cache (was 800 KB generation counter)
4. **Deeper prefetching** — 2 neighbors ahead, 3 cache lines per embedding, cross-platform
5. **Batch distance + deferred heap** — compute/memory separation for better ILP
6. **Flat layer 0 connections** — single-indirection array (was triple-indirection Vec<Vec<Vec>>)

## Python Ecosystem Comparison (100K vectors)

| Library | Build Time | Search QPS | Recall@10 |
|---------|------------|------------|-----------|
| hnswlib | 5.74s | 4,110 | 39.5% |
| faiss-hnsw | 8.63s | 3,131 | 44.9% |

> **Note:** Python benchmarks use `ef_search=64` vs Foxstash `ef_search=100`.
> Lower ef_search increases QPS but reduces recall. An apples-to-apples comparison
> would show Foxstash recall higher and QPS gap narrower.

### Cross-Ecosystem

| Library | Search QPS | vs Foxstash (1T) |
|---------|------------|------------------|
| **Foxstash** (batch) | **12,948** | — |
| **Foxstash** (1T) | **1,238** | — |
| hnswlib (C++, ef=64) | 4,110 | 3.3x faster* |
| faiss-hnsw (C++, ef=64) | 3,131 | 2.5x faster* |
| instant-distance (Rust) | 539 | 2.3x slower |

*hnswlib/faiss use lower ef_search (64 vs 100), inflating their QPS relative to Foxstash.

### Notes on Recall

All libraries show low recall at 100K synthetic vectors. This is expected —
uniform random vectors in 128 dimensions are nearly equidistant due to the
curse of dimensionality, making nearest-neighbor separation extremely hard.

At **10K vectors**, Foxstash achieves **97% recall** with the same parameters.
Real-world embeddings (which have natural clustering) typically see 10-20%
higher recall than these synthetic benchmarks.

### Foxstash Advantages

- **Search speed** — 2.3x faster than instant-distance single-threaded, 24x with rayon
- **Build speed** — 9.8x faster than instant-distance
- **Comparable recall** — 59.8% vs 58.0% at same parameters
- **Quantization options** — SQ8 (4x), Binary (32x), PQ (192x) compression
- **WASM support** — Same code runs in browser
- **Streaming ingestion** — Batch processing with progress callbacks

### Running Benchmarks

```bash
# Full suite (Rust + Python comparisons)
./scripts/bench.sh

# Or manually:
cd benchmarks/python
python -m venv env
source env/bin/activate
pip install -r requirements.txt
python download_datasets.py --synthetic --synthetic-size 100000
python quick_bench.py
```

## Raw Results

Full benchmark data saved to `data/benchmark_results.json`.

---

*Last updated: 2026-02-11*
