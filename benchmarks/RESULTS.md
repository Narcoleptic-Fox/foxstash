# Foxstash Benchmark Results

Comparative benchmarks against industry-standard ANN libraries.

## Test Configuration

- **Dataset:** Synthetic SIFT-like (100K vectors, 128 dimensions, 10K queries)
- **Hardware:** Cortex dev server (Ubuntu 24.04)
- **Date:** 2026-02-11

## Rust Ecosystem Comparison (100K vectors)

| Library | Build Time | Search QPS | Recall@10 |
|---------|------------|------------|-----------|
| **Foxstash** (sequential) | **7.71s** | **8,251** | 63.3% |
| **Foxstash** (parallel) | **7.63s** | 788 | 59.3% |
| instant-distance | 74.21s | 559 | 58.6% |

### Analysis

- **Search Performance:** Foxstash is **14.75x faster** than instant-distance
- **Build Performance:** Foxstash is **9.62x faster** than instant-distance
- **Recall:** Foxstash achieves higher recall (63.3% vs 58.6%) at the same parameters
- **SIMD:** Foxstash uses SIMD-accelerated distance computation

## Python Ecosystem Comparison (100K vectors)

| Library | Build Time | Search QPS | Recall@10 |
|---------|------------|------------|-----------|
| hnswlib | 5.74s | 4,110 | 39.5% |
| faiss-hnsw | 8.63s | 3,131 | 44.9% |

### Cross-Ecosystem

| Library | Search QPS | vs Foxstash |
|---------|------------|-------------|
| **Foxstash** | **8,251** | - |
| hnswlib (C++) | 4,110 | 2.0x slower |
| faiss-hnsw (C++) | 3,131 | 2.6x slower |
| instant-distance (Rust) | 559 | 14.8x slower |

### Notes

- Python libraries use C/C++ backends (not pure Python)
- All benchmarks use M=32, ef_construction=100, ef_search=64 (or equivalent)
- Synthetic random vectors; real-world recall is typically 10-20% higher

### Foxstash Advantages

- **Search speed** - 14.75x faster than instant-distance, 2x faster than hnswlib
- **Build speed** - 9.62x faster than instant-distance
- **Higher recall** - 63.3% vs 58.6% (instant-distance) at same parameters
- **Quantization options** - SQ8 (4x), Binary (32x), PQ (192x) compression
- **WASM support** - Same code runs in browser
- **Streaming ingestion** - Batch processing with progress callbacks

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
