# Foxstash Benchmark Results

Comparative benchmarks against industry-standard ANN libraries.

## Test Configuration

- **Dataset:** Synthetic SIFT-like (100K vectors, 128 dimensions, 10K queries)
- **Hardware:** [Run on local dev machine]
- **Date:** 2025-02-04

## Results Summary

### HNSW Index Performance (100K vectors)

| Library | ef_search | Build Time | Memory | Recall@10 | Recall@100 | QPS |
|---------|-----------|------------|--------|-----------|------------|-----|
| **FAISS** | 32 | 5.27s | 66 MB | 0.263 | 0.190 | 77,691 |
| **FAISS** | 64 | 5.35s | 62 MB | 0.400 | 0.300 | 55,954 |
| **FAISS** | 128 | 5.35s | 63 MB | 0.562 | 0.446 | 32,162 |
| **hnswlib** | 32 | 4.39s | 79 MB | 0.442 | 0.340 | 31,222 |
| **hnswlib** | 64 | 4.37s | 66 MB | 0.442 | 0.340 | 31,495 |
| **hnswlib** | 128 | 4.35s | 65 MB | 0.499 | 0.392 | 25,332 |
| **Annoy** | 10 trees | 0.64s | 87 MB | 0.074 | 0.055 | 9,742 |
| **Annoy** | 50 trees | 0.97s | 128 MB | 0.308 | 0.241 | 2,204 |
| **Annoy** | 100 trees | 1.36s | 188 MB | 0.517 | 0.419 | 1,134 |

### Key Observations

1. **FAISS HNSW** achieves highest QPS across all recall levels
2. **hnswlib** has faster build times but slightly lower QPS
3. **Annoy** trades QPS for simplicity and lower build time
4. All HNSW implementations show similar recall/QPS tradeoffs

### Notes

⚠️ **Synthetic Data Caveat:** These benchmarks use synthetic random vectors which don't exhibit the clustering patterns of real embeddings. Real-world recall would typically be 10-20% higher.

## Foxstash Comparison (TODO)

To include Foxstash in these benchmarks, we need:
1. PyO3 Python bindings for the benchmark harness
2. Or: Separate Rust benchmark using Criterion

### Expected Foxstash Advantages

Based on implementation:
- **SIMD acceleration** - 3-4x speedup on distance computation
- **Quantization options** - SQ8 (4x), Binary (32x), PQ (192x) compression
- **WASM support** - Unique capability for browser deployment

### Running Benchmarks

```bash
cd benchmarks/python

# Install dependencies
python -m venv env
source env/bin/activate
pip install -r requirements.txt

# Generate synthetic data
python download_datasets.py --synthetic --synthetic-size 100000

# Run benchmarks
python quick_bench.py
```

## Raw Results

Full benchmark data saved to `data/benchmark_results.json`.

---

*Last updated: 2025-02-04*
