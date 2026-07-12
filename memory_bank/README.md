# Foxstash Memory Bank

Project context and learnings for AI assistant sessions.

## Project Summary

**Foxstash** is a high-performance, local-first RAG (Retrieval-Augmented Generation) library written in Rust.
- **Version**: 0.2.1
- **License**: MIT
- **Repo**: https://github.com/Narcoleptic-Fox/foxstash

## Architecture Overview

### Workspace Crates
| Crate | Purpose |
|-------|---------|
| `foxstash-core` | Main library - vectors, indexes, storage, embeddings |
| `foxstash-wasm` | WebAssembly bindings (browser) |
| `foxstash-native` | Native bindings (includes ONNX) |
| `foxstash-python` | Python bindings (PyO3) |
| `foxstash-benches` | Benchmark suite |

### Core Modules (`crates/core/src/`)
- **`vector/`** — SIMD ops (`pulp`), quantization (SQ8, Binary, PQ)
- **`index/`** — FlatIndex, HNSWIndex, SQ8/Binary/PQ HNSW variants
- **`storage/`** — File persistence, compression (Gzip/LZ4/Zstd), WAL
- **`embedding/`** — ONNX Runtime integration (feature-gated: `onnx`)
- **`lib.rs`** — Document, SearchResult, RagConfig, RagError

### Key Design Patterns
1. **Struct-of-Arrays (SoA)** — Hot embeddings separate from cold metadata
2. **Generation-based visited tracking** — O(1) reset via counters
3. **Runtime SIMD detection** — AVX2/SSE/NEON via `pulp` with scalar fallback
4. **WAL-based incremental persistence** — Append-only with periodic checkpoints
5. **Quantizer trait** — Symmetric + asymmetric distance abstraction

### Index Types
| Index | Compression | Recall@10 (SIFT10K) | Use Case |
|-------|------------|---------------------|----------|
| FlatIndex | 1x | 100% | Small datasets (<1K) |
| HNSWIndex | 1x | ~97% | General purpose |
| SQ8HNSW | 4x | 100% | Memory-constrained |
| RaBitQHNSW | 32x | 62.6% | Very large datasets |
| BinaryHNSW | 32x | **1.1% — DEPRECATED** | Do not use |
| PQHNSW | 192x | ~80% (unverified) | Extreme compression |

> **Recall numbers here were previously invented.** The table used to claim BinaryHNSW
> got "~85%". Measured on real SIFT10K it gets **1.1%** — its zero threshold sets every
> bit on non-negative data (SIFT, any ReLU embedding), collapsing all codes to all-ones.
> Deprecated in favour of `RaBitQHNSWIndex` (same 32x, centered). Verify with:
> `cargo run --release -p foxstash-benches --example quantizer_sift`.
>
> Anything still marked "unverified" above has *not* been measured on real data and
> should not be quoted. See the note on synthetic data below.

## Build & Test

```bash
# Run tests (no ONNX, works on all platforms)
cargo test -p foxstash-core --no-default-features

# Run with compression features
cargo test -p foxstash-core --features compression-all

# ONNX has linker issues on Windows (MSVC CRT conflicts)
# Works on Linux/macOS CI
```

## Known Issues
- **ONNX on Windows**: `ort` crate causes LNK2005 symbol conflicts with MSVC runtime (static vs dynamic CRT mismatch). Tests pass on Linux/macOS CI. Tracked for fix.
- Default features are empty (`default = []`), so `cargo test` without features works fine.

## Recent Work (as of 2026-02-09)
- Parallel HNSW builder (78x faster, layer-copying from instant-distance)
- Diversity-aware neighbor selection (Algorithm 4)
- Product Quantization (PQ) up to 192x compression
- Incremental persistence (WAL)
- WASM support with IndexedDB
- Refactoring: removed dead code, cleaned up SearchContext
- Performance: 10x search speedup with batch parallelization

## Performance Characteristics

> ⚠️ **The recall claims below are void.** They were measured on synthetic vectors, where
> every ANN collapses to ~60% regardless of quality. "Beats gold standards on recall" was
> an artifact: hnswlib's 40.3% there is 99.99% on real SIFT10K. **Do not quote these
> recall numbers.** The *speed* numbers are probably still directionally fine (throughput
> doesn't depend on cluster structure), but they have not been re-measured on real data.

- **100K vectors, 128d**: ~8,400 QPS — ~~61.4% recall — beats gold standards~~ (recall void)
  - vs hnswlib: ~4,200 QPS — ~~40.3% recall (1.5x recall)~~ (recall void; real = 99.99%)
  - vs instant-distance: ~590 QPS (14x speed) — recall void
- **Build**: Sequential 578.9s vs Parallel 7.4s (78x faster)
- **SIMD**: 3-4x speedup over scalar for distance computations

**TODO:** regenerate all comparison numbers against `benchmarks/data/sift*` and add Foxstash
to `benchmarks/results/report_sift10k.md`, which currently contains only annoy/faiss/hnswlib.

## Benchmarking: never trust synthetic vectors

`benchmarks/RESULTS.md` is measured on "Synthetic SIFT-like" data and its recall column
is **meaningless**. On that data hnswlib scores 39.5% and faiss-hnsw 44.9%; on real
SIFT10K (`benchmarks/results/report_sift10k.md`) the same libraries score **99.99%** and
**100%**. Synthetic vectors have no cluster structure, so every ANN collapses to ~60% and
real graph-connectivity bugs hide behind the noise.

Real data lives in `benchmarks/data/{sift10k,sift100k,sift1m}` as `.npy`. Rust can read it
via the loader in `crates/benches/examples/quantizer_sift.rs`.

**Rule:** any benchmark reporting recall must include an exact/flat control row. If the
control is not ~100%, the loader or the distance metric is wrong and every other row in
the table is void. Foxstash itself does not yet appear in the real-data report — its only
published comparison numbers come from the poisoned synthetic run and need regenerating.
