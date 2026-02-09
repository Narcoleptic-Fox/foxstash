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
| Index | Compression | Recall | Use Case |
|-------|------------|--------|----------|
| FlatIndex | 1x | 100% | Small datasets (<1K) |
| HNSWIndex | 1x | ~97% | General purpose |
| SQ8HNSW | 4x | ~95% | Memory-constrained |
| BinaryHNSW | 32x | ~85% | Very large datasets |
| PQHNSW | 192x | ~80% | Extreme compression |

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
- **100K vectors, 128d**: 8,439 QPS @ 61.4% recall — **beats gold standards**
  - vs hnswlib: 4,245 QPS @ 40.3% recall (2x speed, 1.5x recall)
  - vs instant-distance: 587 QPS @ 62.1% recall (14x speed, similar recall)
- **Build**: Sequential 578.9s vs Parallel 7.4s (78x faster)
- **SIMD**: 3-4x speedup over scalar for distance computations
