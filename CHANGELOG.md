# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Removed

- **Reusable search context** (`Collection::create_search_context`, `Collection::search_with_context`), added below in 0.5.0 on the promise of reduced per-query allocation overhead. Measured on SIFT1M: 4,118 QPS via `search()` vs 4,121 QPS reused — **1.00x**, no measurable difference. The search is memory-latency bound, not allocation bound, so there was nothing for a reused scratch buffer to save. The underlying `HNSWIndex::search_with_context`/`create_search_context` and `search_batch_fast` (measured 0.97x `search_batch` — slower) were deleted for the same reason; see `crates/core/src/index/hnsw.rs`'s `Searcher` doc and `cargo run --release -p foxstash-benches --example search_api_cost` to reproduce. `Collection::search_batch` (also added in 0.5.0, below) is unaffected — it doesn't carry this claim and remains.

## [0.5.0] - 2026-03-01

### Added

- **Parallel batch search** (`Collection::search_batch`): search multiple queries concurrently via rayon with an optional metadata filter, mirroring the single-query `search(query, k, filter)` signature. Uses thread-local pooled `SearchContext`s to minimise allocation overhead. Filtered paths apply the same progressive over-fetch semantics (`2×`, `4×`, `8×`, full scan) as single-query filtered search.
- **Reusable search context** (`Collection::create_search_context`, `Collection::search_with_context`): allocate a `SearchContext` once and reuse it across repeated single-query searches in tight loops, reducing per-query allocation overhead.

### Changed

- SIMD kernels migrated to pulp 0.22 idiomatic API. Manual `while`-loop + `f32s_partial_load`/`cast_lossy` workarounds replaced with `S::as_simd_f32s()` (proper chunk/tail splitting). All renamed intrinsics updated (`f32s_splat` → `splat_f32s`, `f32s_mul_add_e` → `mul_add_e_f32s`, etc.).

### Fixed

- Collection name validation now rejects backslash (`\`) on all platforms. Previously `\` was accepted on Linux because it is a valid filename character there, while being rejected on Windows as a path separator.

### Dependencies

- `pulp` updated from `0.18` to `0.22`
- `tokenizers` updated from `0.19` to `0.22`
- `lru` updated from `0.12` to `0.16`
- `ndarray` updated from `0.15` to `0.17`

## [0.4.1] - 2026-03-01

### Fixed

- Blocked path traversal in collection names to prevent filesystem escape in create/get/delete paths.
- Removed durability races between WAL appends and checkpoint/compaction snapshots that could lose acknowledged writes.
- Hardened atomic persistence writes for manifests/checkpoints and made WAL recovery tolerate torn/truncated tail entries.
- Fixed HNSW search-layer neighbor truncation when `m0 > 64`.
- Hardened WASM/native boundaries by replacing panic-prone paths with validated error returns.
- Restored strict `clippy -D warnings` compatibility across bench/native targets.

## [0.1.1] - 2025-01-31

### Fixed

- Add README to crates.io package

## [0.1.0] - 2025-01-31

### Added

- **HNSW Index**: Hierarchical Navigable Small World graph for fast approximate nearest neighbor search
  - Algorithm 4 diversity heuristic for better graph connectivity
  - Configurable M, ef_construction, ef_search parameters
  - Multi-layer graph construction

- **Vector Quantization**: Memory-efficient vector storage
  - Scalar Quantization (SQ8): 4x compression with ~95% recall
  - Binary Quantization: 32x compression with two-phase reranking
  - Product Quantization (PQ): Up to 192x compression

- **SIMD Acceleration**: Hardware-accelerated vector operations
  - AVX2, SSE4.1, and NEON support
  - Automatic fallback to scalar operations
  - 3-4x speedup for distance computations

- **Streaming Operations**: Memory-efficient batch processing
  - BatchBuilder for large-scale ingestion
  - Progress callbacks with throughput metrics
  - Filtered search with pagination

- **Persistence**: Durable storage options
  - File-based storage with compression (Gzip, LZ4, Zstd)
  - Write-Ahead Log (WAL) for incremental updates
  - Checkpointing for fast recovery

- **Flat Index**: Exact search baseline for comparison

### Notes

- ONNX embedding support available but may have platform-specific limitations on Windows
- WASM support is experimental

[Unreleased]: https://github.com/Narcoleptic-Fox/foxstash/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/Narcoleptic-Fox/foxstash/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/Narcoleptic-Fox/foxstash/compare/v0.4.0...v0.4.1
[0.1.0]: https://github.com/Narcoleptic-Fox/foxstash/releases/tag/v0.1.0
