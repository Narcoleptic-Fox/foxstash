# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/Narcoleptic-Fox/foxstash/compare/v0.4.1...HEAD
[0.4.1]: https://github.com/Narcoleptic-Fox/foxstash/compare/v0.4.0...v0.4.1
[0.1.0]: https://github.com/Narcoleptic-Fox/foxstash/releases/tag/v0.1.0
