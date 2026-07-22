# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`Storage::Warren`** — TurboRabit's 4-bit walk plus a two-level 8+8-bit residual rerank, with **no retained f32**. The arena block is byte-identical to `TurboRabit` (the walk *is* the same code); the residual reranks in rotated space via the identity `grid = u + c_B` (three SIMD dots per candidate, no O(dim²) inverse rotation). Delivers TurboRabit's recall/QPS at ~⅓ the vector memory (1,152 B vs 3,456 B at 768-d). Bulk-build only. Exposed to the Python binding as `warren[N]`. Snapshot format bumped v2 → v3.
- **Filtered search on `HNSWIndex`** — `search_filtered(query, k, &FilterMask)` walks the whole graph (excluded nodes are still traversed, for connectivity) but admits only allowed nodes to the result heap, gated at layer 0. Returns up to `k` allowed nearest neighbours with no over-fetch. Build a reusable `FilterMask` once with `filter_mask` (a predicate over id/content/metadata) or `filter_mask_ids`; reuse it across queries. Unfiltered `search` is unchanged (one predicted branch per candidate). Python: `Filter` + `query_filtered`.
- **Predicate-gated filtered search** — `HNSWIndex::search_filtered_by(query, k, allow)` and `search_batch_filtered_by(..)` filter *during* the walk against a live predicate `allow(id, metadata)`, for one-off filters not worth a prebuilt `FilterMask`. Evaluated lazily on visited nodes only — no O(n) up-front pass, no over-fetch. The mask and predicate paths share one layer-0 gating mechanism (`dyn Fn(usize) -> bool`).

### Changed

- **`Collection` filtered search now uses the graph's native filtered walk** (`search_filtered_by` / `search_batch_filtered_by`) instead of progressive over-fetch (`2×` → `4×` → `8×` → full scan). A filtered query is now a single graph walk that collects up to `k` matching results directly, rather than re-running the walk up to four times and degrading to a full brute-force scan on selective filters.

### Deprecated

- **`Storage::TurboQuant`** (`#[deprecated(since = "0.7.0")]`, scheduled for removal in 0.8). Dominated by `Storage::TurboRabit` (Extended RaBitQ) at every matched bit budget, and — unlike plain TurboQuant — it collapses on out-of-distribution data (recall 0.888 vs TurboRabit's 0.987 on the yandex-200 OOD set). It remains fully functional through 0.7; migrate to `Storage::TurboRabit` or `Storage::SQ8`.

## [0.6.0] - 2026-07-15

The "1.0 audit" release: one index, one config, one metric; −5,380 lines.

### Changed

- **One `HNSWIndex`, quantization is a `Storage` mode** (`F32` / `SQ8` / `RaBitQ`), and every storage mode honours every metric. Replaces the separate `PQHNSWIndex` / `RaBitQHNSWIndex` / `SQ8HNSWIndex` / `BinaryHNSWIndex` types. **Breaking.**
- **Parallel builder recall fixed**: `par_insert` truncated the zero-layer candidate beam to `m0` before the Algorithm-4 diversity heuristic, so it could not diversify and the graph had no long-range bridges — ~15 recall points below the sequential builder at equal `ef_search` on clustered data (+0.5–2 on SIFT1M, which understates it). Now feeds the full `ef_construction` beam. Guarded by `both_builders_reach_similar_recall`, a recall-parity equivalence test.
- Node layout changed from struct-of-arrays to an interleaved arena, fixing a two-random-DRAM-reads-per-node-visit regression that lost to hnswlib at 1M.

### Added

- **RaBitQ storage** (1-bit quantization): wins at ≥768-d, where SQ8's per-node dequant cost dominates and RaBitQ's block is far smaller. SQ8 still wins ≤384-d.
- **Python binding** (`crates/python`, PyO3): `Foxstash(metric, dim, m, ef_construction, storage, rerank_candidates)` with `fit` / `set_query_arguments` / `query`, validated against numpy brute-force ground truth; CI-gated.
- CI gates: clippy `--all-targets` (was not linting tests, where the bugs hid), a generated config×code-path matrix, and the Python recall suite.

### Removed

- **Per-quantizer index types** and ~5,380 lines of dead/duplicate code: a second SQ8 implementation the index never called, an unused streaming module, product-quantize scaffolding, and uniform-random benchmarks (which mask graph-connectivity bugs).
- **Reusable search context** (`Collection::create_search_context`, `Collection::search_with_context`), added in 0.5.0 on the promise of reduced per-query allocation overhead. Measured on SIFT1M: 4,118 QPS via `search()` vs 4,121 QPS reused — **1.00x**, no measurable difference. Search is memory-latency bound, not allocation bound, so there was nothing for a reused scratch buffer to save. `search_batch_fast` (measured 0.97x `search_batch` — slower) deleted for the same reason. `Collection::search_batch` is unaffected and remains.

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

[Unreleased]: https://github.com/Narcoleptic-Fox/foxstash/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/Narcoleptic-Fox/foxstash/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/Narcoleptic-Fox/foxstash/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/Narcoleptic-Fox/foxstash/compare/v0.4.0...v0.4.1
[0.1.0]: https://github.com/Narcoleptic-Fox/foxstash/releases/tag/v0.1.0
