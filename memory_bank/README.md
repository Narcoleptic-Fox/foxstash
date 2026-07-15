# Foxstash Memory Bank

Project context and learnings for AI assistant sessions.

> **Every number in this file is either cited to a specific source or explicitly marked
> unsourced.** If you add a claim, add its citation next to it — a number without one gets
> deleted the next time this file is audited, not captioned "unverified" and kept. See
> `benchmarks/RESULTS.md` for the project's actual benchmark log; this file summarizes it and
> should not drift from it.

## Project Summary

**Foxstash** is a high-performance, local-first RAG (Retrieval-Augmented Generation) library
written in Rust.
- **Version**: 0.5.0 (`Cargo.toml`, `[workspace.package].version`)
- **License**: MIT
- **Repo**: https://github.com/Narcoleptic-Fox/foxstash

## Architecture Overview

### Workspace Crates

Verified against `ls crates/` and `Cargo.toml`'s `[workspace].members` — five crates, no more:

| Crate | Purpose |
|-------|---------|
| `foxstash-core` | Main library — vectors, indexes, storage, embeddings |
| `foxstash-db` | Document store: named collections, metadata filtering, BM25, hybrid search |
| `foxstash-wasm` | WebAssembly bindings (browser), IndexedDB persistence |
| `foxstash-native` | Native bindings, full ONNX support |
| `foxstash-benches` | Benchmark suite (`crates/benches/examples/*.rs`) |

There is no `foxstash-python` crate. (An earlier version of this file listed one; it does not
exist in this workspace and never has, as far as `Cargo.toml`'s member list shows.)

### Core Modules (`crates/core/src/`)
- **`vector/`** — SIMD ops (`pulp`), quantization: scalar (SQ8), RaBitQ (1-bit), product
  quantization (PQ)
- **`index/`** — `FlatIndex` (brute force), `HNSWIndex` (storage modes `F32`/`SQ8`/`RaBitQ` —
  see below), `PQHNSWIndex`
- **`storage/`** — File persistence, compression (Gzip/LZ4/Zstd), WAL
- **`embedding/`** — ONNX Runtime integration (feature-gated: `onnx`)
- **`lib.rs`** — `Document`, `SearchResult`, `RagConfig`, `RagError`

### Key Design Patterns
1. **Struct-of-Arrays (SoA)** — hot embeddings separate from cold metadata
2. **Generation-based visited tracking** — O(1) reset via counters
3. **Runtime SIMD detection** — AVX2/SSE/NEON via `pulp`, with scalar fallback
4. **WAL-based incremental persistence** — append-only with periodic checkpoints
5. **Storage modes, not parallel index types** — `HNSWConfig::storage` (`F32`/`SQ8`/`RaBitQ`)
   selects the traversal encoding on one `HNSWIndex`, rather than shipping a separate index
   type per quantizer. This replaced an earlier design with standalone `SQ8HNSWIndex` and
   `BinaryHNSWIndex` types — see "What used to exist and was deleted" below for why.

## What exists today

Current index types, with their actual measured recall and the benchmark that produced it.
Recall for `HNSWIndex` (any storage mode) is not a single number — `ef_search` trades it
against speed — so these are recall points from `benchmarks/RESULTS.md`'s matched-recall
tables, not the index's ceiling.

| Type | Compression | Measured recall@10 | Source |
|------|-------------|---------------------|--------|
| `FlatIndex` | none | 100% (exact brute-force by construction) | n/a — not approximate |
| `HNSWIndex`, `Storage::F32` | none | matched-recall points 93.0%-99.9% (SIFT1M) | `benchmarks/RESULTS.md` §"SIFT1M — QPS at matched recall" |
| `HNSWIndex`, `Storage::SQ8` | ~2x block size | **1.20x hnswlib's QPS @ 99.5% recall** (SIFT1M, 128-d) | `benchmarks/RESULTS.md` §"Headline"; reproduce with `cargo run --release -p foxstash-benches --example storage_pareto` |
| `HNSWIndex`, `Storage::RaBitQ` | ~3.3x block size | **1.79x hnswlib's QPS @ 98.3% recall** (GIST1M, 960-d) | `benchmarks/RESULTS.md` §"Headline"; reproduce with `cargo run --release -p foxstash-benches --example storage_pareto gist1m` |
| `PQHNSWIndex` | 192x on the vector payload | **~55% recall@10** on clustered data | `crates/core/src/index/hnsw_pq.rs` module doc, backed by test `pq_use_distance_cache_false_retrieves_correctly_on_clustered_data` |

`Storage::SQ8` and `Storage::RaBitQ` are not interchangeable wins — they trade places by
dimension (SQ8 wins at 128-d, is worthless at 960-d; RaBitQ is the reverse). See
`benchmarks/RESULTS.md` §"The right quantizer depends on the dimension" for the full mechanism
and measurements, and the main `README.md`'s "Pick your storage mode by dimension" section for
the condensed version.

## What used to exist and was deleted

Kept here deliberately — this project's most expensive lessons are in *why* these were removed,
and re-deriving them once is enough. The point is not that these names are forbidden to mention;
it's that none of them are a live, selectable option today.

| Type | What it was | Why it's gone | Source |
|------|-------------|----------------|--------|
| `SQ8HNSWIndex` (standalone) | A separate 8-bit index type: own config, `Vec<HashSet<usize>>` adjacency, sequential build, hardcoded L2 metric | Dominated on every axis by `Storage::SQ8` on the unified `HNSWIndex`: 2.7-3.2x the QPS, 9.6x faster build, higher recall at every `ef` tested. Also a metric footgun — no `metric` field, silently always L2 against `HNSWIndex`'s cosine default. | `benchmarks/RESULTS.md` §"Why `SQ8HNSWIndex` was deleted" |
| `BinaryHNSWIndex` | 1-bit quantized index, sign-threshold at zero | Measured **1.2% recall** on real SIFT10K while its rustdoc claimed ~90% — the zero threshold sets every bit on non-negative embeddings (SIFT, any ReLU output), collapsing all codes to all-ones. Deprecated in favour of `Storage::RaBitQ` (also 1-bit, but a centered/rotated estimator, not a zero threshold). | `benchmarks/RESULTS.md` §"The bugs that survived because nothing could fail"; reproduce the failure with `cargo run --release -p foxstash-benches --example quantizer_sift` |
| `search_with_context` / `create_search_context` (public `HNSWIndex` API) | A reusable-scratch search API, documented as "~2-3x faster than `search()`" | The claim was never true — measured 4,118 QPS vs 4,121 QPS reused, **1.00x**. `search()` already reuses its allocator's free list for same-sized bitsets; the search is memory-latency bound, not allocation bound. Replaced by `Searcher` (`index.searcher()`), which exists for `distance_calls()` instrumentation, not speed. | `crates/core/src/index/hnsw.rs` (`Searcher` doc comment); reproduce with `cargo run --release -p foxstash-benches --example search_api_cost` |
| `search_batch_fast` | A batch-search fast path | Measured **0.97x** `search_batch` — slower, not faster. Deleted alongside `search_with_context`. | `crates/benches/examples/search_api_cost.rs` |

## Why the recall numbers in this file changed shape

This file used to carry a "Performance Characteristics" section with numbers measured on
**synthetic** vectors (uniform random, no cluster structure). Every ANN algorithm — foxstash
included — collapses to roughly the same ~60% recall on data like that, because there is no
cluster structure to lose in the first place; it does not distinguish a working index from a
broken one. On that synthetic set, hnswlib scored ~39-40% recall; on real SIFT10K it scores
99.99% (`benchmarks/results/report_sift10k.md`). Those old numbers are not reproduced here
because they measured noise, not the library, and a "78x faster" build claim and a "10x search
speedup" claim that used to sit alongside them had no traceable benchmark run behind them at
all — see "Claims removed from this file" below.

**Rule going forward:** any recall number quoted in this repo must cite a real dataset (SIFT,
GIST) and, ideally, an exact/brute-force control row scoring ~100% on the same run — if the
control isn't ~100%, the loader or metric is wrong and every other number in that run is void.
See `benchmarks/RESULTS.md` §"Four traps, every one of which has caught this project" for the
full list of ways this has gone wrong before (recall on synthetic data, unmatched `ef`,
uncounted competitor threads, concurrent benchmark runs).

The originally-planned fix — regenerate `benchmarks/results/report_sift10k.md` to include
foxstash alongside annoy/faiss/hnswlib — was superseded, not completed: `benchmarks/RESULTS.md`
now carries real SIFT1M and GIST1M matched-recall comparisons against hnswlib and faiss
directly, which is a stronger methodology (1M vectors instead of 10K, matched-recall rather
than matched-`ef`) than what the SIFT10K report format supports. `report_sift10k.md` itself
still contains only annoy/faiss/hnswlib as of this writing — that specific gap is real, it's
just no longer the right thing to fill first.

## Build & Test

```bash
# Run tests (no ONNX, works on all platforms)
cargo test -p foxstash-core --no-default-features

# Run with compression features
cargo test -p foxstash-core --features compression-all
```

Verified against `crates/core/Cargo.toml`: `default = []`, and `compression-all = ["lz4",
"zstd"]` both exist as written.

## Known Issues
- **ONNX on Windows**: `ort` crate causes LNK2005 symbol conflicts with MSVC runtime (static vs
  dynamic CRT mismatch). Tests pass on Linux/macOS CI. Tracked for fix.
- Default features are empty (`default = []`), so `cargo test` without features works fine.

## Claims removed from this file, and why

Per this file's own rule (cite it or delete it), the following were removed rather than kept
with an "unverified" caption, because no benchmark producing them could be found in
`crates/benches` or `benchmarks/RESULTS.md`:

- **"78x faster" parallel vs. sequential HNSW build.** No bench compares `BuildStrategy::
  Sequential` against `BuildStrategy::Parallel` build wall-time directly. (`benchmarks/
  RESULTS.md` does compare foxstash's overall build time against hnswlib/faiss — 2.1x faster
  than hnswlib at 128-d, see §"Build time and memory" — but that is a different comparison
  than Sequential-vs-Parallel, and is cited above where it's actually relevant.)
- **"10x search speedup with batch parallelization."** No `search` vs. `search_batch`
  throughput comparison at a specific speedup ratio was found.
- **"3-4x SIMD speedup over scalar for distance computations."** No SIMD-vs-scalar
  microbenchmark producing this ratio was found in `crates/core/benches` or
  `crates/benches/examples`.
- **`PQHNSW | 192x | ~80% (unverified)`** from the old index-types table. Contradicted the
  measured figure now cited above (~55% recall@10) — the measured one wins, the unverified one
  is gone.

If you have the run that produced any of these, put the number back with a citation to it.

## Orphan claim from the previous rewrite — now RESOLVED

`benchmarks/RESULTS.md` §"Known issues" states `RaBitQHNSWIndex` and `PQHNSWIndex` "have no
`metric` field at all — they are hardcoded L2." This is confirmed still accurate: `grep -n
DistanceMetric crates/core/src/index/hnsw_pq.rs crates/core/src/index/hnsw_quantized.rs`
returns zero hits in both files. Both types are hardcoded L2 with no `metric` field at all,
while `HNSWIndex` defaults to Cosine — the same metric footgun that was already closed for
`SQ8HNSWIndex` by folding it into `Storage::SQ8`. `RaBitQHNSWIndex` is slated for the same
treatment: it is not deleted as of this writing (still a separate type, not in the "What used
to exist and was deleted" table above), but `Storage::RaBitQ` on `HNSWIndex` — see "What exists
today" — is already a near-superset of it, which is the deletion case being built.
`PQHNSWIndex` has no
`Storage::PQ` equivalent, so its version of this gap has no planned fix yet.
