# Foxstash

**High-performance local RAG library for Rust**

[![Crates.io](https://img.shields.io/crates/v/foxstash-core.svg)](https://crates.io/crates/foxstash-core)
[![Documentation](https://docs.rs/foxstash-core/badge.svg)](https://docs.rs/foxstash-core)
[![CI](https://github.com/Narcoleptic-Fox/foxstash/actions/workflows/ci.yml/badge.svg)](https://github.com/Narcoleptic-Fox/foxstash/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Foxstash is a local-first Retrieval-Augmented Generation (RAG) library featuring SIMD-accelerated vector operations, HNSW indexing, vector quantization, ONNX embeddings, hybrid search (BM25 + vector), and WebAssembly support.

## Features

- **SIMD-Accelerated** - AVX2/SSE/NEON vector operations with runtime CPU detection
- **HNSW Indexing** - Hierarchical Navigable Small World graphs for fast similarity search
- **Quantized traversal** - `Storage::SQ8` (8-bit) beats hnswlib **1.20x** at 128-d; `Storage::RaBitQ` (1-bit) beats it **1.79x** at 960-d. They swap places with dimension — [pick by dimension](#-pick-your-storage-mode-by-dimension--the-two-swap-places)
- **Hybrid Search** - Combine BM25 keyword search with vector similarity for best-of-both recall
- **ONNX Embeddings** - Generate embeddings locally with MiniLM-L6-v2 or any ONNX model
- **WASM Support** - Run in the browser with IndexedDB persistence
- **Compression** - Gzip, LZ4, and Zstd support for efficient storage
- **Incremental Persistence** - Write-ahead log for fast updates without full rewrites
- **Local-First** - Your data never leaves your machine

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
foxstash-core = "0.5"
```

### Basic Usage

```rust
use foxstash_core::{Document, RagConfig, IndexType};
use foxstash_core::index::HNSWIndex;

// Create an HNSW index
let mut index = HNSWIndex::with_defaults(384); // 384-dim for MiniLM-L6-v2

// Add documents with embeddings
let doc = Document {
    id: "doc1".to_string(),
    content: "Foxes are clever animals".to_string(),
    embedding: vec![0.1; 384], // Your embedding here
    metadata: None,
};
index.add(doc)?;

// Search for similar documents
let query = vec![0.1; 384];
let results = index.search(&query, 5)?;

for result in results {
    println!("{}: {:.4}", result.id, result.score);
}
```

### Quantized traversal (`Storage::SQ8` / `Storage::RaBitQ`)

This is where the wins come from. The graph is built with **exact** distances; only the bytes
read during traversal are quantized, and a final pool is rescored against the exact vectors — so
recall barely moves while the hot node block shrinks.

**Which code you want depends on your dimension** (see the note below): `SQ8` at 128-d,
`RaBitQ` at 768-d and above. Swap the one field.

```rust
use foxstash_core::index::{BuildStrategy, DistanceMetric, HNSWConfig, HNSWIndex, Storage};

let index = HNSWIndex::build(embeddings, HNSWConfig {
    metric: DistanceMetric::L2,
    storage: Storage::SQ8,     // 8-bit codes in the hot node block
    rerank_candidates: 100,    // rescore the top 100 against exact vectors
    build_strategy: BuildStrategy::Parallel,
    ..Default::default()
});

let results = index.search(&query, 10)?;
```

At 768-d and above, change one field — `Storage::RaBitQ` — and raise the rerank pool, because a
1-bit code ranks more coarsely and needs a deeper pool to rescore from:

```rust
let index = HNSWIndex::build(embeddings, HNSWConfig {
    metric: DistanceMetric::L2,
    storage: Storage::RaBitQ,  // 1-bit codes; 1.79x hnswlib at 960-d
    rerank_candidates: 400,    // deeper pool than SQ8 — a coarse code needs it
    build_strategy: BuildStrategy::Parallel,
    ..Default::default()
});
```

If recall *falls* as you raise `ef_search`, that is the rerank pool, not the quantizer:
distractors crowd a fixed pool and evict true neighbours before the exact rescore sees them.
Raise `rerank_candidates`.

Set `rerank_candidates: 0` to drop the full-precision vectors entirely. This is **not** merely a
"smallest index, at the cost of recall" trade — at high dimension it is one of the best points on
the whole frontier. On GIST (960-d, 100k), `Storage::SQ8` with `rerank_candidates: 0` gives
**98.40% recall@10 in 139 MB** — 3.2x smaller than `RaBitQ + rerank` (440 MB) at the same recall,
trading about 2.6x QPS. If memory is your binding constraint, start here.

> ### ⚠️ Pick your storage mode by dimension — the two swap places
>
> The right code depends on your dimension, and the two **trade places**:
>
> | | SIFT1M (128-d) | GIST1M (960-d) |
> |---|---|---|
> | `Storage::SQ8` | **1.20x hnswlib**, 1.30x faiss | **worthless** — 1.03x F32, and *more* memory |
> | `Storage::RaBitQ` | **~12x slower** than SQ8 | **1.79x hnswlib**, 1.43x faiss |
>
> Every quantized traversal trades ALU work for memory traffic, and the two codes sit on
> opposite sides of it:
>
> * **SQ8** must widen `u8`→`i32`→`f32` before it can compute — ~**3x the ALU per dimension** of
>   plain f32. What it buys, skipped DRAM round-trips, is roughly **fixed** per node visit. Fixed
>   benefit, cost linear in `dim`: **wins small, dies big.**
> * **RaBitQ** compares sign bits against a once-per-query rotated vector — **cheaper per
>   dimension than f32**, no widening. What it pays is a coarser estimate, costing extra graph
>   hops — and that penalty **shrinks sharply as `dim` grows**, because the code is 1 bit *per
>   dimension*: higher-dimensional vectors get proportionally **longer codes**. It gets cheaper
>   *and* more accurate as `dim` rises: **loses small, wins big.**
>
> It's a scissors, not one crossing line. One corpus, prefix-truncated so `dim` is the only
> variable (`--example dim_crossover`, GIST, n=200k, ef=100):
>
> | dim | 64 | 128 | 192 | 256 | 384 | 512 | 768 | 960 |
> |---|---|---|---|---|---|---|---|---|
> | RaBitQ recall@10 | 63.7% | 78.0% | 84.9% | 88.2% | 91.4% | 94.2% | 96.9% | 96.7% |
> | gap vs F32 | −35.8 | −20.6 | −13.4 | −10.0 | −6.4 | −3.8 | **−0.9** | **−0.9** |
> | RaBitQ speed (F32 ns/dist ÷ RaBitQ) | 1.28x | 1.55x | 1.58x | 1.74x | 1.88x | 2.01x | 1.98x | 1.89x |
>
> The accuracy penalty collapses ~40x across that range while the speed advantage grows. Both
> blades close.
>
> ### The rule of thumb needs TWO axes: dimension **and metric**
>
> | | `L2` | `Cosine` (what RAG uses) |
> |---|---|---|
> | **≤ 256-d** | `SQ8` | `SQ8` |
> | **384-d** (MiniLM) | **`SQ8`** — RaBitQ loses (0.72x) | **tie** — take `RaBitQ` for the smaller index |
> | **≥ 768-d** (OpenAI, GIST) | `RaBitQ` | **`RaBitQ`** (1.2–1.3x SQ8) |
>
> **Cosine moves the crossover down**, and this is not a curiosity — *cosine is what RAG actually
> uses*, and every crossover number this project published before now was measured under **L2**.
> At **matched recall**, 384-d (`--example dim_pareto gist1m 384 200000 [l2|cosine]`, n=200k):
>
> | recall@10 | metric | F32 | SQ8 | RaBitQ |
> |---|---|---|---|---|
> | ~97.7% | `L2` | 1,826 | **1,935** | ~1,398 — *loses* |
> | ~97.7% | `Cosine` | 1,636 | 1,897 | **~1,903** — *dead even* |
>
> **Why:** under cosine, RaBitQ encodes a **unit-normalized** copy of each vector. Normalization
> throws away magnitude — precisely the information a 1-bit sign code cannot represent, and
> precisely the information cosine does not care about. Under L2 magnitude *does* matter, so
> discarding it costs real accuracy. **RaBitQ is structurally suited to cosine.**
>
> At 384-d cosine, `RaBitQ` also ships a *smaller* index than `SQ8` (405 MB vs 471 MB — `SQ8 +
> rerank` is the largest of the three, since it keeps the f32 vectors *and* the codes). So on a
> tie, take RaBitQ.
>
> ### Do not read a config decision out of a fixed-`ef` benchmark
>
> At fixed `ef`, RaBitQ looks like the 384-d winner (3,257 QPS vs SQ8's 1,898). That is an
> artifact: it was 6.4 recall points behind, and it was fast **because it stopped looking**. Under
> L2 at matched recall it in fact *loses* there. **A mode that is fast because it stopped finding
> things is not fast.** This repo has published a wrong conclusion from a fixed-`ef` reading once
> already; always compare at matched recall.
>
> Reproduce: `cargo run --release -p foxstash-benches --example storage_pareto gist1m`. Mechanism
> and full tables in `benchmarks/RESULTS.md`.

### Memory (SIFT1M, 1M x 128d)

| configuration | index size | vs hnswlib |
|---|---|---|
| `Storage::SQ8`, `rerank_candidates: 0` | **564 MB** | **0.73x** |
| `Storage::F32` | 948 MB | 1.22x |
| `Storage::SQ8` + rerank (fastest) | 1,076 MB | 1.39x |
| hnswlib / faiss | ~776 MB | 1.00x |

Rerank needs the full-precision vectors, so the fastest configuration is also the largest.
Speed crown or memory crown — not both, yet.

### Product Quantization was removed in 1.0 — it was dominated

`PQHNSWIndex` compressed the vector payload **192x**. It is gone, and the measurement that
killed it is worth reading, because the number we published about it for a year was invalid.

The docs said PQ got **~55% recall@10**. That figure was produced with `rerank_candidates` at
its **default of 0** — the exact-rescoring stage *switched off*. Measured properly on GIST
(960-d, 100k, L2 — PQ's best case, since PQ was L2-only):

| | MB | recall@10 | QPS |
|---|---|---|---|
| `PQHNSWIndex`, no rerank | **18** | **23.07%** | 1,293 |
| `PQHNSWIndex`, rerank 100 | 402 | **62.27%** ← ceiling | 790 |
| `PQHNSWIndex`, rerank 400 | 402 | 60.97% ← *worse* | 446 |
| `Storage::RaBitQ` + rerank | 440 | **97.97%** | **1,970** |
| `Storage::SQ8`, no rerank | **139** | **98.40%** | 760 |

**62% is a ceiling, not a knob.** The graph is *traversed* on PQ codes, so the candidate pool
handed to the rescoring stage does not contain the true neighbours — and you cannot rerank your
way to items you never retrieved. Widening the pool made recall *fall*.

And the compression evaporates exactly when it becomes useful: reaching even 62% requires
retaining the f32 vectors (402 MB), at which point `Storage::RaBitQ` costs 440 MB and delivers
**98%**. PQ's only unique point was 18 MB at 23% recall — which is not a retrieval index.

The [`ProductQuantizer`] primitive is still there (`vector::product_quantize`). It is a fine
quantizer. It is just not a viable way to traverse a graph.

> **The pattern to take away**, because it has now cost this project four times: *a bad number
> produced by a disabled feature makes the feature look inherently bad, and then nobody
> re-measures it.* SQ8 was advertised at 71.4% recall (a metric bug — really 99.33%). PQ's
> reranking was a silent no-op for a release. RaBitQ was nearly deleted on 128-d evidence in a
> library whose users run 384–1536-d. And PQ was judged on a figure taken with its accuracy stage
> off. Check what a number was *measured with* before you let it decide anything.

### Streaming Batch Ingestion

For large datasets, use streaming batch ingestion with progress tracking:

```rust
use foxstash_core::index::{HNSWIndex, BatchBuilder, BatchConfig};

let mut index = HNSWIndex::with_defaults(384);

let config = BatchConfig::default()
    .with_batch_size(1000)
    .with_total(100_000)
    .with_progress(|progress| {
        println!(
            "Indexed {}/{} ({:.1}%) - {:.0} docs/sec",
            progress.completed,
            progress.total.unwrap_or(0),
            progress.percent().unwrap_or(0.0),
            progress.docs_per_sec
        );
    });

let mut builder = BatchBuilder::new(&mut index, config);

for doc in document_iterator {
    builder.add(doc)?;
}

let result = builder.finish();
println!("Indexed {} documents in {}ms", result.documents_indexed, result.elapsed_ms);
```

### Incremental Persistence (WAL)

Avoid rewriting the entire index on every update:

```rust
use foxstash_core::storage::{IncrementalStorage, IncrementalConfig, IndexMetadata};

let config = IncrementalConfig::default()
    .with_checkpoint_threshold(10_000)  // Full snapshot every 10K ops
    .with_wal_sync_interval(100);       // Sync to disk every 100 ops

let mut storage = IncrementalStorage::new("/tmp/my_index", config)?;

// Fast append-only writes to WAL
for doc in new_documents {
    storage.log_add(&doc)?;
    index.add(doc)?;
}

// Periodic checkpoint
if storage.needs_checkpoint() {
    storage.checkpoint(&index, IndexMetadata {
        document_count: index.len(),
        embedding_dim: 384,
        index_type: "hnsw".to_string(),
    })?;
}
```

### With ONNX Embeddings

Enable the `onnx` feature:

```toml
[dependencies]
foxstash-core = { version = "0.5", features = ["onnx"] }
```

```rust
use foxstash_core::embedding::OnnxEmbedder;

let mut embedder = OnnxEmbedder::new(
    "models/model.onnx",
    "models/tokenizer.json"
)?;

let embedding = embedder.embed("Foxes cache food for later retrieval")?;
assert_eq!(embedding.len(), 384);
```

## Database Layer (foxstash-db)

For production use, `foxstash-db` provides a high-level document store with named collections, metadata filtering, BM25 full-text search, and hybrid search built on top of `foxstash-core`.

```toml
[dependencies]
foxstash-db = "0.5"
```

### VectorStore and Collections

```rust
use foxstash_db::{VectorStore, DbConfig, Filter, HybridConfig, MergeStrategy};
use serde_json::json;

// Open a persistent store (recovers existing collections from disk)
let config = DbConfig::default().with_embedding_dim(384);
let store = VectorStore::open("/var/data/my_store", config)?;

// Get or create a collection
let col = store.get_or_create_collection("articles")?;

// Insert documents with optional metadata
col.insert(
    "doc1".to_string(),
    "Foxes are highly adaptable mammals found worldwide".to_string(),
    vec![0.1_f32; 384],  // embedding from your model
    Some(json!({ "category": "biology", "year": 2024 })),
)?;

col.insert(
    "doc2".to_string(),
    "Red foxes cache food in scattered locations for later retrieval".to_string(),
    vec![0.2_f32; 384],
    Some(json!({ "category": "behavior", "year": 2023 })),
)?;

// Upsert (insert or replace) a document
col.upsert(
    "doc1".to_string(),
    "Updated content about fox adaptability".to_string(),
    vec![0.1_f32; 384],
    Some(json!({ "category": "biology", "year": 2025 })),
)?;

// Vector similarity search
let query_embedding = vec![0.15_f32; 384];
let results = col.search(&query_embedding, 5, None)?;

// Vector search with metadata filter
let filter = Filter::eq("category", "biology");
let filtered = col.search(&query_embedding, 5, Some(&filter))?;

// BM25 full-text search
let text_results = col.search_text("fox cache food", 5, None)?;

// Hybrid search: combines vector + BM25 with Reciprocal Rank Fusion
let hybrid_results = col.search_hybrid(
    &query_embedding,
    "fox cache food",
    5,
    None,    // optional Filter
    None,    // optional HybridConfig (uses default if None)
)?;

// Look up a document by ID
if let Some(doc) = col.get("doc1")? {
    println!("Found: {}", doc.content);
}

// Delete a document
col.delete("doc2")?;

// Compact tombstoned entries
col.compact()?;

// Flush WAL to disk
col.flush()?;

// Flush all collections at once
store.flush_all()?;
```

### VectorStore API

| Method | Description |
|--------|-------------|
| `VectorStore::open(path, config)` | Open a store, recovering existing collections from disk |
| `get_or_create_collection(name)` | Return existing collection or create a new one |
| `create_collection(name)` | Create a new collection; error if it already exists |
| `get_collection(name)` | Get an existing collection; error if not found |
| `collections()` | List all collection names |
| `unload_collection(name)` | Remove from memory (files remain; can be re-opened) |
| `delete_collection(name)` | Permanently delete from memory and disk |
| `flush_all()` | Flush all collections to disk |

### Collection API

| Method | Description |
|--------|-------------|
| `insert(id, content, embedding, metadata)` | Insert a document; error on duplicate ID |
| `upsert(id, content, embedding, metadata)` | Insert or replace a document |
| `delete(id)` | Tombstone a document by ID |
| `get(id)` | Retrieve a document by ID |
| `search(query, k, filter)` | Vector similarity search with optional metadata filter |
| `search_batch(queries, k, filter)` | Parallel vector search for multiple queries via rayon |
| `search_text(query, k, filter)` | BM25 keyword search with optional metadata filter |
| `search_hybrid(query, text, k, filter, config)` | Hybrid vector + BM25 search |
| `flush()` | Flush WAL to disk |
| `compact()` | Remove tombstoned entries and rebuild index |

### Metadata Filtering

`Filter` supports dot-notation field access into JSON metadata:

```rust
use foxstash_db::Filter;
use serde_json::json;

// Equality
let f = Filter::eq("category", "biology");

// Inequality
let f = Filter::ne("status", "archived");

// Range comparisons
let f = Filter::gt("year", json!(2020));
let f = Filter::lte("score", json!(0.9));

// Set membership
let f = Filter::is_in("lang", vec![json!("en"), json!("fr")]);

// Field existence
let f = Filter::exists("tags.entity");

// Logical composition
let f = Filter::and(vec![
    Filter::eq("category", "biology"),
    Filter::gt("year", json!(2020)),
]);

let f = Filter::or(vec![
    Filter::eq("status", "active"),
    Filter::eq("status", "pending"),
]);

let f = Filter::not(Filter::eq("archived", true));
```

### Hybrid Search Configuration

```rust
use foxstash_db::{HybridConfig, MergeStrategy};

let config = HybridConfig::default()
    .with_weights(0.7, 0.3)               // vector_weight=0.7, keyword_weight=0.3
    .with_strategy(MergeStrategy::Rrf)    // Reciprocal Rank Fusion (default)
    .with_rrf_k(60.0);                    // RRF smoothing constant

// Alternatively, use WeightedSum with min-max normalized scores
let config = HybridConfig::default()
    .with_weights(0.6, 0.4)
    .with_strategy(MergeStrategy::WeightedSum);
```

| Field | Default | Description |
|-------|---------|-------------|
| `vector_weight` | `0.7` | Weight for vector similarity scores |
| `keyword_weight` | `0.3` | Weight for BM25 keyword scores |
| `merge_strategy` | `Rrf` | `Rrf` (rank-based) or `WeightedSum` (score-based) |
| `rrf_k` | `60.0` | RRF smoothing constant (only used with `Rrf`) |

## Index and Text Index Trait Abstractions

`foxstash-core` exposes `VectorIndex` and `VectorIndexSnapshot` traits that abstract over
concrete index types (HNSW, Flat, SQ8, Binary, PQ). The `foxstash-db` crate additionally
exports a `TextIndex` trait for BM25-backed keyword indexes. These traits make it straightforward
to swap implementations or build generic search pipelines without coupling to a specific type.

```rust
use foxstash_core::index::{VectorIndex, VectorIndexSnapshot};
use foxstash_db::TextIndex;

fn search_any<I: VectorIndex>(index: &I, query: &[f32], k: usize) {
    let results = index.search(query, k).unwrap();
    // ...
}
```

## Crates

| Crate | Description |
|-------|-------------|
| `foxstash-core` | Core library with indexes, embeddings, and storage |
| `foxstash-db` | Document storage, collections, hybrid search, BM25 |
| `foxstash-wasm` | WebAssembly bindings with IndexedDB persistence |
| `foxstash-native` | Native bindings with full ONNX support |

## Architecture

```
foxstash/
├── crates/
│   ├── core/           # Main library
│   │   ├── embedding/  # ONNX Runtime + caching
│   │   ├── index/      # HNSW (F32/SQ8/RaBitQ storage), Flat, PQ
│   │   ├── storage/    # File persistence, compression, WAL
│   │   └── vector/     # SIMD ops, quantization
│   ├── db/             # Database layer
│   │   ├── collection/ # Named collections with WAL
│   │   ├── filter/     # Metadata filtering
│   │   ├── hybrid/     # BM25 + vector hybrid search
│   │   └── store/      # VectorStore (multi-collection manager)
│   ├── wasm/           # Browser target
│   ├── native/         # Desktop/server target
│   └── benches/        # Comprehensive benchmarks
```

## Benchmarks

Measured on **real SIFT (128-d)** and **real GIST (960-d)**, against **hnswlib** and **faiss**,
at **matched recall**, single-threaded, on an idle machine. Competitors are shown at *their own*
Pareto frontier (best of M=16 / M=32), not at a fixed M that flatters us. Full methodology:
[`benchmarks/RESULTS.md`](benchmarks/RESULTS.md).

**Foxstash wins at both dimensions — but with a different storage mode at each:**

| | **SIFT1M (128-d)** | **GIST1M (960-d)** |
|---|---|---|
| use this mode | **`Storage::SQ8`** | **`Storage::RaBitQ`** |
| vs hnswlib | **1.20x** @ 99.5% recall | **1.79x** @ 98.3% recall |
| vs faiss | **1.30x** @ 99.5% recall | **1.43x** @ 98.3% recall |
| the *other* mode | RaBitQ: ~12x slower | SQ8: worthless (1.03x F32) |

The two quantizers **swap places** with dimension, and the reason is mechanical — see the note
under [8-bit traversal](#faster-search-with-8-bit-traversal-storagesq8). Real embeddings are
384-d (MiniLM) to 1536-d (OpenAI), so **reach for `RaBitQ` first**; SIFT's 128-d is the best case
for `SQ8`, not a typical one.

### The honest costs

- **We build slowly at high dimension.** GIST1M: ~1,141 s against faiss's 294–408 s — **faiss
  builds 3–4x faster than us**. Our "2.1x faster builds than hnswlib" is a 128-d result and it
  inverts at 960-d. Not yet explained.
- **Full precision is unremarkable.** `Storage::F32` is 0.88x hnswlib at 128-d, and at 960-d sits
  *between* the two (1.10–1.20x hnswlib, 0.90–0.96x faiss). Every win comes from the quantized
  traversal, not from a better graph or a faster kernel.
- **Memory.** At 128-d the fast config costs 1,076 MB against their ~776 MB; `rerank_candidates:
  0` drops it to **564 MB (0.73x hnswlib)** at a ~98.9% recall ceiling. Pick one.

### SIFT1M — 1,000,000 x 128d, k=10, M=32, single-threaded

QPS at **matched recall** (competitor QPS interpolated along its own curve to Foxstash's recall):

| recall@10 | Foxstash SQ8 | hnswlib | vs hnswlib | faiss | vs faiss |
|-----------|--------------|---------|------------|-------|----------|
| 93.0% | **13,107** | 11,823 | **1.11x** | 10,961 | **1.19x** |
| 98.2% | **7,183** | 6,249 | **1.15x** | 5,673 | **1.27x** |
| 99.5% | **4,254** | 3,537 | **1.20x** | 3,282 | **1.30x** |
| 99.9% | **2,549** | 1,911 | **1.33x** | 1,895 | **1.34x** |

| | Foxstash SQ8 | Foxstash F32 | hnswlib | faiss |
|---|---|---|---|---|
| build, all cores | 167 s | **167 s** | 342 s | 78 s |
| index size | 1,076 MB | 948 MB | ~776 MB | ~776 MB |
| index size, codes only | **564 MB** | — | — | — |

### GIST1M — 1,000,000 x 960d, k=10, single-threaded

The dimension real embeddings actually live at. Competitors at **their own** Pareto frontier
(best of M=16 / M=32 at each recall). Here `Storage::SQ8` is worthless and **`Storage::RaBitQ`
is the win**:

| recall@10 | Foxstash RaBitQ | hnswlib | vs hnswlib | faiss | vs faiss |
|-----------|-----------------|---------|------------|-------|----------|
| 90.6% | **1,487** | 986 | **1.51x** | 1,197 | **1.24x** |
| 93.8% | **1,136** | 719 | **1.58x** | 877 | **1.30x** |
| 96.4% | **842** | 477 | **1.76x** | 591 | **1.43x** |
| 98.3% | **536** | 299 | **1.79x** | 376 | **1.43x** |

| | Foxstash RaBitQ | Foxstash SQ8 | Foxstash F32 | hnswlib | faiss |
|---|---|---|---|---|---|
| ns per distance | **138–157** | 250–269 | 258–277 | — | — |
| build, all cores | 1,209 s | 1,145 s | 1,141 s | 1,059–1,433 s | **294–408 s** |
| index size | 4,404 MB | 5,236 MB | 4,276 MB | — | — |

Two things to read off that: **SQ8 costs more memory than F32 here and buys no speed** (its
block shrank 3.3x and ns/dist did not move — the dequant tax scales with `dim` while the latency
it saves does not). And **faiss builds 3–4x faster than us at 960-d** — a real weakness, and the
inverse of the 128-d picture.

Why it works: HNSW search is **memory-latency bound** — one distance computation costs ~85 ns,
about a DRAM round-trip. Foxstash already computed distances *faster* than faiss and still lost,
because each one waited on memory. `Storage::SQ8` puts 8-bit codes in the hot node block (400
bytes instead of 784) and keeps the f32 vectors in a cold array touched only when rescoring the
final candidates. The graph is still built with exact distances, so recall and the distance
count are unchanged — only `ns/dist` moves, 87 -> 67.

### Reproduce

```bash
benchmarks/fetch-data.sh        # canonical TEXMEX SIFT; verifies an exact-L2 control per corpus
benchmarks/run-scoreboard.sh    # Foxstash vs hnswlib vs faiss, serialized behind an idle gate
```

### Four ways to get this wrong — every one of which this project got wrong

1. **Compare at matched recall, not matched `ef`.** `ef` is a knob, and implementations reach
   a given recall at different settings of it. A fixed-`ef` table measures nothing.
2. **Count the threads.** hnswlib's `knn_query` defaults to `num_threads=-1` — *every core*.
   Timing that against a single-threaded loop produced an "11x slower" claim here that was
   pure artifact.
3. **Never run two benchmarks at once.** A concurrent build halved hnswlib's apparent QPS on
   this machine.
4. **Never benchmark on synthetic vectors.** Random vectors have no cluster structure, so
   *every* ANN collapses to ~60% recall on them regardless of quality. The table that used to
   sit here reported hnswlib at **39.5% recall** — on real SIFT it scores ~99%. That table
   flattered Foxstash, and it hid a quantizer bug (1.2% real recall) for an entire release.

And do not compare recall across datasets: on SIFT10K the 100th neighbour is only 4.7%
further from the query than the 10th, so the true top-10 hides in a near-tie shell. Every
index scores far better on SIFT100K because the task is easier, not because it improved.

## Roadmap

- [x] SQ8 8-bit traversal (`Storage::SQ8`) — beats hnswlib 1.20x at 99.5% recall on SIFT1M
- [x] Streaming add/search for large datasets
- [x] Incremental persistence (WAL + checkpointing)
- [x] ~~Product quantization (PQ)~~ — **removed in 1.0**: dominated. ~62% recall ceiling, and no memory win once the rerank stage it needs is enabled. See above.
- [x] Diversity-aware neighbor selection (Algorithm 4)
- [x] Hybrid search (BM25 + vector, RRF and WeightedSum)
- [x] VectorIndex / TextIndex trait abstractions
- [ ] Constrained graph traversal for efficient pre-filtering
- [ ] Cache-locality optimizations for quantized indices (flattened L0 cache)
- [ ] High-concurrency scaling (sharded-lock or lock-free index updates)
- [ ] GPU acceleration (optional)
- [ ] Multi-vector support (late interaction)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Credits

Built by [Narcoleptic Fox](https://narcolepticfox.com)
