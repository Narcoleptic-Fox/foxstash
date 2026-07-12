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
- **Vector Quantization** - Int8 (4x), Binary (32x), and Product Quantization (192x)
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

### Memory-Efficient Indexing with Quantization

For large datasets, use quantized indexes to reduce memory by 4-192x:

```rust
use foxstash_core::index::{SQ8HNSWIndex, BinaryHNSWIndex, QuantizedHNSWConfig};
use foxstash_core::Document;

// Scalar Quantization (4x compression, 71.4% recall on SIFT10K)
let mut sq8_index = SQ8HNSWIndex::for_normalized(384, QuantizedHNSWConfig::default());

// Binary Quantization (32x compression, use with reranking)
let mut binary_index = BinaryHNSWIndex::with_full_precision(384, QuantizedHNSWConfig::default());

// Add documents
let doc = Document {
    id: "doc1".to_string(),
    content: "Foxes cache food for retrieval".to_string(),
    embedding: vec![0.1; 384],
    metadata: None,
};
sq8_index.add(doc.clone())?;
binary_index.add_with_full_precision(doc)?;

// Search with SQ8 (high quality, 4x memory savings)
let results = sq8_index.search(&query, 10)?;

// Two-phase search with Binary (fast filter, then precise rerank)
let results = binary_index.search_and_rerank(&query, 100, 10)?;
```

### Product Quantization (Extreme Compression)

For massive datasets, use Product Quantization for up to 192x compression:

```rust
use foxstash_core::index::{PQHNSWIndex, PQHNSWConfig};
use foxstash_core::vector::product_quantize::PQConfig;

// Configure PQ: 8 subvectors, 256 centroids each
let pq_config = PQConfig::new(384, 8, 8)
    .with_kmeans_iterations(20);

// Train on sample vectors
let training_data = load_sample_vectors(10_000);
let mut index = PQHNSWIndex::train(pq_config, &training_data, PQHNSWConfig::default())?;

// Add documents (automatically compressed)
for doc in documents {
    index.add(doc)?;
}

// Search using Asymmetric Distance Computation (ADC)
let results = index.search(&query, 10)?;
```

### Memory Comparison (1M vectors, 384 dimensions)

| Index Type | Memory | Compression | Recall |
|------------|--------|-------------|--------|
| HNSW (f32) | 1.5 GB | 1x | ~98% |
| SQ8 HNSW | 384 MB | 4x | ~95% |
| Binary HNSW | 48 MB | 32x | ~90%* |
| PQ HNSW (M=8) | 8 MB | 192x | ~80%** |

*With two-phase reranking. **Using ADC search.

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
| `create_search_context()` | Allocate a reusable `SearchContext` for tight query loops |
| `search_with_context(query, k, ctx)` | Vector search reusing a caller-managed context |
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
│   │   ├── index/      # HNSW, Flat, SQ8, Binary, PQ indexes
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

Measured on **real SIFT**, against **hnswlib** and **faiss**, at **matched recall**,
single-threaded. Full methodology and the 10K/100K tables:
[`benchmarks/RESULTS.md`](benchmarks/RESULTS.md).

**Where Foxstash actually stands: it does not beat hnswlib.** At 1M vectors it serves
~0.85–0.91x hnswlib's QPS at equal recall, and sits at parity with faiss. It does build 2.1x
faster than hnswlib, and it reaches any given recall at a lower `ef` than either of them.

### SIFT1M — 1,000,000 x 128d, k=10, M=32, single-threaded

QPS at **matched recall** (competitor QPS interpolated along its own curve to Foxstash's recall):

| recall@10 | Foxstash | hnswlib | vs hnswlib | faiss | vs faiss |
|-----------|----------|---------|------------|-------|----------|
| 93.0% | 10,044 | 11,787 | 0.85x | 10,961 | 0.91x |
| 98.2% | 5,455 | 6,352 | 0.88x | 5,749 | 0.97x |
| 99.5% | 3,267 | 3,636 | 0.91x | 3,384 | 0.98x |

| | Foxstash | hnswlib | faiss |
|---|---|---|---|
| build, all cores | **167 s** | 342 s | 78 s |
| index size | 947 MB | ~776 MB | ~776 MB |

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

- [x] Int8/Binary quantization (4-32x memory reduction)
- [x] Streaming add/search for large datasets
- [x] Incremental persistence (WAL + checkpointing)
- [x] Product quantization (PQ) - up to 192x compression
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
