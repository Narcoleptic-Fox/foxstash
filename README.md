# 🦊 Foxstash

**High-performance local RAG library for Rust**

[![Crates.io](https://img.shields.io/crates/v/foxstash-core.svg)](https://crates.io/crates/foxstash-core)
[![Documentation](https://docs.rs/foxstash-core/badge.svg)](https://docs.rs/foxstash-core)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Foxstash is a local-first Retrieval-Augmented Generation (RAG) library featuring SIMD-accelerated vector operations, HNSW indexing, vector quantization, ONNX embeddings, and WebAssembly support.

## Features

- 🚀 **SIMD-Accelerated** - AVX2/SSE/NEON vector operations with 3-4x speedup
- 📊 **HNSW Indexing** - Hierarchical Navigable Small World graphs for fast similarity search
- 📉 **Vector Quantization** - Int8 (4x) and Binary (32x) quantization for memory efficiency
- 🧠 **ONNX Embeddings** - Generate embeddings locally with MiniLM-L6-v2 or any ONNX model
- 🌐 **WASM Support** - Run in the browser with IndexedDB persistence
- 🗜️ **Compression** - Gzip, LZ4, and Zstd support for efficient storage
- 🔒 **Local-First** - Your data never leaves your machine

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
foxstash-core = "0.1"
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

For large datasets, use quantized indexes to reduce memory by 4-32x:

```rust
use foxstash_core::index::{SQ8HNSWIndex, BinaryHNSWIndex, QuantizedHNSWConfig};
use foxstash_core::Document;

// Scalar Quantization (4x compression, ~95% recall)
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

// Two-phase search with Binary (fast filter → precise rerank)
let results = binary_index.search_and_rerank(&query, 100, 10)?;
```

#### Memory Comparison (1M vectors × 384 dims)

| Index Type | Memory | Recall | Use Case |
|------------|--------|--------|----------|
| HNSW (f32) | 1.5 GB | ~98% | Default choice |
| SQ8 HNSW | 384 MB | ~95% | Memory constrained |
| Binary HNSW | 48 MB | ~90%* | Massive datasets |

*Binary recall improves significantly with two-phase search (filter + rerank).

### Streaming Batch Ingestion

For large datasets, use streaming batch ingestion with progress tracking:

```rust
use foxstash_core::index::{HNSWIndex, BatchBuilder, BatchConfig};
use foxstash_core::Document;

let mut index = HNSWIndex::with_defaults(384);

// Configure batch processing with progress callback
let config = BatchConfig::default()
    .with_batch_size(1000)
    .with_total(100_000)
    .with_progress(|progress| {
        println!(
            "Indexed {}/{} ({:.1}%) - {:.0} docs/sec, ETA: {}s",
            progress.completed,
            progress.total.unwrap_or(0),
            progress.percent().unwrap_or(0.0),
            progress.docs_per_sec,
            progress.eta_ms().unwrap_or(0) / 1000
        );
    });

let mut builder = BatchBuilder::new(&mut index, config);

// Stream documents from any source
for doc in document_iterator {
    builder.add(doc)?;
}

let result = builder.finish();
println!("Indexed {} documents in {}ms", result.documents_indexed, result.elapsed_ms);
```

### Filtered Search with Pagination

```rust
use foxstash_core::index::{FilteredSearchBuilder, SearchPage};

// Build filtered search
let results = index.search(&query, 100)?;

let filtered = FilteredSearchBuilder::new()
    .min_score(0.7)
    .has_metadata_field("category")
    .metadata_equals("type", serde_json::json!("article"))
    .max_results(50)
    .apply(results);

// Paginate results
let page = SearchPage::from_results(filtered, 0, 10);
println!("Page {}/{}, {} results", page.page + 1, page.total_pages, page.results.len());
```

### With ONNX Embeddings

Enable the `onnx` feature:

```toml
[dependencies]
foxstash-core = { version = "0.1", features = ["onnx"] }
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

## Crates

| Crate | Description |
|-------|-------------|
| `foxstash-core` | Core library with indexes, embeddings, and storage |
| `foxstash-wasm` | WebAssembly bindings with IndexedDB persistence |
| `foxstash-native` | Native bindings with full ONNX support |

## Performance

Benchmarked on Intel i7-12700K:

| Operation | Throughput |
|-----------|------------|
| Cosine similarity (384-dim, SIMD) | 15M ops/sec |
| HNSW search (10K docs, k=10) | 50K queries/sec |
| HNSW insert | 8K docs/sec |
| SQ8 distance (quantized) | 40M ops/sec |
| Binary Hamming distance | 100M ops/sec |
| Embedding generation (MiniLM) | 500 docs/sec |

## Architecture

```
foxstash/
├── crates/
│   ├── core/           # Main library
│   │   ├── embedding/  # ONNX Runtime + caching
│   │   ├── index/      # HNSW + Flat + Quantized indexes
│   │   ├── storage/    # File persistence + compression
│   │   └── vector/     # SIMD ops + quantization
│   ├── wasm/           # Browser target
│   ├── native/         # Desktop/server target
│   └── benches/        # Comprehensive benchmarks
```

## Roadmap

- [x] Int8/Binary quantization (4-32x memory reduction)
- [x] Streaming add/search for large datasets
- [ ] Incremental persistence (delta updates)
- [ ] Product quantization (PQ)
- [ ] GPU acceleration (optional)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Credits

Built by [Narcoleptic Fox](https://narcolepticfox.com) 🦊

---

*Why "Foxstash"? Foxes are famous for caching food and retrieving it later - just like a RAG system stores and retrieves knowledge.*
