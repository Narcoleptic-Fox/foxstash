# Foxstash

**High-performance local RAG library for Rust**

[![Crates.io](https://img.shields.io/crates/v/foxstash-core.svg)](https://crates.io/crates/foxstash-core)
[![Documentation](https://docs.rs/foxstash-core/badge.svg)](https://docs.rs/foxstash-core)
[![CI](https://github.com/Narcoleptic-Fox/foxstash/actions/workflows/ci.yml/badge.svg)](https://github.com/Narcoleptic-Fox/foxstash/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Foxstash is a local-first Retrieval-Augmented Generation (RAG) library featuring SIMD-accelerated vector operations, HNSW indexing, vector quantization, ONNX embeddings, and WebAssembly support.

## Features

- **SIMD-Accelerated** - AVX2/SSE/NEON vector operations with 3-4x speedup
- **HNSW Indexing** - Hierarchical Navigable Small World graphs for fast similarity search
- **Vector Quantization** - Int8 (4x), Binary (32x), and Product Quantization (192x)
- **ONNX Embeddings** - Generate embeddings locally with MiniLM-L6-v2 or any ONNX model
- **WASM Support** - Run in the browser with IndexedDB persistence
- **Compression** - Gzip, LZ4, and Zstd support for efficient storage
- **Incremental Persistence** - Write-ahead log for fast updates without full rewrites
- **Local-First** - Your data never leaves your machine

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

For large datasets, use quantized indexes to reduce memory by 4-192x:

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

## Architecture

```
foxstash/
├── crates/
│   ├── core/           # Main library
│   │   ├── embedding/  # ONNX Runtime + caching
│   │   ├── index/      # HNSW, Flat, SQ8, Binary, PQ indexes
│   │   ├── storage/    # File persistence, compression, WAL
│   │   └── vector/     # SIMD ops, quantization
│   ├── wasm/           # Browser target
│   ├── native/         # Desktop/server target
│   └── benches/        # Comprehensive benchmarks
```

## Benchmarks

Run benchmarks with:

```bash
cargo bench -p foxstash-benches
```

See `crates/benches/` for benchmark implementations.

## Roadmap

- [x] Int8/Binary quantization (4-32x memory reduction)
- [x] Streaming add/search for large datasets
- [x] Incremental persistence (WAL + checkpointing)
- [x] Product quantization (PQ) - up to 192x compression
- [x] Diversity-aware neighbor selection (Algorithm 4)
- [ ] GPU acceleration (optional)
- [ ] Hybrid search (sparse + dense vectors)
- [ ] Multi-vector support (late interaction)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Credits

Built by [Narcoleptic Fox](https://narcolepticfox.com)
