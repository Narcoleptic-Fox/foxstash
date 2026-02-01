# 🦊 Foxstash

**High-performance local RAG library for Rust**

[![Crates.io](https://img.shields.io/crates/v/foxstash-core.svg)](https://crates.io/crates/foxstash-core)
[![Documentation](https://docs.rs/foxstash-core/badge.svg)](https://docs.rs/foxstash-core)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Foxstash is a local-first Retrieval-Augmented Generation (RAG) library featuring SIMD-accelerated vector operations, HNSW indexing, ONNX embeddings, and WebAssembly support.

## Features

- 🚀 **SIMD-Accelerated** - AVX2/SSE/NEON vector operations with 3-4x speedup
- 📊 **HNSW Indexing** - Hierarchical Navigable Small World graphs for fast similarity search
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
| Embedding generation (MiniLM) | 500 docs/sec |

## Architecture

```
foxstash/
├── crates/
│   ├── core/           # Main library
│   │   ├── embedding/  # ONNX Runtime + caching
│   │   ├── index/      # HNSW + Flat indexes
│   │   ├── storage/    # File persistence + compression
│   │   └── vector/     # SIMD-accelerated operations
│   ├── wasm/           # Browser target
│   ├── native/         # Desktop/server target
│   └── benches/        # Comprehensive benchmarks
```

## Roadmap

- [ ] Int8/Binary quantization (4-8x memory reduction)
- [ ] Streaming add/search for large datasets
- [ ] Incremental persistence (delta updates)
- [ ] Product quantization (PQ)
- [ ] GPU acceleration (optional)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Credits

Built by [Narcoleptic Fox](https://narcolepticfox.com) 🦊

---

*Why "Foxstash"? Foxes are famous for caching food and retrieving it later - just like a RAG system stores and retrieves knowledge.*
