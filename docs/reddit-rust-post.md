# r/rust Post

**Title:** Foxstash: Local-first RAG library with SIMD, HNSW, and 192x compression via Product Quantization

**Body:**

Hey r/rust! Just released Foxstash, a vector search library for local RAG applications.

**What it does:**
- HNSW indexing for approximate nearest neighbor search
- SIMD-accelerated distance computation (AVX2/SSE/NEON)
- Vector quantization: SQ8 (4x), Binary (32x), Product Quantization (192x)
- Compiles to WASM for browser deployment
- No runtime dependencies beyond std

**Why another vector library?**

Most vector DBs are cloud-first. We wanted something that:
1. Runs entirely local — no API keys, no data exfiltration
2. Compiles to WASM — same code runs in browser
3. Handles memory constraints — 192x compression for large datasets

**Quick example:**

```rust
use foxstash_core::index::HNSWIndex;
use foxstash_core::Document;

let mut index = HNSWIndex::with_defaults(384);
index.add(Document {
    id: "doc1".to_string(),
    content: "Text content".to_string(),
    embedding: vec![0.1; 384],
    metadata: None,
})?;

let results = index.search(&query, 10)?;
```

**Stats:**
- 190 unit tests, 48 doctests
- 92% code coverage
- CI on Linux/macOS/Windows

Links:
- GitHub: https://github.com/Narcoleptic-Fox/foxstash
- crates.io: https://crates.io/crates/foxstash-core  
- docs.rs: https://docs.rs/foxstash-core

Feedback welcome! Particularly interested in thoughts on the SIMD implementation (uses `pulp` crate) and whether the quantization API feels ergonomic.
