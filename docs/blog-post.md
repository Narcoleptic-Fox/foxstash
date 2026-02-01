# Foxstash: 192x Vector Compression in Pure Rust

We just shipped [Foxstash](https://github.com/Narcoleptic-Fox/foxstash), a local-first RAG library for Rust. It runs entirely on your machine — no API keys, no cloud dependencies, no data leaving your control.

The headline feature: **192x vector compression** using Product Quantization. A million 384-dimensional vectors drops from 1.5GB to 8MB. That's not a typo.

## Why We Built This

Most vector databases assume you'll send your data somewhere else. Pinecone, Weaviate, Qdrant — all great, but they're cloud-first. We wanted something different:

- **Local-first.** Your embeddings stay on your machine.
- **No runtime dependencies.** Pure Rust, compiles to a single binary.
- **Browser-ready.** Ships as WASM for client-side search.

The fox caches food for later retrieval. So does Foxstash.

## The Compression Story

Vector search has a memory problem. Each 384-dim float32 vector is 1.5KB. A million documents? That's 1.5GB just for embeddings.

We implemented three levels of quantization:

### Scalar Quantization (SQ8) — 4x compression
Convert float32 to int8. Simple, fast, ~95% recall.

```rust
let index = SQ8HNSWIndex::for_normalized(384, config);
```

### Binary Quantization — 32x compression  
One bit per dimension. Use with two-phase reranking for ~90% recall.

```rust
let index = BinaryHNSWIndex::with_full_precision(384, config);
let results = index.search_and_rerank(&query, 100, 10)?;
```

### Product Quantization (PQ) — 192x compression
Divide vectors into subvectors, cluster each independently. 8 bytes per vector instead of 1,536.

```rust
let pq_config = PQConfig::new(384, 8, 8);
let index = PQHNSWIndex::train(pq_config, &training_data, config)?;
```

The tradeoff is recall vs memory. Pick your poison:

| Method | Compression | Recall | Use Case |
|--------|-------------|--------|----------|
| Full (f32) | 1x | ~98% | Default |
| SQ8 | 4x | ~95% | Memory constrained |
| Binary | 32x | ~90%* | Large datasets |
| PQ | 192x | ~80%** | Massive scale |

*With reranking. **Using ADC search.

## SIMD: The Boring Performance Win

We spent time on the unglamorous work: SIMD intrinsics for distance computation.

```rust
// Auto-selects best available: AVX2 > SSE4.1 > scalar
let similarity = cosine_similarity_auto(&a, &b);
```

The payoff: 3-4x speedup on vector operations. For a library that's 90% dot products, that matters.

AVX2 on x86, NEON on ARM, automatic fallback when neither is available.

## HNSW: Not Just Another Implementation

HNSW (Hierarchical Navigable Small World) is the standard for approximate nearest neighbor search. We implemented Algorithm 4 from the original paper — the diversity heuristic that most implementations skip.

The difference: instead of just picking the M closest neighbors, we ensure diversity. A candidate only gets selected if it's closer to the query than to any already-selected neighbor. This prevents clustering and improves graph connectivity.

```rust
let config = HNSWConfig::default()
    .with_m(16)
    .with_ef_search(50);

let mut index = HNSWIndex::new(384, config);
```

## Runs in Your Browser

The entire library compiles to WebAssembly. Same code, same performance characteristics, runs client-side.

```bash
wasm-pack build crates/wasm --target web
```

Use cases:
- Offline-capable search in PWAs
- Privacy-sensitive applications
- Edge deployments without server round-trips

[Try the demo →](https://foxstash-demo.vercel.app)

## What's Next

Foxstash 0.1 is the foundation. On the roadmap:

- **Python bindings** (PyO3) — most ML work happens in Python
- **GPU acceleration** — optional CUDA/Metal for large-scale
- **Hybrid search** — sparse + dense vectors together
- **Multi-vector support** — late interaction patterns like ColBERT

## Get Started

```toml
[dependencies]
foxstash-core = "0.1"
```

```rust
use foxstash_core::index::HNSWIndex;
use foxstash_core::Document;

let mut index = HNSWIndex::with_defaults(384);

let doc = Document {
    id: "doc1".to_string(),
    content: "The quick brown fox".to_string(),
    embedding: embed(&text), // Your embedding function
    metadata: None,
};

index.add(doc)?;
let results = index.search(&query_embedding, 10)?;
```

Links:
- [GitHub](https://github.com/Narcoleptic-Fox/foxstash)
- [crates.io](https://crates.io/crates/foxstash-core)
- [docs.rs](https://docs.rs/foxstash-core)
- [Demo](https://foxstash-demo.vercel.app)

---

*Built by [Narcoleptic Fox](https://narcolepticfox.com). The fox looks lazy until it moves. Then it's fast.*
