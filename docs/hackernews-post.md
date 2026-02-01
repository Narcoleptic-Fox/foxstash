# Hacker News Post

**Title:** Show HN: Foxstash – Local-first vector search in Rust with 192x compression

**URL:** https://github.com/Narcoleptic-Fox/foxstash

**First comment (post immediately after submitting):**

Hi HN! I built Foxstash because I wanted vector search without cloud dependencies.

The interesting bits:

**192x compression via Product Quantization.** A million 384-dim vectors goes from 1.5GB to 8MB. The trick is dividing vectors into subvectors and clustering each independently. You lose some recall (~80% vs ~98%) but gain massive memory savings.

**Runs in browser.** The whole thing compiles to WASM. Same Rust code, same HNSW index, runs client-side. Useful for offline-capable apps or when you can't send data to a server.

**SIMD everywhere.** AVX2 on x86, NEON on ARM, automatic fallback. Distance computation is 3-4x faster than naive loops.

**Algorithm 4 from the HNSW paper.** Most implementations skip the diversity heuristic for neighbor selection. We didn't. Better graph connectivity, better recall.

Trade-offs I made:
- No GPU support yet (on roadmap)
- ONNX embeddings optional (pulls in ort dependency)
- Single-threaded indexing (parallel queries work fine)

Demo: [link to vercel demo]
Docs: https://docs.rs/foxstash-core

Would love feedback, especially from folks who've built similar systems.
