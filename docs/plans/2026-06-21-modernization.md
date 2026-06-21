# Foxstash Modernization Plan (v0.6 → v1.0)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement
> the remaining phases task-by-task. Phases 0–1 are already done (see Status); start
> at Phase 2.

**Goal:** Modernize foxstash's index and quantization layers based on a deep-research
pass (2026-06-21) over the 2024–2026 ANN/RAG literature, and fix a recall bug that
research surfaced. Move recall and compression-vs-quality to the current state of the
art while keeping the local-first / WASM positioning.

**Architecture:** Incremental, each phase commits independently with tests passing
before moving on. Quantization advances layer onto the existing `Quantizer` trait and
two-phase rerank; index advances stay within the custom HNSW. No new heavy deps.

**Tech Stack:** Rust, `parking_lot` (RwLock), `pulp` (SIMD), `rand` 0.10 (`RngExt`),
`serde`, `ort` (ONNX). Workspace crates: `core`, `db`, `wasm`, `native`, `benches`.

---

## Background: the research + the recall diagnosis

A deep-research pass (5 angles, 26 sources, adversarial verification) produced these
**verified** conclusions; several roadmap fronts were *not* confirmed and are flagged.

- **Extended RaBitQ** (Gao & Long, SIGMOD 2025, arXiv:2409.09913) — asymptotically
  optimal quantizer, configurable 1–9 bits, beats PQ/OPQ/SQ at equal memory; 4/5/7-bit
  reach ~90/95/99% recall. **The flagship quantization win.** Reuses our two-phase rerank.
- **RaBitQ rescoring** recovers recall after aggressive compression (1-bit ~76% → ~94.7%
  with an SQ8 refine pass; ~3× QPS in Milvus 2.6). Maps to our existing rerank machinery.
- **HNSW diversity neighbor-selection heuristic** (Malkov & Yashunin, Alg. 4) is the
  standard fix for low recall on clustered data — and was the literal cause of our bug.
- **TurboQuant** (arXiv:2504.19874) — data-oblivious, zero-training; attractive for
  WASM streaming, but its "beats PQ recall" claim was *adversarially refuted*. Secondary.
- **SAQ** (arXiv:2509.12086, Sept 2025) — claims gains over Extended RaBitQ but is an
  unreviewed preprint with cherry-picked maxima. **Monitor only.**
- **Not verified** (research rate-limited, needs a fresh pass before roadmap inclusion):
  newer embedding models, ColBERT/multi-vector, Matryoshka, SPLADE, CAGRA/cuVS, DiskANN,
  ACORN. Treat as unproven, not as rejected.

### The recall story (why the README's "61%" was misleading)

The repo's benchmarks generated **uniform-random** vectors. In high dimensions those are
all near-equidistant (curse of dimensionality), so *every* ANN scores ~60% recall@10 —
the number measured the data, not the index. A realistic (clustered) benchmark exposed a
real bug: `BuildStrategy::Parallel` used `select_simple()` (raw M-nearest) and naive
reverse-insert, **ignoring `use_heuristic`**, so on clustered data nodes connected only
within their own cluster and search couldn't cross cluster boundaries.

> **Lesson (do not regress):** never benchmark ANN recall on uniform-random vectors.
> Use clustered or real-embedding data. The canonical bench is now
> `crates/benches/examples/realistic_comparison.rs`.

---

## Status

| Phase | Work | State | Branch / commit |
|-------|------|-------|-----------------|
| 0 | Realistic benchmark + recall diagnosis | ✅ done | `fix/parallel-build-recall` |
| 1 | Fix parallel-build heuristic | ✅ done | `fix/parallel-build-recall` `4133da9` |
| 1b | RaBitQ 1-bit quantizer (standalone) | ✅ done | `feat/rabitq-quantizer` `77c5b12` |
| 2 | Wire RaBitQ into an HNSW index | ⏳ next | — |
| 3 | Extended (B-bit) RaBitQ | ⏳ | — |
| 4 | Quick wins: ef_search default + embedding swap | ⏳ | — |
| 5 | Larger bets (filtered traversal, concurrency) | 🔭 future | — |

**Measured results so far** (20k vectors, 128d, clustered, recall@10):
- Parallel build: **35.6% → 73.5%** (≈ Sequential's ~78%), still ~9× faster than Sequential.
  ef_search now scales recall monotonically (44%→87.5% over ef 50→400); previously flat.
- RaBitQ vs Binary at 32× compression (two-phase rerank): **55.6% vs 28.2% (+27.3 pts)**.

---

## Phase 2: Integrate RaBitQ into an HNSW index

Make the standalone `RaBitQuantizer` usable in real search via a quantized HNSW index
with two-phase rerank, mirroring the existing `SQ8HNSWIndex` / `BinaryHNSWIndex`.

### Task 2.1: Add `RaBitQHNSWIndex`

Mirror the existing quantized index types: store full-precision vectors for rerank, keep
the HNSW graph over RaBitQ estimates, expose `search` and `search_and_rerank`.

**Files:**
- Modify: `crates/core/src/index/hnsw_quantized.rs` (add `RaBitQHNSWIndex` next to
  `SQ8HNSWIndex` / `BinaryHNSWIndex`; reuse `QuantizedHNSWConfig`)
- Modify: `crates/core/src/index/mod.rs` (re-export the new type)
- Reference: `crates/core/src/vector/rabitq.rs` (`prepare_query` / `estimate_dist_sq`)

**Step 1:** Write a test asserting `search_and_rerank` recall@10 > 0.85 on clustered data
(reuse the generator from `quantizer_comparison.rs`).
**Step 2:** Implement the index using `RaBitCode` for graph distances and full vectors
for the rerank stage.
**Step 3:** Run the recall test + full `cargo test -p foxstash-core`.

### Task 2.2: Document + example

**Files:**
- Modify: `README.md` (add RaBitQ to the quantization section + memory/recall table)
- Add: `crates/benches/examples/` usage example (or extend `quantizer_comparison.rs`)

---

## Phase 3: Extended (B-bit) RaBitQ

Generalize the 1-bit sign code to configurable B-bit per dimension (the 4/5/7-bit
operating points → ~90/95/99% recall). Builds on the same rotation + estimator.

### Task 3.1: B-bit encoder/estimator

**Files:**
- Modify: `crates/core/src/vector/rabitq.rs` (add `bits: u8` config; per-dim B-bit
  scalar quantization of the rotated residual + generalized unbiased estimator)

**Step 1:** Test that recall increases monotonically with B (1→4→7) at fixed pool size.
**Step 2:** Implement B-bit packing + estimator; keep the 1-bit path as `bits = 1`.
**Step 3:** Extend `quantizer_comparison.rs` to sweep B and print the recall/memory curve.

---

## Phase 4: Quick wins

### Task 4.1: Raise the default `ef_search`

Now that the graph is correctly connected, recall scales with `ef_search`; the current
default (100) is low for structured data. Benchmark and pick a better default.

**Files:**
- Modify: `crates/core/src/index/hnsw.rs` (`HNSWConfig::default`, ~line 224)

**Step 1:** Sweep ef ∈ {100, 150, 200, 300} on `realistic_comparison.rs`; choose the
recall/QPS knee. **Step 2:** Update the default + doc comment. **Step 3:** Full tests.

### Task 4.2: Replace the canonical benchmark + README numbers

The README's headline table uses the misleading uniform-random `quick_comparison.rs`.

**Files:**
- Modify: `README.md` (replace recall table with `realistic_comparison.rs` numbers;
  remove/annotate the "~61% recall" line)
- Consider: deprecate `quick_comparison.rs` in favour of `realistic_comparison.rs`

### Task 4.3 (needs research): Embedding model upgrade

MiniLM-L6-v2 (384d) → a stronger small ONNX model (bge-small-en-v1.5,
snowflake-arctic-embed-s). **Gated on a fresh research pass** — the embedding-model
findings did not survive verification in the 2026-06-21 run.

---

## Phase 5: Larger bets (future, post-1.0 candidates)

- **Filtered/predicate-aware traversal (ACORN-style)** — the roadmap's "constrained
  graph traversal for pre-filtering"; highest-value scale feature for a *local* lib.
- **Concurrent updates** — sharded-lock or lock-free HNSW updates.
- **Matryoshka dimension truncation** — pairs with quantization for compounding savings.
- **Multi-vector / late interaction (ColBERT/PLAID)** — large architectural change
  (storage is currently one-vector-per-doc); only if demand justifies.
- **TurboQuant** — re-evaluate for WASM streaming (training-free) if it proves needed.

Each of Phase 5 should get its own dated plan doc when picked up.

---

## Deferred / explicitly out of scope for now

- GPU graph indexes (CAGRA/cuVS) and disk-based ANN (DiskANN/SPANN) — least aligned with
  the local-first/WASM positioning; revisit only for a server-scale variant.
- SAQ — monitor for peer review before considering.
