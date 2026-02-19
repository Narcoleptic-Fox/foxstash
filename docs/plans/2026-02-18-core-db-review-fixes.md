# Core + DB Review Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all critical, high, and medium severity findings from the consolidated code review of foxstash-core and foxstash-db.

**Architecture:** Surgical fixes organized in 4 phases by risk priority. Phase 1 fixes safety/UB bugs. Phase 2 fixes db concurrency bugs. Phase 3 applies performance optimizations. Phase 4 is cleanup. Each phase commits independently and tests pass before moving to the next.

**Tech Stack:** Rust, `parking_lot` (RwLock/Mutex), `pulp` (SIMD), `serde_json`, `crc32fast`

---

## Phase 1: Safety (data loss, UB, panics)

These are bugs that can corrupt data, trigger undefined behavior, or panic in production. Small, surgical fixes with high confidence.

---

### Task 1: Fix `delete()` WAL ordering (C1 + H2)

The `delete()` method mutates in-memory state before writing the WAL. A crash between the two operations loses the delete. Additionally, releasing the write lock before acquiring the storage lock creates a TOCTOU race where concurrent inserts can make the WAL inconsistent.

**Fix:** Hold the write lock for the entire operation, WAL-first ordering.

**Files:**
- Modify: `crates/db/src/collection.rs:123-135`
- Test: `crates/db/src/collection.rs` (existing tests in `mod tests`)

**Step 1: Write a test proving delete WAL-first ordering**

Add to `crates/db/src/collection.rs` inside `mod tests`:

```rust
#[test]
fn delete_is_durable_across_reopen() {
    let dir = TempDir::new().unwrap();

    // Session 1: insert two docs, delete one, flush.
    {
        let col = Collection::create("test", dir.path(), cfg(3)).unwrap();
        col.insert("a".into(), "alpha".into(), vec![1.0, 0.0, 0.0], None)
            .unwrap();
        col.insert("b".into(), "beta".into(), vec![0.0, 1.0, 0.0], None)
            .unwrap();
        col.delete("a").unwrap();
        col.flush().unwrap();
    }

    // Session 2: reopen — "a" must still be deleted.
    {
        let col = Collection::open("test", dir.path(), cfg(3)).unwrap();
        assert_eq!(col.len(), 1);
        assert!(col.get("a").unwrap().is_none());
        assert!(col.get("b").unwrap().is_some());
    }
}
```

**Step 2: Run test — expect PASS (existing behavior is correct for non-crash case)**

```bash
cargo test -p foxstash-db delete_is_durable_across_reopen -- --exact
```

Expected: PASS (the test exercises the happy path; the crash bug requires actual crash simulation which we can't unit-test, but the code fix is still necessary).

**Step 3: Fix `delete()` — WAL-first, hold write lock for entire operation**

Replace lines 123-135 in `crates/db/src/collection.rs`:

```rust
/// Soft-delete a document by ID. Returns `true` if the document existed.
pub fn delete(&self, id: &str) -> Result<bool> {
    // Hold write lock for the entire operation to prevent TOCTOU races.
    let mut inner = self.inner.write();
    if !inner.id_map.is_live(id) {
        return Ok(false);
    }

    // WAL first (crash-safe).
    {
        let mut storage = self.storage.lock();
        storage.log_remove(id).map_err(DbError::Core)?;
    }

    // Then apply tombstone in-memory.
    inner.id_map.remove(id);
    Ok(true)
}
```

**Step 4: Run all db tests**

```bash
cargo test -p foxstash-db
```

Expected: all 64+ tests pass.

**Step 5: Commit**

```bash
git add crates/db/src/collection.rs
git commit -m "fix(db): WAL-first ordering and atomic locking in delete()

delete() previously mutated in-memory state before writing the WAL,
and released the write lock between the two operations. This created
two bugs:
1. Crash between tombstone and WAL write loses the delete
2. TOCTOU race: concurrent insert between lock release and WAL write
   can produce inconsistent WAL state

Now holds the write lock for the entire operation and writes WAL
before applying the in-memory tombstone."
```

---

### Task 2: Fix `IdMap::insert()` stale `pos_to_id` leak (H1)

When a previously existing ID is re-inserted, `insert()` overwrites `id_to_pos[id]` with a new position but never removes the old `pos_to_id[old_pos]` entry. This causes `id_at(old_pos)` and `is_pos_tombstoned(old_pos)` to return incorrect results, and `pos_to_id` grows without bound on re-insert-heavy workloads.

**Files:**
- Modify: `crates/db/src/id_map.rs:36-43`
- Test: `crates/db/src/id_map.rs` (existing `mod tests`)

**Step 1: Write a test proving the leak**

Add to `crates/db/src/id_map.rs` inside `mod tests`:

```rust
#[test]
fn reinsert_cleans_old_pos_to_id() {
    let mut map = IdMap::new();
    let old_pos = map.insert("x".into());
    assert_eq!(map.id_at(old_pos), Some("x"));

    // Re-insert "x" — old position should no longer map to "x".
    map.remove("x");
    let new_pos = map.insert("x".into());
    assert_ne!(old_pos, new_pos);
    assert_eq!(map.id_at(new_pos), Some("x"));
    assert_eq!(map.id_at(old_pos), None); // stale entry must be gone
    assert_eq!(map.get("x"), Some(new_pos));
}
```

**Step 2: Run test — expect FAIL**

```bash
cargo test -p foxstash-db reinsert_cleans_old_pos_to_id -- --exact
```

Expected: FAIL — `id_at(old_pos)` returns `Some("x")` instead of `None`.

**Step 3: Fix `insert()` to remove stale reverse mapping**

Replace lines 36-43 in `crates/db/src/id_map.rs`:

```rust
pub fn insert(&mut self, id: String) -> usize {
    self.tombstones.remove(&id);
    // Remove stale reverse mapping if ID already exists at a different position.
    if let Some(&old_pos) = self.id_to_pos.get(&id) {
        self.pos_to_id.remove(&old_pos);
    }
    let pos = self.next_pos;
    self.id_to_pos.insert(id.clone(), pos);
    self.pos_to_id.insert(pos, id);
    self.next_pos += 1;
    pos
}
```

**Step 4: Run test — expect PASS**

```bash
cargo test -p foxstash-db reinsert_cleans_old_pos_to_id -- --exact
```

**Step 5: Run all db tests**

```bash
cargo test -p foxstash-db
```

Expected: all tests pass.

**Step 6: Commit**

```bash
git add crates/db/src/id_map.rs
git commit -m "fix(db): clean stale pos_to_id entry on re-insert in IdMap

IdMap::insert() now removes the old pos_to_id entry when an ID is
re-inserted at a new position. Previously, the stale reverse mapping
leaked, causing id_at(old_pos) to return incorrect results and
pos_to_id to grow without bound on re-insert-heavy workloads."
```

---

### Task 3: Fix `random_level()` missing epsilon clamp in quantized variants (H4)

The quantized HNSW variants (`SQ8HNSWIndex`, `BinaryHNSWIndex`, `PQHNSWIndex`) are missing the `.max(f32::EPSILON)` clamp that the base `HNSWIndex` has. If `rng.gen()` returns 0.0 (~1 in 8M chance), `ln(0) = -Inf` causes `usize::MAX` allocation and OOM abort.

**Files:**
- Modify: `crates/core/src/index/hnsw_quantized.rs:344-348, 787-791`
- Modify: `crates/core/src/index/hnsw_pq.rs:407-411`

**Step 1: Fix SQ8HNSWIndex::random_level()**

In `crates/core/src/index/hnsw_quantized.rs`, line 346, change:

```rust
let uniform: f32 = rng.gen();
```

to:

```rust
let uniform: f32 = rng.gen::<f32>().max(f32::EPSILON);
```

**Step 2: Fix BinaryHNSWIndex::random_level()**

In `crates/core/src/index/hnsw_quantized.rs`, line 789, same change:

```rust
let uniform: f32 = rng.gen::<f32>().max(f32::EPSILON);
```

**Step 3: Fix PQHNSWIndex::random_level()**

In `crates/core/src/index/hnsw_pq.rs`, line 409, same change:

```rust
let uniform: f32 = rng.gen::<f32>().max(f32::EPSILON);
```

**Step 4: Run all core tests**

```bash
cargo test -p foxstash-core
```

Expected: all tests pass (the fix only affects a rare RNG edge case).

**Step 5: Commit**

```bash
git add crates/core/src/index/hnsw_quantized.rs crates/core/src/index/hnsw_pq.rs
git commit -m "fix(core): add epsilon clamp to random_level() in quantized HNSW variants

SQ8HNSWIndex, BinaryHNSWIndex, and PQHNSWIndex were missing the
.max(f32::EPSILON) clamp that the base HNSWIndex already has. Without
the clamp, rng.gen() returning exactly 0.0 causes ln(0) = -Inf,
producing usize::MAX as the level and an immediate OOM abort."
```

---

### Task 4: Add `is_finite()` check to all `add()` paths (H7)

All `add()` methods check for `NaN` but not `Inf`. Infinite values pass into embeddings and corrupt distance computations. Additionally, `serde_json` cannot serialize `Inf`, so checkpoint serialization fails at runtime.

**Files:**
- Modify: `crates/core/src/index/hnsw.rs:572-576, 639-643`
- Modify: `crates/core/src/index/hnsw_quantized.rs` (SQ8 add ~161-173, Binary add_internal ~581-585)
- Modify: `crates/core/src/index/hnsw_pq.rs` (PQ add ~211-215)

**Step 1: Write failing test for Inf rejection**

Add to `crates/core/src/index/hnsw.rs` in the test module:

```rust
#[test]
fn rejects_inf_embedding() {
    let mut index = HNSWIndex::new(3, HNSWConfig::default());
    let doc = Document {
        id: "inf".to_string(),
        content: "test".to_string(),
        embedding: vec![f32::INFINITY, 0.0, 0.0],
        metadata: None,
    };
    assert!(index.add(doc).is_err());

    let doc_neg = Document {
        id: "neg_inf".to_string(),
        content: "test".to_string(),
        embedding: vec![0.0, f32::NEG_INFINITY, 0.0],
        metadata: None,
    };
    assert!(index.add(doc_neg).is_err());
}
```

**Step 2: Run test — expect FAIL**

```bash
cargo test -p foxstash-core rejects_inf_embedding -- --exact
```

Expected: FAIL — Inf passes through the NaN-only check.

**Step 3: Replace NaN checks with `is_finite()` in all add paths**

In `crates/core/src/index/hnsw.rs`, lines 572-576, replace:

```rust
if document.embedding.iter().any(|v| v.is_nan()) {
    return Err(crate::RagError::InvalidInput(
        "embedding contains NaN values".into(),
    ));
}
```

with:

```rust
if document.embedding.iter().any(|v| !v.is_finite()) {
    return Err(crate::RagError::InvalidInput(
        "embedding contains non-finite values (NaN or Inf)".into(),
    ));
}
```

Apply the same change to:
- `hnsw.rs` lines 639-643 (`add_embedding`)
- `hnsw_quantized.rs` SQ8 add (~lines 161-173)
- `hnsw_quantized.rs` Binary add_internal (~lines 581-585)
- `hnsw_pq.rs` PQ add (~lines 211-215)

**Step 4: Run test — expect PASS**

```bash
cargo test -p foxstash-core rejects_inf_embedding -- --exact
```

**Step 5: Run all core tests**

```bash
cargo test -p foxstash-core
```

Expected: all tests pass.

**Step 6: Commit**

```bash
git add crates/core/src/index/hnsw.rs crates/core/src/index/hnsw_quantized.rs crates/core/src/index/hnsw_pq.rs
git commit -m "fix(core): reject Inf embeddings in all add() paths

Changed is_nan() checks to !is_finite() across HNSWIndex, SQ8HNSWIndex,
BinaryHNSWIndex, and PQHNSWIndex. Infinite values corrupted distance
computations and caused serde_json checkpoint serialization failures."
```

---

### Task 5: Guard `SearchContext` capacity in `search_with_context()` (H5)

`create_search_context()` allocates a bitset for the current index size. If documents are added before using the context in `search_with_context()`, `get_unchecked` accesses out of bounds — UB in release mode. `search_batch_fast` already has this guard; `search_with_context` does not.

**Files:**
- Modify: `crates/core/src/index/hnsw.rs:728-733`

**Step 1: Write failing test**

Add to `crates/core/src/index/hnsw.rs` tests:

```rust
#[test]
fn search_context_resizes_after_add() {
    let mut index = HNSWIndex::new(3, HNSWConfig::default());
    index.add(Document {
        id: "a".into(),
        content: "a".into(),
        embedding: vec![1.0, 0.0, 0.0],
        metadata: None,
    }).unwrap();

    let mut ctx = index.create_search_context();

    // Add more docs after context creation.
    for i in 0..10 {
        index.add(Document {
            id: format!("doc-{i}"),
            content: format!("content-{i}"),
            embedding: vec![(i as f32) * 0.1, 1.0 - (i as f32) * 0.1, 0.0],
            metadata: None,
        }).unwrap();
    }

    // Search must not panic or UB even though context is undersized.
    let results = index.search_with_context(&[1.0, 0.0, 0.0], 5, &mut ctx).unwrap();
    assert!(!results.is_empty());
}
```

**Step 2: Run test — expect PASS in release, may panic in debug**

```bash
cargo test -p foxstash-core search_context_resizes_after_add -- --exact
```

In debug mode the `debug_assert!` in `BitsetVisited` will fire. In release mode, this is UB. Either way, the fix is needed.

**Step 3: Add capacity guard at top of `search_with_context()`**

In `crates/core/src/index/hnsw.rs`, at the start of `search_with_context()` (after the existing dimension check), add:

```rust
// Ensure search context is large enough for current index size.
if ctx.capacity < self.len() {
    *ctx = SearchContext::new(self.len());
}
```

**Step 4: Run test — expect PASS**

```bash
cargo test -p foxstash-core search_context_resizes_after_add -- --exact
```

**Step 5: Run all core tests**

```bash
cargo test -p foxstash-core
```

**Step 6: Commit**

```bash
git add crates/core/src/index/hnsw.rs
git commit -m "fix(core): auto-resize SearchContext when index grows after creation

search_with_context() now checks that the SearchContext bitset is
large enough for the current index size and reallocates if needed.
Previously, adding documents after create_search_context() caused
out-of-bounds access via get_unchecked in release mode (UB)."
```

---

### Task 6: Fix stack buffer overflow in `search_layer` batch buffer (H6)

The fixed-size `batch_buf: [(f32, usize); 64]` has no bounds check. Safe at default `m0=64` but UB if `m0 > 64`.

**Files:**
- Modify: `crates/core/src/index/hnsw.rs:1028-1062`

**Step 1: Add bounds check**

In `crates/core/src/index/hnsw.rs`, at line 1060, replace:

```rust
batch_buf[batch_count] = (dist, neighbor_id);
batch_count += 1;
```

with:

```rust
if batch_count < batch_buf.len() {
    batch_buf[batch_count] = (dist, neighbor_id);
    batch_count += 1;
}
```

**Step 2: Run all core tests**

```bash
cargo test -p foxstash-core
```

Expected: all tests pass.

**Step 3: Commit**

```bash
git add crates/core/src/index/hnsw.rs
git commit -m "fix(core): add bounds check to search_layer batch buffer

The fixed-size 64-element stack buffer for neighbor batching had no
bounds check on batch_count. Safe at default m0=64, but caused a
stack buffer overrun with m0 > 64. Now clamps to buffer capacity."
```

---

### Task 7: Validate `m0 <= 255` in HNSWConfig (C3)

`connections_l0_count` uses `u8`, which silently truncates if `m0 > 255`. Add validation at config construction time.

**Files:**
- Modify: `crates/core/src/index/hnsw.rs` (HNSWConfig and `build_l0_cache`)

**Step 1: Write failing test**

```rust
#[test]
fn rejects_m_too_large_for_l0_cache() {
    let config = HNSWConfig::default().with_m(200); // m0 = 400, exceeds u8
    let result = HNSWIndex::build(
        vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        config,
    );
    assert!(result.is_err() || true); // We just need the config to be rejected
}
```

Actually, a better approach — add the assertion in `HNSWConfig::with_m()` and `HNSWIndex::new()`:

**Step 1: Add validation in config**

In `HNSWConfig::with_m()`, add an assertion:

```rust
pub fn with_m(mut self, m: usize) -> Self {
    assert!(m <= 127, "m must be <= 127 (m0 = 2*m must fit in u8)");
    self.m = m;
    self.m0 = m * 2;
    self
}
```

Also add a `debug_assert!` in `build_l0_cache()` at line 534:

```rust
debug_assert!(m0 <= 255, "m0 exceeds u8 capacity for connections_l0_count");
```

**Step 2: Run all core tests**

```bash
cargo test -p foxstash-core
```

**Step 3: Commit**

```bash
git add crates/core/src/index/hnsw.rs
git commit -m "fix(core): validate m <= 127 to prevent u8 truncation in L0 cache

connections_l0_count uses u8 to store neighbor counts. If m0 > 255
(from with_m(128+)), the count silently truncates, corrupting search.
Now asserts m <= 127 in with_m() and debug_asserts in build_l0_cache()."
```

---

### Task 8: Fix WAL rotation safety (C2)

`rotate_wal()` deletes the old WAL before the new WAL is confirmed created. A crash during rotation could leave no WAL at all. Fix: create new WAL first, then delete old.

**Files:**
- Modify: `crates/core/src/storage/incremental.rs:711-730`

**Step 1: Reorder operations in `rotate_wal()`**

Replace lines 711-730:

```rust
fn rotate_wal(&mut self, checkpoint_id: u64) -> Result<()> {
    // Close current WAL
    if let Some(ref mut writer) = self.wal_writer {
        writer.sync()?;
    }

    // Open new WAL FIRST (before deleting old).
    let new_wal_path = self.base_path.join(format!("wal_{:05}.log", checkpoint_id));
    let new_writer = WalWriter::open(&new_wal_path, self.config.sync_on_write)?;
    self.wal_writer = Some(new_writer);

    // Now safe to delete old WAL.
    let old_wal = self
        .base_path
        .join(format!("wal_{:05}.log", checkpoint_id.saturating_sub(1)));
    if old_wal.exists() && old_wal != new_wal_path {
        let _ = fs::remove_file(&old_wal);
    }

    Ok(())
}
```

**Step 2: Run all core tests**

```bash
cargo test -p foxstash-core
```

**Step 3: Commit**

```bash
git add crates/core/src/storage/incremental.rs
git commit -m "fix(core): create new WAL before deleting old in rotate_wal()

Previously, rotate_wal() deleted the old WAL before creating the new
one. A crash between these operations could leave no WAL at all.
Now creates and opens the new WAL first, then deletes the old one."
```

---

## Phase 2: DB Concurrency

These fix race conditions in the db crate's concurrent access patterns.

---

### Task 9: Fix `compact()` dropping concurrent inserts (H3)

`compact()` takes a read snapshot, rebuilds outside any lock, then swaps under write lock — silently dropping any inserts that happened between the snapshot and the swap.

**Fix:** Hold the write lock for the entire compaction. This blocks concurrent reads during compaction, but compaction is infrequent and correctness matters more than throughput.

**Files:**
- Modify: `crates/db/src/collection.rs:181-223`

**Step 1: Write a test for concurrent safety documentation**

Add to `crates/db/src/collection.rs` tests:

```rust
#[test]
fn compact_preserves_all_live_documents() {
    let dir = TempDir::new().unwrap();
    let col = Collection::create("test", dir.path(), cfg(3)).unwrap();

    // Insert docs, delete some, insert more.
    col.insert("a".into(), "alpha".into(), vec![1.0, 0.0, 0.0], None).unwrap();
    col.insert("b".into(), "beta".into(), vec![0.0, 1.0, 0.0], None).unwrap();
    col.insert("c".into(), "gamma".into(), vec![0.0, 0.0, 1.0], None).unwrap();
    col.delete("b").unwrap();

    // Compact.
    col.compact().unwrap();

    // All live docs must survive.
    assert_eq!(col.len(), 2);
    assert!(col.get("a").unwrap().is_some());
    assert!(col.get("b").unwrap().is_none());
    assert!(col.get("c").unwrap().is_some());

    // Search still works.
    let results = col.search(&[1.0, 0.0, 0.0], 5, None).unwrap();
    assert_eq!(results.len(), 2);
    assert!(!results.iter().any(|r| r.id == "b"));
}
```

**Step 2: Run test — expect PASS (single-threaded case works)**

```bash
cargo test -p foxstash-db compact_preserves_all_live_documents -- --exact
```

**Step 3: Restructure `compact()` to hold write lock throughout**

Replace `compact()` in `crates/db/src/collection.rs`:

```rust
/// Compact: rebuild index from live documents only, checkpoint, reclaim tombstones.
pub fn compact(&self) -> Result<()> {
    // Hold write lock for the entire operation to prevent concurrent mutations
    // from being silently dropped during the swap.
    let mut inner = self.inner.write();

    let live_docs = self.collect_live_documents(&inner);
    let doc_count = live_docs.len();

    // Rebuild index + id_map from scratch.
    let mut new_index = HNSWIndex::new(self.config.embedding_dim, self.config.hnsw.clone());
    let mut new_id_map = IdMap::new();

    for doc in &live_docs {
        new_index.add(doc.clone()).map_err(DbError::Core)?;
        new_id_map.insert(doc.id.clone());
    }

    // Checkpoint the compacted state.
    {
        let mut storage = self.storage.lock();
        storage
            .checkpoint(
                &live_docs,
                IndexMetadata {
                    document_count: doc_count,
                    embedding_dim: self.config.embedding_dim,
                    index_type: "hnsw".into(),
                },
            )
            .map_err(DbError::Core)?;
    }

    // Swap in the new state (already holding write lock).
    inner.index = new_index;
    inner.id_map = new_id_map;
    inner.documents = live_docs;

    debug!(name = %self.name, doc_count, "compaction complete");
    Ok(())
}
```

Note: `collect_live_documents` takes `&CollectionInner`, not `&self`, so no borrow conflict.

**Step 4: Run all db tests**

```bash
cargo test -p foxstash-db
```

**Step 5: Commit**

```bash
git add crates/db/src/collection.rs
git commit -m "fix(db): hold write lock during entire compact() to prevent data loss

compact() previously took a read snapshot, rebuilt outside any lock,
then swapped under write lock. Concurrent inserts between the snapshot
and swap were silently dropped from the in-memory index (though they
survived in the WAL for recovery). Now holds the write lock for the
entire operation, ensuring no mutations are lost."
```

---

### Task 10: Add `embedding_dim` validation on recovery (M8)

Recovery loads checkpoint documents without checking that their dimension matches the config. A misconfigured `DbConfig` produces a confusing `DimensionMismatch` error deep in `HNSWIndex::add` instead of a clear upfront error.

**Files:**
- Modify: `crates/db/src/recovery.rs:18-38`

**Step 1: Add dimension validation after checkpoint load**

In `crates/db/src/recovery.rs`, after loading the checkpoint (around line 28), add:

```rust
if meta.embedding_dim != config.embedding_dim {
    return Err(DbError::DimensionMismatch {
        expected: config.embedding_dim,
        actual: meta.embedding_dim,
    });
}
```

**Step 2: Run all db tests**

```bash
cargo test -p foxstash-db
```

**Step 3: Commit**

```bash
git add crates/db/src/recovery.rs
git commit -m "fix(db): validate embedding_dim against checkpoint metadata on recovery

Recovery now checks that the checkpoint's embedding_dim matches the
DbConfig before loading documents. Previously, a misconfigured dim
produced a confusing DimensionMismatch error deep in HNSWIndex::add."
```

---

### Task 11: Clamp over-fetch in `search_unfiltered()` (M9)

`k + tombstone_count` can far exceed index size when tombstones are large. HNSW search cost scales with the fetch parameter.

**Files:**
- Modify: `crates/db/src/collection.rs:263-264`

**Step 1: Clamp fetch to index size**

Replace line 264:

```rust
let fetch = k + inner.id_map.tombstone_count();
```

with:

```rust
let fetch = (k + inner.id_map.tombstone_count()).min(inner.index.len());
```

**Step 2: Run all db tests**

```bash
cargo test -p foxstash-db
```

**Step 3: Commit**

```bash
git add crates/db/src/collection.rs
git commit -m "fix(db): clamp search over-fetch to index size

search_unfiltered() used k + tombstone_count as the fetch parameter,
which could far exceed the index size when tombstones are numerous.
Now clamped to index.len() to avoid unnecessary HNSW work."
```

---

### Task 12: Decouple auto-checkpoint from full compaction (M10)

`maybe_auto_checkpoint()` calls `compact()` which does a full O(N log N) HNSW rebuild inline during `insert()`. Replace with a WAL flush for auto-checkpoint; compact only when explicitly requested or tombstone ratio is high.

**Files:**
- Modify: `crates/db/src/collection.rs:320-335`

**Step 1: Replace `compact()` with `flush()` in auto-checkpoint, add tombstone-ratio trigger**

Replace `maybe_auto_checkpoint()`:

```rust
fn maybe_auto_checkpoint(&self) -> Result<()> {
    if !self.config.auto_checkpoint {
        return Ok(());
    }

    let needs = {
        let storage = self.storage.lock();
        storage.needs_checkpoint()
    };

    if needs {
        // Lightweight: flush WAL to disk and checkpoint current state
        // without rebuilding the entire HNSW index.
        let inner = self.inner.read();
        let live_docs = self.collect_live_documents(&inner);
        let doc_count = live_docs.len();

        let mut storage = self.storage.lock();
        storage
            .checkpoint(
                &live_docs,
                IndexMetadata {
                    document_count: doc_count,
                    embedding_dim: self.config.embedding_dim,
                    index_type: "hnsw".into(),
                },
            )
            .map_err(DbError::Core)?;
    }

    Ok(())
}
```

**Step 2: Run all db tests**

```bash
cargo test -p foxstash-db
```

**Step 3: Commit**

```bash
git add crates/db/src/collection.rs
git commit -m "perf(db): decouple auto-checkpoint from full compaction

maybe_auto_checkpoint() previously called compact(), which rebuilds
the entire HNSW index — O(N log N) inline during insert(). Now
performs a lightweight checkpoint (serialize + flush) without
rebuilding. Full compaction is still available via compact()."
```

---

## Phase 3: Performance

These are optimization changes that improve throughput and reduce allocations. Each is independent and can be applied in any order.

---

### Task 13: Incremental L0 cache update instead of full rebuild (M1)

`build_l0_cache()` is called after every `add()` / `add_embedding()` and rebuilds the entire flat L0 cache from scratch — O(n) per insert. Replace with targeted updates to only the affected nodes.

**Files:**
- Modify: `crates/core/src/index/hnsw.rs:532-546, 608-609, 675-676`

**Step 1: Add `sync_l0_cache_for_node()` method**

Add a new method to `HNSWIndex`:

```rust
/// Update the L0 cache for a single node and its neighbors.
/// Much cheaper than rebuild_l0_cache() — O(m0) instead of O(n * m0).
fn sync_l0_cache_for_nodes(&mut self, node_ids: &[usize]) {
    let m0 = self.config.m0;
    for &node_id in node_ids {
        if node_id < self.connections.len() && !self.connections[node_id].is_empty() {
            let neighbors = &self.connections[node_id][0];
            let count = neighbors.len().min(m0);
            let start = node_id * m0;
            // Ensure cache is large enough (may need to grow for new nodes).
            if start + m0 > self.connections_l0.len() {
                self.connections_l0.resize(self.connections.len() * m0, 0u32);
                self.connections_l0_count.resize(self.connections.len(), 0u8);
            }
            self.connections_l0[start..start + count]
                .copy_from_slice(&neighbors[..count]);
            // Zero out any trailing stale entries.
            for j in count..m0 {
                self.connections_l0[start + j] = 0;
            }
            self.connections_l0_count[node_id] = count as u8;
        }
    }
}
```

**Step 2: Replace `build_l0_cache()` calls in `add()` and `add_embedding()`**

In `add()` (line 608-609) and `add_embedding()` (line 675-676), replace:

```rust
self.build_l0_cache();
```

with logic to collect affected node IDs from `insert_node` and call `sync_l0_cache_for_nodes`. This requires `insert_node` to return the set of affected nodes. Modify `insert_node` to return a `Vec<usize>` of node IDs whose L0 connections were modified (the new node plus all nodes it connected to at layer 0).

The detailed implementation: have `insert_node` / `connect_neighbors` track which nodes were mutated at layer 0 and return them. Then:

```rust
let affected = self.insert_node(node_id, node_level);
self.sync_l0_cache_for_nodes(&affected);
```

Keep `build_l0_cache()` for use in `build()` and `build_parallel()` where a full rebuild is appropriate.

**Step 3: Run all core tests**

```bash
cargo test -p foxstash-core
```

**Step 4: Run benchmarks to verify improvement**

```bash
cargo bench -p foxstash-benches -- insert
```

**Step 5: Commit**

```bash
git add crates/core/src/index/hnsw.rs
git commit -m "perf(core): incremental L0 cache update on add() instead of full rebuild

Replaced build_l0_cache() calls in add()/add_embedding() with
sync_l0_cache_for_nodes() that only updates the new node and its
neighbors. O(m0) per insert instead of O(n * m0). Full rebuild
retained for build()/build_parallel() where it's appropriate."
```

---

### Task 14: Eliminate unnecessary `WalEntry` clone in `verify()` (M7)

`verify()` clones the entire `WalEntry` (including embedding vectors) just to zero the checksum field before recomputing. But `compute_checksum()` already excludes the checksum field from its hash input, making the clone pointless.

**Files:**
- Modify: `crates/core/src/storage/incremental.rs:211-218`

**Step 1: Simplify `verify()`**

Replace lines 211-218:

```rust
pub fn verify(&self) -> bool {
    // compute_checksum() serializes (seq, timestamp, operation) — the checksum
    // field is already excluded from the hash input, so no clone needed.
    self.checksum == self.compute_checksum()
}
```

**Step 2: Run all core tests**

```bash
cargo test -p foxstash-core
```

**Step 3: Commit**

```bash
git add crates/core/src/storage/incremental.rs
git commit -m "perf(core): remove unnecessary WalEntry clone in verify()

verify() cloned the entire WalEntry (including embedding Vec<f32>) to
zero the checksum field before recomputing. Since compute_checksum()
already excludes the checksum field from its hash input, the clone was
unnecessary. Eliminates one heap allocation per WAL entry during recovery."
```

---

### Task 15: Replace `HashSet<usize>` with `Vec<u32>` in quantized HNSW (M2)

The quantized HNSW variants use `HashSet<usize>` for neighbor lists while the main `HNSWIndex` uses `Vec<u32>` for 4-5x faster search. This is a larger refactor that affects `SQ8Node`, `BinaryNode`, and all methods that read/write connections in the quantized indexes.

**Files:**
- Modify: `crates/core/src/index/hnsw_quantized.rs` (SQ8Node:97, BinaryNode:502, and all connection access sites)

**Step 1: Change connection field types**

In both `SQ8Node` and `BinaryNode`, change:

```rust
connections: Vec<HashSet<usize>>,
```

to:

```rust
connections: Vec<Vec<u32>>,
```

**Step 2: Update all connection access patterns**

Replace `.insert(id)` with a duplicate-checked push:

```rust
fn add_connection(connections: &mut Vec<u32>, neighbor: u32, max_connections: usize) {
    if !connections.contains(&neighbor) && connections.len() < max_connections {
        connections.push(neighbor);
    }
}
```

Replace `.contains(&id)` with `connections.contains(&(id as u32))`.

Replace iteration patterns from `HashSet` to `Vec` (mostly transparent since both implement `Iterator`).

**Step 3: Run all core tests**

```bash
cargo test -p foxstash-core
```

**Step 4: Commit**

```bash
git add crates/core/src/index/hnsw_quantized.rs
git commit -m "perf(core): replace HashSet<usize> with Vec<u32> for quantized HNSW connections

Matches the main HNSWIndex pattern for cache-friendly neighbor
traversal. Vec<u32> is 4-5x faster for small neighbor lists (M=16-64)
due to sequential memory access and no hashing overhead."
```

---

### Task 16: Add `SearchContext` reuse to quantized HNSW search (M3)

The quantized `search_layer` functions allocate fresh `HashSet` + 2x `BinaryHeap` on every call. Port the `SearchContext` pattern from the main HNSW.

**Files:**
- Modify: `crates/core/src/index/hnsw_quantized.rs` (search paths for SQ8 and Binary)

This is a larger refactor that follows the same pattern as `SearchContext` in `hnsw.rs:115-156`. Create a `QuantizedSearchContext` with a `BitsetVisited` and reusable heaps, pass it through `search_layer` calls. Reset between searches instead of reallocating.

**Step 1: Implement `QuantizedSearchContext`**

Port the `SearchContext` struct from `hnsw.rs` into the quantized module (or make the existing one public and reuse it).

**Step 2: Thread it through `search()` and `search_layer()` in both SQ8 and Binary**

**Step 3: Run all core tests**

```bash
cargo test -p foxstash-core
```

**Step 4: Commit**

```bash
git add crates/core/src/index/hnsw_quantized.rs
git commit -m "perf(core): reuse SearchContext in quantized HNSW search

Ported the SearchContext pattern (bitset + reusable heaps) from the
main HNSWIndex to SQ8HNSWIndex and BinaryHNSWIndex. Eliminates 3
heap allocations per search_layer call."
```

---

### Task 17: SIMD-ify SQ8 distance functions (M5)

`Sq8L2` and `AsymL2` accept a SIMD token but use scalar loops. Implement proper SIMD with u8->i16 widening.

**Files:**
- Modify: `crates/core/src/vector/quantize.rs:360-382, 412-430`

This requires SIMD intrinsics for u8 subtraction with i16 widening and accumulation. The implementation depends on the `pulp` API for integer SIMD. If `pulp` doesn't expose integer SIMD, fall back to processing 4 u8s at a time using u32 arithmetic (still faster than byte-by-byte).

**Step 1: Implement SIMD inner loop for `Sq8L2`**

Use the `simd` token to process bytes in chunks. For AVX2, load 32 bytes at a time, widen to i16, subtract, square, accumulate.

**Step 2: Run all core tests + benchmarks**

```bash
cargo test -p foxstash-core
cargo bench -p foxstash-benches -- sq8
```

**Step 3: Commit**

```bash
git add crates/core/src/vector/quantize.rs
git commit -m "perf(core): SIMD-accelerate SQ8 L2 and asymmetric L2 distance

Replaced scalar byte-by-byte loops with SIMD-accelerated u8/i16
widening and accumulation in Sq8L2 and AsymL2 distance functions."
```

---

### Task 18: Use full SIMD loads in main loop, partial only for remainder (M6)

All four SIMD distance functions (`FusedCosineDistance`, `DotProduct`, `L2Distance`, `Magnitude`) use `f32s_partial_load` in the full-chunk loop where `lane_count` elements are guaranteed available.

**Files:**
- Modify: `crates/core/src/vector/simd.rs` (lines 219-220, 279-280, 327-328, 374)

**Step 1: Replace `f32s_partial_load` with direct slice cast in full-chunk loops**

For each of the four `with_simd` implementations, replace the hot loop load:

```rust
let a_vec = pulp::cast_lossy::<_, S::f32s>(simd.f32s_partial_load(&a[i..]));
```

with a full-lane load (the exact API depends on `pulp` version — use `pulp::cast::<&[f32], S::f32s>` or equivalent). If pulp doesn't have an explicit full load, use:

```rust
let a_slice = &a[i..i + lane_count];
let a_vec: S::f32s = pulp::cast(*bytemuck::from_bytes::<[f32; LANES]>(
    bytemuck::cast_slice(a_slice)
));
```

Or more practically, verify that `f32s_partial_load` with a slice >= lane_count compiles to the same instruction as a full load (in which case this is a no-op optimization and can be documented as "verified correct").

**Step 2: Run all core tests + benchmarks**

```bash
cargo test -p foxstash-core
cargo bench -p foxstash-benches -- distance
```

**Step 3: Commit**

```bash
git add crates/core/src/vector/simd.rs
git commit -m "perf(core): use full SIMD loads in main distance loop

Replaced f32s_partial_load (designed for remainder handling) with
full-width loads in the main SIMD loop where lane_count elements
are guaranteed available."
```

---

## Phase 4: Cleanup

Lower-priority improvements for robustness and maintainability.

---

### Task 19: Migrate `.meta` files from bincode to serde_json (H8)

The bincode->JSON migration was applied to data content but not to `StorageMetadata` sidecar files. Schema evolution on bincode is fragile.

**Files:**
- Modify: `crates/core/src/storage/file.rs:275-276, 581, 870-871`

**Step 1: Replace bincode serialize/deserialize for StorageMetadata with serde_json**

Change write sites (lines 275, 870):
```rust
let meta_bytes = serde_json::to_vec(&metadata)
    .map_err(|e| RagError::StorageError(format!("metadata serialize failed: {}", e)))?;
```

Change read site (line 581):
```rust
let metadata: StorageMetadata = serde_json::from_slice(&contents)
    .map_err(|e| RagError::StorageError(format!("metadata deserialize failed: {}", e)))?;
```

Consider bumping `STORAGE_VERSION` to signal the format change.

**Step 2: Run all core tests**

```bash
cargo test -p foxstash-core
```

**Step 3: Commit**

```bash
git add crates/core/src/storage/file.rs
git commit -m "refactor(core): migrate StorageMetadata from bincode to serde_json

Completes the bincode-to-JSON migration that was applied to data
content but missed the .meta sidecar files. JSON supports schema
evolution (optional fields, field reordering) unlike bincode."
```

---

### Task 20: Document `Filter::Ne` missing-field semantics (L1)

**Files:**
- Modify: `crates/db/src/filter.rs:12-14`

Add doc comment to the `Ne` variant:

```rust
/// Field does not equal value.
///
/// Returns `true` when the field is missing or `None`, following SQL-like
/// NULL semantics where a missing value is considered "not equal" to any value.
Ne { field: String, value: Value },
```

**Step 1: Add doc comment, run tests, commit**

```bash
cargo test -p foxstash-db
git add crates/db/src/filter.rs
git commit -m "docs(db): document Ne filter missing-field semantics"
```

---

### Task 21: Document single `embedding_dim` constraint on `VectorStore` (L3)

**Files:**
- Modify: `crates/db/src/lib.rs:57-65` and `crates/db/src/store.rs:22-26`

Add doc comments explaining all collections share the same embedding dimension, and that separate `VectorStore` instances are needed for different embedding models.

**Step 1: Add doc comments, run tests, commit**

```bash
cargo test -p foxstash-db
git add crates/db/src/lib.rs crates/db/src/store.rs
git commit -m "docs(db): document single embedding_dim constraint across collections"
```

---

### Task 22: Remove stale `mod cache` / `CachedEmbedder` declarations from embedding/mod.rs

**Files:**
- Modify: `crates/core/src/embedding/mod.rs`

The uncommitted changes declare `mod cache` and `pub use cache::CachedEmbedder` but no `cache.rs` file exists. This would fail to compile with `--features onnx`. Either revert these lines or create the module.

**Step 1: Revert the uncommitted embedding/mod.rs changes**

```bash
git checkout -- crates/core/src/embedding/mod.rs
```

**Step 2: Verify compilation**

```bash
cargo check -p foxstash-core
```

**Step 3: Commit (if there were other staged changes to preserve)**

No commit needed — this is just reverting unstaged changes.

---

## Summary

| Phase | Tasks | Focus | Est. Complexity |
|-------|-------|-------|-----------------|
| 1 | 1-8 | Safety: data loss, UB, panics | Small surgical fixes |
| 2 | 9-12 | DB concurrency + correctness | Medium: locking strategy changes |
| 3 | 13-18 | Performance: hot path optimizations | Medium-Large: SIMD + refactoring |
| 4 | 19-22 | Cleanup: format migration, docs | Small |

**Run full test suite after each phase:**

```bash
cargo test -p foxstash-core && cargo test -p foxstash-db
```

**Commit each phase as a logical batch, or per-task for easier review.**
