//! End-to-end integration tests for foxstash-db.
//!
//! Each test exercises a full stack scenario — open store, manipulate data,
//! verify behavior, optionally reopen to confirm persistence.
//!
//! These tests intentionally use a small embedding dimension (4) to keep
//! test data compact while still exercising real HNSW and BM25 code paths.

use foxstash_db::{DbConfig, Filter, HybridConfig, MergeStrategy, VectorStore};
use serde_json::json;
use tempfile::TempDir;

// ── Helpers ────────────────────────────────────────────────────────────────

/// Config with dim=4, auto-checkpoint disabled (deterministic test behavior).
fn test_cfg() -> DbConfig {
    DbConfig {
        embedding_dim: 4,
        auto_checkpoint: false,
        ..Default::default()
    }
}

/// Build a unit-axis embedding of the given dimension.
/// `axis` selects which component is 1.0; rest are 0.0.
fn axis(axis: usize, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dim];
    v[axis % dim] = 1.0;
    v
}

// ── Scenario 1: Full lifecycle ─────────────────────────────────────────────

/// Open store → create collection → insert 20+ docs with metadata →
/// vector search → text search → hybrid search → verify results →
/// flush → drop store → reopen → verify all data persisted.
#[test]
fn scenario_full_lifecycle() {
    let dir = TempDir::new().unwrap();

    // Phase 1: populate.
    {
        let store = VectorStore::open(dir.path(), test_cfg()).unwrap();
        let col = store.get_or_create_collection("lifecycle").unwrap();

        let categories = ["alpha", "beta", "gamma", "delta"];
        for i in 0..24usize {
            let cat = categories[i % 4];
            let mut emb = axis(i % 4, 4);
            // Slight perturbation so every vector is unique.
            emb[(i + 1) % 4] = (i as f32) * 0.001;

            col.insert(
                format!("doc-{i}"),
                format!("the {cat} document number {i}"),
                emb,
                Some(json!({ "category": cat, "index": i })),
            )
            .unwrap();
        }
        assert_eq!(col.len(), 24);

        // Vector search — query near axis 0 should return docs with cat "alpha".
        let v_results = col.search(&axis(0, 4), 6, None).unwrap();
        assert!(!v_results.is_empty());
        // Top result should be closest to axis 0.
        let top_ids: Vec<&str> = v_results.iter().map(|r| r.id.as_str()).collect();
        assert!(
            top_ids.contains(&"doc-0"),
            "doc-0 should be in top-6 for axis-0 query; got {top_ids:?}"
        );

        // Text search for "alpha".
        let t_results = col.search_text("alpha", 10, None).unwrap();
        assert!(!t_results.is_empty());
        assert!(
            t_results.iter().all(|r| {
                r.content.contains("alpha")
            }),
            "text search for 'alpha' should only return alpha docs"
        );

        // Filtered vector search — only beta docs.
        let filter = Filter::eq("category", "beta");
        let f_results = col.search(&axis(1, 4), 10, Some(&filter)).unwrap();
        assert!(!f_results.is_empty());
        assert!(
            f_results.iter().all(|r| {
                r.metadata
                    .as_ref()
                    .and_then(|m| m.get("category"))
                    .and_then(|v| v.as_str())
                    == Some("beta")
            }),
            "all filtered results must be category=beta"
        );

        // Hybrid search.
        let h_results = col
            .search_hybrid(&axis(0, 4), "alpha document", 8, None, None)
            .unwrap();
        assert!(!h_results.is_empty());
        assert!(h_results.len() <= 8);

        store.flush_all().unwrap();
    }

    // Phase 2: reopen and verify persistence.
    {
        let store = VectorStore::open(dir.path(), test_cfg()).unwrap();
        assert!(store.collections().contains(&"lifecycle".to_string()));

        let col = store.get_collection("lifecycle").unwrap();
        assert_eq!(col.len(), 24, "all 24 docs must survive flush+reopen");

        // Spot-check a specific document.
        // doc-7: i=7, categories[7 % 4] = categories[3] = "delta"
        let doc = col.get("doc-7").unwrap().unwrap();
        assert_eq!(doc.content, "the delta document number 7");
        assert_eq!(doc.metadata.as_ref().unwrap()["category"], "delta");

        // Vector search still works.
        let results = col.search(&axis(2, 4), 3, None).unwrap();
        assert!(!results.is_empty());

        // Text search still works.
        let results = col.search_text("gamma", 10, None).unwrap();
        assert!(!results.is_empty());
        assert!(results.iter().all(|r| r.content.contains("gamma")));
    }
}

// ── Scenario 2: Multi-collection isolation ─────────────────────────────────

/// Open store → create 3 collections → insert different data in each →
/// search across them → delete one collection → verify others unaffected.
#[test]
fn scenario_multi_collection_isolation() {
    let dir = TempDir::new().unwrap();
    let store = VectorStore::open(dir.path(), test_cfg()).unwrap();

    // Populate three independent collections.
    let animals = store.get_or_create_collection("animals").unwrap();
    let fruits = store.get_or_create_collection("fruits").unwrap();
    let colors = store.get_or_create_collection("colors").unwrap();

    let animal_data = [
        ("cat", "the quick cat runs fast", [1.0, 0.0, 0.0, 0.0]),
        ("dog", "a loyal dog barks loudly", [0.9, 0.1, 0.0, 0.0]),
        ("bird", "a small bird sings softly", [0.0, 0.0, 1.0, 0.0]),
    ];
    for (id, content, emb) in &animal_data {
        animals
            .insert(id.to_string(), content.to_string(), emb.to_vec(), None)
            .unwrap();
    }

    let fruit_data = [
        ("apple", "crisp red apple from orchard", [0.0, 1.0, 0.0, 0.0]),
        ("banana", "yellow banana ripe and sweet", [0.0, 0.9, 0.1, 0.0]),
    ];
    for (id, content, emb) in &fruit_data {
        fruits
            .insert(id.to_string(), content.to_string(), emb.to_vec(), None)
            .unwrap();
    }

    colors
        .insert(
            "red".into(),
            "vibrant red color".into(),
            vec![0.0, 0.0, 0.0, 1.0],
            None,
        )
        .unwrap();
    colors
        .insert(
            "blue".into(),
            "calm blue color".into(),
            vec![0.0, 0.0, 0.1, 0.9],
            None,
        )
        .unwrap();

    assert_eq!(animals.len(), 3);
    assert_eq!(fruits.len(), 2);
    assert_eq!(colors.len(), 2);

    // Collections are isolated — text search in "animals" finds nothing from fruits.
    let animal_results = animals.search_text("apple", 10, None).unwrap();
    assert!(
        animal_results.is_empty(),
        "apple should not appear in animals collection"
    );

    let fruit_results = fruits.search_text("apple", 10, None).unwrap();
    assert!(
        !fruit_results.is_empty(),
        "apple should appear in fruits collection"
    );

    // Delete the colors collection.
    store.delete_collection("colors").unwrap();

    // Colors collection gone from registry.
    assert!(store.get_collection("colors").is_err());

    // Animals and fruits are unaffected.
    assert_eq!(animals.len(), 3);
    assert_eq!(fruits.len(), 2);

    let mut names = store.collections();
    names.sort();
    assert_eq!(names, vec!["animals", "fruits"]);

    // Search still works in surviving collections.
    let results = animals.search(&[1.0, 0.0, 0.0, 0.0], 2, None).unwrap();
    assert!(!results.is_empty());
    assert_eq!(results[0].id, "cat");
}

// ── Scenario 3: WAL recovery ───────────────────────────────────────────────

/// Open store → insert docs → flush (checkpoint) → insert more (unflushed) →
/// drop store WITHOUT flush → reopen → verify WAL-logged docs recovered.
#[test]
fn scenario_wal_recovery() {
    let dir = TempDir::new().unwrap();

    // Phase 1: checkpoint batch.
    {
        let store = VectorStore::open(dir.path(), test_cfg()).unwrap();
        let col = store.get_or_create_collection("wal_test").unwrap();

        for i in 0..5usize {
            col.insert(
                format!("checkpoint-{i}"),
                format!("checkpointed document {i}"),
                axis(i % 4, 4),
                Some(json!({ "batch": "checkpoint" })),
            )
            .unwrap();
        }
        assert_eq!(col.len(), 5);
        // Flush to produce a checkpoint.
        store.flush_all().unwrap();
    }

    // Phase 2: WAL-only batch (no flush before drop).
    {
        let store = VectorStore::open(dir.path(), test_cfg()).unwrap();
        let col = store.get_collection("wal_test").unwrap();
        assert_eq!(col.len(), 5, "checkpoint recovered");

        for i in 0..4usize {
            col.insert(
                format!("wal-{i}"),
                format!("wal-only document {i}"),
                axis((i + 2) % 4, 4),
                Some(json!({ "batch": "wal" })),
            )
            .unwrap();
        }
        assert_eq!(col.len(), 9);
        // Intentionally drop WITHOUT flushing — WAL entries must be replayed.
    }

    // Phase 3: reopen and verify full recovery.
    {
        let store = VectorStore::open(dir.path(), test_cfg()).unwrap();
        let col = store.get_collection("wal_test").unwrap();

        assert_eq!(col.len(), 9, "checkpoint docs + WAL docs both recovered");

        // Checkpoint docs still present.
        for i in 0..5usize {
            let doc = col.get(&format!("checkpoint-{i}")).unwrap();
            assert!(
                doc.is_some(),
                "checkpointed doc checkpoint-{i} should be present"
            );
        }

        // WAL-only docs recovered.
        for i in 0..4usize {
            let doc = col.get(&format!("wal-{i}")).unwrap();
            assert!(
                doc.is_some(),
                "wal-only doc wal-{i} should be recovered from WAL"
            );
            assert_eq!(
                doc.unwrap().metadata.as_ref().unwrap()["batch"],
                "wal"
            );
        }

        // Search works across all recovered docs.
        let results = col.search(&axis(0, 4), 5, None).unwrap();
        assert!(!results.is_empty());

        // Text search also works after recovery.
        let results = col.search_text("document", 15, None).unwrap();
        assert_eq!(
            results.len(),
            9,
            "all 9 docs contain 'document' — text index rebuilt correctly"
        );
    }
}

// ── Scenario 4: Compaction ─────────────────────────────────────────────────

/// Insert 50+ docs → delete half → compact → verify remaining docs searchable →
/// flush → reopen → verify compacted state persisted.
#[test]
fn scenario_compaction() {
    let dir = TempDir::new().unwrap();

    // Phase 1: bulk insert and delete half.
    {
        let store = VectorStore::open(dir.path(), test_cfg()).unwrap();
        let col = store.get_or_create_collection("compact_test").unwrap();

        for i in 0..52usize {
            col.insert(
                format!("doc-{i}"),
                format!("compaction document number {i} with unique term term{i}"),
                axis(i % 4, 4),
                Some(json!({ "group": i % 2, "idx": i })),
            )
            .unwrap();
        }
        assert_eq!(col.len(), 52);

        // Delete all even-indexed docs (26 docs).
        for i in (0..52usize).step_by(2) {
            let removed = col.delete(&format!("doc-{i}")).unwrap();
            assert!(removed, "doc-{i} should have been live before delete");
        }
        assert_eq!(col.len(), 26, "26 odd-indexed docs remain");

        // Compact to eliminate tombstones.
        col.compact().unwrap();
        assert_eq!(col.len(), 26, "compact must not lose live docs");

        // All deleted docs absent after compaction.
        for i in (0..52usize).step_by(2) {
            assert!(
                col.get(&format!("doc-{i}")).unwrap().is_none(),
                "deleted doc doc-{i} should not be present after compaction"
            );
        }

        // All surviving docs present and searchable.
        for i in (1..52usize).step_by(2) {
            let doc = col.get(&format!("doc-{i}")).unwrap();
            assert!(doc.is_some(), "live doc doc-{i} should be present");
        }

        // Vector search still works.
        let results = col.search(&axis(1, 4), 10, None).unwrap();
        assert!(!results.is_empty());
        // None of the results should be deleted docs.
        for r in &results {
            let idx: usize = r.id.trim_start_matches("doc-").parse().unwrap();
            assert_eq!(idx % 2, 1, "deleted doc {} appeared in results", r.id);
        }

        // Text search still works.
        let results = col.search_text("compaction", 30, None).unwrap();
        assert_eq!(results.len(), 26, "text search returns all 26 live docs");

        store.flush_all().unwrap();
    }

    // Phase 2: reopen and verify compacted state persisted.
    {
        let store = VectorStore::open(dir.path(), test_cfg()).unwrap();
        let col = store.get_collection("compact_test").unwrap();

        assert_eq!(col.len(), 26, "compacted count must survive reopen");

        // Even docs still gone.
        assert!(col.get("doc-0").unwrap().is_none());
        assert!(col.get("doc-50").unwrap().is_none());

        // Odd docs present.
        let doc = col.get("doc-1").unwrap().unwrap();
        assert!(doc.content.contains("1"));

        // Search still works.
        let results = col.search(&axis(3, 4), 5, None).unwrap();
        assert!(!results.is_empty());
    }
}

// ── Scenario 5: Filter combinations ───────────────────────────────────────

/// Insert docs with rich metadata → test all filter variants.
#[test]
fn scenario_filter_combinations() {
    let dir = TempDir::new().unwrap();
    let store = VectorStore::open(dir.path(), test_cfg()).unwrap();
    let col = store.get_or_create_collection("filters").unwrap();

    // Insert 12 docs with diverse metadata.
    let docs = [
        // id, content, axis, priority, tag, active, score, nested_level
        ("d00", "rust programming language", 0, 1, "lang",   true,  90.0f64),
        ("d01", "python scripting language", 1, 2, "lang",   true,  70.0),
        ("d02", "go systems language",       2, 1, "lang",   false, 80.0),
        ("d03", "rust game development",     0, 3, "dev",    true,  95.0),
        ("d04", "python data science",       1, 1, "data",   true,  75.0),
        ("d05", "go microservices server",   2, 2, "server", false, 85.0),
        ("d06", "rust embedded systems",     3, 3, "dev",    true,  88.0),
        ("d07", "python web framework",      1, 2, "web",    true,  72.0),
        ("d08", "go web server handlers",    2, 1, "web",    false, 82.0),
        ("d09", "rust async runtime crate",  0, 2, "async",  true,  91.0),
        ("d10", "typescript front end",      3, 1, "web",    true,  65.0),
        ("d11", "typescript node server",    3, 3, "server", false, 60.0),
    ];

    for (id, content, ax, priority, tag, active, score) in &docs {
        col.insert(
            id.to_string(),
            content.to_string(),
            axis(*ax, 4),
            Some(json!({
                "priority": priority,
                "tag": tag,
                "active": active,
                "metrics": { "score": score }
            })),
        )
        .unwrap();
    }
    assert_eq!(col.len(), 12);

    // ── Eq ─────────────────────────────────────────────────────────
    let results = col
        .search(&axis(0, 4), 12, Some(&Filter::eq("tag", "lang")))
        .unwrap();
    assert_eq!(results.len(), 3, "exactly 3 lang docs");
    assert!(results.iter().all(|r| r.metadata.as_ref().unwrap()["tag"] == "lang"));

    // ── Ne ─────────────────────────────────────────────────────────
    let results = col
        .search(&axis(0, 4), 12, Some(&Filter::ne("tag", "lang")))
        .unwrap();
    assert_eq!(results.len(), 9, "9 non-lang docs");
    assert!(results.iter().all(|r| r.metadata.as_ref().unwrap()["tag"] != "lang"));

    // ── In ─────────────────────────────────────────────────────────
    let results = col
        .search(
            &axis(0, 4),
            12,
            Some(&Filter::is_in(
                "tag",
                vec![json!("dev"), json!("async")],
            )),
        )
        .unwrap();
    assert_eq!(results.len(), 3, "dev + async docs: d03, d06, d09");

    // ── Gt ─────────────────────────────────────────────────────────
    let results = col
        .search(&axis(0, 4), 12, Some(&Filter::gt("priority", json!(2))))
        .unwrap();
    assert_eq!(results.len(), 3, "priority > 2: d03, d06, d11");

    // ── Lt ─────────────────────────────────────────────────────────
    let results = col
        .search(&axis(0, 4), 12, Some(&Filter::lt("priority", json!(2))))
        .unwrap();
    assert_eq!(results.len(), 5, "priority < 2: d00, d02, d04, d08, d10");

    // ── Gte / Lte ──────────────────────────────────────────────────
    let results = col
        .search(&axis(0, 4), 12, Some(&Filter::gte("priority", json!(2))))
        .unwrap();
    assert_eq!(results.len(), 7, "priority >= 2");

    let results = col
        .search(&axis(0, 4), 12, Some(&Filter::lte("priority", json!(2))))
        .unwrap();
    assert_eq!(results.len(), 9, "priority <= 2");

    // ── And ────────────────────────────────────────────────────────
    let results = col
        .search(
            &axis(0, 4),
            12,
            Some(&Filter::and(vec![
                Filter::eq("active", true),
                Filter::eq("tag", "web"),
            ])),
        )
        .unwrap();
    assert_eq!(results.len(), 2, "active web docs: d07, d10");
    let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
    assert!(ids.contains(&"d07"));
    assert!(ids.contains(&"d10"));

    // ── Or ─────────────────────────────────────────────────────────
    let results = col
        .search(
            &axis(0, 4),
            12,
            Some(&Filter::or(vec![
                Filter::eq("tag", "lang"),
                Filter::eq("tag", "data"),
            ])),
        )
        .unwrap();
    assert_eq!(results.len(), 4, "lang + data: d00, d01, d02, d04");

    // ── Not ────────────────────────────────────────────────────────
    let results = col
        .search(
            &axis(0, 4),
            12,
            Some(&Filter::not(Filter::eq("active", true))),
        )
        .unwrap();
    assert_eq!(results.len(), 4, "inactive docs: d02, d05, d08, d11");
    assert!(results.iter().all(|r| r.metadata.as_ref().unwrap()["active"] == false));

    // ── Exists ─────────────────────────────────────────────────────
    let results = col
        .search(&axis(0, 4), 12, Some(&Filter::exists("metrics.score")))
        .unwrap();
    assert_eq!(results.len(), 12, "all docs have metrics.score");

    let results = col
        .search(
            &axis(0, 4),
            12,
            Some(&Filter::exists("metrics.nonexistent")),
        )
        .unwrap();
    assert_eq!(results.len(), 0, "no docs have metrics.nonexistent");

    // ── Nested dot-notation Gt ─────────────────────────────────────
    let results = col
        .search(
            &axis(0, 4),
            12,
            Some(&Filter::gt("metrics.score", json!(89.0))),
        )
        .unwrap();
    // docs with score > 89: d00(90), d03(95), d06(88 is NOT > 89), d09(91)
    // Actually d06 = 88 < 89, so: d00, d03, d09 = 3 docs
    assert_eq!(results.len(), 3, "score > 89: d00(90), d03(95), d09(91)");

    // ── Compound And+Or ────────────────────────────────────────────
    // (tag == "server" OR tag == "web") AND active == false
    let results = col
        .search(
            &axis(0, 4),
            12,
            Some(&Filter::and(vec![
                Filter::or(vec![
                    Filter::eq("tag", "server"),
                    Filter::eq("tag", "web"),
                ]),
                Filter::eq("active", false),
            ])),
        )
        .unwrap();
    // server+inactive: d05, d11; web+inactive: d08 → 3 docs
    assert_eq!(results.len(), 3, "inactive server/web docs: d05, d08, d11");
}

// ── Scenario 6: Hybrid search strategies ──────────────────────────────────

/// Insert docs with varied content and embeddings → compare RRF vs WeightedSum.
#[test]
fn scenario_hybrid_search_strategies() {
    let dir = TempDir::new().unwrap();
    let store = VectorStore::open(dir.path(), test_cfg()).unwrap();
    let col = store.get_or_create_collection("hybrid").unwrap();

    // Insert docs where some are strong vector matches and some are strong text matches.
    let docs = [
        // Strong vector match for axis 0 query, weak text match.
        ("vec-a", "unrelated noise filler content here", [1.0, 0.0, 0.0, 0.0_f32]),
        ("vec-b", "more noise unrelated terms filler", [0.99, 0.01, 0.0, 0.0]),
        // Strong text match, moderate vector match.
        ("txt-a", "rust async runtime channel message", [0.5, 0.5, 0.0, 0.0]),
        ("txt-b", "rust channel async send receive", [0.4, 0.6, 0.0, 0.0]),
        // Both strong vector and text matches — should rank high in both strategies.
        ("both-a", "rust async runtime fast", [0.95, 0.05, 0.0, 0.0]),
        // Irrelevant to both.
        ("none-a", "completely different topic cooking", [0.0, 0.0, 1.0, 0.0]),
        ("none-b", "gardening and flowers spring time", [0.0, 0.0, 0.0, 1.0]),
    ];

    for (id, content, emb) in &docs {
        col.insert(id.to_string(), content.to_string(), emb.to_vec(), None)
            .unwrap();
    }
    assert_eq!(col.len(), 7);

    let query_vec = [1.0f32, 0.0, 0.0, 0.0];
    let query_text = "rust async";
    let top_k = 5;

    // RRF strategy.
    let rrf_config = HybridConfig::default().with_strategy(MergeStrategy::Rrf);
    let rrf_results = col
        .search_hybrid(&query_vec, query_text, top_k, None, Some(&rrf_config))
        .unwrap();

    // WeightedSum strategy.
    let ws_config = HybridConfig::default().with_strategy(MergeStrategy::WeightedSum);
    let ws_results = col
        .search_hybrid(&query_vec, query_text, top_k, None, Some(&ws_config))
        .unwrap();

    // Both strategies must return a non-empty result set.
    assert!(!rrf_results.is_empty(), "RRF must return results");
    assert!(!ws_results.is_empty(), "WeightedSum must return results");

    // Both must respect the k limit.
    assert!(rrf_results.len() <= top_k);
    assert!(ws_results.len() <= top_k);

    // The "both" doc (strong vector + text) should appear in both result sets.
    let rrf_ids: Vec<&str> = rrf_results.iter().map(|r| r.id.as_str()).collect();
    let ws_ids: Vec<&str> = ws_results.iter().map(|r| r.id.as_str()).collect();
    assert!(
        rrf_ids.contains(&"both-a"),
        "RRF must include both-a (strong on both axes); got {rrf_ids:?}"
    );
    assert!(
        ws_ids.contains(&"both-a"),
        "WeightedSum must include both-a; got {ws_ids:?}"
    );

    // Scores must be finite and positive.
    for r in rrf_results.iter().chain(ws_results.iter()) {
        assert!(r.score.is_finite(), "score must be finite for {}", r.id);
        assert!(r.score >= 0.0, "score must be non-negative for {}", r.id);
    }

    // "none" docs (poor on both axes) should NOT be in top-3 results.
    let rrf_top3_ids: Vec<&str> = rrf_results.iter().take(3).map(|r| r.id.as_str()).collect();
    assert!(
        !rrf_top3_ids.contains(&"none-a") && !rrf_top3_ids.contains(&"none-b"),
        "irrelevant docs should not dominate top-3 in RRF; got {rrf_top3_ids:?}"
    );
}

// ── Scenario 7: Concurrent access ─────────────────────────────────────────

/// Spawn multiple threads doing insert/search/delete on the same collection.
/// Verify no panics and no data corruption.
#[test]
fn scenario_concurrent_access() {
    use std::sync::Arc;
    use std::thread;

    let dir = TempDir::new().unwrap();
    let store = Arc::new(VectorStore::open(dir.path(), test_cfg()).unwrap());
    let col = Arc::new(store.get_or_create_collection("concurrent").unwrap());

    // Seed some initial docs so searches have something to work with.
    for i in 0..10usize {
        col.insert(
            format!("seed-{i}"),
            format!("seed document content {i}"),
            axis(i % 4, 4),
            Some(json!({ "group": "seed" })),
        )
        .unwrap();
    }

    let mut handles = Vec::new();

    // 4 writer threads — each inserts 10 docs with unique IDs.
    for thread_id in 0..4usize {
        let col = Arc::clone(&col);
        let h = thread::spawn(move || {
            for j in 0..10usize {
                let id = format!("t{thread_id}-doc-{j}");
                col.insert(
                    id,
                    format!("concurrent document from thread {thread_id} item {j}"),
                    axis((thread_id + j) % 4, 4),
                    Some(json!({ "thread": thread_id, "item": j })),
                )
                .expect("concurrent insert must not fail");
            }
        });
        handles.push(h);
    }

    // 2 reader threads — search repeatedly.
    for ax in [0usize, 1usize] {
        let col = Arc::clone(&col);
        let h = thread::spawn(move || {
            for _ in 0..20 {
                // May return 0 results if collection is briefly empty — that is fine.
                let _ = col.search(&axis(ax, 4), 5, None);
                let _ = col.search_text("document", 5, None);
            }
        });
        handles.push(h);
    }

    // 1 delete thread — deletes seed docs.
    {
        let col = Arc::clone(&col);
        let h = thread::spawn(move || {
            for i in 0..10usize {
                // Ignore errors — doc may not exist yet or may be double-deleted.
                let _ = col.delete(&format!("seed-{i}"));
            }
        });
        handles.push(h);
    }

    // All threads must finish without panicking.
    for h in handles {
        h.join().expect("thread must not panic");
    }

    // 4 writer threads * 10 docs each = 40 writer docs, 0-10 seeds remain.
    let live = col.len();
    assert!(
        live >= 40,
        "at least 40 writer docs must be present; got {live}"
    );
    assert!(live <= 50, "at most 50 total docs; got {live}");

    // No data corruption: each writer doc must be retrievable.
    for thread_id in 0..4usize {
        for j in 0..10usize {
            let id = format!("t{thread_id}-doc-{j}");
            let doc = col
                .get(&id)
                .expect("get must not error")
                .unwrap_or_else(|| panic!("{id} must be present after all writes completed"));
            assert!(
                doc.content.contains(&format!("thread {thread_id}")),
                "document content corrupted for {id}"
            );
        }
    }
}

// ── Scenario 8: Edge cases ─────────────────────────────────────────────────

/// Empty collection search, search after deleting all docs, rapid upserts,
/// zero-length content.
#[test]
fn scenario_edge_cases() {
    let dir = TempDir::new().unwrap();
    let store = VectorStore::open(dir.path(), test_cfg()).unwrap();
    let col = store.get_or_create_collection("edge").unwrap();

    // ── 8a: Search on empty collection returns empty, not an error. ─
    let v = col.search(&axis(0, 4), 5, None).unwrap();
    assert!(v.is_empty(), "vector search on empty collection must return []");

    let t = col.search_text("anything", 5, None).unwrap();
    assert!(t.is_empty(), "text search on empty collection must return []");

    let h = col
        .search_hybrid(&axis(0, 4), "anything", 5, None, None)
        .unwrap();
    assert!(h.is_empty(), "hybrid search on empty collection must return []");

    // ── 8b: k=0 search returns empty immediately. ───────────────────
    col.insert("x".into(), "some content here".into(), axis(0, 4), None)
        .unwrap();
    let v = col.search(&axis(0, 4), 0, None).unwrap();
    assert!(v.is_empty(), "k=0 vector search must return []");

    let h = col
        .search_hybrid(&axis(0, 4), "content", 0, None, None)
        .unwrap();
    assert!(h.is_empty(), "k=0 hybrid search must return []");
    col.delete("x").unwrap();

    // ── 8c: Search after deleting all docs returns empty. ───────────
    for i in 0..5usize {
        col.insert(
            format!("tmp-{i}"),
            format!("temporary document {i}"),
            axis(i % 4, 4),
            None,
        )
        .unwrap();
    }
    assert_eq!(col.len(), 5);

    for i in 0..5usize {
        col.delete(&format!("tmp-{i}")).unwrap();
    }
    assert_eq!(col.len(), 0);

    let v = col.search(&axis(0, 4), 5, None).unwrap();
    assert!(v.is_empty(), "vector search after all-delete must return []");

    let t = col.search_text("temporary", 5, None).unwrap();
    assert!(t.is_empty(), "text search after all-delete must return []");

    // ── 8d: Upsert same doc 10 times rapidly — count stays at 1. ────
    for i in 0..10usize {
        col.upsert(
            "upsert-target".into(),
            format!("upserted version {i}"),
            axis(i % 4, 4),
            Some(json!({ "version": i })),
        )
        .unwrap();
    }
    assert_eq!(col.len(), 1, "10 upserts of same ID must produce 1 live doc");

    let doc = col.get("upsert-target").unwrap().unwrap();
    assert_eq!(
        doc.content, "upserted version 9",
        "last upsert (version 9) must win"
    );
    assert_eq!(doc.metadata.as_ref().unwrap()["version"], 9);
    col.delete("upsert-target").unwrap();

    // ── 8e: Insert with empty content — valid, text search returns nothing. ─
    col.insert(
        "empty-content".into(),
        String::new(),
        axis(0, 4),
        None,
    )
    .unwrap();
    assert_eq!(col.len(), 1);

    // Vector search still finds it.
    let results = col.search(&axis(0, 4), 1, None).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "empty-content");

    // Text search for any term finds nothing (empty content has no tokens).
    let results = col.search_text("anything", 5, None).unwrap();
    assert!(results.is_empty(), "empty-content doc must not appear in text search");

    // ── 8f: Dimension mismatch returns an error, not a panic. ───────
    let wrong_dim_result = col.insert(
        "wrong-dim".into(),
        "will fail".into(),
        vec![1.0, 2.0],  // dim=2 but store expects dim=4
        None,
    );
    assert!(
        wrong_dim_result.is_err(),
        "dimension mismatch must return Err, not panic"
    );

    let wrong_dim_search = col.search(&[1.0, 0.0], 5, None);
    assert!(
        wrong_dim_search.is_err(),
        "search with wrong dimension must return Err"
    );

    // ── 8g: Delete returns false for non-existent ID. ───────────────
    let removed = col.delete("nonexistent-id").unwrap();
    assert!(!removed, "delete of non-existent ID must return false");

    // ── 8h: list_ids and contains reflect live docs accurately. ─────
    col.insert("a".into(), "content a".into(), axis(0, 4), None).unwrap();
    col.insert("b".into(), "content b".into(), axis(1, 4), None).unwrap();
    col.delete("a").unwrap();

    let ids = col.list_ids();
    assert!(ids.contains(&"b".to_string()), "b must be in list_ids");
    assert!(!ids.contains(&"a".to_string()), "deleted a must not be in list_ids");

    assert!(col.contains("b"), "contains must be true for live doc b");
    assert!(!col.contains("a"), "contains must be false for deleted doc a");
}

// ── Scenario 9: Upsert semantics and WAL consistency ──────────────────────

/// Verify upsert correctly handles the WAL (each version gets a WAL entry) and
/// the text index (old tokens removed, new tokens added) on re-open.
#[test]
fn scenario_upsert_wal_consistency() {
    let dir = TempDir::new().unwrap();

    // Phase 1: upsert same ID multiple times, then flush.
    {
        let store = VectorStore::open(dir.path(), test_cfg()).unwrap();
        let col = store.get_or_create_collection("upsert_wal").unwrap();

        col.upsert(
            "doc-a".into(),
            "original rust content for testing".into(),
            vec![1.0, 0.0, 0.0, 0.0],
            Some(json!({ "version": 1 })),
        )
        .unwrap();

        col.upsert(
            "doc-a".into(),
            "updated python content for testing".into(),
            vec![0.0, 1.0, 0.0, 0.0],
            Some(json!({ "version": 2 })),
        )
        .unwrap();

        col.upsert(
            "doc-a".into(),
            "final go content for production".into(),
            vec![0.0, 0.0, 1.0, 0.0],
            Some(json!({ "version": 3 })),
        )
        .unwrap();

        assert_eq!(col.len(), 1, "3 upserts of same ID must leave 1 live doc");

        let doc = col.get("doc-a").unwrap().unwrap();
        assert_eq!(doc.content, "final go content for production");
        assert_eq!(doc.metadata.unwrap()["version"], 3);

        // Text search: old terms "rust" and "python" must not match this doc.
        let rust_results = col.search_text("rust", 5, None).unwrap();
        assert!(
            rust_results.is_empty(),
            "old term 'rust' must not match after upsert"
        );

        let python_results = col.search_text("python", 5, None).unwrap();
        assert!(
            python_results.is_empty(),
            "old term 'python' must not match after upsert"
        );

        // New terms do match.
        let go_results = col.search_text("production", 5, None).unwrap();
        assert_eq!(go_results.len(), 1);
        assert_eq!(go_results[0].id, "doc-a");

        store.flush_all().unwrap();
    }

    // Phase 2: reopen and verify the final upsert state survived.
    {
        let store = VectorStore::open(dir.path(), test_cfg()).unwrap();
        let col = store.get_collection("upsert_wal").unwrap();

        assert_eq!(col.len(), 1);
        let doc = col.get("doc-a").unwrap().unwrap();
        assert_eq!(doc.content, "final go content for production");

        // Text index rebuilt correctly on recovery — old terms absent.
        let rust_results = col.search_text("rust", 5, None).unwrap();
        assert!(rust_results.is_empty(), "rust term must be absent after reopen");

        let go_results = col.search_text("production", 5, None).unwrap();
        assert_eq!(go_results.len(), 1);
    }
}
