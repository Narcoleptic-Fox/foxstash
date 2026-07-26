//! CI guard for the Collection lifecycle.
//!
//! `examples/lifecycle.rs` measures at scale and reports; it does not run in CI, so
//! nothing protected the properties it exercises against regression. This is the
//! same shape, sized to run in seconds, asserting only what must never break.
//!
//! Deliberately asserts **behaviour, not timings**. A wall-clock threshold in CI
//! either flakes on a loaded runner or gets loosened until it means nothing. The
//! numbers live in the example, run by hand on an idle machine; this guards the
//! invariants those numbers depend on.

use foxstash_core::Document;
use foxstash_db::{Collection, DbConfig};
use tempfile::TempDir;

fn vector(seed: u64, dim: usize) -> Vec<f32> {
    // Dense pseudo-random. Not modular or one-hot: the default metric is Cosine,
    // so structured fixtures produce ties and a "wrong" nearest neighbour that is
    // really a coin flip. Three separate tests were derailed by that.
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..dim)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) as f32 / (1u32 << 31) as f32) - 0.5
        })
        .collect()
}

fn docs(n: usize, dim: usize) -> Vec<Document> {
    (0..n)
        .map(|i| Document {
            id: format!("doc-{i}"),
            content: format!("document number {i} about topic {}", i % 13),
            embedding: vector(i as u64, dim),
            metadata: None,
        })
        .collect()
}

/// Ingest, restart, search, delete, compact, restart again — the whole lifecycle.
///
/// The reopen leg is the one with teeth: it goes through the graph snapshot, which
/// is a cache that must be transparent. If it ever returns something different
/// from a rebuild, this fails.
#[test]
fn collection_survives_a_full_lifecycle() {
    let dim = 16;
    let n = 400;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("c");
    std::fs::create_dir_all(&path).unwrap();
    let cfg = DbConfig::default().with_embedding_dim(dim);
    let corpus = docs(n, dim);

    {
        let c = Collection::create("c", &path, cfg.clone()).unwrap();
        c.insert_many(corpus.clone()).unwrap();
        c.flush().unwrap();
        assert_eq!(c.len(), n);
    }

    let c = Collection::open("c", &path, cfg.clone()).unwrap();
    assert_eq!(c.len(), n, "reopen must restore every document");
    for d in corpus.iter().take(50) {
        let hits = c.search(&d.embedding, 1, None).unwrap();
        assert_eq!(
            hits[0].id, d.id,
            "{} should retrieve itself after reopen",
            d.id
        );
    }

    for d in corpus.iter().take(40) {
        c.delete(&d.id).unwrap();
    }
    c.compact().unwrap();
    assert_eq!(
        c.len(),
        n - 40,
        "compaction must drop exactly the deleted documents"
    );

    drop(c);
    let c = Collection::open("c", &path, cfg).unwrap();
    assert_eq!(c.len(), n - 40, "compaction must survive a restart");
    for d in corpus.iter().skip(40).take(30) {
        assert!(
            c.search(&d.embedding, 3, None)
                .unwrap()
                .iter()
                .any(|h| h.id == d.id),
            "{} should still be findable after compaction + restart",
            d.id
        );
    }
}

/// A quantized collection auto-fits from a plain insert loop and keeps working.
///
/// This is the default configuration, and it is the path that self-deadlocked:
/// `insert` holds the mutation lock and then reaches `fit`. A test that disables
/// auto-fit cannot catch that, which is exactly how it shipped.
#[test]
fn a_quantized_collection_auto_fits_and_keeps_serving() {
    use foxstash_core::index::{HNSWConfig, Storage};

    let dim = 16;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("c");
    std::fs::create_dir_all(&path).unwrap();
    let cfg = DbConfig {
        hnsw: HNSWConfig {
            storage: Storage::SQ8,
            ..Default::default()
        },
        fit_threshold: 100,
        ..DbConfig::default()
    }
    .with_embedding_dim(dim);

    let c = Collection::create("c", &path, cfg.clone()).unwrap();
    let corpus = docs(300, dim);
    for d in &corpus {
        c.insert(d.id.clone(), d.content.clone(), d.embedding.clone(), None)
            .unwrap();
    }
    assert_eq!(c.len(), 300, "auto-fit must not lose or block writes");

    // Quantized search is lossy, so assert membership in a small top-k rather than
    // rank 1 — demanding exactness here would be asserting that quantization does
    // not quantize.
    let found = corpus
        .iter()
        .take(100)
        .filter(|d| {
            c.search(&d.embedding, 5, None)
                .unwrap()
                .iter()
                .any(|h| h.id == d.id)
        })
        .count();
    assert!(
        found >= 90,
        "only {found}/100 documents found themselves after auto-fit"
    );

    c.compact().unwrap();
    assert_eq!(c.len(), 300, "compaction must work on a fitted collection");
    drop(c);
    let re = Collection::open("c", &path, cfg).unwrap();
    assert_eq!(re.len(), 300, "a fitted collection must survive a restart");
}
