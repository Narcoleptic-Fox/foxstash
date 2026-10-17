//! Resident-memory saving from externalized rerank vectors (issue #58).
//!
//! Run: `cargo run --release -p foxstash-benches --example external_rerank_memory -- 45000 768`
//! (defaults to 20000 x 768 if no args.)
//!
//! With reranking on, a quantized `HNSWIndex` keeps a full f32 copy of every vector in its cold
//! `full` pool, to rescore the candidate pool exactly. When the index lives inside a `Collection`,
//! that is a *second* copy of vectors the Collection's document store already holds. The external
//! build drops the index's copy and sources the vectors from the document store at search time, so
//! the f32 lives once instead of twice.
//!
//! This builds the same corpus both ways and prints the index-side footprint from
//! `memory_breakdown()` (no RSS noise — this is the exact structural delta), which is precisely the
//! per-collection saving, plus the RSS to show it is real. The delta is `n * dim * 4` bytes.

use foxstash_core::index::{DistanceMetric, HNSWConfig, HNSWIndex, Storage};
use foxstash_core::Document;

fn rss_mib() -> f64 {
    std::fs::read_to_string("/proc/self/statm")
        .ok()
        .and_then(|s| {
            s.split_whitespace()
                .nth(1)
                .and_then(|p| p.parse::<f64>().ok())
        })
        .map(|pages| pages * 4096.0 / (1024.0 * 1024.0))
        .unwrap_or(f64::NAN)
}

fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

/// Deterministic unit-ish pseudo-random vector (no `rand` dep, reproducible).
fn vector(seed: u64, dim: usize) -> Vec<f32> {
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

fn main() {
    let mut args = std::env::args().skip(1);
    let n: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(20_000);
    let dim: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(768);

    let docs: Vec<Document> = (0..n)
        .map(|i| Document {
            id: i.to_string(),
            content: String::new(),
            embedding: vector(i as u64, dim),
            metadata: None,
        })
        .collect();

    let cfg = HNSWConfig {
        metric: DistanceMetric::Cosine,
        storage: Storage::TurboRabit,
        rabit_bits: 4,
        rerank_candidates: 100,
        ..Default::default()
    };

    println!("external-rerank memory — {n} docs x {dim} dim, TurboRabit4 + rerank 100");
    println!(
        "expected delta (one f32 copy): {:.1} MiB\n",
        mib(n * dim * 4)
    );

    // Standalone: the index owns its f32 rerank pool.
    let base_rss = rss_mib();
    let standalone = HNSWIndex::build_parallel_from_documents(docs.clone(), cfg.clone());
    let standalone_total = standalone.memory_breakdown().total();
    let standalone_rss = rss_mib();
    drop(standalone);

    // External: the index keeps no f32; vectors would come from the Collection's document store.
    let (external, _node_doc) =
        HNSWIndex::build_parallel_from_documents_external(docs.clone(), cfg.clone());
    let external_total = external.memory_breakdown().total();

    println!(
        "index footprint (memory_breakdown):\n  standalone (owns full): {:8.1} MiB\n  external   (no full):   {:8.1} MiB\n  saved:                  {:8.1} MiB",
        mib(standalone_total),
        mib(external_total),
        mib(standalone_total.saturating_sub(external_total)),
    );
    println!(
        "\nRSS: baseline {:.1} MiB -> after standalone build {:.1} MiB (corpus + index + full)",
        base_rss, standalone_rss
    );
    assert!(
        external.vectors_external(),
        "external build must flag the index external"
    );
}
