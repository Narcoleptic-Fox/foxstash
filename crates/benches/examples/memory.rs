//! Index memory footprint, measured as real RSS growth rather than estimated.
//!
//! Run: `cargo run --release -p foxstash-benches --example memory`
//!
//! Layer-0 connections used to live in *both* the nested `connections` structure and the
//! flat `connections_l0` array — a duplicate of the hottest data in the index. The flat
//! array is now the sole owner.
//!
//! hnswlib's own accounting for the same index (10k x 128d, M=32): vectors 5.12 MB +
//! links (m0=64) 2.56 MB + labels ~0.08 MB ~= 7.8 MB.

use foxstash_benches::sift::Dataset;
use foxstash_core::index::hnsw::{BuildStrategy, DistanceMetric, HNSWConfig, HNSWIndex};

/// Resident set size in bytes, from /proc/self/statm (page count * page size).
fn rss() -> usize {
    let statm = std::fs::read_to_string("/proc/self/statm").expect("read /proc/self/statm");
    let resident_pages: usize = statm
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .expect("parse statm");
    resident_pages * 4096
}

fn main() {
    let ds = Dataset::load("benchmarks/data", "sift10k").expect("load sift10k");
    let n = ds.base.len();
    let dim = ds.dim();

    // Drop the queries/truth we don't need, and take a baseline with the corpus already
    // resident, so we measure the *index*, not the dataset.
    let base = ds.base.clone();
    drop(ds);

    let before = rss();
    let index = HNSWIndex::build_parallel(
        base.clone(),
        HNSWConfig {
            metric: DistanceMetric::L2,
            m: 32,
            m0: 64,
            ef_construction: 200,
            ef_search: 100,
            build_strategy: BuildStrategy::Parallel,
            ..Default::default()
        },
    );
    let after = rss();

    let m = index.memory_breakdown();
    let mb = |b: usize| b as f64 / 1e6;
    let vectors = n * dim * 4;
    let links = n * 64 * 4;

    println!("SIFT10K index: {n} vectors x {dim}d, M=32 (m0=64)\n");
    println!("{:<40} {:>10}", "Retained by the index", "MB");
    println!("{:-<52}", "");
    println!("{:<40} {:>10.2}", "  embeddings (f32)", mb(m.embeddings));
    println!(
        "{:<40} {:>10.2}",
        "  layer-0 links (flat)",
        mb(m.layer0_links)
    );
    println!(
        "{:<40} {:>10.2}",
        "  upper-layer links (nested)",
        mb(m.upper_layer_links)
    );
    println!("{:<40} {:>10.2}", "  norms (cosine only)", mb(m.norms));
    println!(
        "{:<40} {:>10.2}",
        "  payload (ids + contents)",
        mb(m.payload)
    );
    println!("{:-<52}", "");
    println!("{:<40} {:>10.2}", "  TOTAL", mb(m.total()));
    println!();
    println!(
        "{:<40} {:>10.2}",
        "theoretical floor (vectors + links)",
        mb(vectors + links)
    );
    println!("{:<40} {:>10.2}", "hnswlib (its own accounting)", 7.80);
    println!(
        "{:<40} {:>10.2}",
        "(RSS delta incl. build transients)",
        mb(after.saturating_sub(before))
    );
    println!(
        "\nindex.len() = {}, so the graph is real and not empty.",
        index.len()
    );
}
