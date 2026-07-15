//! Where does search time actually go? Isolates API overhead from graph traversal.
//!
//! Run: `cargo run --release -p foxstash-benches --example search_profile`
//!
//! hnswlib serves ~32.7k QPS on SIFT10K at ef=500 where foxstash serves ~2.9k. This
//! splits foxstash's per-query cost into layers so we optimise the thing that's actually
//! expensive rather than the thing that looks expensive:
//!
//!   1. `search()`               — allocates a fresh SearchContext per query
//!   2. `search_with_context()`  — reuses the context, still builds SearchResult
//!   3. `search_batch()`         — rayon across queries
//!
//! Anything left over after (2) is graph traversal + distance, i.e. the real algorithm.

use foxstash_benches::sift::Dataset;
use foxstash_core::index::hnsw::{BuildStrategy, DistanceMetric, HNSWConfig, HNSWIndex};
use std::time::Instant;

const EF: usize = 500; // hnswlib's config for its 99.98% row
const K: usize = 10;

fn qps(n: usize, secs: f64) -> f64 {
    n as f64 / secs
}

fn main() {
    let ds = Dataset::load("benchmarks/data", "sift10k").expect("load sift10k");
    let index = HNSWIndex::build_parallel(
        ds.base.clone(),
        HNSWConfig {
            metric: DistanceMetric::L2,
            m: 32,
            m0: 64,
            ef_construction: 200,
            ef_search: EF,
            build_strategy: BuildStrategy::Parallel,
            ..Default::default()
        },
    );

    let n = ds.queries.len();
    println!("SIFT10K, {} queries, ef_search={EF}, k={K}\n", n);
    println!("{:<34} {:>10} {:>12}", "Path", "QPS", "vs hnswlib");
    println!("{:-<58}", "");
    let hnswlib = 32_715.0;
    let row = |label: &str, q: f64| {
        println!("{:<34} {:>10.0} {:>11.1}x", label, q, hnswlib / q);
    };

    // 1. The API everyone actually calls.
    let t = Instant::now();
    for q in &ds.queries {
        std::hint::black_box(index.search(q, K).unwrap());
    }
    row("search()", qps(n, t.elapsed().as_secs_f64()));

    // 2. Same work, but the per-query Searcher allocation is hoisted out.
    let mut searcher = index.searcher();
    let t = Instant::now();
    for q in &ds.queries {
        std::hint::black_box(searcher.search(q, K).unwrap());
    }
    row("search_with_context()", qps(n, t.elapsed().as_secs_f64()));

    // 3. Parallel across queries.
    let t = Instant::now();
    std::hint::black_box(index.search_batch(&ds.queries, K).unwrap());
    row("search_batch() [rayon]", qps(n, t.elapsed().as_secs_f64()));

    println!("\nhnswlib reference: {:.0} QPS (single-threaded)", hnswlib);
}
