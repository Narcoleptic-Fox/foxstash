//! Does reusing a search context buy anything? (No.)
//!
//! Run: `cargo run --release -p foxstash-benches --example search_api_cost [dataset]`
//!
//! This is the evidence behind the 1.0 search API. `HNSWIndex` used to expose four ways to
//! search — `search`, `search_batch`, `search_with_context`, `search_batch_fast` — and the
//! last two carried docstrings promising "~2-3x faster". Nothing had ever checked.
//!
//! `search()` allocates a fresh `SearchContext` per call: a visited bitset sized to the whole
//! index (125 KB at 1M nodes) plus two heaps. A reused context skips that. The saving is O(n)
//! and the search is only O(ef · log n)-ish, so if the claim were ever going to hold it would
//! hold at 1M. Measured, single-threaded, SIFT1M, SQ8 + rerank, ef=100:
//!
//! ```text
//!                                     QPS     us/query
//!                   search()         4118        242.8
//!      search_with_context()         4121        242.7     <- 1.00x
//!
//!             search_batch()        33948         29.5
//!        search_batch_fast()        32785         30.5     <- 0.97x, i.e. SLOWER
//! ```
//!
//! The bitset is the same size every call, so it comes straight back off the allocator's
//! free list and zeroing it costs ~2 us — against a query that spends ~234 us stalled on DRAM
//! for its ~3,500 distance computations. HNSW search is memory-latency bound, not allocation
//! bound. That is the same fact that makes `Storage::SQ8` a 1.20x win: shrink the bytes each
//! hop touches and everything moves; shave an allocation and nothing does.
//!
//! `search_with_context` and `search_batch_fast` were therefore deleted. `Searcher` survives
//! them, honestly documented, because `distance_calls()` is the diagnostic the project runs
//! on — not because it is fast.
//!
//! This example now guards that conclusion: it re-measures `search()` against `searcher()` on
//! the current code and ASSERTS they are within noise. If someone reintroduces a per-query
//! allocation heavy enough to matter, or the scratch stops being reused, this fails.
//!
//! Run on an IDLE machine.

use foxstash_benches::sift::Dataset;
use foxstash_core::index::hnsw::{BuildStrategy, DistanceMetric, HNSWConfig, HNSWIndex, Storage};
use std::time::Instant;

const K: usize = 10;

fn main() {
    let name = std::env::args().nth(1).unwrap_or_else(|| "sift1m".into());
    let ds = Dataset::load("benchmarks/data", &name).unwrap_or_else(|e| panic!("load {name}: {e}"));

    let index = HNSWIndex::build_parallel(
        ds.base.clone(),
        HNSWConfig {
            metric: DistanceMetric::L2,
            m: 32,
            m0: 64,
            ef_construction: 200,
            ef_search: 100,
            storage: Storage::SQ8,
            rerank_candidates: 100,
            build_strategy: BuildStrategy::Parallel,
            ..Default::default()
        },
    );

    println!(
        "{} — {} base x {}d, {} queries, k={K}, ef=100\n",
        ds.name,
        ds.base.len(),
        ds.dim(),
        ds.queries.len()
    );

    // Warm.
    let mut warm = index.searcher();
    for q in ds.queries.iter().take(100) {
        std::hint::black_box(warm.search(q, K).unwrap());
    }
    drop(warm);

    let t = Instant::now();
    for q in &ds.queries {
        std::hint::black_box(index.search(q, K).unwrap());
    }
    let plain = t.elapsed();

    let mut searcher = index.searcher();
    let t = Instant::now();
    for q in &ds.queries {
        std::hint::black_box(searcher.search(q, K).unwrap());
    }
    let reused = t.elapsed();

    let n = ds.queries.len() as f64;
    println!("{:>26} {:>12} {:>12}", "", "QPS", "us/query");
    println!("{:-<52}", "");
    println!(
        "{:>26} {:>12.0} {:>12.1}",
        "search()  (fresh scratch)",
        n / plain.as_secs_f64(),
        plain.as_micros() as f64 / n
    );
    println!(
        "{:>26} {:>12.0} {:>12.1}",
        "searcher() (reused)",
        n / reused.as_secs_f64(),
        reused.as_micros() as f64 / n
    );

    let speedup = plain.as_secs_f64() / reused.as_secs_f64();
    println!("\n  scratch reuse buys: {speedup:.2}x");
    println!(
        "  {} distance computations, {:.1} ns each",
        searcher.distance_calls(),
        reused.as_nanos() as f64 / searcher.distance_calls() as f64
    );

    // The load-bearing claim. Not "reuse is faster" — the opposite: that it is *irrelevant*,
    // because the search is latency-bound. A large deviation either way means the cost model
    // in the module docs above is wrong and the API decision needs revisiting.
    assert!(
        (0.85..=1.15).contains(&speedup),
        "scratch reuse measured {speedup:.2}x, expected ~1.00x. Either a per-query allocation \
         crept back into search(), or the searcher stopped reusing its scratch. The 1.0 API \
         collapse rests on this being ~1.00x — re-derive it before touching the API."
    );
    println!("\n  OK: within noise of 1.00x, as the memory-latency model predicts.");
}
