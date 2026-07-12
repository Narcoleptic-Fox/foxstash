//! Where does the time actually go? Distance computations per query, at matched recall.
//!
//! Run: `cargo run --release -p foxstash-benches --example work_per_query [dataset]`
//!
//! After interleaving the node arena, foxstash is at parity with faiss and ~10% behind
//! hnswlib on SIFT1M. The layout is no longer the difference — our node block is 784 bytes
//! against hnswlib's ~780 — and `get_unchecked` on the hot accessors bought 1.3%, i.e.
//! nothing. So the remaining gap is somewhere else, and there are exactly two somewheres:
//!
//!   1. **We do more work.** More distance computations to reach the same recall — a worse
//!      graph, or a search that stops too late. Shows up as a higher `dist/query`.
//!   2. **We do the same work more slowly.** Same distance count, worse ns-per-distance —
//!      a worse inner loop, or worse memory-latency hiding.
//!
//! These have completely different fixes, and you cannot tell them apart from QPS alone.
//! This prints `ns/dist`, which does.
//!
//! Compare against faiss, which exposes the identical counter as `hnsw_stats.ndis`:
//!   `benchmarks/python/work_per_query.py <dataset>`
//! (hnswlib's Python bindings do not expose theirs.)
//!
//! Run on an IDLE machine. A concurrent build halves the QPS and makes ns/dist a fiction.

use foxstash_benches::sift::Dataset;
use foxstash_core::index::hnsw::{BuildStrategy, DistanceMetric, HNSWConfig, HNSWIndex, Storage};
use std::time::Instant;

const K: usize = 10;

fn main() {
    let name = std::env::args().nth(1).unwrap_or_else(|| "sift1m".into());
    let ds = Dataset::load("benchmarks/data", &name).unwrap_or_else(|e| panic!("load {name}: {e}"));

    let control = ds.exact_control_sampled(K, 200);
    assert!(
        control > 0.99,
        "exact control failed at {:.2}% — harness is broken, numbers below would be void",
        control * 100.0
    );

    let (storage, rerank) = match std::env::args().nth(2).as_deref() {
        Some("sq8") => (Storage::SQ8, 100),
        Some("sq8-norerank") => (Storage::SQ8, 0),
        _ => (Storage::F32, 0),
    };
    let mut index = HNSWIndex::build_parallel(
        ds.base.clone(),
        HNSWConfig {
            metric: DistanceMetric::L2,
            m: 32,
            m0: 64,
            ef_construction: 200,
            ef_search: 100,
            storage,
            rerank_candidates: rerank,
            build_strategy: BuildStrategy::Parallel,
            ..Default::default()
        },
    );
    let mem = index.memory_breakdown();
    println!(
        "storage={storage:?} rerank={rerank}  index {:.0} MB\n",
        mem.total() as f64 / 1e6
    );

    println!(
        "{} — {} base x {}d, {} queries — single-threaded, k={K}, M=32\n",
        ds.name,
        ds.base.len(),
        ds.dim(),
        ds.queries.len()
    );
    println!(
        "{:>6} {:>11} {:>10} {:>12} {:>10}",
        "ef", "recall@10", "QPS", "dist/query", "ns/dist"
    );
    println!("{:-<54}", "");

    for &ef in &[10usize, 20, 50, 100, 200, 500] {
        index.set_ef_search(ef);

        let mut searcher = index.searcher();
        for q in ds.queries.iter().take(50) {
            std::hint::black_box(searcher.search(q, K).unwrap());
        }

        let recall = ds.recall_at(K, |q| {
            index
                .search(q, K)
                .unwrap()
                .into_iter()
                .filter_map(|r| r.id.parse::<usize>().ok())
                .collect()
        });

        let mut searcher = index.searcher();
        searcher.reset_stats();
        let t = Instant::now();
        for q in &ds.queries {
            std::hint::black_box(searcher.search(q, K).unwrap());
        }
        let elapsed = t.elapsed();

        let n = ds.queries.len() as f64;
        let qps = n / elapsed.as_secs_f64();
        let dists = searcher.distance_calls() as f64;
        let per_query = dists / n;
        let ns_per_dist = elapsed.as_nanos() as f64 / dists;

        println!(
            "{:>6} {:>10.2}% {:>10.0} {:>12.0} {:>10.1}",
            ef,
            recall * 100.0,
            qps,
            per_query,
            ns_per_dist
        );
    }

    println!(
        "\nns/dist near DRAM latency (~80-100ns) means the search is memory-latency bound and\n\
         the lever is fewer/cheaper reads (compressed traversal), not a faster kernel."
    );
}
