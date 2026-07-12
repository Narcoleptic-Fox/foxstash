//! Does the graph cost more work than it needs to for the recall it gets?
//!
//! Run: `cargo run --release -p foxstash-benches --example graph_density [dataset]`
//!
//! Measurement (SIFT1M, matched ~98% recall): foxstash performs ~2,172 distance computations
//! per query where faiss performs ~1,925 — about 13% more work — while being *faster* per
//! distance (84 ns vs 98 ns). The inner loop is not the problem. The graph is: we walk more
//! nodes to reach the same recall.
//!
//! The prime suspect is `keep_pruned_connections`, which backfills a node's layer-0 list up to
//! `m0` with candidates the Algorithm-4 diversity heuristic explicitly rejected. hnswlib's
//! `getNeighborsByHeuristic2` simply drops them. Backfilling makes every hop scan more
//! neighbours, which buys recall per `ef` but costs distance computations per unit of recall —
//! precisely the signature measured.
//!
//! This sweeps the density knobs and reports, at each setting, both the recall and what it cost
//! to get there. The column that matters is **dist/query at matched recall**, not QPS at fixed
//! ef: a denser graph flatters itself on a fixed-ef comparison.

use foxstash_benches::sift::Dataset;
use foxstash_core::index::hnsw::{BuildStrategy, DistanceMetric, HNSWConfig, HNSWIndex};
use std::time::Instant;

const K: usize = 10;

struct Variant {
    label: &'static str,
    m: usize,
    m0: usize,
    keep_pruned: bool,
}

fn main() {
    let name = std::env::args().nth(1).unwrap_or_else(|| "sift100k".into());
    let ds = Dataset::load("benchmarks/data", &name).unwrap_or_else(|e| panic!("load {name}: {e}"));

    let control = ds.exact_control_sampled(K, 200);
    assert!(
        control > 0.99,
        "exact control failed at {:.2}% — harness broken, numbers below void",
        control * 100.0
    );

    println!(
        "{} — {} base x {}d, {} queries — single-threaded, k={K}, ef_construction=200\n",
        ds.name,
        ds.base.len(),
        ds.dim(),
        ds.queries.len()
    );

    let variants = [
        Variant {
            label: "m=32 m0=64 keep_pruned=ON  (current default)",
            m: 32,
            m0: 64,
            keep_pruned: true,
        },
        Variant {
            label: "m=32 m0=64 keep_pruned=OFF (hnswlib-like)   ",
            m: 32,
            m0: 64,
            keep_pruned: false,
        },
        Variant {
            label: "m=16 m0=32 keep_pruned=ON                   ",
            m: 16,
            m0: 32,
            keep_pruned: true,
        },
        Variant {
            label: "m=16 m0=32 keep_pruned=OFF                  ",
            m: 16,
            m0: 32,
            keep_pruned: false,
        },
    ];

    for v in &variants {
        let t = Instant::now();
        let mut index = HNSWIndex::build_parallel(
            ds.base.clone(),
            HNSWConfig {
                metric: DistanceMetric::L2,
                m: v.m,
                m0: v.m0,
                ef_construction: 200,
                ef_search: 100,
                keep_pruned_connections: v.keep_pruned,
                build_strategy: BuildStrategy::Parallel,
                ..Default::default()
            },
        );
        let build = t.elapsed().as_secs_f64();
        let mem = index.memory_breakdown().total() as f64 / 1e6;

        println!("{}   build {:.0}s, {:.0} MB", v.label, build, mem);
        println!(
            "  {:>5} {:>11} {:>9} {:>12} {:>9}",
            "ef", "recall@10", "QPS", "dist/query", "ns/dist"
        );

        for &ef in &[20usize, 50, 100, 200] {
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
            let dists = searcher.distance_calls() as f64;
            println!(
                "  {:>5} {:>10.2}% {:>9.0} {:>12.0} {:>9.1}",
                ef,
                recall * 100.0,
                n / elapsed.as_secs_f64(),
                dists / n,
                elapsed.as_nanos() as f64 / dists
            );
        }
        println!();
    }
}
