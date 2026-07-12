//! Does compressed traversal + exact rerank actually beat full-precision HNSW?
//!
//! Run: `cargo run --release -p foxstash-benches --example sq8_pareto [dataset]`
//!
//! The thesis: foxstash's search is memory-latency bound. Measured on SIFT1M, a distance
//! computation costs 77-98 ns — essentially one DRAM round-trip — and foxstash already runs
//! *faster per distance* than faiss (84 ns vs 98). The only lever left is to move fewer
//! bytes: traverse the graph on 8-bit codes (a quarter the vector bytes), then rescore the
//! final candidate pool against the retained f32 vectors so recall survives.
//!
//! This measures whether that trade actually pays, against full-precision HNSW as the
//! control. Both are foxstash, same machine, same queries, same ground truth, so the only
//! variable is the storage/traversal scheme.
//!
//! Run on an IDLE machine — a concurrent build halves the QPS and makes all of this fiction.

use foxstash_benches::sift::Dataset;
use foxstash_core::index::hnsw::{BuildStrategy, DistanceMetric, HNSWConfig, HNSWIndex};
use foxstash_core::index::{QuantizedHNSWConfig, SQ8HNSWIndex};
use foxstash_core::Document;
use std::time::Instant;

const K: usize = 10;

fn doc(i: usize, v: &[f32]) -> Document {
    Document {
        id: i.to_string(),
        content: String::new(),
        embedding: v.to_vec(),
        metadata: None,
    }
}

fn main() {
    let name = std::env::args().nth(1).unwrap_or_else(|| "sift100k".into());
    let ds = Dataset::load("benchmarks/data", &name).unwrap_or_else(|e| panic!("load {name}: {e}"));

    let control = ds.exact_control_sampled(K, 200);
    assert!(
        control > 0.99,
        "exact control failed at {:.2}% — harness broken, everything below is void",
        control * 100.0
    );

    println!(
        "{} — {} base x {}d, {} queries — single-threaded, k={K}, M=32, ef_construction=200",
        ds.name,
        ds.base.len(),
        ds.dim(),
        ds.queries.len()
    );
    println!("exact control: {:.2}%  PASS\n", control * 100.0);

    // ---- Control: full-precision HNSW ----
    let t = Instant::now();
    let mut hnsw = HNSWIndex::build_parallel(
        ds.base.clone(),
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
    let hnsw_build = t.elapsed().as_secs_f64();
    let hnsw_mem = hnsw.memory_breakdown().total() as f64 / 1e6;

    println!("=== full-precision HNSW (control) ===  build {hnsw_build:.0}s, {hnsw_mem:.0} MB");
    println!(
        "{:>6} {:>11} {:>10} {:>12} {:>9}",
        "ef", "recall@10", "QPS", "dist/query", "ns/dist"
    );
    let mut ctx = hnsw.create_search_context();
    for &ef in &[20usize, 50, 100, 200] {
        hnsw.set_ef_search(ef);
        for q in ds.queries.iter().take(50) {
            std::hint::black_box(hnsw.search_with_context(q, K, &mut ctx).unwrap());
        }
        let recall = ds.recall_at(K, |q| {
            hnsw.search(q, K)
                .unwrap()
                .into_iter()
                .filter_map(|r| r.id.parse::<usize>().ok())
                .collect()
        });
        ctx.reset_stats();
        let t = Instant::now();
        for q in &ds.queries {
            std::hint::black_box(hnsw.search_with_context(q, K, &mut ctx).unwrap());
        }
        let el = t.elapsed();
        let n = ds.queries.len() as f64;
        let d = ctx.distance_calls() as f64;
        println!(
            "{:>6} {:>10.2}% {:>10.0} {:>12.0} {:>9.1}",
            ef,
            recall * 100.0,
            n / el.as_secs_f64(),
            d / n,
            el.as_nanos() as f64 / d
        );
    }
    drop(hnsw);

    // ---- SQ8: 8-bit traversal, exact rerank against retained f32 vectors ----
    println!("\n=== SQ8 HNSW — 8-bit traversal + exact rerank ===");
    let t = Instant::now();
    let cfg = QuantizedHNSWConfig {
        m: 32,
        m0: 64,
        ef_construction: 200,
        ef_search: 100,
        ..Default::default()
    };
    let mut sq8 = SQ8HNSWIndex::fit(&ds.base, cfg);
    for (i, v) in ds.base.iter().enumerate() {
        sq8.add_with_full_precision(doc(i, v)).unwrap();
    }
    let sq8_build = t.elapsed().as_secs_f64();
    println!(
        "build {sq8_build:.0}s, {:.0} MB (vs {hnsw_mem:.0} MB full precision)",
        sq8.memory_usage() as f64 / 1e6
    );
    println!("\n{:>10} {:>11} {:>10}", "rerank_n", "recall@10", "QPS");
    println!("{:-<33}", "");

    // Sweep the rerank pool: how many coarse candidates get rescored exactly.
    for &pool in &[10usize, 50, 100, 200, 400] {
        for q in ds.queries.iter().take(50) {
            std::hint::black_box(sq8.search_and_rerank(q, pool, K).unwrap());
        }
        let recall = ds.recall_at(K, |q| {
            sq8.search_and_rerank(q, pool, K)
                .unwrap()
                .into_iter()
                .filter_map(|r| r.id.parse::<usize>().ok())
                .collect()
        });
        let t = Instant::now();
        for q in &ds.queries {
            std::hint::black_box(sq8.search_and_rerank(q, pool, K).unwrap());
        }
        let qps = ds.queries.len() as f64 / t.elapsed().as_secs_f64();
        println!("{:>10} {:>10.2}% {:>10.0}", pool, recall * 100.0, qps);
    }

    println!(
        "\nCompare against the control at MATCHED RECALL, not at matched ef or matched pool.\n\
         If SQ8's recall/QPS curve does not dominate full-precision HNSW's, compressed\n\
         traversal is not paying for itself on this dataset and should not be the default."
    );
}
