//! Matched-recall storage comparison at an ARBITRARY dimension.
//!
//! Run: `cargo run --release -p foxstash-benches --example dim_pareto [dataset] [dim] [n]`
//!   e.g. `... --example dim_pareto gist1m 384 200000`     (MiniLM's dimension)
//!
//! This is the cross of the other two examples, and it exists because neither of them can
//! answer the question that actually matters:
//!
//! * `storage_pareto` sweeps `ef` and compares at matched recall — but only at whatever
//!   dimension the corpus happens to ship with (128 or 960). Nothing in between.
//! * `dim_crossover` truncates the dimension — but pins `ef=100`. At a fixed `ef` a mode that
//!   is fast *because it stopped finding things* looks good. RaBitQ recalls 63.7% at 64-d;
//!   its QPS there is the QPS of giving up early. Those columns cannot be read as a win.
//!
//! So: truncate the dimension AND sweep `ef`, and read the frontier. A quantizer that needs
//! more hops to reach a given recall pays for its cheaper distance kernel in `dist/query`, and
//! matched recall is the only comparison where that bill actually lands.
//!
//! Ground truth is recomputed exactly at the truncated dimension — the shipped GT indexes the
//! full space and is void for a prefix.
//!
//! Run on an IDLE machine.

use foxstash_benches::sift::Dataset;
use foxstash_core::index::hnsw::{BuildStrategy, DistanceMetric, HNSWConfig, HNSWIndex, Storage};
use rayon::prelude::*;
use std::time::Instant;

const K: usize = 10;
const EFS: &[usize] = &[20, 50, 100, 200, 400, 800];

/// Exact top-k by brute force in the truncated space, **under the metric being tested**. The
/// oracle. Computing this under L2 while the index searches under cosine would score every mode
/// against the wrong answer — a silent, total invalidation, and exactly the kind of mistake the
/// control rows exist to catch.
fn exact_gt(
    base: &[Vec<f32>],
    queries: &[Vec<f32>],
    k: usize,
    metric: DistanceMetric,
) -> Vec<Vec<usize>> {
    queries
        .par_iter()
        .map(|q| {
            let qn: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
            let mut d: Vec<(f32, usize)> = base
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let s = match metric {
                        DistanceMetric::L2 => v
                            .iter()
                            .zip(q)
                            .map(|(a, b)| {
                                let x = a - b;
                                x * x
                            })
                            .sum::<f32>(),
                        _ => {
                            let dot: f32 = v.iter().zip(q).map(|(a, b)| a * b).sum();
                            let vn: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                            if vn == 0.0 || qn == 0.0 {
                                1.0
                            } else {
                                1.0 - dot / (vn * qn)
                            }
                        }
                    };
                    (s, i)
                })
                .collect();
            d.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
            d.into_iter().take(k).map(|(_, i)| i).collect()
        })
        .collect()
}

fn main() {
    let name = std::env::args().nth(1).unwrap_or_else(|| "gist1m".into());
    let dim: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(384);
    let n: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200_000);
    // 4th arg: "cosine" | "l2" (default l2).
    //
    // This exists because EVERY crossover number this project has published was measured under
    // L2 — while real RAG embeddings (MiniLM, OpenAI, GloVe, and 5 of 6 standard ANN benchmark
    // datasets) are compared by COSINE. The storage-mode rule of thumb in the README was
    // extrapolated across that gap without anybody noticing. Same species of error as reading a
    // 128-d SIFT result and generalizing it to 960-d: right experiment, wrong regime.
    let metric = match std::env::args().nth(4).as_deref() {
        Some("cosine") => DistanceMetric::Cosine,
        _ => DistanceMetric::L2,
    };

    let ds = Dataset::load("benchmarks/data", &name).unwrap_or_else(|e| panic!("load {name}: {e}"));
    assert!(
        dim <= ds.dim(),
        "asked for {dim}d but {} is only {}d — cannot truncate upward",
        ds.name,
        ds.dim()
    );
    let n = n.min(ds.base.len());
    let nq = 500.min(ds.queries.len());

    let base: Vec<Vec<f32>> = ds.base[..n].iter().map(|v| v[..dim].to_vec()).collect();
    let queries: Vec<Vec<f32>> = ds.queries[..nq].iter().map(|v| v[..dim].to_vec()).collect();
    let truth = exact_gt(&base, &queries, K, metric);

    let hdr = 272usize;
    println!(
        "{} truncated to {dim}d — {n} base, {nq} queries, k={K}, M=32/m0=64, ef_c=200\n\
         metric: {metric:?}   (exact ground truth recomputed at {dim}d UNDER THIS METRIC)\n\n\
         node block:  F32 {} B   SQ8 {} B   RaBitQ {} B\n",
        ds.name,
        hdr + dim * 4,
        hdr + dim,
        hdr + dim.div_ceil(8) + 8
    );

    for (label, storage, rerank) in [
        ("F32 (control)", Storage::F32, 0),
        ("SQ8 + rerank", Storage::SQ8, 100),
        ("RaBitQ + rerank", Storage::RaBitQ, 400),
    ] {
        let t = Instant::now();
        let mut index = HNSWIndex::build_parallel(
            base.clone(),
            HNSWConfig {
                metric,
                m: 32,
                m0: 64,
                ef_construction: 200,
                storage,
                rerank_candidates: rerank,
                build_strategy: BuildStrategy::Parallel,
                ..Default::default()
            },
        );
        let build = t.elapsed();
        let mem = index.memory_breakdown().total() as f64 / 1e6;

        println!(
            "\n=== {label} ===  build {:.0}s, {mem:.0} MB",
            build.as_secs_f64()
        );
        println!(
            "{:>6} {:>11} {:>10} {:>12} {:>9}",
            "ef", "recall@10", "QPS", "dist/query", "ns/dist"
        );

        for &ef in EFS {
            index.set_ef_search(ef);

            let mut hits = 0usize;
            for (qi, q) in queries.iter().enumerate() {
                let got: Vec<usize> = index
                    .search(q, K)
                    .unwrap()
                    .into_iter()
                    .filter_map(|r| r.id.parse::<usize>().ok())
                    .collect();
                hits += got.iter().filter(|i| truth[qi].contains(i)).count();
            }
            let recall = hits as f64 / (queries.len() * K) as f64 * 100.0;

            let mut s = index.searcher();
            for q in queries.iter().take(50) {
                std::hint::black_box(s.search(q, K).unwrap());
            }

            let mut s = index.searcher();
            let t = Instant::now();
            for q in &queries {
                std::hint::black_box(s.search(q, K).unwrap());
            }
            let el = t.elapsed();

            let nq = queries.len() as f64;
            let d = s.distance_calls() as f64;
            println!(
                "{:>6} {:>10.2}% {:>10.0} {:>12.0} {:>9.1}",
                ef,
                recall,
                nq / el.as_secs_f64(),
                d / nq,
                el.as_nanos() as f64 / d
            );
        }
    }

    println!(
        "\nRead ACROSS the blocks at equal recall, not down them at equal ef. RaBitQ buys a cheap\n\
         kernel (low ns/dist) with a coarse estimate (high dist/query); whether that trade is\n\
         profitable is exactly what the recall-matched frontier decides, and it flips with `dim`."
    );
}
