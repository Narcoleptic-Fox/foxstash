//! Where does SQ8 stop winning and RaBitQ start? Sweep `dim`, hold everything else fixed.
//!
//! Run: `cargo run --release -p foxstash-benches --example dim_crossover [dataset] [n]`
//!   e.g. `... --example dim_crossover gist1m 200000`
//!
//! We know the two quantized storage modes swap places:
//!
//! ```text
//!                     SIFT1M (128-d)        GIST1M (960-d)
//!   Storage::SQ8      1.20x hnswlib         worthless (1.03x F32)
//!   Storage::RaBitQ   ~12x slower than SQ8  1.4-1.6x faster than SQ8 or F32
//! ```
//!
//! But that is *bracketed*, not *located* — and the two data points come from two different
//! corpora, so `dim` is confounded with dataset difficulty. Switching datasets to find a
//! crossover cannot separate "384-d is different" from "that corpus is different".
//!
//! So: take ONE corpus and truncate it to a dimension prefix. GIST vectors are concatenated
//! orientation histograms, so a prefix is a real sub-descriptor with real cluster structure —
//! not a synthetic vector, and not a random projection. Everything except `dim` is held fixed:
//! same base vectors, same queries, same graph parameters, same machine.
//!
//! Ground truth is recomputed exactly for each `dim` (the shipped GT indexes the full 960-d
//! space and is meaningless for a prefix — using it would silently score every mode against
//! the wrong answer, which is exactly the class of mistake `MANIFEST` and the control rows
//! exist to prevent).
//!
//! The mechanism under test, which predicts a crossover exists at all:
//!
//! * **SQ8** widens `u8` -> `i32` -> `f32`: ~3x the ALU uops/dim of plain f32. Its benefit —
//!   skipped DRAM round-trips — is roughly FIXED per node visit. Fixed benefit, cost linear in
//!   `dim`. Wins small, dies big.
//! * **RaBitQ** compares sign bits: CHEAPER per dim than f32, no widening. Its cost is a coarse
//!   estimate -> more graph hops, roughly INDEPENDENT of `dim`. Loses small, wins big.
//!
//! Read `ns/dist` — that is where the mechanism lives. `dist/query` is the control: if it moves
//! a lot between modes, the graph walk changed and you are not measuring what you think.
//!
//! Run on an IDLE machine.

use foxstash_benches::sift::Dataset;
use foxstash_core::index::hnsw::{BuildStrategy, DistanceMetric, HNSWConfig, HNSWIndex, Storage};
use rayon::prelude::*;
use std::time::Instant;

const K: usize = 10;
const DIMS: &[usize] = &[64, 128, 192, 256, 384, 512, 768, 960];
const EF: usize = 100;

/// Exact top-k by brute force, in the truncated space. The oracle.
fn exact_gt(base: &[Vec<f32>], queries: &[Vec<f32>], k: usize) -> Vec<Vec<usize>> {
    queries
        .par_iter()
        .map(|q| {
            let mut d: Vec<(f32, usize)> = base
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let s = v
                        .iter()
                        .zip(q)
                        .map(|(a, b)| {
                            let x = a - b;
                            x * x
                        })
                        .sum::<f32>();
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
    let n: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200_000);

    let ds = Dataset::load("benchmarks/data", &name).unwrap_or_else(|e| panic!("load {name}: {e}"));
    let full_dim = ds.dim();
    let n = n.min(ds.base.len());
    let nq = 500.min(ds.queries.len());

    println!(
        "{} — {n} base x (truncated from {full_dim}d), {nq} queries, k={K}, M=32/m0=64, ef={EF}\n",
        ds.name
    );
    println!(
        "Prefix-truncating ONE corpus so `dim` is the only variable. Ground truth recomputed\n\
         exactly at every dim — the shipped GT indexes the full {full_dim}-d space and is void\n\
         for a prefix.\n"
    );
    println!(
        "{:>5} {:>7} {:>7} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "dim", "blkF32", "blkRBQ", "F32 ns", "SQ8 ns", "RBQ ns", "F32 QPS", "SQ8 QPS", "RBQ QPS"
    );
    println!("{:-<86}", "");

    for &d in DIMS {
        if d > full_dim {
            continue;
        }

        let base: Vec<Vec<f32>> = ds.base[..n].iter().map(|v| v[..d].to_vec()).collect();
        let queries: Vec<Vec<f32>> = ds.queries[..nq].iter().map(|v| v[..d].to_vec()).collect();
        let truth = exact_gt(&base, &queries, K);

        let hdr = 272usize;
        let mut row = vec![format!("{d:>5}"), format!("{:>7}", hdr + d * 4)];
        row.push(format!("{:>7}", hdr + d.div_ceil(8) + 8));

        let mut ns = Vec::new();
        let mut qps = Vec::new();
        let mut recalls = Vec::new();

        for (storage, rerank) in [
            (Storage::F32, 0usize),
            (Storage::SQ8, 100),
            (Storage::RaBitQ, 100),
        ] {
            let mut index = HNSWIndex::build_parallel(
                base.clone(),
                HNSWConfig {
                    metric: DistanceMetric::L2,
                    m: 32,
                    m0: 64,
                    ef_construction: 200,
                    ef_search: EF,
                    storage,
                    rerank_candidates: rerank,
                    build_strategy: BuildStrategy::Parallel,
                    ..Default::default()
                },
            );
            index.set_ef_search(EF);

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
            recalls.push(hits as f64 / (queries.len() * K) as f64 * 100.0);

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
            let dc = s.distance_calls() as f64;
            ns.push(el.as_nanos() as f64 / dc);
            qps.push(queries.len() as f64 / el.as_secs_f64());
        }

        for v in &ns {
            row.push(format!("{v:>9.1}"));
        }
        for v in &qps {
            row.push(format!("{v:>9.0}"));
        }
        println!("{}", row.join(" "));
        println!(
            "{:>5} {:>7} {:>7} recall@10:  F32 {:>5.1}%   SQ8 {:>5.1}%   RBQ {:>5.1}%",
            "", "", "", recalls[0], recalls[1], recalls[2]
        );
    }

    println!(
        "\nns/dist is the mechanism. SQ8's dequant tax grows with dim while the DRAM round-trips\n\
         it saves stay ~fixed; RaBitQ's kernel is cheaper per dim than f32 and its cost (a coarse\n\
         estimate -> more hops) is ~dim-independent. The crossover is where those lines meet.\n\
         Recall is printed too: a mode that is fast because it stopped finding things is not fast."
    );
}
