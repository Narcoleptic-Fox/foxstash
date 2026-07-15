//! Does the build strategy change a QUANTIZED index's recall?
//!
//! Run: cargo run -p foxstash-benches --example build_strategy_x_storage --release
//!
//! The parallel builder (`build_parallel` -> `search_zero`/`search_upper`) selects graph
//! edges by RAW f32 distance and quantizes at the end. The sequential builder
//! (`insert_node` -> `search_layer` -> `distance_to_node`) builds against storage. So for a
//! quantized index the two strategies produce different graphs. Every quantizer benchmark in
//! this crate calls `build_parallel` — i.e. the published RaBitQ/SQ8 recall numbers were all
//! measured on the f32-graph path. This isolates whether that matters.
//!
//! Two questions, one table:
//!   1. Sequential vs Parallel recall at each (dim, storage) — does the strategy move recall?
//!   2. Parallel run A vs run B — the builder is documented non-reproducible at a fixed seed;
//!      if that changes the GRAPH it should change RECALL. This makes the spread visible.
//!
//! Clustered data (not uniform random — uniform vectors have no neighbour structure and every
//! ANN scores ~random on them). Ground truth is exact brute-force L2.

use foxstash_core::index::hnsw::{BuildStrategy, DistanceMetric, HNSWConfig, HNSWIndex, Storage};
use std::collections::HashSet;
use std::time::Instant;

const N: usize = 20_000;
const NQ: usize = 200;
const K: usize = 10;
const CLUSTERS: usize = 100;
const SIGMA: f32 = 0.05;
const EF: usize = 100;
const DIMS: [usize; 2] = [128, 768]; // SQ8's turf and RaBitQ's turf

struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Self {
        Rng(s)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn f32u(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn gauss(&mut self) -> f32 {
        let u1 = self.f32u().max(1e-7);
        let u2 = self.f32u();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
    fn unit(&mut self, d: usize) -> Vec<f32> {
        let mut v: Vec<f32> = (0..d).map(|_| self.gauss()).collect();
        norm(&mut v);
        v
    }
}
fn norm(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        v.iter_mut().for_each(|x| *x /= n);
    }
}
fn clustered(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut r = Rng::new(seed);
    let centers: Vec<Vec<f32>> = (0..CLUSTERS).map(|_| r.unit(dim)).collect();
    (0..count)
        .map(|_| {
            let c = &centers[(r.next_u64() as usize) % CLUSTERS];
            let mut v: Vec<f32> = c.iter().map(|&x| x + SIGMA * r.gauss()).collect();
            norm(&mut v);
            v
        })
        .collect()
}
fn exact_gt(q: &[f32], base: &[Vec<f32>]) -> HashSet<usize> {
    let mut d: Vec<(f32, usize)> = base
        .iter()
        .enumerate()
        .map(|(j, v)| (q.iter().zip(v).map(|(a, b)| (a - b).powi(2)).sum::<f32>(), j))
        .collect();
    d.sort_by(|a, b| a.0.total_cmp(&b.0));
    d.iter().take(K).map(|(_, j)| *j).collect()
}
fn recall(idx: &mut HNSWIndex, truth: &[HashSet<usize>], queries: &[Vec<f32>]) -> f32 {
    idx.set_ef_search(EF);
    let mut t = 0.0;
    for (qi, q) in queries.iter().enumerate() {
        let got: HashSet<usize> = idx
            .search(q, K)
            .unwrap()
            .iter()
            .filter_map(|r| r.id.parse::<usize>().ok())
            .collect();
        t += truth[qi].intersection(&got).count() as f32 / K as f32;
    }
    t / queries.len() as f32
}

fn build(base: Vec<Vec<f32>>, dim: usize, storage: Storage, rerank: usize, strat: BuildStrategy) -> (HNSWIndex, u128) {
    let _ = dim;
    let cfg = HNSWConfig {
        metric: DistanceMetric::L2,
        m: 32,
        m0: 64,
        ef_construction: 200,
        ef_search: EF,
        storage,
        rerank_candidates: rerank,
        build_strategy: strat,
        seed: Some(42),
        ..Default::default()
    };
    let start = Instant::now();
    let idx = HNSWIndex::build(base, cfg);
    (idx, start.elapsed().as_millis())
}

fn main() {
    println!("=== Build strategy x storage: does the strategy move QUANTIZED recall? ===");
    println!("{N} vectors, top-{K}, {CLUSTERS} clusters, ef_search={EF}, seed=42\n");
    println!(
        "{:>4}  {:>7}  {:>9}  {:>9}  {:>9}  {:>8}  {:>7}  {:>7}",
        "dim", "storage", "Seq R@10", "Par-A", "Par-B", "Seq-Par", "Seq ms", "Par ms"
    );
    println!("{:-<72}", "");

    for dim in DIMS {
        let base = clustered(N, dim, 42);
        let queries = clustered(NQ, dim, 123);
        let truth: Vec<HashSet<usize>> = queries.iter().map(|q| exact_gt(q, &base)).collect();

        for (storage, rerank, name) in [
            (Storage::F32, 0usize, "F32"),
            (Storage::SQ8, 100, "SQ8"),
            (Storage::RaBitQ, 100, "RaBitQ"),
        ] {
            let (mut seq, seq_ms) = build(base.clone(), dim, storage, rerank, BuildStrategy::Sequential);
            let r_seq = recall(&mut seq, &truth, &queries);

            let (mut par_a, par_ms) = build(base.clone(), dim, storage, rerank, BuildStrategy::Parallel);
            let r_par_a = recall(&mut par_a, &truth, &queries);

            let (mut par_b, _) = build(base.clone(), dim, storage, rerank, BuildStrategy::Parallel);
            let r_par_b = recall(&mut par_b, &truth, &queries);

            println!(
                "{:>4}  {:>7}  {:>8.2}%  {:>8.2}%  {:>8.2}%  {:>+7.2}  {:>7}  {:>7}",
                dim,
                name,
                r_seq * 100.0,
                r_par_a * 100.0,
                r_par_b * 100.0,
                (r_seq - r_par_a) * 100.0,
                seq_ms,
                par_ms
            );
        }
        println!();
    }
    println!("Seq-Par > 0 => sequential recalls higher (parallel's f32 graph hurt the quantized index).");
    println!("Par-A vs Par-B spread => the non-reproducible builder changes graph quality, not just node order.");
}
