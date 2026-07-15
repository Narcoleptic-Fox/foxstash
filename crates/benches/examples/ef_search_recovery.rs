//! Is the parallel builder's zero-layer graph GOOD, with only the upper navigation broken?
//!
//! Run: cargo run -p foxstash-benches --example ef_search_recovery --release
//!
//! Mechanism under test (from reading build_parallel): the parallel builder does NOT build the
//! upper layers with the HNSW algorithm. It PROJECTS them from the zero layer
//! (`UpperNode::from_zero` = zero node's neighbor list truncated to m). Those neighbor ids are the
//! globally-nearest DENSE points, most of which were never promoted into the sparse upper layer,
//! so `search_upper` skips them and the navigation layers are under-connected. The zero layer
//! itself, though, is built properly (par_insert runs the real Algorithm-4 heuristic + reverse
//! links there).
//!
//! If that is the whole story, the defect is a BAD ENTRY POINT into a GOOD zero graph. Raising
//! `ef_search` makes the zero-layer search explore far enough to escape a bad start:
//!   - Parallel recall climbs to meet Sequential as ef_search grows => zero graph is fine, only
//!     the upper/entry navigation is broken. Fix = rebuild upper layers (or raise ef_search).
//!   - Parallel stays below Sequential at every ef_search => the zero graph is ALSO worse; the
//!     defect is deeper than upper-layer projection.
//!
//! ef_construction is held at 200 for both (the recovery sweep already showed build effort is inert).

use foxstash_core::index::hnsw::{BuildStrategy, DistanceMetric, HNSWConfig, HNSWIndex, Storage};
use std::collections::HashSet;
use std::time::Instant;

const N: usize = 20_000;
const NQ: usize = 200;
const K: usize = 10;
const CLUSTERS: usize = 100;
const SIGMA: f32 = 0.05;
const DIMS: [usize; 2] = [128, 768];
const EF_SEARCH_SWEEP: [usize; 5] = [100, 200, 400, 800, 1600];

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
fn recall_at(idx: &mut HNSWIndex, ef: usize, truth: &[HashSet<usize>], queries: &[Vec<f32>]) -> f32 {
    idx.set_ef_search(ef);
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
fn build(base: Vec<Vec<f32>>, strat: BuildStrategy) -> HNSWIndex {
    let cfg = HNSWConfig {
        metric: DistanceMetric::L2,
        m: 32,
        m0: 64,
        ef_construction: 200,
        ef_search: 100,
        storage: Storage::F32,
        build_strategy: strat,
        seed: Some(42),
        ..Default::default()
    };
    HNSWIndex::build(base, cfg)
}

fn main() {
    println!("=== Does raising ef_search recover the parallel deficit? (F32) ===");
    println!("{N} vectors, top-{K}, ef_construction=200, seed=42\n");

    for dim in DIMS {
        let base = clustered(N, dim, 42);
        let queries = clustered(NQ, dim, 123);
        let truth: Vec<HashSet<usize>> = queries.iter().map(|q| exact_gt(q, &base)).collect();

        let t0 = Instant::now();
        let mut seq = build(base.clone(), BuildStrategy::Sequential);
        let seq_ms = t0.elapsed().as_millis();
        let t1 = Instant::now();
        let mut par = build(base.clone(), BuildStrategy::Parallel);
        let par_ms = t1.elapsed().as_millis();

        println!("dim={dim}  (build: seq {seq_ms} ms, par {par_ms} ms)");
        println!("{:>10}  {:>10}  {:>10}  {:>8}", "ef_search", "Seq R@10", "Par R@10", "gap");
        println!("  {:-<42}", "");
        for ef in EF_SEARCH_SWEEP {
            let rs = recall_at(&mut seq, ef, &truth, &queries);
            let rp = recall_at(&mut par, ef, &truth, &queries);
            println!(
                "{:>10}  {:>9.2}%  {:>9.2}%  {:>+7.2}",
                ef,
                rs * 100.0,
                rp * 100.0,
                (rs - rp) * 100.0
            );
        }
        println!();
    }
    println!("gap -> ~0 as ef_search grows => zero-layer graph is fine; only upper/entry nav is broken.");
    println!("gap persists at high ef_search => the zero-layer graph is also worse.");
}
