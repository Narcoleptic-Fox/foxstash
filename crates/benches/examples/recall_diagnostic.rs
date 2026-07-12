//! Recall root-cause diagnostic — isolates which knob tanks recall on clustered data.
//!
//! Run: cargo run -p foxstash-benches --example recall_diagnostic --release
//!
//! Tests the 2x2 of {Sequential, Parallel} build x {heuristic, simple} neighbor
//! selection on clustered (structured) data, where foxstash was observed to get
//! ~25% recall@10 vs instant-distance's ~69% on identical data.

use foxstash_core::index::{BuildStrategy, HNSWConfig, HNSWIndex};
use std::collections::HashSet;
use std::time::Instant;

const N: usize = 20_000;
const NQ: usize = 200;
const DIM: usize = 128;
const K: usize = 10;
const CLUSTERS: usize = 100;
const SIGMA: f32 = 0.05;

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
fn clustered(count: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut r = Rng::new(seed);
    let centers: Vec<Vec<f32>> = (0..CLUSTERS).map(|_| r.unit(DIM)).collect();
    (0..count)
        .map(|_| {
            let c = &centers[(r.next_u64() as usize) % CLUSTERS];
            let mut v: Vec<f32> = c.iter().map(|&x| x + SIGMA * r.gauss()).collect();
            norm(&mut v);
            v
        })
        .collect()
}
fn gt(q: &[f32], base: &[Vec<f32>]) -> HashSet<usize> {
    let mut d: Vec<(f32, usize)> = base
        .iter()
        .enumerate()
        .map(|(j, v)| {
            (
                q.iter().zip(v).map(|(a, b)| (a - b).powi(2)).sum::<f32>(),
                j,
            )
        })
        .collect();
    d.sort_by(|a, b| a.0.total_cmp(&b.0));
    d.iter().take(K).map(|(_, j)| *j).collect()
}
fn recall(idx: &HNSWIndex, base: &[Vec<f32>], queries: &[Vec<f32>]) -> f32 {
    let mut ctx = idx.create_search_context();
    let mut t = 0.0;
    for q in queries.iter().take(NQ) {
        let truth = gt(q, base);
        let got: HashSet<usize> = idx
            .search_with_context(q, K, &mut ctx)
            .unwrap()
            .iter()
            .map(|r| r.id.parse().unwrap())
            .collect();
        t += truth.intersection(&got).count() as f32 / K as f32;
    }
    t / NQ as f32
}

fn main() {
    println!("=== Recall Root-Cause Diagnostic (clustered data) ===");
    println!(
        "{} vectors, {}d, top-{}, {} clusters\n",
        N, DIM, K, CLUSTERS
    );
    let base = clustered(N, 42);
    let queries = clustered(NQ, 123);

    println!(
        "{:<12} {:<10} {:>12} {:>12}",
        "build", "select", "Recall@10", "build ms"
    );
    println!("{:-<48}", "");
    for (sname, strat) in [
        ("Sequential", BuildStrategy::Sequential),
        ("Parallel", BuildStrategy::Parallel),
    ] {
        for (hname, simple) in [("heuristic", false), ("simple", true)] {
            let mut cfg = HNSWConfig::default()
                .with_build_strategy(strat)
                .with_ef_search(100);
            if simple {
                cfg = cfg.with_simple_selection();
            }
            let start = Instant::now();
            let idx = HNSWIndex::build(base.clone(), cfg);
            let ms = start.elapsed().as_millis();
            let r = recall(&idx, &base, &queries);
            println!(
                "{:<12} {:<10} {:>11.2}% {:>12}",
                sname,
                hname,
                r * 100.0,
                ms
            );
        }
    }
}
