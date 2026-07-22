//! Is the parallel builder's ~15-point recall deficit a BUG or a tunable TRADEOFF?
//!
//! Run: cargo run -p foxstash-benches --example parallel_ef_recovery --release
//!
//! `build_strategy_x_storage` established that Parallel recalls ~15 points below Sequential at
//! equal `ef_search`, in EVERY storage mode (so it is graph quality, not quantization). Parallel
//! is also 9-14x faster to build. The open question: does that speed come at the cost of a graph
//! that MORE build effort can fix, or one that is structurally worse?
//!
//! This sweeps the parallel builder's `ef_construction` upward and watches recall. Sequential at
//! the default ef_construction=200 is the bar to clear.
//!   - Recall climbs to meet Sequential  => it's a knob. Parallel just needs more ef_construction.
//!   - Recall plateaus below Sequential   => structural defect in the parallel construction path.
//!
//! F32 only (the deficit is storage-independent) so the graph is the sole variable.

use foxstash_core::index::hnsw::{BuildStrategy, DistanceMetric, HNSWConfig, HNSWIndex, Storage};
use std::collections::HashSet;
use std::time::Instant;

const N: usize = 20_000;
const NQ: usize = 200;
const K: usize = 10;
const CLUSTERS: usize = 100;
const SIGMA: f32 = 0.05;
const EF_SEARCH: usize = 100;
const DIMS: [usize; 2] = [128, 768];
const EF_CONSTRUCTION_SWEEP: [usize; 4] = [200, 400, 800, 1600];

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
fn recall(idx: &mut HNSWIndex, truth: &[HashSet<usize>], queries: &[Vec<f32>]) -> f32 {
    idx.set_ef_search(EF_SEARCH);
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
fn build(base: Vec<Vec<f32>>, ef_c: usize, strat: BuildStrategy) -> (HNSWIndex, u128) {
    let cfg = HNSWConfig {
        metric: DistanceMetric::L2,
        m: 32,
        m0: 64,
        ef_construction: ef_c,
        ef_search: EF_SEARCH,
        storage: Storage::F32,
        build_strategy: strat,
        seed: Some(42),
        ..Default::default()
    };
    let start = Instant::now();
    let idx = HNSWIndex::build(base, cfg);
    (idx, start.elapsed().as_millis())
}

fn main() {
    println!("=== Does more ef_construction recover the parallel builder's recall? (F32) ===");
    println!("{N} vectors, top-{K}, ef_search={EF_SEARCH}, seed=42\n");

    for dim in DIMS {
        let base = clustered(N, dim, 42);
        let queries = clustered(NQ, dim, 123);
        let truth: Vec<HashSet<usize>> = queries.iter().map(|q| exact_gt(q, &base)).collect();

        // The bar: sequential at the default ef_construction=200.
        let (mut seq, seq_ms) = build(base.clone(), 200, BuildStrategy::Sequential);
        let seq_r = recall(&mut seq, &truth, &queries);
        println!(
            "dim={dim}  SEQUENTIAL ef_c=200 -> {:.2}%  ({seq_ms} ms)  <- the bar",
            seq_r * 100.0
        );

        for ef_c in EF_CONSTRUCTION_SWEEP {
            let (mut par, par_ms) = build(base.clone(), ef_c, BuildStrategy::Parallel);
            let par_r = recall(&mut par, &truth, &queries);
            println!(
                "dim={dim}  parallel   ef_c={ef_c:<4} -> {:.2}%  ({par_ms} ms)  gap {:+.2}",
                par_r * 100.0,
                (seq_r - par_r) * 100.0
            );
        }
        println!();
    }
    println!("gap -> 0 as ef_c grows  => tunable knob, parallel just needs more build effort.");
    println!("gap plateaus > 0        => structural defect in the parallel construction path.");
}
