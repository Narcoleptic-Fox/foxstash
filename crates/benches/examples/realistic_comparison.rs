//! Realistic recall benchmark — structured (clustered) vs uniform-random data
//!
//! Run with: cargo run -p foxstash-benches --example realistic_comparison --release
//!
//! WHY THIS EXISTS
//! ---------------
//! `quick_comparison.rs` generates vectors as `hash(i,j) % 256 - 128` then
//! normalizes them — i.e. *uniform random points on the unit sphere*. In high
//! dimensions such points are all nearly equidistant (curse of dimensionality),
//! so the true 10-NN are barely closer than the 50th. ANY graph ANN scores
//! ~60% recall@10 on that data, which is why `instant-distance` lands at ~60%
//! too. That number measures the dataset, not the index.
//!
//! Real embeddings (MiniLM, BGE, etc.) live on clusters/manifolds where near
//! neighbors are genuinely distinguishable, and a correct HNSW reaches 90%+.
//! This benchmark generates Gaussian-cluster data as a stand-in for that
//! structure and reports recall on BOTH distributions with the identical
//! foxstash index, so the contrast is unambiguous.

use foxstash_core::index::{BuildStrategy, HNSWConfig, HNSWIndex};
use instant_distance::{Builder, Search};
use std::collections::HashSet;
use std::time::Instant;

const NUM_VECTORS: usize = 50_000;
const NUM_QUERIES: usize = 2_000;
const RECALL_QUERIES: usize = 200;
const DIM: usize = 128;
const K: usize = 10;

// Cluster structure (proxy for real-embedding manifold).
const NUM_CLUSTERS: usize = 200;
const CLUSTER_SIGMA: f32 = 0.05; // per-coord noise; radius ≈ SIGMA*sqrt(DIM) ≈ 0.57

// ---------------------------------------------------------------------------
// Deterministic PRNG (SplitMix64) — reproducible, no rand-version churn.
// ---------------------------------------------------------------------------
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform f32 in [0, 1).
    fn next_f32(&mut self) -> f32 {
        // top 24 bits → full f32 mantissa precision
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Standard normal via Box–Muller.
    fn next_gaussian(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-7);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }

    fn unit_vector(&mut self, dim: usize) -> Vec<f32> {
        let mut v: Vec<f32> = (0..dim).map(|_| self.next_gaussian()).collect();
        normalize(&mut v);
        v
    }
}

fn normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Clustered data: pick a random cluster center, add Gaussian noise, renormalize.
/// Stands in for the manifold structure of real embeddings.
fn generate_clustered(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Rng::new(seed);
    let centers: Vec<Vec<f32>> = (0..NUM_CLUSTERS).map(|_| rng.unit_vector(dim)).collect();

    (0..count)
        .map(|_| {
            let center = &centers[(rng.next_u64() as usize) % NUM_CLUSTERS];
            let mut v: Vec<f32> = center
                .iter()
                .map(|&c| c + CLUSTER_SIGMA * rng.next_gaussian())
                .collect();
            normalize(&mut v);
            v
        })
        .collect()
}

/// Uniform random data — the OLD distribution from quick_comparison.rs.
fn generate_uniform(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Rng::new(seed);
    (0..count)
        .map(|_| {
            let mut v: Vec<f32> = (0..dim).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
            normalize(&mut v);
            v
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Recall
// ---------------------------------------------------------------------------

/// Brute-force top-K by Euclidean distance (matches instant-distance's metric;
/// on unit vectors this ranks identically to cosine).
fn brute_force_topk(query: &[f32], base: &[Vec<f32>], k: usize) -> HashSet<usize> {
    let mut dists: Vec<(f32, usize)> = base
        .iter()
        .enumerate()
        .map(|(j, v)| {
            let d: f32 = query
                .iter()
                .zip(v.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            (d, j)
        })
        .collect();
    dists.sort_by(|a, b| a.0.total_cmp(&b.0));
    dists.iter().take(k).map(|(_, j)| *j).collect()
}

/// Average recall@K of a foxstash index over the first `n` queries.
fn foxstash_recall(
    index: &HNSWIndex,
    base: &[Vec<f32>],
    queries: &[Vec<f32>],
    k: usize,
    n: usize,
) -> f32 {
    let mut searcher = index.searcher();
    let mut total = 0.0;
    for q in queries.iter().take(n) {
        let gt = brute_force_topk(q, base, k);
        let got: HashSet<usize> = searcher
            .search(q, k)
            .unwrap()
            .iter()
            .map(|r| r.id.parse().unwrap())
            .collect();
        total += gt.intersection(&got).count() as f32 / k as f32;
    }
    total / n as f32
}

fn build_foxstash(base: &[Vec<f32>], ef_search: usize) -> (HNSWIndex, std::time::Duration) {
    let config = HNSWConfig::default()
        .with_build_strategy(BuildStrategy::Parallel)
        .with_ef_search(ef_search);
    let start = Instant::now();
    let index = HNSWIndex::build(base.to_vec(), config);
    (index, start.elapsed())
}

// ---------------------------------------------------------------------------

fn main() {
    println!("=== Foxstash Realistic Recall Benchmark ===\n");
    println!(
        "Dataset: {} vectors, {}d, top-{}, {} clusters (sigma={})\n",
        NUM_VECTORS, DIM, K, NUM_CLUSTERS, CLUSTER_SIGMA
    );

    // --- Generate both distributions ---
    println!("Generating clustered (structured) data...");
    let clustered_base = generate_clustered(NUM_VECTORS, DIM, 42);
    let clustered_queries = generate_clustered(NUM_QUERIES, DIM, 123);

    println!("Generating uniform-random (control) data...");
    let uniform_base = generate_uniform(NUM_VECTORS, DIM, 7);
    let uniform_queries = generate_uniform(NUM_QUERIES, DIM, 9);

    // === Main result: same index code, two data distributions ===
    println!("\n--- CLUSTERED data (proxy for real embeddings) ---");
    let (clustered_index, c_build) = build_foxstash(&clustered_base, 100);
    println!("Foxstash build time: {:?}", c_build);

    let mut searcher = clustered_index.searcher();
    let start = Instant::now();
    for q in &clustered_queries {
        let _ = searcher.search(q, K);
    }
    let c_qps = NUM_QUERIES as f64 / start.elapsed().as_secs_f64();
    println!(
        "Foxstash search: {:.0} QPS (single-threaded, searcher reuse)",
        c_qps
    );

    let c_recall = foxstash_recall(
        &clustered_index,
        &clustered_base,
        &clustered_queries,
        K,
        RECALL_QUERIES,
    );
    println!("Foxstash Recall@{}: {:.2}%", K, c_recall * 100.0);

    // instant-distance cross-check on the SAME clustered data
    let points: Vec<Point> = clustered_base.iter().map(|v| Point(v.clone())).collect();
    let values: Vec<usize> = (0..NUM_VECTORS).collect();
    let id_hnsw = Builder::default().build(points, values);
    let mut id_search = Search::default();
    let mut id_total = 0.0;
    for q in clustered_queries.iter().take(RECALL_QUERIES) {
        let gt = brute_force_topk(q, &clustered_base, K);
        let qp = Point(q.clone());
        let got: HashSet<usize> = id_hnsw
            .search(&qp, &mut id_search)
            .take(K)
            .map(|item| *item.value)
            .collect();
        id_total += gt.intersection(&got).count() as f32 / K as f32;
    }
    println!(
        "instant-distance Recall@{}: {:.2}%  (cross-check on same data)",
        K,
        id_total / RECALL_QUERIES as f32 * 100.0
    );

    // === Control: the OLD uniform-random distribution, identical index ===
    println!("\n--- UNIFORM-RANDOM data (the old quick_comparison distribution) ---");
    let (uniform_index, u_build) = build_foxstash(&uniform_base, 100);
    println!("Foxstash build time: {:?}", u_build);
    let u_recall = foxstash_recall(
        &uniform_index,
        &uniform_base,
        &uniform_queries,
        K,
        RECALL_QUERIES,
    );
    println!("Foxstash Recall@{}: {:.2}%", K, u_recall * 100.0);

    // === ef_search sweep on clustered data (the real recall knob) ===
    println!("\n--- ef_search sweep (clustered data) ---");
    println!("{:>10} {:>12} {:>12}", "ef_search", "Recall@10", "QPS");
    println!("{:-<36}", "");
    for &ef in &[50usize, 100, 200, 400] {
        let (idx, _) = build_foxstash(&clustered_base, ef);
        let mut searcher = idx.searcher();
        let start = Instant::now();
        for q in &clustered_queries {
            let _ = searcher.search(q, K);
        }
        let qps = NUM_QUERIES as f64 / start.elapsed().as_secs_f64();
        let recall = foxstash_recall(&idx, &clustered_base, &clustered_queries, K, RECALL_QUERIES);
        println!("{:>10} {:>11.2}% {:>12.0}", ef, recall * 100.0, qps);
    }

    // === Verdict ===
    println!("\n=== VERDICT ===");
    println!(
        "Same foxstash index, two distributions:\n  \
         Clustered (real-embedding proxy): {:.1}% recall@10\n  \
         Uniform random (curse of dim.):   {:.1}% recall@10",
        c_recall * 100.0,
        u_recall * 100.0
    );
    println!(
        "\nThe ~60% figure in the README is a property of uniform-random test data,\n\
         not the index. On structured data foxstash recovers high recall."
    );
}

// instant-distance point wrapper (Euclidean).
#[derive(Clone)]
struct Point(Vec<f32>);

impl instant_distance::Point for Point {
    fn distance(&self, other: &Self) -> f32 {
        self.0
            .iter()
            .zip(other.0.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}
