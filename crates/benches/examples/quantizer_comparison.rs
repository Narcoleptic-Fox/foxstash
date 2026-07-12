//! Quantizer recall comparison — RaBitQ vs Binary vs SQ8 (two-phase search).
//!
//! Run: cargo run -p foxstash-benches --example quantizer_comparison --release
//!
//! All three rank the corpus by their (query, doc) distance estimate, keep the
//! top `POOL` candidates, then rerank those by exact L2 and take top-K. We
//! report recall@K vs exact brute force. RaBitQ and Binary are both 32x
//! compression (1 bit/dim), so that pair is the apples-to-apples comparison;
//! SQ8 (4x, 8 bit/dim) is a higher-memory reference point.

use foxstash_core::vector::quantize::{BinaryQuantizer, Quantizer, ScalarQuantizer};
use foxstash_core::vector::rabitq::RaBitQuantizer;
use std::collections::HashSet;

const N: usize = 20_000;
const NQ: usize = 200;
const DIM: usize = 128;
const K: usize = 10;
const POOL: usize = 100; // first-stage candidate pool
const CLUSTERS: usize = 100;
const SIGMA: f32 = 0.05;

struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Self {
        Rng(s)
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn f32u(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
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
            let c = &centers[(r.next() as usize) % CLUSTERS];
            let mut v: Vec<f32> = c.iter().map(|&x| x + SIGMA * r.gauss()).collect();
            norm(&mut v);
            v
        })
        .collect()
}
fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}
fn exact_topk(q: &[f32], base: &[Vec<f32>]) -> HashSet<usize> {
    let mut d: Vec<(f32, usize)> = base
        .iter()
        .enumerate()
        .map(|(i, v)| (l2_sq(q, v), i))
        .collect();
    d.sort_by(|a, b| a.0.total_cmp(&b.0));
    d.iter().take(K).map(|(_, i)| *i).collect()
}

/// Recall@K of a two-phase search given a stage-1 scorer (lower = better).
fn rerank_recall(
    base: &[Vec<f32>],
    queries: &[Vec<f32>],
    stage1: impl Fn(&[f32]) -> Vec<(f32, usize)>,
) -> f32 {
    let mut total = 0.0;
    for q in queries.iter().take(NQ) {
        let truth = exact_topk(q, base);
        let mut est = stage1(q);
        est.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut pool: Vec<(f32, usize)> = est
            .iter()
            .take(POOL)
            .map(|&(_, i)| (l2_sq(q, &base[i]), i))
            .collect();
        pool.sort_by(|a, b| a.0.total_cmp(&b.0));
        let got: HashSet<usize> = pool.iter().take(K).map(|(_, i)| *i).collect();
        total += truth.intersection(&got).count() as f32 / K as f32;
    }
    total / NQ as f32
}

fn main() {
    println!("=== Quantizer Recall Comparison (two-phase, clustered data) ===");
    println!(
        "{} vectors, {}d, top-{}, pool={}, {} clusters\n",
        N, DIM, K, POOL, CLUSTERS
    );
    let base = clustered(N, 42);
    let queries = clustered(NQ, 123);

    // RaBitQ (32x): asymmetric unbiased estimate.
    let rabitq = RaBitQuantizer::fit(&base);
    let rb_codes: Vec<_> = base.iter().map(|v| rabitq.encode(v)).collect();
    let rb_recall = rerank_recall(&base, &queries, |q| {
        let prep = rabitq.prepare_query(q);
        rb_codes
            .iter()
            .enumerate()
            .map(|(i, c)| (rabitq.estimate_dist_sq(&prep, c), i))
            .collect()
    });

    // Binary (32x): Hamming distance between quantized query and docs.
    let binq = BinaryQuantizer::new(DIM);
    let bin_codes: Vec<_> = base.iter().map(|v| binq.quantize(v)).collect();
    let bin_recall = rerank_recall(&base, &queries, |q| {
        let qc = binq.quantize(q);
        bin_codes
            .iter()
            .enumerate()
            .map(|(i, c)| (binq.distance_quantized(&qc, c), i))
            .collect()
    });

    // SQ8 (4x): asymmetric, full-precision query vs quantized docs.
    let sq8 = ScalarQuantizer::fit(&base);
    let sq_codes: Vec<_> = base.iter().map(|v| sq8.quantize(v)).collect();
    let sq_recall = rerank_recall(&base, &queries, |q| {
        sq_codes
            .iter()
            .enumerate()
            .map(|(i, c)| (sq8.distance_asymmetric(q, c), i))
            .collect()
    });

    println!("{:<22} {:>10} {:>14}", "Quantizer", "Compress", "Recall@10");
    println!("{:-<48}", "");
    println!(
        "{:<22} {:>10} {:>13.1}%",
        "RaBitQ (1-bit)",
        "32x",
        rb_recall * 100.0
    );
    println!(
        "{:<22} {:>10} {:>13.1}%",
        "Binary (Hamming)",
        "32x",
        bin_recall * 100.0
    );
    println!(
        "{:<22} {:>10} {:>13.1}%",
        "SQ8 (scalar int8)",
        "4x",
        sq_recall * 100.0
    );

    println!(
        "\nAt equal 32x compression, RaBitQ's unbiased estimator recovers {:+.1} pts\n\
         of recall@10 over Binary's Hamming proxy (pool={}, rerank by exact L2).",
        (rb_recall - bin_recall) * 100.0,
        POOL
    );
}
