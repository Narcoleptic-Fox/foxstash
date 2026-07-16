//! QPS smoke test for the arena-packed multi-bit storages.
//!
//! Run: cargo run -p foxstash-benches --example multibit_qps_smoke --release
//!
//! NOT a benchmark — a regression tripwire. The coco 2026-07-15 sweep measured the
//! unpacked side-table TurboQuant at 40–165 QPS vs RaBitQ's 900–2,800 (10–40x). After
//! arena-packing, TurboQuant and TurboRabit should sit in the same order of magnitude
//! as RaBitQ (they read comparable bytes per node visit through the same kind of
//! kernel). If either is still >5x off RaBitQ here, do NOT burn a VIBE sweep — the
//! packing has a hole.
//!
//! Clustered cosine data at 768-d (the RAG case); recall@10 printed alongside so a QPS
//! win from a broken estimator (blind-but-fast) is visible immediately.

use foxstash_core::index::hnsw::{DistanceMetric, HNSWConfig, HNSWIndex, Storage};
use std::collections::HashSet;
use std::time::Instant;

const N: usize = 50_000;
const NQ: usize = 500;
const K: usize = 10;
const CLUSTERS: usize = 200;
const SIGMA: f32 = 0.05;
const DIM: usize = 768;
const EF: usize = 100;
const RERANK: usize = 100;

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
    fn f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn gauss(&mut self) -> f32 {
        let u1 = self.f32().max(1e-7);
        let u2 = self.f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

fn make_clustered(rng: &mut Rng, n: usize) -> Vec<Vec<f32>> {
    let centers: Vec<Vec<f32>> = (0..CLUSTERS)
        .map(|_| (0..DIM).map(|_| rng.gauss()).collect())
        .collect();
    (0..n)
        .map(|i| {
            let c = &centers[i % CLUSTERS];
            c.iter().map(|x| x + SIGMA * rng.gauss()).collect()
        })
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut d, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for (x, y) in a.iter().zip(b) {
        d += x * y;
        na += x * x;
        nb += y * y;
    }
    d / (na.sqrt() * nb.sqrt()).max(1e-9)
}

fn main() {
    let mut rng = Rng::new(42);
    let base = make_clustered(&mut rng, N);
    let queries = make_clustered(&mut rng, NQ);

    println!("ground truth (brute force cosine, {N} x {NQ})...");
    let truth: Vec<HashSet<usize>> = queries
        .iter()
        .map(|q| {
            let mut s: Vec<(f32, usize)> = base
                .iter()
                .enumerate()
                .map(|(i, v)| (cosine(v, q), i))
                .collect();
            s.sort_by(|a, b| b.0.total_cmp(&a.0));
            s.into_iter().take(K).map(|(_, i)| i).collect()
        })
        .collect();

    println!(
        "{:<14} {:>9} {:>10} {:>9}",
        "storage", "build_s", "QPS", "recall@10"
    );
    for (label, storage, tb, rb) in [
        ("f32", Storage::F32, 2, 3),
        ("sq8", Storage::SQ8, 2, 3),
        ("rabitq", Storage::RaBitQ, 2, 3),
        ("turboquant3", Storage::TurboQuant, 3, 3),
        ("turboquant4", Storage::TurboQuant, 4, 3),
        ("turborabit3", Storage::TurboRabit, 2, 3),
        ("turborabit4", Storage::TurboRabit, 2, 4),
    ] {
        let config = HNSWConfig {
            metric: DistanceMetric::Cosine,
            m: 16,
            m0: 32,
            ef_construction: 200,
            ef_search: EF,
            storage,
            turbo_bits: tb,
            rabit_bits: rb,
            rerank_candidates: if storage == Storage::F32 { 0 } else { RERANK },
            seed: Some(7),
            ..Default::default()
        };
        let t0 = Instant::now();
        let index = HNSWIndex::build_parallel(base.clone(), config);
        let build_s = t0.elapsed().as_secs_f64();

        let t0 = Instant::now();
        let mut hits = 0usize;
        for (q, gt) in queries.iter().zip(&truth) {
            let got: HashSet<usize> = index
                .search(q, K)
                .expect("search")
                .into_iter()
                .filter_map(|r| r.id.parse::<usize>().ok())
                .collect();
            hits += gt.intersection(&got).count();
        }
        let qps = NQ as f64 / t0.elapsed().as_secs_f64();
        let recall = hits as f32 / (NQ * K) as f32;
        println!("{label:<14} {build_s:>9.1} {qps:>10.0} {recall:>9.4}");
    }
}
