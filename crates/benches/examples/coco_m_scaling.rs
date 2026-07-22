//! Does lifting the parallel builder's M≤32 cap actually buy anything? (task #13 gate)
//!
//! The parallel builder panics at m>32 / m0>64 (fixed-size arena node arrays); only the slow
//! `Sequential` builder can go denser. hnswlib's coco optimum is M=48, and on the VIBE matched
//! grid hnswlib-M48 owned the very top of the frontier that foxstash-M32 could not reach. Before
//! spending an arena refactor to lift the cap, measure the prize: build f32 at Parallel-M=32 and
//! Sequential-M=48 on the SAME real coco/nomic embeddings and compare the recall/QPS frontier.
//! f32 storage, so this isolates the GRAPH (no quantizer confound).
//!
//! Reads raw dumps of the VIBE coco hdf5 (see COCO_VECS below): train/test f32 row-major, gt i32.
//! Run: COCO_VECS=<dir> cargo run -p foxstash-benches --example coco_m_scaling --release

use foxstash_core::index::hnsw::{BuildStrategy, DistanceMetric, HNSWConfig, HNSWIndex, Storage};
use std::collections::HashSet;
use std::time::Instant;

fn read_f32(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}
fn read_i32(path: &str) -> Vec<i32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn main() {
    let dir = std::env::var("COCO_VECS").unwrap_or_else(|_| {
        "/tmp/claude-1000/-home-dieshen/76045e7d-2c0e-4336-8596-9e5660d19a3d/scratchpad/coco-vecs"
            .into()
    });
    let dim = 768usize;
    let k = 100usize;

    let train_flat = read_f32(&format!("{dir}/train.bin"));
    let test_flat = read_f32(&format!("{dir}/test.bin"));
    let gt_flat = read_i32(&format!("{dir}/gt.bin"));
    let n = train_flat.len() / dim;
    let nq = test_flat.len() / dim;
    let gt_k = gt_flat.len() / nq;
    println!("coco: {n} train x {dim}, {nq} queries, gt top-{gt_k}");

    let train: Vec<Vec<f32>> = train_flat.chunks_exact(dim).map(|r| r.to_vec()).collect();
    let queries: Vec<Vec<f32>> = test_flat.chunks_exact(dim).map(|r| r.to_vec()).collect();
    let truth: Vec<HashSet<usize>> = gt_flat
        .chunks_exact(gt_k)
        .map(|row| row.iter().take(k).map(|&i| i as usize).collect())
        .collect();

    // Graphs at matched efc, f32 storage. Parallel-M32 vs Sequential-M48 conflates degree with
    // builder; Sequential-M32 is the control that isolates each (same builder as M48 / same
    // degree as the parallel run). Set COCO_SEQ32_ONLY=1 to run just that control.
    let configs: Vec<(&str, usize, usize, BuildStrategy)> =
        if std::env::var("COCO_SEQ32_ONLY").is_ok() {
            vec![("sequential M=32", 32, 64, BuildStrategy::Sequential)]
        } else {
            vec![
                ("parallel M=32", 32, 64, BuildStrategy::Parallel),
                ("sequential M=48", 48, 96, BuildStrategy::Sequential),
            ]
        };
    for (label, m, m0, strat) in configs {
        let config = HNSWConfig {
            metric: DistanceMetric::Cosine,
            m,
            m0,
            ef_construction: 500,
            storage: Storage::F32,
            build_strategy: strat,
            seed: Some(7),
            ..Default::default()
        };
        let t0 = Instant::now();
        let mut index = HNSWIndex::build(train.clone(), config);
        let build_s = t0.elapsed().as_secs_f64();
        println!("\n=== {label}  (build {build_s:.0}s) ===");
        println!("{:>6} {:>10} {:>11}", "ef", "recall@100", "QPS");

        for ef in [50usize, 100, 200, 300, 500, 800] {
            index.set_ef_search(ef);
            let t0 = Instant::now();
            let mut hits = 0usize;
            for (q, gt) in queries.iter().zip(&truth) {
                let got: HashSet<usize> = index
                    .search(q, k)
                    .expect("search")
                    .into_iter()
                    .filter_map(|r| r.id.parse::<usize>().ok())
                    .collect();
                hits += gt.intersection(&got).count();
            }
            let qps = nq as f64 / t0.elapsed().as_secs_f64();
            let recall = hits as f64 / (nq * k) as f64;
            println!("{ef:>6} {recall:>10.4} {qps:>11.0}");
        }
    }
}
