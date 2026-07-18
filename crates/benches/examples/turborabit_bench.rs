//! Before/after harness for the turborabit optimization program (#12/#14/#15).
//!
//! Reads raw dumps of a VIBE dataset (train/test f32 row-major, gt i32; meta.txt = "n dim nq gtk")
//! and reports recall@100 + QPS across an ef sweep for one storage mode. Real embeddings, so the
//! numbers are comparable to VIBE (index-set recall@100, single-threaded query).
//!
//! Env:  VECS_DIR (required), STORAGE=turborabit4|sq8|f32 (default turborabit4),
//!       M=32 EFC=500 RERANK=200
//! Run:  VECS_DIR=<dir> cargo run -p foxstash-benches --example turborabit_bench --release

use foxstash_core::index::hnsw::{DistanceMetric, HNSWConfig, HNSWIndex, Storage};
use std::collections::HashSet;
use std::time::Instant;

fn read_f32(p: &str) -> Vec<f32> {
    std::fs::read(p).unwrap_or_else(|e| panic!("read {p}: {e}"))
        .chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}
fn read_i32(p: &str) -> Vec<i32> {
    std::fs::read(p).unwrap_or_else(|e| panic!("read {p}: {e}"))
        .chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect()
}
fn env(k: &str, d: &str) -> String { std::env::var(k).unwrap_or_else(|_| d.into()) }

fn main() {
    let dir = std::env::var("VECS_DIR").expect("set VECS_DIR to a dumped dataset dir");
    let meta = std::fs::read_to_string(format!("{dir}/meta.txt")).expect("meta.txt");
    let m: Vec<usize> = meta.split_whitespace().map(|x| x.parse().unwrap()).collect();
    let (n, dim, nq, gtk) = (m[0], m[1], m[2], m[3]);
    let k = 100usize;

    let storage_s = env("STORAGE", "turborabit4");
    let (storage, rabit_bits, turbo_bits) = match storage_s.as_str() {
        "f32" => (Storage::F32, 3, 2),
        "sq8" => (Storage::SQ8, 3, 2),
        "turborabit4" => (Storage::TurboRabit, 4, 2),
        "turborabit3" => (Storage::TurboRabit, 3, 2),
        other => panic!("unsupported STORAGE {other}"),
    };
    let mm: usize = env("M", "32").parse().unwrap();
    let efc: usize = env("EFC", "500").parse().unwrap();
    let rerank: usize = if storage == Storage::F32 { 0 } else { env("RERANK", "200").parse().unwrap() };

    let train_flat = read_f32(&format!("{dir}/train.bin"));
    let queries: Vec<Vec<f32>> = read_f32(&format!("{dir}/test.bin")).chunks_exact(dim).map(|r| r.to_vec()).collect();
    let gt = read_i32(&format!("{dir}/gt.bin"));
    let truth: Vec<HashSet<usize>> = gt.chunks_exact(gtk).map(|r| r.iter().take(k).map(|&i| i as usize).collect()).collect();
    let train: Vec<Vec<f32>> = train_flat.chunks_exact(dim).map(|r| r.to_vec()).collect();
    println!("{}: {n} x {dim}, {nq} q | {storage_s} M={mm} efc={efc} rerank={rerank}",
             dir.rsplit('/').next().unwrap_or(&dir));

    let config = HNSWConfig {
        metric: DistanceMetric::Cosine, m: mm, m0: mm * 2, ef_construction: efc,
        storage, turbo_bits, rabit_bits, rerank_candidates: rerank, seed: Some(7),
        ..Default::default()
    };
    // Snapshot cache: the parallel builder is non-reproducible at fixed seed, so a fair
    // before/after (AVX-512 kernel, graph reordering) MUST reuse the SAME graph — not rebuild.
    // Build once, snapshot; every later variant loads it and only the thing under test changes.
    // REBUILD=1 forces a fresh build (and re-saves).
    let snap = format!("{dir}/idx_{storage_s}_M{mm}_efc{efc}_r{rerank}.snap");
    let snap_path = std::path::Path::new(&snap);
    let mut index = if snap_path.exists() && std::env::var("REBUILD").is_err() {
        let t0 = Instant::now();
        let ix = HNSWIndex::snapshot_from_file(snap_path).expect("load snapshot");
        println!("loaded snapshot {:.1}s", t0.elapsed().as_secs_f64());
        ix
    } else {
        let t0 = Instant::now();
        let ix = HNSWIndex::build_parallel(train, config);
        println!("build {:.0}s", t0.elapsed().as_secs_f64());
        ix.snapshot_to_file(snap_path).expect("save snapshot");
        ix
    };
    println!("{:>5} {:>11} {:>9}", "ef", "recall@100", "QPS");
    for ef in [100usize, 200, 300, 500, 800] {
        index.set_ef_search(ef);
        let t0 = Instant::now();
        let mut hits = 0usize;
        for (q, gtset) in queries.iter().zip(&truth) {
            let got: HashSet<usize> = index.search(q, k).expect("search").into_iter()
                .filter_map(|r| r.id.parse::<usize>().ok()).collect();
            hits += gtset.intersection(&got).count();
        }
        let qps = nq as f64 / t0.elapsed().as_secs_f64();
        println!("{ef:>5} {:>11.4} {qps:>9.0}", hits as f64 / (nq * k) as f64);
    }
}
