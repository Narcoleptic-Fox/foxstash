//! Spike for #50: how much slower is reopen-by-rebuild than reopen-by-snapshot?
//!
//! Encrypted stores may disable the graph-snapshot cache (no plaintext vectors on disk), so reopen
//! would rebuild the HNSW graph from the checkpoint instead of loading the snapshot. This measures
//! both costs so we can decide whether that is marginal (disable snapshots) or not (encrypt them).
//!
//! Usage: cargo run --release -p foxstash-benches --example snapshot_reopen_spike [dataset] [N...]
//!   dataset defaults to rustdocs45k (real 45k x 768 EmbeddingGemma vectors).
//!   Each N tiles the base (with tiny per-copy noise) up to N vectors; defaults sweep toward 200k.

use foxstash_benches::sift::Dataset;
use foxstash_core::index::{HNSWConfig, HNSWIndex};
use foxstash_core::Document;
use std::time::Instant;

fn tiled_docs(base: &[Vec<f32>], n: usize) -> Vec<Document> {
    (0..n)
        .map(|i| {
            let src = &base[i % base.len()];
            let copy = (i / base.len()) as f32;
            // Tiny deterministic perturbation so tiled copies are not exact duplicates (which would
            // degenerate the graph and distort build time); negligible vs unit-norm vectors.
            let embedding: Vec<f32> = src
                .iter()
                .enumerate()
                .map(|(j, &x)| x + copy * 1e-4 * (((j % 7) as f32) - 3.0))
                .collect();
            Document {
                id: format!("d{i}"),
                content: String::new(),
                embedding,
                metadata: None,
            }
        })
        .collect()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let name = args.next().unwrap_or_else(|| "rustdocs45k".to_string());
    let sizes: Vec<usize> = {
        let rest: Vec<usize> = args.filter_map(|s| s.parse().ok()).collect();
        if rest.is_empty() {
            vec![45_000, 90_000, 180_000]
        } else {
            rest
        }
    };

    let ds = Dataset::load("benchmarks/data", &name).expect("load dataset");
    let dim = ds.dim();
    eprintln!("dataset {name}: {} base x {dim}d\n", ds.base.len());

    println!(
        "{:>10} {:>5} {:>14} {:>14} {:>12} {:>10}",
        "N", "dim", "rebuild(s)", "snap_load(s)", "slowdown", "snap(MB)"
    );
    for &n in &sizes {
        let docs = tiled_docs(&ds.base, n);
        let cfg = HNSWConfig::default();

        let t = Instant::now();
        let index = HNSWIndex::build_parallel_from_documents(docs, cfg);
        let rebuild = t.elapsed().as_secs_f64();

        let tmp = std::env::temp_dir().join(format!("reopen_spike_{n}.snapshot"));
        index.snapshot_to_file(&tmp).expect("snapshot");
        let snap_mb = std::fs::metadata(&tmp)
            .map(|m| m.len() as f64 / 1e6)
            .unwrap_or(0.0);

        let t = Instant::now();
        let loaded = HNSWIndex::snapshot_from_file(&tmp).expect("load snapshot");
        let load = t.elapsed().as_secs_f64();
        assert_eq!(loaded.len(), n, "snapshot must round-trip all vectors");
        let _ = std::fs::remove_file(&tmp);

        println!(
            "{n:>10} {dim:>5} {rebuild:>14.2} {load:>14.3} {:>11.0}x {snap_mb:>10.1}",
            rebuild / load.max(1e-9)
        );
    }
}
