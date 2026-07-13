//! Benchmark foxstash's indexes against **real** SIFT, emitting the same JSON schema
//! the Python harness uses so results merge into one report.
//!
//! Run: `cargo run --release -p foxstash-benches --example sift_bench [dataset]`
//! (dataset defaults to `sift10k`; `sift100k` and `sift1m` also live in benchmarks/data)
//!
//! Foxstash has never appeared in `benchmarks/results/report_sift10k.md` — only annoy,
//! faiss and hnswlib do — because nothing in Rust could read the `.npy` files. Its only
//! published comparison numbers come from `benchmarks/RESULTS.md`, which runs on synthetic
//! vectors where every ANN collapses to ~60% recall. This closes that gap.
//!
//! The `flat` row is a control. If it is not 100% recall, the loader or the metric is
//! wrong and every other row is void.

use foxstash_benches::sift::{l2_sq, Dataset};
use foxstash_core::index::hnsw::{BuildStrategy, DistanceMetric, HNSWConfig, HNSWIndex, Storage};
use foxstash_core::Document;
use serde_json::json;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const DATA_ROOT: &str = "benchmarks/data";
const OUT_DIR: &str = "benchmarks/results";

/// One row of the report, matching the Python harness's record shape.
#[allow(clippy::too_many_arguments)]
fn record(
    algorithm: &str,
    ds: &Dataset,
    build_time: f64,
    memory_mb: f64,
    recall10: f32,
    recall100: f32,
    qps: f64,
    params: serde_json::Value,
    notes: &str,
) -> serde_json::Value {
    json!({
        "library": "foxstash",
        "algorithm": algorithm,
        "dataset": ds.name,
        "index_size": ds.base.len(),
        "build_time_sec": build_time,
        "peak_memory_mb": memory_mb,
        "recall_at_10": recall10,
        "recall_at_100": recall100,
        "qps": qps,
        "parameters": params,
        "notes": notes,
    })
}

fn doc(i: usize, v: &[f32]) -> Document {
    Document {
        id: i.to_string(),
        content: String::new(),
        embedding: v.to_vec(),
        metadata: None,
    }
}

/// Parse the numeric ids the indexes hand back out of `SearchResult`.
fn ids(results: Vec<foxstash_core::SearchResult>) -> Vec<usize> {
    results
        .into_iter()
        .filter_map(|r| r.id.parse::<usize>().ok())
        .collect()
}

/// Time `search` across every query and return queries-per-second.
fn measure_qps(ds: &Dataset, k: usize, search: impl Fn(&[f32]) -> Vec<usize>) -> f64 {
    let start = Instant::now();
    for q in &ds.queries {
        std::hint::black_box(search(q));
    }
    let _ = k;
    ds.queries.len() as f64 / start.elapsed().as_secs_f64()
}

fn main() {
    let name = std::env::args().nth(1).unwrap_or_else(|| "sift10k".into());
    let ds = Dataset::load(DATA_ROOT, &name)
        .unwrap_or_else(|e| panic!("loading {DATA_ROOT}/{name}: {e}"));
    let dim = ds.dim();

    eprintln!(
        "== {} : {} base x {}d, {} queries ==",
        ds.name,
        ds.base.len(),
        dim,
        ds.queries.len()
    );

    let mut out = Vec::new();

    // ---- Control: exact brute force. Must be 100% or nothing below is meaningful. ----
    eprintln!("[1/5] flat (exact control)");
    let t = Instant::now();
    let flat_recall10 = ds.exact_control(10);
    let flat_qps = measure_qps(&ds, 10, |q| {
        let mut d: Vec<(f32, usize)> = ds
            .base
            .iter()
            .enumerate()
            .map(|(i, v)| (l2_sq(q, v), i))
            .collect();
        d.sort_by(|a, b| a.0.total_cmp(&b.0));
        d.into_iter().take(10).map(|(_, i)| i).collect()
    });
    let flat_build = t.elapsed().as_secs_f64();
    assert!(
        flat_recall10 > 0.99,
        "CONTROL FAILED: exact search scored {:.1}% against the shipped ground truth. \
         The loader or the distance metric is wrong; every other row would be void.",
        flat_recall10 * 100.0
    );
    out.push(record(
        "flat",
        &ds,
        flat_build,
        (ds.base.len() * dim * 4) as f64 / 1e6,
        flat_recall10,
        ds.exact_control(100),
        flat_qps,
        json!({ "k": 10 }),
        "Brute force control (must be 100% recall)",
    ));

    // ---- HNSW L2: the row that is actually comparable to faiss/hnswlib ----
    //
    // SIFT's ground truth is exact L2. Until DistanceMetric existed, HNSWIndex was
    // cosine-only and could not be scored against it at all (it read ~55%, measuring the
    // metric gap rather than the graph). This row is the apples-to-apples comparison.
    // Swept over ef_search so the recall/QPS tradeoff is visible, and so the comparison
    // against hnswlib (whose best row uses ef=500) is at matching effort rather than
    // matching only in name.
    for ef in [100usize, 200, 500] {
        eprintln!("[2/7] hnsw-l2 (ef_search={ef})");
        let l2_config = HNSWConfig {
            metric: DistanceMetric::L2,
            m: 32,
            m0: 64,
            ef_construction: 200,
            ef_search: ef,
            build_strategy: BuildStrategy::Parallel,
            ..Default::default()
        };
        let t = Instant::now();
        let l2_index = HNSWIndex::build_parallel(ds.base.clone(), l2_config);
        let l2_build = t.elapsed().as_secs_f64();
        let l2_r10 = ds.recall_at(10, |q| ids(l2_index.search(q, 10).unwrap()));
        let l2_r100 = ds.recall_at(100, |q| ids(l2_index.search(q, 100).unwrap()));
        let l2_qps = measure_qps(&ds, 10, |q| ids(l2_index.search(q, 10).unwrap()));
        out.push(record(
            &format!("hnsw-l2 (ef={ef})"),
            &ds,
            l2_build,
            (ds.base.len() * dim * 4) as f64 / 1e6,
            l2_r10,
            l2_r100,
            l2_qps,
            json!({ "m": 32, "ef_construction": 200, "ef_search": ef,
                    "metric": "L2", "build": "parallel" }),
            "Full precision, L2 metric. Directly comparable to faiss/hnswlib on SIFT's L2 \
             ground truth.",
        ));
    }

    // ---- HNSW cosine, for reference ----
    //
    // Scored against a brute-forced COSINE ground truth, which is the fair question for a
    // cosine index. Not comparable to the L2 rows above — different question entirely.
    eprintln!("      computing cosine ground truth (brute force)...");
    let cos_truth = ds.cosine_truth(100);

    for (label, build_fn) in [
        (
            "hnsw-incremental",
            &(|d: &Dataset, c: HNSWConfig| {
                let mut ix = HNSWIndex::new(d.dim(), c);
                for (i, v) in d.base.iter().enumerate() {
                    ix.add(doc(i, v)).unwrap();
                }
                ix
            }) as &dyn Fn(&Dataset, HNSWConfig) -> HNSWIndex,
        ),
        ("hnsw-sequential", &|d: &Dataset, c: HNSWConfig| {
            HNSWIndex::build(
                d.base.clone(),
                HNSWConfig {
                    build_strategy: BuildStrategy::Sequential,
                    ..c
                },
            )
        }),
        ("hnsw-parallel", &|d: &Dataset, c: HNSWConfig| {
            HNSWIndex::build_parallel(
                d.base.clone(),
                HNSWConfig {
                    build_strategy: BuildStrategy::Parallel,
                    ..c
                },
            )
        }),
    ] {
        eprintln!("[2/5] {label}");
        let config = HNSWConfig {
            m: 32,
            m0: 64,
            ef_construction: 200,
            ef_search: 100,
            ..Default::default()
        };

        let t = Instant::now();
        let index = build_fn(&ds, config);
        let build = t.elapsed().as_secs_f64();

        let cos10 = ds.recall_against(&cos_truth, 10, |q| ids(index.search(q, 10).unwrap()));
        let cos100 = ds.recall_against(&cos_truth, 100, |q| ids(index.search(q, 100).unwrap()));
        let l2_10 = ds.recall_at(10, |q| ids(index.search(q, 10).unwrap()));
        let qps = measure_qps(&ds, 10, |q| ids(index.search(q, 10).unwrap()));

        out.push(record(
            label,
            &ds,
            build,
            (ds.base.len() * dim * 4) as f64 / 1e6,
            cos10,
            cos100,
            qps,
            json!({ "m": 32, "ef_construction": 200, "ef_search": 100,
                    "metric": "cosine", "recall_vs_l2_truth": l2_10 }),
            "Full precision (f32), cosine metric. recall_at_* are vs COSINE ground truth; \
             recall_vs_l2_truth shows what it scores against SIFT's L2 key (metric mismatch, \
             not an index defect).",
        ));
    }

    // ---- RaBitQ (32x), two-phase — now a STORAGE MODE, not a separate index type ----
    // `RaBitQHNSWIndex` was deleted; `Storage::RaBitQ` on the unified index does the same job
    // with the arena layout, a rayon build, and an honest `metric` field. The legacy type's
    // per-query rerank pool survives as `set_rerank_candidates`.
    eprintln!("[5/5] rabitq (Storage::RaBitQ)");
    const POOL: usize = 100;
    let t = Instant::now();
    let mut rb = HNSWIndex::build_parallel(
        ds.base.clone(),
        HNSWConfig {
            metric: DistanceMetric::L2,
            storage: Storage::RaBitQ,
            rerank_candidates: POOL,
            ..Default::default()
        },
    );
    let rb_build = t.elapsed().as_secs_f64();
    let rb_r10 = ds.recall_at(10, |q| ids(rb.search(q, 10).unwrap()));
    rb.set_rerank_candidates(POOL.max(200)).unwrap();
    let rb_r100 = ds.recall_at(100, |q| ids(rb.search(q, 100).unwrap()));
    rb.set_rerank_candidates(POOL).unwrap();
    let rb_qps = measure_qps(&ds, 10, |q| ids(rb.search(q, 10).unwrap()));
    out.push(record(
        "rabitq-storage",
        &ds,
        rb_build,
        rb.memory_breakdown().total() as f64 / 1e6,
        rb_r10,
        rb_r100,
        rb_qps,
        json!({ "compression": "32x", "rerank_pool": POOL }),
        "1-bit RaBitQ first stage + exact rerank. Supersedes BinaryHNSW, which \
         degenerates to chance-level recall on non-negative data — a zero threshold sets \
         every bit, collapsing all codes to all-ones. See benchmarks/RESULTS.md.",
    ));

    // ---- Emit ----
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let path = format!("{OUT_DIR}/results_foxstash_{}_{}.json", ds.name, stamp);
    std::fs::write(&path, serde_json::to_string_pretty(&out).unwrap()).unwrap();

    println!(
        "{:<16} {:>10} {:>10} {:>12} {:>10}",
        "Index", "Recall@10", "Recall@100", "Build (s)", "QPS"
    );
    println!("{:-<64}", "");
    for r in &out {
        println!(
            "{:<20} {:>9.1}% {:>9.1}% {:>11.2} {:>10.0}",
            r["algorithm"].as_str().unwrap(),
            r["recall_at_10"].as_f64().unwrap() * 100.0,
            r["recall_at_100"].as_f64().unwrap() * 100.0,
            r["build_time_sec"].as_f64().unwrap(),
            r["qps"].as_f64().unwrap(),
        );
    }
    eprintln!("\nwrote {path}");
}
