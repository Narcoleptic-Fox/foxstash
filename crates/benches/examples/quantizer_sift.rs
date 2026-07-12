//! Quantizer recall comparison on **real** SIFT10K — RaBitQ vs Binary vs SQ8.
//!
//! Run: cargo run -p foxstash-benches --example quantizer_sift --release
//!
//! Companion to `quantizer_comparison`, which uses synthetic clustered gaussians.
//! Synthetic data is a poor proxy here: it packs N/CLUSTERS vectors into each
//! sigma=0.05 ball on the unit sphere, so the true top-K sit inside one tight
//! cluster and no 1-bit code can separate them. Real SIFT has the cluster
//! structure of natural data without that pathology.
//!
//! Ground truth ships with the dataset (exact L2, 0-indexed), so recall is
//! measured against it rather than a self-computed baseline. The `Exact (flat)`
//! row is a control: it must read ~100%, otherwise the loader or the metric is
//! wrong and every other row is meaningless.

#[allow(deprecated)]
use foxstash_core::index::hnsw_quantized::BinaryHNSWIndex;
use foxstash_core::index::hnsw_quantized::{QuantizedHNSWConfig, RaBitQHNSWIndex};
use foxstash_core::vector::quantize::{BinaryQuantizer, Quantizer, ScalarQuantizer};
use foxstash_core::vector::rabitq::RaBitQuantizer;
use foxstash_core::Document;
use std::collections::HashSet;
use std::path::Path;
use std::time::Instant;

const K: usize = 10;
const POOL: usize = 100; // first-stage candidate pool
const DATA: &str = "benchmarks/data/sift10k";

/// Minimal .npy reader: v1/v2 header, little-endian, C-order only.
/// Returns (rows, cols, raw payload bytes).
fn npy_parse(bytes: &[u8], want_descr: &str) -> (usize, usize, Vec<u8>) {
    assert_eq!(&bytes[0..6], b"\x93NUMPY", "not a .npy file");
    let (header_len, data_start) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 => (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        ),
        v => panic!("unsupported .npy version {v}"),
    };
    let header = std::str::from_utf8(&bytes[data_start..data_start + header_len]).unwrap();
    assert!(
        header.contains(want_descr),
        "expected dtype {want_descr}, header: {header}"
    );
    assert!(
        header.contains("'fortran_order': False"),
        "fortran-order arrays unsupported"
    );

    let shape = header
        .split("'shape':")
        .nth(1)
        .and_then(|s| s.split('(').nth(1))
        .and_then(|s| s.split(')').next())
        .expect("no shape in header");
    let dims: Vec<usize> = shape
        .split(',')
        .filter_map(|t| t.trim().parse().ok())
        .collect();
    assert_eq!(dims.len(), 2, "expected a 2-D array, got shape ({shape})");

    (dims[0], dims[1], bytes[data_start + header_len..].to_vec())
}

fn load_f32(path: &Path) -> Vec<Vec<f32>> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let (rows, cols, payload) = npy_parse(&bytes, "<f4");
    (0..rows)
        .map(|r| {
            (0..cols)
                .map(|c| {
                    let o = (r * cols + c) * 4;
                    f32::from_le_bytes([payload[o], payload[o + 1], payload[o + 2], payload[o + 3]])
                })
                .collect()
        })
        .collect()
}

fn load_i32(path: &Path) -> Vec<Vec<i32>> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let (rows, cols, payload) = npy_parse(&bytes, "<i4");
    (0..rows)
        .map(|r| {
            (0..cols)
                .map(|c| {
                    let o = (r * cols + c) * 4;
                    i32::from_le_bytes([payload[o], payload[o + 1], payload[o + 2], payload[o + 3]])
                })
                .collect()
        })
        .collect()
}

fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Recall@K of a two-phase search against the dataset's own ground truth.
/// `stage1` scores the whole corpus (lower = better); the top `POOL` are then
/// reranked by exact L2 and cut to K.
fn rerank_recall(
    base: &[Vec<f32>],
    queries: &[Vec<f32>],
    truth: &[Vec<i32>],
    stage1: impl Fn(&[f32]) -> Vec<(f32, usize)>,
) -> f32 {
    let mut total = 0.0;
    for (qi, q) in queries.iter().enumerate() {
        let gt: HashSet<usize> = truth[qi].iter().take(K).map(|&i| i as usize).collect();

        let mut est = stage1(q);
        est.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut pool: Vec<(f32, usize)> = est
            .iter()
            .take(POOL)
            .map(|&(_, i)| (l2_sq(q, &base[i]), i))
            .collect();
        pool.sort_by(|a, b| a.0.total_cmp(&b.0));

        let got: HashSet<usize> = pool.iter().take(K).map(|(_, i)| *i).collect();
        total += gt.intersection(&got).count() as f32 / K as f32;
    }
    total / queries.len() as f32
}

/// Recall@K where `search` returns the index's own top-K node ids for a query.
/// Unlike `rerank_recall` this does no brute-force stage 1 — it measures whatever
/// the index actually returns, approximation and all.
fn index_recall(
    queries: &[Vec<f32>],
    truth: &[Vec<i32>],
    search: impl Fn(&[f32]) -> Vec<usize>,
) -> f32 {
    let mut total = 0.0;
    for (qi, q) in queries.iter().enumerate() {
        let gt: HashSet<usize> = truth[qi].iter().take(K).map(|&i| i as usize).collect();
        let got: HashSet<usize> = search(q).into_iter().take(K).collect();
        total += gt.intersection(&got).count() as f32 / K as f32;
    }
    total / queries.len() as f32
}

fn main() {
    let dir = Path::new(DATA);
    let base = load_f32(&dir.join("base.npy"));
    let queries = load_f32(&dir.join("query.npy"));
    let truth = load_i32(&dir.join("groundtruth.npy"));
    let dim = base[0].len();

    println!("=== Quantizer Recall Comparison (two-phase, REAL SIFT10K) ===");
    println!(
        "{} vectors, {}d, {} queries, top-{}, pool={}\n",
        base.len(),
        dim,
        queries.len(),
        K,
        POOL
    );

    // Control: exact L2 over the whole corpus. Must be ~100% or the loader/metric is wrong.
    let exact = rerank_recall(&base, &queries, &truth, |q| {
        base.iter()
            .enumerate()
            .map(|(i, v)| (l2_sq(q, v), i))
            .collect()
    });

    // RaBitQ (32x): asymmetric unbiased estimate.
    let rabitq = RaBitQuantizer::fit(&base);
    let rb_codes: Vec<_> = base.iter().map(|v| rabitq.encode(v)).collect();
    let rb = rerank_recall(&base, &queries, &truth, |q| {
        let prep = rabitq.prepare_query(q);
        rb_codes
            .iter()
            .enumerate()
            .map(|(i, c)| (rabitq.estimate_dist_sq(&prep, c), i))
            .collect()
    });

    // Binary (32x): Hamming between quantized query and docs.
    let binq = BinaryQuantizer::new(dim);
    let bin_codes: Vec<_> = base.iter().map(|v| binq.quantize(v)).collect();
    let bin = rerank_recall(&base, &queries, &truth, |q| {
        let qc = binq.quantize(q);
        bin_codes
            .iter()
            .enumerate()
            .map(|(i, c)| (binq.distance_quantized(&qc, c), i))
            .collect()
    });

    // Binary, mean-centered (32x): identical code path, but thresholded at the
    // per-dimension centroid instead of at zero. SIFT is entirely non-negative,
    // so the zero threshold sets every bit and all codes collapse to all-ones.
    let centroid: Vec<f32> = (0..dim)
        .map(|d| base.iter().map(|v| v[d]).sum::<f32>() / base.len() as f32)
        .collect();
    let center = |v: &[f32]| -> Vec<f32> { v.iter().zip(&centroid).map(|(x, c)| x - c).collect() };
    let cbin_codes: Vec<_> = base.iter().map(|v| binq.quantize(&center(v))).collect();
    let cbin = rerank_recall(&base, &queries, &truth, |q| {
        let qc = binq.quantize(&center(q));
        cbin_codes
            .iter()
            .enumerate()
            .map(|(i, c)| (binq.distance_quantized(&qc, c), i))
            .collect()
    });

    // SQ8 (4x): asymmetric, full-precision query vs quantized docs.
    let sq8 = ScalarQuantizer::fit(&base);
    let sq_codes: Vec<_> = base.iter().map(|v| sq8.quantize(v)).collect();
    let sq = rerank_recall(&base, &queries, &truth, |q| {
        sq_codes
            .iter()
            .enumerate()
            .map(|(i, c)| (sq8.distance_asymmetric(q, c), i))
            .collect()
    });

    // ---- End-to-end through the real HNSW indexes -------------------------------
    // The rows above compare quantizers in isolation (brute-force stage 1 over the
    // whole corpus). These rows exercise the actual index: approximate graph traversal
    // using the quantized estimate, then rerank. This is what a caller actually gets.
    let docs = |v: &Vec<f32>, i: usize| Document {
        id: i.to_string(),
        content: String::new(),
        embedding: v.clone(),
        metadata: None,
    };

    let t0 = Instant::now();
    let mut rb_index = RaBitQHNSWIndex::fit(&base, QuantizedHNSWConfig::default());
    for (i, v) in base.iter().enumerate() {
        rb_index.add_with_full_precision(docs(v, i)).unwrap();
    }
    let rb_build = t0.elapsed();

    let t0 = Instant::now();
    let rb_hnsw = index_recall(&queries, &truth, |q| {
        rb_index
            .search_and_rerank(q, POOL, K)
            .unwrap()
            .into_iter()
            .filter_map(|r| r.id.parse::<usize>().ok())
            .collect()
    });
    let rb_qps = queries.len() as f64 / t0.elapsed().as_secs_f64();

    #[allow(deprecated)]
    let (bin_hnsw, bin_build, bin_qps) = {
        let t0 = Instant::now();
        let mut idx = BinaryHNSWIndex::with_full_precision(dim, QuantizedHNSWConfig::default());
        for (i, v) in base.iter().enumerate() {
            idx.add_with_full_precision(docs(v, i)).unwrap();
        }
        let build = t0.elapsed();

        let t0 = Instant::now();
        let recall = index_recall(&queries, &truth, |q| {
            idx.search_and_rerank(q, POOL, K)
                .unwrap()
                .into_iter()
                .filter_map(|r| r.id.parse::<usize>().ok())
                .collect()
        });
        (
            recall,
            build,
            queries.len() as f64 / t0.elapsed().as_secs_f64(),
        )
    };

    println!("{:<22} {:>10} {:>14}", "Quantizer", "Compress", "Recall@10");
    println!("{:-<48}", "");
    println!(
        "{:<22} {:>10} {:>13.1}%",
        "Exact (flat, control)",
        "1x",
        exact * 100.0
    );
    println!(
        "{:<22} {:>10} {:>13.1}%",
        "RaBitQ (1-bit)",
        "32x",
        rb * 100.0
    );
    println!(
        "{:<22} {:>10} {:>13.1}%",
        "Binary (Hamming)",
        "32x",
        bin * 100.0
    );
    println!(
        "{:<22} {:>10} {:>13.1}%",
        "Binary (centered)",
        "32x",
        cbin * 100.0
    );
    println!(
        "{:<22} {:>10} {:>13.1}%",
        "SQ8 (scalar int8)",
        "4x",
        sq * 100.0
    );

    println!("\n--- End-to-end through the HNSW index (approximate graph traversal) ---");
    println!(
        "{:<22} {:>10} {:>14} {:>10} {:>10}",
        "Index", "Compress", "Recall@10", "Build", "QPS"
    );
    println!("{:-<70}", "");
    println!(
        "{:<22} {:>10} {:>13.1}% {:>9.1}s {:>10.0}",
        "RaBitQHNSWIndex",
        "32x",
        rb_hnsw * 100.0,
        rb_build.as_secs_f64(),
        rb_qps
    );
    println!(
        "{:<22} {:>10} {:>13.1}% {:>9.1}s {:>10.0}",
        "BinaryHNSWIndex (dep.)",
        "32x",
        bin_hnsw * 100.0,
        bin_build.as_secs_f64(),
        bin_qps
    );

    if exact < 0.99 {
        println!(
            "\n!! Control row is {:.1}%, not ~100%. The loader or distance metric is wrong;\n\
             !! disregard the quantizer rows until this reads 100%.",
            exact * 100.0
        );
    } else {
        println!(
            "\nBinary's zero threshold is degenerate here: SIFT is entirely non-negative, so\n\
             every bit sets, all codes collapse to all-ones, and stage 1 ranks arbitrarily.\n\
             Centering the threshold at the corpus centroid recovers {:+.1} pts ({:.1}% -> {:.1}%).",
            (cbin - bin) * 100.0,
            bin * 100.0,
            cbin * 100.0
        );
        println!(
            "\nApples-to-apples at 32x, RaBitQ beats a *working* binary baseline by {:+.1} pts\n\
             ({:.1}% vs {:.1}%). Comparing against the broken {:.1}% baseline would overstate\n\
             RaBitQ's gain as {:+.1} pts. (pool={}, rerank by exact L2, dataset ground truth.)",
            (rb - cbin) * 100.0,
            rb * 100.0,
            cbin * 100.0,
            bin * 100.0,
            (rb - bin) * 100.0,
            POOL
        );
    }
}
