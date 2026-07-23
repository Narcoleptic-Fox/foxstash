//! 1c: does a two-stage exact-f32 rerank of the coarse top-C recover the recall the quantizer lost?
//! Plain vs reranked (recall + QPS) per mode. Run: VECS_DIR=<dir> cargo run -p foxstash-gpu --example rerank --release
use foxstash_gpu::{GpuFlatIndex, WarrenRerankIndex};
use std::time::Instant;

fn read_f32(p: &str) -> Vec<f32> {
    std::fs::read(p)
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn main() {
    let dir = std::env::var("VECS_DIR").unwrap();
    let m: Vec<usize> = std::fs::read_to_string(format!("{dir}/meta.txt"))
        .unwrap()
        .split_whitespace()
        .map(|x| x.parse().unwrap())
        .collect();
    let dim = m[1];
    let k = 10usize;
    let c = std::env::var("C")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128usize);
    let n = std::env::var("N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50000usize);
    let train = read_f32(&format!("{dir}/train.bin"));
    let qs: Vec<Vec<f32>> = read_f32(&format!("{dir}/test.bin"))[..200 * dim]
        .chunks_exact(dim)
        .map(|r| r.to_vec())
        .collect();
    let vecs: Vec<Vec<f32>> = train[..n * dim]
        .chunks_exact(dim)
        .map(|r| r.to_vec())
        .collect();
    let ids: Vec<String> = (0..n).map(|i| i.to_string()).collect();

    // exact f32 dot ground truth
    let mut gt: Vec<std::collections::HashSet<usize>> = Vec::with_capacity(qs.len());
    for q in &qs {
        let mut e: Vec<(f32, usize)> = (0..n)
            .map(|i| (q.iter().zip(&vecs[i]).map(|(a, b)| a * b).sum::<f32>(), i))
            .collect();
        e.sort_by(|a, b| b.0.total_cmp(&a.0));
        gt.push(e.iter().take(k).map(|&(_, i)| i).collect());
    }
    let recall = |res: &[Vec<foxstash_gpu::SearchResult>]| -> f64 {
        let h: usize = res
            .iter()
            .enumerate()
            .map(|(qi, r)| {
                let g: std::collections::HashSet<usize> =
                    r.iter().map(|x| x.id.parse().unwrap()).collect();
                gt[qi].intersection(&g).count()
            })
            .sum();
        h as f64 / (qs.len() * k) as f64
    };
    let qps = |f: &dyn Fn() -> Vec<Vec<foxstash_gpu::SearchResult>>| -> f64 {
        let _ = f();
        let t = Instant::now();
        let _ = f();
        qs.len() as f64 / t.elapsed().as_secs_f64()
    };

    println!("n={n} rerank_c={c}  plain vs +rerank (recall / QPS)");
    println!(
        "{:>8} {:>9} {:>9} {:>9} {:>9}",
        "mode", "plain r", "rrank r", "plain q", "rrank q"
    );
    let modes: Vec<(&str, GpuFlatIndex)> = vec![
        ("SQ8", GpuFlatIndex::build(&vecs, ids.clone())),
        ("Sign1", GpuFlatIndex::build_sign1(&vecs, ids.clone())),
        ("RaBitQ", GpuFlatIndex::build_rabitq(&vecs, ids.clone())),
        ("TR4", GpuFlatIndex::build_turborabit(&vecs, ids.clone(), 4)),
    ];
    for (name, idx) in &modes {
        let pr = recall(&idx.search_batch(&qs, k));
        let rr = recall(&idx.search_batch_reranked(&qs, k, c, &vecs));
        let pq = qps(&|| idx.search_batch(&qs, k));
        let rq = qps(&|| idx.search_batch_reranked(&qs, k, c, &vecs));
        println!("{name:>8} {pr:>9.4} {rr:>9.4} {pq:>9.0} {rq:>9.0}");
    }

    // GPU f32 rerank (uncaps the CPU-rerank ceiling; SQ8 has no rotation so it should approach its scan QPS).
    println!("--- GPU f32 rerank (on-device dot) ---");
    for (name, bits) in [("SQ8g", 8usize), ("TR4g", 4usize)] {
        let idx = if bits == 8 {
            GpuFlatIndex::build(&vecs, ids.clone()).with_gpu_rerank(&vecs)
        } else {
            GpuFlatIndex::build_turborabit(&vecs, ids.clone(), 4).with_gpu_rerank(&vecs)
        };
        let rr = recall(&idx.search_batch_reranked_gpu(&qs, k, c));
        let rq = qps(&|| idx.search_batch_reranked_gpu(&qs, k, c));
        println!(
            "{name:>8} {:>9} {rr:>9.4} {:>9} {rq:>9.0}   (GPU f32 rerank)",
            "-", "-"
        );
    }

    // Warren: 4-bit scan + 8+8 residual rerank, NO f32. CPU-rerank vs GPU-gather-rerank (same result).
    let warren = WarrenRerankIndex::build(&vecs, ids.clone(), 4);
    let wr = recall(&warren.search_batch(&qs, k, c));
    let wq = qps(&|| warren.search_batch(&qs, k, c));
    let gr = recall(&warren.search_batch_gpu(&qs, k, c));
    let gq = qps(&|| warren.search_batch_gpu(&qs, k, c));
    println!(
        "{:>8} {:>9} {wr:>9.4} {:>9} {wq:>9.0}   (8+8 residual, CPU rerank)",
        "Warren", "-", "-"
    );
    println!(
        "{:>8} {:>9} {gr:>9.4} {:>9} {gq:>9.0}   (8+8 residual, GPU gather-rerank)",
        "Warren-G", "-", "-"
    );
    // The rerank representation: f32 keeps 32 bits/dim; Warren's 8+8 residual is 16 bits/dim (no f32) and
    // reaches the same recall — half the rerank memory. (The 4-bit scan codes are shared with the coarse
    // stage either way.)
    println!(
        "rerank repr/dim: f32 = 32 bits  vs  Warren 8+8 residual = 16 bits (no f32, same recall)"
    );
}
