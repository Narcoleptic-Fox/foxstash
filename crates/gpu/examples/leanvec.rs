//! LeanVec: reduced-dim coarse scan + full-dim rerank. Compares plain SQ8 flat (full dim) to LeanVec at a
//! few rdim/C. Run: VECS_DIR=<dir> cargo run -p foxstash-gpu --example leanvec --release
use foxstash_gpu::{GpuFlatIndex, LeanVecIndex};
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

    println!("n={n} dim={dim}  (recall / QPS)");
    let flat = GpuFlatIndex::build(&vecs, ids.clone());
    println!(
        "{:>22} {:>8.4} {:>9.0}",
        "flat SQ8 (full dim)",
        recall(&flat.search_batch(&qs, k)),
        qps(&|| flat.search_batch(&qs, k))
    );

    for rdim in [dim / 2, dim / 3] {
        let lv = LeanVecIndex::build(&vecs, ids.clone(), rdim);
        for c in [64usize, 128] {
            let r = recall(&lv.search_batch(&qs, k, c));
            let q = qps(&|| lv.search_batch(&qs, k, c));
            println!(
                "{:>22} {r:>8.4} {q:>9.0}",
                format!("LeanVec rdim={rdim} C={c}")
            );
        }
    }
}
