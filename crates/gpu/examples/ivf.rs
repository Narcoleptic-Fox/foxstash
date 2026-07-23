//! IVF (Phase 2): scan only the nprobe nearest cells instead of all n. Compares flat SQ8 (O(n)) to IVF
//! at a few nprobe (exact f32 candidates). Run: VECS_DIR=<dir> cargo run -p foxstash-gpu --example ivf --release
use foxstash_gpu::{GpuFlatIndex, GpuIvfIndex};
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
    let nlist = std::env::var("NLIST")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256usize);
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

    println!("n={n} nlist={nlist}  (recall / QPS)");
    let flat = GpuFlatIndex::build(&vecs, ids.clone());
    println!(
        "{:>16} {:>8.4} {:>9.0}",
        "flat SQ8 (O(n))",
        recall(&flat.search_batch(&qs, k)),
        qps(&|| flat.search_batch(&qs, k))
    );

    print!("building IVF (k-means)... ");
    let t = Instant::now();
    let ivf = GpuIvfIndex::build(&vecs, ids, nlist, 5);
    println!("{:.1}s", t.elapsed().as_secs_f64());
    for np in [1usize, 4, 16, 64] {
        let r = recall(&ivf.search_batch(&qs, k, np));
        let q = qps(&|| ivf.search_batch(&qs, k, np));
        println!("{:>16} {r:>8.4} {q:>9.0}", format!("IVF nprobe={np}"));
    }

    // Single-query (b=1) — IVF's real regime: flat has no batch reuse either, and IVF prunes to nprobe cells.
    println!("\n--- single-query (b=1) latency: flat scans all n, IVF prunes ---");
    let one = |f: &dyn Fn(&[Vec<f32>])| -> f64 {
        for q in qs.iter().take(5) {
            f(std::slice::from_ref(q));
        } // warm
        let t = Instant::now();
        for q in &qs {
            f(std::slice::from_ref(q));
        }
        qs.len() as f64 / t.elapsed().as_secs_f64()
    };
    println!(
        "{:>16} {:>9.0}",
        "flat b=1",
        one(&|q| {
            flat.search_batch(q, k);
        })
    );
    for np in [16usize, 64] {
        println!(
            "{:>16} {:>9.0}",
            format!("IVF b=1 np={np}"),
            one(&|q| {
                ivf.search_batch(q, k, np);
            })
        );
    }
}
