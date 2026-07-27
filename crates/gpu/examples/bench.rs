//! End-to-end GpuFlatIndex QPS + recall by mode (SQ8 vs 1-bit Sign) vs corpus size, on real 768-d.
//! Run: VECS_DIR=<dir> cargo run -p foxstash-gpu --example bench --release
use foxstash_gpu::GpuFlatIndex;
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
    let (dim, k) = (m[1], 10usize);
    let train = read_f32(&format!("{dir}/train.bin"));
    let qs: Vec<Vec<f32>> = read_f32(&format!("{dir}/test.bin"))[..200 * dim]
        .chunks_exact(dim)
        .map(|r| r.to_vec())
        .collect();
    let sizes: Vec<usize> = std::env::var("SIZES")
        .unwrap_or("20000,50000,100000,200000".into())
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    println!(
        "{:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "N", "SQ8 q", "SQ8 r", "Sgn1 q", "Sgn1 r", "RbQ q", "RbQ r", "TR4 q", "TR4 r"
    );
    for n in sizes {
        let vecs: Vec<Vec<f32>> = train[..n * dim]
            .chunks_exact(dim)
            .map(|r| r.to_vec())
            .collect();
        let ids: Vec<String> = (0..n).map(|i| i.to_string()).collect();
        // exact f32 ground truth
        let mut gt: Vec<std::collections::HashSet<usize>> = Vec::with_capacity(qs.len());
        for q in &qs {
            let mut e: Vec<(f32, usize)> = (0..n)
                .map(|i| (q.iter().zip(&vecs[i]).map(|(a, b)| a * b).sum::<f32>(), i))
                .collect();
            e.sort_by(|a, b| b.0.total_cmp(&a.0));
            gt.push(e.iter().take(k).map(|&(_, i)| i).collect());
        }
        let bench = |idx: &GpuFlatIndex| -> (f64, f64) {
            let _ = idx.search_batch(&qs, k);
            let t = Instant::now();
            let res = idx.search_batch(&qs, k);
            let qps = qs.len() as f64 / t.elapsed().as_secs_f64();
            let hits: usize = res
                .iter()
                .enumerate()
                .map(|(qi, r)| {
                    let got: std::collections::HashSet<usize> =
                        r.iter().map(|x| x.id.parse().unwrap()).collect();
                    gt[qi].intersection(&got).count()
                })
                .sum();
            (qps, hits as f64 / (qs.len() * k) as f64)
        };
        let (sq8_q, sq8_r) = bench(&GpuFlatIndex::build(&vecs, ids.clone()));
        let (s1_q, s1_r) = bench(&GpuFlatIndex::build_sign1(&vecs, ids.clone()));
        let (rq_q, rq_r) = bench(&GpuFlatIndex::build_rabitq(&vecs, ids.clone()));
        let (tr_q, tr_r) = bench(&GpuFlatIndex::build_turborabit(&vecs, ids, 4));
        println!("{n:>8} {sq8_q:>8.0} {sq8_r:>8.4} {s1_q:>8.0} {s1_r:>8.4} {rq_q:>8.0} {rq_r:>8.4} {tr_q:>8.0} {tr_r:>8.4}");
    }
}
