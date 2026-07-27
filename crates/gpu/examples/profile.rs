//! Where does the time go? Per-stage timing (scan / top-k / readback) for b=1 (single-query RAG) and
//! b=64 (batch), so an optimization targets the actual wall. Run: VECS_DIR=<dir> cargo run -p foxstash-gpu
//! --example profile --release
use foxstash_gpu::GpuFlatIndex;

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
    let train = read_f32(&format!("{dir}/train.bin"));
    let qs: Vec<Vec<f32>> = read_f32(&format!("{dir}/test.bin"))[..64 * dim]
        .chunks_exact(dim)
        .map(|r| r.to_vec())
        .collect();
    let n = std::env::var("N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100000usize);
    let vecs: Vec<Vec<f32>> = train[..n * dim]
        .chunks_exact(dim)
        .map(|r| r.to_vec())
        .collect();
    let ids: Vec<String> = (0..n).map(|i| i.to_string()).collect();

    let reps = 200;
    println!("n={n} reps={reps}  (per-chunk stage time, µs; % of chunk)");
    println!(
        "{:>8} {:>4} {:>10} {:>10} {:>10}   split",
        "mode", "b", "scan µs", "topk µs", "read µs"
    );
    for (name, idx) in [
        ("SQ8", GpuFlatIndex::build(&vecs, ids.clone())),
        ("TR4", GpuFlatIndex::build_turborabit(&vecs, ids.clone(), 4)),
    ] {
        for b in [1usize, 64usize] {
            let q = &qs[..b];
            let _ = idx.profile_pipeline(q, k, 5); // warm up
            let (s, t, r) = idx.profile_pipeline(q, k, reps);
            let (s, t, r) = (s / reps as f64, t / reps as f64, r / reps as f64);
            let tot = s + t + r;
            println!(
                "{name:>8} {b:>4} {:>10.1} {:>10.1} {:>10.1}   scan {:.0}% / topk {:.0}% / read {:.0}%",
                s * 1e6, t * 1e6, r * 1e6, s / tot * 100.0, t / tot * 100.0, r / tot * 100.0
            );
        }
    }

    // Rerank pipeline stage breakdown (SQ8 coarse + f32 rerank), b=64.
    let c = std::env::var("C")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64usize);
    println!(
        "\nrerank pipeline (SQ8 coarse, C={c}, b=64): coarse(scan+topC) / rerank / read+cputopk"
    );
    let idx = GpuFlatIndex::build(&vecs, ids.clone()).with_gpu_rerank(&vecs);
    let q = &qs[..64.min(qs.len())];
    let _ = idx.profile_rerank(q, k, c, 5);
    let (a, b, d) = idx.profile_rerank(q, k, c, reps);
    let (a, b, d) = (a / reps as f64, b / reps as f64, d / reps as f64);
    let tot = a + b + d;
    println!(
        "  coarse {:.1}µs ({:.0}%) / rerank {:.1}µs ({:.0}%) / read {:.1}µs ({:.0}%)",
        a * 1e6,
        a / tot * 100.0,
        b * 1e6,
        b / tot * 100.0,
        d * 1e6,
        d / tot * 100.0
    );
}
