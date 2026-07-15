//! Microbenchmark the distance kernel — the innermost loop of every search.
//!
//! Run: `cargo run --release -p foxstash-benches --example distance_micro`
//!
//! `l2_distance_simd` calls `pulp::Arch::new()` (runtime CPU feature detection) and then
//! `dispatch` on *every* call. At ef_search=500 a single SIFT query touches thousands of
//! nodes, so this runs millions of times per benchmark. It also takes a `sqrt` that
//! ranking does not need — squared L2 is monotonic in L2.

use std::time::Instant;

const DIM: usize = 128;
const N: usize = 2_000_000;

#[inline(always)]
fn l2_sq_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

fn main() {
    // Two vectors that stay in L1 — we're measuring the kernel, not memory.
    let a: Vec<f32> = (0..DIM).map(|i| (i % 37) as f32).collect();
    let b: Vec<f32> = (0..DIM).map(|i| (i % 53) as f32).collect();

    println!("{DIM}d vectors, {N} distance calls each\n");
    println!("{:<44} {:>10} {:>12}", "Kernel", "ns/call", "M calls/s");
    println!("{:-<68}", "");

    let bench = |label: &str, f: &dyn Fn() -> f32| {
        // warm up
        for _ in 0..10_000 {
            std::hint::black_box(f());
        }
        let t = Instant::now();
        for _ in 0..N {
            std::hint::black_box(f());
        }
        let e = t.elapsed().as_secs_f64();
        println!(
            "{:<44} {:>10.1} {:>12.1}",
            label,
            e / N as f64 * 1e9,
            N as f64 / e / 1e6
        );
    };

    bench("l2_distance_simd (Arch::new per call)", &|| {
        foxstash_core::vector::simd::l2_distance_simd(&a, &b)
    });

    bench("cosine_similarity_simd (Arch::new per call)", &|| {
        foxstash_core::vector::simd::cosine_similarity_simd(&a, &b)
    });

    bench("plain scalar squared-L2 (autovectorised)", &|| {
        l2_sq_scalar(&a, &b)
    });

    // Hoist the dispatch: build the Arch once, reuse it. This is what the index should do.
    let arch = pulp::Arch::new();
    bench("pulp, Arch hoisted out of the loop", &|| {
        arch.dispatch(
            #[inline(always)]
            || {
                let mut sum = 0.0f32;
                for (x, y) in a.iter().zip(b.iter()) {
                    let d = *x - *y;
                    sum += d * d;
                }
                sum
            },
        )
    });
}
