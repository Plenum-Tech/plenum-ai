//! Which shape of dot product the compiler turns into the fastest code on this CPU.
//! Variants are timed in turn, several rounds, so heat and other load affect them alike.
//!
//!   cargo run --release --example dotbench

use std::hint::black_box;
use std::time::Instant;

fn dot16(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 16];
    for (x, y) in a.chunks_exact(16).zip(b.chunks_exact(16)) {
        for i in 0..16 {
            acc[i] += x[i] * y[i];
        }
    }
    acc.iter().sum()
}

fn dot32(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 32];
    for (x, y) in a.chunks_exact(32).zip(b.chunks_exact(32)) {
        for i in 0..32 {
            acc[i] += x[i] * y[i];
        }
    }
    acc.iter().sum()
}

fn dot64(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 64];
    for (x, y) in a.chunks_exact(64).zip(b.chunks_exact(64)) {
        for i in 0..64 {
            acc[i] += x[i] * y[i];
        }
    }
    acc.iter().sum()
}

fn fma32(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 32];
    for (x, y) in a.chunks_exact(32).zip(b.chunks_exact(32)) {
        for i in 0..32 {
            acc[i] = x[i].mul_add(y[i], acc[i]);
        }
    }
    acc.iter().sum()
}

fn fma4x8(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [[0f32; 8]; 4];
    for (x, y) in a.chunks_exact(32).zip(b.chunks_exact(32)) {
        for c in 0..4 {
            let (xc, yc) = (&x[c * 8..c * 8 + 8], &y[c * 8..c * 8 + 8]);
            for i in 0..8 {
                acc[c][i] = xc[i].mul_add(yc[i], acc[c][i]);
            }
        }
    }
    acc.iter().flatten().sum()
}

fn main() {
    let n = 4096;
    let a: Vec<f32> = (0..n).map(|i| (i % 97) as f32 * 0.01 - 0.4).collect();
    let b: Vec<f32> = (0..n).map(|i| (i % 89) as f32 * 0.01 - 0.3).collect();
    let variants: [(&str, fn(&[f32], &[f32]) -> f32); 5] = [("dot16", dot16), ("dot32", dot32), ("dot64", dot64), ("fma32", fma32), ("fma4x8", fma4x8)];
    println!("fma detected: {}   avx2: {}   avx512f: {}", is_x86_feature_detected!("fma"), is_x86_feature_detected!("avx2"), is_x86_feature_detected!("avx512f"));
    println!("compiled with fma: {}   avx2: {}", cfg!(target_feature = "fma"), cfg!(target_feature = "avx2"));
    for round in 0..3 {
        for (name, f) in variants {
            let reps = 60_000;
            let t = Instant::now();
            let mut s = 0f32;
            for _ in 0..reps {
                s += f(black_box(&a), black_box(&b));
            }
            let secs = t.elapsed().as_secs_f64();
            println!("round {round}  {name:<7} {:6.2} G mult/s   (sum {s:.1})", (reps * n) as f64 / secs / 1e9);
        }
    }
}
