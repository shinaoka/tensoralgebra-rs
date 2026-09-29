// Rayon pool entry latency: install, par_iter fan-out and broadcast after an
// idle gap on the calling thread. See README.md for the recorded results.
use rayon::prelude::*;
use std::hint::black_box;
use std::time::{Duration, Instant};

fn spin(d: Duration) { let t = Instant::now(); while t.elapsed() < d { std::hint::spin_loop(); } }

fn stats(mut v: Vec<f64>) -> String {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p = |q: f64| v[((v.len() - 1) as f64 * q) as usize];
    format!("p10 {:7.2}  p50 {:7.2}  p90 {:7.2}  p99 {:7.2} us", p(0.1), p(0.5), p(0.9), p(0.99))
}

// measure one op after a caller-side busy gap (caller never sleeps, workers may)
fn after_gap<F: FnMut()>(gap: Duration, reps: usize, mut op: F) -> Vec<f64> {
    (0..reps).map(|_| { spin(gap); let t = Instant::now(); op(); t.elapsed().as_secs_f64() * 1e6 }).collect()
}

fn main() {
    let ncpu = std::thread::available_parallelism().unwrap().get();
    let gaps_us = [0u64, 10, 100, 1_000, 10_000, 50_000];
    for &nt in &[1usize, 4, ncpu] {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(nt).build().unwrap();
        println!("\n=== pool {nt} threads ===");
        // warm up
        for _ in 0..1000 { pool.install(|| black_box(1)); }

        println!("-- install(empty) after caller gap");
        for &g in &gaps_us {
            let reps = if g >= 10_000 { 60 } else { 2000 };
            let v = after_gap(Duration::from_micros(g), reps, || { pool.install(|| black_box(1)); });
            println!("gap {:>6} us: {}", g, stats(v));
        }
        println!("-- install + par_iter over all workers (broadcast-like) after gap");
        for &g in &gaps_us {
            let reps = if g >= 10_000 { 60 } else { 2000 };
            let v = after_gap(Duration::from_micros(g), reps, || {
                pool.install(|| { (0..nt * 4).into_par_iter().for_each(|i| { black_box(i); }); });
            });
            println!("gap {:>6} us: {}", g, stats(v));
        }
        println!("-- pool.broadcast(empty) after gap");
        for &g in &gaps_us {
            let reps = if g >= 10_000 { 60 } else { 2000 };
            let v = after_gap(Duration::from_micros(g), reps, || { pool.broadcast(|c| black_box(c.index())); });
            println!("gap {:>6} us: {}", g, stats(v));
        }
        println!("-- 1000 back-to-back installs inside ONE outer install (re-entry)");
        let t = Instant::now();
        pool.install(|| for _ in 0..1000 { pool.install(|| black_box(1)); });
        println!("per inner install: {:.3} us", t.elapsed().as_secs_f64() * 1e3);
    }
    let t = Instant::now();
    for _ in 0..100_000 { black_box((|| black_box(1))()); }
    println!("\ndirect call on caller: {:.4} us", t.elapsed().as_secs_f64() * 10.0);
}
