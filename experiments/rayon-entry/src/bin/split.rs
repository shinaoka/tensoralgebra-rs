// Split the install round trip: is it the worker wake-up or the caller wake-up?
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::*};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

fn spin(d: Duration) { let t = Instant::now(); while t.elapsed() < d { std::hint::spin_loop(); } }
fn p50(mut v: Vec<f64>) -> String {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    format!("p10 {:6.2} p50 {:6.2} p90 {:6.2} us", v[v.len() / 10], v[v.len() / 2], v[v.len() * 9 / 10])
}

fn main() {
    let gaps = [0u64, 100, 1000, 10_000];
    for nt in [1usize, 4] {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(nt).build().unwrap();
        for _ in 0..1000 { pool.install(|| black_box(1)); }
        println!("\n=== pool {nt} ===");
        for &g in &gaps {
            let reps = if g >= 10_000 { 100 } else { 2000 };
            let a: Vec<f64> = (0..reps).map(|_| { spin(Duration::from_micros(g));
                let t = Instant::now(); pool.install(|| black_box(1)); t.elapsed().as_secs_f64()*1e6 }).collect();
            // spawn, caller busy-waits on an atomic: no caller-side condvar
            let b: Vec<f64> = (0..reps).map(|_| { spin(Duration::from_micros(g));
                let done = Arc::new(AtomicBool::new(false)); let d2 = done.clone();
                let t = Instant::now(); pool.spawn(move || d2.store(true, Release));
                while !done.load(Acquire) { std::hint::spin_loop(); }
                t.elapsed().as_secs_f64()*1e6 }).collect();
            println!("gap {:>5}us  install        {}", g, p50(a));
            println!("gap {:>5}us  spawn+spinwait {}", g, p50(b));
        }
    }
    // raw baselines between two OS threads
    println!("\n=== raw 2-thread round trip ===");
    let flag = Arc::new(AtomicU64::new(0)); let f2 = flag.clone();
    let h = std::thread::spawn(move || { let mut k = 0; loop { while f2.load(Acquire) == k * 2 { std::hint::spin_loop(); }
        if f2.load(Acquire) == u64::MAX { break } f2.store(k*2+2, Release); k += 1; } });
    let mut v = vec![];
    for k in 0..20000u64 { let t = Instant::now(); flag.store(k*2+1, Release); while flag.load(Acquire) != k*2+2 { std::hint::spin_loop(); } v.push(t.elapsed().as_secs_f64()*1e6); }
    flag.store(u64::MAX, Release); h.join().unwrap();
    println!("both spin (atomics)     {}", p50(v));

    let pair = Arc::new((Mutex::new(0u64), Condvar::new(), Condvar::new())); let p2 = pair.clone();
    let h = std::thread::spawn(move || { let (m, req, resp) = &*p2; let mut g = m.lock().unwrap(); let mut k = 0u64;
        loop { while *g == k*2 { g = req.wait(g).unwrap(); } if *g == u64::MAX { break } *g = k*2+2; resp.notify_one(); k += 1; } });
    let (m, req, resp) = &*pair; let mut v = vec![];
    for k in 0..20000u64 { let t = Instant::now(); let mut g = m.lock().unwrap(); *g = k*2+1; req.notify_one();
        while *g != k*2+2 { g = resp.wait(g).unwrap(); } drop(g); v.push(t.elapsed().as_secs_f64()*1e6); }
    *m.lock().unwrap() = u64::MAX; req.notify_one(); h.join().unwrap();
    println!("both sleep (condvar)    {}", p50(v));
}
