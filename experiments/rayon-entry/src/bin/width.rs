//! Fixed 18-worker pool, several active widths k: does `broadcast` with only k
//! active workers cost less than a full-width broadcast? Compared with
//! `install` + `scope` spawning k barrier-free tasks. Empty closures.
use std::hint::black_box;
use std::time::Instant;

fn spin(us: f64) {
    let t = Instant::now();
    while t.elapsed().as_secs_f64() * 1e6 < us {}
}
fn stats(mut v: Vec<f64>) -> String {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    format!("p10 {:7.2} p50 {:7.2} p90 {:7.2}", v[n / 10], v[n / 2], v[n * 9 / 10])
}
fn main() {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(18).build().unwrap();
    for _ in 0..1000 {
        pool.broadcast(|c| black_box(c.index()));
    }
    for gap in [0.0, 1000.0] {
        for k in [1usize, 2, 4, 8, 18] {
            let reps = 2000;
            let mut b = Vec::with_capacity(reps);
            let mut s = Vec::with_capacity(reps);
            for _ in 0..reps {
                spin(gap);
                let t = Instant::now();
                let r = pool.broadcast(|c| if c.index() < k { black_box(c.index()) } else { 0 });
                b.push(t.elapsed().as_secs_f64() * 1e6);
                assert_eq!(r.len(), 18); // every worker is dispatched
                spin(gap);
                let t = Instant::now();
                pool.install(|| rayon::scope(|sc| for i in 0..k { sc.spawn(move |_| { black_box(i); }); }));
                s.push(t.elapsed().as_secs_f64() * 1e6);
            }
            println!("gap {gap:5.0}us k {k:2}: broadcast(18 dispatched) {} | install+scope(k tasks) {}", stats(b), stats(s));
        }
    }
}
