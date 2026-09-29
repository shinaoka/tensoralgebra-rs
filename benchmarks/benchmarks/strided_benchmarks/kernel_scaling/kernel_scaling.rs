//! Kernel scaling coverage for strided-rs#269.
//!
//! Elementwise and ternary select/clamp (erased, typed, raw), reductions (erased plan, typed reduce,
//! raw multi-accumulator) and structural copy plans (erased plan, typed plan,
//! raw memcpy/loop) at an enforced thread count. Setup, allocation, descriptor
//! construction and correctness checks are outside timed regions. The typed
//! `reduce_axis` API allocates its result, so that allocation is inside its
//! timed region by construction of the API.
//!
//! Usage: `kernel_scaling --threads N [--time]`. Without `--time` every case is
//! checked once and nothing is timed. With `--time` the CSV
//! `case,variant,threads,median_ns,samples` is printed on stdout; diagnostics
//! go to stderr.
//!
//! Environment: `BENCH_RUNS` (timed samples, default 11), `BENCH_WARMUP`
//! (default 2), `KERNEL_SCALING_SHRINK` (divide every 2D extent by this
//! factor and every 1D length by its square, default 1, for smoke runs),
//! `BENCH_FILTER` (substring filter on case names).
use num_complex::Complex64;
use std::{
    env,
    hint::black_box,
    mem::MaybeUninit,
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};
use strided_kernel::{
    erased_clamp_into_uninit, erased_map_into, erased_map_into_uninit, erased_select_into_uninit,
    erased_zip_into, erased_zip_into_uninit, map_into, reduce, reduce_axis, zip_map2_into,
    zip_map3_into, ConcatenatePlan, DynamicSlicePlan, ErasedConcatenatePlan,
    ErasedDynamicSlicePlan, ErasedMapOp, ErasedPadPlan, ErasedRawStridedMut, ErasedRawStridedPtr,
    ErasedRawStridedRef, ErasedRawStridedUninitMut, ErasedReducePlan, ErasedReversePlan,
    ErasedSlicePlan, ErasedZipOp, ExecContext, KernelDType, KernelStorageElement, PadPlan,
    RawStridedMut, RawStridedRef, ReduceOp, ReversePlan, SlicePlan, StridedView, StridedViewMut,
};

// ---------------------------------------------------------------------------
// Configuration, thread enforcement and timing
// ---------------------------------------------------------------------------

struct Cfg {
    threads: usize,
    exec: ExecContext,
    timing: bool,
    runs: usize,
    warmup: usize,
    shrink: usize,
    filter: Option<String>,
}

impl Cfg {
    fn enabled(&self, case: &str) -> bool {
        self.filter.as_deref().map_or(true, |f| case.contains(f))
    }

    /// Run `f` once (check mode) or `warmup` + `runs` times and print the median.
    fn measure(&self, case: &str, variant: &str, mut f: impl FnMut()) {
        if !self.timing {
            f();
            return;
        }
        for _ in 0..self.warmup {
            f();
        }
        let mut ns = Vec::with_capacity(self.runs);
        for _ in 0..self.runs {
            let start = Instant::now();
            f();
            ns.push(start.elapsed().as_nanos());
        }
        ns.sort_unstable();
        let n = ns.len();
        let median = if n % 2 == 1 {
            ns[n / 2]
        } else {
            (ns[n / 2 - 1] + ns[n / 2]) / 2
        };
        println!("{case},{variant},{},{median},{n}", self.threads);
    }

    fn ok(&self, case: &str, variant: &str) {
        eprintln!("CHECK {case} {variant} threads={} ok", self.threads);
    }

    fn d(&self, extent: usize) -> usize {
        (extent / self.shrink).max(1)
    }

    fn n1(&self, len: usize) -> usize {
        (len / (self.shrink * self.shrink)).max(1)
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .map(|v| v.parse().unwrap_or_else(|_| panic!("{name} must be an integer")))
        .unwrap_or(default)
}

static SEEN_WORKER: [AtomicBool; 256] = [const { AtomicBool::new(false) }; 256];

/// Number of OS threads in this process, when the platform exposes it.
fn os_thread_count() -> Option<usize> {
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        return status
            .lines()
            .find_map(|l| l.strip_prefix("Threads:"))
            .and_then(|v| v.trim().parse().ok());
    }
    // macOS: `ps -M` prints one header line plus one line per thread.
    let out = std::process::Command::new("ps")
        .args(["-M", "-p", &std::process::id().to_string()])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let lines = String::from_utf8_lossy(&out.stdout).lines().count();
    lines.checked_sub(1)
}

/// Verify inside the pool that the effective thread budget is `cfg.threads`.
fn verify_threads(cfg: &Cfg) {
    let pool = rayon::current_num_threads();
    assert_eq!(pool, cfg.threads, "rayon pool size {pool} != requested {}", cfg.threads);
    assert!(
        rayon::current_thread_index().is_some(),
        "benchmark body must run on a pool worker"
    );
    // Probe: a large typed map records which pool workers executed elements.
    let n = 1 << 22;
    let src = vec![1.0f64; n];
    let mut dst = vec![0.0f64; n];
    let sv: StridedView<f64> = StridedView::new(&src, &[n], &[1], 0).unwrap();
    let mut dv = StridedViewMut::new(&mut dst, &[n], &[1], 0).unwrap();
    for flag in &SEEN_WORKER {
        flag.store(false, Ordering::Relaxed);
    }
    cfg.exec.run(|| {
        map_into(&mut dv, &sv, |x| {
            let index = rayon::current_thread_index().unwrap_or(255).min(255);
            SEEN_WORKER[index].store(true, Ordering::Relaxed);
            x
        })
        .unwrap()
    });
    let observed = SEEN_WORKER
        .iter()
        .filter(|f| f.load(Ordering::Relaxed))
        .count();
    assert!(!SEEN_WORKER[255].load(Ordering::Relaxed), "work ran outside the pool");
    assert!(
        observed <= cfg.threads,
        "probe observed {observed} workers > requested {}",
        cfg.threads
    );
    let os = os_thread_count();
    eprintln!(
        "# threads: requested={} rayon_pool={pool} exec_ctx={:?} probe_workers={observed} os_threads={}",
        cfg.threads,
        cfg.exec,
        os.map_or("unknown".to_string(), |v| v.to_string())
    );
    if let Some(os) = os {
        // Main thread (blocked in the pool entry) plus the pool workers.
        assert!(
            os <= cfg.threads + 1,
            "process has {os} OS threads, more than main + {} workers",
            cfg.threads
        );
    }
    if cfg.threads > 1 && observed == 1 {
        eprintln!("# warning: typed map_into probe stayed on one worker at {} threads", cfg.threads);
    }
}

// ---------------------------------------------------------------------------
// Raw-pointer helpers
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}
impl<T> SendPtr<T> {
    #[inline(always)]
    fn get(self) -> *mut T {
        self.0
    }
}

#[derive(Clone, Copy)]
struct SendConst<T>(*const T);
unsafe impl<T> Send for SendConst<T> {}
unsafe impl<T> Sync for SendConst<T> {}
impl<T> SendConst<T> {
    #[inline(always)]
    fn get(self) -> *const T {
        self.0
    }
}

/// Split `0..n` into `threads` chunks (multiples of `align`) and run `f(start, end)`.
fn par_ranges(threads: usize, n: usize, align: usize, f: impl Fn(usize, usize) + Sync) {
    if threads == 1 || n <= align {
        f(0, n);
        return;
    }
    let per = n.div_ceil(threads).div_ceil(align) * align;
    rayon::scope(|s| {
        let f = &f;
        let mut start = 0;
        while start < n {
            let end = (start + per).min(n);
            s.spawn(move |_| f(start, end));
            start = end;
        }
    });
}

/// Same as `par_ranges`, returning per-chunk results in chunk order.
fn par_map_ranges<R: Send + Default + Clone>(
    threads: usize,
    n: usize,
    align: usize,
    f: impl Fn(usize, usize) -> R + Sync,
) -> Vec<R> {
    if threads == 1 || n <= align {
        return vec![f(0, n)];
    }
    let per = n.div_ceil(threads).div_ceil(align) * align;
    let chunks = n.div_ceil(per);
    let mut out = vec![R::default(); chunks];
    rayon::scope(|s| {
        let f = &f;
        for (k, slot) in out.iter_mut().enumerate() {
            let start = k * per;
            let end = (start + per).min(n);
            s.spawn(move |_| *slot = f(start, end));
        }
    });
    out
}

/// Parallel memcpy of `len` elements.
fn par_copy<T: Copy>(threads: usize, dst: *mut T, src: *const T, len: usize) {
    let (d, s) = (SendPtr(dst), SendConst(src));
    par_ranges(threads, len, 4096, |a, b| unsafe {
        std::ptr::copy_nonoverlapping(s.get().add(a), d.get().add(a), b - a);
    });
}

// ---------------------------------------------------------------------------
// Element types
// ---------------------------------------------------------------------------

trait Elem: KernelStorageElement + Copy + Send + Sync + std::fmt::Debug {
    const NAME: &'static str;
    fn gen(i: usize, which: u32) -> Self;
    fn sentinel() -> Self;
    fn close(a: Self, b: Self) -> bool;
}

impl Elem for f64 {
    const NAME: &'static str = "f64";
    fn gen(i: usize, which: u32) -> Self {
        match which {
            // lhs: mixed sign
            0 => ((i % 97) as f64 - 48.0) / 32.0,
            // rhs: strictly positive, safe divisor
            _ => 0.5 + (i % 89) as f64 / 64.0,
        }
    }
    fn sentinel() -> Self {
        f64::NAN
    }
    fn close(a: Self, b: Self) -> bool {
        a == b || (a - b).abs() <= 1e-12 * a.abs().max(b.abs())
    }
}

impl Elem for Complex64 {
    const NAME: &'static str = "c64";
    fn gen(i: usize, which: u32) -> Self {
        Complex64::new(f64::gen(i, which), f64::gen(i + 13, 1 - which.min(1)))
    }
    fn sentinel() -> Self {
        Complex64::new(f64::NAN, f64::NAN)
    }
    fn close(a: Self, b: Self) -> bool {
        a == b || (a - b).norm() <= 1e-12 * a.norm().max(b.norm())
    }
}

fn nan_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a >= b {
        a
    } else {
        b
    }
}

fn nan_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a <= b {
        a
    } else {
        b
    }
}

fn check_all<T: Copy + std::fmt::Debug>(
    case: &str,
    variant: &str,
    out: &[T],
    close: impl Fn(T, T) -> bool,
    expected: impl Fn(usize) -> T,
) {
    for (k, &v) in out.iter().enumerate() {
        let e = expected(k);
        assert!(close(v, e), "{case} {variant}: index {k}: got {v:?}, expected {e:?}");
    }
}

/// Read a fully overwritten uninitialized buffer.
fn assume_init<T>(buf: &[MaybeUninit<T>]) -> &[T] {
    // SAFETY: callers only use this after a kernel returned Ok and fully
    // overwrote every reachable slot of a compact destination.
    unsafe { std::slice::from_raw_parts(buf.as_ptr().cast::<T>(), buf.len()) }
}

// ---------------------------------------------------------------------------
// Elementwise
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Layout {
    /// 1D compact operands.
    Contig,
    /// 2D column-major destination and rhs, row-major (transposed) lhs.
    Trans,
}

struct EwShape {
    dims: Vec<usize>,
    lhs_strides: Vec<isize>,
    col_strides: Vec<isize>,
    len: usize,
    rows: usize,
    cols: usize,
}

fn ew_shape(cfg: &Cfg, layout: Layout) -> EwShape {
    match layout {
        Layout::Contig => {
            let n = cfg.n1(33_554_432);
            EwShape {
                dims: vec![n],
                lhs_strides: vec![1],
                col_strides: vec![1],
                len: n,
                rows: n,
                cols: 1,
            }
        }
        Layout::Trans => {
            let (r, c) = (cfg.d(8192), cfg.d(4096));
            EwShape {
                dims: vec![r, c],
                lhs_strides: vec![c as isize, 1],
                col_strides: vec![1, r as isize],
                len: r * c,
                rows: r,
                cols: c,
            }
        }
    }
}

/// Offset in the lhs buffer of destination linear index `k` (column-major).
#[inline(always)]
fn lhs_offset(s: &EwShape, k: usize) -> usize {
    if s.cols == 1 {
        k
    } else {
        let (i, j) = (k % s.rows, k / s.rows);
        j + i * s.cols
    }
}

const TILE: usize = 64;

fn layout_label(layout: Layout) -> &'static str {
    match layout {
        Layout::Contig => "contig",
        Layout::Trans => "trans",
    }
}

fn bench_binary<T: Elem>(
    cfg: &Cfg,
    opname: &str,
    layout: Layout,
    op: ErasedZipOp,
    f: impl Fn(T, T) -> T + Sync + Send + Copy,
) {
    let case = format!("ew_{opname}_{}_{}", T::NAME, layout_label(layout));
    if !cfg.enabled(&case) {
        return;
    }
    let s = ew_shape(cfg, layout);
    let a: Vec<T> = (0..s.len).map(|i| T::gen(i, 0)).collect();
    let b: Vec<T> = (0..s.len).map(|i| T::gen(i, 1)).collect();
    let mut out = vec![T::sentinel(); s.len];
    let expected = |k: usize| f(a[lhs_offset(&s, k)], b[k]);

    // raw
    {
        let (pa, pb, pd) = (SendConst(a.as_ptr()), SendConst(b.as_ptr()), SendPtr(out.as_mut_ptr()));
        let (rows, cols, len) = (s.rows, s.cols, s.len);
        let threads = cfg.threads;
        cfg.measure(&case, "raw", || {
            if cols == 1 {
                par_ranges(threads, len, 4096, |lo, hi| unsafe {
                    let d = std::slice::from_raw_parts_mut(pd.get().add(lo), hi - lo);
                    let x = std::slice::from_raw_parts(pa.get().add(lo), hi - lo);
                    let y = std::slice::from_raw_parts(pb.get().add(lo), hi - lo);
                    for ((d, &x), &y) in d.iter_mut().zip(x).zip(y) {
                        *d = f(x, y);
                    }
                });
            } else {
                // Tiled transpose-read loop, parallel over column tiles.
                par_ranges(threads, cols, TILE, |c0, c1| unsafe {
                    let (a, b, d) = (pa.get(), pb.get(), pd.get());
                    for jb in (c0..c1).step_by(TILE) {
                        let je = (jb + TILE).min(c1);
                        for ib in (0..rows).step_by(TILE) {
                            let ie = (ib + TILE).min(rows);
                            for j in jb..je {
                                for i in ib..ie {
                                    *d.add(i + j * rows) = f(*a.add(j + i * cols), *b.add(i + j * rows));
                                }
                            }
                        }
                    }
                });
            }
            black_box(pd.get());
        });
    }
    check_all(&case, "raw", &out, T::close, expected);
    cfg.ok(&case, "raw");

    // typed
    out.fill(T::sentinel());
    {
        let av: StridedView<T> = StridedView::new(&a, &s.dims, &s.lhs_strides, 0).unwrap();
        let bv: StridedView<T> = StridedView::new(&b, &s.dims, &s.col_strides, 0).unwrap();
        let mut dv = StridedViewMut::new(&mut out, &s.dims, &s.col_strides, 0).unwrap();
        cfg.measure(&case, "typed", || {
            cfg.exec
                .run(|| zip_map2_into(&mut dv, black_box(&av), black_box(&bv), f).unwrap());
            black_box(&mut dv);
        });
    }
    check_all(&case, "typed", &out, T::close, expected);
    cfg.ok(&case, "typed");

    // erased (initialized destination)
    out.fill(T::sentinel());
    {
        let ar = ErasedRawStridedRef::from_slice(&a, &s.dims, &s.lhs_strides, 0).unwrap();
        let br = ErasedRawStridedRef::from_slice(&b, &s.dims, &s.col_strides, 0).unwrap();
        let (ap, bp) = (ErasedRawStridedPtr::from_ref(&ar), ErasedRawStridedPtr::from_ref(&br));
        let mut dest = ErasedRawStridedMut::from_slice_mut(&mut out, &s.dims, &s.col_strides, 0).unwrap();
        cfg.measure(&case, "erased", || {
            erased_zip_into(T::DTYPE, op, &cfg.exec, &mut dest, black_box(&ap), black_box(&bp)).unwrap();
            black_box(&mut dest);
        });
    }
    check_all(&case, "erased", &out, T::close, expected);
    cfg.ok(&case, "erased");
    drop(out);

    // erased (uninitialized destination)
    let mut out_u = vec![MaybeUninit::<T>::uninit(); s.len];
    {
        let ar = ErasedRawStridedRef::from_slice(&a, &s.dims, &s.lhs_strides, 0).unwrap();
        let br = ErasedRawStridedRef::from_slice(&b, &s.dims, &s.col_strides, 0).unwrap();
        let (ap, bp) = (ErasedRawStridedPtr::from_ref(&ar), ErasedRawStridedPtr::from_ref(&br));
        let mut dest =
            ErasedRawStridedUninitMut::from_uninit_slice(&mut out_u, &s.dims, &s.col_strides, 0).unwrap();
        cfg.measure(&case, "erased_uninit", || {
            erased_zip_into_uninit(T::DTYPE, op, &cfg.exec, &mut dest, black_box(&ap), black_box(&bp))
                .unwrap();
            black_box(&mut dest);
        });
    }
    check_all(&case, "erased_uninit", assume_init(&out_u), T::close, expected);
    cfg.ok(&case, "erased_uninit");
}

fn bench_unary<T: Elem, U: Elem>(
    cfg: &Cfg,
    opname: &str,
    layout: Layout,
    op: ErasedMapOp,
    f: impl Fn(T) -> U + Sync + Send + Copy,
) {
    let case = format!("ew_{opname}_{}_{}", T::NAME, layout_label(layout));
    if !cfg.enabled(&case) {
        return;
    }
    let s = ew_shape(cfg, layout);
    let a: Vec<T> = (0..s.len).map(|i| T::gen(i, 0)).collect();
    let mut out = vec![U::sentinel(); s.len];
    let expected = |k: usize| f(a[lhs_offset(&s, k)]);

    {
        let (pa, pd) = (SendConst(a.as_ptr()), SendPtr(out.as_mut_ptr()));
        let (rows, cols, len) = (s.rows, s.cols, s.len);
        let threads = cfg.threads;
        cfg.measure(&case, "raw", || {
            if cols == 1 {
                par_ranges(threads, len, 4096, |lo, hi| unsafe {
                    let d = std::slice::from_raw_parts_mut(pd.get().add(lo), hi - lo);
                    let x = std::slice::from_raw_parts(pa.get().add(lo), hi - lo);
                    for (d, &x) in d.iter_mut().zip(x) {
                        *d = f(x);
                    }
                });
            } else {
                par_ranges(threads, cols, TILE, |c0, c1| unsafe {
                    let (a, d) = (pa.get(), pd.get());
                    for jb in (c0..c1).step_by(TILE) {
                        let je = (jb + TILE).min(c1);
                        for ib in (0..rows).step_by(TILE) {
                            let ie = (ib + TILE).min(rows);
                            for j in jb..je {
                                for i in ib..ie {
                                    *d.add(i + j * rows) = f(*a.add(j + i * cols));
                                }
                            }
                        }
                    }
                });
            }
            black_box(pd.get());
        });
    }
    check_all(&case, "raw", &out, U::close, expected);
    cfg.ok(&case, "raw");

    out.fill(U::sentinel());
    {
        let av: StridedView<T> = StridedView::new(&a, &s.dims, &s.lhs_strides, 0).unwrap();
        let mut dv = StridedViewMut::new(&mut out, &s.dims, &s.col_strides, 0).unwrap();
        cfg.measure(&case, "typed", || {
            cfg.exec.run(|| map_into(&mut dv, black_box(&av), f).unwrap());
            black_box(&mut dv);
        });
    }
    check_all(&case, "typed", &out, U::close, expected);
    cfg.ok(&case, "typed");

    out.fill(U::sentinel());
    {
        let ar = ErasedRawStridedRef::from_slice(&a, &s.dims, &s.lhs_strides, 0).unwrap();
        let ap = ErasedRawStridedPtr::from_ref(&ar);
        let mut dest = ErasedRawStridedMut::from_slice_mut(&mut out, &s.dims, &s.col_strides, 0).unwrap();
        cfg.measure(&case, "erased", || {
            erased_map_into(T::DTYPE, op, &cfg.exec, &mut dest, black_box(&ap)).unwrap();
            black_box(&mut dest);
        });
    }
    check_all(&case, "erased", &out, U::close, expected);
    cfg.ok(&case, "erased");
    drop(out);

    let mut out_u = vec![MaybeUninit::<U>::uninit(); s.len];
    {
        let ar = ErasedRawStridedRef::from_slice(&a, &s.dims, &s.lhs_strides, 0).unwrap();
        let ap = ErasedRawStridedPtr::from_ref(&ar);
        let mut dest =
            ErasedRawStridedUninitMut::from_uninit_slice(&mut out_u, &s.dims, &s.col_strides, 0).unwrap();
        cfg.measure(&case, "erased_uninit", || {
            erased_map_into_uninit(T::DTYPE, op, &cfg.exec, &mut dest, black_box(&ap)).unwrap();
            black_box(&mut dest);
        });
    }
    check_all(&case, "erased_uninit", assume_init(&out_u), U::close, expected);
    cfg.ok(&case, "erased_uninit");
}

fn elementwise(cfg: &Cfg) {
    for layout in [Layout::Contig, Layout::Trans] {
        bench_binary::<f64>(cfg, "add", layout, ErasedZipOp::Add, |a, b| a + b);
        bench_binary::<f64>(cfg, "sub", layout, ErasedZipOp::Subtract, |a, b| a - b);
        bench_binary::<f64>(cfg, "mul", layout, ErasedZipOp::Multiply, |a, b| a * b);
        bench_binary::<f64>(cfg, "div", layout, ErasedZipOp::Divide, |a, b| a / b);
        bench_binary::<f64>(cfg, "max", layout, ErasedZipOp::Maximum, nan_max);
        bench_binary::<f64>(cfg, "min", layout, ErasedZipOp::Minimum, nan_min);
        bench_unary::<f64, f64>(cfg, "neg", layout, ErasedMapOp::Negate, |a| -a);
        bench_unary::<f64, f64>(cfg, "abs", layout, ErasedMapOp::Abs, |a| a.abs());
    }
    let c = Layout::Contig;
    bench_binary::<Complex64>(cfg, "add", c, ErasedZipOp::Add, |a, b| a + b);
    bench_binary::<Complex64>(cfg, "mul", c, ErasedZipOp::Multiply, |a, b| a * b);
    bench_binary::<Complex64>(cfg, "div", c, ErasedZipOp::Divide, |a, b| a / b);
    bench_unary::<Complex64, Complex64>(cfg, "neg", c, ErasedMapOp::Negate, |a| -a);
    bench_unary::<Complex64, Complex64>(cfg, "conj", c, ErasedMapOp::Conj, |a| a.conj());
    bench_unary::<Complex64, f64>(cfg, "abs", c, ErasedMapOp::Abs, |a| a.norm());
    bench_unary::<Complex64, Complex64>(cfg, "conj", Layout::Trans, ErasedMapOp::Conj, |a| a.conj());
}

// ---------------------------------------------------------------------------
// Ternary elementwise: select and clamp
// ---------------------------------------------------------------------------

/// Irregular predicate pattern shared with the Julia script (0-based `i`).
fn gen_pred(i: usize) -> bool {
    (i * 7919) % 13 < 6
}

/// Clamp operand: the lhs generator with a sparse NaN to exercise propagation.
fn gen_clamp_x(i: usize) -> f64 {
    if i % 1021 == 0 {
        f64::NAN
    } else {
        f64::gen(i, 0)
    }
}

/// Scalar clamp with the strided semantics: any NaN operand gives NaN, ties
/// return the bound, and `lo > hi` returns `hi`.
#[inline(always)]
fn nan_clamp(x: f64, lo: f64, hi: f64) -> f64 {
    let raised = if lo >= x { lo } else { x };
    let lowered = if hi <= raised { hi } else { raised };
    if x.is_nan() | lo.is_nan() | hi.is_nan() {
        f64::NAN
    } else {
        lowered
    }
}

/// Exact comparison that treats every NaN as equal.
fn same_f64(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

/// `out[k] = f(p[lhs(k)], b[k], c[k])` with f64 output. The first operand uses
/// the lhs layout (row major for `Trans`), the others and the destination are
/// column major. There is no initialized erased select or clamp entry, so the
/// erased rows use the uninitialized destination entries only.
#[allow(clippy::too_many_arguments)]
fn bench_ternary<P: KernelStorageElement + Send + Sync + std::fmt::Debug>(
    cfg: &Cfg,
    opname: &str,
    layout: Layout,
    gen_p: impl Fn(usize) -> P,
    gen_b: impl Fn(usize) -> f64,
    gen_c: impl Fn(usize) -> f64,
    f: impl Fn(P, f64, f64) -> f64 + Sync + Send + Copy,
    erased: impl Fn(
        &ExecContext,
        &mut ErasedRawStridedUninitMut<'_>,
        &ErasedRawStridedPtr<'_>,
        &ErasedRawStridedPtr<'_>,
        &ErasedRawStridedPtr<'_>,
    ),
) {
    let case = format!("ter_{opname}_f64_{}", layout_label(layout));
    if !cfg.enabled(&case) {
        return;
    }
    let s = ew_shape(cfg, layout);
    let p: Vec<P> = (0..s.len).map(gen_p).collect();
    let b: Vec<f64> = (0..s.len).map(gen_b).collect();
    let c: Vec<f64> = (0..s.len).map(gen_c).collect();
    let mut out = vec![f64::NAN; s.len];
    let expected = |k: usize| f(p[lhs_offset(&s, k)], b[k], c[k]);
    let sentinel = -12345.0;

    // raw
    out.fill(sentinel);
    {
        let (pp, pb, pc, pd) = (
            SendConst(p.as_ptr()),
            SendConst(b.as_ptr()),
            SendConst(c.as_ptr()),
            SendPtr(out.as_mut_ptr()),
        );
        let (rows, cols, len) = (s.rows, s.cols, s.len);
        let threads = cfg.threads;
        cfg.measure(&case, "raw", || {
            if cols == 1 {
                par_ranges(threads, len, 4096, |lo, hi| unsafe {
                    let d = std::slice::from_raw_parts_mut(pd.get().add(lo), hi - lo);
                    let x = std::slice::from_raw_parts(pp.get().add(lo), hi - lo);
                    let y = std::slice::from_raw_parts(pb.get().add(lo), hi - lo);
                    let z = std::slice::from_raw_parts(pc.get().add(lo), hi - lo);
                    for (((d, &x), &y), &z) in d.iter_mut().zip(x).zip(y).zip(z) {
                        *d = f(x, y, z);
                    }
                });
            } else {
                par_ranges(threads, cols, TILE, |c0, c1| unsafe {
                    let (p, b, c, d) = (pp.get(), pb.get(), pc.get(), pd.get());
                    for jb in (c0..c1).step_by(TILE) {
                        let je = (jb + TILE).min(c1);
                        for ib in (0..rows).step_by(TILE) {
                            let ie = (ib + TILE).min(rows);
                            for j in jb..je {
                                for i in ib..ie {
                                    let k = i + j * rows;
                                    *d.add(k) = f(*p.add(j + i * cols), *b.add(k), *c.add(k));
                                }
                            }
                        }
                    }
                });
            }
            black_box(pd.get());
        });
    }
    check_all(&case, "raw", &out, same_f64, expected);
    cfg.ok(&case, "raw");

    // typed
    out.fill(sentinel);
    {
        let pv: StridedView<P> = StridedView::new(&p, &s.dims, &s.lhs_strides, 0).unwrap();
        let bv: StridedView<f64> = StridedView::new(&b, &s.dims, &s.col_strides, 0).unwrap();
        let cv: StridedView<f64> = StridedView::new(&c, &s.dims, &s.col_strides, 0).unwrap();
        let mut dv = StridedViewMut::new(&mut out, &s.dims, &s.col_strides, 0).unwrap();
        cfg.measure(&case, "typed", || {
            cfg.exec.run(|| {
                zip_map3_into(&mut dv, black_box(&pv), black_box(&bv), black_box(&cv), f).unwrap()
            });
            black_box(&mut dv);
        });
    }
    check_all(&case, "typed", &out, same_f64, expected);
    cfg.ok(&case, "typed");
    drop(out);

    // erased (uninitialized destination)
    let mut out_u = vec![MaybeUninit::<f64>::uninit(); s.len];
    {
        let pr = ErasedRawStridedRef::from_slice(&p, &s.dims, &s.lhs_strides, 0).unwrap();
        let br = ErasedRawStridedRef::from_slice(&b, &s.dims, &s.col_strides, 0).unwrap();
        let cr = ErasedRawStridedRef::from_slice(&c, &s.dims, &s.col_strides, 0).unwrap();
        let (pp, bp, cp) = (
            ErasedRawStridedPtr::from_ref(&pr),
            ErasedRawStridedPtr::from_ref(&br),
            ErasedRawStridedPtr::from_ref(&cr),
        );
        let mut dest =
            ErasedRawStridedUninitMut::from_uninit_slice(&mut out_u, &s.dims, &s.col_strides, 0).unwrap();
        cfg.measure(&case, "erased_uninit", || {
            erased(&cfg.exec, &mut dest, black_box(&pp), black_box(&bp), black_box(&cp));
            black_box(&mut dest);
        });
    }
    check_all(&case, "erased_uninit", assume_init(&out_u), same_f64, expected);
    cfg.ok(&case, "erased_uninit");
}

fn ternary(cfg: &Cfg) {
    for layout in [Layout::Contig, Layout::Trans] {
        bench_ternary(
            cfg,
            "select",
            layout,
            gen_pred,
            |i| f64::gen(i, 0),
            |i| f64::gen(i, 1),
            |p: bool, a: f64, b: f64| if p { a } else { b },
            |ctx, dest, p, a, b| {
                erased_select_into_uninit(KernelDType::F64, ctx, dest, p, a, b).unwrap()
            },
        );
        bench_ternary(
            cfg,
            "clamp",
            layout,
            gen_clamp_x,
            |_| -0.25,
            |_| 0.25,
            nan_clamp,
            |ctx, dest, x, lo, hi| {
                erased_clamp_into_uninit(KernelDType::F64, ctx, dest, x, lo, hi).unwrap()
            },
        );
    }
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Red {
    Sum,
    Prod,
    Max,
    Min,
}

impl Red {
    fn label(self) -> &'static str {
        match self {
            Red::Sum => "sum",
            Red::Prod => "prod",
            Red::Max => "max",
            Red::Min => "min",
        }
    }
    fn op(self) -> ReduceOp {
        match self {
            Red::Sum => ReduceOp::Sum,
            Red::Prod => ReduceOp::Product,
            Red::Max => ReduceOp::Max,
            Red::Min => ReduceOp::Min,
        }
    }
    fn init(self) -> f64 {
        match self {
            Red::Sum => 0.0,
            Red::Prod => 1.0,
            Red::Max => f64::NEG_INFINITY,
            Red::Min => f64::INFINITY,
        }
    }
    #[inline(always)]
    fn apply(self, a: f64, b: f64) -> f64 {
        match self {
            Red::Sum => a + b,
            Red::Prod => a * b,
            Red::Max => nan_max(a, b),
            Red::Min => nan_min(a, b),
        }
    }
    fn gen(self, i: usize) -> f64 {
        match self {
            Red::Prod => 1.0 + ((i % 1013) as f64 - 506.0) * 1e-9,
            _ => 0.5 + (i % 1013) as f64 / 1013.0 + (i % 7) as f64 * 1e-3,
        }
    }
    fn close(self, a: f64, b: f64) -> bool {
        match self {
            Red::Max | Red::Min => a == b,
            _ => a == b || (a - b).abs() <= 1e-9 * a.abs().max(b.abs()),
        }
    }
}

/// Eight-lane contiguous reduction leaf; max/min use select plus a NaN flag.
#[inline(always)]
fn raw_reduce_contig(red: Red, x: &[f64]) -> f64 {
    let mut acc = [red.init(); 8];
    let mut nan = false;
    let chunks = x.chunks_exact(8);
    let tail = chunks.remainder();
    match red {
        Red::Sum => {
            for c in chunks {
                for l in 0..8 {
                    acc[l] += c[l];
                }
            }
        }
        Red::Prod => {
            for c in chunks {
                for l in 0..8 {
                    acc[l] *= c[l];
                }
            }
        }
        Red::Max => {
            let mut flag = [false; 8];
            for c in chunks {
                for l in 0..8 {
                    flag[l] |= c[l].is_nan();
                    acc[l] = if c[l] > acc[l] { c[l] } else { acc[l] };
                }
            }
            nan = flag.iter().any(|&f| f);
        }
        Red::Min => {
            let mut flag = [false; 8];
            for c in chunks {
                for l in 0..8 {
                    flag[l] |= c[l].is_nan();
                    acc[l] = if c[l] < acc[l] { c[l] } else { acc[l] };
                }
            }
            nan = flag.iter().any(|&f| f);
        }
    }
    let mut r = red.init();
    for &v in acc.iter().chain(tail) {
        r = red.apply(r, v);
    }
    if nan {
        f64::NAN
    } else {
        r
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Axes {
    All,
    Axis0,
    Axis1,
}

fn bench_reduce(cfg: &Cfg, red: Red, axes: Axes, rows0: usize, cols0: usize) {
    let axes_label = match axes {
        Axes::All => "all",
        Axes::Axis0 => "axis0",
        Axes::Axis1 => "axis1",
    };
    let case = format!("red_{}_{axes_label}_{rows0}x{cols0}", red.label());
    if !cfg.enabled(&case) {
        return;
    }
    let (rows, cols) = (cfg.d(rows0), cfg.d(cols0));
    let n = rows * cols;
    let dims = [rows, cols];
    let strides = [1isize, rows as isize];
    let a: Vec<f64> = (0..n).map(|i| red.gen(i)).collect();
    let out_len = match axes {
        Axes::All => 1,
        Axes::Axis0 => cols,
        Axes::Axis1 => rows,
    };
    // Plain sequential reference.
    let reference: Vec<f64> = match axes {
        Axes::All => vec![a.iter().fold(red.init(), |acc, &x| red.apply(acc, x))],
        Axes::Axis0 => (0..cols)
            .map(|j| a[j * rows..(j + 1) * rows].iter().fold(red.init(), |acc, &x| red.apply(acc, x)))
            .collect(),
        Axes::Axis1 => (0..rows)
            .map(|i| (0..cols).fold(red.init(), |acc, j| red.apply(acc, a[i + j * rows])))
            .collect(),
    };
    let check = |variant: &str, out: &[f64]| {
        check_all(&case, variant, out, |x, y| red.close(x, y), |k| reference[k]);
        cfg.ok(&case, variant);
    };
    let threads = cfg.threads;
    let mut out = vec![f64::NAN; out_len];

    // raw
    {
        let pa = SendConst(a.as_ptr());
        let pd = SendPtr(out.as_mut_ptr());
        cfg.measure(&case, "raw", || {
            match axes {
                Axes::All => {
                    let parts = par_map_ranges(threads, n, 4096, |lo, hi| unsafe {
                        raw_reduce_contig(red, std::slice::from_raw_parts(pa.get().add(lo), hi - lo))
                    });
                    let r = parts.into_iter().fold(red.init(), |acc, v| red.apply(acc, v));
                    unsafe { *pd.get() = r };
                }
                Axes::Axis0 => par_ranges(threads, cols, 1, |c0, c1| unsafe {
                    for j in c0..c1 {
                        let col = std::slice::from_raw_parts(pa.get().add(j * rows), rows);
                        *pd.get().add(j) = raw_reduce_contig(red, col);
                    }
                }),
                Axes::Axis1 => par_ranges(threads, rows, 512, |r0, r1| unsafe {
                    let d = std::slice::from_raw_parts_mut(pd.get().add(r0), r1 - r0);
                    d.fill(red.init());
                    let mut nan = false;
                    for j in 0..cols {
                        let x = std::slice::from_raw_parts(pa.get().add(r0 + j * rows), r1 - r0);
                        match red {
                            Red::Sum => d.iter_mut().zip(x).for_each(|(d, &x)| *d += x),
                            Red::Prod => d.iter_mut().zip(x).for_each(|(d, &x)| *d *= x),
                            Red::Max => d.iter_mut().zip(x).for_each(|(d, &x)| {
                                nan |= x.is_nan();
                                *d = if x > *d { x } else { *d };
                            }),
                            Red::Min => d.iter_mut().zip(x).for_each(|(d, &x)| {
                                nan |= x.is_nan();
                                *d = if x < *d { x } else { *d };
                            }),
                        }
                    }
                    if nan {
                        // Rare path: recompute with exact NaN propagation.
                        for (k, d) in d.iter_mut().enumerate() {
                            let i = r0 + k;
                            *d = (0..cols).fold(red.init(), |acc, j| red.apply(acc, *pa.get().add(i + j * rows)));
                        }
                    }
                }),
            }
            black_box(pd.get());
        });
    }
    check("raw", &out);

    // typed
    {
        let av: StridedView<f64> = StridedView::new(&a, &dims, &strides, 0).unwrap();
        let mut last = Vec::new();
        cfg.measure(&case, "typed", || {
            let res = cfg.exec.run(|| match axes {
                Axes::All => vec![reduce(black_box(&av), |x| x, move |p, q| red.apply(p, q), red.init()).unwrap()],
                Axes::Axis0 | Axes::Axis1 => {
                    let axis = if axes == Axes::Axis0 { 0 } else { 1 };
                    let arr = reduce_axis(black_box(&av), axis, |x| x, move |p, q| red.apply(p, q), red.init())
                        .unwrap();
                    arr.into_data()
                }
            });
            last = black_box(res);
        });
        check("typed", &last);
    }

    // erased plan
    out.fill(f64::NAN);
    {
        let (dest_dims, dest_strides): (Vec<usize>, Vec<isize>) = match axes {
            Axes::All => (vec![1], vec![1]),
            Axes::Axis0 => (vec![cols], vec![1]),
            Axes::Axis1 => (vec![rows], vec![1]),
        };
        let plan = match axes {
            Axes::All => ErasedReducePlan::compile(KernelDType::F64, red.op(), &dims, &strides).unwrap(),
            _ => ErasedReducePlan::compile_axes(
                KernelDType::F64,
                red.op(),
                &dims,
                &strides,
                &dest_dims,
                &dest_strides,
                &[if axes == Axes::Axis0 { 0 } else { 1 }],
            )
            .unwrap(),
        };
        let src = ErasedRawStridedRef::from_slice(&a, &dims, &strides, 0).unwrap();
        let mut dest = ErasedRawStridedMut::from_slice_mut(&mut out, &dest_dims, &dest_strides, 0).unwrap();
        cfg.measure(&case, "erased", || {
            plan.execute(&cfg.exec, &mut dest, black_box(&src)).unwrap();
            black_box(&mut dest);
        });
    }
    check("erased", &out);
}

fn reductions(cfg: &Cfg) {
    for (r, c) in [(8192, 4096), (2048, 2048)] {
        for axes in [Axes::All, Axes::Axis0, Axes::Axis1] {
            for red in [Red::Sum, Red::Prod, Red::Max, Red::Min] {
                bench_reduce(cfg, red, axes, r, c);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Structural copy plans
// ---------------------------------------------------------------------------

fn gen_copy(i: usize) -> f64 {
    i as f64 * 0.5 + 1.0
}

/// Destination column-major `rows x cols`, checked against `expected(i, j)`.
fn check_2d(cfg: &Cfg, case: &str, variant: &str, out: &[f64], rows: usize, expected: impl Fn(usize, usize) -> f64) {
    check_all(case, variant, out, |a, b| a == b, |k| expected(k % rows, k / rows));
    cfg.ok(case, variant);
}

fn struct_slice_step2(cfg: &Cfg) {
    let case = "copy_slice_step2";
    if !cfg.enabled(case) {
        return;
    }
    let (r, c) = (cfg.d(8192), cfg.d(4096));
    let (dr, dc) = (r / 2, c);
    let a: Vec<f64> = (0..r * c).map(gen_copy).collect();
    let (od, os) = ([r, c], [1isize, r as isize]);
    let (dd, ds) = ([dr, dc], [1isize, dr as isize]);
    let (starts, limits, steps) = ([0usize, 0], [r, c], [2usize, 1]);
    let expected = |i: usize, j: usize| a[2 * i + j * r];
    let mut out = vec![f64::NAN; dr * dc];
    {
        let (pa, pd, t) = (SendConst(a.as_ptr()), SendPtr(out.as_mut_ptr()), cfg.threads);
        cfg.measure(case, "raw", || {
            par_ranges(t, dc, 1, |c0, c1| unsafe {
                for j in c0..c1 {
                    let d = std::slice::from_raw_parts_mut(pd.get().add(j * dr), dr);
                    let s = std::slice::from_raw_parts(pa.get().add(j * r), 2 * dr);
                    for (d, s) in d.iter_mut().zip(s.chunks_exact(2)) {
                        *d = s[0];
                    }
                }
            });
            black_box(pd.get());
        });
    }
    check_2d(cfg, case, "raw", &out, dr, expected);
    out.fill(f64::NAN);
    {
        let plan = SlicePlan::compile(&od, &os, &dd, &ds, &starts, &limits, &steps).unwrap();
        let src = RawStridedRef::new(&a, &od, &os, 0).unwrap();
        let mut dst = RawStridedMut::new(&mut out, &dd, &ds, 0).unwrap();
        cfg.measure(case, "typed", || {
            cfg.exec.run(|| plan.execute(&mut dst, black_box(&src)).unwrap());
            black_box(&mut dst);
        });
    }
    check_2d(cfg, case, "typed", &out, dr, expected);
    out.fill(f64::NAN);
    {
        let plan = ErasedSlicePlan::compile(KernelDType::F64, &od, &os, &dd, &ds, &starts, &limits, &steps).unwrap();
        let src = ErasedRawStridedRef::from_slice(&a, &od, &os, 0).unwrap();
        let mut dst = ErasedRawStridedMut::from_slice_mut(&mut out, &dd, &ds, 0).unwrap();
        cfg.measure(case, "erased", || {
            plan.execute(&cfg.exec, &mut dst, black_box(&src)).unwrap();
            black_box(&mut dst);
        });
    }
    check_2d(cfg, case, "erased", &out, dr, expected);
}

fn struct_reverse(cfg: &Cfg, axis: usize) {
    let case = if axis == 0 { "copy_reverse_axis0" } else { "copy_reverse_axis1" };
    if !cfg.enabled(case) {
        return;
    }
    let (r, c) = (cfg.d(4096), cfg.d(4096));
    let a: Vec<f64> = (0..r * c).map(gen_copy).collect();
    let (dims, st) = ([r, c], [1isize, r as isize]);
    let expected = |i: usize, j: usize| if axis == 0 { a[(r - 1 - i) + j * r] } else { a[i + (c - 1 - j) * r] };
    let mut out = vec![f64::NAN; r * c];
    {
        let (pa, pd, t) = (SendConst(a.as_ptr()), SendPtr(out.as_mut_ptr()), cfg.threads);
        cfg.measure(case, "raw", || {
            par_ranges(t, c, 1, |c0, c1| unsafe {
                for j in c0..c1 {
                    let d = std::slice::from_raw_parts_mut(pd.get().add(j * r), r);
                    if axis == 0 {
                        let s = std::slice::from_raw_parts(pa.get().add(j * r), r);
                        for (d, &s) in d.iter_mut().zip(s.iter().rev()) {
                            *d = s;
                        }
                    } else {
                        let s = std::slice::from_raw_parts(pa.get().add((c - 1 - j) * r), r);
                        d.copy_from_slice(s);
                    }
                }
            });
            black_box(pd.get());
        });
    }
    check_2d(cfg, case, "raw", &out, r, expected);
    out.fill(f64::NAN);
    {
        let plan = ReversePlan::compile(&dims, &st, &st, &[axis]).unwrap();
        let src = RawStridedRef::new(&a, &dims, &st, 0).unwrap();
        let mut dst = RawStridedMut::new(&mut out, &dims, &st, 0).unwrap();
        cfg.measure(case, "typed", || {
            cfg.exec.run(|| plan.execute(&mut dst, black_box(&src)).unwrap());
            black_box(&mut dst);
        });
    }
    check_2d(cfg, case, "typed", &out, r, expected);
    out.fill(f64::NAN);
    {
        let plan = ErasedReversePlan::compile(KernelDType::F64, &dims, &st, &st, &[axis]).unwrap();
        let src = ErasedRawStridedRef::from_slice(&a, &dims, &st, 0).unwrap();
        let mut dst = ErasedRawStridedMut::from_slice_mut(&mut out, &dims, &st, 0).unwrap();
        cfg.measure(case, "erased", || {
            plan.execute(&cfg.exec, &mut dst, black_box(&src)).unwrap();
            black_box(&mut dst);
        });
    }
    check_2d(cfg, case, "erased", &out, r, expected);
}

fn struct_concat(cfg: &Cfg, axis: usize) {
    let case = if axis == 0 { "copy_concat_axis0" } else { "copy_concat_axis1" };
    if !cfg.enabled(case) {
        return;
    }
    let (r, c) = (cfg.d(4096), cfg.d(4096));
    let (hr, hc) = if axis == 0 { (r / 2, c) } else { (r, c / 2) };
    let a1: Vec<f64> = (0..hr * hc).map(gen_copy).collect();
    let a2: Vec<f64> = (0..hr * hc).map(|i| -gen_copy(i)).collect();
    let (hd, hs) = ([hr, hc], [1isize, hr as isize]);
    let (dd, ds) = ([r, c], [1isize, r as isize]);
    let expected = |i: usize, j: usize| {
        if axis == 0 {
            if i < hr { a1[i + j * hr] } else { a2[(i - hr) + j * hr] }
        } else if j < hc {
            a1[i + j * hr]
        } else {
            a2[i + (j - hc) * hr]
        }
    };
    let mut out = vec![f64::NAN; r * c];
    {
        let (p1, p2, pd, t) = (SendConst(a1.as_ptr()), SendConst(a2.as_ptr()), SendPtr(out.as_mut_ptr()), cfg.threads);
        cfg.measure(case, "raw", || {
            if axis == 0 {
                par_ranges(t, c, 1, |c0, c1| unsafe {
                    for j in c0..c1 {
                        std::ptr::copy_nonoverlapping(p1.get().add(j * hr), pd.get().add(j * r), hr);
                        std::ptr::copy_nonoverlapping(p2.get().add(j * hr), pd.get().add(j * r + hr), hr);
                    }
                });
            } else {
                par_copy(t, pd.get(), p1.get(), hr * hc);
                par_copy(t, unsafe { pd.get().add(hr * hc) }, p2.get(), hr * hc);
            }
            black_box(pd.get());
        });
    }
    check_2d(cfg, case, "raw", &out, r, expected);
    out.fill(f64::NAN);
    let in_dims: [&[usize]; 2] = [&hd, &hd];
    let in_strides: [&[isize]; 2] = [&hs, &hs];
    {
        let plan = ConcatenatePlan::compile(&in_dims, &in_strides, &dd, &ds, axis).unwrap();
        let inputs = [
            RawStridedRef::new(&a1, &hd, &hs, 0).unwrap(),
            RawStridedRef::new(&a2, &hd, &hs, 0).unwrap(),
        ];
        let mut dst = RawStridedMut::new(&mut out, &dd, &ds, 0).unwrap();
        cfg.measure(case, "typed", || {
            cfg.exec.run(|| plan.execute(&mut dst, black_box(&inputs)).unwrap());
            black_box(&mut dst);
        });
    }
    check_2d(cfg, case, "typed", &out, r, expected);
    out.fill(f64::NAN);
    {
        let plan = ErasedConcatenatePlan::compile(KernelDType::F64, &in_dims, &in_strides, &dd, &ds, axis).unwrap();
        let inputs = [
            ErasedRawStridedRef::from_slice(&a1, &hd, &hs, 0).unwrap(),
            ErasedRawStridedRef::from_slice(&a2, &hd, &hs, 0).unwrap(),
        ];
        let mut dst = ErasedRawStridedMut::from_slice_mut(&mut out, &dd, &ds, 0).unwrap();
        cfg.measure(case, "erased", || {
            plan.execute(&cfg.exec, &mut dst, black_box(&inputs)).unwrap();
            black_box(&mut dst);
        });
    }
    check_2d(cfg, case, "erased", &out, r, expected);
}

fn struct_dynamic_slice(cfg: &Cfg, rank: usize) {
    let case = if rank == 1 { "copy_dynslice_rank1" } else { "copy_dynslice_rank2" };
    if !cfg.enabled(case) {
        return;
    }
    // (operand dims, window dims, starts)
    let (od, wd, starts): (Vec<usize>, Vec<usize>, Vec<i64>) = if rank == 1 {
        let w = cfg.n1(16_777_216);
        (vec![2 * w], vec![w], vec![(w / 3) as i64])
    } else {
        let (o, w) = (cfg.d(4608), cfg.d(4096));
        (vec![o, o], vec![w, w], vec![(cfg.d(100)) as i64, (cfg.d(200)) as i64])
    };
    let os = strided_kernel::col_major_strides(&od);
    let ws = strided_kernel::col_major_strides(&wd);
    let (sd, ss) = ([rank], [1isize]);
    let a: Vec<f64> = (0..od.iter().product()).map(gen_copy).collect();
    let wr = wd[0];
    let wc = if rank == 1 { 1 } else { wd[1] };
    let or = od[0];
    let (s0, s1) = (starts[0] as usize, if rank == 1 { 0 } else { starts[1] as usize });
    let expected = |i: usize, j: usize| a[(i + s0) + (j + s1) * or];
    let mut out = vec![f64::NAN; wr * wc];
    {
        let (pa, pd, t) = (SendConst(a.as_ptr()), SendPtr(out.as_mut_ptr()), cfg.threads);
        cfg.measure(case, "raw", || {
            if rank == 1 {
                par_copy(t, pd.get(), unsafe { pa.get().add(s0) }, wr);
            } else {
                par_ranges(t, wc, 1, |c0, c1| unsafe {
                    for j in c0..c1 {
                        std::ptr::copy_nonoverlapping(pa.get().add(s0 + (j + s1) * or), pd.get().add(j * wr), wr);
                    }
                });
            }
            black_box(pd.get());
        });
    }
    check_2d(cfg, case, "raw", &out, wr, expected);
    out.fill(f64::NAN);
    {
        let plan = DynamicSlicePlan::compile(&od, &os, &sd, &ss, &wd, &ws, &wd).unwrap();
        let src = RawStridedRef::new(&a, &od, &os, 0).unwrap();
        let st = RawStridedRef::new(&starts, &sd, &ss, 0).unwrap();
        let mut dst = RawStridedMut::new(&mut out, &wd, &ws, 0).unwrap();
        cfg.measure(case, "typed", || {
            cfg.exec.run(|| plan.execute(&mut dst, black_box(&src), black_box(&st)).unwrap());
            black_box(&mut dst);
        });
    }
    check_2d(cfg, case, "typed", &out, wr, expected);
    out.fill(f64::NAN);
    {
        let plan =
            ErasedDynamicSlicePlan::compile(KernelDType::F64, KernelDType::I64, &od, &os, &sd, &ss, &wd, &ws, &wd)
                .unwrap();
        let src = ErasedRawStridedRef::from_slice(&a, &od, &os, 0).unwrap();
        let st = ErasedRawStridedRef::from_slice(&starts, &sd, &ss, 0).unwrap();
        let mut dst = ErasedRawStridedMut::from_slice_mut(&mut out, &wd, &ws, 0).unwrap();
        cfg.measure(case, "erased", || {
            plan.execute(&cfg.exec, &mut dst, black_box(&src), black_box(&st)).unwrap();
            black_box(&mut dst);
        });
    }
    check_2d(cfg, case, "erased", &out, wr, expected);
}

fn struct_pad(cfg: &Cfg) {
    let case = "copy_pad";
    if !cfg.enabled(case) {
        return;
    }
    let (dr, dc) = (cfg.d(4096), cfg.d(4096));
    let p = cfg.d(48);
    let (r, c) = (dr - 2 * p, dc - 2 * p);
    let a: Vec<f64> = (0..r * c).map(gen_copy).collect();
    let (od, os) = ([r, c], [1isize, r as isize]);
    let (dd, ds) = ([dr, dc], [1isize, dr as isize]);
    let (lo, hi, interior) = ([p as i64; 2], [p as i64; 2], [0i64; 2]);
    let expected = |i: usize, j: usize| {
        if i >= p && i < p + r && j >= p && j < p + c {
            a[(i - p) + (j - p) * r]
        } else {
            0.0
        }
    };
    let mut out = vec![f64::NAN; dr * dc];
    {
        let (pa, pd, t) = (SendConst(a.as_ptr()), SendPtr(out.as_mut_ptr()), cfg.threads);
        cfg.measure(case, "raw", || {
            par_ranges(t, dc, 1, |c0, c1| unsafe {
                for j in c0..c1 {
                    let d = std::slice::from_raw_parts_mut(pd.get().add(j * dr), dr);
                    if j < p || j >= p + c {
                        d.fill(0.0);
                    } else {
                        d[..p].fill(0.0);
                        d[p..p + r].copy_from_slice(std::slice::from_raw_parts(pa.get().add((j - p) * r), r));
                        d[p + r..].fill(0.0);
                    }
                }
            });
            black_box(pd.get());
        });
    }
    check_2d(cfg, case, "raw", &out, dr, expected);
    out.fill(f64::NAN);
    {
        let plan = PadPlan::compile(&od, &os, &dd, &ds, &lo, &hi, &interior).unwrap();
        let src = RawStridedRef::new(&a, &od, &os, 0).unwrap();
        let mut dst = RawStridedMut::new(&mut out, &dd, &ds, 0).unwrap();
        cfg.measure(case, "typed", || {
            cfg.exec.run(|| plan.execute(&mut dst, black_box(&src), 0.0).unwrap());
            black_box(&mut dst);
        });
    }
    check_2d(cfg, case, "typed", &out, dr, expected);
    out.fill(f64::NAN);
    {
        let plan = ErasedPadPlan::compile(KernelDType::F64, &od, &os, &dd, &ds, &lo, &hi, &interior).unwrap();
        let src = ErasedRawStridedRef::from_slice(&a, &od, &os, 0).unwrap();
        let mut dst = ErasedRawStridedMut::from_slice_mut(&mut out, &dd, &ds, 0).unwrap();
        let fill = 0.0f64.to_ne_bytes();
        cfg.measure(case, "erased", || {
            plan.execute(&cfg.exec, &mut dst, black_box(&src), &fill).unwrap();
            black_box(&mut dst);
        });
    }
    check_2d(cfg, case, "erased", &out, dr, expected);
}

fn structural(cfg: &Cfg) {
    struct_slice_step2(cfg);
    struct_reverse(cfg, 0);
    struct_reverse(cfg, 1);
    struct_concat(cfg, 0);
    struct_concat(cfg, 1);
    struct_dynamic_slice(cfg, 1);
    struct_dynamic_slice(cfg, 2);
    struct_pad(cfg);
}

// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = env::args().collect();
    let timing = args.iter().any(|a| a == "--time");
    let threads: usize = args
        .iter()
        .position(|a| a == "--threads")
        .and_then(|p| args.get(p + 1))
        .and_then(|v| v.parse().ok())
        .expect("usage: kernel_scaling --threads N [--time]");
    assert!(threads >= 1 && threads < 255);
    let runs = env_usize("BENCH_RUNS", 11);
    let warmup = env_usize("BENCH_WARMUP", 2);
    let shrink = env_usize("KERNEL_SCALING_SHRINK", 1).max(1);
    assert!(runs > 0);
    let filter = env::var("BENCH_FILTER").ok().filter(|f| !f.is_empty());

    // The global pool is the only Rayon pool in the process and is bounded to
    // `threads`, so no strided or baseline code can fan out wider than that.
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .expect("failed to build the bounded global Rayon pool");
    let exec = if threads == 1 {
        ExecContext::serial()
    } else {
        ExecContext::max_threads(threads).unwrap()
    };
    let cfg = Cfg {
        threads,
        exec,
        timing,
        runs,
        warmup,
        shrink,
        filter,
    };
    eprintln!(
        "# kernel_scaling strided-rs={} threads={threads} timing={timing} runs={runs} warmup={warmup} shrink={shrink}",
        option_env!("STRIDED_RS_REV").unwrap_or("unknown")
    );

    // `rayon::scope` called from the main thread runs its body on a pool
    // worker, so every kernel below executes inside the bounded pool.
    rayon::scope(|_| {
        verify_threads(&cfg);
        if timing {
            println!("case,variant,threads,median_ns,samples");
        }
        elementwise(&cfg);
        ternary(&cfg);
        reductions(&cfg);
        structural(&cfg);
    });
    eprintln!("# done");
}
