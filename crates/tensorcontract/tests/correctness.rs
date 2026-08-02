//! End-to-end correctness against the brute-force oracle.
//!
//! Two complementary strategies:
//!
//! 1. **Randomised small problems with tiny cache blocking.** Setting
//!    `MC`/`KC`/`NC` to a handful of elements forces every level of the
//!    five-loop nest, every partial block and every packing edge case to fire
//!    on tensors small enough for the `O(prod of all extents)` oracle.
//! 2. **Large pure-GEMM problems with default blocking.** These cross the real
//!    `MC`/`KC`/`NC` boundaries and are checked against a straightforward
//!    triple loop, which is fast enough at these sizes.
//!
//! Together they cover the loop arithmetic both structurally and at scale.

use num_complex::Complex;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use tensorcontract::element::{Element, Real};
use tensorcontract::kernel::{Blocking, KernelSet};
use tensorcontract::plan::{ElementOp, Operand};
use tensorcontract::reference::{contract_reference, RefOperand};
use tensorcontract::{Layout, Plan};

// ---------------------------------------------------------------- utilities

fn sample<T: Element>(rng: &mut ChaCha8Rng) -> T {
    let re = T::Real::from_f64(rng.gen_range(-1.0..1.0));
    let im = if T::IS_COMPLEX {
        T::Real::from_f64(rng.gen_range(-1.0..1.0))
    } else {
        T::Real::ZERO
    };
    T::from_parts(re, im)
}

fn fill<T: Element>(n: usize, rng: &mut ChaCha8Rng) -> Vec<T> {
    (0..n).map(|_| sample::<T>(rng)).collect()
}

fn rel_error<T: Element>(got: &[T], want: &[T]) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for (&g, &w) in got.iter().zip(want) {
        num += g.sub(w).norm().powi(2);
        den += w.norm().powi(2);
    }
    if den == 0.0 {
        num.sqrt()
    } else {
        (num / den).sqrt()
    }
}

fn tol<T: Element>() -> f64 {
    if core::mem::size_of::<T::Real>() == 4 {
        2e-4
    } else {
        1e-11
    }
}

/// A dense layout whose modes are contiguous in a random order, producing an
/// arbitrary stride permutation.
fn random_layout(extents: &[i64], rng: &mut ChaCha8Rng) -> Layout {
    let n = extents.len();
    let mut order: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        order.swap(i, rng.gen_range(0..=i));
    }
    let mut strides = vec![0i64; n];
    let mut acc = 1i64;
    for &d in &order {
        strides[d] = acc;
        acc *= extents[d];
    }
    Layout {
        extents: extents.to_vec(),
        strides,
    }
}

fn shuffled<T: Clone>(v: &[T], rng: &mut ChaCha8Rng) -> Vec<T> {
    let mut out = v.to_vec();
    for i in (1..out.len()).rev() {
        out.swap(i, rng.gen_range(0..=i));
    }
    out
}

// ------------------------------------------------------- randomised problems

struct Problem {
    idx_a: Vec<i64>,
    idx_b: Vec<i64>,
    idx_c: Vec<i64>,
    idx_d: Vec<i64>,
    la: Layout,
    lb: Layout,
    lc: Layout,
    ld: Layout,
    conj_a: bool,
    conj_b: bool,
    conj_c: bool,
    conj_d: bool,
    use_c: bool,
}

/// Generate a contraction covering all TAPP index cases the engine supports.
fn random_problem(rng: &mut ChaCha8Rng, complex: bool) -> Problem {
    let mut next_label = 0i64;
    let new_labels = |n: usize, next: &mut i64| -> Vec<i64> {
        (0..n)
            .map(|_| {
                *next += 1;
                *next
            })
            .collect()
    };

    let nm = rng.gen_range(0..=2);
    let nn = rng.gen_range(0..=2);
    let nk = rng.gen_range(1..=2);
    let nh = rng.gen_range(0..=1);
    let nia = rng.gen_range(0..=1); // isolated in A -> reduction
    let nib = rng.gen_range(0..=1); // isolated in B -> reduction

    let m = new_labels(nm, &mut next_label);
    let n = new_labels(nn, &mut next_label);
    let k = new_labels(nk, &mut next_label);
    let h = new_labels(nh, &mut next_label);
    let ia = new_labels(nia, &mut next_label);
    let ib = new_labels(nib, &mut next_label);

    let extent_of = |l: i64, rng: &mut ChaCha8Rng| -> i64 {
        let _ = l;
        rng.gen_range(1..=4)
    };
    let mut ext: Vec<(i64, i64)> = Vec::new();
    for &l in m.iter().chain(&n).chain(&k).chain(&h).chain(&ia).chain(&ib) {
        let e = extent_of(l, rng);
        ext.push((l, e));
    }
    let e_of = |l: i64| ext.iter().find(|x| x.0 == l).unwrap().1;

    let mut idx_a: Vec<i64> = m.iter().chain(&k).chain(&h).chain(&ia).copied().collect();
    let mut idx_b: Vec<i64> = k.iter().chain(&n).chain(&h).chain(&ib).copied().collect();
    let idx_d_base: Vec<i64> = m.iter().chain(&n).chain(&h).copied().collect();

    // With some probability, repeat a label inside A: selects A's diagonal.
    if !idx_a.is_empty() && rng.gen_bool(0.25) {
        let l = idx_a[rng.gen_range(0..idx_a.len())];
        idx_a.push(l);
    }
    if !idx_b.is_empty() && rng.gen_bool(0.15) {
        let l = idx_b[rng.gen_range(0..idx_b.len())];
        idx_b.push(l);
    }

    idx_a = shuffled(&idx_a, rng);
    idx_b = shuffled(&idx_b, rng);
    let idx_d = shuffled(&idx_d_base, rng);
    // C carries the same labels as D but may be laid out differently.
    let idx_c = shuffled(&idx_d_base, rng);

    let shape = |idx: &[i64]| -> Vec<i64> { idx.iter().map(|&l| e_of(l)).collect() };
    Problem {
        la: random_layout(&shape(&idx_a), rng),
        lb: random_layout(&shape(&idx_b), rng),
        lc: random_layout(&shape(&idx_c), rng),
        ld: random_layout(&shape(&idx_d), rng),
        conj_a: complex && rng.gen_bool(0.3),
        conj_b: complex && rng.gen_bool(0.3),
        conj_c: complex && rng.gen_bool(0.3),
        conj_d: complex && rng.gen_bool(0.2),
        use_c: rng.gen_bool(0.6),
        idx_a,
        idx_b,
        idx_c,
        idx_d,
    }
}

fn op(b: bool) -> ElementOp {
    if b {
        ElementOp::Conjugate
    } else {
        ElementOp::Identity
    }
}

fn check_problem<T>(p: &Problem, rng: &mut ChaCha8Rng, blocking: Option<Blocking>) -> f64
where
    T: Element,
    T::Real: KernelSet,
{
    let a: Vec<T> = fill(p.la.storage_len() as usize, rng);
    let b: Vec<T> = fill(p.lb.storage_len() as usize, rng);
    let c: Vec<T> = fill(p.lc.storage_len() as usize, rng);
    let alpha = sample::<T>(rng);
    let beta = if p.use_c { sample::<T>(rng) } else { T::zero() };

    let mut got: Vec<T> = fill(p.ld.storage_len() as usize, rng);
    let mut want = got.clone();

    let mut plan = Plan::new(
        Operand {
            layout: &p.la,
            idx: &p.idx_a,
            op: op(p.conj_a),
        },
        Operand {
            layout: &p.lb,
            idx: &p.idx_b,
            op: op(p.conj_b),
        },
        p.use_c.then(|| Operand {
            layout: &p.lc,
            idx: &p.idx_c,
            op: op(p.conj_c),
        }),
        Operand {
            layout: &p.ld,
            idx: &p.idx_d,
            op: op(p.conj_d),
        },
    )
    .expect("plan");
    if let Some(blk) = blocking {
        plan = plan.with_blocking(blk);
    }

    unsafe {
        plan.run_raw::<T>(
            alpha,
            a.as_ptr(),
            b.as_ptr(),
            beta,
            c.as_ptr(),
            got.as_mut_ptr(),
        );
    }

    contract_reference::<T>(
        alpha,
        &RefOperand {
            data: &a,
            layout: &p.la,
            idx: &p.idx_a,
            op: op(p.conj_a),
        },
        &RefOperand {
            data: &b,
            layout: &p.lb,
            idx: &p.idx_b,
            op: op(p.conj_b),
        },
        beta,
        p.use_c.then_some(&RefOperand {
            data: &c,
            layout: &p.lc,
            idx: &p.idx_c,
            op: op(p.conj_c),
        }),
        &mut want,
        &p.ld,
        &p.idx_d,
        op(p.conj_d),
    )
    .expect("reference");

    rel_error(&got, &want)
}

/// Blocking small enough that a 4x4x4 problem still spans several blocks.
const TINY: Blocking = Blocking {
    mc: 1,
    kc: 2,
    nc: 1,
};

fn randomised_sweep<T>(seed: u64, iters: usize, complex: bool)
where
    T: Element,
    T::Real: KernelSet,
{
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    for i in 0..iters {
        let p = random_problem(&mut rng, complex);
        for blk in [Some(TINY), None] {
            let err = check_problem::<T>(&p, &mut rng, blk);
            assert!(
                err <= tol::<T>(),
                "iteration {i} blocking {blk:?}: relative error {err:e}\n\
                 idx_a={:?} idx_b={:?} idx_c={:?} idx_d={:?}\n\
                 la={:?} lb={:?} lc={:?} ld={:?}\n\
                 conj a/b/c/d = {}/{}/{}/{} use_c={}",
                p.idx_a,
                p.idx_b,
                p.idx_c,
                p.idx_d,
                p.la,
                p.lb,
                p.lc,
                p.ld,
                p.conj_a,
                p.conj_b,
                p.conj_c,
                p.conj_d,
                p.use_c,
            );
        }
    }
}

#[test]
fn randomised_f64() {
    randomised_sweep::<f64>(0xC0FFEE, 300, false);
}

#[test]
fn randomised_f32() {
    randomised_sweep::<f32>(0xBEEF, 200, false);
}

#[test]
fn randomised_c64() {
    randomised_sweep::<Complex<f64>>(0xD00D, 300, true);
}

#[test]
fn randomised_c32() {
    randomised_sweep::<Complex<f32>>(0xFEED, 200, true);
}

// -------------------------------------------------- large blocking boundaries

fn naive_gemm<T: Element>(m: usize, n: usize, k: usize, a: &[T], b: &[T]) -> Vec<T> {
    let mut c = vec![T::zero(); m * n];
    for j in 0..n {
        for p in 0..k {
            let bv = b[p + j * k];
            for i in 0..m {
                c[i + j * m] = c[i + j * m].add(a[i + p * m].mul(bv));
            }
        }
    }
    c
}

/// Sizes chosen to straddle the real `MC`/`KC`/`NC` boundaries with an
/// awkward remainder in every dimension.
fn large_gemm_case<T>(m: usize, n: usize, k: usize, seed: u64)
where
    T: Element,
    T::Real: KernelSet,
{
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let a: Vec<T> = fill(m * k, &mut rng);
    let b: Vec<T> = fill(k * n, &mut rng);
    let mut d = vec![T::zero(); m * n];

    let la = Layout::col_major(&[m as i64, k as i64]);
    let lb = Layout::col_major(&[k as i64, n as i64]);
    let ld = Layout::col_major(&[m as i64, n as i64]);
    let plan = Plan::new(
        Operand::new(&la, &[0, 2]),
        Operand::new(&lb, &[2, 1]),
        None,
        Operand::new(&ld, &[0, 1]),
    )
    .unwrap();
    assert!(plan.stats.is_pure_gemm);

    unsafe {
        plan.run_raw::<T>(
            T::one(),
            a.as_ptr(),
            b.as_ptr(),
            T::zero(),
            d.as_ptr(),
            d.as_mut_ptr(),
        )
    };

    let want = naive_gemm(m, n, k, &a, &b);
    let err = rel_error(&d, &want);
    assert!(err <= tol::<T>(), "{m}x{n}x{k}: relative error {err:e}");
}

#[test]
fn large_gemm_crosses_cache_blocks_f64() {
    // default f64 blocking is mc=256 kc=256 nc>=1536
    large_gemm_case::<f64>(301, 197, 523, 1);
}

#[test]
fn large_gemm_crosses_cache_blocks_c64() {
    large_gemm_case::<Complex<f64>>(211, 143, 401, 2);
}

#[test]
fn large_gemm_crosses_cache_blocks_f32() {
    large_gemm_case::<f32>(401, 233, 797, 3);
}

#[test]
fn large_gemm_crosses_cache_blocks_c32() {
    large_gemm_case::<Complex<f32>>(277, 181, 613, 4);
}

// ------------------------------------------------------------- degenerate cases

#[test]
fn empty_contraction_dimension_scales_c() {
    // D[i,j] = beta * C[i,j] when the contracted extent is zero.
    let la = Layout::col_major(&[3, 0]);
    let lb = Layout::col_major(&[0, 2]);
    let lc = Layout::col_major(&[3, 2]);
    let a: Vec<f64> = vec![];
    let b: Vec<f64> = vec![];
    let c: Vec<f64> = (0..6).map(|x| x as f64).collect();
    let mut d = vec![-1.0f64; 6];
    let plan = Plan::new(
        Operand::new(&la, &[0, 2]),
        Operand::new(&lb, &[2, 1]),
        Some(Operand::new(&lc, &[0, 1])),
        Operand::new(&lc, &[0, 1]),
    )
    .unwrap();
    assert!(plan.has_empty_contraction());
    unsafe {
        plan.run_raw::<f64>(1.0, a.as_ptr(), b.as_ptr(), 2.0, c.as_ptr(), d.as_mut_ptr());
    }
    assert_eq!(d, vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0]);
}

#[test]
fn zero_sized_output_is_a_no_op() {
    let la = Layout::col_major(&[0, 3]);
    let lb = Layout::col_major(&[3, 2]);
    let ld = Layout::col_major(&[0, 2]);
    let b: Vec<f64> = vec![0.0; 6];
    let mut d: Vec<f64> = vec![];
    let plan = Plan::new(
        Operand::new(&la, &[0, 2]),
        Operand::new(&lb, &[2, 1]),
        None,
        Operand::new(&ld, &[0, 1]),
    )
    .unwrap();
    assert!(plan.is_empty());
    unsafe {
        plan.run_raw::<f64>(
            1.0,
            core::ptr::NonNull::dangling().as_ptr(),
            b.as_ptr(),
            0.0,
            d.as_ptr(),
            d.as_mut_ptr(),
        );
    }
}

#[test]
fn scalar_output_full_reduction() {
    // d = sum_{i,j} A[i,j] * B[i,j]  (a full double contraction to a scalar)
    let la = Layout::col_major(&[4, 5]);
    let lb = Layout::col_major(&[4, 5]);
    let ld = Layout::col_major(&[]);
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let a: Vec<f64> = fill(20, &mut rng);
    let b: Vec<f64> = fill(20, &mut rng);
    let mut d = vec![0.0f64];
    let plan = Plan::new(
        Operand::new(&la, &[0, 1]),
        Operand::new(&lb, &[0, 1]),
        None,
        Operand::new(&ld, &[]),
    )
    .unwrap();
    unsafe { plan.run_raw::<f64>(1.0, a.as_ptr(), b.as_ptr(), 0.0, d.as_ptr(), d.as_mut_ptr()) };
    let want: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
    assert!((d[0] - want).abs() < 1e-12, "{} vs {}", d[0], want);
}

#[test]
fn negative_strides_via_reversed_axis() {
    // A[i,k] stored with the i axis reversed: stride -1, base at the end.
    let m = 5usize;
    let k = 4usize;
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    let storage: Vec<f64> = fill(m * k, &mut rng);
    // logical A[i,p] = storage[(m-1-i) + p*m]
    let la = Layout::new(vec![m as i64, k as i64], vec![-1, m as i64]).unwrap();
    let lb = Layout::col_major(&[k as i64, 3]);
    let ld = Layout::col_major(&[m as i64, 3]);
    let b: Vec<f64> = fill(k * 3, &mut rng);
    let mut d = vec![0.0f64; m * 3];

    let plan = Plan::new(
        Operand::new(&la, &[0, 2]),
        Operand::new(&lb, &[2, 1]),
        None,
        Operand::new(&ld, &[0, 1]),
    )
    .unwrap();
    // Base pointer sits at the last element of the first column.
    unsafe {
        plan.run_raw::<f64>(
            1.0,
            storage.as_ptr().add(m - 1),
            b.as_ptr(),
            0.0,
            d.as_ptr(),
            d.as_mut_ptr(),
        );
    }

    for i in 0..m {
        for j in 0..3 {
            let mut want = 0.0;
            for p in 0..k {
                want += storage[(m - 1 - i) + p * m] * b[p + j * k];
            }
            assert!(
                (d[i + j * m] - want).abs() < 1e-12,
                "({i},{j}): {} vs {want}",
                d[i + j * m]
            );
        }
    }
}

#[test]
fn conjugation_matrix_is_consistent() {
    // Check all 16 combinations of conjugation flags against the oracle.
    let mut rng = ChaCha8Rng::seed_from_u64(23);
    let la = Layout::col_major(&[3, 4]);
    let lb = Layout::col_major(&[4, 2]);
    let ld = Layout::col_major(&[3, 2]);
    type T = Complex<f64>;
    let a: Vec<T> = fill(12, &mut rng);
    let b: Vec<T> = fill(8, &mut rng);
    let c: Vec<T> = fill(6, &mut rng);
    let alpha = sample::<T>(&mut rng);
    let beta = sample::<T>(&mut rng);

    for mask in 0..16u8 {
        let (ca, cb, cc, cd) = (mask & 1 != 0, mask & 2 != 0, mask & 4 != 0, mask & 8 != 0);
        let mut got = vec![T::zero(); 6];
        let mut want = vec![T::zero(); 6];
        let plan = Plan::new(
            Operand {
                layout: &la,
                idx: &[0, 2],
                op: op(ca),
            },
            Operand {
                layout: &lb,
                idx: &[2, 1],
                op: op(cb),
            },
            Some(Operand {
                layout: &ld,
                idx: &[0, 1],
                op: op(cc),
            }),
            Operand {
                layout: &ld,
                idx: &[0, 1],
                op: op(cd),
            },
        )
        .unwrap()
        // force multiple K blocks so the accumulate path's conjugation
        // handling is exercised
        .with_blocking(Blocking {
            mc: 1,
            kc: 1,
            nc: 1,
        });
        unsafe {
            plan.run_raw::<T>(
                alpha,
                a.as_ptr(),
                b.as_ptr(),
                beta,
                c.as_ptr(),
                got.as_mut_ptr(),
            )
        };
        contract_reference::<T>(
            alpha,
            &RefOperand {
                data: &a,
                layout: &la,
                idx: &[0, 2],
                op: op(ca),
            },
            &RefOperand {
                data: &b,
                layout: &lb,
                idx: &[2, 1],
                op: op(cb),
            },
            beta,
            Some(&RefOperand {
                data: &c,
                layout: &ld,
                idx: &[0, 1],
                op: op(cc),
            }),
            &mut want,
            &ld,
            &[0, 1],
            op(cd),
        )
        .unwrap();
        let err = rel_error(&got, &want);
        assert!(err < 1e-12, "conj mask {mask:04b}: relative error {err:e}");
    }
}
