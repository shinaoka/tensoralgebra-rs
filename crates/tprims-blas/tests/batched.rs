use num_complex::Complex64;
use strided_view::{StridedView, StridedViewMut};
use tprims_blas::{gemm_batched, BatchIn, BatchStrategy, Conj, Error, Scalar, Selected};
use tprims_exec::{Exec, Pool};

mod common;
use common::{data, dense, from_f64, max_rel_err, naive_gemm};

fn item<'a, T>(d: &'a [T], s: &[isize], r: usize, c: usize, i: usize) -> StridedView<'a, T> {
    StridedView::new(d, &[r, c], &[s[0], s[1]], s[2] * i as isize).unwrap()
}

pub const STRATEGIES: [BatchStrategy; 2] = [BatchStrategy::FaerLoop, BatchStrategy::Tblis];

/// Batch of `nb` GEMMs m x k * k x n. `permuted` stores the batch axis first
/// (smallest stride) instead of last.
#[allow(clippy::too_many_arguments)]
fn check<T: Scalar>(
    exec: &Exec<'_>,
    strategy: BatchStrategy,
    m: usize,
    n: usize,
    k: usize,
    nb: usize,
    permuted: bool,
    ca: Conj,
    beta: T,
) -> Selected {
    let lay = |r: usize, c: usize| -> (Vec<isize>, usize) {
        if permuted {
            (vec![nb as isize, (nb * r) as isize, 1], r * c * nb)
        } else {
            (vec![1, r as isize, (r * c) as isize], r * c * nb)
        }
    };
    let (sa, la) = lay(m, k);
    let (sb, lb) = lay(k, n);
    let (sc, lc) = lay(m, n);
    let ad: Vec<T> = data(la, 11);
    let bd: Vec<T> = data(lb, 12);
    let mut cd: Vec<T> = data(lc, 13);
    let a = StridedView::new(&ad, &[m, k, nb], &sa, 0).unwrap();
    let b = StridedView::new(&bd, &[k, n, nb], &sb, 0).unwrap();
    let alpha = from_f64::<T>(0.75);
    let want: Vec<Vec<T>> = (0..nb)
        .map(|i| {
            let ai = item(&ad, &sa, m, k, i);
            let bi = item(&bd, &sb, k, n, i);
            let ci = item(&cd, &sc, m, n, i);
            naive_gemm(alpha, &ai, ca, &bi, Conj::No, beta, &ci)
        })
        .collect();
    let mut ai = BatchIn::new(&a);
    ai.conj = ca;
    let sel = {
        let mut c = StridedViewMut::new(&mut cd, &[m, n, nb], &sc, 0).unwrap();
        gemm_batched(exec, alpha, ai, BatchIn::new(&b), beta, &mut c, strategy).unwrap()
    };
    for (i, w) in want.iter().enumerate() {
        let got =
            dense(&StridedView::new(&cd, &[m, n], &[sc[0], sc[1]], sc[2] * i as isize).unwrap());
        let tol = if std::mem::size_of::<<T as Scalar>::Re>() == 4 {
            2e-5
        } else {
            1e-12
        };
        let err = max_rel_err(&got, w);
        assert!(
            err < tol,
            "{strategy:?} item {i} m{m} n{n} k{k} nb{nb} perm{permuted}: {err}"
        );
    }
    sel
}

#[test]
fn batched_matches_reference_per_item() {
    for strategy in STRATEGIES {
        for &(m, n, k) in &[(4, 3, 5), (16, 16, 16), (1, 1, 1)] {
            for nb in [1, 3, 64] {
                for permuted in [false, true] {
                    check::<f64>(
                        &Exec::serial(),
                        strategy,
                        m,
                        n,
                        k,
                        nb,
                        permuted,
                        Conj::No,
                        1.25,
                    );
                    check::<Complex64>(
                        &Exec::serial(),
                        strategy,
                        m,
                        n,
                        k,
                        nb,
                        permuted,
                        Conj::Yes,
                        from_f64(0.0),
                    );
                }
            }
        }
        check::<f32>(&Exec::serial(), strategy, 8, 8, 8, 5, false, Conj::No, 0.0);
    }
}

#[test]
fn outer_parallel_on_a_pool_matches_and_is_reported() {
    let tp = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let pool = Pool::borrow(&tp);
    let exec = Exec::rayon(&pool);
    for strategy in STRATEGIES {
        let sel = check::<f64>(&exec, strategy, 16, 16, 16, 2048, false, Conj::No, 0.5);
        assert!(
            matches!(
                sel,
                Selected::FaerLoop {
                    outer_parallel: true
                } | Selected::Tblis {
                    outer_parallel: true
                }
            ),
            "{sel:?}"
        );
        let sel = check::<f64>(&exec, strategy, 2, 2, 2, 3, false, Conj::No, 0.5);
        assert!(
            matches!(
                sel,
                Selected::FaerLoop {
                    outer_parallel: false
                } | Selected::Tblis {
                    outer_parallel: false
                }
            ),
            "{sel:?}"
        );
        check::<f64>(&exec, strategy, 300, 200, 250, 2, false, Conj::No, 0.0);
    }
}

#[test]
fn beta_zero_with_nan_and_aliased_batches() {
    for strategy in STRATEGIES {
        let a: Vec<f64> = data(2 * 3 * 4, 1);
        let av = StridedView::new(&a, &[2, 3, 4], &[1, 2, 6], 0).unwrap();
        let bv = StridedView::new(&a, &[3, 2, 4], &[1, 3, 6], 0).unwrap();
        let mut c = vec![f64::NAN; 16];
        {
            let mut cv = StridedViewMut::new(&mut c, &[2, 2, 4], &[1, 2, 4], 0).unwrap();
            gemm_batched(
                &Exec::serial(),
                1.0,
                BatchIn::new(&av),
                BatchIn::new(&bv),
                0.0,
                &mut cv,
                strategy,
            )
            .unwrap();
        }
        assert!(c.iter().all(|x| x.is_finite()), "{strategy:?}");
        let mut c = vec![9.0; 4];
        {
            let mut cv = StridedViewMut::new(&mut c, &[2, 2, 4], &[1, 2, 0], 0).unwrap();
            let e = gemm_batched(
                &Exec::serial(),
                1.0,
                BatchIn::new(&av),
                BatchIn::new(&bv),
                0.0,
                &mut cv,
                strategy,
            );
            assert_eq!(e, Err(Error::AliasedOutput));
        }
        assert!(c.iter().all(|&x| x == 9.0));
        let short = StridedView::new(&a, &[3, 2, 3], &[1, 3, 6], 0).unwrap();
        let mut c = vec![0.0; 16];
        let mut cv = StridedViewMut::new(&mut c, &[2, 2, 4], &[1, 2, 4], 0).unwrap();
        assert!(matches!(
            gemm_batched(
                &Exec::serial(),
                1.0,
                BatchIn::new(&av),
                BatchIn::new(&short),
                0.0,
                &mut cv,
                strategy
            ),
            Err(Error::Shape(_))
        ));
    }
}
