use strided_view::{StridedView, StridedViewMut};
use tprims_exec::{Exec, Par};

use crate::gemm::{check_gemm, gemm_raw, gemm_width, to_faer, GemmShape, SendConst, SendMut};
use crate::operand::{check_injective, mat2};
use crate::{Conj, Error, Result, Scalar};

/// A rank-3 batched input `[rows, cols, batch]` plus a conjugation flag.
///
/// # Examples
///
/// ```
/// let d = [0.0f64; 8];
/// let v = strided_view::StridedView::new(&d, &[2, 2, 2], &[1, 2, 4], 0).unwrap();
/// assert_eq!(tprims_blas::BatchIn::new(&v).conj, tprims_blas::Conj::No);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct BatchIn<'v, 'a, T> {
    /// The borrowed view, batch axis last.
    pub view: &'v StridedView<'a, T>,
    /// Conjugation applied on read.
    pub conj: Conj,
}

impl<'v, 'a, T> BatchIn<'v, 'a, T> {
    /// Use `view` as stored.
    pub fn new(view: &'v StridedView<'a, T>) -> Self {
        Self {
            view,
            conj: Conj::No,
        }
    }

    /// Conjugate on read.
    pub fn conj(mut self) -> Self {
        self.conj = Conj::Yes;
        self
    }
}

/// Which implementation a batched GEMM uses.
///
/// # Examples
///
/// ```
/// assert_ne!(tprims_blas::BatchStrategy::FaerLoop, tprims_blas::BatchStrategy::Tblis);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BatchStrategy {
    /// Let the library choose (currently [`BatchStrategy::FaerLoop`]; the rule
    /// is replaced once the Phase 1b comparison is recorded).
    Auto,
    /// faer per item, in a loop over items.
    FaerLoop,
    /// TBLIS-style direct kernel (tensorcontract, by Lukas Devos).
    Tblis,
}

/// The implementation and schedule a batched GEMM actually ran.
///
/// # Examples
///
/// ```
/// let s = tprims_blas::Selected::FaerLoop { outer_parallel: false };
/// assert!(matches!(s, tprims_blas::Selected::FaerLoop { .. }));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Selected {
    /// faer per item; `outer_parallel` when items were spread over the pool.
    FaerLoop {
        /// Items ran in parallel, each serially.
        outer_parallel: bool,
    },
    /// TBLIS-style; `outer_parallel` as above.
    Tblis {
        /// Items ran in parallel, each serially.
        outer_parallel: bool,
    },
}

/// Validated batched problem: per-item shape plus batch extent and strides.
pub(crate) struct Batch {
    pub item: GemmShape,
    pub count: usize,
    pub sa: isize,
    pub sb: isize,
    pub sc: isize,
}

fn split3<T>(name: &'static str, v: &[usize], s: &[isize]) -> Result<(usize, isize)> {
    let _ = std::marker::PhantomData::<T>;
    if v.len() != 3 {
        return Err(Error::Rank {
            operand: name,
            expected: 3,
            got: v.len(),
        });
    }
    Ok((v[2], s[2]))
}

pub(crate) fn check_batch<T>(
    a: &StridedView<'_, T>,
    b: &StridedView<'_, T>,
    c: &StridedViewMut<'_, T>,
) -> Result<Batch> {
    let (na, sa) = split3::<T>("A", a.dims(), a.strides())?;
    let (nb, sb) = split3::<T>("B", b.dims(), b.strides())?;
    let (nc, sc) = split3::<T>("C", c.dims(), c.strides())?;
    if na != nb || na != nc {
        return Err(Error::Shape(format!("batch extents {na}, {nb}, {nc}")));
    }
    let item = check_gemm(
        mat2("A", &a.dims()[..2], &a.strides()[..2])?,
        mat2("B", &b.dims()[..2], &b.strides()[..2])?,
        mat2("C", &c.dims()[..2], &c.strides()[..2])?,
    )?;
    let (cd, cs) = (c.dims(), c.strides());
    check_injective(&[(cd[0], cs[0]), (cd[1], cs[1]), (cd[2], cs[2])])?;
    Ok(Batch {
        item,
        count: na,
        sa,
        sb,
        sc,
    })
}

/// How to schedule a batch: `None` = serial loop on the caller;
/// `Some(k)` with `outer` = partition items over `k` workers, each serial;
/// otherwise items one after another with inner width `inner`.
pub(crate) struct Schedule {
    pub outer: Option<usize>,
    pub inner: usize,
}

pub(crate) fn schedule<T: Scalar>(exec: &Exec<'_>, s: &Batch) -> Schedule {
    let (m, n, k) = (s.item.c.rows, s.item.c.cols, s.item.a.cols);
    let inner = gemm_width::<T>(exec, m, n, k);
    let total = gemm_width::<T>(exec, m, n, k.saturating_mul(s.count));
    if inner == 1 && total > 1 && s.count > 1 {
        Schedule {
            outer: Some(total.min(s.count)),
            inner: 1,
        }
    } else {
        Schedule { outer: None, inner }
    }
}

/// Batched `C[:, :, i] = alpha * op(A[:, :, i]) * op(B[:, :, i]) + beta * C[:, :, i]`.
///
/// `strategy` selects faer-per-item or the TBLIS-style kernel; the returned
/// [`Selected`] reports what ran. Small items over a large batch are spread
/// over the pool with each item serial; large items use inner parallelism.
///
/// # Errors
///
/// As [`crate::gemm`], plus [`Error::Shape`] for unequal batch extents and
/// [`Error::AliasedOutput`] when output items overlap.
pub fn gemm_batched<T: Scalar>(
    exec: &Exec<'_>,
    alpha: T,
    a: BatchIn<'_, '_, T>,
    b: BatchIn<'_, '_, T>,
    beta: T,
    c: &mut StridedViewMut<'_, T>,
    strategy: BatchStrategy,
) -> Result<Selected> {
    let s = check_batch(a.view, b.view, c)?;
    let sched = schedule::<T>(exec, &s);
    let (ap, bp, cp) = (
        SendConst(a.view.ptr()),
        SendConst(b.view.ptr()),
        SendMut(c.as_mut_ptr()),
    );
    match strategy {
        BatchStrategy::Tblis => {
            crate::tblis::run(exec, &s, &sched, alpha, a.conj, ap, b.conj, bp, beta, cp)
        }
        BatchStrategy::Auto | BatchStrategy::FaerLoop => {
            let (ca, cb) = (a.conj, b.conj);
            let item = |i: usize, par: faer::Par| {
                let (ap, bp, cp) = (ap, bp, cp);
                let i = i as isize;
                // SAFETY: offsets `i * stride` stay inside the bounds-checked
                // views (i < count); output items are disjoint (check_batch).
                unsafe {
                    gemm_raw(
                        &s.item,
                        alpha,
                        ap.0.offset(i * s.sa),
                        ca,
                        bp.0.offset(i * s.sb),
                        cb,
                        beta,
                        cp.0.offset(i * s.sc),
                        par,
                    )
                }
            };
            match sched.outer {
                Some(k) => {
                    let lanes = exec.with_budget(k).unwrap_or(*exec);
                    lanes.for_each_partition(s.count, &|i| item(i, faer::Par::Seq));
                }
                None => {
                    for i in 0..s.count {
                        exec.install(sched.inner, |par: Par| item(i, to_faer(par)));
                    }
                }
            }
            Ok(Selected::FaerLoop {
                outer_parallel: sched.outer.is_some(),
            })
        }
    }
}
