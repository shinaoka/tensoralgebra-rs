//! Grouped GEMM: independent jobs of different sizes over shared buffers.

use tprims_exec::{Exec, Par};

use crate::gemm::{check_gemm, gemm_raw, gemm_width, GemmShape, SendConst, SendMut};
use crate::operand::{scale_in_place, Mat2};
use crate::{Conj, Error, Result, Scalar, Selected};

/// One job of [`gemm_grouped`]: `C = alpha * op(A) * op(B) + beta * C` on
/// compact column-major blocks at element offsets into the shared buffers.
/// `A` is `rows x inner`, `B` is `inner x cols`, `C` is `rows x cols`.
///
/// The layout matches tenferro's `GroupedGemmJob`.
///
/// # Examples
///
/// ```
/// let j = tprims_blas::GroupedJob { a_offset: 0, b_offset: 0, c_offset: 0, rows: 2, inner: 3, cols: 4 };
/// assert_eq!(j.rows * j.cols, 8);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GroupedJob {
    /// First element of the `A` block.
    pub a_offset: usize,
    /// First element of the `B` block.
    pub b_offset: usize,
    /// First element of the `C` block.
    pub c_offset: usize,
    /// Rows of `A` and `C`.
    pub rows: usize,
    /// Columns of `A`, rows of `B`.
    pub inner: usize,
    /// Columns of `B` and `C`.
    pub cols: usize,
}

impl GroupedJob {
    pub(crate) fn shape(&self) -> Result<GemmShape> {
        let mat = |rows: usize, cols: usize| Mat2 {
            rows,
            cols,
            rs: 1,
            cs: rows.max(1) as isize,
        };
        check_gemm(
            mat(self.rows, self.inner),
            mat(self.inner, self.cols),
            mat(self.rows, self.cols),
        )
    }
}

/// `offset + rows * cols <= len`, with checked arithmetic.
fn block_fits(what: &str, offset: usize, rows: usize, cols: usize, len: usize) -> Result<()> {
    let end = rows
        .checked_mul(cols)
        .and_then(|n| n.checked_add(offset))
        .ok_or_else(|| Error::Shape(format!("{what} block extent overflows")))?;
    if end > len {
        return Err(Error::Shape(format!(
            "{what} block {offset}..{end} exceeds the buffer of {len} elements"
        )));
    }
    Ok(())
}

/// Every job's blocks lie inside their buffers and the non-empty output
/// blocks are pairwise disjoint.
pub(crate) fn check_jobs<T>(a: &[T], b: &[T], c: &[T], jobs: &[GroupedJob]) -> Result<()> {
    let mut outs = Vec::with_capacity(jobs.len());
    for j in jobs {
        j.shape()?;
        if j.rows == 0 || j.cols == 0 {
            continue;
        }
        block_fits("C", j.c_offset, j.rows, j.cols, c.len())?;
        if j.inner > 0 {
            block_fits("A", j.a_offset, j.rows, j.inner, a.len())?;
            block_fits("B", j.b_offset, j.inner, j.cols, b.len())?;
        }
        outs.push((j.c_offset, j.c_offset + j.rows * j.cols));
    }
    outs.sort_unstable();
    if outs.windows(2).any(|w| w[1].0 < w[0].1) {
        return Err(Error::AliasedOutput);
    }
    Ok(())
}

/// Independent GEMMs of different sizes, `C_j = alpha * op(A_j) * op(B_j) +
/// beta * C_j`, on compact column-major blocks of shared buffers.
///
/// Many jobs that are each too small for inner parallelism are spread over
/// the pool with each job serial; otherwise the jobs run one after another,
/// each with the width its size warrants. A job with `inner == 0` scales its
/// `C` block by `beta`; a job with no output elements touches nothing.
/// Returns the schedule that ran ([`Selected::FaerLoop`]).
///
/// # Errors
///
/// [`Error::Shape`] when a block lies outside its buffer or an extent
/// overflows; [`Error::AliasedOutput`] when output blocks overlap. Nothing is
/// written on error.
///
/// # Examples
///
/// ```
/// use tprims_blas::{gemm_grouped, Conj, GroupedJob};
/// let a = [1.0, 2.0, 3.0, 4.0]; // one 2x2 block, column-major
/// let b = [1.0, 0.0, 0.0, 1.0]; // identity
/// let mut c = [0.0; 4];
/// let job = GroupedJob { a_offset: 0, b_offset: 0, c_offset: 0, rows: 2, inner: 2, cols: 2 };
/// gemm_grouped(&tprims_exec::Exec::serial(), 1.0, &a, Conj::No, &b, Conj::No, 0.0, &mut c, &[job]).unwrap();
/// assert_eq!(c, a);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn gemm_grouped<T: Scalar>(
    exec: &Exec<'_>,
    alpha: T,
    a: &[T],
    ca: Conj,
    b: &[T],
    cb: Conj,
    beta: T,
    c: &mut [T],
    jobs: &[GroupedJob],
) -> Result<Selected> {
    gemm_grouped_with(
        exec,
        &crate::GemmConfig::default(),
        alpha,
        a,
        ca,
        b,
        cb,
        beta,
        c,
        jobs,
    )
    .map(|sel| sel.batched.expect("the grouped report is always set"))
}

/// [`gemm_grouped`], with a configuration and a full report.
///
/// The grouped path computes with faer, so a configuration that asks for
/// another engine or a named kernel is refused rather than quietly ignored.
///
/// # Errors
///
/// As [`gemm_grouped`], plus [`crate::Error::Select`] for such a configuration.
#[allow(clippy::too_many_arguments)] // INVARIANT: the grouped GEMM argument set.
pub fn gemm_grouped_with<T: Scalar>(
    exec: &Exec<'_>,
    cfg: &crate::GemmConfig,
    alpha: T,
    a: &[T],
    ca: Conj,
    b: &[T],
    cb: Conj,
    beta: T,
    c: &mut [T],
    jobs: &[GroupedJob],
) -> Result<crate::SelectedGemm> {
    let unsupported = match cfg.engine {
        crate::EngineChoice::Auto | crate::EngineChoice::Faer => {
            cfg.kernel != tprims_gemm_kernel::KernelChoice::Auto || cfg.method.is_some()
        }
        crate::EngineChoice::PrivateGemmX86 | crate::EngineChoice::Packed => true,
    };
    if unsupported {
        return Err(Error::Select(
            tprims_gemm_kernel::SelectError::EngineUnsupported {
                engine: "grouped gemm",
                reason: "the grouped path computes with faer",
            },
        ));
    }
    let report = |selected: Selected| crate::SelectedGemm {
        engine: crate::Engine::Faer,
        family_id: None,
        complex: None,
        mr: 0,
        nr: 0,
        mc: 0,
        nc: 0,
        kc: 0,
        partition: tprims_gemm_kernel::PartitionPolicy::default(),
        batched: Some(selected),
        origin: None,
    };
    check_jobs(a, b, c, jobs)?;
    let live: Vec<(GroupedJob, GemmShape, usize)> = jobs
        .iter()
        .filter(|j| j.rows > 0 && j.cols > 0)
        .map(|j| {
            let s = j.shape().expect("validated");
            (*j, s, gemm_width::<T>(exec, j.rows, j.cols, j.inner))
        })
        .collect();
    let (ap, bp, cp) = (
        SendConst(a.as_ptr()),
        SendConst(b.as_ptr()),
        SendMut(c.as_mut_ptr()),
    );
    let run = |i: usize, par: faer::Par| {
        let (ap, bp, cp) = (ap, bp, cp);
        let (j, s, _) = &live[i];
        // SAFETY: check_jobs proved every referenced block lies inside its
        // buffer and output blocks are disjoint, so jobs on different threads
        // write disjoint elements; `c` is borrowed exclusively.
        unsafe {
            let cj = cp.0.add(j.c_offset);
            if j.inner == 0 || alpha == crate::scalar::zero() {
                scale_in_place(cj, &s.c, beta);
            } else {
                gemm_raw(
                    s,
                    alpha,
                    ap.0.add(j.a_offset),
                    ca,
                    bp.0.add(j.b_offset),
                    cb,
                    beta,
                    cj,
                    par,
                );
            }
        }
    };
    let widest = live.iter().map(|l| l.2).max().unwrap_or(1);
    let work: usize = live
        .iter()
        .map(|(j, _, _)| j.rows.saturating_mul(j.cols).saturating_mul(j.inner.max(1)))
        .fold(0, usize::saturating_add);
    let total = gemm_width::<T>(exec, 1, 1, work);
    if widest == 1 && total > 1 && live.len() > 1 {
        let lanes = exec.with_budget(total.min(live.len())).unwrap_or(*exec);
        lanes.for_each_partition(live.len(), &|i| run(i, faer::Par::Seq));
        return Ok(report(Selected::FaerLoop {
            outer_parallel: true,
        }));
    }
    exec.install(widest, |par: Par| {
        for (i, l) in live.iter().enumerate() {
            let w = l.2.min(par.threads());
            run(
                i,
                if w > 1 {
                    faer::Par::rayon(w)
                } else {
                    faer::Par::Seq
                },
            );
        }
    });
    Ok(report(Selected::FaerLoop {
        outer_parallel: false,
    }))
}
