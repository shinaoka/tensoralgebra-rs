//! TTGT baseline: Transpose-Transpose-GEMM-Transpose.
//!
//! The classical way to do a tensor contraction: permute `A` into a dense
//! `M x K` matrix, `B` into `K x N`, call GEMM, and permute the `M x N` result
//! into the output tensor's layout.
//!
//! This implementation deliberately reuses `tensorcontract`'s *index analysis*
//! (via [`Plan::scatters`]) so that TTGT and the block-scatter engine see the
//! identical M/N/K decomposition and index ordering. The only thing that
//! differs is the execution strategy. That makes the comparison a clean
//! measurement of "materialise a transposed copy, then call a vendor GEMM"
//! versus "fuse the transposition into packing".
//!
//! The permutation itself writes its output contiguously and gathers its
//! input, which is the sensible naive strategy. A production TTGT would use a
//! blocked/vectorised transpose such as HPTT; treat these numbers as an upper
//! bound on TTGT's transposition cost, and TBLIS as the strong baseline.

#![allow(dead_code)] // only reachable with the `blas` feature

use tensorcontract::{Element, Plan};

use crate::blas::GemmScalar;

/// Scratch buffers, reused across repetitions so the benchmark measures the
/// algorithm rather than the allocator.
pub struct TtgtScratch<T> {
    pub am: Vec<T>,
    pub bm: Vec<T>,
    pub cm: Vec<T>,
}

impl<T: Element> TtgtScratch<T> {
    pub fn new(plan: &Plan) -> Self {
        let s = &plan.stats;
        TtgtScratch {
            am: vec![T::zero(); s.m * s.k],
            bm: vec![T::zero(); s.k * s.n],
            cm: vec![T::zero(); s.m * s.n],
        }
    }
}

/// `D = alpha * A * B + beta * C` via TTGT. Batch (Hadamard) indices are
/// handled by looping over them, as a TTGT implementation would.
///
/// # Safety
/// Slices must be large enough for every offset the plan generates.
#[allow(clippy::too_many_arguments)]
pub fn ttgt<T>(
    plan: &Plan,
    alpha: T,
    a: &[T],
    b: &[T],
    beta: T,
    c: &[T],
    d: &mut [T],
    scratch: &mut TtgtScratch<T>,
) where
    T: Element + GemmScalar,
{
    let sc = plan.scatters();
    let (m, n, k) = (plan.stats.m, plan.stats.n, plan.stats.k);
    if m == 0 || n == 0 {
        return;
    }

    for h in 0..plan.stats.batch {
        let (oa, ob, oc, od) = (
            sc.h_a[h] as usize,
            sc.h_b[h] as usize,
            sc.h_c[h] as usize,
            sc.h_d[h] as usize,
        );

        // A -> column-major M x K
        for (p, &kp) in sc.a_k.iter().enumerate() {
            let dst = &mut scratch.am[p * m..(p + 1) * m];
            for (i, &mi) in sc.a_m.iter().enumerate() {
                dst[i] = a[oa + (mi + kp) as usize];
            }
        }
        // B -> column-major K x N
        for (j, &nj) in sc.b_n.iter().enumerate() {
            let dst = &mut scratch.bm[j * k..(j + 1) * k];
            for (p, &kp) in sc.b_k.iter().enumerate() {
                dst[p] = b[ob + (kp + nj) as usize];
            }
        }

        if k > 0 {
            // SAFETY: buffers are sized m*k, k*n, m*n by `TtgtScratch::new`.
            unsafe {
                T::gemm(
                    m,
                    n,
                    k,
                    scratch.am.as_ptr(),
                    scratch.bm.as_ptr(),
                    scratch.cm.as_mut_ptr(),
                );
            }
        } else {
            scratch.cm.iter_mut().for_each(|x| *x = T::zero());
        }

        // M x N -> D
        let beta_zero = beta == T::zero();
        for (j, (&dn, &cn)) in sc.d_n.iter().zip(sc.c_n).enumerate() {
            for (i, (&dm, &cm)) in sc.d_m.iter().zip(sc.c_m).enumerate() {
                let mut v = alpha.mul(scratch.cm[j * m + i]);
                if !beta_zero {
                    v = v.add(beta.mul(c[oc + (cm + cn) as usize]));
                }
                d[od + (dm + dn) as usize] = v;
            }
        }
    }
}
