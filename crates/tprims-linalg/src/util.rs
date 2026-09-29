use strided_view::StridedViewMut;
use tprims_blas::{is_injective_layout, GemmPolicy, Scalar};
use tprims_exec::{Exec, Par, WidthPolicy};

use crate::{Error, Result};

/// Run `op` with a faer parallelism chosen from an estimated real-flop count.
pub(crate) fn install<R: Send>(
    exec: &Exec<'_>,
    flops: f64,
    op: impl FnOnce(faer::Par) -> R + Send,
) -> R {
    install_with(exec, flops, GemmPolicy::default().width, op)
}

/// Serial threshold for faer's blocked LU and QR: measured on an EPYC 7713P
/// (2026-09-30, `benchmarks/benchmarks/tprims/linalg`), their parallel paths
/// are slower than serial at n = 128 (0.6x LU, 0.4x QR at 4 threads) and gain
/// at most 1.3x at n = 512, so they stay serial below an estimated 1 ms.
pub(crate) fn factor_policy() -> WidthPolicy {
    WidthPolicy {
        serial_below_ns: 1.0e6,
        ..WidthPolicy::default()
    }
}

/// [`install`] with a kernel-specific width policy.
pub(crate) fn install_with<R: Send>(
    exec: &Exec<'_>,
    flops: f64,
    policy: WidthPolicy,
    op: impl FnOnce(faer::Par) -> R + Send,
) -> R {
    let p = GemmPolicy::default();
    let k = exec.width_for(flops * p.ns_per_flop, &policy);
    exec.install(k, |par| {
        op(match par {
            Par::Seq => faer::Par::Seq,
            Par::Threads(n) => faer::Par::rayon(n.get()),
        })
    })
}

/// Complex arithmetic costs about four real flops per real flop.
pub(crate) fn flops<T: Scalar>(real: f64) -> f64 {
    if T::IS_COMPLEX_SCALAR {
        4.0 * real
    } else {
        real
    }
}

/// A validated right-hand side: rank 2, `rows` rows, injective.
pub(crate) fn rhs_mut<'v, T: Scalar>(
    b: &'v mut StridedViewMut<'_, T>,
    rows: usize,
) -> Result<faer::MatMut<'v, T>> {
    if b.ndim() != 2 {
        return Err(Error::Rank {
            operand: "B",
            expected: 2,
            got: b.ndim(),
        });
    }
    let (d, s) = (b.dims(), b.strides());
    if d[0] != rows {
        return Err(Error::Shape(format!(
            "B has {} rows, expected {rows}",
            d[0]
        )));
    }
    if !is_injective_layout(&[(d[0], s[0]), (d[1], s[1])]) {
        return Err(Error::AliasedOutput);
    }
    let norm = |e: usize, st: isize| if e <= 1 { 1 } else { st };
    let (m, n, rs, cs) = (d[0], d[1], norm(d[0], s[0]), norm(d[1], s[1]));
    // SAFETY: the view was bounds-checked at construction, is exclusively
    // borrowed for 'v, and is injective (checked above).
    Ok(unsafe { faer::MatMut::from_raw_parts_mut(b.as_mut_ptr(), m, n, rs, cs) })
}

/// Square check.
pub(crate) fn square(rows: usize, cols: usize) -> Result<usize> {
    if rows == cols {
        Ok(rows)
    } else {
        Err(Error::NotSquare { rows, cols })
    }
}

/// |x| in f64.
pub(crate) fn abs<T: Scalar>(x: T) -> f64 {
    use tensorcontract::{Element, Real};
    let (re, im) = (Element::re(x).to_f64(), Element::im(x).to_f64());
    re.hypot(im)
}

/// Machine epsilon of `T`'s real type.
pub(crate) fn eps<T: Scalar>() -> f64 {
    if std::mem::size_of::<T::Re>() == 4 {
        f32::EPSILON as f64
    } else {
        f64::EPSILON
    }
}
