//! A deliberately naive reference implementation, used as the test oracle.
//!
//! It shares no code with the engine: it walks the raw layouts and label lists
//! directly, so it independently defines the semantics of repeated labels
//! (diagonals), isolated labels (reductions), Hadamard labels, conjugation and
//! general strides.

use crate::element::Element;
use crate::error::{Error, Result};
use crate::layout::Layout;
use crate::plan::ElementOp;

/// Operand for the reference implementation.
///
/// Structurally the same as [`crate::TensorView`], and deliberately a separate
/// type: the oracle must not be able to borrow any of the engine's
/// convenience, or it would stop being an independent statement of the
/// semantics.
pub struct RefOperand<'a, T> {
    /// The backing allocation.
    pub data: &'a [T],
    /// Extents and strides, in elements.
    pub layout: &'a Layout,
    /// One index label per mode of `layout`.
    pub idx: &'a [i64],
    /// Element-wise operation applied while reading.
    pub op: ElementOp,
}

/// `D = op_D(alpha * op_A(A) * op_B(B) + beta * op_C(C))`, computed by brute
/// force over every index assignment.
///
/// Exported so that a downstream [`crate::element::Element`] implementation, or
/// a caller who has hit a surprising result, can check the engine against the
/// definition without reimplementing it. It walks the full index space with a
/// linear label search per access, so it is several orders of magnitude slower
/// than the engine: fine for the shapes a test uses, hopeless above a few
/// million output elements.
#[allow(clippy::too_many_arguments)]
pub fn contract_reference<T: Element>(
    alpha: T,
    a: &RefOperand<'_, T>,
    b: &RefOperand<'_, T>,
    beta: T,
    c: Option<&RefOperand<'_, T>>,
    d_data: &mut [T],
    d_layout: &Layout,
    idx_d: &[i64],
    op_d: ElementOp,
) -> Result<()> {
    // Collect label extents, checking consistency.
    let mut labels: Vec<(i64, i64)> = Vec::new();
    let mut add = |idx: &[i64], layout: &Layout, name: &'static str| -> Result<()> {
        if idx.len() != layout.ndim() {
            return Err(Error::LabelCountMismatch {
                tensor: name,
                nmode: layout.ndim(),
                nlabel: idx.len(),
            });
        }
        for (k, &l) in idx.iter().enumerate() {
            let e = layout.extents[k];
            match labels.iter().find(|x| x.0 == l) {
                Some(&(_, e0)) if e0 != e => {
                    return Err(Error::ExtentMismatch {
                        label: l,
                        expected: e0,
                        found: e,
                    })
                }
                Some(_) => {}
                None => labels.push((l, e)),
            }
        }
        Ok(())
    };
    add(a.idx, a.layout, "A")?;
    add(b.idx, b.layout, "B")?;
    if let Some(c) = c {
        add(c.idx, c.layout, "C")?;
    }
    add(idx_d, d_layout, "D")?;

    // Split into output labels (those in D) and contracted labels.
    let out: Vec<(i64, i64)> = labels
        .iter()
        .copied()
        .filter(|(l, _)| idx_d.contains(l))
        .collect();
    let sum: Vec<(i64, i64)> = labels
        .iter()
        .copied()
        .filter(|(l, _)| !idx_d.contains(l))
        .collect();

    let offset = |idx: &[i64], layout: &Layout, assign: &dyn Fn(i64) -> i64| -> i64 {
        idx.iter()
            .enumerate()
            .map(|(k, &l)| assign(l) * layout.strides[k])
            .sum()
    };

    let out_total: i64 = out.iter().map(|x| x.1).product();
    let sum_total: i64 = sum.iter().map(|x| x.1).product();

    let mut ovals = vec![0i64; out.len()];
    for _ in 0..out_total {
        let lookup_out = |l: i64| -> i64 {
            out.iter()
                .position(|x| x.0 == l)
                .map(|p| ovals[p])
                .unwrap_or(0)
        };

        let mut acc = T::zero();
        let mut svals = vec![0i64; sum.len()];
        for _ in 0..sum_total {
            let assign = |l: i64| -> i64 {
                if let Some(p) = out.iter().position(|x| x.0 == l) {
                    ovals[p]
                } else if let Some(p) = sum.iter().position(|x| x.0 == l) {
                    svals[p]
                } else {
                    0
                }
            };
            let av = a.data[offset(a.idx, a.layout, &assign) as usize];
            let bv = b.data[offset(b.idx, b.layout, &assign) as usize];
            let av = if a.op.is_conj() { av.conj() } else { av };
            let bv = if b.op.is_conj() { bv.conj() } else { bv };
            acc = acc.add(av.mul(bv));
            odometer(&mut svals, &sum);
        }

        let mut v = alpha.mul(acc);
        if beta != T::zero() {
            if let Some(c) = c {
                let cv = c.data[offset(c.idx, c.layout, &lookup_out) as usize];
                let cv = if c.op.is_conj() { cv.conj() } else { cv };
                v = v.add(beta.mul(cv));
            }
        }
        if op_d.is_conj() {
            v = v.conj();
        }
        d_data[offset(idx_d, d_layout, &lookup_out) as usize] = v;

        odometer(&mut ovals, &out);
    }
    Ok(())
}

fn odometer(vals: &mut [i64], dims: &[(i64, i64)]) {
    for (v, &(_, e)) in vals.iter_mut().zip(dims) {
        *v += 1;
        if *v < e {
            return;
        }
        *v = 0;
    }
}
