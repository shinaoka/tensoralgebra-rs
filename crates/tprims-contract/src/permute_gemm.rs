//! Permute plus batched GEMM: the approach of tenferro-rs's CPU
//! `dot_general` (`tenferro-cpu/src/{dot_runtime.rs, gemm/mod.rs}` at
//! `5a4e7fd`), reimplemented for strided views on `tprims-blas`.
//!
//! The index groups M (A, C), N (B, C), K (A, B) and H (A, B, C) are each
//! put in one order and fused per operand. When every operand fuses, the
//! contraction is one strided batched GEMM over the caller's memory. The
//! smallest set of operands that makes the rest fuse is copied once into
//! compact column-major buffers in canonical `[M, K, H]`, `[K, N, H]`,
//! `[M, N, H]` order.
use strided_basic::run_with_exec;
use strided_view::{StridedView, StridedViewMut};
use tensorcontract::Element;
use tprims_blas::{gemm_batched, BatchIn, BatchStrategy, Conj, Scalar};
use tprims_exec::Exec;

use crate::config::{DotGeneral, Shape};
use crate::{Error, Result};

/// One index group: per axis, the axis number in each of A, B, C (`None`
/// when the operand does not carry the group) and the extent.
#[derive(Clone, Debug)]
struct Group {
    axes: Vec<[Option<usize>; 3]>,
    extents: Vec<usize>,
}

/// The chosen order (permutation of the group's entries) and, per operand,
/// the fused (extent, stride) when it fuses in that order.
fn fuse(
    group: &Group,
    order: &[usize],
    strides: &[&[isize]; 3],
    op: usize,
) -> Option<(usize, isize)> {
    let mut ext = 1usize;
    let mut first: Option<isize> = None;
    let mut expect = 0isize;
    for &e in order {
        let d = group.extents[e];
        if d == 1 {
            continue;
        }
        let axis = group.axes[e][op]?;
        let s = strides[op][axis];
        match first {
            None => first = Some(s),
            Some(_) if s != expect => return None,
            Some(_) => {}
        }
        expect = s.checked_mul(d as isize)?;
        ext = ext.checked_mul(d)?;
    }
    Some((ext, first.unwrap_or(1)))
}

fn sorted_by(group: &Group, strides: &[&[isize]; 3], op: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..group.axes.len()).collect();
    order.sort_by_key(|&e| group.axes[e][op].map_or(0, |a| strides[op][a].unsigned_abs()));
    order
}

/// The plan: which operands are copied, the order of each group, and the
/// fused layout of the batched GEMM.
#[derive(Clone, Debug)]
pub(crate) struct PgPlan {
    pub materialized: [bool; 3],
    /// Axis order of each operand in its canonical compact layout.
    perms: [Vec<usize>; 3],
    /// Fused `(extent, stride)` per operand for the three batched-matrix
    /// axes (A: M K H, B: K N H, C: M N H), for the non-copied layout.
    fused: [[(usize, isize); 3]; 3],
}

fn groups(cfg: &DotGeneral, s: &Shape, a: &[usize], b: &[usize]) -> [Group; 4] {
    let nm = s.lhs_free.len();
    let nn = s.rhs_free.len();
    let m = Group {
        axes: s
            .lhs_free
            .iter()
            .enumerate()
            .map(|(i, &x)| [Some(x), None, Some(i)])
            .collect(),
        extents: s.lhs_free.iter().map(|&x| a[x]).collect(),
    };
    let n = Group {
        axes: s
            .rhs_free
            .iter()
            .enumerate()
            .map(|(i, &x)| [None, Some(x), Some(nm + i)])
            .collect(),
        extents: s.rhs_free.iter().map(|&x| b[x]).collect(),
    };
    let k = Group {
        axes: cfg
            .lhs_contract
            .iter()
            .zip(&cfg.rhs_contract)
            .map(|(&l, &r)| [Some(l), Some(r), None])
            .collect(),
        extents: cfg.lhs_contract.iter().map(|&x| a[x]).collect(),
    };
    let h = Group {
        axes: cfg
            .lhs_batch
            .iter()
            .zip(&cfg.rhs_batch)
            .enumerate()
            .map(|(i, (&l, &r))| [Some(l), Some(r), Some(nm + nn + i)])
            .collect(),
        extents: cfg.lhs_batch.iter().map(|&x| a[x]).collect(),
    };
    [m, n, k, h]
}

/// Which groups each operand carries, in its batched-matrix axis order.
const LAYOUT: [[usize; 3]; 3] = [[0, 2, 3], [2, 1, 3], [0, 1, 3]]; // A: M K H, B: K N H, C: M N H

pub(crate) fn plan(
    cfg: &DotGeneral,
    s: &Shape,
    dims: [&[usize]; 3],
    strides: [&[isize]; 3],
    no_materialize: bool,
) -> Result<PgPlan> {
    let gs = groups(cfg, s, dims[0], dims[1]);
    // An empty problem never reaches the GEMM (execute returns early).
    if dims.iter().any(|d| d.contains(&0)) {
        return Ok(PgPlan {
            materialized: [false; 3],
            perms: [vec![], vec![], vec![]],
            fused: [[(0, 1); 3]; 3],
        });
    }
    // INVARIANT: element counts were checked in ContractPlan::new.
    let sizes: [usize; 3] = [0, 1, 2].map(|o| dims[o].iter().product());
    // Candidate copy sets, cheapest first (C counts twice: it round-trips).
    let mut sets: Vec<[bool; 3]> = (0u8..8)
        .map(|m| [m & 1 != 0, m & 2 != 0, m & 4 != 0])
        .collect();
    sets.sort_by_key(|m| {
        (0..3)
            .filter(|&o| m[o])
            .map(|o| if o == 2 { 2 * sizes[o] } else { sizes[o] })
            .sum::<usize>()
    });
    for mat in sets {
        let mut orders: Vec<Vec<usize>> = Vec::with_capacity(4);
        let mut ok = true;
        for g in &gs {
            let carriers: Vec<usize> = (0..3)
                .filter(|&o| g.axes.iter().all(|a| a[o].is_some()))
                .collect();
            let live: Vec<usize> = carriers.iter().copied().filter(|&o| !mat[o]).collect();
            let mut cands: Vec<Vec<usize>> =
                live.iter().map(|&o| sorted_by(g, &strides, o)).collect();
            cands.push((0..g.axes.len()).collect());
            match cands
                .into_iter()
                .find(|ord| live.iter().all(|&o| fuse(g, ord, &strides, o).is_some()))
            {
                Some(ord) => orders.push(ord),
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok {
            continue;
        }
        if no_materialize && mat.iter().any(|&m| m) {
            return Err(Error::WouldMaterialize { operands: mat });
        }
        let (gs_ref, orders_ref) = (&gs, &orders);
        let perms: [Vec<usize>; 3] = [0, 1, 2].map(|o| {
            let mut perm = Vec::new();
            for &g in &LAYOUT[o] {
                for &e in &orders_ref[g] {
                    if let Some(axis) = gs_ref[g].axes[e][o] {
                        perm.push(axis);
                    }
                }
            }
            perm
        });
        let fused: [[(usize, isize); 3]; 3] = [0, 1, 2].map(|o| {
            LAYOUT[o].map(|g| {
                if mat[o] {
                    // Placeholder; compact strides are computed at execution.
                    (gs[g].extents.iter().product(), 0)
                } else {
                    fuse(&gs[g], &orders[g], &strides, o).unwrap_or((1, 1))
                }
            })
        });
        return Ok(PgPlan {
            materialized: mat,
            perms,
            fused,
        });
    }
    Err(Error::Backend(
        "no fusable layout (unreachable: copying all operands always fuses)".into(),
    ))
}

/// Compact column-major fused strides for extents `[e0, e1, e2]`.
fn compact(e: [usize; 3]) -> [(usize, isize); 3] {
    let s1 = e[0].max(1);
    let s2 = s1 * e[1].max(1);
    [(e[0], 1), (e[1], s1 as isize), (e[2], s2 as isize)]
}

fn permuted<'a, T: Scalar>(v: &StridedView<'a, T>, perm: &[usize]) -> Result<StridedView<'a, T>> {
    v.permute(perm).map_err(|e| Error::Backend(e.to_string()))
}

fn batch_view<'a, T>(
    data: &'a [T],
    offset: isize,
    f: &[(usize, isize); 3],
) -> Result<StridedView<'a, T>> {
    StridedView::new(
        data,
        &[f[0].0, f[1].0, f[2].0],
        &[f[0].1, f[1].1, f[2].1],
        offset,
    )
    .map_err(|e| Error::Backend(e.to_string()))
}

/// Execute a validated, non-empty, alpha != 0 problem.
#[allow(clippy::too_many_arguments)] // INVARIANT: the contraction argument set.
pub(crate) fn execute<T: Scalar>(
    p: &PgPlan,
    exec: &Exec<'_>,
    alpha: T,
    a: &StridedView<'_, T>,
    ca: Conj,
    b: &StridedView<'_, T>,
    cb: Conj,
    beta: T,
    c: &mut StridedViewMut<'_, T>,
) -> Result<()> {
    let abuf = if p.materialized[0] {
        Some(copy_compact(exec, a, &p.perms[0])?)
    } else {
        None
    };
    let bbuf = if p.materialized[1] {
        Some(copy_compact(exec, b, &p.perms[1])?)
    } else {
        None
    };
    let fa = if p.materialized[0] {
        compact(p.fused[0].map(|x| x.0))
    } else {
        p.fused[0]
    };
    let fb = if p.materialized[1] {
        compact(p.fused[1].map(|x| x.0))
    } else {
        p.fused[1]
    };
    let fc = if p.materialized[2] {
        compact(p.fused[2].map(|x| x.0))
    } else {
        p.fused[2]
    };
    let av = match &abuf {
        Some(buf) => batch_view(buf, 0, &fa)?,
        None => batch_view(a.data(), a.offset(), &fa)?,
    };
    let bv = match &bbuf {
        Some(buf) => batch_view(buf, 0, &fb)?,
        None => batch_view(b.data(), b.offset(), &fb)?,
    };
    let (ai, bi) = (
        with_conj(BatchIn::new(&av), ca),
        with_conj(BatchIn::new(&bv), cb),
    );
    let blas = |e: tprims_blas::Error| Error::Backend(e.to_string());
    if p.materialized[2] {
        let cview = c.as_view();
        let mut cbuf = if beta == <T as Element>::zero() {
            vec![<T as Element>::zero(); cview.len()]
        } else {
            copy_compact(exec, &cview, &p.perms[2])?
        };
        {
            let mut cv = StridedViewMut::new(
                &mut cbuf,
                &[fc[0].0, fc[1].0, fc[2].0],
                &[fc[0].1, fc[1].1, fc[2].1],
                0,
            )
            .map_err(|e| Error::Backend(e.to_string()))?;
            gemm_batched(exec, alpha, ai, bi, beta, &mut cv, BatchStrategy::FaerLoop)
                .map_err(blas)?;
        }
        copy_back(exec, &cbuf, c, &p.perms[2])?;
    } else {
        let off = c.offset();
        let mut cv = StridedViewMut::new(
            c.data_mut(),
            &[fc[0].0, fc[1].0, fc[2].0],
            &[fc[0].1, fc[1].1, fc[2].1],
            off,
        )
        .map_err(|e| Error::Backend(e.to_string()))?;
        gemm_batched(exec, alpha, ai, bi, beta, &mut cv, BatchStrategy::FaerLoop).map_err(blas)?;
    }
    Ok(())
}

fn with_conj<'v, 'a, T>(b: BatchIn<'v, 'a, T>, c: Conj) -> BatchIn<'v, 'a, T> {
    if c == Conj::Yes {
        b.conj()
    } else {
        b
    }
}

/// Copy `src` with its axes permuted by `perm` into a new compact
/// column-major buffer.
fn copy_compact<T: Scalar>(
    exec: &Exec<'_>,
    src: &StridedView<'_, T>,
    perm: &[usize],
) -> Result<Vec<T>> {
    let sp = permuted(src, perm)?;
    let dims = sp.dims().to_vec();
    let len: usize = dims.iter().product();
    let mut buf = vec![<T as Element>::zero(); len];
    let strides = col_major(&dims);
    {
        let mut dst = StridedViewMut::new(&mut buf, &dims, &strides, 0)
            .map_err(|e| Error::Backend(e.to_string()))?;
        run_with_exec(exec, len, |_| strided_basic::copy_into(&mut dst, &sp))
            .map_err(|e| Error::Backend(e.to_string()))?;
    }
    Ok(buf)
}

/// Copy the compact buffer back into `c` (axes of `c` permuted by `perm`).
fn copy_back<T: Scalar>(
    exec: &Exec<'_>,
    buf: &[T],
    c: &mut StridedViewMut<'_, T>,
    perm: &[usize],
) -> Result<()> {
    let dims: Vec<usize> = perm.iter().map(|&x| c.dims()[x]).collect();
    let strides_c: Vec<isize> = perm.iter().map(|&x| c.strides()[x]).collect();
    let off = c.offset();
    let src: StridedView<'_, T> = StridedView::new(buf, &dims, &col_major(&dims), 0)
        .map_err(|e| Error::Backend(e.to_string()))?;
    let len = buf.len();
    let mut dst = StridedViewMut::new(c.data_mut(), &dims, &strides_c, off)
        .map_err(|e| Error::Backend(e.to_string()))?;
    run_with_exec(exec, len, |_| strided_basic::copy_into(&mut dst, &src))
        .map_err(|e| Error::Backend(e.to_string()))
}

fn col_major(dims: &[usize]) -> Vec<isize> {
    let mut s = Vec::with_capacity(dims.len());
    let mut acc = 1isize;
    for &d in dims {
        s.push(acc);
        acc *= d.max(1) as isize;
    }
    s
}
