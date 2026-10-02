//! The micro-tile half of the loop nest: packing of one epoch's panels, the two
//! innermost loops with their direct-path branches and first-slab rule, and the
//! write-back call. Shared by the static and the dynamic scheduler so there is one
//! copy of the arithmetic.

use tprims_kernel::scatter::IRREGULAR;
use tprims_kernel::{Element, Real, UkrAux, UkrFn};

use super::Ctx;

// Common write-back call for the first-K and accumulation paths. Numerical
// loops stay in the kernel layer; foreign types retain the legacy entry.
unsafe fn emit_tile<T: Element>(
    cx: &Ctx<'_, T>,
    ab: *const T::Real,
    mrem: usize,
    nrem: usize,
    beta: T,
    c: *const T,
    cr: &[i64],
    cc: &[i64],
    crs: i64,
    conj_c: bool,
    d: *mut T,
    dr: &[i64],
    dc: &[i64],
    drs: i64,
    conj_d: bool,
) {
    // SAFETY: run_strip supplies the same checked live tile/scatters and
    // initialized accumulator that the former writeback calls used.
    unsafe {
        (cx.emitter)(
            ab, cx.mr, cx.nr, mrem, nrem, cx.alpha, beta, c, cr, cc, crs, conj_c, d, dr, dc, drs,
            conj_d,
        )
    }
}

/// Everything the micro-tile loops need that is fixed for one
/// `(batch, NC panel, KC slab)` epoch: the batch-offset operand bases, the K
/// slab, the NC block and where the packed `B` sliver for a column lives.
/// Shared by the static and the dynamic scheduler so there is one copy of the
/// arithmetic and of the direct-path branches.
#[derive(Clone, Copy)]
pub(super) struct Epoch<T: Element> {
    pub(super) ah: *mut T,
    pub(super) bh: *mut T,
    pub(super) ch: *const T,
    pub(super) dh: *mut T,
    pub(super) pc: usize,
    pub(super) pc_len: usize,
    pub(super) first_k_block: bool,
    pub(super) jc: usize,
    pub(super) jc_len: usize,
    pub(super) b_sliver: usize,
    /// Packed `B` for sliver `q0` of this NC block (the first of the group).
    pub(super) bp: *mut T::Real,
    pub(super) q0: usize,
}

/// One worker's private buffers.
#[derive(Clone, Copy)]
pub(super) struct Bufs<R> {
    pub(super) ap: *mut R,
    pub(super) tile: *mut R,
    pub(super) scratch: *mut R,
}

/// Pack `B` slivers `[w0, w1)` of the epoch's NC block into the shared panel.
///
/// # Safety
/// As [`run_strip`]; `w0..w1` are slivers of the block owned by the caller
/// alone, and the panel slice has room for them.
pub(super) unsafe fn pack_b_slivers<T>(cx: &Ctx<'_, T>, ep: &Epoch<T>, w0: usize, w1: usize)
where
    T: Element,
{
    let (nr, bn, bk, conj_b) = (cx.nr, cx.bn, cx.bk, cx.conj_b);
    let b_n_bs = cx.runs.slice(cx.scatter, cx.runs.b);
    let (pc, pc_len, b_sliver) = (ep.pc, ep.pc_len, ep.b_sliver);
    let c0 = ep.jc + w0 * nr;
    let c1 = (ep.jc + w1 * nr).min(ep.jc + ep.jc_len);
    let out = ep.bp.add((w0 - ep.q0) * b_sliver);
    let (_, pack_b) = cx.packers;
    // SAFETY: the validated scatters and capacity of this epoch.
    unsafe {
        pack_b(
            ep.bh,
            &bn[c0..c1],
            &b_n_bs[c0 / nr..c1.div_ceil(nr)],
            &bk[pc..pc + pc_len],
            nr,
            conj_b,
            out,
        )
    }
}

/// Pack rows `[ic, ic + ic_len)` of the epoch's K slab into `ap`.
///
/// # Safety
/// As [`run_strip`]; `ap` is the caller's private panel with room for `ic_len`
/// rows of the slab.
pub(super) unsafe fn pack_a_rows<T>(
    cx: &Ctx<'_, T>,
    ep: &Epoch<T>,
    ic: usize,
    ic_len: usize,
    ap: *mut T::Real,
) where
    T: Element,
{
    let (mr, am, ak, conj_a) = (cx.mr, cx.am, cx.ak, cx.conj_a);
    let a_m_bs = cx.runs.slice(cx.scatter, cx.runs.a);
    let (pc, pc_len) = (ep.pc, ep.pc_len);
    let (pack_a, _) = cx.packers;
    // SAFETY: the validated scatters and capacity of this epoch.
    unsafe {
        pack_a(
            ep.ah,
            &am[ic..ic + ic_len],
            &a_m_bs[ic / mr..(ic + ic_len).div_ceil(mr)],
            &ak[pc..pc + pc_len],
            mr,
            conj_a,
            ap,
        )
    }
}

/// Loops 2 and 1: every micro-tile of the packed `A` rows `[ic, ic + ic_len)`
/// against the columns `[jr_lo, jr_hi)` of the epoch's NC block, with the
/// write-back. The one copy of the tile arithmetic, the direct-C/direct-B
/// branches and the first-slab/accumulate rule.
///
/// # Safety
/// As [`run_strip`]; `bufs.ap` holds the packed rows, the packed `B` panel (or
/// the in-place operand) covers the columns, and the caller owns the output
/// tiles of the covered block.
#[inline(always)]
pub(super) unsafe fn compute_block<T>(
    cx: &Ctx<'_, T>,
    ep: &Epoch<T>,
    bufs: Bufs<T::Real>,
    ic: usize,
    ic_len: usize,
    jr_lo: usize,
    jr_hi: usize,
) where
    T: Element,
{
    let Ctx {
        plan,
        fam,
        mr,
        nr,
        cm,
        cn,
        dm,
        dn,
        bn,
        bk,
        alpha,
        beta,
        direct_b,
        ..
    } = *cx;
    let Epoch {
        bh,
        ch,
        dh,
        pc,
        pc_len,
        first_k_block,
        jc,
        jc_len,
        b_sliver,
        bp: bp_ptr,
        q0,
        ..
    } = *ep;
    let (ap_ptr, tile_ptr, scratch_ptr) = (bufs.ap, bufs.tile, bufs.scratch);
    let runs = cx.runs;
    let scatter = cx.scatter;
    let b_n_bs = runs.slice(scatter, runs.b);
    let d_m_bs = runs.slice(scatter, runs.dm);
    let d_n_bs = runs.slice(scatter, runs.dn);
    let c_m_bs = runs.slice(scatter, runs.cm);
    let one = T::one();
    let a_sliver = fam.a_per_k * pc_len;
    let mut jr = jr_lo;
    while jr < jr_hi {
        let nrem = nr.min(jc_len - jr);
        let j0 = jc + jr;
        let bpan = bp_ptr.add((jr / nr - q0) * b_sliver);
        // B's k steps are one apart and its columns one constant
        // stride apart, both checked by `pack_b_needed`; otherwise
        // the tile reads the packed panel (k stride `NR`, columns
        // adjacent).
        let (b_base, b_rs, b_cs) = if direct_b {
            // The base carries this k block's and this column
            // block's offsets, so the kernel's own k stride is one.
            (
                bh.offset((bn[j0] + bk[pc]) as isize) as *const T::Real,
                1,
                b_n_bs[j0 / nr] as isize,
            )
        } else {
            (bpan as *const T::Real, nr as isize, 1)
        };

        // ---- loop 1: MR -----------------------------------
        let mut ir = 0;
        while ir < ic_len {
            let mrem = mr.min(ic_len - ir);
            let i0 = ic + ir;
            let apan = ap_ptr.add((ir / mr) * a_sliver);
            let d_rs = *d_m_bs.get_unchecked(i0 / mr);
            // A Direct family writes D itself only where D's own
            // strides make that expressible: the guard was decided
            // once for the call, and both scatters must be regular.
            let direct_tile = matches!(fam.kernel, UkrFn::Direct(_))
                && cx.call.direct_c_allowed
                && d_rs != IRREGULAR
                && *d_n_bs.get_unchecked(j0 / nr) != IRREGULAR;

            match fam.kernel {
                UkrFn::Tile(_) => {
                    // SAFETY: full packed panels and tile per
                    // family contract; an induced family
                    // scales the inner kernel's k itself.
                    unsafe {
                        tprims_kernel::induced::tile_call(
                            &fam,
                            pc_len,
                            apan,
                            bpan,
                            tile_ptr,
                            scratch_ptr,
                        )
                    };
                }
                UkrFn::Direct(func) => {
                    // A direct tile overlaps the accumulator it
                    // scales; a fallback tile is overwritten
                    // (`alpha_d = 0`), and the write-back below
                    // then applies alpha/beta exactly as the
                    // scratch path does.
                    let (d_base, rs_d, cs_d, alpha_d, beta_ab) = if direct_tile {
                        (
                            dh.offset((dm[i0] + dn[j0]) as isize) as *mut T::Real,
                            d_rs as isize,
                            *d_n_bs.get_unchecked(j0 / nr) as isize,
                            if first_k_block {
                                if beta == T::zero() {
                                    T::Real::ZERO
                                } else {
                                    beta.re()
                                }
                            } else {
                                T::Real::ONE
                            },
                            alpha.re(),
                        )
                    } else {
                        (tile_ptr, 1, mr as isize, T::Real::ZERO, T::Real::ONE)
                    };
                    let aux = UkrAux {
                        a_next: if ir + mr < ic_len {
                            apan.add(a_sliver)
                        } else {
                            apan
                        },
                        // In-place B has no panel to point past.
                        b_next: match (direct_b, jr + nr < jc_len) {
                            (true, _) => b_base,
                            (false, true) => bpan.add(b_sliver),
                            (false, false) => bpan,
                        },
                        inner: None,
                        opaque: fam.opaque,
                    };
                    // SAFETY: A is the packed panel with unit row
                    // stride; B and D follow the strides derived
                    // above, and `d_out` is the live tile extent.
                    unsafe {
                        (func)(
                            mrem,
                            nrem,
                            pc_len,
                            d_base,
                            rs_d,
                            cs_d,
                            apan as *const T::Real,
                            fam.a_per_k as isize,
                            b_base,
                            b_rs,
                            b_cs,
                            alpha_d,
                            beta_ab,
                            &aux,
                        )
                    };
                }
            }

            if direct_tile {
                ir += mr;
                continue;
            }
            if first_k_block {
                let c_rs = c_m_bs.get(i0 / mr).copied().unwrap_or(IRREGULAR);
                emit_tile::<T>(
                    cx,
                    tile_ptr,
                    mrem,
                    nrem,
                    beta,
                    ch,
                    &cm[i0..i0 + mrem],
                    &cn[j0..j0 + nrem],
                    c_rs,
                    plan.conj_c,
                    dh,
                    &dm[i0..i0 + mrem],
                    &dn[j0..j0 + nrem],
                    d_rs,
                    plan.conj_d,
                );
            } else {
                // Accumulate: C := D, beta := 1, and conjugate
                // the readback exactly when op_D conjugates.
                emit_tile::<T>(
                    cx,
                    tile_ptr,
                    mrem,
                    nrem,
                    one,
                    dh as *const T,
                    &dm[i0..i0 + mrem],
                    &dn[j0..j0 + nrem],
                    d_rs,
                    plan.conj_d,
                    dh,
                    &dm[i0..i0 + mrem],
                    &dn[j0..j0 + nrem],
                    d_rs,
                    plan.conj_d,
                );
            }
            ir += mr;
        }
        jr += nr;
    }
}
