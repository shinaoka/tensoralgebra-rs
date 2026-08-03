//! The five-loop driver.
//!
//! Structurally identical to BLIS's GEMM: two levels of cache blocking with a
//! packing step at each, wrapped around a register-blocked micro-kernel. The
//! only tensor-specific part is that every matrix access goes through a
//! scatter vector.
//!
//! ```text
//! for each Hadamard (batch) index h          -- offsets all four operands
//!   loop 5: for jc in 0..N step NC
//!     loop 4: for pc in 0..K step KC
//!               pack B[pc:pc+KC, jc:jc+NC] -> Bp    (NR slivers)
//!       loop 3: for ic in 0..M step MC
//!                 pack A[ic:ic+MC, pc:pc+KC] -> Ap  (MR slivers)
//!         loop 2: for jr in 0..NC step NR
//!           loop 1: for ir in 0..MC step MR
//!                     micro-kernel -> MR x NR tile
//!                     write back to C/D through the scatter vectors
//! ```
//!
//! The loop nest is identical for real and for all three complex methods. What
//! the complex method changes is captured entirely by the selected
//! [`Ukr`](crate::kernel::Ukr): its `a_pack`/`b_pack` formats, its per-k sliver
//! widths, its tile size and its `tile_fmt`. Nothing here branches on the
//! method, which is what makes the three genuinely comparable — they share
//! every line of index analysis, packing traversal, loop arithmetic and
//! write-back scatter.
//!
//! `beta` and the `C` operand are consumed on the first `pc` iteration only;
//! later iterations accumulate into `D`.
//!
//! # Threading
//!
//! BLIS-style, over the `M` direction only. The `M` range is cut once into `p`
//! contiguous strips at `MR` granularity; each thread runs loops 3, 2 and 1
//! over its own strip with its own packed-`A` block, and the packed-`B` panel is
//! **shared**: every thread packs a slice of its `NR` slivers and then all of
//! them stream the whole panel out of L3, which is what the L3-sized `NC`
//! budget is for. Two barriers per `(jc, pc)` iteration bracket the packing —
//! one so nobody is still reading the previous panel, one so the new one is
//! complete.
//!
//! Three properties this buys, all of them deliberate:
//!
//! * **No reduction.** Loop 4 (`pc`) accumulates into `D` in place, so
//!   parallelising it would need either a temporary per thread or atomics.
//!   Parallelising `M` instead gives every output element a single owning
//!   thread, which accumulates over the full `K` in the original order.
//! * **Bitwise identical to serial**, therefore, for any thread count — the
//!   floating-point operations per output element are the same operations in the
//!   same order. That is a strong enough invariant to test directly, and
//!   `threaded_matches_serial_bitwise` does.
//! * **The serial path is unchanged.** With `p == 1` the only difference from
//!   the pre-threading driver is two `Option` checks per `(jc, pc)` iteration,
//!   nowhere near the hot loops. Every measurement committed in `DECISIONS.md`
//!   was taken single-threaded and stays comparable.
//!
//! Known limits, in the order they will bite (see the Phase 4 report):
//! parallelism is capped at `ceil(M / MR)` strips, so a skinny-`M` contraction
//! cannot use the cores no matter how much other work it has; `std::thread::scope`
//! spawns per `execute` call rather than reusing a pool; and `NC`'s L3 budget is
//! still charged as if one core owned the cache.

use std::sync::Barrier;

use crate::buffer::Panel;
use crate::element::Element;
use crate::kernel::{config_for_plan, Blocking, KernelSet, Ukr};
use crate::pack::{pack_panel, panel_len};
use crate::plan::Plan;
use crate::scatter::{build_block_scatter, IRREGULAR};
use crate::writeback::{scale_only, writeback};

/// A raw pointer shared across the threads of one [`execute`] call.
///
/// Rust will not send a bare pointer between threads, and rightly, so the
/// promise is made explicitly here rather than silently at each use: the
/// operands are read-only for the duration, and every thread writes only the
/// output rows in its own strip. The strips are disjoint by construction, so no
/// two threads address the same byte of `D`.
#[derive(Clone, Copy)]
struct Shared<T>(*mut T);

// SAFETY: see the type's documentation. The disjointness is a property of the
// strip partition in `execute`, which is the only place `Shared` is created.
unsafe impl<T> Send for Shared<T> {}
unsafe impl<T> Sync for Shared<T> {}

/// Everything one thread of the loop nest needs that does not vary with its
/// row strip. Exists so that the nest can be written once and run either
/// serially or on `p` threads, rather than duplicated.
struct Ctx<'a, T: Element>
where
    T::Real: KernelSet,
{
    plan: &'a Plan,
    ukr: Ukr<T::Real>,
    mr: usize,
    nr: usize,
    mc: usize,
    kc: usize,
    nc: usize,
    m: usize,
    n: usize,
    k: usize,
    am: &'a [i64],
    ak: &'a [i64],
    bk: &'a [i64],
    bn: &'a [i64],
    cm: &'a [i64],
    cn: &'a [i64],
    dm: &'a [i64],
    dn: &'a [i64],
    ha: &'a [i64],
    hb: &'a [i64],
    a_m_bs: &'a [i64],
    b_n_bs: &'a [i64],
    d_m_bs: &'a [i64],
    c_m_bs: &'a [i64],
    conj_a: bool,
    conj_b: bool,
    alpha: T,
    beta: T,
    a: Shared<T>,
    b: Shared<T>,
    c: Shared<T>,
    d: Shared<T>,
    /// The shared packed-`B` panel: written cooperatively, read by everyone.
    bp: Shared<T::Real>,
}

/// How one thread participates in packing the shared `B` panel: which slice of
/// the `NR` slivers it takes, and the barrier that brackets the packing. `None`
/// is the serial case — one thread packs all of them and synchronises with
/// nobody.
type BShare<'a> = Option<(&'a Barrier, usize, usize)>;

/// Execute a plan.
///
/// # Safety
/// * `a`, `b`, `d` must be valid for all offsets generated by the plan's
///   scatter vectors (reads for `a`/`b`, reads and writes for `d`).
/// * `c` must likewise be valid for reads unless `beta` is zero, in which case
///   it is never dereferenced and may be dangling.
/// * `d` must not alias `a` or `b`.
pub unsafe fn execute<T>(
    plan: &Plan,
    alpha: T,
    a: *const T,
    b: *const T,
    beta: T,
    c: *const T,
    d: *mut T,
) where
    T: Element,
    T::Real: KernelSet,
{
    if plan.is_empty() {
        return;
    }

    // `MR` is a plan-level choice, not just a kernel constant: it sets the
    // granularity at which the output's row scatter is blocked, and so which
    // write-back path each block takes. See `Plan::row_block`.
    let cfg = config_for_plan::<T>(plan);
    let cfg = match plan.blocking {
        Some(blk) => cfg.with_blocking(blk),
        None => cfg,
    };
    let ukr = cfg.ukr;
    let (mr, nr) = (ukr.mr, ukr.nr);

    // Row/column orientation. Exchanging `(A, M)` with `(B, N)` computes
    // `D^T = B^T A^T`, which is the same contraction seen through the
    // transposed matrix view of `C` and `D`. Nothing below branches on it
    // again: from here on `am`/`ak`/`ptr_a` *are* the row operand, whichever
    // tensor that is. See `Plan::transposes_gemm` for why it is worth doing.
    //
    // Note this also swaps the two pack formats, which matters only for 1m,
    // where they differ ("1e" for rows, "1r" for columns) — and there it is
    // exactly right, since the kernel's contract is about the row and column
    // panels, not about which user tensor they came from.
    let swap = plan.transposes_gemm(mr);
    let (ptr_a, ptr_b) = if swap { (b, a) } else { (a, b) };
    let (am, ak, conj_a) = if swap {
        (&plan.b_n, &plan.b_k, plan.conj_b)
    } else {
        (&plan.a_m, &plan.a_k, plan.conj_a)
    };
    let (bk, bn, conj_b) = if swap {
        (&plan.a_k, &plan.a_m, plan.conj_a)
    } else {
        (&plan.b_k, &plan.b_n, plan.conj_b)
    };
    let (cm, cn) = if swap {
        (&plan.c_n, &plan.c_m)
    } else {
        (&plan.c_m, &plan.c_n)
    };
    let (dm, dn) = if swap {
        (&plan.d_n, &plan.d_m)
    } else {
        (&plan.d_m, &plan.d_n)
    };
    let (ha, hb) = if swap {
        (&plan.h_b, &plan.h_a)
    } else {
        (&plan.h_a, &plan.h_b)
    };

    let m = am.len();
    let n = bn.len();
    let k = ak.len();

    // Empty contraction dimension: D = op_D(beta * op_C(C)).
    if plan.has_empty_contraction() {
        for h in 0..plan.stats.batch {
            scale_only::<T>(
                beta,
                c.offset(plan.h_c[h] as isize),
                cm,
                cn,
                plan.conj_c,
                d.offset(plan.h_d[h] as isize),
                dm,
                dn,
                plan.conj_d,
            );
        }
        return;
    }

    // Block-scatter metadata. Cheap (O(M/MR + N/NR)) and element-type
    // dependent only through MR/NR, so it lives here rather than in the plan.
    let a_m_bs = build_block_scatter(am, mr);
    let b_n_bs = build_block_scatter(bn, nr);
    // Row block scatter for the output too: the write-back uses it exactly as
    // packing uses the operands', to keep scatter-table loads out of its
    // innermost loop. `C` is only consulted when beta is nonzero.
    let d_m_bs = build_block_scatter(dm, mr);
    let c_m_bs = if beta == T::zero() {
        Vec::new()
    } else {
        build_block_scatter(cm, mr)
    };

    // NOTE (Phase 4): `cfg.blk` is still the untouched Phase 2 heuristic, and
    // `MC`/`NC` in it are sized for a `KC`-deep panel. On a third of the corpus
    // the contraction is far shallower than `KC` (`k = 24` against 256), so the
    // packed `A` block uses a tenth of its budget and `B` is re-streamed
    // `M/MC` times for nothing. Re-deriving against `min(k, KC)` was tried and
    // is *not* a win: it swings individual cases by +13% and −18% with no rule
    // visible, because `MC` has a second constraint this model omits — the
    // strip of `D` that one `jr` pass revisits. See the Phase 4 report; this is
    // what the `MC`/`KC`/`NC` sweep has to settle.
    let Blocking { mc, kc, nc } = cfg.blk;
    let mc = mc.min(m.next_multiple_of(mr));
    let nc = nc.min(n.next_multiple_of(nr));

    // Panel sizes come from the kernel's declared per-k sliver widths, so a
    // method that packs more reals per element (1m's "1e", 3m's sum plane)
    // automatically gets a correspondingly larger buffer. `A` is per thread and
    // allocated inside it, so it is first-touched on the node that will use it;
    // `B` is shared and allocated here.
    let ap_len = panel_len(mc, mr, kc, ukr.a_pack);
    let mut bp = Panel::<T::Real>::new(panel_len(nc, nr, kc, ukr.b_pack));

    let cx = Ctx::<T> {
        plan,
        ukr,
        mr,
        nr,
        mc,
        kc,
        nc,
        m,
        n,
        k,
        am,
        ak,
        bk,
        bn,
        cm,
        cn,
        dm,
        dn,
        ha,
        hb,
        a_m_bs: &a_m_bs,
        b_n_bs: &b_n_bs,
        d_m_bs: &d_m_bs,
        c_m_bs: &c_m_bs,
        conj_a,
        conj_b,
        alpha,
        beta,
        a: Shared(ptr_a as *mut T),
        b: Shared(ptr_b as *mut T),
        c: Shared(c as *mut T),
        d: Shared(d),
        bp: Shared(bp.as_mut_ptr()),
    };

    // Strips are whole `MR` panels, so every thread's row blocks line up with
    // the block scatter and with the write-back's fast path. That caps the
    // useful thread count at the number of panels — a contraction with three
    // row blocks cannot use eight cores here however much work it contains.
    let npanels = m.div_ceil(mr);
    let p = plan.strips(mr);

    if p == 1 {
        let mut ap = Panel::<T::Real>::new(ap_len);
        let mut tile = Panel::<T::Real>::new(ukr.tile);
        run_strip::<T>(&cx, 0, m, ap.as_mut_ptr(), tile.as_mut_ptr(), None);
        return;
    }

    let bar = Barrier::new(p);
    std::thread::scope(|scope| {
        for t in 0..p {
            let cx = &cx;
            let bar = &bar;
            scope.spawn(move || {
                let lo = (t * npanels / p) * mr;
                let hi = (((t + 1) * npanels / p) * mr).min(cx.m);
                let mut ap = Panel::<T::Real>::new(ap_len);
                let mut tile = Panel::<T::Real>::new(cx.ukr.tile);
                // SAFETY: `execute`'s contract covers the accesses; the strips
                // partition the output rows, so this thread's writes are
                // disjoint from every other thread's.
                unsafe {
                    run_strip::<T>(
                        cx,
                        lo,
                        hi,
                        ap.as_mut_ptr(),
                        tile.as_mut_ptr(),
                        Some((bar, t, p)),
                    )
                };
            });
        }
    });
}

/// Loops 5 through 1 over one strip of the `M` range.
///
/// `[m_lo, m_hi)` are whole `MR` panels. Everything above loop 3 is identical
/// across threads, which is what makes the barrier counts match without
/// tracking them.
///
/// # Safety
/// As [`execute`], plus: `ap` and `tile` must be this thread's alone, and no
/// other thread may own an overlapping row strip.
unsafe fn run_strip<T>(
    cx: &Ctx<'_, T>,
    m_lo: usize,
    m_hi: usize,
    ap_ptr: *mut T::Real,
    tile_ptr: *mut T::Real,
    bshare: BShare<'_>,
) where
    T: Element,
    T::Real: KernelSet,
{
    let Ctx {
        plan,
        ukr,
        mr,
        nr,
        mc,
        kc,
        nc,
        n,
        k,
        am,
        ak,
        bk,
        bn,
        cm,
        cn,
        dm,
        dn,
        ha,
        hb,
        a_m_bs,
        b_n_bs,
        d_m_bs,
        c_m_bs,
        conj_a,
        conj_b,
        alpha,
        beta,
        ..
    } = *cx;
    let (ptr_a, ptr_b, c, d, bp_ptr) = (cx.a.0, cx.b.0, cx.c.0 as *const T, cx.d.0, cx.bp.0);
    let one = T::one();

    for h in 0..plan.stats.batch {
        let ah = ptr_a.offset(ha[h] as isize);
        let bh = ptr_b.offset(hb[h] as isize);
        let ch = c.offset(plan.h_c[h] as isize);
        let dh = d.offset(plan.h_d[h] as isize);

        // ---- loop 5: N blocking -------------------------------------------
        let mut jc = 0;
        while jc < n {
            let jc_len = nc.min(n - jc);

            // ---- loop 4: K blocking ---------------------------------------
            let mut pc = 0;
            while pc < k {
                let pc_len = kc.min(k - pc);
                let first_k_block = pc == 0;
                let b_sliver = ukr.b_per_k * pc_len;

                // Pack the shared `B` panel. Threaded, each thread takes a
                // slice of the `NR` slivers; the two barriers say "nobody is
                // still reading the previous panel" and "the new one is
                // complete". Serially this is one call over all of them, with
                // the same arguments the pre-threading driver used.
                let nsliv = jc_len.div_ceil(nr);
                let (q0, q1) = match bshare {
                    Some((bar, t, p)) => {
                        bar.wait();
                        (t * nsliv / p, (t + 1) * nsliv / p)
                    }
                    None => (0, nsliv),
                };
                if q1 > q0 {
                    let c0 = jc + q0 * nr;
                    let c1 = (jc + q1 * nr).min(jc + jc_len);
                    pack_panel::<T>(
                        bh,
                        &bn[c0..c1],
                        &b_n_bs[c0 / nr..c1.div_ceil(nr)],
                        &bk[pc..pc + pc_len],
                        nr,
                        conj_b,
                        ukr.b_pack,
                        bp_ptr.add(q0 * b_sliver),
                    );
                }
                if let Some((bar, _, _)) = bshare {
                    bar.wait();
                }

                // ---- loop 3: M blocking -----------------------------------
                let mut ic = m_lo;
                while ic < m_hi {
                    let ic_len = mc.min(m_hi - ic);

                    pack_panel::<T>(
                        ah,
                        &am[ic..ic + ic_len],
                        &a_m_bs[ic / mr..(ic + ic_len).div_ceil(mr)],
                        &ak[pc..pc + pc_len],
                        mr,
                        conj_a,
                        ukr.a_pack,
                        ap_ptr,
                    );
                    let a_sliver = ukr.a_per_k * pc_len;

                    // ---- loop 2: NR ---------------------------------------
                    let mut jr = 0;
                    while jr < jc_len {
                        let nrem = nr.min(jc_len - jr);
                        let j0 = jc + jr;
                        let bpan = bp_ptr.add((jr / nr) * b_sliver);

                        // ---- loop 1: MR -----------------------------------
                        let mut ir = 0;
                        while ir < ic_len {
                            let mrem = mr.min(ic_len - ir);
                            let i0 = ic + ir;
                            let apan = ap_ptr.add((ir / mr) * a_sliver);

                            (ukr.func)(pc_len, apan, bpan, tile_ptr);

                            let d_rs = *d_m_bs.get_unchecked(i0 / mr);
                            if first_k_block {
                                let c_rs = c_m_bs.get(i0 / mr).copied().unwrap_or(IRREGULAR);
                                writeback::<T>(
                                    tile_ptr,
                                    ukr.tile_fmt,
                                    mr,
                                    nr,
                                    mrem,
                                    nrem,
                                    alpha,
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
                                writeback::<T>(
                                    tile_ptr,
                                    ukr.tile_fmt,
                                    mr,
                                    nr,
                                    mrem,
                                    nrem,
                                    alpha,
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
                    ic += mc;
                }
                pc += kc;
            }
            jc += nc;
        }
    }
}
