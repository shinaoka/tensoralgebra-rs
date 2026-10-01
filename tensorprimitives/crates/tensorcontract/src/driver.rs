//! The five-loop driver.
//!
//! tprims: threads come only from a [`tprims_exec::Exec`] (not upstream); this
//! crate spawns none, and without an `Exec` the driver runs serially.
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
//! BLIS-style, and two-dimensional: the output is cut into a `pm x pn` grid of
//! contiguous row strips of whole `MR` panels by column groups of whole `NR`
//! blocks, one thread each. [`Plan::partition`] decides `(pm, pn)` and is the
//! single definition of it; `pn > 1` only when `ceil(M / MR) < p`, so on all but
//! four of the 49 corpus cases this is still a 1-D partition of `M` and behaves
//! exactly as the first version did.
//!
//! The two axes are cut in different places, and deliberately:
//!
//! * The **row strips** are cut once, outside everything, and each thread runs
//!   loops 3, 2 and 1 over its own strip with its own packed-`A` block.
//! * The **column groups** are cut *inside* loop 5, per `NC` block: every thread
//!   iterates the same `(h, jc, pc)` sequence over the whole of `N` and `K`, and
//!   within each `jc` block takes its group's contiguous range of `NR` slivers.
//!
//! Cutting `N` inside loop 5 rather than over the whole range is what keeps the
//! packed-`B` panel single and shared. It stays exactly the L3-sized panel the
//! `NC` budget is derived for — a top-level split of `N` would need `pn` panels
//! and `pn` times the L3 — and it keeps loops 5 and 4 identical across all `p`
//! threads, which is what makes the barrier counts agree without anyone tracking
//! them (see `run_strip`).
//!
//! `B` is packed cooperatively: within a column group, the `pm` threads that
//! share it split its slivers, so each thread packs slivers it will itself read
//! and no thread packs anything it will not. Two barriers per `(jc, pc)` bracket
//! the packing — one so nobody is still reading the previous panel, one so the
//! new one is complete — and they are **per column group**, of `pm` threads,
//! because a group's slice of the panel is written and read only by its own
//! threads. A one-thread column group therefore needs no barrier at all, which
//! is the case a pure `N` split (`pm == 1`) degenerates to: no synchronisation
//! anywhere in the loop nest.
//!
//! The packed `A` block is per thread and so is **duplicated `pn` times**: the
//! `pn` threads of one row strip each pack that strip for themselves. That is
//! deliberate rather than merely convenient. It costs `ceil(panels/pm) * MR * K`
//! element moves per thread against `ceil(panels/pm) * ceil(blocks/pn) * MR * NR
//! * K` lane-FMAs, i.e. one packed element per `NR * ceil(blocks/pn)` FMA slots,
//! and it buys back the alternative's cost: a single packer per row strip needs a
//! barrier *inside* loop 3, at `M/MC` times the frequency of the ones above it,
//! and leaves the block in one thread's L2 for the others to pull across L3
//! instead of each having it in its own. `Plan::partition` prices the duplication
//! explicitly and will not split `N` when a thread would be left with too few
//! `NR` blocks to amortise it over.
//!
//! Four properties this buys, all of them deliberate:
//!
//! * **No reduction.** Loop 4 (`pc`) accumulates into `D` in place, so
//!   parallelising it would need either a temporary per thread or atomics.
//!   Partitioning the *output* instead gives every element a single owning
//!   thread, which accumulates over the full `K` in the original order. Both
//!   axes have this property; `K` is the one that does not, and it is left
//!   serial.
//! * **Bitwise identical to serial**, therefore, for any thread count and any
//!   `(pm, pn)` — the floating-point operations per output element are the same
//!   operations in the same order. That is a strong enough invariant to test
//!   directly, and the `threaded_matches_serial_*` tests do.
//! * **Blocks stay aligned with the block scatter.** Strips are whole `MR`
//!   panels and groups whole `NR` slivers, so every thread's micro-tiles are the
//!   ones the serial driver would have used. The write-back fast path, the
//!   row-block rule and the orientation rule are untouched by threading.
//! * **The serial path is unchanged.** With `pm == pn == 1` the only difference
//!   from the pre-threading driver is two `Option` checks and a handful of
//!   integer divisions per `(jc, pc)` iteration, nowhere near the hot loops.
//!   Every measurement committed in `docs/notebook/` was taken single-threaded and
//!   stays comparable.
//!
//! Known limits, in the order they will bite (see the Phase 4 report):
//! parallelism is capped at `ceil(M / MR) * ceil(N / NR)`, and a column group can
//! only be as wide as the `jc` block it is cut from, so a tail `NC` block with
//! fewer slivers than groups leaves some threads idle for that block; and `NC`'s
//! L3 budget is still charged as if one core owned the cache.
//!
//! The per-call spawn cost that used to head that list — `std::thread::scope`
//! rather than a pool, ~20–36 µs per thread and the whole story below a megabyte
//! (A43, D46) — is gone: the threads of one call are the workers of the host's
//! [`tprims_exec::Pool`], co-scheduled by [`tprims_exec::Exec::broadcast`]. For
//! many small contractions, [`crate::batch`] parallelises over a *batch*
//! instead of inside each contraction.

use std::sync::Barrier;

use tprims_exec::{Exec, WorkspaceProvider, WorkspaceReq};

use crate::buffer::Panel;
mod dynamic;
use crate::element::Element;
use crate::kernel::{config_for_plan, Blocking, KernelSet};
use crate::pack::{pack_panel, panel_len};
use crate::plan::Plan;
use crate::scatter::IRREGULAR;
use crate::writeback::{scale_only, writeback};
pub use dynamic::{dynamic_report, Assignment, DynSnapshot, DynStats, DynamicReport};
use tprims_kernel::scatter::append_block_scatter;
use tprims_kernel::{Axis, BAccess, DriverFamily, Real, UkrAux, UkrFn};

/// Operand-dependent decisions the driver makes once per execute, kept
/// separate from the plan-level resolution so tests can pin the real rule.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResolvedCall {
    /// The shared B panel must be packed this call.
    pub pack_b_needed: bool,
    /// Eligible tiles may be updated directly, without a scratch tile.
    pub direct_c_allowed: bool,
}

/// The same two decisions the driver takes, for a given plan, resolution and
/// operand pair.
#[doc(hidden)]
pub fn driver_decisions<T>(
    plan: &Plan,
    rg: &tprims_kernel::ResolvedGemm<T::Real>,
    c: *const T,
    d: *mut T,
    beta: T,
) -> ResolvedCall
where
    T: Element,
{
    // Which user operand plays the kernel's column role, exactly as
    // `execute_capped` decides it.
    let swap = plan.transposes_gemm(rg.mr);
    let (bk, bn) = if swap {
        (&plan.a_k, &plan.a_m)
    } else {
        (&plan.b_k, &plan.b_n)
    };
    ResolvedCall {
        pack_b_needed: pack_b_needed(rg.family().b_access, bk, bn, rg.nr),
        direct_c_allowed: direct_c_allowed(plan, c, d, beta),
    }
}

/// Whether the family must pack this call's column operand: it must accept an
/// in-place B, B's k stride must be one, and every `NR` block's column offsets
/// must be an arithmetic progression, so one stride expresses the tile.
///
/// This reads the operand's own scatter rather than its block scatter, so it
/// can decide before any scratch buffer exists.
pub(crate) fn pack_b_needed(b_access: BAccess, bk: &[i64], bn: &[i64], nr: usize) -> bool {
    if !matches!(
        b_access,
        BAccess::Direct {
            unit_stride: Axis::Col
        }
    ) {
        return true;
    }
    if bk.len() > 1 && !bk.windows(2).all(|w| w[1] - w[0] == 1) {
        return true;
    }
    !tprims_kernel::scatter::block_scatter_regular(bn, nr)
}

/// Whether the Direct kernel may write D in place. Real storage only, no C or D
/// conjugation, and either C is not read at all or C *is* D — because the
/// kernel has a single output pointer it scales as `alpha_d*D + beta_ab*A*B`.
fn direct_c_allowed<T: Element>(plan: &Plan, c: *const T, d: *mut T, beta: T) -> bool {
    !T::IS_COMPLEX
        && !plan.conj_c
        && !plan.conj_d
        && (beta == T::zero()
            || (core::ptr::eq(c, d as *const T)
                && plan.c_m == plan.d_m
                && plan.c_n == plan.d_n
                && plan.h_c == plan.h_d))
}

/// Where the five block-scatter vectors sit inside one reused scatter buffer.
///
/// They are built together into the team's buffer so a steady-state execute
/// allocates nothing; these offsets are what keeps them separate slices.
#[derive(Clone, Copy, Default)]
struct ScatterRuns {
    a: (usize, usize),
    b: (usize, usize),
    dm: (usize, usize),
    dn: (usize, usize),
    cm: (usize, usize),
}

impl ScatterRuns {
    fn slice<'a>(&self, buf: &'a [i64], run: (usize, usize)) -> &'a [i64] {
        &buf[run.0..run.1]
    }
}

/// A raw pointer shared across the threads of one [`execute`] call.
///
/// Rust will not send a bare pointer between threads, and rightly, so the
/// promise is made explicitly here rather than silently at each use: the
/// operands are read-only for the duration, and every thread writes only the
/// output elements of its own `(row strip, column group)` cell. The cells
/// partition the output's `(i, j)` index space by construction, so no two
/// threads write the same element — and no two write the same *byte*, because
/// `D`'s scatter is injective, which the serial write-back's read-modify-write
/// on every `pc` block past the first already requires.
#[derive(Clone, Copy)]
struct Shared<T>(*mut T);

// SAFETY: see the type's documentation. The disjointness is a property of the
// strip partition in `execute`, which is the only place `Shared` is created.
//
// `T: Send` on both, and it is `Send` rather than `Sync` that is wanted even for
// the `Sync` impl: `Shared` is written through from several threads at once, so
// it is morally a split `&mut T` rather than a shared `&T`, and the obligation
// it discharges is that values of `T` may be produced and dropped on a thread
// other than the one that created them. Every instantiation is `T: Element`,
// which is already `Copy + Send + Sync + 'static`, so the bound costs nothing
// here -- it is there so the impls cannot silently start covering a `T` that
// does not deserve them. `buffer::Panel` bounds its `Send` the same way.
unsafe impl<T: Send> Send for Shared<T> {}
unsafe impl<T: Send> Sync for Shared<T> {}

/// Everything one thread of the loop nest needs that does not vary with its
/// row strip. Exists so that the nest can be written once and run either
/// serially or on `p` threads, rather than duplicated.
struct Ctx<'a, T: Element>
where
    T::Real: KernelSet,
{
    plan: &'a Plan,
    fam: DriverFamily<T::Real>,
    packers: Option<(tprims_kernel::PackFn<T>, tprims_kernel::PackFn<T>)>,
    emitter: Option<tprims_kernel::EmitFn<T>>,
    /// This call's operand-dependent decisions, when a resolution is in use.
    call: Option<ResolvedCall>,
    /// Whether the kernel's column role takes B in place this call.
    direct_b: bool,
    mr: usize,
    nr: usize,
    mc: usize,
    kc: usize,
    nc: usize,
    m: usize,
    n: usize,
    k: usize,
    /// Reals between one column group's slice of the packed `B` panel and the
    /// next. See `execute` for why the panel is cut per group and not per sliver.
    b_group: usize,
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
    /// Every block-scatter vector this call needs, laid out by `runs`.
    scatter: &'a [i64],
    runs: ScatterRuns,
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
) where
    T::Real: KernelSet,
{
    // SAFETY: run_strip supplies the same checked live tile/scatters and
    // initialized accumulator that the former writeback calls used.
    if let Some(emit) = cx.emitter {
        unsafe {
            emit(
                ab, cx.mr, cx.nr, mrem, nrem, cx.alpha, beta, c, cr, cc, crs, conj_c, d, dr, dc,
                drs, conj_d,
            )
        }
    } else {
        unsafe {
            writeback::<T>(
                ab,
                cx.fam.tile_fmt,
                cx.mr,
                cx.nr,
                mrem,
                nrem,
                cx.alpha,
                beta,
                c,
                cr,
                cc,
                crs,
                conj_c,
                d,
                dr,
                dc,
                drs,
                conj_d,
            )
        }
    }
}

/// Where one thread sits in the `pm x pn` partition, as far as loop 5 and the
/// shared packed-`B` panel are concerned.
///
/// Each `jc` block of `nsliv` slivers is cut into `pn` contiguous groups; this
/// thread computes over group `g`'s slivers and packs the fraction `r` of `pm`
/// of them. Both ranges come from [`BPart::ranges`], which is the only place the
/// arithmetic lives, so "which slivers do I own" and "which slivers do I pack"
/// cannot drift apart.
///
/// [`BPart::SERIAL`] is the degenerate `1 x 1` case and reproduces the
/// pre-threading driver exactly: one thread, all the slivers, no barrier.
#[derive(Clone, Copy)]
struct BPart<'a> {
    /// This thread's column group, in `0..pn`.
    g: usize,
    pn: usize,
    /// This thread's row strip, in `0..pm` — its index *within* the column
    /// group, which is what decides its share of the group's packing.
    r: usize,
    pm: usize,
    /// Barrier shared by the `pm` threads of this column group, and by nobody
    /// else: they are the only threads that touch the group's slice of the
    /// panel. `None` when the group has one thread, which then packs and reads
    /// only what it wrote and needs no synchronisation at all.
    bar: Option<&'a Barrier>,
}

impl BPart<'_> {
    /// The serial partition: one cell covering everything, no barrier.
    const SERIAL: BPart<'static> = BPart {
        g: 0,
        pn: 1,
        r: 0,
        pm: 1,
        bar: None,
    };

    /// `(compute, pack)` sliver ranges out of a `jc` block's `nsliv` slivers:
    /// this thread's whole column group, and its share of packing that group.
    ///
    /// Both are half-open and both partition exactly — the `pn` groups tile
    /// `0..nsliv` and the `pm` packing shares tile their group — which is what
    /// makes every output element owned once and every sliver packed once. A
    /// group can come out empty when a tail `jc` block has fewer slivers than
    /// there are groups; that thread then does no work for the block, but still
    /// takes its barriers.
    #[inline]
    fn ranges(&self, nsliv: usize) -> ((usize, usize), (usize, usize)) {
        let q0 = self.g * nsliv / self.pn;
        let q1 = (self.g + 1) * nsliv / self.pn;
        let span = q1 - q0;
        let w0 = q0 + self.r * span / self.pm;
        let w1 = q0 + (self.r + 1) * span / self.pm;
        ((q0, q1), (w0, w1))
    }
}

fn lcm(a: usize, b: usize) -> usize {
    fn gcd(mut a: usize, mut b: usize) -> usize {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    }
    a / gcd(a, b) * b
}

/// Execute a plan.
///
/// # Safety
/// * `a`, `b`, `d` must be valid for all offsets generated by the plan's
///   scatter vectors (reads for `a`/`b`, reads and writes for `d`).
/// * `c` must likewise be valid for reads unless `beta` is zero, in which case
///   it is never dereferenced and may be dangling.
/// * `d` must not alias `a` or `b`.
/// * Selection/configuration must be valid for T and every active width,
///   including serial fallback. `Plan::run` validates this automatically;
///   raw built-in callers can validate with `Plan::resolved::<T>()` and
///   `with_threads(1)` before execution.
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
    // SAFETY: forwarded unchanged; the serial `Exec` is the only thread source
    // when the caller supplies none.
    unsafe {
        execute_capped(
            plan,
            alpha,
            a,
            b,
            beta,
            c,
            d,
            &Exec::Serial,
            None,
            None,
            None,
        )
    }
}

/// Execute an already resolved built-in family on `exec`, with `workspace`
/// (or, when `None`, the one `exec` lends). The width is `exec.budget()`;
/// effective blocking uses the active grid width, not the plan's requested width.
///
/// # Safety
/// All `execute` pointer/alias obligations apply. `rg` must be validated for
/// `T`, the plan's conjugations, scratch ABI, and every active width (including
/// serial fallback); its public geometry/blocking fields must remain valid.
///
/// # Examples
/// ```
/// use tensorcontract::{Layout, Operand, Plan};
/// let l = Layout::col_major(&[2,2]);
/// let p = Plan::new(Operand::new(&l,&[0,2]), Operand::new(&l,&[2,1]),
///     None, Operand::new(&l,&[0,1]))?.with_threads(1);
/// let rg = p.resolved::<f64>()?.with_threads(1)?;
/// let a = [1.,2.,3.,4.]; let b = [5.,6.,7.,8.]; let mut d = [0.;4];
/// // SAFETY: full checked layouts/buffers, valid resolution; beta=0 ignores C.
/// unsafe { tensorcontract::execute_resolved(&p,&rg,&tprims_exec::Exec::serial(),None,
///     1.,a.as_ptr(),b.as_ptr(),0.,std::ptr::null(),d.as_mut_ptr()); }
/// assert_eq!(d, [23.,34.,31.,46.]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// An `exec` that declines a multi-worker broadcast is retried serially with
/// this same frozen family/policy; it never re-selects the plan default.
pub unsafe fn execute_resolved<T: Element>(
    plan: &Plan,
    rg: &tprims_kernel::ResolvedGemm<T::Real>,
    exec: &Exec<'_>,
    workspace: Option<&dyn WorkspaceProvider>,
    alpha: T,
    a: *const T,
    b: *const T,
    beta: T,
    c: *const T,
    d: *mut T,
) where
    T::Real: KernelSet,
{
    // SAFETY: raw pointer and descriptor obligations forwarded unchanged.
    unsafe {
        execute_capped(
            plan,
            alpha,
            a,
            b,
            beta,
            c,
            d,
            exec,
            workspace,
            Some(rg),
            None,
        )
    }
}

/// [`execute_resolved`] with opt-in [`DynStats`] counters, for tests and
/// benchmarks of `PartitionPolicy::DynamicTiles`. Ordinary execution carries no
/// counters.
///
/// # Safety
/// As [`execute_resolved`].
#[doc(hidden)]
pub unsafe fn execute_resolved_instrumented<T: Element>(
    plan: &Plan,
    rg: &tprims_kernel::ResolvedGemm<T::Real>,
    exec: &Exec<'_>,
    workspace: Option<&dyn WorkspaceProvider>,
    stats: &DynStats,
    alpha: T,
    a: *const T,
    b: *const T,
    beta: T,
    c: *const T,
    d: *mut T,
) where
    T::Real: KernelSet,
{
    // SAFETY: raw pointer and descriptor obligations forwarded unchanged.
    unsafe {
        execute_capped(
            plan,
            alpha,
            a,
            b,
            beta,
            c,
            d,
            exec,
            workspace,
            Some(rg),
            Some(stats),
        )
    }
}

/// [`execute`] on the co-scheduled threads of `exec` (tprims addition).
///
/// # Safety
///
/// As [`execute`].
#[allow(clippy::too_many_arguments)] // INVARIANT: same argument set as `execute_capped`.
pub unsafe fn execute_with<T>(
    plan: &Plan,
    exec: &Exec<'_>,
    workspace: Option<&dyn WorkspaceProvider>,
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
    // SAFETY: forwarded unchanged.
    unsafe { execute_capped(plan, alpha, a, b, beta, c, d, exec, workspace, None, None) }
}

/// [`execute`] on `exec`: its budget is the width this call may use, and its
/// workspace (or `workspace`, when given) is where the call's buffers live.
///
/// Passing [`Exec::Serial`] is how a caller runs an item serially, for example
/// when the parallel axis is a batch of items.
///
/// # Safety
///
/// As [`execute`].
// Eight arguments, against clippy's seven: seven of them are the contraction
// itself — `alpha`, four operands, `beta` and the plan — and bundling them into a
// struct to satisfy a count would put a layer between the ABI-facing entry points
// and the loop nest for no reader's benefit.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn execute_capped<T>(
    plan: &Plan,
    alpha: T,
    a: *const T,
    b: *const T,
    beta: T,
    c: *const T,
    d: *mut T,
    exec: &Exec<'_>,
    workspace: Option<&dyn WorkspaceProvider>,
    resolution: Option<&tprims_kernel::ResolvedGemm<T::Real>>,
    stats: Option<&dynamic::DynStats>,
) where
    T: Element,
    T::Real: KernelSet,
{
    // INVARIANT: safe run/run_with validate selection (including the serial
    // NC bound) first; raw callers promise a valid selection for this dtype.
    let resolved = match resolution {
        Some(rg) => Some(*rg),
        None => crate::resolve::builtin::<T>(plan)
            .map(|r| r.expect("raw execution requires valid kernel selection")),
    };
    if plan.is_empty() {
        return;
    }

    // Built-ins consume the plan's cached descriptor; foreign scalar types
    // retain the original KernelSet path without a new trait bound.
    let (fam, blk) = match resolved {
        Some(rg) => (
            rg.family()
                .driver_family()
                .expect("validated family ABI for this scheme"),
            Blocking {
                mc: rg.mc,
                kc: rg.kc,
                nc: rg.nc,
            },
        ),
        None => {
            let cfg = config_for_plan::<T>(plan);
            let cfg = match plan.blocking {
                Some(blk) => cfg.with_blocking(blk),
                None => cfg,
            };
            let ukr = cfg.ukr;
            (
                DriverFamily {
                    mr: ukr.mr,
                    nr: ukr.nr,
                    a_per_k: ukr.a_per_k,
                    b_per_k: ukr.b_per_k,
                    tile: ukr.tile,
                    a_pack: ukr.a_pack,
                    b_pack: ukr.b_pack,
                    tile_fmt: ukr.tile_fmt,
                    b_access: BAccess::Packed,
                    kernel: UkrFn::Tile(ukr.func),
                    opaque: core::ptr::null(),
                    inner: None,
                    method: None,
                },
                cfg.blk,
            )
        }
    };
    let (mr, nr) = (fam.mr, fam.nr);

    // Row/column orientation. Exchanging `(A, M)` with `(B, N)` computes
    // `D^T = B^T A^T`, which is the same contraction seen through the
    // transposed matrix view of `C` and `D`. Nothing below branches on it
    // again: from here on `am`/`ak`/`ptr_a` *are* the row operand, whichever
    // tensor that is. See `Plan::transposes_gemm` for why it is worth doing.
    //
    // Packing formats remain fixed to kernel roles: row-A uses 1e and
    // column-B uses 1r under 1m, whichever user tensor fills that role.
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

    let workspace = workspace.or_else(|| exec.workspace());
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

    // Operand-dependent decisions, made once for the whole call: the family's
    // B access, this B's strides, and whether D can be updated in place. Both
    // are allocation-free, because the partition below depends on them and the
    // team's buffer is only borrowed once the shape is known.
    let call = resolved.map(|rg| ResolvedCall {
        pack_b_needed: pack_b_needed(rg.family().b_access, bk, bn, nr),
        direct_c_allowed: direct_c_allowed(plan, c, d, beta),
    });
    let direct_b = call.is_some_and(|c| !c.pack_b_needed);

    // NOTE (Phase 4): `cfg.blk` is still the untouched Phase 2 heuristic, and
    // `MC`/`NC` in it are sized for a `KC`-deep panel. On a third of the corpus
    // the contraction is far shallower than `KC` (`k = 24` against 256), so the
    // packed `A` block uses a tenth of its budget and `B` is re-streamed
    // `M/MC` times for nothing. Re-deriving against `min(k, KC)` was tried and
    // is *not* a win: it swings individual cases by +13% and −18% with no rule
    // visible, because `MC` has a second constraint this model omits — the
    // strip of `D` that one `jr` pass revisits. See the Phase 4 report; this is
    // what the `MC`/`KC`/`NC` sweep has to settle.
    // Row strips are whole `MR` panels and column groups whole `NR` slivers, so
    // every thread's micro-tiles line up with the block scatter and with the
    // write-back's fast path. `Plan::partition` owns the choice of how many of
    // each; it caps them at the panel and block counts, so a contraction with
    // three row panels and two column blocks uses six threads at most however
    // many were asked for and however much work it contains.
    // The width is the `Exec`'s budget, not the plan's: threads come from the
    // host alone.
    let want = exec.budget();
    // An explicit grid is clamped to the width this call may use; the default
    // grid is the plan's own cost model, which already respects it.
    let explicit_grid = match resolved.map(|rg| rg.partition) {
        Some(tprims_kernel::PartitionPolicy::StaticGrid { pm, pn }) if pm != 0 => Some((pm, pn)),
        _ => None,
    };
    // `DynamicTiles`: the team claims jobs, so the grid is `p x 1` (all in the
    // row direction, which only fixes the barrier's size). The width is the
    // host's budget capped by the jobs that exist at the widest NC block.
    let dyn_jobs = match resolved.map(|rg| rg.partition) {
        Some(tprims_kernel::PartitionPolicy::DynamicTiles { job_m, job_n }) => {
            let nc_serial = resolved
                .and_then(|rg| rg.with_threads(1).ok())
                .map_or(n, |rg| rg.nc);
            let jobs = dynamic::job_count(m, n, nr, nc_serial, job_m, job_n);
            Some((job_m, job_n, jobs))
        }
        _ => None,
    };
    let (mut pm, mut pn) = match (explicit_grid, dyn_jobs) {
        (_, Some((_, _, jobs))) => (want.min(jobs), 1),
        (Some((pm, pn)), None) => (pm, pn),
        (None, None) => plan.partition_with(mr, nr, want),
    };
    // A pinned partition (`PartitionMode::Pin` or an explicit grid)
    // ignores the thread count, so shrink it to the budget.
    if dyn_jobs.is_none() {
        while pm * pn > want {
            if pn > 1 {
                pn -= 1;
            } else {
                pm -= 1;
            }
        }
    }
    let p = pm * pn;
    // A direct-B kernel reads B where it lies, one column group at a time, so
    // there is no shared panel to publish and no cross-thread barrier: the
    // partition must be a pure split of `N` (`pm == 1`). The total width is
    // unchanged, so the blocking below still matches the active thread count.
    if direct_b && dyn_jobs.is_none() {
        pm = 1;
        pn = p;
    }
    let blocking = if let Some(rg) = resolved {
        // INVARIANT: safe execution checked serial's maximal NC, and p >= 1.
        let rg = rg
            .with_threads(p)
            .expect("validated effective-width blocking");
        Blocking {
            mc: rg.mc,
            kc: rg.kc,
            nc: rg.nc,
        }
    } else {
        blk
    };
    let Blocking { mc, kc, nc } = blocking;
    let mc = mc.min(m.next_multiple_of(mr));
    let nc = nc.min(n.next_multiple_of(nr));
    // C-line aligned strips, when asked for: one line of `C` is 64 bytes, so
    // the boundary is a multiple of both the panel and the line.
    let align = match resolved.map(|rg| rg.opts.align_c_lines) {
        Some(true) => lcm(mr, (64 / core::mem::size_of::<T>()).max(1)),
        _ => 0,
    };

    // Panel sizes come from the kernel's declared per-k sliver widths, so a
    // method that packs more reals per element (1m's "1e", 3m's sum plane)
    // automatically gets a correspondingly larger buffer. `A` is per thread and
    // allocated inside it, so it is first-touched on the node that will use it;
    // `B` is shared and allocated here.
    //
    // The `B` panel is cut into one slice per *column group*, not indexed by
    // absolute sliver, and the two differ: a group's sliver range moves between
    // `jc` blocks, because a tail block has fewer slivers to divide, so with
    // absolute indexing one group's next block would land on top of another
    // group's current one. There is nothing to order them — the barriers are per
    // group by design, and at `pm == 1` there are none at all — so it has to be
    // structural. It costs at most one sliver of padding per group, and it is
    // also a small win: a group's slice is contiguous, so groups do not share
    // cache lines at their boundaries.
    //
    // The stride between slices is the *worst-case* sliver size, `kc` deep and
    // not `pc_len` deep, precisely so that two groups sitting on different `pc`
    // blocks at the same moment still cannot overlap.
    let ap_len = panel_len(mc, mr, kc, fam.a_pack);
    let group_cap = nc.div_ceil(nr).div_ceil(pn);
    let b_group = panel_len(group_cap * nr, nr, kc, fam.b_pack);
    // What this call needs from the owner, if it has one. Everything is in
    // bytes except the element counts of the scratch vectors.
    let element = core::mem::size_of::<T::Real>();
    let req = WorkspaceReq {
        a_bytes: ap_len * element,
        tile_bytes: (fam.tile + fam.induced_scratch(kc)) * element,
        worker_scatter: 0,
        // A direct-B call never touches the panel, so it asks for none.
        b_bytes: if direct_b { 0 } else { pn * b_group * element },
        team_scatter: am.len().div_ceil(mr)
            + bn.len().div_ceil(nr)
            + dm.len().div_ceil(mr)
            + dn.len().div_ceil(nr)
            + if beta == T::zero() {
                0
            } else {
                cm.len().div_ceil(mr)
            },
        barriers: if pm > 1 { pn } else { 0 },
    };
    if let Some(st) = stats {
        st.note_call(p, req.b_bytes);
    }
    let mut lease = workspace.map(|ws| ws.take_team(&req, pm, pn));
    // The panel lives in the lease when there is one, and in a call-local
    // allocation otherwise; a direct-B call sizes it to zero either way. The
    // pointer is taken before the scatter borrow, which is held to the end.
    let has_lease = lease.is_some();
    let mut local_bp = match has_lease || direct_b {
        true => None,
        false => Some(Panel::<T::Real>::new(pn * b_group)),
    };
    let bp_ptr = match lease.as_mut() {
        Some(l) => l.panel(req.b_bytes) as *mut T::Real,
        None => match local_bp.as_mut() {
            Some(p) => p.as_mut_ptr(),
            // A direct-B call reads B in place, so it has no panel at all.
            None => std::ptr::NonNull::<T::Real>::dangling().as_ptr(),
        },
    };
    // A leased team set also carries the scatter vectors and the barriers, so a
    // steady-state call reuses their capacity instead of allocating five
    // vectors and a barrier list. Both are taken as shared slices once the
    // filling is done, so they can be borrowed together.
    let mut local_scatter = Vec::new();
    let local_bars: Vec<Barrier>;
    let runs = {
        let buf = match lease.as_mut() {
            Some(l) => &mut l.scatter,
            None => &mut local_scatter,
        };
        buf.clear();
        buf.reserve(req.team_scatter);
        ScatterRuns {
            a: append_block_scatter(buf, am, mr),
            b: append_block_scatter(buf, bn, nr),
            dm: append_block_scatter(buf, dm, mr),
            dn: append_block_scatter(buf, dn, nr),
            cm: match beta == T::zero() {
                true => (buf.len(), buf.len()),
                false => append_block_scatter(buf, cm, mr),
            },
        }
    };
    let (scatter_buf, bars) = match lease.as_ref() {
        Some(l) => (l.scatter.as_slice(), l.barriers.as_slice()),
        None => {
            local_bars = (0..pn).map(|_| Barrier::new(pm)).collect();
            (local_scatter.as_slice(), local_bars.as_slice())
        }
    };

    let cx = Ctx::<T> {
        plan,
        fam,
        packers: resolved.map(|rg| rg.packers::<T>()),
        emitter: resolved.map(|rg| rg.emitter::<T>()),
        call,
        direct_b,
        mr,
        nr,
        mc,
        kc,
        nc,
        m,
        n,
        k,
        b_group,
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
        scatter: scatter_buf,
        runs,
        conj_a,
        conj_b,
        alpha,
        beta,
        a: Shared(ptr_a as *mut T),
        b: Shared(ptr_b as *mut T),
        c: Shared(c as *mut T),
        d: Shared(d),
        bp: Shared(bp_ptr),
    };

    // Worker buffers come from the owner when there is one, and from a fresh
    // per-call panel otherwise; either way they are the caller-of-`f`'s for the
    // duration of `f`, and the 4m scratch is carved out of the tile.
    let with_buffers = |f: &mut dyn FnMut(*mut T::Real, *mut T::Real)| match workspace {
        Some(ws) => {
            ws.with_worker(&req, &mut |a, tile, _| {
                f(a as *mut T::Real, tile as *mut T::Real)
            });
        }
        None => {
            let mut ap = Panel::<T::Real>::new(ap_len);
            let mut tile = Panel::<T::Real>::new(fam.tile + fam.induced_scratch(kc));
            f(ap.as_mut_ptr(), tile.as_mut_ptr());
        }
    };
    let scratch_off = fam.tile;

    if p == 1 {
        with_buffers(&mut |ap, tile| {
            // SAFETY: `execute`'s contract, and these buffers are exclusive to
            // this call for its duration.
            unsafe { run_strip::<T>(&cx, 0, m, ap, tile, tile.add(scratch_off), BPart::SERIAL) }
        });
        return;
    }

    // One barrier per column group, each shared by exactly the `pm` threads
    // that write and read that group's slice of the packed `B` panel. Groups
    // never need to synchronise with each other, so they do not: the barrier is
    // `pm`-way, not `p`-way. At `pm == 1` there is nothing to synchronise and
    // the threads take no barrier at all — which `bars` being empty expresses.

    // One thread's whole job, as a function of its index in the `pm x pn` grid.
    // Written once; the broadcast below runs it on the `Exec`'s workers. The
    // partition, the strips and therefore the arithmetic are the same at every
    // width, which is why the result stays bitwise identical to serial.
    let claim = std::sync::atomic::AtomicUsize::new(0);
    let cell = |t: usize| {
        let cx = &cx;
        if let Some((job_m, job_n, _)) = dyn_jobs {
            with_buffers(&mut |ap, tile| {
                let bufs = Bufs {
                    ap,
                    tile,
                    scratch: tile.add(scratch_off),
                };
                // SAFETY: `execute`'s contract covers the accesses; the claim
                // counter hands each job to one worker, and jobs partition the
                // output within an epoch. The barrier has `p` participants.
                unsafe {
                    dynamic::run_dynamic::<T>(cx, t, p, job_m, job_n, &claim, &bars[0], bufs, stats)
                };
            });
            return;
        }
        let (r, g) = (t / pn, t % pn);
        let bpart = BPart {
            g,
            pn,
            r,
            pm,
            bar: (pm > 1).then(|| &bars[g]),
        };
        let (lo, hi) = tprims_kernel::partition::strip(r, pm, cx.m, mr, align);
        with_buffers(&mut |ap, tile| {
            // SAFETY: `execute`'s contract covers the accesses; the strips and
            // column groups partition the output, so this thread's writes are
            // disjoint from every other thread's.
            unsafe { run_strip::<T>(cx, lo, hi, ap, tile, tile.add(scratch_off), bpart) };
        });
    };

    // `broadcast` runs nothing when it declines (a serial `Exec`, a width the
    // pool or budget cannot serve, or a caller already on the pool's workers),
    // so this is a real either/or and never a partial execution.
    if exec.broadcast(p, &cell).is_ok() {
        return;
    }
    // Declined: nothing ran, so running serially is safe, and no thread is ever
    // spawned behind the host's back. The caller does the work with this call's
    // own buffers and panel, so the refusal costs no second allocation and no
    // second lease.
    with_buffers(&mut |ap, tile| {
        // SAFETY: forwarded unchanged from this call's contract; a serial strip
        // is the degenerate partition and needs no barrier.
        unsafe { run_strip::<T>(&cx, 0, m, ap, tile, tile.add(scratch_off), BPart::SERIAL) }
    });
}

/// Loops 5 through 1 over one cell of the `pm x pn` partition.
///
/// `[m_lo, m_hi)` are whole `MR` panels; `bpart` selects whole `NR` slivers
/// within each `jc` block.
///
/// **Loops 5 and 4 are traversed identically by every thread** — the same `h`,
/// the same `jc`, the same `pc`, over the whole of `N` and `K` — and the
/// barriers live between them and loop 3. That is what makes the barrier counts
/// match without anyone counting: a thread's cell affects only *how much work it
/// does inside* an iteration, never how many iterations there are. Nothing here
/// may make a barrier conditional on `m_lo`, `m_hi` or `bpart`; a thread with an
/// empty cell in some `jc` block still takes that block's barriers and then does
/// nothing, which is why the skip below sits after them and not before.
///
/// # Safety
/// As [`execute`], plus: `ap` and `tile` must be this thread's alone, and no
/// other thread may own an overlapping cell.
unsafe fn run_strip<T>(
    cx: &Ctx<'_, T>,
    m_lo: usize,
    m_hi: usize,
    ap_ptr: *mut T::Real,
    tile_ptr: *mut T::Real,
    scratch_ptr: *mut T::Real,
    bpart: BPart<'_>,
) where
    T: Element,
    T::Real: KernelSet,
{
    let Ctx {
        plan,
        fam,
        nr,
        mc,
        kc,
        nc,
        n,
        k,
        b_group,
        ha,
        hb,
        direct_b,
        ..
    } = *cx;
    let (ptr_a, ptr_b, c, d, bp_ptr) = (cx.a.0, cx.b.0, cx.c.0 as *const T, cx.d.0, cx.bp.0);

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
                let b_sliver = fam.b_per_k * pc_len;

                // Pack the shared `B` panel. `(q0, q1)` are the slivers this
                // thread will compute over — its column group — and `(w0, w1)`
                // its share of packing them. The two barriers say "nobody is
                // still reading the previous panel" and "the new one is
                // complete", and they bracket only the group's own slice
                // because only the group's own threads touch it. Serially both
                // ranges are all the slivers and there are no barriers, which
                // is exactly the call the pre-threading driver made.
                let nsliv = jc_len.div_ceil(nr);
                let ((q0, q1), (w0, w1)) = bpart.ranges(nsliv);
                // This column group's private slice of the panel. Sliver `s` of
                // the group lives at `(s - q0)` within it, not at `s`: see
                // `execute` for why the panel is cut per group.
                debug_assert!(
                    (q1 - q0) * b_sliver <= b_group,
                    "column group overruns its slice of the packed B panel"
                );
                let bp_ptr = bp_ptr.add(bpart.g * b_group);
                let ep = Epoch::<T> {
                    ah,
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
                };
                let bufs = Bufs {
                    ap: ap_ptr,
                    tile: tile_ptr,
                    scratch: scratch_ptr,
                };
                // A direct-B call reads B where it lies: no panel is written,
                // so neither barrier is taken. The decision is per call, so
                // every thread of a group skips the same pair.
                if !direct_b {
                    if let Some(bar) = bpart.bar {
                        bar.wait();
                    }
                    if w1 > w0 {
                        // SAFETY: slivers `w0..w1` are this thread's alone.
                        unsafe { pack_b_slivers::<T>(cx, &ep, w0, w1) };
                    }
                    if let Some(bar) = bpart.bar {
                        bar.wait();
                    }
                }

                // This thread's slice of loop 2, in columns of the `jc` block.
                let (jr_lo, jr_hi) = (q0 * nr, (q1 * nr).min(jc_len));

                // ---- loop 3: M blocking -----------------------------------
                // Skipped wholesale when this thread's column group is empty in
                // this `jc` block — possible only in a tail block with fewer
                // slivers than groups — since there is no point packing an `A`
                // block no micro-kernel call will read. Both barriers above have
                // already been taken, which is what keeps their counts equal.
                let mut ic = if jr_lo < jr_hi { m_lo } else { m_hi };
                while ic < m_hi {
                    let ic_len = mc.min(m_hi - ic);
                    pack_a_rows::<T>(cx, &ep, ic, ic_len, ap_ptr);
                    compute_block::<T>(cx, &ep, bufs, ic, ic_len, jr_lo, jr_hi);
                    ic += mc;
                }
                pc += kc;
            }
            jc += nc;
        }
    }
}

/// Everything the micro-tile loops need that is fixed for one
/// `(batch, NC panel, KC slab)` epoch: the batch-offset operand bases, the K
/// slab, the NC block and where the packed `B` sliver for a column lives.
/// Shared by the static and the dynamic scheduler so there is one copy of the
/// arithmetic and of the direct-path branches.
#[derive(Clone, Copy)]
struct Epoch<T: Element> {
    ah: *mut T,
    bh: *mut T,
    ch: *const T,
    dh: *mut T,
    pc: usize,
    pc_len: usize,
    first_k_block: bool,
    jc: usize,
    jc_len: usize,
    b_sliver: usize,
    /// Packed `B` for sliver `q0` of this NC block (the first of the group).
    bp: *mut T::Real,
    q0: usize,
}

/// One worker's private buffers.
#[derive(Clone, Copy)]
struct Bufs<R> {
    ap: *mut R,
    tile: *mut R,
    scratch: *mut R,
}

/// Pack `B` slivers `[w0, w1)` of the epoch's NC block into the shared panel.
///
/// # Safety
/// As [`run_strip`]; `w0..w1` are slivers of the block owned by the caller
/// alone, and the panel slice has room for them.
unsafe fn pack_b_slivers<T>(cx: &Ctx<'_, T>, ep: &Epoch<T>, w0: usize, w1: usize)
where
    T: Element,
    T::Real: KernelSet,
{
    let (nr, bn, bk, conj_b, fam) = (cx.nr, cx.bn, cx.bk, cx.conj_b, cx.fam);
    let b_n_bs = cx.runs.slice(cx.scatter, cx.runs.b);
    let (pc, pc_len, b_sliver) = (ep.pc, ep.pc_len, ep.b_sliver);
    let c0 = ep.jc + w0 * nr;
    let c1 = (ep.jc + w1 * nr).min(ep.jc + ep.jc_len);
    let out = ep.bp.add((w0 - ep.q0) * b_sliver);
    if let Some((_, pack_b)) = cx.packers {
        // SAFETY: same validated scatters/capacity as legacy pack_panel.
        pack_b(
            ep.bh,
            &bn[c0..c1],
            &b_n_bs[c0 / nr..c1.div_ceil(nr)],
            &bk[pc..pc + pc_len],
            nr,
            conj_b,
            out,
        );
    } else {
        pack_panel::<T>(
            ep.bh,
            &bn[c0..c1],
            &b_n_bs[c0 / nr..c1.div_ceil(nr)],
            &bk[pc..pc + pc_len],
            nr,
            conj_b,
            fam.b_pack,
            out,
        );
    }
}

/// Pack rows `[ic, ic + ic_len)` of the epoch's K slab into `ap`.
///
/// # Safety
/// As [`run_strip`]; `ap` is the caller's private panel with room for `ic_len`
/// rows of the slab.
unsafe fn pack_a_rows<T>(cx: &Ctx<'_, T>, ep: &Epoch<T>, ic: usize, ic_len: usize, ap: *mut T::Real)
where
    T: Element,
    T::Real: KernelSet,
{
    let (mr, am, ak, conj_a, fam) = (cx.mr, cx.am, cx.ak, cx.conj_a, cx.fam);
    let a_m_bs = cx.runs.slice(cx.scatter, cx.runs.a);
    let (pc, pc_len) = (ep.pc, ep.pc_len);
    if let Some((pack_a, _)) = cx.packers {
        // SAFETY: same validated scatters/capacity as legacy pack_panel.
        pack_a(
            ep.ah,
            &am[ic..ic + ic_len],
            &a_m_bs[ic / mr..(ic + ic_len).div_ceil(mr)],
            &ak[pc..pc + pc_len],
            mr,
            conj_a,
            ap,
        );
    } else {
        pack_panel::<T>(
            ep.ah,
            &am[ic..ic + ic_len],
            &a_m_bs[ic / mr..(ic + ic_len).div_ceil(mr)],
            &ak[pc..pc + pc_len],
            mr,
            conj_a,
            fam.a_pack,
            ap,
        );
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
unsafe fn compute_block<T>(
    cx: &Ctx<'_, T>,
    ep: &Epoch<T>,
    bufs: Bufs<T::Real>,
    ic: usize,
    ic_len: usize,
    jr_lo: usize,
    jr_hi: usize,
) where
    T: Element,
    T::Real: KernelSet,
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
                && cx.call.is_some_and(|c| c.direct_c_allowed)
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
