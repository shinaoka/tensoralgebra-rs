//! [`Plan`]: a prepared contraction.
//!
//! A [`Plan`] is built from a validated [`Problem`] and a [`PlanConfig`]. It
//! owns one immutable metadata record and one strategy payload:
//!
//! ```text
//! Plan<T>
//!   problem   original layouts, normalized roles, ops, C mode, checked spans
//!   strategy  Packed(PackedPlan + resolved family) | Faer(FaerPlan) | Elementwise(ElementPlan)
//!   report    immutable PlanReport
//! ```
//!
//! # Strategy selection
//!
//! Chosen once, at construction, by these rules in order:
//!
//! 1. An explicit kernel, selector, partition, complex method, blocking, cache
//!    model or write-back request forces the **packed** driver, including for
//!    an all-batch problem. It is honoured or refused.
//! 2. An all-batch problem runs the **elementwise** pass, which implements the
//!    full `op_C` / `op_D` / separate-`C` semantics.
//! 3. A problem that fuses to one strided batched GEMM without copying any
//!    operand, with full semantics, runs on **faer**.
//! 4. Everything else runs on the packed driver.
//!
//! Runtime pointers, `alpha` and `beta` may pick the packed driver's existing
//! direct-B and direct-C applicability fallbacks, never another family.
//!
//! # Execution
//!
//! After one operation boundary checks layout compatibility and the
//! accumulation source, an empty output returns; `alpha == 0` or an empty
//! contraction computes `op_D(beta * op_C(C))` and reads no `A` or `B`;
//! otherwise the call dispatches once to the prepared strategy. `beta == 0`
//! never reads `C` or `D`. A plan holds no pointer to an operand and works with
//! different buffers and different executor budgets; the actual width is chosen
//! from the work estimate and the budget at each call.

mod analysis;
mod config;
mod orientation;
mod report;
#[cfg(test)]
pub(crate) mod test_support;

use core::marker::PhantomData;

use strided_view::{StridedView, StridedViewMut};
use tprims_exec::{ArenaProvider, Exec, WidthPolicy, WorkspaceProvider};
use tprims_kernel::{Element, KernelCatalog, ResolvedGemm, SelectError};

pub(crate) use analysis::PackedPlan;
pub use analysis::{Axis, PlanStats};
pub use config::{CacheModel, Partition, PlanConfig, Writeback};
pub use orientation::{Orient, RowBlock};
pub use report::{Algorithm, PackedReport, PlanReport};

use crate::api::{
    AccumulationSource, AliasError, CSpec, ConfigError, Diagnostics, LayoutError, OperandId,
    PreparedContraction, Problem, Result, Scalar,
};
use crate::driver;
use crate::resolve;
use crate::select::Chooser;
use crate::strategy::elementwise::{CRead, ElementPlan, Expr, Inputs};
use crate::strategy::faer::{self as faer_strategy, FaerPlan};

/// Estimated serial time per real flop (provisional: 20 GFLOP/s); the one
/// default the width choice uses everywhere.
pub const NS_PER_FLOP: f64 = 0.05;

#[derive(Debug)]
struct Packed<T: Scalar> {
    plan: PackedPlan,
    rg: ResolvedGemm<<T as Scalar>::Re>,
}

#[derive(Debug)]
enum Strategy<T: Scalar> {
    Packed(Box<Packed<T>>),
    Faer(FaerPlan),
    Elementwise(ElementPlan),
}

/// A prepared contraction for fixed layouts, reusable across executions.
///
/// # Examples
///
/// ```
/// use strided_view::{StridedView, StridedViewMut};
/// use tprims_contract::api::{CSpec, DType, Labels, LayoutSpec, OperandSpec, Problem};
/// use tprims_contract::{Plan, PlanConfig};
/// use tprims_exec::Exec;
///
/// let l = |d: &[usize], s: &[isize]| OperandSpec::new(LayoutSpec::new(d, s, 0).unwrap());
/// // D[i,k] = sum_j A[i,j] B[j,k]
/// let problem = Problem::from_labels(
///     DType::F64,
///     l(&[2, 2], &[1, 2]),
///     l(&[2, 2], &[1, 2]),
///     CSpec::Absent,
///     l(&[2, 2], &[1, 2]),
///     &Labels::new(&[0, 1], &[1, 2], &[0, 2]),
/// ).unwrap();
/// let plan = Plan::<f64>::new(&problem, &PlanConfig::default()).unwrap();
/// let a = [1.0, 2.0, 3.0, 4.0];
/// let b = [1.0, 0.0, 0.0, 1.0];
/// let mut d = [0.0; 4];
/// let av = StridedView::new(&a, &[2, 2], &[1, 2], 0).unwrap();
/// let bv = StridedView::new(&b, &[2, 2], &[1, 2], 0).unwrap();
/// let mut dv = StridedViewMut::new(&mut d, &[2, 2], &[1, 2], 0).unwrap();
/// plan.execute_into(&Exec::serial(), 1.0, &av, &bv, &mut dv).unwrap();
/// assert_eq!(d, a);
/// ```
#[derive(Debug)]
pub struct Plan<T: Scalar> {
    problem: Problem,
    strategy: Strategy<T>,
    /// The pass over the output's elements: `alpha == 0` and empty `K`.
    output: ElementPlan,
    report: PlanReport,
    diagnostics: Diagnostics,
    /// Lent to serial executions so their steady state allocates nothing; a
    /// borrowed pool lends its own arena instead.
    workspace: ArenaProvider,
    _t: PhantomData<fn() -> T>,
}

impl<T: Scalar> Plan<T> {
    /// Plan `problem` under `config`.
    ///
    /// # Errors
    ///
    /// [`ConfigError::DtypeMismatch`] when `T` is not the problem's dtype, any
    /// [`PlanConfig::validate`] error, [`Error::Select`] for an unknown or
    /// unusable kernel, blocking or partition request, and
    /// [`ShapeError::Overflow`](crate::api::ShapeError) for a role whose scatter
    /// vector cannot be addressed.
    pub fn new(problem: &Problem, config: &PlanConfig) -> Result<Self> {
        Self::build(problem, config, None)
    }

    /// [`Plan::new`] on the packed driver, choosing the kernel family with a
    /// caller-supplied selector over a caller-supplied [`KernelCatalog`]
    /// (issue #28).
    ///
    /// The selector sees metadata only -- the validated, oriented shape and the
    /// explicit method -- and returns an opaque handle. It is called at most
    /// once, here, outside every lock and worker broadcast; execution never
    /// calls it, and the plan keeps only the chosen trusted handle, so the
    /// selector and catalog may be dropped. A selector is a requirement: it
    /// forces the packed driver, including for an all-batch problem.
    ///
    /// # Errors
    ///
    /// As [`Plan::new`], plus [`Error::Select`] for an explicit kernel id
    /// alongside the selector (ambiguous), no admissible candidate, the
    /// selector's own `Err` (unchanged) or a handle it should not have
    /// returned. Zero-size problems are selected and validated too.
    pub fn new_with_selector(
        problem: &Problem,
        config: &PlanConfig,
        catalog: &KernelCatalog<T>,
        selector: &mut Chooser<'_, T>,
    ) -> Result<Self> {
        if let tprims_kernel::KernelChoice::Id(id) = &config.kernel {
            return Err(SelectError::Incompatible {
                id: id.clone(),
                reason: "a forced kernel id and a custom selector are ambiguous",
            }
            .into());
        }
        Self::build(problem, config, Some((catalog, selector)))
    }

    fn build(
        problem: &Problem,
        config: &PlanConfig,
        selection: Option<(&KernelCatalog<T>, &mut Chooser<'_, T>)>,
    ) -> Result<Self> {
        if problem.dtype() != T::STORAGE {
            return Err(ConfigError::DtypeMismatch {
                plan: T::STORAGE.name(),
                problem: problem.dtype().name(),
            }
            .into());
        }
        config.validate()?;
        let packed = config.requires_packed() || selection.is_some();
        let strategy = if packed {
            Strategy::Packed(Box::new(Self::plan_packed(problem, config, selection)?))
        } else if problem.all_batch() {
            Strategy::Elementwise(ElementPlan::product(problem))
        } else if let Some(f) = faer_strategy::plan(problem) {
            Strategy::Faer(f)
        } else {
            Strategy::Packed(Box::new(Self::plan_packed(problem, config, None)?))
        };
        let (algorithm, packed_report) = match &strategy {
            Strategy::Packed(p) => (Algorithm::Packed, Some(packed_report(p))),
            Strategy::Faer(_) => (Algorithm::Faer, None),
            Strategy::Elementwise(_) => (Algorithm::Elementwise, None),
        };
        Ok(Self {
            output: ElementPlan::output(problem),
            report: PlanReport {
                algorithm,
                materialized: [false; 3],
                packed: packed_report,
            },
            diagnostics: Diagnostics::new("tprims-contract", algorithm.name()),
            problem: problem.clone(),
            strategy,
            workspace: ArenaProvider::new(),
            _t: PhantomData,
        })
    }

    fn plan_packed(
        problem: &Problem,
        config: &PlanConfig,
        selection: Option<(&KernelCatalog<T>, &mut Chooser<'_, T>)>,
    ) -> Result<Packed<T>> {
        let plan = PackedPlan::from_problem(problem, config)?;
        let handle = match selection {
            Some((catalog, selector)) => Some(crate::select::choose::<T>(
                problem, &plan, config, catalog, selector,
            )?),
            None => None,
        };
        let rg = resolve::resolve::<T>(&plan, config, handle.as_ref())?;
        Ok(Packed { plan, rg })
    }

    /// What this plan decided: the algorithm and, for the packed strategy, the
    /// family, blocking and grid. A lookup, not a re-selection.
    pub fn report(&self) -> &PlanReport {
        &self.report
    }

    /// The validated problem this plan was built from.
    pub fn problem(&self) -> &Problem {
        &self.problem
    }

    fn check_layout(&self, which: OperandId, dims: &[usize], strides: &[isize]) -> Result<()> {
        let planned = match which {
            OperandId::A => self.problem.a().layout(),
            OperandId::B => self.problem.b().layout(),
            OperandId::D => self.problem.d().layout(),
            OperandId::C => match self.problem.c_spec() {
                CSpec::Separate(c) => c.layout(),
                _ => self.problem.d().layout(),
            },
        };
        // The offset is part of the problem's address-range check; the view
        // brings its own origin and has validated its own storage.
        if planned.dims() != dims || planned.strides() != strides {
            return Err(LayoutError::Mismatch { operand: which }.into());
        }
        Ok(())
    }

    /// `D = op_D(alpha * dot_general(op_A(A), op_B(B)))`: the overwrite form.
    /// No previous output value is read.
    ///
    /// # Errors
    ///
    /// [`LayoutError::Mismatch`] when a view differs from the plan (nothing is
    /// written); [`Error::Exec`] or [`Error::Backend`] from a lower layer.
    pub fn execute_into(
        &self,
        exec: &Exec<'_>,
        alpha: T,
        a: &StridedView<'_, T>,
        b: &StridedView<'_, T>,
        d: &mut StridedViewMut<'_, T>,
    ) -> Result<()> {
        self.validate_views(a, b, None, d)?;
        // SAFETY: the views' layouts equal the plan's validated layouts, so
        // their origins address every offset the plan generates; `d` is an
        // exclusive borrow and cannot alias `a` or `b`.
        unsafe {
            self.run(
                exec,
                alpha,
                a.ptr(),
                b.ptr(),
                <T as Element>::zero(),
                CRead::None,
                d.as_mut_ptr(),
            )
        }
    }

    /// `D = op_D(alpha * dot_general(op_A(A), op_B(B)) + beta * op_C(C))`.
    ///
    /// `beta == 0` reads no previous C value; `alpha == 0` or an empty
    /// contraction reads no A or B value. `source` must match the planned C
    /// mode: [`AccumulationSource::Output`] only when C maps to D (a problem
    /// built with [`CSpec::Output`]), and [`AccumulationSource::Separate`] only
    /// for a separately described C.
    ///
    /// # Errors
    ///
    /// As [`Plan::execute_into`], and [`LayoutError::CMode`] for a source that
    /// does not match the planned C mode (including any accumulation from a
    /// problem built with [`CSpec::Absent`]).
    pub fn execute_into_accum(
        &self,
        exec: &Exec<'_>,
        alpha: T,
        a: &StridedView<'_, T>,
        b: &StridedView<'_, T>,
        beta: T,
        source: AccumulationSource<'_, T>,
        d: &mut StridedViewMut<'_, T>,
    ) -> Result<()> {
        let c = self.validate_views(a, b, Some(source), d)?;
        // SAFETY: as `execute_into`; a separate C is an immutable borrow
        // distinct from the exclusive `d`.
        unsafe { self.run(exec, alpha, a.ptr(), b.ptr(), beta, c, d.as_mut_ptr()) }
    }

    /// Check the views of one operation against the plan, before any write,
    /// and say where the accumulation term is read from.
    ///
    /// `source` is `None` for the overwrite form.
    pub(crate) fn validate_views(
        &self,
        a: &StridedView<'_, T>,
        b: &StridedView<'_, T>,
        source: Option<AccumulationSource<'_, T>>,
        d: &StridedViewMut<'_, T>,
    ) -> Result<CRead<T>> {
        self.check_layout(OperandId::A, a.dims(), a.strides())?;
        self.check_layout(OperandId::B, b.dims(), b.strides())?;
        self.check_layout(OperandId::D, d.dims(), d.strides())?;
        Ok(match (self.problem.c_spec(), source) {
            (_, None) => CRead::None,
            (CSpec::Output(_), Some(AccumulationSource::Output)) => CRead::InPlace,
            (CSpec::Separate(_), Some(AccumulationSource::Separate(c))) => {
                self.check_layout(OperandId::C, c.dims(), c.strides())?;
                CRead::Separate(c.ptr())
            }
            _ => return Err(LayoutError::CMode.into()),
        })
    }

    /// Run one item whose views [`Plan::validate_views`] has accepted.
    ///
    /// # Safety
    ///
    /// The views are the ones that were validated, `d` is exclusive, and a
    /// separate C is distinct from `d`.
    pub(crate) unsafe fn run_validated(
        &self,
        exec: &Exec<'_>,
        alpha: T,
        a: &StridedView<'_, T>,
        b: &StridedView<'_, T>,
        beta: T,
        c: CRead<T>,
        d: &mut StridedViewMut<'_, T>,
    ) -> Result<()> {
        // SAFETY: forwarded.
        unsafe { self.run(exec, alpha, a.ptr(), b.ptr(), beta, c, d.as_mut_ptr()) }
    }

    /// Execute on raw origins: the pointers are the elements at logical index
    /// zero of each operand. This is the entry of the C adapter, which holds
    /// pointers and no Rust references.
    ///
    /// The same semantic preflight as the safe path runs first: `D`
    /// overlapping an input, `C` and `D` overlapping without being the same
    /// mapping at the same origin. Input errors leave `D` untouched. Null
    /// pointers are the caller's to refuse (they are an FFI status, not a
    /// contraction error): `a` and `b` must be non-null unless the call reads
    /// neither, `c` unless it reads a separate C, and `d` unless the output is
    /// empty.
    ///
    /// `c` is read only for a problem built with [`CSpec::Separate`] and a
    /// nonzero `beta`; for [`CSpec::Output`] the previous `D` is read through
    /// `d`.
    ///
    /// # Safety
    ///
    /// Every pointer the call reads or writes is the origin of a live allocation that covers the
    /// problem's address range for that operand (see
    /// [`Problem::span`](crate::api::Problem::span), relative to the operand's
    /// logical offset); `d` is valid for writes and nothing else accesses any of
    /// the memory for the duration of the call.
    #[allow(clippy::too_many_arguments)] // INVARIANT: the contraction argument set.
    pub unsafe fn execute_raw(
        &self,
        exec: &Exec<'_>,
        alpha: T,
        a: *const T,
        b: *const T,
        beta: T,
        c: *const T,
        d: *mut T,
    ) -> Result<()> {
        let p = &self.problem;
        if p.out_empty() {
            return Ok(());
        }
        let zero = <T as Element>::zero();
        let reads_ab = !(p.k_empty() || alpha == zero);
        let c_read = match p.c_spec() {
            CSpec::Absent => CRead::None,
            CSpec::Output(_) => CRead::InPlace,
            CSpec::Separate(_) if beta == zero => CRead::None,
            CSpec::Separate(_) => CRead::Separate(c),
        };
        let range = |o: OperandId, origin: *const T| -> Option<(usize, usize)> {
            let span = p.span(o)?;
            let off = match o {
                OperandId::A => p.a().layout().offset(),
                OperandId::B => p.b().layout().offset(),
                OperandId::D => p.d().layout().offset(),
                OperandId::C => match p.c_spec() {
                    CSpec::Separate(c) => c.layout().offset(),
                    _ => p.d().layout().offset(),
                },
            } as i128;
            let base = origin as usize as i128;
            let size = core::mem::size_of::<T>() as i128;
            let lo = base + (span.lo() - off) * size;
            let hi = base + (span.hi() - off + 1) * size;
            Some((usize::try_from(lo).ok()?, usize::try_from(hi).ok()?))
        };
        let overlap = |x: Option<(usize, usize)>, y: Option<(usize, usize)>| match (x, y) {
            (Some(x), Some(y)) => x.0 < y.1 && y.0 < x.1,
            _ => false,
        };
        let rd = range(OperandId::D, d);
        if reads_ab {
            if overlap(rd, range(OperandId::A, a)) {
                return Err(AliasError::OutputOverlapsInput {
                    operand: OperandId::A,
                }
                .into());
            }
            if overlap(rd, range(OperandId::B, b)) {
                return Err(AliasError::OutputOverlapsInput {
                    operand: OperandId::B,
                }
                .into());
            }
        }
        if let CRead::Separate(cp) = c_read {
            // A separate C may be D itself only as the same mapping at the
            // same origin (an in-place update).
            let same = core::ptr::eq(cp, d as *const T) && p.c_matches_d();
            if !same && overlap(rd, range(OperandId::C, cp)) {
                return Err(AliasError::CDOverlap.into());
            }
        }
        // SAFETY: preflight passed; the caller's contract covers the rest.
        unsafe { self.run(exec, alpha, a, b, beta, c_read, d) }
    }

    /// The width the work estimate asks for at the executor's budget.
    fn width(&self, exec: &Exec<'_>) -> usize {
        let flops = 2.0 * self.problem.macs() as f64 * if T::IS_COMPLEX { 4.0 } else { 1.0 };
        exec.width_for(flops * NS_PER_FLOP, &WidthPolicy::default())
    }

    /// After preflight: output-empty returns; `alpha == 0` / empty `K` uses the
    /// output-update helper; otherwise one dispatch to the prepared strategy.
    ///
    /// # Safety
    ///
    /// As [`Plan::execute_raw`], with the preflight already done.
    #[allow(clippy::too_many_arguments)] // INVARIANT: the contraction argument set.
    unsafe fn run(
        &self,
        exec: &Exec<'_>,
        alpha: T,
        a: *const T,
        b: *const T,
        beta: T,
        c: CRead<T>,
        d: *mut T,
    ) -> Result<()> {
        let p = &self.problem;
        if p.out_empty() {
            return Ok(());
        }
        let zero = <T as Element>::zero();
        let (conj_a, conj_b) = (p.a().op().is_conj(), p.b().op().is_conj());
        let (conj_c, conj_d) = (p.op_c().is_conj(), p.d().op().is_conj());
        // No C term means beta is zero, whatever the caller passed.
        let beta = if matches!(c, CRead::None) { zero } else { beta };
        let expr = Expr {
            alpha,
            beta,
            conj_a,
            conj_b,
            conj_c,
            conj_d,
        };
        if p.k_empty() || alpha == zero {
            // D = op_D(beta * op_C(C)); A and B are not referenced. An
            // in-place identity update changes nothing.
            if matches!(c, CRead::InPlace) && beta == <T as Element>::one() && !conj_c && !conj_d {
                return Ok(());
            }
            // SAFETY: the caller's contract.
            unsafe { self.output.run(exec, expr, Inputs::None, c, d) };
            return Ok(());
        }
        match &self.strategy {
            Strategy::Packed(pk) => {
                let exec = exec.with_budget(self.width(exec)).unwrap_or(*exec);
                // A borrowed pool lends its own arena; a serial context uses the
                // plan's, which is why a serial plan's steady state allocates
                // nothing either.
                let workspace: &dyn WorkspaceProvider = exec.workspace().unwrap_or(&self.workspace);
                let cp = match c {
                    CRead::Separate(c) => c,
                    _ => d as *const T,
                };
                // SAFETY: the caller's contract; `pk.rg` was validated for this
                // plan when it was built.
                unsafe {
                    driver::execute_packed(
                        &pk.plan,
                        &pk.rg,
                        &exec,
                        Some(workspace),
                        alpha,
                        a,
                        b,
                        beta,
                        cp,
                        d,
                    )
                };
            }
            Strategy::Faer(f) => {
                // SAFETY: the caller's contract; the fusion was proven copy-free
                // over exactly this problem's layouts.
                unsafe { f.run(exec, alpha, a, b, beta, d) };
            }
            Strategy::Elementwise(e) => {
                // SAFETY: the caller's contract.
                unsafe { e.run(exec, expr, Inputs::Product(a, b), c, d) };
            }
        }
        Ok(())
    }
}

fn packed_report<T: Scalar>(pk: &Packed<T>) -> PackedReport {
    let rg = &pk.rg;
    let fam = rg.family();
    let element = core::mem::size_of::<<T as Scalar>::Re>();
    let a_reals = fam.a_per_k / fam.mr;
    let b_reals = fam.b_per_k / fam.nr;
    let scratch_bytes = (rg.mc * rg.kc * a_reals + rg.nc * rg.kc * b_reals) * element;
    PackedReport {
        family_id: fam.id,
        origin: fam.origin,
        complex: fam.complex,
        mr: rg.mr,
        nr: rg.nr,
        mc: rg.mc,
        nc: rg.nc,
        kc: rg.kc,
        partition: rg.partition,
        align_c_lines: rg.opts.align_c_lines,
        dynamic: driver::dynamic_report(&pk.plan, rg, usize::MAX),
        stats: pk.plan.stats.clone(),
        scratch_bytes,
    }
}

impl<T: Scalar> PreparedContraction<T> for Plan<T> {
    fn execute_into(
        &self,
        exec: &Exec<'_>,
        alpha: T,
        a: &StridedView<'_, T>,
        b: &StridedView<'_, T>,
        d: &mut StridedViewMut<'_, T>,
    ) -> Result<()> {
        Plan::execute_into(self, exec, alpha, a, b, d)
    }

    fn execute_into_accum(
        &self,
        exec: &Exec<'_>,
        alpha: T,
        a: &StridedView<'_, T>,
        b: &StridedView<'_, T>,
        beta: T,
        source: AccumulationSource<'_, T>,
        d: &mut StridedViewMut<'_, T>,
    ) -> Result<()> {
        Plan::execute_into_accum(self, exec, alpha, a, b, beta, source, d)
    }

    fn diagnostics(&self) -> &Diagnostics {
        &self.diagnostics
    }
}
