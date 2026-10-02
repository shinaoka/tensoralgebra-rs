//! Safe user-defined kernel selection (tprims addition, issue #28).
//!
//! [`Plan::with_selector`] lets a caller pass their own
//! [`KernelCatalog`] together with a selection callback. The callback receives
//! metadata only — a [`SelectionContext`] and one [`KernelCandidate`] per
//! admissible family — and returns an opaque [`KernelHandle`]. The plan keeps
//! that trusted handle; execution never calls the callback, consults the
//! catalog or looks an id up again.
//!
//! The candidate facts are computed here, with the same helpers the driver
//! uses at execution (`Plan::transposes_gemm`, `Plan::partition_with`, the
//! direct-B rule), so what a selector is shown is what the driver will do.

use crate::{
    driver, kernel::ComplexMethod, Element, Error, KernelSet, Operand, Plan, PlanStats, Result,
};
use core::any::Any;
use tprims_kernel::{
    CpuFeatures, Families, KernelCatalog, KernelHandle, Method, PartitionPolicy, SelectError,
};

/// What a selector returns: the trusted handle it chose, or why it declined.
pub type Selection<T> = core::result::Result<KernelHandle<T>, SelectError>;

/// A selector as the planners hold it while building one plan.
pub type Chooser<'a, T> =
    dyn FnMut(&SelectionContext<'_>, &[KernelCandidate<T>]) -> Selection<T> + 'a;

/// One operand's layout as the caller described it, borrowed from the plan's
/// construction arguments (nothing tensor-sized is copied).
#[derive(Clone, Copy, Debug)]
pub struct OperandMeta<'a> {
    /// Original extents.
    pub extents: &'a [i64],
    /// Original signed element strides (zero strides are broadcasts).
    pub strides: &'a [i64],
    /// Index labels, one per mode.
    pub labels: &'a [i64],
    /// Whether the operand is read conjugated.
    pub conj: bool,
}

/// Problem-level facts shown to a selector. Candidate-specific consequences
/// (orientation, direct B, active width) are in [`KernelCandidate`], because
/// they depend on each candidate's geometry.
///
/// Bounds and pointer-aliasing facts (for example whether `C` and `D` are the
/// same buffer) are not known when a plan is built; they stay execution
/// guards, not selector guarantees.
#[derive(Clone, Copy, Debug)]
pub struct SelectionContext<'a> {
    /// Storage scalar name: `"f32"`, `"f64"`, `"c32"` or `"c64"`.
    pub dtype: &'static str,
    /// Whether the storage type is complex.
    pub is_complex: bool,
    /// Folded matrix shape, batch count and axis structure the index analysis
    /// reached (`m`, `n`, `k`, `batch`, `is_pure_gemm`, folded axes).
    pub stats: &'a PlanStats,
    /// A, B, C and D. A plan without a C operand reports D's layout as C.
    pub operands: [OperandMeta<'a>; 4],
    /// Requested complex method, or none for the family's own.
    pub method: Option<ComplexMethod>,
    /// Host thread budget of the plan.
    pub threads: usize,
    /// Grid policy the plan will freeze.
    pub partition: PartitionPolicy,
    /// CPU/OS instruction sets available; candidates already satisfy it.
    pub cpu: CpuFeatures,
    /// Whether the problem has no work (a zero extent); the selection and its
    /// validation still run.
    pub is_empty: bool,
}

/// One admissible family with facts derived under *its own* geometry.
#[derive(Clone, Copy, Debug)]
pub struct KernelCandidate<T: Families> {
    /// The handle to return to select this family; also carries its metadata.
    pub handle: KernelHandle<T>,
    /// Whether the driver exchanges (A, M) with (B, N) for this family's `MR`.
    pub swapped: bool,
    /// Whether this family reads the column operand in place (no B buffer).
    pub direct_b: bool,
    /// Threads that would actually run at the plan's budget (`pm * pn`).
    pub active_width: usize,
    /// The `(pm, pn)` grid behind `active_width`.
    pub grid: (usize, usize),
}

/// The family a selector chose, per storage dtype (selection is typed).
#[derive(Clone, Copy, Debug)]
pub(crate) enum Forced {
    F32(KernelHandle<f32>),
    F64(KernelHandle<f64>),
    C32(KernelHandle<crate::C32>),
    C64(KernelHandle<crate::C64>),
}

impl Forced {
    fn new<T: Families>(handle: KernelHandle<T>) -> Forced {
        let any: &dyn Any = &handle;
        if let Some(h) = any.downcast_ref::<KernelHandle<f32>>() {
            Forced::F32(*h)
        } else if let Some(h) = any.downcast_ref::<KernelHandle<f64>>() {
            Forced::F64(*h)
        } else if let Some(h) = any.downcast_ref::<KernelHandle<crate::C32>>() {
            Forced::C32(*h)
        } else {
            // INVARIANT: Families is sealed to the four storage dtypes.
            Forced::C64(
                *any.downcast_ref::<KernelHandle<crate::C64>>()
                    .expect("sealed storage dtype"),
            )
        }
    }

    pub(crate) fn id(&self) -> &'static str {
        match self {
            Forced::F32(h) => h.id(),
            Forced::F64(h) => h.id(),
            Forced::C32(h) => h.id(),
            Forced::C64(h) => h.id(),
        }
    }

    /// The handle, if the plan is being resolved for its own dtype.
    pub(crate) fn handle<T: Families>(&self) -> core::result::Result<KernelHandle<T>, SelectError> {
        let (any, id): (&dyn Any, &'static str) = match self {
            Forced::F32(h) => (h, h.id()),
            Forced::F64(h) => (h, h.id()),
            Forced::C32(h) => (h, h.id()),
            Forced::C64(h) => (h, h.id()),
        };
        any.downcast_ref::<KernelHandle<T>>()
            .copied()
            .ok_or_else(|| SelectError::DtypeMismatch {
                id: id.into(),
                dtype: T::DTYPE,
            })
    }
}

pub(crate) fn method_kind(method: ComplexMethod) -> Method {
    match method {
        ComplexMethod::Planar => Method::Native,
        ComplexMethod::OneM => Method::OneM,
        ComplexMethod::ThreeM => Method::ThreeM,
    }
}

/// Why `handle` cannot serve `plan`, using the checks a built-in resolution
/// applies; `None` when it can.
pub(crate) fn inadmissible<T: Families>(
    plan: &Plan,
    handle: &KernelHandle<T>,
) -> Option<&'static str> {
    if !handle.caps().scatter_pack {
        return Some("the packed driver needs scatter packing");
    }
    // Mirrors the built-in resolution: conjugation is checked against the
    // plan's operands as the user wrote them.
    if (plan.conj_a && !handle.caps().conj_a) || (plan.conj_b && !handle.caps().conj_b) {
        return Some("operand conjugation unsupported");
    }
    if T::IS_COMPLEX {
        if let Some(m) = plan.method {
            if handle.complex().is_none_or(|s| s.method != method_kind(m)) {
                return Some("family implements a different complex method");
            }
        }
    }
    None
}

fn candidate<T: Families>(plan: &Plan, handle: KernelHandle<T>) -> KernelCandidate<T> {
    let (mr, nr) = (handle.mr(), handle.nr());
    let swapped = plan.transposes_gemm(mr);
    let (bk, bn) = if swapped {
        (&plan.a_k, &plan.a_m)
    } else {
        (&plan.b_k, &plan.b_n)
    };
    let direct_b = !driver::pack_b_needed(handle.b_access(), bk, bn, nr);
    let (mut pm, mut pn) = plan.partition_with(mr, nr, plan.threads());
    // The driver turns a direct-B grid into a pure split of N, same width.
    if direct_b {
        pn *= pm;
        pm = 1;
    }
    KernelCandidate {
        handle,
        swapped,
        direct_b,
        active_width: pm * pn,
        grid: (pm, pn),
    }
}

fn operand_meta(o: Operand<'_>) -> OperandMeta<'_> {
    OperandMeta {
        extents: o.layout.extents(),
        strides: o.layout.strides(),
        labels: o.idx,
        conj: o.op.is_conj(),
    }
}

impl Plan {
    /// Choose this plan's kernel family with a caller-supplied selector over a
    /// caller-supplied [`KernelCatalog`].
    ///
    /// Call this **last**: the selection is made once, here, for the storage
    /// dtype `T`, and the plan keeps only the chosen trusted handle. Thread
    /// count, blocking and complex-method builders applied afterwards still
    /// apply, and are re-checked against the chosen family when the plan is
    /// resolved. The plan then runs only for `T`; resolving it for another
    /// dtype is a typed `DtypeMismatch`.
    ///
    /// `operands` are the A, B, C, D descriptions the plan was built from (C
    /// repeats D when there is none); they are reported to the selector as
    /// metadata only and are not trusted for anything.
    ///
    /// The selector runs synchronously on this thread, outside every lock and
    /// worker broadcast, and is not retained. It need not be `Send`, `Sync` or
    /// `'static`. A panic in it unwinds out of this call before any compute.
    ///
    /// # Errors
    /// `KernelSelection` with: `NoCandidates` when the catalog has nothing
    /// admissible (CPU, conjugation, complex method, scatter packing);
    /// whatever the selector returned in `Err`, unchanged (no fallback is
    /// substituted); `ForeignHandle` for a handle another catalog minted;
    /// `CpuUnsupported` or `NotACandidate` for a catalog member that was not
    /// among the offered candidates; or `Incompatible` for a plan that already
    /// carries a forced kernel id (ambiguous with a selector).
    pub fn with_selector<T, F>(
        mut self,
        operands: [Operand<'_>; 4],
        catalog: &KernelCatalog<T>,
        selector: F,
    ) -> Result<Plan>
    where
        T: Element + Families,
        T::Real: KernelSet,
        F: FnOnce(
            &SelectionContext<'_>,
            &[KernelCandidate<T>],
        ) -> core::result::Result<KernelHandle<T>, SelectError>,
    {
        if let Some(tprims_kernel::KernelChoice::Id(id)) = &self.kernel {
            return Err(Error::KernelSelection(SelectError::Incompatible {
                id: id.clone(),
                reason: "a forced kernel id and a custom selector are ambiguous",
            }));
        }
        let cpu = CpuFeatures::detect();
        let candidates: Vec<KernelCandidate<T>> = catalog
            .handles()
            .filter(|h| cpu.contains(h.required()) && inadmissible(&self, h).is_none())
            .map(|h| candidate(&self, h))
            .collect();
        if candidates.is_empty() {
            return Err(Error::KernelSelection(SelectError::NoCandidates {
                dtype: T::DTYPE,
            }));
        }
        let context = SelectionContext {
            dtype: T::DTYPE,
            is_complex: T::IS_COMPLEX,
            stats: &self.stats,
            operands: operands.map(operand_meta),
            method: self.method,
            threads: self.threads(),
            partition: PartitionPolicy::default(),
            cpu,
            is_empty: self.is_empty(),
        };
        let chosen = selector(&context, &candidates).map_err(Error::KernelSelection)?;
        catalog
            .admit(&chosen, cpu)
            .map_err(Error::KernelSelection)?;
        if let Some(reason) = inadmissible(&self, &chosen) {
            return Err(Error::KernelSelection(SelectError::NotACandidate {
                id: chosen.id().into(),
                reason,
            }));
        }
        self.forced = Some(Forced::new(chosen));
        self.resolved_cache = Default::default();
        // Freeze and validate now (blocking overflow, serial NC bound), so a
        // chosen family that cannot run is an error from this call.
        crate::resolve::validate::<T>(&self)?;
        Ok(self)
    }
}
