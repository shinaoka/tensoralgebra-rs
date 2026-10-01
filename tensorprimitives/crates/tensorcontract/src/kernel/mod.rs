//! Kernel selection glue between tensorcontract plans and the kernel layer.
pub use tprims_kernel::blocking as cache;
#[cfg(target_arch = "aarch64")]
#[doc(hidden)]
pub use tprims_kernel::kernels::aarch64;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[doc(hidden)]
pub use tprims_kernel::kernels::x86;
pub use tprims_kernel::kernels::{reference::scalar, KernelSet};
pub use tprims_kernel::{
    Blocking, ComplexMethod, KernelConfig, PackFormat, ParseError, TileFormat, Ukr,
};

/// The register block `(MR, NR)` and cache blocking the engine will use for
/// element type `T`. Exposed for diagnostics and for harnesses that want to
/// report block-scatter regularity at the same granularity the engine sees.
pub fn selected_config<T>(method: ComplexMethod) -> (usize, usize, Blocking)
where
    T: crate::element::Element,
    T::Real: KernelSet,
{
    let cfg = config_for::<T>(method);
    (cfg.ukr.mr, cfg.ukr.nr, cfg.blk)
}

/// Name of the micro-kernel selected for `T`.
pub fn selected_kernel_name<T>(method: ComplexMethod) -> &'static str
where
    T: crate::element::Element,
    T::Real: KernelSet,
{
    config_for::<T>(method).ukr.name
}

/// Pick the kernel configuration for an element type and complex method.
/// `method` is ignored for real element types.
pub(crate) fn config_for<T>(method: ComplexMethod) -> KernelConfig<T::Real>
where
    T: crate::element::Element,
    T::Real: KernelSet,
{
    if T::IS_COMPLEX {
        <T::Real as KernelSet>::config_cplx(method)
    } else {
        <T::Real as KernelSet>::config_real()
    }
}

/// [`config_for`] plus the plan's choice of micro-tile row block.
///
/// Used while resolving the legacy menu (and directly for foreign scalars).
/// Built-in numerical execution consumes the cached descriptor instead.
/// It differs from [`config_for`] only when the
/// output's stride pattern makes a shape other than the kernel set's default
/// worth having — see [`crate::Plan::row_block`] — and falls back to the
/// default whenever the requested shape does not exist.
///
/// The plan's thread count is applied to the blocking here, because `nc` is a
/// share of a cache the threads of one call contend for and the process default
/// is not necessarily what this plan runs with. Under the legacy derivation this
/// changes nothing.
pub(crate) fn config_for_plan<T>(plan: &crate::plan::Plan) -> KernelConfig<T::Real>
where
    T: crate::element::Element,
    T::Real: KernelSet,
{
    let method = plan.complex_method();
    let menu = <T::Real as KernelSet>::row_blocks(T::IS_COMPLEX, method);
    plan.row_block(menu)
        .and_then(|i| <T::Real as KernelSet>::config_at(T::IS_COMPLEX, method, i))
        .unwrap_or_else(|| config_for::<T>(method))
        .retarget_threads(plan.threads())
}

/// Legacy KernelSet menu's register block and cache blocking for a plan.
///
/// This preserves the legacy diagnostic API and excludes forced registered
/// choices and explicit plan blocking. For the canonical typed configuration
/// (including registered families), use [`crate::Plan::resolved`].
pub fn plan_config<T>(plan: &crate::plan::Plan) -> (usize, usize, Blocking)
where
    T: crate::element::Element,
    T::Real: KernelSet,
{
    let cfg = config_for_plan::<T>(plan);
    (cfg.ukr.mr, cfg.ukr.nr, cfg.blk)
}
