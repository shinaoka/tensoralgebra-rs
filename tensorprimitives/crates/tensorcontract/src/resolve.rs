//! Typed plan-owned family caches. No workspace or ambient executor lives here.
use crate::{ComplexMethod, Element, KernelSet, Plan};
use std::{
    any::{Any, TypeId},
    sync::OnceLock,
};
use tprims_gemm_kernel::{
    CpuFeatures, Families, Isa, KernelChoice, KernelImpl, Method, Origin, Registry, ResolvedGemm,
    SelectError, C32, C64,
};

#[derive(Debug, Default)]
pub(crate) struct Cache {
    real32: OnceLock<Result<ResolvedGemm<f32>, SelectError>>,
    complex32: OnceLock<Result<ResolvedGemm<f32>, SelectError>>,
    real64: OnceLock<Result<ResolvedGemm<f64>, SelectError>>,
    complex64: OnceLock<Result<ResolvedGemm<f64>, SelectError>>,
}
impl Clone for Cache {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl Cache {
    pub(crate) fn resolved<T: Families>(
        &self,
        p: &Plan,
    ) -> Result<ResolvedGemm<T::Real>, SelectError>
    where
        T::Real: KernelSet,
    {
        let result: &dyn Any = if TypeId::of::<T>() == TypeId::of::<f32>() {
            self.real32.get_or_init(|| resolve::<f32>(p))
        } else if TypeId::of::<T>() == TypeId::of::<C32>() {
            self.complex32.get_or_init(|| resolve::<C32>(p))
        } else if TypeId::of::<T>() == TypeId::of::<f64>() {
            self.real64.get_or_init(|| resolve::<f64>(p))
        } else {
            self.complex64.get_or_init(|| resolve::<C64>(p))
        };
        // INVARIANT: Families is sealed to these four storage types. Their
        // Element impls bind Real to the exact cache real type chosen above.
        result
            .downcast_ref::<Result<ResolvedGemm<T::Real>, SelectError>>()
            .expect("sealed dtype cache real type")
            .clone()
    }
}

pub(crate) fn default_choice() -> &'static KernelChoice {
    KernelChoice::from_env()
}

fn legacy_isa() -> Isa {
    use tprims_kernel_tensorcontract::{kernel_force, KernelForce};
    if kernel_force() == KernelForce::Scalar {
        return Isa::Portable;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    return match tprims_kernel_tensorcontract::x86::selected_isa() {
        Some(tprims_kernel_tensorcontract::x86::Isa::Avx2) => Isa::Avx2,
        Some(tprims_kernel_tensorcontract::x86::Isa::Avx512) => Isa::Avx512,
        None => Isa::Portable,
    };
    #[cfg(target_arch = "aarch64")]
    return Isa::Neon;
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    Isa::Portable
}

fn method_kind(method: ComplexMethod) -> Method {
    match method {
        ComplexMethod::Planar => Method::Native,
        ComplexMethod::OneM => Method::OneM,
        ComplexMethod::ThreeM => Method::ThreeM,
    }
}

fn resolve<T: Families>(p: &Plan) -> Result<ResolvedGemm<T::Real>, SelectError>
where
    T::Real: KernelSet,
{
    p.freeze_execution_switches();
    tprims_kernel_tensorcontract::register();
    let choice = p.kernel.as_ref().unwrap_or_else(|| default_choice());
    let legacy_auto = matches!(choice, KernelChoice::Auto);
    let default = if p.kernel.is_none() && p.method.is_none() {
        Some(tprims_gemm_kernel::process_default::<T>()?)
    } else {
        None
    };
    let chosen;
    let choice = if legacy_auto {
        let method = method_kind(p.complex_method());
        let isa = legacy_isa();
        let candidates = Registry::families::<T>(CpuFeatures::detect(), false);
        let candidates: Vec<_> = candidates
            .into_iter()
            .filter(|f| {
                // The legacy menu is the compiled KernelSet list; an induced
                // variant of it is a different family that only an explicit id
                // may choose.
                f.imp != KernelImpl::Induced
                    && f.origin == Origin::Tensorcontract
                    && f.isa == isa
                    && (!T::IS_COMPLEX || f.complex.is_some_and(|s| s.method == method))
            })
            .collect();
        let menu = <T::Real as KernelSet>::row_blocks(T::IS_COMPLEX, p.complex_method());
        let shape = p.row_block(menu).map(|i| menu[i]);
        let family = candidates
            .iter()
            .find(|f| shape.is_none_or(|(mr, nr)| (f.mr, f.nr) == (mr, nr)))
            .or_else(|| candidates.first());
        let family = family.ok_or_else(|| SelectError::Incompatible {
            id: "Auto".into(),
            reason: "legacy default family is unavailable",
        })?;
        chosen = KernelChoice::Id(family.id.into());
        &chosen
    } else {
        choice
    };
    let mut rg =
        match default.filter(|r| matches!(choice, KernelChoice::Id(id) if id == r.family().id)) {
            Some(r) => r.with_threads(p.threads())?,
            None => ResolvedGemm::<T::Real>::resolve::<T>(choice, p.threads())?,
        };
    if T::IS_COMPLEX
        && p.method.is_some_and(|m| {
            rg.family()
                .complex
                .is_none_or(|s| s.method != method_kind(m))
        })
    {
        return Err(SelectError::Incompatible {
            id: rg.family().id.into(),
            reason: "family implements a different complex method",
        });
    }
    // A family whose kernel arm the driver cannot express (unsupported complex
    // scheme) must never reach execution; validation rejects it the same way
    // unsupported descriptors are rejected at registration.
    if rg.family().driver_family().is_none() {
        return Err(SelectError::Incompatible {
            id: rg.family().id.into(),
            reason: "family has no driver-expressible kernel",
        });
    }
    if (p.conj_a && !rg.family().caps.conj_a) || (p.conj_b && !rg.family().caps.conj_b) {
        return Err(SelectError::Incompatible {
            id: rg.family().id.into(),
            reason: "operand conjugation unsupported",
        });
    }
    if legacy_auto
        && tprims_gemm_kernel::cache::block_model() == tprims_gemm_kernel::cache::BlockModel::Legacy
    {
        // Preserve old percentage-before-register-rounding semantics. The
        // canonical resolver above already checked override multiplication;
        // the legacy raw seed is no larger than its aligned descriptor seed.
        // This runs once in planning, never in the built-in execute path.
        rg = rg.with_blocking(crate::kernel::config_for_plan::<T>(p).blk)?;
    }
    if let Some(blk) = p.blocking {
        rg = rg.with_blocking(blk)?;
    }
    Ok(rg)
}

pub(crate) fn validate<T: Element>(p: &Plan) -> crate::Result<()>
where
    T::Real: KernelSet,
{
    if let Some(result) = builtin::<T>(p) {
        // Serial has the largest model NC; validating it also bounds every
        // smaller active-width retarget before the raw driver allocates.
        result
            .and_then(|rg| rg.with_threads(1))
            .map(|_| ())
            .map_err(crate::Error::KernelSelection)
    } else if let KernelChoice::Id(id) = p.kernel.as_ref().unwrap_or_else(|| default_choice()) {
        Err(crate::Error::KernelSelection(SelectError::Incompatible {
            id: id.clone(),
            reason: "registered families require a built-in storage dtype",
        }))
    } else {
        Ok(())
    }
}

/// Built-in dispatch without narrowing existing Element/KernelSet execution.
/// Foreign scalars return None and continue through their existing KernelSet.
pub(crate) fn builtin<T: Element>(p: &Plan) -> Option<Result<ResolvedGemm<T::Real>, SelectError>>
where
    T::Real: KernelSet,
{
    macro_rules! dtype {
        ($t:ty) => {
            if TypeId::of::<T>() == TypeId::of::<$t>() {
                let result = p.resolved::<$t>();
                let result: &dyn Any = &result;
                // INVARIANT: exact storage TypeId implies the fixed Element
                // impl's Real type; downcast is safe and allocation-free.
                return Some(
                    result
                        .downcast_ref::<Result<ResolvedGemm<T::Real>, SelectError>>()
                        .expect("built-in dtype real type")
                        .clone(),
                );
            }
        };
    }
    dtype!(f32);
    dtype!(f64);
    dtype!(C32);
    dtype!(C64);
    None
}
