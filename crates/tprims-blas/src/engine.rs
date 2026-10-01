//! Which engine computes a GEMM, and what the planner chose.
//!
//! Three engines can compute the same product:
//!
//! * [`Engine::Faer`] — the workspace's long-standing matrix engine, whose
//!   blocking and threading are faer's;
//! * [`Engine::PrivateGemmX86`] — `private-gemm-x86` called directly
//!   ([`tprims_kernel_pgx86`]), on x86-64 with AVX2 and FMA only;
//! * [`Engine::Packed`] — the packed micro-kernel driver of
//!   `tensorcontract`, which is where a *named* kernel family, the direct-C
//!   guard and the workspace live.
//!
//! `Auto` keeps today's behaviour: faer for a matrix GEMM. The other two are
//! opt-in, per call ([`GemmConfig::engine`]) or process-wide
//! (`TPRIMS_GEMM_ENGINE`).
use strided_view::StridedViewMut;
use tprims_exec::Exec;
/// Re-exported because it is part of [`GemmConfig`]'s public surface.
pub use tprims_gemm_kernel::KernelChoice;

use crate::scalar::zero;
use crate::{Error, MatIn, Result, Scalar};

/// Which engine a GEMM runs on.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum EngineChoice {
    /// The process default: faer, unless `TPRIMS_GEMM_ENGINE` says otherwise.
    #[default]
    Auto,
    /// faer's `matmul`, with the operand views as they are.
    Faer,
    /// `private-gemm-x86` called directly.
    PrivateGemmX86,
    /// The packed micro-kernel driver, with [`GemmConfig::kernel`] selectable.
    Packed,
}

/// What the engine actually used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Engine {
    /// faer's `matmul`.
    Faer,
    /// `private-gemm-x86`.
    PrivateGemmX86,
    /// The packed driver.
    Packed,
}

/// One GEMM's configuration.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GemmConfig {
    /// Which engine to use.
    pub engine: EngineChoice,
    /// Registered kernel family for [`Engine::Packed`]; `Auto` uses the
    /// process default set.
    pub kernel: tprims_gemm_kernel::KernelChoice,
    /// Complex method for [`Engine::Packed`]; `None` keeps the family's own or
    /// the process default.
    pub method: Option<tprims_gemm_kernel::Method>,
}

/// What a GEMM used, for diagnostics: the engine, the resolved family and the
/// blocking it ran with.
///
/// The batched variants wrap [`Selected`](crate::Selected) instead of replacing
/// it, because that is the type the pinned tenferro consumer already matches
/// on; `batched` carries the older report when there is one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectedGemm {
    /// The engine that computed the product.
    pub engine: Engine,
    /// The resolved family's id, for [`Engine::Packed`].
    pub family_id: Option<&'static str>,
    /// The resolved family's complex scheme, if it has one.
    pub complex: Option<tprims_gemm_kernel::ComplexScheme>,
    /// Micro-tile rows.
    pub mr: usize,
    /// Micro-tile columns.
    pub nr: usize,
    /// Cache block rows.
    pub mc: usize,
    /// Cache block columns.
    pub nc: usize,
    /// Cache block depth.
    pub kc: usize,
    /// The grid the driver used.
    pub partition: tprims_gemm_kernel::PartitionPolicy,
    /// The report the batched strategies already produced, when there is one.
    pub batched: Option<crate::Selected>,
    /// The resolved family's provenance (provider crate and license), for
    /// [`Engine::Packed`]. A downstream kernel reports
    /// [`Origin::External`](tprims_gemm_kernel::Origin::External).
    pub origin: Option<tprims_gemm_kernel::Origin>,
}

impl SelectedGemm {
    /// The report as JSON, hand-written so the crate needs no serializer.
    ///
    /// # Examples
    /// ```
    /// use tprims_blas::{Engine, SelectedGemm};
    /// use tprims_gemm_kernel::PartitionPolicy;
    /// let s = SelectedGemm {
    ///     engine: Engine::Faer, family_id: None, complex: None,
    ///     mr: 0, nr: 0, mc: 0, nc: 0, kc: 0,
    ///     partition: PartitionPolicy::default(), batched: None, origin: None,
    /// };
    /// assert!(s.to_json().starts_with("{\"engine\":\"Faer\""));
    /// ```
    pub fn to_json(&self) -> String {
        let family = match self.family_id {
            Some(id) => format!("\"{id}\""),
            None => "null".to_string(),
        };
        let complex = match self.complex {
            Some(scheme) => format!(
                "{{\"method\":\"{:?}\",\"mr\":{},\"nr\":{}}}",
                scheme.method, self.mr, self.nr
            ),
            None => "null".to_string(),
        };
        let provider = match self.origin {
            Some(o) => format!(
                "{{\"crate\":\"{}\",\"license\":\"{}\"}}",
                o.crate_name(),
                o.license()
            ),
            None => "null".to_string(),
        };
        format!(
            "{{\"engine\":\"{:?}\",\"family_id\":{family},\"complex\":{complex},\
             \"mr\":{},\"nr\":{},\"mc\":{},\"nc\":{},\"kc\":{},\"provider\":{provider}}}",
            self.engine, self.mr, self.nr, self.mc, self.nc, self.kc
        )
    }
}

/// The process default engine, read once from `TPRIMS_GEMM_ENGINE`.
///
/// Unset or unrecognised means faer, which is what every existing caller gets.
///
/// # Examples
/// ```
/// // faer unless the environment selected another engine at startup.
/// let _ = tprims_blas::default_engine();
/// ```
pub fn default_engine() -> Engine {
    static DEFAULT: std::sync::OnceLock<Engine> = std::sync::OnceLock::new();
    *DEFAULT.get_or_init(|| {
        #[cfg(feature = "std")]
        if let Ok(v) = std::env::var("TPRIMS_GEMM_ENGINE") {
            if v.eq_ignore_ascii_case("pgx86") || v.eq_ignore_ascii_case("private-gemm-x86") {
                return Engine::PrivateGemmX86;
            }
            if v.eq_ignore_ascii_case("packed") {
                return Engine::Packed;
            }
        }
        Engine::Faer
    })
}

/// Register the kernel crates this build has, once.
///
/// The tensorcontract families are always available; the gemm and pgx86
/// providers are behind their features, and their id prefixes are declared
/// even when they are absent so an id naming them reports the feature to
/// enable rather than "unknown".
pub(crate) fn register_built() {
    static REGISTER: std::sync::Once = std::sync::Once::new();
    REGISTER.call_once(|| {
        tprims_kernel_tensorcontract::register();
        #[cfg(feature = "kernel-gemm")]
        tprims_kernel_gemm::register();
        tprims_gemm_kernel::register_known_prefix("gemm.", "kernel-gemm");
        tprims_gemm_kernel::register_known_prefix("pgx86.", "kernel-pgx86");
    });
}

/// `private-gemm-x86` for one matrix GEMM, or the error that says why not.
#[cfg(feature = "kernel-pgx86")]
fn run_pgx86<T: Scalar>(
    exec: &Exec<'_>,
    alpha: T,
    a: MatIn<'_, '_, T>,
    b: MatIn<'_, '_, T>,
    beta: T,
    c: &mut StridedViewMut<'_, T>,
    s: &crate::gemm::GemmShape,
) -> Result<()> {
    if !tprims_kernel_pgx86::available() {
        return Err(Error::Select(
            tprims_gemm_kernel::SelectError::EngineUnsupported {
                engine: "pgx86",
                reason: "x86-64 with AVX2 and FMA is required",
            },
        ));
    }
    if beta != crate::scalar::one() {
        // SAFETY: shapes validated; `c` is exclusive and injective.
        unsafe { crate::operand::scale_in_place(c.as_mut_ptr(), &s.c, beta) };
    }
    let (m, n, k) = (s.c.rows, s.c.cols, s.a.cols);
    let (ap, bp) = (
        crate::gemm::SendConst(a.view.ptr()),
        crate::gemm::SendConst(b.view.ptr()),
    );
    let cp = crate::gemm::SendMut(c.as_mut_ptr());
    let (ca, cb) = (a.conj, b.conj);
    let width = crate::gemm::gemm_width::<T>(exec, m, n, k);
    exec.install(width, move |par| {
        let threads = match par {
            tprims_exec::Par::Seq => 1,
            tprims_exec::Par::Threads(nn) => nn.get(),
        };
        let (ap, bp, cp) = (ap, bp, cp);
        // The engine is generic over four concrete scalars while this entry
        // point is generic over `Scalar`, whose four implementers those are.
        // The cast below is therefore the identity for the matching arm, and
        // the size/complexity pair names exactly one arm per scalar.
        macro_rules! call {
            ($scalar:ty) => {{
                let cp = cp.0 as *mut $scalar;
                let ap = ap.0 as *const $scalar;
                let bp = bp.0 as *const $scalar;
                // SAFETY: the cast is the identity in the arm that runs.
                let alpha = unsafe { *(&alpha as *const T as *const $scalar) };
                // SAFETY: shapes validated; `D` is exclusive and does not alias
                // the operands; `beta` was applied above, so the engine adds.
                unsafe {
                    tprims_kernel_pgx86::gemm(
                        m,
                        n,
                        k,
                        cp,
                        s.c.rs,
                        s.c.cs,
                        beta == zero(),
                        ap,
                        s.a.rs,
                        s.a.cs,
                        ca == crate::Conj::Yes,
                        bp,
                        s.b.rs,
                        s.b.cs,
                        cb == crate::Conj::Yes,
                        alpha,
                        threads,
                    )
                }
            }};
        }
        match (core::mem::size_of::<T>(), T::IS_COMPLEX_SCALAR) {
            (4, false) => call!(f32),
            (8, false) => call!(f64),
            (8, true) => call!(num_complex::Complex<f32>),
            (16, true) => call!(num_complex::Complex<f64>),
            _ => unreachable!("Scalar is sealed to the four supported types"),
        }
    });
    Ok(())
}

/// Without the provider the engine cannot run, and says which feature to build.
#[cfg(not(feature = "kernel-pgx86"))]
fn run_pgx86<T: Scalar>(
    _exec: &Exec<'_>,
    _alpha: T,
    _a: MatIn<'_, '_, T>,
    _b: MatIn<'_, '_, T>,
    _beta: T,
    _c: &mut StridedViewMut<'_, T>,
    _s: &crate::gemm::GemmShape,
) -> Result<()> {
    Err(Error::Select(
        tprims_gemm_kernel::SelectError::EngineUnsupported {
            engine: "pgx86",
            reason: "the kernel-pgx86 feature is not enabled",
        },
    ))
}

/// Every registered family, as a diagnostic table. Registers the kernel crates
/// this build has first.
///
/// # Examples
/// ```
/// let kernels = tprims_blas::list_kernels::<f64>();
/// assert!(kernels.iter().any(|k| k.id.starts_with("tc.")));
/// ```
pub fn list_kernels<T: Scalar>() -> Vec<tprims_gemm_kernel::KernelInfo> {
    register_built();
    tprims_gemm_kernel::list_kernels::<T>()
}

/// `C = alpha * op(A) * op(B) + beta * C`, on the configured engine.
///
/// Identical to [`gemm`](crate::gemm) at [`GemmConfig::default`], and reports
/// what it chose.
///
/// # Errors
///
/// Everything [`gemm`](crate::gemm) returns, plus [`Error::Select`] when the
/// chosen engine or family cannot be used — an unavailable CPU, an unbuilt
/// provider, an unknown id or an unsupported partition. Nothing is written then.
#[allow(clippy::too_many_arguments)] // INVARIANT: the GEMM argument set.
pub fn gemm_with<T: Scalar>(
    exec: &Exec<'_>,
    cfg: &GemmConfig,
    alpha: T,
    a: MatIn<'_, '_, T>,
    b: MatIn<'_, '_, T>,
    beta: T,
    c: &mut StridedViewMut<'_, T>,
) -> Result<SelectedGemm> {
    let s = crate::gemm::check_gemm(
        crate::operand::mat2("A", a.view.dims(), a.view.strides())?,
        crate::operand::mat2("B", b.view.dims(), b.view.strides())?,
        crate::operand::mat2("C", c.dims(), c.strides())?,
    )?;
    let engine = match cfg.engine {
        EngineChoice::Auto => default_engine(),
        EngineChoice::Faer => Engine::Faer,
        EngineChoice::PrivateGemmX86 => Engine::PrivateGemmX86,
        EngineChoice::Packed => Engine::Packed,
    };
    let (m, n, k) = (s.c.rows, s.c.cols, s.a.cols);
    // Every engine agrees that an empty K or a zero alpha only scales C, and
    // none of them may read A or B then.
    if m == 0 || n == 0 || k == 0 || alpha == zero() {
        // SAFETY: shapes validated; `c` is exclusive and injective.
        unsafe { crate::operand::scale_in_place(c.as_mut_ptr(), &s.c, beta) };
        return Ok(SelectedGemm {
            engine,
            family_id: None,
            complex: None,
            mr: 0,
            nr: 0,
            mc: 0,
            nc: 0,
            kc: 0,
            partition: tprims_gemm_kernel::PartitionPolicy::default(),
            batched: None,
            origin: None,
        });
    }
    match engine {
        Engine::Faer => {
            let width = crate::gemm::gemm_width::<T>(exec, m, n, k);
            let (ap, bp) = (
                crate::gemm::SendConst(a.view.ptr()),
                crate::gemm::SendConst(b.view.ptr()),
            );
            let cp = crate::gemm::SendMut(c.as_mut_ptr());
            let (ca, cb) = (a.conj, b.conj);
            exec.install(width, move |par| {
                let (ap, bp, cp) = (ap, bp, cp);
                // SAFETY: shapes validated against the views, which were
                // bounds-checked at construction; `c` is an exclusive borrow,
                // injective by `check_gemm`.
                unsafe {
                    crate::gemm::gemm_raw(
                        &s,
                        alpha,
                        ap.0,
                        ca,
                        bp.0,
                        cb,
                        beta,
                        cp.0,
                        crate::gemm::to_faer(par),
                    )
                }
            });
            Ok(SelectedGemm {
                engine,
                family_id: None,
                complex: None,
                mr: 0,
                nr: 0,
                mc: 0,
                nc: 0,
                kc: 0,
                partition: tprims_gemm_kernel::PartitionPolicy::default(),
                batched: None,
                origin: None,
            })
        }
        Engine::PrivateGemmX86 => {
            run_pgx86(exec, alpha, a, b, beta, c, &s)?;
            Ok(SelectedGemm {
                engine,
                family_id: None,
                complex: None,
                mr: 0,
                nr: 0,
                mc: 0,
                nc: 0,
                kc: 0,
                partition: tprims_gemm_kernel::PartitionPolicy::default(),
                batched: None,
                origin: None,
            })
        }
        // SAFETY: the shapes and strides were validated against the views,
        // which were bounds-checked at construction; `c` is an exclusive,
        // injective borrow of a region that aliases neither operand.
        Engine::Packed => unsafe {
            crate::tblis::run_one_plan(
                exec,
                cfg,
                alpha,
                a.view.ptr(),
                a.conj,
                a.view.dims(),
                a.view.strides(),
                b.view.ptr(),
                b.conj,
                b.view.dims(),
                b.view.strides(),
                beta,
                c,
                None,
                true,
            )
        },
    }
}
