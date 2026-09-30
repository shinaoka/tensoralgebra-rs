//! GEMM family contracts and project-owned portable kernels.
//! Arithmetic traits, packing, scatter and write-back were moved from Lukas
//! Devos's tensorcontract: lkdvos/tensorprimitives-rs; MIT OR Apache-2.0.
#![warn(missing_docs)]
#![warn(missing_debug_implementations)]

/// Read one `TENSORCONTRACT_*` variable once per process, or fall back.
///
/// Nine switches were spelling this out by hand, and the copies had drifted:
/// `partition_override` returned the *legacy* rule without `std` where the
/// `std` default is the domain-aware one, so a `--no-default-features` build
/// silently partitioned differently. Naming the default once, outside the
/// `cfg`, makes that class of divergence unrepresentable — the two arms cannot
/// disagree because there is only one expression.
///
/// The variable name stays a literal at each call site on purpose, so
/// `grep TENSORCONTRACT_` still finds every switch in the crate.
///
/// `$ty` must be `Copy`; every switch is a small enum, `bool` or `usize`.
#[doc(hidden)]
#[macro_export]
macro_rules! env_once {
    ($ty:ty, $var:literal, $default:expr, $parse:expr) => {{
        #[cfg(feature = "std")]
        {
            use std::sync::OnceLock;
            static ENV: OnceLock<$ty> = OnceLock::new();
            *ENV.get_or_init(|| match std::env::var($var) {
                Ok(v) => ($parse)(v.as_str()),
                Err(_) => $default,
            })
        }
        #[cfg(not(feature = "std"))]
        {
            $default
        }
    }};
}

pub mod cache;
mod cpu;
pub mod element;
mod family;
pub mod induced;
#[doc(hidden)]
pub mod pack;
pub mod portable;
mod registry;
mod resolved;
pub mod scatter;
mod types;
#[doc(hidden)]
pub mod writeback;

pub use cpu::{kernel_force, CpuFeatures, Isa, KernelForce};
pub use element::{Element, Real, C32, C64};
pub use family::*;
pub use registry::{
    list_kernels, register, register_known_prefix, Families, KernelInfo, RealSlot, Registry,
    SelectError,
};
pub use resolved::{process_default, KernelChoice, ResolvedGemm};
pub use scatter::IRREGULAR;
pub use types::*;

/// `TENSORCONTRACT_THREADS=<n>` sets the default thread count. Read once per
/// process. Unset means **1**: see `tensorcontract::Plan::threads` for why that is the default
/// while Phase 4 is still measuring, and note that it keeps every committed
/// single-core number reproducible from a bare checkout.
///
/// Visible to the crate because the blocking model needs a thread count when it
/// is asked for a configuration without a plan — one definition of the default,
/// rather than two readers of one variable.
#[doc(hidden)]
pub fn env_threads() -> usize {
    env_once!(usize, "TENSORCONTRACT_THREADS", 1, |v: &str| v
        .trim()
        .parse::<usize>()
        .unwrap_or(1)
        .max(1))
}
