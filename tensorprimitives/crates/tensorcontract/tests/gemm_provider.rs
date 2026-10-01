//! The `gemm-f64`/`gemm-f32` provider families, driven through the real
//! planner and driver: the adapter's argument mapping has to survive packing,
//! the direct-C guard and the batch loop, not just a direct kernel call.
mod common;

use common::{all_families, check_family_vs_oracle, Opts};
use tensorcontract::element::Element;
use tensorcontract::kernel::KernelSet;
use tprims_gemm_kernel::Families;

fn run_family<T>(id: &'static str)
where
    T: Element + Families,
    T::Real: KernelSet,
{
    let tol = if core::mem::size_of::<T::Real>() == 4 {
        3e-5
    } else {
        1e-11
    };
    for row_major_d in [false, true] {
        // The direct-C write is only reachable with `beta == 0` or an aliasing
        // `C`, so both cases have to be run for the guard's arms to be real.
        for (beta, alias) in [(0.0, false), (-0.4, false), (-0.4, true)] {
            check_family_vs_oracle::<T>(
                id,
                Opts {
                    row_major_d,
                    alias,
                    ..Opts::real(1.3, beta)
                },
                tol,
            );
        }
    }
}

#[test]
fn gemm_families_match_the_oracle_for_both_dtypes() {
    tprims_kernel_gemm::register();
    for id in all_families::<f64>() {
        if id.starts_with("gemm.") {
            run_family::<f64>(id);
        }
    }
    for id in all_families::<f32>() {
        if id.starts_with("gemm.") {
            run_family::<f32>(id);
        }
    }
}
