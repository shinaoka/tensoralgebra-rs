//! What the analytical blocking model predicts on *this* machine, against the
//! hardcoded constants that ship.
//!
//! ```text
//! cargo run --release -p tensorcontract --example blocking_model
//! ```
//!
//! This is a **computation, not a measurement**: it reads the cache descriptors,
//! evaluates both derivations for every dtype, complex method and thread count,
//! and prints them. It touches no data, times nothing and costs no CPU, so it is
//! safe to run while a benchmark is in flight — the same standing as `tcbench
//! shapes` and `tcbench orient`.
//!
//! Both columns are shown *after* the register-block rounding the driver
//! requires, so they are what execution would really use. The `A`/`B` columns
//! are the packed footprints those parameters imply, which is the quantity the
//! model is actually reasoning about: `A` should sit inside the L2 and `B`
//! inside the L3.

use tensorcontract::kernel::cache::{self, CacheLevel};
use tensorcontract::kernel::{Blocking, ComplexMethod, KernelConfig, KernelSet, Ukr};
use tprims_kernel::KernelForce;

fn main() {
    let h = cache::hierarchy();
    println!("cache descriptors ({}):", h.source.name());
    let show = |lvl: &CacheLevel, name: &str| {
        println!(
            "  {name:3}  {:>7} KiB  {:>2}-way  {:>6} sets  {:>3} B lines  shared by {:>2} logical CPU(s)",
            lvl.size / 1024,
            lvl.ways,
            lvl.sets,
            lvl.line,
            lvl.shared_by,
        );
    };
    show(&h.l1d, "L1d");
    if let Some(l2) = &h.l2 {
        show(l2, "L2");
    }
    if let Some(l3) = &h.l3 {
        show(l3, "L3");
    }
    println!(
        "  {} logical CPU(s) per core, so the L2 belongs to {} core(s) and the L3 to {}",
        h.threads_per_core(),
        h.l2.map(|l| h.cores_sharing(&l)).unwrap_or(0),
        h.l3.map(|l| h.cores_sharing(&l)).unwrap_or(0),
    );
    println!();

    for threads in [1usize, 8] {
        println!("threads = {threads}");
        println!(
            "{:<6} {:<7} {:>3} {:>3} | {:>6} {:>4} {:>7} {:>7} {:>7} | {:>6} {:>4} {:>7} {:>7} {:>7}",
            "dtype", "method", "MR", "NR", //
            "mc", "kc", "nc", "A KiB", "B KiB", //
            "mc", "kc", "nc", "A KiB", "B KiB",
        );
        row::<f64>("f64", None, threads);
        row::<f32>("f32", None, threads);
        for m in ComplexMethod::ALL {
            row::<f64>("c64", Some(m), threads);
        }
        for m in ComplexMethod::ALL {
            row::<f32>("c32", Some(m), threads);
        }
        println!();
    }
    println!("left block: legacy (hardcoded)   right block: analytical model");
}

/// One dtype and method: the shipped blocking and the model's, side by side.
fn row<T: KernelSet>(dtype: &str, method: Option<ComplexMethod>, threads: usize) {
    let (cfg, a_reals, b_reals) = match method {
        None => (T::config_real(KernelForce::Auto), 1, 1),
        Some(m) => {
            let c = T::config_cplx(KernelForce::Auto, m);
            let (a, b) = (
                c.ukr.a_pack.reals_per_element(),
                c.ukr.b_pack.reals_per_element(),
            );
            (c, a, b)
        }
    };
    let ukr = cfg.ukr;
    let real_bytes = core::mem::size_of::<T>();
    // Recompute both arms explicitly, so the table does not depend on which
    // derivation a plan would use.
    let legacy = rounded(ukr, Blocking::derive(real_bytes, a_reals, b_reals));
    let model = rounded(ukr, Blocking::model(&ukr, threads));
    let kib = |b: usize| b / 1024;
    let cells = |b: Blocking| {
        format!(
            "{:>6} {:>4} {:>7} {:>7} {:>7}",
            b.mc,
            b.kc,
            b.nc,
            kib(b.mc * b.kc * a_reals * real_bytes),
            kib(b.nc * b.kc * b_reals * real_bytes),
        )
    };
    println!(
        "{dtype:<6} {:<7} {:>3} {:>3} | {} | {}",
        method.map(|m| m.name()).unwrap_or("-"),
        ukr.mr,
        ukr.nr,
        cells(legacy),
        cells(model),
    );
}

/// The register-block rounding the driver imposes, applied through the same
/// code path execution uses.
fn rounded<T>(ukr: Ukr<T>, blk: Blocking) -> Blocking {
    KernelConfig { ukr, blk }.with_blocking(blk).blk
}
