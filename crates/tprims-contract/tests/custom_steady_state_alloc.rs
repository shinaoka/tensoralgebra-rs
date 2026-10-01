//! A custom selection adds nothing to a steady-state execute: the selector ran
//! at planning, and execution allocates exactly what the same plan on a
//! built-in kernel allocates.
//!
//! Its own binary, because the allocation counter is global.
use std::alloc::{GlobalAlloc, Layout as AllocLayout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use strided_view::{StridedView, StridedViewMut};
use tprims_blas::{Conj, GemmConfig, KernelCatalog};
use tprims_contract::{ContractPlan, DotGeneral, Flags, Strategy};
use tprims_contract_testkit::custom_kernels as own;
use tprims_exec::Exec;
use tprims_kernel::KernelChoice;

static COUNT: AtomicUsize = AtomicUsize::new(0);

struct Counting;
// SAFETY: forwards to the system allocator unchanged; the counter is atomic.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: AllocLayout) -> *mut u8 {
        COUNT.fetch_add(1, Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: AllocLayout) {
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static ALLOC: Counting = Counting;

const DIMS_A: [usize; 3] = [30, 24, 20];
const DIMS_B: [usize; 3] = [20, 28, 24];
const DIMS_C: [usize; 2] = [30, 28];

fn strides(dims: &[usize]) -> Vec<isize> {
    let mut s = Vec::new();
    let mut acc = 1isize;
    for &d in dims {
        s.push(acc);
        acc *= d as isize;
    }
    s
}

/// Allocations performed by `reps` executes of an already warm plan.
fn executes(plan: &ContractPlan<f64>, reps: usize) -> usize {
    let a = vec![1.0; DIMS_A.iter().product()];
    let b = vec![2.0; DIMS_B.iter().product()];
    let mut c = vec![0.0; DIMS_C.iter().product()];
    let (sa, sb, sc) = (strides(&DIMS_A), strides(&DIMS_B), strides(&DIMS_C));
    let av = StridedView::new(&a, &DIMS_A, &sa, 0).unwrap();
    let bv = StridedView::new(&b, &DIMS_B, &sb, 0).unwrap();
    let exec = Exec::serial();
    let mut run = || {
        let mut cv = StridedViewMut::new(&mut c, &DIMS_C, &sc, 0).unwrap();
        plan.execute(&exec, 1.0, &av, &bv, 0.0, &mut cv).unwrap();
    };
    // Warm up: first touch of the plan's own workspace.
    run();
    run();
    let before = COUNT.load(Relaxed);
    for _ in 0..reps {
        run();
    }
    COUNT.load(Relaxed) - before
}

#[test]
fn custom_selection_adds_no_steady_state_allocation() {
    let cfg = DotGeneral::new(&[1, 2], &[2, 0], &[], &[]);
    let (sa, sb, sc) = (strides(&DIMS_A), strides(&DIMS_B), strides(&DIMS_C));
    // SAFETY: see `selector_blas::catalog`.
    let cat = unsafe { KernelCatalog::<f64>::from_static_families(own::f64_families()) }.unwrap();
    let calls = Cell::new(0);
    let custom = ContractPlan::<f64>::new_with_selector(
        &Exec::serial(),
        &GemmConfig::default(),
        &cat,
        |_, cands| {
            calls.set(calls.get() + 1);
            Ok(cands[0].handle)
        },
        &cfg,
        (&DIMS_A, &sa),
        (&DIMS_B, &sb),
        (&DIMS_C, &sc),
        (Conj::No, Conj::No),
        Strategy::Tblis,
        Flags::default(),
    )
    .unwrap();
    // The same plan with the equivalent *built-in* geometry by id.
    let builtin = ContractPlan::<f64>::new_with(
        &GemmConfig {
            kernel: KernelChoice::Id("ref.f64.real.4x4".into()),
            ..Default::default()
        },
        &cfg,
        (&DIMS_A, &sa),
        (&DIMS_B, &sb),
        (&DIMS_C, &sc),
        (Conj::No, Conj::No),
        Strategy::Tblis,
        Flags::default(),
    )
    .unwrap();
    let with_custom = executes(&custom, 20);
    let with_builtin = executes(&builtin, 20);
    assert_eq!(calls.get(), 1, "the selector ran at planning only");
    assert_eq!(
        with_custom, with_builtin,
        "custom selection must add no allocation to a steady-state execute"
    );
}
