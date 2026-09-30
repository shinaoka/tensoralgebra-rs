//! Every registered CPU-available GEMM family gets the same small, non-trivial GEMM.
use tensorcontract::element::{Element, Real};
use tensorcontract::kernel::KernelSet;
use tensorcontract::plan::{ElementOp, Operand};
use tensorcontract::reference::{contract_reference, RefOperand};
use tensorcontract::{KernelChoice, Layout, Plan};
use tprims_gemm_kernel::{CpuFeatures, Families, Registry};

fn pinned_orientation() -> bool {
    std::env::var_os("TENSORCONTRACT_ORIENT").is_some()
}

fn value<T: Element>(i: usize) -> T {
    let r = T::Real::from_f64((i as f64 + 1.0) * 0.13);
    let im = T::Real::from_f64((i as f64 - 2.0) * 0.07);
    T::from_parts(r, if T::IS_COMPLEX { im } else { T::Real::ZERO })
}

fn run_family<T>(id: &'static str)
where
    T: Element + Families,
    T::Real: KernelSet,
{
    let probe = Plan::new(
        Operand::new(&Layout::col_major(&[2, 2]), &[0, 2]),
        Operand::new(&Layout::col_major(&[2, 2]), &[2, 1]),
        None,
        Operand::new(&Layout::col_major(&[2, 2]), &[0, 1]),
    )
    .unwrap()
    .with_kernel(KernelChoice::Id(id.into()))
    .unwrap();
    let rg = probe.resolved::<T>().unwrap();
    assert_eq!(rg.family().id, id);

    let m = rg.mr + 1;
    let n = rg.nr.max(rg.mr) + 1;
    let k = 3;
    for row_major in [false, true] {
        let la = Layout::col_major(&[m as i64, k as i64]);
        let lb = Layout::col_major(&[k as i64, n as i64]);
        let ld = if row_major {
            Layout::row_major(&[m as i64, n as i64])
        } else {
            Layout::col_major(&[m as i64, n as i64])
        };
        let lc = if row_major {
            Layout::col_major(&[m as i64, n as i64])
        } else {
            Layout::row_major(&[m as i64, n as i64])
        };
        for conj_a in [false, true] {
            for conj_b in [false, true] {
                let ia = vec![0, 2];
                let ib = vec![2, 1];
                let idd = vec![0, 1];
                let a: Vec<T> = (0..la.storage_len() as usize).map(value).collect();
                let b: Vec<T> = (0..lb.storage_len() as usize).map(value).collect();
                let c: Vec<T> = (0..lc.storage_len() as usize)
                    .map(|i| value(i + 17))
                    .collect();
                let mut got: Vec<T> = (0..ld.storage_len() as usize)
                    .map(|i| value(i + 31))
                    .collect();
                let mut want = got.clone();
                let alpha = T::from_parts(T::Real::from_f64(1.3), T::Real::from_f64(0.2));
                let beta = T::from_parts(T::Real::from_f64(-0.4), T::Real::from_f64(0.1));
                let plan = Plan::new(
                    Operand {
                        layout: &la,
                        idx: &ia,
                        op: if conj_a {
                            ElementOp::Conjugate
                        } else {
                            ElementOp::Identity
                        },
                    },
                    Operand {
                        layout: &lb,
                        idx: &ib,
                        op: if conj_b {
                            ElementOp::Conjugate
                        } else {
                            ElementOp::Identity
                        },
                    },
                    Some(Operand::new(&lc, &idd)),
                    Operand::new(&ld, &idd),
                )
                .unwrap()
                .with_kernel(KernelChoice::Id(id.into()))
                .unwrap()
                .with_threads(1)
                .with_blocking(tensorcontract::Blocking {
                    mc: rg.mr,
                    kc: 2,
                    nc: rg.nr,
                });
                if !pinned_orientation() {
                    assert_eq!(plan.transposes_gemm(rg.mr), row_major);
                }
                assert!(core::ptr::eq(
                    plan.resolved::<T>().unwrap().family(),
                    plan.resolved::<T>().unwrap().family()
                ));
                unsafe {
                    plan.run_raw::<T>(
                        alpha,
                        a.as_ptr(),
                        b.as_ptr(),
                        beta,
                        c.as_ptr(),
                        got.as_mut_ptr(),
                    )
                };
                contract_reference::<T>(
                    alpha,
                    &RefOperand {
                        data: &a,
                        layout: &la,
                        idx: &ia,
                        op: if conj_a {
                            ElementOp::Conjugate
                        } else {
                            ElementOp::Identity
                        },
                    },
                    &RefOperand {
                        data: &b,
                        layout: &lb,
                        idx: &ib,
                        op: if conj_b {
                            ElementOp::Conjugate
                        } else {
                            ElementOp::Identity
                        },
                    },
                    beta,
                    Some(&RefOperand {
                        data: &c,
                        layout: &lc,
                        idx: &idd,
                        op: ElementOp::Identity,
                    }),
                    &mut want,
                    &ld,
                    &idd,
                    ElementOp::Identity,
                )
                .unwrap();
                let diff: f64 = got
                    .iter()
                    .zip(&want)
                    .map(|(x, y)| x.sub(*y).norm().powi(2))
                    .sum();
                let scale: f64 = want.iter().map(|x| x.norm().powi(2)).sum();
                let err = (diff / scale.max(1.0)).sqrt();
                let tol = if core::mem::size_of::<T::Real>() == 4 {
                    3e-5
                } else {
                    5e-12
                };
                assert!(
                    err < tol,
                    "family {id}, row_major={row_major}, conj={conj_a}/{conj_b}: {err:e}"
                );
            }
        }
    }
}

fn all_families<T>() -> Vec<&'static str>
where
    T: Element + Families,
{
    tprims_kernel_tensorcontract::register();
    Registry::families::<T>(CpuFeatures::detect(), false)
        .into_iter()
        .map(|f| f.id)
        .collect()
}

macro_rules! families_test {
    ($name:ident, $ty:ty) => {
        #[test]
        fn $name() {
            for id in all_families::<$ty>() {
                run_family::<$ty>(id);
            }
        }
    };
}

families_test!(all_f32_families, f32);
families_test!(all_f64_families, f64);
families_test!(all_c32_families, tensorcontract::C32);
families_test!(all_c64_families, tensorcontract::C64);

#[test]
fn selection_rejects_unknown_and_wrong_dtype() {
    let plan = Plan::new(
        Operand::new(&Layout::col_major(&[2, 2]), &[0, 2]),
        Operand::new(&Layout::col_major(&[2, 2]), &[2, 1]),
        None,
        Operand::new(&Layout::col_major(&[2, 2]), &[0, 1]),
    )
    .unwrap();
    assert!(matches!(
        plan.clone()
            .with_kernel(KernelChoice::Id("definitely-not-a-family".into())),
        Err(tensorcontract::Error::KernelSelection(
            tprims_gemm_kernel::SelectError::UnknownId { .. }
        ))
    ));
    let id = all_families::<f32>()[0];
    assert!(matches!(
        plan.with_kernel(KernelChoice::Id(id.into()))
            .unwrap()
            .resolved::<f64>(),
        Err(tprims_gemm_kernel::SelectError::DtypeMismatch { .. })
    ));
}
