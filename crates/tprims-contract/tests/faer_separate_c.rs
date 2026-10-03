//! A separately described C on a GEMM-fusable problem runs on faer (Phase 2
//! W2): `D = op_D(alpha * op_A(A) * op_B(B) + beta * op_C(C))` against the
//! independent label oracle, for a C that is D itself (in place, same mapping
//! and origin), a C of D's layout in its own buffer, and a C of another
//! layout, over every conjugation combination and the scalar corner cases
//! (`beta == 0` reads no C, `alpha == 0` reads no A or B), in f64 and c64.
use num_complex::Complex64 as C64;
use tprims_contract::api::{CSpec, DType, Labels, LayoutSpec, Op, OperandSpec, Problem, Scalar};
use tprims_contract::{Algorithm, Plan, PlanConfig};
use tprims_exec::Exec;
use tprims_kernel::Element;
use tprims_testkit::fixtures::{col_major, row_major, seeded, storage_len};
use tprims_testkit::oracle::{contract_reference, RefOperand, RefOutput};

struct Shape {
    name: &'static str,
    a: Vec<usize>,
    b: Vec<usize>,
    d: Vec<usize>,
    la: Vec<i64>,
    lb: Vec<i64>,
    ld: Vec<i64>,
}

/// GEMM-fusable shapes: a plain matmul, a batched one, and two fused K axes.
fn shapes() -> Vec<Shape> {
    vec![
        Shape {
            name: "matmul",
            a: vec![7, 5],
            b: vec![5, 6],
            d: vec![7, 6],
            la: vec![0, 2],
            lb: vec![2, 1],
            ld: vec![0, 1],
        },
        Shape {
            name: "batched",
            a: vec![4, 3, 5],
            b: vec![3, 6, 5],
            d: vec![4, 6, 5],
            la: vec![0, 2, 9],
            lb: vec![2, 1, 9],
            ld: vec![0, 1, 9],
        },
        Shape {
            name: "two fused K",
            a: vec![3, 4, 5],
            b: vec![4, 5, 6],
            d: vec![3, 6],
            la: vec![0, 1, 2],
            lb: vec![1, 2, 3],
            ld: vec![0, 3],
        },
    ]
}

fn op(c: bool) -> Op {
    if c {
        Op::Conjugate
    } else {
        Op::Identity
    }
}

fn spec(dims: &[usize], strides: &[isize], conj: bool) -> OperandSpec {
    OperandSpec::new(LayoutSpec::new(dims, strides, 0).unwrap()).with_op(op(conj))
}

#[derive(Clone, Copy, Debug)]
enum Source {
    /// C is D itself: same mapping, same origin.
    InPlace,
    /// A separate buffer with D's layout.
    SameLayout,
    /// A separate buffer, row-major while D is column-major.
    OtherLayout,
}

fn dtype<T: Scalar>() -> DType {
    T::STORAGE
}

fn check<T: Scalar + Element>(
    shape: &Shape,
    source: Source,
    flags: [bool; 4],
    alpha: T,
    beta: T,
    nan_c: bool,
) {
    let [ca, cb, cc, cd] = flags;
    let (sa, sb, sd) = (col_major(&shape.a), col_major(&shape.b), col_major(&shape.d));
    let sc = match source {
        Source::InPlace | Source::SameLayout => sd.clone(),
        Source::OtherLayout => row_major(&shape.d),
    };
    let a: Vec<T> = seeded(1, storage_len(&shape.a, &sa));
    let b: Vec<T> = seeded(2, storage_len(&shape.b, &sb));
    let start: Vec<T> = seeded(3, storage_len(&shape.d, &sd));
    let c_sep: Vec<T> = if nan_c {
        let nan = <T as Element>::from_parts(
            <<T as Element>::Real as tprims_kernel::Real>::from_f64(f64::NAN),
            <<T as Element>::Real as tprims_kernel::Real>::from_f64(f64::NAN),
        );
        vec![nan; storage_len(&shape.d, &sc)]
    } else {
        seeded(4, storage_len(&shape.d, &sc))
    };

    let problem = Problem::from_labels(
        dtype::<T>(),
        spec(&shape.a, &sa, ca),
        spec(&shape.b, &sb, cb),
        CSpec::Separate(spec(&shape.d, &sc, cc)),
        spec(&shape.d, &sd, cd),
        &Labels::new(&shape.la, &shape.lb, &shape.ld).with_c(&shape.ld),
    )
    .unwrap();
    let plan = Plan::<T>::new(&problem, &PlanConfig::default()).unwrap();
    assert_eq!(
        plan.report().algorithm,
        Algorithm::Faer,
        "{} {source:?}: GEMM-fusable, so faer",
        shape.name
    );

    let c_data: &[T] = match source {
        Source::InPlace => &start,
        _ => &c_sep,
    };
    let mut want = start.clone();
    contract_reference(
        alpha,
        &RefOperand {
            data: &a,
            dims: &shape.a,
            strides: &sa,
            offset: 0,
            labels: &shape.la,
            conj: ca,
        },
        &RefOperand {
            data: &b,
            dims: &shape.b,
            strides: &sb,
            offset: 0,
            labels: &shape.lb,
            conj: cb,
        },
        beta,
        Some(&RefOperand {
            data: c_data,
            dims: &shape.d,
            strides: &sc,
            offset: 0,
            labels: &shape.ld,
            conj: cc,
        }),
        &mut RefOutput {
            data: &mut want,
            dims: &shape.d,
            strides: &sd,
            offset: 0,
            labels: &shape.ld,
            conj: cd,
        },
    )
    .unwrap();

    let mut got = start.clone();
    let dp = got.as_mut_ptr();
    let cp = match source {
        Source::InPlace => dp as *const T,
        _ => c_sep.as_ptr(),
    };
    // SAFETY: every pointer addresses its operand's full layout; D is
    // exclusive, and C is D itself only for the in-place mapping.
    unsafe {
        plan.execute_raw(&Exec::serial(), alpha, a.as_ptr(), b.as_ptr(), beta, cp, dp)
            .unwrap();
    }
    let err = got
        .iter()
        .zip(&want)
        .map(|(g, w)| g.sub(*w).norm())
        .fold(0.0, f64::max);
    assert!(
        err < 1e-12 * 50.0 && got.iter().all(|z| z.norm().is_finite()),
        "{} {source:?} flags {flags:?} alpha {alpha:?} beta {beta:?}: err {err}",
        shape.name
    );
}

fn scalars<T: Element>() -> Vec<(T, T)> {
    let s = |re: f64, im: f64| {
        T::from_parts(
            <<T as Element>::Real as tprims_kernel::Real>::from_f64(re),
            <<T as Element>::Real as tprims_kernel::Real>::from_f64(im),
        )
    };
    vec![
        (s(1.0, 0.0), s(0.0, 0.0)),
        (s(0.7, -0.3), s(0.0, 0.0)),
        (s(1.0, 0.0), s(1.0, 0.0)),
        (s(0.7, -0.3), s(1.0, 0.0)),
        (s(0.7, -0.3), s(-0.4, 0.6)),
        (s(0.0, 0.0), s(-0.4, 0.6)),
    ]
}

fn sweep<T: Scalar + Element>() {
    for shape in shapes() {
        for source in [Source::InPlace, Source::SameLayout, Source::OtherLayout] {
            for bits in 0..16u32 {
                let flags = [bits & 1 != 0, bits & 2 != 0, bits & 4 != 0, bits & 8 != 0];
                for (alpha, beta) in scalars::<T>() {
                    check(&shape, source, flags, alpha, beta, false);
                }
            }
        }
    }
}

#[test]
fn a_separate_c_matches_the_oracle_in_f64() {
    sweep::<f64>();
}

#[test]
fn a_separate_c_matches_the_oracle_in_c64() {
    sweep::<C64>();
}

/// With `beta == 0` the separate C is never read: NaN in it must not leak.
#[test]
fn beta_zero_does_not_read_a_separate_c() {
    let zero = <C64 as Element>::zero();
    for shape in shapes() {
        for source in [Source::SameLayout, Source::OtherLayout] {
            for bits in [0u32, 15] {
                let flags = [bits & 1 != 0, bits & 2 != 0, bits & 4 != 0, bits & 8 != 0];
                check::<C64>(&shape, source, flags, C64::new(0.7, -0.3), zero, true);
            }
        }
    }
}

/// A TAPP product describes C separately (here with D's layout): the plan
/// still reports faer, not materializing any operand.
#[test]
fn a_tapp_style_gemm_product_reports_faer() {
    let (sa, sb, sd) = (col_major(&[8, 9]), col_major(&[9, 7]), col_major(&[8, 7]));
    let problem = Problem::from_labels(
        DType::F64,
        spec(&[8, 9], &sa, false),
        spec(&[9, 7], &sb, false),
        CSpec::Separate(spec(&[8, 7], &sd, false)),
        spec(&[8, 7], &sd, false),
        &Labels::new(&[0, 2], &[2, 1], &[0, 1]).with_c(&[0, 1]),
    )
    .unwrap();
    let plan = Plan::<f64>::new(&problem, &PlanConfig::default()).unwrap();
    assert_eq!(plan.report().algorithm, Algorithm::Faer);
    assert_eq!(plan.report().materialized, [false; 3]);
}
