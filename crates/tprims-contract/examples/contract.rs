//! Using the engine: four contractions a matrix multiply cannot express.
//!
//! ```text
//! cargo run --release -p tensorcontract --example contract
//! ```
//!
//! This is **documentation that compiles**, not a measurement: the tensors are
//! tiny, nothing is timed, and every result is checked against a value written
//! out by hand. It costs no CPU and is safe to run while a benchmark is in
//! flight. For what the engine's performance actually is, see `docs/results.md`.
//!
//! Each section is the smallest case that shows one feature of the index
//! notation, in the order the crate documentation introduces them: a batch
//! index, a reduction, a diagonal, and complex conjugation with plan reuse.

use tensorcontract::kernel::ComplexMethod;
use tensorcontract::plan::Operand;
use tensorcontract::{contract, parse_einsum, Layout, Plan, TensorView, TensorViewMut, C64};

// Every fallible call below is `?`, not `.unwrap()`, because this file is read
// as an example of how to call the crate and the difference is visible.
fn main() -> tensorcontract::Result<()> {
    batch_index()?;
    reduction()?;
    diagonal()?;
    complex_with_plan_reuse()?;
    println!("\nall four contractions matched their expected values");
    Ok(())
}

/// A label in `A`, `B` *and* `D` is a batch index: the contraction runs
/// independently for each of its values, with no reshaping and no loop here.
fn batch_index() -> tensorcontract::Result<()> {
    // D[h,i,j] = sum_k A[h,i,k] * B[h,k,j], for each h.
    let (ia, ib, id) = parse_einsum("hik,hkj->hij")?;
    let l = Layout::col_major(&[2, 2, 2]);

    // A[h,i,k] = h + 2i + 4k + 1, since the layout is column-major over (h,i,k).
    let a: Vec<f64> = (1..=8).map(|x| x as f64).collect();
    // Per batch, B is diag(2, 1) over (k, j): it doubles k=0 and passes k=1.
    let b = vec![2.0f64, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];
    let mut d = vec![0.0f64; 8];

    contract(
        1.0,
        TensorView::new(&a, &l, &ia),
        TensorView::new(&b, &l, &ib),
        0.0,
        None,
        TensorViewMut::new(&mut d, &l, &id),
    )?;

    // B being diagonal means only k=j contributes, so D[h,i,j] = B[h,j,j]*A[h,i,j]:
    // the j=0 half of D (offsets 0..4) doubles, and the j=1 half passes through.
    assert_eq!(d, vec![2.0, 4.0, 6.0, 8.0, 5.0, 6.0, 7.0, 8.0]);
    report("batch index", "hik,hkj->hij", &d);
    Ok(())
}

/// A label in one input only is summed over, and needs no temporary: it becomes
/// a contraction index with stride 0 in the other operand, which the
/// block-scatter machinery treats as a perfectly regular access.
fn reduction() -> tensorcontract::Result<()> {
    // D[i,j] = sum_{k,l} A[i,k,l] * B[k,j] — `l` appears nowhere else.
    let (ia, ib, id) = parse_einsum("ikl,kj->ij")?;
    let la = Layout::col_major(&[2, 2, 3]);
    let lb = Layout::col_major(&[2, 2]);
    let ld = Layout::col_major(&[2, 2]);

    // Every element 1.0, so summing over l (extent 3) triples each entry.
    let a = vec![1.0f64; 12];
    let identity = vec![1.0f64, 0.0, 0.0, 1.0];
    let mut d = vec![0.0f64; 4];

    contract(
        1.0,
        TensorView::new(&a, &la, &ia),
        TensorView::new(&identity, &lb, &ib),
        0.0,
        None,
        TensorViewMut::new(&mut d, &ld, &id),
    )?;

    assert_eq!(d, vec![3.0, 3.0, 3.0, 3.0]);
    report("reduction", "ikl,kj->ij", &d);
    Ok(())
}

/// A label repeated *within one operand* selects that operand's diagonal, again
/// with no copy: the two modes' strides are simply summed.
fn diagonal() -> tensorcontract::Result<()> {
    // D[i] = sum_j A[i,j,j] * B[i] — `j` twice in A takes its diagonal.
    let la = Layout::col_major(&[2, 2, 2]);
    let lv = Layout::col_major(&[2]);
    let (i, j) = (b'i' as i64, b'j' as i64);

    // A[i,j,j] picks offsets i + 2j + 4j = i, i+6 → elements 0,1 and 6,7.
    let a: Vec<f64> = (0..8).map(|x| x as f64).collect();
    let b = vec![1.0f64, 1.0];
    let mut d = vec![0.0f64; 2];

    contract(
        1.0,
        TensorView::new(&a, &la, &[i, j, j]),
        TensorView::new(&b, &lv, &[i]),
        0.0,
        None,
        TensorViewMut::new(&mut d, &lv, &[i]),
    )?;

    // i=0: A[0,0,0] + A[0,1,1] = 0 + 6. i=1: A[1,0,0] + A[1,1,1] = 1 + 7.
    assert_eq!(d, vec![6.0, 8.0]);
    report("diagonal", "ijj,i->i", &d);
    Ok(())
}

/// Complex contraction, conjugation, and the reason to build a [`Plan`] by hand:
/// planning is `O(M + N + K)`, so a repeated shape should pay it once.
fn complex_with_plan_reuse() -> tensorcontract::Result<()> {
    let (ia, ib, id) = parse_einsum("ik,kj->ij")?;
    let l = Layout::col_major(&[2, 2]);

    // Conjugation is folded into packing, so it is fixed when the plan is built.
    // The views handed to `run` must agree, or `run` rejects the call.
    let plan = Plan::new(
        Operand::new(&l, &ia).conj(),
        Operand::new(&l, &ib),
        None,
        Operand::new(&l, &id),
    )?
    .with_complex_method(ComplexMethod::Planar);

    let identity = vec![
        C64::new(1.0, 0.0),
        C64::new(0.0, 0.0),
        C64::new(0.0, 0.0),
        C64::new(1.0, 0.0),
    ];

    // One plan, two different operands.
    for scale in [1.0f64, 10.0] {
        let a: Vec<C64> = (1..=4)
            .map(|x| C64::new(x as f64 * scale, x as f64))
            .collect();
        let mut d = vec![C64::new(0.0, 0.0); 4];

        plan.run(
            C64::new(1.0, 0.0),
            TensorView::new(&a, &l, &ia).conj(),
            TensorView::new(&identity, &l, &ib),
            C64::new(0.0, 0.0),
            None,
            TensorViewMut::new(&mut d, &l, &id),
        )?;

        let want: Vec<C64> = a.iter().map(|z| z.conj()).collect();
        assert_eq!(d, want);
        println!(
            "complex, conjugated  ik,kj->ij  scale {scale:>4}  ->  [{}]",
            d.iter()
                .map(|z| format!("{}{:+}i", z.re, z.im))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    // Handing `run` a view that disagrees with its plan is an error, not a
    // silently unconjugated result.
    let a = vec![C64::new(1.0, 1.0); 4];
    let mut d = vec![C64::new(0.0, 0.0); 4];
    let err = plan
        .run(
            C64::new(1.0, 0.0),
            TensorView::new(&a, &l, &ia), // plan says conjugate; this does not
            TensorView::new(&identity, &l, &ib),
            C64::new(0.0, 0.0),
            None,
            TensorViewMut::new(&mut d, &l, &id),
        )
        .unwrap_err();
    println!("mismatched op rejected: {err}");
    Ok(())
}

fn report(what: &str, spec: &str, d: &[f64]) {
    println!("{what:<20} {spec:<14} ->  {d:?}");
}
