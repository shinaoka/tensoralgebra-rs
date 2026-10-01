//! Many independent contractions, with the **batch** as the parallel axis.
//!
//! Parallelising over a batch of `n` contractions needs no barrier: items are
//! wholly independent, so the work is plain fork-join over the host's
//! [`Exec`], and this crate spawns no thread of its own.
//!
//! # What this does not do
//!
//! * **The batch axis is the only parallel axis.** Each item runs serially
//!   however wide the `Exec` is, so the two axes never nest.
//! * **The split is static and contiguous**, by item count. Balancing it by
//!   estimated work is the obvious refinement and is deliberately not guessed at:
//!   D47 measured that block-scatter load imbalance does not cost on the dense
//!   path, and the case where it plausibly does — **block-sparse, where items
//!   differ in size rather than in regularity** — is not built either. That is
//!   where a dynamic claim over the batch belongs, and it is what TBLIS uses a
//!   dynamic atomic-claim scheduler for while keeping static partitioning for
//!   dense (part 15). Same division, reached independently.

use std::sync::Mutex;

use tprims_exec::Exec;

use crate::element::Element;
use crate::error::Result;
use crate::kernel::KernelSet;
use crate::plan::Plan;
use crate::{TensorView, TensorViewMut};

/// One contraction of a batch: a plan and the operands to run it against.
///
/// Holding these in a `&mut [BatchItem]` is what makes the parallel execution
/// sound without a single unsafe block on the caller's side: each item's `d` is a
/// `&mut` borrow, so the borrow checker has already proved the outputs disjoint.
/// The plans are shared references and may all be the *same* plan, which is the
/// common case — a batch of identically shaped contractions.
///
/// The derived [`Debug`](core::fmt::Debug) carries a `T: Debug` bound, which
/// constrains nothing that can occur: [`Element`] requires `Debug` as a
/// supertrait, so every `T` that reaches [`contract_batched`] already has it.
/// There is deliberately no `Clone` or `Copy` — `d` is an exclusive borrow, and
/// that borrow is the whole soundness argument above.
#[derive(Debug)]
pub struct BatchItem<'a, T> {
    /// The plan. Built once and shared across items where the shape allows.
    pub plan: &'a Plan,
    /// Scales the product.
    pub alpha: T,
    /// Left operand.
    pub a: TensorView<'a, T>,
    /// Right operand.
    pub b: TensorView<'a, T>,
    /// Scales `c`. Ignored when `c` is `None`, exactly as in [`crate::contract`].
    pub beta: T,
    /// Optional accumulate-from operand. `None` means `d` is overwritten.
    pub c: Option<TensorView<'a, T>>,
    /// Output. The exclusive borrow is load-bearing: see the type's documentation.
    pub d: TensorViewMut<'a, T>,
}

/// Run every item, parallelising over the batch on at most `exec.budget()`
/// threads of `exec` (serially on [`Exec::Serial`]).
///
/// # Ordering and results
///
/// Items are independent, so the batch imposes no order between them and the
/// result of each is **bitwise identical to running it alone** — each item runs on
/// exactly one thread, through the same serial path
/// [`Plan::run`](crate::plan::Plan::run) takes. The batch axis adds no reduction
/// and no accumulation, so there is nothing for a width to change.
///
/// # All or nothing
///
/// Every item's bounds are checked against its plan's scatter vectors **before any
/// item runs**, so a batch containing one bad item writes to no output at all. That
/// is a stronger guarantee than looping over [`Plan::run`] gives, and it is the
/// reason to prefer this even when serial: a partially executed batch leaves the
/// caller unable to say which outputs are valid.
///
/// # Threads
///
/// The budget bounds the *batch* axis, and the items themselves run serially. It
/// is further capped by the item count: the lanes are `min(budget, items)`.
///
/// ```
/// use tensorcontract::batch::{contract_batched, BatchItem};
/// use tensorcontract::plan::Operand;
/// use tensorcontract::{Layout, Plan, TensorView, TensorViewMut};
///
/// let (i, j, k) = (b'i' as i64, b'j' as i64, b'k' as i64);
/// let l = Layout::col_major(&[2, 2]);
/// let (ia, ib, id) = ([i, k], [k, j], [i, j]);
///
/// // One plan, shared by every item — the common case for a batch.
/// let plan = Plan::new(
///     Operand::new(&l, &ia),
///     Operand::new(&l, &ib),
///     None,
///     Operand::new(&l, &id),
/// )
/// .unwrap();
///
/// let a0 = vec![1.0f64, 2.0, 3.0, 4.0];
/// let a1 = vec![5.0f64, 6.0, 7.0, 8.0];
/// let identity = vec![1.0f64, 0.0, 0.0, 1.0];
/// let mut d0 = vec![0.0f64; 4];
/// let mut d1 = vec![0.0f64; 4];
///
/// // The outputs are distinct `&mut` borrows, which is what proves them
/// // disjoint — no unsafe on the caller's side.
/// let mut items = vec![
///     BatchItem {
///         plan: &plan,
///         alpha: 1.0,
///         a: TensorView::new(&a0, &l, &ia),
///         b: TensorView::new(&identity, &l, &ib),
///         beta: 0.0,
///         c: None,
///         d: TensorViewMut::new(&mut d0, &l, &id),
///     },
///     BatchItem {
///         plan: &plan,
///         alpha: 1.0,
///         a: TensorView::new(&a1, &l, &ia),
///         b: TensorView::new(&identity, &l, &ib),
///         beta: 0.0,
///         c: None,
///         d: TensorViewMut::new(&mut d1, &l, &id),
///     },
/// ];
///
/// contract_batched(&mut items, &tprims_exec::Exec::serial()).unwrap();
/// drop(items);                       // release the borrows on d0 / d1
///
/// assert_eq!(d0, a0);                // multiplying by the identity
/// assert_eq!(d1, a1);
/// ```
pub fn contract_batched<T>(items: &mut [BatchItem<'_, T>], exec: &Exec<'_>) -> Result<()>
where
    T: Element + Send + Sync,
    T::Real: KernelSet,
{
    // Validate everything first. `check_bounds` is what `Plan::run` would do per
    // item; doing it for all of them up front converts "some outputs are written"
    // into "no outputs are written" on failure.
    for it in items.iter() {
        crate::resolve::validate::<T>(it.plan)?;
        it.plan.check_bounds(
            it.a.data.len(),
            it.b.data.len(),
            it.c.as_ref().map(|c| c.data.len()),
            it.d.data.len(),
        )?;
    }
    if items.is_empty() {
        return Ok(());
    }

    // Width one is the plain loop, with no allocation.
    if exec.budget() == 1 || items.len() == 1 {
        items.iter_mut().for_each(run_item);
        return Ok(());
    }

    // Lanes of contiguous items, one per worker; at width one this is the
    // plain loop on the caller. `for_each_partition` takes `Fn`, so each item
    // sits behind its own mutex, locked exactly once and never contended: the
    // `&mut` outputs already prove the items disjoint.
    let cells: Vec<Mutex<&mut BatchItem<'_, T>>> = items.iter_mut().map(Mutex::new).collect();
    exec.for_each_partition(cells.len(), &|i| {
        let mut it = cells[i].lock().unwrap_or_else(|e| e.into_inner());
        run_item(&mut it);
    });
    Ok(())
}

/// One item, serially. Bounds are already checked by the caller.
fn run_item<T>(it: &mut BatchItem<'_, T>)
where
    T: Element,
    T::Real: KernelSet,
{
    let cptr = match &it.c {
        Some(c) => c.data.as_ptr(),
        None => it.d.data.as_ptr(),
    };
    let beta = if it.c.is_none() { T::zero() } else { it.beta };
    // SAFETY: bounds were validated for every item before any of them ran, and
    // `d` is an exclusive borrow so it cannot alias `a`, `b` or `c`. The serial `Exec`
    // makes the item run on its own thread, which is what makes the batch the
    // only parallel axis.
    unsafe {
        crate::driver::execute_capped(
            it.plan,
            it.alpha,
            it.a.data.as_ptr(),
            it.b.data.as_ptr(),
            beta,
            cptr,
            it.d.data.as_mut_ptr(),
            &Exec::Serial,
            None,
            None,
            None,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::Layout;

    /// `n` independent `ij,jk->ik` contractions with distinct data, run batched and
    /// run one at a time, compared **bitwise**. That equivalence is the whole
    /// contract of this module, and it is checked at several batch widths so a
    /// chunking error on the tail is not missed.
    #[test]
    fn batched_matches_one_at_a_time_bitwise() {
        for n in [1usize, 2, 3, 7, 16] {
            for threads in [1usize, 2, 4, 8] {
                let tp = rayon::ThreadPoolBuilder::new()
                    .num_threads(8)
                    .build()
                    .unwrap();
                let pool = tprims_exec::Pool::borrow(&tp);
                let exec = Exec::rayon(&pool).with_budget(threads).unwrap();
                let (m, k, nn) = (12usize, 9usize, 7usize);
                let la = Layout::col_major(&[m as i64, k as i64]);
                let lb = Layout::col_major(&[k as i64, nn as i64]);
                let ld = Layout::col_major(&[m as i64, nn as i64]);
                let plan = Plan::new(
                    crate::plan::Operand::new(&la, &[0, 2]),
                    crate::plan::Operand::new(&lb, &[2, 1]),
                    None,
                    crate::plan::Operand::new(&ld, &[0, 1]),
                )
                .unwrap();

                // Distinct data per item, so a batch that ran the wrong item, ran
                // one twice, or skipped one cannot pass.
                let a: Vec<Vec<f64>> = (0..n)
                    .map(|i| (0..m * k).map(|j| (i * 31 + j) as f64 * 0.5).collect())
                    .collect();
                let b: Vec<Vec<f64>> = (0..n)
                    .map(|i| (0..k * nn).map(|j| (i * 17 + j) as f64 * 0.25).collect())
                    .collect();

                let mut want = vec![vec![0.0f64; m * nn]; n];
                for i in 0..n {
                    plan.run(
                        1.0,
                        TensorView::new(&a[i], &la, &[0, 2]),
                        TensorView::new(&b[i], &lb, &[2, 1]),
                        0.0,
                        None,
                        TensorViewMut::new(&mut want[i], &ld, &[0, 1]),
                    )
                    .unwrap();
                }

                let mut got = vec![vec![0.0f64; m * nn]; n];
                {
                    let mut items: Vec<BatchItem<'_, f64>> = got
                        .iter_mut()
                        .enumerate()
                        .map(|(i, d)| BatchItem {
                            plan: &plan,
                            alpha: 1.0,
                            a: TensorView::new(&a[i], &la, &[0, 2]),
                            b: TensorView::new(&b[i], &lb, &[2, 1]),
                            beta: 0.0,
                            c: None,
                            d: TensorViewMut::new(d, &ld, &[0, 1]),
                        })
                        .collect();
                    contract_batched(&mut items, &exec).unwrap();
                }
                assert_eq!(got, want, "n = {n}, threads = {threads}");
            }
        }
    }

    /// A bad item must stop the whole batch **before** anything is written. This is
    /// the guarantee a loop over `Plan::run` cannot give, so it gets its own test.
    #[test]
    fn a_bad_item_leaves_every_output_untouched() {
        let la = Layout::col_major(&[4, 3]);
        let lb = Layout::col_major(&[3, 2]);
        let ld = Layout::col_major(&[4, 2]);
        let plan = Plan::new(
            crate::plan::Operand::new(&la, &[0, 2]),
            crate::plan::Operand::new(&lb, &[2, 1]),
            None,
            crate::plan::Operand::new(&ld, &[0, 1]),
        )
        .unwrap();
        let a = vec![1.0f64; 12];
        let b = vec![1.0f64; 6];
        let short = vec![1.0f64; 2]; // too small for `a`: the plan will reject it
        let mut d0 = vec![0.0f64; 8];
        let mut d1 = vec![0.0f64; 8];
        {
            let (first, second) = (&mut d0, &mut d1);
            let mut items = vec![
                BatchItem {
                    plan: &plan,
                    alpha: 1.0,
                    a: TensorView::new(&a, &la, &[0, 2]),
                    b: TensorView::new(&b, &lb, &[2, 1]),
                    beta: 0.0,
                    c: None,
                    d: TensorViewMut::new(first, &ld, &[0, 1]),
                },
                BatchItem {
                    plan: &plan,
                    alpha: 1.0,
                    a: TensorView::new(&short, &la, &[0, 2]),
                    b: TensorView::new(&b, &lb, &[2, 1]),
                    beta: 0.0,
                    c: None,
                    d: TensorViewMut::new(second, &ld, &[0, 1]),
                },
            ];
            assert!(contract_batched(&mut items, &Exec::serial()).is_err());
        }
        assert!(
            d0.iter().all(|&x| x == 0.0),
            "the good item ran despite a later item being invalid"
        );
        assert!(d1.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn an_empty_batch_is_ok() {
        let mut items: Vec<BatchItem<'_, f64>> = Vec::new();
        contract_batched(&mut items, &Exec::serial()).unwrap();
    }
}
