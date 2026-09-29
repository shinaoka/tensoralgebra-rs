use tprims_exec::{Exec, Pool, WidthPolicy};

use super::schedule;
use crate::util::factor_policy;

#[test]
fn many_parallel_sized_items_are_split_across_the_batch() {
    let tp = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let pool = Pool::borrow(&tp);
    let exec = Exec::rayon(&pool);
    // n = 128 LU solve: about 2.1 Mflop per item. With the generic policy one
    // item would take 3 threads; with 16 items the batch must be split.
    let item = 2.1e6;
    let s = schedule(&exec, 16, item, WidthPolicy::default());
    assert_eq!(s.outer, Some(4), "{s:?}");
    // Two huge items: inner parallelism, items in turn.
    let s = schedule(&exec, 2, 5e9, WidthPolicy::default());
    assert_eq!(s.outer, None);
    assert!(s.inner > 1);
    // LU items below the factor threshold never get inner threads.
    let s = schedule(&exec, 3, item, factor_policy());
    assert_eq!(s.inner, 1);
    // Serial context.
    let s = schedule(&Exec::serial(), 1000, item, WidthPolicy::default());
    assert_eq!((s.outer, s.inner), (None, 1));
}
