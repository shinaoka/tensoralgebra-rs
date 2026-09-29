use tprims_contract::{add, permute, Error};
use tprims_exec::{Exec, Pool};

mod common;
use common::*;

#[test]
fn permute_copies_with_an_axis_permutation() {
    let a = T::<f64>::new(&[3, 4, 5], 1);
    let perm = [2, 0, 1];
    let mut c = T::<f64>::new(&[5, 3, 4], 9).reversed();
    permute(&Exec::serial(), &a.view(), &perm, &mut c.view_mut()).unwrap();
    for_each_index(&[3, 4, 5], |i| {
        assert_eq!(c.get(&[i[2], i[0], i[1]]), a.get(i))
    });
    let mut bad = T::<f64>::new(&[5, 4, 3], 9);
    assert!(matches!(
        permute(&Exec::serial(), &a.view(), &perm, &mut bad.view_mut()),
        Err(Error::Shape(_))
    ));
    assert!(matches!(
        permute(&Exec::serial(), &a.view(), &[0, 0, 1], &mut bad.view_mut()),
        Err(Error::Config(_))
    ));
}

#[test]
fn add_scales_and_accumulates_including_beta_zero() {
    let tp = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let pool = Pool::borrow(&tp);
    for exec in [Exec::serial(), Exec::rayon(&pool)] {
        let a = T::<f64>::new(&[300, 200], 1).restride(&[1, 0]);
        let c0 = T::<f64>::new(&[300, 200], 2);
        let mut c = c0.clone();
        add(&exec, 2.0, &a.view(), 0.5, &mut c.view_mut()).unwrap();
        for_each_index(&[300, 200], |i| {
            assert_eq!(c.get(i), 2.0 * a.get(i) + 0.5 * c0.get(i))
        });
        let mut n = T::<f64> {
            data: vec![f64::NAN; 60000],
            dims: vec![300, 200],
            strides: vec![1, 300],
            offset: 0,
        };
        add(&exec, 1.0, &a.view(), 0.0, &mut n.view_mut()).unwrap();
        for_each_index(&[300, 200], |i| assert_eq!(n.get(i), a.get(i)));
    }
}
