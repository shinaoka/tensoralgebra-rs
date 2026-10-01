use tensorcontract::Element;
use tprims_blas::{GemmPolicy, Scalar};
use tprims_exec::Exec;

#[derive(Clone, Copy)]
struct Ptr<T>(*mut T);
// SAFETY: dereferenced only at validated, disjoint (for writes) offsets.
unsafe impl<T> Send for Ptr<T> {}
unsafe impl<T> Sync for Ptr<T> {}

impl<T> Ptr<T> {
    fn get(self) -> *mut T {
        self.0
    }
}

/// Elementwise update `c = f(c, [in_0, ..])` over a strided iteration space.
///
/// Axes of extent one are dropped (their strides are unconstrained by the
/// view), the rest are walked with C's smallest-|stride| axis innermost, and
/// the outermost axis is split into barrier-free lanes on `exec`. When
/// `read_c` is false, `f` receives zero instead of reading C.
///
/// # Safety
///
/// `c` and every input origin are the elements at logical index 0 of
/// non-empty, bounds-checked views with these extents and strides; C is
/// exclusively borrowed and injective; inputs do not alias C.
pub(crate) unsafe fn zip_update<T: Scalar, const N: usize>(
    exec: &Exec<'_>,
    dims: &[usize],
    c: (*mut T, &[isize]),
    ins: [(*const T, &[isize]); N],
    read_c: bool,
    f: &(dyn Fn(T, [T; N]) -> T + Sync),
) {
    if dims.contains(&0) {
        return;
    }
    // (extent, C stride, input strides), unit axes dropped, C-stride order.
    let mut axes: Vec<(usize, isize, [isize; N])> = (0..dims.len())
        .filter(|&k| dims[k] > 1)
        .map(|k| (dims[k], c.1[k], ins.map(|(_, s)| s[k])))
        .collect();
    axes.sort_by_key(|a| a.1.unsigned_abs());
    let total: usize = axes.iter().map(|a| a.0).product();
    let (cp, ip) = (Ptr(c.0), ins.map(|(p, _)| Ptr(p as *mut T)));
    let (outer, inner) = match axes.split_last() {
        Some((o, rest)) => (*o, rest.to_vec()),
        None => ((1, 0, [0; N]), vec![]),
    };
    let p = GemmPolicy::default();
    let lanes = exec
        .width_for(total as f64 * 2.0 * p.ns_per_flop, &p.width)
        .min(outer.0)
        .max(1);
    let body = |l: usize| {
        let mut idx = vec![0usize; inner.len()];
        for t in l * outer.0 / lanes..(l + 1) * outer.0 / lanes {
            idx.iter_mut().for_each(|i| *i = 0);
            let mut co = t as isize * outer.1;
            let mut io: [isize; N] = std::array::from_fn(|j| t as isize * outer.2[j]);
            loop {
                // SAFETY: (idx, t) lies inside the validated extents, so every
                // offset is inside its view; lanes own disjoint t ranges and C
                // is injective.
                unsafe {
                    let cq = cp.get().offset(co);
                    let old = if read_c { *cq } else { <T as Element>::zero() };
                    let vals: [T; N] = std::array::from_fn(|j| *ip[j].get().offset(io[j]));
                    *cq = f(old, vals);
                }
                let mut k = 0;
                while k < inner.len() {
                    idx[k] += 1;
                    co += inner[k].1;
                    for j in 0..N {
                        io[j] += inner[k].2[j];
                    }
                    if idx[k] < inner[k].0 {
                        break;
                    }
                    let back = inner[k].0 as isize;
                    co -= inner[k].1 * back;
                    for j in 0..N {
                        io[j] -= inner[k].2[j] * back;
                    }
                    idx[k] = 0;
                    k += 1;
                }
                if k == inner.len() {
                    break;
                }
            }
        }
    };
    exec.for_each_partition(lanes, &body);
}

/// `C = beta * C`; `beta == 0` writes zeros without reading.
///
/// # Safety
///
/// As [`zip_update`] for C alone.
pub(crate) unsafe fn scale<T: Scalar>(origin: *mut T, dims: &[usize], strides: &[isize], beta: T) {
    let read = beta != <T as Element>::zero();
    // SAFETY: forwarded.
    unsafe {
        zip_update(
            &Exec::serial(),
            dims,
            (origin, strides),
            [],
            read,
            &move |c, []| Element::mul(beta, c),
        )
    };
}

/// A kernel-selection failure as the interface error, keeping the typed
/// [`SelectError`](tprims_kernel::SelectError) as the downcastable source.
pub(crate) fn select_err(e: tprims_kernel::SelectError) -> crate::Error {
    crate::Error::backend(e)
}
