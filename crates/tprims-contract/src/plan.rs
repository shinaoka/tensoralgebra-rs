use strided_view::{StridedView, StridedViewMut};
use tensorcontract::Element;
use tprims_blas::{is_injective_layout, Conj, Scalar};
use tprims_exec::Exec;

use crate::config::{DotGeneral, Shape};
use crate::permute_gemm::{self, PgPlan};
use crate::tblis::{self, TbPlan};
use crate::{Error, Result};

/// Which implementation a plan uses.
///
/// # Examples
///
/// ```
/// assert_ne!(tprims_contract::Strategy::PermuteGemm, tprims_contract::Strategy::Tblis);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Strategy {
    /// Library choice: an elementwise pass for all-batch (Hadamard)
    /// problems, otherwise [`Strategy::PermuteGemm`] (faster than TBLIS-style
    /// on every fusable case of the Phase 1c corpus).
    Auto,
    /// Fuse to strided batched GEMM, copying non-fusable operands once.
    PermuteGemm,
    /// TBLIS-style direct contraction (tensorcontract, by Lukas Devos).
    Tblis,
}

/// Planning options.
///
/// # Examples
///
/// ```
/// assert!(!tprims_contract::Flags::default().no_materialize);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Flags {
    /// Refuse a plan that would copy any operand ([`Error::WouldMaterialize`]).
    pub no_materialize: bool,
}

/// What a plan runs.
///
/// # Examples
///
/// ```
/// let s = tprims_contract::Selected::PermuteGemm { materialized: [false; 3] };
/// assert!(matches!(s, tprims_contract::Selected::PermuteGemm { .. }));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Selected {
    /// Batched GEMM; which of A, B, C are copied into compact buffers.
    PermuteGemm {
        /// A, B, C copied.
        materialized: [bool; 3],
    },
    /// Direct contraction; nothing is copied.
    Tblis,
    /// All axes are batch axes (Hadamard product): one elementwise pass.
    Elementwise,
}

#[derive(Debug)]
enum Inner {
    Pg(Box<PgPlan>),
    Tb(Box<TbPlan>),
    /// A and B axis of each output (batch) axis.
    Elementwise {
        a_axes: Vec<usize>,
        b_axes: Vec<usize>,
    },
}

/// A validated contraction for fixed layouts, reusable across calls.
#[derive(Debug)]
pub struct ContractPlan<T> {
    layouts: [(Vec<usize>, Vec<isize>); 3],
    conj: (Conj, Conj),
    k_empty: bool,
    inner: Inner,
    _t: std::marker::PhantomData<fn() -> T>,
}

type Lay<'a> = (&'a [usize], &'a [isize]);

impl<T: Scalar> ContractPlan<T> {
    /// Validate `cfg` against the layouts (extents and element strides of A,
    /// B and C) and choose a strategy.
    ///
    /// # Errors
    ///
    /// [`Error::Config`], [`Error::Shape`] (including C's extents),
    /// [`Error::AliasedOutput`], [`Error::WouldMaterialize`] under
    /// `no_materialize`, [`Error::Backend`] when tensorcontract rejects the
    /// problem.
    pub fn new(
        cfg: &DotGeneral,
        a: Lay<'_>,
        b: Lay<'_>,
        c: Lay<'_>,
        conj: (Conj, Conj),
        strategy: Strategy,
        flags: Flags,
    ) -> Result<Self> {
        for (name, l) in [("A", &a), ("B", &b), ("C", &c)] {
            if l.0.len() != l.1.len() {
                return Err(Error::Shape(format!(
                    "{name}: {} extents, {} strides",
                    l.0.len(),
                    l.1.len()
                )));
            }
        }
        let shape: Shape = cfg.validate(a.0, b.0)?;
        for (name, l) in [("A", &a), ("B", &b), ("C", &c)] {
            let n = l.0.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d));
            if n.and_then(|n| n.checked_mul(std::mem::size_of::<T>()))
                .is_none_or(|b| b > isize::MAX as usize)
            {
                return Err(Error::Shape(format!("{name}: element count overflows")));
            }
        }
        if shape.out_dims != c.0 {
            return Err(Error::Shape(format!(
                "C has extents {:?}, expected {:?}",
                c.0, shape.out_dims
            )));
        }
        let cl: Vec<(usize, isize)> = c.0.iter().copied().zip(c.1.iter().copied()).collect();
        if !is_injective_layout(&cl) {
            return Err(Error::AliasedOutput);
        }
        let dims = [a.0, b.0, c.0];
        let strides = [a.1, b.1, c.1];
        let k_empty = cfg.lhs_contract.iter().any(|&x| a.0[x] == 0);
        let all_batch =
            shape.lhs_free.is_empty() && shape.rhs_free.is_empty() && cfg.lhs_contract.is_empty();
        let inner = match strategy {
            Strategy::Auto | Strategy::PermuteGemm if all_batch => Inner::Elementwise {
                a_axes: cfg.lhs_batch.clone(),
                b_axes: cfg.rhs_batch.clone(),
            },
            Strategy::Tblis => Inner::Tb(Box::new(tblis::plan(cfg, &shape, dims, strides, conj)?)),
            Strategy::Auto | Strategy::PermuteGemm => Inner::Pg(Box::new(permute_gemm::plan(
                cfg,
                &shape,
                dims,
                strides,
                flags.no_materialize,
            )?)),
        };
        Ok(Self {
            layouts: [0, 1, 2].map(|o| (dims[o].to_vec(), strides[o].to_vec())),
            conj,
            k_empty,
            inner,
            _t: std::marker::PhantomData,
        })
    }

    /// The implementation this plan runs.
    pub fn selected(&self) -> Selected {
        match &self.inner {
            Inner::Pg(p) => Selected::PermuteGemm {
                materialized: p.materialized,
            },
            Inner::Tb(_) => Selected::Tblis,
            Inner::Elementwise { .. } => Selected::Elementwise,
        }
    }

    /// `C = alpha * contract(op(A), op(B)) + beta * C` on views with exactly
    /// the planned layouts. `beta == 0` never reads C; `alpha == 0` or an
    /// empty contraction never reads A or B.
    ///
    /// # Errors
    ///
    /// [`Error::LayoutMismatch`] when a view differs from the plan (nothing is
    /// written); [`Error::Backend`] from a lower layer.
    pub fn execute(
        &self,
        exec: &Exec<'_>,
        alpha: T,
        a: &StridedView<'_, T>,
        b: &StridedView<'_, T>,
        beta: T,
        c: &mut StridedViewMut<'_, T>,
    ) -> Result<()> {
        let views = [
            (a.dims(), a.strides()),
            (b.dims(), b.strides()),
            (c.dims(), c.strides()),
        ];
        for (o, (name, v)) in ["A", "B", "C"].iter().zip(views).enumerate() {
            if v.0 != self.layouts[o].0.as_slice() || v.1 != self.layouts[o].1.as_slice() {
                return Err(Error::LayoutMismatch(format!(
                    "{name}: {:?} / {:?}",
                    v.0, v.1
                )));
            }
        }
        if c.dims().contains(&0) {
            return Ok(());
        }
        if self.k_empty || alpha == <T as Element>::zero() || a.is_empty() || b.is_empty() {
            let (dims, strides) = (c.dims().to_vec(), c.strides().to_vec());
            // SAFETY: C is non-empty and bounds-checked; exclusive borrow.
            unsafe { crate::util::scale(c.as_mut_ptr(), &dims, &strides, beta) };
            return Ok(());
        }
        match &self.inner {
            Inner::Pg(p) => {
                permute_gemm::execute(p, exec, alpha, a, self.conj.0, b, self.conj.1, beta, c)
            }
            Inner::Tb(p) => tblis::execute(p, exec, alpha, a, b, beta, c),
            Inner::Elementwise { a_axes, b_axes } => {
                let (dims, cs) = (c.dims().to_vec(), c.strides().to_vec());
                let sa: Vec<isize> = a_axes.iter().map(|&x| a.strides()[x]).collect();
                let sb: Vec<isize> = b_axes.iter().map(|&x| b.strides()[x]).collect();
                let (ca, cb) = self.conj;
                let op = |x: T, cj: Conj| if cj == Conj::Yes { Element::conj(x) } else { x };
                let read = beta != <T as Element>::zero();
                // SAFETY: all three views are non-empty and bounds-checked;
                // A and B axes are permuted onto C's (batch) axes, whose
                // extents they share; C is exclusive and injective.
                unsafe {
                    crate::util::zip_update(
                        exec,
                        &dims,
                        (c.as_mut_ptr(), &cs),
                        [(a.ptr(), &sa), (b.ptr(), &sb)],
                        read,
                        &move |y, [x, z]| {
                            Element::add(
                                Element::mul(alpha, Element::mul(op(x, ca), op(z, cb))),
                                Element::mul(beta, y),
                            )
                        },
                    )
                };
                Ok(())
            }
        }
    }
}
