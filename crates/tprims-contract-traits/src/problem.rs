use crate::{Error, Result, Scalar};

/// Conjugation applied to an operand while it is read.
///
/// # Examples
///
/// ```
/// assert_eq!(tprims_contract_traits::Conj::default(), tprims_contract_traits::Conj::No);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Conj {
    /// Use the elements as stored.
    #[default]
    No,
    /// Use their complex conjugates (no effect for real types).
    Yes,
}

/// Which axes of the two operands are contracted and which are batch axes.
///
/// Free axes (all others) keep their order. The output layout is
/// `[lhs_free..., rhs_free..., batch...]`, batch axes in the listed order.
///
/// # Examples
///
/// ```
/// let bmm = tprims_contract_traits::DotGeneral::new(&[2], &[1], &[0], &[0]);
/// assert_eq!(bmm.validate(&[3, 5, 4], &[3, 4, 2]).unwrap().out_dims, vec![5, 2, 3]);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DotGeneral {
    /// Contracted axes of A.
    pub lhs_contract: Vec<usize>,
    /// Contracted axes of B, paired with `lhs_contract`.
    pub rhs_contract: Vec<usize>,
    /// Batch axes of A.
    pub lhs_batch: Vec<usize>,
    /// Batch axes of B, paired with `lhs_batch`.
    pub rhs_batch: Vec<usize>,
}

/// A validated problem shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shape {
    /// Free axes of A, in order.
    pub lhs_free: Vec<usize>,
    /// Free axes of B, in order.
    pub rhs_free: Vec<usize>,
    /// Output extents `[lhs_free, rhs_free, batch]`.
    pub out_dims: Vec<usize>,
}

impl DotGeneral {
    /// A configuration from axis lists.
    pub fn new(
        lhs_contract: &[usize],
        rhs_contract: &[usize],
        lhs_batch: &[usize],
        rhs_batch: &[usize],
    ) -> Self {
        Self {
            lhs_contract: lhs_contract.to_vec(),
            rhs_contract: rhs_contract.to_vec(),
            lhs_batch: lhs_batch.to_vec(),
            rhs_batch: rhs_batch.to_vec(),
        }
    }

    /// Validate against operand extents.
    ///
    /// # Errors
    ///
    /// [`Error::Config`] for unequal list lengths, repeated or out-of-range
    /// axes; [`Error::Shape`] for mismatched extents.
    pub fn validate(&self, a: &[usize], b: &[usize]) -> Result<Shape> {
        if self.lhs_contract.len() != self.rhs_contract.len()
            || self.lhs_batch.len() != self.rhs_batch.len()
        {
            return Err(Error::Config(
                "contract/batch lists of A and B differ in length".into(),
            ));
        }
        let check = |name: &str, rank: usize, c: &[usize], bt: &[usize]| -> Result<Vec<usize>> {
            let mut seen = vec![false; rank];
            for &x in c.iter().chain(bt) {
                if x >= rank {
                    return Err(Error::Config(format!(
                        "{name} axis {x} out of range for rank {rank}"
                    )));
                }
                if seen[x] {
                    return Err(Error::Config(format!("{name} axis {x} listed twice")));
                }
                seen[x] = true;
            }
            Ok((0..rank).filter(|&x| !seen[x]).collect())
        };
        let lhs_free = check("A", a.len(), &self.lhs_contract, &self.lhs_batch)?;
        let rhs_free = check("B", b.len(), &self.rhs_contract, &self.rhs_batch)?;
        for (&l, &r) in self
            .lhs_contract
            .iter()
            .zip(&self.rhs_contract)
            .chain(self.lhs_batch.iter().zip(&self.rhs_batch))
        {
            if a[l] != b[r] {
                return Err(Error::Shape(format!(
                    "A axis {l} has extent {}, B axis {r} has {}",
                    a[l], b[r]
                )));
            }
        }
        let out_dims = lhs_free
            .iter()
            .map(|&x| a[x])
            .chain(rhs_free.iter().map(|&x| b[x]))
            .chain(self.lhs_batch.iter().map(|&x| a[x]))
            .collect();
        Ok(Shape {
            lhs_free,
            rhs_free,
            out_dims,
        })
    }
}

/// Extents and signed element strides of one operand.
///
/// # Examples
///
/// ```
/// let l = tprims_contract_traits::Layout::new(&[2, 3], &[1, 2]);
/// assert!(l.matches(&[2, 3], &[1, 2]));
/// assert!(!l.matches(&[2, 3], &[3, 1]));
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    /// Extent of each axis.
    pub dims: Vec<usize>,
    /// Signed element stride of each axis.
    pub strides: Vec<isize>,
}

impl Layout {
    /// A layout from extents and strides.
    pub fn new(dims: &[usize], strides: &[isize]) -> Self {
        Self {
            dims: dims.to_vec(),
            strides: strides.to_vec(),
        }
    }

    /// Whether a view with these extents and strides has exactly this layout.
    pub fn matches(&self, dims: &[usize], strides: &[isize]) -> bool {
        self.dims == dims && self.strides == strides
    }
}

/// The semantic contraction problem: what to compute, on which layouts.
///
/// The storage dtype is the scalar type parameter of the backend and plan.
/// The operation is `C = alpha * dot_general(op(A), op(B)) + beta * C`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Problem {
    /// Contracted and batch axes.
    pub dot: DotGeneral,
    /// Layout of A.
    pub a: Layout,
    /// Layout of B.
    pub b: Layout,
    /// Layout of the output C.
    pub c: Layout,
    /// Conjugation of A and B.
    pub conj: (Conj, Conj),
}

/// What the consumer demands of the plan beyond the numerical result.
///
/// # Examples
///
/// ```
/// let r = tprims_contract_traits::Requirements::new().no_materialize(true);
/// assert!(r.no_materialize);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct Requirements {
    /// Refuse a plan that copies any operand ([`Error::WouldMaterialize`]).
    pub no_materialize: bool,
}

impl Requirements {
    /// No requirements.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the copy-refusal flag.
    #[must_use]
    pub fn no_materialize(mut self, yes: bool) -> Self {
        self.no_materialize = yes;
        self
    }
}

/// The host width a plan is expected to run on; a planning hint, not a
/// binding. A plan runs on any compatible host width.
///
/// # Examples
///
/// ```
/// assert_eq!(tprims_contract_traits::PlanningBudget::serial().threads, 1);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanningBudget {
    /// Expected most threads an execution occupies (at least 1).
    pub threads: usize,
}

impl PlanningBudget {
    /// A budget of `threads` (clamped to at least one).
    pub fn new(threads: usize) -> Self {
        Self {
            threads: threads.max(1),
        }
    }

    /// One thread.
    pub fn serial() -> Self {
        Self { threads: 1 }
    }
}

/// The result of canonical validation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Validated {
    /// Free axes and output extents.
    pub shape: Shape,
    /// Some contracted extent is zero: the product has no terms.
    pub k_empty: bool,
    /// Every axis is a batch axis (Hadamard product).
    pub all_batch: bool,
}

type Lay<'a> = (&'a [usize], &'a [isize]);

/// Whether distinct indices of a layout map to distinct addresses.
///
/// Sufficient test on the sorted strides; an empty layout is injective.
pub fn is_injective_layout(extents_strides: &[(usize, isize)]) -> bool {
    if extents_strides.iter().any(|&(e, _)| e == 0) {
        return true;
    }
    let mut axes: Vec<(usize, usize)> = extents_strides
        .iter()
        .filter(|(e, _)| *e > 1)
        .map(|&(e, s)| (e, s.unsigned_abs()))
        .collect();
    axes.sort_by_key(|&(_, s)| s);
    let mut span = 1usize;
    for (e, s) in axes {
        if s < span {
            return false;
        }
        match s.checked_mul(e) {
            Some(next) => span = next,
            None => return false,
        }
    }
    true
}

/// The canonical validation every backend shares: rank agreement, axis
/// roles, extents, bounded element counts for scalar `T`, the output extents
/// and an injective output mapping. Runs before any no-op shortcut.
///
/// # Errors
///
/// [`Error::Config`], [`Error::Shape`], [`Error::AliasedOutput`].
pub fn validate_layouts<T: Scalar>(
    dot: &DotGeneral,
    a: Lay<'_>,
    b: Lay<'_>,
    c: Lay<'_>,
) -> Result<Validated> {
    for (name, l) in [("A", &a), ("B", &b), ("C", &c)] {
        if l.0.len() != l.1.len() {
            return Err(Error::Shape(format!(
                "{name}: {} extents, {} strides",
                l.0.len(),
                l.1.len()
            )));
        }
    }
    let shape = dot.validate(a.0, b.0)?;
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
    let k_empty = dot.lhs_contract.iter().any(|&x| a.0[x] == 0);
    let all_batch =
        shape.lhs_free.is_empty() && shape.rhs_free.is_empty() && dot.lhs_contract.is_empty();
    Ok(Validated {
        shape,
        k_empty,
        all_batch,
    })
}

impl Problem {
    /// A problem from its parts.
    pub fn new(dot: DotGeneral, a: Layout, b: Layout, c: Layout, conj: (Conj, Conj)) -> Self {
        Self { dot, a, b, c, conj }
    }

    /// Canonical validation, see [`validate_layouts`].
    ///
    /// # Errors
    ///
    /// As [`validate_layouts`].
    pub fn validate<T: Scalar>(&self) -> Result<Validated> {
        validate_layouts::<T>(
            &self.dot,
            (&self.a.dims, &self.a.strides),
            (&self.b.dims, &self.b.strides),
            (&self.c.dims, &self.c.strides),
        )
    }

    /// Check actual view layouts against the problem, before any write.
    ///
    /// # Errors
    ///
    /// [`Error::LayoutMismatch`] naming the first differing operand.
    pub fn check_views(&self, a: Lay<'_>, b: Lay<'_>, c: Lay<'_>) -> Result<()> {
        for (name, planned, v) in [("A", &self.a, a), ("B", &self.b, b), ("C", &self.c, c)] {
            if !planned.matches(v.0, v.1) {
                return Err(Error::LayoutMismatch(format!(
                    "{name}: {:?} / {:?}",
                    v.0, v.1
                )));
            }
        }
        Ok(())
    }
}
