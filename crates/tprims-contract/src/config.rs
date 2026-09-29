use crate::{Error, Result};

/// Which axes of the two operands are contracted and which are batch axes.
///
/// Free axes (all others) keep their order. The output layout is
/// `[lhs_free..., rhs_free..., batch...]`, batch axes in the listed order.
///
/// # Examples
///
/// ```
/// let bmm = tprims_contract::DotGeneral::new(&[2], &[1], &[0], &[0]);
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
