//! Error type.

use core::fmt;

/// Everything that can go wrong when building or executing a contraction plan.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// `extents.len() != strides.len()`.
    RankMismatch { extents: usize, strides: usize },
    /// The number of index labels does not match the tensor rank.
    LabelCountMismatch {
        tensor: &'static str,
        nmode: usize,
        nlabel: usize,
    },
    /// A label appears with different extents in different operands, or a
    /// label repeated inside one tensor has inconsistent extents.
    ExtentMismatch {
        label: i64,
        expected: i64,
        found: i64,
    },
    /// A label appears only in the output. TAPP calls this "case 5"
    /// (isolated output index / broadcast) and does not require support.
    BroadcastIndexUnsupported { label: i64 },
    /// `C` and `D` must describe the same set of labels.
    OutputLabelMismatch,
    /// A negative extent was supplied.
    NegativeExtent { label: i64, extent: i64 },
    /// The requested element type is not supported by this entry point.
    UnsupportedDatatype,
    /// A null data pointer was supplied for a non-empty tensor.
    NullPointer { tensor: &'static str },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::RankMismatch { extents, strides } => {
                write!(f, "rank mismatch: {extents} extents but {strides} strides")
            }
            Error::LabelCountMismatch {
                tensor,
                nmode,
                nlabel,
            } => write!(
                f,
                "tensor {tensor} has {nmode} modes but {nlabel} index labels"
            ),
            Error::ExtentMismatch {
                label,
                expected,
                found,
            } => write!(
                f,
                "index label {label} has inconsistent extents: {expected} vs {found}"
            ),
            Error::BroadcastIndexUnsupported { label } => write!(
                f,
                "index label {label} appears only in the output; broadcasting \
                 (TAPP case 5) is not supported"
            ),
            Error::OutputLabelMismatch => {
                write!(f, "C and D must have the same set of index labels")
            }
            Error::NegativeExtent { label, extent } => {
                write!(f, "index label {label} has negative extent {extent}")
            }
            Error::UnsupportedDatatype => write!(f, "unsupported element type"),
            Error::NullPointer { tensor } => {
                write!(f, "null data pointer for non-empty tensor {tensor}")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

pub type Result<T> = core::result::Result<T, Error>;
