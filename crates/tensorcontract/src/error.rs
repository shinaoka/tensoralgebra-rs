//! Error type.
//!
//! Everything here describes a contraction that was rejected while being
//! *planned*, which is the only place this crate reports failure. The variants
//! fall into four groups:
//!
//! * **the description is internally inconsistent** — [`Error::RankMismatch`],
//!   [`Error::LabelCountMismatch`], [`Error::ExtentMismatch`],
//!   [`Error::NegativeExtent`], [`Error::OutputLabelMismatch`];
//! * **the contraction is well formed but out of scope** —
//!   [`Error::BroadcastIndexUnsupported`] (a label only in the output) and
//!   [`Error::UnsupportedDatatype`];
//! * **the shape cannot be addressed at all** —
//!   [`Error::ExtentProductOverflow`];
//! * **the data does not back the shape** — [`Error::NullPointer`], the one
//!   variant raised by [`crate::Plan::run`] rather than by
//!   [`crate::Plan::new`], because slice lengths are not known until then.
//!
//! Consequently a [`crate::Plan`] that exists is a plan that will run: there is
//! no execution-time failure mode left once the data has been bounds-checked.

use core::fmt;

/// Everything that can go wrong when building or executing a contraction plan.
///
/// Every variant is a rejected *description* of a contraction — a shape, label
/// or stride inconsistency — caught while planning. There is deliberately no
/// variant for a failure during execution: once [`crate::Plan::new`] has
/// returned, the only thing that can still be wrong is the data slices, and
/// [`crate::Plan::run`] checks those before touching anything.
///
/// `#[non_exhaustive]` because the mapping onto TAPP's error codes is the
/// stable contract, not this enumeration; match with a `_` arm.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// `extents.len() != strides.len()`.
    RankMismatch {
        /// Number of extents supplied.
        extents: usize,
        /// Number of strides supplied.
        strides: usize,
    },
    /// The number of index labels does not match the tensor rank.
    LabelCountMismatch {
        /// Which operand: `"A"`, `"B"`, `"C"` or `"D"`.
        tensor: &'static str,
        /// Modes the layout has.
        nmode: usize,
        /// Labels supplied for it.
        nlabel: usize,
    },
    /// A label appears with different extents in different operands, or a
    /// label repeated inside one tensor has inconsistent extents.
    ExtentMismatch {
        /// The offending label.
        label: i64,
        /// Extent seen first, and taken as authoritative for the message.
        expected: i64,
        /// The conflicting extent.
        found: i64,
    },
    /// A label appears only in the output. TAPP calls this "case 5"
    /// (isolated output index / broadcast) and does not require support.
    BroadcastIndexUnsupported {
        /// The output-only label.
        label: i64,
    },
    /// `C` and `D` must describe the same set of labels.
    ///
    /// They need not have the same strides or even the same mode order — `C` is
    /// read through its own scatter vectors — but a label in one and not the
    /// other would leave `beta * C` undefined at some output element.
    OutputLabelMismatch,
    /// A negative extent was supplied. Negative *strides* are legal and
    /// supported; negative extents are not a layout, they are a mistake.
    NegativeExtent {
        /// The label whose mode has the negative extent.
        label: i64,
        /// The extent as supplied.
        extent: i64,
    },
    /// The product of a tensor's extents does not fit an `i64`, so the scatter
    /// vector it implies cannot be addressed — let alone allocated.
    ///
    /// Checked because the alternative is worse than an error. The engine sizes
    /// each scatter vector by that product; unchecked, a wrapping product yields
    /// a plan that silently computes nothing in release and aborts the process
    /// in debug, and neither is an acceptable answer to give a C caller who
    /// passed extents that merely do not fit. Found by the TAPP conformance
    /// suite, which is the only caller that can supply extents this large
    /// without also allocating the memory for them.
    ExtentProductOverflow {
        /// Which operand: `"A"`, `"B"`, `"C"` or `"D"`.
        tensor: &'static str,
    },
    /// The requested element type is not supported by this entry point.
    /// Raised at the TAPP boundary for `TAPP_F16` / `TAPP_BF16`, which have no
    /// [`crate::Element`] impl here.
    UnsupportedDatatype,
    /// A data slice or pointer cannot hold every offset the plan will generate
    /// — including the null-pointer-for-a-non-empty-tensor case that gives the
    /// variant its name.
    NullPointer {
        /// Which operand: `"A"`, `"B"`, `"C"` or `"D"`.
        tensor: &'static str,
    },
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
            Error::ExtentProductOverflow { tensor } => {
                write!(f, "extents of tensor {tensor} overflow when multiplied")
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

/// `Result` with this crate's [`Error`].
pub type Result<T> = core::result::Result<T, Error>;
