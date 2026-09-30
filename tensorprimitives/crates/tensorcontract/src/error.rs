//! Error type.
//!
//! Almost everything here describes a contraction that was rejected while being
//! *planned*. Typed kernel resolution may also reject execution before any
//! computation begins. The variants fall into six groups:
//!
//! * **the description is internally inconsistent** — [`Error::RankMismatch`],
//!   [`Error::LabelCountMismatch`], [`Error::ExtentMismatch`],
//!   [`Error::NegativeExtent`], [`Error::OutputLabelMismatch`];
//! * **the contraction is well formed but out of scope** —
//!   [`Error::BroadcastIndexUnsupported`] (a label only in the output) and
//!   [`Error::UnsupportedDatatype`];
//! * **the shape cannot be addressed at all** —
//!   [`Error::ExtentProductOverflow`];
//! * **the call disagrees with the plan** — [`Error::NullPointer`], when the
//!   data cannot hold every offset the plan generates, and
//!   [`Error::ElementOpMismatch`], when a view's conjugation differs from the
//!   one recorded in the plan. These are raised by [`crate::Plan::run`]
//!   because neither slice lengths nor the views exist until then;
//! * **kernel selection/configuration is invalid** — [`Error::KernelSelection`],
//!   including dtype, CPU, registration and blocking failures. Safe execution
//!   propagates typed resolution errors before shortcuts or numerical calls;
//! * **there was never a contraction to reject** — [`Error::EinsumSyntax`], the
//!   one variant raised before planning starts, by [`crate::parse_einsum`]
//!   failing on the *notation* rather than on anything the labels mean.
//!
//! So a [`crate::Plan`] that exists describes a contraction that is expressible;
//! execution still checks its storage dtype's kernel resolution and whether
//! the supplied arguments are the ones it was built for.

use core::fmt;

/// Everything that can go wrong when building or executing a contraction plan.
///
/// Most variants are a rejected *description* of a contraction — a shape, label
/// or stride inconsistency — caught while planning. [`crate::Plan::run`] also
/// checks typed kernel resolution ([`Error::KernelSelection`]), slice bounds
/// ([`Error::NullPointer`]) and view operations ([`Error::ElementOpMismatch`]).
/// Nothing is reported once the engine starts
/// computing. [`Error::EinsumSyntax`] is the outlier and comes from the string
/// convenience helper, before any of that.
///
/// `#[non_exhaustive]` because the mapping onto TAPP's error codes is the
/// stable contract, not this enumeration; match with a `_` arm.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// Invalid registered-kernel selection/configuration, before computation.
    KernelSelection(tprims_gemm_kernel::SelectError),
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
    /// An operand handed to [`crate::Plan::run`] carries a different
    /// [`crate::plan::ElementOp`] than the one the plan was built with.
    ///
    /// Conjugation is folded into the packing and write-back traversals, so it
    /// is fixed when [`crate::Plan::new`] analyses the contraction — a
    /// conjugated plan and an unconjugated one are different plans. The
    /// alternative to this error was to ignore the argument silently, which
    /// returned the unconjugated result to a caller who had asked for the
    /// conjugated one. Rebuild the plan with [`crate::plan::Operand::conj`], or
    /// use [`crate::contract`], which derives the plan from the same views it
    /// executes and so cannot disagree with itself.
    ElementOpMismatch {
        /// Which operand: `"A"`, `"B"`, `"C"` or `"D"`.
        tensor: &'static str,
    },
    /// A spec handed to [`crate::parse_einsum`] is not in the notation it
    /// accepts: `"ab,bc->ac"`, exactly one `->` and one `,`.
    ///
    /// The offending spec is deliberately not carried: the caller still owns
    /// it, so a static `reason` suffices. Description errors retain their small
    /// payloads; `KernelSelection` separately carries owned kernel diagnostics.
    EinsumSyntax {
        /// Which piece of the notation was missing. A short static phrase — the
        /// set is closed and matching on it is not supported.
        reason: &'static str,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::KernelSelection(source) => fmt::Display::fmt(source, f),
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
            Error::ElementOpMismatch { tensor } => write!(
                f,
                "tensor {tensor} was given a different element-wise op than the \
                 plan was built with; conjugation is fixed at Plan::new"
            ),
            Error::EinsumSyntax { reason } => {
                write!(f, "cannot parse einsum spec: {reason}")
            }
        }
    }
}

/// `core::error::Error`, not `std::error::Error`, and unconditionally.
///
/// The two are the same trait — `std` has re-exported `core`'s since Rust 1.81,
/// well under this crate's MSRV — so naming the `core` one costs nothing and
/// removes a `#[cfg(feature = "std")]` that only ever subtracted.
///
/// What the gate did was leave a `--no-default-features` build with an [`Error`]
/// that could not be `?`'d into a `Box<dyn Error>`, handed to `anyhow`, or used
/// as a `#[source]` — the impl is what makes an error type interoperable, and
/// without it this one was merely printable. What the gate bought in exchange was
/// **nothing**: this crate is not `no_std` (see the `std` feature's own note in
/// `Cargo.toml` — no `#![no_std]`, and `Vec` in the scatter vectors), so the
/// trait was always reachable and only the impl was withheld. The cost fell on
/// exactly the callers the feature is *for*, the ones turning it off to be rid of
/// the `TENSORCONTRACT_*` reads.
///
/// It survived because CI builds `--no-default-features` and building is not
/// using: a missing impl is not a compile error until something needs it, and
/// nothing did until `tests/traits.rs`.
impl core::error::Error for Error {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::KernelSelection(source) => Some(source),
            _ => None,
        }
    }
}

/// `Result` with this crate's [`Error`].
pub type Result<T> = core::result::Result<T, Error>;
