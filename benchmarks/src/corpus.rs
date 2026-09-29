//! Benchmark corpora: shapes recorded from real workloads (spec 1e, P1) or
//! written by hand, replayed by `contract --corpus` and `blas --corpus`.
//!
//! ```json
//! { "source": { "tool": "...", "commit": "..." },
//!   "entries": [
//!     { "name": "...", "op": "dot_general", "dtype": "f64",
//!       "a": {"dims": [..], "strides": [..]}, "b": {..}, "c": {..},
//!       "lc": [..], "rc": [..], "lb": [..], "rb": [..], "conj": [false, false],
//!       "calls": 12, "time_share": 0.3 },
//!     { "name": "...", "op": "gemm_batched", "dtype": "c64",
//!       "m": 8, "n": 8, "k": 8, "batch": 1024,
//!       "a": {"dims": [m, k, batch], ..}, "b": {"dims": [k, n, batch], ..},
//!       "c": {"dims": [m, n, batch], ..} } ] }
//! ```
//!
//! Strides are in elements and may be negative; a replay stores each operand
//! so that its lowest addressed element is element 0 ([`Operand::span`]).
use serde::Deserialize;

/// Element type of an entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Dtype {
    /// `f32`.
    F32,
    /// `f64`.
    F64,
    /// `Complex<f32>`.
    C32,
    /// `Complex<f64>`.
    C64,
}

impl Dtype {
    /// Short tag used in case names.
    pub fn tag(self) -> &'static str {
        match self {
            Dtype::F32 => "f32",
            Dtype::F64 => "f64",
            Dtype::C32 => "c32",
            Dtype::C64 => "c64",
        }
    }
}

/// One operand layout.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct Operand {
    /// Extents.
    pub dims: Vec<usize>,
    /// Element strides, one per extent.
    pub strides: Vec<isize>,
}

impl Operand {
    /// `(offset, len)`: storing the operand in `len` elements with the view
    /// starting at `offset` puts its lowest addressed element at 0. Empty
    /// operands give `(0, 0)`.
    pub fn span(&self) -> (usize, usize) {
        if self.dims.contains(&0) {
            return (0, 0);
        }
        let (mut lo, mut hi) = (0isize, 0isize);
        for (&d, &s) in self.dims.iter().zip(&self.strides) {
            let reach = (d as isize - 1) * s;
            if reach < 0 {
                lo += reach;
            } else {
                hi += reach;
            }
        }
        ((-lo) as usize, (hi - lo + 1) as usize)
    }

    fn check(&self, what: &str) -> Result<(), String> {
        if self.dims.len() != self.strides.len() {
            return Err(format!(
                "{what}: {} dims but {} strides",
                self.dims.len(),
                self.strides.len()
            ));
        }
        Ok(())
    }

    fn pairs(&self) -> Vec<(usize, isize)> {
        self.dims
            .iter()
            .copied()
            .zip(self.strides.iter().copied())
            .collect()
    }
}

/// A binary contraction entry (`dot_general` semantics of tprims-contract).
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct DotGeneralEntry {
    /// Case name.
    pub name: String,
    /// Element type.
    pub dtype: Dtype,
    /// Left operand.
    pub a: Operand,
    /// Right operand.
    pub b: Operand,
    /// Output.
    pub c: Operand,
    /// Contracted axes of `a`.
    pub lc: Vec<usize>,
    /// Contracted axes of `b`, paired with `lc`.
    pub rc: Vec<usize>,
    /// Batch axes of `a`.
    pub lb: Vec<usize>,
    /// Batch axes of `b`, paired with `lb`.
    pub rb: Vec<usize>,
    /// Conjugate `a`, `b`.
    #[serde(default)]
    pub conj: [bool; 2],
    /// Calls in the recorded workload.
    #[serde(default)]
    pub calls: Option<u64>,
    /// Share of the recorded workload's time in this shape.
    #[serde(default)]
    pub time_share: Option<f64>,
}

/// A batched GEMM entry: `C_i = A_i B_i` over `[rows, cols, batch]` views.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct GemmBatchedEntry {
    /// Case name.
    pub name: String,
    /// Element type.
    pub dtype: Dtype,
    /// Rows of `C`.
    pub m: usize,
    /// Columns of `C`.
    pub n: usize,
    /// Inner extent.
    pub k: usize,
    /// Items.
    pub batch: usize,
    /// `[m, k, batch]`.
    pub a: Operand,
    /// `[k, n, batch]`.
    pub b: Operand,
    /// `[m, n, batch]`.
    pub c: Operand,
    /// Calls in the recorded workload.
    #[serde(default)]
    pub calls: Option<u64>,
    /// Share of the recorded workload's time in this shape.
    #[serde(default)]
    pub time_share: Option<f64>,
}

/// One corpus entry.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Entry {
    /// See [`DotGeneralEntry`].
    DotGeneral(DotGeneralEntry),
    /// See [`GemmBatchedEntry`].
    GemmBatched(GemmBatchedEntry),
}

impl Entry {
    /// Case name.
    pub fn name(&self) -> &str {
        match self {
            Entry::DotGeneral(d) => &d.name,
            Entry::GemmBatched(g) => &g.name,
        }
    }
}

/// A validated corpus.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Corpus {
    /// Where the shapes came from (tool, commits, workload); kept verbatim.
    #[serde(default)]
    pub source: serde_json::Value,
    /// Entries in file order.
    pub entries: Vec<Entry>,
}

impl Corpus {
    /// Parse and validate a corpus.
    ///
    /// # Errors
    ///
    /// Malformed JSON, an unknown dtype, rank/stride disagreement, a
    /// configuration tprims-contract rejects, an output whose dims differ
    /// from the configuration's, or an aliased (non-injective) output.
    pub fn parse(json: &str) -> Result<Self, String> {
        let c: Corpus = serde_json::from_str(json).map_err(|e| e.to_string())?;
        for e in &c.entries {
            validate(e).map_err(|m| format!("entry {}: {m}", e.name()))?;
        }
        Ok(c)
    }

    /// [`Corpus::parse`] of a file.
    ///
    /// # Errors
    ///
    /// As [`Corpus::parse`], or the file cannot be read.
    pub fn load(path: &str) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        Self::parse(&text).map_err(|e| format!("{path}: {e}"))
    }
}

fn validate(e: &Entry) -> Result<(), String> {
    let (a, b, c) = match e {
        Entry::DotGeneral(d) => (&d.a, &d.b, &d.c),
        Entry::GemmBatched(g) => (&g.a, &g.b, &g.c),
    };
    a.check("a")?;
    b.check("b")?;
    c.check("c")?;
    let out = match e {
        Entry::DotGeneral(d) => {
            let cfg = tprims_contract::DotGeneral::new(&d.lc, &d.rc, &d.lb, &d.rb);
            cfg.validate(&a.dims, &b.dims)
                .map_err(|e| e.to_string())?
                .out_dims
        }
        Entry::GemmBatched(g) => {
            let want = |op: &Operand, dims: [usize; 3], what: &str| {
                if op.dims != dims {
                    Err(format!("{what}: dims {:?}, expected {dims:?}", op.dims))
                } else {
                    Ok(())
                }
            };
            want(a, [g.m, g.k, g.batch], "a")?;
            want(b, [g.k, g.n, g.batch], "b")?;
            vec![g.m, g.n, g.batch]
        }
    };
    if c.dims != out {
        return Err(format!(
            "c: dims {:?}, the configuration gives {out:?}",
            c.dims
        ));
    }
    if !tprims_blas::is_injective_layout(&c.pairs()) {
        return Err("c: aliased (non-injective) output layout".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests;
