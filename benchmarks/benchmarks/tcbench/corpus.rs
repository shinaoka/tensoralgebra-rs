//! The TCCG benchmark corpus.
//!
//! The contraction set used by Springer & Bientinesi, "Design of a
//! High-Performance GEMM-like Tensor-Tensor Multiplication" (arXiv:1607.00145)
//! and by the accompanying `HPAC/tccg` repository's `benchmark/benchmark.py`.
//! They are drawn from CCSD, CCSD(T), AO-to-MO transformation, and
//! tensor-times-matrix workloads.
//!
//! # On the case count
//!
//! The brief for this project says "the TCCG set of 48 contractions". Reading
//! the upstream `benchmark.py` shows the exact structure is:
//!
//! * with `_fullbenchmark = 1`: 19 CCSD + 3 AO2MO + 8 InTensLi + 18 CCSD(T)
//!   + 1 transpose-C = **49 distinct** contractions;
//! * with `_fullbenchmark = 0`: a 25-case reduced set;
//! * `_sortedTCs` is **not** a separate group. It is those same reduced-set
//!   cases re-written in TCCG's normalised labels and ordered from
//!   bandwidth-bound to compute-bound. All 24 of its entries de-duplicate
//!   against the reduced set, which is why naively concatenating every list
//!   yields 25, not 49.
//!
//! This module therefore uses the **full 49-case set** as `corpus()` (a strict
//! superset of both published variants) and keeps `sorted_order()` as the
//! canonical presentation ordering for the 24 headline cases.
//!
//! # Extent selection
//!
//! Reproduced from `benchmark.py` so results are comparable with the
//! published numbers:
//!
//! * `avg = (tensor_size_bytes / sizeof(elem)) ** (1 / max_ndim)`
//! * an index that is stride-1 in *any* operand is rounded **up** to a
//!   multiple of 24;
//! * every other index is rounded to the **nearest** multiple of 4 (min 4);
//! * tensors are column-major, so the first label of each operand is the
//!   stride-1 one.
//!
//! One deliberate deviation: the element size used for sizing is *fixed* at
//! `f64` (8 bytes) for every data type. TCCG re-sizes per precision, which
//! would give complex runs smaller tensors than real ones and make the
//! real-vs-complex ratio — the headline metric of this project — meaningless.
//! Holding the shape fixed keeps the comparison apples-to-apples; it does mean
//! a `c64` run touches 2x the bytes of the `f64` run of the same case, which
//! is exactly the effect being measured.

use tensorcontract::Layout;

/// One contraction: `C[idx_c] = A[idx_a] * B[idx_b]`, written the way
/// `benchmark.py` writes it, `"c-a-b"`.
#[derive(Clone, Debug)]
pub struct Case {
    pub name: &'static str,
    pub group: &'static str,
    pub c: &'static str,
    pub a: &'static str,
    pub b: &'static str,
}

/// Concrete shapes for a case.
#[derive(Clone, Debug)]
pub struct Sized {
    pub case: Case,
    pub labels: Vec<char>,
    pub extents: Vec<i64>,
    pub la: Layout,
    pub lb: Layout,
    pub lc: Layout,
    pub idx_a: Vec<i64>,
    pub idx_b: Vec<i64>,
    pub idx_c: Vec<i64>,
}

impl Sized {
    #[allow(dead_code)]
    pub fn extent_of(&self, ch: char) -> i64 {
        self.extents[self.labels.iter().position(|&c| c == ch).unwrap()]
    }

    /// Multiply-accumulate count = product of all distinct index extents.
    pub fn macs(&self) -> u64 {
        self.extents.iter().map(|&e| e as u64).product()
    }

    /// Allocation size, which exceeds the logical extent product when the
    /// layout is a padded strided view.
    pub fn elems_a(&self) -> usize {
        self.la.storage_len() as usize
    }
    pub fn elems_b(&self) -> usize {
        self.lb.storage_len() as usize
    }
    pub fn elems_c(&self) -> usize {
        self.lc.storage_len() as usize
    }

    /// `m`, `n`, `k` of the equivalent matrix multiplication.
    pub fn mnk(&self) -> (u64, u64, u64) {
        let inm = |ch: char| self.case.c.contains(ch) && self.case.a.contains(ch);
        let inn = |ch: char| self.case.c.contains(ch) && self.case.b.contains(ch);
        let mut m = 1u64;
        let mut n = 1u64;
        let mut k = 1u64;
        for (i, &ch) in self.labels.iter().enumerate() {
            let e = self.extents[i] as u64;
            if inm(ch) {
                m *= e;
            } else if inn(ch) {
                n *= e;
            } else {
                k *= e;
            }
        }
        (m, n, k)
    }
}

const CCSD: &[&str] = &[
    "ij-ik-kj",
    "ij-ikl-ljk",
    "ij-kil-lkj",
    "ijk-ikl-lj",
    "ijk-il-jlk",
    "ijk-ilk-jl",
    "ijk-ilk-lj",
    "ijk-ilmk-mjl",
    "ijkl-imjn-lnkm",
    "ijkl-imjn-nlmk",
    "ijkl-imkn-jnlm",
    "ijkl-imkn-njml",
    "ijkl-imln-jnkm",
    "ijkl-imln-njmk",
    "ijkl-imnj-nlkm",
    "ijkl-imnk-njml",
    "ijkl-minj-nlmk",
    "ijkl-mink-jnlm",
    "ijkl-minl-njmk",
];

const AO2MO: &[&str] = &["aqrs-pa-pqrs", "abrs-qb-aqrs", "abcs-rc-abrs"];

const INTENSLI: &[&str] = &[
    "abj-bka-kj",
    "ajb-kba-jk",
    "abjc-cbka-kj",
    "ajbc-ckba-jk",
    "abjc-kbac-jk",
    "abjcd-dkbac-jk",
    "adbjc-cbdka-kj",
    "ajbdc-ckbad-jk",
];

const CCSD_T: &[&str] = &[
    "abcijk-ijma-mkbc",
    "abcijk-ijmb-mkac",
    "abcijk-ijmc-mkab",
    "abcijk-ikma-mjbc",
    "abcijk-ikmb-mjac",
    "abcijk-ikmc-mjab",
    "abcijk-jkma-mibc",
    "abcijk-jkmb-miac",
    "abcijk-jkmc-miab",
    "abcijk-eiab-jkec",
    "abcijk-eiac-jkeb",
    "abcijk-eibc-jkea",
    "abcijk-ejab-ikec",
    "abcijk-ejac-ikeb",
    "abcijk-ejbc-ikea",
    "abcijk-ekab-ijec",
    "abcijk-ekac-ijeb",
    "abcijk-ekbc-ijea",
];

const TRANS_C: &[&str] = &["abc-bk-akc"];

/// TCCG's `_sortedTCs`: the 24 headline cases in normalised labels, ordered
/// from bandwidth-bound to compute-bound as measured on a Haswell core. Used
/// for presentation order and for picking representative subsets; every entry
/// is the normalised form of a case already in [`corpus()`].
pub const SORTED_ORDER: &[&str] = &[
    "abcde-efbad-cf",
    "abcde-efcad-bf",
    "abcd-dbea-ec",
    "abcde-ecbfa-fd",
    "abcd-deca-be",
    "abc-bda-dc",
    "abcd-ebad-ce",
    "abcdef-dega-gfbc",
    "abcdef-dfgb-geac",
    "abcdef-degb-gfac",
    "abcdef-degc-gfab",
    "abc-dca-bd",
    "abcd-ea-ebcd",
    "abcd-eb-aecd",
    "abcd-ec-abed",
    "abc-adec-ebd",
    "ab-cad-dcb",
    "ab-acd-dbc",
    "abc-acd-db",
    "abc-adc-bd",
    "ab-ac-cb",
    "abcd-aebf-fdec",
    "abcd-eafd-fbec",
    "abcd-aebf-dfce",
];

/// The full 49-case corpus, de-duplicated after label normalisation.
pub fn corpus() -> Vec<Case> {
    let mut out = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    let groups: [(&'static str, &[&'static str]); 5] = [
        ("ccsd", CCSD),
        ("ao2mo", AO2MO),
        ("intensli", INTENSLI),
        ("ccsd_t", CCSD_T),
        ("transC", TRANS_C),
    ];
    for (group, specs) in groups {
        for spec in specs {
            let norm = normalize(spec);
            if seen.contains(&norm) {
                continue;
            }
            seen.push(norm);
            let mut it = spec.split('-');
            let c = it.next().unwrap();
            let a = it.next().unwrap();
            let b = it.next().unwrap();
            out.push(Case {
                name: spec,
                group,
                c,
                a,
                b,
            });
        }
    }
    out
}

/// The corpus ordered as TCCG presents it: bandwidth-bound first,
/// compute-bound last, per `_sortedTCs`. Cases with no `_sortedTCs` entry
/// (they only exist in the full set) are appended, keeping group order.
pub fn sorted_cases() -> Vec<Case> {
    let all = corpus();
    let rank = |c: &Case| -> usize {
        let n = normalize(c.name);
        SORTED_ORDER
            .iter()
            .position(|s| normalize(s) == n)
            .unwrap_or(usize::MAX)
    };
    let mut out = all;
    out.sort_by_key(rank);
    out
}

/// TCCG's label normalisation, used only for de-duplication.
fn normalize(spec: &str) -> String {
    let mut it = spec.split('-');
    let mut c: Vec<char> = it.next().unwrap().chars().collect();
    let mut a: Vec<char> = it.next().unwrap().chars().collect();
    let mut b: Vec<char> = it.next().unwrap().chars().collect();
    let mut next = 'A';
    let rename = |from: char, to: char, a: &mut Vec<char>, b: &mut Vec<char>, c: &mut Vec<char>| {
        for v in [a, b, c] {
            for x in v.iter_mut() {
                if *x == from {
                    *x = to;
                }
            }
        }
    };
    let cc = c.clone();
    for ch in cc {
        rename(ch, next, &mut a, &mut b, &mut c);
        next = (next as u8 + 1) as char;
    }
    let ac = a.clone();
    for ch in ac {
        if !c.contains(&ch) && ch.is_lowercase() {
            rename(ch, next, &mut a, &mut b, &mut c);
            next = (next as u8 + 1) as char;
        }
    }
    format!(
        "{}-{}-{}",
        c.iter().collect::<String>(),
        a.iter().collect::<String>(),
        b.iter().collect::<String>()
    )
    .to_lowercase()
}

/// How to perturb the TCCG extents.
///
/// TCCG deliberately rounds every stride-1 extent up to a multiple of 24, which
/// makes the extents regular **at a register block that divides 24** — and only
/// there. That is a narrower guarantee than it looks, and reading it as "the
/// corpus is fully regular" steered this project's conclusions for three phases.
/// The shipped `f32`/`c32` blocks are `MR` 16, 32 and 48, none of which divides
/// 24, so on the arm the orientation rule actually picks, `reg_a < 1.0` on
/// **42.9%** of the 392 case-dtype-methods at `--size 64` — 45 of them at
/// `reg_a = 0.0`, i.e. *entirely* on the gather path. Both qualifiers belong to
/// the number: it is 11.7% on an AVX2 node, where `MR = 8` for `f64` does divide
/// 24, and 40.6% at `tcbench orient`'s default size, because the extents scale
/// with it.
///
/// So the unperturbed corpus does exercise the gather path, and any claim about
/// awkward strides must name its [`Stress`] mode and quote the observed `reg_a`.
/// What TCCG cannot produce is *aperiodic* irregularity — the extents are round,
/// so a block either straddles a boundary on a fixed period or never does. That
/// is what the perturbations below are for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Stress {
    /// TCCG's own sizing, unmodified.
    #[default]
    None,
    /// Subtract one from every extent, so nothing divides the register block
    /// and every index-boundary crossing produces an irregular block.
    Ragged,
    /// TCCG sizing, but each tensor is a strided view into a larger buffer
    /// with its leading dimension padded by one element.
    Padded,
}

impl Stress {
    pub fn parse(s: &str) -> Option<Stress> {
        match s {
            "none" => Some(Stress::None),
            "ragged" => Some(Stress::Ragged),
            "padded" => Some(Stress::Padded),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Stress::None => "none",
            Stress::Ragged => "ragged",
            Stress::Padded => "padded",
        }
    }
}

/// Apply the TCCG sizing rule, unperturbed.
#[allow(dead_code)]
pub fn size_case(case: &Case, tensor_bytes: f64) -> Sized {
    size_case_stressed(case, tensor_bytes, Stress::None)
}

/// Apply the TCCG sizing rule, then the requested [`Stress`] perturbation.
pub fn size_case_stressed(case: &Case, tensor_bytes: f64, stress: Stress) -> Sized {
    let max_ndim = case.a.len().max(case.b.len()).max(case.c.len());
    // Sizing always uses 8 bytes/element so shapes are precision-independent.
    let avg = (tensor_bytes / 8.0).powf(1.0 / max_ndim as f64);

    let mut labels: Vec<char> = Vec::new();
    for s in [case.c, case.a, case.b] {
        for ch in s.chars() {
            if !labels.contains(&ch) {
                labels.push(ch);
            }
        }
    }
    let stride1: Vec<char> = [case.c, case.a, case.b]
        .iter()
        .filter_map(|s| s.chars().next())
        .collect();

    let extents: Vec<i64> = labels
        .iter()
        .map(|&ch| {
            if stride1.contains(&ch) {
                (((avg + 23.0) / 24.0) as i64) * 24
            } else {
                let up = (((avg + 3.0) / 4.0) as i64) * 4;
                let down = (((avg / 4.0) as i64) * 4).max(4);
                if (avg - up as f64).abs() < (avg - down as f64).abs() {
                    up
                } else {
                    down
                }
            }
        })
        .collect();

    let extents: Vec<i64> = match stress {
        Stress::Ragged => extents.iter().map(|&e| (e - 1).max(1)).collect(),
        _ => extents,
    };

    let ext_of = |ch: char| extents[labels.iter().position(|&c| c == ch).unwrap()];
    let shape = |s: &str| -> Vec<i64> { s.chars().map(ext_of).collect() };
    let idx = |s: &str| -> Vec<i64> { s.chars().map(|c| c as i64).collect() };
    let lay = |s: &str| -> Layout {
        let sh = shape(s);
        match stress {
            // Pad the leading dimension so the tensor is a non-contiguous
            // strided view, as a slice of a bigger array would be.
            Stress::Padded if !sh.is_empty() => {
                let mut strides = Vec::with_capacity(sh.len());
                let mut acc = 1i64;
                for (d, &e) in sh.iter().enumerate() {
                    strides.push(acc);
                    acc *= if d == 0 { e + 1 } else { e };
                }
                Layout::new(sh, strides).expect("one stride pushed per extent")
            }
            _ => Layout::col_major(&sh),
        }
    };

    Sized {
        labels: labels.clone(),
        extents: extents.clone(),
        la: lay(case.a),
        lb: lay(case.b),
        lc: lay(case.c),
        idx_a: idx(case.a),
        idx_b: idx(case.b),
        idx_c: idx(case.c),
        case: case.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corpus_is_the_full_49_case_set() {
        // 19 + 3 + 8 + 18 + 1, none colliding under TCCG normalisation.
        assert_eq!(corpus().len(), 49, "TCCG corpus size");
    }

    #[test]
    fn sorted_order_entries_are_all_in_the_corpus() {
        // `_sortedTCs` is a re-labelling of existing cases, not new ones.
        let norms: Vec<String> = corpus().iter().map(|c| normalize(c.name)).collect();
        for s in SORTED_ORDER {
            assert!(
                norms.contains(&normalize(s)),
                "{s} from _sortedTCs is not a corpus case"
            );
        }
    }

    #[test]
    fn sizing_matches_tccg_rule() {
        // "ij-ik-kj" is rank 2 everywhere, so avg = sqrt(200Mi/8) = 5120.
        // `i` is stride-1 in C and A, `k` is stride-1 in B: both round up to a
        // multiple of 24 -> 5136. `j` is never stride-1, so it rounds to the
        // nearest multiple of 4 -> 5120.
        let c = corpus().into_iter().find(|c| c.name == "ij-ik-kj").unwrap();
        let s = size_case(&c, 200.0 * 1024.0 * 1024.0);
        assert_eq!(s.extent_of('i'), 5136);
        assert_eq!(s.extent_of('k'), 5136);
        assert_eq!(s.extent_of('j'), 5120);
    }

    #[test]
    fn mnk_is_consistent_with_macs() {
        for c in corpus() {
            let s = size_case(&c, 8.0 * 1024.0 * 1024.0);
            let (m, n, k) = s.mnk();
            assert_eq!(m * n * k, s.macs(), "case {}", c.name);
        }
    }
}
