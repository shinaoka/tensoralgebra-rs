use super::*;

const OK: &str = r#"{
  "source": {"tool": "test"},
  "entries": [
    {"name": "mm", "op": "dot_general", "dtype": "f64",
     "a": {"dims": [4, 3], "strides": [1, 4]},
     "b": {"dims": [3, 5], "strides": [5, 1]},
     "c": {"dims": [4, 5], "strides": [1, 4]},
     "lc": [1], "rc": [0], "lb": [], "rb": [], "conj": [false, true],
     "calls": 7, "time_share": 0.25},
    {"name": "bmm", "op": "gemm_batched", "dtype": "c32",
     "m": 2, "n": 3, "k": 4, "batch": 5,
     "a": {"dims": [2, 4, 5], "strides": [1, 2, 8]},
     "b": {"dims": [4, 3, 5], "strides": [1, 4, 12]},
     "c": {"dims": [2, 3, 5], "strides": [1, 2, 6]}}
  ]
}"#;

#[test]
fn parses_both_entry_kinds() {
    let c = Corpus::parse(OK).unwrap();
    assert_eq!(c.entries.len(), 2);
    match &c.entries[0] {
        Entry::DotGeneral(d) => {
            assert_eq!(d.name, "mm");
            assert_eq!(d.dtype, Dtype::F64);
            assert_eq!(d.conj, [false, true]);
            assert_eq!(d.calls, Some(7));
        }
        e => panic!("{e:?}"),
    }
    match &c.entries[1] {
        Entry::GemmBatched(g) => {
            assert_eq!((g.m, g.n, g.k, g.batch), (2, 3, 4, 5));
            assert_eq!(g.dtype, Dtype::C32);
        }
        e => panic!("{e:?}"),
    }
}

fn with(entry: &str) -> String {
    format!(r#"{{"source": {{}}, "entries": [{entry}]}}"#)
}

#[test]
fn rejects_malformed_entries() {
    let base = r#"{"name": "x", "op": "dot_general", "dtype": "f64",
      "a": {"dims": [4, 3], "strides": [1, 4]}, "b": {"dims": [3, 5], "strides": [5, 1]},
      "c": {"dims": [4, 5], "strides": [1, 4]}, "lc": [1], "rc": [0], "lb": [], "rb": []}"#;
    assert!(Corpus::parse(&with(base)).is_ok());
    for (bad, why) in [
        (
            base.replace(r#""dtype": "f64""#, r#""dtype": "i32""#),
            "dtype",
        ),
        (
            base.replace(r#""strides": [1, 4]}, "b""#, r#""strides": [1]}, "b""#),
            "rank",
        ),
        (base.replace(r#""rc": [0]"#, r#""rc": [1]"#), "extent"),
        (
            base.replace(
                r#""c": {"dims": [4, 5], "strides": [1, 4]}"#,
                r#""c": {"dims": [4, 5], "strides": [1, 1]}"#,
            ),
            "aliased",
        ),
        (
            base.replace(r#""c": {"dims": [4, 5]"#, r#""c": {"dims": [4, 6]"#),
            "output dims",
        ),
    ] {
        assert!(Corpus::parse(&with(&bad)).is_err(), "{why} accepted");
    }
}

#[test]
fn span_offset_puts_the_lowest_element_at_zero() {
    let op = Operand {
        dims: vec![3, 2],
        strides: vec![-1, 3],
    };
    assert_eq!(op.span(), (2, 6));
    let empty = Operand {
        dims: vec![0, 2],
        strides: vec![1, 1],
    };
    assert_eq!(empty.span(), (0, 0));
}
