//! `--partition static|dynamic:JM,JN` for the packed-driver benchmarks.
//!
//! The comparison of the default static grid with the opt-in
//! `PartitionPolicy::DynamicTiles` is run later, with the `tprims-benchmark`
//! protocol; this flag only makes the two separately selectable and their rows
//! separately identifiable. Absent, every benchmark behaves exactly as before.
use tprims_blas::GemmConfig;
use tprims_kernel::PartitionPolicy;

/// The requested policy, or `None` for the default (static) behaviour.
///
/// # Panics
/// On a malformed value: a benchmark run must not silently fall back.
pub fn from_args() -> Option<PartitionPolicy> {
    let args: Vec<String> = std::env::args().collect();
    let i = args.iter().position(|a| a == "--partition")?;
    let value = args
        .get(i + 1)
        .unwrap_or_else(|| panic!("--partition needs a value: static or dynamic:JM,JN"));
    parse(value).unwrap_or_else(|| panic!("bad --partition {value}; use static or dynamic:JM,JN"))
}

/// Parse `static` (the default, reported as no request) or `dynamic:JM,JN`.
pub fn parse(value: &str) -> Option<Option<PartitionPolicy>> {
    if value == "static" {
        return Some(None);
    }
    let (job_m, job_n) = value.strip_prefix("dynamic:")?.split_once(',')?;
    Some(Some(PartitionPolicy::DynamicTiles {
        job_m: job_m.parse().ok()?,
        job_n: job_n.parse().ok()?,
    }))
}

/// Row-label suffix identifying the policy: empty for static.
pub fn suffix(policy: Option<PartitionPolicy>) -> String {
    match policy {
        Some(PartitionPolicy::DynamicTiles { job_m, job_n }) => format!("_dyn{job_m}x{job_n}"),
        _ => String::new(),
    }
}

/// `cfg` with the policy applied.
pub fn apply(mut cfg: GemmConfig, policy: Option<PartitionPolicy>) -> GemmConfig {
    if let Some(p) = policy {
        cfg.partition = p;
    }
    cfg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_static_and_dynamic() {
        assert_eq!(parse("static"), Some(None));
        assert_eq!(
            parse("dynamic:16,32"),
            Some(Some(PartitionPolicy::DynamicTiles {
                job_m: 16,
                job_n: 32
            }))
        );
        assert_eq!(parse("dynamic:16"), None);
        assert_eq!(parse("dynamic:a,b"), None);
        assert_eq!(parse("grid"), None);
        assert_eq!(
            suffix(parse("dynamic:8,24").unwrap()),
            "_dyn8x24".to_string()
        );
    }
}
