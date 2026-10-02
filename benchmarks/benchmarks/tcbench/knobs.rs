//! The measurement knobs that used to be read from the environment inside the
//! library, parsed once here into explicit plan configuration (tprims addition).
//!
//! `TENSORCONTRACT_{KERNEL,COMPLEX,MC,KC,NC,MC_PCT,NC_PCT,KC_COUPLE,BLOCKMODEL,
//! L3_DOMAINS,WRITEBACK,ORIENT,ROWBLOCK,PARTITION}` keep their spellings so the
//! sweep scripts still work. An unset variable is the library baseline; an
//! unparseable value is an error rather than a silent default. The variables that
//! no longer exist (`TENSORCONTRACT_THREADS`, `TENSORCONTRACT_POOL`,
//! `TPRIMS_GEMM_KERNEL`, `TPRIMS_GEMM_ENGINE`) are rejected by name: threads
//! come from a `tprims_exec::Exec` only.

use std::sync::OnceLock;

use tensorcontract::kernel::{ComplexMethod, Tuning};
use tensorcontract::{Orient, PartitionMode, Plan, RowBlock};
use tprims_kernel::{blocking::BlockModel, KernelForce};

/// Variables that were removed together with the library's environment reads.
const REMOVED: [(&str, &str); 4] = [
    (
        "TENSORCONTRACT_THREADS",
        "threads come only from a tprims_exec::Exec",
    ),
    (
        "TENSORCONTRACT_POOL",
        "the parked-thread pool was removed with the Spmd seam",
    ),
    (
        "TPRIMS_GEMM_KERNEL",
        "select a family with Plan::with_kernel or a benchmark flag",
    ),
    (
        "TPRIMS_GEMM_ENGINE",
        "select the engine with a benchmark flag",
    ),
];

/// The parsed knobs.
#[derive(Clone, Debug, Default)]
pub struct Knobs {
    tuning: Tuning,
    method: Option<ComplexMethod>,
    orient: Orient,
    row_block: RowBlock,
    partition: PartitionMode,
    l3_domains: Option<usize>,
}

fn parse_with<T>(
    get: &impl Fn(&str) -> Option<String>,
    name: &str,
    parse: impl Fn(&str) -> Option<T>,
) -> Result<Option<T>, String> {
    match get(name) {
        None => Ok(None),
        Some(v) => parse(&v)
            .map(Some)
            .ok_or_else(|| format!("{name}={v}: unrecognised value")),
    }
}

fn parse_usize(v: &str) -> Option<usize> {
    v.trim().parse().ok().filter(|&n| n > 0)
}

impl Knobs {
    /// Parse from a variable lookup (the process environment in `main`).
    pub fn parse(get: impl Fn(&str) -> Option<String>) -> Result<Knobs, String> {
        for (name, why) in REMOVED {
            if get(name).is_some() {
                return Err(format!("{name} was removed in tprims-rs#37: {why}"));
            }
        }
        let mut k = Knobs::default();
        if let Some(f) = parse_with(&get, "TENSORCONTRACT_KERNEL", KernelForce::parse)? {
            k.tuning.kernel_force = f;
        }
        k.method = parse_with(&get, "TENSORCONTRACT_COMPLEX", ComplexMethod::parse)?;
        if let Some(m) = parse_with(&get, "TENSORCONTRACT_BLOCKMODEL", BlockModel::parse)? {
            k.tuning.block_model = m;
        }
        let b = &mut k.tuning.blocking;
        b.mc = parse_with(&get, "TENSORCONTRACT_MC", parse_usize)?;
        b.kc = parse_with(&get, "TENSORCONTRACT_KC", parse_usize)?;
        b.nc = parse_with(&get, "TENSORCONTRACT_NC", parse_usize)?;
        b.mc_pct = parse_with(&get, "TENSORCONTRACT_MC_PCT", parse_usize)?;
        b.nc_pct = parse_with(&get, "TENSORCONTRACT_NC_PCT", parse_usize)?;
        k.tuning.kc_couple = parse_with(&get, "TENSORCONTRACT_KC_COUPLE", parse_usize)?;
        k.tuning.writeback_gather =
            parse_with(&get, "TENSORCONTRACT_WRITEBACK", |v: &str| {
                match v.trim().to_ascii_lowercase().as_str() {
                    "gather" | "scatter" => Some(true),
                    "fast" | "default" => Some(false),
                    _ => None,
                }
            })?
            .unwrap_or(false);
        k.orient = parse_with(&get, "TENSORCONTRACT_ORIENT", Orient::parse)?.unwrap_or_default();
        k.row_block =
            parse_with(&get, "TENSORCONTRACT_ROWBLOCK", RowBlock::parse)?.unwrap_or_default();
        k.partition =
            parse_with(&get, "TENSORCONTRACT_PARTITION", PartitionMode::parse)?.unwrap_or_default();
        k.l3_domains = parse_with(&get, "TENSORCONTRACT_L3_DOMAINS", parse_usize)?;
        Ok(k)
    }

    /// The kernel-layer tuning inputs.
    pub fn tuning(&self) -> &Tuning {
        &self.tuning
    }

    /// Apply every knob to a plan.
    pub fn apply(&self, plan: Plan) -> Plan {
        let mut plan = plan
            .with_tuning(self.tuning)
            .with_orientation(self.orient)
            .with_row_block(self.row_block)
            .with_partition_mode(self.partition);
        if let Some(m) = self.method {
            plan = plan.with_complex_method(m);
        }
        if let Some(n) = self.l3_domains {
            plan = plan.with_l3_domains(n);
        }
        plan
    }

    /// The forced L3 domain count, if any.
    pub fn l3_domains(&self) -> Option<usize> {
        self.l3_domains
    }
}

static KNOBS: OnceLock<Knobs> = OnceLock::new();

/// Read the process environment once, at startup. An error is printed by the
/// caller; later [`get`] calls then see the baseline.
pub fn init_from_env() -> Result<(), String> {
    let k = Knobs::parse(|n| std::env::var(n).ok())?;
    let _ = KNOBS.set(k);
    Ok(())
}

/// The parsed knobs (the baseline before [`init_from_env`]).
pub fn get() -> &'static Knobs {
    KNOBS.get_or_init(Knobs::default)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |n| {
            pairs
                .iter()
                .find(|(k, _)| *k == n)
                .map(|(_, v)| v.to_string())
        }
    }

    #[test]
    fn unset_is_the_baseline_and_knobs_parse() {
        let base = Knobs::parse(env(&[])).unwrap();
        assert_eq!(base.tuning, Tuning::default());
        assert_eq!(base.partition, PartitionMode::Domain);
        let k = Knobs::parse(env(&[
            ("TENSORCONTRACT_KERNEL", "avx2"),
            ("TENSORCONTRACT_PARTITION", "4x2"),
            ("TENSORCONTRACT_MC", "64"),
            ("TENSORCONTRACT_WRITEBACK", "gather"),
            ("TENSORCONTRACT_L3_DOMAINS", "3"),
        ]))
        .unwrap();
        assert_eq!(k.tuning.kernel_force, KernelForce::Avx2);
        assert_eq!(k.partition, PartitionMode::Pin(4, 2));
        assert_eq!(k.tuning.blocking.mc, Some(64));
        assert!(k.tuning.writeback_gather);
        assert_eq!(k.l3_domains(), Some(3));
    }

    #[test]
    fn removed_and_malformed_knobs_are_rejected() {
        for name in [
            "TENSORCONTRACT_THREADS",
            "TENSORCONTRACT_POOL",
            "TPRIMS_GEMM_KERNEL",
            "TPRIMS_GEMM_ENGINE",
        ] {
            let e = Knobs::parse(env(&[(name, "1")])).unwrap_err();
            assert!(e.contains(name) && e.contains("removed"), "{e}");
        }
        assert!(Knobs::parse(env(&[("TENSORCONTRACT_KERNEL", "avx3")])).is_err());
        assert!(Knobs::parse(env(&[("TENSORCONTRACT_MC", "0")])).is_err());
    }
}
