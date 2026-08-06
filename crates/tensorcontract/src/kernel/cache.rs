//! Hardware cache descriptors, probed at run time, and the analytical
//! blocking model they drive.
//!
//! # Why this exists
//!
//! [`Blocking::derive`](super::Blocking::derive) hardcodes three numbers —
//! `kc = 384` for 4-byte reals and `256` otherwise, a 512 KiB budget for the
//! packed `A` block and a 3 MiB budget for the packed `B` panel. They are half
//! the L2 and a slice of the L3 of *one* machine (`ccqlin038`, Cascade Lake,
//! 1 MiB L2 per core, 25 MiB L3 per socket) and they are the only
//! machine-specific quantities in the engine's blocking. Everything else is
//! derived. This module replaces them with a model whose inputs are queried
//! from the hardware, so that a decent choice transfers to a machine nobody
//! measured on.
//!
//! It is **off by default**: `TENSORCONTRACT_BLOCKMODEL=model` opts in, and
//! `legacy` (the default) keeps the historical constants. See
//! [`block_model`] for why.
//!
//! # The model
//!
//! Low, Igual, Smith & Quintana-Ortí, *"Analytical Modeling Is Enough for
//! High-Performance BLIS"*, ACM TOMS 43(2):12, 2016 (DOI 10.1145/2925987; the
//! extended version is FLAME Working Note #74, and the equation numbers below
//! are the same in both). Its thesis is exactly what is wanted here: the cache
//! blocking of a BLIS-shaped GEMM can be *computed* from the cache geometry
//! rather than searched for empirically.
//!
//! The paper's architecture model gives each cache level `Li` a line size
//! `C_Li`, an associativity `W_Li`, a size `S_Li` and a set count `N_Li` with
//! `S_Li = N_Li * C_Li * W_Li`; `S_DATA` is the size of one scalar. What the
//! paper derives, and what is implemented in [`analytical`]:
//!
//! * `kc` — from the L1. A micro-panel `A_r` (`mr x kc`) occupies `C_Ar` lines
//!   in every L1 set, a micro-panel `B_r` (`kc x nr`) occupies `C_Br`, and one
//!   line per set is reserved for the unpacked micro-tile `C_r`, so
//!   `C_Ar + C_Br <= W_L1 - 1` (eq. 5). `C_Br = ceil(nr/mr * C_Ar)`, so
//!   `C_Ar <= floor((W_L1 - 1) / (1 + nr/mr))`, taken as large as possible, and
//!   `kc = C_Ar * N_L1 * C_L1 / (mr * S_DATA)` (eq. 4). Sizing `A_r` to a whole
//!   number of ways is what makes the *next* micro-panel of `A_c` land on the
//!   same sets as the one it replaces, which is how `B_r` survives in L1. For a
//!   2-way L1 that construction leaves no room for `B_r` at all and the paper
//!   falls back to `kc = N_L1 * C_L1 / (2 * mr * S_DATA)` (eq. 6).
//! * `mc` — from the L2, by the same argument one level out: `A_c` (`mc x kc`)
//!   is the resident block, micro-panels of `B_c` stream past it, one way is
//!   reserved for `C_c`.
//! * `nc` — from the L3, one level out again: `B_c` (`kc x nc`) is resident and
//!   `A_c` streams past it.
//!
//! **What is the paper's and what is inferred.** The `mr`/`nr` equations (1–3)
//! and the `kc` equations (4–6) are quoted above and implemented as written;
//! the register blocks themselves are *not* taken from the model here, because
//! this project measures them (see `kernel::x86`). For `mc` and `nc` the paper
//! says only that they "can be derived in a similar manner" (§4.3.1) and never
//! writes the inequalities down. The `mc` rule implemented here — reserve
//! `ceil(nr*kc*S_DATA / (N_L2*C_L2))` ways for the streaming `B_c` micro-panel
//! and one for `C_c`, give the rest to `A_c` — is therefore a reconstruction,
//! but not a guess: it reproduces the paper's own Table III `mc` exactly for
//! three of its four architectures (Intel SandyBridge 96, AMD Kaveri 1792, TI
//! C6678 128) with the paper's `kc`, `mr` and `nr` as inputs. It does *not*
//! reproduce the Intel Dunnington row (it predicts 1280 against the paper's
//! 384), and the Dunnington `kc` does not follow eq. 4 either (2 ways of `A_r`
//! where the formula asks for 3), so that row appears to carry an unstated
//! extra constraint. `nc` is a further inference by the same symmetry, and the
//! paper explicitly declines to validate `nc` because three of its four
//! machines have no L3.
//!
//! # Two things the paper does not cover
//!
//! * **Reals per element.** This engine's packed panels carry a
//!   method-dependent number of reals per *logical* element — 2 for planar, 4
//!   in `A` for 1m, 3 for 3m (see [`super::PackFormat`]) — so every footprint
//!   below is in reals, not elements. That is what makes 1m automatically get a
//!   smaller `mc` from the same L2, and getting it wrong would rig the
//!   project's three-way method comparison.
//! * **Threading.** The driver threads over `M` with a *shared* packed `B`
//!   panel, so `nc`'s L3 budget is a per-socket resource that is **not** divided
//!   among threads — but the per-thread packed `A` blocks all live in the same
//!   L3, so they are charged against it once per thread. See [`analytical`].
//!
//! # What the model still cannot see
//!
//! `mc` is bounded from *both* sides (assumption A13 in `docs/notebook/`): from
//! below by packed-`A` residency in L2, which is what this model computes, and
//! from above by the strip of `D` that one `jr` pass revisits, which it does
//! not model at all. A pending measurement (Phase 4 report part 7) is designed
//! to separate the two, so no attempt is made to guess the upper bound here.

use super::Blocking;

// ---------------------------------------------------------------------------
// Descriptors
// ---------------------------------------------------------------------------

/// One level of data cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheLevel {
    /// 1, 2 or 3.
    pub level: u8,
    /// Total size in bytes.
    pub size: usize,
    /// Line size in bytes (`C_Li` in the paper).
    pub line: usize,
    /// Associativity (`W_Li`).
    pub ways: usize,
    /// Number of sets (`N_Li`).
    pub sets: usize,
    /// How many *logical* CPUs share this cache. Load-bearing: it is what
    /// decides whether a thread count divides a budget or not.
    pub shared_by: usize,
}

impl CacheLevel {
    /// Bytes in one way, i.e. `N_Li * C_Li`. Every footprint in the model is
    /// expressed in these units, because the model reasons in whole ways.
    pub const fn bytes_per_way(&self) -> usize {
        self.sets * self.line
    }
}

/// Where the descriptors came from. Reported by `tcbench info` so a number can
/// be traced to a probe rather than to a guess.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheSource {
    /// Linux `/sys/devices/system/cpu/cpu0/cache`. The only source that
    /// reports cache *sharing* directly, and it needs no `unsafe`.
    Sysfs,
    /// x86 `CPUID` leaf 4, or `0x8000001D` on AMD.
    Cpuid,
    /// The conservative built-in fallback.
    Builtin,
}

impl CacheSource {
    /// Short name for reports: `"sysfs"`, `"cpuid"` or `"builtin"`.
    pub fn name(self) -> &'static str {
        match self {
            CacheSource::Sysfs => "sysfs",
            CacheSource::Cpuid => "cpuid",
            CacheSource::Builtin => "builtin",
        }
    }
}

/// The data cache hierarchy of one core.
///
/// L1d is mandatory — without it there is no model — so the probe falls back to
/// [`BUILTIN`] rather than reporting nothing. L2 and L3 are optional because
/// plenty of targets lack one or both, and the model degrades level by level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheHierarchy {
    /// The L1 data cache, which fixes `kc`.
    pub l1d: CacheLevel,
    /// The L2, which fixes `mc`. `None` leaves `mc` at the model's fallback.
    pub l2: Option<CacheLevel>,
    /// The L3, which fixes `nc`. `None` on most non-server parts.
    pub l3: Option<CacheLevel>,
    /// Which probe produced these numbers, so a blocking parameter can be
    /// traced back to a measurement of the machine rather than to [`BUILTIN`].
    pub source: CacheSource,
}

/// Conservative defaults: a small 32 KiB L1d, a 256 KiB L2 and an 8 MiB L3,
/// all with plausible geometry. Deliberately smaller than any machine this is
/// likely to run on — under-blocking costs some bandwidth, over-blocking falls
/// off a cliff — and never a reason to fail a contraction.
pub const BUILTIN: CacheHierarchy = CacheHierarchy {
    l1d: CacheLevel {
        level: 1,
        size: 32 * 1024,
        line: 64,
        ways: 8,
        sets: 64,
        shared_by: 1,
    },
    l2: Some(CacheLevel {
        level: 2,
        size: 256 * 1024,
        line: 64,
        ways: 8,
        sets: 512,
        shared_by: 1,
    }),
    l3: Some(CacheLevel {
        level: 3,
        size: 8 * 1024 * 1024,
        line: 64,
        ways: 16,
        sets: 8192,
        shared_by: 4,
    }),
    source: CacheSource::Builtin,
};

impl CacheHierarchy {
    /// Logical CPUs per physical core, inferred from how many share the L1d.
    /// 2 on a hyperthreaded x86, 1 otherwise.
    pub fn threads_per_core(&self) -> usize {
        self.l1d.shared_by.max(1)
    }

    /// How many *physical cores* share `lvl`.
    ///
    /// The model assumes one thread per physical core, which is what every
    /// BLIS-shaped library assumes and what `scripts/env.sh` pins. Under that
    /// assumption a private-but-hyperthread-shared L2 (`shared_by == 2` on this
    /// machine) is one core's to itself, while a socket L3 (`shared_by == 16`)
    /// is contended by 8. Oversubscribing the siblings would halve the L2 and
    /// L1 a thread really gets; the project's own measurement rule already
    /// treats that configuration as invalid, so it is not modelled.
    pub fn cores_sharing(&self, lvl: &CacheLevel) -> usize {
        (lvl.shared_by / self.threads_per_core()).max(1)
    }

    /// How many **L3 domains** a run of `threads` threads spans.
    ///
    /// This is the input the partition rule was missing (A36): the shared packed
    /// `B` panel is sized for *an* L3, so whether "shared" means what the design
    /// assumed depends on how many separate L3s the thread set covers. One L3
    /// per socket gives 1 at every thread count up to the socket; a chiplet
    /// machine with a 4-core L3 gives 1 at 4 threads and 16 at 64.
    ///
    /// **Compact placement is assumed**: `threads` threads occupy `threads`
    /// consecutive physical cores, filling one domain before starting the next.
    /// That is what `scripts/phase4f-threads.sh` does by construction and what a
    /// whole-node run does anyway; a *scattered* placement spans more domains
    /// than this reports, and the error is in the safe direction — it under-counts,
    /// so the rule falls back to the behaviour every committed number was measured
    /// with. `TENSORCONTRACT_L3_DOMAINS` overrides it for exactly that case; see
    /// [`l3_domains`].
    ///
    /// A machine with no L3 at all has nothing shared to spread, so every core is
    /// its own domain.
    pub fn l3_domains(&self, threads: usize) -> usize {
        let threads = threads.max(1);
        match &self.l3 {
            Some(l3) => threads.div_ceil(self.cores_sharing(l3)),
            None => threads,
        }
    }
}

/// How many L3 domains a `threads`-wide run spans on *this* machine.
///
/// [`CacheHierarchy::l3_domains`] over [`hierarchy`], with
/// `TENSORCONTRACT_L3_DOMAINS` overriding the derivation. The override exists
/// because the derivation assumes compact placement: it is how a scattered
/// cpuset (the same thread count spread one-per-domain instead of packed) can be
/// measured against a packed one without a rebuild, and it is how the rule is
/// exercised on a machine that has only one domain.
///
/// Read once per process, like every other environment switch here.
pub fn l3_domains(threads: usize) -> usize {
    #[cfg(feature = "std")]
    {
        use std::sync::OnceLock;
        static ENV: OnceLock<Option<usize>> = OnceLock::new();
        let forced = *ENV.get_or_init(|| {
            std::env::var("TENSORCONTRACT_L3_DOMAINS")
                .ok()
                .and_then(|v| v.trim().parse::<usize>().ok())
                .filter(|&n| n > 0)
        });
        if let Some(n) = forced {
            return n.min(threads.max(1));
        }
    }
    hierarchy().l3_domains(threads)
}

/// The cache hierarchy of this machine: sysfs, then `CPUID`, then [`BUILTIN`].
///
/// Probed once per process and cached, the same way `env_blocking` and
/// `orient_override` are. A probe that fails is never an error: it degrades to
/// the next source, and the last source always succeeds.
pub fn hierarchy() -> CacheHierarchy {
    #[cfg(feature = "std")]
    {
        use std::sync::OnceLock;
        static H: OnceLock<CacheHierarchy> = OnceLock::new();
        *H.get_or_init(probe)
    }
    #[cfg(not(feature = "std"))]
    {
        probe()
    }
}

fn probe() -> CacheHierarchy {
    #[cfg(feature = "std")]
    if let Some(h) = probe_sysfs() {
        return h;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if let Some(h) = probe_cpuid() {
        return h;
    }
    BUILTIN
}

// ---------------------------------------------------------------------------
// Source 1: Linux sysfs
// ---------------------------------------------------------------------------

/// The attributes of one `/sys/.../cache/index*` directory, as raw text.
///
/// Parsing is expressed against this rather than against the filesystem so it
/// can be tested on fixtures — the layout of sysfs is not something a unit test
/// should need a particular kernel to exercise.
#[cfg(feature = "std")]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SysfsIndex<'a> {
    pub level: &'a str,
    pub kind: &'a str,
    pub size: &'a str,
    pub ways: &'a str,
    pub line: &'a str,
    /// `number_of_sets`, optional: derived from size/line/ways when absent.
    pub sets: &'a str,
    pub shared: &'a str,
}

/// Assemble a hierarchy from sysfs text. `None` if there is no usable L1d.
#[cfg(feature = "std")]
pub(crate) fn from_sysfs(indices: &[SysfsIndex<'_>]) -> Option<CacheHierarchy> {
    let mut h = CacheHierarchy {
        l1d: BUILTIN.l1d,
        l2: None,
        l3: None,
        source: CacheSource::Sysfs,
    };
    let mut have_l1d = false;
    for idx in indices {
        // Skip the instruction cache: only data caches hold packed panels.
        // "Unified" counts, which is what L2/L3 normally report.
        if idx.kind.trim().eq_ignore_ascii_case("Instruction") {
            continue;
        }
        let Some(lvl) = parse_level(idx) else {
            continue;
        };
        match lvl.level {
            1 if !have_l1d => {
                h.l1d = lvl;
                have_l1d = true;
            }
            2 if h.l2.is_none() => h.l2 = Some(lvl),
            3 if h.l3.is_none() => h.l3 = Some(lvl),
            _ => {}
        }
    }
    have_l1d.then_some(h)
}

#[cfg(feature = "std")]
fn parse_level(idx: &SysfsIndex<'_>) -> Option<CacheLevel> {
    let level: u8 = idx.level.trim().parse().ok()?;
    if !(1..=3).contains(&level) {
        return None;
    }
    let size = parse_size(idx.size)?;
    let line = parse_num(idx.line)?;
    let ways = parse_num(idx.ways)?;
    if size == 0 || line == 0 || ways == 0 {
        return None;
    }
    // `number_of_sets` is what the model wants; derive it when the kernel does
    // not export it, which some architectures do not.
    let sets = parse_num(idx.sets).unwrap_or(0);
    let sets = if sets > 0 { sets } else { size / (line * ways) };
    if sets == 0 {
        return None;
    }
    Some(CacheLevel {
        level,
        size,
        line,
        ways,
        sets,
        shared_by: parse_cpu_list(idx.shared).unwrap_or(1),
    })
}

#[cfg(feature = "std")]
fn parse_num(s: &str) -> Option<usize> {
    s.trim().parse().ok()
}

/// sysfs sizes carry a unit suffix: `32K`, `1024K`, `25344K`.
#[cfg(feature = "std")]
fn parse_size(s: &str) -> Option<usize> {
    let s = s.trim();
    let (digits, mult) = match s.chars().last()? {
        'K' | 'k' => (&s[..s.len() - 1], 1024),
        'M' | 'm' => (&s[..s.len() - 1], 1024 * 1024),
        'G' | 'g' => (&s[..s.len() - 1], 1024 * 1024 * 1024),
        _ => (s, 1),
    };
    digits.trim().parse::<usize>().ok()?.checked_mul(mult)
}

/// Count the CPUs in a `shared_cpu_list`: `0,16` is 2, `0-7,16-23` is 16.
///
/// This is the one piece of information no other source reports as directly,
/// and the model needs it to know whether an L3 budget is one core's or a
/// socket's.
#[cfg(feature = "std")]
fn parse_cpu_list(s: &str) -> Option<usize> {
    let mut n = 0usize;
    for part in s.trim().split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        match part.split_once('-') {
            Some((a, b)) => {
                let a: usize = a.trim().parse().ok()?;
                let b: usize = b.trim().parse().ok()?;
                n += b.checked_sub(a)?.checked_add(1)?;
            }
            None => {
                part.parse::<usize>().ok()?;
                n += 1;
            }
        }
    }
    (n > 0).then_some(n)
}

#[cfg(feature = "std")]
fn probe_sysfs() -> Option<CacheHierarchy> {
    const BASE: &str = "/sys/devices/system/cpu/cpu0/cache";
    let read = |dir: &str, name: &str| -> String {
        std::fs::read_to_string(format!("{dir}/{name}")).unwrap_or_default()
    };
    let mut raw: Vec<[String; 7]> = Vec::new();
    // `index*` directories are numbered contiguously from 0; stop at the first
    // gap. The bound is a safety net, not a real limit — no CPU has 16 levels.
    for i in 0..16 {
        let dir = format!("{BASE}/index{i}");
        let level = read(&dir, "level");
        if level.trim().is_empty() {
            break;
        }
        raw.push([
            level,
            read(&dir, "type"),
            read(&dir, "size"),
            read(&dir, "ways_of_associativity"),
            read(&dir, "coherency_line_size"),
            read(&dir, "number_of_sets"),
            read(&dir, "shared_cpu_list"),
        ]);
    }
    let idx: Vec<SysfsIndex<'_>> = raw
        .iter()
        .map(|r| SysfsIndex {
            level: &r[0],
            kind: &r[1],
            size: &r[2],
            ways: &r[3],
            line: &r[4],
            sets: &r[5],
            shared: &r[6],
        })
        .collect();
    from_sysfs(&idx)
}

// ---------------------------------------------------------------------------
// Source 2: x86 CPUID
// ---------------------------------------------------------------------------

/// Decode one deterministic-cache-parameters leaf (`CPUID.4` on Intel,
/// `CPUID.8000001D` on AMD — the two use the same register encoding).
///
/// Returns the cache type (1 data, 2 instruction, 3 unified) and the level,
/// or `None` for the null subleaf that terminates the enumeration. Split out
/// from the `CPUID` call so the bit-twiddling can be unit-tested.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) fn decode_cache_leaf(eax: u32, ebx: u32, ecx: u32) -> Option<(u32, CacheLevel)> {
    let kind = eax & 0x1f;
    if kind == 0 {
        return None; // null subleaf: enumeration over
    }
    let level = ((eax >> 5) & 0x7) as u8;
    let shared_by = (((eax >> 14) & 0xfff) + 1) as usize;
    let ways = (((ebx >> 22) & 0x3ff) + 1) as usize;
    let partitions = (((ebx >> 12) & 0x3ff) + 1) as usize;
    let line = ((ebx & 0xfff) + 1) as usize;
    let sets = (ecx as usize) + 1;
    Some((
        kind,
        CacheLevel {
            level,
            size: ways * partitions * line * sets,
            line,
            ways,
            // The model reasons in `sets * line` bytes per way, which is only
            // the true way size when a line is one partition. Fold the
            // partition count in so `bytes_per_way * ways == size` holds.
            sets: sets * partitions,
            shared_by,
        },
    ))
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn probe_cpuid() -> Option<CacheHierarchy> {
    #[cfg(target_arch = "x86")]
    use core::arch::x86 as arch;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64 as arch;

    // `__cpuid_count` became a *safe* function only in Rust 1.94 (stdarch #1935
    // — every x86 target feature postdates `CPUID`, so its availability is
    // implied). It is `unsafe` on the workspace MSRV of 1.89, so the block is
    // required there; `unused_unsafe` is allowed because the same block is
    // redundant from 1.94 on, and CI compiles with `-D warnings` on both.
    //
    // Keeping the block rather than raising the MSRV is deliberate: D20 chose
    // 1.89 because that is where the AVX-512 intrinsics stabilised, and paying
    // five Rust releases of compatibility for two braces is the wrong trade.
    //
    // When the MSRV does reach 1.94, the `allow` and the `unsafe` block come out
    // **together**. Dropping only the `allow` is a hard error on 1.89..1.94;
    // dropping only the block leaves a bare `allow` suppressing nothing, which
    // will outlast anyone's memory of why it was there.
    //
    // SAFETY: `CPUID` is unconditionally available on every x86 CPU that can
    // run this code — it predates every target feature the dispatch tests for —
    // and the instruction only reads processor identification registers.
    #[allow(unused_unsafe)]
    let leaf = |leaf: u32, sub: u32| unsafe { arch::__cpuid_count(leaf, sub) };

    let max_basic = leaf(0, 0).eax;
    let max_ext = leaf(0x8000_0000, 0).eax;
    // Leaf 4 is Intel's; AMD mirrors the same encoding at 0x8000001D. Try the
    // basic one first and fall through when it enumerates nothing.
    let candidates = [
        (max_basic >= 4).then_some(4u32),
        (max_ext >= 0x8000_001D).then_some(0x8000_001D),
    ];

    for base in candidates.into_iter().flatten() {
        let mut h = CacheHierarchy {
            l1d: BUILTIN.l1d,
            l2: None,
            l3: None,
            source: CacheSource::Cpuid,
        };
        let mut have_l1d = false;
        for sub in 0..16 {
            let r = leaf(base, sub);
            let Some((kind, lvl)) = decode_cache_leaf(r.eax, r.ebx, r.ecx) else {
                break;
            };
            if kind == 2 {
                continue; // instruction cache
            }
            match lvl.level {
                1 if !have_l1d => {
                    h.l1d = lvl;
                    have_l1d = true;
                }
                2 if h.l2.is_none() => h.l2 = Some(lvl),
                3 if h.l3.is_none() => h.l3 = Some(lvl),
                _ => {}
            }
        }
        if have_l1d {
            return Some(h);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// The switch
// ---------------------------------------------------------------------------

/// Which blocking derivation is in force.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BlockModel {
    /// The hardcoded Phase 2 constants: today's shipping behaviour.
    #[default]
    Legacy,
    /// [`analytical`], driven by [`hierarchy`].
    Analytical,
}

impl BlockModel {
    /// Parse the `TENSORCONTRACT_BLOCKMODEL` spelling. Several synonyms are
    /// accepted for each arm so that a sweep script can say `on`/`off` and a
    /// report can say `legacy`/`model`.
    pub fn parse(s: &str) -> Option<BlockModel> {
        match s.trim().to_ascii_lowercase().as_str() {
            "legacy" | "phase2" | "const" | "off" => Some(BlockModel::Legacy),
            "model" | "analytic" | "analytical" | "blis" | "on" => Some(BlockModel::Analytical),
            _ => None,
        }
    }

    /// The canonical name, which [`BlockModel::parse`] round-trips. Printed
    /// alongside every measurement, because two arms of a sweep taken under
    /// different derivations are not comparable.
    pub fn name(self) -> &'static str {
        match self {
            BlockModel::Legacy => "legacy",
            BlockModel::Analytical => "model",
        }
    }
}

/// `TENSORCONTRACT_BLOCKMODEL=legacy|model`, read once per process.
///
/// **The default is `legacy`, deliberately.** Two reasons, both concrete:
/// the pending `MC`/`KC`/`NC` grid (`scripts/phase4e-blocking.sh`) defines its
/// arms *relative to the derived defaults*, so changing the derivation would
/// silently change what that measurement means; and every performance number in
/// `docs/notebook/` was taken against the hardcoded constants, which the project
/// requires be comparable through a run-time switch rather than a
/// build-to-build diff (A15). Flip the default only after an end-to-end A/B in
/// the configuration that ships (A20).
pub fn block_model() -> BlockModel {
    env_once!(
        BlockModel,
        "TENSORCONTRACT_BLOCKMODEL",
        BlockModel::Legacy,
        |v| BlockModel::parse(v).unwrap_or_default()
    )
}

// ---------------------------------------------------------------------------
// The model
// ---------------------------------------------------------------------------

/// The packed-panel geometry the model needs, all of it read off the selected
/// micro-kernel.
///
/// `a_reals`/`b_reals` are the reals a packed panel carries per *logical*
/// element ([`super::PackFormat::reals_per_element`]), which is what makes the
/// three complex methods see the same cache budget rather than the same element
/// count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PanelGeom {
    /// `S_DATA`: bytes in one real scalar.
    pub real_bytes: usize,
    /// Reals a packed `A` panel carries per logical element: 1 real, 2 planar,
    /// 4 for 1m's "1e", 3 for 3m.
    pub a_reals: usize,
    /// Reals a packed `B` panel carries per logical element. Differs from
    /// `a_reals` only under 1m, whose `B` stays in planar "1r" form.
    pub b_reals: usize,
    /// Logical rows of the micro-tile.
    pub mr: usize,
    /// Logical columns of the micro-tile.
    pub nr: usize,
}

impl PanelGeom {
    /// Bytes one k-step of an `A` micro-panel occupies (`mr * S_DATA` in the
    /// paper, which has one real per element).
    const fn a_step(&self) -> usize {
        self.mr * self.a_reals * self.real_bytes
    }
    /// Bytes one k-step of a `B` micro-panel occupies.
    const fn b_step(&self) -> usize {
        self.nr * self.b_reals * self.real_bytes
    }
}

/// Guard rails. Blocking parameters larger than this are meaningless (the
/// driver clamps to the problem anyway) and only invite overflow on a machine
/// that reports nonsense.
const MAX_BLOCK: usize = 1 << 20;

/// The analytical blocking for one kernel geometry, thread count and machine.
///
/// `threads` is the count the *plan* will execute with. It enters in exactly one
/// place: the L3. The driver threads over `M` with a **shared** packed `B`
/// panel, so `nc`'s L3 budget is a per-socket resource that all threads use
/// cooperatively and must *not* be divided by the thread count — but each
/// thread keeps its **own** packed `A` block, and those all live in the same
/// L3, so `A`'s claim on it is charged `t` times. `mc` and `kc` come from the
/// L2 and L1, which are private to a core under the one-thread-per-core
/// assumption ([`CacheHierarchy::cores_sharing`]), so they do not depend on the
/// thread count at all.
pub fn analytical(geom: PanelGeom, threads: usize, h: &CacheHierarchy) -> Blocking {
    let geom = PanelGeom {
        real_bytes: geom.real_bytes.max(1),
        a_reals: geom.a_reals.max(1),
        b_reals: geom.b_reals.max(1),
        mr: geom.mr.max(1),
        nr: geom.nr.max(1),
    };
    let kc = model_kc(&geom, &h.l1d);
    let mc = match &h.l2 {
        Some(l2) => model_mc(&geom, kc, l2),
        // No L2: the packed block has nowhere to be resident, so make it one
        // micro-panel and let `A` stream. Inference, not the paper's.
        None => geom.mr,
    };
    let nc = match &h.l3 {
        Some(l3) => model_nc(&geom, kc, mc, threads, h, l3),
        // No L3, which is the common case the paper declines to model: `nc` is
        // "for all practical purposes redundant" (§5.2), so take a panel wide
        // enough not to re-stream `A` needlessly and leave it there.
        None => NO_L3_NC,
    };
    Blocking {
        mc: mc.clamp(1, MAX_BLOCK),
        kc: kc.clamp(1, MAX_BLOCK),
        nc: nc.clamp(1, MAX_BLOCK),
    }
}

/// `nc` when the machine has no L3 at all. BLIS uses values of this order for
/// the same reason: large enough that `A_c` is not re-packed often, small
/// enough that the panel is a sane allocation. Inferred, not the paper's.
const NO_L3_NC: usize = 4096;

/// `kc` from the L1, following eq. (4)–(6) of the paper.
fn model_kc(g: &PanelGeom, l1: &CacheLevel) -> usize {
    let way = l1.bytes_per_way().max(1);
    let (a, b) = (g.a_step().max(1), g.b_step().max(1));
    // `C_Ar <= floor((W_L1 - 1) / (1 + nr/mr))`, in bytes so that the panel
    // formats' differing reals-per-element are accounted for; taken as large as
    // possible, which is the paper's advice.
    let c_ar = l1.ways.saturating_sub(1) * a / (a + b);
    if c_ar == 0 {
        // Eq. (6): a 2-way (or effectively 2-way) L1 cannot host whole ways of
        // `A_r` and a resident `B_r` at once, so split it in half instead.
        (way / (2 * a)).max(1)
    } else {
        (c_ar * way / a).max(1)
    }
}

/// `mc` from the L2. Reconstruction of the paper's "similar manner"; see the
/// module docs for what it reproduces and what it does not.
fn model_mc(g: &PanelGeom, kc: usize, l2: &CacheLevel) -> usize {
    let way = l2.bytes_per_way().max(1);
    // Ways the streaming `B_c` micro-panel occupies while `A_c` is resident.
    let c_bc = (g.b_step() * kc).div_ceil(way).max(1);
    // ... and one more for the micro-tiles of `C_c`, exactly as eq. (5) does
    // in the L1.
    let c_ac = l2.ways.saturating_sub(1).saturating_sub(c_bc).max(1);
    // NOTE (A13): this is the *lower* bound on `mc` only — the size that keeps
    // packed `A` resident in L2. `mc` is also bounded from **above** by the
    // strip of `D` that one `jr` pass revisits, which no cache-residency model
    // can see and which is why re-deriving at `min(k, KC)` swung two corpus
    // cases +13% and -18%. Phase 4 part 7's grid is designed to separate them;
    // until it has run there is deliberately nothing here.
    (c_ac * way / (kc * g.a_reals * g.real_bytes).max(1)).max(g.mr)
}

/// `nc` from the L3, with the threading correction described on [`analytical`].
fn model_nc(
    g: &PanelGeom,
    kc: usize,
    mc: usize,
    threads: usize,
    h: &CacheHierarchy,
    l3: &CacheLevel,
) -> usize {
    let way = l3.bytes_per_way().max(1);
    // One packed `A` block per thread, all of them streaming through the L3
    // that holds the one shared `B` panel. Threads beyond the cores that share
    // this cache are on another socket with an L3 of its own.
    let sharers = threads.clamp(1, h.cores_sharing(l3));
    let a_block = mc * kc * g.a_reals * g.real_bytes;
    let c_ac = (a_block * sharers).div_ceil(way).max(1);
    let c_bc = l3.ways.saturating_sub(1).saturating_sub(c_ac).max(1);
    (c_bc * way / (kc * g.b_reals * g.real_bytes).max(1)).max(g.nr)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// This machine (`ccqlin038`, Xeon Gold 6244), as its sysfs reports it: a
    /// 32 KiB 8-way L1d shared by a hyperthread pair, a 1 MiB 16-way L2
    /// likewise, and a 25344 KiB 11-way L3 shared by a whole socket. Written
    /// out rather than parsed so that the model's tests do not depend on the
    /// probe's, and so that the probe has something exact to be checked against.
    const CASCADE: CacheHierarchy = CacheHierarchy {
        l1d: CacheLevel {
            level: 1,
            size: 32 * 1024,
            line: 64,
            ways: 8,
            sets: 64,
            shared_by: 2,
        },
        l2: Some(CacheLevel {
            level: 2,
            size: 1024 * 1024,
            line: 64,
            ways: 16,
            sets: 1024,
            shared_by: 2,
        }),
        l3: Some(CacheLevel {
            level: 3,
            size: 25344 * 1024,
            line: 64,
            ways: 11,
            sets: 36864,
            shared_by: 16,
        }),
        source: CacheSource::Sysfs,
    };

    fn cascade_lake() -> CacheHierarchy {
        CASCADE
    }

    /// This machine's `/sys/devices/system/cpu/cpu0/cache/index*`, verbatim.
    #[cfg(feature = "std")]
    fn cascade_lake_fixture() -> Vec<[&'static str; 7]> {
        vec![
            ["1", "Data", "32K", "8", "64", "64", "0,16"],
            ["1", "Instruction", "32K", "8", "64", "64", "0,16"],
            ["2", "Unified", "1024K", "16", "64", "1024", "0,16"],
            ["3", "Unified", "25344K", "11", "64", "36864", "0-7,16-23"],
        ]
    }

    #[cfg(feature = "std")]
    fn as_indices(raw: &[[&'static str; 7]]) -> Vec<SysfsIndex<'static>> {
        raw.iter()
            .map(|r| SysfsIndex {
                level: r[0],
                kind: r[1],
                size: r[2],
                ways: r[3],
                line: r[4],
                sets: r[5],
                shared: r[6],
            })
            .collect()
    }

    #[cfg(feature = "std")]
    #[test]
    fn sysfs_fixture_parses() {
        let raw = cascade_lake_fixture();
        let h = from_sysfs(&as_indices(&raw)).expect("fixture has an L1d");
        // Including that the instruction cache at level 1 was not mistaken for
        // the data cache, and that `0,16` is two CPUs while `0-7,16-23` is 16.
        assert_eq!(h, CASCADE);
        // Every level's geometry must be self-consistent, or the model's
        // way-counting is meaningless.
        for lvl in [Some(h.l1d), h.l2, h.l3].into_iter().flatten() {
            assert_eq!(lvl.bytes_per_way() * lvl.ways, lvl.size);
        }
        // 2 logical CPUs per core, so the L2 is one core's and the L3 is eight.
        assert_eq!(h.threads_per_core(), 2);
        assert_eq!(h.cores_sharing(&h.l2.unwrap()), 1);
        assert_eq!(h.cores_sharing(&h.l3.unwrap()), 8);
    }

    /// The input the partition rule was missing (A36), on the two topologies
    /// Phase 4 measured: a socket-wide L3 spans one domain right up to the socket
    /// and a chiplet L3 starts spanning them almost immediately. That difference
    /// is the whole content of the rule, so it is pinned here rather than left to
    /// whatever machine happens to run the suite.
    #[test]
    fn l3_domains_separates_a_socket_l3_from_a_chiplet_one() {
        let socket = CASCADE; // 16 logical CPUs on the L3, SMT 2 -> 8 cores
        assert_eq!(socket.l3_domains(1), 1);
        assert_eq!(socket.l3_domains(8), 1);
        assert_eq!(socket.l3_domains(9), 2); // a second socket, or oversubscribed
        assert_eq!(socket.l3_domains(32), 4);

        // Zen2: four cores per 16 MiB L3, SMT off in the measured allocation.
        let mut chiplet = CASCADE;
        chiplet.l1d.shared_by = 1;
        chiplet.l3 = Some(CacheLevel {
            shared_by: 4,
            ..CASCADE.l3.unwrap()
        });
        assert_eq!(chiplet.l3_domains(4), 1); // the point null in all four dtypes
        assert_eq!(chiplet.l3_domains(16), 4); // 1.17-1.27x for the column axis
        assert_eq!(chiplet.l3_domains(64), 16); // up to 4.3x

        // No L3 at all: nothing is shared, so every thread is its own domain.
        let none = CacheHierarchy {
            l3: None,
            ..CASCADE
        };
        assert_eq!(none.l3_domains(8), 8);
    }

    #[cfg(feature = "std")]
    #[test]
    fn sysfs_derives_missing_set_count() {
        // Not every kernel exports `number_of_sets`.
        let raw = vec![["1", "Data", "32K", "8", "64", "", "0"]];
        let h = from_sysfs(&as_indices(&raw)).unwrap();
        assert_eq!(h.l1d.sets, 64);
        assert_eq!(h.l1d.shared_by, 1);
        assert!(h.l2.is_none() && h.l3.is_none());
    }

    #[cfg(feature = "std")]
    #[test]
    fn sysfs_rejects_junk() {
        // No data cache at all: the probe must decline rather than invent one.
        let raw = vec![["1", "Instruction", "32K", "8", "64", "64", "0"]];
        assert!(from_sysfs(&as_indices(&raw)).is_none());
        // Unparseable fields are skipped level by level, not fatal.
        let raw = vec![
            ["1", "Data", "", "8", "64", "64", "0"],
            ["2", "Unified", "1024K", "zero", "64", "1024", "0"],
        ];
        assert!(from_sysfs(&as_indices(&raw)).is_none());
        // A level out of range is ignored.
        let raw = vec![
            ["1", "Data", "32K", "8", "64", "64", "0"],
            ["4", "Unified", "128M", "16", "64", "131072", "0-63"],
        ];
        let h = from_sysfs(&as_indices(&raw)).unwrap();
        assert!(h.l2.is_none() && h.l3.is_none());
    }

    #[cfg(feature = "std")]
    #[test]
    fn size_and_cpu_list_parsing() {
        assert_eq!(parse_size("32K"), Some(32 * 1024));
        assert_eq!(parse_size(" 1024K\n"), Some(1024 * 1024));
        assert_eq!(parse_size("8M"), Some(8 * 1024 * 1024));
        assert_eq!(parse_size("512"), Some(512));
        assert_eq!(parse_size(""), None);
        assert_eq!(parse_size("K"), None);
        assert_eq!(parse_cpu_list("0"), Some(1));
        assert_eq!(parse_cpu_list("0,16"), Some(2));
        assert_eq!(parse_cpu_list("0-7,16-23\n"), Some(16));
        assert_eq!(parse_cpu_list("0-3"), Some(4));
        assert_eq!(parse_cpu_list(""), None);
        assert_eq!(parse_cpu_list("7-0"), None, "a reversed range is junk");
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn cpuid_leaf_decoding() {
        // Hand-encoded 1 MiB 16-way unified L2 with 64-byte lines and 1024
        // sets, shared by 2 logical CPUs — this machine's, as CPUID.4 would
        // report it.
        let (kind_in, level_in, shared_in) = (3u32, 2u32, 2u32);
        let (line_in, partitions_in, ways_in, sets_in) = (64u32, 1u32, 16u32, 1024u32);
        let eax = kind_in | (level_in << 5) | ((shared_in - 1) << 14);
        let ebx = (line_in - 1) | ((partitions_in - 1) << 12) | ((ways_in - 1) << 22);
        let ecx = sets_in - 1;
        let (kind, lvl) = decode_cache_leaf(eax, ebx, ecx).unwrap();
        assert_eq!(kind, 3);
        assert_eq!(lvl.level, 2);
        assert_eq!(lvl.size, 1 << 20);
        assert_eq!(lvl.ways, 16);
        assert_eq!(lvl.sets, 1024);
        assert_eq!(lvl.line, 64);
        assert_eq!(lvl.shared_by, 2);
        assert_eq!(lvl.bytes_per_way() * lvl.ways, lvl.size);
        // The null subleaf terminates enumeration.
        assert!(decode_cache_leaf(0, 0, 0).is_none());
    }

    /// The paper's own validation table, as a check that the reconstruction of
    /// `mc` is the paper's and not something else. Intel SandyBridge (E3-1220)
    /// from Table III: 32 KiB 8-way L1 with 64 sets, 256 KiB 8-way L2 with 512
    /// sets, `mr = 8`, `nr = 4`, double precision. The paper derives
    /// `kc = 256`, `mc = 96`.
    #[test]
    fn reproduces_the_papers_sandybridge_row() {
        let h = CacheHierarchy {
            l1d: CacheLevel {
                level: 1,
                size: 32 * 1024,
                line: 64,
                ways: 8,
                sets: 64,
                shared_by: 1,
            },
            l2: Some(CacheLevel {
                level: 2,
                size: 256 * 1024,
                line: 64,
                ways: 8,
                sets: 512,
                shared_by: 1,
            }),
            l3: None,
            source: CacheSource::Builtin,
        };
        let g = PanelGeom {
            real_bytes: 8,
            a_reals: 1,
            b_reals: 1,
            mr: 8,
            nr: 4,
        };
        let blk = analytical(g, 1, &h);
        assert_eq!(blk.kc, 256, "eq. (4) with C_Ar = 4");
        assert_eq!(blk.mc, 96, "Table III");
    }

    /// The TI C6678 row of the same table: 32 KiB 4-way L1 with 256 sets,
    /// 512 KiB 4-way L2 with 2048 sets, `mr = nr = 4`, giving `kc = 256`,
    /// `mc = 128`.
    #[test]
    fn reproduces_the_papers_c6678_row() {
        let h = CacheHierarchy {
            l1d: CacheLevel {
                level: 1,
                size: 32 * 1024,
                line: 32,
                ways: 4,
                sets: 256,
                shared_by: 1,
            },
            l2: Some(CacheLevel {
                level: 2,
                size: 512 * 1024,
                line: 64,
                ways: 4,
                sets: 2048,
                shared_by: 1,
            }),
            l3: None,
            source: CacheSource::Builtin,
        };
        let g = PanelGeom {
            real_bytes: 8,
            a_reals: 1,
            b_reals: 1,
            mr: 4,
            nr: 4,
        };
        let blk = analytical(g, 1, &h);
        assert_eq!(blk.kc, 256);
        assert_eq!(blk.mc, 128);
        assert_eq!(blk.nc, NO_L3_NC, "no L3 in this machine");
    }

    /// Every footprint the model reasons about must actually fit the budget it
    /// was derived from. This is the invariant that would break first if a
    /// formula were transcribed wrongly.
    #[test]
    fn footprints_fit_their_budgets() {
        let h = cascade_lake();
        let l2 = h.l2.unwrap();
        let l3 = h.l3.unwrap();
        for real_bytes in [4usize, 8] {
            for (a_reals, b_reals) in [(1, 1), (2, 2), (4, 2), (3, 3)] {
                for mr in [4usize, 8, 12, 16, 24, 32, 48] {
                    for nr in [3usize, 4, 6, 8, 10, 12] {
                        let g = PanelGeom {
                            real_bytes,
                            a_reals,
                            b_reals,
                            mr,
                            nr,
                        };
                        for threads in [1usize, 8, 64] {
                            let b = analytical(g, threads, &h);
                            let a_panel = mr * b.kc * a_reals * real_bytes;
                            assert!(
                                a_panel <= h.l1d.size,
                                "A micro-panel {a_panel} over L1 for {g:?}"
                            );
                            let a_block = b.mc * b.kc * a_reals * real_bytes;
                            assert!(a_block <= l2.size, "A block {a_block} over L2 for {g:?}");
                            let b_panel = b.nc * b.kc * b_reals * real_bytes;
                            let threads_here = threads.min(h.cores_sharing(&l3));
                            assert!(
                                b_panel + a_block * threads_here <= l3.size,
                                "B panel {b_panel} + {threads_here} A blocks over L3 for {g:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// 1m packs four reals per complex element in `A` where planar packs two,
    /// so the same L2 must buy it a smaller `mc`. Getting this wrong would rig
    /// the project's headline three-way comparison, which is why it is asserted
    /// on the model as well as on the shipped configs.
    #[test]
    fn one_m_gets_a_smaller_mc_than_planar() {
        let h = cascade_lake();
        for real_bytes in [4usize, 8] {
            let planar = analytical(
                PanelGeom {
                    real_bytes,
                    a_reals: 2,
                    b_reals: 2,
                    mr: 16,
                    nr: 6,
                },
                1,
                &h,
            );
            let onem = analytical(
                PanelGeom {
                    real_bytes,
                    a_reals: 4,
                    b_reals: 2,
                    mr: 16,
                    nr: 6,
                },
                1,
                &h,
            );
            // At equal micro-tile shape, doubling the reals per element halves
            // both the depth the L1 can host and the rows the L2 can host.
            assert!(
                onem.kc <= planar.kc,
                "1m kc {} vs planar {}",
                onem.kc,
                planar.kc
            );
            let a_bytes = |b: &Blocking, r: usize| b.mc * b.kc * r * real_bytes;
            assert!(
                a_bytes(&onem, 4) <= a_bytes(&planar, 2) * 3 / 2,
                "1m packed A {} vs planar {}",
                a_bytes(&onem, 4),
                a_bytes(&planar, 2)
            );
        }
    }

    /// Threads shrink `nc` (their private `A` blocks crowd the shared `B`
    /// panel out of L3) but never `mc` or `kc`, and never below one micro-panel.
    #[test]
    fn threads_only_affect_nc() {
        let h = cascade_lake();
        let g = PanelGeom {
            real_bytes: 8,
            a_reals: 2,
            b_reals: 2,
            mr: 16,
            nr: 6,
        };
        let one = analytical(g, 1, &h);
        let eight = analytical(g, 8, &h);
        let over = analytical(g, 64, &h);
        assert_eq!((one.mc, one.kc), (eight.mc, eight.kc));
        assert!(eight.nc < one.nc, "eight threads must not get one's nc");
        assert_eq!(
            eight.nc, over.nc,
            "threads beyond the cores sharing this L3 are on another socket"
        );
        assert!(eight.nc >= g.nr);
    }

    /// Absurd inputs must produce usable numbers, never zero, never a panic.
    /// A probe that reports nonsense is a performance question, not a
    /// correctness one.
    #[test]
    fn absurd_inputs_stay_sane() {
        let tiny = CacheHierarchy {
            l1d: CacheLevel {
                level: 1,
                size: 64,
                line: 64,
                ways: 1,
                sets: 1,
                shared_by: 1,
            },
            l2: Some(CacheLevel {
                level: 2,
                size: 64,
                line: 64,
                ways: 1,
                sets: 1,
                shared_by: 1,
            }),
            l3: Some(CacheLevel {
                level: 3,
                size: 64,
                line: 64,
                ways: 1,
                sets: 1,
                shared_by: 1,
            }),
            source: CacheSource::Builtin,
        };
        let huge = PanelGeom {
            real_bytes: 16,
            a_reals: 4,
            b_reals: 3,
            mr: 4096,
            nr: 4096,
        };
        for h in [tiny, BUILTIN] {
            for g in [
                huge,
                PanelGeom {
                    real_bytes: 0,
                    a_reals: 0,
                    b_reals: 0,
                    mr: 0,
                    nr: 0,
                },
            ] {
                for threads in [0usize, 1, 1024] {
                    let b = analytical(g, threads, &h);
                    assert!(b.mc >= 1 && b.kc >= 1 && b.nc >= 1, "{b:?} for {g:?}");
                    assert!(b.mc <= MAX_BLOCK && b.kc <= MAX_BLOCK && b.nc <= MAX_BLOCK);
                }
            }
        }
    }

    #[test]
    fn a_level_without_the_next_one_still_models() {
        let mut h = cascade_lake();
        let g = PanelGeom {
            real_bytes: 8,
            a_reals: 2,
            b_reals: 2,
            mr: 16,
            nr: 6,
        };
        let full = analytical(g, 1, &h);
        h.l3 = None;
        let no_l3 = analytical(g, 1, &h);
        assert_eq!((no_l3.mc, no_l3.kc), (full.mc, full.kc));
        assert_eq!(no_l3.nc, NO_L3_NC);
        h.l2 = None;
        let no_l2 = analytical(g, 1, &h);
        assert_eq!(no_l2.kc, full.kc);
        assert_eq!(
            no_l2.mc, g.mr,
            "one micro-panel when nothing can be resident"
        );
    }

    #[test]
    fn switch_parses_and_defaults_to_legacy() {
        assert_eq!(BlockModel::default(), BlockModel::Legacy);
        for s in ["legacy", "LEGACY", " phase2 ", "off"] {
            assert_eq!(BlockModel::parse(s), Some(BlockModel::Legacy));
        }
        for s in ["model", "analytical", "blis", "on"] {
            assert_eq!(BlockModel::parse(s), Some(BlockModel::Analytical));
        }
        assert_eq!(BlockModel::parse("maybe"), None);
        for m in [BlockModel::Legacy, BlockModel::Analytical] {
            assert_eq!(BlockModel::parse(m.name()), Some(m));
        }
    }

    /// The probe on whatever machine the tests run on must produce something
    /// self-consistent, whichever source answers.
    #[test]
    fn probe_is_self_consistent() {
        let h = hierarchy();
        for lvl in [Some(h.l1d), h.l2, h.l3].into_iter().flatten() {
            assert!(lvl.size > 0 && lvl.line > 0 && lvl.ways > 0 && lvl.sets > 0);
            assert!(lvl.shared_by >= 1);
        }
        assert!(h.l1d.level == 1);
    }
}
