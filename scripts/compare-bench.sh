#!/bin/bash
# The engine against its baselines, measured the way this project measures now.
#
#   scripts/compare-bench.sh prep                      # build, and only build
#   scripts/compare-bench.sh OUTDIR [cpu] [size] [reps] [filter]
#
# e.g.  scripts/compare-bench.sh bench-results/$(hostname -s)-cascadelake
#
# This is `scripts/phase3-bench.sh` brought up to the standard the Rusty sessions
# set. That script must not simply be re-run, for three reasons, and each is fixed
# here:
#
#   * **It writes flat into `bench-results/`** and would overwrite the committed
#     Phase 3 CSVs that Phase 1's headline and A35 both rest on. This one takes an
#     output directory and refuses to write outside it.
#   * **It has no warm-up arm.** It predates A31: an exclusive node's opening arm
#     runs at single-core boost on a cold package and nothing else does, which came
#     back once as a uniform 4.4% "noise floor". One arm, run and thrown away.
#   * **It compiles in the middle of the measurement**, between the TBLIS 2.0 group
#     and the TBLIS 1.3.0 one. Both binaries are built up front by `prep` and this
#     script will not compile at all — `node-session.sh`'s rule, applied here.
#
# Two further things it does that the old one could not:
#
#   * **Two floors at two distances (A32).** `A2` repeats `A` immediately and `A3`
#     repeats it at the end of the session, so the drift over a minute and the
#     drift over the whole run are both measured rather than assumed. Judge a
#     cross-session claim against `floor-session.txt`, not `floor-near.txt`, and
#     judge neither against +-1.3%/+-6% from `DECISIONS.md` — those are one
#     session's numbers on a different machine.
#   * **Per-arm L3-domain occupancy**, through `scripts/run-arms.py`, so
#     co-tenancy is a recorded fact per arm. Two Phase 4 conclusions had to be
#     retracted because it was not.
#
# Everything runs **sequentially on one pinned core**. Concurrent one-arm-per-L3
# placement is rejected here on principle: A27 confirmed it on the corpus (+0.3%)
# and refuted it on the memory-bound half (-3.2%), and the baselines are exactly
# the arms that change memory traffic.
#
# Cost: about 3.5 h at the defaults. Nothing else may run on the machine,
# including a compile or an analysis script.
#
# Naming any TBLIS number without its version is a project rule, not a style
# preference: 1.3.0 and 2.0-dev differ by ~5x on complex and swap the
# TYPE_DOUBLE/TYPE_SCOMPLEX ABI enumerators. Both binaries' `tcbench info` output
# is captured into the output directory, and each self-checks its ABI at startup.
# pipefail matters here rather than being hygiene: every measurement in this
# script is piped into `tee`, so without it a failing `verify` or a failing
# `run-arms.py` is reported by `tee`'s exit status, which is always 0.
set -e -o pipefail
[ "${BASH_SOURCE[0]}" = "$0" ] || { echo "run me, do not source me" >&2; return 1; }
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# One target directory per TBLIS ABI. The two builds differ only in a cargo
# feature, so a shared target directory would make them evict each other, and the
# binary would have to be copied out from under a name cargo owns. Separate
# directories mean each build is its own artefact, `prep` is idempotent, and two
# machines sharing this checkout over GPFS cannot truncate each other's binary.
# The alternative -- build, copy, delete the original -- was tried and is how job
# 6753197 died: cargo's fingerprint still said "fresh", so the deleted file was
# never relinked, and a concurrent job's copy left a 0-byte executable behind
# that every arm then ran happily for 0.0 seconds.
BIN2=./target/tblis2/release/tcbench
BIN13=./target/tblis13/release/tcbench

# `runs BIN` -- is this a working binary, not merely a present one?
runs() { [ -x "$1" ] && [ -s "$1" ] && "$1" info >/dev/null 2>&1; }

# ---------------------------------------------------------------------------
# prep: the only thing here that compiles
# ---------------------------------------------------------------------------
if [ "${1:-}" = "prep" ]; then
    : "${TBLIS_ROOT_2X:?point at the TBLIS 2.0-dev install}"
    : "${TBLIS_ROOT_13:?point at the TBLIS v1.3.0 install}"
    module load gcc/13.3.0 openblas >/dev/null 2>&1 || true

    echo "building the TBLIS 2.0-dev binary"
    CARGO_TARGET_DIR=target/tblis2 TBLIS_ROOT=$TBLIS_ROOT_2X \
        cargo build --release -p tensorprimitives-bench --features tblis,blas

    echo "building the TBLIS v1.3.0 binary (note the tblis13 feature and ABI)"
    CARGO_TARGET_DIR=target/tblis13 TBLIS_ROOT=$TBLIS_ROOT_13 \
        cargo build --release -p tensorprimitives-bench --features tblis13,blas

    ls -l "$BIN2" "$BIN13"
    for b in "$BIN2" "$BIN13"; do
        LD_LIBRARY_PATH="$TBLIS_ROOT_2X/lib:$TBLIS_ROOT_13/lib:${OPENBLAS_ROOT:+$OPENBLAS_ROOT/lib:}${LD_LIBRARY_PATH:-}" \
            runs "$b" || { echo "FATAL: $b was built but does not run" >&2; exit 1; }
    done
    echo
    echo "prep done, and both binaries were executed once to prove it."
    echo "Nothing else in this script compiles."
    exit 0
fi

OUT=${1:?usage: scripts/compare-bench.sh OUTDIR [cpu] [size] [reps] [filter], or \`prep\`}
CPU=${2:-4}
SIZE=${3:-64}
REPS=${4:-3}
FILTER=${5:-}
# 200 MiB is the size Phase 1 used, kept so the efficiency-against-a-GEMM-ceiling
# metric stays comparable across phases. The corpus sweeps run smaller because at
# 200 MiB one case is ~5 min per engine and the corpus would take a day.
PREMISE_SIZE=${PREMISE_SIZE:-200}
VERIFY_SIZE=${VERIFY_SIZE:-8}
ENGINES=${ENGINES:-planar,1m,3m,ttgt,tblis}

: "${TBLIS_ROOT_2X:?point at the TBLIS 2.0-dev install}"
: "${TBLIS_ROOT_13:?point at the TBLIS v1.3.0 install}"
module load gcc/13.3.0 openblas >/dev/null 2>&1 || true

# Present is not the same as working. Both binaries are *executed* here, before
# anything is timed, because an unrunnable one does not fail an arm loudly -- it
# produces an arm that takes 0.0 s, returns rc=0 and writes no CSV at all.
for b in "$BIN2" "$BIN13"; do
    [ -e "$b" ] || { echo "no $b -- run: scripts/compare-bench.sh prep" >&2; exit 1; }
done
LD_LIBRARY_PATH="$TBLIS_ROOT_2X/lib:${OPENBLAS_ROOT:+$OPENBLAS_ROOT/lib:}${LD_LIBRARY_PATH:-}" \
    runs "$BIN2" || { echo "FATAL: $BIN2 does not run. Rebuild: scripts/compare-bench.sh prep" >&2; exit 1; }
LD_LIBRARY_PATH="$TBLIS_ROOT_13/lib:${OPENBLAS_ROOT:+$OPENBLAS_ROOT/lib:}${LD_LIBRARY_PATH:-}" \
    runs "$BIN13" || { echo "FATAL: $BIN13 does not run. Rebuild: scripts/compare-bench.sh prep" >&2; exit 1; }

others=$(pgrep -a -u "$USER" -f 'cargo|rustc|tcbench|kernel_shapes' \
         | grep -v "compare-bench\.sh\|pgrep" || true)
if [ -n "$others" ]; then
    echo "REFUSING: something of yours is running:" >&2
    echo "$others" >&2
    exit 1
fi

mkdir -p "$OUT"
LOG=$OUT/compare.log
: > "$LOG"
say() { echo "$@" | tee -a "$LOG"; }
stage() { say ""; say "########## $* ##########"; say ""; }

export TBLIS_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
libpath() { echo "$1/lib:${OPENBLAS_ROOT:+$OPENBLAS_ROOT/lib:}${LD_LIBRARY_PATH:-}"; }

FILT=()
[ -n "$FILTER" ] && FILT=(--case "$FILTER")
# arms JOBFILE BIN [extra run-arms.py args...]
# Trailing arguments override the defaults, because argparse takes the last
# occurrence -- that is how the premise and ragged stages change `--size`,
# `--engines` and `--subcommand` without a second copy of this invocation.
arms() {
    local jobs=$1 bin=$2; shift 2
    scripts/run-arms.py "$jobs" --outdir "$OUT" --bin "$bin" \
        --size "$SIZE" --reps "$REPS" --engines "$ENGINES" \
        "${FILT[@]}" --cpus "$CPU" --sequential "$@" | tee -a "$LOG"
}

# `produced TAG...` -- did those arms actually measure something? An arm that
# runs a broken binary reports rc=0 in 0.0 s and writes no CSV, so the absence of
# an error is not evidence that an arm ran. Check the output, every stage.
produced() {
    local tag bad=0
    for tag in "$@"; do
        if [ ! -s "$OUT/$tag.csv" ] || [ "$(wc -l < "$OUT/$tag.csv")" -lt 2 ]; then
            echo "FATAL: arm '$tag' produced no measurements ($OUT/$tag.csv)" >&2
            bad=1
        fi
    done
    [ "$bad" = 0 ] || { echo "Read $OUT/<tag>.txt for what the arm actually said." >&2
                        exit 1; }
}

say "comparison benchmark on cpu$CPU; size ${SIZE} MiB, ${REPS} reps, engines $ENGINES"
say "premise size ${PREMISE_SIZE} MiB; filter ${FILTER:-<none>}"
say "started $(date -Is) on $(hostname -s)"
scripts/topology.py --json "$OUT/topology.json" | tee -a "$LOG"

# ---------------------------------------------------------------------------
# 1. What is this machine, and is the engine correct on it?
# ---------------------------------------------------------------------------
# Cheap, and it fails loudly before four hours of measurement rather than after:
# a wrong ISA dispatch, a TBLIS ABI mismatch, or a baseline that will not load.
stage "environment and correctness"
export LD_LIBRARY_PATH=$(libpath "$TBLIS_ROOT_2X")
"$BIN2" info 2>&1 | tee "$OUT/info-tblis2.txt" | tee -a "$LOG"
for m in planar 1m 3m; do
    say "--- verify, $m ---"
    TENSORCONTRACT_COMPLEX=$m "$BIN2" verify --size "$VERIFY_SIZE" 2>&1 \
        | tee "$OUT/verify-$m.txt" | tail -5 | tee -a "$LOG"
done

# ---------------------------------------------------------------------------
# 2. The warm-up arm, run and discarded (A31)
# ---------------------------------------------------------------------------
stage "warm-up (discarded)"
printf '# discarded: this arm exists to leave the package hot (A31)\nwarm-f64c64 f64,c64\nwarm-f32c32 f32,c32\n' \
    > "$OUT/arms-warm.jobs"
arms "$OUT/arms-warm.jobs" "$BIN2"
produced warm-f64c64 warm-f32c32

# ---------------------------------------------------------------------------
# 3. The corpus, against TBLIS 2.0-dev and TTGT -- and its near repeat
# ---------------------------------------------------------------------------
stage "corpus sweep A, and its immediate repeat A2"
{ echo "A-f64c64  f64,c64"; echo "A-f32c32  f32,c32"
  echo "A2-f64c64 f64,c64"; echo "A2-f32c32 f32,c32"; } > "$OUT/arms-sweep.jobs"
arms "$OUT/arms-sweep.jobs" "$BIN2"
produced A-f64c64 A-f32c32 A2-f64c64 A2-f32c32

# ---------------------------------------------------------------------------
# 4. Efficiency against a same-shape GEMM ceiling -- the headline metric
# ---------------------------------------------------------------------------
stage "premise at ${PREMISE_SIZE} MiB, against TBLIS 2.0-dev"
{ echo "premise-f64c64 f64,c64"; echo "premise-f32c32 f32,c32"; } > "$OUT/arms-premise.jobs"
arms "$OUT/arms-premise.jobs" "$BIN2" --subcommand premise --size "$PREMISE_SIZE"
produced premise-f64c64 premise-f32c32

# ---------------------------------------------------------------------------
# 5. Irregular strides: the only mode that exercises the gather path
# ---------------------------------------------------------------------------
# The unperturbed TCCG corpus is fully regular (regA = 1.00 throughout), so any
# awkward-stride sentence in the README has to come from here and has to say so.
stage "ragged stress, c64"
echo "ragged-c64 c64" > "$OUT/arms-ragged.jobs"
arms "$OUT/arms-ragged.jobs" "$BIN2" --stress ragged --engines planar,1m,3m,tblis
produced ragged-c64

# ---------------------------------------------------------------------------
# 6. The session-span repeat
# ---------------------------------------------------------------------------
# A3 is the same computation as A, hours later. A2 - A prices a minute of drift;
# A3 - A prices the span every number above was taken across. They are different
# numbers and the second is the one a cross-stage claim has to clear (A32).
stage "corpus sweep A3, the session-span repeat"
{ echo "A3-f64c64 f64,c64"; echo "A3-f32c32 f32,c32"; } > "$OUT/arms-sweep3.jobs"
arms "$OUT/arms-sweep3.jobs" "$BIN2"
produced A3-f64c64 A3-f32c32

# ---------------------------------------------------------------------------
# 7. Against TBLIS v1.3.0 -- the latest stable release, and a different ABI
# ---------------------------------------------------------------------------
# A separate binary, built during prep, so this costs a library-path change and
# not a compile. The harness self-checks the ABI at startup and aborts on a
# mismatch, which is the only protection against plausible wrong numbers here.
stage "premise at ${PREMISE_SIZE} MiB, against TBLIS v1.3.0"
export LD_LIBRARY_PATH=$(libpath "$TBLIS_ROOT_13")
"$BIN13" info 2>&1 | tee "$OUT/info-tblis13.txt" | tee -a "$LOG"
echo "premise13-f64c64 f64,c64" > "$OUT/arms-premise13.jobs"
arms "$OUT/arms-premise13.jobs" "$BIN13" --subcommand premise \
    --size "$PREMISE_SIZE" --engines planar,tblis
produced premise13-f64c64

# ---------------------------------------------------------------------------
# 8. The two floors, and the provenance record
# ---------------------------------------------------------------------------
stage "this session's floors"
say "--- near floor: A2 against A (minutes apart) ---"
scripts/compare-sweeps.py "$OUT/A-f64c64.csv,$OUT/A-f32c32.csv" \
    "$OUT/A2-f64c64.csv,$OUT/A2-f32c32.csv" | tee "$OUT/floor-near.txt" | tee -a "$LOG"
say "--- session floor: A3 against A (the whole run apart) ---"
scripts/compare-sweeps.py "$OUT/A-f64c64.csv,$OUT/A-f32c32.csv" \
    "$OUT/A3-f64c64.csv,$OUT/A3-f32c32.csv" | tee "$OUT/floor-session.txt" | tee -a "$LOG"

{
    echo "Produced by scripts/compare-bench.sh on $(hostname -f)."
    echo
    echo "  started   : see the head of compare.log"
    echo "  finished  : $(date -Is)"
    echo "  slurm job : ${SLURM_JOB_ID:-<not under slurm>}"
    echo "  node      : ${SLURMD_NODENAME:-$(hostname -s)}"
    echo "  commit    : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "  invocation: scripts/compare-bench.sh $OUT $CPU $SIZE $REPS ${FILTER:-}"
    echo "  premise   : ${PREMISE_SIZE} MiB     engines: $ENGINES"
    echo
    echo "TBLIS baselines, both named because 1.3.0 and 2.0-dev differ by ~5x on"
    echo "complex and swap two ABI enumerators:"
    echo "  2.0-dev : $TBLIS_ROOT_2X"
    echo "  v1.3.0  : $TBLIS_ROOT_13"
    echo "Each binary's self-reported ABI is in info-tblis2.txt / info-tblis13.txt."
    echo
    echo "Arms, in the order they ran: warm (discarded), A, A2, premise, ragged,"
    echo "A3, premise13. One at a time, pinned to cpu$CPU, per-arm L3-domain"
    echo "occupancy in the .cpu files and in arms.jsonl."
    echo
    echo "Floors measured here, not imported: floor-near.txt (A2 vs A, minutes)"
    echo "and floor-session.txt (A3 vs A, the whole run). Nothing in this"
    echo "directory is comparable to a number measured on another machine."
} > "$OUT/PROVENANCE.txt"

say ""
say "done $(date -Is). Everything is in $OUT; read PROVENANCE.txt first."
say "Judge any claim against floor-session.txt, and name the TBLIS version."
