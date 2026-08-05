#!/bin/bash
# Phase 4 item 4: does threading still pay on *small* contractions?
#
# The threading default (D22) rests on `phase4f-threads.sh`, and every case in it
# has 64 MiB operands. That is the wrong half of the corpus for this question.
# Threads are spawned per `execute` call — `std::thread::scope`, no pool — so the
# fixed cost is paid once per contraction and amortises over the work, which
# means the interesting quantity is not a speedup but a **crossover size**.
#
# It matters more here than it would elsewhere: Phase 1 named *small repeated
# contractions* as this project's real headroom, so the regime with no threaded
# measurement is exactly the regime the engine is eventually for. A default
# turned on from large-case data alone could regress it silently.
#
# What this measures, and why it is shaped like this:
#
#   * `timed()` takes the **best of `reps`, each rep one `execute` call**, so the
#     spawn is inside the measurement and is never amortised across repetitions.
#     That is the property this whole script depends on — do not "optimise" it by
#     timing a loop of calls.
#   * Sizes descend to 0.25 MiB. TCCG sizing keeps each case's shape and scales
#     its extents to the nominal tensor size, so a small arm is the same 49
#     contractions on smaller tensors, not a different corpus.
#   * `reps` scales up as the size scales down, because a 5 ms case timed three
#     times is noise. The wall clock stays roughly flat across sizes as a result.
#   * The 64 MiB arm is deliberately **not** repeated here: `phase4f` already has
#     it, and re-running it would double the cost of this script to reproduce a
#     number that exists. Read the two together.
#
# The output to look at is the last section: speedup against `t1` **at the same
# size**. Somewhere between 0.25 and 64 MiB it crosses 1.0, and where it crosses
# is the answer. The implied fixed cost per call falls out of the same table —
# if `t64` is 0.5x at a size where a serial call takes `T`, the overhead is
# roughly `T` itself.
#
# If the crossover turns out to be high, the fix is not necessarily a thread
# pool: a work threshold in `Plan::threads` ("below this many flops, run serial")
# is a smaller change and needs no dependency. Decide that on this data, not
# before it.
#
# Usage: scripts/phase4g-small.sh [cpuset|auto] [outdir] [filter]
#   SIZES="0.25 1 4 16"   nominal tensor sizes in MiB
#   THREADS="1 2 4 8"     override the thread counts
#   NO_WARMUP=1           skip the discarded warm-up arm (do not, without a reason)
set -e
[ "${BASH_SOURCE[0]}" = "$0" ] || { echo "run me, do not source me" >&2; return 1; }
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CPUS=${1:-auto}
OUT=${2:-bench-results/phase4g}
FILTER=${3:-}
BIN=${TC_TARGET:-target}/release/tcbench
[ -x "$BIN" ] || { echo "no $BIN in $PWD -- cargo build --release -p tensorprimitives-bench" >&2; exit 1; }
FILT=()
[ -n "$FILTER" ] && FILT=(--case "$FILTER")
mkdir -p "$OUT"

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

scripts/topology.py --json "$OUT/topology.json" | tee "$OUT/topology.txt"

# Same cpuset derivation as `phase4f-threads.sh`: one logical CPU per physical
# core, on the socket contributing the most cores, ordered so that consecutive
# cores fill L3 domains one at a time. Compact placement, so the domain count is
# a known function of the thread count (A37).
if [ "$CPUS" = auto ]; then
    CPUS=$(python3 - "$OUT/topology.json" <<'PY'
import json, sys
sys.path.insert(0, "scripts")
import topology

t = json.load(open(sys.argv[1]))
allowed = set(t["allowed_cpus"])
by_sock = {k: [c for c in v if c in allowed] for k, v in t["sockets"].items()}
best = max(by_sock.items(), key=lambda kv: len(kv[1]))[1]
dom_of = {c: d["id"] for d in t["domains"] for c in d["cpus"]}
smt_primary = set()
for d in t["domains"]:
    for c in d["cpus"]:
        sibs = topology.expand(
            open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list").read().strip())
        if c == min(sibs):
            smt_primary.add(c)
print(",".join(str(c) for c in
                sorted((c for c in best if c in smt_primary),
                       key=lambda c: (dom_of.get(c, 0), c))))
PY
)
fi
NCPU=$(echo "$CPUS" | tr ',' '\n' | wc -l)

SIZES=${SIZES:-"0.25 1 4 16"}
if [ -z "${THREADS:-}" ]; then
    THREADS=""; nt=1
    while [ "$nt" -lt "$NCPU" ]; do THREADS="$THREADS $nt"; nt=$((nt * 2)); done
    THREADS="$THREADS $NCPU"
fi
TOP=$(echo $THREADS | tr ' ' '\n' | sort -n | tail -1)

echo "cpus $CPUS ($NCPU cores); sizes: $SIZES MiB; thread counts:$THREADS"

cpuset_for() { echo "$CPUS" | tr ',' '\n' | head -n "$1" | paste -sd,; }

# Repetitions per size, so a small arm is not three samples of something that
# takes microseconds. Roughly constant work per arm.
reps_for() {
    python3 -c "
import sys
s = float('$1')
print(max(3, min(200, int(round(24 / s)))))"
}

run() {  # run <tag> <threads> <dtypes> <size> <reps>
    local tag=$1 nt=$2 dt=$3 sz=$4 rp=$5 set
    set=$(cpuset_for "$nt")
    env TENSORCONTRACT_THREADS="$nt" taskset -c "$set" $BIN sweep \
        --size "$sz" --reps "$rp" "${FILT[@]}" \
        --engines planar,1m,3m --dtype "$dt" --csv "$OUT/$tag.csv" \
        > "$OUT/$tag.txt" 2>&1
    echo "$tag threads=$nt cpuset=$set size=${sz}MiB reps=$rp" >> "$OUT/placement.log"
    echo "  $(date +%H:%M:%S) $tag done"
}

# The discarded warm-up (A31/A40): the session's opening arm must not be one that
# is read. At the top thread count and the largest size, so it loads every core.
if [ -z "${NO_WARMUP:-}" ]; then
    BIG=$(echo $SIZES | tr ' ' '\n' | sort -g | tail -1)
    echo "warm-up arm (discarded): t$TOP at ${BIG} MiB"
    run "wu-discard" "$TOP" f64,c64 "$BIG" 3
fi

for pair in f64,c64 f32,c32; do
    t=${pair/,/}
    for sz in $SIZES; do
        rp=$(reps_for "$sz")
        for nt in $THREADS; do
            run "sm-s${sz}-t$nt-$t" "$nt" "$pair" "$sz" "$rp"
        done
    done
    echo "done $t"
done

echo
echo "=========================================================================="
echo "speedup against t1 AT THE SAME SIZE. The crossover is the answer."
echo "=========================================================================="
python3 - "$OUT" "$SIZES" "$THREADS" <<'PY'
import csv, math, os, sys
out, sizes, threads = sys.argv[1], sys.argv[2].split(), [int(x) for x in sys.argv[3].split()]

def load(p):
    if not os.path.exists(p):
        return None
    d = {}
    for r in csv.DictReader(open(p)):
        if r["engine"] in ("ttgt", "tblis"):
            continue
        me = r["engine"] if r["dtype"].startswith("c") else "real"
        d[(r["case"], r["dtype"], me)] = float(r["gflops"])
    return d

def gm(v):
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")

for pair, dts in (("f64c64", ("f64", "c64")), ("f32c32", ("f32", "c32"))):
    print(f"\n--- {pair}")
    print(f"{'MiB':>7}{'dtype':>7}" + "".join(f"{'t%d' % n:>8}" for n in threads if n != 1)
          + f"{'serial ms':>11}")
    for sz in sizes:
        one = load(f"{out}/sm-s{sz}-t1-{pair}.csv")
        if one is None:
            continue
        for dt in dts:
            ks = [k for k in one if k[1] == dt]
            row = []
            for n in threads:
                if n == 1:
                    continue
                a = load(f"{out}/sm-s{sz}-t{n}-{pair}.csv")
                row.append(gm([a[k] / one[k] for k in ks if a and k in a]) if a else None)
            # Median serial time per call, for reading the fixed overhead off the
            # table: a `tN` of 0.5 at a serial time of T means the overhead is ~T.
            secs = []
            for r in csv.DictReader(open(f"{out}/sm-s{sz}-t1-{pair}.csv")):
                if r["dtype"] == dt and r["engine"] not in ("ttgt", "tblis"):
                    secs.append(float(r["seconds"]) * 1e3)
            secs.sort()
            med = secs[len(secs) // 2] if secs else float("nan")
            print(f"{sz:>7}{dt:>7}"
                  + "".join(f"{v:>8.2f}" if v else f"{'-':>8}" for v in row)
                  + f"{med:>11.2f}")
print("\nA column that falls below 1.00 is a size at which threading COSTS.")
print("Compare with bench-results/*/phase4f, which is the same corpus at 64 MiB.")
PY
