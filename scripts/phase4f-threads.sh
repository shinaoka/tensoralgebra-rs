#!/bin/bash
# Phase 4 item 4: thread scaling.
#
# The driver partitions the output into a `pm x pn` grid of row strips of whole
# `MR` panels by column groups of whole `NR` blocks, with a shared packed-`B`
# panel (see the driver's module docs). This measures what that is worth on the
# corpus and — more usefully — *where it stops*, since the interesting output is
# not the mean speedup but the population of cases that fail to scale and why.
#
# Design notes, and why this is not just the sweep with more cores:
#
#   * Threads run on **physical cores of one socket** by default. Two threads
#     sharing a core would share the L1d and L2 that the packed `A` block is
#     sized for, which measures a different question; crossing sockets adds a
#     second L3 and a NUMA hop to a run whose packed `B` panel is sized for one
#     L3. The cross-socket arm is available (`--cross-socket`) and is **labelled
#     as a different question**, not folded into the scaling curve.
#   * Every arm runs on the same cpuset, so the 1-thread arm is *not* the
#     committed single-core number — it is the denominator of a self-consistent
#     ratio measured in the same session.
#   * `t1` runs first and last, bracketing the treatments, as everywhere else in
#     Phase 4. That pair is also **this node's noise floor** for a threaded arm,
#     and it is the only floor that applies here: the published +-1.3% geomean /
#     +-6% per case is `ccqlin038`'s.
#   * Ratios come from `scripts/compare-sweeps.py`, which already reports per-case
#     and per-dtype/method geomeans of one CSV against another. No new analysis.
#   * Occupancy over the **whole allocation** is recorded per arm, not just the
#     cores in use. At `t1` on a 128-core node that is 127 cores which must read
#     as idle; if they do not, the allocation is not exclusive and the arm is
#     void. This is the check that would have caught the two measurements Phase 4
#     had to retract.
#
# The scaling ceiling is `ceil(M / MR) * ceil(N / NR)` cells, which the corpus
# never comes close to, but the *rule* in `Plan::partition` does not always use
# every cell — splitting `N` costs each thread a packed `A` block of its own, so
# the rule declines when the columns are few. Predict that population before
# reading the curve:
#
#   scripts/thread-width.py -p 1,2,4,8,16,32,64,128 FEATURES.csv BASELINE.csv...
#
# and generate FEATURES.csv on *this* machine (`tcbench orient --csv`, free, no
# CPU): the register blocks differ by ISA, so an AVX2 node has 3-6x more row
# panels than the AVX-512 reference and a completely different partition-limited
# population. The arms that price the rule against its two extremes:
#
#   TENSORCONTRACT_PARTITION=m        1-D over `M`, i.e. the partition before 2-D
#   TENSORCONTRACT_PARTITION=n        1-D over `N`, the other extreme
#   TENSORCONTRACT_PARTITION=domain   the domain-aware gate: the shipped rule
#                                     except that the `panels >= p` early return
#                                     gives way to the column axis when the thread
#                                     set spans several L3s (A36)
#   (unset)                           the shipped rule
#
# The `domain` arm is the one this script now exists to settle. It should
# reproduce the `n` arm on the memory-bound family and the rule everywhere else,
# which is a prediction and not a hope: `scripts/partition-score-rule.py` derives
# it from the committed grids before the node is booked, and the run either meets
# it or the gate is wrong. On a one-L3-per-socket machine the arm is bit-identical
# to the rule at every thread count up to the socket, so a difference there is
# contention and nothing else — which makes Ice Lake a free null control.
#
# The sweep CSV's notes column carries `t<threads>/<pm>x<pn>`, so which partition
# ran is recoverable from the data rather than only from the run log.
#
# A **discarded warm-up arm** runs first (A31). The session's opening arm runs on
# a cold package at single-core boost and nothing else does, which came back as a
# uniform 3-5% "noise floor" on three separate nodes while inflating every ratio
# measured against it. The warm-up is at the top thread count, so it costs about
# 1/TOP of a `t1` arm and loads every core — which is the state the rest of the
# session runs in. Whether it worked is visible in the output: `t1` against `t1b`
# should now come back near 1.000 rather than 0.95-0.97.
#
# Cost: the `t1` arms dominate, so roughly (2 + sum over thread counts of 1/nt)
# arm-times per dtype pair. Nothing else may run on the node, and this one wants
# the whole socket, not one core.
#
# Usage: scripts/phase4f-threads.sh [cpuset|auto] [outdir] [size_mib] [reps] [filter]
#   THREADS="1 2 4 8"     override the thread counts
#   CROSS_SOCKET=1        add a whole-node arm at the full core count, labelled
#                         separately because it is a different question
#   NO_WARMUP=1           skip the discarded warm-up arm (do not, without a reason)
set -e
[ "${BASH_SOURCE[0]}" = "$0" ] || { echo "run me, do not source me" >&2; return 1; }
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CPUS=${1:-auto}
OUT=${2:-bench-results/phase4f}
SIZE=${3:-64}
REPS=${4:-3}
FILTER=${5:-}
BIN=./target/release/tcbench
[ -x "$BIN" ] || { echo "no $BIN in $PWD -- cargo build --release -p tensorprimitives-bench" >&2; exit 1; }
FILT=()
[ -n "$FILTER" ] && FILT=(--case "$FILTER")
mkdir -p "$OUT"

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

scripts/topology.py --json "$OUT/topology.json" | tee "$OUT/topology.txt"

# Derive the cpuset from the allocation rather than hardcoding one: one logical
# CPU per physical core, on the socket with the most cores available, and in the
# allocation's own affinity mask so this describes the job and not the node.
# `auto` is the default because a hardcoded `0-7` is a statement about
# `ccqlin038` that is false everywhere else.
if [ "$CPUS" = auto ]; then
    CPUS=$(python3 - "$OUT/topology.json" <<'PY'
import json, sys
sys.path.insert(0, "scripts")
import topology

t = json.load(open(sys.argv[1]))
allowed = set(t["allowed_cpus"])
# One primary SMT sibling per physical core, on whichever socket contributes the
# most cores to this allocation. Ordered so that consecutive cores fill L3
# domains one at a time — on Zen2 that means the first four cores are one CCX
# with a private 16 MiB L3, which is what makes a compact scaling curve
# interpretable.
primary = {min(d["cpus"]): d for d in t["domains"]}
by_sock = {}
for k, v in t["sockets"].items():
    by_sock[k] = [c for c in v if c in allowed]
best = max(by_sock.items(), key=lambda kv: len(kv[1]))[1]
dom_of = {c: d["id"] for d in t["domains"] for c in d["cpus"]}
smt_primary = set()
for d in t["domains"]:
    for c in d["cpus"]:
        sibs = topology.expand(
            open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list").read().strip())
        if c == min(sibs):
            smt_primary.add(c)
cores = sorted((c for c in best if c in smt_primary),
               key=lambda c: (dom_of.get(c, 0), c))
print(",".join(str(c) for c in cores))
PY
)
fi

# Verify the cpuset is one thread per physical core: two logical CPUs sharing a
# core would silently measure hyperthread contention instead of scaling. This
# guard is load-bearing — keep it.
NCPU=$(python3 - "$CPUS" <<'PY'
import sys
def expand(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out
cpus = expand(sys.argv[1])
seen = {}
for c in cpus:
    with open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list") as f:
        key = f.read().strip()
    seen.setdefault(key, []).append(c)
bad = {k: v for k, v in seen.items() if len(v) > 1}
print(f"cpuset {sys.argv[1]}: {len(cpus)} logical CPUs on {len(seen)} physical cores",
      file=sys.stderr)
if bad:
    print(f"REFUSING: these share a core: {bad}", file=sys.stderr)
    sys.exit(1)
print(len(cpus))
PY
)

# Thread counts: powers of two up to the cpuset size, so the same script gives
# 1/2/4/8 on the reference machine's socket and 1..64 on a cluster node.
if [ -z "${THREADS:-}" ]; then
    THREADS=""
    nt=1
    while [ "$nt" -lt "$NCPU" ]; do THREADS="$THREADS $nt"; nt=$((nt * 2)); done
    THREADS="$THREADS $NCPU"
fi
TOP=${TOP:-$(echo $THREADS | tr ' ' '\n' | sort -n | tail -1)}

echo "threads will run on cpus $CPUS ($NCPU cores); thread counts:$THREADS"
echo "partition arms at $TOP threads"

# Each arm gets exactly as many cores as it has threads, taken in order from the
# front of the list, so placement is defined rather than left to the scheduler.
# The alternative — one wide cpuset for every arm — lets `nt` threads roam over
# `NCPU` cores, and on a machine whose L3 domain is smaller than a socket that
# makes "which caches were shared" a property of the scheduler's mood. Compact
# placement instead means the sharing is a known function of `nt`: on a 4-core
# L3 domain, `t4` is one domain and `t8` is two.
cpuset_for() {  # cpuset_for <nthreads>
    echo "$CPUS" | tr ',' '\n' | head -n "$1" | paste -sd,
}

snap() { python3 -c "
import sys
out={}
for l in open('/proc/stat'):
    f=l.split()
    if f[0].startswith('cpu') and f[0][3:].isdigit():
        out[int(f[0][3:])]=[int(x) for x in f[1:]]
print(repr(out))"; }

run() {  # run <tag> <threads> <dtypes> [partition] [cpuset]
    local tag=$1 nt=$2 dt=$3 part=${4:-} set=${5:-} before after
    local pin=()
    [ -n "$part" ] && pin=(TENSORCONTRACT_PARTITION="$part")
    [ -z "$set" ] && set=$(cpuset_for "$nt")
    before=$(snap)
    env TENSORCONTRACT_THREADS="$nt" "${pin[@]}" \
        taskset -c "$set" $BIN sweep \
        --size $SIZE --reps $REPS "${FILT[@]}" \
        --engines planar,1m,3m --dtype "$dt" --csv "$OUT/$tag.csv" \
        > "$OUT/$tag.txt" 2>&1
    after=$(snap)
    echo "$tag threads=$nt cpuset=$set partition=${part:-rule}" >> "$OUT/placement.log"
    # Occupancy over the whole allocation, split into the cores this arm was
    # allowed to use and everything else. The second group is the exclusivity
    # check: on an exclusive node it must be idle, and if it is not, the arm is
    # void rather than merely noisy.
    python3 - "$set" "$OUT/topology.json" "$before" "$after" > "$OUT/$tag.cpu" <<'PY'
import ast, json, sys
def expand(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(out)
mine = set(expand(sys.argv[1]))
topo = json.load(open(sys.argv[2]))
allowed = topo["allowed_cpus"]
a, b = ast.literal_eval(sys.argv[3]), ast.literal_eval(sys.argv[4])
def busy(c):
    if c not in a or c not in b:
        return None
    d = [y - x for x, y in zip(a[c], b[c])]
    tot, idle = sum(d), d[3] + d[4]
    return 100.0 * (tot - idle) / tot if tot else 0.0
used = {c: busy(c) for c in sorted(mine)}
other = {c: busy(c) for c in allowed if c not in mine}
uv = [v for v in used.values() if v is not None]
ov = [v for v in other.values() if v is not None]
print("in-cpuset  n=%d busy min %.0f%% mean %.0f%% max %.0f%%"
      % (len(uv), min(uv), sum(uv) / len(uv), max(uv)) if uv else "in-cpuset  none")
if ov:
    hot = {c: v for c, v in other.items() if (v or 0) > 5.0}
    print("rest of allocation n=%d busy mean %.1f%% max %.1f%%"
          % (len(ov), sum(ov) / len(ov), max(ov)))
    print("NOT EXCLUSIVE: " + ", ".join(f"cpu{c}={v:.0f}%" for c, v in sorted(hot.items()))
          if hot else "exclusivity ok: nothing else in the allocation above 5%")
else:
    print("rest of allocation: empty (the cpuset is the whole allocation)")
for c, v in used.items():
    print(f"cpu{c} busy {v:.1f}%")
PY
    # Three lines, because the third is the exclusivity verdict and it is the one
    # that decides whether the arm counts at all.
    echo "  $(date +%H:%M:%S) $tag done: $(head -3 "$OUT/$tag.cpu" | tr '\n' '|')"
}

# The discarded warm-up (A31). It is deliberately the *first* thing that runs and
# deliberately never read: its only job is that some other arm is not the one that
# meets a cold package. At the top thread count it loads every core in the cpuset
# for a fraction of a `t1` arm's wall clock.
if [ -z "${NO_WARMUP:-}" ]; then
    echo "warm-up arm (discarded, A31): t$TOP"
    run "wu-t$TOP-discard" "$TOP" f64,c64
fi

for pair in f64,c64 f32,c32; do
    t=${pair/,/}
    for nt in $THREADS; do
        run "th-t$nt-$t" "$nt" "$pair"
    done
    # The two 1-D extremes at the top thread count. Only the narrow-row
    # case-dtype-methods can differ from the rule's arm at all, so everything
    # else in these is a control group — which is exactly what makes them
    # readable: a difference outside that population is contention, not the
    # partition.
    run "th-t$TOP-pm-$t" "$TOP" "$pair" m
    run "th-t$TOP-pn-$t" "$TOP" "$pair" n
    # The candidate rule, over the whole corpus, so the cases it must *not* touch
    # are measured alongside the ones it must. That is the column a ratio against
    # the rule alone cannot supply, and it is what caught a contaminated A/B in
    # part 7.
    run "th-t$TOP-dom-$t" "$TOP" "$pair" domain
    # `PARTITION_SWEEP=1` prices the axis choice *as a function of thread count*,
    # on the memory-bound family only so it is cheap. On Zen2 a 1-D `N` partition
    # beat the rule by up to 4.3x on exactly these cases (part 8b) and the
    # mechanism was not identified. Thread count is the discriminator that costs
    # nothing: at 4 threads all of them share one L3 domain on a CCX machine, at
    # the top count they span many, so if the gap opens with the number of L3
    # domains crossed rather than with the thread count itself, the mechanism is
    # cross-domain and not the partition. On a machine with one L3 per socket the
    # same sweep crosses no domains at all until it crosses a socket, which is why
    # running it on a second topology is worth more than running it again here.
    if [ -n "${PARTITION_SWEEP:-}" ]; then
        for nt in $THREADS; do
            [ "$nt" = 1 ] && continue
            for arm in m n domain; do
                FILT=(--case "${PARTITION_CASE:-abcijk}")
                run "ps-t$nt-$arm-$t" "$nt" "$pair" "$arm"
            done
            FILT=(--case "${PARTITION_CASE:-abcijk}")
            run "ps-t$nt-rule-$t" "$nt" "$pair"
        done
        FILT=()
        [ -n "$FILTER" ] && FILT=(--case "$FILTER")
    fi
    run "th-t1b-$t" 1 "$pair"   # the bracketing repeat, and this node's floor
    echo "done $t"
done

if [ -n "${CROSS_SOCKET:-}" ]; then
    # A different question, deliberately not part of the curve above: a second L3
    # and a NUMA hop, against a packed `B` panel sized for one L3. Recorded under
    # its own tag so it cannot be mistaken for a scaling point.
    ALL=$(python3 -c "
import json, sys
sys.path.insert(0, 'scripts')
import topology
t = json.load(open('$OUT/topology.json'))
def primary(c):
    p = f'/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list'
    try:
        sibs = topology.expand(open(p).read().strip())
    except OSError:
        return True
    return c == min(sibs)
print(','.join(str(c) for c in sorted(t['allowed_cpus']) if primary(c)))")
    NALL=$(echo "$ALL" | tr ',' '\n' | wc -l)
    echo "cross-socket arm: $NALL cores over all sockets"
    for pair in f64,c64 f32,c32; do
        t=${pair/,/}
        run "xs-t$NALL-$t" "$NALL" "$pair" "" "$ALL"
    done
fi

echo
echo "this node's noise floor (t1 vs its bracketing repeat):"
for pair in f64c64 f32c32; do
    echo "== $pair t1 vs t1b"
    scripts/compare-sweeps.py "$OUT/th-t1-$pair.csv" "$OUT/th-t1b-$pair.csv" | head -20
done

echo
echo "scaling, each arm against the 1-thread arm:"
for pair in f64c64 f32c32; do
    for nt in $THREADS; do
        [ "$nt" = 1 ] && continue
        echo "== $pair t$nt"
        scripts/compare-sweeps.py "$OUT/th-t1-$pair.csv" "$OUT/th-t$nt-$pair.csv" \
            | head -20
    done
done

echo
echo "partition arms, each against the rule at the same thread count:"
for pair in f64c64 f32c32; do
    for arm in pm pn dom; do
        echo "== $pair t$TOP partition=$arm"
        scripts/compare-sweeps.py "$OUT/th-t$TOP-$pair.csv" "$OUT/th-t$TOP-$arm-$pair.csv" \
            | head -20
    done
done

# And the prediction the `dom` arm above is being held to, re-derived here from
# the committed grids so the run log carries both numbers side by side. It needs
# the domain count, which `topology.py` has already printed.
if [ -f "$OUT/../features.csv" ]; then
    DOM=$(python3 -c "
import json
t = json.load(open('$OUT/topology.json'))
per = max(1, min(len(d['cpus']) for d in t['domains']))
print(max(1, -(-$TOP // per)))" 2>/dev/null || echo 1)
    echo
    echo "predicted from the committed grid (L3 domains spanned at t$TOP: $DOM):"
    scripts/partition-score-rule.py -p "$TOP" -d "$DOM" "$OUT/.." || true
fi
