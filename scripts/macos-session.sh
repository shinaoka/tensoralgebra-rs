#!/bin/bash
# The measurement session for an Apple Silicon machine.
#
#   scripts/macos-session.sh prep  OUTDIR   # the only compile
#   scripts/macos-session.sh bench OUTDIR   # the arms, from the snapshots
#
# Exists because **17 of this repository's scripts cannot run on Darwin** and the
# reasons are structural, not cosmetic: `taskset` has no macOS equivalent,
# `topology.py` dies on `os.sched_getaffinity`, `run-arms.py` samples `/proc/stat`,
# and `arch-label.sh` needed a `sysctl` arm. This is the one driver that works
# here. Do not reach for `compare-bench.sh`, `ab.sh`, `node-session.sh` or any
# `phase4*` -- each dies before measuring anything.
#
# Modelled on `phase3-bench.sh`'s structure, which is the only near-portable
# driver in the tree, with two corrections it predates:
#
#   * **It never compiles during `bench`.** `phase3-bench.sh` rebuilds between
#     measurement groups, which replaces the binary a running arm invokes (D51).
#     `prep` builds all three baseline variants once and snapshots them into
#     `OUTDIR/bin`; `bench` runs only what is there.
#   * **A discarded warm-up arm, then three *identical* arms (A31, A32).** On an
#     idle machine the opening arm runs at single-core boost and nothing else
#     does, which came back once as a uniform 4.4% "floor". Three identical arms
#     give two independent A-vs-A ratios, which is what a *spread* needs -- one
#     ratio is a point estimate and this machine's noise is the least understood
#     in the project.
#
# It also never writes flat into `bench-results/`: `phase3-bench.sh` does, and
# would overwrite the committed Phase 3 CSVs that Phase 1's headline and A35 rest
# on.
#
# ## What this machine cannot give you, and it is not a tooling gap
#
# **There is no CPU pinning on Darwin.** No `sched_setaffinity`, and
# `thread_policy_set(THREAD_AFFINITY_POLICY)` is a no-op on Apple Silicon. Every
# number in this repository is a pinned single-core measurement; nothing measured
# here is that. Add a laptop's DVFS and thermal envelope, and 12 P-cores beside 4
# E-cores the scheduler may move you between, and the honest position is: derive
# the floor in-session, quote it with every ratio, and expect it to be the worst
# in `DECISIONS.md`'s table. The compensations below are cheap and all of them
# matter; none of them is a substitute for an exclusive pinned core.
#
# The one thing that is *better* here: no SMT. The hyperthread sibling sharing
# L1d and L2 -- which invalidated two Phase 4 conclusions -- does not exist.
#
# ## The three baselines, and how to get them
#
#   brew install openblas cmake
#   export OPENBLAS_ROOT=$(brew --prefix openblas)
#
# OpenBLAS needs nothing else: its dylib carries an absolute install name, so no
# `DYLD_LIBRARY_PATH` -- which matters, because SIP strips `DYLD_*` across
# `/bin/sh` wrappers and every Linux driver here exports `LD_LIBRARY_PATH`.
# Worth knowing before quoting it as the roofline: OpenBLAS 0.3.34 is
# DYNAMIC_ARCH and selects its **neoversen1** kernel on Apple P-cores, not
# `vortex`.
#
# Accelerate needs no install at all, and is not a roofline. See `bench` below.
#
# TBLIS 2.0-dev, which was an open question on this platform and works:
#
#   git clone https://github.com/MatthewsResearchGroup/tblis.git tblis-2.0-src
#   cd tblis-2.0-src && git checkout 555320c   # the commit the Linux runs used
#   cmake -S . -B build-arm64 -DCMAKE_BUILD_TYPE=Release \
#         -DCMAKE_INSTALL_PREFIX=$PWD/../tblis-2.0-install \
#         -DBLIS_CONFIG_FAMILY=arm64 -DBLIS_THREAD_MODEL=pthread \
#         -DCMAKE_INSTALL_RPATH=$PWD/../tblis-2.0-install/lib \
#         -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
#   cmake --build build-arm64 -j 10 && cmake --install build-arm64
#
# `arm64`, not `auto`, for the reason DECISIONS.md gives for preferring `x86_64`
# on a non-AVX-512 node: it is the multi-configuration family with runtime
# dispatch. It includes BLIS's Apple `firestorm` config, and **BLIS selects
# `firestorm` on an M3 Max** -- confirm it with `BLIS_ARCH_DEBUG=1`, which prints
# `libblis: selecting sub-configuration 'firestorm'`. Verify the kernels are real
# ARM assembly and not the reference path with `nm`, never `strings` (A45):
#
#   nm -gU $TBLIS_ROOT/lib/libtblis.dylib | grep -E 'bli_[sd]gemm_armv8a'
#
# It needs only `libc++` under `pthread` threading -- no OpenMP, so build.rs
# emits no `-lomp` unless `LIBOMP_ROOT` says to. **TBLIS 1.3.0 is deliberately
# not built**: it has no complex micro-kernel outside Sandy Bridge and no ARM
# kernel, so on this machine it would be a reference path measured against a
# reference path.
set -e
[ "${BASH_SOURCE[0]}" = "$0" ] || { echo "run me, do not source me" >&2; return 1; }
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

STAGE=${1:?usage: scripts/macos-session.sh prep|bench OUTDIR [size] [reps]}
OUT=${2:?give an output directory, e.g. bench-results/$(hostname -s)-m3max}
SIZE=${3:-64}
PREMISE_SIZE=${PREMISE_SIZE:-200}
REPS=${4:-5}

[ "$(uname -s)" = "Darwin" ] || { echo "this driver is for Darwin; on Linux use compare-bench.sh" >&2; exit 1; }

BINDIR=$OUT/bin
mkdir -p "$BINDIR"

# `pgrep -fl`, not `pgrep -a`. BSD `pgrep` accepts `-a` and does NOT print the
# command line, so the guard every other driver here uses cannot filter out its
# own subshell and refuses to start on this platform. Silent, and it looks like a
# real co-tenant.
guard() {
    local others
    others=$(pgrep -fl -u "$USER" 'cargo|rustc|tcbench|kernel_shapes' \
             | grep -v -E "macos-session|pgrep|grep" || true)
    [ -n "$others" ] && { echo "REFUSING: something of yours is running:" >&2; echo "$others" >&2; exit 1; }
    return 0
}

case "$STAGE" in
prep)
    guard
    echo "building three baseline variants into $BINDIR"
    : "${OPENBLAS_ROOT:=$(brew --prefix openblas 2>/dev/null || true)}"
    [ -n "$OPENBLAS_ROOT" ] || { echo "set OPENBLAS_ROOT, or brew install openblas" >&2; exit 1; }
    export OPENBLAS_ROOT

    # OpenBLAS: the roofline, and the only baseline comparable with the Linux runs.
    cargo build --release -p tensorprimitives-bench --features blas
    cp target/release/tcbench "$BINDIR/tcbench-openblas"

    # Accelerate: reported beside it and labelled off-ISA -- see the note in
    # `bench` below and the `accelerate` feature's own.
    cargo build --release -p tensorprimitives-bench --features accelerate
    cp target/release/tcbench "$BINDIR/tcbench-accelerate"

    # TBLIS, if an install exists. Optional on purpose: it is the one baseline
    # whose availability on this platform was an open question.
    if [ -n "${TBLIS_ROOT:-}" ]; then
        cargo build --release -p tensorprimitives-bench --features tblis,blas
        cp target/release/tcbench "$BINDIR/tcbench-tblis"
    else
        echo "TBLIS_ROOT unset: skipping the TBLIS variant"
    fi
    echo "prep done. Nothing after this compiles."
    ;;

bench)
    guard
    BO=$BINDIR/tcbench-openblas
    [ -x "$BO" ] || { echo "no $BO -- run the prep stage first" >&2; exit 1; }
    LOG=$OUT/session.log
    : > "$LOG"

    # Pin every library single-threaded. `VECLIB_MAXIMUM_THREADS` is Accelerate's
    # and has no API equivalent -- it is read once at first use, so the harness
    # cannot set it on the caller's behalf.
    export TENSORCONTRACT_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
    export TBLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

    run() { echo -e "\n\n########## $* ##########\n" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; }

    # The compensations for what this platform cannot pin. Recorded rather than
    # asserted: a thermal event mid-session is the kind of thing that shows up as
    # a plausible 10% effect, and the only defence is a before/after record in
    # the same file as the numbers.
    {
        echo "session   : $(date -Iseconds)"     # -Iseconds, not -Is: BSD date rejects -Is
        echo "host      : $(hostname -s) / $(scripts/arch-label.sh)"
        echo "power     : $(pmset -g batt | head -1)"
        echo "thermal   : $(pmset -g therm | tr '\n' ' ')"
        echo "no pinning: Darwin has no CPU affinity API; see this script's header"
    } | tee -a "$LOG"

    run "$BO" info

    # --- zero-CPU structural analysis, first because it costs nothing ---------
    #
    # `orient` only. **`shapes` is deliberately absent**: it skips a case whose
    # register-block menu is empty, and `row_blocks` returns `&[]` off x86, so it
    # would write a header and no rows. `TENSORCONTRACT_ROWBLOCK` is inert here
    # for the same reason.
    run "$BO" orient --size "$SIZE" --csv "$OUT/orient.csv"

    # --- correctness on a new architecture ------------------------------------
    for m in planar 1m 3m; do
        TENSORCONTRACT_COMPLEX=$m run "$BO" verify --size 32
    done

    # --- efficiency against a same-shape GEMM ceiling -------------------------
    #
    # The most informative measurement on this machine, and the one whose
    # denominator has to be named. **OpenBLAS is the roofline**; Accelerate is a
    # separate arm because its GEMM reaches Apple's undocumented AMX matrix
    # coprocessor, which no NEON kernel can target -- not this engine's, not
    # OpenBLAS's, not TBLIS's. A3's "vendor GEMM as the achievable ceiling" is
    # therefore not the same quantity against the two, and a ratio against
    # Accelerate is an off-ISA comparison. Do not merge these two CSVs.
    run "$BO" premise --size "$PREMISE_SIZE" --reps "$REPS" \
        --engines planar,1m,3m,ttgt --dtype f64,c64 --csv "$OUT/premise-openblas-f64c64.csv"
    run "$BINDIR/tcbench-accelerate" premise --size "$PREMISE_SIZE" --reps "$REPS" \
        --engines planar,1m,3m,ttgt --dtype f64,c64 --csv "$OUT/premise-accelerate-f64c64.csv"

    # --- the corpus: warm-up, then three identical arms -----------------------
    #
    # The warm-up is discarded (A31). The three A arms are identical by
    # construction, so A-vs-A' and A-vs-A'' are this session's floor and the
    # spread between them is what says how readable a per-case ratio is here.
    ENGINES=planar,1m,3m,ttgt
    for arm in warm A A2 A3; do
        for pair in f64,c64 f32,c32; do
            t=${pair/,/}
            run "$BO" sweep --size "$SIZE" --reps "$REPS" \
                --engines "$ENGINES" --dtype "$pair" --csv "$OUT/$arm-$t.csv"
        done
    done

    # --- the only source of irregularity on this machine ----------------------
    #
    # CLAUDE.md's "42.9% of case-dtype-methods have reg_a < 1.0" is an x86
    # register-block artefact: it needs MR in {16,32,48}, none of which divides
    # 24. The scalar path's MR is 4, which *does*, so the unstressed corpus is
    # genuinely regular here and `ragged` is the only thing that is not.
    run "$BO" sweep --size "$SIZE" --reps "$REPS" --stress ragged \
        --engines planar,1m,3m --dtype c64 --csv "$OUT/ragged-c64.csv"

    if [ -x "$BINDIR/tcbench-tblis" ]; then
        run "$BINDIR/tcbench-tblis" sweep --size "$SIZE" --reps "$REPS" \
            --engines planar,tblis --dtype f64,c64 --csv "$OUT/tblis-f64c64.csv"
    fi

    {
        echo "ended     : $(date -Iseconds)"
        echo "thermal   : $(pmset -g therm | tr '\n' ' ')"
    } | tee -a "$LOG"

    # --- this session's floor, from its own repeats ---------------------------
    echo -e "\n###### floor: A' against A ######" | tee -a "$LOG"
    scripts/compare-sweeps.py "$OUT/A-f64c64.csv,$OUT/A-f32c32.csv" \
        "$OUT/A2-f64c64.csv,$OUT/A2-f32c32.csv" | tee "$OUT/floor-A2.txt"
    echo -e "\n###### floor: A'' against A ######" | tee -a "$LOG"
    scripts/compare-sweeps.py "$OUT/A-f64c64.csv,$OUT/A-f32c32.csv" \
        "$OUT/A3-f64c64.csv,$OUT/A3-f32c32.csv" | tee "$OUT/floor-A3.txt"

    echo
    echo "Quote no ratio from this session without the floor in floor-A2/A3.txt."
    echo "Do not compare any number here with a number from another machine."
    ;;

*)
    echo "unknown stage '$STAGE'; use prep or bench" >&2
    exit 1
    ;;
esac
