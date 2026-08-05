#!/usr/bin/env python3
"""Run independent sweep arms concurrently, one per L3 domain, and record where.

A grid of 20 arms at ~22 min each is 7.5 h sequential. On a node with 32 L3
domains it can be under an hour — but only if the placement does not corrupt the
quantity being measured, and that is a hypothesis, not a convenience. This runs
the arms under an explicit placement and records enough per arm to check the
hypothesis afterwards:

  * one measurement thread per L3 domain, every other core in that domain idle;
  * SMT siblings idle (`scripts/topology.py` picks primary siblings as leaders);
  * concurrency capped below the domain count by default, because memory
    bandwidth and the interconnect stay shared however threads are placed;
  * `/proc/stat` sampled for **every core in the arm's own L3 domain** across
    exactly the arm's window, so co-tenancy is a recorded fact per arm rather
    than an assumption — the reference machine's scripts sample the pinned core
    and its sibling, and that recording is what made two earlier retractions
    detectable;
  * the wall-clock window of every arm, so the *observed* concurrency during
    each arm is recoverable after the fact and a slow arm that ran alone is
    distinguishable from one that ran against 19 others.

**This does not validate the placement, it only makes validation possible.** The
comparison that validates it is one arm run solo against the same arm run under
the placement, which is what `scripts/validate-placement.sh` does.

The job file is one arm per line, whitespace-separated:

    <tag>  <dtype-list>  [ENV=VAL ...]

Blank lines and `#` comments are ignored. Every arm gets the same `--size`,
`--reps`, `--case`, `--engines`, `--subcommand` and `--stress`, so an arm differs
from another only by its environment, which is the whole point of the
runtime-switch discipline (A15). A job file that needs a different measurement
— `premise` rather than `sweep`, or a stress mode — is a separate invocation, so
that "these arms are comparable" stays true by construction.

Usage:
    scripts/run-arms.py --outdir DIR --jobs N [--sequential] JOBFILE
"""
import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import topology  # noqa: E402


def proc_stat():
    """cpu -> jiffy counters, for every cpu the kernel reports."""
    out = {}
    with open("/proc/stat") as f:
        for line in f:
            fields = line.split()
            if fields[0].startswith("cpu") and fields[0][3:].isdigit():
                out[int(fields[0][3:])] = [int(x) for x in fields[1:]]
    return out


def busy_pct(before, after, cpu):
    a, b = before.get(cpu), after.get(cpu)
    if not a or not b:
        return None
    d = [y - x for x, y in zip(a, b)]
    total = sum(d)
    idle = d[3] + d[4]  # idle + iowait
    return 100.0 * (total - idle) / total if total else 0.0


def parse_jobs(path):
    jobs = []
    with open(path) as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) < 2:
                sys.exit(f"bad job line (need at least tag and dtypes): {raw!r}")
            jobs.append({"tag": fields[0], "dtype": fields[1], "env": fields[2:]})
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("jobfile")
    ap.add_argument("--outdir", required=True)
    # `TC_TARGET` gives a cluster job its own build tree, so an editing session
    # on the submit host cannot relink the binary a running arm invokes. See
    # `scripts/README.md`. Unset means the ordinary `target/`.
    ap.add_argument(
        "--bin",
        default=os.path.join(os.environ.get("TC_TARGET", "target"), "release", "tcbench"),
    )
    ap.add_argument("--size", default="64")
    ap.add_argument("--reps", default="3")
    ap.add_argument("--case", default="")
    ap.add_argument("--engines", default="planar,1m,3m")
    ap.add_argument("--subcommand", default="sweep", choices=("sweep", "premise"),
                    help="which tcbench measurement to run for every arm in this "
                         "invocation (default sweep). `premise` takes the same "
                         "options and is the efficiency-against-a-GEMM-ceiling "
                         "measurement, so it needs the same occupancy record")
    ap.add_argument("--stress", default="",
                    help="none|ragged|padded, passed straight through; the only "
                         "way to exercise the block-scatter gather path")
    ap.add_argument("--jobs", type=int, help="concurrent arms (default: 75%% of L3 domains)")
    ap.add_argument("--sequential", action="store_true",
                    help="one arm at a time on a single domain; the fallback regime")
    ap.add_argument("--cpus", metavar="LIST",
                    help="place arms on exactly these cpus instead of chosen domain leaders; "
                         "`--cpus 4` reproduces the reference machine's single-pinned-core "
                         "regime, occupancy recording included")
    ap.add_argument("--fake", metavar="CMD",
                    help="run CMD instead of tcbench, to test placement without measuring")
    ap.add_argument("--deadline", type=float, metavar="EPOCH",
                    help="stop starting new arms after this unix time; arms already "
                         "running are allowed to finish. Skipped arms are reported "
                         "loudly, never silently dropped")
    a = ap.parse_args()

    if not a.fake and not os.access(a.bin, os.X_OK):
        sys.exit(f"no {a.bin} -- cargo build --release -p tensorprimitives-bench")

    os.makedirs(a.outdir, exist_ok=True)
    topo = topology.discover()
    jobs = parse_jobs(a.jobfile)
    if a.cpus:
        # Explicit placement. Each cpu still carries its whole L3 domain, so the
        # occupancy record is the same shape as in the chosen-leader case.
        want_cpus = topology.expand(a.cpus)
        allowed = set(topo["allowed_cpus"])
        missing = [c for c in want_cpus if c not in allowed]
        if missing:
            sys.exit(f"cpus {missing} are not in this allocation "
                     f"({topology.compress(sorted(allowed))})")
        by_cpu = {c: d for d in topo["domains"] for c in d["cpus"]}
        slots = [{"domain": by_cpu[c]["id"], "cpu": c, "numa": by_cpu[c]["numa"],
                  "domain_cpus": by_cpu[c].get("all_cpus") or by_cpu[c]["cpus"]}
                 for c in want_cpus]
        if a.sequential:
            slots = slots[:1]
        topo["placement"] = {"concurrency": len(slots),
                             "domains_available": len(topo["domains"]),
                             "slots": slots, "explicit": True}
    else:
        want = 1 if a.sequential else a.jobs
        topo["placement"] = topology.plan(topo, want)
        slots = topo["placement"]["slots"]
    with open(os.path.join(a.outdir, "topology.json"), "w") as f:
        json.dump(topo, f, indent=2)

    print(f"host {topo['host']}  {topo['physical_cores']} physical cores, "
          f"SMT width {topo['smt_width']}, {len(topo['domains'])} L"
          f"{topo['cache_level_for_domains']} domains")
    print(f"{len(jobs)} arms, {len(slots)} concurrent, "
          f"cpus {topology.compress(s['cpu'] for s in slots)}")
    if topo["smt_enabled"]:
        print("NOTE: SMT is on. Leaders are primary siblings and their siblings are "
              "left idle, but nothing prevents the OS from using them.")
    sys.stdout.flush()

    pending = queue.Queue()
    for j in jobs:
        pending.put(j)
    free = queue.Queue()
    for s in slots:
        free.put(s)

    results = []
    lock = threading.Lock()
    t0 = time.time()

    def run_one(job, slot):
        cpu = slot["cpu"]
        dom = slot["domain_cpus"]
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        for assign in job["env"]:
            k, _, v = assign.partition("=")
            env[k] = v
        csv = os.path.join(a.outdir, f"{job['tag']}.csv")
        if a.fake:
            cmd = ["taskset", "-c", str(cpu)] + a.fake.split()
        else:
            cmd = ["taskset", "-c", str(cpu), a.bin, a.subcommand,
                   "--size", a.size, "--reps", a.reps,
                   "--engines", a.engines, "--dtype", job["dtype"], "--csv", csv]
            if a.case:
                cmd += ["--case", a.case]
            if a.stress:
                cmd += ["--stress", a.stress]
        before, start = proc_stat(), time.time()
        with open(os.path.join(a.outdir, f"{job['tag']}.txt"), "w") as log:
            rc = subprocess.call(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        after, end = proc_stat(), time.time()

        occ = {c: busy_pct(before, after, c) for c in dom}
        with open(os.path.join(a.outdir, f"{job['tag']}.cpu"), "w") as f:
            f.write(f"# arm {job['tag']} on cpu{cpu}, L{topo['cache_level_for_domains']} "
                    f"domain {slot['domain']} = cpus {topology.compress(dom)}, "
                    f"numa {slot['numa']}\n")
            f.write(f"# window {end - start:.1f}s; env {' '.join(job['env']) or '(none)'}\n")
            f.write(f"# tcbench {a.subcommand}"
                    + (f" --stress {a.stress}" if a.stress else "")
                    + f" --engines {a.engines} --size {a.size} --reps {a.reps}\n")
            for c in sorted(occ):
                mark = " <- measurement thread" if c == cpu else ""
                f.write(f"cpu{c} busy {occ[c]:.1f}%{mark}\n")
            others = [f"cpu{c}={occ[c]:.1f}%" for c in sorted(occ)
                      if c != cpu and (occ[c] or 0) > 5.0]
            f.write("# domain co-tenancy: "
                    + (", ".join(others) if others else "none above 5%") + "\n")
        rec = dict(tag=job["tag"], dtype=job["dtype"], env=job["env"], cpu=cpu,
                   subcommand=a.subcommand, stress=a.stress or None,
                   engines=a.engines, size=a.size, reps=a.reps,
                   domain=slot["domain"], numa=slot["numa"], rc=rc,
                   start=start, end=end, seconds=end - start,
                   own_busy=occ.get(cpu), domain_busy=occ)
        with lock:
            results.append(rec)
            co = ", ".join(f"cpu{c}={occ[c]:.0f}%" for c in sorted(occ)
                           if c != cpu and (occ[c] or 0) > 5.0)
            print(f"  {time.strftime('%H:%M:%S')} {job['tag']:<16} cpu{cpu:<4} "
                  f"dom{slot['domain']:<3} {end - start:7.1f}s  rc={rc}  "
                  f"self {occ.get(cpu) or 0:.0f}%  "
                  f"co-tenants: {co or 'none'}")
            sys.stdout.flush()

    skipped = []

    def worker():
        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            if a.deadline and time.time() >= a.deadline:
                with lock:
                    skipped.append(job["tag"])
                pending.task_done()
                continue
            slot = free.get()
            try:
                run_one(job, slot)
            finally:
                free.put(slot)
                pending.task_done()

    threads = [threading.Thread(target=worker) for _ in slots]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Observed concurrency per arm: how many other arms overlapped its window.
    for r in results:
        r["overlap"] = sum(
            1 for o in results
            if o is not r and o["start"] < r["end"] and r["start"] < o["end"]
        )
    results.sort(key=lambda r: r["start"])
    with open(os.path.join(a.outdir, "arms.json"), "w") as f:
        json.dump({"wall_seconds": time.time() - t0, "arms": results}, f, indent=2)
    # Also append to a cumulative record, because an output directory usually
    # holds several invocations (a solo arm, then the same arm placed) and
    # `arms.json` describes only the last of them. The placement of an arm is
    # part of its provenance, so losing it to a later run is losing data.
    with open(os.path.join(a.outdir, "arms.jsonl"), "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    bad = [r for r in results if r["rc"] != 0]
    print()
    print(f"{len(results)} arms in {(time.time() - t0) / 60:.1f} min wall "
          f"(sum of arm times {sum(r['seconds'] for r in results) / 60:.1f} min)")
    if results:
        print("observed overlap per arm: "
              f"min {min(r['overlap'] for r in results)}, "
              f"max {max(r['overlap'] for r in results)}")
    if skipped:
        # A bounded run that does not say what it dropped reads as complete
        # coverage. Say it, in both places anyone will look.
        print(f"DEADLINE: {len(skipped)} arms never started: {' '.join(skipped)}")
        with open(os.path.join(a.outdir, "skipped-arms.txt"), "w") as f:
            f.write("\n".join(skipped) + "\n")
    if bad:
        print(f"FAILED arms: {[r['tag'] for r in bad]}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
