#!/usr/bin/env python3
"""What machine is this, and where may a measurement thread be placed?

Every performance number in `docs/notebook/` is single-core on `ccqlin038`, whose
topology is written down by hand in the Environment section. Nothing about that
description transfers to a cluster node, and the *placement* of concurrent arms
on a many-core node is a hypothesis that has to be stated before it can be
validated. This prints the machine description that hypothesis rests on, and
emits it as JSON so the runner and the report read the same facts.

Three things it reports and why each one matters here:

  * **L3 domains.** The packed `B` panel is sized for an L3 (`NC`), so two arms
    sharing an L3 perturb exactly the quantity the blocking grid measures. On
    Cascade Lake one L3 is a whole 8-core socket; on Zen2 it is a 4-core CCX,
    which means one arm per CCX gets a *private* L3 — better isolation than the
    reference machine ever had. The count of domains is therefore the ceiling on
    concurrent arms, and it is a property of the node, not of the schedule.
  * **SMT siblings.** The pinned core's sibling shares L1d and L2, which is what
    every cache-blocking measurement in this project turns on and what forced
    two Phase 4 retractions. Rusty's CPU nodes report `2:64:1` etc. in `sinfo`,
    i.e. SMT off, so on those the concern is absent rather than handled — but
    that is a fact to check on the node, not to assume from `sinfo`.
  * **NUMA nodes.** Memory bandwidth and the interconnect stay shared however
    the threads are placed, so concurrent arms are spread across NUMA nodes
    rather than packed into one, and the spread is recorded.

It reports the *allowed* CPU set (`sched_getaffinity`), not every CPU the kernel
can see, so inside a Slurm allocation it describes the allocation and not the
node. That is deliberate: it is also how the measurement scripts derive their
cpuset instead of hardcoding one.

Usage:
    scripts/topology.py                     # human summary
    scripts/topology.py --json topo.json    # ... and machine-readable
    scripts/topology.py --jobs 20           # plan placement for 20 arms
"""
import argparse
import json
import os
import sys
from collections import OrderedDict, defaultdict

SYS_CPU = "/sys/devices/system/cpu"
SYS_NODE = "/sys/devices/system/node"


def read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def expand(spec):
    """Expand a Linux cpu-list ("0-3,8,12-13") into a sorted list of ints."""
    out = []
    if not spec:
        return out
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def compress(cpus):
    """Inverse of expand, for printing: [0,1,2,5] -> "0-2,5"."""
    cpus = sorted(cpus)
    runs, i = [], 0
    while i < len(cpus):
        j = i
        while j + 1 < len(cpus) and cpus[j + 1] == cpus[j] + 1:
            j += 1
        runs.append(str(cpus[i]) if i == j else f"{cpus[i]}-{cpus[j]}")
        i = j + 1
    return ",".join(runs)


def caches(cpu):
    """Cache descriptors for one CPU: level -> dict, from sysfs."""
    out = {}
    d = f"{SYS_CPU}/cpu{cpu}/cache"
    if not os.path.isdir(d):
        return out
    for idx in sorted(os.listdir(d)):
        if not idx.startswith("index"):
            continue
        lvl = read(f"{d}/{idx}/level")
        typ = read(f"{d}/{idx}/type")
        if lvl is None or typ == "Instruction":
            continue
        out[int(lvl)] = {
            "type": typ,
            "size": read(f"{d}/{idx}/size"),
            "ways": read(f"{d}/{idx}/ways_of_associativity"),
            "shared_cpu_list": read(f"{d}/{idx}/shared_cpu_list"),
        }
    return out


def discover():
    allowed = sorted(os.sched_getaffinity(0))
    online = expand(read(f"{SYS_CPU}/online")) or allowed

    # SMT: group the allowed CPUs by their thread-sibling set.
    sibs = {}
    for c in allowed:
        s = read(f"{SYS_CPU}/cpu{c}/topology/thread_siblings_list")
        sibs[c] = expand(s) if s else [c]
    cores = OrderedDict()  # sibling-set key -> allowed CPUs in it
    for c in allowed:
        cores.setdefault(tuple(sibs[c]), []).append(c)
    smt_width = max((len(k) for k in cores), default=1)

    # L3 domains, keyed by the cache's shared_cpu_list. Fall back to L2 if the
    # node reports no L3 at all (the analytical-model paper's own machines
    # include such parts, so this is not hypothetical).
    per_cpu_cache = {c: caches(c) for c in allowed}
    lvl = 3 if any(3 in v for v in per_cpu_cache.values()) else 2
    doms = OrderedDict()
    for c in allowed:
        info = per_cpu_cache[c].get(lvl)
        key = info["shared_cpu_list"] if info else "unknown"
        doms.setdefault(key, {"level": lvl, "info": info, "cpus": [], "all_cpus": expand(key)})
        doms[key]["cpus"].append(c)

    numa = OrderedDict()
    if os.path.isdir(SYS_NODE):
        for n in sorted(
            (d for d in os.listdir(SYS_NODE) if d.startswith("node")),
            key=lambda d: int(d[4:]),
        ):
            cl = expand(read(f"{SYS_NODE}/{n}/cpulist"))
            mine = [c for c in cl if c in set(allowed)]
            if mine:
                numa[int(n[4:])] = mine
    pkg = defaultdict(list)
    for c in allowed:
        p = read(f"{SYS_CPU}/cpu{c}/topology/physical_package_id")
        pkg[int(p) if p is not None else 0].append(c)

    cpu_to_numa = {}
    for n, cs in numa.items():
        for c in cs:
            cpu_to_numa[c] = n

    domains = []
    for i, (key, d) in enumerate(doms.items()):
        leaders = [c for c in d["cpus"] if c == min(sibs[c])]
        domains.append(
            {
                "id": i,
                "level": d["level"],
                # `cpus` is the allowed subset and drives *placement*; `all_cpus` is
                # the whole physical domain from sysfs and drives *occupancy*. They
                # differ whenever this runs under a taskset or a narrow cgroup, and
                # conflating them silently reduces co-tenancy recording to the
                # measurement core alone — which is the one thing that recording
                # exists to catch.
                "cpus": d["cpus"],
                "all_cpus": d["all_cpus"],
                "shared_cpu_list": key,
                "cpus_in_node": len(d["all_cpus"]),
                "size": (d["info"] or {}).get("size"),
                "ways": (d["info"] or {}).get("ways"),
                "leader": leaders[0] if leaders else d["cpus"][0],
                "numa": cpu_to_numa.get(d["cpus"][0]),
            }
        )

    # The CPU model string. Recorded because it was missing when it was wanted:
    # `bench-results/worker6156-icelake` had cores, caches, NUMA and SMT width but
    # nothing that named the part, so a README could not say which CPU produced
    # its headline table without guessing. A provenance file that cannot name the
    # CPU is not provenance.
    model = ""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass

    return {
        "host": read("/proc/sys/kernel/hostname") or os.uname().nodename,
        "cpu_model": model,
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
        "online_cpus": len(online),
        "allowed_cpus": allowed,
        "allowed_count": len(allowed),
        "physical_cores": len(cores),
        "smt_width": smt_width,
        "smt_enabled": smt_width > 1,
        "sockets": {str(k): v for k, v in sorted(pkg.items())},
        "numa": {str(k): v for k, v in numa.items()},
        "cache_level_for_domains": lvl,
        "l1": per_cpu_cache[allowed[0]].get(1) if allowed else None,
        "l2": per_cpu_cache[allowed[0]].get(2) if allowed else None,
        "l3": per_cpu_cache[allowed[0]].get(3) if allowed else None,
        "domains": domains,
    }


def plan(topo, jobs=None, fraction=0.75):
    """Choose which core each concurrent arm runs on.

    One arm per L3 domain, every other core in that domain idle, SMT siblings
    idle. Deliberately *not* every domain: memory bandwidth and the
    interconnect stay shared no matter how the threads are placed, so the
    default leaves a quarter of the domains empty and spreads the rest evenly
    over the NUMA nodes. `jobs` asks for a specific concurrency and is clamped
    to the number of domains.
    """
    doms = topo["domains"]
    cap = len(doms)
    want = cap if jobs is None else int(jobs)
    if jobs is None:
        want = max(1, int(cap * fraction))
    want = max(1, min(want, cap))

    # Round-robin over NUMA nodes so a partial round is spread, not packed.
    by_numa = OrderedDict()
    for d in doms:
        by_numa.setdefault(d["numa"], []).append(d)
    order, i = [], 0
    while len(order) < cap:
        added = False
        for n in by_numa:
            if i < len(by_numa[n]):
                order.append(by_numa[n][i])
                added = True
        if not added:
            break
        i += 1
    chosen = order[:want]
    return {
        "concurrency": len(chosen),
        "domains_available": cap,
        "slots": [
            {
                "domain": d["id"],
                "cpu": d["leader"],
                "numa": d["numa"],
                # The whole physical domain, so occupancy records co-tenants even
                # when this process's own affinity is narrower (see `all_cpus`).
                "domain_cpus": d.get("all_cpus") or d["cpus"],
            }
            for d in chosen
        ],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", metavar="PATH", help="write the full description here")
    ap.add_argument("--jobs", type=int, help="plan placement for this many concurrent arms")
    ap.add_argument("--fraction", type=float, default=0.75,
                    help="fraction of L3 domains to fill when --jobs is absent (default 0.75)")
    ap.add_argument("--quiet", action="store_true", help="only write the JSON")
    a = ap.parse_args()

    topo = discover()
    topo["placement"] = plan(topo, a.jobs, a.fraction)

    if not a.quiet:
        t = topo
        print(f"host            : {t['host']}")
        if t.get("cpu_model"):
            print(f"cpu             : {t['cpu_model']}")
        if t["slurm_job"]:
            print(f"slurm           : job {t['slurm_job']} on {t['slurm_nodelist']}")
        print(f"cpus allowed    : {t['allowed_count']} of {t['online_cpus']} online "
              f"({compress(t['allowed_cpus'])})")
        print(f"physical cores  : {t['physical_cores']}   SMT width {t['smt_width']}"
              f"{'' if t['smt_enabled'] else '  (SMT off: no sibling to contend with)'}")
        print(f"sockets         : " + "  ".join(
            f"{k}:{len(v)}c" for k, v in t["sockets"].items()))
        print(f"numa nodes      : " + "  ".join(
            f"{k}:{len(v)}c" for k, v in t["numa"].items()))
        for lv in (1, 2, 3):
            c = t[f"l{lv}"]
            if c:
                print(f"L{lv}              : {c['size']} {c['ways']}-way, "
                      f"shared by {len(expand(c['shared_cpu_list']))} logical cpus")
        d0 = t["domains"][0]
        print(f"L{t['cache_level_for_domains']} domains      : {len(t['domains'])}, "
              f"{d0['cpus_in_node']} cpus each ({d0['size']} per domain)")
        p = t["placement"]
        print()
        print(f"placement       : {p['concurrency']} concurrent arms of "
              f"{p['domains_available']} domains, one core per domain, "
              f"rest of each domain idle")
        print("                  cpus " + compress(s["cpu"] for s in p["slots"]))
        per_numa = defaultdict(int)
        for s in p["slots"]:
            per_numa[s["numa"]] += 1
        print("                  per numa node: " + "  ".join(
            f"{k}:{v}" for k, v in sorted(per_numa.items(), key=lambda kv: (kv[0] is None, kv[0]))))

    if a.json:
        with open(a.json, "w") as f:
            json.dump(topo, f, indent=2)
        if not a.quiet:
            print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
