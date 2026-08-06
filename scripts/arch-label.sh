#!/bin/bash
# Print a short microarchitecture label for this machine, from what it reports.
#
#   scripts/arch-label.sh          # -> cascadelake | icelake | zen2 | ...
#
# A directory name under `bench-results/` is a provenance claim, and hardcoding
# one filed an Ice Lake run under a Zen2 name once already -- see
# `bench-results/worker6016-icelake/PROVENANCE.txt`. The label belongs to the
# machine, so it is read from the machine.
#
# The microarchitecture, not the instruction set, is the unit that matters here:
# A34 refuted the assumption that one measurement per ISA is enough, because
# Cascade Lake and Ice Lake disagree by up to 13% on three of eight register
# blocks with the same ISA and the same 32 registers.
#
# Two older copies of this logic exist, in `scripts/rusty-phase4.sbatch` (which
# is the one this was lifted from) and in `scripts/node-session.sh` (which is
# weaker -- it labels every EPYC `zen`). They are left alone because they have
# produced committed data; this is the authoritative one for anything new.
# Darwin has no `/proc/cpuinfo`, so the model name comes from `sysctl` there.
# Apple parts are labelled by *part*, not by core name: `m3max`, not `everest`.
# The core names are Apple's internal ones, they are not in any datasheet, and
# the P-core/E-core split means one part has two of them -- whereas the cache
# hierarchy a measurement turns on is a property of the part.
set -e
exec python3 -c "
import platform, re, subprocess
m = ''
try:
    for line in open('/proc/cpuinfo'):
        if line.startswith('model name'):
            m = line.split(':', 1)[1].strip().lower()
            break
except OSError:
    try:
        m = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                           capture_output=True, text=True,
                           check=True).stdout.strip().lower()
    except (OSError, subprocess.CalledProcessError):
        pass
# Apple Silicon first: an Intel Mac reports an Intel brand string and falls
# through to the x86 table below, which is what it should do.
apple = re.search(r'apple m([0-9]+)\s*(pro|max|ultra)?', m)
if apple:
    print('m' + apple.group(1) + (apple.group(2) or ''))
    raise SystemExit
for pat, tag in (('epyc 7[0-9]*2', 'zen2'), ('epyc 9', 'zen4'), ('epyc', 'zen'),
                 ('platinum 83', 'icelake'), ('gold 63', 'skylake'),
                 ('gold 62', 'cascadelake'), ('xeon', 'intel')):
    if re.search(pat, m):
        print(tag)
        break
else:
    print(re.sub(r'[^a-z0-9]+', '-', m)[:16] or platform.machine().lower() or 'unknown')
"
