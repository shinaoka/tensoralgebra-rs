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
set -e
exec python3 -c "
import re
m = ''
for line in open('/proc/cpuinfo'):
    if line.startswith('model name'):
        m = line.split(':', 1)[1].strip().lower()
        break
for pat, tag in (('epyc 7[0-9]*2', 'zen2'), ('epyc 9', 'zen4'), ('epyc', 'zen'),
                 ('platinum 83', 'icelake'), ('gold 63', 'skylake'),
                 ('gold 62', 'cascadelake'), ('xeon', 'intel')):
    if re.search(pat, m):
        print(tag)
        break
else:
    print(re.sub(r'[^a-z0-9]+', '-', m)[:16] or 'x86')
"
