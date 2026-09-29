#!/usr/bin/env python3
import gzip
import json
import re
from pathlib import Path

root = Path(__file__).parent
cases = ['copy_contiguous', 'copy_transpose', 'copy_lm', 'copy_small', 'copy_negative', 'copy_rank6']
def read(p):
    return p.read_text() if p.exists() else gzip.open(str(p) + '.gz', 'rt').read()
report = {}
for stage in ['initial', 'final']:
    rows = {}
    for case in cases:
        values = []
        for mode in [1, 0]:
            prefix = root / stage / f'map{mode}-{case}'
            log = read(Path(str(prefix) + '.log'))
            assert f'CHECK {case} passed; threads=1 policy=Sequential samples=3' in log
            profile = read(Path(str(prefix) + '.callgrind'))
            assert 'events: Ir\n' in profile
            values.append(int(re.search(r'^summary: (\d+)$', profile, re.M)[1]) / 3)
        rows[case] = dict(map_Ir=values[0], copy_Ir=values[1], reduction_percent=100*(1-values[1]/values[0]))
    report[stage] = rows
(root / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
