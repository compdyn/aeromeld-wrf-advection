#!/usr/bin/env python
"""
Merge the per-seed results of affx.py runs that differ only in their seeds
(e.g. seeds 0-2, 3-6 and 7-9 run as separate jobs) into one results json, with
the mean over all their seeds, for affx_table.py; the model files stay where
they are (the json records them in 'parts').

Usage: python affx_merge.py OUT_DIR NAME RUN.json [RUN.json ...]
"""
import json
import sys

import numpy as np

out_dir, name, files = sys.argv[1], sys.argv[2], sys.argv[3:]
runs = [json.load(open(f)) for f in files]
merged = dict(args=dict(runs[0]['args'], name=name), seeds=dict(), parts=files,
              wall_s=sum(r['wall_s'] for r in runs))
for r in runs:
    for si, v in r['seeds'].items():
        assert si not in merged['seeds'], f'seed {si} twice'
        merged['seeds'][si] = v
merged['mean'] = dict()
for part in ['test', 'mix16']:
    keys = next(iter(merged['seeds'].values()))[part].keys()
    merged['mean'][part] = {k: float(np.mean([v[part][k] for v in merged['seeds'].values()])) for k in keys}
with open(f'{out_dir}/{name}.json', 'w') as fp:
    json.dump(merged, fp, indent=1)
print(f'{name}: {len(merged["seeds"])} seeds, test geo {100 * merged["mean"]["test"]["geo"]:.2f}%')
