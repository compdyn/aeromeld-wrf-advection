#!/usr/bin/env python
"""
Write build_run.py latent directories for our models from the source latents
encoded by encode_sources.sh (affx_wrf3_npz.py on the three 1000-particle
source files): WORK/latents/<fpidx>/latent_scenario_<s>_0001_00000013.txt, the 9
latents z (particle means, z >= 0) and then a placeholder 0 for the ntot latent
(WRF never reads it; the input builders set ntot from the scenario values).

fpidx = <model>.<seed>.0, e.g. twin_k16.0.0. The golden test model
05_nonlinvlong.0.0 gets its stored latents from reference_inputs.json, so the
build needs neither partnn nor its archive.

Usage: python write_latents.py WORK ZSRC_DIR REFERENCE_JSON
"""
import json
import sys
from pathlib import Path

import numpy as np

work, zdir, refjson = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
SCEN = (561, 635, 812)
NAMES = dict(k16='twin_k16', k32='twin_k32', pairs='twin_pairs', none='twin_none', mix0='twin_mix0', ax='ax_rel1')


def write(fpidx, rows):
    d = work / 'latents' / fpidx
    if d.exists():
        print('exists', d); return
    d.mkdir(parents=True)
    for s, v in zip(SCEN, rows):
        (d / ('latent_scenario_%d_0001_%08d.txt' % (s, 13))).write_text(''.join('%.9e\n' % x for x in v))
    print('wrote', d)


ref = json.load(open(refjson))['latents']['05_nonlinvlong.0.0']
write('05_nonlinvlong.0.0', [list(ref[str(s)]) + [0.0] for s in SCEN])
for f in sorted(zdir.glob('zsrc_s*.npz')):
    si = int(f.stem.split('_s')[1])
    z = np.load(f)
    pops = (z["pops"] // 25).tolist()                 # dataset index = 25 scenario + output; output 12 = file 13
    assert pops == list(SCEN), pops
    for key, name in NAMES.items():
        if key + '_z' in z.files:
            zz = z[key + '_z']
            assert zz.shape == (3, 9) and (zz >= 0).all(), (key, zz)
            write('%s.%d.0' % (name, si), [list(r) + [0.0] for r in zz])
