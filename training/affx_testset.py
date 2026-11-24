#!/usr/bin/env python
"""
Test-set predictions and per-population errors of the paper's models, for
figures made without Perlmutter: for each model seed, the paper's eight
metrics (paper_errors.py) of every test population; for seed 0, also the true
and decoded diagnostics of every test population (456 values per particle,
physical units, float32) and the test populations' n_tot, scenario and time.

Usage (in $SCRATCH/aeromeld/affx, with the pytorch module):
    [AFFX_CACHE=cache_optfix.npz] python affx_testset.py OUT_DIR NAME=RUN_PREFIX[,RUN_PREFIX...] ...
"""
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_errors import paper_metrics, ROWS                         # noqa: E402

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
c = np.load(os.environ.get('AFFX_CACHE', '/pscratch/sd/m/mwest/aeromeld/affx/cache.npz'))   # AFFX_CACHE: other targets
X, ntot, i_snr = c['X'], c['ntot'], c['i_snr']
splits = np.load('/pscratch/sd/m/mwest/aeromeld/models/test_idx.npz')
import importlib.util                                                 # the decoder of affx_mixk.py, without its CLI
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'affx_mixk.py')).read()
start = src.index('BLOCKS = [')
ns = {}
exec('import numpy as np\nimport torch\n' + src[start:src.index('rows = []')].replace(
    "c = np.load(args.cache)\nX, ntot = c['X'], c['ntot']\nsplits = np.load(args.splits)\n", ''), ns)
units, decoder = ns['units'], ns['decoder']
GEO = ns['GEO']
for arg in sys.argv[2:]:
    name, prefixes = arg.split('=', 1)
    for run in prefixes.split(','):
        for f in sorted(glob.glob(f'{run}_s*.pt')):
            m = re.fullmatch(re.escape(run) + r'_s(\d+)\.pt', f)
            if not m:
                continue
            si = int(m.group(1))
            tst = np.sort(splits['idx'][si])
            Z = np.load(f'{run}_s{si}_Z.npy').astype(np.float64)
            Xd = decoder(run, si)(Z[tst])
            e = {ROWS[k]: v for k, v in paper_metrics(units(X[tst], ntot[tst]), units(Xd, ntot[tst])).items()}
            res = dict(test_idx=tst, scenario=i_snr[tst], **{'err_' + k: np.asarray(e[k], np.float32) for k in GEO})
            if si == 0:
                res.update(X_dec=Xd.astype(np.float32))
                truth = os.path.join(out, 'truth_s0.npz')
                if not os.path.exists(truth):
                    np.savez_compressed(truth, test_idx=tst, X=X[tst].astype(np.float32), ntot=ntot[tst],
                                        scenario=i_snr[tst])
            np.savez_compressed(os.path.join(out, f'{name}.{si}.npz'), **res)
            print(name, si, 'geo %.2f%%' % (100 * np.exp(np.mean([np.log(np.mean(e[k])) for k in GEO]))), flush=True)
