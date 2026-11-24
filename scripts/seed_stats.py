#!/usr/bin/env python
"""Spectrum errors and their attribution for every seed of the paper's models.

For each run of the registry sets <model>_seeds (training/README.md: the
nonlinear twins and the affine AX-rel1, ten seeds each, each run in
WRF-AeroMELD), over the mixed cells at the final instant:

  smape    the spectrum error (analysis.spectrum_error), mean over mixed cells, %
  terms    mean and RMS over mixed cells of each attribution term
           (paperfigs.mixing: mixing, transport, endmember, sampling, number)
           and of the total log10 ratio

Each entry's 'transport' is 'online' (WRF-AeroMELD) or the offline replay's
scheme (the run key's suffix, e.g. 'sep'). For replay runs, 'spectra_exact'
holds the same statistics against the exact answer (per-source transport of the
sources' particle-resolved diagnostics; paperfigs.transport_schemes), with the
decoding term split into 'offmanifold' and 'mixing'.

Products are built as needed (seconds per run from the run extracts). Writes
data/seed_stats.json (committed), which the figures that show seed spread read.

Usage (from scripts/):  python seed_stats.py [--sets twin_k16_seeds ...]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paperfigs import config, mixing, registry  # noqa: E402

SETS = ('twin_none_seeds', 'twin_pairs_seeds', 'twin_k16_seeds', 'twin_k32_seeds', 'ax_rel1_seeds', 'ax_k16_seeds',
        'twin_none_of_seeds', 'twin_pairs_of_seeds', 'twin_k16_of_seeds', 'twin_k32_of_seeds', 'ax_rel1_of_seeds',
        'ax_k16_of_seeds',
        'twin_none_of_seeds_sep', 'twin_pairs_of_seeds_sep', 'twin_k16_of_seeds_sep', 'twin_k32_of_seeds_sep',
        'ax_rel1_of_seeds_sep')
OUT = config.DATA_DIR / 'seed_stats.json'


#: Run-key suffix of an offline replay run -> its scheme key in the replay (transport/schemes.py).
REPLAY_SCHEMES = {'donor': 'donor', 'shB': 'shB:{m}', 'sep': 'sep:{m}', 'persource': 'oracle'}


def transport_of(run):
    suffix = run.key.rsplit('_', 1)[1]
    return suffix if suffix in REPLAY_SCHEMES else 'online'


def run_stats(run, m=None):
    m = m or mixing.compute(run)
    mixed = m['mixed']
    out = {}
    for sp in m['spectra']:
        d = dict(name=sp['name'], span=sp['span'], smape=100 * float(np.nanmean(sp['error'][mixed])))
        for t, v in list(sp['terms'].items()) + [('total', sp['total'])]:
            x = v[mixed]
            x = x[np.isfinite(x)]
            d['mean_' + t] = float(x.mean()) if x.size else 0.0
            d['rms_' + t] = float(np.sqrt((x ** 2).mean())) if x.size else 0.0
        out[sp['group']] = d
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--sets', nargs='+', default=list(SETS))
    args = ap.parse_args()
    reg = registry.load()
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for s in args.sets:
        for run in reg.select(set_name=s):
            tr = transport_of(run)
            res[run.key] = dict(model=run.model.key, name=run.model.key.split('.')[0], transport=tr,
                                seed=run.model.seed_idx, label=run.model.label, spectra=run_stats(run))
            if tr != 'online':
                from paperfigs import transport_schemes as T
                key = REPLAY_SCHEMES[tr].format(m=T.run_model(run.model.key))
                ex = T.evaluate_exact(run.model.key, Path(run.wrf_dir).name, keys=[key])[key]
                res[run.key]['spectra_exact'] = run_stats(run, ex)
            sp = res[run.key]['spectra']
            print('%-22s ' % run.key + '  '.join('%s %5.1f' % (g[:4], d['smape']) for g, d in sp.items()),
                  flush=True)
    OUT.write_text(json.dumps(res, indent=1, sort_keys=True))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
