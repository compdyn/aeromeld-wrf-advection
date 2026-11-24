#!/usr/bin/env python
"""Run extracts (data/wrfml_runs/<run>.npz) from an offline transport replay.

Same format as training/wrf/extract_runs.py, so a replay is an ordinary affx run
for the paper's products and figures: the surface latents z and ntot at the
start and at 04 UTC, the three sampling points at every 15-minute output, the
surface at every output (z_surf, ntot_surf), and the source latents. One
extract per (model, scheme); run key <model key>_<scheme>, e.g.
twin_k16_of.0.0_sep.

Schemes (transport/schemes.py): donor, shB, sep, and persource (the oracle:
the source latents mixed by per-source transport, i.e. AeroMELD with the
reference transport).

Usage (from the repo root): python transport/extract_replay.py [--replay r3]
       [--models twin_k16_of.0.0,...] [--schemes donor,shB,sep,persource] [--tag 12h]
The paper's main run since 2026-11-13: --replay all1_12h --models twin_k16_of.1.0 --schemes sep --tag 12h
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))

from paperfigs import config, products  # noqa: E402
from paperfigs import transport_schemes as T  # noqa: E402

#: run-key suffix -> scheme key template in the replay
SCHEMES = {'donor': 'donor', 'shB': 'shB:{m}', 'sep': 'sep:{m}', 'persource': 'oracle'}
POINTS = dict(zip(config.LOC_LABELS, config.LOCATIONS))      # WRF 1-based (i, j) (config.LOCATIONS)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--replay', default='r3')
    ap.add_argument('--models', default=','.join(T.MODELS))
    ap.add_argument('--schemes', default=','.join(SCHEMES))
    ap.add_argument('--tag', default='', help='appended to the run key, e.g. 12h: twin_k16_of.1.0_sep12h')
    args = ap.parse_args()
    s = T.load(args.replay)
    t_s = s['time_s']
    times = np.array([config.TIMESTAMP_START[:11] + '%02d:%02d:00' % (t // 3600, (t % 3600) // 60) for t in t_s.astype(int)])
    out_dir = config.DATA_DIR / 'wrfml_runs'
    for model in args.models.split(','):
        rm = T.run_model(model)
        z_star = products.load('populations', model=model)['z'].values.astype(float)
        for suffix in args.schemes.split(','):
            key = SCHEMES[suffix].format(m=rm)
            z = np.stack([T.latent(s, key, z_star, rm, it) for it in range(len(t_s))]).astype(np.float32)  # (t, d, y, x)
            n = np.stack([T.number(s, key, it) for it in range(len(t_s))]).astype(np.float32)            # (t, y, x)
            if not (np.isfinite(z).all() and np.isfinite(n).all()):
                raise ValueError('non-finite latents in %s %s' % (model, key))
            ij = np.array(list(POINTS.values()))
            z_pts = np.stack([z[:, :, j - 1, i - 1] for i, j in ij]).transpose(0, 2, 1)   # (point, latent, time)
            n_pts = np.stack([n[:, j - 1, i - 1] for i, j in ij])
            run_key = '%s_%s%s' % (model, suffix, args.tag)
            path = out_dir / ('%s.npz' % run_key)
            np.savez_compressed(path, z_dom=z[[0, -1]], ntot_dom=n[[0, -1]], dom_times=times[[0, -1]], z_pts=z_pts,
                                ntot_pts=n_pts, times=times, points=list(POINTS), point_ij=ij,
                                zsrc=z_star.astype(np.float32), scenarios=np.array(config.POP_SCENARIOS),
                                fpidx=model, z_surf=z, ntot_surf=n, replay=args.replay, scheme=key)
            print('wrote', path.name, z.shape)


if __name__ == '__main__':
    main()
