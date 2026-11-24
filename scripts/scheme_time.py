#!/usr/bin/env python
"""data/scheme_time.json: each transport scheme's error against the exact answer
over time, from the 12-hour replays (2026-11-12).

For each (model, replay) and every output time from 0.5 h on (every --every
hours), transport_schemes.budget gives, per scheme (donor-cell, shared bounded
limiter, separate limiters, per-source transport) and diagnostic (CCN/CN,
scattering, absorption, frozen fraction), the spectrum SMAPE over the mixed
cells and the RMS of the three budget terms (transport scheme, decoding
mixtures, end-member). Alongside, per time: the number of mixed surface cells,
of surface cells holding at least 5 % urban number, and the surface urban
number relative to its value at the start. The urban plume reaches the
southern boundary from about 6 h, but the cells holding at least 5 % urban
number keep growing to about 9.5 h (1724 at 4 h, 4815 at 9.5 h, 4318 at 12 h),
so the late-time statistics do not rest on fewer cells.

Usage (from scripts/): python scheme_time.py [--runs twin_k16_of.6.0:rep6_12h,ax_rel1_of.5.0:rep5_12h]
                       [--every 0.5] [--out ../data/scheme_time.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from paperfigs import analysis
from paperfigs import transport_schemes as T

SCHEMES = ('donor', 'shB:{m}', 'sep:{m}', 'oracle')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--runs', default='twin_k16_of.6.0:rep6_12h,ax_rel1_of.5.0:rep5_12h')
    ap.add_argument('--every', type=float, default=0.5)
    ap.add_argument('--out', default=str(Path(__file__).resolve().parents[1] / 'data' / 'scheme_time.json'))
    args = ap.parse_args()
    out = {}
    for item in args.runs.split(','):
        model, run = item.split(':')
        s = T.load(run)
        rm = T.run_model(model)
        keys = [k.format(m=rm) for k in SCHEMES]
        t_end = float(s['time_s'][-1]) / 3600
        hours = np.arange(args.every, t_end + 1e-9, args.every)
        n0 = s['oracle'][0].astype(float)
        urban0 = n0[list(T.ORACLE_GROUPS[1])].sum()
        rec = {'hours': [], 'n_mixed': [], 'n_urban_cells': [], 'urban_surface_rel': [], 'schemes': {k: [] for k in keys}}
        for h in hours:
            it = T.time_index(s, h)
            f = T.fractions(s, 'oracle', it)
            n = s['oracle'][it].astype(float)
            rec['hours'].append(float(h))
            rec['n_mixed'].append(int(((f >= analysis.MIX_MIN_FRAC).sum(0) >= 2).sum()))
            rec['n_urban_cells'].append(int((f[1] >= 0.05).sum()))
            rec['urban_surface_rel'].append(float(n[list(T.ORACLE_GROUPS[1])].sum() / urban0))
            b = T.budget(model, run, keys=keys, hours=float(h))
            for k in keys:
                rec['schemes'][k].append({g: {'smape': 100 * v['error'], 'rms': v['rms']} for g, v in b[k].items()})
            print('%s %s %5.2f h: mixed %d, sep scattering %.1f%%, donor %.1f%%' % (
                model, run, h, rec['n_mixed'][-1], rec['schemes'][keys[2]][-1]['scattering']['smape'],
                rec['schemes'][keys[0]][-1]['scattering']['smape']), flush=True)
        out['%s:%s' % (model, run)] = rec
    Path(args.out).write_text(json.dumps(out, indent=1))
    print('wrote', args.out)


if __name__ == '__main__':
    main()
