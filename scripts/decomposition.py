#!/usr/bin/env python
"""data/decomposition.json: the four-term error decomposition (paperfigs/errors.py) of every
decoder, training seed and transport scheme in the case, at one time (default the end of the
run, 12 h), from the 12-hour replays all<seed>_12h (transport/rep12_array.sbatch).

Per decoder (recipe), seed and scheme ('donor', 'shB', 'sep', 'oracle' = per-source transport):
for each term (off-mixture, fraction, mixture, reconstruction), group (transport, decoder) and
the total, the error of each of the six diagnostics and their mean ('overall'), averaged over
the mixed cells of per-source transport. Also, per decoder, the representative seed: the one
whose six total errors with separate limiters are jointly closest to the ten-seed medians
(least sum over the diagnostics of |ln(error / median)|).

With --time MODEL:REPLAY,...: the same through the run (every --every hours, from the 30-minute
extracts of the representative seeds' replays), with the number of mixed cells and of cells
holding at least 5 % urban number, written to data/decomposition_time.json.

Usage (from scripts/): python decomposition.py [--hours 12] [--seeds 0-9] [--out ../data/decomposition.json]
                       python decomposition.py --time twin_k16_of.1.0:all1_12h,ax_rel1_of.5.0:all5_12h
"""
import argparse
import json
from pathlib import Path

import numpy as np

from paperfigs import config, errors
from paperfigs import transport_schemes as T

RECIPES = tuple(config.recipe(b) for b in config.BASES)
SCHEMES = ('donor', 'shB', 'sep', 'oracle')


def scheme_key(scheme, rm):
    return '%s:%s' % (scheme, rm) if scheme in ('shB', 'sep') else scheme


def representative(seeds):
    """The seed whose six total errors (separate limiters) are jointly closest to the medians."""
    tot = {k: np.array([v['sep']['total'][d] for d in errors.SIX]) for k, v in seeds.items()}
    med = np.median(np.array(list(tot.values())), axis=0)
    dist = {k: float(np.abs(np.log(t / med)).sum()) for k, t in tot.items()}
    return min(dist, key=dist.get), dist


def time_series(items, every, out_path):
    """--time: the decomposition through the run for (model, replay) pairs, every `every` hours,
    with the number of mixed cells and of cells holding at least 5 % urban number."""
    from paperfigs import analysis
    out = {}
    for item in items.split(','):
        model, run = item.split(':')
        rm = T.run_model(model)
        s = T.load(run)
        hours = [h for h in np.asarray(s['time_s']) / 3600 if h > 0 and np.isclose(h / every, round(h / every))]
        rec = {'hours': [], 'n_mixed': [], 'n_urban_cells': [], 'schemes': {sch: [] for sch in SCHEMES}}
        for h in hours:
            f = T.fractions(s, 'oracle', T.time_index(s, h))
            r = errors.decompose(model, run, float(h), keys=[scheme_key(sch, rm) for sch in SCHEMES])
            rec['hours'].append(float(h))
            rec['n_mixed'].append(r['mixed'])
            rec['n_urban_cells'].append(int((f[1] >= analysis.MIX_MIN_FRAC).sum()))
            for sch in SCHEMES:
                rec['schemes'][sch].append(r['schemes'][scheme_key(sch, rm)])
            print('%s %s %5.2f h: mixed %d, total sep %.2f donor %.2f' % (
                model, run, h, r['mixed'], 100 * rec['schemes']['sep'][-1]['total']['overall'],
                100 * rec['schemes']['donor'][-1]['total']['overall']), flush=True)
        out['%s:%s' % (model, run)] = rec
    Path(out_path).write_text(json.dumps(out, indent=1))
    print('wrote', out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--hours', type=float, default=12.0)
    ap.add_argument('--seeds', default='0-9')
    ap.add_argument('--recipes', default=','.join(RECIPES))
    ap.add_argument('--out', default=str(Path(__file__).resolve().parents[1] / 'data' / 'decomposition.json'))
    ap.add_argument('--time', help='MODEL:REPLAY,... : the decomposition through the run instead '
                    '(writes data/decomposition_time.json unless --out is given)')
    ap.add_argument('--every', type=float, default=0.5)
    args = ap.parse_args()
    if args.time:
        out = args.out if args.out != ap.get_default('out') else str(Path(args.out).with_name('decomposition_time.json'))
        return time_series(args.time, args.every, out)
    lo, hi = (int(x) for x in args.seeds.split('-'))
    out = {'hours': args.hours, 'six': list(errors.SIX), 'terms': list(errors.TERMS) + list(errors.GROUPS),
           'models': {}}
    for recipe in args.recipes.split(','):
        rm = T.run_model(recipe + '.0.0')
        seeds = {}
        for k in range(lo, hi + 1):
            model = '%s.%d.0' % (recipe, k)
            r = errors.decompose(model, config.REPLAY_12H % k, args.hours, keys=[scheme_key(s, rm) for s in SCHEMES])
            seeds[str(k)] = {s: r['schemes'][scheme_key(s, rm)] for s in SCHEMES}
            out['mixed'] = r['mixed']
            print('%s s%d: overall total %s | sep: offmix %.2f frac %.2f mix %.2f recon %.2f' % (
                recipe, k, ' '.join('%s %.2f' % (s, 100 * seeds[str(k)][s]['total']['overall']) for s in SCHEMES),
                *(100 * seeds[str(k)]['sep'][t]['overall'] for t in errors.TERMS)), flush=True)
        rep, dist = representative(seeds)
        out['models'][recipe] = {'seeds': seeds, 'representative_seed': int(rep), 'rep_distance': dist}
        print('%s representative seed %s' % (recipe, rep), flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print('wrote', args.out)


if __name__ == '__main__':
    main()
