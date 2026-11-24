#!/usr/bin/env python
"""dry_comparison: the paper's comparison (Figs 4 and 5: scorecard and error against tracers) with
the number and speciated-mass size distributions binned by DRY diameter (2026-11-16: dry
diameter is the usual coordinate in atmospheric science; would reviewers find the wet comparison
unfair to the sectional schemes?).

Everything is as in the paper (transport_terms.json; paperfigs/comparison.py) except the two size
distributions, which are binned by dry diameter (H2O excluded from the diameter and from the mass,
14 species; dist_checks.size_dists) for the reference, the four schemes and WRF-AeroMELD:
  - reference Q0: the source particle files binned by dry diameter, mixed in number with f_PS;
  - schemes: their representative particles (R.particles), as dist_checks.py;
  - WRF-AeroMELD: a decoder retrained on dry targets with the encoder frozen at twin_k16_of's
    (training round 36: training/dataset/dry_cache.py, tasks/r36/r36_dry_k16.txt), so the latents
    and the transport runs are the paper's; models data/models/twin_k16dry.<seed>.npz. Its other
    four diagnostics stay the paper decoder's. Control: twin_k16wetctl.<seed> (seeds 1 and 6), the
    same frozen-encoder retraining on the paper's wet targets.
Three errors per diagnostic, means over the 5965 mixed cells at 12 h, as transport_terms.json:
  total           e(Q4, Q0)   the error (Q4 = the scheme's 12-h state after separate-limiter transport)
  representation  e(Q2, Q0)   Q2 = the scheme's representation of the exact mixture
  transport       e(Q4, Q2)
The wet values are recomputed beside the dry ones and checked against transport_terms.json.

Writes
  data/dry_comparison_schemes.json, data/dry_comparison_cells_schemes.npz   (--part schemes; Perlmutter)
  data/dry_comparison_aeromeld.json                                          (--part aeromeld; locally)
  data/transport_terms_dry.json  (--merge): transport_terms.json's layout ('cases' -> case -> '12h'
      -> total / transport / representation -> diagnostic and 'overall'), the two size
      distributions dry, 'overall' the mean of the six; read by paperfigs/comparison.py when
      PAPERFIGS_DRY=1. Case 'aeromeld' is the representative seed (1), as in the paper;
      'aeromeld_median' the median over the ten seeds of each value.

Usage (from scripts/, like dist_checks.py):
  python ../baselines/dry_comparison.py --part schemes [--cells 200]
  python ../baselines/dry_comparison.py --part aeromeld [--seeds 0-9]
  python ../baselines/dry_comparison.py --merge
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import diag  # noqa: E402
import dist_checks as DC  # noqa: E402
from dist_checks import BASES, DATA, LABELS, SCHEMES, blocks_of, rep_dists, scores, size_dists  # noqa: E402

AEROMELD = DC.AEROMELD                      # twin_k16_of
DRY_MODEL = 'twin_k16dry'
CTL_MODEL = 'twin_k16wetctl'
TERMS = ('total', 'representation', 'transport')
SIZE = {'n_prthst': 'e_n', 'm_chmprthst': 'e_m'}
CASES = ('aeromeld',) + SCHEMES


def three(d2, n2, d4, n4, d0, n0):
    """Per cell {term: {metric: (n,)}}: total Q4 vs Q0, representation Q2 vs Q0, transport Q4 vs Q2."""
    return {'total': scores(*d4, n4, *d0, n0), 'representation': scores(*d2, n2, *d0, n0),
            'transport': scores(*d4, n4, *d2, n2)}


def means(sc):
    return {t: {q: float(np.nanmean(sc[t][k])) for q, k in SIZE.items()} for t in TERMS}


# --- schemes (Perlmutter) ------------------------------------------------------------------------

def part_schemes(ncells=None):
    sc = DC.ST.Case()
    f_O, n_O, mask, fm, nO = sc.cells(12.0)
    rho = nO / n_O[mask]
    sel = slice(None) if not ncells else slice(0, ncells)
    ref = DC.Reference(sc.x_pr)
    d0 = {b: ref.mixed(fm[sel], b) for b in BASES}
    res = {'mixed_12h': int(mask.sum()), 'cells_used': int(len(fm[sel])),
           'check_reference_wet_vs_particle_files': ref.check_wet, 'models': {}}
    cells = {'mask': mask}
    for model in SCHEMES:
        t0 = time.perf_counter()
        R = sc.reps[model]
        x2, n2 = (fm @ sc.zsrc[model + '_z'])[sel], nO[sel]
        xs, nS, path = DC.transported_state(sc, model, mask)
        x4, n4 = xs[sel], (nS * rho)[sel]
        d2, d4 = rep_dists(R, x2), rep_dists(R, x4)
        r = res['models'][model] = {'label': LABELS[model], 'transported_file': path}
        for b in BASES:
            s = three(d2[b], n2, d4[b], n4, d0[b], nO[sel])
            r[b] = means(s)
            for t in TERMS:
                for q, k in SIZE.items():
                    cells['%s__%s__%s__%s' % (model, b, t, q)] = s[t][k].astype(np.float32)
        print('%-8s (%.0f s): ' % (model, time.perf_counter() - t0) + '  '.join(
            '%s %s N/M %.2f/%.2f' % (b, t, 100 * r[b][t]['n_prthst'], 100 * r[b][t]['m_chmprthst'])
            for b in BASES for t in TERMS), flush=True)
    tag = '' if not ncells else '_test%d' % ncells
    DC._write(DATA / ('dry_comparison_schemes%s.json' % tag), res)
    np.savez_compressed(DATA / ('dry_comparison_cells_schemes%s.npz' % tag), **cells)
    print('wrote', DATA / ('dry_comparison_schemes%s.json' % tag))


# --- WRF-AeroMELD (locally) ----------------------------------------------------------------------

def decoded(name, z):
    """(N (n, 15), M (n, C, 15)) of the decoder `name` (<model>.<seed>): wet for the paper's and the
    control, dry (H2O row dropped: 14 species) for the dry decoder."""
    N, M = blocks_of(DC.errors.decode_cells(name, z))
    return (N, M[:, diag.DRY]) if name.startswith(DRY_MODEL) else (N, M)


def aeromeld_seed(seed, src):
    T, errors, products, analysis = DC.T, DC.errors, DC.products, DC.analysis
    model = '%s.%d.0' % (AEROMELD, seed)
    s = T.load('all%d_12h' % seed)
    it = T.time_index(s, 12.0)
    rm = T.run_model(model)
    pop = products.load('populations', model=model)
    z_star = pop['z'].values.astype(float)
    f_O, n_O = T.fractions(s, 'oracle', it), T.number(s, 'oracle', it)
    mask = (f_O >= analysis.MIX_MIN_FRAC).sum(0) >= 2
    rho = 1.0 / errors.alt_surface(12.0)[mask]
    fm, nO = f_O[:, mask].T, n_O[mask] * rho
    z4 = T.latent(s, 'sep:%s' % rm, z_star, rm, it)[:, mask].T
    n4 = T.number(s, 'sep:%s' % rm, it)[mask] * rho
    z2 = fm @ z_star
    xw = errors.source_vectors(pop, 'pmc')
    Nw, Mw = blocks_of(xw)
    chk = float(max(np.abs(src['wet'][0] - Nw).max() / np.abs(Nw).max(),
                    np.abs(src['wet'][1] - Mw).max() / np.abs(Mw).max()))
    if chk > 1e-5:
        raise ValueError('source order: particle files and populations product differ (%.2e)' % chk)
    d0 = {'wet': (fm @ Nw, np.einsum('ns,skb->nkb', fm, Mw)),
          'dry': (fm @ src['dry'][0], np.einsum('ns,skb->nkb', fm, src['dry'][1]))}
    out = {'model': model, 'mixed_12h': int(mask.sum()), 'check_sources_wet_rel': chk}
    names = {'wet': '%s.%d' % (AEROMELD, seed), 'dry': '%s.%d' % (DRY_MODEL, seed)}
    ctl = diag.REPO / 'data' / 'models' / ('%s.%d.npz' % (CTL_MODEL, seed))
    if ctl.exists():
        names['wet_control'] = '%s.%d' % (CTL_MODEL, seed)
    for b, name in names.items():
        d2, d4 = decoded(name, z2), decoded(name, z4)
        out[b] = means(three(d2, nO, d4, n4, d0['dry' if b == 'dry' else 'wet'], nO))
    # the latents of the dry decoder's encoder must be the paper's (frozen): checked in --check-z
    return out, mask


def part_aeromeld(seeds):
    m, w = diag.sources()
    src = {b: size_dists(m, w, b) for b in BASES}
    dec = json.loads((diag.REPO / 'data' / 'decomposition.json').read_text())['models'][AEROMELD]
    res = {'recipe': AEROMELD, 'dry_model': DRY_MODEL, 'representative_seed': int(dec['representative_seed']),
           'seeds': {}}
    for k in seeds:
        t0 = time.perf_counter()
        out, mask = aeromeld_seed(k, src)
        d = dec['seeds'][str(k)]
        out['decomposition_json_wet'] = {'total': {q: d['sep']['total'][q] for q in SIZE},
                                         'transport': {q: d['sep']['transport'][q] for q in SIZE},
                                         'representation': {q: d['oracle']['decoder'][q] for q in SIZE},
                                         'other_four': {t: {q: v for q, v in d[g][key].items() if q not in SIZE}
                                                        for t, g, key in (('total', 'sep', 'total'),
                                                                          ('transport', 'sep', 'transport'),
                                                                          ('representation', 'oracle', 'decoder'))}}
        res['seeds'][str(k)] = out
        print('seed %d (%.1f s): ' % (k, time.perf_counter() - t0) + '  '.join(
            '%s %s N/M %.2f/%.2f' % (b, t, 100 * out[b][t]['n_prthst'], 100 * out[b][t]['m_chmprthst'])
            for b in ('wet', 'dry', 'wet_control') if b in out for t in TERMS), flush=True)
    DC._write(DATA / 'dry_comparison_aeromeld.json', res)
    print('wrote', DATA / 'dry_comparison_aeromeld.json')


# --- merge ---------------------------------------------------------------------------------------

def merge():
    tt = json.loads((DATA / 'transport_terms.json').read_text())
    sch = json.loads((DATA / 'dry_comparison_schemes.json').read_text())
    am = json.loads((DATA / 'dry_comparison_aeromeld.json').read_text())
    six = tt['six']
    rep = str(am['representative_seed'])
    out = {'note': 'transport_terms.json with the number and speciated-mass size distributions binned by '
                   'DRY diameter (baselines/dry_comparison.py); total, transport and representation only',
           'six': six, 'cases': {}, 'checks': {}}

    def case(c, dry, wet_check):
        d = {'12h': {}}
        for t in TERMS:
            v = dict(tt['cases'][c]['12h'][t])
            for q in SIZE:
                out['checks'].setdefault(c, {}).setdefault(t, {})[q] = {
                    'published_wet': v[q], 'recomputed_wet': wet_check[t][q], 'dry': dry[t][q]}
                v[q] = dry[t][q]
            v['overall'] = float(np.mean([v[q] for q in six]))
            d['12h'][t] = v
        return d

    for c in SCHEMES:
        out['cases'][c] = case(c, sch['models'][c]['dry'], sch['models'][c]['wet'])
    out['cases']['aeromeld'] = case('aeromeld', am['seeds'][rep]['dry'], am['seeds'][rep]['wet'])
    out['cases']['aeromeld']['seed'] = int(rep)
    # the median over seeds of each value (the other four diagnostics from decomposition.json)
    per = []
    for k, v in am['seeds'].items():
        p = {}
        for t in TERMS:
            p[t] = dict(v['decomposition_json_wet']['other_four'][t])
            p[t].update(v['dry'][t])
            p[t]['overall'] = float(np.mean([p[t][q] for q in six]))
        per.append(p)
    out['cases']['aeromeld_median'] = {'12h': {t: {q: float(np.median([p[t][q] for p in per]))
                                                   for q in list(six) + ['overall']} for t in TERMS},
                                       'seeds': sorted(int(k) for k in am['seeds'])}
    diffs = [abs(x['published_wet'] - x['recomputed_wet']) for c in out['checks'].values()
             for t in c.values() for x in t.values()]
    out['checks_max_abs_diff_wet'] = float(max(diffs))
    # the wet control (frozen-encoder retraining on the paper's targets)
    out['wet_control'] = {k: {'paper_decoder': v['wet'], 'retrained_wet': v['wet_control'], 'retrained_dry': v['dry']}
                          for k, v in am['seeds'].items() if 'wet_control' in v}
    DC._write(DATA / 'transport_terms_dry.json', out)
    print('wrote', DATA / 'transport_terms_dry.json', ' max |wet recomputed - published| %.2e'
          % out['checks_max_abs_diff_wet'])
    print('\nerror (total), %%: wet -> dry')
    for c in CASES + ('aeromeld_median',):
        w = tt['cases'][c if c != 'aeromeld_median' else 'aeromeld']['12h']['total']
        d = out['cases'][c]['12h']['total']
        print('  %-16s N %5.1f -> %5.1f   M %5.1f -> %5.1f   overall %5.2f -> %5.2f' % (
            c, 100 * w['n_prthst'], 100 * d['n_prthst'], 100 * w['m_chmprthst'], 100 * d['m_chmprthst'],
            100 * w['overall'], 100 * d['overall']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', choices=('schemes', 'aeromeld'))
    ap.add_argument('--cells', type=int, default=0, help='schemes: only the first N mixed cells (a test)')
    ap.add_argument('--seeds', default='0-9', help='aeromeld: seeds a-b')
    ap.add_argument('--merge', action='store_true')
    args = ap.parse_args()
    if args.merge:
        return merge()
    if args.part == 'schemes':
        diag.set_threads(1)
        return part_schemes(args.cells or None)
    if args.part == 'aeromeld':
        lo, hi = (int(v) for v in args.seeds.split('-')) if '-' in args.seeds else (int(args.seeds),) * 2
        return part_aeromeld(list(range(lo, hi + 1)))
    ap.error('--part or --merge')


if __name__ == '__main__':
    main()
