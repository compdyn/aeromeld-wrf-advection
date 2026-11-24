#!/usr/bin/env python
"""dry_variants: every baseline number the paper quotes, with the two size distributions binned by DRY
diameter (2026-11-16, the clean dry rebuild; H2O excluded from the diameter and from the mass).

Only the number and speciated-mass size distributions depend on the binning; the CCN spectrum, frozen
fraction and optics are those of the wet computation (static_terms*.json, transport_terms.json,
matrix_sensitivity.json), and are copied from there. The two dry distributions are computed for every
stage of the decomposition (static_terms.py, transport_terms.py; errors.py):

  Q0 = sum_s f_PS,s q_PR,s           the reference: the source particle files binned by dry diameter
  Q1 = sum_s f_PS,s q_R(x_s)         each source represented alone, then mixed
  Q2 = q_R(sum_s f_PS,s x_s)         the exact mixture in the representation (N_PS)
  Q3 = q_R(x3) on N_S                the transported state's nearest mixture (transport cases only)
  Q4 = q_R(x_S) on N_S               the transported state at 12 h (transport cases only)

from each representation's representative particles (dist_checks.rep_dists, the code path of
common.evaluate), scored with the paper's error (dist_checks.scores: paper_errors' relerr on
paper_transform's units). The overall error of each term is then the mean of the six diagnostics.

Cases: the transport runs of transport_terms.CASES (sec8, tomas40, tomas40_nocs, mam4l, mam4la, mam4,
mam4a, matrix1, sec20w, and sec20: 20 bins in dry diameter, replayed 2026-11-16; all seven terms);
static-only representations (MATRIX's other mechanisms and matrix1_oc, as matrix_sensitivity.py;
three static terms). The sec20 case (its replay ran after the rest) goes in with --cases sec20
--static '' --tag _sec20.

Writes data/dry_variants.json: {model: {term: {six diagnostics, 'overall'}}}, plus 'checks' (the wet
size errors recomputed here against the published ones).

--merge writes data/transport_terms_dry.json, read by paperfigs/comparison.py: transport_terms.json's
layout ('cases' -> case -> '12h' -> term -> diagnostic and 'overall'), every transport case of
dry_variants.json, and WRF-AeroMELD from the dry decomposition (data/decomposition.json, the
round-37 models, config.recipe('twin_k16'), separate limiters): 'aeromeld' its representative seed,
'aeromeld_median' the median over the seeds of each value; its term 'decoder' is the schemes'
'representation'.

Usage (Perlmutter CPU, from the scripts/ directory of a repo copy, like dist_checks.sbatch):
  python ../baselines/dry_variants.py [--cases sec8,...] [--static sec20,...] [--cells 200]
  python ../baselines/dry_variants.py --merge     (locally, after the download)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import diag  # noqa: E402
import dist_checks as DC  # noqa: E402
import matrix  # noqa: E402
import static_terms as ST  # noqa: E402
import transport_terms as TT  # noqa: E402
from paperfigs import config, errors, mixing  # noqa: E402

DATA = HERE / 'data'
SIZE = {'n_prthst': 'e_n', 'm_chmprthst': 'e_m'}
CASES = ('sec8', 'tomas40', 'tomas40_nocs', 'mam4l', 'mam4la', 'mam4', 'mam4a', 'matrix1', 'sec20w', 'sec20')
STATIC = ('matrix2', 'matrix3', 'matrix4', 'matrix5', 'matrix6', 'matrix7', 'matrix8', 'matrix1_oc')
STATIC_PAIRS = {'mixture': (2, 1), 'reconstruction': (1, 0), 'representation': (2, 0)}


def wet_other_four():
    """{model: {term: {diag: value}}} of the wet computation (the four diagnostics that do not
    depend on the binning, and the wet size errors for the check)."""
    out = {}
    tt = json.loads((DATA / 'transport_terms.json').read_text())['cases']
    for c, v in tt.items():
        out[c] = v['12h']
    st = json.loads((DATA / 'static_terms.json').read_text())['12h']
    for tag in ('a', 'm', 't1'):
        p = DATA / ('static_terms_add_%s.json' % tag)
        st.update(json.loads(p.read_text())['12h'])
    for m, v in st.items():
        out.setdefault(m + ':static', v)
    ms = json.loads((DATA / 'matrix_sensitivity.json').read_text())['variants']
    for m, v in ms.items():
        out.setdefault(m + ':static', v['12h'])
    return out


def score(a, b):
    """dist_checks.scores between two (N, M, n) stages -> {diag: (n,)} for the two size diagnostics."""
    s = DC.scores(*a, *b)
    return {q: s[k] for q, k in SIZE.items()}


def mix_dists(fm, d_src):
    """Each source's (N, M) mixed in number fractions fm (n, 3)."""
    N, M = d_src
    return fm @ N, np.einsum('ns,skb->nkb', fm, M)


def run(cases, static, ncells, tag=''):
    sc = ST.Case()
    f_O, n_O, mask, fm_all, nO_all = sc.cells(12.0)
    rho_all = nO_all / n_O[mask]
    sel = slice(None) if not ncells else slice(0, ncells)
    fm, nO, rho = fm_all[sel], nO_all[sel], rho_all[sel]
    ref = DC.Reference(sc.x_pr)
    Q0 = {b: (*ref.mixed(fm, b), nO) for b in ('wet', 'dry')}
    out = {'mixed_12h': int(mask.sum()), 'cells_used': int(len(fm)), 'models': {}, 'checks': {}}
    wet = wet_other_four()

    def finish(name, key_wet, terms_dry, terms_wet):
        r = {}
        for t, v in terms_dry.items():
            base = dict(wet[key_wet][t])
            out['checks'].setdefault(name, {})[t] = {q: [float(base[q]), float(np.nanmean(terms_wet[t][q]))] for q in SIZE}
            for q in SIZE:
                base[q] = float(np.nanmean(v[q]))
            base['overall'] = float(np.mean([base[q] for q in errors.SIX]))
            r[t] = base
        out['models'][name] = r
        dev = max(abs(a - b) for t in out['checks'][name].values() for a, b in t.values())
        print('%-13s overall dry %s  | wet size check max |diff| %.1e' % (
            name, '  '.join('%s %.2f' % (t, 100 * v['overall']) for t, v in r.items()), dev), flush=True)

    for case in cases:
        t0 = time.perf_counter()
        if case not in wet:                     # no wet numbers yet (sec20): all six diagnostics, as transport_terms.py
            wet[case] = TT.Run(case, sc).terms(12.0)['terms']
        run_dir, key, rep = TT.CASES[case]
        R, xs = sc.reps[rep], sc.zsrc[rep + '_z']
        x, nS, path = DC.transported_state(sc, case, mask)            # 12 h, from surface_end.npz
        x, nS = x[sel], nS[sel]
        w = TT.fit_weights(xs, TT.species_groups(R))                    # as transport_terms.Run
        f, off = TT.implied_fractions_weighted(x, xs, w)
        x3 = mixing.f_bar(f).T @ xs
        d_src = DC.rep_dists(R, xs)
        st = {0: Q0['dry'], 1: mix_dists(fm, d_src['dry']) + (nO,)}
        stw = {0: Q0['wet'], 1: mix_dists(fm, d_src['wet']) + (nO,)}
        for k, xk, n in ((2, fm @ xs, nO), (3, x3, nS * rho), (4, x, nS * rho)):
            d = DC.rep_dists(R, xk)
            st[k], stw[k] = d['dry'] + (n,), d['wet'] + (n,)
        terms = {t: score(st[a], st[b]) for t, (a, b) in TT.PAIRS.items()}
        terms_w = {t: score(stw[a], stw[b]) for t, (a, b) in TT.PAIRS.items()}
        finish(case, case, terms, terms_w)
        print('   %s (%.0f s)' % (path, time.perf_counter() - t0), flush=True)

    m, w = diag.sources()
    for name in static:
        t0 = time.perf_counter()
        if name == 'sec20':
            R, xs = sc.reps['sec20'], sc.zsrc['sec20_z']
        else:
            R = matrix.Matrix('matrix1', ocar_as='OC') if name == 'matrix1_oc' else matrix.Matrix(name)
            xs = R.project(m, w)
        d_src = DC.rep_dists(R, xs)
        d2 = DC.rep_dists(R, fm @ xs)
        st = {0: Q0['dry'], 1: mix_dists(fm, d_src['dry']) + (nO,), 2: d2['dry'] + (nO,)}
        stw = {0: Q0['wet'], 1: mix_dists(fm, d_src['wet']) + (nO,), 2: d2['wet'] + (nO,)}
        terms = {t: score(st[a], st[b]) for t, (a, b) in STATIC_PAIRS.items()}
        terms_w = {t: score(stw[a], stw[b]) for t, (a, b) in STATIC_PAIRS.items()}
        finish(name, name + ':static', terms, terms_w)
        print('   (%.0f s)' % (time.perf_counter() - t0), flush=True)
    tag = tag if not ncells else '_test%d' % ncells
    path = DATA / ('dry_variants%s.json' % tag)
    path.write_text(json.dumps(out, indent=1))
    print('wrote', path)


def merge():
    dv = {'models': {}}
    for p in sorted(DATA.glob('dry_variants*.json')):
        if '_test' not in p.name:
            dv['models'].update(json.loads(p.read_text())['models'])
    dec = json.loads((config.REPO / 'data' / 'decomposition.json').read_text())
    six = list(errors.SIX)
    assert dec['six'] == six, dec['six']
    out = {'note': 'transport_terms.json with the number and speciated-mass size distributions binned by DRY '
                   'diameter (baselines/dry_variants.py); WRF-AeroMELD the dry-trained decoders '
                   '(data/decomposition.json)', 'six': six, 'cases': {}}
    for c, v in dv['models'].items():
        if 'total' in v:                                             # the transport cases
            out['cases'][c] = {'12h': v}
    name = config.recipe('twin_k16')
    m = dec['models'][name]
    rename = {'decoder': 'representation'}

    def terms(d):
        return {rename.get(t, t): dict(v) for t, v in d.items()}

    rep = str(m['representative_seed'])
    out['cases']['aeromeld'] = {'12h': terms(m['seeds'][rep]['sep']), 'seed': int(rep), 'model': name}
    per = [terms(v['sep']) for v in m['seeds'].values()]
    out['cases']['aeromeld_median'] = {'12h': {t: {q: float(np.median([p[t][q] for p in per]))
                                                   for q in six + ['overall']} for t in per[0]},
                                       'seeds': sorted(int(k) for k in m['seeds']), 'model': name}
    for t in ('total', 'transport', 'representation'):
        lo, hi = (f([p[t]['overall'] for p in per]) for f in (min, max))
        out['cases']['aeromeld_median'].setdefault('range_overall', {})[t] = [lo, hi]
    path = DATA / 'transport_terms_dry.json'
    path.write_text(json.dumps(out, indent=1))
    print('wrote', path)
    for c, v in out['cases'].items():
        print('  %-16s ' % c + '  '.join('%s %.2f' % (t, 100 * v['12h'][t]['overall'])
                                        for t in ('total', 'transport', 'representation')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', default=','.join(CASES))
    ap.add_argument('--static', default=','.join(STATIC))
    ap.add_argument('--cells', type=int, default=0)
    ap.add_argument('--merge', action='store_true')
    ap.add_argument('--tag', default='', help='output data/dry_variants<tag>.json (merged with the others by --merge)')
    args = ap.parse_args()
    if args.merge:
        return merge()
    diag.set_threads(1)
    run([c for c in args.cases.split(',') if c], [s for s in args.static.split(',') if s], args.cells or None, args.tag)


if __name__ == '__main__':
    main()
