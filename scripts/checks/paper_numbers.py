#!/usr/bin/env python
"""Every number the paper quotes from the case and the test set, printed from the data files, so
the text can be checked against them (2026-11-19, the final rebuild: dry diameter, every mass
including water; config.SIZE_BASIS 'drydiam').

Sources (paperfigs.comparison and config give the paths of the paper's basis):
  baselines/data/drydiam/transport_terms.json   the comparison and every scheme variant (12 h, mixed
                                                cells, all seven terms; transport_terms.py --merge)
  baselines/data/drydiam/matrix_sensitivity.json MATRIX's other mechanisms (static terms)
  baselines/data/transport_terms.json           the wet-diameter comparison (wet-trained models), for
                                                the appendix paragraph on wet diameter
  data/decomposition.json                       WRF-AeroMELD: decoders x transport schemes x seeds
  notes/affx_stats/mixk_r39_lr.json             the decoders on the test set
  baselines/data/drydiam/kcurves_baselines.json the schemes on the test set

Usage (from scripts/): python checks/paper_numbers.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paperfigs import comparison as CMP, config, errors  # noqa: E402

B = config.REPO / 'baselines' / 'data'
SIX = list(errors.SIX)
SHORT = dict(zip(SIX, ('N', 'M', 'CCN', 'FF', 'scat', 'abs')))
SCHEMES = ('sec8', 'tomas40', 'mam4l', 'matrix1')


def pc(v):
    return '%.1f' % (100 * v)


def row(d):
    return '  '.join('%s %s' % (SHORT[q], pc(d[q])) for q in SIX) + '  | overall %.2f' % (100 * d['overall'])


def no_frozen(d):
    return float(np.mean([d[q] for q in SIX if q != 'logfrznfrac_tmp']))


def comparison(path, label):
    tt = json.loads(path.read_text())['cases']
    print('\n== comparison, %s (%s) ==' % (label, path.name))
    am = tt['aeromeld']['12h']
    for c in ('aeromeld', 'aeromeld_median') + SCHEMES:
        if c in tt:
            t = tt[c]['12h']
            print('%-16s total  %s  | w/o FF %.2f' % (c, row(t['total']), 100 * no_frozen(t['total'])))
            print('%-16s transport %.2f  representation %.2f' % ('', 100 * t['transport']['overall'],
                                                                100 * t['representation']['overall']))
    r = [tt[c]['12h']['total']['overall'] / am['total']['overall'] for c in SCHEMES]
    rn = [no_frozen(tt[c]['12h']['total']) / no_frozen(am['total']) for c in SCHEMES]
    rr = [tt[c]['12h']['representation']['overall'] / am['representation']['overall'] for c in SCHEMES]
    print('ratio scheme/AeroMELD: overall %.2f-%.2f, w/o FF %.2f-%.2f, representation %.2f-%.2f'
          % (min(r), max(r), min(rn), max(rn), min(rr), max(rr)))
    for q in SIX:
        best = min(SCHEMES, key=lambda c: tt[c]['12h']['total'][q])
        print('  %-5s AeroMELD %s, best scheme %s %s, ratio %.2f' % (
            SHORT[q], pc(am['total'][q]), best, pc(tt[best]['12h']['total'][q]),
            tt[best]['12h']['total'][q] / am['total'][q]))
    if 'aeromeld_median' in tt and 'range_overall' in tt['aeromeld_median']:
        print('AeroMELD seeds, total overall range %s' % tt['aeromeld_median']['range_overall']['total'])


def variants():
    tt = json.loads(CMP.TRANSPORT_TERMS.read_text())['cases']
    print('\n== scheme variants, %s (%s) ==' % (CMP.BASIS, CMP.TRANSPORT_TERMS.relative_to(config.REPO)))
    for c, v in tt.items():
        for t in ('total', 'representation', 'transport', 'mixture', 'reconstruction', 'fraction', 'offmixture'):
            if t in v['12h']:
                print('%-16s %-15s %s' % (c, t, row(v['12h'][t])))
    ms = CMP.TRANSPORT_TERMS.with_name('matrix_sensitivity.json')
    if ms.exists():
        print('\n== MATRIX mechanisms, static terms (%s) ==' % ms.relative_to(config.REPO))
        for c, v in json.loads(ms.read_text())['variants'].items():
            for t in ('representation', 'mixture', 'reconstruction'):
                print('%-13s d %2d %-15s %s' % (c, v['n_tracers'], t, row(v['12h'][t])))


def aeromeld():
    dec = json.loads((config.DATA_DIR / 'decomposition.json').read_text())
    print('\n== WRF-AeroMELD decomposition (decomposition.json) ==')
    for name, m in dec['models'].items():
        seeds = m['seeds']
        rep = str(m['representative_seed'])
        print('%s: representative seed %s' % (name, rep))
        for sch in ('oracle', 'donor', 'shB', 'sep'):
            if sch not in seeds[rep]:
                continue
            for t in ('total', 'transport', 'decoder', 'fraction', 'offmixture', 'mixture', 'reconstruction'):
                v = [seeds[k][sch][t]['overall'] for k in seeds]
                print('  %-6s %-14s median %5.2f  range %5.2f-%5.2f  rep %5.2f' % (
                    sch, t, 100 * np.median(v), 100 * min(v), 100 * max(v), 100 * seeds[rep][sch][t]['overall']))
            fz = [seeds[k][sch]['total']['logfrznfrac_tmp'] for k in seeds]
            print('  %-6s %-14s median %5.2f (frozen fraction, total)' % (sch, '', 100 * np.median(fz)))


def points():
    """Results 4.2: the errors at sampling points A, B, C (config.LOCATIONS) of the main run (the
    representative seed of the recommended decoder, separate limiters), from errors.decompose per cell,
    and each point's mixture by number (per-source transport)."""
    from paperfigs import transport_schemes as T
    dec = json.loads((config.DATA_DIR / 'decomposition.json').read_text())['models']
    name = config.recipe('twin_k16')
    seed = int(dec[name]['representative_seed'])
    model, run = '%s.%d.0' % (name, seed), config.REPLAY_12H % seed
    rm = T.run_model(model)
    key = 'sep:%s' % rm
    r = errors.decompose(model, run, 12.0, keys=[key], per_cell=True)
    mask = r['mask']
    idx = np.full(mask.shape, -1)
    idx[mask] = np.arange(mask.sum())
    s = T.load(run)
    f = T.fractions(s, 'oracle', T.time_index(s, 12.0))
    print('\n== points (%s, %s, separate limiters, 12 h) ==' % (model, run))
    for lab, (i, j) in zip(config.LOC_LABELS, config.LOCATIONS):
        c = idx[j - 1, i - 1]
        if c < 0:
            print('%s (%d, %d): not a mixed cell' % (lab, i, j))
            continue
        cell = {t: {q: float(r['cells'][key][t][q][c]) for q in SIX} for t in ('total', 'transport', 'decoder')}
        print('%s (%d, %d) number bg/urb/emi %s | total overall %.2f  transport overall %.2f  decoder overall %.2f' % (
            lab, i, j, ' '.join('%.2f' % x for x in f[:, j - 1, i - 1]), 100 * np.mean(list(cell['total'].values())),
            100 * np.mean(list(cell['transport'].values())), 100 * np.mean(list(cell['decoder'].values()))))
        print('   total  ' + '  '.join('%s %.1f' % (SHORT[q], 100 * cell['total'][q]) for q in SIX))


def schemes_appendix():
    """Appendix B: the transport schemes with the recommended decoder (medians over the seeds), the
    frozen fraction under donor-cell and separate limiters, the time series (decomposition_time.json)
    and every decoder under every scheme."""
    dec = json.loads((config.DATA_DIR / 'decomposition.json').read_text())['models']
    print('\n== transport schemes and decoders (decomposition.json, medians over seeds) ==')
    med = lambda m, sch, t, q='overall': 100 * float(np.median([v[sch][t][q] for v in dec[m]['seeds'].values()]))
    for m in dec:
        print('%-14s total: donor %.2f shB %.2f sep %.2f oracle %.2f | sep-oracle %.2f shB-oracle %.2f | '
              'transport: donor %.2f shB %.2f sep %.2f | recon %.2f averaging %.2f' % (
                  m, *(med(m, x, 'total') for x in ('donor', 'shB', 'sep', 'oracle')),
                  med(m, 'sep', 'total') - med(m, 'oracle', 'total'), med(m, 'shB', 'total') - med(m, 'oracle', 'total'),
                  *(med(m, x, 'transport') for x in ('donor', 'shB', 'sep')),
                  med(m, 'oracle', 'reconstruction'), med(m, 'oracle', 'mixture')))
    k16 = config.recipe('twin_k16')
    print('%s frozen fraction, total: donor %.1f  sep %.1f  (medians); size distributions and CCN, donor: %s' % (
        k16, med(k16, 'donor', 'total', 'logfrznfrac_tmp'), med(k16, 'sep', 'total', 'logfrznfrac_tmp'),
        '  '.join('%s %.1f' % (SHORT[q], med(k16, 'donor', 'total', q)) for q in SIX)))
    nl = [med(config.recipe(b), 'oracle', 'reconstruction') for b in ('twin_none', 'twin_pairs', 'twin_k16', 'twin_k32')]
    print('nonlinear reconstruction / affine: %.2f-%.2f' % (min(nl) / med(config.recipe('ax_rel1'), 'oracle', 'reconstruction'),
                                                           max(nl) / med(config.recipe('ax_rel1'), 'oracle', 'reconstruction')))
    p = config.DATA_DIR / 'decomposition_time.json'
    if p.exists():
        for k, rec in json.loads(p.read_text()).items():
            h = np.array(rec['hours'])
            ov = lambda sch, t: 100 * np.array([x[t]['overall'] for x in rec['schemes'][sch]])
            d = ov('sep', 'total') - ov('oracle', 'total')
            om, fr = ov('sep', 'offmixture'), ov('sep', 'fraction')
            print('%s: sep - oracle %.2f to %.2f pp; sep off-mixture max %.2f at %g h (12 h %.2f), fraction max %.2f; '
                  'mixed cells %d (0.5 h) to %d (max at %g h; 8 h %d)' % (
                      k, d.min(), d.max(), om.max(), h[om.argmax()], om[-1], fr.max(), rec['n_mixed'][0],
                      max(rec['n_mixed']), h[int(np.argmax(rec['n_mixed']))], rec['n_mixed'][int(np.flatnonzero(h == 8)[0])]))
            dn = ov('donor', 'total')
            print('   donor-cell total: 0.5 h %.2f, 4 h %.2f, max %.2f at %g h, 12 h %.2f' % (
                dn[0], dn[int(np.flatnonzero(h == 4)[0])], dn.max(), h[dn.argmax()], dn[-1]))


def testset():
    p = config.REPO / 'notes' / 'affx_stats' / {'_dd': 'mixk_r39_lr.json', '_dry': 'mixk_r37_lr.json', '_of': 'mixk_r33_lr.json'}[config.SUFFIX]
    if not p.exists():
        print('\n(no %s yet)' % p.name)
        return
    print('\n== test set (%s): six-diagnostic mean, %%, mean over seeds [min-max] ==' % p.name)
    d = json.loads(p.read_text())
    six = lambda r: 100 * float(np.mean([r[q] for q in SIX]))
    for name, v in d.items():
        Ks = sorted(int(k) for k in v['K'])
        cells = []
        for K in Ks:
            x = [six(r['mix']) for r in v['K'][str(K)]]
            cells.append('K=%d %.2f [%.1f-%.1f]' % (K, np.mean(x), min(x), max(x)))
        print('%-14s %s' % (name, '  '.join(cells)))
        k1 = [six(r['mix']) for r in v['K']['1']]
        print('%-14s K=1 per seed: %s; K=5000 per diagnostic (mean): %s' % (
            '', ' '.join('%.1f' % x for x in k1),
            '  '.join('%s %.1f' % (SHORT[q], 100 * np.mean([r['mix'][q] for r in v['K']['5000']])) for q in SIX)))
    extra = p.with_name({'_dd': 'mixk_r40_lr.json', '_dry': 'mixk_r38_lr.json', '_of': 'mixk_r35_lr.json'}[config.SUFFIX])
    if extra.exists():
        print('-- %s (the checks of the decoder trained without mixtures)' % extra.name)
        for name, v in json.loads(extra.read_text()).items():
            Ks = sorted(int(k) for k in v['K'])
            print('%-16s seeds %s  %s' % (name, v.get('seeds'), '  '.join(
                'K=%d %.2f [%.1f-%.1f]' % (K, np.mean([six(r['mix']) for r in v['K'][str(K)]]),
                                           min(six(r['mix']) for r in v['K'][str(K)]),
                                           max(six(r['mix']) for r in v['K'][str(K)])) for K in Ks)))


def main():
    print('size basis %s, models %s' % (CMP.BASIS, config.SUFFIX))
    if CMP.TRANSPORT_TERMS.exists():
        comparison(CMP.TRANSPORT_TERMS, CMP.BASIS)
    comparison(B / 'transport_terms.json', 'wet diameter, wet-trained models')
    variants()
    if (config.DATA_DIR / 'decomposition.json').exists():
        aeromeld()
        points()
        schemes_appendix()
    testset()


if __name__ == '__main__':
    main()
