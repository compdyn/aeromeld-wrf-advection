#!/usr/bin/env python
"""Does the optics fix (notes/pmc_diagnostics_check.md) change the results?

Compares each model trained on the original targets with its round-33 twin trained
on the optics-corrected targets (<model>_of):

  test set   the eight metrics (training/paper_errors.py), mean over test populations
             and seeds, each model against its own training targets
             (data/affx_testset/, data/affx_testset_of/)
  WRF        spectrum error over the mixed cells at 04 UTC and the attribution's
             end-member term (mean and RMS of the log10 ratio), mean over seeds
             (data/seed_stats.json)

Usage (from scripts/): python checks/optics_fix_compare.py
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
NAMES = ('twin_none', 'twin_pairs', 'twin_k16', 'twin_k32', 'ax_rel1', 'ax_k16')
GEO = ['ccn_cdf', 'logqs_pop', 'logqa_pop', 'logfrznfrac_tmp', 'm_chmprthst', 'n_prthst', 'm_prthst', 'm_chmprt']
SHORT = dict(ccn_cdf='CCN', logqs_pop='scat', logqa_pop='abs', logfrznfrac_tmp='FF', m_chmprthst='Mspec',
             n_prthst='N', m_prthst='Mtot', m_chmprt='Mbulk')


def test_errors(d, name):
    rows = []
    for f in sorted(d.glob('%s.*.npz' % name)):
        with np.load(f) as z:
            rows.append([z['err_' + k].mean() for k in GEO])
    if not rows:
        return None
    r = np.array(rows)                                           # (seeds, metrics)
    geo = np.exp(np.log(r).mean(1))
    return len(rows), 100 * r.mean(0), 100 * geo.mean()


print('Test set: each model against its own targets, mean over seeds (%)')
print('| model | targets | seeds | geo | ' + ' | '.join(SHORT[k] for k in GEO) + ' |')
print('|---' * (4 + len(GEO)) + '|')
for n in NAMES:
    for tag, d, key in (('original', REPO / 'data' / 'affx_testset', n),
                        ('optics fixed', REPO / 'data' / 'affx_testset_of', n + '_of')):
        t = test_errors(d, key)
        if t:
            print('| %s | %s | %d | %.2f | %s |' % (n, tag, t[0], t[2], ' | '.join('%.1f' % v for v in t[1])))

st = json.loads((REPO / 'data' / 'seed_stats.json').read_text())
G = ('ccn', 'scattering', 'absorption', 'frozen_fraction')
print('\nWRF-AeroMELD, mixed cells at 04 UTC, mean over seeds: spectrum error (%) / end-member term mean, RMS')
print('| model | targets | ' + ' | '.join(G) + ' |')
print('|---' * (2 + len(G)) + '|')
for n in NAMES:
    for tag, key in (('original', n), ('optics fixed', n + '_of')):
        rs = [r for r in st.values() if r['name'] == key]
        if not rs:
            continue
        cells = []
        for g in G:
            e = np.mean([r['spectra'][g]['smape'] for r in rs])
            em = np.mean([r['spectra'][g]['mean_endmember'] for r in rs])
            er = np.mean([r['spectra'][g]['rms_endmember'] for r in rs])
            cells.append('%.1f / %+.3f, %.3f' % (e, em, er))
        print('| %s | %s | %s |' % (n, tag, ' | '.join(cells)))
